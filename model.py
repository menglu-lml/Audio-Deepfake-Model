import torch
from torch import nn
import torch.nn.functional as F

from transformer import LayerNorm, Transformer




def next_power_of_2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()

def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)


class SincConv(nn.Module):   #SincConv_fast
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size=128, sample_rate=16000, in_channels=1,
                 stride=1, padding=0, dilation=1, bias=False, groups=1, min_low_hz=50, min_band_hz=50):

        super(SincConv,self).__init__()

        if in_channels != 1:
            #msg = (f'SincConv only support one input channel '
            #       f'(here, in_channels = {in_channels:d}).')
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size%2==0:
            self.kernel_size=self.kernel_size+1
            
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(self.to_mel(low_hz),
                          self.to_mel(high_hz),
                          self.out_channels + 1)
        hz = self.to_hz(mel)
        

        # filter lower frequency (out_channels, 1)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))

        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Hamming window
        #self.window_ = torch.hamming_window(self.kernel_size)
        n_lin=torch.linspace(0, (self.kernel_size/2)-1, steps=int((self.kernel_size/2))) # computing only half of the window
        self.window_=0.54-0.46*torch.cos(2*math.pi*n_lin/self.kernel_size);


        # (1, kernel_size/2)
        n = (self.kernel_size - 1) / 2.0
        self.band_pass = 2*math.pi*torch.arange(-n, 0).view(1, -1) / self.sample_rate # Due to symmetry, I only need half of the time axes

 


    def forward(self, x):
        self.band_pass = self.band_pass.to(x.device)

        self.window_ = self.window_.to(x.device)

        low = self.min_low_hz  + torch.abs(self.low_hz_)
        
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_),self.min_low_hz,self.sample_rate/2)
        band=(high-low)[:,0]
        
        f_times_t_low = torch.matmul(low, self.band_pass)
        f_times_t_high = torch.matmul(high, self.band_pass)

        band_pass_left=((torch.sin(f_times_t_high)-torch.sin(f_times_t_low))/(self.band_pass/2))*self.window_ # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations. 
        band_pass_center = 2*band.view(-1,1)
        band_pass_right= torch.flip(band_pass_left,dims=[1])
        
        
        band_pass=torch.cat([band_pass_left,band_pass_center,band_pass_right],dim=1)

        
        band_pass = band_pass / (2*band[:,None])
        

        self.filters = (band_pass).view(self.out_channels, 1, self.kernel_size)

        return F.conv1d(x, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                         bias=None, groups=1) 


class PatchEmbed(nn.Module):
    def __init__(self, feature_size, patch_size, embed_dim, in_chans=1):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.grid_size = (feature_size[0] // patch_size[0], feature_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, H, W = x.shape
        x = x.view(B, 1, H, W)
        x = self.proj(x)
        x = x.flatten(2)
        return x

class EmbedReduce(nn.Module):
    def __init__(self, current_len, seq_size):
        
        super(EmbedReduce,self).__init__()
        self.linear1=nn.Linear(current_len, seq_size[0])
        self.lin_ln1 = nn.LayerNorm(seq_size[0])
        self.linear2=nn.Linear(seq_size[0], seq_size[1])
        self.lin_ln2 = nn.LayerNorm(seq_size[1])
        self.linear3=nn.Linear(seq_size[1], seq_size[2])
        self.lin_ln3 = nn.LayerNorm(seq_size[2])
        
    def forward(self, x):
        #print(x.shape)
        x = self.linear1(x)
        x = self.lin_ln1(x)
        #print(x.shape)
        x = self.linear2(x)
        x = self.lin_ln2(x)
        #print(x.shape)
        x = self.linear3(x)
        x = self.lin_ln3(x)
        #print(x.shape)
        return x.transpose(1, 2) 




class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)  # wont be trained by the model wont update
        

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False) 
        return self.dropout(x)    



class Box(nn.Module):
    def __init__(self, model_args, device):
        super(Box, self).__init__()
        
        self.device = device

        self.Sinc_conv = SincConv(out_channels = model_args['num_filter'],
                                  kernel_size = model_args['filt_len'],
                                  in_channels = model_args['in_channels'])
        self.feature_dim = int((model_args['samp_len']-model_args['filt_len']+1)/model_args['max_pool_len'])
        self.lnorm1 = LayerNorm([model_args['num_filter'],self.feature_dim])
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        self.d_embed = model_args['patch_embed']
        self.patchEmbed = PatchEmbed(feature_size = (model_args['num_filter'], self.feature_dim),
                                     patch_size = model_args['patch_size'],
                                     embed_dim = self.d_embed)
        self.seq_len = (model_args['num_filter']//model_args['patch_size'])*(self.feature_dim//model_args['patch_size'])
        self.EmbedReduce = EmbedReduce(current_len = self.seq_len, seq_size = model_args['seq_size'])
        
        self.posEncode = PositionalEncoding(d_model = self.d_embed, dropout = model_args['drop_out'])
        
        self.transformer = Transformer(d_model = self.d_embed, 
                                       d_hidden = model_args['encoder_hidden'], 
                                       N = model_args['num_block'],
                                       h = model_args['num_head'],
                                       dropout = model_args['drop_out'])
        
        
        self.mlp = nn.Sequential(nn.LayerNorm(model_args['seq_size'][2]),
                                 nn.Linear(in_features = model_args['seq_size'][2], out_features = model_args['seq_size'][2]),
                                 nn.Linear(in_features = model_args['seq_size'][2], out_features = model_args['nb_classes']))
        
       
    def forward(self, x, y = None,is_test=False):
        batch = x.shape[0]
        len_seq = x.shape[1]
        x = x.view(batch,1,len_seq)
        
        x = self.Sinc_conv(x)    # Fixed sinc filters convolution
        x = F.max_pool1d(torch.abs(x), 3)
        x = self.lnorm1(x)
        x = self.leaky_relu(x)
        
        x = self.patchEmbed(x)
        x = self.EmbedReduce(x)
        
        x = self.posEncode(x)
        
        x = self.transformer(x).transpose(1,2)
        
        x = self.mlp(x.mean(dim=1))
        output=F.softmax(x,dim=1)
        return output