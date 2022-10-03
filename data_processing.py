import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split
import collections
import numpy as np
import librosa, librosa.display
import scipy
from scipy.io import wavfile
import soundfile as sf
import sys 
import os
import math
import copy


AudioFile = collections.namedtuple('AudioFile',
    ['file_name','path','label', 'key'])


class ADDDataset(Dataset):
    def __init__(self, data_path=None, label_path=None,transform=None,
                 is_train=True,is_eval=False,feature=None,track=None):
        self.data_path = data_path
        self.label_path = label_path
        self.transform = transform
        self.track = track
        self.feature = feature
        
        self.dset_name = 'eval' if is_eval else 'train' if is_train else 'dev'
        if (self.dset_name == 'eval'):
            cache_fname = 'cache_ADD_{}_{}.npy'.format(self.dset_name,self.track)
            self.cache_fname = os.path.join("/home/menglu/123/Deepfake/built", cache_fname)
        elif self.feature == None:
            cache_fname = 'cache_ADD_{}.npy'.format(self.dset_name)
            self.cache_fname = os.path.join("/home/menglu/123/Deepfake/built", cache_fname)
        else:   
            cache_fname = 'cache_ADD_{}_{}.npy'.format(self.dset_name, self.feature)
            self.cache_fname = os.path.join("/home/menglu/123/Deepfake/built", cache_fname)
               
            
        if os.path.exists(self.cache_fname):
            self.data_x, self.data_y, self.files_meta = torch.load(self.cache_fname)
            print('Dataset loaded from cache', self.cache_fname)
        else: 
            self.files_meta = self.parse_protocols_file(self.label_path)
            data = list(map(self.read_file, self.files_meta))
            self.data_x, self.data_y= map(list, zip(*data))
            if self.transform:
                self.data_x = Parallel(n_jobs=5, prefer='threads')(delayed(self.transform)(x) for x in self.data_x)                          
            torch.save((self.data_x, self.data_y, self.files_meta), self.cache_fname)
        
    def __len__(self):
        self.length = len(self.data_x)
        return self.length
   
    def __getitem__(self, idx):
        x = self.data_x[idx]
        y = self.data_y[idx]
        return x, y
    
    def read_file(self, meta):   
        data_x, sample_rate = librosa.load(meta.path,sr=16000)       
        data_y = meta.key
        return data_x, float(data_y)
      
    def parse_line(self,line):
        tokens = line.strip().split(' ')
        audio_path=os.path.join(self.data_path, tokens[0]).replace('\\','/')
        return AudioFile(file_name=tokens[0], path = audio_path,
                         label=tokens[1], key=int(tokens[1] == 'genuine'))
        
    def parse_protocols_file(self, label_path):
        lines = open(label_path).readlines()
        files_meta = map(self.parse_line, lines)
        return list(files_meta)