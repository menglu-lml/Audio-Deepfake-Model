from torch.utils.data import DataLoader, Dataset, random_split
import torchaudio
import numpy as np
import glob
import collections
from joblib import Parallel, delayed

import torch
from torch import Tensor
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from tensorboardX import SummaryWriter

import sys 
import os
import math
import yaml
import copy
from scipy import signal
import wave
import cv2
from sklearn import svm

from data_processing import ADDDataset
from model import Box



######### helper functions ############

## For data
def pad(x, max_len=48000):
    
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    
    return padded_x


## For Optimizer
def rate(step, model_size, factor, warmup):
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


def train_epoch(data_loader, model, lr, optim, device, scheduler = None ):
    running_loss = 0
    num_correct = 0.0
    num_total = 0.0
    ii = 0
    model.train()
    weight = torch.FloatTensor([1.0, 9.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    for batch_x, batch_y in tqdm(data_loader):
    #for batch_x, batch_y in data_loader:
       
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x,batch_y)
        batch_loss = criterion(batch_out, batch_y)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
        running_loss += (batch_loss.item() * batch_size)
        optim.zero_grad()
        batch_loss.backward()
        optim.step()
        if scheduler !=None:
            scheduler.step()
       
    running_loss /= num_total
    train_accuracy = (num_correct/num_total)*100
    return running_loss, train_accuracy


def evaluate_accuracy(data_loader, model, device):
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    for batch_x, batch_y in tqdm(data_loader):
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x,batch_y)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
    return 100 * (num_correct / num_total)





# Dataloader TRAINING
database_path = "/home/menglu/123/Dataset/ADD2022/ADD_train_dev/train"
label_path = "/home/menglu/123/Dataset/ADD2022/label/train_label.txt"
transform = transforms.Compose([
    lambda x: pad(x),
    lambda x: Tensor(x)
])
batch_size = 32

train_set = ADDDataset(data_path=database_path,label_path=label_path,is_train=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,drop_last=True)

# Dataloader  VALIDATION
dev_data_path = "/home/menglu/123/Dataset/ADD2022/ADD_train_dev/dev"
dev_label_path = "/home/menglu/123/Dataset/ADD2022/label/dev_label.txt"

dev_set = ADDDataset(data_path = dev_data_path,label_path = dev_label_path,is_train=False, transform=transform)
dev_loader = DataLoader(dev_set, batch_size=batch_size, shuffle=True)



# TRAINING Preparation
np.random.seed(1234)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# GPU device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Parameter
config = yaml.safe_load(open('model_config.yaml'))
lr = config['lr']
warmup = config['warmup']
num_epochs = config['epoch']

d_model = config['model']['patch_embed']
num_filter = config['model']['num_filter']
num_block = config['model']['num_block']
num_head = config['model']['num_head']

# Model Initialization
model = Box(config['model'],device).to(device)

nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
print(nb_params)

# Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), 
                             lr = lr, betas=(0.9, 0.98), eps=1e-9)
lr_scheduler = LambdaLR(optimizer=optimizer,
                        lr_lambda=lambda step: rate(step, d_model, factor=1, warmup=config["warmup"]),)


model_tag = 'Top_path_{}_{}_{}_{}_{}_{}'.format(batch_size,d_model, num_filter, num_block, num_head,lr)

model_save_path = os.path.join('/home/menglu/123/Deepfake/built', model_tag)
os.makedirs(model_save_path)



writer = SummaryWriter('logs/{}'.format(model_tag))
best_acc = 40
for epoch in range(num_epochs):
    running_loss, train_accuracy = train_epoch(train_loader,model, lr,optimizer, device, lr_scheduler)
    valid_accuracy = evaluate_accuracy(dev_loader, model, device)
    writer.add_scalar('train_accuracy', train_accuracy, epoch)
    writer.add_scalar('valid_accuracy', valid_accuracy, epoch)
    writer.add_scalar('loss', running_loss, epoch)
    print('\n{} - {} - {:.4f} - {:.4f}'.format(epoch,
                                               running_loss, train_accuracy, valid_accuracy))
    best_acc = max(valid_accuracy, best_acc)
    torch.save(model.state_dict(), os.path.join(model_save_path, 'epoch_{}.pth'.format(epoch)))

writer.close()