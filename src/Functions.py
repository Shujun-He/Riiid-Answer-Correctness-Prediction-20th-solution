import numpy as np
import pandas as pd
import gc
import random
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import os
import time


try:
    #from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

def train_epoch(model, train_iterator, optim, criterion, device="cpu", lr_schedule=None, steps_per_epoch=3200):
    model.train()
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    train_loss = 0
    num_corrects = 0
    num_total = 0
    #labels = []
    #outs = []
    loss_weight=torch.tensor(np.arange(0,1,1/(129-1))).float().unsqueeze(0).cuda()
    #tbar = tqdm(range(steps_per_epoch))
    t=time.time()
    #steps_per_epoch=len(train_iterator)
    for step,item in enumerate(train_iterator):
        if lr_schedule is not None:
            lr_schedule.step()
        x = item[0].to(device).long()
        label = item[1].to(device).float()
        xa = item[2].to(device).long()
        et = item[3].to(device).float()
        pq = item[4].to(device).long()
        lt = item[5].to(device).long()
        attention_mask = item[6].to(device).bool()
        mask = item[7].to(device).bool()
        community = item[8].to(device).long()
        tags = item[9].to(device).float()
        target_mask = (x != 13523)

        optim.zero_grad()
        output = model(x, xa, et, lt, pq, attention_mask, mask, community, tags)

        loss = criterion(output, label)*loss_weight
        loss = loss.mean()

        output = torch.masked_select(output, target_mask)
        label = torch.masked_select(label, target_mask)


        with amp.scale_loss(loss, optim) as scaled_loss:
            scaled_loss.backward()
        optim.step()
        train_loss+=loss.item()
        pred = (torch.sigmoid(output) >= 0.5).long()

        num_corrects += (pred == label).sum().item()
        num_total += len(label)

        print ("Step [{}/{}] Loss: {:.3f} Time: {:.1f}"
                           .format(step+1, steps_per_epoch, train_loss/(step+1), time.time()-t),end='\r',flush=True) #total_loss/(step+1)
        if (step+1)>steps_per_epoch:
            break
        #break
    print('')
    train_loss/=(step+1)
    acc = num_corrects / num_total

    return train_loss, acc
def val_epoch(model, val_iterator, criterion, device="cpu",steps_per_epoch=200):
    model.eval()

    train_loss = []
    num_corrects = 0
    num_total = 0
    labels = []
    outs = []

    #steps_per_epoch=len(val_iterator)
    t=time.time()
    for step,item in enumerate(val_iterator):
        x = item[0].to(device).long()
        label = item[1].to(device).float()
        xa = item[2].to(device).long()
        et = item[3].to(device).float()
        pq = item[4].to(device).long()
        lt = item[5].to(device).long()
        attention_mask = item[6].to(device).bool()
        mask = item[7].to(device).bool()
        community = item[8].to(device).long()
        tags = item[9].to(device).float()
        target_mask = (x != 13523)
        target_mask[:,:-32]=False

        with torch.no_grad():
            output = model(x, xa, et, lt, pq, attention_mask, mask, community, tags)

        output = torch.masked_select(output, target_mask)
        label = torch.masked_select(label, target_mask)

        loss = criterion(output, label)
        train_loss.append(loss.item())

        pred = (torch.sigmoid(output) >= 0.5).long()

        num_corrects += (pred == label).sum().item()
        num_total += len(label)

        labels.extend(label.view(-1).data.cpu().numpy())
        outs.extend(output.view(-1).data.cpu().numpy())

        print ("Step [{}/{}] Time: {:.1f}"
                           .format(step+1, steps_per_epoch, time.time()-t),end='\r',flush=True) #total_loss/(step+1)
        if (step+1)>steps_per_epoch:
            break
    print('')

    acc = num_corrects / num_total
    auc = roc_auc_score(labels, outs)
    loss = np.average(train_loss)

    return loss, acc, auc

def full_validation(model, val_iterator, criterion, device="cpu",steps_per_epoch=200):
    model.eval()

    train_loss = []
    num_corrects = 0
    num_total = 0
    labels = []
    outs = []

    steps_per_epoch=len(val_iterator)
    t=time.time()
    for step,item in enumerate(val_iterator):
        x = item[0].to(device).long()
        label = item[1].to(device).float()
        xa = item[2].to(device).long()
        et = item[3].to(device).float()
        pq = item[4].to(device).long()
        lt = item[5].to(device).long()
        attention_mask = item[6].to(device).bool()
        mask = item[7].to(device).bool()
        community = item[8].to(device).long()
        tags = item[9].to(device).float()
        target_mask = (x != 13523)
        #print(target_mask)
        target_mask[:,:-1]=False
        #print(target_mask.shape)
        #print((target_mask==False).sum())
        #exit()

        with torch.no_grad():
            output = model(x, xa, et, lt, pq, attention_mask, mask, community, tags)

        output = torch.masked_select(output, target_mask)
        label = torch.masked_select(label, target_mask)

        loss = criterion(output, label)
        train_loss.append(loss.item())

        pred = (torch.sigmoid(output) >= 0.5).long()

        num_corrects += (pred == label).sum().item()
        num_total += len(label)

        labels.extend(label.view(-1).data.cpu().numpy())
        outs.extend(output.view(-1).data.cpu().numpy())

        print ("Step [{}/{}] Time: {:.1f}"
                           .format(step+1, steps_per_epoch, time.time()-t),end='\r',flush=True) #total_loss/(step+1)
        if (step+1)>steps_per_epoch:
            break
    print('')

    acc = num_corrects / num_total
    auc = roc_auc_score(labels, outs)
    loss = np.average(train_loss)

    return loss, acc, auc



def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        lr=param_group['lr']
    return lr

class lr_AIAYN():
    '''
    Learning rate scheduler from the paper:
    Attention is All You Need
    '''
    def __init__(self,optimizer,d_model,warmup_steps=4000,factor=1):
        self.optimizer=optimizer
        self.d_model=d_model
        self.warmup_steps=warmup_steps
        self.step_num=0
        self.factor=factor

    def step(self):
        self.step_num+=1
        lr=self.d_model**-0.5*np.min([self.step_num**-0.5,
                                      self.step_num*self.warmup_steps**-1.5])*self.factor
        update_lr(self.optimizer,lr)
        return lr
