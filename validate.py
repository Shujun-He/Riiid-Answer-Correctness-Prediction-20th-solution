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
from Dataset import *
from Network import *
from Functions import *

try:
    #from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")
from ranger import Ranger
import pickle
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0,1',  help='which gpu to use')
    parser.add_argument('--path', type=str, default='../../data', help='path of csv file with DNA sequences and labels')
    parser.add_argument('--epochs', type=int, default=120, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=2048, help='size of each batch during training')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight dacay used in optimizer')
    parser.add_argument('--save_freq', type=int, default=1, help='saving checkpoints per save_freq epochs')
    parser.add_argument('--dropout', type=float, default=.1, help='transformer dropout')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--nfolds', type=int, default=5, help='number of cross validation folds')
    parser.add_argument('--fold', type=int, default=0, help='which fold to train')
    parser.add_argument('--val_freq', type=int, default=1, help='which fold to train')
    parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('--expansion', type=int, default=64, help='number of expansion pixels')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient_accumulation_steps')
    parser.add_argument('--max_seq', type=int, default=128, help='max_seq')
    parser.add_argument('--embed_dim', type=int, default=128, help='batch_size')
    #parser.add_argument('--batch_size', type=int, default=2048, help='batch_size')
    parser.add_argument('--nlayers', type=int, default=1, help='nlayers')
    opts = parser.parse_args()
    return opts

args=get_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')# device = torch.device("cpu")


question_cluster=pd.read_csv(os.path.join(args.path,'question_cmnts.csv'))

question_df=pd.read_csv(os.path.join(args.path,'questions.csv'))
possible_tags=[]
for i, tags in enumerate(question_df.tags):
    try:
        tags=tags.split()
        for tag in tags:
            tag=int(tag)
            if tag not in possible_tags:
                possible_tags.append(tag)
    except:
        pass

tag_encoding=np.zeros((len(question_df),len(possible_tags)))
for i, tags in enumerate(question_df.tags):
    try:
        tags=tags.split()
        for tag in tags:
            tag=int(tag)
            tag_encoding[i,tag]=1
    except:

        #exit()
        #print(i)
        pass#exit()
# try:
t=time.time()
with open(os.path.join(args.path,'group_w_lag_time.p'),'rb') as f:
    group=pickle.load(f)
n_skill = 13523
print(f"time taken to read group: {time.time()-t} sec")
print("###Group loaded###")

from sklearn.model_selection import KFold

kf = KFold(n_splits=args.nfolds,random_state=2020,shuffle=True)

train=group.iloc[list(kf.split(group))[args.fold][0]]
val=group.iloc[list(kf.split(group))[args.fold][1]]


val_dataset = SAKTDataset(val, question_cluster, tag_encoding, n_skill, args.max_seq, train=False)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size*2, shuffle=True, num_workers=args.workers)
del val

criterion = nn.BCEWithLogitsLoss()

# MODELS=[]
# for i in range(3):
#
model = SAKTModel(n_skill, embed_dim=args.embed_dim,
                  max_seq=args.max_seq, nlayers=args.nlayers,
                  dropout=args.dropout).to(device)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.99, weight_decay=0.005)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# Initialization
opt_level = 'O1'
model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
model = nn.DataParallel(model)


pytorch_total_params = sum(p.numel() for p in model.parameters())
print('Total number of paramters: {}'.format(pytorch_total_params))
#model.load_state_dict(torch.load("models/model1_epoch60.pth"))
model.load_state_dict(torch.load("models/model1.pth"))
#model.load_state_dict(torch.load("checkpoints_fold0/epoch{}.ckpt".format(i)))
model.eval()
#     MODELS.append(model)
#
# dict=MODELS[0].module.state_dict()
# for key in dict:
#     for i in range(1,len(MODELS)):
#         dict[key]=dict[key]+MODELS[i].module.state_dict()[key]
#
#     dict[key]=dict[key]/float(len(MODELS))
#
# MODELS[0].module.load_state_dict(dict)
# model=MODELS[0]

val_loss, val_acc, val_auc = full_validation(model, val_dataloader, criterion, device, 200)
#val_loss, val_acc, val_auc = full_validation(model, val_dataloader, criterion, device, len(val_dataloader))
print("val_loss - {:.2f} acc - {:.3f} auc - {:.3f}".format(val_loss, val_acc, val_auc))

with open("validation.txt",'w+') as f:
    f.write("val_loss - {:.3f} acc - {:.4f} auc - {:.4f}".format(val_loss, val_acc, val_auc))
