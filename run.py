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

#get tags
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

#exit()

# try:
t=time.time()
with open(os.path.join(args.path,'group_w_lag_time.p'),'rb') as f:
    group=pickle.load(f)
n_skill = 13523
print(f"time taken to read group: {time.time()-t} sec")
print("###Group loaded###")
# except:
#     dtype = {'timestamp': 'int64',
#              'user_id': 'int32' ,
#              'content_id': 'int16',
#              'content_type_id': 'int8',
#              'answered_correctly':'int8',
#              'prior_question_elapsed_time':'float64',
#              'prior_question_had_explanation':'int'}
#
#     t=time.time()
#     train_df = pd.read_csv(os.path.join(args.path,'train.csv'), usecols=[1, 2, 3, 4, 7, 8, 9], dtype=dtype)
#     train_df['prior_question_elapsed_time']=train_df['prior_question_elapsed_time'].fillna(0)
#     #train_df['prior_question_elapsed_time']=train_df['prior_question_elapsed_time']//1000
#     train_df['prior_question_elapsed_time']=train_df['prior_question_elapsed_time'].values.astype('int32')
#     train_df['prior_question_had_explanation']=train_df['prior_question_had_explanation'].fillna(2)
#     #exit()
#     print(f"time taken to read csv: {time.time()-t} sec")
#
#     train_df = train_df[train_df.content_type_id == False]
#
#     #arrange by timestamp
#     train_df = train_df.sort_values(['timestamp'], ascending=True).reset_index(drop = True)
#
#
#     #preprocess
#     skills = train_df["content_id"].unique()
#     n_skill = len(skills)
#     print("number skills", len(skills))
#
#     group = train_df[['user_id', 'content_id', 'answered_correctly','prior_question_elapsed_time','prior_question_had_explanation','timestamp']].groupby('user_id').apply(lambda r: (
#                 r['content_id'].values,
#                 r['answered_correctly'].values,
#                 r['prior_question_elapsed_time'].values,
#                 r['prior_question_had_explanation'].values,
#                 r['timestamp'].values))
#
#     del train_df
#     with open(os.path.join(args.path,'group_raw2.p'),'wb+') as f:
#         pickle.dump(group,f)
#
#     print("###Group loaded and saved###")

#exit()

from sklearn.model_selection import KFold

kf = KFold(n_splits=args.nfolds,random_state=2020,shuffle=True)

train=group.iloc[list(kf.split(group))[args.fold][0]]
val=group.iloc[list(kf.split(group))[args.fold][1]]

train_dataset = SAKTDataset(train, question_cluster, tag_encoding, n_skill, args.max_seq)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
del train

val_dataset = SAKTDataset(val, question_cluster, tag_encoding, n_skill, args.max_seq, train=False)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size*2, shuffle=True, num_workers=args.workers)
del val

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

#initialize model
model = SAKTModel(n_skill, embed_dim=args.embed_dim,
                  max_seq=args.max_seq, nlayers=args.nlayers,
                  dropout=args.dropout).to(device)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.99, weight_decay=0.005)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#optimizer = Ranger(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()


opt_level = 'O1'
model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

model=nn.DataParallel(model)

#model.load_state_dict(torch.load('models/model1_epoch6.pth'))

from Logger import *
os.system('mkdir logs')
logger=CSVLogger(['epoch','train_loss','train_acc','val_loss','val_acc','val_auc'],f'logs/log_fold{args.fold}.csv')

os.system('mkdir models')

over_fit = 0
last_auc = 0
best_auc=0
cos_epoch=int(args.epochs*0.75)
lr_schedule=lr_AIAYN(optimizer,args.embed_dim)
for epoch in range(args.epochs):

    train_loss, train_acc = train_epoch(model, train_dataloader, optimizer, criterion, device, lr_schedule)
    print("epoch - {} train_loss - {:.2f} acc - {:.3f}".format(epoch, train_loss, train_acc))

    val_loss, val_acc, val_auc = val_epoch(model, val_dataloader, criterion, device)
    print("epoch - {} val_loss - {:.2f} acc - {:.3f} auc - {:.3f}".format(epoch, val_loss, val_acc, val_auc))

    logger.log([epoch+1,train_loss,train_acc,val_loss,val_acc,val_auc])
    if val_auc > last_auc:
        last_auc = val_auc
        over_fit = 0
    else:
        over_fit += 1

    if val_auc > best_auc:
        best_auc=val_auc
        torch.save(model.state_dict(),f'models/model{args.fold}.pth')
    ##torch.save(model.state_dict(),f'models/model{args.fold}_epoch{epoch+1}.pth')

#     if (epoch+1)%80==0:
#         update_lr(optimizer,get_lr(optimizer)*1e-1)
