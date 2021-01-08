import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm

def task_mask(tasks):
    seq_length=len(tasks)
    future_mask = np.triu(np.ones((seq_length, seq_length)), k=1).astype('bool')
    container_mask= np.ones((seq_length, seq_length))
    container_mask=(container_mask*tasks.reshape(1,-1))==(container_mask*tasks.reshape(-1,1))
    #comparison_mask=np.ones((seq_length, seq_length))*tasks.reshape(-1,1)
    #mask=future_mask(task)
    future_mask=future_mask+container_mask
    np.fill_diagonal(future_mask,0)
    return future_mask


                # r['content_id'].values,
                # r['answered_correctly'].values,
                # r['prior_question_elapsed_time'].values,
                # r['prior_question_had_explanation'].values,
                # r['timestamp'].values,
                # r['task_container_id'].values,))

class SAKTDataset(Dataset):
    def __init__(self, group, question_cluster, tag_encoding, n_skill, max_seq=160, train=True): #HDKIM 100
        super(SAKTDataset, self).__init__()
        self.max_seq = max_seq
        self.n_skill = n_skill
        self.samples = group
        self.question_commnity=np.append(question_cluster.community.values,[5])
        self.tag_encoding=np.concatenate([tag_encoding,np.zeros((1,188))],0)
        self.train=train
#         self.user_ids = [x for x in group.index]
        self.user_ids = []
        for user_id in group.index:
            item = group[user_id]
            if len(item[0]) > 2: #HDKIM 10
                self.user_ids.append(user_id)

            #HDKIM Memory reduction
            #if len(q)>self.max_seq:
            #    group[user_id] = (q[-self.max_seq:],qa[-self.max_seq:])
        self._get_indices()


    def _get_indices(self):
        self.indices=[]
        for i, user_id in tqdm(enumerate(self.user_ids)):
            item = self.samples[user_id]
            # if len(q)%self.max_seq==0:
            #     n_seg=len(q)//self.max_seq
            # else:
            #     n_seg=len(q)//self.max_seq+1
            for j in range(1,len(item[0])):
            #for j in range(1,512):
                self.indices.append([i,j])


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i, user_index=self.indices[index]
        user_id = self.user_ids[i]
        q_, qa_, et_, pq_, ts_, tasks_, lt_ = self.samples[user_id]
        # print(ts_)
        # exit()
        #q_=q_+1
        seq_len = len(q_)

        q = np.zeros(self.max_seq, dtype=int)
        q[:]=13523
        qa = np.zeros(self.max_seq, dtype=int)
        et = np.zeros(self.max_seq, dtype=int)
        pq = np.zeros(self.max_seq, dtype=int)
        ts = np.zeros(self.max_seq, dtype=int)
        tasks = np.zeros(self.max_seq, dtype=int)
        lt = np.zeros(self.max_seq, dtype=int)

        if user_index > self.max_seq:
            #HDKIM
            # if self.train:
            #     random_sampling=np.clip(np.random.randint(self.max_seq),0,seq_len-user_index-self.max_seq)
            # else:
            #     random_sampling=0
            random_sampling=0
            q[:] = q_[user_index-self.max_seq:user_index]
            qa[:] = qa_[user_index-self.max_seq:user_index]
            et[:] = et_[user_index-self.max_seq:user_index]
            pq[:] = pq_[user_index-self.max_seq:user_index]
            ts[:] = ts_[user_index-self.max_seq:user_index]
            tasks[:] = tasks_[user_index-self.max_seq:user_index]
            lt[:] = lt_[user_index-self.max_seq:user_index]
        else:
            # if (seq_len-user_index)<=0:
            #     print('shit')
            q[-user_index:] = q_[:user_index]
            qa[-user_index:] = qa_[:user_index]
            et[-user_index:] = et_[:user_index]
            pq[-user_index:] = pq_[:user_index]
            ts[-user_index:] = ts_[:user_index]
            tasks[-user_index:] = tasks_[:user_index]
            lt[-user_index:] = lt_[:user_index]

        target_id = q[1:]
        et = et[1:]//1000
        ts = (ts[1:]-ts[:-1])/1000
        #print(ts)
        for i in range(len(ts)-1):
            if tasks[i+1]==tasks[i+2]:
                ts[i+1]=ts[i]
        # print(ts)
        # exit()
        #lt = lt[1:]/1000
        et=np.clip(et,0,300)
        ts=np.clip(ts,0,1440)
        #lt=np.around((np.clip(lt,-50,550)+50)/4).astype('int')
        #print(ts)
        #print(ts[2:12]-ts[1:11])
        #exit()
        pq = pq[1:]
        label = qa[1:]
        xa = qa[:-1].copy()
        #x += (qa[:-1] == 1) * self.n_skill
        mask=(target_id==13523)
        mask[0]=False

        #attention_mask=task_mask(tasks[1:])
        #attention_mask[:,0]=False
        attention_mask = 0

        community=self.question_commnity[target_id]
        tags=self.tag_encoding[target_id]
        return target_id, label, xa, et, pq, ts, attention_mask, mask, community, tags
