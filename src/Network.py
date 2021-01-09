import torch.nn as nn
import numpy as np
import torch
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, offset=0):
        x = x + self.pe[offset:x.size(0)+offset, :]
        return self.dropout(x)


def future_mask(seq_length):
    future_mask = np.triu(np.ones((seq_length, seq_length)), k=1).astype('bool')
    return torch.from_numpy(future_mask)


class SAKTModel(nn.Module):
    def __init__(self, n_skill, max_seq=100, embed_dim=128, nlayers=2,dropout=0.1, nheads=8):
        super(SAKTModel, self).__init__()
        self.n_skill = n_skill
        self.embed_dim = embed_dim
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        self.embedding = nn.Embedding(n_skill+1, embed_dim)
        self.correctness_embedding = nn.Embedding(2, embed_dim)
        self.explantion_embedding = nn.Embedding(3, embed_dim)
        self.community_embedding = nn.Embedding(6, embed_dim)
        self.tag_embedding = nn.Linear(188, embed_dim)
        #self.pos_embedding = nn.Embedding(max_seq, embed_dim)
        self.et_embedding = nn.Linear(1, embed_dim)
        self.ts_embedding = nn.Linear(1, embed_dim)
        #self.e_embedding = nn.Embedding(n_skill+1, embed_dim)
        self.layer_normal = nn.LayerNorm(embed_dim)
        encoder_layers = nn.TransformerEncoderLayer(embed_dim, nheads, embed_dim*4, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.nheads = nheads
        self.pred = nn.Linear(embed_dim, 1)

    def forward(self, x, xa, et, ts, pq, att_mask, mask, community, tags):
        device = x.device
        correctness=self.correctness_embedding(xa)
        et = self.et_embedding(et.unsqueeze(-1)/300)
        ts = self.ts_embedding(ts.unsqueeze(-1)/1440)
        pq = self.explantion_embedding(pq)
        x = self.embedding(x)
        community = self.community_embedding(community)
        tags = self.tag_embedding(tags)

        x = x + et + ts + pq + correctness + community + tags

        x = x.permute(1, 0, 2)
        x = self.pos_encoder(x)
        x = self.layer_normal(x)

        att_mask = future_mask(x.size(0)).to(device)
        #att_mask = att_mask.expand(self.nheads, *att_mask.shape).transpose(1,0).reshape(-1,*att_mask.shape[1:])
        x = self.transformer_encoder(x, mask=att_mask, src_key_padding_mask=mask)

        x = x.permute(1, 0, 2)



        output = self.pred(x)

        return output.squeeze(-1)
