"""Recurrent Mark Density Estimator
"""
import sys
from collections import defaultdict
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
from tqdm import tqdm

from ..explain.integrated_gradient import batch_integrated_gradient
from ..utils.misc import AverageMeter
from ..utils.torch import ResidualLayer, generate_sequence_mask, set_eval_mode

import math



def subsequent_mask(size):
    "mask out subsequent positions"
    atten_shape = (1,size,size)
    # np.triu: Return a copy of a matrix with the elements below the k-th diagonal zeroed.
    mask = np.triu(np.ones(atten_shape),k=1).astype('uint8')
    aaa = torch.from_numpy(mask) == 0
    return aaa


class MaskBatch():
    "object for holding a batch of data with mask during training"
    def __init__(self,src,pad, device):
        self.src = src
        self.src_mask = self.make_std_mask(self.src, pad, device)

    @staticmethod
    def make_std_mask(tgt,pad,device):
        "create a mask to hide padding and future input"
        # torch.cuda.set_device(device)
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & torch.transpose(tgt_mask,1,2) & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)).to(device)
        return tgt_mask


# class GELU(nn.Module):
#     """
#     Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
#     """

#     def forward(self, x):
#         return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class EventSeqDataset(Dataset):
    """Construct a dataset for store event sequences.

    Args:
        event_seqs (list of list of 2-tuples):
    """

    def __init__(self, event_seqs, min_length=1, sort_by_length=False):

        self.min_length = min_length
        self._event_seqs = [
            torch.FloatTensor(seq)
            for seq in event_seqs
            if len(seq) >= min_length
        ]
        if sort_by_length:
            self._event_seqs = sorted(self._event_seqs, key=lambda x: -len(x))

    def __len__(self):
        return len(self._event_seqs)

    def __getitem__(self, i):
        # TODO: can instead compute the elapsed time between events
        return self._event_seqs[i]

    @staticmethod
    def collate_fn(X):
        return nn.utils.rnn.pad_sequence(X, batch_first=True)

class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """
    def forward(self, query, key, value, mask=None, dropout=None):

        # scores = torch.matmul(query, key.transpose(-2, -1)) \
        #          / math.sqrt(query.size(-1))
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0., -1e9)
            # print('query and mask and score', query.shape, mask.shape, scores.shape)
        # print('mask',mask.shape, mask)

        p_attn = F.softmax(scores, dim=-1)
        if mask is not None:
            p_attn = p_attn.masked_fill(mask == 0., 0.)
        # print('scores after', p_attn)

        # if dropout is not None:
        #     p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    """
    Take in models size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_model = d_model
        self.d_k = d_model // h
        self.h = h
        # self.gelu = GELU()

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model, bias=True) for _ in range(3)])
        # self.linear_layers1 = nn.ModuleList([nn.Linear(d_model, d_model, bias=True) for _ in range(3)])
        # self.linear_layers2 = nn.ModuleList([nn.Linear(d_model, d_model, bias=True) for _ in range(3)])

        self.output_alpha = nn.Linear(d_model, d_model, bias=True)

        # self.alpha_layer = nn.Sequential(
        #     nn.Linear(int(self.d_k*self.h/2), self.d_k, bias=True),
        #     self.gelu
        # )
        self.alpha_layer = nn.Sequential(
            nn.Linear(int(self.d_k*self.h/2), self.d_k, bias=True)
            # ,nn.Softplus(beta=10.0)
            , nn.Sigmoid()
            # ,nn.Linear(self.d_k, self.d_k, bias=True)
        )

        self.gamma_layer = nn.Sequential(
            nn.Linear(int(self.d_k*self.h/2), self.d_k, bias=True)
            # ,nn.Softplus(beta=10.0)
            , nn.Sigmoid()
        )

        self.mu_layer = nn.Sequential(
            nn.Linear(self.d_model, self.d_k, bias=True)
            # ,nn.Softplus(beta=10.0)
            , nn.Sigmoid()
        )

        # self.alpha_layer = nn.Sequential(
        #     nn.Linear(int(self.d_k*self.h/2), self.d_k, bias=True)
        #     ,nn.Sigmoid()
        # )

        # self.gamma_layer = nn.Sequential(
        #     nn.Linear(int(self.d_k*self.h/2), self.d_k, bias=True)
        #     ,nn.Sigmoid()
        # )

        # self.mu_layer = nn.Sequential(
        #     nn.Linear(self.d_model, self.d_k, bias=True)
        #     ,nn.Sigmoid()
        # )

        # self.mu_layer = nn.Sequential(
        #     nn.Linear(self.d_model, self.d_k, bias=True),
        #     self.gelu
        # )

        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # the same mask applies to all heads
            # unsqueeze Returns a new tensor with a dimension of size one
            # inserted at the specified position.
            mask = mask.unsqueeze(1)
        # print('mask',mask.shape)
        # batch_size = query.size(0)
        batch_size, T = query.size()[:2]
        # print('batch_size, T',batch_size, T)

        # # test multi layer
        # query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        #                      for l, x in zip(self.linear_layers1, (query, key, value))]
        # # print('query shape2', query.shape, query)

        # # 2) Apply attention on all the projected vectors in batch.
        # v_mu0, _ = self.attention.forward(query, key, value, mask=mask, dropout=self.dropout)
        # # print('v_mu0.shape', v_mu0.shape, v_mu0)
        # v_mu0 = v_mu0.transpose(1, 2).reshape(batch_size, -1, self.h *self.d_k)

        # query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        #                      for l, x in zip(self.linear_layers2, (v_mu0, v_mu0, v_mu0))]
        # # print('query shape3', query.shape , query)
        # v_mu, attn = self.attention.forward(query, key, value, mask=mask, dropout=self.dropout)
        # # print('v_mu.shape', v_mu.shape, v_mu)
        # # print('attn.shape', attn.shape, attn)
        # # sys.exit()

        # # test multi layer

        #original
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        # print('query, key, value shape', query.shape)

        # 2) Apply attention on all the projected vectors in batch.
        v_mu, attn = self.attention.forward(query, key, value, mask=mask, dropout=self.dropout)
        # print('v_mu and attn shape',v_mu.shape, attn.shape)
        #original

        # 3) "Concat" using a view and apply a final linear.
        attn_repeat = torch.repeat_interleave(attn.unsqueeze(-1),self.d_k, -1)
        # print('attn_repeat.shape',attn_repeat.shape)
        # value_repeat = value.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        value_repeat = torch.repeat_interleave(value.unsqueeze(2),T,2)
        # print('value_repeat.shape',value_repeat.shape)

        mask = mask.permute((0, 2, 3, 1))
        # print('mask',mask.shape,mask)
        # sys.exit()
        v_alpha = 8 * self.alpha_layer((attn_repeat[:,:int(self.h/2)]*value_repeat[:,:int(self.h/2)]).contiguous().view(batch_size, T, T, -1)) # (B x L x L x K)
        v_alpha = v_alpha.masked_fill(mask == 0., 0.)
        v_gamma = 8 * self.gamma_layer((attn_repeat[:,int(self.h/2):]*value_repeat[:,int(self.h/2):]).contiguous().view(batch_size, T, T, -1)) + 0.1 # (B x L x L x K)
        v_gamma = v_gamma.masked_fill(mask == 0., 0.)
        v_mu = 5 * self.mu_layer(v_mu.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k))
        # print("v_alpha.shape,v_gamma.shape, v_mu.shape", v_alpha.shape,v_gamma.shape, v_mu.shape) #B, L, L, K
        # print(v_alpha[...,0])

        return v_mu, v_alpha, v_gamma

class InstancewiseSelfAttentiveHawkesProcesses(nn.Module):
    def __init__(
        self,
        n_types: int,
        embedding_dim: int = 59, #32,
        hidden_size: int = 60, #32,
        dropout: float = 0.0,
        num_head: int = 6,
        **kwargs,
    ):
        super().__init__()
        self.n_types = n_types

        self.embed = nn.Linear(n_types, embedding_dim, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.m_dt = nn.ReLU()
        self.softplus_state_decay = nn.Softplus(beta=10.0)

        self.multiheadattention = MultiHeadedAttention(h=num_head, d_model=hidden_size) #self.d_model
        # self.seq_encoder = getattr(nn, rnn)(
        #     input_size=embedding_dim + 1,
        #     hidden_size=hidden_size,
        #     batch_first=True,
        #     dropout=dropout,
        # )

        # self.bases = [Unity()]
        # if basis_type == "equal":
        #     loc, scale = [], []
        #     for i in range(n_bases):
        #         loc.append(i * max_mean / (n_bases - 1))
        #         scale.append(max_mean / (n_bases - 1))
        # elif basis_type == "dyadic":
        #     L = max_mean / 2 ** (n_bases - 1)
        #     loc, scale = [0], [L / 3]
        #     for i in range(1, n_bases):
        #         loc.append(L)
        #         scale.append(L / 3)
        #         L *= 2
        # else:
        #     raise ValueError(f"unrecognized basis_type={basis_type}")

        # self.bases.append(Normal(loc=loc, scale=scale))
        # self.bases = nn.ModuleList(self.bases)

        # self.shallow_net = ResidualLayer(hidden_size, n_types * (n_bases + 1))

    def state_decay(self, v_mu, v_alpha, v_gamma, dt_arr): # (B, L-1, K), (B, L-1, L-1, K) , (B, L-1, L-1, K), (B, L-1, L-1, 1)
        # * element-wise product
        # cell_t = torch.tanh(self.softplus_state_decay(v_mu + torch.sum(v_alpha * torch.exp(-v_gamma * dt_arr),-3)))
        # cell_t = self.softplus_state_decay(v_mu + torch.sum(v_alpha * torch.exp(-v_gamma * dt_arr),-3))
        cell_t = torch.tanh(v_mu + torch.sum(v_alpha * v_gamma * torch.exp(-v_gamma * dt_arr),-3))
        # cell_t = torch.tanh(self.softplus_state_decay(v_mu + torch.sum(v_alpha * torch.exp(-v_gamma * dt_arr),-3)))
        return cell_t # (B, L-1, K)

    def forward(
        self, event_seqs, src_mask, onehot=False, target_type=-1
    ):
        """[summary]

        Args:
          event_seqs (Tensor): shape=[batch_size, T, 2]
            or [batch_size, T, 1 + n_types]. The last dimension
            denotes the timestamp and the type of an event, respectively.

          onehot (bool): whether the event types are represented by one-hot
            vetors.

          target_type (int): whether to only predict for a specific type

        Returns:
           log_intensities (Tensor): shape=[batch_size, T, n_types],
             log conditional intensities evaluated at each event for each type
             (i.e. starting at t1).
           weights (Tensor, optional): shape=[batch_size, T, n_types, n_bases],
             basis weights intensities evaluated at each previous event (i.e.,
             tarting at t0). Returned only when `need_weights` is `True`.

        """
        assert event_seqs.size(-1) == 1 + (
            self.n_types if onehot else 1
        ), event_seqs.size()

        batch_size, T = event_seqs.size()[:2]
        # print("batch_size, T", batch_size, T)

        # (0,t1, t2, ..., t_n)
        self.ts = F.pad(event_seqs[:, :, 0], (1, 0)) #(t+1)
        # print('ts', self.ts.shape, self.ts)
        # self.ts = event_seqs[:, :, 0]
        # (0, t1 - t0, ..., t_{n} - t_{n - 1})
        # dt = F.pad(self.ts[:, 1:] - self.ts[:, :-1], (1, 0))
        # (t1, t2 - t1, ..., t_{n} - t_{n - 1}) L-1
        dt = self.ts[:, 1:] - self.ts[:, :-1] #(t)
        # dt = self.m_dt(dt)
        # torch.set_printoptions(threshold=10000,edgeitems=100)
        # print('dt', dt.shape, dt)
        # (t1, t2 - t1, ..., t_{n - 1} - t_{n - 2})
        temp_feat = dt[:, :-1].unsqueeze(-1) #[64, 336, 1] t-1

        # (z_1, ..., z_{n - 1})
        if onehot:
            type_feat = self.embed(event_seqs[:, :-1, 1:])
        else:
            type_feat = self.embed(
                F.one_hot(event_seqs[:, :-1, 1].long(), self.n_types).float()
            )
        # print('one_hot', F.one_hot(event_seqs[:, :-1, 1].long()).shape, F.one_hot(event_seqs[:, :-1, 1].long()))
        # print('type_feat',type_feat.shape)
        # type_feat = F.pad(type_feat, (0, 0, 1, 0)) #[64, 336, 64]

        feat = torch.cat([temp_feat, type_feat], dim=-1) #[64, 336, 20]
        # print('feat', feat.shape)
        # print('temp_feat',temp_feat.shape)
        # print('type_feat',type_feat.shape)
        # print('feat',feat.shape)
        # sys.exit()
        # v_mu, v_alpha, v_gamma = self.multiheadattention(feat,feat,feat) # (B, L, K), (B, L, L, K) , (B, L, L, K)
        # return v_mu, v_alpha, v_gamma
        v_mu, v_alpha, v_gamma = self.multiheadattention.forward(feat,feat,feat, mask=src_mask) #

        return v_mu, v_alpha, v_gamma
        # return v_mu, v_alpha, v_gamma
        # history_emb = self.dropout(history_emb)

        # # [B, T, K or 1, R]
        # log_basis_weights = self.shallow_net(history_emb).view(
        #     batch_size, T, self.n_types, -1
        # )
        # if target_type >= 0:
        #     log_basis_weights = log_basis_weights[
        #         :, :, target_type : target_type + 1
        #     ]

        # # [B, T, 1, R]
        # basis_values = torch.cat(
        #     [basis.log_prob(dt[:, 1:, None]) for basis in self.bases], dim=2
        # ).unsqueeze(-2)

        # log_intensities = (log_basis_weights + basis_values).logsumexp(dim=-1)

        # if need_weights:
        #     return log_intensities, log_basis_weights
        # else:
        #     return log_intensities

    # def _eval_cumulants(self, batch, log_basis_weights):
    #     """Evaluate the cumulants (i.e., integral of CIFs at each location)
    #     """
    #     ts = batch[:, :, 0]
    #     # (t1 - t0, ..., t_n - t_{n - 1})
    #     dt = (ts - F.pad(ts[:, :-1], (1, 0))).unsqueeze(-1)
    #     # [B, T, R]
    #     integrals = torch.cat(
    #         [
    #             basis.cdf(dt) - basis.cdf(torch.zeros_like(dt))
    #             for basis in self.bases
    #         ],
    #         dim=-1,
    #     )
    #     cumulants = integrals.unsqueeze(2).mul(log_basis_weights.exp()).sum(-1)
    #     return cumulants

    def _eval_nll(
        self, event_seqs, src_mask, mask, v_mu, v_alpha, v_gamma, device=None, n_mc_samples = 20
    ):  
        n_batch = self.ts.size(0)
        n_times = self.ts.size(1) - 2

        dt_arr = torch.tril(torch.cdist(event_seqs[:, :, 0:1], event_seqs[:, :, 0:1], p=2))[:,1:,:-1] #(B, L-1, L-1)
        # print('dt_arr.shape',dt_arr.shape, dt_arr)
        dt_seq = torch.diagonal(dt_arr, offset=0, dim1=1, dim2=2) #(B, L-1)
        torch.set_printoptions(threshold=10000,edgeitems=100)
        # print('dt_seq', torch.max(dt_seq,1))
        # print('dt_seq.shape',dt_seq.shape,dt_seq)
        dt_meta = torch.tril(torch.repeat_interleave(torch.unsqueeze(dt_seq,-1),n_times,-1)).masked_fill(src_mask == 0., 0.) #(B, L-1, L-1)
        dt_offset = (dt_arr - dt_meta).masked_fill(src_mask == 0., 0.)
        # print('dt_meta.shape',dt_meta.shape,dt_meta)
        # print('dt_offset.shape',dt_offset.shape,dt_offset)
        # dt_offset = torch.max(dt_arr - dt_meta, torch.zeros(dt_meta.shape).to(device)) #check !!

        # print('dt_meta[0], dt_offset[0]', dt_meta[0], dt_offset[0])

        type_mask = F.one_hot(event_seqs[:, 1:, 1].long(), self.n_types).float()
        # print('type_mask',type_mask.shape, type_mask)


        
        cell_t = self.state_decay(v_mu, v_alpha, v_gamma, dt_arr[:,:,:,None]) #(B, L-1, K)
        log_intensities = cell_t.log()  # log intensities
        # seq_mask = seq_onehot_types[:, 1:]
        # seq_mask =  mask[:,1:] # check, should be seq mask for event type with shape (B, L-1, 1) [1:L]
        # print('log_intensities', log_intensities.shape, log_intensities)
        # log_sum = (log_intensities * type_mask).sum(dim=(2, 1)) #B
        # print((log_intensities * type_mask).sum(-1).shape, mask.shape)
        # print((log_intensities * type_mask).sum(-1))
        # print(mask)
        log_sum = (log_intensities * type_mask).sum(-1).masked_select(mask).sum() #B x L-1 -> B
        # print('log_sum', log_sum.shape, log_sum)

        taus = torch.rand(n_batch, n_times, n_times, 1, n_mc_samples).to(device)# self.process_dim replaced 1 (B,L-1,L-1,1,20)
        taus = dt_meta[:, :, :, None, None] * taus  # inter-event times samples) (B,L-1,L-1,1,20).
        # print('taus', taus.shape, taus)
        taus =  taus + dt_offset[:,:,:,None,None] #(B,L-1,L-1,1,20).
        # print('dt_offset', dt_offset.shape, dt_offset)
        # print("taus", torch.max(taus))
        # print('v_mu', torch.max(v_mu))
        # print('v_alpha', torch.max(v_alpha))
        # print('v_gamma', torch.max(v_gamma))

        cell_tau = self.state_decay(
            v_mu[:,:,:,None],
            v_alpha[:,:,:,:,None],
            v_gamma[:,:,:,:,None],
            taus) #(B,L-1, k, 20)
        # print("cell_tau", torch.max(cell_tau), torch.min(cell_tau))
        total_intens_samples = cell_tau.sum(dim=2) #sum over k (B,L-1, 20)
        # print('total_intens_samples',total_intens_samples.shape, total_intens_samples)
        partial_integrals = dt_seq * total_intens_samples.mean(dim=2)
        # print('dt_seq',dt_seq.shape,dt_seq)
        # print('partial_integrals', partial_integrals.shape, partial_integrals)
        partial_integrals = partial_integrals.masked_select(mask) #average samples (B,L-1)
        # print('mask',mask.shape,mask)
        # print('partial_integrals', partial_integrals.shape, partial_integrals)

        # print('total_intens_samples.mean(dim=2)', torch.max(total_intens_samples.mean(dim=2)))
        # print('partial_integrals',partial_integrals.shape)

        # print('dt_seq', torch.max(dt_seq), torch.min(dt_seq))
        # print('total_intens_samples',torch.max(total_intens_samples),torch.min(total_intens_samples))

        integral_ = partial_integrals.sum() #B

        res = torch.sum(- log_sum + integral_)/n_batch
        log_sum = torch.sum(- log_sum)/n_batch
        integral = torch.sum(integral_)/n_batch
        # print('log_sum',log_sum.shape)
        # print(log_sum, integral_)
        # sys.exit()

        return res, integral, log_sum

    def _eval_acc(self, batch, intensities, mask):
        types_pred = intensities.argmax(dim=-1).masked_select(mask)
        types_true = batch[:, :, 1].long().masked_select(mask)
        return (types_pred == types_true).float().mean()

    def train_epoch(
        self,
        train_dataloader,
        optim,
        valid_dataloader=None,
        device=None,
        **kwargs,
    ):
        self.train()

        train_metrics = defaultdict(AverageMeter)

        for batch in train_dataloader:
            if device:
                batch = batch.to(device)
            seq_length = (batch.abs().sum(-1) > 0).sum(-1)
            # torch.set_printoptions(threshold=10000,edgeitems=100)
            # print('batch',batch.shape, batch)
            # print('seq_length', seq_length)
            mask = generate_sequence_mask(seq_length)[:,1:] #pad mask
            # print('padding mask', mask.shape, mask)

            masked_seq_types = MaskBatch(batch[:,1:,0], pad=0., device=device)
            # print('masked_seq_types.src_mask.shape', masked_seq_types.src_mask.shape, masked_seq_types.src_mask)

            reg_masked_seq_types = MaskBatch(batch[:,1:,0], pad=0., device=device)
            reg_src_mask = reg_masked_seq_types.src_mask.unsqueeze(-1)
            # print('src_mask', reg_src_mask.shape, reg_src_mask[1])
            reg_type_mask = F.one_hot(batch[:, :-1, 1].long(), self.n_types).bool().unsqueeze(1)
            # print('type_mask',reg_type_mask.shape)
            type_reg_mask = torch.repeat_interleave((reg_src_mask * reg_type_mask).unsqueeze(-2),self.n_types,-2)

            # type_reg_mask = reg_src_mask * reg_type_mask
            # print('type_reg_mask',type_reg_mask.shape,type_reg_mask[0,:,:,0])
            # print('type_reg_mask',type_reg_mask.shape,type_reg_mask[0,:,:,1])
            # print('type_reg_mask',type_reg_mask.shape,type_reg_mask[0,:,:,2])
            # print('type_reg_mask',type_reg_mask.shape,type_reg_mask[0,:,:,3])
            # print('type_reg_mask',type_reg_mask.shape,type_reg_mask[0,:,:,4])
            # print('type_reg_mask',type_reg_mask.shape,type_reg_mask[1,:,:,0])
            # print('type_reg_mask',type_reg_mask.shape,type_reg_mask[1,:,:,1])
            # print('type_reg_mask',type_reg_mask.shape,type_reg_mask[1,:,:,2])
            # print('type_reg_mask',type_reg_mask.shape,type_reg_mask[1,:,:,3])
            # print('type_reg_mask',type_reg_mask.shape,type_reg_mask[1,:,:,4])
            # sys.exit()

            v_mu, v_alpha, v_gamma = self.forward(
                batch, masked_seq_types.src_mask
            )
            # print('v_mu, v_alpha, v_gamma', v_mu.shape, v_alpha.shape, v_gamma.shape)
            # print(v_mu, v_alpha, v_gamma)

            nll, integral, log_sum = self._eval_nll(batch, masked_seq_types.src_mask,
                mask, v_mu, v_alpha, v_gamma, device=device
            )
            # print('nll', nll)
            # sys.exit()

            # type_reg_score = (v_alpha / v_gamma)
            type_reg_score = v_alpha
            # print('type_reg_score', type_reg_score.shape)
            # type_reg_score[type_reg_score != type_reg_score] = 0.
            history_grouptype_reg_list = []
            history_grouptype_mean_list = []
            # history_groupsparce_reg_list = []
            for i in range(type_reg_mask.size(-1)):
                masked_score_array = type_reg_score.masked_select(type_reg_mask[...,i])
                # print('masked_score_array', masked_score_array)
                # print('masked_score_array.nelement()', masked_score_array.nelement())
                if masked_score_array.nelement() != 0:
                    history_group_mean = masked_score_array.mean()
                    history_grouptype_reg = (masked_score_array - history_group_mean).pow(2).sum().sqrt()
                    # history_type_var = torch.var(masked_score_array)
                    # history_type_l1 = masked_score_array.pow(1).mean()
                    # print('history_type_var', history_type_var)
                    # print('history_type_l1', history_type_l1)
                    history_grouptype_reg_list.append(history_grouptype_reg)
                    history_grouptype_mean_list.append(history_group_mean)
                    # history_type_l1_list.append(history_type_l1)
            # print(history_type_var_list)
            grouptype_reg = torch.stack(history_grouptype_reg_list).mean()
            groupsparse_reg = torch.stack(history_grouptype_mean_list).pow(1).sum()
            # print('avg_var', avg_var)


            # type_reg_score_masked = type_reg_score * type_reg_mask

            # print('type_reg_score_masked', type_reg_score_masked.shape, type_reg_score_masked[-1,:,:,-1,0])
            # print('type_reg_score_masked', type_reg_score_masked.shape, type_reg_score_masked[-1,:,:,-1,1])
            # print('type_reg_score_masked', type_reg_score_masked.shape, type_reg_score_masked[-1,:,:,-1,2])
            # print('type_reg_score_masked', type_reg_score_masked.shape, type_reg_score_masked[-1,:,:,-1,3])
            # print('type_reg_score_masked', type_reg_score_masked.shape, type_reg_score_masked[-1,:,:,-1,4])
            # sys.exit()

            if kwargs["type_reg"] > 0:
                type_reg = (
                    kwargs["type_reg"]
                    * grouptype_reg
                )
            else:
                type_reg = 0.0

            if kwargs["l1_reg"] > 0:
                l1_reg = (
                    kwargs["l1_reg"]
                    * groupsparse_reg
                )
            else:
                l1_reg = 0.0

            loss = nll + type_reg + l1_reg #add reg

            # loss = nll

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5) #self.max_grad_norm
            optim.step()

            train_metrics["loss"].update(loss, batch.size(0))
            train_metrics["nll"].update(nll, batch.size(0))
            # train_metrics["log_sum"].update(log_sum, batch.size(0))
            train_metrics["type_reg"].update(type_reg, batch.size(0))
            train_metrics["l1_reg"].update(l1_reg, batch.size(0))
            # sys.exit()

            # train_metrics["l2_reg"].update(l2_reg, seq_length.sum())
            # train_metrics["acc"].update(
            #     self._eval_acc(batch, log_intensities, mask), seq_length.sum()
            # )

        if valid_dataloader:
            valid_metrics = self.evaluate(valid_dataloader, device=device)
        else:
            valid_metrics = None

        # print('valid nll',valid_metrics["nll"])
        # print('stop here')
        # sys.exit()

        return train_metrics, valid_metrics

    def evaluate(self, dataloader, device=None):
        metrics = defaultdict(AverageMeter)

        self.eval()
        with torch.no_grad():
            for batch in dataloader:
                if device:
                    batch = batch.to(device)

                seq_length = (batch.abs().sum(-1) > 0).sum(-1)
                mask = generate_sequence_mask(seq_length)[:,1:] #pad mask
                masked_seq_types = MaskBatch(batch[:,1:,0], pad=0., device=device)

                v_mu, v_alpha, v_gamma = self.forward(
                    batch, masked_seq_types.src_mask
                )

                nll, integral, log_sum = self._eval_nll(batch, masked_seq_types.src_mask, 
                    mask, v_mu, v_alpha, v_gamma, device=device
                )

                metrics["nll"].update(nll, batch.size(0))
                metrics["log_sum"].update(log_sum, batch.size(0))

                # metrics["acc"].update(
                #     self._eval_acc(batch, log_intensities, mask),
                #     seq_length.sum(),
                # )

        return metrics

    def get_infectivity(
        self,
        dataloader,
        device=None,
        **kwargs,
    ):
        A = torch.zeros(self.n_types, self.n_types)
        type_counts = torch.zeros(self.n_types, self.n_types)
        self.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader):
                if device:
                    batch = batch.to(device)
                batch_size, T = batch.size()[:2]
                seq_length = (batch.abs().sum(-1) > 0).sum(-1)

                # type_mask = F.one_hot(batch[:, :-1, 1].long(), self.n_types).float().detach().cpu() #b, l_j, k
                type_mask = F.one_hot(batch[:, :-1, 1].long(), self.n_types).float().detach().cpu() #b, l_j, k
                type_mask = torch.repeat_interleave(type_mask.unsqueeze(1),self.n_types,1) #b, k, l_j, k
                # print('type_mask', type_mask.shape, type_mask)

                mask = generate_sequence_mask(seq_length)[:,1:] #pad mask
                masked_seq_types = MaskBatch(batch[:,1:,0], pad=0., device=device)
                src_mask = masked_seq_types.src_mask
                src_mask = src_mask.unsqueeze(-1)
                # print('masked_seq_types', src_mask.shape)

                _, v_alpha, _ = self.forward(
                    batch, masked_seq_types.src_mask
                )
                v_score = v_alpha
                # print('v_score', v_score.shape)
                v_score = v_score.masked_fill(src_mask == 0., 0.).detach().cpu() 
                v_score = v_score.permute((0, 3, 1, 2)) #b,k,l_i,l_j

                # print('v_score', v_score.shape, v_score[0])
                v_score_agg_j = torch.matmul(v_score, type_mask) #b,k,l_i,k

                # print('v_score_agg_j', v_score_agg_j.shape, v_score_agg_j)
                # print('A', torch.sum(v_score_agg_j, (0,2)).shape, torch.sum(v_score_agg_j, (0,2)))
                # print('type_counts', torch.sum(type_mask, (0,2)).shape, torch.sum(type_mask, (0,2)))
                A += torch.sum(v_score_agg_j, (0,2)) #k,k
                type_counts += torch.sum(type_mask, (0,2))  #k,k
                # print('type_mask',type_mask.shape,type_mask[0])
                # sys.exit()

        return A/(type_counts+1)

    # def train_epoch(
    #     self,
    #     train_dataloader,
    #     optim,
    #     valid_dataloader=None,
    #     device=None,
    #     **kwargs,
    # ):
    #     self.train()

    #     train_metrics = defaultdict(AverageMeter)

    #     for batch in train_dataloader:
    #         if device:
    #             batch = batch.to(device)
    #         seq_length = (batch.abs().sum(-1) > 0).sum(-1)
    #         mask = generate_sequence_mask(seq_length)

    #         log_intensities, log_basis_weights = self.forward(
    #             batch, need_weights=True
    #         )
    #         nll = self._eval_nll(
    #             batch, log_intensities, log_basis_weights, mask
    #         )
    #         if kwargs["l2_reg"] > 0:
    #             l2_reg = (
    #                 kwargs["l2_reg"]
    #                 * log_basis_weights.permute(2, 3, 0, 1)
    #                 .masked_select(mask)
    #                 .exp()
    #                 .pow(2)
    #                 .mean()
    #             )
    #         else:
    #             l2_reg = 0.0
    #         loss = nll + l2_reg

    #         optim.zero_grad()
    #         loss.backward()
    #         optim.step()

    #         train_metrics["loss"].update(loss, batch.size(0))
    #         train_metrics["nll"].update(nll, batch.size(0))
    #         train_metrics["l2_reg"].update(l2_reg, seq_length.sum())
    #         train_metrics["acc"].update(
    #             self._eval_acc(batch, log_intensities, mask), seq_length.sum()
    #         )

    #     if valid_dataloader:
    #         valid_metrics = self.evaluate(valid_dataloader, device=device)
    #     else:
    #         valid_metrics = None

    #     return train_metrics, valid_metrics

    # def evaluate(self, dataloader, device=None):
    #     metrics = defaultdict(AverageMeter)

    #     self.eval()
    #     with torch.no_grad():
    #         for batch in dataloader:
    #             if device:
    #                 batch = batch.to(device)

    #             seq_length = (batch.abs().sum(-1) > 0).sum(-1)
    #             mask = generate_sequence_mask(seq_length)

    #             log_intensities, log_basis_weights = self.forward(
    #                 batch
    #             )
    #             nll = self._eval_nll(
    #                 batch, mask, device=device
    #             )

    #             metrics["nll"].update(nll, batch.size(0))
    #             metrics["acc"].update(
    #                 self._eval_acc(batch, log_intensities, mask),
    #                 seq_length.sum(),
    #             )

    #     return metrics

    def predict_next_event(
        self, dataloader, predict_type=False, n_samples=100, device=None
    ):
        """[summary]

        Args:
            dataloader (DataLoader):
            predict_type (bool, optional): Defaults to False.
            device (optional): Defaults to None.

        Raises:
            NotImplementedError: if `predict_type = True`.

        Returns:
            event_seqs_pred (List[List[Union[Tuple, float]]]):
        """

        basis_max_vals = torch.cat([basis.maximum for basis in self.bases]).to(
            device
        )

        event_seqs_pred = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                batch = batch.to(device)

                seq_length = (batch.abs().sum(-1) > 0).sum(-1)
                mask = generate_sequence_mask(seq_length)
                N = seq_length.sum()

                _, log_basis_weights = self.forward(batch)
                # sum up weights for all event types
                basis_weights = log_basis_weights.exp().sum(dim=2)
                # [N, R]
                basis_weights = basis_weights.masked_select(
                    mask.unsqueeze(-1)
                ).view(N, -1)

                t = torch.zeros(N * n_samples, device=device)
                # the index for unfinished samples
                idx = torch.arange(N * n_samples, device=device)
                M = basis_weights[idx // n_samples] @ basis_max_vals
                while len(idx) > 0:
                    # get the index for the corresponding basis_weights
                    idx1 = idx // n_samples
                    M_idx = M[idx1]
                    dt = torch.distributions.Exponential(rate=M_idx).sample()
                    t[idx] += dt
                    U = torch.rand(len(idx), device=device)

                    basis_values = torch.cat(
                        [
                            basis.log_prob(t[idx, None]).exp()
                            for basis in self.bases
                        ],
                        dim=-1,
                    )
                    intensity = (basis_weights[idx1] * basis_values).sum(-1)
                    flag = U < (intensity / M_idx)
                    idx = idx[~flag]

                t_pred = t.view(-1, n_samples).mean(-1)
                i = 0
                for b, L in enumerate(seq_length):
                    # reconstruct the actually timestamps
                    seq = t_pred[i : i + L] + F.pad(
                        batch[b, : L - 1, 0], (1, 0)
                    )
                    # TODO: pad the event type as type prediction hasn't been
                    # implemented yet.
                    seq = F.pad(seq[:, None], (0, 1)).cpu().numpy()
                    event_seqs_pred.append(seq)
                    i += L

        return event_seqs_pred

    # def get_infectivity(
    #     self,
    #     dataloader,
    #     device=None,
    #     steps=50,
    #     occurred_type_only=False,
    #     **kwargs,
    # ):
    #     def func(X, target_type):
    #         _, log_basis_weights = self.forward(
    #             X, onehot=True, target_type=target_type
    #         )
    #         cumulants = self._eval_cumulants(X, log_basis_weights)
    #         # drop index=0 as it corresponds to (t_0, t_1)
    #         return cumulants[:, 1:]

    #     set_eval_mode(self)
    #     # freeze the model parameters to reduce unnecessary backpropogation.
    #     for param in self.parameters():
    #         param.requires_grad_(False)

    #     A = torch.zeros(self.n_types, self.n_types, device=device)
    #     type_counts = torch.zeros(self.n_types, device=device).long()

    #     for batch in tqdm(dataloader):
    #         if device:
    #             batch = batch.to(device)

    #         batch_size, T = batch.size()[:2]
    #         seq_lengths = (batch.abs().sum(-1) > 0).sum(-1)

    #         inputs = torch.cat(
    #             [
    #                 batch[:, :, :1],
    #                 F.one_hot(batch[:, :, 1].long(), self.n_types).float(),
    #             ],
    #             dim=-1,
    #         )
    #         baselines = F.pad(inputs[:, :, :1], (0, self.n_types))
    #         mask = generate_sequence_mask(seq_lengths - 1, device=device)

    #         if occurred_type_only:
    #             occurred_types = set(
    #                 batch[:, :, 1]
    #                 .masked_select(generate_sequence_mask(seq_lengths))
    #                 .long()
    #                 .tolist()
    #             )
    #         else:
    #             occurred_types = range(self.n_types)

    #         event_scores = torch.zeros(
    #             self.n_types, batch_size, T - 1, device=device
    #         )
    #         for k in occurred_types:
    #             ig = batch_integrated_gradient(
    #                 partial(func, target_type=k),
    #                 inputs,
    #                 baselines=baselines,
    #                 mask=mask.unsqueeze(-1),
    #                 steps=steps,
    #             )
    #             event_scores[k] = ig[:, :-1].sum(-1)

    #         # shape=[K, B, T - 1]
    #         A.scatter_add_(
    #             1,
    #             index=batch[:, :-1, 1]
    #             .long()
    #             .view(1, -1)
    #             .expand(self.n_types, -1),
    #             src=event_scores.view(self.n_types, -1),
    #         )

    #         ks = (
    #             batch[:, :, 1]
    #             .long()
    #             .masked_select(generate_sequence_mask(seq_lengths))
    #         )
    #         type_counts.scatter_add_(0, index=ks, src=torch.ones_like(ks))

    #     # plus one to avoid divison by zero
    #     A /= type_counts[None, :].float() + 1

    #     return A.detach().cpu()
