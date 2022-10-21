"""ISAHP model
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

from ..utils.misc import AverageMeter
from ..utils.torch import ResidualLayer, generate_sequence_mask, set_eval_mode

import math



def subsequent_mask(size):
    "mask out subsequent positions"
    atten_shape = (1,size,size)
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
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & torch.transpose(tgt_mask,1,2) & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)).to(device)
        return tgt_mask

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
        return self._event_seqs[i]

    @staticmethod
    def collate_fn(X):
        return nn.utils.rnn.pad_sequence(X, batch_first=True)

class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """
    def forward(self, query, key, value, mask=None, dropout=None):

        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0., -1e9)

        p_attn = F.softmax(scores, dim=-1)
        if mask is not None:
            p_attn = p_attn.masked_fill(mask == 0., 0.)

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

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model, bias=True) for _ in range(3)])

        self.output_alpha = nn.Linear(d_model, d_model, bias=True)

        self.alpha_layer = nn.Sequential(
            nn.Linear(int(self.d_k*self.h/2), self.d_k, bias=True)
            ,nn.Softplus(beta=1.0)
        )

        self.gamma_layer = nn.Sequential(
            nn.Linear(int(self.d_k*self.h/2), self.d_k, bias=True)
            ,nn.Softplus(beta=10.0)
        )

        self.mu_layer = nn.Sequential(
            nn.Linear(self.d_model, self.d_k, bias=True)
            , nn.Sigmoid()
        )

        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        batch_size, T = query.size()[:2]

        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        v_mu, attn = self.attention.forward(query, key, value, mask=mask, dropout=self.dropout)
        attn_repeat = torch.repeat_interleave(attn.unsqueeze(-1),self.d_k, -1)
        value_repeat = torch.repeat_interleave(value.unsqueeze(2),T,2)

        mask = mask.permute((0, 2, 3, 1))

        v_alpha = self.alpha_layer((attn_repeat[:,:int(self.h/2)]*value_repeat[:,:int(self.h/2)]).contiguous().view(batch_size, T, T, -1))# (B x L x L x K)
        v_alpha = v_alpha.masked_fill(mask == 0., 0.)
        v_gamma = self.gamma_layer((attn_repeat[:,int(self.h/2):]*value_repeat[:,int(self.h/2):]).contiguous().view(batch_size, T, T, -1))# (B x L x L x K) 0.1
        v_gamma = v_gamma.masked_fill(mask == 0., 0.)
        v_mu = self.mu_layer(v_mu.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k))

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
        self.softplus_state_decay = nn.Softplus(beta=1.0)

        self.multiheadattention = MultiHeadedAttention(h=num_head, d_model=hidden_size) #self.d_model

    def state_decay(self, v_mu, v_alpha, v_gamma, dt_arr): # (B, L-1, K), (B, L-1, L-1, K) , (B, L-1, L-1, K), (B, L-1, L-1, 1)
        # * element-wise product
        cell_t = torch.tanh(v_mu + torch.sum(v_alpha * v_gamma * torch.exp(-v_gamma * dt_arr),-3)) #+ 1e-3
        return cell_t # (B, L-1, K)

    def forward(
        self, event_seqs, src_mask, onehot=False, target_type=-1
    ):

        assert event_seqs.size(-1) == 1 + (
            self.n_types if onehot else 1
        ), event_seqs.size()

        batch_size, T = event_seqs.size()[:2]
        self.ts = F.pad(event_seqs[:, :, 0], (1, 0)) #(t+1)
        dt = self.ts[:, 1:] - self.ts[:, :-1] #(t)
        temp_feat = dt[:, :-1].unsqueeze(-1) #[64, 336, 1] t-1

        if onehot:
            type_feat = self.embed(event_seqs[:, :-1, 1:])
        else:
            type_feat = self.embed(
                F.one_hot(event_seqs[:, :-1, 1].long(), self.n_types).float()
            )

        feat = torch.cat([temp_feat, type_feat], dim=-1) #[64, 336, 20]
        v_mu, v_alpha, v_gamma = self.multiheadattention.forward(feat,feat,feat, mask=src_mask) #

        return v_mu, v_alpha, v_gamma

    def _eval_nll(
        self, event_seqs, src_mask, mask, v_mu, v_alpha, v_gamma, device=None, n_mc_samples = 20
    ):  
        n_batch = self.ts.size(0)
        n_times = self.ts.size(1) - 2

        dt_arr = torch.tril(torch.cdist(event_seqs[:, :, 0:1], event_seqs[:, :, 0:1], p=2))[:,1:,:-1] #(B, L-1, L-1)
        dt_seq = torch.diagonal(dt_arr, offset=0, dim1=1, dim2=2) #(B, L-1)
        torch.set_printoptions(threshold=10000,edgeitems=100)
        dt_meta = torch.tril(torch.repeat_interleave(torch.unsqueeze(dt_seq,-1),n_times,-1)).masked_fill(src_mask == 0., 0.) #(B, L-1, L-1)
        dt_offset = (dt_arr - dt_meta).masked_fill(src_mask == 0., 0.)
        type_mask = F.one_hot(event_seqs[:, 1:, 1].long(), self.n_types).float()


        
        cell_t = self.state_decay(v_mu, v_alpha, v_gamma, dt_arr[:,:,:,None]) #(B, L-1, K)
        log_intensities = cell_t.log()  # log intensities
        log_sum = (log_intensities * type_mask).sum(-1).masked_select(mask).sum() #B x L-1 -> B

        taus = torch.rand(n_batch, n_times, n_times, 1, n_mc_samples).to(device)# self.process_dim replaced 1 (B,L-1,L-1,1,20)
        taus = dt_meta[:, :, :, None, None] * taus  # inter-event times samples) (B,L-1,L-1,1,20).
        taus =  taus + dt_offset[:,:,:,None,None] #(B,L-1,L-1,1,20).

        cell_tau = self.state_decay(
            v_mu[:,:,:,None],
            v_alpha[:,:,:,:,None],
            v_gamma[:,:,:,:,None],
            taus) #(B,L-1, k, 20)

        total_intens_samples = cell_tau.sum(dim=2) #sum over k (B,L-1, 20)
        partial_integrals = dt_seq * total_intens_samples.mean(dim=2)
        partial_integrals = partial_integrals.masked_select(mask) #average samples (B,L-1)

        integral_ = partial_integrals.sum() #B

        res = torch.sum(- log_sum + integral_)/n_batch
        log_sum = torch.sum(- log_sum)/n_batch
        integral = torch.sum(integral_)/n_batch

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
            mask = generate_sequence_mask(seq_length)[:,1:] #pad mask
            masked_seq_types = MaskBatch(batch[:,1:,0], pad=0., device=device)

            reg_masked_seq_types = MaskBatch(batch[:,1:,0], pad=0., device=device)
            reg_src_mask = reg_masked_seq_types.src_mask.unsqueeze(-1)
            reg_type_mask = F.one_hot(batch[:, :-1, 1].long(), self.n_types).bool().unsqueeze(1)
            torch.set_printoptions(threshold=10000,edgeitems=100)
            type_reg_mask = torch.repeat_interleave((reg_src_mask * reg_type_mask).unsqueeze(-2),self.n_types,-2)

            v_mu, v_alpha, v_gamma = self.forward(
                batch, masked_seq_types.src_mask
            )

            nll, integral, log_sum = self._eval_nll(batch, masked_seq_types.src_mask,
                mask, v_mu, v_alpha, v_gamma, device=device
            )

            type_reg_score = v_alpha
            history_grouptype_reg_list = []
            history_grouptype_mean_list = []

            for i in range(type_reg_mask.size(-1)):
                masked_score_array = type_reg_score.masked_select(type_reg_mask[...,i])
                masked_score_array = masked_score_array.reshape(self.n_types,-1)

                if masked_score_array.nelement() > self.n_types:
                    history_group_mean = torch.mean(masked_score_array,-1)
                    history_grouptype_reg = torch.var(masked_score_array,-1)
                    history_grouptype_reg_list.append(history_grouptype_reg)
                    history_grouptype_mean_list.append(history_group_mean)

            grouptype_reg = torch.stack(history_grouptype_reg_list).sum()
            groupsparse_reg = torch.stack(history_grouptype_mean_list).pow(1).sum()

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

            loss = nll + type_reg + l1_reg

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5) #self.max_grad_norm
            optim.step()

            train_metrics["loss"].update(loss, batch.size(0))
            train_metrics["nll"].update(nll, batch.size(0))

            train_metrics["type_reg"].update(type_reg, batch.size(0))
            train_metrics["l1_reg"].update(l1_reg, batch.size(0))

        if valid_dataloader:
            valid_metrics = self.evaluate(valid_dataloader, device=device)
        else:
            valid_metrics = None

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

                type_mask_j = F.one_hot(batch[:, :-1, 1].long(), self.n_types).float().detach().cpu() #b, l_j, k
                type_mask_i = F.one_hot(batch[:, 1:, 1].long(), self.n_types).float().detach().cpu() #b, l_i, k
                type_mask_i_repeat = torch.repeat_interleave(type_mask_i.unsqueeze(1),T-1,1) #b, l_j, l_i, k
                type_mask_i_repeat = type_mask_i_repeat.permute((0, 1, 3, 2)) #b, l_j, k, l_i

                masked_seq_types = MaskBatch(batch[:,1:,0], pad=0., device=device)
                src_mask = masked_seq_types.src_mask
                src_mask = src_mask.unsqueeze(-1)

                _, v_alpha, _ = self.forward(
                    batch, masked_seq_types.src_mask
                )
                v_score = v_alpha
                v_score = v_score.masked_fill(src_mask == 0., 0.).detach().cpu() #b,l_i,l_j,k
                v_score = v_score.permute((0, 2, 1, 3)) #b,l_j,l_i,k
                v_score = torch.matmul(v_score, type_mask_i_repeat) #b,l_j,l_i,l_i
                v_score_instance = v_score.diagonal(offset=0, dim1=2, dim2=3) #b,l_j,l_i

                count_type = torch.triu(torch.ones(v_score_instance.shape)) #b,l_j,l_i

                v_score_agg_i = torch.matmul(v_score_instance, type_mask_i).permute((0, 2, 1)) #b,k_i,l_j
                v_score_agg = torch.matmul(v_score_agg_i, type_mask_j) #b,k_i,k_j

                count_agg_i = torch.matmul(count_type, type_mask_i).permute((0, 2, 1)) #b,k_i,l_j
                count_agg = torch.matmul(count_agg_i, type_mask_j) #b,k_i,k_j

                A += torch.sum(v_score_agg, 0) #k,k
                type_counts += torch.sum(count_agg, 0)  #k,k

        return A/(type_counts+1)

    def predict_next_event_type(self, dataloader, device=None):
        self.eval()
        event_seqs_pred_type = []
        event_seqs_truth_type = []
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

                src_mask = masked_seq_types.src_mask

                n_batch = self.ts.size(0)
                n_times = self.ts.size(1) - 2

                dt_arr = torch.tril(torch.cdist(batch[:, :, 0:1], batch[:, :, 0:1], p=2))[:,1:,:-1] #(B, L-1, L-1)
                dt_seq = torch.diagonal(dt_arr, offset=0, dim1=1, dim2=2) #(B, L-1)
                dt_meta = torch.tril(torch.repeat_interleave(torch.unsqueeze(dt_seq,-1),n_times,-1)).masked_fill(src_mask == 0., 0.) #(B, L-1, L-1)
                dt_offset = (dt_arr - dt_meta).masked_fill(src_mask == 0., 0.)

                intensities = self.state_decay(v_mu, v_alpha, v_gamma, dt_arr[:,:,:,None]) #(B, L-1, K)

                k_pred = intensities.argmax(-1).masked_select(mask).cpu().numpy()
                event_seqs_pred_type.append(k_pred)
                event_seqs_truth_type.append(batch[:, 1:, 1].long().masked_select(mask).cpu().numpy())

        return event_seqs_pred_type, event_seqs_truth_type

