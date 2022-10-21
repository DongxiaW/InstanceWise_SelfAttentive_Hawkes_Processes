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

# class SublayerConnection(nn.Module):
#     """
#     A residual connection followed by a layer norm.
#     Note for code simplicity the norm is first as opposed to last.
#     """

#     def __init__(self, size, dropout):
#         super(SublayerConnection, self).__init__()
#         self.norm = LayerNorm(size)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x, sublayer):
#         "Apply residual connection to any sublayer with the same size."
#         return x + self.dropout(sublayer(self.norm(x)))

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation.forward(self.w_1(x))))



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
        scores = torch.exp(torch.matmul(query, key.transpose(-2, -1))) \
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
        self.output_linear = nn.Linear(d_model, d_model, bias=True)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # the same mask applies to all heads
            # unsqueeze Returns a new tensor with a dimension of size one
            # inserted at the specified position.
            mask = mask.unsqueeze(1)

        batch_size, T = query.size()[:2]

        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        # print('query, key, value shape', query.shape)

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention.forward(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        x = self.output_linear(x)

        return x, attn

class SelfAttentiveHawkesProcesses(nn.Module):
    def __init__(
        self,
        n_types: int,
        embedding_dim: int = 31, #32,
        hidden_size: int = 32, #32,
        dropout: float = 0.0,
        num_head: int = 6,
        **kwargs,
    ):
        super().__init__()
        self.n_types = n_types
        self.nLayers = 1

        self.embed = nn.Linear(n_types, embedding_dim, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.m_dt = nn.ReLU()
        self.softplus_state_decay = nn.Softplus(beta=1.0)
        self.d_model = hidden_size
        self.gelu = GELU()
        self.process_dim = self.d_model *4
        self.d_k = self.d_model // num_head
        self.h = num_head

        # self.input_sublayer = SublayerConnection(size=self.d_model, dropout=dropout)
        # self.output_sublayer = SublayerConnection(size=self.d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=self.d_model, d_ff=self.d_model * 4, dropout=dropout)
        # self.output_layer = nn.Linear(self.d_model, self.d_k, bias=True)

        self.start_layer = nn.Sequential(
            nn.Linear(self.d_model, self.d_model, bias=True),
            self.gelu
        )

        self.converge_layer = nn.Sequential(
            nn.Linear(self.d_model, self.d_model, bias=True),
            self.gelu
        )

        self.decay_layer = nn.Sequential(
            nn.Linear(self.d_model, self.d_model, bias=True)
            ,nn.Softplus(beta=10.0)
        )

        self.intensity_layer = nn.Sequential(
            nn.Linear(self.d_k, self.d_k, bias = True)
            ,nn.Softplus(beta=1.)
        )

        self.multiheadattention = MultiHeadedAttention(h=num_head, d_model=hidden_size) #self.d_model
        self.norm = LayerNorm(self.d_model)


    def state_decay(self, start_point, converge_point, omega, duration_t): # (B, L-1, K), (B, L-1, L-1, K) , (B, L-1, L-1, K), (B, L-1, L-1, 1)
        # * element-wise product
        # cell_t = self.softplus_state_decay(torch.tanh(v_mu + torch.sum(v_alpha * v_gamma * torch.exp(-v_gamma * dt_arr),-3)))
        # cell_t = self.softplus_state_decay(v_mu + torch.sum(v_alpha * torch.exp(-v_gamma * dt_arr),-3))
        cell_t = torch.tanh(converge_point + (start_point - converge_point) * torch.exp(-omega * duration_t)) #+ 1e-3
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
        dt = self.ts[:, 1:] - self.ts[:, :-1] #(t)

        temp_feat = dt[:, :-1].unsqueeze(-1) #[64, 336, 1] t-1

        # (z_1, ..., z_{n - 1})
        if onehot:
            type_feat = self.embed(event_seqs[:, :-1, 1:])
        else:
            type_feat = self.embed(
                F.one_hot(event_seqs[:, :-1, 1].long(), self.n_types).float()
            )

        feat = torch.cat([temp_feat, type_feat], dim=-1) #[64, 336, 20]
        for i in range(self.nLayers):
            output, attn = self.multiheadattention.forward(self.norm(feat), self.norm(feat), self.norm(feat), mask=src_mask)
            feat = feat + output
            # feat = self.input_sublayer(feat, lambda _x: self.attention.forward(_x, _x, _x, mask=src_mask))
            # feat = self.output_sublayer(self.norm(feat), self.feed_forward)
            feat = feat + self.feed_forward(self.norm(feat))

        # embed_info = self.output_layer(feat)
        embed_info = feat
        start_point = self.start_layer(embed_info).contiguous().view(batch_size, -1, self.h, self.d_k).mean(-2)
        converge_point = self.converge_layer(embed_info).view(batch_size, -1, self.h, self.d_k).mean(-2)
        omega = self.decay_layer(embed_info).view(batch_size, -1, self.h, self.d_k).mean(-2)

        # print('start_point',start_point.shape)
        # print('attn',attn.shape)

        # v_mu, v_alpha, v_gamma = self.multiheadattention.forward(feat,feat,feat, mask=src_mask) #

        return start_point, converge_point, omega, attn

    def _eval_nll(
        self, event_seqs, src_mask, mask, start_point, converge_point, omega, device=None, n_mc_samples = 20
    ):  
        n_batch = self.ts.size(0)
        n_times = self.ts.size(1) - 2

        dt_arr = torch.tril(torch.cdist(event_seqs[:, :, 0:1], event_seqs[:, :, 0:1], p=2))[:,1:,:-1] #(B, L-1, L-1)
        dt_seq = torch.diagonal(dt_arr, offset=0, dim1=1, dim2=2) #(B, L-1)
        # dt_meta = torch.tril(torch.repeat_interleave(torch.unsqueeze(dt_seq,-1),n_times,-1)).masked_fill(src_mask == 0., 0.) #(B, L-1, L-1)
        # dt_offset = (dt_arr - dt_meta).masked_fill(src_mask == 0., 0.)

        type_mask = F.one_hot(event_seqs[:, 1:, 1].long(), self.n_types).float()

        cell_t = self.state_decay(start_point, converge_point, omega, dt_seq[:,:,None]) #(B, L-1, K)
        cell_t = self.intensity_layer(cell_t)
        log_intensities = cell_t.log()  # log intensities
        # print('cell_t',cell_t)

        log_sum = (log_intensities * type_mask).sum(-1).masked_select(mask).sum() #B x L-1 -> B

        taus = torch.rand(n_batch, n_times, 1, n_mc_samples).to(device)# self.process_dim replaced 1 (B,L-1,L-1,1,20)
        taus = dt_seq[:, :, None, None] * taus  # inter-event times samples) (B,L-1,L-1,1,20).
        # taus =  taus + dt_offset[:,:,:,None,None] #(B,L-1,L-1,1,20).

        cell_tau = self.state_decay(
            converge_point[:,:,:,None],
            start_point[:,:,:,None],
            omega[:,:,:,None],
            taus) #(B,L-1, k, 20)
        cell_tau = cell_tau.transpose(2, 3)
        cell_tau = self.intensity_layer(cell_tau).transpose(2,3)
        total_intens_samples = cell_tau.sum(dim=2) #sum over k (B,L-1, 20)
        # print('cell_tau',cell_tau.shape)
        partial_integrals = dt_seq * total_intens_samples.mean(dim=2)
        partial_integrals = partial_integrals.masked_select(mask) #average samples (B,L-1)

        integral_ = partial_integrals.sum() #B

        res = torch.sum(- log_sum + integral_)/n_batch
        log_sum = torch.sum(- log_sum)/n_batch
        integral = torch.sum(integral_)/n_batch
        # print('res',res, 'log_sum',log_sum,'integral',integral)
        # sys.exit()

        return res, integral, log_sum

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

            start_point, converge_point, omega,_ = self.forward(
                batch, masked_seq_types.src_mask
            )
            # print('v_mu, v_alpha, v_gamma', v_mu.shape, v_alpha.shape, v_gamma.shape)
            # print(v_mu, v_alpha, v_gamma)

            nll, integral, log_sum = self._eval_nll(batch, masked_seq_types.src_mask,
                mask,start_point, converge_point, omega, device=device
            )
            # print('nll', nll)


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

                start_point, converge_point, omega,_ = self.forward(
                    batch, masked_seq_types.src_mask
                )

                nll, integral, log_sum = self._eval_nll(batch, masked_seq_types.src_mask,
                    mask,start_point, converge_point, omega, device=device
                )

                metrics["nll"].update(nll, batch.size(0))
                metrics["log_sum"].update(log_sum, batch.size(0))

                # metrics["acc"].update(
                #     self._eval_acc(batch, log_intensities, mask),
                #     seq_length.sum(),
                # )

        return metrics

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

                start_point, converge_point, omega,_ = self.forward(
                    batch, masked_seq_types.src_mask
                )

                src_mask = masked_seq_types.src_mask

                n_batch = self.ts.size(0)
                n_times = self.ts.size(1) - 2

                dt_arr = torch.tril(torch.cdist(batch[:, :, 0:1], batch[:, :, 0:1], p=2))[:,1:,:-1] #(B, L-1, L-1)
                dt_seq = torch.diagonal(dt_arr, offset=0, dim1=1, dim2=2) #(B, L-1)

                intensities = self.state_decay(start_point, converge_point, omega, dt_seq[:,:,None]) #(B, L-1, K)
                intensities = self.intensity_layer(intensities)
                torch.set_printoptions(threshold=10000,edgeitems=100)
                # print('intensities',intensities.shape,intensities)
                # sys.exit()

                k_pred = intensities.argmax(-1).masked_select(mask).cpu().numpy()
                event_seqs_pred_type.append(k_pred)
                event_seqs_truth_type.append(batch[:, 1:, 1].long().masked_select(mask).cpu().numpy())

        return event_seqs_pred_type, event_seqs_truth_type


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
                # type_mask_i_repeat = torch.repeat_interleave(type_mask_i.unsqueeze(1),T-1,1) #b, l_j, l_i, k
                # type_mask_i_repeat = type_mask_i_repeat.permute((0, 1, 3, 2)) #b, l_j, k, l_i
                # print('type_mask', type_mask.shape)

                masked_seq_types = MaskBatch(batch[:,1:,0], pad=0., device=device)
                src_mask = masked_seq_types.src_mask
                # src_mask = src_mask.unsqueeze(-1)
                # print('masked_seq_types', src_mask.shape)

                _, _, _, attn = self.forward(
                    batch, masked_seq_types.src_mask
                )
                v_score = attn.mean(1) #b, l,l
                # print('v_score', v_score.shape)
                # print('src_mask',src_mask.shape)
                v_score = v_score.masked_fill(src_mask == 0., 0.).detach().cpu() #b,l_i,l_j,k
                v_score_instance = v_score.permute((0, 2, 1))
                # print(' v_score_instance',v_score_instance)
                # v_score = v_score.permute((0, 2, 1, 3)) #b,l_j,l_i,k
                # # print('v_score', v_score.shape, 'type_mask_i_repeat',type_mask_i_repeat.shape)
                # v_score = torch.matmul(v_score, type_mask_i_repeat) #b,l_j,l_i,l_i
                # v_score_instance = v_score.diagonal(offset=0, dim1=2, dim2=3) #b,l_j,l_i
                # print('v_score sum', v_score.sum())

                count_type = torch.triu(torch.ones(v_score_instance.shape)) #b,l_j,l_i

                # print('v_score', v_score.shape, v_score[1])
                # print('count_type',count_type.shape,count_type[1])

                v_score_agg_i = torch.matmul(v_score_instance, type_mask_i).permute((0, 2, 1)) #b,k_i,l_j
                v_score_agg = torch.matmul(v_score_agg_i, type_mask_j) #b,k_i,k_j
                # print('v_score_agg', v_score_agg.shape, v_score_agg[0])
                # print('v_score_agg sum', v_score_agg.sum())

                count_agg_i = torch.matmul(count_type, type_mask_i).permute((0, 2, 1)) #b,k_i,l_j
                count_agg = torch.matmul(count_agg_i, type_mask_j) #b,k_i,k_j
                # print('count_agg', count_agg.shape, count_agg[0])

                # print('v_score_agg_j', v_score_agg_j.shape, v_score_agg_j)
                # print('A', torch.sum(v_score_agg_j, (0,2)).shape, torch.sum(v_score_agg_j, (0,2)))
                # print('type_counts', torch.sum(type_mask, (0,2)).shape, torch.sum(type_mask, (0,2)))
                A += torch.sum(v_score_agg, 0) #k,k
                type_counts += torch.sum(count_agg, 0)  #k,k
        print('A',A/(type_counts+1))

        return A/(type_counts+1)
