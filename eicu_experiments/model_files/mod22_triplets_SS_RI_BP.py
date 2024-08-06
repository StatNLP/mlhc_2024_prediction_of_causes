# %%
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
tqdm.pandas()
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau

# %%
# data settings
test_cond = 0
sepsis_check=1
data_path = '/home/mitarb/fracarolli/eicu/final/eicu_preprocessed_n.pkl'
sample_divisor = 1
number_of_epochs = 1000
V, D = 98, 17
fore_max_len = 2640

# %%
def inv_list(l, start=0):  # Create vind
    d = {}
    for i in range(len(l)):
        d[l[i]] = i+start
    return d


def f(x):
    mask   = [0 for i in range(V)]
    values = [0 for i in range(V)]
    for vv in x:  # tuple of ['vind','value']
        v = int(vv[0])-1  # shift index of vind
        mask[v] = 1
        values[v] = vv[1]  # get value
    return values+mask  # concat


def pad(x):
    if len(x) > 880:
        print(len(x))
    return x+[0]*(fore_max_len-len(x))


# %%
fore_train_op = joblib.load(open('/home/mitarb/fracarolli/eicu/240401_Experimente/fore_train_op.pkl', 'rb'))
fore_valid_op = joblib.load(open('/home/mitarb/fracarolli/eicu/240401_Experimente/fore_valid_op.pkl', 'rb'))
fore_train_ip = joblib.load(open('/home/mitarb/fracarolli/eicu/240401_Experimente/fore_train_ip.pkl', 'rb'))
fore_valid_ip = joblib.load(open('/home/mitarb/fracarolli/eicu/240401_Experimente/fore_valid_ip.pkl', 'rb'))

# %% [markdown]
# ## Define model architecture.

# %%
class CVE(nn.Module):
    def __init__(self, hid_units, output_dim):
        super(CVE, self).__init__()
        self.hid_units = hid_units
        self.output_dim = output_dim
        self.W1 = nn.Parameter(torch.Tensor(1, self.hid_units))
        self.b1 = nn.Parameter(torch.Tensor(self.hid_units))
        self.W2 = nn.Parameter(torch.Tensor(self.hid_units, self.output_dim))
        nn.init.xavier_uniform_(self.W1)
        nn.init.zeros_(self.b1)
        nn.init.xavier_uniform_(self.W2)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = torch.matmul(torch.tanh(torch.add(torch.matmul(x, self.W1), self.b1)), self.W2)
        return x


class Attention(nn.Module):
    def __init__(self, hid_dim):
        super(Attention, self).__init__()
        self.hid_dim = hid_dim
        self.W = nn.Parameter(torch.Tensor(50, self.hid_dim))
        self.b = nn.Parameter(torch.Tensor(self.hid_dim))
        self.u = nn.Parameter(torch.Tensor(self.hid_dim, 1))
        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.b)
        nn.init.xavier_uniform_(self.u)

    def forward(self, x, mask, mask_value=-1e30):
        attn_weights = torch.matmul(torch.tanh(torch.add(torch.matmul(x, self.W), self.b)), self.u)
        mask = mask.unsqueeze(-1)
        attn_weights = mask * attn_weights + (1 - mask) * mask_value
        attn_weights = F.softmax(attn_weights, dim=-2)
        return attn_weights


class Transformer(nn.Module):
    def __init__(self, N=2, h=8, dk=None, dv=None, dff=None, dropout=0, d=8):
        super(Transformer, self).__init__()
        self.N, self.h, self.dk, self.dv, self.dff, self.dropout = N, h, dk, dv, dff, dropout
        self.epsilon = torch.finfo(torch.float32).eps * torch.finfo(torch.float32).eps
        self.Wq = nn.Parameter(torch.Tensor(N, h, d, dk))
        self.Wk = nn.Parameter(torch.Tensor(N, h, d, dk))
        self.Wv = nn.Parameter(torch.Tensor(N, h, d, dv))
        self.Wo = nn.Parameter(torch.Tensor(N, dv * h, d))
        self.W1 = nn.Parameter(torch.Tensor(N, d, dff))
        self.b1 = nn.Parameter(torch.Tensor(N, dff))
        self.W2 = nn.Parameter(torch.Tensor(N, dff, d))
        self.b2 = nn.Parameter(torch.Tensor(N, d))
        self.gamma = nn.Parameter(torch.Tensor(2 * N))
        self.beta = nn.Parameter(torch.Tensor(2 * N))
        nn.init.xavier_uniform_(self.Wq)
        nn.init.xavier_uniform_(self.Wk)
        nn.init.xavier_uniform_(self.Wv)
        nn.init.xavier_uniform_(self.Wo)
        nn.init.xavier_uniform_(self.W1)
        nn.init.zeros_(self.b1)
        nn.init.xavier_uniform_(self.W2)
        nn.init.zeros_(self.b2)
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        self.training = False
        
    def forward(self, x, mask, mask_value=-1e-30):
        mask = mask.unsqueeze(-2)
        for i in range(self.N):
            # MHA
            mha_ops = []
            for j in range(self.h):
                q = torch.matmul(x, self.Wq[i, j, :, :])          
                k = torch.matmul(x, self.Wk[i, j, :, :]).permute(0, 2, 1)
                v = torch.matmul(x, self.Wv[i, j, :, :])
                A = torch.matmul(q, k)
                # Mask unobserved steps.
                A = mask * A + (1 - mask) * mask_value
                # Mask for attention dropout.
                if self.training:
                    dp_mask = (torch.rand(A.shape, device=A.device) >= self.dropout).float()
                    A = A * dp_mask + (1 - dp_mask) * mask_value
                A = F.softmax(A, dim=-1)
                mha_ops.append(torch.matmul(A, v))
            conc = torch.cat(mha_ops, dim=-1)
            proj = torch.matmul(conc, self.Wo[i, :, :])
            # Dropout.
            if self.training:
                proj = F.dropout(proj, p=self.dropout)
            # Add & LN
            x = x + proj
            mean = x.mean(dim=-1, keepdim=True)
            variance = torch.mean(torch.square(x - mean), dim=-1, keepdim=True)
            std = torch.sqrt(variance + self.epsilon)
            x = (x - mean) / std
            x = x * self.gamma[2 * i] + self.beta[2 * i]
            # FFN
            ffn_op = torch.add(torch.matmul(F.relu(torch.add(
                torch.matmul(x, self.W1[i, :, :]), self.b1[i, :])),
                                            self.W2[i, :, :]), self.b2[i, :])
            # Dropout.
            if self.training:
                ffn_op = F.dropout(ffn_op, p=self.dropout)
            # Add & LN
            x = x + ffn_op
            mean = x.mean(dim=-1, keepdim=True)
            variance = torch.mean(torch.square(x - mean), dim=-1, keepdim=True)
            std = torch.sqrt(variance + self.epsilon)
            x = (x - mean) / std
            x = x * self.gamma[2 * i + 1] + self.beta[2 * i + 1]
        return x


class StratsModel(nn.Module):
    def __init__(self, D, max_len, V, d, N, he, dropout, forecast=False, autoregressive=True):
        super(StratsModel, self).__init__()
        self.forecast = forecast
        self.autoregressive=autoregressive
        self.demo_enc = nn.Sequential(nn.Linear(D, D * d),
                                      nn.Tanh(),
                                      nn.Linear(D * d, d),
                                      nn.Tanh())
        self.varis_emb = nn.Embedding(V + 1, d)
        nn.init.uniform_(self.varis_emb.weight, a=-0.05, b=0.05)
        self.values_emb = CVE(int(np.sqrt(d)), d)
        self.times_emb = CVE(int(np.sqrt(d)), d)
        self.transformer = Transformer(N, he, dk=d//he, dv=d//he, dff=2*d, dropout=dropout, d=d)

        if autoregressive:
            self.linearize = nn.Linear(50, V)
            self.dec = nn.TransformerDecoderLayer(d_model=V, dim_feedforward=V, nhead=1, batch_first=True)
        else:
            self.attention = Attention(2 * d)
            self.attentions = nn.ModuleList([Attention(2*d) for i in range(24)])
            self.fore_op_calc = nn.Linear(2 * d, V)
            self.op_calc = nn.Linear(V, 1)
            self.sig = nn.Sigmoid()        
        

    def forward(self, demo, times, values, varis, trainn=False, tgt=torch.zeros(32, 1, V).cuda(), schedule=True, probability=0.5, deterministic=False):
        demo_enc = self.demo_enc(demo)
        varis_emb = self.varis_emb(varis)
        values_emb = self.values_emb(values)
        times_emb = self.times_emb(times)

        comb_emb = (torch.sum(torch.stack([varis_emb, values_emb, times_emb]), dim=0))
        self.trainn = trainn
        mask = torch.clip(varis, 0, 1)
        cont_emb = self.transformer(comb_emb, mask)
        
        if trainn and schedule:
            output = torch.zeros(cont_emb.size(0), 1, V).cuda()
            output.requires_grad=True
            linear_memory  = self.linearize(cont_emb)
            for i in range(24):
                res = self.dec(output, linear_memory)
                if deterministic:
                    sample = i/24
                else:
                    sample = torch.rand(1).item()
                # print(sample, probability)
                if sample < probability:
                    output = torch.concat([output, tgt[:,i:i+1,:]], dim=1)
                else:
                    output = torch.concat([output, res[:,-1:,:]], dim=1)
            return output[:, 1:, :] 
        if self.autoregressive:
            if self.trainn:
                mask_new = torch.nn.Transformer.generate_square_subsequent_mask(25).cuda()
                linear_memory = self.linearize(cont_emb)
                start_input = torch.zeros(32, 1, V).cuda()
                tgt = torch.concat([start_input, tgt], dim=1)
                result=self.dec(tgt, linear_memory, mask_new)
                return result[:, 1:, :]
            if not self.trainn:
                output = torch.zeros(32, 1, V).cuda()
                linear_memory  = self.linearize(cont_emb)
                for i in range(24):
                    res = self.dec(output.detach(), linear_memory)
                    output = torch.concat([output, res[:,-1:,:]], dim=1)
                return output[:, 1:, :]
                
        if not self.autoregressive:
            fused_embs = []
            for attention in self.attentions:
                attn_weights = attention(cont_emb, mask)
                fused_emb = torch.matmul(attn_weights.transpose(-1, -2), cont_emb).squeeze(-2)
                fused_embs.append(fused_emb)
            fused_emb = torch.stack(fused_embs, axis=1)
            concat = torch.cat([fused_emb, demo_enc.unsqueeze(axis=-2).repeat(1, 24, 1)], axis=-1)
            x = self.fore_op_calc(concat)
            if self.forecast:
                return x
            else:
                x = self.sig(self.op_calc(x)).mean(axis=-2)
                return x

print("Model definitions loaded")


# %% [markdown]
# ## Pretrain on forecasting.

# %%
lr, batch_size, samples_per_epoch, patience = 0.0005, 32, int(102400/sample_divisor), 6
d, N, he, dropout = 50, 2, 4, 0.2
print('number of parameters: ', V)

model = StratsModel(D, fore_max_len, V, d, N, he, dropout, forecast=True).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3)
# Pretrain fore_model.
best_val_loss = np.inf
N_fore = len(fore_train_op)
fore_savepath = 'eicu_24h_strats_no_interp_with_ss_fore_strats_ims_sampling_backprop_schedule_random_new.pytorch'
loss_func = torch.nn.MSELoss(reduction="none")
for e in range(number_of_epochs):
    e_indices = np.random.choice(range(N_fore), size=samples_per_epoch, replace=False)
    e_loss = 0
    pbar = tqdm(range(0, len(e_indices), batch_size))
    model.train()
    for start in pbar:
        ind = e_indices[start:start+batch_size]
        prob=min([1.0, 1.0+(0.25-1.0)*(1-e/200)])
        output = model(*[torch.tensor(ip[ind], dtype=torch.float32).cuda() if i < 3 
                         else torch.tensor(ip[ind], dtype=torch.int32).cuda() 
                         for i, ip in enumerate(fore_train_ip)], trainn=True, tgt=torch.tensor(fore_train_op[ind, :, :V], dtype=torch.float32).cuda(), schedule=True, deterministic=False, probability=prob)
        loss = torch.tensor(fore_train_op[ind, :, V:], dtype=torch.float32).cuda()*(output-torch.tensor(fore_train_op[ind, :, :V], dtype=torch.float32).cuda())**2
        loss = loss.sum(axis=-1).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        e_loss += loss.detach()
        pbar.set_description('%f' % e_loss)
    val_loss = 0
    model.eval()
    loss_list = []
    pbar = tqdm(range(0, len(fore_valid_op)-batch_size, batch_size))  # len(fore_valid_op)           ####################   maybe also batch_size instead of 32
    for start in pbar:
        output = model(*[torch.tensor(ip[start:start+batch_size], dtype=torch.float32).cuda() 
                         if i < 3 else torch.tensor(ip[start:start+batch_size], dtype=torch.int32).cuda() 
                         for i, ip in enumerate(fore_valid_ip)], trainn=False, tgt=torch.tensor(fore_valid_op[start:start+batch_size], dtype=torch.float32).cuda())
        loss = torch.tensor(fore_valid_op[start:start+batch_size,:, V:], dtype=torch.float32).cuda()*loss_func(
            output, torch.tensor(fore_valid_op[start:start+batch_size,:, :V], dtype=torch.float32).cuda())
        loss_list.extend(loss.sum(axis=-1).mean(axis=-1).detach().cpu().tolist())
        loss = loss.sum(axis=-1).mean()
        val_loss += loss.detach().cpu()
        pbar.set_description('%f' % val_loss)
    loss_p = e_loss*batch_size/samples_per_epoch
    val_loss_p = val_loss*batch_size/len(fore_valid_op)
    print('Epoch', e, 'loss', loss_p, 'val loss', val_loss_p, "mean and std", np.mean(loss_list), np.std(loss_list))
    with open('loss_values_log', 'a') as f:
        f.write(str(e)+' ' + str(loss_p.item()) + ' ' + str(val_loss_p.item())+ '\n')
    scheduler.step(val_loss_p.item())
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), fore_savepath)
        best_epoch = e
    if (e-best_epoch) > patience:
        break
print('Training has ended.')

#STRATS, IMS, Sampling, Detach


