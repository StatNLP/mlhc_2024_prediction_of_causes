import joblib
import numpy as np
import pandas as pd

from tqdm import tqdm
tqdm.pandas()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau

from datetime import datetime

print('Start time: ', datetime.now())


fore_train_op = joblib.load(open('fore_train_op_dense.pkl', 'rb'))
fore_valid_op = joblib.load(open('fore_valid_op_dense.pkl', 'rb'))
fore_train_ip = joblib.load(open('fore_train_ip_dense.pkl', 'rb'))
fore_valid_ip = joblib.load(open('fore_valid_ip_dense.pkl', 'rb'))

d_model, e_layers, d_layers, d_ff = 256, 2, 2, 2048
number_of_epochs = 100
V, D = 98, 17
fore_max_len = 2640
print('loading of joblib completed')


import argparse
parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

# basic config
parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
parser.add_argument('--train_only', type=bool, required=False, default=False, help='perform training on full input dataset without validation and testing')
parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')
parser.add_argument('--model', type=str, required=False, default='Autoformer',
                    help='model name, options: [Autoformer, Informer, Transformer]')

# data loader
parser.add_argument('--data', type=str, required=False, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=24, help='input sequence length')
parser.add_argument('--label_len', type=int, default=0, help='start token length')
parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')


# DLinear
parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')
# Formers 
#Michi: bisher nur default=3 zum laufen gebracht
parser.add_argument('--embed_type', type=int, default=3, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=V*2, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
parser.add_argument('--dec_in', type=int, default=V, help='decoder input size')
parser.add_argument('--c_out', type=int, default=V, help='output size')
parser.add_argument('--d_model', type=int, default=d_model, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=e_layers, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=d_layers, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=d_ff, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

args = parser.parse_args(args=[])
import importlib


import models.DenseAutoregressive as autoformer
importlib.reload(autoformer)
model = autoformer.Model(args).cuda()

lr, batch_size, samples_per_epoch, patience = 0.0005, 32, 102400, 6
d, N, he, dropout = 50, 2, 4, 0.2
number_of_epochs = 600


optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3)
# Pretrain fore_model.
best_val_loss = np.inf
N_fore = len(fore_train_op)
#print(N_fore)
fore_savepath = 'eicu_mod10_Dense_IMS_SF_BP.pytorch'
loss_func = torch.nn.MSELoss(reduction="none")

# torch.compile(model)
for e in range(number_of_epochs):
    e_indices = np.random.choice(range(N_fore), size=samples_per_epoch, replace=False)
    e_loss = 0
    pbar = tqdm(range(0, len(e_indices), batch_size))
    model.train()
    for start in pbar:
        ind = e_indices[start:start+batch_size]

        matrix = torch.tensor(fore_train_op[ind], dtype=torch.float32).cuda()
        #torch.Size([32, 48, 258])
        input_matrix = matrix[:, :24, :V*2]
        #torch.Size([32, 24, 129])
        input_mark = torch.arange(0, input_matrix.size(1)).unsqueeze(0).repeat(input_matrix.size(0), 1).cuda()
        #torch.Size([32, 24])
        input_mask = matrix[:, :24, V:]
        #torch.Size([32, 24, 129])
        output_matrix = matrix[:, 24:, :V]
        #torch.Size([32, 24, 129])
        
        output_mark = torch.arange(0, output_matrix.size(1)).unsqueeze(0).repeat(input_matrix.size(0), 1).cuda()
        #torch.Size([32, 24])
        output_mask = matrix[:, 24:, V:]
        #torch.Size([32, 24, 129])
        dec_inp = torch.zeros_like(output_matrix[:, -args.pred_len:, :]).float()
        if 'Linear' not in str(type(model)) and "Autoregressive" in str(type(model)):
            output = model(input_matrix, input_mark, dec_inp, output_mark, tgt=output_matrix, trainn=False, backprop=True)#, enc_self_mask=input_mask, dec_self_mask=output_mask)
        elif "Linear" not in str(type(model)):
            output = model(input_matrix, input_mark, dec_inp, output_mark)#, enc_self_mask=input_mask, dec_self_mask=output_mask)

        else:
            output = model(input_matrix)[:, :, :V]
        loss = output_mask[:, -args.pred_len:, :]*(
        output-output_matrix[:, -args.pred_len:, :])**2
        loss = loss.sum(axis=-1).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        e_loss += loss.detach()
        pbar.set_description('%f' % e_loss)
    val_loss = 0
    #raise Exception
    loss_list = []
    mask_list = []
    model.eval()
    pbar = tqdm(range(0, len(fore_valid_op), batch_size))  # len(fore_valid_op)           ####################   maybe also batch_size instead of 32
    for start in pbar:
        matrix = torch.tensor(fore_valid_op[start:start+batch_size], dtype=torch.float32).cuda()
        #torch.Size([32, 48, 258])
        input_matrix = matrix[:, :24, :V*2]
        #torch.Size([32, 24, 129])
        input_mark = torch.arange(0, input_matrix.size(1)).unsqueeze(0).repeat(input_matrix.size(0), 1).cuda()
        #torch.Size([32, 24])
        input_mask = matrix[:, :24, V:]
        #torch.Size([32, 24, 129])
        output_matrix = matrix[:, 24:, :V]
        #torch.Size([32, 24, 129])
        output_mark = torch.arange(0, output_matrix.size(1)).unsqueeze(0).repeat(input_matrix.size(0), 1).cuda()
        #torch.Size([32, 24])
        output_mask = matrix[:, 24:, V:]
        #torch.Size([32, 24, 129])
        dec_inp = torch.zeros_like(output_matrix[:, -args.pred_len:, :]).float()
        if 'Linear' not in str(type(model)) and "Autoregressive" in str(type(model)):
            output = model(input_matrix, input_mark, dec_inp, output_mark, trainn=False)#, enc_self_mask=input_mask, dec_self_mask=output_mask)
        elif "Linear" not in str(type(model)):
            output = model(input_matrix, input_mark, dec_inp, output_mark)#, enc_self_mask=input_mask, dec_self_mask=output_mask)

        else:
            output = model(input_matrix)[:, :, :V]
        loss = output_mask[:, -args.pred_len:, :]*(
        output-output_matrix[:, -args.pred_len:, :])**2
        loss_list.extend(loss.sum(axis=-1).mean(axis=-1).detach().cpu().tolist())
        mask_list.extend(output_mask[:, -args.pred_len:, :].sum(axis=-1).mean(axis=-1).detach().cpu().tolist())
        loss = loss.sum(axis=-1).mean()
        val_loss += loss.detach().cpu()
        pbar.set_description('%f' % val_loss)
    loss_p = e_loss*batch_size/samples_per_epoch
    val_loss_p = val_loss*batch_size/len(fore_valid_op)

    print('Epoch ', e, 'loss', loss_p.item(), 'val loss', val_loss_p.item(), "mean ", np.mean(loss_list), "std ", np.std(loss_list))
    entry = 'Model 10 Epoch ' +str(e) + ' loss ' + str(loss_p.item()) + ' val loss ' + str(val_loss_p.item()) + " mean " + str(np.mean(loss_list)) +' std '+ str(np.std(loss_list)) + '\n'
    with open('epoch_dense_log.txt','a') as f:
        f.write(entry)
    with open('loss_values_dense_log', 'a') as f:
        f.write(str(e)+' ' + str(loss_p.item()) + ' ' + str(val_loss_p.item())+ '\n')
    scheduler.step(val_loss_p.item())
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), fore_savepath)
        best_epoch = e
    if (e-best_epoch) > patience:
        break
    
print('Training has ended.')

print('End time: ', datetime.now())

#Informer IMS Sampling Detach
