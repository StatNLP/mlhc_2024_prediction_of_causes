#!/usr/bin/env python
# coding: utf-8

# *ToDo*
# 
# 02) pred_window und obs_window ausprobieren.
# 03) output nochmal genau anschauen.
# 04) was ist maskiert anschauen - und wie genau?
# 05) VRAM wenig ausgelastet. Batch size mal mit 320 ausprobieren, aber auch durchziehen.
# 06) Task pred: scheduler einfügen.
# 07) Analyse each var (sparsity)
# 08) loss per var/quality of varis loss (is this already ablation to forecast only one var)
# 09) Include sepsis definition
# 10) 

# In[1]:


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


# In[2]:
sample_divisor = 1

mean_std_dict = joblib.load(open('/home/mitarb/fracarolli/eicu/final/mean_std_dict.pkl', 'rb'))
var_to_ind = dict()
with open("/home/mitarb/fracarolli/eicu/final/var_to_ind.csv") as file:
    for line in file:
        key, value = line.strip().split(",")
        var_to_ind[key]=int(value)

fore_train_op = joblib.load(open('/home/mitarb/fracarolli/eicu/240401_Experimente/fore_train_op_dense.pkl', 'rb'))
fore_valid_op = joblib.load(open('/home/mitarb/fracarolli/eicu/240401_Experimente/fore_valid_op_dense.pkl', 'rb'))
fore_train_ip = joblib.load(open('/home/mitarb/fracarolli/eicu/240401_Experimente/fore_train_ip_dense.pkl', 'rb'))
fore_valid_ip = joblib.load(open('/home/mitarb/fracarolli/eicu/240401_Experimente/fore_valid_ip_dense.pkl', 'rb'))
d_model, e_layers, d_layers, d_ff = 256, 2, 2, 2048
number_of_epochs = 100
V, D = 98, 17
fore_max_len = 2640
print('loading of joblib completed')


# ## Load forecast dataset into matrices.

# In[ ]:


def get_sofa(matrix, var_to_ind): #24xV matrix
    # GCS: min_eye, min_motor, min_verbal = 5, 5, 5
    print(matrix.size())
    #raise Exception
    key ="GCS eye"
    var_to_ind = {x:i-1 for x,i in var_to_ind.items()}
    a=matrix[:, var_to_ind[key]][torch.nonzero(matrix[:, var_to_ind[key]], as_tuple=True)]
    print(a)
    min_eye = min(matrix[:, var_to_ind[key]][torch.nonzero(matrix[:, var_to_ind[key]], as_tuple=True)]*mean_std_dict[key][1]+mean_std_dict[key][0], default=4)
    key = "GCS motor"
    min_motor = min(matrix[:, var_to_ind[key]][torch.nonzero(matrix[:, var_to_ind[key]], as_tuple=True)]*mean_std_dict[key][1]+mean_std_dict[key][0], default=6)
    key = "GCS verbal"
    min_verbal = min(matrix[:, var_to_ind[key]][torch.nonzero(matrix[:, var_to_ind[key]], as_tuple=True)]*mean_std_dict[key][1]+mean_std_dict[key][0], default=5)
    

    GCS = min_eye + min_motor + min_verbal
    if GCS > 14: GCS_sofa = 0
    elif GCS > 12: GCS_sofa = 1
    elif GCS > 9:  GCS_sofa = 2
    elif GCS > 5:  GCS_sofa = 3
    else: GCS_sofa = 4
    #print('GCS_sofa is', GCS_sofa, ';     GCS is', GCS,'; GCS eye', min_eye, '; GCS motor', min_motor, '; GCS verbal', min_verbal)

    key = "Bilirubin (Total)"
    bilir = max(matrix[:, var_to_ind[key]][torch.nonzero(matrix[:, var_to_ind[key]], as_tuple=True)]*mean_std_dict[key][1]+mean_std_dict[key][0], default=1)
    if bilir > 12: bilir_sofa = 4
    elif bilir > 6: bilir_sofa = 3
    elif bilir > 2: bilir_sofa = 2
    elif bilir > 1.2: bilir_sofa = 1
    else: bilir_sofa = 0
    #print('bilir_Sofa is', bilir_sofa, ';   bilirubin is', bilir)
    
    # Coagulation (Platelets)
    key = "Platelets"
    plate = max(matrix[:, var_to_ind[key]][torch.nonzero(matrix[:, var_to_ind[key]], as_tuple=True)]*mean_std_dict[key][1]+mean_std_dict[key][0], default=160)
    if plate > 150: plate_sofa = 0
    elif plate > 100: plate_sofa = 1
    elif plate > 50: plate_sofa = 2
    elif plate > 20: plate_sofa = 3
    else: plate_sofa = 4
    #print('plate_sofa is', plate_sofa, ';   platelet count is', plate)
    
    # print('Urinmenge 24h', sum(data_var[data_var['variable']=='Urine']['value2']))

    key = "Urine"
    urine = sum(matrix[:, var_to_ind[key]][torch.nonzero(matrix[:, var_to_ind[key]], as_tuple=True)]*mean_std_dict[key][1]+mean_std_dict[key][0])
    key = "Creatinine (Blood)"
    creat = max(matrix[:, var_to_ind[key]][torch.nonzero(matrix[:, var_to_ind[key]], as_tuple=True)]*mean_std_dict[key][1]+mean_std_dict[key][0], default=1)
    
    if (urine < 200) or (creat > 5): renal_sofa = 4
    elif  (urine < 500) or (creat > 3.5): renal_sofa = 3
    elif creat > 2.0: renal_sofa = 2
    elif creat > 1.2: renal_sofa = 1
    else: renal_sofa = 0
    #print('renal_sofa:',renal_sofa,';       urine 24:',urine,'; creat:', creat)
    
    CS_data = get_CS(matrix, var_to_ind)
    cs_sofa = CS_SOFA(CS_data)
    
    #cs_sofa = 0
    key="FiO2"
    fio2 = (matrix[:, var_to_ind[key]]*mean_std_dict[key][1]+mean_std_dict[key][0])
    key="PaO2"
    po2 = (matrix[:, var_to_ind[key]]*mean_std_dict[key][1]+mean_std_dict[key][0])
    PaO2FiO2 = 100*po2/fio2
    print("size", PaO2FiO2.size())
    PaO2FiO2 = PaO2FiO2[torch.nonzero(PaO2FiO2, as_tuple=True)]
    pao2fio2 = min(PaO2FiO2)
    if pao2fio2<100: resp=4
    elif pao2fio2<200: resp=3
    elif pao2fio2<300:resp=2
    elif pao2fio2<400:resp=1
    else: resp=0
    return GCS_sofa, cs_sofa, resp, plate_sofa, bilir_sofa, renal_sofa

def get_CS(matrix, var_to_ind):
    #data_var = data_pat[data_pat['variable'].isin(['Dobutamine','Dopamine','Epinephrine','Norepinephrine','Weight'])]
    #data_var['value2'] = data_var['value']*data_var['std']+data_var['mean']
    
    #weight = min(data_var[data_var['variable']=='Weight']['value2'], default=80)  # set default weight to 80kg.
    key = "Weight"
    weight = 80# min(matrix[:, var_to_ind[key]][torch.nonzero(matrix[:, var_to_ind[key]], as_tuple=True)], default=80)#*mean_std_dict[key][1]+mean_std_dict[key][0], default=1)
    key = "d Dopamine ratio"
    try:
        data_dop = max(matrix[:, var_to_ind[key]][torch.nonzero(matrix[:, var_to_ind[key]], as_tuple=True)]*mean_std_dict[key][1]+mean_std_dict[key][0], default=1)
        data_dop = data_dop /60/weight*1000
    except:
        data_dop = 0

    key = "d Dobutamine ratio"
    
    try:
        data_dobu = max(matrix[:, var_to_ind[key]][torch.nonzero(matrix[:, var_to_ind[key]], as_tuple=True)]*mean_std_dict[key][1]+mean_std_dict[key][0], default=1)
        data_dobu = data_dobu  /60/weight*1000
    except:
        data_dobu = 0
    key = "d Epinephrine ratio"
    
    try:
        data_epi = max(matrix[:, var_to_ind[key]][torch.nonzero(matrix[:, var_to_ind[key]], as_tuple=True)]*mean_std_dict[key][1]+mean_std_dict[key][0], default=1)
        data_epi = data_epi  /60/weight*1000
    except:
        data_epi = 0
    key = "d Norepinephrine ratio"
    
    try:
        data_nore = max(matrix[:, var_to_ind[key]][torch.nonzero(matrix[:, var_to_ind[key]], as_tuple=True)]*mean_std_dict[key][1]+mean_std_dict[key][0], default=1)
        data_nore = data_nore /60/weight*1000
    except:
        data_nore = 0
        


    key = "SBP"
    SBP = (matrix[:, var_to_ind[key]]*mean_std_dict[key][1]+mean_std_dict[key][0])

    key = "DBP"
    DBP = (matrix[:, var_to_ind[key]]*mean_std_dict[key][1]+mean_std_dict[key][0])

    MAP = 2/3 * DBP + 1/3 * SBP
    MAP = min(MAP[torch.nonzero(MAP, as_tuple=True)], default=100)
                 
        
    return MAP, data_dop, data_dobu, data_epi, data_nore 
    
def CS_SOFA(data):
    map = data[0]
    dop, dobu, epi, nore = data[1:5]
    # print('CS data: mdden', data)
    if (dop > 15) or (epi > 0.1) or (nore > 0.01): CS = 4
    elif (dop > 5) or (epi > 0) or (nore > 0): CS = 3
    elif (dop > 0) or (dobu > 0): CS = 2
    elif map < 70: CS = 1
    else: CS = 0
    # print('CS Sofa is:', CS)
    return CS 
    
factor_keys =["GCS eye", "GCS motor", "GCS verbal", "Bilirubin (Total)", "Platelets", "Urine", "Creatinine (Blood)", "FiO2", "PaO2", "d Dopamine ratio", "d Dobutamine ratio",
"d Epinephrine ratio", "d Norepinephrine ratio", "SBP", "DBP"]


# In[ ]:


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


# In[ ]:


import models.InformerAutoregressiveFullRegression as autoformer
importlib.reload(autoformer)
model = autoformer.Model(args).cuda()

lr, batch_size, samples_per_epoch, patience = 0.0005, 32, int(102400/sample_divisor), 6
d, N, he, dropout = 50, 2, 4, 0.2
V=131
print('number of parameters: ', V)


#a = summary(model, [(32, 2), (32, 880), (32, 880), (32, 880)],  # shape of fore_train_ip
#            dtypes=[torch.float, torch.float, torch.float, torch.long])
#print(a)  # Model summary
# raise Exception
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3)
# Pretrain fore_model.
best_val_loss = np.inf
N_fore = len(fore_train_op)
#print(N_fore)
fore_savepath = 'informer_regression_eicu.pytorch'
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
        input_matrix = matrix[:, :24, :98*2]
        #torch.Size([32, 24, 129])
        input_mark = torch.arange(0, input_matrix.size(1)).unsqueeze(0).repeat(input_matrix.size(0), 1).cuda()
        #torch.Size([32, 24])
        input_mask = matrix[:, :24, 98:]
        #torch.Size([32, 24, 129])
        output_matrix = matrix[:, 24:, :98]
        #torch.Size([32, 24, 129])
        
        output_mark = torch.arange(0, output_matrix.size(1)).unsqueeze(0).repeat(input_matrix.size(0), 1).cuda()
        #torch.Size([32, 24])
        output_mask = matrix[:, 24:, 98:]
        #torch.Size([32, 24, 129])
        dec_inp = torch.zeros_like(output_matrix[:, -args.pred_len:, :]).float()
        #print(dec_inp.size())
        #dec_inp = torch.cat([output_matrix[:, :args.label_len, :], dec_inp], dim=1).float().cuda()
        #raise Exception
        #print(str(type(model)))
        #print(model)

        output = model(input_matrix, input_mark, dec_inp, output_mark, tgt=output_matrix, trainn=False, backprop=True)#, enc_self_mask=input_mask, dec_self_mask=output_mask)
        #print(output.size())
        loss = 0
        for outpu, real, mask, input_mat in zip(output, output_matrix, output_mask, input_matrix):
            b = sum(get_sofa(real, var_to_ind))
            loss_term = (outpu-b)**2
            loss+=loss_term
            #print(loss_term)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        e_loss += loss.detach()
        pbar.set_description('%f' % e_loss)
    val_loss = 0
    #raise Exception
    loss_list = []
    model.eval()
    pbar = tqdm(range(0, len(fore_valid_op), 32))  # len(fore_valid_op)           ####################   maybe also batch_size instead of 32
    for start in pbar:
        matrix = torch.tensor(fore_valid_op[start:start+batch_size], dtype=torch.float32).cuda()
        #torch.Size([32, 48, 258])
        input_matrix = matrix[:, :24, :98*2]
        #torch.Size([32, 24, 129])
        input_mark = torch.arange(0, input_matrix.size(1)).unsqueeze(0).repeat(input_matrix.size(0), 1).cuda()
        #torch.Size([32, 24])
        input_mask = matrix[:, :24, 98:]
        #torch.Size([32, 24, 129])
        output_matrix = matrix[:, 24:, :98]
        #torch.Size([32, 24, 129])
        output_mark = torch.arange(0, output_matrix.size(1)).unsqueeze(0).repeat(input_matrix.size(0), 1).cuda()
        #torch.Size([32, 24])
        output_mask = matrix[:, 24:, 98:]
        #torch.Size([32, 24, 129])
        dec_inp = torch.zeros_like(output_matrix[:, -args.pred_len:, :]).float()
        #print(dec_inp.size())
        #dec_inp = torch.cat([output_matrix[:, :args.label_len, :], dec_inp], dim=1).float().cuda()
        #raise Exception
        #print(repr(model))

        output = model(input_matrix, input_mark, dec_inp, output_mark, trainn=False)#, enc_self_mask=input_mask, dec_self_mask=output_mask)
        loss = 0
        for outpu, real, mask, input_mat in zip(output, output_matrix, output_mask, input_matrix):
            b = sum(get_sofa(real, var_to_ind))
            loss_term = (outpu-b)**2
            loss+=loss_term
        val_loss += loss.detach().cpu()
        pbar.set_description('%f' % val_loss)
    loss_p = e_loss*batch_size/samples_per_epoch
    val_loss_p = val_loss*batch_size/len(fore_valid_op)
    print('Epoch', e, 'loss', loss_p, 'val loss', val_loss_p, "mean and std", np.mean(loss_list), np.std(loss_list))
    with open('loss_values_log_regression_eicu', 'a') as f:
        f.write(str(e)+' ' + str(loss_p.item()) + ' ' + str(val_loss_p.item())+ '\n')
    scheduler.step(val_loss_p.item())
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), fore_savepath)
        best_epoch = e
    if (e-best_epoch) > patience:
        break
    
print('Training has ended.')

#Informer IMS, Sampling, Backprop


# In[ ]:




