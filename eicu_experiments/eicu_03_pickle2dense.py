import pickle
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
    if len(x) > 3*880:
        print('bigger than 880*3', len(x))
    return x+[0]*(fore_max_len-len(x))


## for train and val set:
# Read data.
data_path = '/home/mitarb/fracarolli/eicu/final/eicu_preprocessed_n.pkl'
data, train_ind, valid_ind, test_ind = joblib.load(open(data_path, 'rb'))
# Remove test patients.
test_sub = data.loc[data.ts_ind.isin(test_ind)].ts_ind.unique()
data = data.loc[~data.ts_ind.isin(test_sub)]

# Get static data with mean fill and missingness indicator.
static_varis = ['i CCU-CTICU', 'i CSICU', 'i CTICU', 'i Cardiac ICU', 'i MICU', 'i Med-Surg ICU', 'i Neuro ICU', 'i SICU', 
                's Admissionheight', 's Admissionweight', 's Age', 's Gender', 's Hospitaladmitoffset', 's Hospitaladmittime24', 
                's Hospitaldischargeyear', 's Patienthealthsystemstayid', 's Unitvisitnumber']  # 17
ii = data.variable.isin(static_varis)
static_data = data.loc[ii]
data = data.loc[~ii]  # ~ binary flip
# print('data\n',data)

static_var_to_ind = inv_list(static_varis)
D = len(static_varis)  # 17 demographic variables
N = data.ts_ind.max()+1  # 77.704 number of stays
demo = np.zeros((int(N), int(D)))
for row in tqdm(static_data.itertuples()):
    demo[int(row.ts_ind), static_var_to_ind[row.variable]] = row.value
# Normalize static data.
means = demo.mean(axis=0, keepdims=True)  # quite sparse
stds = demo.std(axis=0, keepdims=True)
stds = (stds == 0)*1 + (stds != 0)*stds
demo = (demo-means)/stds
# Get variable indices.
varis = sorted(list(set(data.variable)))
V = len(varis)
print('varis', varis, V)
var_to_ind = inv_list(varis, start=1)
data['vind'] = data.variable.map(var_to_ind)
data = data[['ts_ind', 'vind', 'hour', 'value']].sort_values(by=['ts_ind', 'vind', 'hour'])
# Find max_len.
fore_max_len = 2640  # hard coded max_len of vars
# Get forecast inputs and outputs.
fore_times_ip, fore_values_ip, fore_varis_ip = [], [], []
fore_op, fore_op_awesome, fore_inds = [], [], []
for w in tqdm(range(25, 124, 4)):
    pred_data = data.loc[(data.hour>=w)&(data.hour<=w+24)]
    pred_data = pred_data.groupby(['ts_ind', 'vind']).agg({'value':'first'}).reset_index()
    pred_data['vind_value'] = pred_data[['vind', 'value']].values.tolist()
    pred_data = pred_data.groupby('ts_ind').agg({'vind_value':list}).reset_index()
    pred_data['vind_value'] = pred_data['vind_value'].apply(f)   

    obs_data = data.loc[(data.hour < w) & (data.hour >= w-24)]
    obs_data = obs_data.loc[obs_data.ts_ind.isin(pred_data.ts_ind)]
    obs_data = obs_data.groupby('ts_ind').head(fore_max_len)
    obs_data = obs_data.groupby('ts_ind').agg({'vind': list, 'hour': list, 'value': list}).reset_index()
    for pred_window  in range(-24, 24, 1):
        pred_data = data.loc[(data.hour >= w+pred_window) & (data.hour <= w+1 +pred_window)]
        pred_data = pred_data.groupby(['ts_ind', 'vind']).agg({'value': 'first'}).reset_index()
        pred_data['vind_value'+str(pred_window)] = pred_data[['vind', 'value']].values.tolist()
        pred_data = pred_data.groupby('ts_ind').agg({'vind_value'+str(pred_window): list}).reset_index()
        pred_data['vind_value'+str(pred_window)] = pred_data['vind_value'+str(pred_window)].apply(f)  # 721 entries with 2*129 vind_values
        obs_data = obs_data.merge(pred_data, on='ts_ind')

    for col in ['vind', 'hour', 'value']:
        obs_data[col] = obs_data[col].apply(pad)
    fore_op_awesome.append(np.array(list([list(obs_data['vind_value'+str(pred_window)]) for pred_window in range(-24, 24, 1)])))
    fore_inds.append(np.array([int(x) for x in list(obs_data.ts_ind)]))
    fore_times_ip.append(np.array(list(obs_data.hour)))
    fore_values_ip.append(np.array(list(obs_data.value)))
    fore_varis_ip.append(np.array(list(obs_data.vind)))
    
del data
fore_times_ip = np.concatenate(fore_times_ip, axis=0)
fore_values_ip = np.concatenate(fore_values_ip, axis=0)
fore_varis_ip = np.concatenate(fore_varis_ip, axis=0)

fore_op_awesome = np.concatenate(fore_op_awesome, axis=1)
fore_op = np.swapaxes(fore_op_awesome, 0, 1)
print(fore_op.shape)

fore_inds = np.concatenate(fore_inds, axis=0)
fore_demo = demo[fore_inds]
# Generate sets of inputs and outputs.
train_ind = np.argwhere(np.in1d(fore_inds, train_ind)).flatten()
valid_ind = np.argwhere(np.in1d(fore_inds, valid_ind)).flatten()
fore_train_ip = [ip[train_ind] for ip in [fore_demo, fore_times_ip, fore_values_ip, fore_varis_ip]]
fore_valid_ip = [ip[valid_ind] for ip in [fore_demo, fore_times_ip, fore_values_ip, fore_varis_ip]]
del fore_times_ip, fore_values_ip, fore_varis_ip, demo, fore_demo
fore_train_op = fore_op[train_ind]
fore_valid_op = fore_op[valid_ind]
del fore_op

joblib.dump(fore_train_op, 'fore_train_op_dense.pkl')
joblib.dump(fore_valid_op, 'fore_valid_op_dense.pkl')
joblib.dump(fore_train_ip, 'fore_train_ip_dense.pkl')
joblib.dump(fore_valid_ip, 'fore_valid_ip_dense.pkl')

# print('lengths of rem_sub, fore_train_ip[1], fore_valid_ip[0]')
# print(len(rem_sub), fore_train_ip[1].shape, fore_valid_ip[0].shape)
# 214 min



## for test sets
# Read data.
data_path = '/home/mitarb/fracarolli/eicu/final/eicu_preprocessed_n.pkl'
data, train_ind, valid_ind, test_ind = joblib.load(open(data_path, 'rb'))

# Only test patients
data = data.loc[data.ts_ind.isin(test_ind)]
data = data.loc[(data.hour>=0) & (data.hour<=48)]

means_stds = data.groupby("variable").agg({"mean":"first", "std":"first"})
mean_std_dict = dict()
for pos, row in means_stds.iterrows():
    mean_std_dict[pos] = (float(row["mean"]), float(row["std"]))
joblib.dump(mean_std_dict, 'mean_std_dict.pkl')

# Get static data with mean fill and missingness indicator.
static_varis = ['i CCU-CTICU', 'i CSICU', 'i CTICU', 'i Cardiac ICU', 'i MICU', 'i Med-Surg ICU', 'i Neuro ICU', 'i SICU', 
                's Admissionheight', 's Admissionweight', 's Age', 's Gender', 's Hospitaladmitoffset', 's Hospitaladmittime24', 
                's Hospitaldischargeyear', 's Patienthealthsystemstayid', 's Unitvisitnumber']  # 17
ii = data.variable.isin(static_varis)
static_data = data.loc[ii]
data = data.loc[~ii]  # ~ binary flip

static_var_to_ind = inv_list(static_varis)
D = len(static_varis)  # 17 demographic variables
N = data.ts_ind.max()+1  # 77.704 number of stays
demo = np.zeros((int(N), int(D)))
for row in tqdm(static_data.itertuples()):
    demo[int(row.ts_ind), static_var_to_ind[row.variable]] = row.value

# Normalize static data.
means = demo.mean(axis=0, keepdims=True)  # quite sparse
stds = demo.std(axis=0, keepdims=True)
stds = (stds == 0)*1 + (stds != 0)*stds
demo = (demo-means)/stds

# Get variable indices.
varis = sorted(list(set(data.variable)))
V = len(varis)
print('V', V, 'varis', varis)
var_to_ind = inv_list(varis, start=1)
data['vind'] = data.variable.map(var_to_ind)
data = data[['ts_ind', 'vind', 'hour', 'value']].sort_values(by=['ts_ind', 'vind', 'hour'])
# Find max_len.
fore_max_len = 880*3  # hard coded max_len of vars
# Get forecast inputs and outputs.
fore_times_ip, fore_values_ip, fore_varis_ip = [], [], []
fore_op, fore_op_awesome, fore_inds = [], [], []
pred_data = data.loc[(data.hour>=24)&(data.hour<=48)]
pred_data = pred_data.groupby(['ts_ind', 'vind']).agg({'value':'first'}).reset_index()
pred_data['vind_value'] = pred_data[['vind', 'value']].values.tolist()
pred_data = pred_data.groupby('ts_ind').agg({'vind_value':list}).reset_index()
pred_data['vind_value'] = pred_data['vind_value'].apply(f)   

obs_data = data.loc[(data.hour < 24) & (data.hour >= 0)]
obs_data = obs_data.loc[obs_data.ts_ind.isin(pred_data.ts_ind)]
obs_data = obs_data.groupby('ts_ind').head(fore_max_len)
obs_data = obs_data.groupby('ts_ind').agg({'vind': list, 'hour': list, 'value': list}).reset_index()
# Take 24 hours before and after a fixed timepoint (w=24 hours)
for pred_window  in range(-24, 24, 1):
    pred_data = data.loc[(data.hour >= 24+pred_window) & (data.hour <= 25 + pred_window)]
    pred_data = pred_data.groupby(['ts_ind', 'vind']).agg({'value': 'first'}).reset_index()
    pred_data['vind_value'+str(pred_window)] = pred_data[['vind', 'value']].values.tolist()
    pred_data = pred_data.groupby('ts_ind').agg({'vind_value'+str(pred_window): list}).reset_index()
    pred_data['vind_value'+str(pred_window)] = pred_data['vind_value'+str(pred_window)].apply(f)  # 721 entries with 2*129 vind_values
    obs_data = obs_data.merge(pred_data, on='ts_ind')

for col in ['vind', 'hour', 'value']:
    obs_data[col] = obs_data[col].apply(pad)
fore_op_awesome.append(np.array(list([list(obs_data['vind_value'+str(pred_window)]) for pred_window in range(-24, 24, 1)])))
fore_inds.append(np.array([int(x) for x in list(obs_data.ts_ind)]))
fore_times_ip.append(np.array(list(obs_data.hour)))
fore_values_ip.append(np.array(list(obs_data.value)))
fore_varis_ip.append(np.array(list(obs_data.vind)))
    
del data
fore_times_ip = np.concatenate(fore_times_ip, axis=0)
fore_values_ip = np.concatenate(fore_values_ip, axis=0)
fore_varis_ip = np.concatenate(fore_varis_ip, axis=0)

fore_op_awesome = np.concatenate(fore_op_awesome, axis=1)
fore_op = np.swapaxes(fore_op_awesome, 0, 1)
print(fore_op.shape)

fore_inds = np.concatenate(fore_inds, axis=0)
fore_demo = demo[fore_inds]

# Generate 3 sets of inputs and outputs.
test_ind = np.argwhere(np.in1d(fore_inds, test_ind)).flatten()
fore_test_ip = [ip[test_ind] for ip in [fore_demo, fore_times_ip, fore_values_ip, fore_varis_ip]]
del fore_times_ip, fore_values_ip, fore_varis_ip, demo, fore_demo
fore_test_op = fore_op[test_ind]
del fore_op

joblib.dump(fore_test_op, 'fore_test_op_dense.pkl')
joblib.dump(fore_test_ip, 'fore_test_ip_dense.pkl')
