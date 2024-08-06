import numpy as np
import pickle


data_path = '../240130_Preprocess_MIMIC3/mimic_iii_preprocessed.pkl'
data, oc, train_ind, valid_ind, test_ind = pickle.load(open(data_path, 'rb'))

all_ts_ind = np.concatenate((train_ind, valid_ind, test_ind))
datain = data[data['ts_ind'].isin(all_ts_ind)]  # 

AB = datain[datain['variable']=='Antibiotics']  # 1 490 822
BC = datain[datain['variable']=='Blood Culture'] # 57 270

AB_1d = AB[AB['hour']<24]  # 177 728
AB_1d = AB_1d.groupby('ts_ind').count().reset_index()
AB_1d = AB_1d[['ts_ind','variable']]  # 8488
AB_1d = AB_1d.rename(columns={'variable':"AB"})

BC_1d = BC[BC['hour']<24]  # 16 190
BC_1d = BC_1d.groupby('ts_ind').count().reset_index()
BC_1d = BC_1d[['ts_ind','variable']]  # 10 688
BC_1d = BC_1d.rename(columns={'variable':"BC"})

AB_1d.merge(BC_1d, on='ts_ind', how='outer')  # 15357 mit mind. einem
ABC = AB_1d.merge(BC_1d, on='ts_ind') # 3819 mit beiden

infec_train = ABC[ABC['ts_ind'].isin(train_ind)]  # 2483
infec_test = ABC[ABC['ts_ind'].isin(test_ind)]  # 743
infec_valid = ABC[ABC['ts_ind'].isin(valid_ind)]  # 593

np.savetxt('infec_train.csv',infec_train.ts_ind)
np.savetxt('infec_test.csv',infec_test.ts_ind)
np.savetxt('infec_valid.csv',infec_valid.ts_ind)