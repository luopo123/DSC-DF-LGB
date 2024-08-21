## python 3.7
import os
import sklearn.ensemble
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from deepforest import CascadeForestRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
import numpy as np
from evaluation import cal
import joblib
import matplotlib.pyplot as plt
import torch
from torch import nn
from CNN import DSC

import warnings
warnings.filterwarnings('ignore')

root_path = "F:\\China_XCO2\\"
model_path = 'checkpoint\\DF-LGB_after_CNN'
dataset_path = root_path + "dataset_v2\\"
total_path = dataset_path + "total\\"
CNN_path = 'checkpoint\\DSC_checkpoint.tar'
CNNinputdata = np.load(total_path + "CNN_inputdata.npy")
inputdata = np.load(total_path + "inputdata.npy")
pointDataset = np.load(total_path + "outputdata.npy")
inputdata = inputdata.astype(np.float32)
# [CAMS, dem, r, t, blh, sp, tp, lat, lon, year, doy]

# 判断是否有可用GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"There are {torch.cuda.device_count()} GPU(s) available.")
    print("Device name:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("No GPU available, using the CPU instead.")

CNNinputdata = torch.tensor(CNNinputdata, dtype=torch.float32).to(device)
model = DSC(4).to(device)
checkpoint = torch.load(CNN_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
CNN_feature = model.get_feature(CNNinputdata)
CNN_feature = CNN_feature.to('cpu').detach().numpy()

inputdata = np.concatenate((inputdata, CNN_feature), axis=1)

print(inputdata.shape)

# X_train, X_test, Y_train, Y_test = train_test_split(inputdata, pointDataset, test_size=0.1)

# model = CascadeForestRegressor(use_predictor=True)
# model.set_predictor(LGBMRegressor())
# model.fit(X_train, Y_train)

# '''==================train==================='''
#
# Y_train_p = model.predict(X_train)
#
# train_R2 = cal.cal_R2(Y_train_p, Y_train)
# print('train R2:', train_R2)
#
# train_MAE_loss = cal.cal_MAE(Y_train_p, Y_train)
# print('train MAE:', train_MAE_loss)
#
# train_RMSE_loss = cal.cal_RMSE(Y_train_p, Y_train)
# print('train RMSE:', train_RMSE_loss)
#
# train_MAPE = cal.cal_MAPE(Y_train_p,Y_train)
# print('train MAPE:', train_MAPE)
#
# '''===================test===================='''
#
# predict = model.predict(X_test)
#
# test_R2 = cal.cal_R2(predict, Y_test)
# print('test R2:', test_R2)
#
# test_MAE_loss = cal.cal_MAE(predict, Y_test)
# print('test MAE:', test_MAE_loss)
#
# test_RMSE_loss = cal.cal_RMSE(predict, Y_test)
# print('test RMSE:', test_RMSE_loss)
#
# test_MAPE = cal.cal_MAPE(predict, Y_test)
# print('test MAPE:', test_MAPE)

# 10折交叉验证
X = inputdata
Y = pointDataset
R2s = list()
MAEs = list()
RMSEs = list()
MAPEs = list()
n = 0
kfold = KFold(n_splits=10, shuffle=True)

for train_index, test_index in kfold.split(X, Y):
    n = n + 1
    print("第{}次".format(n))
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    # model = LGBMRegressor()
    model = CascadeForestRegressor(use_predictor=True)
    # model.set_predictor(RandomForestRegressor())
    model.set_predictor(LGBMRegressor())
    # model.set_predictor(ExtraTreesRegressor())
    model.fit(X_train, Y_train)
    model.save(model_path + "_" + str(n))
    predict = model.predict(X_test)

    R2 = cal.cal_R2(predict, Y_test)
    MAE = cal.cal_MAE(predict, Y_test)
    RMSE = cal.cal_RMSE(predict, Y_test)
    MAPE = cal.cal_MAPE(predict, Y_test)
    print("R2={0}, MAE={1}, RMSE={2}, MAPE={3}".format(R2, MAE, RMSE, MAPE))

    R2s.append(R2)
    MAEs.append(MAE)
    RMSEs.append(RMSE)
    MAPEs.append(MAPE)
    del model

for k in range(len(R2s)):
    print("Fold {} R2: {:.4f}".format(k + 1, R2s[k]))
R2s = np.array(R2s)
print("     Average R2: {:.4f}".format(R2s.mean()))

for k in range(len(MAEs)):
    print("Fold {} MAE: {:.4f}".format(k + 1, MAEs[k]))
MAEs = np.array(MAEs)
print("     Average MAE: {:.4f}".format(MAEs.mean()))

for k in range(len(RMSEs)):
    print("Fold {} RMSE: {:.4f}".format(k + 1, RMSEs[k]))
RMSEs = np.array(RMSEs)
print("     Average RMSE: {:.4f}".format(RMSEs.mean()))

for k in range(len(MAPEs)):
    print("Fold {} MAPE: {:.6f}".format(k + 1, MAPEs[k]))
MAPEs = np.array(MAPEs)
print("     Average MAPE: {:.6f}".format(MAPEs.mean()))

