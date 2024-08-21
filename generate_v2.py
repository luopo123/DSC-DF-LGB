import os
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from deepforest import CascadeForestRegressor
from lightgbm import LGBMRegressor
import os
import time
import datetime
import torch
from torch import nn
from train.CNN import DSC
# 提取出来的同时，用数组的方式保存改位置索引，跑完代码后按索引放入nc文件，并制作新的掩码数组

root_path = "F:\\China_XCO2\\"
generate_path = root_path + "generated XCO2\\"
CAMS_path = root_path + "CAMS\\"
Dem_path = root_path + "Dem\\"
LAI_path = root_path + "LAI_v2\\"
FVC_path = root_path + "FVC_v2\\"
LandScan_path = root_path + "LandScan\\nc\\"
ODIAC_path = root_path + "ODIAC\\temp_nc\\"

era5_path = root_path + "era5_v2\\"

r_path = era5_path + "r\\"
t_path = era5_path + "t\\"
blh_path = era5_path + "blh\\"
sp_path = era5_path + "sp\\"
tp_path = era5_path + "tp\\"
era5_list = ["r", "t", "blh", "sp", "tp"]
path_list = [r_path, t_path, blh_path, sp_path, tp_path]

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"There are {torch.cuda.device_count()} GPU(s) available.")
    print("Device name:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("No GPU available, using the CPU instead.")

CNN_path = 'train\\checkpoint\\DSC_checkpoint.tar'
CNN = DSC(4).to(device)
checkpoint = torch.load(CNN_path)
CNN.load_state_dict(checkpoint['model_state_dict'])
CNN.eval()

model_file = root_path + "train\\checkpoint\\DF-LGB_after_CNN_6"
model = CascadeForestRegressor(use_predictor=True)
model.set_predictor(LGBMRegressor())
model.load(model_file)

# 将CAMS数据复制进generate_path中，直接覆盖，就不需要再创建一个.nc文件了
for root, dirs, files in os.walk(generate_path):
    for file in files:
        if(file[0:4] == "CAMS"):
            lon = None
            lat = None
            CNN_inputdata = list()
            inputdata = list()
            index_list = list()
            name = file[-11:]  # 20150101.nc
            year = name[0:4]
            month = name[0:6]
            day = name[0:-3]
            print(day)
            CAMS_file = Dataset(generate_path + file, "r+")
            CAMS_XCO2 = CAMS_file.variables["CAMS"][:]
            CAMS_XCO2 = np.ma.ravel(CAMS_XCO2)
            # print(len(XCO2.compressed()))

            if lon == None or lat == None:
                lon = np.zeros((474, 617))
                for k in range(0, 474):
                    lon[k] = np.array(CAMS_file.variables['lon'][:])
                lon = np.ravel(lon)

                lat = np.zeros((617, 474))
                for k in range(0, 617):
                    lat[k] = np.array(CAMS_file.variables['lat'][:])
                lat = lat.T
                lat = np.ravel(lat)

            Dem_file = Dataset(Dem_path + "dem.nc")
            # Dem = Dem_file.variables["dem"][:]
            # Dem = np.ma.ravel(Dem)

            doy = time.strftime("%j", time.strptime(day, '%Y%m%d'))
            LAI_doy = ((int(doy)-1)//8)*8 + 1
            eight_day = time.strftime("%Y%m%d", time.strptime(str(year) + str(LAI_doy), '%Y%j'))
            print(eight_day)
            LAI_file = Dataset(LAI_path + "LAI_" + eight_day + ".nc")
            FVC_file = Dataset(FVC_path + "FVC_" + eight_day + ".nc")
            LAI_array = LAI_file.variables["LAI"][:].filled(fill_value=0)
            FVC_array = FVC_file.variables["FVC"][:].filled(fill_value=0)
            # LAI = np.ma.ravel(LAI)
            # FVC = np.ma.ravel(FVC)

            r_file = Dataset(r_path + "r_" + day + ".nc")
            t_file = Dataset(t_path + "t_" + day + ".nc")
            blh_file = Dataset(blh_path + "blh_" + day + ".nc")
            sp_file = Dataset(sp_path + "sp_" + day + ".nc")
            tp_file = Dataset(tp_path + "tp_" + day + ".nc")

            ODIAC_file = Dataset(ODIAC_path + "emi_" + month + ".nc")
            ODIAC_array = ODIAC_file.variables["emissions"][:].filled(fill_value=0)
            # ODIAC = np.ma.ravel(ODIAC)
            # print(len(EDGAR.compressed()))

            LS_file = Dataset(LandScan_path + year + ".nc")
            LS_array = LS_file.variables["population"][:].filled(fill_value=0)
            # LS = np.ma.ravel(LS)
            # print(len(LS.compressed()))
            for i in range(0, len(CAMS_XCO2)):
                if CAMS_XCO2[i] > 0:
                    lon_value = lon[i]
                    lat_value = lat[i]
                    lat_value = round(lat_value, 2)
                    lon_value = round(lon_value, 2)
                    lat_index = np.where(np.around(Dem_file.variables['lat'][:], 2) == lat_value)[0]
                    lon_index = np.where(np.around(Dem_file.variables['lon'][:], 2) == lon_value)[0]
                    ## emi和pop
                    lat_index_1 = np.where(np.around(LS_file.variables['lat'][:], 2) == lat_value)[0]
                    lon_index_1 = np.where(np.around(LS_file.variables['lon'][:], 2) == lon_value)[0]
                    ## LAI和FVC
                    lat_index_2 = np.where(np.around(LAI_file.variables['lat'][:], 2) == lat_value)[0]
                    lon_index_2 = np.where(np.around(LAI_file.variables['lon'][:], 2) == lon_value)[0]
                    ## era5
                    lat_index_3 = np.where(np.around(r_file.variables['lat'][:], 2) == lat_value)[0]
                    lon_index_3 = np.where(np.around(r_file.variables['lon'][:], 2) == lon_value)[0]
                    if LS_file.variables['population'][lat_index_1, lon_index_1].mask \
                            or ODIAC_file.variables['emissions'][lat_index_1, lon_index_1].mask\
                            or LAI_file.variables['LAI'][lat_index_2, lon_index_2].mask\
                            or FVC_file.variables['FVC'][lat_index_2, lon_index_2].mask\
                            or r_file.variables['r'][lat_index_3, lon_index_3].mask:  ## 掩膜如果不存在
                        continue

                    index_list.append(i)
                    # CNN输入变量
                    pop = LS_array[lat_index_1[0] - 9:lat_index_1[0] + 10, lon_index_1[0] - 9:lon_index_1[0] + 10]
                    emi = ODIAC_array[lat_index_1[0] - 9:lat_index_1[0] + 10, lon_index_1[0] - 9:lon_index_1[0] + 10]
                    LAI = LAI_array[lat_index_2[0] - 9:lat_index_2[0] + 10, lon_index_2[0] - 9:lon_index_2[0] + 10]
                    FVC = FVC_array[lat_index_2[0] - 9:lat_index_2[0] + 10, lon_index_2[0] - 9:lon_index_2[0] + 10]
                    # 剩余变量
                    CAMS = CAMS_XCO2.data[i]
                    dem = Dem_file.variables['dem'][lat_index, lon_index].data[0, 0]
                    r = r_file.variables['r'][lat_index_3, lon_index_3].data[0, 0]
                    t = t_file.variables['t'][lat_index_3, lon_index_3].data[0, 0]
                    blh = blh_file.variables['blh'][lat_index_3, lon_index_3].data[0, 0]
                    sp = sp_file.variables['sp'][lat_index_3, lon_index_3].data[0, 0]
                    tp = tp_file.variables['tp'][lat_index_3, lon_index_3].data[0, 0]

                    CNN_input_arr = np.array([LAI, FVC, pop, emi])
                    CNN_inputdata.append(CNN_input_arr)
                    input_arr = np.array([CAMS, dem, r, t, blh, sp, tp, lat_value, lon_value, year, doy])
                    inputdata.append(input_arr)
            inputdata = np.array(inputdata)
            CNN_inputdata = np.array(CNN_inputdata)
            CNN_inputdata = torch.tensor(CNN_inputdata, dtype=torch.float32).to(device)
            print("送入DSC模型提取局部特征")
            CNN_feature = CNN.get_feature(CNN_inputdata)
            CNN_feature = CNN_feature.to('cpu').detach().numpy()

            inputdata = np.concatenate((inputdata, CNN_feature), axis=1)
            print(inputdata.shape, "送入DF-LGB模型")

            # 送入模型
            output = model.predict(inputdata)
            print("模型运行完毕，开始重建图像")
            for i, v in enumerate(index_list):
                CAMS_XCO2.data[v] = output[i]
            generated_XCO2 = CAMS_XCO2.reshape(474, 617)
            CAMS_file.variables["CAMS"][:] = generated_XCO2
            print("重建完成，进行重命名")
            CAMS_file.variables['CAMS'].long_name = 'DF-LGB_XCO2'
            CAMS_file.renameVariable("CAMS", "XCO2")
            CAMS_file.close()
            newfile = "XCO2_" + name
            os.rename(root + file, root + newfile)