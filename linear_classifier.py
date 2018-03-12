import numpy as np
import matplotlib.pyplot as plt
from data_reader import *

DATAPATH = 'D:\ML\ArtiD\kaggle\GoogleLandmark\dataset\\resnetFeatures'

data = read_all(DATAPATH)
data = np.array(data)
#std_all = np.std(data)
#print('entire std: ', std_all)
means = []
stds = []
for i in range(data.shape[0]):
    mean = np.mean(data[i], axis=0)
    means.append(mean)
    std = np.std(data[i], axis=0)
    stds.append(std)
mean_std = np.mean(np.array(stds))
print('per class std', mean_std)
# means = np.mean(data, axis=1)
# stds = np.std(data, axis=1)
# mean_std_per_class = np.mean(stds, axis=1)
# mean_cls_std = np.mean(mean_std_per_class)
# plt.plot(mean_std_per_class)
# plt.show()
# mean_all = np.mean(data)
# std_all = np.std(data)
# print('entire std : ', std_all)
# print('per class mean std : ', mean_cls_std)