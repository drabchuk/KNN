import numpy as np
import matplotlib.pyplot as plt
from data_reader import *

DATAPATH = 'D:\ML\ArtiD\kaggle\GoogleLandmark\dataset\\resnet_small'

data = read_all(DATAPATH)
data = np.array(data)
means = np.mean(data, axis=1)
stds = np.std(data, axis=1)
mean_std_per_class = np.mean(stds, axis=1)
mean_cls_std = np.mean(mean_std_per_class)
plt.plot(mean_std_per_class)
plt.show()
mean_all = np.mean(data)
std_all = np.std(data)
print('entire std : ', std_all)
print('per class mean std : ', mean_cls_std)