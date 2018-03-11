import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

def filename(cls):
    return 'resnetFeatures' + cls + '.csv'

ds_path = "D:\\ML\\ArtiD\\kaggle\\GoogleLandmark\\dataset\\resnetFeatures"
classes = ['428', '2338', '3804', '3924', '7092', '9029', '10045', '10184', '12172', '12718']

df428 = pd.read_csv(os.path.join(ds_path, filename(classes[0])), sep=',', header=None)
data428 = df428.values
print(data428.shape)
means428 = np.mean(data428, axis=0)
stds428 = np.mean(data428, axis=0)
print('end')
df7092 = pd.read_csv(os.path.join(ds_path, filename(classes[4])), sep=',', header=None)
data7092 = df7092.values
print(data7092.shape)
means7092 = np.mean(data7092, axis=0)
stds7092 = np.mean(data7092, axis=0)
print('end')
df10045 = pd.read_csv(os.path.join(ds_path, filename(classes[5])), sep=',', header=None)
data10045 = df10045.values
print(data10045.shape)
means10045 = np.mean(data10045, axis=0)
stds10045 = np.mean(data10045, axis=0)
print('end')
dif = means428 - means7092
mean_dif = np.mean(dif)
std_dif = np.std(dif)
plt.plot(means428)
plt.plot(means7092)
plt.plot(means10045)
plt.show()
print('e')
