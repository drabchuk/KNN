import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

def filename(cls):
    return 'resnetFeatures' + cls + '.csv'

#ds_path = "D:\\ML\\ArtiD\\kaggle\\GoogleLandmark\\dataset\\resnetFeatures"
#classes = ['428', '2338', '3804', '3924', '7092', '9029', '10045', '10184', '12172', '12718']

def read_cls(ds_path, cls_data):
    df = pd.read_csv(os.path.join(ds_path, cls_data), sep=',', header=None)
    data = df.values
    return data

def read_all(ds_path):
    classes = os.listdir(ds_path)
    all = []
    counter = 0
    for cls in classes:
        counter += 1
        print(counter)
        print('reading', cls)
        data = read_cls(ds_path, cls)
        all.append(data)
    print('reading completed')
    return classes, all

def read_bounded(ds_path, max_samples):
    classes = os.listdir(ds_path)
    all = []
    counter = 0
    for cls in classes:
        counter += 1
        print(counter)
        print('reading', cls)
        data = read_cls(ds_path, cls)
        if len(data) > max_samples:
            data = data[:max_samples]
        all.append(data)
    print('reading completed')
    return classes, all

def read_part(ds_path, max_classes):
    classes = os.listdir(ds_path)
    all = []
    counter = 0
    for cls in classes:
        counter += 1
        if counter > max_classes:
            break
        print(counter)
        print('reading', cls)
        data = read_cls(ds_path, cls)
        all.append(data)
    print('reading completed')
    return classes[:counter - 1], all