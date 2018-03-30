import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import csv

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
    n_classes = len(classes)
    means = np.zeros((n_classes, 2048), dtype=float);
    test_set = []
    test_set_labels = []
    counter = 0
    for cls in classes:
        print(counter)
        print('reading', cls)
        X = read_cls(ds_path, cls)
        y = np.zeros((X.shape[0],), dtype=int)
        y.fill(counter)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
        for sample in X_train:
            means[counter] += sample
        means[counter] /= X_train.shape[0]
        counter += 1
        test_set.append(X_test)
        test_set_labels.append(y_test)
    print('reading completed')
    with open('D:\ML\ArtiD\kaggle\GoogleLandmark\KNN\means_features.csv', "w") as f:
        writer = csv.writer(f)
        writer.writerows(means)

    return classes, all

read_all('C:\Artid\output\output')