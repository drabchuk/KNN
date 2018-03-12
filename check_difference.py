import numpy as np
import matplotlib.pyplot as plt
from data_reader import *

#DATAPATH = 'D:\ML\ArtiD\kaggle\GoogleLandmark\dataset\\resnetFeatures'

#sh428 = read_cls(DATAPATH, 'resnetFeatures428.csv')
#sh2338 = read_cls(DATAPATH, 'resnetFeatures2338.csv')

from keras.applications.resnet50 import ResNet50
import cv2
import os
import numpy as np

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
#fine_tuned_model = load_model('D:\ML\ArtiD\kaggle\GoogleLandmark\models\\res_from140.h5')
#weights = fine_tuned_model.get_weights()
#base_model.set_weights(weights[:-1])

path = 'D:\ML\ArtiD\kaggle\GoogleLandmark\\train_set'
# cv2.imshow('dst_rt', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
dirs = os.listdir(path)

sh428 = cv2.imread('D:\ML\ArtiD\kaggle\GoogleLandmark\\train_set\\428\\35.jpg')
sh2338 = cv2.imread('D:\ML\ArtiD\kaggle\GoogleLandmark\\train_set\\2338\\3167.jpg')
human2338 = cv2.imread('D:\ML\ArtiD\kaggle\GoogleLandmark\\train_set\\2338\\3485.jpg')
#cv2.imshow('sh428', sh428)
#cv2.imshow('sh2338', sh2338)
#cv2.imshow('human2338', human2338)
#print('please, don\'t close pictures, type ESC')
#cv2.waitKey(0)
#cv2.destroyAllWindows()
sh428_features = base_model.predict(sh428.reshape((1, 224, 224, 3)))
sh2338_features = base_model.predict(sh2338.reshape((1, 224, 224, 3)))
human2338_features = base_model.predict(human2338.reshape((1, 224, 224, 3)))
sh428_features = sh428_features.reshape(2048)
sh2338_features = sh2338_features.reshape(2048)
human2338_features = human2338_features.reshape(2048)

sh_towers_dist = np.linalg.norm(sh428_features - sh2338_features)
sh428_human_dist = np.linalg.norm(sh428_features - human2338_features)
sh2338_human_dist = np.linalg.norm(sh2338_features - human2338_features)
print('sh towers dist: ', sh_towers_dist)
print('sh 428 human dist: ', sh428_human_dist)
print('sh 2338 human dist: ', sh2338_human_dist)

sh_towers_dist = np.linalg.norm(sh428_features - sh2338_features, ord=4)
sh428_human_dist = np.linalg.norm(sh428_features - human2338_features, ord=4)
sh2338_human_dist = np.linalg.norm(sh2338_features - human2338_features, ord=4)
print('sh towers 4d dist: ', sh_towers_dist)
print('sh 428 human 4d dist: ', sh428_human_dist)
print('sh 2338 human 4d dist: ', sh2338_human_dist)

sh_towers_dist = np.sum(np.abs(sh428_features - sh2338_features))
sh428_human_dist = np.sum(np.abs(sh428_features - human2338_features))
sh2338_human_dist = np.sum(np.abs(sh2338_features - human2338_features))
print('sh towers manhattan dist: ', sh_towers_dist)
print('sh 428 human manhattan dist: ', sh428_human_dist)
print('sh 2338 human manhattan dist: ', sh2338_human_dist)

plt.plot(sh428_features, label='sh248')
plt.plot(sh2338_features, label='sh2338')
plt.plot(human2338_features, label='human2338')
plt.show()
