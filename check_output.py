from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout
from keras.utils import plot_model
from keras.models import load_model
import cv2
import os
import csv
import numpy as np

#base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
fine_tuned_model = load_model('D:\ML\ArtiD\kaggle\GoogleLandmark\models\\res_from140.h5')
#weights = fine_tuned_model.get_weights()
#base_model.set_weights(weights[:-1])

path = 'D:\ML\ArtiD\kaggle\GoogleLandmark\\train_set\\'
cls = '10184'
img_name = '23151.jpg'
img = cv2.imread(os.path.join(path, cls, img_name))
cv2.imshow('dst_rt', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
out = fine_tuned_model.predict(img.reshape((1, 224, 224, 3)))
vec = out.reshape(out.shape[1])
print(vec)
