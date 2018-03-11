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

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
fine_tuned_model = load_model('D:\ML\ArtiD\kaggle\GoogleLandmark\models\\res_from140.h5')
weights = fine_tuned_model.get_weights()
base_model.set_weights(weights[:-1])

path = 'D:\ML\ArtiD\kaggle\GoogleLandmark\\train_set'
# cv2.imshow('dst_rt', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
dirs = os.listdir(path)
for cls in dirs:
    cls_path = path + '\\' + cls
    imgs = os.listdir(cls_path)
    csv_matrix = []
    i = 1
    for img_name in imgs[2500:]:
        img = cv2.imread(cls_path + '\\' + img_name)
        out = base_model.predict(img.reshape((1, 224, 224, 3)))
        vec = out.reshape(2048)
        vec_round = vec.round(decimals=4)
        csv_matrix.append(list(vec_round.astype(str)))
        i += 1
        if i > 100:
            break
        print(i)

    with open('D:\ML\ArtiD\kaggle\GoogleLandmark\dataset\\resnet_small\\' + cls + "_test_features.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(csv_matrix)