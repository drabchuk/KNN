import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('D:\ML\ArtiD\kaggle\GoogleLandmark\\train_set\\12172\\26753.jpg')
img2 = cv2.imread('D:\ML\ArtiD\kaggle\GoogleLandmark\\train_set\\428\\57.jpg')
img = cv2.imread('D:\ML\ArtiD\kaggle\GoogleLandmark\\train_set\\3804\\6604.jpg')
rows, cols, ch = img.shape
top = 0
left = 50
height = 200
width = 200
squeeze_y = 0.8
squeeze_x = 0.6
ground = 60
pts1 = np.float32([[left, top],
                   [left + squeeze_x * width, top + (height * (1.0 - squeeze_y)) / 2.0 + ground],
                   [left, top + width],
                   [left + squeeze_x * width, top + height - height * squeeze_y / 2.0 + ground]])
pad = 50
size = 200
pts2 = np.float32([[pad, pad],
                   [size + pad, pad],
                   [pad, size + pad],
                   [size + pad, size + pad]])
M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(img,M,(300,300))
plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()