from sklearn import linear_model
from data_reader import *
from sklearn.model_selection import train_test_split
import pickle
from sklearn.externals import joblib
import csv

#data = read_all('D:\ML\ArtiD\kaggle\GoogleLandmark\dataset\\resnet_small')
classes, data = read_all('D:\ML\ArtiD\kaggle\GoogleLandmark\dataset\\resnetFeatures_from140')
X = []
y = []
for cls, cls_samples in enumerate(data):
    for sample in cls_samples:
        X.append(sample)
        y.append(cls)
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.33, random_state=42)

print('reading completed')
clf = linear_model.SGDClassifier()
clf.fit(X_train, y_train)
joblib.dump(clf, 'linear_10cls.pkl')
print(clf.score(X_test, y_test))

with open('D:\ML\ArtiD\kaggle\GoogleLandmark\dataset\\resnet_small\\calsses_order.csv', "w") as f:
    writer = csv.writer(f)
    writer.writerows([classes])