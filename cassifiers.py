from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from data_reader import *
from sklearn.model_selection import train_test_split
import pickle
from sklearn.externals import joblib
import csv

#data = read_all('D:\ML\ArtiD\kaggle\GoogleLandmark\dataset\\resnet_small')
#data = read_all('D:\ML\ArtiD\kaggle\GoogleLandmark\dataset\\resnetFeatures_from140')
classes, data = read_part('C:\Artid\output\output', 1000)
X = []
y = []
for cls, cls_samples in enumerate(data):
    for sample in cls_samples:
        X.append(sample)
        y.append(cls)
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.33, random_state=42)

num_classes = len(classes)

print('reading completed')

clf = linear_model.SGDClassifier()
clf.fit(X_train, y_train)

#check
sample = X_test[0].reshape((1, -1))
res = clf.predict(sample)
print('sample 0 :', res)
joblib.dump(clf, 'linear_'+str(num_classes)+'.pkl')
print('linear classifier', clf.score(X_test, y_test))

clf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=0)
clf.fit(X_train, y_train)
joblib.dump(clf, 'random_forest_'+str(num_classes)+'.pkl')
print('random forest train', clf.score(X_train, y_train))
print('random forest', clf.score(X_test, y_test))
print(clf.feature_importances_)
#print(clf.predict([[0, 0, 0, 0]]))

# clf = KNeighborsClassifier(n_neighbors=30, p=1, algorithm='ball_tree')
# clf.fit(X_train, y_train)
# joblib.dump(clf, 'knn_'+str(num_classes)+'.pkl')
# print('model training completed')
# print('KNN', clf.score(X_test, y_test))

clf = MLPClassifier(hidden_layer_sizes=(2048), activation='relu', solver='adam')
clf.fit(X_train, y_train)
joblib.dump(clf, 'mlp_'+str(num_classes)+'.pkl')
print('model trained')
print('MLP train', clf.score(X_train, y_train))
print('MLP', clf.score(X_test, y_test))