from sklearn.ensemble import RandomForestClassifier
from data_reader import *
from sklearn.model_selection import train_test_split

#data = read_all('D:\ML\ArtiD\kaggle\GoogleLandmark\dataset\\resnet_small')
data = read_all('D:\ML\ArtiD\kaggle\GoogleLandmark\dataset\\resnetFeatures')
X = []
y = []
for cls, cls_samples in enumerate(data):
    for sample in cls_samples:
        X.append(sample)
        y.append(cls)
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.33, random_state=42)

print('reading completed')
clf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=0)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
print(clf.feature_importances_)
#print(clf.predict([[0, 0, 0, 0]]))