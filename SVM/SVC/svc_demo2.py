import numpy as np
from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])
from sklearn.svm import SVC
# clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf = SVC()
clf.fit(X, y)
clf.predict([[-0.8,-0.1]])
print(clf.predict([[-0.8,-0.1]]))
print(clf.predict([[0.8,0.1],[-0.1,-0.5]]))
