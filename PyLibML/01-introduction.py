import sys
print("Python 버전:", sys.version)

import pandas as pd
print("pandas 버전:", pd.__version__)

import matplotlib
print("matplotlib 버전:", matplotlib.__version__)

import numpy as np
print("NumPy 버전:", np.__version__)

import scipy as sp
print("SciPy 버전:", sp.__version__)

import IPython
print("IPython 버전:", IPython.__version__)

import sklearn
print("scikit-learn 버전:", sklearn.__version__)

from sklearn.datasets import load_iris
iris_dataset = load_iris()

import mglearn

print("iris_dataset의 키:\n", iris_dataset.keys())
print(iris_dataset['DESCR'][:193] + "\n...")
print("타깃의 이름:", iris_dataset['target_names'])
print("특성의 이름:\n", iris_dataset['feature_names'])
print("data의 타입:", type(iris_dataset['data']))
print("data의 크기:", iris_dataset['data'].shape)
print("data의 처음 다섯 행:\n", iris_dataset['data'][:5])
print("target의 타입:", type(iris_dataset['target']))
print("target의 크기:", iris_dataset['target'].shape)
print("타깃:\n", iris_dataset['target'])

### 성과 측정 : 훈련 데이터와 테스트 데이터

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)

print("X_train 크기:", X_train.shape)
print("y_train 크기:", y_train.shape)

print("X_test 크기:", X_test.shape)
print("y_test 크기:", y_test.shape)

### 가장 먼저 할 일 : 데이터 살펴보기

# X_train 데이터를 사용해서 데이터프레임을 만듭니다.
# 열의 이름은 iris_dataset.feature_names에 있는 문자열을 사용합니다.
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# 데이터프레임을 사용해 y_train에 따라 색으로 구분된 산점도 행렬을 만듭니다.
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
                           hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)

### 첫 번째 머신 러닝 모델 : k - 최근접 이웃 알고리즘
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)
# 예측하기
X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape:", X_new.shape)

prediction = knn.predict(X_new)
print("예측:", prediction)
print("예측한 타깃의 이름:", 
       iris_dataset['target_names'][prediction])

# 모델 평가하기

y_pred = knn.predict(X_test)
print("테스트 세트에 대한 예측값:\n", y_pred)
print("테스트 세트의 정확도: {:.2f}".format(np.mean(y_pred == y_test)))
print("테스트 세트의 정확도: {:.2f}".format(knn.score(X_test, y_test)))

X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

print("테스트 세트의 정확도: {:.2f}".format(knn.score(X_test, y_test)))

