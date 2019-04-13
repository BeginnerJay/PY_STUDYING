### 지도 학습 알고리즘
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
import numpy as np

print("cancer.keys():\n", cancer.keys())
print("유방암 데이터의 형태:", cancer.data.shape)
print("클래스별 샘플 갯수:\n",
      {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))})
print("특성 이름:\n", cancer.feature_names)

from sklearn.datasets import load_boston
boston = load_boston()
print("데이터의 형태:", boston.data.shape)

from sklearn.model_selection import train_test_split
X, y = mglearn.datasets.make_forge()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)