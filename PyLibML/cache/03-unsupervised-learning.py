##### 3. 비지도 학습과 데이터 전처리
    #### 3.1 비지도 학습의 종류
        ### 1) 비지도 변환
            ## 데이터를 새롭게 표현하여, 사람이나 다른 ML Algorithm이 원래 데이터보다 알아보기 쉽도록 하는 것
                # ex) 고차원 데이터의 차원 수를 줄이면서 필요한 데이터로 표현하는 차원 축소
                # ex) 텍스트 문서에서 주제 추출하기
        ### 2) 군집
            ## 데이터를 비슷한 것끼리 그룹으로 묶는 것
    #### 3.2 비지도 학습의 도전 과제
        ### 알고리즘이 뭔가 유용한 것을 학습했는지 평가
        ### 무엇이 올바른 출력인지 모르는 경우가 많음
            ## 알고리즘에게 우리가 원하는 것을 알려줄 방법이 없음
            ## 직접 확인하는 것이 유일한 방법일 경우도 있음.
        ### 탐색적 분석 단계에서 많이 사용함
        ### 비지도 학습은 지도 학습의 전처리 단계에서도 사용함
    #### 3.3 데이터 전처리와 스케일 조정
        ### DL, SVM 같은 algorithm은 data scale에 매우 민감하다.
        ### 3.3.1 여러 가지 전처리 방법
            ## 1) StandardScaler : 각 특성의 평균을 0, 분산을 1로 변경(표준화)
                # min, max 제한 않음 -> outlier에 취약
            ## 2) RobustScaler : 평균과 분산 대신 중간값과 사분위값을 이용한 표준화
                # 이상치에 취약하지 않음
            ## 3) MinMaxScaler : 모든 특성이 0과 1 사이에 위치하도록 데이터를 변경
                # 2차원 데이터 셋의 경우에는 모든 데이터가 넓이 1인 정사각형에 담긴다.
            ## 4) Normalizer : 특정 벡터의 유클리디안 길이가 1이 되도록 데이터 포인트 조정
                # 즉 지름이 1인 원(구)에 데이터 포인트 투영
                # 각 데이터 포인트가 다른 비율로(길이에 반비례하여) 스케일이 조정된다는 뜻.
                # 특성 벡터의 길이는 상관 없고, 데이터의 방향(or 각도)만 중요할 때에 많이 사용함.
        ### 3.3.2 데이터 변환 적용하기


import sys
print("Python 버전:", sys.version)

import pandas as pd
print("pandas 버전:", pd.__version__)

import matplotlib
from matplotlib import pyplot as plt
plt.show()
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


mglearn.plots.plot_scaling()

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
                                                    random_state=1)
print(X_train.shape)
print(X_test.shape)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)

# 데이터 변환
X_train_scaled = scaler.transform(X_train)
# 스케일이 조정된 후 데이터셋의 속성을 출력합니다
print("변환된 후 크기:", X_train_scaled.shape)
print("스케일 조정 전 특성별 최소값:\n", X_train.min(axis=0))
print("스케일 조정 전 특성별 최대값:\n", X_train.max(axis=0))
print("스케일 조정 후 특성별 최소값:\n", X_train_scaled.min(axis=0))
print("스케일 조정 후 특성별 최대값:\n", X_train_scaled.max(axis=0))

# 테스트 데이터 변환
X_test_scaled = scaler.transform(X_test)
# 스케일이 조정된 후 테스트 데이터의 속성을 출력합니다
print("스케일 조정 후 특성별 최소값:\n", X_test_scaled.min(axis=0))
print("스케일 조정 후 특성별 최대값:\n", X_test_scaled.max(axis=0))

# 동일한 방법으로 훈련 데이터와 테스트 데이터의 스케일을 조정하기
# matplotlib 3.0 버전에서는 scatter 함수에 색깔을 지정할 때
# 하나의 RGB 포맷 문자열이나 Colormap의 리스트를 지정해야 합니다.
# 경고를 피하기 위해 mglearn에서 만든 ListedColormap 객체의 colors 속성의 원소를
# 직접 선택하여 RGB 포맷 문자열을 지정합니다.

from sklearn.datasets import make_blobs
# 인위적인 데이터셋 생성
X, _ = make_blobs(n_samples=50, centers=5, random_state=4, cluster_std=2)
# 훈련 세트와 테스트 세트로 나눕니다
X_train, X_test = train_test_split(X, random_state=5, test_size=.1)

# 훈련 세트와 테스트 세트의 산점도를 그립니다
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].scatter(X_train[:, 0], X_train[:, 1],
                c=mglearn.cm2.colors[0], label="훈련 세트", s=60)
axes[0].scatter(X_test[:, 0], X_test[:, 1], marker='^',
                c=mglearn.cm2.colors[1], label="테스트 세트", s=60)
axes[0].legend(loc='upper left')
axes[0].set_title("원본 데이터")

# MinMaxScaler를 사용해 스케일을 조정합니다
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 스케일이 조정된 데이터의 산점도를 그립니다
axes[1].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1],
                c=mglearn.cm2.colors[0], label="훈련 세트", s=60)
axes[1].scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], marker='^',
                c=mglearn.cm2.colors[1], label="테스트 세트", s=60)
axes[1].set_title("스케일 조정된 데이터")

# 테스트 세트의 스케일을 따로 조정합니다
# 테스트 세트의 최솟값은 0, 최댓값은 1이 됩니다
# 이는 예제를 위한 것으로 절대로 이렇게 사용해서는 안됩니다
test_scaler = MinMaxScaler()
test_scaler.fit(X_test)
X_test_scaled_badly = test_scaler.transform(X_test)

# 잘못 조정된 데이터의 산점도를 그립니다
axes[2].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1],
                c=mglearn.cm2.colors[0], label="training set", s=60)
axes[2].scatter(X_test_scaled_badly[:, 0], X_test_scaled_badly[:, 1],
                marker='^', c=mglearn.cm2.colors[1], label="test set", s=60)
axes[2].set_title("잘못 조정된 데이터")

for ax in axes:
    ax.set_xlabel("특성 0")
    ax.set_ylabel("특성 1")
fig.tight_layout()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# 메소드 체이닝(chaining)을 사용하여 fit과 transform을 연달아 호출합니다
X_scaled = scaler.fit(X_train).transform(X_train)
# 위와 동일하지만 더 효율적입니다
X_scaled_d = scaler.fit_transform(X_train)

# 지도 학습에서 데이터 전처리 효과

#사이킷런 0.20 버전에서 SVC 클래스의 gamma 매개변수 옵션에 auto외에 scale이 추가되었습니다.
# auto는 1/n_features, 즉 특성 개수의 역수입니다.
# scale은 1/(n_features * X.std())로 스케일 조정이 되지 않은 특성에서 더 좋은 결과를 만듭니다.
# 사이킷런 0.22 버전부터는 gamma 매개변수의 기본값이 auto에서 scale로 변경됩니다.
# 서포트 벡터 머신을 사용하기 전에 특성을 표준화 전처리하면 scale과 auto는 차이가 없습니다.

from sklearn.svm import SVC

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
                                                    random_state=0)

svm = SVC(C=100)
svm.fit(X_train, y_train)
print("테스트 세트 정확도: {:.2f}".format(svm.score(X_test, y_test)))

# 0~1 사이로 스케일 조정
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 조정된 데이터로 SVM 학습
svm.fit(X_train_scaled, y_train)

# 스케일 조정된 테스트 세트의 정확도
print("스케일 조정된 테스트 세트의 정확도: {:.2f}".format(svm.score(X_test_scaled, y_test)))

# 평균 0, 분산 1을 갖도록 스케일 조정
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 조정된 데이터로 SVM 학습
svm.fit(X_train_scaled, y_train)

# 스케일 조정된 테스트 세트의 정확도
print("SVM test accuracy: {:.2f}".format(svm.score(X_test_scaled, y_test)))





















































