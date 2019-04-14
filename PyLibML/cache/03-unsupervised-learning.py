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
            ##  스케일을 조정하는 전처리 메서드들은 보통 지도 학습 알고리즘을 적용하기 전에 적용.
            ##  전처리 후에 만들어진 지도 학습 모델을 평가하려면 훈련 세트와 테스트 세트로 나눠야 한다.
                # 이 때 훈련 세트와 테스트 세트를 동일한 방법으로 변환시켜야 한다.
    #### 3.4 차원 축소, 특성 추출, 매니폴드 학습\
        ### 3.4.1 주성분 분석(PCA)
            ## 특성들이 통계적으로 상관관계가 없도록 데이터셋을 회전시키는 기술.
                # 첫 번쨰 방향과 직각인 방향 중에서 가장 많은 정보를 담은 방향을 찾는다.
                # 2차원에서는 가능한 직각 방향이 하나지만, 고차원에서는 무한이 많을 수 있다.
                # 이런 과정을 거쳐 찾은 방향을, 데이터에 있는 주된 분산의 방향이라고 해서
                # 주성분이라고 한다. 일반적으로는 원본 특성 개수만큼 주성분이 존재한다.
            ## PCA에 의해 회전된 축은 연관되어 있지 않다.
                # 변환된 데이터의 상관관계 행렬이 대각행렬이 된다.
                    # 상관관계 행렬 : 공분산 행렬을 정규화한 것.
            ## 전체 데이터에서 평균을 빼서 중심을 원점에 맞춘 후
            ## 축에 나란하도록 회전한다. 그리고 나서
            ## 다시 평균을 더하고 반대로 회전시킨다.
            ## 데이터셋에 특성이 많은 경우 산점도 행렬 대신 히스토그램을 그린다.
                # 히스토그램은 분포에 대한 정보는 주지만, 상호작용에 대한 정보는 없다.
            ## PCA 객체를 생성하고, fit 메서드로 주성문을 찾고, transform 메서드로 데이터를 회전시키고 차원을 축소한다.
                # 데이터를 줄이려면 PCA 객체를 만들 때 얼마나 많은 성분을 유지할지 알려주어야 한다.\
            ## PCA는 비지도 학습 -> 회전축을 찾을 때 어떤 클래스 정보도 사용하지 않음.
            ## 단점 : 그래프의 두 축을 해석하기 쉽지 않다.
            ## PCA는 특성 추출에도 이용한다
                # 특성 추출 -> 원본 데이터 표현보다 분석하기에 더 적합한 표현 찾아보기
                # 분류기로는 클래스별 훈련 데이터가 너무 적고, 매번 재 훈련 필요
                # 원본 픽셀 공간에서 거리를 계산하는 것은 매우 나쁨
                    # PCA의 화이트닝 옵션 적용 -> 주성분의 스케일이 같아지도록 조정한다.
                    # 이는 화이트닝 없이 변환 후에 StandardScaler 적용하는 것과 같다.
            ## PCA는 데이터를 회전시키고 분산이 작은 주성분을 덜어내는 것이다.
            ## PCA를 이해하는 또 다른 방법은 몇 개의 주성분을 사용해 원본 데이터를 재구성하는 것이다.
                # 주성분을 더 많이 사용할수록 원본에 더 가까워진다.
        ### 3.4.2 비음수 행렬 분해(Non-negative Matrix Factorization)
            ## 유용한 특성 뽑기, 차원 축소에 사용 가능.
            ## NMF에서는 음수가 아닌 성분과 계수의 값을 찾는다. -> 음수인 데이터에 사용불가
                # PCA에서는 데이터의 분산이 가장 크고 수직인 성분 찾았음
            ## 여러 사람 목소리 분해, 악기 분해, 텍스트 데이터 등(덧붙이는 구조를 가진 데이터)에 특히 유용함.
            ## NMF의 주성분이 대체로 더 해석하기 쉽다.
                # 음수로 된 성분이나 계수가 만드는 상쇄 효과를 이해하기 어려운 PCA보다
            ## 주어진 데이터가 양수인지 확인
                # 데이터가 원점에서 상대적으로 어디에 놓여 있는지가 중요하다.
            ## 성분이 특성 개수만큼 많다면
                # 알고리즘은 데이터의 각 특성의 끝에 위치한 포인트를 가리키는 방향 선택.
            ## 성분을 하나만 사용한다면
                # 평균으로 향하는 성분을 만든다.
            ## 성분 개수를 줄이면 특정 방향이 제거될 뿐만 아니라 전체 성분이 완전히 바뀐다!
                # NMF에서 성분은 특정 방식으로 정렬되어 있지도 않다(순서가 없다).
                # 모든 성분을 동등하게 취급한다.
            ## 난수 생성 초깃값에 따라 결과가 달라진다.
                # 데이터가 복잡한 경우에는 난수가 큰 차이 만들 수도 있다.
            ## NMF는 데이터 인코딩, 재구성보다는 데이터의 패턴 찾기에 더 유용하다.
                # PCA가 재구성 측면에서 최선의 방향을 찾는다.
            ## 성분과 계수에 있는 제약을 설명하려면 확률 이론이 필요하다.
        ### 3.4.3 t-SNE를 이용한 매니폴드 학습
            ## 훨씬 복잡한 매핑을 만들어 더 나은 시각화를 제공한다.
            ## 새로운 데이터에는 적용하지 못한다.
                # 탐색적 분석에는 유용하지만, 지도 학습용으로는 사용하지 못한다.
            ## t-SNE는 새 데이터를 변환하는 기능을 제공하지 않음
                # TSBE 모델에는 transform 메서드가 없음
                # 대신 모델을 만들자마자 데이터를 변환시키는 fit_transform 메서드 사용가능
            ## 클래스 레이블 정보를 사용하지 않음 -> 완전한 비지도 학습


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
# fit 메서드로 학습한 변환을 적용하려면,
# 즉 실제로 훈련 데이터의 스케일을 조정하려면 스케일 객체의 transform 메서드를 사용
X_train_scaled = scaler.transform(X_train)
# 스케일이 조정된 후 데이터셋의 속성을 출력합니다
print("변환된 후 크기:", X_train_scaled.shape)
print("스케일 조정 전 특성별 최소값:\n", X_train.min(axis=0))
print("스케일 조정 전 특성별 최대값:\n", X_train.max(axis=0))
print("스케일 조정 후 특성별 최소값:\n", X_train_scaled.min(axis=0))
print("스케일 조정 후 특성별 최대값:\n", X_train_scaled.max(axis=0))

# 테스트 데이터 변환
# 이 데이터에 SVM을 적용하려면 테스트 세트도 변환해야 한다.
X_test_scaled = scaler.transform(X_test) # 같은 변환!!!
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
X_test_scaled_badly = test_scaler.transform(X_test) # 둘 다 0에서 1까지 정렬되어버림

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

#### 3.4
    ### 3.4.1

mglearn.plots.plot_pca_illustration()
        ## 시각화를 위해 유방암 데이터셋에 PCA 적용하기
fig, axes = plt.subplots(15, 2, figsize=(10, 20))
malignant = cancer.data[cancer.target == 0]
benign = cancer.data[cancer.target == 1]

ax = axes.ravel()
# 산점도 행렬로는 분석 불가(435개), 히스토그램 그린다.
for i in range(30):
    _, bins = np.histogram(cancer.data[:, i], bins=50)
    ax[i].hist(malignant[:, i], bins=bins, color=mglearn.cm3(0), alpha=.5)
    ax[i].hist(benign[:, i], bins=bins, color=mglearn.cm3(2), alpha=.5)
    ax[i].set_title(cancer.feature_names[i])
    ax[i].set_yticks(())
ax[0].set_xlabel("특성 크기")
ax[0].set_ylabel("빈도")
ax[0].legend(["악성", "양성"], loc="best")
fig.tight_layout()


from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()


scaler = StandardScaler()
scaler.fit(cancer.data)
X_scaled = scaler.transform(cancer.data)

from sklearn.decomposition import PCA
# 데이터의 처음 두 개 주성분만 유지시킵니다
pca = PCA(n_components=2)
# 유방암 데이터로 PCA 모델을 만듭니다
pca.fit(X_scaled)

# 처음 두 개의 주성분을 사용해 데이터를 변환합니다
X_pca = pca.transform(X_scaled)
print("원본 데이터 형태:", str(X_scaled.shape))
print("축소된 데이터 형태:", str(X_pca.shape))


# 클래스를 색깔로 구분하여 처음 두 개의 주성분을 그래프로 나타냅니다.
plt.figure(figsize=(8, 8))
mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], cancer.target)
plt.legend(["악성", "양성"], loc="best")
plt.gca().set_aspect("equal")
plt.xlabel("첫 번째 주성분")
plt.ylabel("두 번째 주성분")


print("PCA 주성분 형태:", pca.components_.shape)
print("PCA 주성분:", pca.components_)


plt.matshow(pca.components_, cmap='viridis')
plt.yticks([0, 1], ["첫 번째 주성분", "두 번째 주성분"])
plt.colorbar()
plt.xticks(range(len(cancer.feature_names)),
           cancer.feature_names, rotation=60, ha='left')
plt.xlabel("특성")
plt.ylabel("주성분")
        ## 고유 얼굴 특성 추출



from sklearn.datasets import fetch_lfw_people
people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape

fig, axes = plt.subplots(2, 5, figsize=(15, 8),
                         subplot_kw={'xticks': (), 'yticks': ()})
for target, image, ax in zip(people.target, people.images, axes.ravel()):
    ax.imshow(image)
    ax.set_title(people.target_names[target])


people.target[0:10], people.target_names[people.target[0:10]]
print("people.images.shape:", people.images.shape)
print("클래스 개수:", len(people.target_names))


# 각 타깃이 나타난 횟수 계산
counts = np.bincount(people.target)
# 타깃별 이름과 횟수 출력
for i, (count, name) in enumerate(zip(counts, people.target_names)):
    print("{0:25} {1:3}".format(name, count), end='   ')
    if (i + 1) % 3 == 0:
        print()


mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1

X_people = people.data[mask]
y_people = people.target[mask]

# 0~255 사이의 흑백 이미지의 픽셀 값을 0~1 사이로 스케일 조정합니다.
# (옮긴이) MinMaxScaler를 적용하는 것과 거의 동일합니다.
X_people = X_people / 255.


from sklearn.neighbors import KNeighborsClassifier
# 데이터를 훈련 세트와 테스트 세트로 나눕니다
X_train, X_test, y_train, y_test = train_test_split(
    X_people, y_people, stratify=y_people, random_state=0)
# 이웃 개수를 한 개로 하여 KNeighborsClassifier 모델을 만듭니다
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
print("1-최근접 이웃의 테스트 세트 점수: {:.2f}".format(knn.score(X_test, y_test)))

mglearn.plots.plot_pca_whitening()


pca = PCA(n_components=100, whiten=True, random_state=0).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

print("X_train_pca.shape:", X_train_pca.shape)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train_pca, y_train)
print("테스트 세트 정확도: {:.2f}".format(knn.score(X_test_pca, y_test)))

print("pca.components_.shape:", pca.components_.shape)


fig, axes = plt.subplots(3, 5, figsize=(15, 12),
                         subplot_kw={'xticks': (), 'yticks': ()})
for i, (component, ax) in enumerate(zip(pca.components_, axes.ravel())):
    ax.imshow(component.reshape(image_shape), cmap='viridis')
    ax.set_title("주성분 {}".format((i + 1)))


from matplotlib.offsetbox import OffsetImage, AnnotationBbox

image_shape = people.images[0].shape
plt.figure(figsize=(20, 3))
ax = plt.gca()

imagebox = OffsetImage(people.images[0], zoom=2, cmap="gray")
ab = AnnotationBbox(imagebox, (.05, 0.4), pad=0.0, xycoords='data')
ax.add_artist(ab)

for i in range(4):
    imagebox = OffsetImage(pca.components_[i].reshape(image_shape), zoom=2,
                           cmap="viridis")

    ab = AnnotationBbox(imagebox, (.285 + .2 * i, 0.4),
                        pad=0.0, xycoords='data')
    ax.add_artist(ab)
    if i == 0:
        plt.text(.155, .3, 'x_{} *'.format(i), fontdict={'fontsize': 30})
    else:
        plt.text(.145 + .2 * i, .3, '+ x_{} *'.format(i),
                 fontdict={'fontsize': 30})

plt.text(.95, .3, '+ ...', fontdict={'fontsize': 30})

plt.rc('text')
plt.text(.12, .3, '=', fontdict={'fontsize': 30})
plt.axis("off")
plt.show()
plt.close()
plt.rc('text')

mglearn.plots.plot_pca_faces(X_train, X_test, image_shape)

mglearn.discrete_scatter(X_train_pca[:, 0], X_train_pca[:, 1], y_train)
plt.xlabel("첫 번째 주성분")
plt.ylabel("두 번째 주성분")

    ### 3.4.2
        ## 인공 데이터에 NMF 적용
mglearn.plots.plot_nmf_illustration()
        ## 얼굴 이미지에 NMF 적용
mglearn.plots.plot_nmf_faces(X_train, X_test, image_shape)


from sklearn.decomposition import NMF
nmf = NMF(n_components=15, random_state=0)
nmf.fit(X_train)
X_train_nmf = nmf.transform(X_train)
X_test_nmf = nmf.transform(X_test)

fig, axes = plt.subplots(3, 5, figsize=(15, 12),
                         subplot_kw={'xticks': (), 'yticks': ()})
for i, (component, ax) in enumerate(zip(nmf.components_, axes.ravel())):
    ax.imshow(component.reshape(image_shape))
    ax.set_title("성분 {}".format(i))


compn = 3
# 4번째 성분으로 정렬하여 처음 10개 이미지를 출력합니다
inds = np.argsort(X_train_nmf[:, compn])[::-1]
fig, axes = plt.subplots(2, 5, figsize=(15, 8),
                         subplot_kw={'xticks': (), 'yticks': ()})
for i, (ind, ax) in enumerate(zip(inds, axes.ravel())):
    ax.imshow(X_train[ind].reshape(image_shape))

compn = 7
# 8번째 성분으로 정렬하여 처음 10개 이미지를 출력합니다
inds = np.argsort(X_train_nmf[:, compn])[::-1]
fig, axes = plt.subplots(2, 5, figsize=(15, 8),
                         subplot_kw={'xticks': (), 'yticks': ()})
for i, (ind, ax) in enumerate(zip(inds, axes.ravel())):
    ax.imshow(X_train[ind].reshape(image_shape))

    ### 3.4.3
from sklearn.datasets import load_digits
digits = load_digits()

fig, axes = plt.subplots(2, 5, figsize=(10, 5),
                         subplot_kw={'xticks':(), 'yticks': ()})
for ax, img in zip(axes.ravel(), digits.images):
    ax.imshow(img)


# PCA 모델을 생성합니다
pca = PCA(n_components=2)
pca.fit(digits.data)
# 처음 두 개의 주성분으로 숫자 데이터를 변환합니다
digits_pca = pca.transform(digits.data)
colors = ["#476A2A", "#7851B8", "#BD3430", "#4A2D4E", "#875525",
          "#A83683", "#4E655E", "#853541", "#3A3120","#535D8E"]
plt.figure(figsize=(10, 10))
plt.xlim(digits_pca[:, 0].min(), digits_pca[:, 0].max())
plt.ylim(digits_pca[:, 1].min(), digits_pca[:, 1].max())
for i in range(len(digits.data)):
    # 숫자 텍스트를 이용해 산점도를 그립니다
    plt.text(digits_pca[i, 0], digits_pca[i, 1], str(digits.target[i]),
             color = colors[digits.target[i]],
             fontdict={'weight': 'bold', 'size': 9})
plt.xlabel("첫 번째 주성분")
plt.ylabel("두 번째 주성분")


from sklearn.manifold import TSNE
tsne = TSNE(random_state=42)
# TSNE에는 transform 메소드가 없으므로 대신 fit_transform을 사용합니다
digits_tsne = tsne.fit_transform(digits.data)

plt.figure(figsize=(10, 10))
plt.xlim(digits_tsne[:, 0].min(), digits_tsne[:, 0].max() + 1)
plt.ylim(digits_tsne[:, 1].min(), digits_tsne[:, 1].max() + 1)
for i in range(len(digits.data)):
    # 숫자 텍스트를 이용해 산점도를 그립니다
    plt.text(digits_tsne[i, 0], digits_tsne[i, 1], str(digits.target[i]),
             color = colors[digits.target[i]],
             fontdict={'weight': 'bold', 'size': 9})
plt.xlabel("t-SNE 특성 0")
plt.ylabel("t-SNE 특성 1")















