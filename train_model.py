import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow import keras
import joblib

# 단계 1: Dataset 전처리

# 1-1. Data 불러오기
fires = pd.read_csv("./sanbul2district-divby100.csv", sep=",")

# 1-2. 데이터 탐색 및 카테고리형 특성 확인
print("2021810051 LeeMinJun")
print("=== fires.head() ===")
print(fires.head())
print("\n=== fires.info() ===")
print(fires.info())
print("\n=== fires.describe() ===")
print(fires.describe())
print("\n=== 카테고리형 특성 month, day value_counts() ===")
print(fires['month'].value_counts())
print(fires['day'].value_counts())

# 1-3. 데이터 시각화 (전체 특성 히스토그램)
fires.hist(bins=50, figsize=(20, 15))
plt.suptitle("1-3. Feature Histograms Before Log Transformation 2021810051 LeeMinJun")
plt.show()

# 1-4. 특성 burned_area 왜곡 현상 개선 (로그 함수 적용 및 히스토그램 비교)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
fires['burned_area'].hist(bins=50)
plt.title("Before Log Transformation (burned_area) 2021810051 LeeMinJun")

# y = ln(burned_area + 1) 변환
fires['burned_area'] = np.log(fires['burned_area'] + 1)

plt.subplot(1, 2, 2)
fires['burned_area'].hist(bins=50)
plt.title("After Log Transformation (burned_area) 2021810051 LeeMinJun")
plt.show()

# 1-5. StratifiedShuffleSplit을 이용한 데이터 분리
print("2021810051 LeeMinJun")
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(fires, fires["month"]):
    strat_train_set = fires.loc[train_index]
    strat_test_set = fires.loc[test_index]

# 1-6. Pandas scatter_matrix() 함수를 이용하여 4개 이상의 특성에 대해 matrix 출력
attributes = ["burned_area", "max_temp", "avg_temp", "max_wind_speed"]
scatter_matrix(strat_train_set[attributes], figsize=(12, 8))
plt.suptitle("1-6. Scatter Matrix 2021810051 LeeMinJun")
plt.show()

# 1-7. 지역별로 ‘burned_area’에 대해 plot 하기
strat_train_set.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
                     s=strat_train_set["max_temp"] * 5, label="max_temp", figsize=(10,7),
                     c="burned_area", cmap=plt.get_cmap("jet"), colorbar=True)
plt.title("1-7. Geographic Plot of Burned Area 2021810051 LeeMinJun")
plt.legend()
plt.show()

# 1-8. 카테고리형 특성 month, day에 대해 OneHotEncoder()를 이용한 인코딩/출력
fires = strat_train_set.drop(["burned_area"], axis=1) # drop labels for training set
fires_labels = strat_train_set["burned_area"].copy()

# --- 첫 번째 채워야 할 코드 시작 ---
fires_cat = fires[["month", "day"]]
print(fires_cat.head(10))

from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
fires_cat_1hot = cat_encoder.fit_transform(fires_cat)
print(fires_cat_1hot)

print(fires_cat_1hot.toarray())
# --- 첫 번째 채워야 할 코드 끝 ---

fires_num = fires.drop(["month", "day"], axis=1)

# --- 두 번째 채워야 할 코드 시작 ---
print("=== month categories ===")
print(cat_encoder.categories_[0])

print("\n=== day categories ===")
print(cat_encoder.categories_[1])
# --- 두 번째 채워야 할 코드 끝 ---

# 1-9. Scikit-Learn의 Pipeline, StandardScaler를 이용하여 카테고리형 특성을 인코딩한 training set 생성
print("2021810051 LeeMinJun")
print("\n\n########################################################################")
print("Now let's build a pipeline for preprocessing the numerical attributes:")

num_attribs = list(fires_num)
cat_attribs = ["month", "day"]

num_pipeline = Pipeline([
    ('std_scaler', StandardScaler()),
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(handle_unknown='ignore'), cat_attribs),
])

fires_prepared = full_pipeline.fit_transform(fires)

# 단계 2: Keras model 개발 (Regression MLP)
print("2021810051 LeeMinJun")
model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=fires_prepared.shape[1:]),
    keras.layers.Dense(10, activation="relu"),
    keras.layers.Dense(1)
])

model.compile(loss="mean_squared_error", optimizer="sgd")
model.fit(fires_prepared, fires_labels, epochs=50)

# 모델 및 전처리 파이프라인 저장
print("2021810051 LeeMinJun")
model.save("forest_fire_model.keras")
joblib.dump(full_pipeline, "full_pipeline.pkl")
print("Model and pipeline saved successfully.")