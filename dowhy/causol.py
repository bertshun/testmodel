# =========================
# Causal Inference with EconML + DoWhy
# Dataset: UCI Bank Marketing
# =========================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from econml.dr import DRLearner
import dowhy
from dowhy import CausalModel
import matplotlib.pyplot as plt

# =========================
# 1. データ取得
# =========================
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional-full.csv"
df = pd.read_csv(url, sep=';')

# =========================
# 2. 欠損処理・変数選定
# =========================
df = df.dropna()

# 処置(T): 電話をかけたかどうか
T_col = 'contact'  # 'cellular' or 'telephone'
df[T_col] = (df[T_col] == 'cellular').astype(int)

# 目的変数(Y): 定期預金加入
Y_col = 'y'
df[Y_col] = (df[Y_col] == 'yes').astype(int)

# 説明変数
X_cols = ['age', 'job', 'education', 'marital', 'housing', 'loan', 'month', 'campaign', 'pdays', 'previous']

# =========================
# 3. 前処理
# =========================
num_cols = df[X_cols].select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = list(set(X_cols) - set(num_cols))

ct = ColumnTransformer([
    ('num', RobustScaler(), num_cols),
    ('cat', OneHotEncoder(drop='first'), cat_cols)
])

X = ct.fit_transform(df[X_cols])
T = df[T_col].values
Y = df[Y_col].values

# =========================
# 4. モデル構築 (DoWhy)
# =========================
causal_model = CausalModel(
    data=pd.DataFrame({'Y': Y, 'T': T}),
    treatment='T',
    outcome='Y',
    common_causes='X'  # dummy entry
)

# =========================
# 5. EconML: Doubly Robust Learner
# =========================
model_y = RandomForestRegressor(n_estimators=100, random_state=42)
model_t = RandomForestClassifier(n_estimators=100, random_state=42)
est = DRLearner(model_regression=model_y, model_propensity=model_t)

est.fit(Y, T, X=X)
treatment_effects = est.effect(X)

# =========================
# 6. 結果解析
# =========================
ATE = np.mean(treatment_effects)
print(f"\n📊 Average Treatment Effect (ATE): {ATE:.4f}")

plt.hist(treatment_effects, bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Individual Treatment Effects (ITE)')
plt.xlabel('Estimated Effect')
plt.ylabel('Frequency')
plt.show()

# =========================
# 7. 評価（簡易）
# =========================
# 平均処置効果: E[Y|T=1] - E[Y|T=0] と比較
true_ate = df.loc[df[T_col]==1, Y_col].mean() - df.loc[df[T_col]==0, Y_col].mean()
print(f"Ground-truth-like naive ATE (difference in means): {true_ate:.4f}")
