import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import requests, zipfile
from io import BytesIO

# ========================
# 1. データ取得
# ========================
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
print("📦 Downloading dataset...")
r = requests.get(url)
r.raise_for_status()

zf = zipfile.ZipFile(BytesIO(r.content))
csv_file = [f for f in zf.namelist() if "bank-additional-full.csv" in f][0]
df = pd.read_csv(zf.open(csv_file), sep=';')
print("✅ Loaded:", df.shape)

# ========================
# 2. 前処理
# ========================
df['y'] = (df['y'] == 'yes').astype(int)
df['housing'] = df['housing'].replace({'yes': 1, 'no': 0, 'unknown': np.nan})
df = df.dropna(subset=['housing'])

X_cols = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate',
           'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
T_col = 'housing'
Y_col = 'y'

X = df[X_cols]
T = df[T_col].astype(int)  # 明示的にintにキャスト
Y = df[Y_col]

# ========================
# 3. Split
# ========================
X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
    X, T, Y, test_size=0.3, random_state=42
)

# ========================
# 4. CausalForestDML
# ========================
print("🌲 Training CausalForestDML...")

cf = CausalForestDML(
    model_y=RandomForestRegressor(n_estimators=100, min_samples_leaf=10),
    model_t=RandomForestClassifier(n_estimators=100, min_samples_leaf=10),
    discrete_treatment=True,   # 🔥 これが超重要！！
    random_state=42
)

cf.fit(Y_train, T_train, X=X_train)

# ========================
# 5. 推定と評価
# ========================
ite = cf.effect(X_test)
ate = np.mean(ite)

print(f"Average Treatment Effect (ATE): {ate:.4f}")
lb, ub = cf.effect_interval(X_test)
print(f"95% CI: [{np.mean(lb):.3f}, {np.mean(ub):.3f}]")

# ========================
# 6. 数式での意味
# ========================
"""
CausalForestDMLの推定式:

Y = g(X) + τ(X) * T + ε

ここで τ(X) は 条件付き平均処置効果 (CATE):

τ(X) = E[Y(1) - Y(0) | X]

DMLでは E[Y|X] と E[T|X] を機械学習で推定して残差化し、
残差化されたデータで CATE を学習します。
"""

# ========================
# 7. 可視化
# ========================
import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(ite, kde=True)
plt.title("Estimated Individual Treatment Effects (CATE)")
plt.xlabel("τ(X)")
plt.show()
