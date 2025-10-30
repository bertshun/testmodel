# -------------------------
# 0. 必要パッケージ
# -------------------------
# pip install numpy pandas scikit-learn torch matplotlib seaborn econml dowhy requests

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from econml.dml import CausalForestDML
from dowhy import CausalModel

# -------------------------
# 1. データ読み込み（医療以外データ例として UCI Bank Data）
# -------------------------
import requests, zipfile, io
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
r = requests.get(url)
r.raise_for_status()
zf = zipfile.ZipFile(io.BytesIO(r.content))
csvfile = [f for f in zf.namelist() if "bank-additional-full.csv" in f][0]
df = pd.read_csv(zf.open(csvfile), sep=';')

# -------------------------
# 2. 前処理・変数設定
# -------------------------
df = df.dropna(subset=["housing"])
df['y_bin'] = (df['y']=='yes').astype(int)

X_cols = ['age', 'duration', 'campaign', 'pdays', 'previous']
T_col = 'housing'
Y_col = 'y_bin'

df[T_col] = df[T_col].replace({'no':0, 'yes':1, 'unknown':0}).astype(int)
X = df[X_cols].values
Y = df[Y_col].values
T = df[T_col].values

# スケーリング
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
    X_scaled, T, Y, test_size=0.3, random_state=42
)

# -------------------------
# 3. Representation Balancing モデル
# -------------------------
class RepBalanceNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.h0 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim,1))
        self.h1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim,1))

    def forward(self, x, t):
        phi = self.encoder(x)
        y_hat = self.h1(phi) * t + self.h0(phi) * (1 - t)
        return y_hat, phi

def mmd_loss(phi_t, phi_c, gamma=1.0):
    if len(phi_t)==0 or len(phi_c)==0:
        return torch.tensor(0.0, device=phi_t.device if len(phi_t)>0 else phi_c.device)
    def gaussian_kernel(x, y, gamma):
        x_norm = (x**2).sum(1).view(-1,1)
        y_norm = (y**2).sum(1).view(1,-1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
        return torch.exp(-gamma * dist)
    K_tt = gaussian_kernel(phi_t, phi_t, gamma)
    K_cc = gaussian_kernel(phi_c, phi_c, gamma)
    K_tc = gaussian_kernel(phi_t, phi_c, gamma)
    mmd = K_tt.mean() + K_cc.mean() - 2*K_tc.mean()
    return mmd

device = torch.device("mps" if torch.cuda.is_available() else "cpu")
model_rb = RepBalanceNet(input_dim=X_train.shape[1]).to(device)
optimizer = optim.Adam(model_rb.parameters(), lr=1e-3)
mse_loss = nn.MSELoss()

X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
T_train_t = torch.tensor(T_train.reshape(-1,1), dtype=torch.float32, device=device)
Y_train_t = torch.tensor(Y_train.reshape(-1,1), dtype=torch.float32, device=device)

for epoch in range(200):
    model_rb.train()
    optimizer.zero_grad()
    y_pred, phi = model_rb(X_train_t, T_train_t)
    loss_pred = mse_loss(y_pred, Y_train_t)

    phi_t = phi[T_train_t.squeeze()==1]
    phi_c = phi[T_train_t.squeeze()==0]
    loss_mmd = mmd_loss(phi_t, phi_c)

    loss = loss_pred + 0.1 * loss_mmd
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, pred_loss={loss_pred.item():.4f}, mmd_loss={loss_mmd.item():.4f}")

# 評価
model_rb.eval()
with torch.no_grad():
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    y0_hat, _ = model_rb(X_test_t, torch.zeros_like(T_test.reshape(-1,1)).float().to(device))
    y1_hat, _ = model_rb(X_test_t, torch.ones_like(T_test.reshape(-1,1)).float().to(device))
    ite_rb = (y1_hat - y0_hat).cpu().numpy().flatten()
ate_rb = np.mean(ite_rb)
print("Representation Balancing – Estimated ATE:", ate_rb)

# -------------------------
# 4. CausalForestDML + DoWhy
# -------------------------
df_dowhy = pd.DataFrame(np.hstack([X_scaled, T.reshape(-1,1), Y.reshape(-1,1)]),
                        columns=X_cols + [T_col, Y_col])

causal_model = CausalModel(
    data=df_dowhy,
    treatment=T_col,
    outcome=Y_col,
    common_causes=X_cols
)
identified_estimand = causal_model.identify_effect()
print(identified_estimand)

est = CausalForestDML(
    model_y=RandomForestRegressor(n_estimators=100, min_samples_leaf=10),
    model_t=RandomForestClassifier(n_estimators=100, min_samples_leaf=10),
    discrete_treatment=True,
    random_state=42
)
est.fit(Y_train, T_train, X=X_train)
ite_cf = est.effect(X_test)
ate_cf = np.mean(ite_cf)
lb_cf, ub_cf = est.effect_interval(X_test)
print("CausalForestDML – ATE:", ate_cf, "CI:", (np.mean(lb_cf), np.mean(ub_cf)))

# -------------------------
# 5. 評価指標・可視化
# -------------------------
# 数式
# ITE: \hat{\tau}(x_i) = \hat{Y}(x_i,1) - \hat{Y}(x_i,0)
# ATE: \hat{\tau}_{ATE} = (1/n) ∑ \hat{\tau}(x_i)

import matplotlib.pyplot as plt
plt.hist(ite_cf, bins=30, alpha=0.6, label="CausalForest ITE")
plt.hist(ite_rb, bins=30, alpha=0.6, label="RepBal ITE")
plt.legend()
plt.title("Comparison of Individual Treatment Effects (ITE)")
plt.show()

print("\nSummary:")
print("RepBal ATE:", ate_rb)
print("CausalForest ATE:", ate_cf, "CI:", (np.mean(lb_cf), np.mean(ub_cf)))
