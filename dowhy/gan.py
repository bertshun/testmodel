# causal_experiments.py
import time
import io
import zipfile
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from econml.dml import CausalForestDML
from dowhy import CausalModel
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

# reproducibility
RND = 42
np.random.seed(RND)
torch.manual_seed(RND)

OUTDIR = Path("results_causal_experiments")
OUTDIR.mkdir(exist_ok=True)

# -----------------------
# 1) データ取得・前処理
# -----------------------
print("Downloading dataset...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
r = requests.get(url, timeout=60)
r.raise_for_status()
zf = zipfile.ZipFile(io.BytesIO(r.content))
csvfile = [f for f in zf.namelist() if "bank-additional-full.csv" in f][0]
df = pd.read_csv(zf.open(csvfile), sep=';')
print("Rows:", len(df), "Cols:", df.shape[1])

# target y: 'y' -> binary
df = df.dropna(subset=["housing"])  # ensure treatment exists
df['y_bin'] = (df['y'] == 'yes').astype(int)

# for simplicity pick numeric columns and a few categorical mapped
NUM_COLS = ['age','duration','campaign','pdays','previous']
# fill numeric nulls
df[NUM_COLS] = df[NUM_COLS].fillna(df[NUM_COLS].median())

# treatment T: 'housing' (yes/no/unknown) -> binary
df['T'] = df['housing'].replace({'yes':1, 'no':0, 'unknown':0}).astype(int)

# select subset to keep runtime small
df_small = df.sample(n=8000, random_state=RND).reset_index(drop=True)

# features X
X = df_small[NUM_COLS].copy()
# optionally add some categorical encodings (use one-hot for 'job' top categories)
top_jobs = df_small['job'].value_counts().index[:6]
for j in top_jobs:
    X[f"job_{j}"] = (df_small['job'] == j).astype(int)

Y = df_small['y_bin'].values
T = df_small['T'].values

# standardize numeric
scaler = StandardScaler()
X[NUM_COLS] = scaler.fit_transform(X[NUM_COLS])

# train/test split
X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
    X.values, T, Y, test_size=0.3, random_state=RND
)
print("Train size:", X_train.shape[0], "Test size:", X_test.shape[0])

# -----------------------
# 2) 評価関数
# -----------------------
def eval_classification(y_true, y_prob, thresh=0.5):
    y_pred = (y_prob >= thresh).astype(int)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float('nan')
    return {"precision":prec,"recall":rec,"f1":f1,"auc":auc}

# -----------------------
# 3) Baseline predictive classifier (RandomForest)
#    - used as predictive baseline and also to derive naive ITE by toggling T
# -----------------------
print("\n[Baseline] Training RandomForestClassifier...")
t0 = time.time()
rf = RandomForestClassifier(n_estimators=200, random_state=RND)
rf.fit(np.hstack([X_train, T_train.reshape(-1,1)]), Y_train)
train_time_rf = time.time() - t0
# predict probability on test
y_prob_rf = rf.predict_proba(np.hstack([X_test, T_test.reshape(-1,1)]))[:,1]
metrics_rf = eval_classification(Y_test, y_prob_rf)
print("Baseline metrics:", metrics_rf, "train_time:", train_time_rf)

# naive ITE from predictive model (difference when toggling T)
def estimate_ite_from_predictive(model, X, t0=0, t1=1):
    X = np.array(X)
    p0 = model.predict_proba(np.hstack([X, np.full((len(X),1), t0)]))[:,1]
    p1 = model.predict_proba(np.hstack([X, np.full((len(X),1), t1)]))[:,1]
    return p1 - p0

ite_rf = estimate_ite_from_predictive(rf, X_test)
ate_rf = np.mean(ite_rf)
print("Baseline ATE (naive):", ate_rf)

# -----------------------
# 4) CausalForestDML
# -----------------------
print("\n[CausalForestDML] fitting...")
t0 = time.time()
cf = CausalForestDML(
    model_y=RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=RND),
    model_t=RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=RND),
    discrete_treatment=True,
    random_state=RND
)
cf.fit(Y_train, T_train, X=X_train)
train_time_cf = time.time() - t0
ite_cf = cf.effect(X_test)
ate_cf = np.mean(ite_cf)
# For classification evaluation we need predicted probabilities — derive via Y|T=actual predicted by models inside CF? We'll use naive predictive wrapper:
# Use cf.const_marginal_effect? here we report ITE-based ATE and ITE distribution
print("CausalForest ATE:", ate_cf, "train_time:", train_time_cf)

# -----------------------
# 5) Representation Balancing (RepBalNet) with mini-batch MMD
# -----------------------
print("\n[RepBalNet] training (mini-batch MMD)...")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
class RepBalanceNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
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

def mmd_loss_batch(phi_t, phi_c, batch_size=128, gamma=1.0):
    # downsample batches from each group
    if len(phi_t) == 0 or len(phi_c) == 0:
        return torch.tensor(0.0, device=phi_t.device if len(phi_t)>0 else phi_c.device)
    bt = phi_t[torch.randperm(len(phi_t))[:min(batch_size,len(phi_t))]]
    bc = phi_c[torch.randperm(len(phi_c))[:min(batch_size,len(phi_c))]]
    def gaussian_kernel(x,y):
        x_norm = (x**2).sum(1).view(-1,1)
        y_norm = (y**2).sum(1).view(1,-1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
        return torch.exp(-gamma * dist)
    K_tt = gaussian_kernel(bt, bt)
    K_cc = gaussian_kernel(bc, bc)
    K_tc = gaussian_kernel(bt, bc)
    return K_tt.mean() + K_cc.mean() - 2*K_tc.mean()

Xtr = torch.tensor(X_train, dtype=torch.float32, device=device)
Ttr = torch.tensor(T_train.reshape(-1,1), dtype=torch.float32, device=device)
Ytr = torch.tensor(Y_train.reshape(-1,1), dtype=torch.float32, device=device)
Xte = torch.tensor(X_test, dtype=torch.float32, device=device)
Tte = torch.tensor(T_test.reshape(-1,1), dtype=torch.float32, device=device)
Yte = torch.tensor(Y_test.reshape(-1,1), dtype=torch.float32, device=device)

rb = RepBalanceNet(input_dim=X_train.shape[1]).to(device)
opt = optim.Adam(rb.parameters(), lr=1e-3)
mse = nn.MSELoss()
n_epochs = 150
batch_sz = 512

for epoch in range(n_epochs):
    perm = np.random.permutation(len(X_train))
    for i in range(0, len(perm), batch_sz):
        idx = perm[i:i+batch_sz]
        xb = Xtr[idx]
        tb = Ttr[idx]
        yb = Ytr[idx]
        opt.zero_grad()
        ypred, phi = rb(xb, tb)
        loss_pred = mse(ypred, yb)
        # compute MMD across full phi groups in this batch
        phi_t = phi[tb.squeeze()==1]
        phi_c = phi[tb.squeeze()==0]
        loss_mmd = mmd_loss_batch(phi_t, phi_c, batch_size=64)
        loss = loss_pred + 0.05 * loss_mmd
        loss.backward()
        opt.step()
    if epoch % 50 == 0:
        print(f"RB Epoch {epoch}: pred_loss={loss_pred.item():.4f}, mmd={loss_mmd.item():.6f}")

# Evaluate RB: estimate y|T=0 and y|T=1
rb.eval()
with torch.no_grad():
    y0_hat_rb, _ = rb(Xte, torch.zeros_like(Tte))
    y1_hat_rb, _ = rb(Xte, torch.ones_like(Tte))
ite_rb = (y1_hat_rb - y0_hat_rb).cpu().numpy().flatten()
ate_rb = np.mean(ite_rb)
print("RepBal ATE:", ate_rb)

# Also produce a classifier probability from RB by using mixed rule: p = sigmoid( y_hat ) (since y is binary 0/1)
import scipy.special as sps
y_prob_rb = sps.expit(((y1_hat_rb + y0_hat_rb)/2).cpu().numpy().flatten())  # approximation
metrics_rb = eval_classification(Y_test, y_prob_rb)
print("RepBal classification metrics (approx):", metrics_rb)

# -----------------------
# 6) GAN-based representation model (simple)
# -----------------------
print("\n[GAN-Rep] training generator-only regression (simple stable variant)...")
class GANRepNet(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, latent_dim))
        self.dec = nn.Sequential(nn.Linear(latent_dim+1, 64), nn.ReLU(), nn.Linear(64,1))
    def forward(self, x, t):
        z = self.enc(x)
        zt = torch.cat([z, t], dim=1)
        return self.dec(zt)

gan = GANRepNet(input_dim=X_train.shape[1]).to(device)
opt_gan = optim.Adam(gan.parameters(), lr=1e-3)
mse = nn.MSELoss()
for epoch in range(120):
    perm = np.random.permutation(len(X_train))
    for i in range(0, len(perm), batch_sz):
        idx = perm[i:i+batch_sz]
        xb = Xtr[idx]
        tb = Ttr[idx]
        yb = Ytr[idx]
        opt_gan.zero_grad()
        ypred = gan(xb, tb)
        loss = mse(ypred, yb)
        loss.backward()
        opt_gan.step()
    if epoch % 40 == 0:
        print(f"GAN Epoch {epoch}: loss={loss.item():.4f}")

gan.eval()
with torch.no_grad():
    y0_gan = gan(Xte, torch.zeros_like(Tte)).cpu().numpy().flatten()
    y1_gan = gan(Xte, torch.ones_like(Tte)).cpu().numpy().flatten()
ite_gan = y1_gan - y0_gan
ate_gan = np.mean(ite_gan)
y_prob_gan = sps.expit((y0_gan + y1_gan)/2.0)
metrics_gan = eval_classification(Y_test, y_prob_gan)
print("GAN-based ATE:", ate_gan, "classification metrics:", metrics_gan)

# -----------------------
# 7) Survival / hazard-style evaluation (synthetic times)
#    - We don't have real time-to-event in this dataset, so synthesize a time variable
#    - We synthesize in a way dependent on features and treatment, then fit CoxPH to test discrimination (C-index).
# -----------------------
print("\n[Survival] Generate synthetic time-to-event and evaluate C-index")

# create baseline hazard and make time dependent on X and T (so models that estimate treatment effect should relate)
# time = Exponential(scale = base * exp(-beta^T x - gamma * t))  => shorter times = higher risk
beta = np.random.randn(X_train.shape[1]) * 0.5
gamma = 0.8
all_X = np.vstack([X_train, X_test])
linear = all_X.dot(beta)
# draw times
rng = np.random.RandomState(RND)
base = 10.0
times = rng.exponential(scale= base * np.exp(-linear))
# introduce effect of T by dividing times for treated samples
T_all = np.hstack([T_train, T_test])
times = times * np.exp(-gamma * T_all)
# censoring
censor = rng.uniform(0, base*2, size=len(times))
observed = (times <= censor).astype(int)
obs_time = np.minimum(times, censor)
# split back
ntr = len(X_train)
times_train = obs_time[:ntr]; event_train=observed[:ntr]
times_test = obs_time[ntr:]; event_test=observed[ntr:]

# prepare DataFrame for lifelines CoxPH
df_surv_train = pd.DataFrame(np.hstack([all_X[:ntr], T_all[:ntr].reshape(-1,1)]),
                             columns=[f"x{i}" for i in range(all_X.shape[1])] + ['T'])
df_surv_train['time'] = times_train
df_surv_train['event'] = event_train

df_surv_test = pd.DataFrame(np.hstack([all_X[ntr:], T_all[ntr:].reshape(-1,1)]),
                             columns=[f"x{i}" for i in range(all_X.shape[1])] + ['T'])
df_surv_test['time'] = times_test
df_surv_test['event'] = event_test

# Fit CoxPH
cph = CoxPHFitter()
cph.fit(df_surv_train, duration_col='time', event_col='event', show_progress=False)
# predict partial hazard (higher = higher risk)
pred_train = cph.predict_partial_hazard(df_surv_train).values.flatten()
pred_test = cph.predict_partial_hazard(df_surv_test).values.flatten()
cindex = concordance_index(df_surv_test['time'].values, -pred_test, df_surv_test['event'].values)  # negative because larger hazard -> shorter time
print("CoxPH C-index on synthetic test:", cindex)

# We can also evaluate how ITE correlates with hazard: e.g. does larger estimated ITE -> lower survival?
# correlate ITE (from CausalForest, RepBal, GAN) with test hazards
# get predicted hazards (we have pred_test). compute Spearman correlation
from scipy.stats import spearmanr
# ensure lengths match
# use ite_cf, ite_rb, ite_gan all on X_test
corr_cf = spearmanr(ite_cf, pred_test)[0]
corr_rb = spearmanr(ite_rb, pred_test)[0]
corr_gan = spearmanr(ite_gan, pred_test)[0]
print("Spearman corr between ITE and predicted hazard (test): CF, RB, GAN =", corr_cf, corr_rb, corr_gan)

# -----------------------
# 8) Summarize metrics in a DataFrame and save
# -----------------------
summary = []
summary.append({
    "model":"Baseline-RF",
    "precision":metrics_rf['precision'],
    "recall":metrics_rf['recall'],
    "f1":metrics_rf['f1'],
    "auc":metrics_rf['auc'],
    "ate":ate_rf
})
summary.append({
    "model":"CausalForestDML",
    "precision":float('nan'),  # CF is not directly classifier
    "recall":float('nan'),
    "f1":float('nan'),
    "auc":float('nan'),
    "ate":float(ate_cf)
})
summary.append({
    "model":"RepBalanceNet",
    "precision":metrics_rb['precision'],
    "recall":metrics_rb['recall'],
    "f1":metrics_rb['f1'],
    "auc":metrics_rb['auc'],
    "ate":float(ate_rb)
})
summary.append({
    "model":"GAN-Rep",
    "precision":metrics_gan['precision'],
    "recall":metrics_gan['recall'],
    "f1":metrics_gan['f1'],
    "auc":metrics_gan['auc'],
    "ate":float(ate_gan)
})

df_summary = pd.DataFrame(summary)
print("\n----- Summary -----")
print(df_summary)
df_summary.to_csv(OUTDIR / "model_comparison_summary.csv", index=False)
print("Saved summary to", OUTDIR / "model_comparison_summary.csv")

# Optionally: plot ITE distributions
plt.figure(figsize=(8,5))
sns.kdeplot(ite_cf, label="CausalForest ITE")
sns.kdeplot(ite_rb, label="RepBal ITE")
sns.kdeplot(ite_gan, label="GAN ITE")
plt.legend(); plt.title("ITE distributions")
plt.savefig(OUTDIR / "ite_distributions.png")
print("Saved ITE plot to", OUTDIR / "ite_distributions.png")
plt.close()

print("All done.")
