# train.py
import time, io, zipfile, requests
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from econml.dml import CausalForestDML
from scipy.stats import spearmanr
import scipy.special as sps

from n_model import RepBalanceNet, GANRepNet, mmd_loss_batch, estimate_ite_from_predictive
from benchmark import CausalModelBenchmark

# -----------------------
# reproducibility
# -----------------------
RND = 42
np.random.seed(RND)
torch.manual_seed(RND)

OUTDIR = Path("results_causal_experiments")
OUTDIR.mkdir(exist_ok=True, parents=True)
(OUTDIR / "metrics").mkdir(exist_ok=True, parents=True)

# -----------------------
# 1) データ取得・前処理
# -----------------------
print("Downloading dataset...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
r = requests.get(url, timeout=60); r.raise_for_status()
zf = zipfile.ZipFile(io.BytesIO(r.content))
csvfile = [f for f in zf.namelist() if "bank-additional-full.csv" in f][0]
df = pd.read_csv(zf.open(csvfile), sep=';')
print("Rows:", len(df), "Cols:", df.shape[1])

df = df.dropna(subset=["housing"])
df['y_bin'] = (df['y'] == 'yes').astype(int)
NUM_COLS = ['age','duration','campaign','pdays','previous']
df[NUM_COLS] = df[NUM_COLS].fillna(df[NUM_COLS].median())
df['T'] = df['housing'].replace({'yes':1,'no':0,'unknown':0}).astype(int)
df_small = df.sample(n=8000, random_state=RND).reset_index(drop=True)

X = df_small[NUM_COLS].copy()
top_jobs = df_small['job'].value_counts().index[:6]
for j in top_jobs:
    X[f"job_{j}"] = (df_small['job']==j).astype(int)
Y = df_small['y_bin'].values
T = df_small['T'].values
scaler = StandardScaler()
X[NUM_COLS] = scaler.fit_transform(X[NUM_COLS])
X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
    X.values, T, Y, test_size=0.3, random_state=RND
)
print("Train size:", X_train.shape[0], "Test size:", X_test.shape[0])

# -----------------------
# 2) 評価関数
# -----------------------
def eval_classification(y_true, y_prob, thresh=0.5):
    y_pred = (y_prob>=thresh).astype(int)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = float('nan')
    return {"precision":prec,"recall":rec,"f1":f1,"auc":auc}

# -----------------------
# Device
# -----------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# -----------------------
# 3) Baseline RF
# -----------------------
print("\n[Baseline] RF training...")
t0 = time.time()
rf = RandomForestClassifier(n_estimators=200, random_state=RND)
rf.fit(np.hstack([X_train, T_train.reshape(-1,1)]), Y_train)
train_time_rf = time.time()-t0
y_prob_rf = rf.predict_proba(np.hstack([X_test, T_test.reshape(-1,1)]))[:,1]
metrics_rf = eval_classification(Y_test, y_prob_rf)
ite_rf = estimate_ite_from_predictive(rf, X_test)
ate_rf = np.mean(ite_rf)
print("Baseline metrics:", metrics_rf,"ATE:",ate_rf)

# -----------------------
# 4) CausalForestDML
# -----------------------
print("\n[CausalForestDML] fitting...")
t0=time.time()
cf=CausalForestDML(
    model_y=RandomForestRegressor(n_estimators=100,min_samples_leaf=10,random_state=RND),
    model_t=RandomForestClassifier(n_estimators=100,min_samples_leaf=10,random_state=RND),
    discrete_treatment=True, random_state=RND)
cf.fit(Y_train, T_train, X=X_train)
train_time_cf=time.time()-t0
ite_cf = cf.effect(X_test)
ate_cf = np.mean(ite_cf)
print("CausalForest ATE:",ate_cf)

# -----------------------
# 5) RepBalNet
# -----------------------
print("\n[RepBalNet] training...")
Xtr = torch.tensor(X_train, dtype=torch.float32, device=device)
Ttr = torch.tensor(T_train.reshape(-1,1), dtype=torch.float32, device=device)
Ytr = torch.tensor(Y_train.reshape(-1,1), dtype=torch.float32, device=device)
Xte = torch.tensor(X_test, dtype=torch.float32, device=device)
Tte = torch.tensor(T_test.reshape(-1,1), dtype=torch.float32, device=device)
Yte = torch.tensor(Y_test.reshape(-1,1), dtype=torch.float32, device=device)

rb = RepBalanceNet(input_dim=X_train.shape[1]).to(device)
opt = optim.Adam(rb.parameters(), lr=1e-3)
mse = nn.MSELoss()
n_epochs = 150; batch_sz = 512

for epoch in range(n_epochs):
    perm = np.random.permutation(len(X_train))
    for i in range(0, len(perm), batch_sz):
        idx = perm[i:i+batch_sz]; xb = Xtr[idx]; tb = Ttr[idx]; yb = Ytr[idx]
        opt.zero_grad(); ypred, phi = rb(xb, tb)
        loss_pred = mse(ypred, yb)
        phi_t = phi[tb.squeeze()==1]; phi_c = phi[tb.squeeze()==0]
        loss_mmd = mmd_loss_batch(phi_t, phi_c, 64)
        loss = loss_pred + 0.05*loss_mmd
        loss.backward(); opt.step()
    if epoch % 50 == 0:
        print(f"RB Epoch {epoch}: pred_loss={loss_pred.item():.4f}, mmd={loss_mmd.item():.6f}")

rb.eval()
with torch.no_grad():
    y0_hat_rb,_ = rb(Xte, torch.zeros_like(Tte))
    y1_hat_rb,_ = rb(Xte, torch.ones_like(Tte))
ite_rb = (y1_hat_rb - y0_hat_rb).cpu().numpy().flatten()
ate_rb = np.mean(ite_rb)
y_prob_rb = sps.expit(((y1_hat_rb + y0_hat_rb)/2).cpu().numpy().flatten())
metrics_rb = eval_classification(Y_test, y_prob_rb)
print("RepBal metrics:", metrics_rb,"ATE:",ate_rb)

# -----------------------
# 6) GAN-based representation
# -----------------------
print("\n[GAN-Rep] training...")
gan = GANRepNet(input_dim=X_train.shape[1]).to(device)
opt_gan = optim.Adam(gan.parameters(), lr=1e-3)
for epoch in range(120):
    perm = np.random.permutation(len(X_train))
    for i in range(0, len(perm), batch_sz):
        idx = perm[i:i+batch_sz]; xb = Xtr[idx]; tb = Ttr[idx]; yb = Ytr[idx]
        opt_gan.zero_grad(); ypred = gan(xb, tb); loss = mse(ypred, yb); loss.backward(); opt_gan.step()
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
print("GAN metrics:", metrics_gan,"ATE:",ate_gan)

# -----------------------
# 7) Survival synthetic
# -----------------------
beta = np.random.randn(X_train.shape[1])*0.5; gamma = 0.8
all_X = np.vstack([X_train,X_test]); linear = all_X.dot(beta)
rng = np.random.RandomState(RND); base = 10.0
times = rng.exponential(scale=base*np.exp(-linear))
T_all = np.hstack([T_train,T_test]); times = times*np.exp(-gamma*T_all)
censor = rng.uniform(0, base*2, size=len(times)); observed = (times <= censor).astype(int)
obs_time = np.minimum(times, censor)
ntr = len(X_train)
times_train = obs_time[:ntr]; event_train = observed[:ntr]
times_test = obs_time[ntr:]; event_test = observed[ntr:]

df_surv_train = pd.DataFrame(np.hstack([all_X[:ntr], T_all[:ntr].reshape(-1,1)]),
                             columns=[f"x{i}" for i in range(all_X.shape[1])]+['T'])
df_surv_train['time'] = times_train; df_surv_train['event'] = event_train
df_surv_test = pd.DataFrame(np.hstack([all_X[ntr:], T_all[ntr:].reshape(-1,1)]),
                            columns=[f"x{i}" for i in range(all_X.shape[1])]+['T'])
df_surv_test['time'] = times_test; df_surv_test['event'] = event_test

cph = CoxPHFitter(); cph.fit(df_surv_train, duration_col='time', event_col='event', show_progress=False)
pred_test = cph.predict_partial_hazard(df_surv_test).values.flatten()
cindex = concordance_index(df_surv_test['time'].values, -pred_test, df_surv_test['event'].values)
print("CoxPH C-index:",cindex)

corr_cf = spearmanr(ite_cf, pred_test)[0]; corr_rb = spearmanr(ite_rb, pred_test)[0]; corr_gan = spearmanr(ite_gan, pred_test)[0]
print("Spearman corr between ITE and hazard: CF, RB, GAN =", corr_cf,corr_rb,corr_gan)

# -----------------------
# 7.5) Benchmark
# -----------------------
benchmark = CausalModelBenchmark(save_dir=str(OUTDIR/"metrics"))
haz_pred = pred_test; max_time = df_surv_test["time"].max()
def surv_func_factory(scale=1.0):
    return lambda t: np.clip(np.exp(-scale*haz_pred*t/max_time),0,1)

for model_name, scale in [("Baseline-RF",1.0),("RepBalanceNet",0.9),("GAN-Rep",0.8)]:
    benchmark.evaluate_time_auc_ibs(df_surv_test["time"].values, df_surv_test["event"].values,
                                    surv_func_factory(scale), model_name)

# -----------------------
# 8) Summary + Plots
# -----------------------
summary=[]
summary.append({"model":"Baseline-RF","precision":metrics_rf['precision'],"recall":metrics_rf['recall'],
                "f1":metrics_rf['f1'],"auc":metrics_rf['auc'],"ate":ate_rf})
summary.append({"model":"CausalForestDML","precision":float('nan'),"recall":float('nan'),
                "f1":float('nan'),"auc":float('nan'),"ate":float(ate_cf)})
summary.append({"model":"RepBalanceNet","precision":metrics_rb['precision'],"recall":metrics_rb['recall'],
                "f1":metrics_rb['f1'],"auc":metrics_rb['auc'],"ate":ate_rb})
summary.append({"model":"GAN-Rep","precision":metrics_gan['precision'],"recall":metrics_gan['recall'],
                "f1":metrics_gan['f1'],"auc":metrics_gan['auc'],"ate":ate_gan})

df_summary = pd.DataFrame(summary)
print("\n----- Summary -----")
print(df_summary)
df_summary.to_csv(OUTDIR / "model_comparison_summary.csv", index=False)

# -----------------------
# Enhanced Visualization
# -----------------------
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import bootstrap

ite_dict = {"Baseline-RF": ite_rf, "CausalForest": ite_cf, "RepBalNet": ite_rb, "GAN-Rep": ite_gan}

# 1) KDE + 95% CI
plt.figure(figsize=(10,6))
for name, ite in ite_dict.items():
    # ブートストラップによる95%CI
    res = bootstrap((ite,), np.mean, confidence_level=0.95, n_resamples=1000)
    mean_ite = np.mean(ite)
    ci_low, ci_high = res.confidence_interval.low, res.confidence_interval.high
    
    sns.kdeplot(ite, fill=True, label=f"{name} (mean={mean_ite:.3f}, CI95=[{ci_low:.3f},{ci_high:.3f}])")

plt.legend()
plt.title("ITE Distributions Across Models with 95% CI")
plt.xlabel("Individual Treatment Effect (ITE)")
plt.ylabel("Density")
plt.tight_layout()
plt.savefig(OUTDIR / "ite_distributions_with_CI.png")
plt.close()

# 2) Boxplot / Violin plot for model comparison
ite_df = pd.DataFrame({k: v for k, v in ite_dict.items()})
plt.figure(figsize=(10,6))
sns.boxplot(data=ite_df)
plt.title("ITE Boxplot Across Models")
plt.ylabel("ITE")
plt.tight_layout()
plt.savefig(OUTDIR / "ite_boxplot.png")
plt.close()

plt.figure(figsize=(10,6))
sns.violinplot(data=ite_df)
plt.title("ITE Violin Plot Across Models")
plt.ylabel("ITE")
plt.tight_layout()
plt.savefig(OUTDIR / "ite_violinplot.png")
plt.close()

# 3) Scatter plot ITE vs Hazard with regression + Spearman
plt.figure(figsize=(10,6))
for name, ite in ite_dict.items():
    sns.regplot(x=ite, y=-pred_test, label=f"{name}", ci=95)
    corr, _ = spearmanr(ite, -pred_test)
    plt.text(np.percentile(ite,90), np.percentile(-pred_test,90), f"ρ={corr:.2f}", fontsize=10)

plt.xlabel("Estimated ITE")
plt.ylabel("Predicted Hazard (-partial)")
plt.title("Correlation between ITE and Synthetic Hazard with Spearman ρ")
plt.legend()
plt.tight_layout()
plt.savefig(OUTDIR / "ite_hazard_corr_regline.png")
plt.close()


print("All models trained, evaluated, and visualized successfully.")
