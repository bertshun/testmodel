"""
Causal Lattice Model - Full Stable & Visual Version
---------------------------------------------------
拡張データ（Wine x100）、欠損処理、NaN安定化、学習アニメーション付き
"""

import os, math, torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import numpy as np, matplotlib.pyplot as plt, seaborn as sns, networkx as nx
from matplotlib.animation import FuncAnimation

# =====================================================
# Utility
# =====================================================
def pairwise_distances(x):
    x = torch.nan_to_num(x)
    xx = (x ** 2).sum(dim=1, keepdim=True)
    dist2 = xx + xx.t() - 2 * (x @ x.t())
    return torch.sqrt(torch.clamp(dist2, min=0.0) + 1e-8)

def knn_adj(z, k=6):
    dist = pairwise_distances(z)
    dist.fill_diagonal_(float('inf'))
    _, idx = torch.topk(-dist, k=k, dim=1)
    A = torch.zeros((z.size(0), z.size(0)), device=z.device)
    rows = torch.arange(z.size(0), device=z.device).unsqueeze(1).repeat(1, k)
    A[rows.flatten(), idx.flatten()] = 1.0
    A = ((A + A.t()) > 0).float()
    return A

# =====================================================
# Model
# =====================================================
class CausalEncoder(nn.Module):
    def __init__(self, in_dim, z_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, z_dim)
        )
    def forward(self, x):
        x = torch.nan_to_num(x)
        return self.net(x)

class LatticeEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, k=6):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.k = k
        self._A = None
    def forward(self, z):
        z = torch.nan_to_num(z)
        A = knn_adj(z, k=self.k)
        self._A = A.detach().cpu()
        h = F.relu(self.fc1(z))
        h = (A @ h) / (A.sum(dim=1, keepdim=True) + 1e-6)
        h = F.relu(self.fc2(h))
        return h, A
    def get_graph(self):
        if self._A is None: return None
        A = self._A.numpy()
        return nx.from_numpy_array((A > 0.5).astype(int))

class CausalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.to_q, self.to_k, self.to_v = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
    def forward(self, H, T):
        H, T = torch.nan_to_num(H), torch.nan_to_num(T)
        dist = pairwise_distances(T)
        scale = 1.0 / (1.0 + dist.mean(dim=1, keepdim=True))
        Q, K, V = self.to_q(H), self.to_k(H), self.to_v(H)
        attn = torch.softmax(Q @ K.T / math.sqrt(Q.size(-1)), dim=-1)
        A = (attn * scale)
        return self.out(A @ V)

class MetaUtility(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim // 2), nn.ReLU(), nn.Linear(dim // 2, 1), nn.Sigmoid())
    def forward(self, A): return A * self.net(A)

class OutputHead(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim // 2), nn.ReLU(), nn.Linear(dim // 2, num_classes))
    def forward(self, x): return self.net(x)

class FullModel(nn.Module):
    def __init__(self, in_dim, num_classes=3):
        super().__init__()
        self.causal_encoder = CausalEncoder(in_dim)
        self.lattice_encoder = LatticeEncoder(16)
        self.causal_attention = CausalAttention(64)
        self.meta_utility = MetaUtility(64)
        self.output_head = OutputHead(64, num_classes)
    def forward(self, X, T):
        z = self.causal_encoder(X)
        H, A_lat = self.lattice_encoder(z)
        A = self.causal_attention(H, T)
        O = self.meta_utility(A)
        Y_hat = self.output_head(O)
        return Y_hat, z, H, A

# =====================================================
# Loss
# =====================================================
def task_loss(Y_hat, Y): return F.cross_entropy(Y_hat, Y)
def causal_loss(Y_hat, Y, T):
    err = (Y_hat.softmax(dim=1).max(dim=1).values -
           F.one_hot(Y, num_classes=Y_hat.size(1)).float().max(dim=1).values)
    return torch.abs(torch.mean(err * T.mean(dim=1)))
def geometric_loss(H): return torch.cdist(H, H).mean()
def regularization(z, Y): return 1e-3 * (z ** 2).mean()

# =====================================================
# Config & Data
# =====================================================
EPOCHS, λ1, λ2 = 50, 0.5, 0.2
device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
os.makedirs("results/frames", exist_ok=True)

wine = load_wine()
X, y = wine.data.astype(np.float32), wine.target.astype(np.int64)
scaler = StandardScaler().fit(X)
X = scaler.transform(X)
# --- 100倍拡張 + 欠損導入 & 処理 ---
X_aug = np.tile(X, (100, 1)) + np.random.normal(0, 0.02, (len(X)*100, X.shape[1]))
y_aug = np.tile(y, 100)
mask = np.random.rand(*X_aug.shape) < 0.01
X_aug[mask] = np.nan
X_aug = np.nan_to_num(X_aug, nan=np.nanmean(X_aug))

X = torch.tensor(X_aug, dtype=torch.float32)
y = torch.tensor(y_aug, dtype=torch.long)
T = X.clone()
dataset = torch.utils.data.TensorDataset(X, y, T)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)

# =====================================================
# Train
# =====================================================
model = FullModel(in_dim=X.size(1)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

frames = []
for epoch in range(EPOCHS):
    model.train()
    for Xb, Yb, Tb in dataloader:
        Xb, Yb, Tb = Xb.to(device), Yb.to(device), Tb.to(device)
        Y_hat, z, H, A = model(Xb, Tb)
        L_task = task_loss(Y_hat, Yb)
        L_causal = causal_loss(Y_hat, Yb, Tb)
        L_geo = geometric_loss(H)
        L_reg = regularization(z, Yb)
        L_total = L_task + λ1*L_causal + λ2*L_geo + L_reg
        optimizer.zero_grad()
        L_total.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    print(f"[Epoch {epoch+1:03d}] Total={L_total.item():.4f}, Task={L_task.item():.4f}")

    # --- 各epochごとに可視化 ---
    with torch.no_grad():
        z_all = model.causal_encoder(X.to(device)).detach().cpu().numpy()
        z_all = np.nan_to_num(z_all)
        z_emb = TSNE(n_components=2, perplexity=40, init='random').fit_transform(z_all)
        fig, ax = plt.subplots(figsize=(6,5))
        sns.scatterplot(x=z_emb[:,0], y=z_emb[:,1], hue=y, palette='Spectral', s=10, alpha=0.8, ax=ax, legend=False)
        ax.set_title(f"Epoch {epoch+1} | L={L_total.item():.3f}\n$L=L_t+λ_1L_c+λ_2L_g+L_r$")
        ax.set_xlabel("z₁"); ax.set_ylabel("z₂")
        frame_path = f"results/frames/frame_{epoch:03d}.png"
        plt.savefig(frame_path); plt.close()
        frames.append(frame_path)

# =====================================================
# Evaluation & Summary Plots
# =====================================================
model.eval()
with torch.no_grad():
    Y_hat, _, _, _ = model(X.to(device), T.to(device))
    preds = Y_hat.argmax(dim=1)
    acc = (preds == y.to(device)).float().mean().item()
print(f"✅ Accuracy: {acc:.4f}")

# --- Animate ---
import imageio
gif_path = "results/latent_evolution.gif"
imageio.mimsave(gif_path, [imageio.imread(f) for f in frames], duration=0.4)

# --- Final Density (改良版) ---
z = model.causal_encoder(X.to(device)).detach().cpu().numpy()
z = np.nan_to_num(z)  # NaN/inf 安定化

# t-SNE埋め込み
z_emb = TSNE(n_components=2, perplexity=40, init='random').fit_transform(z)

# 有効データのみ
mask = np.all(np.isfinite(z_emb), axis=1)
z_clean = z_emb[mask]

plt.figure(figsize=(6,5))
try:
    # KDEプロット安定化
    sns.kdeplot(
        x=z_clean[:,0],
        y=z_clean[:,1],
        fill=True,
        cmap="inferno",
        levels=20,  # 安定化
        thresh=0.05
    )
    plt.title(f"Final Density | Acc={acc:.3f}, λ₁={λ1}, λ₂={λ2}")

    # ばらつき・密集度を数値表示
    dispersion = np.mean(np.linalg.norm(z_clean - z_clean.mean(axis=0), axis=1))
    density = 1.0 / (1.0 + dispersion)
    plt.text(
        0.02, 0.95,
        f"Dispersion={dispersion:.3f}\nDensity={density:.3f}",
        transform=plt.gca().transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
    )

    plt.xlabel("z₁"); plt.ylabel("z₂")
    plt.tight_layout()
    plt.savefig("results/final_density.png", dpi=300)
    plt.close()
except Exception as e:
    print(f"⚠ KDEPlot skipped due to: {e}")

# --- Causal Lattice ---
G = model.lattice_encoder.get_graph()
if G is not None:
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(6,6))
    nx.draw(G, pos, node_color='orange', edge_color='gray', node_size=40)
    plt.title(f"Causal Lattice\nAcc={acc:.3f}, λ₁={λ1}, λ₂={λ2}")
    plt.savefig("results/final_lattice.png", dpi=300)
    plt.close()

print(f"✅ All results saved to ./results/")
