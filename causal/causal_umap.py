"""
causal_lattice_full.py
Complete single-file implementation:
- data 100x expansion (sklearn wine)
- model, train, eval
- visualizations: t-SNE, UMAP, density heatmap, lattice graph
- clustering: DBSCAN + HDBSCAN (when available), silhouette
- saves PNGs into ./results/

Requires:
pip install torch scikit-learn matplotlib seaborn networkx umap-learn hdbscan
(hdbscan optional; umap-learn recommended)
"""

import os
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA

# try optional libs
try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except Exception:
    HDBSCAN_AVAILABLE = False

# --------------- Utilities ---------------
def pairwise_distances(x):
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

# --------------- Model ---------------
class CausalEncoder(nn.Module):
    def __init__(self, in_dim, z_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, z_dim)
        )
    def forward(self, x): return self.net(x)

class LatticeEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, k=6):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.k = k
        self._A = None
    def forward(self, z):
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
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
    def forward(self, H, T):
        # scale uses mean causal distance (simple proxy)
        dist = pairwise_distances(T)
        scale = 1.0 / (1.0 + dist.mean(dim=1, keepdim=True))
        Q, K, V = self.to_q(H), self.to_k(H), self.to_v(H)
        attn = torch.softmax(Q @ K.T / math.sqrt(Q.size(-1)), dim=-1)
        A = (attn * scale)
        return self.out(A @ V)

class MetaUtility(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.ReLU(),
            nn.Linear(dim // 2, 1), nn.Sigmoid()
        )
    def forward(self, A):
        u = self.net(A)
        return A * u

class OutputHead(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.ReLU(),
            nn.Linear(dim // 2, num_classes)
        )
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

# --------------- Losses ---------------
def task_loss(Y_hat, Y): 
    # Cross-entropy: implemented by torch
    return F.cross_entropy(Y_hat, Y)

def causal_loss(Y_hat, Y, T):
    # our simple proxy: correlation between prediction error and T
    # err_i = max_prob_i - one_hot_max_of_label_i  (scalar per sample)
    probs = F.softmax(Y_hat, dim=1)
    maxprob_vals, _ = probs.max(dim=1)
    # one-hot max (always 1 for correct class), so we compute difference between maxprob and 1 for correct label
    # but simpler: build scalar error as (maxprob - indicator_of_true)
    true_one_hot_max = F.one_hot(Y, num_classes=Y_hat.size(1)).float().max(dim=1).values
    err = maxprob_vals - true_one_hot_max
    # T summary per sample: mean across features as proxy
    T_mean = T.mean(dim=1)
    corr = torch.abs(torch.mean(err * T_mean))
    return corr

def geometric_loss(H):
    # average pairwise distance (coarse)
    return torch.cdist(H, H).mean()

def regularization(z, Y, alpha=1e-3):
    return alpha * (z ** 2).mean()

# --------------- Config ---------------
EPOCHS = 40
LAMBDA1 = 0.5  # lambda for causal loss
LAMBDA2 = 0.2  # lambda for geo loss
BATCH_SIZE = 512
LR = 1e-3
DEVICE = torch.device('mps' if torch.cuda.is_available() else 'cpu')
RESULT_DIR = "results/umap"
os.makedirs(RESULT_DIR, exist_ok=True)

# --------------- Data: load + expand 100x ---------------
wine = load_wine()
X0 = wine.data.astype(np.float32)
y0 = wine.target.astype(np.int64)
scaler = StandardScaler().fit(X0)
X_scaled = scaler.transform(X0)

repeat_factor = 100
# repeat each sample and add small gaussian noise (preserve distribution)
X_big = np.repeat(X_scaled, repeat_factor, axis=0)
noise = np.random.normal(loc=0.0, scale=0.02, size=X_big.shape).astype(np.float32)
X_big += noise
y_big = np.repeat(y0, repeat_factor, axis=0)

X = torch.tensor(X_big, dtype=torch.float32)
y = torch.tensor(y_big, dtype=torch.long)
T = X.clone()

print(f"Dataset expanded: {X.shape[0]} samples, {X.shape[1]} features (repeat={repeat_factor})")

dataset = torch.utils.data.TensorDataset(X, y, T)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

# --------------- Model init ---------------
model = FullModel(in_dim=X.size(1), num_classes=len(np.unique(y0))).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# --------------- Training loop (your exact structure) ---------------
start_time = time.time()
for epoch in range(EPOCHS):
    running = {}
    for Xb, Yb, Tb in dataloader:
        Xb, Yb, Tb = Xb.to(DEVICE), Yb.to(DEVICE), Tb.to(DEVICE)
        # 1. forward
        Y_hat, z, H, A = model(Xb, Tb)
        # 2. loss components
        L_task = task_loss(Y_hat, Yb)
        L_causal = causal_loss(Y_hat, Yb, Tb)
        L_geo = geometric_loss(H)
        L_reg = regularization(z, Yb)
        L_total = L_task + LAMBDA1 * L_causal + LAMBDA2 * L_geo + L_reg
        # 3. backward
        optimizer.zero_grad()
        L_total.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running = {"L_total": L_total.item(), "L_task": L_task.item(), "L_causal": L_causal.item(), "L_geo": L_geo.item()}
    print(f"[{epoch+1:03d}/{EPOCHS}] L_total={running['L_total']:.4f} task={running['L_task']:.4f} causal={running['L_causal']:.6f} geo={running['L_geo']:.4f}")

elapsed = time.time() - start_time
print(f"Training done in {elapsed/60:.2f} min")

# --------------- Evaluation (full data) ---------------
model.eval()
with torch.no_grad():
    Y_hat_all, z_all, H_all, A_all = model(X.to(DEVICE), T.to(DEVICE))
    preds = Y_hat_all.argmax(dim=1).cpu().numpy()
    acc = (preds == y.numpy()).mean()
    print(f"Evaluation accuracy (full expanded data): {acc:.4f}")

    # confusion & classification report on original classes
    cm = confusion_matrix(y.numpy(), preds)
    report = classification_report(y.numpy(), preds, digits=4)
    print("Confusion matrix:\n", cm)
    print("Classification report:\n", report)

# Save model weights
torch.save(model.state_dict(), os.path.join(RESULT_DIR, "model_state.pth"))

# --------------- Embeddings for visualization (subsample for speed) ---------------
N_VIS = min(3000, X.shape[0])
subset_idx = np.random.choice(len(X), size=N_VIS, replace=False)
X_vis = X[subset_idx].to(DEVICE)
y_vis = y[subset_idx].numpy()

with torch.no_grad():
    z_vis = model.causal_encoder(X_vis).cpu().numpy()
# optionally also H embeddings:
with torch.no_grad():
    _, z_all_full, H_all_full, _ = model(X.to(DEVICE), T.to(DEVICE))
    H_all_np = H_all_full.cpu().numpy()
    z_all_np = z_all_full.cpu().numpy()

# --------------- Dimensionality reduction: t-SNE + UMAP + PCA ---------------
print("Computing embeddings for visualization...")
tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
z_tsne = tsne.fit_transform(z_vis)  # (N_vis, 2)

if UMAP_AVAILABLE:
    reducer = umap.UMAP(n_components=2, random_state=42)
    z_umap = reducer.fit_transform(z_vis)
else:
    z_umap = None

# also PCA for a quick baseline
pca = PCA(n_components=2)
z_pca = pca.fit_transform(z_vis)

# --------------- Save t-SNE scatter with parameters text ---------------
plt.figure(figsize=(8,6))
sns.scatterplot(x=z_tsne[:,0], y=z_tsne[:,1], hue=y_vis, palette='Spectral', s=25, alpha=0.9, linewidth=0)
plt.title(f"t-SNE Latent Causal Space (N={N_VIS})\nλ1={LAMBDA1}, λ2={LAMBDA2}, epochs={EPOCHS}, acc={acc:.4f}")
plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
plt.legend(title="Class", loc='best')
plt.tight_layout()
fn = os.path.join(RESULT_DIR, "latent_tsne.png")
plt.savefig(fn, dpi=200); plt.close()
print("Saved", fn)

# --------------- Save UMAP scatter (if available) ---------------
if z_umap is not None:
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=z_umap[:,0], y=z_umap[:,1], hue=y_vis, palette='Spectral', s=25, alpha=0.9, linewidth=0)
    plt.title(f"UMAP Latent Causal Space (N={N_VIS})\nλ1={LAMBDA1}, λ2={LAMBDA2}, epochs={EPOCHS}, acc={acc:.4f}")
    plt.xlabel("UMAP 1"); plt.ylabel("UMAP 2")
    plt.legend(title="Class", loc='best')
    plt.tight_layout()
    fn = os.path.join(RESULT_DIR, "latent_umap.png")
    plt.savefig(fn, dpi=200); plt.close()
    print("Saved", fn)
else:
    print("UMAP not available; skip.")

# --------------- Density heatmap (kde) ---------------
plt.figure(figsize=(7,6))
sns.kdeplot(x=z_tsne[:,0], y=z_tsne[:,1], fill=True, cmap="mako", thresh=0.05, levels=100)
plt.title("Density heatmap on t-SNE latent space")
plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
plt.tight_layout()
fn = os.path.join(RESULT_DIR, "density_heatmap_tsne.png")
plt.savefig(fn, dpi=200); plt.close()
print("Saved", fn)

# --------------- Lattice graph visualization ---------------
G = model.lattice_encoder.get_graph()
if G is not None:
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(6,6))
    nx.draw(G, pos, node_color='orange', edge_color='gray', node_size=30, alpha=0.8)
    plt.title(f"Learned causal lattice (k={model.lattice_encoder.k})")
    plt.tight_layout()
    fn = os.path.join(RESULT_DIR, "causal_lattice.png")
    plt.savefig(fn, dpi=200); plt.close()
    print("Saved", fn)
else:
    print("No lattice graph available")

# --------------- Clustering: DBSCAN + HDBSCAN ---------------
# Use z_vis (reduced latent vectors) for clustering (can choose t-SNE, UMAP, or raw z)
embedding_for_clust = z_vis  # raw latent z (not reduced) is also fine; use raw or UMAP/TSNE
# convert to numpy
emb_np = z_vis.cpu().numpy() if torch.is_tensor(z_vis) else z_vis

# DBSCAN - choose eps via heuristic (small dataset: 0.5 as start)
db = DBSCAN(eps=0.5, min_samples=10).fit(emb_np)
labels_db = db.labels_
n_clusters_db = len(set(labels_db)) - (1 if -1 in labels_db else 0)
print(f"DBSCAN clusters: {n_clusters_db} (including noise label -1)")

# silhouette (only when more than 1 cluster)
try:
    if n_clusters_db > 1:
        sil_db = silhouette_score(emb_np, labels_db)
    else:
        sil_db = None
except Exception:
    sil_db = None

# HDBSCAN if available
labels_hdb = None
n_clusters_hdb = 0
sil_hdb = None
if HDBSCAN_AVAILABLE:
    clusterer = hdbscan.HDBSCAN(min_cluster_size=20)
    labels_hdb = clusterer.fit_predict(emb_np)
    n_clusters_hdb = len(set(labels_hdb)) - (1 if -1 in labels_hdb else 0)
    if n_clusters_hdb > 1:
        sil_hdb = silhouette_score(emb_np, labels_hdb)
    print(f"HDBSCAN clusters: {n_clusters_hdb} (including noise -1)")

# --------------- Save clustering scatter + silhouette info ---------------
plt.figure(figsize=(8,6))
sns.scatterplot(x=z_tsne[:,0], y=z_tsne[:,1], hue=labels_db, palette='tab10', s=35, alpha=0.9)
plt.title(f"DBSCAN clustering on t-SNE (clusters={n_clusters_db}, silhouette={sil_db})")
plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
plt.legend(title="DBSCAN label", loc='best', bbox_to_anchor=(1.05,1), borderaxespad=0.)
plt.tight_layout()
fn = os.path.join(RESULT_DIR, "dbscan_tsne.png")
plt.savefig(fn, dpi=200); plt.close()
print("Saved", fn)

if labels_hdb is not None:
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=z_tsne[:,0], y=z_tsne[:,1], hue=labels_hdb, palette='tab10', s=35, alpha=0.9)
    plt.title(f"HDBSCAN clustering on t-SNE (clusters={n_clusters_hdb}, silhouette={sil_hdb})")
    plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
    plt.legend(title="HDBSCAN label", loc='best', bbox_to_anchor=(1.05,1), borderaxespad=0.)
    plt.tight_layout()
    fn = os.path.join(RESULT_DIR, "hdbscan_tsne.png")
    plt.savefig(fn, dpi=200); plt.close()
    print("Saved", fn)

# --------------- Save a summary text file ---------------
summary_path = os.path.join(RESULT_DIR, "summary.txt")
with open(summary_path, "w") as f:
    f.write("Causal Lattice Model - run summary\n")
    f.write(f"Date: {time.asctime()}\n")
    f.write(f"Data shape (expanded): {X.shape}\n")
    f.write(f"Repeat factor: {repeat_factor}\n")
    f.write(f"EPOCHS: {EPOCHS}\n")
    f.write(f"LAMBDA1 (causal): {LAMBDA1}\n")
    f.write(f"LAMBDA2 (geo): {LAMBDA2}\n")
    f.write(f"Batch size: {BATCH_SIZE}\n")
    f.write(f"LR: {LR}\n")
    f.write(f"Final accuracy: {acc:.4f}\n")
    f.write(f"DBSCAN clusters: {n_clusters_db}, silhouette: {sil_db}\n")
    if labels_hdb is not None:
        f.write(f"HDBSCAN clusters: {n_clusters_hdb}, silhouette: {sil_hdb}\n")
    else:
        f.write("HDBSCAN not available\n")
    f.write("\nClassification report:\n")
    f.write(report)
print("Saved summary to", summary_path)

print("All outputs saved to:", os.path.abspath(RESULT_DIR))
