# train_gcn.py
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt

# --- PyG imports ---
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected

# -------------------------
# Reproducibility
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# -------------------------
# Config
# -------------------------
DATA_ROOT = "data"
DATASET_NAME = "Cora"   # Cora を使ったノード分類サンプル
DEVICE = torch.device("mps" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 200
LR = 0.01
WEIGHT_DECAY = 5e-4
HIDDEN_DIM = 64
PATIENCE = 20  # early stopping patience

# -------------------------
# Model
# -------------------------
class GCNNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        # edge_index expected as [2, E]
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x  # logits

# -------------------------
# Load dataset (Cora)
# -------------------------
os.makedirs(DATA_ROOT, exist_ok=True)
dataset = Planetoid(root=DATA_ROOT, name=DATASET_NAME)
data = dataset[0].to(DEVICE)
# ensure undirected
data.edge_index = to_undirected(data.edge_index)

# -------------------------
# Prepare model, optimizer
# -------------------------
model = GCNNet(dataset.num_node_features, HIDDEN_DIM, dataset.num_classes).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# -------------------------
# Training / Eval helpers
# -------------------------
def train_one_epoch():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)  # [num_nodes, num_classes]
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(mask):
    model.eval()
    out = model(data.x, data.edge_index)
    logits = out[mask].cpu().numpy()
    preds = logits.argmax(axis=1)
    labels = data.y[mask].cpu().numpy()
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    cm = confusion_matrix(labels, preds)
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "cm": cm, "preds": preds, "labels": labels}

# -------------------------
# Training loop with early stopping
# -------------------------
best_val = 0.0
best_state = None
patience_cnt = 0
history = {"train_loss": [], "val_acc": [], "val_f1": []}

for epoch in range(1, NUM_EPOCHS+1):
    loss = train_one_epoch()
    train_loss = loss
    val_metrics = evaluate(data.val_mask)
    val_acc = val_metrics["acc"]
    val_f1 = val_metrics["f1"]

    history["train_loss"].append(train_loss)
    history["val_acc"].append(val_acc)
    history["val_f1"].append(val_f1)

    # Early stopping on validation F1 (or acc)
    if val_f1 > best_val:
        best_val = val_f1
        best_state = model.state_dict()
        patience_cnt = 0
    else:
        patience_cnt += 1

    print(f"Epoch {epoch:03d} | Loss {train_loss:.4f} | ValAcc {val_acc:.4f} | ValF1 {val_f1:.4f} | Pat {patience_cnt}")

    if patience_cnt >= PATIENCE:
        print("Early stopping triggered.")
        break

# restore best
if best_state is not None:
    model.load_state_dict(best_state)

# -------------------------
# Final test evaluation
# -------------------------
test_metrics = evaluate(data.test_mask)
print("\n=== Final Test Metrics ===")
print(f"Test Accuracy: {test_metrics['acc']:.4f}")
print(f"Test Precision (macro): {test_metrics['prec']:.4f}")
print(f"Test Recall (macro): {test_metrics['rec']:.4f}")
print(f"Test F1 (macro): {test_metrics['f1']:.4f}")
print("Confusion Matrix:\n", test_metrics["cm"])

# -------------------------
# Save model & history
# -------------------------
torch.save(model.state_dict(), "gcn_cora_best.pt")
import json
with open("train_history.json", "w") as fh:
    json.dump(history, fh)

# -------------------------
# Plot loss / val_f1
# -------------------------
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(history["train_loss"], label="train_loss")
plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
plt.subplot(1,2,2)
plt.plot(history["val_f1"], label="val_f1")
plt.xlabel("epoch"); plt.ylabel("val_f1"); plt.legend()
plt.tight_layout()
plt.savefig("training_plots.png")
print("Saved training_plots.png")
