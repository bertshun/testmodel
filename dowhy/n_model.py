# n_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# -----------------------
# RepBalNet
# -----------------------
class RepBalanceNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.h0 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.h1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, t):
        phi = self.encoder(x)
        y_hat = self.h1(phi) * t + self.h0(phi) * (1 - t)
        return y_hat, phi

# -----------------------
# GAN-based representation
# -----------------------
class GANRepNet(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.dec = nn.Sequential(
            nn.Linear(latent_dim + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, t):
        z = self.enc(x)
        zt = torch.cat([z, t], dim=1)
        return self.dec(zt)

# -----------------------
# MMD loss for RepBalNet
# -----------------------
def mmd_loss_batch(phi_t, phi_c, batch_size=128, gamma=1.0):
    if len(phi_t) == 0 or len(phi_c) == 0:
        return torch.tensor(0.0, device=phi_t.device if len(phi_t) > 0 else phi_c.device)

    bt = phi_t[torch.randperm(len(phi_t))[:min(batch_size, len(phi_t))]]
    bc = phi_c[torch.randperm(len(phi_c))[:min(batch_size, len(phi_c))]]

    def gaussian_kernel(x, y):
        x_norm = (x**2).sum(1).view(-1,1)
        y_norm = (y**2).sum(1).view(1,-1)
        dist = x_norm + y_norm - 2*torch.mm(x, y.t())
        return torch.exp(-gamma * dist)

    K_tt = gaussian_kernel(bt, bt)
    K_cc = gaussian_kernel(bc, bc)
    K_tc = gaussian_kernel(bt, bc)
    return K_tt.mean() + K_cc.mean() - 2*K_tc.mean()

# -----------------------
# Utility: ITE estimation from predictive classifier
# -----------------------
def estimate_ite_from_predictive(model, X, t0=0, t1=1):
    """
    Input:
        model: sklearn classifier with predict_proba
        X: feature matrix (numpy)
        t0, t1: treatment values
    Returns:
        ite: individual treatment effect (numpy array)
    """
    import numpy as np
    p0 = model.predict_proba(np.hstack([X, np.full((len(X),1), t0)]))[:,1]
    p1 = model.predict_proba(np.hstack([X, np.full((len(X),1), t1)]))[:,1]
    return p1 - p0
