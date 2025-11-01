# benchmark.py
import os
import numpy as np
import matplotlib.pyplot as plt
from sksurv.metrics import (
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.util import Surv
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

class CausalModelBenchmark:
    """
    医療系指標 (時間依存AUC, IBS) + 通常分類評価
    """

    def __init__(self, save_dir="results_causal_experiments/metrics"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------
    # 通常分類タスクの評価
    # ------------------------------------------------
    def evaluate_classification(self, y_true, y_prob, model_name):
        y_pred = (y_prob >= 0.5).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = np.nan

        # グラフ保存
        plt.figure(figsize=(4,4))
        plt.bar(["F1", "Precision", "AUC"], [f1, precision, auc], color=["#4C72B0","#55A868","#C44E52"])
        plt.title(f"{model_name} — Classification Metrics")
        plt.ylim(0,1)
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{model_name}_basic_metrics.png")
        plt.close()

        return {"precision":precision, "recall":recall, "f1":f1, "auc":auc}

    # ------------------------------------------------
    # 時間依存評価 (AUC(t) と IBS)
    # ------------------------------------------------
    def evaluate_time_auc_ibs(self, event_time, event_observed, surv_prob_func, model_name):
        """
        event_time: array-like, 生存時間
        event_observed: array-like, イベント発生 (1/0)
        surv_prob_func: callable, tを渡すと survival probability (Nサンプルのベクトル) を返す関数
        """
        times = np.linspace(0.1, np.max(event_time)*0.9, 40)
        y = Surv.from_arrays(event_observed.astype(bool), event_time)

        # survival probability matrix: shape (n_samples, len(times))
        surv_probs = np.row_stack([surv_prob_func(t) for t in times]).T
        aucs, mean_auc = cumulative_dynamic_auc(y, y, surv_probs, times)
        ibs = integrated_brier_score(y, y, surv_probs, times)

        # --- plot AUC(t)
        plt.figure(figsize=(6,4))
        plt.plot(times, aucs, color='C0')
        plt.title(f"Time-dependent AUC — {model_name}\nMean AUC={mean_auc:.3f}")
        plt.xlabel("Time")
        plt.ylabel("AUC(t)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{model_name}_time_auc.png")
        plt.close()

        # --- plot IBS(t)
        plt.figure(figsize=(6,4))
        plt.plot(times, np.mean((1 - surv_probs) ** 2, axis=0), label=f"IBS={ibs:.3f}")
        plt.title(f"Integrated Brier Score — {model_name}")
        plt.xlabel("Time")
        plt.ylabel("Brier Score")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{model_name}_ibs.png")
        plt.close()

        return {"time_auc_mean":mean_auc, "ibs":ibs}


# ------------------------------------------------
# 数式の理論メモ（説明用コメント）
# ------------------------------------------------
"""
Time-dependent AUC:
AUC(t) = P( Ŝ(t|X_i) < Ŝ(t|X_j) | T_i <= t < T_j )

Brier Score at time t:
BS(t) = (1/N) Σ_i [ Ŝ(t|X_i) - I(T_i > t) ]²

Integrated Brier Score (IBS):
IBS = (1/τ) ∫₀^τ BS(t) dt

これらは時間発展的な予測の信頼性を評価する。
"""
