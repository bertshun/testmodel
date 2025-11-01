# benchmark.py
import numpy as np
import matplotlib.pyplot as plt
from lifelines.utils import concordance_index

class CausalModelBenchmark:
    def __init__(self, save_dir="results_causal_experiments/metrics"):
        self.save_dir = save_dir

    def evaluate_time_auc_ibs(self, event_time, event_observed, surv_prob_func, model_name):
        grid_t = np.linspace(0, np.max(event_time), 100)
        ibs_vals, auc_vals = [], []
        n = len(event_time)

        for t in grid_t:
            surv_pred = surv_prob_func(t)
            brier = np.mean((event_time > t - surv_pred)**2)
            ibs_vals.append(brier)

            # pseudo-AUC: concordance at time t
            auc_t = concordance_index(event_time, -surv_pred, event_observed)
            auc_vals.append(auc_t)

        ibs = np.mean(ibs_vals)
        auc_mean = np.mean(auc_vals)

        plt.figure(figsize=(6,4))
        plt.plot(grid_t, auc_vals, label=f"{model_name} AUC(t)")
        plt.plot(grid_t, ibs_vals, label=f"{model_name} Brier(t)")
        plt.title(f"Time-dependent AUC and IBS for {model_name}")
        plt.xlabel("Time"); plt.legend()
        plt.savefig(f"{self.save_dir}/{model_name}_time_metrics.png")
        plt.close()

        return {"time_auc_mean": auc_mean, "ibs": ibs}
