# Causal Experiments Pipeline - Improvement Tasks

## Overview
Current pipeline integrates multiple models (Baseline-RF, CausalForestDML, RepBalNet, GAN-Rep) for causal inference and survival prediction using the UCI bank dataset.  
While functional, several issues reduce scientific rigor and interpretability, especially from a causality-focused, academic perspective.

---

## 1. True Effect Evaluation (Simulation Environment)

**Issue:**  
- Real dataset does not provide ground truth for ATE/ITE.  
- Model comparison is therefore qualitative only.

**Improvement:**  
- Implement a synthetic simulation environment where true treatment effects are known.  
- Evaluate RMSE, bias, coverage of ITE estimates.  
- Optionally, include confounding structures to test robustness.

**Priority:** High

**Implementation:**  
- Add a `simulate_data.py` module generating X, T, Y with known ITE/ATE.  
- Replace UCI dataset for benchmark evaluation, or use in parallel.

---

## 2. Statistical Confidence & Uncertainty Quantification

**Issue:**  
- No confidence intervals, p-values, or bootstrap-based uncertainty in ITE/ATE estimates.  
- Survival metrics (C-index, Time-AUC, IBS) are point estimates only.

**Improvement:**  
- Apply bootstrap to ITE and ATE to compute 95% CI.  
- Add confidence bands to Time-AUC/IBS curves.  
- Include statistical tests (e.g., Wilcoxon, paired t-test) for model comparisons.

**Priority:** High

**Implementation:**  
- Update `eval_classification` and `CausalModelBenchmark` to include CI computation.  
- Extend ITE visualization with shaded CI regions.

---

## 3. GNN Causal Discovery Integration

**Issue:**  
- `torch_geometric` GNN causal discovery is optional and not fully integrated.  
- No demonstration of how learned causal graph affects ITE estimation.

**Improvement:**  
- Implement pipeline: `GNN -> DAG adjacency -> feature selection -> ITE model`.  
- Compare ITE accuracy with vs. without causal graph guidance.

**Priority:** Medium

**Implementation:**  
- Add `gnn_causal_discovery.py` module.  
- Output adjacency matrix and selected features to influence `RepBalNet` or `GAN-Rep`.

---

## 4. Survival Analysis Improvements

**Issue:**  
- CoxPH synthetic data is overly simplistic (linear effects, exponential baseline).  
- Time-dependent metrics lack confidence intervals.

**Improvement:**  
- Introduce non-linear and interaction terms in survival data simulation.  
- Compute CI for C-index, Time-AUC, and IBS.

**Priority:** Medium

**Implementation:**  
- Extend survival simulation in `simulate_data.py`.  
- Use bootstrap or jackknife for confidence intervals in `CausalModelBenchmark`.

---

## 5. Visualization Enhancements

**Issue:**  
- Scatter plots show correlations but no p-values or regression confidence bands.  
- Boxplots/KDEs lack combined display with mean ± CI markers.

**Improvement:**  
- Include 95% confidence bands in scatter regression lines.  
- Combine violin + boxplot + mean marker for ITE distributions.  
- Uniform scale for metrics across models.

**Priority:** Medium

**Implementation:**  
- Update plotting functions in `train.py`.  
- Use `seaborn.lineplot(..., ci=95)` and `sns.violinplot(..., inner="point")`.

---

## 6. Code Scalability & MPS Optimization

**Issue:**  
- Current PyTorch training may not scale to large datasets.  
- Batch processing and device placement are functional but not optimized.

**Improvement:**  
- Use DataLoader for batching.  
- Ensure all tensors moved to device efficiently.  
- Profile memory usage for large N.

**Priority:** Low

**Implementation:**  
- Refactor RepBalNet/GAN training loops to use `torch.utils.data.DataLoader`.  
- Add device checks and `.to(device)` consistently.

---

## Summary

| Issue | Priority |
|-------|---------|
| True effect evaluation via simulation | High |
| Statistical confidence (CI, p-values) | High |
| GNN causal discovery integration | Medium |
| Survival analysis improvements | Medium |
| Visualization enhancements | Medium |
| Code scalability & MPS optimization | Low |

> Implementing the above will strengthen the scientific rigor, reproducibility, and interpretability of the causal experiments pipeline.  
> Target improvement: move from 68/100 → 80+/100 in professor-level assessment.

