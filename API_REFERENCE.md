
# NeuroCluster API Reference

This document provides a high-level overview of the NeuroCluster API for external users. It includes class descriptions, method details, input/output expectations, and usage examples.

## Overview

NeuroCluster is a Python package for performing non-parametric cluster-based permutation testing on time-frequency representations (TFR) of neural data, such as EEG, MEG, or intracranial recordings.

---

## `TFR_Cluster_Test` Class

### Description
A class that performs regression analysis at each time-frequency point of the input TFR data and identifies clusters of significant effects using non-parametric permutation testing.

### Initialization
```python
TFR_Cluster_Test(tfr_data, predictor_data, target_variable, demo_channel, alternative='two-sided')
```

#### Parameters
- `tfr_data` (np.ndarray): 4D array of shape `(n_trials, n_channels, n_freqs, n_times)`
- `predictor_data` (pd.DataFrame): Behavioral regressors of shape `(n_trials, n_regressors)`
- `target_variable` (str): Column name in `predictor_data` to be tested
- `demo_channel` (str): Channel to visualize during plotting/debugging
- `alternative` (str): Statistical test type: `'two-sided'`, `'greater'`, or `'less'`

### Example
```python
cluster_test = TFR_Cluster_Test(tfr_data, predictor_data, 'accuracy', 'Cz')
```

---

## Methods

### `run_regression()`
Performs linear regression at each time-frequency point.

```python
betas, tvals = cluster_test.run_regression()
```

**Returns:**
- `betas` (np.ndarray): Regression coefficients, shape `(n_channels, n_freqs, n_times)`
- `tvals` (np.ndarray): T-values associated with the regressor of interest

---

### `find_clusters(tvals)`
Identifies clusters in the true data based on a significance threshold.

```python
clusters = cluster_test.find_clusters(tvals)
```

**Returns:**
- `clusters` (list): List of clusters, where each cluster is a list of (freq_idx, time_idx) tuples

---

### `compute_null_distribution()`
Generates a null distribution of cluster-level statistics using permutations.

```python
null_distribution = cluster_test.compute_null_distribution(n_permutations=1000)
```

**Returns:**
- `null_distribution` (np.ndarray): Max cluster stats from permuted datasets

---

### `calculate_p_values(true_clusters, null_distribution)`
Calculates p-values for the observed clusters.

```python
pvals = cluster_test.calculate_p_values(true_clusters, null_distribution)
```

**Returns:**
- `pvals` (list): P-values corresponding to each cluster

---

### `plot_clusters(clusters, tvals)`
Visualizes significant clusters on the time-frequency map.

```python
cluster_test.plot_clusters(clusters, tvals)
```

---

## Example Pipeline
```python
cluster_test = TFR_Cluster_Test(tfr_data, predictor_data, 'accuracy', 'Cz')
tvals = cluster_test.run_regression()[1]
true_clusters = cluster_test.find_clusters(tvals)
null_dist = cluster_test.compute_null_distribution(n_permutations=1000)
pvals = cluster_test.calculate_p_values(true_clusters, null_dist)
cluster_test.plot_clusters(true_clusters, tvals)
```

---

## Notes
- Ensure that `tfr_data` and `predictor_data` are aligned across trials
- Recommended to z-score continuous regressors beforehand
- Consider adjusting the cluster-forming threshold for your statistical needs
- Consider adjusting the number of permutations based on experimental considerations and computational resources. 

For a full demo, see the Jupyter notebook: `notebooks/NeuroCluster_template.ipynb`
