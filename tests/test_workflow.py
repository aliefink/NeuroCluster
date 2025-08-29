# tests/test_workflow.py
import numpy as np
import pandas as pd
import os
from glob import glob
from pathlib import Path
import pytest
import NeuroCluster

# ---------- helpers to coerce current API outputs ----------

def _coerce_tcrit_or_mask(tcrit_or_mask, tstats):
    if isinstance(tcrit_or_mask, (int, float, np.floating)):
        return float(tcrit_or_mask), None
    if isinstance(tcrit_or_mask, np.ndarray):
        assert tcrit_or_mask.shape == tstats.shape
        return None, tcrit_or_mask
    if isinstance(tcrit_or_mask, list):
        assert len(tcrit_or_mask) > 0
        first = np.asarray(tcrit_or_mask[0])
        assert first.shape == tstats.shape
        return None, first
    raise TypeError("threshold_tfr_tstat returned unsupported type")

def _pick_best_cluster(clusters_or_dict):
    if isinstance(clusters_or_dict, dict):
        c = clusters_or_dict
    elif isinstance(clusters_or_dict, list):
        assert len(clusters_or_dict) > 0
        c = max(clusters_or_dict, key=lambda d: abs(float(d.get("cluster_stat", 0.0))))
    else:
        raise TypeError("max_tfr_cluster returned unsupported type")
    return {
        "cluster_stat": float(c["cluster_stat"]),
        "freq_idx": (int(c["freq_idx"][0]), int(c["freq_idx"][1])),
        "time_idx": (int(c["time_idx"][0]), int(c["time_idx"][1])),
    }

def _coerce_null(null_like):
    arr = np.asarray(null_like, dtype=float).reshape(-1)
    assert arr.size >= 1
    return arr

def _coerce_p(p_like):
    if isinstance(p_like, (int, float, np.floating)):
        p = float(p_like)
    else:
        flat = np.asarray(p_like, dtype=float).reshape(-1)
        assert flat.size >= 1
        p = float(flat[0])
    assert 0.0 <= p <= 1.0
    return p

# ----------------- fixture -----------------

@pytest.fixture(scope="module")
def cluster_test_and_data():
    data_path = os.path.join(os.path.dirname(NeuroCluster.__file__), "data")
    sample_ieeg_files = glob(os.path.join(data_path, "*.npy"))
    sample_ieeg_dict = {os.path.basename(f).split('.')[0]: np.load(f) for f in sample_ieeg_files}
    sample_behav = pd.read_csv(os.path.join(data_path, 'sample_behavior.csv'))
    freqs = np.logspace(*np.log10([2, 200]), num=30)
    target_var = 'error'
    predictor_data = sample_behav[['outcome', 'error']].copy()
    demo_channel = 'channel_4'
    tfr_data = sample_ieeg_dict[demo_channel]
    ct = NeuroCluster.TFR_Cluster_Test(
        tfr_data, predictor_data, target_var, demo_channel, alternative='two-sided'
    )
    # If your class supports seeding, you can uncomment for deterministic CI:
    # if hasattr(ct, "set_random_state"): ct.set_random_state(123)
    return ct, freqs

# ----------------- unit tests -----------------

@pytest.mark.unit
def test_tfr_regression(cluster_test_and_data):
    ct, _ = cluster_test_and_data
    betas, tstats = ct.tfr_regression()
    assert isinstance(betas, np.ndarray) and isinstance(tstats, np.ndarray)
    assert betas.shape == tstats.shape
    assert np.isfinite(tstats).all()

@pytest.mark.unit
def test_threshold_tfr_tstat(cluster_test_and_data):
    ct, _ = cluster_test_and_data
    _, tstats = ct.tfr_regression()
    tcrit_or_mask = ct.threshold_tfr_tstat(tstats)
    tcrit, mask = _coerce_tcrit_or_mask(tcrit_or_mask, tstats)
    assert (tcrit is not None) or (mask is not None)

@pytest.mark.unit
def test_max_tfr_cluster(cluster_test_and_data):
    ct, _ = cluster_test_and_data
    _, tstats = ct.tfr_regression()
    raw = ct.max_tfr_cluster(tstats, max_cluster_output="all")
    best = _pick_best_cluster(raw)
    assert isinstance(best, dict)
    assert {"cluster_stat","freq_idx","time_idx"} <= set(best.keys())

@pytest.mark.unit
def test_compute_null_cluster_stats(cluster_test_and_data):
    ct, _ = cluster_test_and_data
    raw = ct.compute_null_cluster_stats(num_permutations=5)
    null_dist = _coerce_null(raw)
    assert isinstance(null_dist, np.ndarray) and null_dist.ndim == 1
    assert null_dist.size >= 5  # allow multiples (e.g., per-tail/per-predictor)

@pytest.mark.unit
def test_cluster_significance_test(cluster_test_and_data):
    ct, _ = cluster_test_and_data
    _, tstats = ct.tfr_regression()
    best = _pick_best_cluster(ct.max_tfr_cluster(tstats, max_cluster_output="all"))
    null_dist = _coerce_null(ct.compute_null_cluster_stats(num_permutations=5))
    p = _coerce_p(ct.cluster_significance_test(best, null_dist))
    assert 0.0 <= p <= 1.0

@pytest.mark.unit
def test_plot_results(cluster_test_and_data):
    ct, freqs = cluster_test_and_data
    betas, tstats = ct.tfr_regression()
    tcrit_or_mask = ct.threshold_tfr_tstat(tstats)
    tcrit, mask = _coerce_tcrit_or_mask(tcrit_or_mask, tstats)
    best = _pick_best_cluster(ct.max_tfr_cluster(tstats, max_cluster_output="all"))
    null_dist = _coerce_null(ct.compute_null_cluster_stats(num_permutations=5))
    p = _coerce_p(ct.cluster_significance_test(best, null_dist))
    tcrit_for_plot = tcrit if tcrit is not None else np.nan

    figs = NeuroCluster.plot_neurocluster_results(
        betas, ct, best, null_dist, tstats, tcrit_for_plot, p, freqs
    )
    assert isinstance(figs, tuple) and len(figs) == 6

# ----------------- integration (writes files) -----------------

@pytest.mark.integration
def test_full_real_data_cluster_workflow(cluster_test_and_data, tmp_path: Path):
    ct, freqs = cluster_test_and_data
    betas, tstats = ct.tfr_regression()
    tcrit_or_mask = ct.threshold_tfr_tstat(tstats)
    tcrit, mask = _coerce_tcrit_or_mask(tcrit_or_mask, tstats)
    best = _pick_best_cluster(ct.max_tfr_cluster(tstats, max_cluster_output="all"))
    null_dist = _coerce_null(ct.compute_null_cluster_stats(num_permutations=5))
    p = _coerce_p(ct.cluster_significance_test(best, null_dist))
    tcrit_for_plot = tcrit if tcrit is not None else np.nan

    figs = NeuroCluster.plot_neurocluster_results(
        betas, ct, best, null_dist, tstats, tcrit_for_plot, p, freqs
    )
    assert isinstance(figs, tuple) and len(figs) == 6

    names = ["tcrit_plot.png","beta_plot.png","tstat_plot.png",
             "cluster_plot.png","max_cluster_plot.png","null_distribution_plot.png"]
    for fig, name in zip(figs, names):
        fig.savefig(tmp_path / name, dpi=150, bbox_inches="tight")
    for name in names:
        assert (tmp_path / name).exists()

