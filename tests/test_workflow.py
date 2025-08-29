# tests/test_workflow.py
import numpy as np
import pandas as pd
import os
from glob import glob
from pathlib import Path
import pytest
import NeuroCluster

# ----------------- helpers to coerce current API outputs -----------------

def _coerce_tcrit_or_mask(tcrit_or_mask, tstats):
    """
    Accepts either:
      - float (critical |t|), or
      - ndarray mask (F, T), or
      - list of ndarray masks
    Returns a tuple: (tcrit: float|None, mask: np.ndarray|None)
    """
    if isinstance(tcrit_or_mask, (int, float, np.floating)):
        return float(tcrit_or_mask), None
    if isinstance(tcrit_or_mask, np.ndarray):
        assert tcrit_or_mask.shape == tstats.shape
        return None, tcrit_or_mask
    if isinstance(tcrit_or_mask, list):
        # assume a list of masks; take the first and validate
        assert len(tcrit_or_mask) > 0
        first = np.asarray(tcrit_or_mask[0])
        assert first.shape == tstats.shape
        return None, first
    raise TypeError("threshold_tfr_tstat returned unsupported type")

def _pick_best_cluster(clusters_or_dict):
    """
    Accepts either:
      - dict describing a single cluster, or
      - list of dicts. Returns dict for the max |cluster_stat|.
    Ensures native Python types.
    """
    if isinstance(clusters_or_dict, dict):
        c = clusters_or_dict
    elif isinstance(clusters_or_dict, list):
        assert len(clusters_or_dict) > 0
        best = max(
            clusters_or_dict,
            key=lambda c: abs(float(c.get("cluster_stat", 0.0)))
        )
        c = best
    else:
        raise TypeError("max_tfr_cluster returned unsupported type")

    return {
        "cluster_stat": float(c["cluster_stat"]),
        "freq_idx": (int(c["freq_idx"][0]), int(c["freq_idx"][1])),
        "time_idx": (int(c["time_idx"][0]), int(c["time_idx"][1])),
    }

def _coerce_null(null_like, expected_n=None):
    """
    Accepts list / nested list / ndarray and returns a 1-D ndarray of floats.
    """
    arr = np.asarray(null_like, dtype=float).reshape(-1)
    if expected_n is not None:
        assert arr.size == expected_n
    return arr

def _coerce_p(p_like):
    """
    Accepts float or list-like; returns a single float in [0,1].
    If list-like, take the first element.
    """
    if isinstance(p_like, (int, float, np.floating)):
        p = float(p_like)
    elif isinstance(p_like, (list, tuple, np.ndarray)):
        flat = np.asarray(p_like, dtype=float).reshape(-1)
        assert flat.size >= 1
        p = float(flat[0])
    else:
        raise TypeError("cluster_significance_test returned unsupported type")
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
    # If class supports seeding RNGs, you can uncomment for deterministic CI:
    # if hasattr(ct, "set_random_state"):
    #     ct.set_random_state(123)
    return ct, freqs

# ----------------- unit tests (fast) -----------------

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
    # at least one of these must exist and be valid
    assert (tcrit is not None and isinstance(tcrit, float)) or (mask is not None and mask.shape == tstats.shape)

@pytest.mark.unit
def test_max_tfr_cluster(cluster_test_and_data):
    ct, _ = cluster_test_and_data
    _, tstats = ct.tfr_regression()
    raw = ct.max_tfr_cluster(tstats, max_cluster_output="all")
    max_cluster = _pick_best_cluster(raw)
    assert isinstance(max_cluster, dict)
    assert {"cluster_stat","freq_idx","time_idx"} <= set(max_cluster.keys())

@pytest.mark.unit
def test_compute_null_cluster_stats(cluster_test_and_data):
    ct, _ = cluster_test_and_data
    raw = ct.compute_null_cluster_stats(num_permutations=5)
    null_dist = _coerce_null(raw, expected_n=5)
    assert isinstance(null_dist, np.ndarray) and null_dist.ndim == 1

@pytest.mark.unit
def test_cluster_significance_test(cluster_test_and_data):
    ct, _ = cluster_test_and_data
    _, tstats = ct.tfr_regression()
    raw_cluster = ct.max_tfr_cluster(tstats, max_cluster_output="all")
    max_cluster = _pick_best_cluster(raw_cluster)
    raw_null = ct.compute_null_cluster_stats(num_permutations=5)
    null_dist = _coerce_null(raw_null, expected_n=5)
    raw_p = ct.cluster_significance_test(max_cluster, null_dist)
    p = _coerce_p(raw_p)
    assert 0.0 <= p <= 1.0

@pytest.mark.unit
def test_plot_results(cluster_test_and_data):
    ct, freqs = cluster_test_and_data
    betas, tstats = ct.tfr_regression()
    tcrit_or_mask = ct.threshold_tfr_tstat(tstats)
    tcrit, mask = _coerce_tcrit_or_mask(tcrit_or_mask, tstats)
    raw_cluster = ct.max_tfr_cluster(tstats, max_cluster_output="all")
    max_cluster = _pick_best_cluster(raw_cluster)
    null_dist = _coerce_null(ct.compute_null_cluster_stats(num_permutations=5), expected_n=5)
    p = _coerce_p(ct.cluster_significance_test(max_cluster, null_dist))

    # plot_neurocluster_results expects a scalar tcrit; if we only have a mask,
    # pass a placeholder scalar (e.g., np.nan) that your plotting code can handle,
    # OR modify your plotting utility to accept either scalar/mask.
    tcrit_for_plot = tcrit if tcrit is not None else np.nan

    figs = NeuroCluster.plot_neurocluster_results(
        betas, ct, max_cluster, null_dist, tstats, tcrit_for_plot, p, freqs
    )
    assert isinstance(figs, tuple) and len(figs) == 6

# ----------------- integration test (writes files) -----------------

@pytest.mark.integration
def test_full_real_data_cluster_workflow(cluster_test_and_data, tmp_path: Path):
    ct, freqs = cluster_test_and_data
    betas, tstats = ct.tfr_regression()
    tcrit_or_mask = ct.threshold_tfr_tstat(tstats)
    tcrit, mask = _coerce_tcrit_or_mask(tcrit_or_mask, tstats)
    raw_cluster = ct.max_tfr_cluster(tstats, max_cluster_output="all")
    max_cluster = _pick_best_cluster(raw_cluster)
    null_dist = _coerce_null(ct.compute_null_cluster_stats(num_permutations=5), expected_n=5)
    p = _coerce_p(ct.cluster_significance_test(max_cluster, null_dist))

    tcrit_for_plot = tcrit if tcrit is not None else np.nan
    figs = NeuroCluster.plot_neurocluster_results(
        betas, ct, max_cluster, null_dist, tstats, tcrit_for_plot, p, freqs
    )
    assert isinstance(figs, tuple) and len(figs) == 6

    names = ["tcrit_plot.png","beta_plot.png","tstat_plot.png",
             "cluster_plot.png","max_cluster_plot.png","null_distribution_plot.png"]
    for fig, name in zip(figs, names):
        fig.savefig(tmp_path / name, dpi=150, bbox_inches="tight")
    for name in names:
        assert (tmp_path / name).exists()
