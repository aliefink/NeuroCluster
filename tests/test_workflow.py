import numpy as np
import pandas as pd
import os
from glob import glob
import pytest
import NeuroCluster

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
    cluster_test = NeuroCluster.TFR_Cluster_Test(
        tfr_data, predictor_data, target_var, demo_channel, alternative='two-sided'
    )
    return cluster_test, freqs

# -------- unit tests (fast) --------
@pytest.mark.unit
def test_tfr_regression(cluster_test_and_data):
    ct, _ = cluster_test_and_data
    betas, tstats = ct.tfr_regression()
    assert betas.shape == tstats.shape
    assert np.isfinite(tstats).all()

@pytest.mark.unit
def test_threshold_tfr_tstat(cluster_test_and_data):
    ct, _ = cluster_test_and_data
    _, tstats = ct.tfr_regression()
    tcrit = ct.threshold_tfr_tstat(tstats)
    assert isinstance(tcrit, float)

@pytest.mark.unit
def test_max_tfr_cluster(cluster_test_and_data):
    ct, _ = cluster_test_and_data
    _, tstats = ct.tfr_regression()
    max_cluster = ct.max_tfr_cluster(tstats, max_cluster_output="all")
    assert isinstance(max_cluster, dict)

@pytest.mark.unit
def test_compute_null_cluster_stats(cluster_test_and_data):
    ct, _ = cluster_test_and_data
    null_dist = ct.compute_null_cluster_stats(num_permutations=5)
    assert isinstance(null_dist, np.ndarray) and null_dist.ndim == 1

@pytest.mark.unit
def test_cluster_significance_test(cluster_test_and_data):
    ct, _ = cluster_test_and_data
    _, tstats = ct.tfr_regression()
    max_cluster = ct.max_tfr_cluster(tstats, max_cluster_output="all")
    null_dist = ct.compute_null_cluster_stats(num_permutations=5)
    p = ct.cluster_significance_test(max_cluster, null_dist)
    assert 0.0 <= p <= 1.0

@pytest.mark.unit
def test_plot_results(cluster_test_and_data):
    ct, freqs = cluster_test_and_data
    betas, tstats = ct.tfr_regression()
    tcrit = ct.threshold_tfr_tstat(tstats)
    max_cluster = ct.max_tfr_cluster(tstats, max_cluster_output="all")
    null_dist = ct.compute_null_cluster_stats(num_permutations=5)
    p = ct.cluster_significance_test(max_cluster, null_dist)
    figs = NeuroCluster.plot_neurocluster_results(
        betas, ct, max_cluster, null_dist, tstats, tcrit, p, freqs
    )
    assert isinstance(figs, tuple) and len(figs) == 6

# -------- integration test (writes files) --------
@pytest.mark.integration
def test_full_real_data_cluster_workflow(cluster_test_and_data):
    ct, freqs = cluster_test_and_data
    betas, tstats = ct.tfr_regression()
    tcrit = ct.threshold_tfr_tstat(tstats)
    max_cluster = ct.max_tfr_cluster(tstats, max_cluster_output="all")
    null_dist = ct.compute_null_cluster_stats(num_permutations=5)
    p = ct.cluster_significance_test(max_cluster, null_dist)

    figs = NeuroCluster.plot_neurocluster_results(
        betas, ct, max_cluster, null_dist, tstats, tcrit, p, freqs
    )
    assert isinstance(figs, tuple) and len(figs) == 6

    # Save to tests/test_outputs (matches your manual step)
    out_dir = os.path.join(os.path.dirname(__file__), "test_outputs")
    os.makedirs(out_dir, exist_ok=True)
    names = ["tcrit_plot.png","beta_plot.png","tstat_plot.png",
             "cluster_plot.png","max_cluster_plot.png","null_distribution_plot.png"]
    for fig, name in zip(figs, names):
        fig.savefig(os.path.join(out_dir, name), dpi=150, bbox_inches="tight")

