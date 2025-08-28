import numpy as np
import pandas as pd
import os
from glob import glob
import pytest
import NeuroCluster

# ---------- Test setup fixture ----------
@pytest.fixture(scope="module")
def cluster_test_and_data():
    # Resolve the data path from the installed NeuroCluster package
    data_path = os.path.join(os.path.dirname(NeuroCluster.__file__), "data")

    # Load sample data
    sample_ieeg_files = glob(os.path.join(data_path, "*.npy"))
    sample_ieeg_dict = {
        os.path.basename(f).split('.')[0]: np.load(f)
        for f in sample_ieeg_files
    }
    sample_behav = pd.read_csv(os.path.join(data_path, 'sample_behavior.csv'))

    freqs = np.logspace(*np.log10([2, 200]), num=30)
    target_var = 'error'
    predictor_data = sample_behav[['outcome', 'error']].copy()
    demo_channel = 'channel_4'
    tfr_data = sample_ieeg_dict[demo_channel]

    # Create cluster_test object
    cluster_test = NeuroCluster.TFR_Cluster_Test(
        tfr_data, predictor_data, target_var, demo_channel, alternative='two-sided'
    )

    return cluster_test, freqs


# ---------- Individual step tests ----------
def test_tfr_regression(cluster_test_and_data):
    cluster_test, _ = cluster_test_and_data
    betas, tstats = cluster_test.tfr_regression()
    assert betas.shape == tstats.shape
    assert np.isfinite(tstats).all()


def test_threshold_tfr_tstat(cluster_test_and_data):
    cluster_test, _ = cluster_test_and_data
    _, tstats = cluster_test.tfr_regression()
    tcrit = cluster_test.threshold_tfr_tstat(tstats)
    assert isinstance(tcrit, float)


def test_max_tfr_cluster(cluster_test_and_data):
    cluster_test, _ = cluster_test_and_data
    _, tstats = cluster_test.tfr_regression()
    max_cluster_data = cluster_test.max_tfr_cluster(tstats, max_cluster_output="all")
    assert isinstance(max_cluster_data, dict)


def test_compute_null_cluster_stats(cluster_test_and_data):
    cluster_test, _ = cluster_test_and_data
    null_cluster_distribution = cluster_test.compute_null_cluster_stats(num_permutations=5)
    assert isinstance(null_cluster_distribution, np.ndarray)
    assert null_cluster_distribution.ndim == 1


def test_cluster_significance_test(cluster_test_and_data):
    cluster_test, _ = cluster_test_and_data
    _, tstats = cluster_test.tfr_regression()
    max_cluster_data = cluster_test.max_tfr_cluster(tstats, max_cluster_output="all")
    null_cluster_distribution = cluster_test.compute_null_cluster_stats(num_permutations=5)
    pval = cluster_test.cluster_significance_test(max_cluster_data, null_cluster_distribution)
    assert 0.0 <= pval <= 1.0


def test_plot_results(cluster_test_and_data):
    cluster_test, freqs = cluster_test_and_data
    betas, tstats = cluster_test.tfr_regression()
    tcrit = cluster_test.threshold_tfr_tstat(tstats)
    max_cluster_data = cluster_test.max_tfr_cluster(tstats, max_cluster_output="all")
    null_cluster_distribution = cluster_test.compute_null_cluster_stats(num_permutations=5)
    cluster_pvalue = cluster_test.cluster_significance_test(max_cluster_data, null_cluster_distribution)

    plots = NeuroCluster.plot_neurocluster_results(
        betas, cluster_test, max_cluster_data,
        null_cluster_distribution, tstats, tcrit,
        cluster_pvalue, freqs
    )

    assert isinstance(plots, tuple) and len(plots) == 6


# ---------- Full workflow integration test ----------
def test_full_real_data_cluster_workflow(cluster_test_and_data, tmp_path):
    cluster_test, freqs = cluster_test_and_data
    betas, tstats = cluster_test.tfr_regression()
    tcrit = cluster_test.threshold_tfr_tstat(tstats)
    max_cluster_data = cluster_test.max_tfr_cluster(tstats, max_cluster_output="all")
    null_cluster_distribution = cluster_test.compute_null_cluster_stats(num_permutations=5)
    cluster_pvalue = cluster_test.cluster_significance_test(max_cluster_data, null_cluster_distribution)

    plots = NeuroCluster.plot_neurocluster_results(
        betas, cluster_test, max_cluster_data,
        null_cluster_distribution, tstats, tcrit,
        cluster_pvalue, freqs
    )

    assert isinstance(plots, tuple) and len(plots) == 6

    # Save plots
    plot_names = [
        "tcrit_plot.png", "beta_plot.png", "tstat_plot.png",
        "cluster_plot.png", "max_cluster_plot.png", "null_distribution_plot.png"
    ]
    for fig, name in zip(plots, plot_names):
        fig.savefig(tmp_path / name, dpi=150, bbox_inches="tight")
