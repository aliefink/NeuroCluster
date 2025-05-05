import numpy as np
import pandas as pd
import os
from glob import glob
import NeuroCluster

def test_real_data_cluster_workflow():
    # Dynamically resolve the project root
    project_root = os.path.dirname(os.path.dirname(__file__))
    sample_data_dir = os.path.join(project_root, 'data')

    # Load real data
    sample_ieeg_files = glob(f'{sample_data_dir}/*.npy')
    sample_ieeg_dict = {
        os.path.basename(f).split('.')[0]: np.load(f)
        for f in sample_ieeg_files
    }

    sample_behav = pd.read_csv(os.path.join(sample_data_dir, 'sample_behavior.csv'))
    freqs = np.logspace(*np.log10([2, 200]), num=30)

    target_var = 'error'
    predictor_data = sample_behav[['outcome', 'error']].copy()
    demo_channel = 'channel_4'
    tfr_data = sample_ieeg_dict[demo_channel]

    # Full NeuroCluster workflow
    cluster_test = NeuroCluster.TFR_Cluster_Test(
        tfr_data, predictor_data, target_var, demo_channel, alternative='two-sided'
    )

    betas, tstats = cluster_test.tfr_regression()
    tstat_threshold = cluster_test.threshold_tfr_tstat(tstats)
    max_cluster_data = cluster_test.max_tfr_cluster(tstats, max_cluster_output='all')
    null_cluster_distribution = cluster_test.compute_null_cluster_stats(num_permutations=10)
    cluster_pvalue = cluster_test.cluster_significance_test(max_cluster_data, null_cluster_distribution)

    plots = NeuroCluster.plot_neurocluster_results(
        betas, cluster_test, max_cluster_data,
        null_cluster_distribution, tstats, tstat_threshold,
        cluster_pvalue, freqs
    )

    # Assertions
    assert isinstance(plots, tuple) and len(plots) == 6

    output_dir = os.path.join(os.path.dirname(__file__), 'test_outputs')
    os.makedirs(output_dir, exist_ok=True)

    plot_names = [
        "tcrit_plot.png", "beta_plot.png", "tstat_plot.png",
        "cluster_plot.png", "max_cluster_plot.png", "null_distribution_plot.png"
    ]

    for fig, name in zip(plots, plot_names):
        fig.savefig(os.path.join(output_dir, name), dpi=150, bbox_inches='tight')