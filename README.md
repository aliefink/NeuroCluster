# NeuroCluster

[![NeuroCluster Continuous Integration](https://github.com/alliefink/NeuroCluster/actions/workflows/tests.yml/badge.svg)](https://github.com/alliefink/NeuroCluster/actions/workflows/tests.yml)


A Python pipeline for non-parametric cluster-based permutation testing for electrophysiological signals related to computational cognitive model variables.

**Statement of Need**: Time-varying, continuous latent variables from computational cognitive models enable model-based neural analyses that identify how cognitive processes are encoded in the brain, informing the development of neural-inspired algorithms. However, current statistical methods for linking these variables to electrophysiological signals are limited, which may hinder the understanding of neurocomputational mechanisms. To address this methodological limitation, we propose a multivariate linear regression that leverages non-parametric cluster-based permutation testing strategy.

This repository contains code and example data for performing NeuroCluster. 

**Contributions and Code of Conduct**: We welcome contributions to this project and are grateful for your interest in helping improve it! Whether you're fixing bugs, adding features, improving documentation, or sharing ideas, your input is valued. Before getting started, please take a moment to review our [Contributing Guidelines](https://github.com/aliefink/NeuroCluster/blob/main/CONTRIBUTING.md)  and [Code of Conduct](https://github.com/aliefink/NeuroCluster/blob/main/CODE_OF_CONDUCT.md), both of which are included in the repository. These documents outline the process for submitting contributions and the standards we uphold to ensure a respectful and inclusive community.

## Full API Reference

This repository includes two levels of documentation:

- **Setup instructions and basic overview** are available throughout this `README.md`, including installation and walk-through of template notebook for basic use-case.
-  **API documentation**, including class descriptions, method parameters, expected inputs/outputs, and example pipelines, can be found in:[API_REFERENCE.md](https://github.com/aliefink/NeuroCluster/blob/main/API_REFERENCE.md)


## Installation

```
# Create and activate a clean virtual environment (recommended)
conda create -n neurocluster_env python=3.10 -y
conda activate neurocluster_env

# Install NeuroCluster from GitHub
pip install git+https://github.com/aliefink/NeuroCluster.git

```

## Updating

```
pip install --upgrade git+https://github.com/aliefink/NeuroCluster.git

```

## Run Automatic Testing

We provide a test script which validates the full statistical analysis workflow implemented in the NeuroCluster toolbox. It runs a regression-based cluster-based permutation test on synthetic data using pytest, and produces a visual output of the detected significant clusters. This test ensures that the core functionality of the toolbox—including regression modeling, thresholding, cluster formation, null distribution generation, p-value computation, and plotting works as expected.

```
# Activate the virtual environment where NeuroCluster has been installed
conda activate neurocluster_env

# Navigate to the project directory
cd NeuroCluster

# Install pytest if not already in your environment
pip install pytest

# Run Tests

# Run only individual unit tests (fast, isolated functions – 6 total)
# These step through each aspect of the pipeline:

# 1. TFR regression (compute betas and t-stats)
pytest -k test_tfr_regression -m unit

# 2. Threshold t-stats (determine critical value)
pytest -k test_threshold_tfr_tstat -m unit

# 3. Extract max cluster from t-stats
pytest -k test_max_tfr_cluster -m unit

# 4. Compute null distribution via permutations
pytest -k test_compute_null_cluster_stats -m unit

# 5. Perform cluster significance test
pytest -k test_cluster_significance_test -m unit

# 6. Plot results (generate figures from analysis)
pytest -k test_plot_results -m unit

# 1. Run all individual unit tests at once 
pytest -m unit

# 2. Run only the integration test (end-to-end workflow, writes output plots)
pytest -m integration

# 3. Run all tests (unit + integration)
pytest

# View test outputs (for integration tests)
open tests/test_outputs

```

## Quick Start 

```/notebooks/NeuroCluster_template.ipynb```: This Jupyter notebook uses example data stored within ```/data/``` directory to perform Neurocluster for one example electrode. 

Below is a schematic of the NeuroCluster workflow and quick summary. A more detailed step-by-step summary follows. By following these steps, researchers can identify significant time-frequency clusters and assess their statistical validity using non-parametric methods! 

![neurocluster workflow](https://github.com/christinamaher/NeuroCluster/blob/main/workflow/workflow.png)

**Summary of Workflow:** 

1. **Initialize Analysis**: Create the ```TFR_Cluster_Test``` object.

2. **Perform Regression**: Extract beta coefficients and t-statistics.

3. **Identify Clusters**: Find the largest significant clusters in the TFR data.

4. **Generate Null Distribution**: Permute data to create a null cluster distribution.

5. **Compute Significance**: Compare observed clusters against the null distribution.

6. **Visualize and Save Results**: Plot and save all key outputs for interpretation. 

# NeuroCluster Single Electrode Workflow - *detailed overview* 

## **Step 1: Create TFR_Cluster_Test Object**

```cluster_test = NeuroCluster.TFR_Cluster_Test(tfr_data, predictor_data, target_var, demo_channel, alternative='two-sided')```

**Explanation**:

**Purpose**: Initialize a ```TFR_Cluster_Test``` object.

**Inputs**:

```tfr_data```: Time-frequency representation data.

```predictor_data```: Data for the independent variable(s) (e.g., behavioral regressors).

```target_var```: Dependent variable of interest.This variable will be permuted to compute non-parametric p-value.

```demo_channel```: The channel to analyze (e.g., electrode name).

```alternative```: Specifies the type of test ('two-sided', 'greater', or 'less').

**Output**: A ```TFR_Cluster_Test``` object ready for subsequent analysis. 

## **Step 2: Run TFR Regression and Threshold t-statistics**

```betas, tstats = cluster_test.tfr_regression()```

```tstat_threshold = cluster_test.threshold_tfr_tstat(tstats)```

**Explanation:**

**```tfr_regression()```**: 

* **Purpose**: Performs a time-frequency regression analysis to compute:

* ```betas```: Beta coefficients for the predictor of interest.

* ```tstats```: t-statistics for each pixel in the time-frequency representation (TFR).

**```threshold_tfr_tstat()```**: 

* **Purpose**: Determines which t-statistics are significant based on a critical t-value.

* **Output**: A thresholded t-statistic matrix where non-significant values are removed.

## **Step 3: Identify Largest Clusters**

```max_cluster_data = cluster_test.max_tfr_cluster(tstats, max_cluster_output='all')```

**Explanation**:

* **Purpose**: Identifies the largest contiguous clusters of significant t-statistics.

* **Inputs**: 

* ```tstats```:  t-statistics matrix.

* ```max_cluster_output```: Specifies the type of output ('all' for full cluster details).

* **Output**: 

* ```max_cluster_data```: Contains the maximum cluster statistics and their corresponding time-frequency indices.

## **Step 4: Compute Null Distribution**

```null_cluster_distribution = cluster_test.compute_null_cluster_stats(num_permutations=100)```

**Explanation**:

* **Purpose**: Creates a null distribution of maximum cluster statistics by permuting the data.

* **Inputs**:

* ```num_permutations```: Number of permutations to generate the null distribution.

* **Outputs**:

* ```null_cluster_distribution```: A distribution of maximum cluster statistics under the null hypothesis.

## **Step 5: Compute non-parametric p-value**

**Explanation**

* **Purpose**: Calculates the statistical significance of the observed clusters.

* **Inputs**:

* ```max_cluster_data```: Data for the largest observed cluster(s).

* ```null_cluster_distribution```: Null distribution of cluster statistics.

* **Output**:

* ```cluster_pvalue```: Non-parametric p-value for the observed cluster(s).

## **Step 6 (optional): Generate and save plots**

```beta_plot, tstat_plot, cluster_plot, max_cluster_plot, null_distribution_plot = NeuroCluster.plot_neurocluster_results(betas, cluster_test, max_cluster_data, null_cluster_distribution, tstats, tstat_threshold, cluster_pvalue,freqs)```
    
```output_directory = f'{results_dir}/{demo_channel}_{target_var}'```

```NeuroCluster.create_directory(output_directory)```

```NeuroCluster.save_plot_to_pdf(beta_plot, output_directory, 'beta_plot.png')```

```NeuroCluster.save_plot_to_pdf(tstat_plot, output_directory, 'tstat_plot.png')```

```NeuroCluster.save_plot_to_pdf(cluster_plot, output_directory, 'cluster_plot.png')```

```NeuroCluster.save_plot_to_pdf(max_cluster_plot, output_directory, 'max_cluster_plot.png')```

```NeuroCluster.save_plot_to_pdf(null_distribution_plot, output_directory, 'null_distribution_plot.png')```

**Explanation:**

* **```plot_neurocluster_results()```:**

* **Purpose**: Generates visualizations for each step of the analysis:

* **```beta_plot```:** Visualizes beta coefficients.

* **```tstat_plot```:** Displays the t-statistics matrix.

* **```cluster_plot```:** Shows significant clusters.

* **```max_cluster_plot```:** Highlights the maximum observed cluster.

* **```null_distribution_plot```:** Plots the null distribution of cluster statistics.


* **```create_directory()```:** Ensures the output directory exists.

* **```save_plot_to_pdf()```:** Saves the generated plots to the specified directory in ```.png``` format.

# Example Data

This directory contains de-identified example neural (local field potential) and behavioral data for running NeuroCluster. Each electrode's neural data is stored in a separate ```.npy``` file. Behavioral data stored in ```sample_behavior.csv``` contains continuous model-based (expected value, RPE) and discrete model-free (reward outcome, condition) corresponding with the neural data provided. This example data provides users an opportunity to experiment locally with the NeuroCluster method and provides a template for formatting data to be used with this pipeline. 

# Generating Synthetic Data  

This directory contains a script for generating synthetic time-frequency representation (TFR) data with a known association to a simulated continuous variable. This was done to validate the method using a dataset with a known ground truth.  

The notebook that generates the data is located at:  
`/data/synthetic_validation_tfr_data/Generate_Synthetic_TFR_data.ipynb`  

Findings from data generated using this notebook are reported in the manuscript (Fig. 3). The generated data can be found in the same path.  

