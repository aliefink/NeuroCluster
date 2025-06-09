---
title: "NeuroCluster: A Python toolbox for nonparametric cluster-based statistical testing of neurophysiological data with respect to continuous predictors."
tags:
  - Python
  - neurophysiology
  - non-parametric statistics
  - cluster-based permutation testing
  - spectro-temporal resolution
  - human intracranial electrophysiology
  - complex behavioral predictors
authors:
  - name: Alexandra Fink Skular
    orcid: 0000-0003-1648-4604
    equal-contrib: true
    affiliation: "1, 2, 3"
  - name: Christina Maher
    orcid: 0009-0003-8188-2083
    equal-contrib: true
    affiliation: "1, 2, 3"
  - name: Salman Qasim
    orcid: 0000-0001-8739-5962
    corresponding: false
    affiliation: "2, 3"
  - name: Ignacio Saez
    orcid: 0000-0003-0651-2069
    corresponding: true
    affiliation: "1, 2, 3, 4, 5"
affiliations:
  - name: 'The Nash Family Department of Neuroscience, Icahn School of Medicine at Mount Sinai, NY, NY'
    index: 1
  - name: 'The Nash Family Center for Advanced Circuit Therapeutics, The Mount Sinai Hospital, NY, NY'
    index: 2
  - name: 'The Center for Computational Psychiatry, Icahn School of Medicine at Mount Sinai, NY, NY'
    index: 3
  - name: 'Department of Neurosurgery, The Mount Sinai Hospital, NY, NY'
    index: 4
  - name: 'Department of Neurology, The Mount Sinai Hospital, NY, NY'
    index: 5
date: "2025-02-27"
bibliography: paper.bib
---
# Summary

Cognitive neurophysiology offers a unique framework for studying cognitive brain-behavior relationships by relating electrophysiological signals to complex behaviors. With the advent of new technical and behavioral paradigms, researchers can design cognitive experiments that leverage both the spatiotemporal resolution of electrophysiological data and the complexity of continuous behavioral variables. Analyzing these data requires sophisticated statistical methods that can interpret multidimensional neurophysiological data and dynamic, continuous behavioral variables. Often used statistical frameworks for nonparametric, cluster-based statistical tests are specifically focused on the contrast between discrete behavioral conditions but are not suitable for assessing how continuous variables predict the occurrence of clusters in neurophysiological data. NeuroCluster is an open-source Python toolbox for analysis of two-dimensional electrophysiological data (e.g. time-frequency representations)  related to multivariate and continuous behavioral variables.  NeuroCluster introduces a statistical approach which uses nonparametric cluster-based permutation testing in tandem with linear regression to identify two-dimensional clusters of neurophysiological activity that significantly encodes time-varying, continuous behavioral variables. Uniquely, it also supports multivariate analyses by allowing for multiple behavioral predictors to model neural activity. NeuroCluster addresses a methodological gap in statistical approaches to relate continuous, cognitive predictors to underlying electrophysiological activity with time and frequency resolution, to determine the neurocomputational processes giving rise to complex behaviors. 

# Statement of need

`NeuroCluster` addresses a methodological gap in cognitive and behavioral neuroscience, by providing a Python-based statistical toolbox to relate continuous predictors to two-dimensional neurophysiological activity. Continuous predictors vary over an experimental session, reflecting dynamic behaviors, underlying cognitive processes, complex movements, trial-varying experimental conditions, perceptual signals, or value-based trial outcomes [@collins2022; @hoy2021; @mathis2020; @odoherty2007a]. Standard analytical approaches for relating complex behavioral variables to neuronal activity sacrifice the complexity of neurophysiological signals by reducing the dimensionality of neuronal timeseries data (e.g., averaging across temporal, spectral, or spatial domains, or dimensionality reduction) [@crosse2016; @lopezpersem2020; @rey2015; @saboo2019; @saez2018; @stokes2016]. Conversely, analysis methods that preserve the complexity of neurophysiological data (i.e., two-dimensional timeseries) constrain behavioral predictors to discrete conditions [@candiarivera2022; @domenech2020; @kosciessa2020; @marcopallares2015; @maris2007]. Directly linking continuous experimental variables to two-dimensional physiological timeseries data offers a rigorous way to study brain-behavior relationships, by maintaining the complexity of dynamic behavior, without sacrificing the resolution of event-related neurophysiological activity.

`NeuroCluster` uses cluster-based permutation testing to identify significant two-dimensional clusters with respect to continuous task variables. Cluster-based nonparametric statistical testing is a standard approach to analyze two-dimensional event-related time series data, while controlling for multiple comparisons and reducing family-wise error rates [@cohen2014; @groppe2011; @maris2012; @maris2007; @nichols2002]. Neurophysiological activity is typically aggregated by condition to perform a two-sample cluster-based permutation test, which tests whether the neuronal encoding patterns differ between two discrete task conditions, rather than continuous, trial-varying features [@bullmore1999; @maris2007]. While two-sample cluster-based permutation tests provide a nonparametric statistical inference tool for identifying the presence of significant clusters of activity between two conditions, they are insufficient for identifying the presence of clusters as a function of continuously varying predictors. `NeuroCluster` provides a solution to this analytical gap by performing linear regressions at individual points across the 2D neural matrix. This approach enables users to quantify the degree to which a continuous predictor is related to neurophysiological activity at the pixel-level and to perform analyses with multivariate behavioral data, by incorporating multiple continuous or categorical covariates in the regression models. The t-statistics corresponding to the predictor of interest from the pixel-wise regressions are thresholded by a critical t-statistic to control for the FDR, creating a binary 2D matrix [@genovese2002]. The binary 2D matrix is then used to identify putative 2D clusters of activation related to the continuous predictor of interest. This process is repeated many times with the predictor of interest randomly permuted to produce a surrogate distribution of 2D clusters. Clusters that survive cluster-based permutation testing are classified as significant regions of activation with respect to the specified continuous predictor.

`NeuroCluster` is applicable for numerous analysis goals; the major use cases are performing an initial exploratory analysis to generate specific hypotheses, determine data-driven windows interest, or to identify regional patterns of significant clusters within and between subjects. Future adaptations of `NeuroCluster` may implement mixed effects regressions, nonlinear mapping models, or group-level analysis frameworks [@bianchi2019; @ivanova2022; @konig2024; @yu2022]. We demonstrate our approach with human intracranial local field potential data, but NeuroCluster is applicable for all types of two-dimensional neurophysiological measures (e.g., spatiotemporal clusters from EEG/MEG, cross-frequency interactions). To our knowledge, `NeuroCluster` presents a novel Python-based statistical software package. `NeuroCluster` is designed to supplement existing Python-based electrophysiological analysis toolboxes [@donoghue2020; @gramfort2013; @kosciessa2020; @whitten2011], particularly MNE-Python.


# NeuroCluster Documentation

`NeuroCluster` is accompanied by a detailed tutorial which outlines the workflow (Fig. 1) for implementing this approach with time-frequency power estimates from multi-region LFP recording.


![NeuroCluster workflow. This approach involves three key steps: (1) determine cluster statistic in true data, (2) generate a null distribution of cluster statistics by permuting dataset, (3) determine significance of true cluster statistic against null distribution.](../NeuroCluster/workflow/Figure1_workflow.png){#fig:Fig1}

Below we outline the statistical approach implemented by `NeuroCluster` for performing nonparametric permutation-based cluster testing using time-frequency resolved power estimates from neural data estimated using [@python-mne] and continuous predictors (i.e., latent cognitive processes, behavior, or experimental conditions). In these example data, we are testing the hypothesis that RPEs are significantly encoded in the electrophysiological signal from a given iEEG channel time-frequency representation (TFR). The following methodological description is based on data collected from a neurosurgical epilepsy patient undergoing stereotactic EEG (sEEG) monitoring for treatment-resistant depression. During the monitoring period, the patient performed a value-based decision-making task while local field potentials (LFPs) were recorded from both cortical and subcortical sites. By analyzing the patient's behavior during the task, we derived continuous variables representing hypothesized latent cognitive processes—such as the trial-by-trial computation of reward prediction errors (RPEs)—to examine their relationship with neural activity.

## 1.	Determine cluster statistic in true data 

#### A.   Define clusters: At each time-frequency index, we perform a linear univariate (or multivariate) regression using behaviorally-derived independent variables (e.g., latent cognitive variables, behavioral measures, task conditions) to predict neuronal activity (i.e., power). The β coefficient represents the strength and direction of the relationship between each independent variable and the dependent variable. It is estimated from the regression model and reflects how changes in the independent variable are associated with changes in power at the specific time-frequency pair. Pixel-wise regressions are parallelized for speed. For each time-frequency pair, the β coefficient for the regressor of interest (the independent variable of primary interest) is extracted from the regression results (Fig 2A). A t-statistic is computed for the β coefficient to capture how significantly different it is from zero (Fig 2B). A significance threshold is applied to the t-statistics of the β coefficient for the regressor of interest. If the t-statistic for a time-frequency pair exceeds the significance threshold, the pair is deemed significant. Clusters are then defined as adjacent time-frequency pairs where all pairs within the cluster have t-statistics exceeding the threshold, according to the test's desired tails (Fig 2C).

#### B.	Compute cluster statistics: For each identified cluster, sum the t-statistics of all time-frequency pairs within the cluster. In a two-tailed test (the default), compute both the maximum and minimum cluster sums (Fig 2D).


## 2. Generate null distribution of cluster statistics

#### A.	Permutation procedure: Labels for the behavioral predictor of interest are shuffled for the desired number of permutations. 

#### B.	Recalculate cluster statistic: Steps 1A/1B are repeated to define clusters and compute cluster statistics for each permuted dataset. 

#### C.	Construct null distribution: The cluster statistics from all permutations are compiled to create a null distribution, representing the distribution of cluster statistics under the null hypothesis (Fig 2E). The permuted TFR regressions are also parallelized at the pixel-level, while each permutation is performed sequentially. We tested many iterations of these functions with different parallelization approaches and sequential permutation-level computations with pixel-level parallelization within each TFR regression was the fastest method. 

## 3.	Determine cluster significance 

#### A.	Compare true cluster statistic to null distribution to compute p-values: The proportion of cluster statistics in the null distribution falling above (or below) the true cluster statistic(s) determines the p-value associated with the cluster(s) identified in the true data (Fig 2E). 

![NeuroCluster methods. A. β coefficients for continuous predictor of interest (RPE) predicting power in given time-frequency pair (red outline = maximum positive cluster; blue outline = maximum negative cluster). B. T-statistics corresponding with βRPE coefficients. C. Clusters as determined using t-critical threshold. D. Maximum positive and negative clusters determined by summing t-statistics in identified clusters. E. Null distribution of cluster statistics generated by permuting dataset for predictor of interest (100 permutations; red dashed line = true cluster statistic.](../NeuroCluster/workflow/Figure2.png){#fig:Fig2}


## 4. Comparison of results to existing methods. 

To evaluate the advantages of NeuroCluster, we compared its results to those obtained using MNE-Python's two-sample cluster-based permutation test. This approach requires discretizing the continuous variable of interest (RPE) into distinct categories, which reduces the resolution of the behavioral predictor. Additionally, MNE-Python's implementation does not support multivariate analyses, limiting the ability to model multiple behavioral covariates simultaneously. When applying the two-sample cluster test to our data, we did not identify any significant clusters of increased or decreased activity related to RPE. In contrast, NeuroCluster successfully detected significant clusters (Fig. 2), demonstrating its ability to preserve the richness of continuous behavioral variables and reduce the likelihood of false negatives. This comparison highlights NeuroCluster as a powerful and flexible alternative to existing statistical methods for analyzing continuous brain-behavior relationships.

## 5. Metric validation in synthetic data with known ground truth. 

Thus far, we have demonstrated NeuroCluster using biological data. However, because these data are experimental, there is no definitive ground truth for the observed neural fluctuations associated with behavioral predictors. To validate the NeuroCluster method, we generated synthetic TFR data (2-200 Hz, sampling rate = 250, -1 to +1 seconds around "choice", 1 channel, 100 trials) with a known linear association between power in a specific time-frequency region and a continuous behavioral variable—in this case, the expected value of choice. Code for simulating these data is provided in the NeuroCluster repository. We then applied NeuroCluster to the synthetic dataset and, as expected, successfully identified a significant positive cluster corresponding to the known association embedded in the data (Fig. 3). This validation confirms the accuracy of NeuroCluster and provides evidence against its susceptibility to false positives.

![NeuroCluster validation in synthetic data. A. Time-frequency representation (TFR) showing power differences between high (>0.50) and low (<0.50) expected value trials in synthetic data (1 channel, 100 trials, time-locked to "choice"). B. A significant positive cluster identified in the expected time-frequency region, consistent with the predefined association embedded in the synthetic dataset.](../NeuroCluster/workflow/Figure3.png){#fig:Fig3}


# Acknowledgements

We acknowledge feedback from Shawn Rhoads and support from Xiaosi Gu and Angela Radulescu during the genesis of this project.

# References