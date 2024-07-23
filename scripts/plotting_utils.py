import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import mne 

def plot_beta_coef(betas, cluster_test):
    """
    Plots the beta coefficients for regressor of interest from a linear regression. 

    Args:
    - betas (np.array): 2D array of beta coefficients (frequency x time) from a linear regression.
    - cluster_test (NeuroCluster object): object containing the model specifications. 

    Returns:
    - None: displays a plot of the beta coefficients.

    """
    fig = plt.figure(figsize=(4,4))

    plt.imshow(betas, interpolation = 'Bicubic',cmap='Spectral_r', aspect='auto',origin='lower') 
    cbar = plt.colorbar()
    cbar.set_label('Beta Coefficient')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time')

    # make title dynamic depending on whether or not you are controlling for other variables
    if cluster_test.predictor_data.columns.tolist() == [cluster_test.permute_var]:
        plt.title(f'{cluster_test.ch_name} encoding \n{cluster_test.permute_var}')
    else:
        beh_variables = cluster_test.predictor_data.columns.tolist().copy()
        control_variables = [var for var in beh_variables if var != cluster_test.permute_var] 
        control_variables_str = ", ".join(control_variables)
        plt.title(f'{cluster_test.ch_name} encoding {cluster_test.permute_var} \ncontrolling for {control_variables_str}')
    plt.show()

    return fig
    

def plot_tstats(tstats, cluster_test):
    """
    Plots the t statistics for the regressor of interest.

    Args:
    - tstats (np.array): 2D array of t statistics (frequency x time) corresponding with the beta coefficients for a linear regression. 
    - cluster_test (NeuroCluster object): object containing the model specifications.

    Returns:
    - None: displays a plot of the t statistics.

    """
    fig = plt.figure(figsize=(4,4))
    
    plt.imshow(tstats, interpolation = 'Bicubic',cmap='Spectral_r', aspect='auto',origin='lower') 
    cbar = plt.colorbar()
    cbar.set_label('T Statistic')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (ms)')
    # make title dynamic depending on whether or not you are controlling for other variables
    if cluster_test.predictor_data.columns.tolist() == [cluster_test.permute_var]:
        plt.title(f'{cluster_test.ch_name} encoding \n{cluster_test.permute_var}')
    else:
        beh_variables = cluster_test.predictor_data.columns.tolist().copy()
        control_variables = [var for var in beh_variables if var != cluster_test.permute_var] 
        control_variables_str = ", ".join(control_variables)
        plt.title(f'{cluster_test.ch_name} encoding {cluster_test.permute_var} \ncontrolling for {control_variables_str}')
    plt.show()

    return fig

def plot_clusters(tstat_threshold,alternative='two-sided'):
    """
    Plots clusters based on pixels with significant t-statistics for regressor of interest.

    Args:
    tstat_threshold (list): List (either length 2 if alternative == 'two-sided' or length 1 if alternative == 'less' | 'greater') of binary matrices (frequency x time) for significant t-statistics.
    max (bool): If True, plots the maximum cluster. If False, plots all clusters. Default is True.
    alternative (str): Type of test. Default is 'two-sided' but can also be 'greater' or 'less'. 

    Returns:
    - None: displays a plot of the significant t statistics.

    """
    if alternative == 'two-sided':
        fig, axs = plt.subplots(1,2,figsize=(8, 4))

        for i in range(len(tstat_threshold)):
            if i == 0:
                axs[i].imshow(tstat_threshold[i], interpolation = 'Bicubic',cmap='Reds', aspect='auto',origin='lower')
                axs[i].set_ylabel('Frequency (Hz)')
                axs[i].set_xlabel('Time (ms)')
                axs[i].set_title(f'T Statistic \n Above +Threshold')
            else:
                axs[i].imshow(tstat_threshold[i], interpolation = 'Bicubic',cmap='Blues', aspect='auto',origin='lower')
                axs[i].set_ylabel('Frequency (Hz)')
                axs[i].set_xlabel('Time (ms)')
                axs[i].set_title(f'T Statistic \n Below -Threshold')
            plt.tight_layout()
        plt.show()
    elif alternative == 'greater':
        plt.imshow(tstat_threshold[0], interpolation = 'Bicubic',cmap='Reds', aspect='auto',origin='lower')
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (ms)')
        plt.title('T Statistic Above +Threshold')
        plt.show()
    elif alternative == 'less':
        plt.imshow(tstat_threshold[0], interpolation = 'Bicubic',cmap='Blues', aspect='auto',origin='lower')
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (ms)')
        plt.title('T Statistic \n Below -Threshold')
        plt.show()
    
    return fig


def plot_max_clusters(max_cluster_data, tstats):
    """
    Plots significant clusters (positive and negative).

    Args:
    - max_cluster_data (list): List of dictionaries containing the significant cluster information.
    - tstats (np.array): 2D array of t statistics (frequency x time) corresponding with the beta coefficients for a linear regression.

    Returns:
    - None: displays a plot of the maximum cluster(s).

    """

    if len(max_cluster_data) > 1:
        fig,axs = plt.subplots(1,2,figsize=(8, 4))
        # Loop through the list of dictionaries
        for i,cluster in enumerate(max_cluster_data):
        
            # Initialize an array the same shape as the tstat
            masked_tstat_plot = np.zeros_like(tstats)

            # Extract the indices from the dictionary
            freq_start, freq_end = cluster['freq_idx']
            time_start, time_end = cluster['time_idx']

            # Copy the values from tstat_plot to masked_tstat_plot for the significant cluster range
            masked_tstat_plot[freq_start:freq_end+1, time_start:time_end+1] = 1
            
            if cluster['cluster_stat'] > 0:
            # Plot the masked tstat plot
                axs[i].imshow(masked_tstat_plot, interpolation='bicubic', cmap='binary', aspect='auto', origin='lower')
                axs[i].set_ylabel('Freq')
                axs[i].set_xlabel('Time')
                axs[i].set_title(f'Positive cluster')
            elif cluster['cluster_stat'] < 0:
                axs[i].imshow(masked_tstat_plot, interpolation='bicubic', cmap='binary', aspect='auto', origin='lower')
                axs[i].set_ylabel('Freq')
                axs[i].set_xlabel('Time')
                axs[i].set_title(f'Negative cluster')
            plt.tight_layout()
        plt.show()
    else:

        # Initialize an array the same shape as the tstat
        masked_tstat_plot = np.zeros_like(tstats)

        # Extract the indices from the dictionary
        freq_start, freq_end = max_cluster_data[0]['freq_idx']
        time_start, time_end = max_cluster_data[0]['time_idx']

        # Copy the values from tstat_plot to masked_tstat_plot for the significant cluster range
        masked_tstat_plot[freq_start:freq_end+1, time_start:time_end+1] = 1

        if max_cluster_data[0]['cluster_stat'] > 0:
            # Plot the masked tstat plot
            plt.imshow(masked_tstat_plot, interpolation='bicubic', cmap='binary', aspect='auto', origin='lower')
            plt.ylabel('Freq')
            plt.xlabel('Time')
            plt.title(f'Positive cluster')
        elif max_cluster_data[0]['cluster_stat'] < 0:
            plt.imshow(masked_tstat_plot, interpolation='bicubic', cmap='binary', aspect='auto', origin='lower')
            plt.ylabel('Freq')
            plt.xlabel('Time')
            plt.title(f'Negative cluster')
        plt.show() 

        return fig


def plot_null_distribution(null_cluster_distribution, max_cluster_data, axs=None):
    """
    Plots the null distribution of the cluster permutation test.

    Args:
    - null_cluster_distribution (np.array): 1D array of cluster statistics from the permutation test.
    - max_cluster_data (list): List of dictionaries containing the significant cluster(s) statistics.
    - axs (matplotlib.axes): List of axes to plot on. Default is None.


    Returns:
    - None: displays a plot of the null distribution.

    """
        # initialize plots
    fig, axs = plt.subplots(1, len(max_cluster_data), figsize=(8,4))
    for i, cluster in enumerate(max_cluster_data):
        axs[i].hist(null_cluster_distribution[i], bins=20, color='gray',edgecolor='black')
        axs[i].axvline(cluster['cluster_stat'], color='red', linestyle='dashed', linewidth=2)
        axs[i].set_xlabel('Cluster Statistic')
        axs[i].set_ylabel('Frequency')
        if cluster['cluster_stat'] > 0:
            axs[i].set_title(f' Null Distribution\n Positive Cluster')
        else:
            axs[i].set_title(f' Null Distribution\n Negative Cluster')
    plt.tight_layout()
    plt.show()

    return fig

# plot all results in one grid using each of the above functions

def plot_neurocluster_results(betas,cluster_test, max_cluster_data, null_cluster_distribution, tstats, tstat_threshold):
    """
    Plots all the results from a NeuroCluster object.

    Args:
    - betas (np.array): 2D array of beta coefficients (frequency x time) from a linear regression.
    - cluster_test (NeuroCluster object): object containing the model specifications.
    - max_cluster_data (list): List of dictionaries containing the significant cluster(s) statistics.
    - null_cluster_distribution (np.array): 1D array of cluster statistics from the permutation test.
    - tstats (np.array): 2D array of t statistics (frequency x time) corresponding with the beta coefficients for a linear regression.
    - tstat_threshold (list): List (either length 2 if alternative == 'two-sided' or length 1 if alternative == 'less' | 'greater') of binary matrices (frequency x time) for significant t-statistics.

    Returns:
    - None: displays beta coefficients, t statistics, significant clusters, maximum cluster(s), and null distribution plots.

    """
    beta_plot = plot_beta_coef(betas, cluster_test)
    tstat_plot = plot_tstats(tstats, cluster_test)
    cluster_plot = plot_clusters(tstat_threshold, alternative='two-sided')
    max_cluster_plot= plot_max_clusters(max_cluster_data, tstats)
    null_distribution_plot = plot_null_distribution(null_cluster_distribution, max_cluster_data)

    return beta_plot,tstat_plot,cluster_plot,max_cluster_plot,null_distribution_plot

