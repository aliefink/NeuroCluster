import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import mne 

# DEV - functions for plotting neurocluster results 

def plot_beta_coef(betas, cluster_test):

    plt.imshow(betas, interpolation = 'Bicubic',cmap='Spectral_r', aspect='auto',origin='lower',vmin=-.5,vmax=.5) 
    plt.colorbar()
    plt.ylabel('Freq')
    plt.xlabel('Time')
    # make title dynamic depending on whether or not you are controlling for other variables
    if cluster_test.predictor_data.columns.tolist() == [cluster_test.permute_var]:
        plt.title(f'Beta coefficients from {cluster_test.ch_name} encoding {cluster_test.permute_var}')
    else:
        beh_variables = cluster_test.predictor_data.columns.tolist().copy()
        control_variables = beh_variables.remove(cluster_test.permute_var) # to do - fix this bc its not printing as it should 
        plt.title(f'Beta coefficients from {cluster_test.ch_name} encoding {cluster_test.permute_var} controlling for {control_variables}')
    plt.show()


def plot_tstats(tstats, cluster_test):

    plt.imshow(tstats, interpolation = 'Bicubic',cmap='Spectral_r', aspect='auto',origin='lower',vmin=-3,vmax=3) 
    plt.colorbar()
    plt.ylabel('Freq')
    plt.xlabel('Time')
    # make title dynamic depending on whether or not you are controlling for other variables
    if cluster_test.predictor_data.columns.tolist() == [cluster_test.permute_var]:
        plt.title(f'T-statistics for beta coefficients from {cluster_test.ch_name} encoding {cluster_test.permute_var}')
    else:
        beh_variables = cluster_test.predictor_data.columns.tolist().copy()
        control_variables = beh_variables.remove(cluster_test.permute_var) # to do - fix this because its not printing as it should 
        plt.title(f'T-statistics for beta coefficents from {cluster_test.ch_name} encoding {cluster_test.permute_var} controlling for {control_variables}')
    plt.show()

def plot_clusterstats(cluster_data, tstats, cluster_test):

    # Loop through the list of dictionaries
    for cluster in cluster_data:
        # Initialize an array the same shape as the tstat
        masked_tstat_plot = np.zeros_like(tstats)

        # Extract the indices from the dictionary
        freq_start, freq_end = cluster['freq_idx']
        time_start, time_end = cluster['time_idx']

        # Copy the values from tstat_plot to masked_tstat_plot for the significant cluster range
        masked_tstat_plot[freq_start:freq_end+1, time_start:time_end+1] = 1

        # Plot the masked tstat plot
        plt.imshow(masked_tstat_plot, interpolation='bicubic', cmap='Spectral_r', aspect='auto', origin='lower', vmin=-3, vmax=3)
        plt.ylabel('Freq')
        plt.xlabel('Time')
        plt.title(f'Significant cluster from {cluster_test.ch_name} encoding {cluster_test.permute_var}')
        plt.show()

