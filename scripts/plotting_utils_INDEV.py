import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import mne 

# DEV - functions for plotting neurocluster results 

def plot_beta_coefs(betas, tfr, freqs, times, title, save_path=None):
    """
    Plot beta coefficients from pixel-wise regression. 

    Args:
    - betas (numpy array): Beta coefficients from pixel-wise regression.
    - tfr (numpy array): Time-frequency representation of data.
    - freqs (numpy array): Frequency values.
    - times (numpy array): Time values.
    - title (str): Title for plot.
    - save_path (str): Path to save plot. 

    Returns:
    - fig, ax : plt.figure
        Figure and axis of plot.
    """
    
    fig, ax = plt.subplots(1,1,figsize=(10,10))
    im = ax.imshow(betas, aspect='auto', origin='lower', cmap='RdBu_r', 
                   extent=[times[0], times[-1], freqs[0], freqs[-1]])
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    plt.colorbar(im, ax=ax)
    
    if save_path:
        plt.savefig(save_path)
    
    return fig, ax

def plot_t_stats(tstats, tfr, freqs, times, title, save_path=None):
    """
    Plot t-statistics from pixel-wise regression. 

    Args:
    - tstats (numpy array): T-statistics from pixel-wise regression.
    - tfr (numpy array): Time-frequency representation of data.
    - freqs (numpy array): Frequency values.
    - times (numpy array): Time values.
    - title (str): Title for plot.
    - save_path (str): Path to save plot. 

    Returns:
    - fig, ax : plt.figure
        Figure and axis of plot.
    """
    
    fig, ax = plt.subplots(1,1,figsize=(10,10))
    im = ax.imshow(tstats, aspect='auto', origin='lower', cmap='RdBu_r', 
                   extent=[times[0], times[-1], freqs[0], freqs[-1]])
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    plt.colorbar(im, ax=ax)
    
    if save_path:
        plt.savefig(save_path)
    
    return fig, ax

def plot_sig_clusters(tfr_tstats, tfr, freqs, times, title, save_path=None):
    """
    Plot significant positive and negative clusters from pixel-wise regression. 

    Args:
    - tfr_tstats (numpy array): T-statistics from pixel-wise regression.
    - tfr (numpy array): Time-frequency representation of data.
    - freqs (numpy array): Frequency values.
    - times (numpy array): Time values.
    - title (str): Title for plot.
    - save_path (str): Path to save plot. 

    Returns:
    - fig, ax : plt.figure
        Figure and axis of plot.
    """
    
    fig, ax = plt.subplots(1,1,figsize=(10,10))
    im = ax.imshow(tfr_tstats, aspect='auto', origin='lower', cmap='RdBu_r', 
                   extent=[times[0], times[-1], freqs[0], freqs[-1])
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    plt.colorbar(im, ax=ax)
    
    if save_path:
        plt.savefig(save_path)
    
    return fig, ax

def plot_null_distribution(null_dist, title, observed_stat=None, save_path=None):
    """
    Plot null distribution of cluster statistics. 

    Args:
    - null_dist (numpy array): Null distribution of cluster statistics.
    - title (str): Title for plot.
    - save_path (str): Path to save plot. 
    - observed_stat (float): Observed cluster statistic.

    Returns:
    - fig, ax : plt.figure
        Figure and axis of plot.
    """
    
    fig, ax = plt.subplots(1,1,figsize=(10,10))
    ax.hist(null_dist, bins=50, color='b', alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel('T-statistic')
    ax.set_ylabel('Frequency')

    # add vertical line at observed cluster statistic if available
    if observed_stat:
        ax.axvline(observed_stat, color='r', linestyle='--', label='Observed Statistic')
        ax.legend()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig, ax

def plot_all(tfr, betas, tstats, tfr_tstats, null_dist, freqs, times, title, save_path=None):
    """
    Plot all relevant plots for nonparametric pixel-wise regression using the plotting functions above. 

    Args:
    - tfr (numpy array): Time-frequency representation of data.
    - betas (numpy array): Beta coefficients from pixel-wise regression.
    - tstats (numpy array): T-statistics from pixel-wise regression.
    - tfr_tstats (numpy array): T-statistics from pixel-wise regression.
    - null_dist (numpy array): Null distribution of cluster statistics.
    - freqs (numpy array): Frequency values.
    - times (numpy array): Time values.
    - title (str): Title for plot containing electrode and condition information.
    - save_path (str): Path to save plot.

    Returns:
    - fig, ax : plt.figure
        Figure and axis of plot.
    """

    fig, ax = plt.subplots(2,2,figsize=(20,20))

    plot_beta_coefs(betas, tfr, freqs, times, 'Beta Coefficients', save_path=None)
    plot_t_stats(tstats, tfr, freqs, times, 'T-Statistics', save_path=None)
    plot_sig_clusters(tfr_tstats, tfr, freqs, times, 'Significant Clusters', save_path=None)
    plot_null_distribution(null_dist, 'Null Distribution of Cluster Statistics', save_path=None)

    # add title to figure
    fig.suptitle(title)

    if save_path:
        plt.savefig(save_path)

    return fig, ax

def plot_sig_electrodes(sig_electrodes, title, save_path=None):

    """
    Plot the proportion of electrodes with significant clusters for a given condition

    Args:
    - sig_electrodes (numpy array): Proportion of electrodes with significant clusters.
    - conditions (string): Conditions for which the proportion of electrodes is calculated.
    - save_path (str): Path to save plot. 

    Returns:
    - fig, ax : plt.figure
        Figure and axis of plot.
    """

    fig, ax = plt.subplots(1,1,figsize=(10,10))
    ax.plot(sig_electrodes, color='b')
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel({f'Proportion of electrodes encoding {condition'})

    if save_path:
        plt.savefig(save_path)

    return fig, ax




