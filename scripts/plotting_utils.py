import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.text import OffsetFrom



def plot_beta_coef(betas, cluster_test,figsize=(6,4),dpi=150,sns_context='talk',cmap='Spectral_r'):
    """
    Plots the beta coefficients for regressor of interest from a linear regression. 

    Args:
    - betas : (np.array): 2D array of beta coefficients (frequency x time) from a linear regression.
    - cluster_test : (NeuroCluster object): object containing the model specifications. 
    - figsize (tuple): size of the figure. Default is (8,4).
    - dpi (int): dots per inch for the plot. Default is 150.
    - sns_context (str): seaborn context for the plot. Default is 'talk'.
    - cmap (str): colormap for the plot. Default is 'Spectral_r'.

    Returns:
    - None: displays a plot of the beta coefficients.

    """
    sns.set_context(sns_context)
    fig = plt.figure(figsize=figsize,dpi = dpi)

    plt.imshow(betas, interpolation = 'Bicubic', cmap=cmap, aspect='auto', 
               origin='lower')
    cbar = plt.colorbar()
    # cbar.set_label(r'$Beta $Coefficient')
    cbar.set_label(fr'$\beta_{{{cluster_test.permute_var}}}$'+  ' coefficient')
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
    plt.close(fig) 
    return fig
    

def plot_tstats(tstats, cluster_test,figsize=(6,4),dpi=150,sns_context='talk',cmap='Spectral_r'):
    """
    Plots the t statistics for the regressor of interest.

    Args:
    - tstats (np.array): 2D array of t statistics (frequency x time) corresponding with the beta coefficients for a linear regression. 
    - cluster_test (NeuroCluster object): object containing the model specifications.
    - figsize (tuple): size of the figure. Default is (8,4).
    - dpi (int): dots per inch for the plot. Default is 150.
    - sns_context (str): seaborn context for the plot. Default is 'talk'.
    - cmap (str): colormap for the plot. Default is 'Spectral_r'.

    Returns:
    - None: displays a plot of the t statistics.

    """
    
    sns.set_context(sns_context)
    fig = plt.figure(figsize=figsize,dpi=dpi)
    plt.imshow(tstats, interpolation = 'Bicubic', cmap=cmap, aspect='auto', origin='lower')     
    
    cbar = plt.colorbar()
    cbar.set_label(r'$t-statistic$')
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
    plt.close(fig) 
    return fig

def plot_clusters(tstats,cluster_test,alternative='two-sided',figsize=(8,4),dpi=150,sns_context='talk'):
    """
    Plots clusters based on pixels with significant t-statistics for regressor of interest.

    Args:
    tstat_threshold (list): List (either length 2 if alternative == 'two-sided' or length 1 if alternative == 'less' | 'greater') of binary matrices (frequency x time) for significant t-statistics.
    max (bool): If True, plots the maximum cluster. If False, plots all clusters. Default is True.
    alternative (str): Type of test. Default is 'two-sided' but can also be 'greater' or 'less'. 
    figsize (tuple): size of the figure. Default is (8,4).
    dpi (int): dots per inch for the plot. Default is 150.
    sns_context (str): seaborn context for the plot. Default is 'talk'.

    Returns:
    - None: displays a plot of the significant t statistics.

    """
    
    sns.set_context(sns_context)

    if alternative == 'two-sided':
        # binary tstat matrices thresholded by tcritical value
        tstat_threshold = cluster_test.threshold_tfr_tstat(tstats)
        tcritical = cluster_test.compute_tcritical()

        fig, axs = plt.subplots(1,2,figsize=figsize,dpi=dpi,constrained_layout=True)
        for i in range(len(tstat_threshold)):
            if i == 0:
                axs[i].imshow(tstat_threshold[i], interpolation = 'Bicubic',cmap='Reds', aspect='auto',origin='lower')
                axs[i].set_ylabel('Frequency (Hz)')
                axs[i].set_xlabel('Time (ms)')
                axs[i].set_title(fr'$t_{{pixel}}>t_{{critical}}={np.round(tcritical,2)}$',fontsize=15)
            
            else:
                axs[i].imshow(tstat_threshold[i], interpolation = 'Bicubic',cmap='Blues', aspect='auto',origin='lower')
                axs[i].set_xlabel('Time (ms)')
                axs[i].set_title(fr'$t_{{pixel}}<t_{{critical}}={np.negative(np.round(tcritical,2))}$',fontsize=15)
        
        fig.suptitle('Threshold Significant Pixels')
    
    elif alternative == 'greater':
        # binary tstat matrices thresholded by tcritical value
        tstat_threshold = cluster_test.threshold_tfr_tstat(tstats)
        tcritical = cluster_test.compute_tcritical()
        
        fig = plt.figure(figsize=figsize,dpi=dpi,constrained_layout=True)
        plt.imshow(tstat_threshold[0], interpolation = 'Bicubic',cmap='Reds', aspect='auto',origin='lower')
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (ms)')
        plt.title(fr'$t_{{pixel}}>t_{{critical}}={np.round(tcritical,2)}$',fontsize=15)
        fig.suptitle('Threshold Significant Pixels')
        plt.show()
    
    elif alternative == 'less':
        # binary tstat matrices thresholded by tcritical value
        tstat_threshold = cluster_test.threshold_tfr_tstat(tstats)
        tcritical = cluster_test.compute_tcritical()
        fig = plt.figure(figsize=figsize,dpi=dpi,constrained_layout=True)
        plt.imshow(tstat_threshold[0], interpolation = 'Bicubic',cmap='Blues', aspect='auto',origin='lower')
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (ms)')
        plt.title(fr'$t_{{pixel}}<t_{{critical}}={np.negative(np.round(tcritical,2))}$',fontsize=15)
        fig.suptitle('Threshold Significant Pixels')
    
    plt.close(fig) 
    return fig


def plot_max_clusters(cluster_test,tstats,alternative='two-sided',which_cluster='all',figsize=(8,4),dpi=150,sns_context='talk'):
    """
    Plots significant clusters (positive and negative).

    Args:
    - max_cluster_data (list): List of dictionaries containing the significant cluster information.
    - tstats (np.array): 2D array of t statistics (frequency x time) corresponding with the beta coefficients for a linear regression.
    - alternative (str): Type of test. Default is 'two-sided' but can also be 'greater' or 'less'.
    - which_cluster (str): For two-sided tests, indicate which clusters to plot: 'all', 'positive', or 'negative'. Default is 'all'. 
    - figsize (tuple): size of the figure. Default is (8,4).
    - dpi (int): dots per inch for the plot. Default is 150.
    - sns_context (str): seaborn context for the plot. Default is 'talk'.
    - sns_style (str): seaborn style for the plot. Default is 'white'.

    Returns:
    - None: displays a plot of the maximum cluster(s).

    """
    sns.set_context(sns_context,rc={'axes.linewidth': 1.5})

    if alternative=='two-sided':
        # Compute the max cluster statistics with expanded output to get 2D cluster coordinates
        max_cluster_info = cluster_test.max_tfr_cluster(tstats,output='expanded',alternative=alternative)

        if which_cluster=='all':

            fig,axs = plt.subplots(1,len(max_cluster_info),figsize=figsize,dpi=dpi,constrained_layout=True)

            # Loop through the list of dictionaries
            for i,cluster in enumerate(max_cluster_info):

                # Extract max cluster pixel indices by finding 2D time-freq indices of max cluster 
                cluster_freqs,cluster_times = np.where(cluster['all_clusters']==cluster['max_label'])
                
                # Initialize an array the same shape as the tstat
                masked_tstat_plot = np.zeros_like(tstats)

                # Copy the values from tstat_plot to masked_tstat_plot for the significant cluster 
                masked_tstat_plot[cluster_freqs, cluster_times] = 1
                
                if cluster['cluster_stat'] > 0:
                    # Plot the masked tstat plot
                    axs[i].imshow(masked_tstat_plot, interpolation='bicubic', cmap='Reds', aspect='auto', origin='lower')
                    axs[i].set_ylabel('Frequency (Hz)')
                    axs[i].set_xlabel('Time (ms)')
                    axs[i].text(0.95,0.95,('').join([r'$Max Cluster_{{statistic}}$',f'\n',
                        r'$\sum$ $t_{{pixel}} = $',f'{np.round(max_cluster_info[i]["cluster_stat"],2)}']),color='k',fontsize=11,
                        va='top',ha='right',transform=axs[i].transAxes)                
                    axs[i].set_title(r'Positive TFR Cluster',fontsize=16)

                elif cluster['cluster_stat'] < 0:
                    axs[i].imshow(masked_tstat_plot, interpolation='bicubic', cmap='Blues', aspect='auto', origin='lower')
                    # axs[i].set_ylabel('Frequency (Hz)')
                    axs[i].set_xlabel('Time (ms)')
                    axs[i].text(0.95,0.95,('').join([r'$Max Cluster_{{statistic}}$',f'\n',
                        r'$\sum$ $t_{{pixel}} = $',f'{np.round(max_cluster_info[i]["cluster_stat"],2)}']),color='k',fontsize=11,
                        va='top',ha='right',transform=axs[i].transAxes)
                    axs[i].set_title(r'Negative TFR Cluster',fontsize=16)

        elif which_cluster=='positive':
            cluster = max_cluster_info.copy()[0]

            # Extract max cluster pixel indices by finding 2D time-freq indices of max cluster 
            cluster_freqs,cluster_times = np.where(cluster['all_clusters']==cluster['max_label'])
            
            # Initialize an array the same shape as the tstat
            masked_tstat_plot = np.zeros_like(tstats)

            # Copy the values from tstat_plot to masked_tstat_plot for the significant cluster 
            masked_tstat_plot[cluster_freqs, cluster_times] = 1
            
            fig,ax = plt.subplots(1,1,figsize=figsize,dpi=dpi,constrained_layout=True)

            # Plot the masked tstat plot
            ax.imshow(masked_tstat_plot, interpolation='bicubic', cmap='Reds', aspect='auto', origin='lower')
            ax.set_ylabel('Frequency (Hz)')
            ax.set_xlabel('Time (ms)')
            ax.text(0.95,0.95,('').join([r'$Max Cluster_{{statistic}}$',f'\n',
                r'$\sum$ $t_{{pixel}} = $',f'{np.round(cluster["cluster_stat"],2)}']),color='k',fontsize=11,
                va='top',ha='right',transform=ax.transAxes)                
            ax.set_title(r'Positive TFR Cluster',fontsize=16)
        
        elif which_cluster=='negative':
            cluster = max_cluster_info.copy()[1]

            # Extract max cluster pixel indices by finding 2D time-freq indices of max cluster 
            cluster_freqs,cluster_times = np.where(cluster['all_clusters']==cluster['max_label'])
            
            # Initialize an array the same shape as the tstat
            masked_tstat_plot = np.zeros_like(tstats)

            # Copy the values from tstat_plot to masked_tstat_plot for the significant cluster 
            masked_tstat_plot[cluster_freqs, cluster_times] = 1
            
            # Plot the masked tstat plot
            fig,ax = plt.subplots(1,1,figsize=figsize,dpi=dpi,constrained_layout=True)

            ax.imshow(masked_tstat_plot, interpolation='bicubic', cmap='Blues', aspect='auto', origin='lower')
            ax.set_ylabel('Frequency (Hz)')
            ax.set_xlabel('Time (ms)')
            ax.text(0.95,0.95,('').join([r'$Max Cluster_{{statistic}}$',f'\n',
                    r'$\sum$ $t_{{pixel}} = $',f'{np.round(cluster["cluster_stat"],2)}']),color='k',fontsize=11,
                    va='top',ha='right',transform=ax.transAxes)
            ax.set_title(r'Negative TFR Cluster',fontsize=16)


    else:
        # Compute the max cluster statistics with expanded output to get 2D cluster coordinates
        max_cluster_info = cluster_test.max_tfr_cluster(tstats,output='expanded',alternative=alternative)
        
        # Initialize an array the same shape as the tstat
        masked_tstat_plot = np.zeros_like(tstats)

        # Extract max cluster pixel indices by finding 2D time-freq indices of max cluster 
        cluster_freqs,cluster_times = np.where(max_cluster_info[0]['all_clusters']==max_cluster_info[0]['max_label'])

        # Copy the values from tstat_plot to masked_tstat_plot for the significant cluster 
        masked_tstat_plot[cluster_freqs, cluster_times] = 1

        if max_cluster_info[0]['cluster_stat'] > 0:
            fig,ax = plt.subplots(1,1,figsize=figsize,dpi=dpi,constrained_layout=True)
            
            # Plot the masked tstat plot
            plt.imshow(masked_tstat_plot, interpolation='bicubic', cmap='Reds', aspect='auto', origin='lower')
            plt.ylabel('Frequency (Hz)')
            plt.xlabel('Time (ms)')
            ax.text(0.95,0.95,('').join([r'$Max Cluster_{{statistic}}$',f'\n',
                    r'$\sum$ $t_{{pixel}} = $',f'{np.round(max_cluster_info[0]["cluster_stat"],2)}']),color='k',fontsize=11,
                    va='top',ha='right',transform=ax.transAxes)
            plt.title(r'Positive TFR Cluster',fontsize=16)

        elif max_cluster_info[0]['cluster_stat'] < 0:
            fig,ax = plt.subplots(1,1,figsize=figsize,dpi=dpi,constrained_layout=True)
            plt.imshow(masked_tstat_plot, interpolation='bicubic', cmap='Blues', aspect='auto', origin='lower')
            plt.ylabel('Frequency (Hz)')
            plt.xlabel('Time (ms)')
            ax.text(0.95,0.95,('').join([r'$Max Cluster_{{statistic}}$',f'\n',
                    r'$\sum$ $t_{{pixel}} = $',f'{np.round(max_cluster_info[0]["cluster_stat"],2)}']),color='k',fontsize=11,
                    va='top',ha='right',transform=ax.transAxes)
            plt.title(r'Negative TFR Cluster',fontsize=16)

    plt.close(fig) 

    return fig


def plot_null_distribution(null_cluster_distribution, max_cluster_data, pvalue,figsize=(9,4),dpi=150,sns_context='talk'):
    """
    Plots the null distribution of the cluster permutation test.

    Args:
    - null_cluster_distribution (np.array): 1D array of cluster statistics from the permutation test.
    - max_cluster_data (list): List of dictionaries containing the significant cluster(s) statistics.
    - pvalue (float): p-value associated with the cluster permutation test.
    - figsize (tuple): size of the figure. Default is (12,4).
    - dpi (int): dots per inch for the plot. Default is 150.
    - sns_context (str): seaborn context for the plot. Default is 'talk'.
    - sns_style (str): seaborn style for the plot. Default is 'white'.


    Returns:
    - None: displays a plot of the null distribution.

    """
    sns.set_context(sns_context,rc={'axes.linewidth': 1.5})
    
    # initialize plots
    fig, axs = plt.subplots(1, len(max_cluster_data), figsize=figsize,dpi=dpi)
    for i, cluster in enumerate(max_cluster_data):
        axs[i].hist(null_cluster_distribution[i], bins=20, color='gray',edgecolor='black')
        axs[i].axvline(cluster['cluster_stat'], color='red', linestyle='dashed', linewidth=2)
        axs[i].set_xlabel('Cluster Statistic')
        axs[i].set_ylabel('Count')

        if cluster['cluster_stat'] > 0:
            axs[i].set_title(f' Null Distribution\n Positive Cluster')
            if len(pvalue) == 1:
                axs[i].text(0.97,0.95,('').join([r'$p = $',f'{np.round(pvalue[0],4)}']),color='k',fontsize=11,
                    va='top',ha='right', transform=axs[i].transAxes)
                axs[i].annotate(f'True \nCluster',xy=(cluster['cluster_stat'],axs[i].get_ylim()[1]),xycoords='data',
                                xytext=(7,-45),color='red',
                        fontsize=10,textcoords='offset points')
            else:
                axs[i].text(0.97,0.95,('').join([r'$p = $',f'{np.round(pvalue[0],4)}']),color='k',fontsize=11,
                    va='top',ha='right', transform=axs[i].transAxes)
                axs[i].annotate(f'True \nCluster',xy=(cluster['cluster_stat'],axs[i].get_ylim()[1]),xycoords='data',
                                xytext=(7,-45),color='red',fontsize=10,textcoords='offset points')
        else:
            axs[i].set_title(f' Null Distribution\n Negative Cluster')

            if len(pvalue) == 1:
                axs[i].text(0.2,0.95,('').join([r'$p = $',f'{np.round(pvalue[0],4)}']),color='k',fontsize=11,
                    va='top',ha='right', transform=axs[i].transAxes)
                axs[i].annotate(f'True \nCluster',xy=(cluster['cluster_stat'],axs[i].get_ylim()[1]),xycoords='data',
                                xytext=(-40,-25),color='red',
                        fontsize=10,textcoords='offset points')
            else:
                axs[i].text(0.2,0.95,('').join([r'$p = $',f'{np.round(pvalue[1],4)}']),color='k',fontsize=11,
                    va='top',ha='right', transform=axs[i].transAxes)
                axs[i].annotate(f'True \nCluster',xy=(cluster['cluster_stat'],axs[i].get_ylim()[1]),xycoords='data',
                                xytext=(-40,-25),color='red',
                        fontsize=10,textcoords='offset points')
    plt.tight_layout()
    plt.close(fig) 
    return fig


def plot_neurocluster_results(betas,cluster_test, max_cluster_data, null_cluster_distribution, tstats, tstat_threshold,cluster_pvalue):
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
    cluster_plot = plot_clusters(tstats,cluster_test)
    max_cluster_plot= plot_max_clusters(cluster_test,tstats)
    null_distribution_plot = plot_null_distribution(null_cluster_distribution, max_cluster_data,cluster_pvalue)

    return beta_plot,tstat_plot,cluster_plot,max_cluster_plot,null_distribution_plot

