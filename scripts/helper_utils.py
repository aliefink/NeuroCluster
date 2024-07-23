from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def subset_channels(data, ch_names):
    '''
    Extracts a subset of channels from a MNE Raw object.

    Parameters
    ----------
    data : mne.io.Raw
        MNE Raw object.

    ch_names : list of str
        List of channel names to extract.

    Returns
    -------
    np.ndarray
        4D array of shape (n_epochs, n_channels, n_times) containing the data
        from the selected channels.

    '''
    if ch_names == 'all':
        return data._data[:, :, :, :]
    else:
        ch_idx = [data.ch_names.index(ch) for ch in ch_names]
        return data._data[:, ch_idx, :, :]
    

def prepare_regressor_df(power_epochs):
    '''
    Prepare a DataFrame containing the behavioral variables.

    Parameters
    ----------
    power_epochs : mne.Epochs
        MNE Epochs object containing the power data.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the behavioral variables.

    '''
    
    beh_df = []

    beh_variables = [col for col in power_epochs.metadata if col not in power_epochs.ch_names] # get behavioral variables
    beh_df = power_epochs.metadata[beh_variables] # extract behavioral variables from metadata

    # present user with list of beahvioral variables and have them decide if they want to keep them or not 
    for col in beh_df.columns:
        keep = input(f'Would you like to keep {col}? (yes or no): ')
        if keep == 'no':
            beh_df.drop(col, axis=1, inplace=True)

    # present user with list of behavioral variables and have them mark as categorical or continuous
    for col in beh_df.columns:
            data_type = input(f'Please specify data type for {col} (category or float64).')
            beh_df[col] = beh_df[col].astype(data_type)

    # present user with list of behavioral variables that are not marked as category and ask if they want to z-score them
    for col in beh_df.columns:
        if beh_df[col].dtype != 'category':
            z_score = input(f'Would you like to z-score {col}? (yes or no): ')
            if z_score == 'yes':
                beh_df[col] = (beh_df[col] - beh_df[col].mean()) / beh_df[col].std()

    return beh_df

def prepare_anat_dic(roi, file_path):
    '''
    Prepare a dictionary mapping each channel to its anatomical region.

    Args:
    roi : str
        Region of interest (e.g., 'ofc', 'hippocampus', etc.).

    file_path : str
        Path to the file containing the anatomical information. 

    Returns:
    dict
        Dictionary mapping each channel to its anatomical region.

    '''
    # check if file name is csv, if not raise error
    if file_path.split('.')[-1] != 'csv':
        raise ValueError('Anat file must be a csv file.')
    
    # read in anatomical info
    anat_df = pd.read_csv(file_path)

    # subset rows for specified ROI
    roi_info_df = anat_df[anat_df['roi'].isin(roi)]

    # get unique subj_ids for ROI
    roi_subj_ids = roi_info_df.subj_id.unique().tolist()

    # create dict with subj_id as key and elecs as values
    anat_dic = {f'{subj_id}':roi_info_df.reref_ch_names[roi_info_df.subj_id == subj_id].unique().tolist() for subj_id in roi_subj_ids}

    return anat_dic


# def run_permutation_test(cluster_test, num_permutations):
#     def permutation_test(cluster_test):
#         permuted_cluster_test = cluster_test.permute_predictor()  # Permute predictor variable
#         _, tstats = permuted_cluster_test.tfr_multireg()  # Run regression on permuted data
#         cluster_stat = cluster_test.max_tfr_cluster(tstats, output='cluster_stat')  # Get cluster statistics
#         del permuted_cluster_test, tstats  # Delete objects to free up memory
#         return cluster_stat

#     # Run permutation tests in parallel
#     perm_cluster_list = Parallel(n_jobs=-1, verbose=12)(
#         delayed(permutation_test)(cluster_test) for _ in range(num_permutations)
#     )
#     return perm_cluster_list

def create_directory(directory_path):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def save_plot_to_pdf(fig, directory, filename):
    """Save a plot to the specified directory with the given filename."""
    filepath = os.path.join(directory, filename)
    fig.savefig(filepath, dpi=300,bbox_inches='tight')
    plt.close(fig)  # Close the figure to avoid display and memory issues
  