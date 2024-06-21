from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    

#def prepare_regressor_df(zscore,dropnans,data_types,regressors_to_include):


def prepare_anat_dic(roi, file_path):
    '''
    Prepare a dictionary mapping each channel to its anatomical region.

    Parameters
    ----------
    roi : str
        Region of interest (e.g., 'ofc', 'hippocampus', etc.).

    file_path : str
        Path to the file containing the anatomical information. 

    Returns
    -------
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


def run_permutation_test(cluster_test, num_permutations=1000):

    '''
    Run a permutation test for a given cluster test.

    Parameters
    ----------
    cluster_test : TFR_Cluster_Test
        Instance of TFR_Cluster_Test class.

    num_permutations : int
        Number of permutations to run. Default is 1000.

    Returns
    -------
    list
        List of cluster statistics for each permutation.

    '''
    # Initialize list to store cluster statistics
    perm_cluster_list = []

    # Run permutation test
    for _ in range(num_permutations):
        permuted_cluster_test = cluster_test.permute_predictor() # permute predictor variable
        _, tstats = permuted_cluster_test.tfr_multireg() # run regression on permuted data
        perm_cluster_list.append(cluster_test.max_tfr_cluster(tstats,output='cluster_stat')) # get cluster statistics - only cluster stat (positive and negative) for each permutation
        del permuted_cluster_test, tstats # delete objects to free up memory

    return perm_cluster_list


def run_permutation_test(cluster_test, num_permutations):
    def permutation_test(cluster_test):
        permuted_cluster_test = cluster_test.permute_predictor()  # Permute predictor variable
        _, tstats = permuted_cluster_test.tfr_multireg()  # Run regression on permuted data
        cluster_stat = cluster_test.max_tfr_cluster(tstats, output='cluster_stat')  # Get cluster statistics
        del permuted_cluster_test, tstats  # Delete objects to free up memory
        return cluster_stat

    # Run permutation tests in parallel
    perm_cluster_list = Parallel(n_jobs=-1, verbose=12)(
        delayed(permutation_test)(cluster_test) for _ in range(num_permutations)
    )
    return perm_cluster_list

  