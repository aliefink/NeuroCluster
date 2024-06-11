import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import mne 

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