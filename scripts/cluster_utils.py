from joblib import Parallel, delayed
import numpy as np 
from scipy.ndimage import label 
from scipy.stats import zscore, t
import statsmodels.api as sm 
import statsmodels.formula.api as smf
import operator

'''
Cluster utils V2 contains the functions from TFR_Cluster_Test class code as stand-alone operations + the previous version of cluster_utils functions from V1
'''


def max_tfr_cluster(tfr_tstats,predictor_data,alternative='two-sided',clust_struct=np.ones(shape=(3,3))):
    
    max_cluster_data = []
    for binary_mat in threshold_tfr_tstat(tfr_tstats,predictor_data,alternative ='two-sided'):
        cluster_label, num_clusters = label(binary_mat,clust_struct)
        # use argmax to find index of largest absolute value of cluster t statistic sums 
        max_label = np.argmax(np.abs([np.sum(tfr_tstats[cluster_label==i+1]) for i in range(num_clusters)]))
        # use max_label index to compute cluster tstat sum (without absolute value)
        max_clust_stat = np.sum(tfr_tstats[cluster_label==max_label+1])
        clust_freqs, clust_times = [(np.min(arr),np.max(arr)) for arr in np.where(cluster_label == max_label)]
        
        max_cluster_data.append({'clust_stat':max_clust_stat,'freq_idx':clust_freqs,'time_idx':clust_times})        

    return max_cluster_data

def make_pixel_df(epoch_data,predictor_data):
    # predictor data should be predictor df of single subj with only predictors included in model as columns
    return predictor_data.assign(pow=epoch_data)

def pixel_multi_regression(pixel_df,permute_var):

    """
    Run pixel-wise OLS regression model to extraxct beta coefficient and t-statistic. 

    Args:
    - y (numpy array): single pixel, single subject power x epochs 
    - regressor_data (numpy array): Feature matrix as pandas df with columns for each variable for multiple regression

    Returns:
    - beta_coeff (numpy array): Beta coefficient(s) from pixel-wise regression.
    - tstat_pixel (numpy array): Observed t-statistic(s) from pixel-wise regression.
    """

    # formula should be in form 'col_name + col_name' if col is categorical then should be 'C(col_name)'  
    formula    = ' + '.join(['pow ~ 1 ',' + '.join( [''.join(['C(',col,')']) if pd.api.types.is_categorical_dtype(pixel_df[col])
                                else col for col in pixel_df.columns[~pixel_df.columns.isin(['pow'])].tolist()])])
    
    pixel_model = smf.ols(formula,pixel_df,missing='drop').fit() # fit regression model

    return (pixel_model.params[permute_var],pixel_model.tvalues[permute_var])

def tfr_multireg(tfr_data,predictor_data,tfr_dims,permute_var):

    # preallocate np arrays for betas + tstats
    tfr_betas = np.zeros((tfr_dims))
    tfr_tstats = np.zeros((tfr_dims))

    iter_tup = expand_tfr_indices(tfr_dims)


    # Prepare arguments for the permutation function`
    start = time.time()
    # either precompute pixel_args before passing to parallel, or run all together in loop. - check later!! 
    pixel_args = [make_pixel_df(tfr_data[:,freq_idx,time_idx],predictor_data) for freq_idx,time_idx in iter_tup]
    
    # run pixel permutations in parallel 
    expanded_results = Parallel(n_jobs=-1, verbose=12)(
                        delayed(pixel_multi_regression)(args,permute_var)
                        for args in pixel_args)  
    
    print(f'pixel regression time: ', '{:.2f}'.format(time.time()-start))

    for count,(freq_idx,time_idx) in enumerate(iter_tup):
        tfr_betas[freq_idx,time_idx] = expanded_results[count][0]
        tfr_tstats[freq_idx,time_idx] = expanded_results[count][1]

    return tfr_betas,tfr_tstats
  
def expand_tfr_indices(tfr_dims):
    iter_tup = list(map(tuple,np.unravel_index(np.dstack(([*np.indices(tfr_dims)])),np.product(tfr_dims))[0].
                    reshape(np.product(np.dstack(([*np.indices(tfr_dims)])).shape[:2]),-1)))

    return iter_tup


def compute_tcritical(predictor_data,tails=2, alternative = 'two-sided',alpha=0.05):
    """
    Calculate critical t-values for regression model.

    Args:
    - predictor_dims (tuple): Dimensions of data matrix. Tuple of (n_samples, n_predictors). 
    - tails (int): Number of tails for t-distribution. Default is 2. Options are 1 or 2.
    - alternative (str): Type of test. Default is 'two-sided'. Options are 'two-sided', 'greater', 'less'.
    - alpha (float): Significance level. Default is 0.05.

    Returns:
    - tcritical (float): Critical t-value.
    """
    
    deg_free = float(len(predictor_data)-len(predictor_data.columns)-tails)
    
    return t.ppf(1-alpha/tails,deg_free) if alternative != 'less' else np.negative(t.ppf(1-alpha/tails,deg_free))

def threshold_tfr_tstat(tfr_tstats,predictor_data,alternative ='two-sided'):

    # tcrit = self.compute_tcritical()
    # tfr_tstat_binary = (tfr_tstats>tcrit).astype(int)
    if alternative == 'two-sided':
        return ((tfr_tstats>compute_tcritical(predictor_data)), (tfr_tstats<np.negative(compute_tcritical(predictor_data)).astype(int)))
    elif alternative == 'greater':
        return (tfr_tstats>compute_tcritical(predictor_data)).astype(int)
    else: #alternative = less
        return (tfr_tstats<compute_tcritical(predictor_data)).astype(int)

def pixel_regression(X,y):

    """
    Run pixel-wise OLS regression model to extraxct beta coefficient and t-statistic. 

    Args:
    - y (numpy array): single pixel, single subject power x epochs 
    - regressor_data (numpy array): Feature matrix as pandas df with columns for each variable for multiple regression

    Returns:
    - beta_coeff (numpy array): Beta coefficient(s) from pixel-wise regression.
    - tstat_pixel (numpy array): Observed t-statistic(s) from pixel-wise regression.
    """

    # format outcome + predictor vars 
    pixel_model = sm.OLS(y, sm.add_constant(X)).fit() # fit regression model
    # extract beta coefficient(s) excluding intercept + t statistic(s) - do not save into memory

    return pixel_model.params[1:], pixel_model.tvalues[1:] 

    
def tfr_cluster_test(elec_data, predictor_data, tcritical, zscore=False, clust_def=None,output='all'):
    '''
    Perform OLS regression on each pixel (freq x time) of electrode data with task-based regressors.

    Args:
    elec_data (numpy array): Electrode data matrix (num_epochs x num_freq x num_time).
    predictor_data (numpy array): Task-based regressor data (num_epochs x num_predictors).
    tcritical (float): Critical t-value for regression model. Either one or two values depending on tails. 
    zscore (function): Flag to zscore regressor data. Either True or False. Default is False. 
    

    Returns:
    elec_cluster_data (dict): Dictionary of electrode-level cluster statistics. The dictionary contains the following keys:
    - results_betas (numpy array): Matrix of beta coefficients for each pixel (freq x time).
    - tstat_observed (numpy array): Matrix of t-statistics for each pixel (freq x time).
    - sig_tstat_observed_pos (numpy array): Binary matrix for positive t-statistics (freq x time).
    - sig_tstat_observed_neg (numpy array): Binary matrix for negative t-statistics (freq x time).
    - pos_clust_data (dict): Dictionary of positive cluster statistics.
    - neg_clust_data (dict): Dictionary of negative cluster statistics.

    '''
    
    num_freq = elec_data.shape[0] # num frequencies 
    num_time = elec_data.shape[1] # num time points 
   
    # pixel-wise univariate regression matrices 
    results_betas  = np.zeros((num_freq, num_time)) # pixel beta coefficients 
    tstat_observed = np.zeros((num_freq, num_time)) # pixel t statistics

    for f in range(num_freq): 
        freq_data = elec_data[f,:] # matrix is num_epochs x num_time
        
        for t in range(num_time):
            time_data = freq_data[t] # matrix is num_epochs x 1 

            # vectorize handling of NaNs in both neural (due to artifact rejection etc.) and behavioral (due to participant lapse etc.) data (remove epochs with NaNs in either data type)
            # valid_mask = ~np.isnan(time_data) & ~np.isnan(predictor_data)
            # time_data = time_data[valid_mask]
            # predictor_data_t = predictor_data[valid_mask]
            predictor_data_t = predictor_data.copy()

            # zscore regressor vector if flagged
            if zscore==True:
                predictor_data_t = zscore(predictor_data_t) # defined in step 1

            # Compute predictor beta and tstat from univariate regression
            beta_coefficient, tstat_pixel = pixel_regression(time_data, predictor_data_t)
            # Save pixel stats 
            results_betas[f,t] = beta_coefficient # save beta coeff 
            tstat_observed[f,t] = tstat_pixel # save tstat
            
    # update binary matrix for whether pixel t statistic is greater than tcritical (split pos/neg)
    sig_tstat_observed_pos = (tstat_observed>tcritical).astype(int) # 1 = GREATER THAN +T CRIT
    sig_tstat_observed_neg = (tstat_observed<np.negative(tcritical)).astype(int) # 1 = LESS THAN -T CRIT
    
                
    # Electrode-level cluster statistics 
    # extract maximum positive cluster 
    pos_clust_data = get_max_cluster(sig_tstat_observed_pos, tstat_observed)
    # extract maximum negative cluster 
    neg_clust_data = get_max_cluster(sig_tstat_observed_neg, tstat_observed)

    if output == 'all':
        elec_cluster_data = {'results_betas':results_betas,
                            'tstat_observed':tstat_observed, # matrix of t statistics single electrode
                            'sig_tstat_observed_pos':sig_tstat_observed_pos, # binary pos matrix for cluster image labels
                            'sig_tstat_observed_neg':sig_tstat_observed_neg, # binary neg matrix for cluster image labels
                            'pos_clust_data':pos_clust_data,
                            'neg_clust_data':neg_clust_data}
        
                        
        return elec_cluster_data
    else:
        return (pos_clust_data['max_cluster_tstat'],neg_clust_data['max_cluster_tstat'])



def get_max_cluster(binary_tstat_mat, elec_tstat_mat, clust_struct=np.ones(shape=(3,3),dtype=int)):  
    """
    Find maximum cluster (positive or negative) for single electrode 

    Args:
    - binary_tstat_mat: binary matrix for indices of pixel tstat >/< t_crit (np.array freq x time)
    - elec_tstat_mat:   matrix of pixel tstats from regression (np.array freq x time)
    - clust_struct:     specify structure for scipy.ndimage.label fn (scipy default excludes diagonals - see 
                        https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html
    Returns:
    - max_clust_data: dictionary with max cluster statistic (sum tstat) & cluster location indices (freq x time)
    """
    
    # find clusters in binary tstat matrix 
    cluster_label, num_clusters = label(binary_tstat_mat,structure=clust_struct)

    # take the sum of t stat in each identified cluster
    cluster_sum_tstat = [np.sum(elec_tstat_mat[cluster_label==i+1]) for i in range(num_clusters)] #add 1 to iterator! 

    # find the largest cluster
    max_cluster  = np.max(np.abs(cluster_sum_tstat)) # max of abs value for negative cluster stats 
    max_label    = np.where(np.abs(cluster_sum_tstat) == max_cluster)[0] + 1 #clust nums start at 1
    # get freq x time indices for largest cluster 
    max_cluster_indices = np.where(cluster_label==max_label[0]) # tuple of indices (freq,time) for max cluster 

    # save max cluster data into dict 
    max_clust_data = {'max_cluster_tstat':max_cluster, # max cluster statistic (sum tstat)
                      'max_cluster_indices':max_cluster_indices} # max cluster indices (freq x time) 
    
    return max_clust_data


def permuted_tfr_cluster_test(real_elec_data,predictor_data,tcritical,poi=None,output='tstat'):
    '''
    Perform OLS regression on each pixel (freq x time) of permuted data within electrode. Labels for predictor of interest are shuffled across epochs.


    Args:
    ral_elec_data (numpy array): Electrode data matrix (num_epochs x num_freq x num_time).
    predictor_data (numpy array): Task-based regressor data (num_epochs x num_predictors).
    tcritical (float): Critical t-value for regression model. Either one or two values depending on tails. 
    poi (int): Predictor of interest in multivariate regression. Index of predictor of interest in predictor_data. Default is None - only needed if multiple predictors are passed in predictor_data matrix.
    output (str): Default = 'tstat'. Argument to reduce output type of tfr_cluster_test to only max positive/negative cluster tstats
                    output= 'all' it will return permuted_cluster_data dict
                  

    Returns:
    if output = 'tstat' (default):
    permuted_cluster_data (tuple): Tuple of maximum and minimum cluster t stastics for all permutations. 
    
    if output = 'all':
    permuted_cluster_data (dict): Dictionary of electrode-level cluster statistics. The dictionary contains the following keys:
    - results_betas (numpy array): Matrix of beta coefficients for each pixel (freq x time).
    - tstat_observed (numpy array): Matrix of t-statistics for each pixel (freq x time).
    - sig_tstat_observed_pos (numpy array): Binary matrix for positive t-statistics (freq x time).
    - sig_tstat_observed_neg (numpy array): Binary matrix for negative t-statistics (freq x time).
    - pos_clust_data (dict): Dictionary of positive cluster statistics.
    - neg_clust_data (dict): Dictionary of negative cluster statistics.

    '''
    # np.random.permutation uses first axis to permute - should be epochs. 
    if poi is None:
        permuted_predictor_data = np.random.permutation(predictor_data) # for univariate regression, predictor data is a one-dimensional array (num_epochs,)
    else:
        permuted_predictor_data = np.random.permutation(predictor_data[:,poi]) # shuffle predictor of interest labels across epochs. Let's shuffle here so that the function is more robust to multivariate regression. 
    
    # run pixel-wise tfr cluster test 
    permuted_cluster_data = tfr_cluster_test(real_elec_data,permuted_predictor_data,tcritical,output=output)

    return permuted_cluster_data


def tfr_cluster_test_v2(elec_data, predictor_data, tcritical, zscore=False, clust_def=None):
    '''
    IN DEV: Function to perform cluster-based permutation testing on TFR data. Vectorized for speed. 

    Parameters
    ----------
    elec_data : np.array
        3D array of neural data (num_epochs, num_freq, num_time)
    predictor_data : np.array
        1D array of behavioral data (num_epochs)
    tcritical : float
        critical t value for the given alpha level and degrees of freedom
    zscore : bool
        whether to zscore the predictor data
    clust_def : dict
        dictionary containing cluster definition parameters

    Returns
    -------
    elec_cluster_data : dict
        dictionary containing results of the cluster-based permutation test
    '''

    # make reg_data a 2D array for the regression
    predictor_data_expanded = np.expand_dims(predictor_data, axis=1)

    # make elec_data a 2D array for the regression (num_epochs, num_time * num_freq)
    elec_data_expanded = elec_data.reshape(elec_data.shape[0], elec_data.shape[1]*elec_data.shape[2])

    # drop epochs with NaNs from neural data (marked as NaN during artifact rejection: )
    predictor_data_expanded = predictor_data_expanded[~np.isnan(elec_data_expanded).any(axis=1)]
    elec_data_expanded = elec_data_expanded[~np.isnan(elec_data_expanded).any(axis=1)]

    # drop epochs with NaNs from predictor data (missing behavioral data due to participant lapse etc. )
    elec_data_expanded = elec_data_expanded[~np.isnan(predictor_data_expanded).any(axis=1)]
    predictor_data_expanded = predictor_data_expanded[~np.isnan(predictor_data_expanded).any(axis=1)]

    # perform pixel-wise regression for each column in the expanded neural data
    results_betas = np.zeros((elec_data_expanded.shape[1], predictor_data_expanded.shape[1]))
    tstat_observed = np.zeros((elec_data_expanded.shape[1], predictor_data_expanded.shape[1]))

    # zscore regressor vector if flagged
    if zscore==True:
        predictor_data_expanded = zscore(predictor_data_expanded) # defined in step 1

    for i in range(elec_data_expanded.shape[1]):
        beta_coefficient, tstat_pixel = pixel_regression(elec_data_expanded[:,i], predictor_data_expanded)
        results_betas[i] = beta_coefficient
        tstat_observed[i] = tstat_pixel

    # reshape the results to match the original data shape
    results_betas = results_betas.reshape(elec_data.shape[1], elec_data.shape[2])
    tstat_observed = tstat_observed.reshape(elec_data.shape[1], elec_data.shape[2])

    # update binary matrix for whether pixel t statistic is greater than tcritical (split pos/neg)
    sig_tstat_observed_pos = (tstat_observed>tcritical).astype(int) # 1 = GREATER THAN +T CRIT
    sig_tstat_observed_neg = (tstat_observed<np.negative(tcritical)).astype(int) # 1 = LESS THAN -T CRIT

    # Electrode-level cluster statistics 
    # extract maximum positive cluster 
    pos_clust_data = get_max_cluster(sig_tstat_observed_pos, tstat_observed)
    # extract maximum negative cluster 
    neg_clust_data = get_max_cluster(sig_tstat_observed_neg, tstat_observed)

    elec_cluster_data = {'results_betas':results_betas,
                            'tstat_observed':tstat_observed, # matrix of t statistics single electrode
                            'sig_tstat_observed_pos':sig_tstat_observed_pos, # binary pos matrix for cluster image labels
                            'sig_tstat_observed_neg':sig_tstat_observed_neg, # binary neg matrix for cluster image labels
                            'pos_clust_data':pos_clust_data,
                            'neg_clust_data':neg_clust_data}

    return elec_cluster_data


