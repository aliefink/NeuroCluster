import numpy as np
import pandas as pd
from scipy.ndimage import label 
from scipy.stats import  t
from joblib import Parallel, delayed
import statsmodels.api as sm 
from scipy.ndimage import label 
import statsmodels.formula.api as smf
import tqdm
from scipy.ndimage import label 
import time
import inspect



class TFR_Cluster_Test(object):
    '''
    Single-electrode neurophysiology object class to identify time-frequency resolved neural activity correlates of complex behavioral variables using non-parametric 
    cluster-based permutation testing.   

    Attributes
    ----------
    tfr_data       : (np.array) Single electrode tfr data matrix. Array of floats (n_epochs,n_freqs,n_times). 
    tfr_dims       : (tuple) Frequency and time dimensions of tfr_data. Tuple of integers (n_freq,n_times). 
    ch_name        : (str) Unique electrode identification label. String of characters.
    predictor_data : (pd.DataFrame) Regressors from task behavior with continuous, discreet, or categorical data. DataFrame of (rows=n_epochs,columns=n_regressors). 
    permute_var    : (str) Column label for primary regressor of interest.
      
    Methods
    ----------
    **To-do: fill in methods info
    '''

    def __init__(self, tfr_data, predictor_data, permute_var, ch_name, **kwargs):
        '''
        Args:
        - tfr_data       : (np.array) Single electrode tfr data matrix. Array of floats (n_epochs,n_freqs,n_times). 
        - predictor_data : (pd.DataFrame) Task-based regressor data with dtypes continuous/discreet(int64/float) or categorical(pd.Categorical). DataFrame of (n_epochs,n_regressors).
        - permute_var    : (str) Column label for primary regressor of interest. Array of 1d integers or floats (n_epochs,).
        - ch_name        : (str) Unique electrode identification label. String of characters.  
        - **kwargs       : (optional) alternative, alpha, cluster_shape
        '''

        self.tfr_data        = tfr_data  # single electrode tfr data
        self.tfr_dims        = self.tfr_data.shape[1:] # time-frequency dims of electrode data (n_freqs x n_times)
        self.ch_name         = ch_name # channel name for single electrode tfr data
        self.predictor_data  = predictor_data # single subject behav data
        self.permute_var     = permute_var # variable to permute in regression model 
        self.ols_dmatrix     = pd.get_dummies(predictor_data,drop_first=True) # converts only categorical variables into one dummy coded vector
        self.permute_var_idx = np.where(self.ols_dmatrix.columns  == permute_var)[0][0] # column index of regressor of interest in dummy coded dmatrix
        

    def tfr_regression(self):
        '''
        Performs univariate or multivariate OLS regression across tfr matrix for all pixel-level time-frequency power data and task-based predictor variables. Regressions are parallelized across pixels.

        Returns:
        - tfr_betas  : (np.array) Matrix of beta coefficients for predictor of interest for each pixel regression. Array of (n_freqs,n_times). 
        - tfr_tstats : (np.array) Matrix of t-statistics from coefficient estimates for predictor of interest for each pixel regression. Array of (n_freqs,n_times). 
        '''
        # run pixel permutations in parallel    
        expanded_results = Parallel(n_jobs=-1, verbose=5)(delayed(self.pixel_regression)(pixel_data)
                                                           for pixel_data in np.resize(self.tfr_data,(self.tfr_data.shape[0],np.prod(self.tfr_dims))).T) 
        
        tfr_betas,tfr_tstats = list(zip(*expanded_results))
        
        return np.resize(np.array(tfr_betas),(self.tfr_data.shape[1],self.tfr_data.shape[2])), np.resize(np.array(tfr_tstats),
                                                                                                         (self.tfr_data.shape[1],self.tfr_data.shape[2]))
        # # Prepare arguments for parallelization`using tfr matrix indices converted to list of tuples (freq x power)
        # pixel_args = [self.make_pixel_df(self.tfr_data[:,freq_idx,time_idx]) for freq_idx,time_idx in self.expand_tfr_indices()]
        
        # # run pixel permutations in parallel 
        # expanded_results = Parallel(n_jobs=-1, verbose=5)(
        #                 delayed(self.pixel_regression)(args)
        #                     for args in pixel_args)      

        # # preallocate np arrays for betas + tstats
        # tfr_betas  = np.zeros((self.tfr_dims))
        # tfr_tstats = np.zeros((self.tfr_dims))

        # # expanded_results is a list of tuples (beta,tstat) for every pixel 
        # for count,(freq_idx,time_idx) in enumerate(self.expand_tfr_indices()):
        #     tfr_betas[freq_idx,time_idx]  = expanded_results[count][0]
        #     tfr_tstats[freq_idx,time_idx] = expanded_results[count][1]
        
        # return tfr_betas, tfr_tstats

    def pixel_regression(self,pixel_data,permuted=False):
        '''        
        Fit pixel-wise univariate or multivariate OLS regression model and extract beta coefficient and t-statistic for predictor of interest (self.permute_var). 

        Args:
        - pixel_df   : (pd.DataFrame) Pixel-level regression dataframe with power epochs data and behavioral regressors. DataFrame of (n_epochs, n_regressors+1). 
                                      Regressor column data must be continuous(dtype=float), discrete(dtype=int), or categorical(dtype=pd.Categorical). 
        
        Returns:
        - pixel_beta : (np.array) Beta coefficient for predictor of interest from pixel-wise regression. Array of 1d float (1,)
        - pixel_tval : (np.array) Observed t-statistic for predictor of interest from pixel-wise regression. Array of 1d float (1,)
        '''


        # # formula should be in form 'col_name + col_name' if col is categorical then should be 'C(col_name)'  
        # formula    = '+ '.join(['pow ~ 1 ',(' + ').join([''.join(['C(',col,')']) if pd.api.types.is_categorical_dtype(pixel_df[col])
        #                     else col for col in pixel_df.columns[~pixel_df.columns.isin(['pow'])].tolist()])])
        
        # pixel_model = smf.ols(formula,pixel_df,missing='drop').fit()

        if permuted: ###### make clear that this permanently updates ols_dmatrix data!!!!
            self.ols_dmatrix[self.permute_var] = np.random.permutation(self.ols_dmatrix[self.permute_var].values)
            pixel_model = sm.OLS(pixel_data,sm.add_constant(self.ols_dmatrix.to_numpy()),missing='drop').fit()
        
        else: 
            pixel_model = sm.OLS(pixel_data,sm.add_constant(self.ols_dmatrix.to_numpy()),missing='drop').fit()

        return (pixel_model.params[self.permute_var_idx + 1],pixel_model.tvalues[self.permute_var_idx + 1])

    def max_tfr_cluster(self,tfr_tstats,alternative='two-sided',output='all',clust_struct=np.ones(shape=(3,3))):

        '''
        Identify time-frequency clusters of neural activity that are significantly correlated with the predictor of interest (self.permute_var). Clusters are identified 
        from neighboring pixel regression t-statistics for the predictor of interest that exceed the tcritical threshold from the alternate hypothesis. 

        Args:
        - tfr_tstats       : (np.array) Pixel regression tstatistic from coefficient estimates for predictor of interest. Array of floats (n_freqs,n_times). 
        - alternative      : (str) Alternate hypothesis for t-test. Must be 'two-sided','greater', or 'less'. Default is 'two-sided'. 
        - output           : (str) Output format for max cluster statistics. Must be 'all', 'cluster_stat', or 'freq_time'. Default is 'all'.
        - clust_struct     : (np.array) Binary matrix to specify cluster structure for scipy.ndimage.label. Array of (3,3). 
                                        Default is np.ones.shape(3,3), to allow diagonal cluster pixels (Not the scipy.ndimage.label default).
                                        https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html

        Returns:
        - max_cluster_data : (list) Beta coefficient for predictor of interest for each pixel regression. List (len=2 if 'two-sided') of dict(s).
                                    If output = 'all', return dictionary of maximum cluster statistic ('cluster_stat' : sum of pixel t-statistics), 
                                    cluster frequency indices ('freq_idx':(freq_x,freq_y)), and cluster time indices ('time_idx':(time_x,time_y)). 
                                    If output = 'cluster_stat', return only [{cluster_stat}]. If output = 'freq_time', return only {freq_idx,time_idx}
                                    ** If no clusters are found, max_cluster_data contains list of empty dictionaries
        *** add docstring for expanded output
        '''
        
        max_cluster_data = []
        # Create binary matrix from tfr_tstats by thresholding pixel t-statistics by tcritical. (1 = pixel t-statistic exceeded tcritical threshold)
        for binary_mat in self.threshold_tfr_tstat(tfr_tstats,alternative):

            # test whether there are any pixels above tcritical threshold
            if np.sum(binary_mat) != 0: 
                # Find clusters of pixels with t-statistics exceeding tcritical
                cluster_label, num_clusters = label(binary_mat,clust_struct)
                # use argmax to find index of largest absolute value of cluster t statistic sums 
                max_label = np.argmax([np.abs(np.sum(tfr_tstats[cluster_label==i+1])) for i in range(num_clusters)])+1
                # use max_label index to compute cluster tstat sum (without absolute value)
                max_clust_stat = np.sum(tfr_tstats[cluster_label==max_label])
                # find 2D indices of minimum/maximum cluster frequencies and times 
                clust_freqs, clust_times = [(np.min(arr),np.max(arr)) for arr in np.where(cluster_label == max_label)]

                if output == 'all':
                    max_cluster_data.append({'cluster_stat':max_clust_stat,'freq_idx':clust_freqs,'time_idx':clust_times})
                elif output == 'cluster_stat':
                    max_cluster_data.append({'cluster_stat':max_clust_stat})
                elif output == 'freq_time':
                    max_cluster_data.append({'freq_idx':clust_freqs,'time_idx':clust_times})
                elif output == 'expanded':
                    max_cluster_data.append({'cluster_stat':max_clust_stat,'freq_idx':clust_freqs,'time_idx':clust_times,
                                            'max_label':max_label,'all_clusters':cluster_label})
            
            else: # if there is no cluster, return max_cluster_data with empty dictionaries
                if output == 'all':
                    max_cluster_data.append({'cluster_stat':0,'freq_idx':0,'time_idx':0})
                elif output == 'cluster_stat':
                    max_cluster_data.append({'cluster_stat':0})
                elif output == 'freq_time':
                    max_cluster_data.append({'freq_idx':0,'time_idx':0})
                elif output == 'expanded':
                    max_cluster_data.append({'cluster_stat':0,'freq_idx':0,'time_idx':0,'max_label':0,'all_clusters':0})            
        
        return max_cluster_data

    def compute_tcritical(self,alternative ='two-sided',alpha=0.05):
        '''
        Calculate critical t-values for regression model.
        
        Args:
        - alternative : (str) Alternate hypothesis for t-test. Must be 'two-sided','greater', or 'less'. Default is 'two-sided'.
        - alpha       : (float) Significance level. Default is 0.05.

        Returns:
        - tcritical   : (float) Critical t-statistic for hypothesis test. Positive value when alternative = 'two-sided' or 'greater'. Negative when alternative = 'less'. 
        '''

        # Set number of tails for t-tests using 'alternative' parameter input string. 
            # tails = 2 if alternative = 'two-sided' (two tailed hypothesis test)
            # tails = 1 if alternative = 'greater' or 'less' (one tailed hypothesis test)
        tails = len(alternative.split('-')) 

        # Calculate degrees of freedom (N-k-1) 
        deg_free = float(len(self.ols_dmatrix)-len(self.ols_dmatrix.columns)-1) #### predictor data must only include regressors in columns

        # Return tcritical from t-distribution. Significance level is alpha/2 for two tailed hypothesis tests (alternative = 'two-sided').
        return (t.ppf(1-(alpha/tails),deg_free) if alternative != 'less' else np.negative(t.ppf(1-(alpha/tails),deg_free)))

    def threshold_tfr_tstat(self,tfr_tstats,alternative='two-sided'):
        '''
        Threshold tfr t-statistic matrix using tcritical.

        Args:
        - tfr_tstats  : (np.array) Matrix of t-statistics from pixel-wise regressions. Array of floats (n_freqs, n_times). 
        - alternative : (str) Type of hypothesis test for t-distribution. Must be 'two-sided', 'greater', 'less'. Default is 'two-sided'.

        Returns:
        - binary_mat  : (np.array) Binary matrix results of pixel-wise t-tests. Pixel = 1 when tstatistic > tcritical, else pixel = 0. List of array(s) (n_freqs, n_times).
        '''

        if alternative == 'two-sided': 
            return [(tfr_tstats>self.compute_tcritical()).astype(int), (tfr_tstats<np.negative(self.compute_tcritical())).astype(int)]

        elif alternative == 'greater':
            return [(tfr_tstats>self.compute_tcritical(tails=1,alternative='greater')).astype(int)]

        elif alternative == 'less':
            return [(tfr_tstats<self.compute_tcritical(tails=1,alternative='less')).astype(int)] 
        else: 
            raise ValueError('Alternative hypothesis must be two-sided, greater, or less not {alternative}')
    
    # def expand_tfr_indices(self):
    #     '''
    #     Create list of tfr pixel indices for parallelized tfr_regression.

    #     Returns:
    #     - iter_tup : (list) Time-frequency indices for all pixels in tfr_data. List of tuples [(freq_x_index,freq_y_index),(time_x_index,time_y_index)]        
    #     '''

    #     return list(map(tuple,np.unravel_index(np.dstack(([*np.indices(self.tfr_dims)])),np.product(self.tfr_dims)
    #                         )[0].reshape(np.product(np.dstack(([*np.indices(self.tfr_dims)])).shape[:2]),-1)))

    # def make_pixel_df(self,epoch_data,permuted=False):
    #     '''
    #     Format input data for pixel regression.  input data. Make pixel-level (frequency x timepoint) dataframe. Add tfr power data for single pixel to predictor_df. 

    #     Args:
    #     - epoch_data : (str) Alternate hypothesis for t-test. Must be 'two-sided','greater', or'less'. Default is 'two-sided'. Array of 1d integers or floats (n_epochs,).
        
    #     Returns:
    #     - pixel_df   : (pd.DataFrame) Pixel regression input dataframe containing power epochs and task-based behavioral regressor data (dtype=int/float/pd.Categorical). 
    #                                   DataFrame of (n_epochs, n_regressors+1). 
        
    #     ##### to-do add docstring info for permuted kwargs
    #     '''
        
    #     if permuted: ###### make clear that this permanently updates predictor data!!!!
    #         self.predictor_data[self.permute_var] = np.random.permutation(self.predictor_data[self.permute_var].values)
    #         return self.predictor_data.assign(pow=epoch_data)
    #     else: 
    #         return self.predictor_data.assign(pow=epoch_data) 

###### UNTESTED PERMUTATION FUNCTIONS!!!

    def compute_null_cluster_stats(self,num_permutations=None):

        '''
        To-do add docstring & test

        Args:
        - null_cluster_distribution : (list) 
        '''

        #### for every permutation:
            # permute predictor of interest, then make pixel df 
            # run tfr regression & extract permutation t stats 
            # find max cluster statistics for permutation  

        null_tstat = Parallel(n_jobs=-1, verbose=5)(delayed
                                            (self.permuted_tfr_regression(output='cluster_stat'))(self.permuted_tfr_regression() for n in range(num_permutations)))
        return null_cluster_distribution


    def permuted_tfr_regression(self,n_jobs=-1,verbose=0):
        '''
        Run tfr regression for single permutation
        
        Args:
        - perm_tstats : (np.array)

        '''

        permuted_results = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(self.pixel_regression)(pixel_data,permuted=True)
                                                            for pixel_data in np.resize(self.tfr_data,(self.tfr_data.shape[0],np.prod(self.tfr_dims))).T) 
        
        _,perm_tstats = list(zip(*permuted_results))
        
        return np.resize(np.array(perm_tstats), (self.tfr_data.shape[1],self.tfr_data.shape[2]))

        # iter_tup = self.expand_tfr_indices()

        # # either precompute pixel_args before passing to parallel, or run all together in loop. - check later!! 
        # perm_args = [self.make_pixel_df(self.tfr_data[:,freq_idx,time_idx],permuted=True) for freq_idx,time_idx in iter_tup]

        # # Run regression on permuted data + extract tstats only

        # # run pixel permutations in parallel 
        # permuted_results = Parallel(n_jobs=-1, verbose=5)(
        #                 delayed(self.pixel_regression)(args)
        #                     for args in perm_args)      
        
        # # preallocate np arrays for betas + tstats
        # perm_tstats = np.zeros((self.tfr_dims))

        # # expanded_results is a list of tuples (beta,tstat) for every pixel 
        # for count,(freq_idx,time_idx) in enumerate(iter_tup):
        #     perm_tstats[freq_idx,time_idx] = permuted_results[count][1]
        
        # yield perm_tstats

    # def cluster_significance_test(self, null_distribution,max_cluster_stat,alpha=0.05,alternative='two-sided'):
        '''
        Compute non-param etric pvalue from cluster permutation data 
        
        Args:
         - alpha (float): Significance level. Default is 0.05.
         
        '''
        # null_df = pd.concat([pd.DataFrame(dict,index=[0]) for dict in null_distribution]).reset_index(drop=True)
        # null_df['sign'] = ['positive' if row.cluster_stat > 0 else 'negative' for row in null_df.iterrows()]
        # for sign in null_df.sign.unique(): #### one loop option 
        # for cluster in max_cluster_stat: ### another loop option
        #     null_max_clusters = null_df.cluster_stat[null_df.sign == sign]

        
    #     return cluster_pvalue