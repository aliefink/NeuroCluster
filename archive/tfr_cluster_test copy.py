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


class TFR_Cluster_Test(object):
  
  """Class for time-frequency resolved cluster permutation testing.

  Parameters
  ----------
  tfr_data : array, float, shape(freqs,times,n_epochs)
      Single electrode tfr data 
  predictor_data : DataFrame, shape(n_epochs,n_regressors)
      Single subject behav data 

  """

  def __init__(self, tfr_data, predictor_data, permute_var, ch_name, **kwargs):

    """Constructor for the Environment class
    This function runs every time we create an instance of the class Environment
    To learn more about how constructors work: https://www.udacity.com/blog/2021/11/__init__-in-python-an-overview.html"""

    # "self" is just a convention that binds the attributes and methods of a class with the arguments of a given instance

    self.tfr_data       = tfr_data  # single electrode tfr data
    self.predictor_data = predictor_data # single subject behav data
    self.tfr_dims       = self.tfr_data.shape[1:] # dims of single electrode tfr data (n_freqs x n_times)
    self.permute_var    = permute_var # variable to permute in regression model
    self.ch_name        = ch_name # channel name for single electrode tfr data


  def tfr_regression(self):

    # preallocate np arrays for betas + tstats
    tfr_betas = np.zeros((self.tfr_dims))
    tfr_tstats = np.zeros((self.tfr_dims))

    iter_tup = self.expand_tfr_indices()

    # Prepare arguments for the permutation function`
    start = time.time()
    # either precompute pixel_args before passing to parallel, or run all together in loop. - check later!! 
    pixel_args = [self.make_pixel_df(self.tfr_data[:,freq_idx,time_idx]) for freq_idx,time_idx in iter_tup]
    
    # run pixel permutations in parallel 
    expanded_results = Parallel(n_jobs=-1, verbose=12)(
                      delayed(self.pixel_multi_regression)(args)
                        for args in pixel_args)      
    
    # expanded_results is a list of tuples (beta,tstat) for every pixel 
    for count,(freq_idx,time_idx) in enumerate(iter_tup):
      tfr_betas[freq_idx,time_idx] = expanded_results[count][0]
      tfr_tstats[freq_idx,time_idx] = expanded_results[count][1]
    
    print(f'pixel regression time: ', '{:.2f}'.format(time.time()-start))


    return tfr_betas, tfr_tstats


  def expand_tfr_indices(self):
    iter_tup = list(map(tuple,np.unravel_index(np.dstack(([*np.indices(self.tfr_dims)])),np.product(self.tfr_dims))[0].
                        reshape(np.product(np.dstack(([*np.indices(self.tfr_dims)])).shape[:2]),-1)))
        
    return iter_tup
  
  def make_pixel_df(self,epoch_data):
    """
    Make pixel-level (frequency x timepoint) dataframe. 
    """
    return self.predictor_data.assign(pow=epoch_data)


  def pixel_multi_regression(self,pixel_df):

      """
      Run pixel-wise OLS regression model to extraxct beta coefficient and t-statistic.

      Args:
      - pixel_df (pandas df): regression dataframe (insert details here)

      Returns:
      - beta_coeff (numpy array): Beta coefficient(s) from pixel-wise regression.
      - tstat_pixel (numpy array): Observed t-statistic(s) from pixel-wise regression.
      """

      # formula should be in form 'col_name + col_name' if col is categorical then should be 'C(col_name)'
      formula    = 'pow ~ 1 + ' + (' + ').join(['C('+col+')' if pd.api.types.is_categorical_dtype(pixel_df[col])
                                  else col for col in pixel_df.columns[~pixel_df.columns.isin(['pow'])].tolist()])
      
      pixel_model = smf.ols(formula,pixel_df,missing='drop').fit() # fit regression model

      return (pixel_model.params[self.permute_var],pixel_model.tvalues[self.permute_var])
  
  def compute_tcritical(self,tails=2, alternative ='two-sided',alpha=0.05):
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
      
      # Calculate degrees of freedom
      deg_free = float(len(self.predictor_data)-len(self.predictor_data.columns)-tails)

      return (t.ppf(1-alpha/tails,deg_free) if alternative != 'less' else np.negative(t.ppf(1-alpha/tails,deg_free)))


  def threshold_tfr_tstat(self,tfr_tstats,alternative='two-sided'):
      if alternative == 'two-sided':
          return [(tfr_tstats>self.compute_tcritical()).astype(int), (tfr_tstats<np.negative(self.compute_tcritical()).astype(int))]
    
      elif alternative == 'greater':
          return [(tfr_tstats>self.compute_tcritical(tails=1,alternative='greater')).astype(int)]

      else: #alternative = less
          return [(tfr_tstats<self.compute_tcritical(tails=1,alternative='less')).astype(int)]

  def max_tfr_cluster(self,tfr_tstats,alternative='two-sided',clust_struct=np.ones(shape=(3,3)),output='all'):
      
      max_cluster_data = []
      for binary_mat in self.threshold_tfr_tstat(tfr_tstats,alternative = alternative):
          cluster_label, num_clusters = label(binary_mat,clust_struct)
          # use argmax to find index of largest absolute value of cluster t statistic sums 
          max_label = np.argmax([np.abs(np.sum(tfr_tstats[cluster_label==i+1])) for i in range(num_clusters)])
          # use max_label index to compute cluster tstat sum (without absolute value)
          max_clust_stat = np.sum(tfr_tstats[cluster_label==max_label+1])
          clust_freqs, clust_times = [(np.min(arr),np.max(arr)) for arr in np.where(cluster_label == max_label)]

          if output == 'all':
              max_cluster_data.append({'clust_stat':max_clust_stat,'freq_idx':clust_freqs,'time_idx':clust_times})
          elif output == 'cluster_stat':
              max_cluster_data.append({'clust_stat':max_clust_stat})
          elif output == 'freq_time':
              max_cluster_data.append({'freq_idx':clust_freqs,'time_idx':clust_times})

      return max_cluster_data
  

  def permute_predictor(self):
      """
      Permute predictor variable for permutation test.
      """

      self.predictor_data[self.permute_var] = np.random.permutation(self.predictor_data[self.permute_var].values)

      return self
  


