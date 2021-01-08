'''
Created on Mar 28, 2012

@author: jan
'''
import random
import numpy as np
from scipy.stats import norm, expon, gamma
from scipy.spatial.distance import squareform


def gaussian_influence(mu, width):
    ''' creates a 2D-function for gaussian influence sphere'''
    return lambda x, y: np.exp(-width * ((x - mu[0]) ** 2 + (y - mu[1]) ** 2))

def correlated_samples(cov, num_sampl, marginal_dist):
    ''' creates correlated samples with a gaussian copula 
        
        cov: covariance matrix (copula)
        num_sampl: numbers of sample to be drawn
        marginal_dist: marginal distribution samples are drawn from
    '''

    # Create Gaussian Copula
    dependence = np.random.multivariate_normal([0] * cov.shape[0], cov, num_sampl)
    dependence_dist = norm()
    uniform_dependence = dependence_dist.cdf(dependence)

    #Transform marginals 
    dependend_samples = marginal_dist.ppf(uniform_dependence)
    return dependend_samples

def group_covmtx(rho_intra, rho_inter, num_groups, num_objects):
    ''' create a covarince matrix with groups
    
    create covariance matrix for num_groups*num_objects variables 
    in each group are num_objects with a covariance of rho_intra. 
    objects between groups have a covariance of rho_intra 
    '''

    intra_mtx_size = int((num_objects ** 2 - num_objects) / 2)
    intra_cov = 1 - squareform([1 - rho_intra] * intra_mtx_size)

    cov = rho_inter * np.ones((num_groups * num_objects, num_groups * num_objects))
    for group_num in range(num_groups):
        cov[group_num * num_objects:(group_num + 1) * num_objects,
            group_num * num_objects:(group_num + 1) * num_objects] = intra_cov
    return cov

def adjusted_gamma(mean, var):
    ''' create a gamma distribution with defined mean and variance '''

    scale = var / mean
    shape = mean / scale
    if shape > 1:
        print('!!! Warning !!! - shape parameter: ', str(shape))
    return gamma(shape, scale=scale)

def crosscor(a1, a2):
    '''calculate crosscorrelation between two matrices'''

    num_var = a1.shape[0]
    return np.corrcoef(np.vstack((a1, a2)))[num_var:, :num_var]

class Dataset():
    """Surrogate dataset of sources and their mixed observations
    
    Keyword arguments: 
    param : dictionary of parameters
            shape: tuple with spatial extent of observed area in pixel
            gridpoints: number of sources in one dimension
            width: width of spatial influence
            latents: number of sources
            covgroups: number of correlated source groups
            cov: correlation of activation within a source group
            mean: mean source activation
            var: source activation variance
            no_samples: number of independent observations (stimuli)
            act_time: model time cours of activation
            noisevar: sigma of gaussian pixel noise
    """


    def __init__(self, param):

        # create spatial sources
        num_grid = param.get('gridpoints', 9)
        pixel = np.indices(param['shape'])
        p_dist = param['shape'][0] / num_grid
        self.points = np.indices((num_grid, num_grid)) * p_dist + p_dist
        self.points = list(zip(self.points[0].flatten(), self.points[1].flatten()))
        random.shuffle(self.points)
        components = [gaussian_influence(mu, param['width'])(pixel[0], pixel[1])
                  for mu in self.points[:param['latents']]]
        self.spt_sources = np.array([i.flatten() for i in components])

        # generate activation timcourses
        covgroups = param.get('covgroups', 4)
        self.cov = group_covmtx(param['cov'], 0.1, covgroups, int(param['latents'] / covgroups))
        marginal_dist = adjusted_gamma(param['mean'], param['var'])
        self.activ_pre = correlated_samples(self.cov, param['no_samples'],
                                             marginal_dist).T
        self.activ_pre[np.isnan(self.activ_pre)] = 0
        # fold with single stim timecourse
        if param['act_time']:
            self.activation = np.vstack([np.outer(i, param['act_time']).flatten()
                                    for i in self.activ_pre]).T
        self.observed_raw = np.dot(self.activation, self.spt_sources)

        # add noise
        noise = param['noisevar'] * np.random.randn(*self.observed_raw.shape)
        self.observed = self.observed_raw.copy() + noise

    def cor2source(self, estimator):
        """match sources to their best estimate and calc correlation
        
        each source is matched to the estimator it's exhibits the highest spatial
        correlation
        
        Parameters
        ----------
        estimator: ImageAnalysisComponents.TimeSeries
        
        Returns
        ------- 
        matchid: numpy.array
            i-th entry contains index of estimator matched to i-th source
        st_cor: numpy.array
            i-th entry contains temporal correlation of estimator for i-th source
        sp_cor: numpy.array
            i-th entry contains spatial correlation of estimator for i-th source
        """

        # temporal corellation with all sources
        tmp_cor = crosscor(self.activation.T, estimator._series.T)
        # spatial correlations with all sources
        sp_cor = crosscor(self.spt_sources, estimator.base._series)
        matchid = np.nanargmax(np.abs(sp_cor), 0)

        # temporal correlation at best spatial corralation
        st_cor = np.abs(tmp_cor[matchid, range(self.spt_sources.shape[0])])
        return matchid, st_cor, sp_cor

    def mse2source(self, estimator, local=0):
        """match sources to their best estimate and calc mean squared error (MSE)
        
        each source is matched to the estimator it's exhibits the lowest MSE
        
        Parameters
        ----------
        estimator: ImageAnalysisComponents.TimeSeries
        
        Returns
        ------- 
        mse: numpy.array
            i-th entry contains MSE of estimator matched to i-th source
        """

        best_mse = []
        for source_ind in range(self.activation.shape[1]):

            mf_pixelpart = estimator.base._series
            data_pixelpart = self.spt_sources[source_ind]
            if local:
                mask = self.spt_sources[source_ind] > local
                mf_pixelpart = mf_pixelpart[:, mask]
                data_pixelpart = data_pixelpart[mask]

            source = np.outer(self.activation[:, source_ind], data_pixelpart)
            source_norm = np.linalg.norm(source)
            mse_components = []
            for e_ind in range(estimator.num_objects):
                estimate = np.outer(estimator._series[:, e_ind], mf_pixelpart[e_ind])
                mse = np.linalg.norm(source - estimate) / source_norm
                mse_components.append(mse)
            best_mse.append(np.min(mse_components))
        return best_mse
