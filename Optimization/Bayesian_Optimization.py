from typing import Tuple
import numpy as np
import random
from scipy.optimize import minimize
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor

class BayesOpt:
    def __init__(self, f, mu, eps, pbounds, _X_init, _Y_init, iters=30):
        """
        Bayes Opt Class:
            f: Function to optimize
            mu: nu value for the Matern Kernel
            eps: added noise
            pbounds: Bounds for variables to explore
        """
        #Record Previously Sampled vals
        self.sampled = dict()
        
        self._X_sample = _X_init
        self._Y_sample = _Y_init
        self.pbounds = pbounds
        self.p_dict = dict()#TODO MAKE LIST OF PBOUNDS A DICT SO WE CAN PASS TUPLES INTO THE SAMPLING FUNCTION
        #Init regressor
        self.mu = mu #for kernal initialization

        self.regressor = GaussianProcessRegressor(kernel=Matern(nu=mu, length_scale=0.0001), alpha=eps, normalize_y=True, 
                                                    n_restarts_optimizer=5)
        self.running = True
        self.eps = eps
        self.f = f
        self.curr_iter = 0
        self.iters=iters
    
    #def fit_GP()
    
    def sample_next_points(self, acquisition, gpr, xi, num_restarts=25):
        """
        Fit and optimize the acquisition function to allow us to find the new Sample points
        """
        best_x = None
        best_acquisition_value = 1
        n_params = self._X_sample.shape[1]
        def min_obj(x): #Define our objective for lbfgs-b optimizer
            return -acquisition(x.reshape(-1, n_params), self._X_sample, gpr, xi)
        sample_from_bounds = np.random.uniform(self.pbounds[:, 0], self.pbounds[:, 1], size=(num_restarts, n_params))
        for x0 in sample_from_bounds:
            res = minimize(fun=min_obj,
                           x0=x0,
                           bounds=self.pbounds,
                           method='L-BFGS-B')
            if res.fun < best_acquisition_value:
                best_acquisition_value = res.fun
                best_x = res.x
        return best_x

    def run(self, acquisition, xi):
        """
        run bayes opt step for iters
        xi: Exploration, exploitation tradeoff
        """
        self._X_sample = self._X_sample
        self._Y_sample = self._Y_sample
        gpr = self.regressor
        gpr.n_features_in_ = self._X_sample.shape[1]
        gpr.fit(self._X_sample, self._Y_sample)
        next_X = self.sample_next_points(acquisition, gpr, xi)
        self._X_sample = np.vstack((self._X_sample, next_X))
        self.curr_iter = self.curr_iter+1
        if self.curr_iter >= self.iters:
            self.running = False

        return next_X
        
    def append_y(self, yval):
        """
        use this after evaluating model
        """
        self._Y_sample = np.vstack((self._Y_sample, yval))