# -*- coding: utf-8 -*-
"""Contains factory capable of constructing decision makers (for M-step)."""

import torch

from aux import flatten_first_two_dims, retrieve_param, retrieve_param_opt, format_value, is_valid
from aux_time import timing
from numerical_optimization import optimal_h_numerically_ty, optimal_h_numerically_ty_scipy


def kill_gradients(f):
    """Makes sure that gradients are not propagated through the output tensor."""
    def wrapper(*args, **kwargs): 
        result = f(*args, **kwargs)
        result = torch.tensor(result.cpu().detach().numpy(), requires_grad=False)         
        return result
    wrapper.__name__ = f.__name__
    return wrapper 
    

class HOptimizerFactory:
    """Creates methods that allow for calculation of optimal h by 
       first, sampling y and then, 
       optimizing a requested approximation to the utility-dependent term (or gain)."""

    def __init__(self, **kwargs):        
        self.H_NSAMPLES_UTILITY_TERM_THETA = retrieve_param("H_NSAMPLES_UTILITY_TERM_THETA", kwargs)
        self.H_NSAMPLES_UTILITY_TERM_Y = retrieve_param("H_NSAMPLES_UTILITY_TERM_Y", kwargs)
        
        self.H_NUMERICAL_MAX_NITER = retrieve_param_opt("H_NUMERICAL_MAX_NITER", kwargs, 10000)
        self.H_NUMERICAL_MAX_NITER_TOL = retrieve_param_opt("H_NUMERICAL_MAX_NITER_TOL", kwargs, 0.0001)
        self.H_NUMERICAL_MAX_NITER_TOL_GOAL = retrieve_param_opt("H_NUMERICAL_MAX_NITER_TOL_GOAL", kwargs, 1e-10)
        self.H_NUMERICAL_LR = retrieve_param_opt("H_NUMERICAL_LR", kwargs, 0.1)
        self.H_NUMERICAL_START_FROM_PREVIOUS = retrieve_param_opt("H_NUMERICAL_START_FROM_PREVIOUS", kwargs, False)

        self.sample_predictive_y0 = retrieve_param("sample_predictive_y0", kwargs) #default sampler of y
      
        
        print("[HOptimizerFactory] Configuration: %s" % 
                " ".join("%s=%s" % (k, format_value(v)) for k, v in vars(self).items() 
                                    if k.upper()==k or k.startswith("sample")) )

        self.optimal_h_bayes_estimator = kwargs.get("optimal_h_bayes_estimator", None)
        if kwargs.get("H_NUMERICAL_OPT_SCIPY", False):
            print("[HOptimizerFactory] Choosing SciPy numerical optimization.")
            self.optimal_h_numerically = optimal_h_numerically_ty_scipy
        else:
            print("[HOptimizerFactory] Choosing PyTorch numerical optimization.")
            self.optimal_h_numerically = optimal_h_numerically_ty


    def create(self, FIND_H, u, optimal_h_bayes_estimator=None, utility_aggregator=None):
        """
            Returns a requested method that calculates the optimal h.            

            Args:
                FIND_H: The name of the optimizer: 
                    bayes - use Bayes estimator / num - optimize numerically gain / 
                    num-util - optimize numerically a selected approximation to the utility-dependent term.
                u: Utility function. 
                optimal_h_bayes_estimator: A method to calculate optimal h analytically (=Bayes estimator).
                utility_aggregator: An approximation to the utility-dependent term;
                                    reduces a matrix of utilities into a single value, 
                                    for example gain. Used when optimizing numerically.
        """
        if optimal_h_bayes_estimator is not None: 
            self.optimal_h_bayes_estimator = optimal_h_bayes_estimator


        @kill_gradients
        @timing
        def optimize_h_with_bayes_estimator(*args):
            """Assumes that optimal_h calculates gain-optimal value analytically."""
            ys = flatten_first_two_dims( self.sample_predictive_y0(*args, 
                      nsamples_theta=self.H_NSAMPLES_UTILITY_TERM_THETA, 
                      nsamples_y=self.H_NSAMPLES_UTILITY_TERM_Y) )
            h = self.optimal_h_bayes_estimator(ys)

            if optimize_h_with_bayes_estimator.counter<3 and not is_valid(h): 
                print("WARNING: your optimal_h_bayes_estimator=%s returns invalid decisions! Are we good here?" % 
                        self.optimal_h_bayes_estimator.__name__)
            optimize_h_with_bayes_estimator.counter += 1
            return h
        optimize_h_with_bayes_estimator.counter = 0


        @kill_gradients
        @timing
        def optimize_h_for_gain_numerically(*args):
            """Finds gain-optimal value using numerical optimization."""
            ys =  self.sample_predictive_y0(*args, 
                          nsamples_theta=self.H_NSAMPLES_UTILITY_TERM_THETA, 
                          nsamples_y=self.H_NSAMPLES_UTILITY_TERM_Y) 
            h = self.optimal_h_numerically(ys, u, #utility_aggregator=gain,
                    start=optimize_h_for_gain_numerically.start if self.H_NUMERICAL_START_FROM_PREVIOUS else None,
                    max_niter=self.H_NUMERICAL_MAX_NITER, tol=self.H_NUMERICAL_MAX_NITER_TOL, 
                    tol_goal=self.H_NUMERICAL_MAX_NITER_TOL_GOAL, lr=self.H_NUMERICAL_LR)
            optimize_h_for_gain_numerically.start = h #next time start from the last optimum
            return h
        optimize_h_for_gain_numerically.start = None            


        @kill_gradients
        @timing
        def optimize_h_for_utilterm_numerically(*args):
            """Finds utility_aggregator-optimal value using numerical optimization."""
            if utility_aggregator is None: raise ValueError("utility_aggregator must be fixed!")            

            ys = self.sample_predictive_y0(*args, 
                          nsamples_theta=self.H_NSAMPLES_UTILITY_TERM_THETA, 
                          nsamples_y=self.H_NSAMPLES_UTILITY_TERM_Y)
            h = self.optimal_h_numerically(ys, u, utility_aggregator=utility_aggregator,
                    start = optimize_h_for_utilterm_numerically.start if self.H_NUMERICAL_START_FROM_PREVIOUS else None,
                    max_niter=self.H_NUMERICAL_MAX_NITER, tol=self.H_NUMERICAL_MAX_NITER_TOL, 
                    tol_goal=self.H_NUMERICAL_MAX_NITER_TOL_GOAL, lr=self.H_NUMERICAL_LR)
            optimize_h_for_utilterm_numerically.start = h #next time start from the last optimum
            return h
        optimize_h_for_utilterm_numerically.start = None

         
        H_OPTIMIZERS = { "analytical":  optimize_h_with_bayes_estimator, 
                         "bayes":       optimize_h_with_bayes_estimator,
                         "num":         optimize_h_for_gain_numerically, 
                         "num-gain":    optimize_h_for_gain_numerically, 
                         "num-util":    optimize_h_for_utilterm_numerically,
                        }
        FIND_H = FIND_H.lower().replace("sgd", "num").replace("utility_term", "util")
        if FIND_H not in H_OPTIMIZERS: 
            raise KeyError("[HOptimizerFactory] Unknown name (%s)! Try: %s" % (FIND_H, list(H_OPTIMIZERS.keys())) )
        return  H_OPTIMIZERS[FIND_H]   


