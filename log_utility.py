# -*- coding: utf-8 -*-
"""Contains factories capable of constructing various approximations of the utility-dependent term and 
   various wrappers allowing to optimize utility-dependent term w.r.t. h using sampling from posteriors."""

import torch

from aux import sel_with_prefix, sel_without_prefix, flatten_first_two_dims
from losses import optimal_h_numerically_ty, optimal_h_numerically, \
                   optimal_h_numerically_ty_scipy, optimal_h_numerically_scipy 



def _retrieve_param(name, kwargs):
    if name not in kwargs or kwargs[name] is None:
        raise NameError("Param <%s> is not set!" % name)
    return kwargs[name]


def _retrieve_param_opt(name, kwargs, default=None):
    if name not in kwargs or kwargs[name] is None: 
        return default
    return kwargs[name]  
    

def _format_value(v):
    if hasattr(v, "__name__"): return v.__name__
    if hasattr(v, "shape"): return v.shape
    return str(v)[:15]       
           


###############################################################################


# When utilities are close to 0, logs or dividing by them leads to NaN-s
# adding eps increase the numerical stability
EPS_NAIVE = 1e-16
EPS_JENSEN = 1e-16
EPS_JENSEN_IS = 1e-16
EPS_TAYLOR1 = 1e-16
EPS_TAYLOR2 = 1e-8


class UtilityAggregatorFactory:
    """ Constructs methods that take a matrix of utilities (utilities for samples of y)
        and calculate an approximation to the utility-dependent term.
        In practice, the methods aggregate multiple utilities into single value.
    """

    def create(self, UTILITY_AGGREGATOR):
        """ Returns the requested approximation to the utility-dependent term.
            Args:
               UTILITY_AGGREGATOR: Selects an approximation to the utility-dependent term: 'vi'/'naive'/'jensen'/'linearized'.
        """

        def utility_term_naive(us, data_mask):
            """
                Args:
                    us: Utility matrix where dim 0 is over output samples y, 
                        dim 1 is over latent variables theta and remaining dims are over input.
                    data_mask: A mask over input values that selects which data points should be included.
            """            
            point_utility_term = (us.mean(0) + EPS_NAIVE).log().mean(0)   
            assert point_utility_term.shape==data_mask.shape, "%s=datashape!=mask.shape=%s" % (point_utility_term.shape,data_mask.shape)    
            return torch.masked_select(point_utility_term, data_mask).sum()


        def utility_term_Jensen(us, data_mask):
            #print("[utility_term_Jensen] us=%s" % us.shape)
            us = flatten_first_two_dims(us)
            point_utility_term = (us + EPS_JENSEN).log().mean(0)
            assert point_utility_term.shape==data_mask.shape, "%s=datashape!=mask.shape=%s" % (point_utility_term.shape,data_mask.shape)
            return torch.masked_select(point_utility_term, data_mask).sum()


        def _logExpected_Taylor(us):
            mu_utility_term, mu, sigma =  (us + EPS_TAYLOR1).log().mean(0), us.mean(0), us.std(0)     
            return mu_utility_term + sigma**2 / (2 * mu**2 + EPS_TAYLOR2)   


        def utility_term_Taylor(us, data_mask):
            point_utility_term = _logExpected_Taylor(us).mean(0)
            assert point_utility_term.shape==data_mask.shape, "%s=datashape!=mask.shape=%s" % (point_utility_term.shape,data_mask.shape)
            return torch.masked_select(point_utility_term, data_mask).sum()


        def utility_term_linearized(us, data_mask):
            point_utility_term = us.mean(0).mean(0)        
            assert point_utility_term.shape==data_mask.shape, "%s=datashape!=mask.shape=%s" % (point_utility_term.shape,data_mask.shape) 
            return torch.masked_select(point_utility_term, data_mask).sum()


        def utility_term_Jensen_IS(us, weights, data_mask): 
            """ Calculates a lower bound to the original utility-bound using weights.
                Args:
                    us: Utility matrix where dim 0 is over output samples y and remaining dims are over input.
                    weights: A matrix matching us and containing weights for respective y-s. 
                             Weights are assumed to be normalized.
                    data_mask: A mask over input values that selects which data points are included.
            """
            point_utility_term = ( (us + EPS_JENSEN_IS).log() * weights ).sum(0)
            assert point_utility_term.shape==data_mask.shape, "%s=datashape!=mask.shape=%s" % (point_utility_term.shape,data_mask.shape)
            return torch.masked_select(point_utility_term, data_mask).sum()


        def utility_term_linearized_IS(us, weights, data_mask):     
            point_utility_term = (us * weights).sum(0)
            assert point_utility_term.shape==data_mask.shape, "%s=datashape!=mask.shape=%s" % (point_utility_term.shape,data_mask.shape)
            return torch.masked_select(point_utility_term, data_mask).sum()


        def utility_term_vi(us, data_mask):
            return us.sum()*0.0


        UTILITY_AGGREGATORS = {"vi": utility_term_vi, "naive": utility_term_naive, "linearized": utility_term_linearized,
                               "jensen": utility_term_Jensen, "taylor": utility_term_Taylor,
                               "jensen-is": utility_term_Jensen_IS, "linearized-is": utility_term_linearized_IS}
        UTILITY_AGGREGATOR = dict(enumerate(["vi", "naive", "linearized", "jensen", "taylor", "jensen-is", "linearized-is"])).get(UTILITY_AGGREGATOR, UTILITY_AGGREGATOR).lower() #maps: 0->vi, ..., 6->linearized-is
        if UTILITY_AGGREGATOR not in UTILITY_AGGREGATORS: 
            raise KeyError("[UtilityAggregatorFactory] Unknown name (%s)! Try: %s" % (UTILITY_AGGREGATOR, list(UTILITY_AGGREGATORS.keys())) )
        return UTILITY_AGGREGATORS[UTILITY_AGGREGATOR]
        

###############################################################################


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
       first, sampling y and then, optimizing a requested approximation to the utility-dependent term."""

    def __init__(self, **kwargs):

        params = globals()
        
        self.H_NSAMPLES_UTILITY_TERM_THETA = _retrieve_param("H_NSAMPLES_UTILITY_TERM_THETA", kwargs)
        self.H_NSAMPLES_UTILITY_TERM_Y = _retrieve_param("H_NSAMPLES_UTILITY_TERM_Y", kwargs)
        
        self.H_NUMERICAL_MAX_NITER = _retrieve_param_opt("H_NUMERICAL_MAX_NITER", kwargs, 10000)
        self.H_NUMERICAL_MAX_NITER_TOL = _retrieve_param_opt("H_NUMERICAL_MAX_NITER_TOL", kwargs, 0.0001)
        self.H_NUMERICAL_MAX_NITER_TOL_GOAL = _retrieve_param_opt("H_NUMERICAL_MAX_NITER_TOL_GOAL", kwargs, 1e-10)
        self.H_NUMERICAL_LR = _retrieve_param_opt("H_NUMERICAL_LR", kwargs, 0.1)
        self.H_NUMERICAL_START_FROM_PREVIOUS = _retrieve_param_opt("H_NUMERICAL_START_FROM_PREVIOUS", kwargs, False)

        self.sample_predictive_y0 = _retrieve_param("sample_predictive_y0", kwargs) #default sampler of y
        self.sample_predictive_y_IS = _retrieve_param_opt("sample_predictive_y_IS", kwargs) #optinal sampler with weights
        
        print("[HOptimizerFactory] Configuration: %s" % 
                " ".join("%s=%s" % (k, _format_value(v)) for k, v in vars(self).items() 
                                    if k.upper()==k or k.startswith("sample")) )  
      
        self.optimal_h_analytical = kwargs.get("optimal_h", None)
        if kwargs.get("H_NUMERICAL_OPT_SCIPY", False):
            print("[HOptimizerFactory] Using SciPy numerical optimization.")
            self.optimal_h_num_weighted = optimal_h_numerically_scipy
            self.optimal_h_num = optimal_h_numerically_ty_scipy
        else:
            print("[HOptimizerFactory] Using PyTorch numerical optimization.")
            self.optimal_h_num_weighted = optimal_h_numerically
            self.optimal_h_num = optimal_h_numerically_ty


    def create(self, FIND_H, u, optimal_h=None, utility_aggregator=None):
        """
            Returns a requested method that calculates the optimal h.            

            Args:
                FIND_H: The name of the optimizer: 
                    analytical - use Bayes estimator / num - optimize numerically gain / 
                    num-util - optimize numerically a selected approximation to the utility-dependent term.
                u: Utility function. 
                optimal_h: A method to calculate optimal h analytically (Bayes estimator).
                utility_aggregator: An approximation to the utility-dependent term;
                                    reduces a matrix of utilities into a single value, 
                                    for example gain.
        """

        if optimal_h is not None:   self.optimal_h_analytical = optimal_h

        @kill_gradients        
        def optimize_h_analytically(*args, **kwargs):
            """Assumes that optimal_h calculates gain-optimal value analytically."""
            nsamples_theta, nsamples_y = kwargs.get("nsamples_theta", self.H_NSAMPLES_UTILITY_TERM_THETA), kwargs.get("nsamples_y", self.H_NSAMPLES_UTILITY_TERM_Y)
            
            ys = flatten_first_two_dims( self.sample_predictive_y0(*args, nsamples_theta=nsamples_theta, nsamples_y=nsamples_y) )
            h = self.optimal_h_analytical(ys)
            return h


        @kill_gradients        
        def optimize_h_numerically(*args, **kwargs):
            """Finds gain-optimal value using numerical optimization."""
            nsamples_theta, nsamples_y = kwargs.get("nsamples_theta", self.H_NSAMPLES_UTILITY_TERM_THETA), kwargs.get("nsamples_y", self.H_NSAMPLES_UTILITY_TERM_Y)
            
            ys =  self.sample_predictive_y0(*args, nsamples_theta=nsamples_theta, nsamples_y=nsamples_y) 
            h = self.optimal_h_num(ys, u, 
                    start=optimize_h_numerically.start if self.H_NUMERICAL_START_FROM_PREVIOUS else None,
                    max_niter=self.H_NUMERICAL_MAX_NITER, tol=self.H_NUMERICAL_MAX_NITER_TOL, 
                    tol_goal=self.H_NUMERICAL_MAX_NITER_TOL_GOAL, lr=self.H_NUMERICAL_LR)
            optimize_h_numerically.start = h #next time start from the last optimum
            return h
        optimize_h_numerically.start = None            


        @kill_gradients        
        def optimize_h_utility_numerically(*args, **kwargs):
            """Finds utility_aggregator-optimal value using numerical optimization."""
            nsamples_theta, nsamples_y = kwargs.get("nsamples_theta", self.H_NSAMPLES_UTILITY_TERM_THETA), kwargs.get("nsamples_y", self.H_NSAMPLES_UTILITY_TERM_Y)
            if utility_aggregator is None: raise ValueError("utility_aggregator must be fixed!")            

            ys = self.sample_predictive_y0(*args, nsamples_theta=nsamples_theta, nsamples_y=nsamples_y)
            h = self.optimal_h_num(ys, u, utility_aggregator=utility_aggregator,
                    start = optimize_h_utility_numerically.start if self.H_NUMERICAL_START_FROM_PREVIOUS else None,
                    max_niter=self.H_NUMERICAL_MAX_NITER, tol=self.H_NUMERICAL_MAX_NITER_TOL, 
                    tol_goal=self.H_NUMERICAL_MAX_NITER_TOL_GOAL, lr=self.H_NUMERICAL_LR)
            optimize_h_utility_numerically.start = h #next time start from the last optimum
            return h
        optimize_h_utility_numerically.start = None


        @kill_gradients        
        def optimize_h_numerically_IS(*args, **kwargs):
            """Finds gain-optimal value using numerical optimization on weighted samples y."""
            nsamples_theta, nsamples_y = kwargs.get("nsamples_theta", self.H_NSAMPLES_UTILITY_TERM_THETA), kwargs.get("nsamples_y", self.H_NSAMPLES_UTILITY_TERM_Y)    
            
            ys, weights = self.sample_predictive_y_IS(*args, nsamples_theta=nsamples_theta, nsamples_y=nsamples_y) 
            h = self.optimal_h_num_weighted(ys, u, weights=weights, 
                    start = optimize_h_numerically_IS.start if self.H_NUMERICAL_START_FROM_PREVIOUS else None,
                    max_niter=self.H_NUMERICAL_MAX_NITER, tol=self.H_NUMERICAL_MAX_NITER_TOL, 
                    tol_goal=self.H_NUMERICAL_MAX_NITER_TOL_GOAL, lr=self.H_NUMERICAL_LR)
            optimize_h_numerically_IS.start = h #next time start from the last optimum                   
            return h
        optimize_h_numerically_IS.start = None            


        @kill_gradients        
        def optimize_h_utility_numerically_IS(*args, **kwargs):
            """Finds utility_aggregator-optimal value using numerical optimization on weighted samples y."""
            nsamples_theta, nsamples_y = kwargs.get("nsamples_theta", self.H_NSAMPLES_UTILITY_TERM_THETA), kwargs.get("nsamples_y", self.H_NSAMPLES_UTILITY_TERM_Y)  
            if utility_aggregator is None: raise ValueError("utility_aggregator must be fixed!")            
              
            ys, weights = self.sample_predictive_y_IS(*args, nsamples_theta=nsamples_theta, nsamples_y=nsamples_y) 
            h = self.optimal_h_num_weighted(ys, u, weights=weights, utility_aggregator=utility_aggregator,
                    start = optimize_h_utility_numerically_IS.start if self.H_NUMERICAL_START_FROM_PREVIOUS else None,
                    max_niter=self.H_NUMERICAL_MAX_NITER, tol=self.H_NUMERICAL_MAX_NITER_TOL, 
                    tol_goal=self.H_NUMERICAL_MAX_NITER_TOL_GOAL, lr=self.H_NUMERICAL_LR)
            optimize_h_utility_numerically_IS.start = h #next time start from the last optimum                   
            return h
        optimize_h_utility_numerically_IS.start = None            


        H_OPTIMIZERS = { "analytical":  optimize_h_analytically, 
                         "num":         optimize_h_numerically, 
                         "num-is":      optimize_h_numerically_IS,
                         "num-util":    optimize_h_utility_numerically,
                         "num-util-is": optimize_h_utility_numerically_IS}
        FIND_H = FIND_H.lower().replace("sgd", "num").replace("utility_term", "util")
        if FIND_H not in H_OPTIMIZERS: 
            raise KeyError("[HOptimizerFactory] Unknown name (%s)! Try: %s" % (FIND_H, list(H_OPTIMIZERS.keys())) )
        return  H_OPTIMIZERS[FIND_H]   
        
                 
###############################################################################
###############################################################################
###############################################################################


class UtilityDependentTermFactory:
    """ Creates methods that allow for calculation of the utility-dependent term approximation. 
        First, the optimal h is obtained using selected H_OPTIMIZER, 
        then, an approximation to the utility-dependent term is constructed using resampled y-s."""

    def __init__(self, **kwargs):
        self.UTILITY_TERM_SCALE = _retrieve_param("UTILITY_TERM_SCALE", kwargs)
        
        self.H_OPTIMIZER = _retrieve_param("H_OPTIMIZER", kwargs)
        self.UTILITY_TERM_MASK = _retrieve_param("utility_term_mask", kwargs)
        
        self.sample_predictive_y0 = _retrieve_param("sample_predictive_y0", kwargs)
        self.sample_predictive_y_IS = _retrieve_param_opt("sample_predictive_y_IS", kwargs)      
        
        if kwargs.get("verbose", True):
            print("[UtilityDependentTermFactory] Configuration: %s" % 
                " ".join("%s=%s" % (k, _format_value(v)) for k, v in vars(self).items() 
                                    if k.upper()==k or k.startswith("sample")) )                

    def create(self, UTILITY_TERM, u):
        """ Returns a requested method for calculation of the utility-dependent term approximation
            
            Args:
                UTILITY_TERM: Selects an approximation to the utility-dependent term: 'vi'/'naive'/'jensen'/'linearized'.        
                u: Utility function.
        """

        def _log_utility_approximator(utility_term, multiply_result_by=1.0):
            """ Decorator that calculates optimal h, resamples ys using sample_predictive_y0
                (separate set of samples is used for optimal h and utilities calculation),
                calculates utility values (assuming that utility function u is defined)
                and measures time using  decorator.
                The result is multiplied by multiply_result_by."""
            def approximating_decorator(*args, **kwargs): 
                calculate_optimal_h = kwargs.pop("calculate_optimal_h", self.H_OPTIMIZER)
                utility_term_mask = kwargs.pop("utility_term_mask", self.UTILITY_TERM_MASK)        

                h = calculate_optimal_h(*args, **sel_with_prefix(kwargs, "h_"))       
                ys = self.sample_predictive_y0(*args, **sel_without_prefix(kwargs, "h_"))   
                assert h.shape==ys.shape[2:], "[_log_utility_approximator] %s=h.shape != ys.shape[2:]=%s" % (h.shape, ys.shape)
                return utility_term(u(h,ys), utility_term_mask) * multiply_result_by
            approximating_decorator.__name__ = utility_term.__name__.replace("_term", "")
            return approximating_decorator



        def _log_utility_approximator_IS(utility_term, multiply_result_by=1.0):
            """ Decorator that calculates optimal h, resamples ys using sample_predictive_y_IS
                (separate set of samples is used for optimal h and utilities calculation),
                calculates utility values (assuming that utility function u is defined)
                and measures time using  decorator.
                The result is multiplied by multiply_result_by."""
            def approximating_decorator(*args, **kwargs): 
                calculate_optimal_h = kwargs.pop("calculate_optimal_h", self.H_OPTIMIZER)
                utility_term_mask = kwargs.pop("utility_term_mask", self.UTILITY_TERM_MASK)        

                h = calculate_optimal_h(*args, **sel_with_prefix(kwargs, "h_"))       
                ys, weights = self.sample_predictive_y_IS(*args, **sel_without_prefix(kwargs, "h_")) #RESAMPLING 
                assert h.shape==ys.shape[1:], "[_log_utility_approximator_IS] h.shape!=ys.shape[1:]"
                return utility_term(u(h,ys), weights, utility_term_mask) * multiply_result_by
            approximating_decorator.__name__ = utility_term.__name__.replace("_term", "")
            return approximating_decorator


        f = UtilityAggregatorFactory()
        log_utility_naive  = _log_utility_approximator(f.create("naive"), self.UTILITY_TERM_SCALE)
        log_utility_Jensen = _log_utility_approximator(f.create("jensen"), self.UTILITY_TERM_SCALE)
        log_utility_Taylor = _log_utility_approximator(f.create("taylor"), self.UTILITY_TERM_SCALE)
        log_utility_linearized = _log_utility_approximator(f.create("linearized"), self.UTILITY_TERM_SCALE)

        log_utility_Jensen_IS = _log_utility_approximator_IS(f.create("jensen-is"), self.UTILITY_TERM_SCALE)
        log_utility_linearized_IS = _log_utility_approximator_IS(f.create("linearized-is"), self.UTILITY_TERM_SCALE)


        
        def log_utility_vi(*args, **kwargs):
            """ Vanilla VI utility-dependent term.
                Always returns 0.0 but depends on all reparametrized parameters 
                so gradients can be calculated in the same way like for LCVI."""
            v = sum(a.rsample().sum() for a in args if issubclass(a.__class__, torch.distributions.Distribution) and a.has_rsample)
            return v*0.0 
            

        UTILITY_TERM_CALCULATORS = {"vi": log_utility_vi, "naive": log_utility_naive, "linearized": log_utility_linearized,
                            "jensen": log_utility_Jensen, "taylor": log_utility_Taylor,
                            "jensen-is": log_utility_Jensen_IS, "linearized-is": log_utility_linearized_IS}
        #maps: 0->vi, ..., 6->linearized-is
        UTILITY_TERM = dict(enumerate(["vi", "naive", "linearized", "jensen", "taylor", "jensen-is", "linearized-is"])).get(UTILITY_TERM, UTILITY_TERM).lower() 
        if UTILITY_TERM not in UTILITY_TERM_CALCULATORS: 
            raise KeyError("[UtilityDependentTermFactory] Unknown name (%s) ! Try: %s" % (UTILITY_TERM, list(UTILITY_TERM_CALCULATORS.keys())) )
        return UTILITY_TERM_CALCULATORS[UTILITY_TERM]
        
        
                        

                                   
                                   
