# -*- coding: utf-8 -*-
"""Contains factory capable of constructing various estimators to the utility-dependent term."""

import torch

from aux import flatten_first_two_dims


# When utilities are close to 0, logs or dividing by them leads to NaN-s
# adding eps increases the numerical stability
EPS_NAIVE = 1e-16
EPS_JENSEN = 1e-16
EPS_TAYLOR1 = 1e-16
EPS_TAYLOR2 = 1e-8


class UtilityAggregatorFactory:
    """ Constructs methods that take a matrix of utilities 
        and calculate an approximation to the utility-dependent term.
        In practice, the methods aggregate multiple utilities into single value.
    """

    def create(self, UTILITY_AGGREGATOR, a=None, b=None):
        """ Returns the requested approximation to the utility-dependent term.

            Args:
              a,b  Parameters of the linear transformation of the utility (b+a*u). 
        """
        print("[UtilityAggregatorFactory] creating %s (a=%s, b=%s)" % (UTILITY_AGGREGATOR, a, b))

        def utility_term_naive(us, data_mask):
            """
                Args:
                    UTILITY_AGGREGATOR: Selects an approximation to the utility-dependent term: 'vi'/'naive'/'jensen'/'linearized'.
                    us: Utility matrix where dim 0 is over output samples y, 
                        dim 1 is over latent variables theta and remaining dims are over input.
                    data_mask: A mask over input values that selects which data points should be included.
            """            
            point_utility_term = (us.mean(0) + EPS_NAIVE).log().mean(0)   
            assert point_utility_term.shape==data_mask.shape, "%s=datashape!=mask.shape=%s" % (point_utility_term.shape,data_mask.shape)    
            return torch.masked_select(point_utility_term, data_mask).sum()


        def utility_term_naive_transformed(us, data_mask):
            point_utility_term = (b + a * us.mean(0)).log().mean(0) # transformed utility
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


        def utility_term_vi(us, data_mask):
            return us.sum()*0.0


        UTILITY_AGGREGATORS = {"vi": utility_term_vi, 
                               "naive": utility_term_naive, 
                               "linearized": utility_term_linearized,
                               "gain": utility_term_linearized,
                               "linear": utility_term_linearized,
                               "jensen": utility_term_Jensen, 
                               "taylor": utility_term_Taylor,
                               "naive-transformed": utility_term_naive_transformed}
        if UTILITY_AGGREGATOR not in UTILITY_AGGREGATORS: 
            raise KeyError("[UtilityAggregatorFactory] Unknown name (%s)! Try: %s" % (UTILITY_AGGREGATOR, list(UTILITY_AGGREGATORS.keys())) )
        return UTILITY_AGGREGATORS[UTILITY_AGGREGATOR]



