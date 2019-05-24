# -*- coding: utf-8 -*-

import torch
from aux_time import timing
from aux import print2, format_value
from numerical_optimization import optimal_h_numerically, optimal_h_numerically_scipy


_last_result_cache = {}
def cache_function_last_result(f, 
        msg1="[Evaluation] Reusing previous result of <name>",
        msg2="[Evaluation] Recalculating and memorizing result of <name>"):
    key = f.__name__
    print2("[Evaluation] Results of %s will be cached." % key)

    def same_objects_on_lists(l1, l2):        
        return len(l1)==len(l2) and 0==sum((e1 is not e2) for e1, e2 in zip(l1, l2))

    def caching_function(*args): 

        args2result = _last_result_cache.get(key, ([], None))
        prev_args = args2result[0]
        if same_objects_on_lists(prev_args, args): 
            if msg1 is not None: print2( msg1.replace("<name>",key) )
        else:
            if msg2 is not None: print2( msg2.replace("<name>",key) )
            result = f(*args)   
            _last_result_cache[key] = args2result = (args, result)    
        return args2result[1]

    caching_function.__name__ = key
    return caching_function


class Measures:

    def __init__(self, y, loss, u, 
                 sample_predictive_y, 
                 optimal_h_bayes_estimator=None,
                 y_mask=None,
                 GAIN_OPTIMAL_H_NUMERICALLY = True,
                 RISK_OPTIMAL_H_NUMERICALLY = False,
                 EVAL_NSAMPLES_UTILITY_TERM_THETA = 1000,
                 EVAL_NSAMPLES_UTILITY_TERM_Y = 1,
                 EVAL_MAX_NITER = 10000,
                 EVAL_SGD_PREC = 0.0001,
                 EVAL_LR = 0.01,
                 EVAL_RESAMPLE_EVERY_TIME = False
                 ):
        """
            Args:
              y       Evaluation data.
              y_mask  A mask selecting data points for evaluation (default: all).    
              loss    A function y x h -> loss used to calculate risks.
              u       A function y x h -> utility used to calculate gains.
              sample_predictive_y  A function that for each data point from y, generates samples from predictive posterior.
              EVAL_RESAMPLE_EVERY_TIME  Can results of sample_predictive_y, optimal_h_for_gain and optimal_h_for_risk be cached? 
        """
        self.y = y
        self.y_mask = y_mask
        if self.y_mask is None: 
            print2("[Evaluation] WARNING: using default all data points in evaluation.")        
            env = torch if "cpu" in str(self.y.device).lower() else torch.cuda
            self.y_mask = torch.ones_like(self.y).type(env.ByteTensor)

        self.loss = loss
        self.utility = u
        self.sample_predictive_y = sample_predictive_y
        self.optimal_h_bayes_estimator = optimal_h_bayes_estimator
        if (self.optimal_h_bayes_estimator is None) and \
           (not GAIN_OPTIMAL_H_NUMERICALLY or not RISK_OPTIMAL_H_NUMERICALLY):
            print2("[Evaluation] WARNING: Optimal decisions h for both Risk and Gain will be obtained numerically.")
            self.optimal_h_bayes_estimator = lambda ys: None
            GAIN_OPTIMAL_H_NUMERICALLY, RISK_OPTIMAL_H_NUMERICALLY = True, True

        self.GAIN_OPTIMAL_H_NUMERICALLY = GAIN_OPTIMAL_H_NUMERICALLY
        self.RISK_OPTIMAL_H_NUMERICALLY = RISK_OPTIMAL_H_NUMERICALLY

        self.EVAL_NSAMPLES_UTILITY_TERM_THETA = EVAL_NSAMPLES_UTILITY_TERM_THETA
        self.EVAL_NSAMPLES_UTILITY_TERM_Y = EVAL_NSAMPLES_UTILITY_TERM_Y
        self.EVAL_MAX_NITER = EVAL_MAX_NITER
        self.EVAL_SGD_PREC = EVAL_SGD_PREC
        self.EVAL_LR = EVAL_LR

        print("[Evaluation] Configuration: %s" % 
                " ".join("%s=%s" % (k, format_value(v)) for k, v in vars(self).items() ) )

        if not EVAL_RESAMPLE_EVERY_TIME:
            self.optimal_h_for_gain = cache_function_last_result( self.optimal_h_for_gain )
            self.optimal_h_for_risk = cache_function_last_result( self.optimal_h_for_risk )
            self.sample_predictive_posterior = cache_function_last_result( self.sample_predictive_posterior )


    def sample_predictive_posterior(self, *args):
        ys = self.sample_predictive_y(*args, nsamples_theta=self.EVAL_NSAMPLES_UTILITY_TERM_THETA, 
                                             nsamples_y=self.EVAL_NSAMPLES_UTILITY_TERM_Y)
        return ys    


    def optimal_h_for_gain(self, ys):
        if self.GAIN_OPTIMAL_H_NUMERICALLY:
            print2("[Evaluation] Numerically optimizing h for qGain. May take time...")
            h = optimal_h_numerically(ys, self.utility, 
                                      data_mask=self.y_mask, 
                                      start=None, #self.optimal_h_bayes_estimator(ys), #start from Bayes estimator for Risk
                                      max_niter=self.EVAL_MAX_NITER, tol=self.EVAL_SGD_PREC, 
                                      tol_goal=-1, debug=True, lr=self.EVAL_LR)
            #h = optimal_h_numerically_scipy(ys, self.utility, data_mask=self.y_mask,
            #                      max_niter=self.EVAL_MAX_NITER, tol=self.EVAL_SGD_PREC, tol_goal=-1, 
            #                      lr=self.EVAL_LR, start=None, optimizer="COBYLA",
            #                      verbose=True, debug=False, sparse_verbose=True)
        else:
            h = self.optimal_h_bayes_estimator(ys)
        print2("[Evaluation:optimal_h_for_gain] h=%s" % str(h)[:200])   
        return h


    def optimal_h_for_risk(self, ys):
        if self.RISK_OPTIMAL_H_NUMERICALLY: 
            print2("[Evaluation] Numerically optimizing h for qRisk. May take time...")
            utility = lambda h, y: -self.loss(h, y)
            h = optimal_h_numerically(ys, utility, 
                                      data_mask=self.y_mask, 
                                      start=None, #self.optimal_h_bayes_estimator(ys), #start from Bayes estimator for Risk
                                      max_niter=self.EVAL_MAX_NITER, tol=self.EVAL_SGD_PREC, 
                                      tol_goal=-1, debug=True, lr=self.EVAL_LR)     
        else:
            h = self.optimal_h_bayes_estimator(ys)
        print2("[Evaluation:optimal_h_for_risk] h=%s" % str(h)[:200])   
        return h


    @timing
    def qrisk(self, *args):   
        """qRisk coming from data approximate predictive posteriors.""" 
        ys = self.sample_predictive_posterior(*args)
        h = self.optimal_h_for_risk(ys)        
        ys = self.sample_predictive_posterior(*args)
        assert h.shape==ys.shape[1: ]
        lu = torch.masked_select(self.loss(h, ys), self.y_mask).mean()    
        return lu


    @timing
    def empirical_risk(self, *args):
        """Empirical risk."""
        ys = self.sample_predictive_posterior(*args)
        h = self.optimal_h_for_risk(ys)        
        assert h.shape==self.y.shape
        lu = torch.masked_select(self.loss(h, self.y), self.y_mask).mean()
        return lu


    @timing
    def qgain(self, *args):    
        """qGain coming from data approximate predictive posteriors."""    
        ys = self.sample_predictive_posterior(*args)
        h = self.optimal_h_for_gain(ys)
        ys = self.sample_predictive_posterior(*args)
        assert h.shape==ys.shape[1: ]
        lu = torch.masked_select(self.utility(h, ys), self.y_mask).mean()    
        return lu


    @timing
    def empirical_gain(self, *args):
        """Empirical gain."""                
        ys = self.sample_predictive_posterior(*args)
        h = self.optimal_h_for_gain(ys)
        assert h.shape==self.y.shape
        lu = torch.masked_select(self.utility(h, self.y), self.y_mask).mean()
        return lu

