# -*- coding: utf-8 -*-
"""Contains functions to calculate losses and transform losses into utilities."""

import torch
from torch.nn.functional import softplus
import scipy.optimize
import numpy as np

from aux import tonumpy, print2, is_valid, print_numtimes, sparse_print, retrieve_param, retrieve_param_opt, format_value #get_traceback


#TILTED_Q = 0.2 #h being too large is punished much, but h being to small is fine
#TILTED_Q = 0.9 #h being too small is punished much, but h being to large is fine


def tilted_loss(h, y, q):
    e = (y-h)        
    return torch.max(q*e, (q-1)*e)

  
def tilted_optimal_h(ys, q):   
    quantiles = np.percentile(tonumpy(ys), int(q*100), axis=0) #warning: it blocks gradients!
    return torch.tensor(quantiles, requires_grad=False, dtype=torch.float32)


###############################################################################

    
#LINEX_C = 0.75 #h being too large is punished much, but h being to small is fine
#LINEX_C = -0.75 #h being too small is punished much, but h being to large is fine


def linex(h, y, c):
    return torch.exp(c*(y-h)) - c*(y-h) - 1


def linex_optimal_h(ys, c): 
    return -torch.log( torch.exp(-c*ys).mean(0) ) / c  #allows gradients through    
    
    
###############################################################################

    
def squared_loss(h, y):
    return (y-h)**2
  
  
def squared_loss_optimal_h(ys): 
    return ys.mean(0) #allows gradients through      


###############################################################################


class LossFactory:
    """Selecting loss along with analytical (if possible) optimizer and fixing loss parameters."""
            
    def __init__(self, **kwargs):
        self.TILTED_Q = retrieve_param_opt("TILTED_Q", kwargs)
        self.LINEX_C = retrieve_param_opt("LINEX_C", kwargs)
        
        print("[LossFactory] Configuration: %s" % 
                " ".join("%s=%s" % (k, format_value(v)) for k, v in vars(self).items() if k.upper()==k) )
        
    def create(self, LOSS, TILTED_Q=None, LINEX_C=None):
        """ Returns a requested loss with fixed parameters.

            Args:
                LOSS: loss name (squared, tilted, linex, exptilted).
        """
        def bayes_estimator_not_implemented(y):
            print_numtimes("[LossFactory:create] Warning: Bayes estimator is not implemented!", "not_implemented_bayes_estimator", 1) 
            return torch.ones(y.shape[1:])*float("nan")

        
        def _fix_tilted_q():
            if TILTED_Q is not None: self.TILTED_Q = TILTED_Q
            if self.TILTED_Q is None: raise LookupError("Param TILTED_Q is not set!")

        def tilted_loss_fixedq(h, y):
            _fix_tilted_q()
            return tilted_loss(h, y, q=self.TILTED_Q)

        def tilted_optimal_h_fixedq(y):
            _fix_tilted_q()
            return tilted_optimal_h(y, q=self.TILTED_Q)
                        
    
        def expsquared(h,y): #falls in range [0,1]
            return 1.0-torch.exp( -squared_loss(h, y) )

        def exptilted_loss_fixedq(h,y): #falls in range [0,1]
            return 1.0-torch.exp( -tilted_loss_fixedq(h, y) )

        def exptilted125_loss_fixedq(h,y): #falls in range [0.25,1.25]
            return 1.25-torch.exp( -tilted_loss_fixedq(h, y) )               

                
        def _fix_linex_c():
            if LINEX_C is not None: self.LINEX_C = LINEX_C
            if self.LINEX_C is None: raise LookupError("Param LINEX_C is not set!")        

        def linex_fixedc(h, y):
            _fix_linex_c()
            return linex(h, y, c=self.LINEX_C)

        def linex_optimal_h_fixedc(y):
            _fix_linex_c()
            return linex_optimal_h(y, c=self.LINEX_C)                  


        #Choose which loss to use
        LOSSES = {"squared":        (squared_loss, squared_loss_optimal_h),
                  "tilted":         (tilted_loss_fixedq, tilted_optimal_h_fixedq),
                  "expsquared":     (expsquared,     bayes_estimator_not_implemented),
                  "exptilted":      (exptilted_loss_fixedq,     bayes_estimator_not_implemented),
                  "exptilted125":   (exptilted125_loss_fixedq,  bayes_estimator_not_implemented),
                  "linex":          (linex_fixedc, linex_optimal_h_fixedc)}
        assert LOSS.lower() in LOSSES, "[LossFactory] Unknown loss: %s!" % LOSS                 
        loss, optimal_h = LOSSES[LOSS.lower()]
        return loss, optimal_h
                

###############################################################################


class UtilityFactory:
    """Transformations from loss to utility."""
    
    def __init__(self, **kwargs):
        self.M = retrieve_param("M", kwargs)
        self.GAMMA = retrieve_param_opt("GAMMA", kwargs, 1.0)
        
        print("[UtilityFactory] Configuration: %s" % 
                " ".join("%s=%s" % (k, format_value(v)) for k, v in vars(self).items() if k.upper()==k) )
            
    def create(self, UTIL, loss):
        """ Returns a requested transformation to the loss. """

        def uid(h,y):
            return -loss(h, y)

        def ubigM(h, y):    
            return self.M - loss(h, y)

        def uexp(h, y):
            return torch.exp( -self.GAMMA*loss(h,y) )

        def usoftplus(h, y):
            return softplus( -loss(h,y) )

        def exptilted125(h, y): #falls in range [0.25,1.25] for exptilted125 loss
            return -(loss(h,y)-1.25) + 0.25
        
        def exptilted(h, y): #falls in range [0.0,1.0] for exptilted loss
            return -(loss(h,y)-1.0)         

        def expsquared(h, y): #falls in range [0.0,1.0] for exptilted loss
            return 1.0-loss(h,y)

        UTILS = {"bigm": ubigM, "exp": uexp, "id": uid, "softplus": usoftplus, "exptilted125": exptilted125, "exptilted": exptilted, "expsquared": expsquared}
        assert UTIL.lower() in UTILS, "[UtilityFactory] Unknown utility: %s!" % UTIL                         
        u = UTILS[UTIL.lower()]
        return u




