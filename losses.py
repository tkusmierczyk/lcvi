# -*- coding: utf-8 -*-
"""Contains functions to calculate losses, transform losses into utilities and obtain optimal decisions."""

import torch
from torch.nn.functional import softplus
import scipy.optimize

import numpy as np
import time

from inspect import signature

from aux import tonumpy, print2, is_valid, print_numtimes, sparse_print 



#TILTED_Q = 0.2 #h being too large is punished much, but h being to small is fine
#TILTED_Q = 0.9 #h being too small is punished much, but h being to large is fine


def tilted_loss(h, y, q):
    e = (y-h)        
    return torch.max(q*e, (q-1)*e)

  
def tilted_optimal_h(ys, q):   
    quantiles = np.percentile(tonumpy(ys), int(q*100), axis=0) #blocks gradients
    return torch.tensor(quantiles, requires_grad=False, dtype=torch.float32)


###############################################################################

    
#LINEX_C = 0.75 #h being too large is punished much, but h being to small is fine
#LINEX_C = -0.75 #h being too small is punished much, but h being to large is fine


def linex(h, y, c):
    return torch.exp(c*(h-y)) - c*(h-y) - 1


def linex_optimal_h(ys, c): 
    return -torch.log( torch.exp(-c*ys).mean(0) ) / c  #allows gradients through    
    
    
###############################################################################

    
def squared_loss(h, y):
    return (h-y)**2
  
  
def squared_loss_optimal_h(ys): 
    return ys.mean(0) #allows gradients through      
    

###############################################################################


def _format_value(v):
    if hasattr(v, "__name__"): return v.__name__
    if hasattr(v, "shape"): return v.shape
    return str(v)[:15]       
    

def _retrieve_param(name, kwargs):
    if name not in kwargs or kwargs[name] is None:
        raise LookupError("Param <%s> is not set!" % name)
    return kwargs[name]    
    
    
def _retrieve_param_opt(name, kwargs):
    if name not in kwargs or kwargs[name] is None: 
        return None
    return kwargs[name]       


###############################################################################


class LossFactory:
    """Selecting loss along with analytical (if possible) optimizer and fixing loss parameters."""
            
    def __init__(self, **kwargs):
        self.TILTED_Q = _retrieve_param_opt("TILTED_Q", kwargs)
        self.LINEX_C = _retrieve_param_opt("LINEX_C", kwargs)
        
        print("[LossFactory] Configuration: %s" % 
                " ".join("%s=%s" % (k, _format_value(v)) for k, v in vars(self).items() if k.upper()==k) )
        
    def create(self, LOSS, TILTED_Q=None, LINEX_C=None):
        """ Returns a requested loss with fixed parameters.

            Args:
                LOSS: loss name (squared, tilted, linex, exptilted).
        """
        
        def _fix_tilted_q():
            if TILTED_Q is not None: self.TILTED_Q = TILTED_Q
            if self.TILTED_Q is None: raise LookupError("Param TILTED_Q is not set!")


        def tilted_loss_fixedq(h, y):
            _fix_tilted_q()
            return tilted_loss(h, y, q=self.TILTED_Q)


        
        def tilted_optimal_h_fixedq(y):
            _fix_tilted_q()
            return tilted_optimal_h(y, q=self.TILTED_Q)
                        
    
        def exptilted_loss_fixedq(h,y): #falls in range [0,1]
            return 1.0-torch.exp( -tilted_loss_fixedq(h, y) )
        
        
        
        def exptilted_optimal_h_fixedq(y):
            print_numtimes("exptilted_optimal_h_fixedq", 
                "[exptilted_optimal_h_fixedq] Error: exptilted_optimal_h_fixedq is not implemented!", 1) 
            return torch.ones(y.shape[1:])*float("nan")
                
                
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
        LOSSES = {"squared":    (squared_loss, squared_loss_optimal_h),
                  "tilted":     (tilted_loss_fixedq, tilted_optimal_h_fixedq),
                  "exptilted":  (exptilted_loss_fixedq, exptilted_optimal_h_fixedq),
                  "linex":      (linex_fixedc, linex_optimal_h_fixedc)}
        assert LOSS.lower() in LOSSES, "[LossFactory] Unknown loss: %s!" % LOSS                 
        loss, optimal_h = LOSSES[LOSS.lower()]
        return loss, optimal_h
                

###############################################################################


class UtilityFactory:
    """Transformations from loss to utility."""
    
    def __init__(self, **kwargs):
        self.M = _retrieve_param("M", kwargs)
        
        print("[UtilityFactory] Configuration: %s" % 
                " ".join("%s=%s" % (k, _format_value(v)) for k, v in vars(self).items() if k.upper()==k) )
            
    def create(self, UTIL, loss):
        """ Returns a requested transformation to the loss. """

        def uid(h,y):
            return -loss(h, y)


        def ubigM(h, y):    
            return self.M - loss(h, y)


        def uexp(h, y):
            return torch.exp( -loss(h,y) )


        def usoftplus(h, y):
            return softplus( -loss(h,y) )


        UTILS = {"bigm": ubigM, "exp": uexp, "id": uid, "softplus": usoftplus}
        assert UTIL.lower() in UTILS, "[UtilityFactory] Unknown utility: %s!" % UTIL                         
        u = UTILS[UTIL.lower()]
        return u


    
###############################################################################
###############################################################################
###############################################################################
#Numerical optimization w.r.t h
    
             
def gain(us, data_mask):
    """ Args: 
            us: Utility matrix (#y-samples x #theta-samples x data-size). 
    """
    point_utility_term = us.mean(0).mean(0)      
    assert point_utility_term.shape==data_mask.shape, "%s=point_utility_term.shape!=data_mask.shape=%s" % (point_utility_term.shape,data_mask.shape)
    return torch.masked_select(point_utility_term, data_mask).sum()    
    
    
def optimal_h_numerically_ty(ys, u, utility_aggregator=gain, data_mask=None,
                        max_niter=10000, tol=1e-4, tol_goal=-1, lr=0.01, start=None, optimizer=torch.optim.Adam,
                        verbose=False, debug=False, sparse_verbose=True): 
    """ Using numerical optimization finds optimal h for utility-dependent term expressed by utility_aggregator.
    
        Args:
            ys: Samples matrix. The dimensionality should match what utility_aggregator takes:
                #y-samples x #theta-samples x data-size.
            u:  Utility function u(h, ys) -> utilities matrix (the same shape as ys).
            utility_aggregator: A function that calculate utility-dependent term. 
                                Should take exactly 2 params: utilites and data_mask.
            data_mask: A mask passed to utility_aggregator.
    """    
    printv = lambda txt: (print2("[optimal_h_numerically_ty] "+txt) if verbose else None)
    if sparse_verbose and not verbose: printv = lambda txt: sparse_print("optimal_h_numerically_ty", "[optimal_h_numerically_ty]"+txt, 100) 
    printd = lambda txt: (print2("[optimal_h_numerically_ty] "+txt) if debug else None)

    assert len( signature(utility_aggregator).parameters )==2, \
            "[optimal_h_numerically_ty] Your utility_aggregator=%s takes wrong number of params! perhaps requires weights?" % utility_aggregator

    env = torch if "cpu" in str(ys.device) else torch.cuda
    if data_mask is None: #No data mask provided. Using all data points
        data_mask = (torch.ones(ys.shape[2: ]) if len(ys.shape)>2 else torch.tensor(1)).type(env.ByteTensor)
    
    y = torch.tensor( tonumpy(ys), requires_grad=False)  
              
    if start is None: 
        h = y.mean(0).mean(0).clone().detach().requires_grad_(True) #start from E(y)   
    elif start is None or not is_valid(torch.tensor(start)):
        printd("start point is invalid. ignoring!")
        h = y.mean(0).mean(0).clone().detach().requires_grad_(True) #start from E(y)       
    else: 
        h = torch.tensor( tonumpy(start), requires_grad=True)
    
    optimizer = optimizer([h], lr=lr)
    prev_h, prev_goal = torch.tensor(tonumpy(h)), float("inf")
    start = time.time()    
    for i in range(max_niter):
            
        goal = -utility_aggregator(u(h, y), data_mask)         
        optimizer.zero_grad()
        goal.backward(retain_graph=False)
        optimizer.step()   
        
        #check for convergence:
        if (prev_h-h).abs().max() <= tol:
            printv("[%.2f][ys=%s max_niter=%i tol=%s tol_goal=%s lr=%s u=%s->%s] Converged in %i. iter (tolerance reached): obj=%.4f max-err=%.8f mean-err=%.8f" % 
                   (time.time()-start, tuple(ys.shape), max_niter, tol, tol_goal, lr, u.__name__, utility_aggregator.__name__, 
                   i+1,goal.item(),(prev_h-h).abs().max(),(prev_h-h).abs().mean()))    
            break
        if abs(prev_goal-goal.item()) <= tol_goal:
            printv("[%.2f][ys=%s max_niter=%i tol=%s tol_goal=%s lr=%s u=%s->%s] Converged in %i. iter (objective tolerance reached): obj=%.4f max-err=%.8f mean-err=%.8f" % 
                   (time.time()-start, tuple(ys.shape), max_niter, tol, tol_goal, lr, u.__name__, utility_aggregator.__name__, 
                   i+1,goal.item(),(prev_h-h).abs().max(),(prev_h-h).abs().mean()))    
            break      
        if i>=max_niter-1: 
            printv("[%.2f][ys=%s max_niter=%i tol=%s tol_goal=%s lr=%s u=%s->%s] Converged in %i. iter (max number reached): obj=%.4f max-err=%.8f mean-err=%.8f" % 
                    (time.time()-start, tuple(ys.shape), max_niter, tol, tol_goal, lr, u.__name__, utility_aggregator.__name__, 
                    i+1,goal.item(),(prev_h-h).abs().max(),(prev_h-h).abs().mean()))    
            break         
                                   
        if i%(max_niter//10)==0: printd("[%.2f] iter %i: objective=%.4f err=%.6f" % 
                                        (time.time()-start,i,goal.item(),(prev_h-h).abs().max()))
        
        prev_h = torch.tensor(tonumpy(h))
        prev_goal = goal.item()
                                                
    return h    
    

def gain_weighted(us, weights, data_mask):   
    """ 
        Args: 
            us: Utility matrix (#y-samples x data-size). 
    """  
    point_utility_term = (us * weights).sum(0)
    assert point_utility_term.shape==data_mask.shape, "%s=point_utility_term.shape!=data_mask.shape=%s" % (point_utility_term.shape,data_mask.shape)
    return torch.masked_select(point_utility_term, data_mask).sum()
    
    
def optimal_h_numerically(ys, u, weights=None, utility_aggregator=gain_weighted, data_mask=None,
                                  max_niter=10000, tol=1e-4, tol_goal=-1, lr=0.01, start=None, optimizer=torch.optim.Adam,
                                  verbose=False, debug=False, sparse_verbose=True): 
    """ Using numerical optimization finds optimal h for utility-dependent term expressed by utility_aggregator.
    
        Args:
            ys: Samples matrix. The dimensionality should match what utility_aggregator takes: #y-samples x data-size.
            u:  Utility function u(h, ys) -> utilities matrix (the same shape as ys).
            utility_aggregator: A function that calculate utility-dependent term. 
                Should take exactly 3 params: utilities, weights and data_mask.
                By default gain calculated of weighted samples. 
            data_mask: A mask passed to utility_aggregator.
    """    
    printv = lambda txt: (print2("[optimal_h_numerically]"+txt) if verbose else None)
    if sparse_verbose and not verbose: printv = lambda txt: sparse_print("optimal_h_numerically", "[optimal_h_numerically]"+txt, 100) 
    printd = lambda txt: (print2("[optimal_h_numerically]"+txt) if debug else None)

    assert len( signature(utility_aggregator).parameters )==3, \
            "[optimal_h_numerically] Your utility_aggregator=%s takes wrong number of params! does not support weights?" % utility_aggregator

    env = torch if "cpu" in str(ys.device) else torch.cuda
    if data_mask is None:
        data_mask = (torch.ones(ys.shape[1: ]) if len(ys.shape)>1 else torch.tensor(1)).type(env.ByteTensor)

    if weights is None: weights = torch.ones_like(ys)
    weights /= weights.sum(0) #enforce normalization   
    weights = torch.tensor(tonumpy(weights), requires_grad=False)

    printd("[ys=%s weights=%s data_mask=%s max_niter=%i tol=%s tol_goal=%s lr=%s u=%s->%s] " % 
            (tuple(ys.shape), tuple(weights.shape), tuple(data_mask.shape), max_niter, tol, tol_goal, lr, 
            u.__name__, utility_aggregator.__name__))
    
    y = torch.tensor( tonumpy(ys), requires_grad=False)   
                      
    if start is None: 
        h = (y*weights).sum(0).clone().detach().requires_grad_(True) #start from E(y)   
    elif start is None or not is_valid(torch.tensor(start)):
        printd("start point is invalid. ignoring!")
        h = (y*weights).sum(0).clone().detach().requires_grad_(True) #start from E(y)   
    else: 
        h = torch.tensor( tonumpy(start), requires_grad=True)    
    
    optimizer = optimizer([h], lr=lr)
    prev_h, prev_goal = torch.tensor(tonumpy(h)), float("inf")
    start = time.time()
    for i in range(max_niter):
            
        goal = -utility_aggregator(u(h, y), weights, data_mask)         
        optimizer.zero_grad()
        goal.backward(retain_graph=False)
        optimizer.step()   
        
        #check for convergence:
        if (prev_h-h).abs().max() <= tol:
            printv("[%.2f][ys=%s max_niter=%i tol=%s tol_goal=%s lr=%s u=%s->%s] Converged in %i. iter (tolerance reached): obj=%.4f max-err=%.8f mean-err=%.8f" % 
                    (time.time()-start, tuple(ys.shape), max_niter, tol, tol_goal, lr, u.__name__, utility_aggregator.__name__, 
                    i+1,goal.item(),(prev_h-h).abs().max(),(prev_h-h).abs().mean()))    
            break
        if abs(prev_goal-goal.item()) <= tol_goal:
            printv("[%.2f][ys=%s max_niter=%i tol=%s tol_goal=%s lr=%s u=%s->%s] Converged in %i. iter (objective tolerance reached): obj=%.4f max-err=%.8f mean-err=%.8f" % 
                    (time.time()-start, tuple(ys.shape), max_niter, tol, tol_goal, lr, u.__name__, utility_aggregator.__name__, 
                    i+1,goal.item(),(prev_h-h).abs().max(),(prev_h-h).abs().mean()))    
            break      
        if i>=max_niter-1: 
            printv("[%.2f][ys=%s max_niter=%i tol=%s tol_goal=%s lr=%s u=%s->%s] Converged in %i. iter (max number reached): obj=%.4f max-err=%.8f mean-err=%.8f" % 
                   (time.time()-start, tuple(ys.shape), max_niter, tol, tol_goal, lr, u.__name__, utility_aggregator.__name__, 
                   i+1,goal.item(),(prev_h-h).abs().max(),(prev_h-h).abs().mean()))    
            break                     
            
        if i%(max_niter//10)==0: printd("[%.2f] iter %i: objective=%.4f err=%.6f" % 
                                        (time.time()-start,i,goal.item(),(prev_h-h).abs().max()))
                                        
        prev_h = torch.tensor(tonumpy(h))
        prev_goal = goal.item()
                                        
    return h    


def _scipy_minimize(fun, x0, method="COBYLA", max_niter=10000, tol=1e-4, tol_goal=-1, lr=0.01, debug=False):
    if method=="COBYLA":
        return scipy.optimize.minimize(fun, x0, args=(), method='COBYLA', 
                                constraints=(), tol=None, callback=None, 
                                options={'rhobeg': 1.0, 'maxiter': max_niter, 'disp': False, 'catol': tol, 'tol': max(tol_goal, 1e-8)})        
    if method=="CG":
        return scipy.optimize.minimize(fun, x0, args=(), method='CG', jac=None, tol=None, callback=None, 
                    options={'gtol': tol, 'eps': 1.4901161193847656e-08, 'maxiter': max_niter, 'disp': False, 'return_all': False})

    if method=="SLSQP":
        return scipy.optimize.minimize(fun, x0, args=(), method='SLSQP', jac=None, bounds=None, constraints=(), tol=None, callback=None, 
            options={'maxiter': max_niter, 'ftol': 1e-06, 'iprint': 1, 'disp': False, 'eps': 1.4901161193847656e-08})

    if method=="Powell":
        return scipy.optimize.minimize(fun, x0, args=(), method='Powell', tol=None, callback=None, 
            options={'xtol': 0.0001, 'ftol': 0.0001, 'maxiter': max_niter, 'maxfev': None, 'disp': False, 'direc': None, 'return_all': False})

    raise LookupError("Not supported SciPy method (%s)!" % method) 

    
def optimal_h_numerically_ty_scipy(ys, u,  weights=None, utility_aggregator=gain, data_mask=None,
                                  max_niter=10000, tol=1e-4, tol_goal=-1, lr=0.01, start=None, optimizer="COBYLA",
                                  verbose=False, debug=False, sparse_verbose=True): 
    """ Using numerical optimization (SciPy) finds optimal h for utility-dependent term expressed by utility_aggregator.

        Compatible with optimal_h_numerically_ty.
    """    
    printv = lambda txt: (print2("[optimal_h_numerically_ty_scipy]"+txt) if verbose else None)
    if sparse_verbose and not verbose: printv = lambda txt: sparse_print("optimal_h_numerically_ty_scipy", "[optimal_h_numerically_scipy]"+txt, 100) 
    printd = lambda txt: (print2("[optimal_h_numerically_ty_scipy]"+txt) if debug else None)

    assert len( signature(utility_aggregator).parameters )==2, \
            "[optimal_h_numerically_ty] Your utility_aggregator=%s takes wrong number of params! perhaps requires weights?" % utility_aggregator

    env = torch if "cpu" in str(ys.device) else torch.cuda
    if data_mask is None:
        data_mask = (torch.ones(ys.shape[2: ]) if len(ys.shape)>2 else torch.tensor(1)).type(env.ByteTensor)        
    
    y = torch.tensor(tonumpy(ys), requires_grad=False)   
                      
    if start is None: 
        h = y.mean(0).mean(0) #start from E(y)   
    elif start is None or not is_valid(torch.tensor(start)):
        printd("start point is invalid. ignoring!")
        h = y.mean(0).mean(0) #start from E(y)       
    else: 
        h = start

    start = time.time()
    x0 = tonumpy(h).flatten()
    fun = lambda h: -utility_aggregator(u(torch.tensor(h.reshape(y.shape[2:]), dtype=y.dtype), y), data_mask).item()          
    result = _scipy_minimize(fun, x0, method=optimizer, max_niter=max_niter, tol=tol, tol_goal=tol_goal, lr=lr, debug=debug)
    if verbose or True:
        printv("[%.4f][optimizer=%s ys=%s max_niter=%i tol=%s tol_goal=%s lr=%s u=%s->%s] %s" % 
                                    (time.time()-start, optimizer, tuple(ys.shape), max_niter, tol, 
                                     tol_goal, lr, u.__name__, utility_aggregator.__name__, 
                                     str(result).replace("\n", ";")[:200]))
    return torch.tensor(result["x"].reshape(y.shape[2:]), dtype=ys.dtype)

        
def optimal_h_numerically_scipy(ys, u,  weights=None, utility_aggregator=gain_weighted, data_mask=None,
                                  max_niter=10000, tol=1e-4, tol_goal=-1, lr=0.01, start=None, optimizer="COBYLA",
                                  verbose=False, debug=False, sparse_verbose=True): 
    """ Using numerical optimization (SciPy) finds optimal h for utility-dependent term expressed by utility_aggregator.

        Compatible with optimal_h_numerically.
    """    
    printv = lambda txt: (print2("[optimal_h_numerically_scipy]"+txt) if verbose else None)
    if sparse_verbose and not verbose: printv = lambda txt: sparse_print("optimal_h_numerically_scipy", "[optimal_h_numerically_scipy]"+txt, 100) 
    printd = lambda txt: (print2("[optimal_h_numerically_scipy]"+txt) if debug else None)

    assert len( signature(utility_aggregator).parameters )==3, \
            "[optimal_h_numerically_scipy] Your utility_aggregator=%s takes wrong number of params! does not support weights?" % utility_aggregator

    env = torch if "cpu" in str(ys.device) else torch.cuda
    if data_mask is None:
        data_mask = (torch.ones(ys.shape[1: ]) if len(ys.shape)>1 else torch.tensor(1)).type(env.ByteTensor)

    if weights is None: weights = torch.ones_like(ys)
    weights /= weights.sum(0) #enforce normalization   
    weights = torch.tensor(tonumpy(weights), requires_grad=False)
    
    y = torch.tensor( tonumpy(ys), requires_grad=False)   
                      
    if start is None: 
        h = (y*weights).sum(0) #start from E(y)   
    elif start is None or not is_valid(torch.tensor(start)):
        printd("start point is invalid. ignoring!")
        h = (y*weights).sum(0) #start from E(y)   
    else: 
        h = start

    start = time.time()
    x0 = tonumpy(h).flatten()        
    fun = lambda h: -utility_aggregator(u(torch.tensor(h.reshape(y.shape[1:]), dtype=y.dtype), y), weights, data_mask).item()        
    result = _scipy_minimize(fun, x0, method=optimizer, max_niter=max_niter, tol=tol, tol_goal=tol_goal, lr=lr, debug=debug)
    if verbose:
        printv("[%.4f][optimizer=%s ys=%s max_niter=%i tol=%s tol_goal=%s lr=%s u=%s->%s] %s" % 
                                    (time.time()-start, optimizer, tuple(ys.shape), max_niter, tol, 
                                     tol_goal, lr, u.__name__, utility_aggregator.__name__, 
                                     str(result).replace("\n", ";")[:200]))
    return torch.tensor(result["x"].reshape(y.shape[1:]), dtype=ys.dtype)
    


