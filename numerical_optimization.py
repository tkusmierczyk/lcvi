# -*- coding: utf-8 -*-
"""Numerical optimization to obtain optimal decisions."""

import torch
from torch.nn.functional import softplus
import scipy.optimize
import numpy as np
import time
from inspect import signature

from aux import tonumpy, print2, is_valid, print_numtimes, sparse_print #get_traceback


             
def gain(us, data_mask):
    """ 
        Args: 
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
            printv("[%.2f][ys=%s max_niter=%i tol=%s tol_goal=%s lr=%s u=%s->%s] Finished at %i. iter (tolerance reached): obj=%.4f max-err=%.8f mean-err=%.8f" % 
                   (time.time()-start, tuple(ys.shape), max_niter, tol, tol_goal, lr, u.__name__, utility_aggregator.__name__, 
                   i+1,goal.item(),(prev_h-h).abs().max(),(prev_h-h).abs().mean()))    
            break
        if abs(prev_goal-goal.item()) <= tol_goal:
            printv("[%.2f][ys=%s max_niter=%i tol=%s tol_goal=%s lr=%s u=%s->%s] Finished at %i. iter (objective tolerance reached): obj=%.4f max-err=%.8f mean-err=%.8f" % 
                   (time.time()-start, tuple(ys.shape), max_niter, tol, tol_goal, lr, u.__name__, utility_aggregator.__name__, 
                   i+1,goal.item(),(prev_h-h).abs().max(),(prev_h-h).abs().mean()))    
            break      
        if i>=max_niter-1: 
            printv("[%.2f][ys=%s max_niter=%i tol=%s tol_goal=%s lr=%s u=%s->%s] Finished at %i. iter (max number reached): obj=%.4f max-err=%.8f mean-err=%.8f" % 
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
        printd("[optimal_h_numerically] start point is invalid. ignoring!")
        h = (y*weights).sum(0).clone().detach().requires_grad_(True) #start from E(y)   
    else: 
        h = torch.tensor( tonumpy(start), requires_grad=True)    
    
    optimizer = optimizer([h], lr=lr)
    prev_h, prev_goal = torch.tensor(tonumpy(h)), float("inf")
    start = time.time()
    for i in range(max_niter):
            
        us = u(h, y)
        #printd("h=%s... y=%s... -> us=%s..." % (str(h)[:100], str(y)[:100], str(us)[:100]))
        goal = -utility_aggregator(us, weights, data_mask)         
        optimizer.zero_grad()
        goal.backward(retain_graph=False)
        optimizer.step()   
        
        #check for convergence:
        if (prev_h-h).abs().max() <= tol:
            printv("[%.2f][ys=%s max_niter=%i tol=%s tol_goal=%s lr=%s u=%s->%s] Finished at %i. iter (tolerance reached): obj=%.4f max-err=%.8f mean-err=%.8f" % 
                    (time.time()-start, tuple(ys.shape), max_niter, tol, tol_goal, lr, u.__name__, utility_aggregator.__name__, 
                    i+1,goal.item(),(prev_h-h).abs().max(),(prev_h-h).abs().mean()))    
            break
        if abs(prev_goal-goal.item()) <= tol_goal:
            printv("[%.2f][ys=%s max_niter=%i tol=%s tol_goal=%s lr=%s u=%s->%s] Finished at %i. iter (objective tolerance reached): obj=%.4f max-err=%.8f mean-err=%.8f" % 
                    (time.time()-start, tuple(ys.shape), max_niter, tol, tol_goal, lr, u.__name__, utility_aggregator.__name__, 
                    i+1,goal.item(),(prev_h-h).abs().max(),(prev_h-h).abs().mean()))    
            break      
        if i>=max_niter-1: 
            printv("[%.2f][ys=%s max_niter=%i tol=%s tol_goal=%s lr=%s u=%s->%s] Finished at %i. iter (max number reached): obj=%.4f max-err=%.8f mean-err=%.8f" % 
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
    


