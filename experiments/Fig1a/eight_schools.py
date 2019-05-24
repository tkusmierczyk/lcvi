#!/usr/bin/env python
# coding: utf-8

# #  The Eight Schools model illustration

# In[166]:


#get_ipython().system('pip install torch pystan seaborn matplotlib pandas numpy')


# In[167]:
import sys
sys.path.append("../../.")

import torch
from torch.nn.functional import softplus
from torch.distributions import Normal, Cauchy


# In[168]:


if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    env = torch.cuda
    device = torch.device('cuda')
    print("Using GPU")
else:
    torch.set_default_tensor_type('torch.FloatTensor')
    env = torch
    device = torch.device('cpu')
    print("Using CPU")


# In[169]:


import sys
import time
import pandas as pd
import numpy as np
import random
np.random.seed(seed=0)
random.seed(0)


# In[170]:


from aux import print2, tonumpy, tonumpy2, flatten_first_two_dims, parse_script_args


# In[171]:


import losses
import utility_term_estimation
import optimal_decisions
import evaluation 


# In[172]:


from importlib import reload  # Python 3.4+ only.
reload(losses)
reload(evaluation)
reload(utility_term_estimation)
reload(optimal_decisions)


# # Configuration

# In[173]:


args = parse_script_args()


# In[174]:


# optimization general parmeters
SEED = args.get("SEED", 123)
NITER  = 30001 # number of iterations - around 20k is advised
LR = 0.05

# number of samples used to approximate ELBO term
NSAMPLES = 11


# In[175]:


# selected loss: tilted/squared
LOSS = "tilted"
TILTED_Q = 0.2 
# loss-to-utility transformation (bigM: utility=M-loss / id: utility=-loss)
UTIL =  "bigM"
M = (10000.0 if UTIL.lower()=="bigm" else 1.0)


# In[176]:


# constructing an approximation to the utility-dependent term:
UTILITY_TERM = "linearized" # vi/naive/linearized/jensen
# the utility-dependent term is multiplied by this value: 
#  should be 1/M for linearized and M for the others
UTILITY_TERM_SCALE = 1.0/7.0 # 1.0 / ~90% percentile of losses for VI
#   ( if "linearized" in UTILITY_TERM else M) #for linearized we use U~=-risk 

# how many samples of latent variables
NSAMPLES_UTILITY_TERM_THETA = 123
# how many samples of y for each latent variable
NSAMPLES_UTILITY_TERM_Y = 11
# update gradients of the utility term in every x iterations
UTILITY_TERM_EVERY_NITER = 1


# In[177]:


# Evaluation configuration

# how often to evaulate (=report losses and (optionally) gains)
EVAL_NITER = 1000

# Usually to evaluate risks we rely on Bayes estimators whereas for gains we use numerical optimization.
GAIN_OPTIMAL_H_NUMERICALLY = False #True!
RISK_OPTIMAL_H_NUMERICALLY = False

# evaluation parameters (these numbers should be sufficently large 
# if we want to trust our evaluation):
#  number of samples of latent variables
EVAL_NSAMPLES_UTILITY_TERM_THETA = 10000
#  number of samples of y for each latent variable
EVAL_NSAMPLES_UTILITY_TERM_Y = 1
# parameters of the numerical optimization w.r.t. h used in evaluation
EVAL_MAX_NITER = 10000
EVAL_SGD_PREC = 0.0001


# # Data

# In[178]:


schools_dat = {'J': 8,
               'y': [28,  8, -3,  7, -1,  1, 18, 12],
               'sigma': [15, 10, 16, 11,  9, 11, 10, 18]}


# # Model definition

# In[179]:


def jacobian_softplus(x):
       return 1.0/(1.0 + torch.exp(-x))

    
def model_log_prob(data, mu, tau, theta):    
    y = torch.tensor(data["y"], dtype=torch.float32)
    sigma = torch.tensor(data["sigma"], dtype=torch.float32)
    taup = softplus(tau)
    
    lik = Normal(theta, sigma)
    prior_theta = Normal(mu, taup)
    prior_mu = Normal(0.0, 5.0)
    prior_tau = Cauchy(0.0, 5.0)  #half-Cauchy (=2*pdf of Cauchy)
    
    return lik.log_prob(y).sum()             + prior_theta.log_prob(theta).sum()             + prior_mu.log_prob(mu)             + torch.tensor(2.0, dtype=torch.float32).log()+prior_tau.log_prob(taup)             + torch.log(jacobian_softplus(tau))


# In[180]:


def sample_predictive_y0(data, q_theta, nsamples_theta, nsamples_y):
    """ Returns a tensor with samples     
        (nsamples_y samples of y for each theta x 
         nsamples_theta samples of latent variables).
    """
    sigma = torch.tensor(data["sigma"], dtype=torch.float32)
    
    theta =  q_theta.rsample(torch.Size([nsamples_theta]))
    ys = Normal(theta, sigma).rsample(torch.Size([nsamples_y]))    
    return ys


def sample_predictive_y(data, q_theta, nsamples_theta, nsamples_y):
    """ Returns a tensor with samples (nsamples_y x nsamples_theta).
        Flattents the first two dimensions 
        (samples of y for different thetas) from sample_predictive_y0.
    """    
    ys = sample_predictive_y0(data, q_theta, nsamples_theta, nsamples_y)
    return flatten_first_two_dims(ys)


# # Constructing losses and utilities

# In[181]:


# include all (training) data points in utility-dependent term
utility_term_mask = torch.ones(schools_dat["J"]).type(env.ByteTensor) 

loss, optimal_h_bayes_estimator = losses.LossFactory(**globals()).create(LOSS)
print2("> <%s> loss: %s with (analytical/Bayes estimator) h: %s" % 
        (LOSS, loss.__name__, optimal_h_bayes_estimator.__name__))
        
u = losses.UtilityFactory(**globals()).create(UTIL, loss)
print2("> utility: %s" % u.__name__)            


# In[182]:


utility_term_factory = utility_term_estimation.UtilityAggregatorFactory()


# # Evaluation

# In[183]:


measures = evaluation.Measures(
                 torch.tensor(schools_dat["y"], dtype=torch.float32), 
                 loss, u, sample_predictive_y, 
                 optimal_h_bayes_estimator=optimal_h_bayes_estimator,
                 y_mask=utility_term_mask, # all data points
                 GAIN_OPTIMAL_H_NUMERICALLY = GAIN_OPTIMAL_H_NUMERICALLY,
                 RISK_OPTIMAL_H_NUMERICALLY = RISK_OPTIMAL_H_NUMERICALLY,
                 EVAL_NSAMPLES_UTILITY_TERM_THETA = EVAL_NSAMPLES_UTILITY_TERM_THETA,
                 EVAL_NSAMPLES_UTILITY_TERM_Y = EVAL_NSAMPLES_UTILITY_TERM_Y,
                 EVAL_MAX_NITER = EVAL_MAX_NITER,
                 EVAL_SGD_PREC = EVAL_SGD_PREC) 


# # VI

# In[136]:


# intialization
torch.manual_seed(SEED)
np.random.seed(SEED)

q_mu_loc = torch.randn((1), requires_grad=True)
q_mu_scale = torch.ones((1), requires_grad=True)
q_tau_loc = torch.randn((1), requires_grad=True) 
q_tau_scale = torch.tensor((1.0), requires_grad=True)
q_theta_loc = torch.randn((schools_dat["J"]), requires_grad=True)
q_theta_scale = torch.ones((schools_dat["J"]), requires_grad=True)


# In[137]:


optimizer = torch.optim.Adam([q_mu_loc, q_mu_scale, q_tau_loc, q_tau_scale, 
                                q_theta_loc, q_theta_scale], lr=LR)


# In[138]:


start = time.time()
report = []
losses = []
trajectory_vi = {}
for i in range(NITER):    
    q_mu = Normal(q_mu_loc, softplus(q_mu_scale))    
    q_tau = Normal(q_tau_loc, softplus(q_tau_scale))    
    q_theta = Normal(q_theta_loc, softplus(q_theta_scale))    
    
    elbo = 0.0
    for _ in range(NSAMPLES):
        mu =  q_mu.rsample()               
        tau =  q_tau.rsample()               
        theta =  q_theta.rsample()               

        elbo += model_log_prob(schools_dat, mu, tau, theta)                 -q_mu.log_prob(mu).sum() -q_tau.log_prob(tau).sum() -q_theta.log_prob(theta).sum()
    elbo = elbo/NSAMPLES            
    
    optimizer.zero_grad()    
    (-elbo).backward()
    optimizer.step()

    if i%EVAL_NITER==0 or i==NITER-1:
        
        # store losses
        ys = sample_predictive_y(schools_dat, q_theta, nsamples_theta=100, nsamples_y=1)        
        h = optimal_h_bayes_estimator(ys)
        losses.append(loss(h,ys))        
        
        # store trajectories of posteriors
        trajectory_vi.setdefault("qmu_loc",[]).append(tonumpy2(q_mu_loc)) 
        trajectory_vi.setdefault("qmu_scale",[]).append(tonumpy2(softplus(q_mu_scale)))         
        trajectory_vi.setdefault("qtau_loc",[]).append(tonumpy2(softplus(q_tau_loc))) 
        trajectory_vi.setdefault("qtau_scale",[]).append(tonumpy2(softplus(q_tau_scale))) 
        trajectory_vi.setdefault("qtheta_loc",[]).append(tonumpy2(q_theta_loc) ) 
        trajectory_vi.setdefault("qtheta_scale",[]).append(tonumpy2(softplus(q_theta_scale)))                 
        
        print("iter/epoch %i. evaluating..." % i)        
        rmse = (tonumpy(q_theta_loc)-np.array(schools_dat['y']))**2
        training_rmse = rmse.mean()
        r = (time.time()-start, "VI", SEED, i, elbo.item(), 
                 training_rmse, 
                 measures.qrisk(schools_dat, q_theta).item(), 
                 measures.empirical_risk(schools_dat, q_theta).item(),
                 measures.qgain(schools_dat, q_theta).item(), 
                 measures.empirical_gain(schools_dat, q_theta).item(),
                 softplus(q_tau_loc).item(), softplus(q_tau_scale).item(),
                 q_mu_loc.item(), softplus(q_mu_scale).item() )
        report.append(r)         
        print2(("""[%.1fs][%s][%s][%s] elbo: %.2f;  training rmse:%.4f  qRisk:%.4f eRisk:%.4f 
                  qGain:%.4f eGain:%.4f""" % (r[:10])) + " q_tau_loc: %.2f" % softplus(q_tau_loc).item())     
        
        LOSSSTR = (("%s%s" % (LOSS, TILTED_Q)) if "tilted" in LOSS else LOSS)
        OUT = "8s_VI_%i_%s_%s" % (SEED, LOSSSTR, UTIL)
        pd.DataFrame(report).to_csv(OUT+".csv", header=False, index=False)         



# # LCVI

# In[184]:


utility_term = utility_term_factory.create(UTILITY_TERM)
print2("> utility_term: %s" % utility_term.__name__)  


# In[185]:


# intialization
torch.manual_seed(SEED)
np.random.seed(SEED)

q_mu_loc = torch.randn((1), requires_grad=True)
q_mu_scale = torch.ones((1), requires_grad=True)
q_tau_loc = torch.randn((1), requires_grad=True) 
q_tau_scale = torch.tensor((1.0), requires_grad=True)
q_theta_loc = torch.randn((schools_dat["J"]), requires_grad=True)
q_theta_scale = torch.ones((schools_dat["J"]), requires_grad=True)

h = torch.randn((schools_dat['J']), requires_grad=True)


# In[186]:


optimizer = torch.optim.Adam([q_mu_loc, q_mu_scale, q_tau_loc, q_tau_scale, 
                              q_theta_loc, q_theta_scale, h], lr=LR) 


# In[ ]:


start = time.time()
report = []
trajectory_lcvi = {}
for i in range(NITER):    

    q_mu = Normal(q_mu_loc, softplus(q_mu_scale))    
    q_tau = Normal(q_tau_loc, softplus(q_tau_scale))    
    q_theta = Normal(q_theta_loc, softplus(q_theta_scale))    
    
    elbo = 0.0
    for _ in range(NSAMPLES):
        mu =  q_mu.rsample()               
        tau =  q_tau.rsample()               
        theta =  q_theta.rsample()               

        elbo += model_log_prob(schools_dat, mu, tau, theta)                 -q_mu.log_prob(mu).sum() -q_tau.log_prob(tau).sum() -q_theta.log_prob(theta).sum()
    elbo = elbo/NSAMPLES    
                             
    ys = sample_predictive_y0(schools_dat, q_theta, 
                                    nsamples_theta=NSAMPLES_UTILITY_TERM_THETA,
                                    nsamples_y=NSAMPLES_UTILITY_TERM_Y)        
      
    optimizer.zero_grad()            
    objective = -elbo -utility_term(u(h,ys), utility_term_mask)*UTILITY_TERM_SCALE
    objective.backward(retain_graph=False)
    optimizer.step()
        
    if i%EVAL_NITER==0 or i==NITER-1: #report results
        
        #store trajectories of posteriors
        trajectory_lcvi.setdefault("qmu_loc",[]).append(tonumpy2(q_mu_loc)) 
        trajectory_lcvi.setdefault("qmu_scale",[]).append(tonumpy2(softplus(q_mu_scale)))         
        trajectory_lcvi.setdefault("qtau_loc",[]).append(tonumpy2(softplus(q_tau_loc))) 
        trajectory_lcvi.setdefault("qtau_scale",[]).append(tonumpy2(softplus(q_tau_scale))) 
        trajectory_lcvi.setdefault("qtheta_loc",[]).append(tonumpy2(q_theta_loc)) 
        trajectory_lcvi.setdefault("qtheta_scale",[]).append(tonumpy2(softplus(q_theta_scale)))         
                
        print("iter/epoch %i. evaluating..." % i)
        rmse = (tonumpy(q_theta_loc)-np.array(schools_dat['y']))**2
        training_rmse = rmse.mean()
        r = (time.time()-start, utility_term.__name__, SEED, i, elbo.item(), 
                 training_rmse, 
                 measures.qrisk(schools_dat, q_theta).item(), 
                 measures.empirical_risk(schools_dat, q_theta).item(),
                 measures.qgain(schools_dat, q_theta).item(), 
                 measures.empirical_gain(schools_dat, q_theta).item(),
                 softplus(q_tau_loc).item(), softplus(q_tau_scale).item(),
                 q_mu_loc.item(), softplus(q_mu_scale).item() )
        report.append(r)    
        print2(("""[%.1fs][%s][%s][%s] elbo: %.2f;  training rmse:%.4f  qRisk:%.4f eRisk:%.4f 
                  qGain:%.4f eGain:%.4f""" % (r[:10])) + " q_tau_loc: %.2f" % softplus(q_tau_loc).item())        
        
        LOSSSTR = (("%s%s" % (LOSS, TILTED_Q)) if "tilted" in LOSS else LOSS)
        OUT = "8s_LCVI_%s%s_%i_%s_%s" % (UTILITY_TERM, UTILITY_TERM_SCALE, SEED, LOSSSTR, UTIL)
        pd.DataFrame(report).to_csv(OUT+".csv", header=False, index=False)
        

