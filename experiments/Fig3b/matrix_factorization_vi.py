#!/usr/bin/env python
# coding: utf-8

# # Matrix Factorization using Last.fm data

# In[1]:


#get_ipython().system('pip install torch seaborn matplotlib pandas numpy')


# In[2]:
import sys
sys.path.append("../../.")

import math
import time
import sys
import pickle

import numpy as np
import pandas as pd

import torch
from torch.distributions import Normal
from torch.nn.functional import softplus


# In[3]:


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


# In[4]:


from aux import print2, tonumpy, tonumpy2, flatten_first_two_dims, assert_valid, dict2str, parse_script_args
import aux_time


# In[5]:


import losses
import utility_term_estimation
import optimal_decisions
import evaluation 


# In[6]:


from importlib import reload  
reload(losses)
reload(evaluation)
reload(utility_term_estimation)
reload(optimal_decisions)


# # Configuration

# To run standard VI change UTILITY_TERM to "vi".

# In[7]:


args = parse_script_args()


# In[8]:


# optimization general parmeters
SEED = args.get("SEED", 0)
NITER  = 30001 # number of iterations - around 70k is the right number for mininbatch=10
LR = 0.01 # 0.1 is the right number for the full batch, 0.001 advised for mininbatch=10
MINIBATCH = 100 # how many rows of the matrix in minibatch

# model
K = 20 # number of latent variables

# number of samples used to approximate ELBO term
NSAMPLES = 11


# In[9]:


# selected loss: tilted/squared 
#  you can also put "exptilted" (that is in [0,1]), but then fix M=1.25
#  and UTILITY_TERM_SCALE=1/1.25 for linearized and UTILITY_TERM_SCALE=1 otherwise
#  also change FIND_H=num-util and EVAL_SKIP_GAIN=False to get an appropriate evaluation
LOSS = args.get("LOSS", "squared")
TILTED_Q = args.get("TILTED_Q", 0.2)
# loss-to-utility transformation (bigM: utility=M-loss / id: utility=-loss)
UTIL = args.get("UTIL", "exp")
M = args.get("M", 1.0)
GAMMA = 2.0/M

# In[10]:


# constructing an approximation to the utility-dependent term:
UTILITY_TERM = args.get("UTILITY_TERM", "vi") # vi/naive/linearized/jensen
# the utility-dependent term is multiplied by this value: 
#  should be 1/M for linearized and M for the others
UTILITY_TERM_SCALE = args.get("UTILITY_TERM_SCALE", 1.0)  #for linearized we use U~=-risk 
# how many samples of latent variables
NSAMPLES_UTILITY_TERM_THETA = 300
# how many samples of y for each latent variable
NSAMPLES_UTILITY_TERM_Y = 1


# In[11]:


# how often to evaulate (=report losses and (optionally) gains)
EVAL_NITER = 30000

# Usually to evaluate risks we rely on Bayes estimators whereas for gains we use numerical optimization.
GAIN_OPTIMAL_H_NUMERICALLY = False #True!
RISK_OPTIMAL_H_NUMERICALLY = False

# evaluation parameters (these numbers should be sufficently large 
# if we want to trust our evaluation):
#  number of samples of latent variables
EVAL_NSAMPLES_UTILITY_TERM_THETA = 1000
#  number of samples of y for each latent variable
EVAL_NSAMPLES_UTILITY_TERM_Y = 1
# parameters of the numerical optimization w.r.t. h used in evaluation
EVAL_MAX_NITER = 20000
EVAL_SGD_PREC = 0.0001


# In[12]:


print("CONFIGURATION SUMMARY: %s" % dict2str(globals()))


# # Data

# In[13]:


def lastfm_data(N=1000, D=100, url="../../data/lastfm_data.csv"):    
    """LastFm views."""
    df = pd.read_csv(url, header=None)
    x_ = df.values #users in rows, artists in colums
    x_ = x_[:N, :D]
    N, D = x_.shape

    x_ = np.log(1+x_)
    mask_ = np.ones( (N,D), dtype=bool) #non-missing values
    return x_, mask_


# In[14]:


# Prepares data to work with pytorch
x_, mask_ = lastfm_data()
x    = torch.tensor(x_, dtype = torch.float, requires_grad=False)
mask = torch.tensor(mask_.astype(np.uint8)).type(env.ByteTensor) # non-missing values
N, D = x.shape


# ## Training vs. test split

# In[15]:


def random_mask(x, testing_prob=0.5, seed=123): 
    np.random.seed(seed)
    N, D = x.shape
    testing_mask = np.random.choice([0, 1], (N, D), (1.0-testing_prob, testing_prob))
    training_mask = np.ones((N, D))-testing_mask
    return training_mask, testing_mask


# In[16]:


# Prepares masks to work with pytorch
training_mask, testing_mask = random_mask(x)
training_mask = torch.tensor((training_mask*mask_).astype(np.uint8)).type(env.ByteTensor)
testing_mask  = torch.tensor((testing_mask*mask_).astype(np.uint8)).type(env.ByteTensor)
N, D = training_mask.shape


# # Model

# In[17]:


def jacobian_softplus(x):
       return 1.0/(1.0 + torch.exp(-x))
    
    
def model_log_prob(x, w, z, mask=None, sgd_scale=1.0):
    if mask is None: mask = torch.ones_like(x).type(env.ByteTensor)
    
    xhat = z.matmul(w)
    likelihood = Normal(xhat, 1.) 
    prior = Normal(0, 10)
    assert likelihood.loc.shape[1: ] == x.shape
    
    return torch.masked_select(likelihood.log_prob(x), mask).sum()*sgd_scale             + prior.log_prob(w).sum() + prior.log_prob(z).sum()*sgd_scale


# ## Sampling from predictive posterior distribution

# In[18]:


def sample_predictive_y0(qw, qz, nsamples_theta, nsamples_y):  
    """ Returns a tensor with samples     
        (nsamples_y samples of y for each theta x 
         nsamples_theta samples of latent variables)."""
    w = qw.rsample(torch.Size([nsamples_theta]))
    z = qz.rsample(torch.Size([nsamples_theta]))    
    
    xhat = z.matmul(w)
    likelihood = Normal(xhat, 1.)
    y_samples = likelihood.rsample(torch.Size([nsamples_y]))
    return y_samples


def sample_predictive_y(qw, qz, nsamples_theta, nsamples_y):  
    """ Returns a tensor with samples (nsamples_y x nsamples_theta).
        Flattents the first two dimensions 
        (samples of y for different thetas) from sample_predictive_y0.
    """
    return flatten_first_two_dims(sample_predictive_y0(qw, qz, nsamples_theta, nsamples_y))


# # Constructing losses and utilities

# In[19]:


# mask used to select points to the utility-dependent term: use only training data
utility_term_mask = training_mask 
 
loss, optimal_h_bayes_estimator = losses.LossFactory(**globals()).create(LOSS)
print2("> <%s> loss: %s with (analytical/Bayes estimator) h: %s" % 
        (LOSS, loss.__name__, optimal_h_bayes_estimator.__name__))
        
u = losses.UtilityFactory(**globals()).create(UTIL, loss)
print2("> utility: %s" % u.__name__)            


# In[20]:


utility_term_factory = utility_term_estimation.UtilityAggregatorFactory()


# # Evaluation

# In[21]:


train_measures = evaluation.Measures(
                 x, loss, u, 
                 sample_predictive_y, 
                 optimal_h_bayes_estimator=optimal_h_bayes_estimator,
                 y_mask=training_mask,
                 GAIN_OPTIMAL_H_NUMERICALLY = GAIN_OPTIMAL_H_NUMERICALLY,
                 RISK_OPTIMAL_H_NUMERICALLY = RISK_OPTIMAL_H_NUMERICALLY,
                 EVAL_NSAMPLES_UTILITY_TERM_THETA = EVAL_NSAMPLES_UTILITY_TERM_THETA,
                 EVAL_NSAMPLES_UTILITY_TERM_Y = EVAL_NSAMPLES_UTILITY_TERM_Y,
                 EVAL_MAX_NITER = EVAL_MAX_NITER,
                 EVAL_SGD_PREC = EVAL_SGD_PREC) 


test_measures = evaluation.Measures(
                 x, loss, u, 
                 sample_predictive_y, 
                 optimal_h_bayes_estimator=optimal_h_bayes_estimator,
                 y_mask=testing_mask,
                 GAIN_OPTIMAL_H_NUMERICALLY = GAIN_OPTIMAL_H_NUMERICALLY,
                 RISK_OPTIMAL_H_NUMERICALLY = RISK_OPTIMAL_H_NUMERICALLY,
                 EVAL_NSAMPLES_UTILITY_TERM_THETA = EVAL_NSAMPLES_UTILITY_TERM_THETA,
                 EVAL_NSAMPLES_UTILITY_TERM_Y = EVAL_NSAMPLES_UTILITY_TERM_Y,
                 EVAL_MAX_NITER = EVAL_MAX_NITER,
                 EVAL_SGD_PREC = EVAL_SGD_PREC) 


# # Minibatch optimization

# In[22]:


def yield_minibatch_rows(i, N, MINIBATCH):
    """ Minibatch optimization via rows subset selection.
    
        Args:
          i  Iteration number 0,1,2,...
    """
    if MINIBATCH>N: MINIBATCH=N

    nbatches_per_epoch = int( np.ceil(N/MINIBATCH) )
    batch_no = i%nbatches_per_epoch    
    if batch_no==0: # shuffle order
        yield_minibatch_rows.rows_order = np.random.permutation(range(N))
    six, eix = batch_no*MINIBATCH, (batch_no+1)*MINIBATCH
    rows = yield_minibatch_rows.rows_order[six: eix] # batch rows
    
    # makes sure that for full-batch the order of rows is preserved
    if MINIBATCH>=N: rows = list(range(N)) 
      
    sgd_scale = N/len(rows) 
    epoch_no = i//nbatches_per_epoch
    return rows, epoch_no, sgd_scale
yield_minibatch_rows.rows_order = None  


# # LCVI

# In[ ]:


utility_term = utility_term_factory.create(UTILITY_TERM)
print2("> utility_term: %s" % utility_term.__name__)  


# In[ ]:


torch.manual_seed(SEED)

qz_loc = torch.randn([N, K], requires_grad=True)
qz_scale = torch.randn([N, K], requires_grad=True)
qw_loc = torch.randn([K, D], requires_grad=True)
qw_scale = torch.randn([K, D], requires_grad=True)

h = torch.randn((N, D), requires_grad=True)

optimizer = torch.optim.Adam([qw_loc, qw_scale, qz_loc, qz_scale, h], lr=LR) 

report = []


# In[ ]:


start = time.time()    
for i in range(NITER):
    with aux_time.AccumulatingTimer("VI"):
    
        rows, epoch_no, sgd_scale = yield_minibatch_rows(i, N, MINIBATCH)
        if i<20 or i%500==0: print2("[%.2fs] %i. iteration, %i. epoch" % (time.time()-start, i, epoch_no))

        #######################################################        
        # preparation: selecting minibatch rows

        qz_loc0 = qz_loc[rows, :]
        qz_scale0 = qz_scale[rows, :]    
        qw = Normal(qw_loc, softplus(qw_scale))
        qz = Normal(qz_loc0, softplus(qz_scale0))

        h0 = h[rows,:]
        x0 = x[rows,:]
        training_mask0 = training_mask[rows, :]
        utility_term_mask0 = utility_term_mask[rows, :] 

        #######################################################
        # optimization

        w = qw.rsample(torch.Size([NSAMPLES]))
        z = qz.rsample(torch.Size([NSAMPLES]))
        elbo = model_log_prob(x0, w, z, training_mask0, sgd_scale).sum()                         -qw.log_prob(w).sum() -qz.log_prob(z).sum()*sgd_scale 
        elbo = elbo/NSAMPLES

        ys = sample_predictive_y0(qw, qz, 
                                  nsamples_theta=NSAMPLES_UTILITY_TERM_THETA,
                                  nsamples_y=NSAMPLES_UTILITY_TERM_Y)        

        optimizer.zero_grad()            
        objective = -elbo 
        objective.backward(retain_graph=False)
        optimizer.step()    
    
    #######################################################
    # reporting
    
    if (i>0) and (i%EVAL_NITER==0 or i==NITER-1): #report results
        print("evaluating (performed on the full data batch; may take some time)...")        
        qw = Normal(qw_loc, softplus(qw_scale))
        qz = Normal(qz_loc, softplus(qz_scale))
            
        rmse = (qz.rsample(torch.Size([])).mm(qw.rsample(torch.Size([])))-x)**2
        training_rmse = torch.masked_select(rmse, training_mask).mean()
        testing_rmse = torch.masked_select(rmse, testing_mask).mean() 
            
        r = (aux_time.AccumulatingTimer.get_elapsed("VI"),
                 utility_term.__name__, SEED, i, epoch_no,
                 elbo.item(), 
                 training_rmse.item(), 
                 train_measures.qrisk(qw, qz).item(), 
                 train_measures.empirical_risk(qw, qz).item(),
                 train_measures.qgain(qw, qz).item(), 
                 train_measures.empirical_gain(qw, qz).item(),
                 testing_rmse.item(),  
                 test_measures.qrisk(qw, qz).item(),  
                 test_measures.empirical_risk(qw, qz).item(),
                 test_measures.qgain(qw, qz).item(),  
                 test_measures.empirical_gain(qw, qz).item(),
                 qw_loc.mean().item(), softplus(qw_scale).mean().item(), 
                 qz_loc.mean().item(), softplus(qz_scale).mean().item(),
                 qw_loc.std().item(),  softplus(qw_scale).std().item(), 
                 qz_loc.std().item(),  softplus(qz_scale).std().item())        
        report.append(r) # append to report            
        print("[%.1fs][%s][%s][%s][%s] elbo: %.2f;   TRAINING: rmse:%.4f  qRisk:%.4f eRisk:%.4f \t\t\t                 qGain:%.4f eGain:%.4f;  \t\t\t TEST: rmse:%.4f  qRisk: %4f eRisk: %.4f                 qGain:%.4f eGain:%.4f  qw:%.3f+/-%.3f qz:%.3f+/-%.3f" % (r[:20]))                                                               
        
        assert_valid(qw.loc); 
        assert_valid(qz.loc); 
        assert_valid(qw.scale); 
        assert_valid(qz.scale);  
        
        LOSSSTR = (("%s%s" % (LOSS, TILTED_Q)) if "tilted" in LOSS else LOSS)
        UTILSTR = "%s%s" % (UTIL, GAMMA)
        OUT = "MF_VI_%i_%s_%s" % (SEED, LOSSSTR, UTILSTR)
        pd.DataFrame(report).to_csv(OUT+".csv", header=False, index=False)




