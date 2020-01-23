# Variational Bayesian Decision-making for Continuous Utilities

-----------------------------------------------------------------------------------------------------------------

### Publication

The code was used in and allows to reproduce results from the following paper:

**T. Kuśmierczyk, J. Sakaya, A. Klami: Variational Bayesian Decision-making for Continuous Utilities.** NeurIPS 2019. [(arXiv preprint)](https://arxiv.org/pdf/1902.00792.pdf) [(poster)](poster.pdf)


### Abstract

Bayesian decision theory outlines a rigorous framework for making optimal decisions based on maximizing expected utility over a model posterior. However, practitioners often do not have access to the full posterior and resort to approximate inference strategies. In such cases, taking the eventual decision-making task into account while performing the inference allows for calibrating the posterior approximation to maximizethe utility. We present an automatic pipeline that co-opts continuous utilities into variational inference algorithms to account for decision-making. We provide practical strategies for approximating and maximizing gain, and empirically demonstrate consistent improvement when calibrating approximations for specific utilities.


### Main files 
 - [eight_schools.ipynb](eight_schools.ipynb) - Jupter Notebook illustrating VI, LCVI and LCVI-EM on The Eight Schools model.
 - [matrix_factorization.ipynb](matrix_factorization.ipynb) - Jupter Notebook illustrating VI/LCVI/LCVI-EM on Matrix Factorization model with Last.fm data.


### Additional files 
 
 - [evaluation.py](evaluation.py) - Calculation of empirical risks and gains
 - [losses.py](losses.py) - Python code allowing to calculate losses and utilities 
 - [optimal_decisions.py](optimal_decisions.py) - Code to optimize w.r.t. decisions, both in closed-forms and numerically (M-step).
 - [numerical_optimization.py](numerical_optimization.py) - Numerical optimization of decisions 
 - [utility_term_estimation.py](utility_term_estimation.py) - Python code allowing to construct various approximations of the utility-dependent term.
 - aux.py / aux_time.py / aux_plt.py - Auxiliary functions for printing, plotting, measuring time and handling PyTorch tensors.


### Data

Last.fm data can be found in *data* directory. The main file is lastfm_data.csv. 
The data was extracted using *data/lastfm_data_preparation.ipynb*.


### Pre-installation Requirements

The code was tested using Python 3.6 from Anaconda 2018.*.
It requires PyTorch 1.0, numpy, pandas, pystan, seaborn, and matplotlib to be preinstalled.


### Experiments

Experiments are placed in *experiments* directory.
Additional details can be found in *experiments/README.txt*.
For each of the experiments, we provide the intermediate results that 
were used to create plots using REPORT*.ipynb Jupyter notebooks. 
Thanks to that, additional plots/processing can be achieved 
without running full experiments.
