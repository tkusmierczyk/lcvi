Code necessary to reproduce results from the following paper:

**T. Kuśmierczyk, J. Sakaya, A. Klami: Variational Bayesian Decision-making for Continuous Utilities.** 

-----------------------------------------------------------------------------------------------------------------

### Sources   

Seven files are included:
 - eight_schools.ipynb - Jupter Notebook illustrating VI and LCVI on The Eight Schools model.
 - matrix_factorization.ipynb - Jupter Notebook illustrating VI/LCVI on Matrix Factorization model with Last.fm data.
 - lastfm_data.csv  - CSV files with Last.fm data.
 - losses.py - Python code allowing to calculate losses, utilities and to optimize gains and risks both analytically and numerically.
 - log_utility.py - Python code allowing to construct various approximations of the utility-dependent term and 
   various wrappers to optimize the utility-dependent term w.r.t. h.
 - aux.py / aux_plt.py - Python auxiliary functions for printing, plotting and handling PyTorch tensors.


### Pre-installation Requirements

The code was tested using Python 3.6 from Anaconda 2018.*.
It requires PyTorch 1.0, numpy, pandas, pystan, seaborn, and matplotlib to be preinstalled.

