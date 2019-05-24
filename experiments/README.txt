The experiments were carried out on the cluster managed with Slurm Workload Manager (https://en.wikipedia.org/wiki/Slurm_Workload_Manager).
First, for each experiment Slurm jobs were generated (and then, run) using GENERATE_SLURM_JOBS*.py script.
Each of the jobs produces one or more *.csv files. 
These *.csv files were merged into a single RESULTS.csv file using the following command:
grep "" *.csv > RESULTS.csv
and then placed into newly created RESULTS directory.
Final plots were generated using appropriate REPORT*.ipynb Jupyter notebooks 
that load RESULTS.csv from RESULTS directory.
For Fig2 additional *.pickle files are generated that need to be placed in RESULTS_PICKLES directory.
