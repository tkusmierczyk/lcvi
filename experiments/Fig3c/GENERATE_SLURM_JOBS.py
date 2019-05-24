

code_gpu = """#!/bin/bash
#SBATCH --job-name=%s
#SBATCH -o %s.log
#SBATCH -c 1
#SBATCH --mem-per-cpu=12000
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 6:00:00
srun python %s
"""

code_cpu = """#!/bin/bash
#SBATCH --job-name=%s
#SBATCH -o %s.log
#SBATCH -c 1
#SBATCH --mem-per-cpu=8000
#SBATCH -p short
#SBATCH -t 8:00:00
srun python %s 
"""

for seed in [0,1,2,3,4,5,6,7,8,9,10]:


    #percentiles=[1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99] => 
    #loss_threshold_values=0.0002, 0.0047, 0.0187, 0.0742, 0.1651, 0.2884, 0.4372, 0.5997, 0.7583, 0.8908, 0.9753, 0.9949, 0.9999
    PERCENTILE_THRESHOLDS = [0.0742, 0.4372, 0.8908, 1.0] 

    for scale in PERCENTILE_THRESHOLDS:
        name = "linearized%s_SEED%s" % (scale, seed)
        f = open(name+".job", "w")
        f.write(code_cpu % (name, name, "matrix_factorization.py LOSS=[expsquared],UTIL=[expsquared],UTILITY_TERM=[linearized],M=%s,SEED=%s" % (scale, seed)))
        f.close()
        print("sbatch %s.job" % name)


    for utility_term in ["vi", "naive"]:                    
        name = "%s_SEED%s" % (utility_term, seed)
        f = open(name+".job", "w")
        f.write(code_cpu % (name, name, "matrix_factorization.py LOSS=[expsquared],UTIL=[expsquared],UTILITY_TERM=[%s],SEED=%s" % (utility_term, seed)))
        f.close()
        print("sbatch %s.job" % name)

