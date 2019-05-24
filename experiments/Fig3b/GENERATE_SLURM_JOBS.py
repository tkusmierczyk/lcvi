
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
#SBATCH -t 4:00:00
srun python %s 
"""

PERCENTILE_THRESHOLD = [0.0002, 0.0052, 0.0207, 0.0841, 0.1945, 0.3603, 0.5962, 0.9291, 1.4077, 2.1528, 3.5538, 5.0556, 8.76, 5, 10, 50, 100, 500, 1000, 10000, 1, 0.1, 0.5, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005, 0.00001]
for seed in [0,1,2,3,4,5,6,7,8,9]:


    for scale in PERCENTILE_THRESHOLD:
        name = "MF_LIN_%s_%s" % (scale, seed)
        f = open(name+".job", "w")
        f.write(code_cpu % (name, name, "matrix_factorization_lin.py M=%s,SEED=%s" % (scale, seed)))
        f.close()
        print("sbatch %s.job" % name)



    for scale in PERCENTILE_THRESHOLD:
        name = "MF_EXP_%s_%s" % (scale, seed)
        f = open(name+".job", "w")
        f.write(code_cpu % (name, name, "matrix_factorization_exp.py M=%s,SEED=%s" % (scale, seed)))
        f.close()
        print("sbatch %s.job" % name)

