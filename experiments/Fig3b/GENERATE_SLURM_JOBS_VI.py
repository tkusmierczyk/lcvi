
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


for seed in [0,1,2,3,4,5,6,7,8,9]:

    name = "MF_VI_%s" % (seed)
    f = open(name+".job", "w")
    f.write(code_cpu % (name, name, "matrix_factorization_vi.py SEED=%s" % (seed)))
    f.close()
    print("sbatch %s.job" % name)

