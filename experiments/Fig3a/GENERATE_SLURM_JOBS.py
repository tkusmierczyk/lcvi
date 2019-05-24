

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

for seed in [0,1,2,3,4,5,6,7,8,9]: #,10,11]:
    code = code_cpu if seed<10 else code_gpu

    name = "MF_VI_SEED%s" % (seed)
    f = open(name+".job", "w")
    f.write(code % (name, name, "matrix_factorization_VI.py SEED=%s" % (seed)))
    f.close()
    print("sbatch %s.job" % name)

    name = "MF_LCVIJO_SEED%s" % (seed)
    f = open(name+".job", "w")
    f.write(code % (name, name, "matrix_factorization_JO.py SEED=%s" % (seed)))
    f.close()
    print("sbatch %s.job" % name)

    name = "MF_LCVIEM10_SEED%s" % (seed)
    f = open(name+".job", "w")
    f.write(code % (name, name, "matrix_factorization_EM.py SEED=%s,H_NSAMPLES_UTILITY_TERM_THETA=10" % (seed)))
    f.close()
    print("sbatch %s.job" % name)


    name = "MF_LCVIEM100_SEED%s" % (seed)
    f = open(name+".job", "w")
    f.write(code % (name, name, "matrix_factorization_EM.py SEED=%s,H_NSAMPLES_UTILITY_TERM_THETA=100" % (seed)))
    f.close()
    print("sbatch %s.job" % name)

    name = "MF_LCVIEM300_SEED%s" % (seed)
    f = open(name+".job", "w")
    f.write(code % (name, name, "matrix_factorization_EM.py SEED=%s,H_NSAMPLES_UTILITY_TERM_THETA=300" % (seed)))
    f.close()
    print("sbatch %s.job" % name)


    name = "MF_LCVIEM5_SEED%s" % (seed)
    f = open(name+".job", "w")
    f.write(code % (name, name, "matrix_factorization_EM.py SEED=%s,H_NSAMPLES_UTILITY_TERM_THETA=5" % (seed)))
    f.close()
    print("sbatch %s.job" % name)

    name = "MF_LCVIEM50_SEED%s" % (seed)
    f = open(name+".job", "w")
    f.write(code % (name, name, "matrix_factorization_EM.py SEED=%s,H_NSAMPLES_UTILITY_TERM_THETA=50" % (seed)))
    f.close()
    print("sbatch %s.job" % name)


    name = "MF_LCVIEMNUM10_SEED%s" % (seed)
    f = open(name+".job", "w")
    f.write(code % (name, name, "matrix_factorization_EMNUM.py SEED=%s,H_NSAMPLES_UTILITY_TERM_THETA=10" % (seed)))
    f.close()
    print("sbatch %s.job" % name)

    name = "MF_LCVIEMNUM100_SEED%s" % (seed)
    f = open(name+".job", "w")
    f.write(code % (name, name, "matrix_factorization_EMNUM.py SEED=%s,H_NSAMPLES_UTILITY_TERM_THETA=100" % (seed)))
    f.close()
    print("sbatch %s.job" % name)

    name = "MF_LCVIEMNUM5_SEED%s" % (seed)
    f = open(name+".job", "w")
    f.write(code % (name, name, "matrix_factorization_EMNUM.py SEED=%s,H_NSAMPLES_UTILITY_TERM_THETA=5" % (seed)))
    f.close()
    print("sbatch %s.job" % name)


    name = "MF_LCVIEMNUM50_SEED%s" % (seed)
    f = open(name+".job", "w")
    f.write(code % (name, name, "matrix_factorization_EMNUM.py SEED=%s,H_NSAMPLES_UTILITY_TERM_THETA=50" % (seed)))
    f.close()
    print("sbatch %s.job" % name)

    name = "MF_LCVIEMNUM300_SEED%s" % (seed)
    f = open(name+".job", "w")
    f.write(code % (name, name, "matrix_factorization_EMNUM.py SEED=%s,H_NSAMPLES_UTILITY_TERM_THETA=300" % (seed)))
    f.close()
    print("sbatch %s.job" % name)




