

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

for seed in [0,1,2,3,4,5,6,7,8,9]:

  name = "lcvi_q02_SEED%s" % (seed)
  f = open(name+".job", "w")
  f.write(code_cpu % (name, name, "matrix_factorization.py LOSS=[tilted],TILTED_Q=0.2,UTIL=[exp],UTILITY_TERM=[naive],SEED=%s,GAMMA=1.62" % (seed)))
  f.close()
  print("sbatch %s.job" % name)

  name = "lcvi_q05_SEED%s" % (seed)
  f = open(name+".job", "w")
  f.write(code_cpu % (name, name, "matrix_factorization.py LOSS=[tilted],TILTED_Q=0.5,UTIL=[exp],UTILITY_TERM=[naive],SEED=%s,GAMMA=1.0" % (seed)))
  f.close()
  print("sbatch %s.job" % name)

  name = "lcvi_q08_SEED%s" % (seed)
  f = open(name+".job", "w")
  f.write(code_cpu % (name, name, "matrix_factorization.py LOSS=[tilted],TILTED_Q=0.8,UTIL=[exp],UTILITY_TERM=[naive],SEED=%s,GAMMA=1.62" % (seed)))
  f.close()
  print("sbatch %s.job" % name)

  name = "lcvi_squared_SEED%s" % (seed)
  f = open(name+".job", "w")
  f.write(code_cpu % (name, name, "matrix_factorization.py LOSS=[squared],UTIL=[exp],UTILITY_TERM=[naive],SEED=%s,GAMMA=0.3" % (seed)))
  f.close()
  print("sbatch %s.job" % name)


  name = "vi_q02_SEED%s" % (seed)
  f = open(name+".job", "w")
  f.write(code_cpu % (name, name, "matrix_factorization.py LOSS=[tilted],TILTED_Q=0.2,UTIL=[exp],UTILITY_TERM=[vi],SEED=%s" % (seed)))
  f.close()
  print("sbatch %s.job" % name)

  name = "vi_q05_SEED%s" % (seed)
  f = open(name+".job", "w")
  f.write(code_cpu % (name, name, "matrix_factorization.py LOSS=[tilted],TILTED_Q=0.5,UTIL=[exp],UTILITY_TERM=[vi],SEED=%s" % (seed)))
  f.close()
  print("sbatch %s.job" % name)

  name = "vi_q08_SEED%s" % (seed)
  f = open(name+".job", "w")
  f.write(code_cpu % (name, name, "matrix_factorization.py LOSS=[tilted],TILTED_Q=0.8,UTIL=[exp],UTILITY_TERM=[vi],SEED=%s" % (seed)))
  f.close()
  print("sbatch %s.job" % name)

  name = "vi_squared_SEED%s" % (seed)
  f = open(name+".job", "w")
  f.write(code_cpu % (name, name, "matrix_factorization.py LOSS=[squared],UTIL=[exp],UTILITY_TERM=[vi],SEED=%s" % (seed)))
  f.close()
  print("sbatch %s.job" % name)



