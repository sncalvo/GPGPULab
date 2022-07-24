#!/bin/bash
#SBATCH --job-name=mitrabajo
#SBATCH --ntasks=1
#SBATCH --mem=65536
#SBATCH --time=00:30:00

#SBATCH --partition=besteffort

#SBATCH --qos=besteffort_gpu

#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sncalvo5@gmail.com
#SBATCH -o salida.out

export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

rm ./solution

nvcc ./main.cu -o solution

# echo '============================'
# echo 'NORMAL RUN STARTING'
# echo '============================'

# ./solution 10 10

tests=( 100 1000 10000 12500 15000 )

for test in "${tests[@]}"
do
  echo '============================'
  echo 'TESTING TIME WITH ARG $test'
  echo '============================'

  nvprof ./solution $test $test

  echo '============================'
  echo 'TESTING EFFICIENCY WITH $test END'
  echo '============================'
  nvprof --metrics gld_efficiency,gst_efficiency,shared_efficiency,shared_replay_overhead ./solution $test $test
done
