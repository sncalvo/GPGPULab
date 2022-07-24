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
#SBATCH -o salida3.out

export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

rm ./solution3

nvcc ./main3.cu -lineinfo -o solution3

# ./solution3 10 10

tests=( 100 1000 10000 12500 15000 )
# tests=( 10 )

for test in "${tests[@]}"
do
  echo '============================'
  echo 'TESTING TIME WITH ARG $test'
  echo '============================'

  nvprof ./solution3 $test $test

  echo '============================'
  echo 'TESTING EFFICIENCY WITH $test END'
  echo '============================'
  nvprof --metrics gld_efficiency,gst_efficiency,shared_efficiency,atomic_replay_overhead,atomic_throughput,atomic_transactions ./solution3 $test $test
done
