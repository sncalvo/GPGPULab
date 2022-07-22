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

nvcc ./main.cu -o solution

echo '============================'
echo 'NORMAL RUN STARTING'
echo '============================'

./solution 10000 10000

echo '============================'
echo 'TESTING TIME'
echo '============================'

# nvprof ./solution 10000 10000

# echo '============================'
# echo 'TESTING EFFICIENCY'
# echo '============================'

# nvprof --metrics gld_efficiency,gst_efficiency,shared_efficiency ./solution 10000 10000

# echo '============================'
# echo 'END'
# echo '============================'

tests=( 100 1000 10000 20000 50000 )

for test in "${tests[@]}"
do
  echo '============================'
  echo 'TESTING TIME WITH ARG $test'
  echo '============================'

  nvprof ./solution $test $test

  echo '============================'
  echo 'TESTING EFFICIENCY WITH $test END'
  echo '============================'
  nvprof --metrics gld_efficiency,gst_efficiency,shared_efficiency ./solution $test $test
done
