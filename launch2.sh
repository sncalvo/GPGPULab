#!/bin/bash
#SBATCH --job-name=mitrabajo
#SBATCH --ntasks=1
#SBATCH --mem=8192
#SBATCH --time=00:10:00

#SBATCH --partition=besteffort

#SBATCH --qos=besteffort_gpu

#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sncalvo5@gmail.com
#SBATCH -o salida2.out

export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

nvcc ./main.cu -o solution2

./solution2 10 10

echo '============================'
echo 'NORMAL RUN STARTING'
echo '============================'

./solution2 10000 10000

echo '============================'
echo 'TESTING TIME'
echo '============================'

nvprof ./solution2 10000 10000

echo '============================'
echo 'TESTING EFFICIENCY'
echo '============================'

nvprof --metrics gld_efficiency,gst_efficiency,shared_efficiency ./solution2 10000 10000

echo '============================'
echo 'END'
echo '============================'
