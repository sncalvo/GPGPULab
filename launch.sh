#!/bin/bash
#SBATCH --job-name=mitrabajo
#SBATCH --ntasks=1
#SBATCH --mem=2048
#SBATCH --time=00:10:00

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

nvprof ./solution 10000 10000

echo '============================'
echo 'TESTING EFFICIENCY'
echo '============================'

nvprof --metrics gld_efficiency,gst_efficiency,shared_efficiency 10000 10000

echo '============================'
echo 'END'
echo '============================'
