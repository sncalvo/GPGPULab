#!/bin/bash
#SBATCH --job-name=mitrabajo
#SBATCH --ntasks=1
#SBATCH --mem=8192
#SBATCH --time=00:02:00

#SBATCH --partition=besteffort

#SBATCH --qos=besteffort_gpu

#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sncalvo5@gmail.com
#SBATCH -o salida4.out

export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

rm ./solution4

nvcc --version

nvcc ./cudasample.cu -lcusparse -o solution4

echo '============================'
echo 'NORMAL RUN STARTING'
echo '============================'

# cuda-memcheck ./solution3 10000 10000
./solution4 10000 10000
