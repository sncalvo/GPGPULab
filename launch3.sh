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
#SBATCH -o salida3.out

export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

rm ./solution3

nvcc --version

nvcc ./cusparse.cu -lcusparse -o solution3

echo '============================'
echo 'NORMAL RUN STARTING'
echo '============================'

cuda-memcheck ./solution2 10000 10000
# ./solution3 10000 10000
