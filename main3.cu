#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

#include "./types.h"
#include "./csr_product.cuh"

#include "./utils.h"

int main(int argc, char *argv[]){
  if (argc < 2) {
    printf("./programa blFil blCol \n");
    exit(1);
  }

  srand(0); // Inicializa la semilla aleatoria

  unsigned int blFilN = atoi(argv[1]);
  unsigned int blColN = atoi(argv[2]);

  BlMat A;

  gen_matriz_bloques(&A, blFilN, blColN);

  CSRMat A_csr;

  bloques_a_CSR(&A, &A_csr);

  printf("\n \n \n");

  // print_matriz_bloques_en_COO(&A);

  VALUE *vector = (VALUE*) malloc(A_csr.colN*sizeof(VALUE));

  random_vector(vector, A.blColN * 8);

  VALUE *d_res;
  CUDA_CHK(cudaMalloc((void **)&d_res, A_csr.colN*sizeof(VALUE)));

  int *d_blStart;
  CUDA_CHK(cudaMalloc((void **)&d_blStart, (A.nBlocks+1)*sizeof(int)));
  CUDA_CHK(cudaMemcpy(d_blStart, A.blStart, (A.nBlocks+1)*sizeof(int), cudaMemcpyHostToDevice));

  int *d_blColIdx;
  CUDA_CHK(cudaMalloc((void **)&d_blColIdx, A.nBlocks*sizeof(int)));
  CUDA_CHK(cudaMemcpy(d_blColIdx, A.blColIdx, A.nBlocks*sizeof(int), cudaMemcpyHostToDevice));

  unsigned long long *d_blBmp;
  CUDA_CHK(cudaMalloc((void **)&d_blBmp, A.nBlocks*sizeof(unsigned long long)));
  CUDA_CHK(cudaMemcpy(d_blBmp, A.blBmp, A.nBlocks*sizeof(unsigned long long), cudaMemcpyHostToDevice));

  int *d_blRowPtr;
  CUDA_CHK(cudaMalloc((void **)&d_blRowPtr, (A.blFilN+1)*sizeof(int)));
  CUDA_CHK(cudaMemcpy(d_blRowPtr, A.blRowPtr, (A.blFilN+1)*sizeof(int), cudaMemcpyHostToDevice));

  VALUE *d_val;
  CUDA_CHK(cudaMalloc((void **)&d_val, A.nnz*sizeof(VALUE)));
  CUDA_CHK(cudaMemcpy(d_val, A.val, A.nnz*sizeof(VALUE), cudaMemcpyHostToDevice));

  A.blStart = d_blStart;
  A.blColIdx = d_blColIdx;
  A.blBmp = d_blBmp;
  A.blRowPtr = d_blRowPtr;
  A.val = d_val;

  VALUE *d_vector;
  CUDA_CHK(cudaMalloc((void **)&d_vector, A_csr.colN*sizeof(VALUE)));
  CUDA_CHK(cudaMemcpy(d_vector, vector, A_csr.colN*sizeof(VALUE), cudaMemcpyHostToDevice));

	dim3 dimBlock(8, 8);
	dim3 dimGrid(A.blFilN, A.blColN);
  bsr_vector_kernel_3<<<dimGrid, dimBlock>>>(A, d_vector, d_res);

  CUDA_CHK(cudaGetLastError());
  CUDA_CHK(cudaDeviceSynchronize());
  VALUE *res = (VALUE*) malloc(A_csr.colN*sizeof(VALUE));
  CUDA_CHK(cudaMemcpy(res, d_res, A_csr.colN*sizeof(VALUE), cudaMemcpyDeviceToHost));

  printf("\n");

  for (int i = 0; i < 10 * 8; ++i)
  {
    printf("%.2f\n", res[i]);
  }

  cudaFree(d_vector);
  cudaFree(d_res);
  cudaFree(d_blStart);
  cudaFree(d_blColIdx);
  cudaFree(d_blBmp);
  cudaFree(d_blRowPtr);
  cudaFree(d_val);

	free(vector);
  free(res);
	return 0;
}
