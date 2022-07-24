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

  // print_matriz_bloques(&A);
  // print_matriz_bloques_en_COO(&A);

  CSRMat A_csr;

  bloques_a_CSR(&A, &A_csr);
  // print_CSR(&A_csr);

  VALUE *vector = (VALUE*) malloc(blColN*8*sizeof(VALUE));

  random_vector(vector, blColN*8);

  VALUE *d_vector = (VALUE*) malloc(blColN*8*sizeof(VALUE));
  cudaMalloc((void **)&d_vector, blColN*8*sizeof(VALUE));
	cudaMemcpy(d_vector, vector, blColN*8*sizeof(VALUE), cudaMemcpyHostToDevice);

  VALUE *d_val = (VALUE*) malloc(A_csr.rowPtr[A_csr.filN]*sizeof(VALUE));
  int *d_colIdx = (int*) malloc(A_csr.rowPtr[A_csr.filN]*sizeof(int));
  int *d_rowPtr = (int*) malloc((A_csr.filN+1)*sizeof(int));
  cudaMalloc((void **)&d_val, A_csr.rowPtr[A_csr.filN]*sizeof(VALUE));
  cudaMalloc((void **)&d_colIdx, A_csr.rowPtr[A_csr.filN]*sizeof(int));
  cudaMalloc((void **)&d_rowPtr, (A_csr.filN+1)*sizeof(int));

  CUDA_CHK(cudaMemcpy(d_val, A_csr.val, A_csr.rowPtr[A_csr.filN]*sizeof(VALUE), cudaMemcpyHostToDevice));
  CUDA_CHK(cudaMemcpy(d_colIdx, A_csr.colIdx, A_csr.rowPtr[A_csr.filN]*sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHK(cudaMemcpy(d_rowPtr, A_csr.rowPtr, (A_csr.filN+1)*sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHK(cudaGetLastError());
  CUDA_CHK(cudaDeviceSynchronize());
  A_csr.val = d_val;
  A_csr.colIdx = d_colIdx;
  A_csr.rowPtr = d_rowPtr;

  VALUE *d_res;
  CUDA_CHK(cudaMalloc((void **)&d_res, blColN*8*sizeof(VALUE)));

	// dim3 dimBlock(256);
  // Fast ceil(A_csr.colN/256)
	// dim3 dimGrid((A_csr.colN + 256 - 1) / 256);

  // for (int i = 0; i < A_csr.colN; ++i)
  // {
  //   printf("%.2f\n", vector[i]);
  // }

  dim3 dimBlock(256, 1);
  const int gridRowSquare = ceil(sqrt(A_csr.filN));
  dim3 dimGrid((A_csr.colN + 256 - 1) / 256, gridRowSquare, gridRowSquare);
  // printf("%d\n", dimBlock.x);
  // printf("x: %d, y: %d\n", dimGrid.x, dimGrid.y);

  // spmv_csr_kernel<<<dimGrid, dimBlock>>>(A_csr, d_vector, d_res);
  spmv_csr_kernel_2<<<dimGrid, dimBlock>>>(A_csr, d_vector, d_res);

  CUDA_CHK(cudaGetLastError());
  CUDA_CHK(cudaDeviceSynchronize());
  VALUE *res = (VALUE*) malloc(blColN*8*sizeof(VALUE));
  CUDA_CHK(cudaMemcpy(res, d_res, blColN*8*sizeof(VALUE), cudaMemcpyDeviceToHost));

  printf("\n");

  // for (int i = 0; i < 10 * 8; ++i)
  // {
  //   printf("%.2f\n", res[i]);
  // }

  cudaFree(d_vector);
  cudaFree(d_val);
  cudaFree(d_colIdx);
  cudaFree(d_rowPtr);
  cudaFree(d_res);

	free(vector);
  free(res);
	return 0;
}
