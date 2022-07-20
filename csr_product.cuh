#include <stdio.h>
#include "./types.h"

// Kernel that implements spmv product using CSR matrix
__global__ void spmv_csr_kernel(
  const CSRMat A, // CSR Matrix A
  const VALUE *x, // Vector x
  VALUE *result // Result vector
) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= A.colN) {
    return;
  }

  __shared__ int near_row[1024];

  // const int start = A.rowPtr[row];
  // const int end = A.rowPtr[row + 1];
  near_row[threadIdx.x] = A.rowPtr[row];
  __syncthreads();

  int start = near_row[threadIdx.x];
  int end = near_row[threadIdx.x + 1];

  __shared__ VALUE near_val[1024];
  __shared__ int near_col[1024];

  near_val[threadIdx.x] = A.val[start];
  near_col[threadIdx.x] = A.colIdx[start];
  __syncthreads();

  VALUE sum = 0;
  for (int i = start; i < end; i++) {
    sum += near_val[threadIdx.x] * x[near_col[threadIdx.x]];
  }

  // for (int i = start; i < end; i++) {
  //   const int col = A.colIdx[i];
  //   sum += A.val[i] * x[col];
  // }

  result[row] = sum;
}

__device__ VALUE ** create_dense_block(
  const unsigned long long bitMap,
  const int start,
  VALUE *values,
  int i
) {
  // SHOULD IT BE MALLOC???
  VALUE **block = (VALUE **)malloc(sizeof(VALUE *) * 32);

  for (int j = 0; j < 32; j++) {
    if (bitMap & (1 << j)) {
      block[j] = (VALUE *)malloc(sizeof(VALUE) * 32);
      for (int k = 0; k < 32; k++) {
        block[j][k] = values[i++];
      }
    }
  }

  return block;
}

// Kernel that implements spmv product using Block CSR matrix
__global__ void bcsr_spmv_kernel_thread_per_row_row_major_matrix (
  BlMat A,
  const VALUE *x,
  VALUE *result
) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= A.blFilN) {
    return;
  }

  const int start = A.blRowPtr[idx];
  const int end = A.blRowPtr[idx + 1];

  VALUE sum = 0;

  for (int i = start; i < end; i++) {
    const int col = A.blColIdx[i];
    sum += A.val[i] * x[col];

    // Create dense block
    VALUE **dense_block = create_dense_block(
      A.blBmp[i],
      A.blStart[i],
      A.val,
      i
    );

    // Multiply dense block by dense vector
    for (int j = 0; j < 32; j++) {
      if (dense_block[j] != NULL) {
        for (int k = 0; k < 32; k++) {
          sum += dense_block[j][k] * x[col];
        }
      }
    }
  }
}
