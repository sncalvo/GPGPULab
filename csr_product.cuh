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

  result[row] = sum;
}

// Kernel that implements spmv product using Block CSR matrix
__global__ void bsr_vector_kernel(
  BlMat A,
  const VALUE *x,
  VALUE *result
) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx + 1 >= A.blFilN) {
    return;
  }

  const int rowStart = A.blRowPtr[idx];
  const int rowEnd = A.blRowPtr[idx + 1];

  for (int i = rowStart; i < rowEnd; i++) {
    const int col = A.blColIdx[i];

    // Create dense block
    VALUE block[8][8];
    unsigned long long bitMap = A.blBmp[i];
    const int start = A.blStart[i];
    const int end = A.blStart[i + 1];

    for (int j = 0; j < 8; j++) {
      for (int k = 0; k < 8; k++) {
        if (bitMap & (1 << (j*k)) && start + j + k < end) {
          block[j][k] = A.val[start + j + k];
        } else {
          block[j][k] = 0;
        }
      }
    }

    // Multiply dense block by dense vector
    for (int j = 0; j < 8; j++) {
      for (int k = 0; k < 8; k++) {
        result[idx * 8 + j] += block[j][k] * x[col + k];
        // const VALUE tmp = block[j][k];
        // const VALUE tmp2 = x[col + k];
        // const VALUE tmp3 = result[idx * 8 + j];

        // if (k > 6 && threadIdx.x == 0) {
        //   printf("%f %f %d %d\n", tmp, tmp3, col, i);
        // }
      }
    }
  }
}
