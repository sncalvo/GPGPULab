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

  if (idx >= A.blFilN + 1) {
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

    int index = 0;
    for (int j = 0; j < 8; j++) {
      for (int k = 0; k < 8; k++) {
        if (bitMap & (0x8000000000000000 >> (j*8 + k))) {
          block[j][k] = A.val[start + index];
          index += 1;
        } else {
          block[j][k] = 0;
        }
      }
    }

    // Multiply dense block by dense vector
    for (int j = 0; j < 8; j++) {
      for (int k = 0; k < 8; k++) {
        result[idx * 8 + j] += block[j][k] * x[col * 8 + k];
      }
    }
  }
}

// Kernel that implements spmv product using Block CSR matrix
__global__ void bsr_vector_kernel_3(
  BlMat A,
  const VALUE *x,
  VALUE *result
) {
  __shared__ VALUE block[8][8];

  if (threadIdx.x == 0 && blockIdx.x == 0 && threadIdx.y == 0) {
    for (int i = 0; i < A.blColN * 8; i++) {
      printf("%f ", x[i]);
    }
    printf("\n");

    // Print matrix
    // for (int i = 0; i < A.nnz; i++) {
    //   printf("idx: %d, %f \n", i, A.val[i]);
    // }
    // printf("\n");
  }

  const int i = threadIdx.x;
  const int j = threadIdx.y;

  const int rowIdx = blockIdx.x;
  const int rowStart = A.blRowPtr[rowIdx] + blockIdx.y;
  const int rowEnd = A.blRowPtr[rowIdx + 1];

  // if (rowStart >= rowEnd) {
  //   block[j][i] = 0;
  //   return;
  // }

  const int col = A.blColIdx[rowStart];

  unsigned long long bitMap = A.blBmp[rowStart];
  const int start = A.blStart[rowStart];

  const int numberOfVals = __popcll(bitMap >> (64 - (j*8 + i)));

  // printf("%llu\n", bitMap);
  printf("%llu\n", bitMap & (0x8000000000000000 >> (j*8 + i)));
  if (bitMap & (0x8000000000000000 >> (j*8 + i)) != 0) {
    printf("Writing: %.2f, with: start: %d, nValues: %d \n", A.val[start + numberOfVals], start, numberOfVals);
    block[j][i] = A.val[start + numberOfVals];
  } else {
    block[j][i] = 0;
  }

  __syncthreads();

  // printf("%d %d %d %d %d %d %d %.2f\n", i, j, rowIdx, rowStart, rowEnd, col, numberOfVals, block[j][i]);

  if (j == 0) {
    for (int k = 0; k < 8; k++) {
      VALUE sumRow = 0;
      for (int l = 0; l < 8; l++) {
        sumRow += block[k][l] * x[col * 8 + l];
      }

      if (sumRow != 0) {
        atomicAdd(&result[rowIdx * 8 + k], sumRow);
      }
    }
  }
}
