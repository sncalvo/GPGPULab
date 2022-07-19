#include <stdio.h>
#include "./types.h"

// Kernel that implements spmv product using CSR matrix
__global__ void spmv_csr_kernel(
  const CSRMat A, // CSR Matrix A
  const VALUE *x, // Vector x
  VALUE *result // Result vector
) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;

  printf("AH!!! %d: %f\n", row, A.colN);

  if (row >= A.colN) {
    return;
  }

  const int start = A.rowPtr[row];
  const int end = A.rowPtr[row + 1];

  VALUE sum = 0;
  for (int i = start; i < end; i++) {
    const int col = A.colIdx[i];
    sum += A.val[i] * x[col];
  }

  printf("AH!!! %d: %f\n", row, sum);

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
  const BlMat *A,
  const VALUE *x,
  VALUE *result,
  const int n
) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= A->blFilN) {
    return;
  }

  const int start = A->blRowPtr[idx];
  const int end = A->blRowPtr[idx + 1];

  VALUE sum = 0;

  for (int i = start; i < end; i++) {
    const int col = A->blColIdx[i];
    sum += A->val[i] * x[col];

    // Create dense block
    VALUE **dense_block = create_dense_block(
      A->blBmp[i],
      A->blStart[i],
      A->val,
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

// TODO: Investigate and use __restricted__
// Kernel that implements spmv product using Block CSR matrix
__global__ void bcsr_spmv_kernel_thread_per_row_row_major_matrix_1 (
  const int n_block_rows,
  const int bs,
  const int *col_ids,
  const int *row_ptr,
  const VALUE *data,
  const VALUE *x,
  VALUE *y
) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = idx % bs;
  const int block_row = idx / bs;
  const int first_block = row_ptr[block_row];
  const int last_block = row_ptr[block_row + 1];

  if (row < bs && block_row < n_block_rows) {
    VALUE local_out = 0.0;

    for (int block = first_block; block < last_block; block++) {
      const int first_col = col_ids[block] * bs;

      for (int col = 0; col < bs; col++) {
        local_out += x[first_col + col] * data[block * bs * bs + row * bs + col];
      }
    }

    y[block_row * bs + row] = local_out;
  }
}
