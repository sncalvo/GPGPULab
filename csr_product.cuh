#include <stdio.h>
#include "./types.h"

__inline__ __device__ VALUE warp_reduce_sum(VALUE val) {
  for (int mask = warpSize/2; mask > 0; mask /= 2)
    val += __shfl_xor(val, mask);
  return val;
}

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

  __shared__ int near_row[257];

  near_row[threadIdx.x] = A.rowPtr[row];
  __syncthreads();

  int start = near_row[threadIdx.x];
  int end = near_row[threadIdx.x + 1];

  VALUE sum = 0;
  for (int i = start; i < end; i++) {
    sum += A.val[i] * x[A.colIdx[i]];
  }

  result[row] = sum;
}

__global__ void spmv_csr_kernel_2(
  const CSRMat A, // CSR Matrix A
  const VALUE *x, // Vector x
  VALUE *result // Result vector
) {
  const int row = blockIdx.y * gridDim.y + blockIdx.z;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ int start;
  __shared__ int end;

  __shared__ VALUE vals[256];
  __shared__ VALUE shared_x[256];

  if (threadIdx.x == 0) {
    start = A.rowPtr[row];
    end = A.rowPtr[row + 1];
  }
  __syncthreads();

  if (start + col < end) {
    vals[threadIdx.x] = A.val[start + col];
    shared_x[threadIdx.x] = x[A.colIdx[start + col]];
  }

  __syncthreads();

  VALUE sum = 0;
  if (start + col < end) {
    sum = vals[threadIdx.x] * shared_x[threadIdx.x];
  } else {
    sum = 0;
  }

  warp_reduce_sum(sum);

  if (sum != 0) {
    atomicAdd(&result[row], sum);
  }
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

    const int col = A.blColIdx[i];
    // Multiply dense block by dense vector
    for (int j = 0; j < 8; j++) {
      VALUE sumRow = 0;
      for (int k = 0; k < 8; k++) {
        sumRow += block[j][k] * x[col * 8 + k];
      }
      result[idx * 8 + j] = sumRow;
    }
  }
}

// Kernel that implements spmv product using Block CSR matrix
__global__ void bsr_vector_kernel_2(
  BlMat A,
  const VALUE *x,
  VALUE *result
) {
  __shared__ VALUE block[8][8];

  const int i = threadIdx.x;
  const int j = threadIdx.y;

  const int rowIdx = blockIdx.x;
  const int rowStart = A.blRowPtr[rowIdx] + blockIdx.y;
  const int rowEnd = A.blRowPtr[rowIdx + 1];

  if (rowStart >= rowEnd) {
    block[j][i] = 0;
    return;
  }

  const int col = A.blColIdx[rowStart];

  unsigned long long bitMap = A.blBmp[rowStart];
  const int start = A.blStart[rowStart];
  // const int end = A.blStart[rowStart + 1];

  const int numberOfVals = __popcll(bitMap >> (64 - (j*8 + i)));

  if (bitMap & (0x8000000000000000 >> (j*8 + i))) {
    block[j][i] = A.val[start + numberOfVals];
  } else {
    block[j][i] = 0;
  }

  __syncthreads();

  if (j == 0 && i == 0) {
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

// Kernel that implements spmv product using Block CSR matrix
__global__ void bsr_vector_kernel_3(
  BlMat A,
  const VALUE *x,
  VALUE *result
) {
  const int i = threadIdx.x;
  const int j = threadIdx.y;

  const int rowIdx = blockIdx.x;

  const int rowStart = A.blRowPtr[rowIdx] + blockIdx.y;
  const int rowEnd = A.blRowPtr[rowIdx + 1];

  VALUE sumRow = 0;
  __shared__ int col;
  __shared__ unsigned long long bitMap;
  __shared__ int start;
  __shared__ VALUE vals[64];
  __shared__ VALUE shared_x[64];

  if (threadIdx.x == 0) {
    col = A.blColIdx[rowStart];
    bitMap = A.blBmp[rowStart];
    start = A.blStart[rowStart];
  }

  shared_x[threadIdx.x] = x[col * 8 + i];

  __syncthreads();

  const int numberOfVals = __popcll(bitMap >> (64 - (j*8 + i)));
  vals[threadIdx.x] = A.val[start + numberOfVals];

  __syncthreads();

  if (bitMap & (0x8000000000000000 >> (j*8 + i)) && rowStart < rowEnd) {
    sumRow = vals[threadIdx.x] * shared_x[threadIdx.x];
  } else {
    sumRow = 0;
  }

  warp_reduce_sum(sumRow);

  if (sumRow != 0) {
    atomicAdd(&result[rowIdx * 8 + j], sumRow);
  }
}
