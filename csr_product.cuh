// Kernel that implements spmv product using CSR matrix
template <typename data_type>
__global__ void spmv_csr_kernel(const int *row_ptr, const int *col_ind, const data_type *val, const data_type *x, data_type *y, const int n) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < n) {
    const int start = row_ptr[row];
    const int end = row_ptr[row + 1];

    data_type sum = 0;
    for (int i = start; i < end; i++) {
      const int col = col_ind[i];
      sum += val[i] * x[col];
    }

    y[row] = sum;
  }
}

// TODO: Investigate and use __restricted__
// Kernel that implements spmv product using Block CSR matrix
template <typename data_type>
__global__ void bcsr_spmv_kernel_thread_per_row_row_major_matrix (
  const int n_block_rows,
  const int bs,
  const int *col_ids,
  const int *row_ptr,
  const data_type *data,
  const data_type *x,
  data_type *y
) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = idx % bs;
  const int block_row = idx / bs;
  const int first_block = row_ptr[block_row];
  const int last_block = row_ptr[block_row + 1];

  if (row < bs && block_row < n_block_rows) {
    data_type local_out = 0.0;

    for (int block = first_block; block < last_block; block++) {
      const int first_col = col_ids[block] * bs;

      for (int col = 0; col < bs; col++) {
        local_out += x[first_col + col] * data[block * bs * bs + row * bs + col];
      }
    }

    y[block_row * bs + row] = local_out;
  }
}
