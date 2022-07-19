#ifndef TYPES_H

#define TYPES_H

#define VALUE float

typedef struct {
  VALUE *val;
  int *blStart;
  unsigned long long *blBmp;
  int *blColIdx;
  int *blRowPtr;
  int blFilN, blColN, nBlocks, nnz;
} BlMat;

typedef struct {
  VALUE *val;
  int *colIdx;
  int *rowPtr;
  int filN, colN;
} CSRMat;

#define CUDA_CHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#endif
