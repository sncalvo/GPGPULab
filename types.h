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

#endif
