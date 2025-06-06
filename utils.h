#ifndef UTILS_H

#define UTILS_H

void gen_matriz_bloques(BlMat *A, int blFilN, int blColN) {
  A->blFilN = blFilN;
  A->blColN = blColN;

  // genero comienzos de fila de bloques
  A->blRowPtr = (int*) malloc((blFilN + 1) * sizeof(int));
  A->blRowPtr[0] = 0;
  int n_bl = 0;
  for (int i = 1; i < blFilN + 1; ++i) {
    n_bl += 1 + rand()%(blColN-1);
    A->blRowPtr[i]=n_bl;
  }

  A->nBlocks= n_bl;

  A->blStart = (int*) malloc((n_bl+1)*sizeof(int));
  A->blBmp = (unsigned long long*) malloc((n_bl)*sizeof(unsigned long long));
  A->blColIdx = (int*) malloc((n_bl)*sizeof(int));

  // sorteo índice de columna de bloque
  for (int i = 0; i < blFilN; ++i)
  {
    int colidx =- 1;
    int nBlFil = A->blRowPtr[i+1]-A->blRowPtr[i];
    for (int k = A->blRowPtr[i]; k < A->blRowPtr[i+1]; ++k)
    {
      colidx += 1 + (rand()%(blColN-nBlFil-colidx));
      nBlFil--;
      A->blColIdx[k]=colidx;
    }
  }

  A->blStart[0] = 0;
  // sorteo número de noceros de cada bloque
  int ctr = 0;
  for (int i = 1; i < n_bl+1; ++i) {
    ctr += 1 + rand()%20;
    A->blStart[i] = ctr;
  }
  A->nnz=ctr;

  // genero valores
  A->val = (VALUE*) malloc(ctr*sizeof(VALUE));
  for (int i = 0; i < ctr; ++i) {
    A->val[i] = 1.0;
  }

  // genero bmps aleatorios
  for (int i = 0; i < n_bl; ++i) {
    int nnz_bl=A->blStart[i+1]-A->blStart[i];
    unsigned long long bmp=0;
    int bit=-1;
    for (int j = A->blStart[i]; j < A->blStart[i+1]; ++j) {
      bit+=1+(rand()%(64-nnz_bl-bit));
      nnz_bl--;
      // printf("bit=%d ", bit);
      bmp=bmp + (1ULL << bit);

      // printf("nnz_bl=%d 2^bit=%llx bmp=%llx\n", nnz_bl, (unsigned long long)(1ULL<<bit), bmp);
    }
    A->blBmp[i]=bmp;
  }
}

void print_matriz_bloques(BlMat * A) {
  printf("blfil=%d blcol=%d blocks=%d\n", A->blFilN, A->blColN, A->blRowPtr[A->blFilN]); fflush(0);

  for (int i = 0; i <  A->blFilN; ++i) {
    for (int k = A->blRowPtr[i]; k < A->blRowPtr[i+1]; ++k) {
      printf("bloque (%d,%d,%llx)\n", i, A->blColIdx[k], A->blBmp[k]);
      printf("%d valores\n", A->blStart[k+1]- A->blStart[k]);
      printf("\n");
    }
  }
}

void print_matriz_bloques_en_COO(BlMat * A) {
  printf("blfil=%d blcol=%d blocks=%d\n", A->blFilN, A->blColN, A->blRowPtr[A->blFilN]); fflush(0);
  int ctr = 0;

  for (int i = 0; i <  A->blFilN; ++i)
    for (int k = A->blRowPtr[i]; k < A->blRowPtr[i+1]; ++k) {
        unsigned long long bmp = A->blBmp[k];
        int ii = A->blStart[k];
        int blCol = A->blColIdx[k];

      for (int jj = 63; jj >= 0; --jj) {
        if (((1ULL << jj) & bmp) != 0) {
          printf("A(%d,%d)=%.2f\n", i*8+(63-jj)/8, blCol*8+(63-jj)%8, A->val[ii]);
          ii++;
          ctr++;
        }
      }
    }

  printf("NNZ=%d\n",ctr);
}

void bloques_a_CSR(BlMat * A_bl, CSRMat *A_csr) {
  A_csr->filN = A_bl->blFilN*8;
  A_csr->colN = A_bl->blColN*8;

  int nnz = A_bl->nnz;

  A_csr->val = (VALUE*) malloc(nnz * sizeof(VALUE));
  A_csr->colIdx = (int*) malloc(nnz * sizeof(int));
  A_csr->rowPtr = (int*) malloc((A_csr->filN + 1) * sizeof(int));

  for (int i = 0; i < (A_csr->filN + 1); ++i) {
    A_csr->rowPtr[i] = 0;
  }

  // inicializo rowptr con la cantidad de elems de cada fila
  for (int i = 0; i <  A_bl->blFilN; ++i)
    for (int k = A_bl->blRowPtr[i]; k < A_bl->blRowPtr[i+1]; ++k)
      for (int jj = 63; jj >= 0; --jj)
        if (((1ULL << jj) & A_bl->blBmp[k]) != 0)
          A_csr->rowPtr[1+i*8+(63-jj)/8]++;

  // suma prefija para obtener los desplazamientos
  for (int i = 1; i < A_csr->filN+1; ++i)
    A_csr->rowPtr[i] += A_csr->rowPtr[i-1];

  // reservo un vector temporal de desplazamientos
  int *offsets = (int*) malloc((A_csr->filN)*sizeof(int));

  // recorro matriz de bloques y pongo cada elemento en su fila de A_csr
  for (int i = 0; i <  A_bl->blFilN; ++i) {
    for (int k = A_bl->blRowPtr[i]; k < A_bl->blRowPtr[i+1]; ++k) {
      unsigned long long bmp = A_bl->blBmp[k];
      int ii = A_bl->blStart[k];
      int blCol = A_bl->blColIdx[k];

      for (int jj = 63; jj >= 0; --jj) {
        if (((1ULL << jj) & bmp) != 0) {
          int fil = i * 8 + (63 - jj) / 8;
          A_csr->colIdx[A_csr->rowPtr[fil] + offsets[fil]] = blCol*8+(63-jj)%8;
          A_csr->val[A_csr->rowPtr[fil] + offsets[fil]] = A_bl->val[ii];
          ii++;
          offsets[fil]++;
        }
      }
    }
  }

  free(offsets);
}


void print_CSR(CSRMat *A_csr) {
  printf("A_csr: %d fils, %d cols, %d nz\n",  A_csr->filN,  A_csr->colN, A_csr->rowPtr[A_csr->filN]- A_csr->rowPtr[0]);

  for (int i = 0; i < A_csr->filN; ++i)
  {
    for (int k = A_csr->rowPtr[i]; k < A_csr->rowPtr[i+1]; ++k)
    {
      printf("(%d,%d,%.2f)\n", i, A_csr->colIdx[k], A_csr->val[k]);
    }
  }
}

void random_vector(VALUE *v, int n) {
  for (int i = 0; i < n; ++i)
    v[i] = 1.0;
}

#endif
