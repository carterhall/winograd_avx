#include "../helpers.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void test_matmul() {
  int M = 24, N = 16, P = 32;
  float *A = aligned_alloc(64, sizeof(float)*M*N);
  float *B = aligned_alloc(64, sizeof(float)*N*P);
  float *B_T = aligned_alloc(64, sizeof(float)*P*N);

  float *swizzle_mem = aligned_alloc(64, sizeof(float)*N*P);

  for (int i = 0; i < M*N; ++i) {
    A[i] = (float)rand()/(float)RAND_MAX;
    //A[i] = i;
  }
  for (int i = 0; i < N*P; ++i) {
    B[i] = (float)rand()/(float)RAND_MAX;
    //B[i] = i;
  }

  transpose_matrix(B, N, P, B_T);

  float *Cgemm = aligned_alloc(64, sizeof(float)*M*P);
  float *Cnaive = aligned_alloc(64, sizeof(float)*M*P);
  memset(Cgemm, 0, M*P*sizeof(float));
  memset(Cnaive, 0, M*P*sizeof(float));

  naive_matmul(A, B, Cnaive, M, N, P);
  gemm_preallocated(A, B_T, Cgemm, M, N, P, swizzle_mem);

  for (int i = 0; i < N*N; ++i) {
    if (Cnaive[i] != Cgemm[i]) {
      printf("MISMATCH: index %d, naive = %f, gemm = %f\n", i, Cnaive[i], Cgemm[i]);
      goto done;
    }
  }
  printf("All matrix entries match.\n");

done:
  free(A); free(B); free(B_T); free(Cgemm); free(Cnaive); free(swizzle_mem);
}

int main() {
  test_matmul();
}
