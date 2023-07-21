#pragma once
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <immintrin.h>

#define MIN(x, y) (((x) < (y)) ? (x) : (y))

typedef struct _Tensor { int shape[4]; float *data; } Tensor;  // Allow up to order 4

Tensor view(Tensor t, int new_shape[4]) {    // Only modifies shape, not data
  Tensor v;
  for (int i = 0; i < 4; ++i) v.shape[i] = new_shape[i];
  v.data = t.data;
  return v;
}

int index0(Tensor t, int i0) {  // Get a 3-tensor from a 4-tensor (index into the first dim)
  return i0*t.shape[1]*t.shape[2]*t.shape[3];
}

int index01(Tensor t, int i0, int i1) {  // Get a matrix from a 4-tensor 
  int stride23 = t.shape[2]*t.shape[3];
  return i0*t.shape[1]*stride23 + i1*stride23;
}

int index012(Tensor t, int i0, int i1, int i2) {  // Get a vector from a 4-tensor
  int stride23 = t.shape[2]*t.shape[3];
  return i0*t.shape[1]*stride23 + i1*stride23 + i2*t.shape[3];
}

int index0123(Tensor t, int i0, int i1, int i2, int i3) {  // Get a scalar from a 4-tensor
  int stride23 = t.shape[2]*t.shape[3];
  return i0*t.shape[1]*stride23 + i1*stride23 + i2*t.shape[3] + i3;
}

// This one might as well just allocate a new tensor and return it. It's an expensive operation however you slice it.
Tensor tensor_transpose(Tensor in, int d0, int d1) {
  int total_size = in.shape[0]*in.shape[1]*in.shape[2]*in.shape[3];
  Tensor out;
  out.data = aligned_alloc(64, sizeof(float)*in.shape[0]*in.shape[1]*in.shape[2]*in.shape[3]);
  for (int i = 0; i < 4; ++i) {
    if      (i == d0) out.shape[i] = in.shape[d1]; 
    else if (i == d1) out.shape[i] = in.shape[d0]; 
    out.shape[i] = in.shape[i];
  }

  for (int i0 = 0; i0 < in.shape[0]; ++i0) {
    int o0 = 0 == d0 ? d1 : 0 == d1 ? d0 : i0;
    for (int i1 = 0; i1 < in.shape[1]; ++i1) {
      int o1 = 1 == d0 ? d1 : 1 == d1 ? d0 : i1;
      for (int i2 = 0; i2 < in.shape[2]; ++i2) {
        int o2 = 2 == d0 ? d1 : 2 == d1 ? d0 : i2;
        for (int i3 = 0; i3 < in.shape[3]; ++i3) {
          int o3 = 3 == d0 ? d1 : 3 == d1 ? d0 : i3;
          out.data[index0123(out, o0, o1, o2, o3)] = in.data[index0123(in, i0, i1, i2, i3)];
        }
      }
    }
  }

  return out;
}


void matrix_get_tile(float *tile, float *matrix, int RM, int CM, int RT, int CT, int rmstart, int cmstart) {
  int RT_bounded = MIN(RT, RM - rmstart);
  int CT_bounded = MIN(CT, CM - cmstart);
  for (int rt = 0; rt < RT_bounded; ++rt) {
    for (int ct = 0; ct < CT_bounded; ++ct) {
      tile[rt*CT + ct] = matrix[(rmstart + rt)*CM + (cmstart + ct)];
    }
  }
}


// Strided in the matrix, not the tile.
// Skips over 'column_stride' elements, but 'cmstart' is still a raw index!
void matrix_get_tile_column_strided(float *tile, float *matrix, int RM, int CM, int RT, int CT, int rmstart, int cmstart, int column_stride) {
  int RT_bounded = MIN(RT, RM - rmstart);
  int CT_bounded = MIN(CT, CM - cmstart);
  //int CT_bounded = CT;

  /*
  for (int rt = 0; rt < RT_bounded; ++rt) {
    for (int ct = 0; ct < CT_bounded; ++ct) {
      tile[(rt*CT + ct)] = matrix[(rmstart + rt)*CM + (cmstart + column_stride*ct)];
    }
  }
  */
  // @Speed this is the dumb way to do it because my brain is tired
  // TODO Correct the CT_bounded calculation above
  for (int rt = 0; rt < RT; ++rt) {
    for (int ct = 0; ct < CT; ++ct) {
      if (rmstart + rt >= RM) {
        tile[(rt*CT + ct)] = 0.f;
      }
      else if (cmstart + column_stride*ct >= CM) {
        tile[(rt*CT + ct)] = 0.f;
      }
      else {
        tile[(rt*CT + ct)] = matrix[(rmstart + rt)*CM + (cmstart + column_stride*ct)];
      }
    }
  }
}

// Strided in the matrix, not the tile.
// Skips over 'column_stride' elements, but 'cmstart' is still a raw index!
void matrix_get_tile_column_strided_avx(__m256 *tile, float *matrix, int RM, int CM, int RT, int CT, int rmstart, int cmstart, int column_stride) {
  int RT_bounded = MIN(RT, RM - rmstart);
  int CT_bounded = MIN(CT, CM - cmstart);
  //int CT_bounded = CT;

  //for (int rt = 0; rt < RT_bounded; ++rt) {
    //for (int ct = 0; ct < CT_bounded; ++ct) {
      //tile[(rt*CT + ct)] = matrix[(rmstart + rt)*CM + (cmstart + column_stride*ct)];
    //}
  //}
  // @Speed this is the dumb way to do it because my brain is tired
  // TODO Correct the CT_bounded calculation above
  for (int rt = 0; rt < RT; ++rt) {
    for (int ct = 0; ct < CT; ++ct) {
      if (rmstart + rt >= RM) {
        tile[(rt*CT + ct)] = _mm256_setzero_ps();
      }
      else if (cmstart + column_stride*ct >= CM) {
        tile[(rt*CT + ct)] = _mm256_setzero_ps();
      }
      else {
        tile[(rt*CT + ct)] = *((__m256*)&matrix[(rmstart + rt)*CM + (cmstart + column_stride*ct)]);
      }
    }
  }
}

// Strided in the TILE, not the matrix, unlike above functions.
// Skips over 'column_stride' elements, but 'cmstart' is still a raw index!
void matrix_set_tile_column_strided(float *tile, float *matrix, int RM, int CM, int RT, int CT, int rmstart, int cmstart, int column_stride, int ctstart) {
  int RT_bounded = MIN(RT, RM - rmstart);
  int CT_bounded = MIN(CT, CM - cmstart);
  for (int rt = 0; rt < RT; ++rt) {
    for (int ct = 0; ct < CT; ++ct) {
      if (rmstart + rt < RM && cmstart + ct < CM) {
        matrix[(rmstart + rt)*CM + (cmstart + ct)] = tile[(rt*CT + ct)*column_stride + ctstart];
        //tile[(rt*CT + ct)] = matrix[(rmstart + rt)*CM + (cmstart + column_stride*ct)];
      }
    }
  }
}

void matrix_set_tile(float *tile, float *matrix, int RM, int CM, int RT, int CT, int rmstart, int cmstart) {
  RT = MIN(RT, RM - rmstart);
  CT = MIN(CT, CM - cmstart);
  for (int rt = 0; rt < RT; ++rt) {
    for (int ct = 0; ct < CT; ++ct) {
      matrix[(rmstart + rt)*CM + (cmstart + ct)] = tile[rt*CT + ct];
    }
  }
}

void transpose_matrix(float *in, int R, int C, float *out) {
  for (int r = 0; r < R; ++r) for (int c = 0; c < C; ++c) out[c*R + r] = in[r*C + c];
}

void print_matrix(FILE *file, float *M, int R, int C) {  // Formatted as Python list
  fprintf(file, "   [\n");
  for (int r = 0; r < R; ++r) {
    fprintf(file, "    [");
    for (int c = 0; c < C; ++c) {
      fprintf(file, "%f, ", M[r*C + c]);
    }
    fprintf(file, "],\n");
  }
  fprintf(file, "   ],\n");
}

void print_tensor(FILE *file, Tensor t) {  // Formatted as Python list
  fprintf(file, "[\n");
  for (int i0 = 0; i0 < t.shape[0]; ++i0) {
    fprintf(file, "  [\n");
    for (int i1 = 0; i1 < t.shape[1]; ++i1) {
      float *mat = &t.data[index01(t, i0, i1)];
      print_matrix(file, mat, t.shape[2], t.shape[3]);
    }
    fprintf(file, "  ],\n");
  }
  fprintf(file, "]\n");
}

int matmul_flops(int R, int S, int C) { return 2*R*S*C; }

void naive_matmul(float *A, float *B, float *Y, int Na, int Nc, int Nb) { // Shared dim in middle??
  for (int ia = 0; ia < Na; ++ia) {      // A outer dim
    for (int ib = 0; ib < Nb; ++ib) {    // B outer dim
      float acc = 0.f;
      for (int ic = 0; ic < Nc; ++ic)  acc += A[ia*Nc + ic] * B[ic*Nb + ib];
      Y[ia*Nb + ib] = acc;
    }
  }
}


uint64_t nanos() {
  struct timespec start;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  return (uint64_t)start.tv_sec*1000000000 + (uint64_t)start.tv_nsec;
}

#define BLOCK 8
#define BLOCK_Y 4
#define BLOCK_X 2

void preswizzle(float *Bf, float *B, int N, int P) {
  for (int y = 0; y < P; y+=8) {
    for (int x = 0; x < N; x++) {
      for (int iy = 0; iy < 8; iy++) {
        Bf[y*N + x*8 + iy] = B[(y+iy)*N + x];
      }
    }
  }
}

void gemm_preallocated(float *A, float *B, float *C, int M, int N, int P, float *Bf) {
  preswizzle(Bf, B, N, P);

  __m256 *Cm = (__m256*)C;     // SIMD version of output
  __m256 *Bfm = (__m256*)Bf;   // Pre-swizzled SIMD version of B
  __m256 *Bm = (__m256*)B;  

  // Bf = (y/8, k, 8)
  for (int y = 0; y < M; y+=BLOCK_Y) {
    for (int x = 0; x < P; x+=BLOCK*BLOCK_X) {
      __m256 acc[BLOCK_Y][BLOCK_X] = {};
      for (int k = 0; k < N; k++) {
        for (int iy = 0; iy < BLOCK_Y; iy++) {
          __m256 ta = _mm256_broadcast_ss(&A[(y+iy)*N + k]);  
          for (int ix = 0; ix < BLOCK_X; ix++) {
            acc[iy][ix] = _mm256_fmadd_ps(ta, Bfm[((x+ix*BLOCK)*N + k*8)/8], acc[iy][ix]);
          }
        }
      }

      for (int iy = 0; iy < BLOCK_Y; iy++) {
        for (int ix = 0; ix < BLOCK_X; ix++) {
          __m256 _acc = acc[iy][ix];
          Cm[((y+iy)*P + x + ix * BLOCK)/8] = _acc;
        }
      }
    }
  }
}

void simple_gemm(float *A, float *B, float *C, int M, int N, int P, float *unused) {
  for (int y = 0; y < M; y+=1) {
    for (int x = 0; x < P; x+=1) {
      float acc = 0.f;
      for (int k = 0; k < N; k++) {
        acc += A[N*y + k]*B[N*x + k];
      }
      C[P*y + x] = acc;
    }
  }
}
