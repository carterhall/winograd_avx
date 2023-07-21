#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>

#include "helpers.h"

//#define PROFILE

#define OUTPUT_TILE_DIM 2   // This must be 2, our transforms are hardcoded for (2x2,3x3).
#define ALPHA (OUTPUT_TILE_DIM + 3 - 1)  // Input tile dim

void transform_kernel_avx(__m256 *kernel, __m256 *transformed_kernel, int stride) {  // G @ kernel @ G_T
  __m256 tmp[4*3];
  for (int i = 0; i < 3; ++i) {
    tmp[0*3 + i] = kernel[stride*(0*3 + i)];
    tmp[1*3 + i] = _mm256_mul_ps(_mm256_set1_ps(0.5f), _mm256_add_ps(kernel[stride*(0*3 + i)], _mm256_add_ps(kernel[stride*(1*3 + i)], kernel[stride*(2*3 + i)])));
    tmp[2*3 + i] = _mm256_mul_ps(_mm256_set1_ps(0.5f), _mm256_sub_ps(kernel[stride*(0*3 + i)], _mm256_sub_ps(kernel[stride*(1*3 + i)], kernel[stride*(2*3 + i)])));
    tmp[3*3 + i] = kernel[stride*(2*3 + i)];
  }

  for (int i = 0; i < 4; ++i) {
    transformed_kernel[0 + 4*i] = tmp[0 + 3*i];
    transformed_kernel[1 + 4*i] = _mm256_mul_ps(_mm256_set1_ps(0.5f), _mm256_add_ps(tmp[0 + 3*i], _mm256_add_ps(tmp[1 + 3*i], tmp[2 + 3*i])));
    transformed_kernel[2 + 4*i] = _mm256_mul_ps(_mm256_set1_ps(0.5f), _mm256_sub_ps(tmp[0 + 3*i], _mm256_sub_ps(tmp[1 + 3*i], tmp[2 + 3*i])));
    transformed_kernel[3 + 4*i] = tmp[2 + 3*i];
  }
}

void transform_kernels(Tensor kernels, Tensor U) {
  __m256 u[ALPHA*ALPHA];
  int K = kernels.shape[0], C = kernels.shape[3];

  for (int k = 0; k < K; ++k) {
    for (int c = 0; c < C; c += 8) {
      float *g_kc = &kernels.data[index0(kernels, k)] + c;
      transform_kernel_avx((__m256*)g_kc, (__m256*)u, C/8);
      for (int tx = 0; tx < ALPHA; ++tx) {
        for (int ty = 0; ty < ALPHA; ++ty) {
          __m256 *Ublock = (__m256*) &U.data[index01(U, tx, ty)];
          Ublock[(k*C + c) >> 3] = u[tx*ALPHA + ty];
        }
      }
    }
  }
}

void transform_input_tile_avx(__m256 *input_tile, __m256 *transformed_tile) {  // Hardcoded B_T @ d_cb @ B
  __m256 tmp[16];
  for (int i = 0; i < 4; ++i) {    // tmp = B_T @ input_tile
    tmp[0*4 + i] = _mm256_sub_ps(input_tile[0*4 + i], input_tile[2*4 + i]);
    tmp[1*4 + i] = _mm256_add_ps(input_tile[1*4 + i], input_tile[2*4 + i]);
    tmp[2*4 + i] = _mm256_sub_ps(input_tile[2*4 + i], input_tile[1*4 + i]);
    tmp[3*4 + i] = _mm256_sub_ps(input_tile[1*4 + i], input_tile[3*4 + i]);
  }
  for (int i = 0; i < 4; ++i) {    // m256_transformed_tile = tmp @ B
    transformed_tile[0 + 4*i] = _mm256_sub_ps(tmp[4*i + 0], tmp[4*i + 2]);
    transformed_tile[1 + 4*i] = _mm256_add_ps(tmp[4*i + 1], tmp[4*i + 2]);
    transformed_tile[2 + 4*i] = _mm256_sub_ps(tmp[4*i + 2], tmp[4*i + 1]);
    transformed_tile[3 + 4*i] = _mm256_sub_ps(tmp[4*i + 1], tmp[4*i + 3]);
  }
}

void transform_inputs(Tensor inputs, Tensor V) {
  __m256 v[ALPHA*ALPHA];
  __m256 d_cb[ALPHA*ALPHA];
  __m256 tmp[ALPHA*ALPHA];

  int BS = inputs.shape[0];   // Paper calls this 'N' and indexes it with 'i'
  int H  = inputs.shape[1]; 
  int W  = inputs.shape[2];
  int C  = inputs.shape[3]; 

  int num_tiles = BS * ceilf((float)H/OUTPUT_TILE_DIM) * ceilf((float)W/OUTPUT_TILE_DIM);
  int num_tiles_per_col = ceilf((float)BS*H/OUTPUT_TILE_DIM);
  int num_tiles_per_row = ceilf((float)W/OUTPUT_TILE_DIM);

  int itile = 0;
  for (int rtile = 0; rtile < BS*H; rtile += OUTPUT_TILE_DIM) {
    for (int ctile = 0; ctile < W; ctile += OUTPUT_TILE_DIM) {
      for (int c = 0; c < C; c += 8) {
        matrix_get_tile_column_strided_avx(d_cb, inputs.data, BS*H, W*C, ALPHA, ALPHA, rtile, C*ctile + c, C);
        transform_input_tile_avx(d_cb, v);
        for (int tx = 0; tx < ALPHA; ++tx) {
          for (int ty = 0; ty < ALPHA; ++ty) {
            __m256 *Vblock = (__m256*) &V.data[index01(V, tx, ty)];
            Vblock[(itile*C + c) >> 3] = v[tx*ALPHA + ty];  // Vblock is made of m256 now! Div index by 8!
          }
        }
      }
      itile += 1;
    }
  }
}

void transform_output_tile_avx(__m256 *output_tile, __m256 *transformed_tile) {  // Hardcoded B_T @ d_cb @ B
  __m256 tmp[16];
  for (int i = 0; i < 4; ++i) {    // tmp = A_T @ output_tile
    tmp[0*4 + i] = _mm256_add_ps(output_tile[0*4 + i], _mm256_add_ps(output_tile[1*4 + i], output_tile[2*4 + i]));
    tmp[1*4 + i] = _mm256_sub_ps(output_tile[1*4 + i], _mm256_add_ps(output_tile[2*4 + i], output_tile[3*4 + i]));
  }
  for (int i = 0; i < 2; ++i) {    // transformed_tile = tmp @ A
    transformed_tile[0 + 2*i] = _mm256_add_ps(tmp[4*i + 0], _mm256_add_ps(tmp[4*i + 1], tmp[4*i + 2]));
    transformed_tile[1 + 2*i] = _mm256_sub_ps(tmp[4*i + 1], _mm256_add_ps(tmp[4*i + 2], tmp[4*i + 3]));
  }
}

void transform_outputs(Tensor M, Tensor outputs) {
  __m256 Mpart[ALPHA*ALPHA];
  __m256 ytile[OUTPUT_TILE_DIM*OUTPUT_TILE_DIM];

  int K = M.shape[2];
  int BS = outputs.shape[1];
  int H  = outputs.shape[2];
  int W  = outputs.shape[3];

  int num_tiles = M.shape[3];
  int num_tiles_per_row = ceilf((float)W/OUTPUT_TILE_DIM);

  int Y_shape[4] = {K, BS*H, W, 1};
  Tensor Y = view(outputs, Y_shape);

  for (int k = 0; k < K; ++k) {
    float *Y_k = &Y.data[index0(Y, k)];

    for (int itile = 0; itile < num_tiles; itile += 8) {
      for (int tx = 0; tx < ALPHA; ++tx) {
        for (int ty = 0; ty < ALPHA; ++ty) {
          Mpart[tx*ALPHA + ty] = *((__m256*) &M.data[index0123(M, tx, ty, k, itile)]);
        }
      }

      transform_output_tile_avx(Mpart, ytile);

      for (int i = 0; i < 8; ++i) {
        int rtile = ((itile + i) / num_tiles_per_row) * OUTPUT_TILE_DIM;
        int ctile = ((itile + i) % num_tiles_per_row) * OUTPUT_TILE_DIM;

        // Strided in the tile, not the matrix! Opposite of our other strided function!
        matrix_set_tile_column_strided((float*)ytile, Y_k, BS*H, W, OUTPUT_TILE_DIM, OUTPUT_TILE_DIM, rtile, ctile, 8, i);
      }
    }
  }
}

float *allocate_winograd_temp_storage(int BS, int K, int C, int H, int W) {
  int num_tiles = BS * ceilf((float)H/OUTPUT_TILE_DIM) * ceilf((float)W/OUTPUT_TILE_DIM);
  int Umem = ALPHA*ALPHA*K*C, Vmem = ALPHA*ALPHA*C*num_tiles, Mmem = ALPHA*ALPHA*K*num_tiles;
  int gemm_mem = C*num_tiles;

  printf("Input tensor: %f MB, output tensor: %f MB.\n", sizeof(float)*BS*C*H*W/1e6, sizeof(float)*K*C*H*W/1e6);
  printf("Preallocating %f MB for temp storage.\n", (Umem+Vmem+Mmem+gemm_mem)*sizeof(float)/1e6);
  return aligned_alloc(64, sizeof(float)*(Umem+Vmem+Mmem+gemm_mem));
}

void winograd_3x3(Tensor kernels, Tensor inputs, Tensor outputs, float *temp_storage) {
  uint64_t t_start = nanos();

  int BS = inputs.shape[0], H = inputs.shape[1], W = inputs.shape[2], C = inputs.shape[3], K = kernels.shape[0];
  int num_tiles = BS * ceilf((float)H/OUTPUT_TILE_DIM) * ceilf((float)W/OUTPUT_TILE_DIM);

  Tensor U = { .shape={ALPHA,ALPHA,K,C},         .data=temp_storage };  temp_storage += ALPHA*ALPHA*K*C;
  Tensor V = { .shape={ALPHA,ALPHA,num_tiles,C}, .data=temp_storage };  temp_storage += ALPHA*ALPHA*C*num_tiles;
  Tensor M = { .shape={ALPHA,ALPHA,K,num_tiles}, .data=temp_storage };  temp_storage += ALPHA*ALPHA*K*num_tiles;
  float *gemm_mem = temp_storage;

  transform_kernels(kernels, U);
  uint64_t t_U = nanos();

  transform_inputs(inputs, V);
  uint64_t t_V = nanos();

  for (int tx = 0; tx < ALPHA; ++tx) {
    for (int ty = 0; ty < ALPHA; ++ty) {
      float *Ublock = &U.data[index01(U, tx, ty)];
      float *Vblock = &V.data[index01(V, tx, ty)];
      float *Mblock = &M.data[index01(M, tx, ty)];
      gemm_preallocated(Ublock, Vblock, Mblock, K, C, num_tiles, gemm_mem); // Note: Vblock is transposed
    }
  }
  uint64_t t_gemm = nanos();

  transform_outputs(M, outputs);
  uint64_t t_Y = nanos();

  #ifdef PROFILE
  double U_seconds = (t_U - t_start)*1e-9, V_seconds = (t_V - t_U)*1e-9, gemm_seconds = (t_gemm - t_V)*1e-9;
  double Y_seconds = (t_Y - t_gemm)*1e-9,  total_seconds = (t_Y - t_start)*1e-9;
  printf("Breakdown of actual Winograd execution: %d loops transforming kernels, %d for inputs, %d for outputs.\n", K*C, num_tiles*C, num_tiles*K);
  printf("  %f%% (%fus): transforming kernels\n", U_seconds/total_seconds*100., U_seconds*1e6);
  printf("  %f%% (%fus): transforming inputs\n", V_seconds/total_seconds*100., V_seconds*1e6);
  printf("  %f%% (%fus): multiplying the large matrices\n", gemm_seconds/total_seconds*100., gemm_seconds*1e6);
  printf("  %f%% (%fus): transforming outputs\n\n", Y_seconds/total_seconds*100., Y_seconds*1e6);
  #endif
}

int main() {
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  //int BS = 32, K = 64, C = 64, H = 32, W = 32; // Torch conv2d beats us with these params (using FFT?)
  int BS = 4, K = 512, C = 512, H = 24, W = 24; // VGG 4.2ish; Winograd should beat FFT on CPU with these params

  float *temp_storage = allocate_winograd_temp_storage(BS, K, C, H, W);
  Tensor inputs  = { .shape={BS,H,W,C}, .data=aligned_alloc(64, sizeof(float)*C*BS*H*W) };
  Tensor kernels = { .shape={K,3,3,C},  .data=aligned_alloc(64, sizeof(float)*K*C*3*3) };
  Tensor outputs = { .shape={K,BS,H,W}, .data=aligned_alloc(64, sizeof(float)*K*BS*H*W) };

  #if 1  // Initialize with random numbers if 1, sequential if 0
  srand(1337);
  for (int i = 0; i < C*BS*H*W; ++i) inputs.data[i] = 2.f*(float)rand()/(float)(RAND_MAX) - 1.f; 
  for (int i = 0; i < K*C*3*3; ++i) kernels.data[i] = 2.f*(float)rand()/(float)(RAND_MAX) - 1.f;
  #else    // Sequential inputs for easier debugging
  int itotal = 0;
  for (int batch = 0; batch < BS; ++batch) // Leave this the same for all data layouts
    for (int c = 0; c < C; ++c)
      for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w)
          inputs.data[index0123(inputs, batch, h, w, c)] = itotal++;  // Update this for a particular layout
  for (int k = 0; k < K; ++k)
    for (int c = 0; c < C; ++c) 
      for (int kr = 0; kr < 3; ++kr) 
        for (int kc = 0; kc < 3; ++kc) 
          kernels.data[index0123(kernels, k, kr, kc, c)] = itotal++;
  #endif

  // Calculate the number of ops (treating the various transforms as actual matrix multiplications)
  float num_tiles = (float)BS * ceilf((float)H/OUTPUT_TILE_DIM) * ceilf((float)W/OUTPUT_TILE_DIM);
  float u_flop = (float)K*C*(matmul_flops(ALPHA,3,3)+matmul_flops(ALPHA,3,ALPHA));
  float v_flop = (float)num_tiles*C*(matmul_flops(ALPHA,ALPHA,ALPHA) + matmul_flops(ALPHA,ALPHA,ALPHA));
  float matmul_flop = (float)ALPHA*ALPHA*matmul_flops(K,C,num_tiles);
  float y_flop = (float)K*num_tiles*(matmul_flops(OUTPUT_TILE_DIM,ALPHA,ALPHA) + matmul_flops(OUTPUT_TILE_DIM,ALPHA,OUTPUT_TILE_DIM));
  float total_flop = u_flop + v_flop + matmul_flop + y_flop;
  printf("Batch size %d with %d channels, %d kernels, image size %dx%d\n", BS, C, K, W, H);
  printf("Winograd theoretical complexity: %f GFLOPs.\n", total_flop*1e-9);

  int n_iters = 16;    // Run a number of iterations to smooth out any warmup inconsistencies
  //int n_iters = 256;
  //for (int iter = 0; iter < n_iters; ++iter)  winograd_3x3(kernels, inputs, outputs, temp_storage); // Warmup
  printf("Calling winograd %d times...\n", n_iters);
  uint64_t start = nanos();
  for (int iter = 0; iter < n_iters; ++iter)  winograd_3x3(kernels, inputs, outputs, temp_storage);
  uint64_t end = nanos();
  double seconds = (end - start)*1e-9 / n_iters;

  FILE *file = fopen("tmp_data.py", "w");    // Write data that Python can load, for validation
  fprintf(file, "import torch\n");
  fprintf(file, "OUTPUT_TILE_DIM = %d\ntotal_flop = %f\n", OUTPUT_TILE_DIM, total_flop);
  fprintf(file, "BS, C, H, W, K = %d, %d, %d, %d, %d\n", BS, C, H, W, K);
  fprintf(file, "inputs = torch.tensor(");   print_tensor(file, inputs);   fprintf(file, ")\n");
  fprintf(file, "kernels = torch.tensor(");  print_tensor(file, kernels);  fprintf(file, ")\n");
  fprintf(file, "outputs = torch.tensor(");  print_tensor(file, outputs);  fprintf(file, ")\n");
  fclose(file);

  #ifdef PROFILE
  printf("Theoretical complexity breakdown for current parameters:\n");
  printf("  %.2f%%: transforming kernels\n", 100.f*u_flop / total_flop);
  printf("  %.2f%%: transforming inputs\n", 100.f*v_flop / total_flop);
  printf("  %.2f%%: multiplying the large matrices\n", 100.f*matmul_flop / total_flop);
  printf("  %.2f%%: transforming outputs\n", 100.f*y_flop / total_flop);
  #endif

  printf("\nWe're achieving %f GFLOP/S, %f ms per iteration.\n", total_flop/seconds*1e-9, seconds*1e3);
  free(inputs.data); free(kernels.data); free(outputs.data); free(temp_storage);
}
