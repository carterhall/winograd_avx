#include "../helpers.h"

void test_indexing() {
  printf("Testing tensor indexing...\n");

  Tensor t = { .shape={2,3,4,5}, .data=malloc(sizeof(float)*2*3*4*5) };
  for (int i = 0; i < 2*3*4*5; ++i) t.data[i] = i;

  printf("Printing 3-tensor retrieved with index0\n");
  float *t3 = &t.data[index0(t, 1)];
  for (int i = 0; i < 3; ++i) {
    printf("--------\n");
    print_matrix(stdout, &t3[i*4*5], 4, 5);
  }
  printf("--------\n\n");

  printf("Printing matrix retrieved with index01\n");
  float *m = &t.data[index01(t, 1, 2)]; 
  print_matrix(stdout, m, 4, 5);
  printf("\n");

  printf("Printing vector retrieved with index012\n");
  float *v = &t.data[index012(t, 1, 0, 0)];
  for (int i = 0; i < 5; ++i) printf("%f, ", v[i]);
  printf("\n\n");

  printf("Printing scalar retrieved with index0123\n");
  float *s = &t.data[index0123(t, 1, 0, 1, 0)];
  printf("%f", s[0]);
  printf("\n");

  free(t.data);
  printf("\n");
}

void test_tiling() { 
  printf("Testing matrix_get_tile...\n");

  int RM = 8;  // Matrix dims
  int CM = 6;
  int RT = 3;  // Tile dims
  int CT = 2;
  int rmstart = 1; // Start index within matrix
  int cmstart = 2;

  float *matrix = malloc(sizeof(float)*RM*CM);
  for (int i = 0; i < RM*CM; ++i) matrix[i] = i;

  float *tile   = malloc(sizeof(float)*RT*CT);
  matrix_get_tile(tile, matrix, RM, CM, RT, CT, rmstart, cmstart);

  printf("Original matrix:\n");
  print_matrix(stdout, matrix, RM, CM);
  printf("\n");

  printf("Tile, size (%d, %d), start index (%d, %d):\n", RT, CT, rmstart, cmstart);
  print_matrix(stdout, tile, RT, CT);
  printf("\n");

  printf("Negating that tile within the original matrix:\n");
  for (int i = 0; i < RT*CT; ++i) tile[i] = -tile[i];
  matrix_set_tile(tile, matrix, RM, CM, RT, CT, rmstart, cmstart);
  print_matrix(stdout, matrix, RM, CM);
  printf("\n");

  free(matrix); free(tile);
}

void test_strided_tiling() { 
  printf("Testing matrix_get_tile_column_strided...\n");

  int RM = 8;  // Matrix dims
  int CM = 6;
  int RT = 3;  // Tile dims
  int CT = 2;
  int rmstart = 1; // Start index within matrix
  int cmstart = 2;

  int column_stride = 3;

  float *matrix = calloc(RM*CM, sizeof(float));
  for (int i = 0; i < RM*CM; ++i) matrix[i] = i;

  float *tile   = calloc(RT*CT, sizeof(float));
  matrix_get_tile_column_strided(tile, matrix, RM, CM, RT, CT, rmstart, cmstart, column_stride);

  printf("Original matrix:\n");
  print_matrix(stdout, matrix, RM, CM);
  printf("\n");

  printf("Tile, size (%d, %d), start index (%d, %d), column stride = %d:\n", RT, CT, rmstart, cmstart, column_stride);
  print_matrix(stdout, tile, RT, CT);
  printf("\n");

  free(matrix); free(tile);
}

int main() {
  test_indexing(); 
  test_tiling();
  test_strided_tiling();
}
