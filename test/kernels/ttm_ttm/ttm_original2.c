#ifndef TACO_C_HEADERS
#define TACO_C_HEADERS
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <complex.h>
#include <string.h>
#include <omp.h>
#if _OPENMP
#include <omp.h>
#endif
#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
#define TACO_MAX(_a,_b) ((_a) > (_b) ? (_a) : (_b))
#define TACO_DEREF(_a) (((___context___*)(*__ctx__))->_a)
#ifndef TACO_TENSOR_T_DEFINED
#define TACO_TENSOR_T_DEFINED
typedef enum { taco_mode_dense, taco_mode_sparse } taco_mode_t;
typedef struct {
  int32_t      order;         // tensor order (number of modes)
  int32_t*     dimensions;    // tensor dimensions
  int32_t      csize;         // component size
  int32_t*     mode_ordering; // mode storage ordering
  taco_mode_t* mode_types;    // mode storage types
  uint8_t***   indices;       // tensor index data (per mode)
  uint8_t*     vals;          // tensor values
  int32_t      vals_size;     // values array size
} taco_tensor_t;
#endif
#if !_OPENMP
int omp_get_thread_num() { return 0; }
int omp_get_max_threads() { return 1; }
#endif
int cmp(const void *a, const void *b) {
  return *((const int*)a) - *((const int*)b);
}
int taco_binarySearchAfter(int *array, int arrayStart, int arrayEnd, int target) {
  if (array[arrayStart] >= target) {
    return arrayStart;
  }
  int lowerBound = arrayStart; // always < target
  int upperBound = arrayEnd; // always >= target
  while (upperBound - lowerBound > 1) {
    int mid = (upperBound + lowerBound) / 2;
    int midValue = array[mid];
    if (midValue < target) {
      lowerBound = mid;
    }
    else if (midValue > target) {
      upperBound = mid;
    }
    else {
      return mid;
    }
  }
  return upperBound;
}
int taco_binarySearchBefore(int *array, int arrayStart, int arrayEnd, int target) {
  if (array[arrayEnd] <= target) {
    return arrayEnd;
  }
  int lowerBound = arrayStart; // always <= target
  int upperBound = arrayEnd; // always > target
  while (upperBound - lowerBound > 1) {
    int mid = (upperBound + lowerBound) / 2;
    int midValue = array[mid];
    if (midValue < target) {
      lowerBound = mid;
    }
    else if (midValue > target) {
      upperBound = mid;
    }
    else {
      return mid;
    }
  }
  return lowerBound;
}
taco_tensor_t* init_taco_tensor_t(int32_t order, int32_t csize,
                                  int32_t* dimensions, int32_t* mode_ordering,
                                  taco_mode_t* mode_types) {
  taco_tensor_t* t = (taco_tensor_t *) malloc(sizeof(taco_tensor_t));
  t->order         = order;
  t->dimensions    = (int32_t *) malloc(order * sizeof(int32_t));
  t->mode_ordering = (int32_t *) malloc(order * sizeof(int32_t));
  t->mode_types    = (taco_mode_t *) malloc(order * sizeof(taco_mode_t));
  t->indices       = (uint8_t ***) malloc(order * sizeof(uint8_t***));
  t->csize         = csize;
  for (int32_t i = 0; i < order; i++) {
    t->dimensions[i]    = dimensions[i];
    t->mode_ordering[i] = mode_ordering[i];
    t->mode_types[i]    = mode_types[i];
    switch (t->mode_types[i]) {
      case taco_mode_dense:
        t->indices[i] = (uint8_t **) malloc(1 * sizeof(uint8_t **));
        break;
      case taco_mode_sparse:
        t->indices[i] = (uint8_t **) malloc(2 * sizeof(uint8_t **));
        break;
    }
  }
  return t;
}
void deinit_taco_tensor_t(taco_tensor_t* t) {
  for (int i = 0; i < t->order; i++) {
    free(t->indices[i]);
  }
  free(t->indices);
  free(t->dimensions);
  free(t->mode_ordering);
  free(t->mode_types);
  free(t);
}
#endif

int assemble(taco_tensor_t *A1542, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
  int A15421_dimension = (int)(A1542->dimensions[0]);
  int A15423_dimension = (int)(A1542->dimensions[2]);
  int* restrict A15422_pos = (int*)(A1542->indices[1][0]);
  int* restrict A15422_crd = (int*)(A1542->indices[1][1]);
  double* restrict A1542_vals = (double*)(A1542->vals);
  int B1_dimension = (int)(B->dimensions[0]);
  int* restrict B2_pos = (int*)(B->indices[1][0]);
  int* restrict B2_crd = (int*)(B->indices[1][1]);

  A15422_pos = (int32_t*)malloc(sizeof(int32_t) * (A15421_dimension + 1));
  A15422_pos[0] = 0;
  for (int32_t pA15422 = 1; pA15422 < (A15421_dimension + 1); pA15422++) {
    A15422_pos[pA15422] = 0;
  }
  int32_t A15422_crd_size = 1048576;
  A15422_crd = (int32_t*)malloc(sizeof(int32_t) * A15422_crd_size);
  int32_t i1548A1542 = 0;

  for (int32_t i1552 = 0; i1552 < ((B1_dimension + 15) / 16); i1552++) {
    for (int32_t i1553 = 0; i1553 < 16; i1553++) {
      int32_t i1547 = i1552 * 16 + i1553;
      if (i1547 >= B1_dimension)
        continue;

      int32_t pA15422_begin = i1548A1542;

      for (int32_t i1548B = B2_pos[i1547]; i1548B < B2_pos[(i1547 + 1)]; i1548B++) {
        int32_t i1548 = B2_crd[i1548B];
        if (A15422_crd_size <= i1548A1542) {
          A15422_crd = (int32_t*)realloc(A15422_crd, sizeof(int32_t) * (A15422_crd_size * 2));
          A15422_crd_size *= 2;
        }
        A15422_crd[i1548A1542] = i1548;
        i1548A1542++;
      }

      A15422_pos[i1547 + 1] = i1548A1542 - pA15422_begin;
    }
  }

  int32_t csA15422 = 0;
  for (int32_t pA154220 = 1; pA154220 < (A15421_dimension + 1); pA154220++) {
    csA15422 += A15422_pos[pA154220];
    A15422_pos[pA154220] = csA15422;
  }

  A1542_vals = (double*)malloc(sizeof(double) * (i1548A1542 * A15423_dimension));

  A1542->indices[1][0] = (uint8_t*)(A15422_pos);
  A1542->indices[1][1] = (uint8_t*)(A15422_crd);
  A1542->vals = (uint8_t*)A1542_vals;
  return 0;
}

int compute(taco_tensor_t *A1542, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
  int A15421_dimension = (int)(A1542->dimensions[0]);
  int A15423_dimension = (int)(A1542->dimensions[2]);
  int* restrict A15422_pos = (int*)(A1542->indices[1][0]);
  double* restrict A1542_vals = (double*)(A1542->vals);
  int B1_dimension = (int)(B->dimensions[0]);
  int* restrict B2_pos = (int*)(B->indices[1][0]);
  int* restrict B2_crd = (int*)(B->indices[1][1]);
  int* restrict B3_pos = (int*)(B->indices[2][0]);
  int* restrict B3_crd = (int*)(B->indices[2][1]);
  double* restrict B_vals = (double*)(B->vals);
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  double* restrict C_vals = (double*)(C->vals);
  int D1_dimension = (int)(D->dimensions[0]);
  int D2_dimension = (int)(D->dimensions[1]);
  double* restrict D_vals = (double*)(D->vals);

//   int32_t i1548A1542 = 0;

  #pragma omp parallel for schedule(static)
  for (int32_t pA1542 = 0; pA1542 < (A15422_pos[A15421_dimension] * A15423_dimension); pA1542++) {
    A1542_vals[pA1542] = 0.0;
  }

  #pragma omp parallel for schedule(runtime)
  for (int32_t i1552 = 0; i1552 < ((B1_dimension + 15) / 16); i1552++) {
    for (int32_t i1553 = 0; i1553 < 16; i1553++) {
      int32_t i1547 = i1552 * 16 + i1553;
      if (i1547 >= B1_dimension)
        continue;

      for (int32_t i1548B = B2_pos[i1547]; i1548B < B2_pos[(i1547 + 1)]; i1548B++) {
        for (int32_t i1549B = B3_pos[i1548B]; i1549B < B3_pos[(i1548B + 1)]; i1549B++) {
          int32_t i1549 = B3_crd[i1549B];
          for (int32_t i1550 = 0; i1550 < D1_dimension; i1550++) {
            int32_t i1550C = i1549 * C2_dimension + i1550;
            for (int32_t i1551 = 0; i1551 < D2_dimension; i1551++) {
            //   int32_t i1551A1542 = i1548A1542 * A15423_dimension + i1551;
              int32_t i1551A1542 = i1548B * A15423_dimension + i1551;
              int32_t i1551D = i1550 * D2_dimension + i1551;
              A1542_vals[i1551A1542] = A1542_vals[i1551A1542] + (B_vals[i1549B] * C_vals[i1550C]) * D_vals[i1551D];
            }
          }
        }
        // i1548A1542++;
      }
    }
  }
  return 0;
}
#include "/home/min/a/kadhitha/workspace/my_taco/taco/test/kernels/ttm_ttm/ttm_original2.h"
int _shim_assemble(void** parameterPack) {
  return assemble((taco_tensor_t*)(parameterPack[0]), (taco_tensor_t*)(parameterPack[1]), (taco_tensor_t*)(parameterPack[2]), (taco_tensor_t*)(parameterPack[3]));
}
int _shim_compute(void** parameterPack) {
  return compute((taco_tensor_t*)(parameterPack[0]), (taco_tensor_t*)(parameterPack[1]), (taco_tensor_t*)(parameterPack[2]), (taco_tensor_t*)(parameterPack[3]));
}
