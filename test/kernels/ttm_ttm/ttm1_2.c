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

int assemble(taco_tensor_t *A3056, taco_tensor_t *B, taco_tensor_t *A2886) {
  int A30561_dimension = (int)(A3056->dimensions[0]);
  int A30563_dimension = (int)(A3056->dimensions[2]);
  int* restrict A30562_pos = (int*)(A3056->indices[1][0]);
  int* restrict A30562_crd = (int*)(A3056->indices[1][1]);
  double* restrict A3056_vals = (double*)(A3056->vals);
  int B1_dimension = (int)(B->dimensions[0]);
  int* restrict B2_pos = (int*)(B->indices[1][0]);
  int* restrict B2_crd = (int*)(B->indices[1][1]);

  A30562_pos = (int32_t*)malloc(sizeof(int32_t) * (A30561_dimension + 1));
  A30562_pos[0] = 0;
  for (int32_t pA30562 = 1; pA30562 < (A30561_dimension + 1); pA30562++) {
    A30562_pos[pA30562] = 0;
  }
  int32_t A30562_crd_size = 1048576;
  A30562_crd = (int32_t*)malloc(sizeof(int32_t) * A30562_crd_size);
  int32_t i1543A3056 = 0;

  for (int32_t i1547 = 0; i1547 < ((B1_dimension + 15) / 16); i1547++) {
    for (int32_t i1548 = 0; i1548 < 16; i1548++) {
      int32_t i1542 = i1547 * 16 + i1548;
      if (i1542 >= B1_dimension)
        continue;

      int32_t pA30562_begin = i1543A3056;

      for (int32_t i1543B = B2_pos[i1542]; i1543B < B2_pos[(i1542 + 1)]; i1543B++) {
        int32_t i1543 = B2_crd[i1543B];
        if (A30562_crd_size <= i1543A3056) {
          A30562_crd = (int32_t*)realloc(A30562_crd, sizeof(int32_t) * (A30562_crd_size * 2));
          A30562_crd_size *= 2;
        }
        A30562_crd[i1543A3056] = i1543;
        i1543A3056++;
      }

      A30562_pos[i1542 + 1] = i1543A3056 - pA30562_begin;
    }
  }

  int32_t csA30562 = 0;
  for (int32_t pA305620 = 1; pA305620 < (A30561_dimension + 1); pA305620++) {
    csA30562 += A30562_pos[pA305620];
    A30562_pos[pA305620] = csA30562;
  }

  A3056_vals = (double*)malloc(sizeof(double) * (i1543A3056 * A30563_dimension));

  A3056->indices[1][0] = (uint8_t*)(A30562_pos);
  A3056->indices[1][1] = (uint8_t*)(A30562_crd);
  A3056->vals = (uint8_t*)A3056_vals;
  return 0;
}

int compute(taco_tensor_t *A3056, taco_tensor_t *B, taco_tensor_t *A2886) {
  int A30561_dimension = (int)(A3056->dimensions[0]);
  int A30563_dimension = (int)(A3056->dimensions[2]);
  double* restrict A3056_vals = (double*)(A3056->vals);
  int B1_dimension = (int)(B->dimensions[0]);
  int* restrict B2_pos = (int*)(B->indices[1][0]);
  int* restrict B2_crd = (int*)(B->indices[1][1]);
  int* restrict B3_pos = (int*)(B->indices[2][0]);
  int* restrict B3_crd = (int*)(B->indices[2][1]);
  double* restrict B_vals = (double*)(B->vals);
  int A28861_dimension = (int)(A2886->dimensions[0]);
  int A28862_dimension = (int)(A2886->dimensions[1]);
  double* restrict A2886_vals = (double*)(A2886->vals);

  // int32_t i1543A3056 = 0;

  #pragma omp parallel for schedule(runtime)
  for (int32_t i1547 = 0; i1547 < ((B1_dimension + 15) / 16); i1547++) {
    for (int32_t i1548 = 0; i1548 < 16; i1548++) {
      int32_t i1542 = i1547 * 16 + i1548;
      if (i1542 >= B1_dimension)
        continue;

      for (int32_t i1543B = B2_pos[i1542]; i1543B < B2_pos[(i1542 + 1)]; i1543B++) {
        for (int32_t i1546 = 0; i1546 < A28862_dimension; i1546++) {
          // int32_t i1546A3056 = i1543A3056 * A30563_dimension + i1546;
          int32_t i1546A3056 = i1543B * A30563_dimension + i1546;
          double ti1544A3056_val = 0.0;
          for (int32_t i1544B = B3_pos[i1543B]; i1544B < B3_pos[(i1543B + 1)]; i1544B++) {
            int32_t i1544 = B3_crd[i1544B];
            int32_t i1546A2886 = i1544 * A28862_dimension + i1546;
            ti1544A3056_val += B_vals[i1544B] * A2886_vals[i1546A2886];
          }
          A3056_vals[i1546A3056] = ti1544A3056_val;
        }
        // i1543A3056++;
      }
    }
  }
  return 0;
}
#include "/home/min/a/kadhitha/workspace/my_taco/taco/test/kernels/ttm_ttm/ttm1_2.h"
int _shim_assemble(void** parameterPack) {
  return assemble((taco_tensor_t*)(parameterPack[0]), (taco_tensor_t*)(parameterPack[1]), (taco_tensor_t*)(parameterPack[2]));
}
int _shim_compute(void** parameterPack) {
  return compute((taco_tensor_t*)(parameterPack[0]), (taco_tensor_t*)(parameterPack[1]), (taco_tensor_t*)(parameterPack[2]));
}
