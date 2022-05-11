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

int assemble(taco_tensor_t *A2593, taco_tensor_t *A2398, taco_tensor_t *D) {
  int A25931_dimension = (int)(A2593->dimensions[0]);
  int A25933_dimension = (int)(A2593->dimensions[2]);
  int* restrict A25932_pos = (int*)(A2593->indices[1][0]);
  int* restrict A25932_crd = (int*)(A2593->indices[1][1]);
  double* restrict A2593_vals = (double*)(A2593->vals);
  int A23981_dimension = (int)(A2398->dimensions[0]);
  int* restrict A23982_pos = (int*)(A2398->indices[1][0]);
  int* restrict A23982_crd = (int*)(A2398->indices[1][1]);

  A25932_pos = (int32_t*)malloc(sizeof(int32_t) * (A25931_dimension + 1));
  A25932_pos[0] = 0;
  for (int32_t pA25932 = 1; pA25932 < (A25931_dimension + 1); pA25932++) {
    A25932_pos[pA25932] = 0;
  }
  int32_t A25932_crd_size = 1048576;
  A25932_crd = (int32_t*)malloc(sizeof(int32_t) * A25932_crd_size);
  int32_t i1543A2593 = 0;

  for (int32_t i1547 = 0; i1547 < ((A23981_dimension + 15) / 16); i1547++) {
    for (int32_t i1548 = 0; i1548 < 16; i1548++) {
      int32_t i1542 = i1547 * 16 + i1548;
      if (i1542 >= A23981_dimension)
        continue;

      int32_t pA25932_begin = i1543A2593;

      for (int32_t i1543A2398 = A23982_pos[i1542]; i1543A2398 < A23982_pos[(i1542 + 1)]; i1543A2398++) {
        int32_t i1543 = A23982_crd[i1543A2398];
        if (A25932_crd_size <= i1543A2593) {
          A25932_crd = (int32_t*)realloc(A25932_crd, sizeof(int32_t) * (A25932_crd_size * 2));
          A25932_crd_size *= 2;
        }
        A25932_crd[i1543A2593] = i1543;
        i1543A2593++;
      }

      A25932_pos[i1542 + 1] = i1543A2593 - pA25932_begin;
    }
  }

  int32_t csA25932 = 0;
  for (int32_t pA259320 = 1; pA259320 < (A25931_dimension + 1); pA259320++) {
    csA25932 += A25932_pos[pA259320];
    A25932_pos[pA259320] = csA25932;
  }

  A2593_vals = (double*)malloc(sizeof(double) * (i1543A2593 * A25933_dimension));

  A2593->indices[1][0] = (uint8_t*)(A25932_pos);
  A2593->indices[1][1] = (uint8_t*)(A25932_crd);
  A2593->vals = (uint8_t*)A2593_vals;
  return 0;
}

int compute(taco_tensor_t *A2593, taco_tensor_t *A2398, taco_tensor_t *D) {
  int A25931_dimension = (int)(A2593->dimensions[0]);
  int A25933_dimension = (int)(A2593->dimensions[2]);
  double* restrict A2593_vals = (double*)(A2593->vals);
  int A23981_dimension = (int)(A2398->dimensions[0]);
  int A23983_dimension = (int)(A2398->dimensions[2]);
  int* restrict A23982_pos = (int*)(A2398->indices[1][0]);
  int* restrict A23982_crd = (int*)(A2398->indices[1][1]);
  double* restrict A2398_vals = (double*)(A2398->vals);
  int D1_dimension = (int)(D->dimensions[0]);
  int D2_dimension = (int)(D->dimensions[1]);
  double* restrict D_vals = (double*)(D->vals);

//   int32_t i1543A2593 = 0;

  #pragma omp parallel for schedule(runtime)
  for (int32_t i1547 = 0; i1547 < ((A23981_dimension + 15) / 16); i1547++) {
    for (int32_t i1548 = 0; i1548 < 16; i1548++) {
      int32_t i1542 = i1547 * 16 + i1548;
      if (i1542 >= A23981_dimension)
        continue;

      for (int32_t i1543A2398 = A23982_pos[i1542]; i1543A2398 < A23982_pos[(i1542 + 1)]; i1543A2398++) {
        for (int32_t i1546 = 0; i1546 < D2_dimension; i1546++) {
        //   int32_t i1546A2593 = i1543A2593 * A25933_dimension + i1546;
          int32_t i1546A2593 = i1543A2398 * A25933_dimension + i1546;
          double ti1545A2593_val = 0.0;
          for (int32_t i1545 = 0; i1545 < D1_dimension; i1545++) {
            int32_t i1545A2398 = i1543A2398 * A23983_dimension + i1545;
            int32_t i1546D = i1545 * D2_dimension + i1546;
            ti1545A2593_val += A2398_vals[i1545A2398] * D_vals[i1546D];
          }
          A2593_vals[i1546A2593] = ti1545A2593_val;
        }
        // i1543A2593++;
      }
    }
  }
  return 0;
}
#include "/home/min/a/kadhitha/workspace/my_taco/taco/test/kernels/ttm_ttm/ttm2.h"
int _shim_assemble(void** parameterPack) {
  return assemble((taco_tensor_t*)(parameterPack[0]), (taco_tensor_t*)(parameterPack[1]), (taco_tensor_t*)(parameterPack[2]));
}
int _shim_compute(void** parameterPack) {
  return compute((taco_tensor_t*)(parameterPack[0]), (taco_tensor_t*)(parameterPack[1]), (taco_tensor_t*)(parameterPack[2]));
}
