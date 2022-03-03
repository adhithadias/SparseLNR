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

int assemble(taco_tensor_t *A2886, taco_tensor_t *C, taco_tensor_t *D) {
  int A28861_dimension = (int)(A2886->dimensions[0]);
  int A28862_dimension = (int)(A2886->dimensions[1]);
  double* restrict A2886_vals = (double*)(A2886->vals);

  A2886_vals = (double*)malloc(sizeof(double) * (A28861_dimension * A28862_dimension));

  A2886->vals = (uint8_t*)A2886_vals;
  return 0;
}

int compute(taco_tensor_t *A2886, taco_tensor_t *C, taco_tensor_t *D) {
  int A28861_dimension = (int)(A2886->dimensions[0]);
  int A28862_dimension = (int)(A2886->dimensions[1]);
  double* restrict A2886_vals = (double*)(A2886->vals);
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  double* restrict C_vals = (double*)(C->vals);
  int D1_dimension = (int)(D->dimensions[0]);
  int D2_dimension = (int)(D->dimensions[1]);
  double* restrict D_vals = (double*)(D->vals);

  #pragma omp parallel for schedule(static)
  for (int32_t pA2886 = 0; pA2886 < (A28861_dimension * A28862_dimension); pA2886++) {
    A2886_vals[pA2886] = 0.0;
  }

  #pragma omp parallel for schedule(runtime)
  for (int32_t i1551 = 0; i1551 < ((C1_dimension + 31) / 32); i1551++) {
    for (int32_t i1553 = 0; i1553 < ((D1_dimension + 31) / 32); i1553++) {
      for (int32_t i1555 = 0; i1555 < ((D2_dimension + 31) / 32); i1555++) {
        for (int32_t i1552 = 0; i1552 < 32; i1552++) {
          int32_t i1544 = i1551 * 32 + i1552;
          if (i1544 >= C1_dimension)
            continue;

          for (int32_t i1554 = 0; i1554 < 32; i1554++) {
            int32_t i1545 = i1553 * 32 + i1554;
            int32_t i1545C = i1544 * C2_dimension + i1545;
            if (i1545 >= D1_dimension)
              continue;

            for (int32_t i1556 = 0; i1556 < 32; i1556++) {
              int32_t i1546 = i1555 * 32 + i1556;
              int32_t i1546D = i1545 * D2_dimension + i1546;
              int32_t i1546A2886 = i1544 * A28862_dimension + i1546;
              if (i1546 >= D2_dimension)
                continue;

              A2886_vals[i1546A2886] = A2886_vals[i1546A2886] + C_vals[i1545C] * D_vals[i1546D];
            }
          }
        }
      }
    }
  }
  return 0;
}
#include "/home/min/a/kadhitha/workspace/my_taco/taco/test/kernels/ttm_ttm/gemm.h"
int _shim_assemble(void** parameterPack) {
  return assemble((taco_tensor_t*)(parameterPack[0]), (taco_tensor_t*)(parameterPack[1]), (taco_tensor_t*)(parameterPack[2]));
}
int _shim_compute(void** parameterPack) {
  return compute((taco_tensor_t*)(parameterPack[0]), (taco_tensor_t*)(parameterPack[1]), (taco_tensor_t*)(parameterPack[2]));
}
