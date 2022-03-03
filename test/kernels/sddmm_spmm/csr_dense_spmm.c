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

int assemble(taco_tensor_t *A2535, taco_tensor_t *A2531, taco_tensor_t *A1455) {
  int A25352_dimension = (int)(A2535->dimensions[1]);
  double* restrict A2535_vals = (double*)(A2535->vals);

  A2535_vals = (double*)malloc(sizeof(double) * (5 * A25352_dimension));

  A2535->vals = (uint8_t*)A2535_vals;
  return 0;
}

int compute(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B) {
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  double* restrict C_vals = (double*)(C->vals);
  int A1_dimension = (int)(A->dimensions[0]);
  int* restrict A2_pos = (int*)(A->indices[1][0]);
  int* restrict A2_crd = (int*)(A->indices[1][1]);
  double* restrict A_vals = (double*)(A->vals);
  int B1_dimension = (int)(B->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  double* restrict B_vals = (double*)(B->vals);

  #pragma omp parallel for schedule(static)
  for (int32_t pC = 0; pC < (C1_dimension * C2_dimension); pC++) {
    C_vals[pC] = 0.0;
  }

  #pragma omp parallel for schedule(dynamic, 1)
  for (int32_t i0 = 0; i0 < ((A1_dimension + 15) / 16); i0++) {
    for (int32_t i1 = 0; i1 < 16; i1++) {
      int32_t i = i0 * 16 + i1;
      if (i >= A1_dimension)
        continue;

      for (int32_t jpos0 = A2_pos[i] / 4; jpos0 < ((A2_pos[(i + 1)] + 3) / 4); jpos0++) {
        int32_t jposA = jpos0 * 4;
        if (jpos0 * 4 < A2_pos[i] || (jpos0 * 4 + 4) + ((jpos0 * 4 + 4) - jpos0 * 4) >= A2_pos[(i + 1)]) {
          for (int32_t k = 0; k < B2_dimension; k++) {
            int32_t kC = i * C2_dimension + k;
            for (int32_t jpos1 = 0; jpos1 < 4; jpos1++) {
              int32_t jposA = jpos0 * 4 + jpos1;
              if (jposA < A2_pos[i] || jposA >= A2_pos[(i + 1)])
                continue;

              int32_t j = A2_crd[jposA];
              int32_t kB = j * B2_dimension + k;
              C_vals[kC] = C_vals[kC] + A_vals[jposA] * B_vals[kB];
            }
          }
        }
        else {
          #pragma clang loop interleave(enable) vectorize(enable)
          for (int32_t k = 0; k < B2_dimension; k++) {
            int32_t kC = i * C2_dimension + k;
            for (int32_t jpos1 = 0; jpos1 < 4; jpos1++) {
              int32_t jposA = jpos0 * 4 + jpos1;
              int32_t j = A2_crd[jposA];
              int32_t kB = j * B2_dimension + k;
              C_vals[kC] = C_vals[kC] + A_vals[jposA] * B_vals[kB];
            }
          }
        }
      }
    }
  }
  return 0;
}
#include "/home/min/a/kadhitha/workspace/my_taco/taco/test/kernels/sddmm_spmm/csr_dense_spmm.h"
int _shim_assemble(void** parameterPack) {
  return assemble((taco_tensor_t*)(parameterPack[0]), (taco_tensor_t*)(parameterPack[1]), (taco_tensor_t*)(parameterPack[2]));
}
int _shim_compute(void** parameterPack) {
  return compute((taco_tensor_t*)(parameterPack[0]), (taco_tensor_t*)(parameterPack[1]), (taco_tensor_t*)(parameterPack[2]));
}
