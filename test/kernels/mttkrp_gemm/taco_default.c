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

int assemble(taco_tensor_t *A1538, taco_tensor_t *matmul_5_5_5, taco_tensor_t *A1475, taco_tensor_t *A1416, taco_tensor_t *A1479) {
  int A15381_dimension = (int)(A1538->dimensions[0]);
  int A15382_dimension = (int)(A1538->dimensions[1]);
  double* restrict A1538_vals = (double*)(A1538->vals);

  A1538_vals = (double*)malloc(sizeof(double) * (A15381_dimension * A15382_dimension));

  A1538->vals = (uint8_t*)A1538_vals;
  return 0;
}

int compute(taco_tensor_t *A1538, taco_tensor_t *matmul_5_5_5, taco_tensor_t *A1475, taco_tensor_t *A1416, taco_tensor_t *A1479) {
  int A15381_dimension = (int)(A1538->dimensions[0]);
  int A15382_dimension = (int)(A1538->dimensions[1]);
  double* restrict A1538_vals = (double*)(A1538->vals);
  int* restrict matmul_5_5_51_pos = (int*)(matmul_5_5_5->indices[0][0]);
  int* restrict matmul_5_5_51_crd = (int*)(matmul_5_5_5->indices[0][1]);
  int* restrict matmul_5_5_52_pos = (int*)(matmul_5_5_5->indices[1][0]);
  int* restrict matmul_5_5_52_crd = (int*)(matmul_5_5_5->indices[1][1]);
  int* restrict matmul_5_5_53_pos = (int*)(matmul_5_5_5->indices[2][0]);
  int* restrict matmul_5_5_53_crd = (int*)(matmul_5_5_5->indices[2][1]);
  double* restrict matmul_5_5_5_vals = (double*)(matmul_5_5_5->vals);
  int A14751_dimension = (int)(A1475->dimensions[0]);
  int A14752_dimension = (int)(A1475->dimensions[1]);
  double* restrict A1475_vals = (double*)(A1475->vals);
  int A14161_dimension = (int)(A1416->dimensions[0]);
  int A14162_dimension = (int)(A1416->dimensions[1]);
  double* restrict A1416_vals = (double*)(A1416->vals);
  int A14791_dimension = (int)(A1479->dimensions[0]);
  int A14792_dimension = (int)(A1479->dimensions[1]);
  double* restrict A1479_vals = (double*)(A1479->vals);

  #pragma omp parallel for schedule(static)
  for (int32_t pA1538 = 0; pA1538 < (A15381_dimension * A15382_dimension); pA1538++) {
    A1538_vals[pA1538] = 0.0;
  }

  #pragma omp parallel for schedule(runtime)
  for (int32_t i1542matmul_5_5_5 = matmul_5_5_51_pos[0]; i1542matmul_5_5_5 < matmul_5_5_51_pos[1]; i1542matmul_5_5_5++) {
    int32_t i1542 = matmul_5_5_51_crd[i1542matmul_5_5_5];
    for (int32_t i1546 = 0; i1546 < A14792_dimension; i1546++) {
      int32_t i1546A1538 = i1542 * A15382_dimension + i1546;
      double ti1543A1538_val = 0.0;
      for (int32_t i1543matmul_5_5_5 = matmul_5_5_52_pos[i1542matmul_5_5_5]; i1543matmul_5_5_5 < matmul_5_5_52_pos[(i1542matmul_5_5_5 + 1)]; i1543matmul_5_5_5++) {
        int32_t i1543 = matmul_5_5_52_crd[i1543matmul_5_5_5];
        for (int32_t i1544matmul_5_5_5 = matmul_5_5_53_pos[i1543matmul_5_5_5]; i1544matmul_5_5_5 < matmul_5_5_53_pos[(i1543matmul_5_5_5 + 1)]; i1544matmul_5_5_5++) {
          int32_t i1544 = matmul_5_5_53_crd[i1544matmul_5_5_5];
          for (int32_t i1545 = 0; i1545 < A14791_dimension; i1545++) {
            int32_t i1545A1475 = i1544 * A14752_dimension + i1545;
            int32_t i1545A1416 = i1543 * A14162_dimension + i1545;
            int32_t i1546A1479 = i1545 * A14792_dimension + i1546;
            ti1543A1538_val += ((matmul_5_5_5_vals[i1544matmul_5_5_5] * A1475_vals[i1545A1475]) * A1416_vals[i1545A1416]) * A1479_vals[i1546A1479];
          }
        }
      }
      A1538_vals[i1546A1538] = ti1543A1538_val;
    }
  }
  return 0;
}
#include "/home/min/a/kadhitha/workspace/my_taco/taco/test/kernels/mttkrp_gemm/taco_default.h"
int _shim_assemble(void** parameterPack) {
  return assemble((taco_tensor_t*)(parameterPack[0]), (taco_tensor_t*)(parameterPack[1]), (taco_tensor_t*)(parameterPack[2]), (taco_tensor_t*)(parameterPack[3]), (taco_tensor_t*)(parameterPack[4]));
}
int _shim_compute(void** parameterPack) {
  return compute((taco_tensor_t*)(parameterPack[0]), (taco_tensor_t*)(parameterPack[1]), (taco_tensor_t*)(parameterPack[2]), (taco_tensor_t*)(parameterPack[3]), (taco_tensor_t*)(parameterPack[4]));
}
