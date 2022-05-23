#include "taco/cuda.h"
#include "taco/index_notation/index_notation.h"
#include "taco/ir_tags.h"
#include "taco/tensor.h"
#include "test.h"
#include "util.h"
#include <climits>
#include <experimental/filesystem>
#include "gtest/gtest.h"
#include <cstdint>


#define NUM_THREADS_TO_USE 1

TEST(scheduling_eval, sddmmFusedWithSyntheticData) {
  if (should_use_CUDA_codegen()) {
    return;
  }

  taco_set_num_threads(NUM_THREADS_TO_USE);

  std::default_random_engine gen(0);
  std::uniform_real_distribution<double> unif(0.0, 1.0);

  Format csr({dense, sparse});
  Format  rm({dense, dense});
  int ldim = 4;
  int kdim = 8;

  // uncomment this for reading the csr matrix saved in mtx file
  std::cout << "reading B mat mtx\n";

  int NUM_I = 1021/10;
  int NUM_J = 1039/10;
  float SPARSITY = .3;
  Tensor<double> B("B", {NUM_I, NUM_J}, csr);
  srand(75883);
  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        B.insert({i, j}, (double) ((int) (rand_float*3/SPARSITY)));
      }
    }
  }
  B.pack();


  std::cout << "B dim0: " << B.getDimension(0) << ", dim1: " << B.getDimension(1) << std::endl;
  std::cout << "adding c mat\n";
  Tensor<double> C({B.getDimension(0), kdim}, rm);
  for (int i = 0; i < C.getDimension(0); ++i) {
    for (int j = 0; j < C.getDimension(1); ++j) {
      C.insert({i,j}, unif(gen));
    }
  }
  std::cout << "packing C mat\n";
  C.pack();

  Tensor<double> D({B.getDimension(1), kdim}, rm);
  for (int i = 0; i < D.getDimension(0); ++i) {
    for (int j = 0; j < D.getDimension(1); ++j) {
      D.insert({i,j}, unif(gen));
    }
  }
  std::cout << "packing D mat\n";
  D.pack();

  Tensor<double> F({B.getDimension(1), ldim}, rm);
  for (int i = 0; i < F.getDimension(0); ++i) {
    for (int j = 0; j < F.getDimension(1); ++j) {
      F.insert({i,j}, unif(gen));
    }
  }
  std::cout << "packing F mat\n";
  F.pack();

  Tensor<double> A({B.getDimension(0), ldim}, rm);
  Tensor<double> ref({B.getDimension(0), ldim}, rm);
  IndexVar i, j, k, l;
  A(i,l)=B(i,j)*C(i,k)*D(j,k)*F(j,l);

  // IndexStmt stmt = A.getAssignment().concretize();
  IndexStmt stmt = makeReductionNotation(A.getAssignment());
  stmt = makeConcreteNotation(stmt);
  printToFile("fusedMMConcrete", stmt);
  
  stmt = reorderLoopsTopologically(stmt);
  printToFile("fusedMMOrdered", stmt);
  
  stmt = loopFusionOverFission(stmt, A.getAssignment(), "b", 1);
  printToFile("fusedMMFused", stmt);

  stmt = insertTemporaries(stmt);
  printToFile("fusedMMWithTemps", stmt);
  stmt = parallelizeOuterLoop(stmt); 
  printToFile("fusedMMFusedPar", stmt);

  A.compile(stmt);
  // We can now call the functions taco generated to assemble the indices of the
  // output matrix and then actually compute the MTTKRP.
  A.assemble();


  ref(i,l)=B(i,j)*C(i,k)*D(j,k)*F(j,l);
  IndexStmt refStmt = makeReductionNotation(ref.getAssignment());
  refStmt = makeConcreteNotation(refStmt);
  refStmt = insertTemporaries(refStmt);
  refStmt = parallelizeOuterLoop(refStmt);
  ref.compile(refStmt);
  ref.assemble();

  Tensor<double> ref1({B.getDimension(0), B.getDimension(1)}, csr);
  Tensor<double> ref2({B.getDimension(0), ldim}, rm);
  ref1(i,j)=B(i,j)*C(i,k)*D(j,k);
  ref2(i,l)=ref1(i,j)*F(j,l);

  IndexStmt ref1Stmt = makeReductionNotation(ref1.getAssignment());
  ref1Stmt = makeConcreteNotation(ref1Stmt);
  ref1Stmt = insertTemporaries(ref1Stmt);
  ref1Stmt = parallelizeOuterLoop(ref1Stmt);
  ref1.compile(ref1Stmt);
  ref1.assemble();

  IndexStmt ref2Stmt = makeReductionNotation(ref2.getAssignment());
  ref2Stmt = makeConcreteNotation(ref2Stmt);
  ref2Stmt = insertTemporaries(ref2Stmt);
  ref2Stmt = parallelizeOuterLoop(ref2Stmt);
  ref2.compile(ref2Stmt);
  ref2.assemble();

  std::cout << "compute start\n";
  taco::util::TimeResults timevalue;
  bool time                = true;
  TOOL_BENCHMARK_TIMER(ref.compute(), "\n\nReference Kernel: ", timevalue);
  TOOL_BENCHMARK_TIMER(A.compute(), "\n\nFused Kernel: ", timevalue);

  // check results
  for (int q = 0; q < A.getDimension(0); ++q) {
    for (int w = 0; w < A.getDimension(1); ++w) {
      if ( abs(A(q,w) - ref(q,w))/abs(ref(q,w)) > ERROR_MARGIN) {
        std::cout << "error: results don't match A("<< q << "," << w << "): " 
          << A(q,w) << ", ref: " << ref(q,w) << std::endl;
        ASSERT_TRUE(false);
      }
    }
  }
  // ASSERT_TENSOR_EQ(A, ref);
  TOOL_BENCHMARK_TIMER(ref1.compute(), "\n\nSDDMM Kernel: ", timevalue);
  TOOL_BENCHMARK_TIMER(ref2.compute(), "\n\nSpMM Kernel: ", timevalue);

  for (int q = 0; q < ref2.getDimension(0); ++q) {
    for (int w = 0; w < ref2.getDimension(1); ++w) {
      if ( abs(ref2(q,w) - ref(q,w))/abs(ref(q,w)) > ERROR_MARGIN) {
        std::cout << "error: results don't match A("<< q << "," << w << "): " 
          << ref2(q,w) << ", ref: " << ref(q,w) << std::endl;
        ASSERT_TRUE(false);
      }
    }
  }

}


IndexStmt scheduleSDDMMCPU_forfuse(IndexStmt stmt, Tensor<double> B, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  IndexVar i, j, k, l, m;
  IndexVar i0("i0"), i1("i1"), kpos("kpos"), kpos0("kpos0"), kpos1("kpos1");
  return stmt.split(i, i0, i1, CHUNK_SIZE)
          .pos(k, kpos, B(i,k))
          .split(kpos, kpos0, kpos1, UNROLL_FACTOR)
          .reorder({i0, i1, kpos0, j, kpos1})
          .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
          .parallelize(kpos1, ParallelUnit::CPUVector, OutputRaceStrategy::ParallelReduction);
}


TEST(scheduling_eval, sddmmFused) {
  if (should_use_CUDA_codegen()) {
    return;
  }

  taco_set_num_threads(NUM_THREADS_TO_USE);

  std::default_random_engine gen(0);
  std::uniform_real_distribution<double> unif(0.0, 1.0);

  Format csr({dense, sparse});
  Format rm({dense, dense});
  int ldim = 128;
  int kdim = 128;

  ofstream statfile;
  std::string stat_file = "";
  if (getenv("STAT_FILE")) {
    stat_file = getenv("STAT_FILE");
    std::cout << "stat file: " << stat_file << " is defined\n";
  } else {
    return;
  }
  statfile.open(stat_file, std::ios::app);

  std::string matfile = "";
  if (getenv("TENSOR_FILE")) {
    matfile = getenv("TENSOR_FILE");
    std::cout << "tensor_file: " << matfile << ";\n";
  } else {
    return;
  }

  std::cout << "reading B mat mtx\n";
  Tensor<double> B = read(matfile, csr);
  B.pack();
  std::cout << "B dim0: " << B.getDimension(0) << ", dim1: " << B.getDimension(1) << std::endl;
  
  std::cout << "adding C mat\n";
  Tensor<double> C({B.getDimension(0), kdim}, rm);
  for (int i = 0; i < C.getDimension(0); ++i) {
    for (int j = 0; j < C.getDimension(1); ++j) {
      C.insert({i,j}, unif(gen));
    }
  }
  std::cout << "packing C mat\n";
  C.pack();

  std::cout << "adding D mat\n";
  Tensor<double> D({B.getDimension(1), kdim}, rm);
  for (int i = 0; i < D.getDimension(0); ++i) {
    for (int j = 0; j < D.getDimension(1); ++j) {
      D.insert({i,j}, unif(gen));
    }
  }
  std::cout << "packing D mat\n";
  D.pack();

  Tensor<double> F({B.getDimension(1), ldim}, rm);
  for (int i = 0; i < F.getDimension(0); ++i) {
    for (int j = 0; j < F.getDimension(1); ++j) {
      F.insert({i,j}, unif(gen));
    }
  }
  std::cout << "packing F mat\n";
  F.pack();

  Tensor<double> A({B.getDimension(0), ldim}, rm);
  Tensor<double> ref({B.getDimension(0), ldim}, rm);
  IndexVar i, j, k, l, m;
  IndexVar i0("i0"), i1("i1"), jpos("jpos"), jpos0("jpos0"), jpos1("jpos1"), k0("k0"), k1("k1");
  A(i,l)=B(i,j)*C(i,k)*D(j,k)*F(j,l);

  // IndexStmt stmt = A.getAssignment().concretize();
  IndexStmt stmt = makeReductionNotation(A.getAssignment());
  stmt = makeConcreteNotation(stmt);
  stmt = reorderLoopsTopologically(stmt);
  stmt = loopFusionOverFission(stmt, A.getAssignment(), "b", 1);
  stmt = stmt
    .split(i, i0, i1, 16);
  stmt = insertTemporaries(stmt);
  stmt = parallelizeOuterLoop(stmt); 

  A.compile(stmt);
  A.assemble();


  ref(i,l)=B(i,j)*C(i,k)*D(j,k)*F(j,l);
  IndexStmt refStmt = makeReductionNotation(ref.getAssignment());
  refStmt = makeConcreteNotation(refStmt);
  refStmt = insertTemporaries(refStmt);
  refStmt = refStmt
    .split(i, i0, i1, 32)
    .reorder({i0, i1, k, j, l})
    .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
  stmt = insertTemporaries(stmt);
  ref.compile(refStmt);
  ref.assemble();

  Tensor<double> ref1({B.getDimension(0), B.getDimension(1)}, csr);
  Tensor<double> ref2({B.getDimension(0), ldim}, rm);
  ref1(i,j)=B(i,j)*C(i,k)*D(j,k);
  ref2(i,l)=ref1(i,j)*F(j,l);

  // IndexStmt ref1Stmt = ref1.getAssignment().concretize(); // anyway Ryan's kernel is used here
  IndexStmt ref1Stmt = makeReductionNotation(ref1.getAssignment());
  ref1Stmt = makeConcreteNotation(ref1Stmt);
  ref1Stmt = ref1Stmt.split(i, i0, i1, 16); // assemble gives compilation error with more directives
          // .split(k, k0, k1, 8);
          // .reorder({i0, i1, jpos0, k, jpos1});
          // .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
          // .parallelize(jpos1, ParallelUnit::CPUVector, OutputRaceStrategy::ParallelReduction);
  // ref1Stmt.split(i, );
  // stmt = scheduleSDDMMCPU_forfuse(ref1Stmt, B);
  // IndexStmt ref1Stmt = makeReductionNotation(ref1.getAssignment());
  // ref1Stmt = makeConcreteNotation(ref1Stmt);
  ref1Stmt = insertTemporaries(ref1Stmt);
  // ref1Stmt = parallelizeOuterLoop(ref1Stmt);
  ref1.compile(ref1Stmt);
  ref1.assemble();

  IndexStmt ref2Stmt = makeReductionNotation(ref2.getAssignment()); // Ryan's SpMM kernel is used here
  ref2Stmt = makeConcreteNotation(ref2Stmt);
  //ref2(i,l)=ref1(i,j)*F(j,l);
  ref2Stmt = ref2Stmt
    .split(i, i0, i1, 32)
    .pos(j, jpos, ref1(i,j))
    .split(jpos, jpos0, jpos1, 4)
    .reorder({i0, i1, jpos0, l, jpos1})
    .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
    .parallelize(l, ParallelUnit::CPUVector, OutputRaceStrategy::IgnoreRaces);
  ref2Stmt = insertTemporaries(ref2Stmt);
  ref2.compile(ref2Stmt);
  ref2.assemble();

  std::cout << "compute start\n";
  taco::util::TimeResults timevalue;
  vector<double> timeValues(4);
  
  // fused kernel
  // std::string sofile_fused = "/home/min/a/kadhitha/workspace/my_taco/taco/test/kernels/sddmm_spmm/fused_kernel.so";
  A.compute(timevalue);
  timeValues[0] = timevalue.mean;


  // sddmm and then spmm execution
  cout << "\nseparate execution\n";
  
  ref1.compute(timevalue);
  timeValues[1] = timevalue.mean;
  
  ref2.compute(timevalue);
  timeValues[2] = timevalue.mean;

  // reference execution
  cout << "\nreference execution \n";

  ref.compute(timevalue);
  timeValues[3] = timevalue.mean;

  string dataset = matfile.substr(matfile.find_last_of("/\\") + 1);
  if (statfile.is_open()) {
    statfile
      << dataset.substr(0, dataset.find_first_of(".")) << ", "
      << timeValues[0] << ", "
      << timeValues[1] + timeValues[2] << ", "
      << timeValues[3]
      << endl
      ;
  }

  double* A_vals = (double*) (A.getTacoTensorT()->vals);
  double* ref_vals = (double*) (ref.getTacoTensorT()->vals);
  double* ref2_vals = (double*) (ref2.getTacoTensorT()->vals);

  // int* A2_pos = (double*) (ref.getTacoTensorT()->vals);

  // for (size_t q=0; q < B.getStorage().getValues().getSize(); q++) {
  //   if ( abs(A_vals[q] - ref_vals[q])/abs(ref_vals[q]) > ERROR_MARGIN) {
  //     std::cout << "error: results don't match i: " << q << ", avals: " << A_vals[q] << " "
  //       << "refvals: " << ref_vals[q] << std::endl;
  //     ASSERT_TRUE(false);
  //   }
  // }

  for (size_t q=0; q < A.getDimension(0)* A.getDimension(1); q++) {
    if ( abs(A_vals[q] - ref_vals[q])/abs(ref_vals[q]) > ERROR_MARGIN) {
      std::cout << "error: results don't match i: " << q << ", avals: " << A_vals[q] << " "
        << "refvals: " << ref_vals[q] << std::endl;
      ASSERT_TRUE(false);
    }
  }
  for (size_t q=0; q < A.getDimension(0)* A.getDimension(1); q++) {
    if ( abs(A_vals[q] - ref2_vals[q])/abs(ref2_vals[q]) > ERROR_MARGIN) {
      std::cout << "error: results don't match i: " << q << ", avals: " << A_vals[q] << " "
        << "refvals: " << ref2_vals[q] << std::endl;
      ASSERT_TRUE(false);
    }
  }
  // // for (int q= 0; q< A_vals
  // for (int q = 0; q < A.getDimension(0); ++q) {
  //   for (int w = 0; w < A.getDimension(1); ++w) {
  //     if ( abs(A(q,w) - ref(q,w))/abs(ref(q,w)) > ERROR_MARGIN) {
  //       std::cout << "error: results don't match A("<< q << "," << w << "): " 
  //         << A(q,w) << ", ref: " << ref(q,w) << std::endl;
  //       ASSERT_TRUE(false);
  //     }
  //   }
  // }
  // ASSERT_TENSOR_EQ(A, ref);


  if (statfile.is_open()) {
    statfile.close();
  }

}


TEST(scheduling_eval, hadamardFused) {
  if (should_use_CUDA_codegen()) {
    return;
  }

  taco_set_num_threads(NUM_THREADS_TO_USE);

  std::default_random_engine gen(0);
  std::uniform_real_distribution<double> unif(0.0, 1.0);

  Format csr({dense, sparse});
  Format rm({dense, dense});
  int kdim = 128;
  int ldim = 128;

  ofstream statfile;
  std::string stat_file = "";
  if (getenv("STAT_FILE")) {
    stat_file = getenv("STAT_FILE");
    std::cout << "stat file: " << stat_file << " is defined\n";
  } else {
    return;
  }
  statfile.open(stat_file, std::ios::app);

  std::string matfile = "";
  if (getenv("TENSOR_FILE")) {
    matfile = getenv("TENSOR_FILE");
    std::cout << "tensor_file: " << matfile << ";\n";
  } else {
    return;
  }

  std::cout << "reading B mat mtx\n";
  Tensor<double> B = read(matfile, csr, true);
  B.setName("B");
  B.pack();
  std::cout << "B dim0: " << B.getDimension(0) << ", dim1: " << B.getDimension(1) << std::endl;
  
  std::cout << "adding C mat\n";
  Tensor<double> C({B.getDimension(1), kdim}, rm);
  for (int i = 0; i < C.getDimension(0); ++i) {
    for (int j = 0; j < C.getDimension(1); ++j) {
      C.insert({i,j}, unif(gen));
    }
  }
  std::cout << "packing C mat\n";
  C.pack();

  std::cout << "adding D mat\n";
  Tensor<double> D({B.getDimension(1), kdim}, rm);
  for (int i = 0; i < D.getDimension(0); ++i) {
    for (int j = 0; j < D.getDimension(1); ++j) {
      D.insert({i,j}, unif(gen));
    }
  }
  std::cout << "packing D mat\n";
  D.pack();

  std::cout << "adding F mat\n";
  Tensor<double> F({kdim, ldim}, rm);
  for (int i = 0; i < F.getDimension(0); ++i) {
    for (int j = 0; j < F.getDimension(1); ++j) {
      F.insert({i,j}, unif(gen));
    }
  }
  std::cout << "packing F mat\n";
  F.pack();

  Tensor<double> A({B.getDimension(0), ldim}, rm);
  Tensor<double> ref({B.getDimension(0), ldim}, rm);
  IndexVar i, j, k, l, m;
  IndexVar i0("i0"), i1("i1"), l0("l0"), l1("l1"), jpos("jpos"), jpos0("jpos0"), jpos1("jpos1"), k0("k0"), k1("k1");
  A(i,l)=B(i,j)*C(j,k)*D(j,k)*F(k,l);

  // IndexStmt stmt = A.getAssignment().concretize();
  IndexStmt stmt = makeReductionNotation(A.getAssignment());
  stmt = makeConcreteNotation(stmt);
  stmt = reorderLoopsTopologically(stmt);
  stmt = stmt.reorder({i, j, k, l});
  stmt = loopFusionOverFission(stmt, A.getAssignment(), "b", 1);
  stmt = stmt
    .split(i, i0, i1, 16);
  stmt = insertTemporaries(stmt);
  stmt = parallelizeOuterLoop(stmt); 
  printToFile("fusedMMFusedPar", stmt);

  A.compile(stmt);
  A.assemble();


  ref(i,l)=B(i,j)*C(j,k)*D(j,k)*F(k,l);
  IndexStmt refStmt = makeReductionNotation(ref.getAssignment());
  refStmt = makeConcreteNotation(refStmt);
  refStmt = refStmt
    .split(i, i0, i1, 16)
    .reorder({i0, i1, j, k, l});
  refStmt = insertTemporaries(refStmt);
  refStmt = parallelizeOuterLoop(refStmt);
  ref.compile(refStmt);
  ref.assemble();

  Tensor<double> ref1({B.getDimension(0), kdim}, rm);
  Tensor<double> ref2({B.getDimension(0), ldim}, rm);
  ref1(i,k)=B(i,j)*C(j,k)*D(j,k);
  ref2(i,l)=ref1(i,k)*F(k,l);

  // IndexStmt ref1Stmt = ref1.getAssignment().concretize();
  
  // ref1Stmt = ref1Stmt.split(i, i0, i1, 16);
  //         // .pos(j, jpos, B(i,j));
  //         // .split(k, k0, k1, 8);
  //         // .reorder({i0, i1, jpos0, k, jpos1});
  //         // .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
  //         // .parallelize(jpos1, ParallelUnit::CPUVector, OutputRaceStrategy::ParallelReduction);
  // // ref1Stmt.split(i, );
  // // stmt = scheduleSDDMMCPU_forfuse(ref1Stmt, B);
  IndexStmt ref1Stmt = makeReductionNotation(ref1.getAssignment());
  ref1Stmt = makeConcreteNotation(ref1Stmt);
  ref1Stmt = ref1Stmt
    .split(i, i0, i1, 16)
    .reorder({i0, i1, j, k});
    // .pos(j, jpos, B(i,j))
    // .split(jpos, jpos0, jpos1, 32)
    // .split(k, k0, k1, 32)
    // .reorder({i0, i1, jpos0, k0, jpos1, k1});
  ref1Stmt = insertTemporaries(ref1Stmt);
  ref1Stmt = parallelizeOuterLoop(ref1Stmt);
  ref1.compile(ref1Stmt);
  ref1.assemble();

  IndexStmt ref2Stmt = makeReductionNotation(ref2.getAssignment());
  ref2Stmt = makeConcreteNotation(ref2Stmt);
  ref2Stmt = ref2Stmt
    .split(i, i0, i1, 32)
    .split(k, k0, k1, 32)
    .split(l, l0, l1, 32)
    .reorder({i0, k0, l0, i1, k1, l1});
  ref2Stmt = insertTemporaries(ref2Stmt);
  ref2Stmt = parallelizeOuterLoop(ref2Stmt);
  ref2.compile(ref2Stmt);
  ref2.assemble();

  std::cout << "compute start\n";
  taco::util::TimeResults timevalue;
  vector<double> timeValues(4);
  
  // fused kernel
  A.compute(timevalue);
  timeValues[0] = timevalue.mean;
  
  // hadamard produce kernel execution
  ref1.compute(timevalue);
  timeValues[1] = timevalue.mean;
  
  // gemm kernel
  ref2.compute(timevalue);
  timeValues[2] = timevalue.mean;

  // reference kernel
  ref.compute(timevalue);
  timeValues[3] = timevalue.mean;

  string dataset = matfile.substr(matfile.find_last_of("/\\") + 1);
  if (statfile.is_open()) {
    statfile
      << dataset.substr(0, dataset.find_first_of(".")) << ", "
      << timeValues[0] << ", "
      << timeValues[1] + timeValues[2] << ", "
      << timeValues[3]
      << endl
      ;
  }

  double* A_vals = (double*) (A.getTacoTensorT()->vals);
  double* ref_vals = (double*) (ref.getTacoTensorT()->vals);
  double* ref2_vals = (double*) (ref2.getTacoTensorT()->vals);

  // // int* A2_pos = (double*) (ref.getTacoTensorT()->vals);

  for (int q=0; q < A.getDimension(0)* A.getDimension(1); q++) {
    if ( abs(A_vals[q] - ref_vals[q])/abs(ref_vals[q]) > ERROR_MARGIN) {
      std::cout << "error: results don't match i: " << q << ", avals: " << A_vals[q] << " "
        << "refvals: " << ref_vals[q] << std::endl;
      ASSERT_TRUE(false);
    }
  }

  for (int q=0; q < A.getDimension(0)* A.getDimension(1); q++) {
    if ( abs(A_vals[q] - ref2_vals[q])/abs(ref2_vals[q]) > ERROR_MARGIN) {
      std::cout << "error: results don't match i: " << q << ", avals: " << A_vals[q] << " "
        << "refvals: " << ref2_vals[q] << std::endl;
      ASSERT_TRUE(false);
    }
  }

  if (statfile.is_open()) {
    statfile.close();
  }

}


TEST(scheduling_eval, mttkrpFusedWithSyntheticData) {
  if (should_use_CUDA_codegen()) {
    return;
  }
  taco_set_num_threads(NUM_THREADS_TO_USE);

  std::default_random_engine gen(0);
  std::uniform_real_distribution<double> unif(0.0, 1.0);
  // Predeclare the storage formats that the inputs and output will be stored as.
  // To define a format, you must specify whether each dimension is dense or 
  // sparse and (optionally) the order in which dimensions should be stored. The 
  // formats declared below correspond to compressed sparse fiber (csf) and 
  // row-major dense (rm).
  Format csf({Sparse,Sparse,Sparse});
  Format rm({Dense,Dense});
  Format sd({Dense,Dense});

  int NUM_I = 102/10;
  int NUM_J = 103/10;
  int NUM_K = 105/10;
  int NUM_L = 122/10;
  int NUM_M = 121/10;
  float SPARSITY = .1;
  Tensor<double> A("A", {NUM_I, NUM_M}, sd);
  Tensor<double> B("B", {NUM_I, NUM_K, NUM_L}, csf);
  Tensor<double> C("C", {NUM_K, NUM_J}, rm);
  Tensor<double> D("D", {NUM_L, NUM_J}, rm);
  Tensor<double> E("E", {NUM_J, NUM_M}, rm);
  Tensor<double> ref({NUM_I, NUM_M}, sd);

  srand(549694);
  for (int i = 0; i < NUM_I; i++) {
    for (int k = 0; k < NUM_K; k++) {
      for (int l = 0; l < NUM_L; l++) {
        float rand_float = (float) rand() / (float) (RAND_MAX);
        if (rand_float < SPARSITY) {
          B.insert({i, k, l}, (double) ((int) (rand_float * 3 / SPARSITY)));
        }
      }
    }
  }
  B.pack();
  // write("/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/synthetic.tns", B);

  // Generate a random dense matrix and store it in row-major (dense) format. 
  // Matrices correspond to order-2 tensors in taco.
  for (int k = 0; k < NUM_K; k++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      C.insert({k, j}, (double) ((int) (rand_float*3)));
    }
  }
  C.pack();

  for (int l = 0; l < NUM_L; l++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      D.insert({l, j}, (double) ((int) (rand_float*3)));
    }
  }
  D.pack();

  for (int i = 0; i < E.getDimension(0); ++i) {
    for (int j = 0; j < E.getDimension(1); ++j) {
      E.insert({i,j}, unif(gen));
    }
  }
  E.pack();

  // Define the MTTKRP computation using index notation.
  IndexVar i, k, l, j, m;
  A(i,m) = B(i,k,l) * D(l,j) * C(k,j) * E(j, m);


  IndexStmt stmt = makeReductionNotation(A.getAssignment());
  stmt = makeConcreteNotation(stmt);
  printToFile("fusedMTTKRPConcrete", stmt);
  
  stmt = reorderLoopsTopologically(stmt);
  printToFile("fusedMTTKRPOrdered", stmt);
  
  stmt = loopFusionOverFission(stmt, A.getAssignment(), "b", 1);
  printToFile("fusedMTTKRPFused", stmt);

  stmt = insertTemporaries(stmt);
  printToFile("fusedMTTKRPWithTemps", stmt);
  stmt = parallelizeOuterLoop(stmt); 
  printToFile("fusedMTTKRPFusedPar", stmt);

  
  // At this point, we have defined how entries in the output matrix should be
  // computed from entries in the input tensor and matrices but have not actually
  // performed the computation yet. To do so, we must first tell taco to generate
  // code that can be executed to compute the MTTKRP operation.
  A.compile(stmt);
  // We can now call the functions taco generated to assemble the indices of the
  // output matrix and then actually compute the MTTKRP.
  A.assemble();


  ref(i,m) = B(i,k,l) * D(l,j) * C(k,j) * E(j, m);
  IndexStmt refStmt = makeReductionNotation(ref.getAssignment());
  refStmt = makeConcreteNotation(refStmt);
  refStmt = insertTemporaries(refStmt);
  refStmt = parallelizeOuterLoop(refStmt);
  ref.compile(refStmt);
  ref.assemble();  

  // Tensor<double> ref2({NUM_I, NUM_J}, sd);
  // ref2(i,j) = B(i,k,l) * D(l,j) * C(k,j);
  // IndexStmt ref2Stmt = makeReductionNotation(ref2.getAssignment());
  // ref2Stmt = makeConcreteNotation(ref2Stmt);
  // ref2Stmt = insertTemporaries(ref2Stmt);
  // ref2Stmt = parallelizeOuterLoop(ref2Stmt);
  // ref2.compile(ref2Stmt);
  // ref2.assemble(); 

  // Tensor<double> ref3({NUM_I, NUM_M}, sd);
  // ref3(i,m) = ref2(i,j) * E(j,m);
  // IndexStmt ref3Stmt = makeReductionNotation(ref3.getAssignment());
  // ref3Stmt = makeConcreteNotation(ref3Stmt);
  // ref3Stmt = insertTemporaries(ref3Stmt);
  // ref3Stmt = parallelizeOuterLoop(ref3Stmt);
  // ref3.compile(ref3Stmt);
  // ref3.assemble();  
  
  std::cout << "compute start\n";
  taco::util::TimeResults timevalue;
  bool time                = true;
  // TOOL_BENCHMARK_TIMER(ref.compute(), "\n\nReference ISPC: ", timevalue);
  TOOL_BENCHMARK_TIMER(A.compute(), "\n\nFused MTTKRP+SPMM: ", timevalue);
  TOOL_BENCHMARK_TIMER(ref.compute(), "\n\nReference MTTKRP+SPMM: ", timevalue);
  // TOOL_BENCHMARK_TIMER(ref2.compute(), "\n\nReference MTTKRP: ", timevalue);
  // TOOL_BENCHMARK_TIMER(ref3.compute(), "\n\nReference SPMM: ", timevalue);
  ASSERT_TENSOR_EQ(ref, A);
  // ASSERT_TENSOR_EQ(ref, ref3);

}


TEST(scheduling_eval, mttkrpFused) {
  if (should_use_CUDA_codegen()) {
    return;
  }

  taco_set_num_threads(NUM_THREADS_TO_USE);

  std::default_random_engine gen(0);
  std::uniform_real_distribution<double> unif(0.0, 1.0);
  // Predeclare the storage formats that the inputs and output will be stored as.
  // To define a format, you must specify whether each dimension is dense or 
  // sparse and (optionally) the order in which dimensions should be stored. The 
  // formats declared below correspond to compressed sparse fiber (csf) and 
  // row-major dense (rm).
  Format csf({Dense,Sparse,Sparse});
  Format rm({Dense,Dense});
  Format sd({Dense,Dense}); // sd also keep dense because 1st dimension is mostly dense
  int jDim = 32;
  int mDim = 64;

  ofstream statfile;
  std::string stat_file = "";
  if (getenv("STAT_FILE")) {
    stat_file = getenv("STAT_FILE");
    std::cout << "stat file: " << stat_file << " is defined\n";
  } else {
    return;
  }
  statfile.open(stat_file, std::ios::app);

  std::string matfile = "";
  if (getenv("TENSOR_FILE")) {
    matfile = getenv("TENSOR_FILE");
    std::cout << "tensor_file: " << matfile << ";\n";
  } else {
    return;
  }

  std::cout << "reading B mat mtx\n";
  Tensor<double> B = read(matfile, csf, true);

  Tensor<double> C({B.getDimension(1), jDim}, rm);
  for (int i = 0; i < C.getDimension(0); ++i) {
    for (int j = 0; j < C.getDimension(1); ++j) {
      C.insert({i,j}, unif(gen));
    }
  }
  C.pack();

  Tensor<double> D({B.getDimension(2), jDim}, rm);
  for (int i = 0; i < D.getDimension(0); ++i) {
    for (int j = 0; j < D.getDimension(1); ++j) {
      D.insert({i,j}, unif(gen));
    }
  }
  D.pack();

  Tensor<double> E({jDim, mDim}, rm);
  for (int i = 0; i < E.getDimension(0); ++i) {
    for (int j = 0; j < E.getDimension(1); ++j) {
      E.insert({i,j}, unif(gen));
    }
  }
  E.pack();

  Tensor<double> A({B.getDimension(0), mDim}, sd);
  Tensor<double> ref({B.getDimension(0), mDim}, sd);

  // Define the MTTKRP computation using index notation.
  IndexVar i, k, l, j, m;
  IndexVar i1("i1"), i2("i2"), j1("j1"), j2("j2"), m1("m1"), m2("m2");

  A(i,m) = B(i,k,l) * D(l,j) * C(k,j) * E(j, m);

  IndexStmt stmt = makeReductionNotation(A.getAssignment());
  stmt = makeConcreteNotation(stmt);
  stmt = reorderLoopsTopologically(stmt);
  // stmt = stmt.reorder({i,j,k,l,m});
  stmt = loopFusionOverFission(stmt, A.getAssignment(), "b", 1);
  stmt = stmt.split(i, i1, i2, 32);
  stmt = insertTemporaries(stmt);
  stmt = parallelizeOuterLoop(stmt); 
  printToFile("fusedMTTKRPFusedPar", stmt);
  A.compile(stmt);
  A.assemble();

  ref(i,m) = B(i,k,l) * D(l,j) * C(k,j) * E(j, m);
  IndexStmt refStmt = makeReductionNotation(ref.getAssignment());
  refStmt = makeConcreteNotation(refStmt);
  refStmt = refStmt
    .split(i, i1, i2, 16);
  refStmt = insertTemporaries(refStmt);
  refStmt = parallelizeOuterLoop(refStmt);
  ref.compile(refStmt);
  ref.assemble();

  Tensor<double> ref2({B.getDimension(0), jDim}, sd);
  ref2(i,j) = B(i,k,l) * D(l,j) * C(k,j);
  IndexStmt ref2Stmt = makeReductionNotation(ref2.getAssignment());
  ref2Stmt = makeConcreteNotation(ref2Stmt);
  ref2Stmt = ref2Stmt
    .split(i, i1, i2, 16);
  ref2Stmt = insertTemporaries(ref2Stmt);
  ref2Stmt = parallelizeOuterLoop(ref2Stmt);
  ref2.compile(ref2Stmt);
  ref2.assemble(); 

  Tensor<double> ref2_ryan({B.getDimension(0), jDim}, sd);
  ref2_ryan(i,j) = B(i,k,l) * D(l,j) * C(k,j);

  IndexStmt ref2RyanStmt = makeReductionNotation(ref2_ryan.getAssignment());
  ref2RyanStmt = makeConcreteNotation(ref2RyanStmt);
  
  IndexExpr precomputeExpr = ref2RyanStmt.as<Forall>().getStmt().as<Forall>().getStmt()
                                 .as<Forall>().getStmt().as<Forall>().getStmt()
                                 .as<Assignment>().getRhs().as<Mul>().getA();
  TensorVar w("w", Type(Float64, {Dimension(j)}), taco::dense);
  ref2RyanStmt = ref2RyanStmt.split(i, i1, i2, 16)
          .reorder({i1, i2, k, l, j})
          .precompute(precomputeExpr, j, j, w)
          .parallelize(i1, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
  ref2RyanStmt = insertTemporaries(ref2RyanStmt);
  // ref2RyanStmt = parallelizeOuterLoop(ref2RyanStmt);
  ref2_ryan.compile(ref2RyanStmt);
  ref2_ryan.assemble(); 

  Tensor<double> ref3({B.getDimension(0), mDim}, sd);
  ref3(i,m) = ref2(i,j) * E(j,m);
  IndexStmt ref3Stmt = makeReductionNotation(ref3.getAssignment());
  ref3Stmt = makeConcreteNotation(ref3Stmt);
  ref3Stmt = ref3Stmt
    .split(i, i1, i2, 16)
    .split(j, j1, j2, 16)
    .split(m, m1, m2, 16)
    .reorder({i1, j1, m1, i2, j2, m2})
    .parallelize(i1, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
  ref3Stmt = insertTemporaries(ref3Stmt);
  ref3Stmt = parallelizeOuterLoop(ref3Stmt);
  ref3.compile(ref3Stmt);
  ref3.assemble(); 


  std::cout << "compute start\n";
  taco::util::TimeResults timevalue;
  // default mttkrp, ryan mttkrp, gemm, reference, fused
  vector<double> timeValues(5);

  ref2.compute(timevalue);
  timeValues[0] = timevalue.mean;

  ref2_ryan.compute(timevalue);
  timeValues[1] = timevalue.mean;

  double* ref2_vals = (double*) (ref2.getTacoTensorT()->vals);
  double* ref2_ryan_vals = (double*) (ref2_ryan.getTacoTensorT()->vals);
  for (int q=0; q < B.getDimension(0)* jDim; q++) {
    if ( abs(ref2_vals[q] - ref2_ryan_vals[q])/abs(ref2_ryan_vals[q]) > ERROR_MARGIN) {
      std::cout << "error: results don't match i: " << q << ", avals: " << ref2_vals[q] << " "
        << "refvals: " << ref2_ryan_vals[q] << std::endl;
      ASSERT_TRUE(false);
    }
  }

  ref3.compute(timevalue);
  timeValues[2] = timevalue.mean;

  ref.compute(timevalue);
  timeValues[3] = timevalue.mean;

  double* ref3_vals = (double*) (ref3.getTacoTensorT()->vals);
  double* ref_vals = (double*) (ref.getTacoTensorT()->vals);
  for (int q=0; q < B.getDimension(0)* mDim; q++) {
    if ( abs(ref3_vals[q] - ref_vals[q])/abs(ref_vals[q]) > ERROR_MARGIN) {
      std::cout << "error: results don't match i: " << q << ", avals: " << ref3_vals[q] << " "
        << "refvals: " << ref_vals[q] << std::endl;
      ASSERT_TRUE(false);
    }
  }

  A.compute(timevalue);
  timeValues[4] = timevalue.mean;

  double* A_vals = (double*) (A.getTacoTensorT()->vals);
  for (int q=0; q < B.getDimension(0)* mDim; q++) {
    if ( abs(A_vals[q] - ref_vals[q])/abs(ref_vals[q]) > ERROR_MARGIN) {
      std::cout << "error: results don't match i: " << q << ", avals: " << A_vals[q] << " "
        << "refvals: " << ref_vals[q] << std::endl;
      ASSERT_TRUE(false);
    }
  }

  string dataset = matfile.substr(matfile.find_last_of("/\\") + 1);
  if (statfile.is_open()) {
    statfile 
      << dataset.substr(0, dataset.find_first_of(".")) << ", "
      << timeValues[4] << ", "
      << min(timeValues[0], timeValues[1]) + timeValues[2] << ", "
      << timeValues[3]
      << endl
      ;
  } else { std::cout << " stat file is not open\n"; }

  if (statfile.is_open()) {
    statfile.close();
  }

}


TEST(scheduling_eval, ttmFusedWithSyntheticData) {
  if (should_use_CUDA_codegen()) {
    return;
  }

  taco_set_num_threads(NUM_THREADS_TO_USE);

  std::default_random_engine gen(0);
  std::uniform_real_distribution<double> unif(0.0, 1.0);
  Format csf({Sparse,Sparse,Sparse});
  Format custom({Sparse,Sparse,Dense});
  Format rm({Dense,Dense});

  int NUM_I = 5;
  int NUM_J = 5;
  int NUM_K = 5;
  int NUM_L = 64;
  int NUM_M = 32;
  float SPARSITY = .1;

  Tensor<double> B("B", {NUM_I, NUM_J, NUM_K}, csf);
  srand(549694);
  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      for (int k = 0; k < NUM_K; k++) {
        float rand_float = (float) rand() / (float) (RAND_MAX);
        if (rand_float < SPARSITY) {
          B.insert({i, j, k}, (double) ((int) (rand_float * 3 / SPARSITY)));
        }
      }
    }
  }
  B.pack();
  // write("/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/synthetic.tns", B);

  // Generate a random dense matrix and store it in row-major (dense) format. 
  // Matrices correspond to order-2 tensors in taco.
  Tensor<double> C({B.getDimension(2), NUM_L}, rm);
  for (int i = 0; i < C.getDimension(0); ++i) {
    for (int j = 0; j < C.getDimension(1); ++j) {
      C.insert({i,j}, unif(gen));
    }
  }
  C.pack();

  // Generate another random dense matrix and store it in row-major format.
  Tensor<double> D({NUM_L, NUM_M}, rm);
  for (int i = 0; i < D.getDimension(0); ++i) {
    for (int j = 0; j < D.getDimension(1); ++j) {
      D.insert({i,j}, unif(gen));
    }
  }
  D.pack();

  Tensor<double> A({B.getDimension(0), B.getDimension(1), NUM_M}, custom);
  Tensor<double> ref({B.getDimension(0), B.getDimension(1), NUM_M}, custom);

  // Define the MTTKRP computation using index notation.
  IndexVar i, j, k, l, m;
  A(i,j,m) = B(i,j,k) * C(k,l) * D(l,m);

  IndexStmt stmt = makeReductionNotation(A.getAssignment());
  stmt = makeConcreteNotation(stmt);
  printToFile("fusedTTMTTKRPConcrete", stmt);
  
  stmt = reorderLoopsTopologically(stmt);
  printToFile("fusedTTMOrdered", stmt);
  
  stmt = loopFusionOverFission(stmt, A.getAssignment(), "b", 1);
  printToFile("fusedTTMFused", stmt);

  stmt = insertTemporaries(stmt);
  printToFile("fusedTTMWithTemps", stmt);
  stmt = parallelizeOuterLoop(stmt); 
  printToFile("fusedTTMFinal", stmt);

  
  // At this point, we have defined how entries in the output matrix should be
  // computed from entries in the input tensor and matrices but have not actually
  // performed the computation yet. To do so, we must first tell taco to generate
  // code that can be executed to compute the MTTKRP operation.
  A.compile(stmt);
  // We can now call the functions taco generated to assemble the indices of the
  // output matrix and then actually compute the MTTKRP.
  A.assemble();


  ref(i,j,m) = B(i,j,k) * C(k,l) * D(l,m);
  IndexStmt refStmt = makeReductionNotation(ref.getAssignment());
  refStmt = makeConcreteNotation(refStmt);
  refStmt = insertTemporaries(refStmt);
  refStmt = parallelizeOuterLoop(refStmt);
  printToFile("tacoFusedTTM", refStmt);
  ref.compile(refStmt);
  ref.assemble(); 

  Tensor<double> ref1({B.getDimension(0), B.getDimension(1), NUM_L}, custom);
  ref1(i,j,l) = B(i,j,k) * C(k,l);
  IndexStmt ref1Stmt = makeReductionNotation(ref1.getAssignment());
  ref1Stmt = makeConcreteNotation(ref1Stmt);
  ref1Stmt = insertTemporaries(ref1Stmt);
  ref1Stmt = parallelizeOuterLoop(ref1Stmt);
  ref1.compile(ref1Stmt);
  ref1.assemble();  

  Tensor<double> ref2({B.getDimension(0), B.getDimension(1), NUM_M}, custom);
  ref2(i,j,m) = ref1(i,j,l) * D(l,m);
  IndexStmt ref2Stmt = makeReductionNotation(ref2.getAssignment());
  ref2Stmt = makeConcreteNotation(ref2Stmt);
  ref2Stmt = insertTemporaries(ref2Stmt);
  ref2Stmt = parallelizeOuterLoop(ref2Stmt);
  ref2.compile(ref2Stmt);
  ref2.assemble(); 

  Tensor<double> ref3({B.getDimension(2), NUM_M}, rm);
  ref3(k,m) = C(k,l) * D(l,m);
  IndexStmt ref3Stmt = makeReductionNotation(ref3.getAssignment());
  ref3Stmt = makeConcreteNotation(ref3Stmt);
  ref3Stmt = insertTemporaries(ref3Stmt);
  ref3Stmt = parallelizeOuterLoop(ref3Stmt);
  ref3.compile(ref3Stmt);
  ref3.assemble();  

  Tensor<double> ref4({B.getDimension(0), B.getDimension(1), NUM_M}, custom);
  ref4(i,j,m) = B(i,j,k) * ref3(k,m);
  IndexStmt ref4Stmt = makeReductionNotation(ref4.getAssignment());
  ref4Stmt = makeConcreteNotation(ref4Stmt);
  ref4Stmt = insertTemporaries(ref4Stmt);
  ref4Stmt = parallelizeOuterLoop(ref4Stmt);
  ref4.compile(ref4Stmt);
  ref4.assemble();

  std::cout << "compute start\n";
  taco::util::TimeResults timevalue;
  bool time                = true;
  // TOOL_BENCHMARK_TIMER(ref.compute(), "\n\nReference ISPC: ", timevalue);
  TOOL_BENCHMARK_TIMER(A.compute(), "\n\nFused TTM->TTM: ", timevalue);
  TOOL_BENCHMARK_TIMER(ref.compute(), "\n\nReference TTM->TTM: ", timevalue);
  TOOL_BENCHMARK_TIMER(ref1.compute(), "\n\nTTM1: ", timevalue);
  TOOL_BENCHMARK_TIMER(ref2.compute(), "\n\nTTM1: ", timevalue);
  TOOL_BENCHMARK_TIMER(ref3.compute(), "\n\ndense: ", timevalue);
  TOOL_BENCHMARK_TIMER(ref4.compute(), "\n\nTTM after dense: ", timevalue);
  ASSERT_TENSOR_EQ(ref, A);
  ASSERT_TENSOR_EQ(ref, ref2);
  ASSERT_TENSOR_EQ(ref, ref4);

  for (int q = 0; q < A.getDimension(0); ++q) {
    for (int w = 0; w < A.getDimension(1); ++w) {
      for (int z = 0; z < A.getDimension(2); ++z) {
        // std::cout << "(" << q << "," << w << "," << z << ")" 
        //   << "a: " << A(q,w,z) << ", ref: " << ref(q,w,z) << std::endl;
        if ( abs(A(q,w,z) - ref(q,w,z))/abs(ref(q,w,z)) > ERROR_MARGIN) {
          std::cout << "error: results don't match A: " 
            << A(q,w,z) << ", ref: " << ref(q,w,z) << std::endl;
          ASSERT_TRUE(false);
        }
      }
    }
  }

}


TEST(scheduling_eval, ttmFused) {
  if (should_use_CUDA_codegen()) {
    return;
  }

  taco_set_num_threads(NUM_THREADS_TO_USE);

  std::default_random_engine gen(0);
  std::uniform_real_distribution<double> unif(0.0, 1.0);
  Format csf({Dense,Sparse,Sparse});
  Format custom({Dense,Sparse,Dense});
  Format rm({Dense,Dense});
  int ldim = 32;
  int mdim = 64;

  ofstream statfile;
  std::string stat_file = "";
  if (getenv("STAT_FILE")) {
    stat_file = getenv("STAT_FILE");
    std::cout << "stat file: " << stat_file << " is defined\n";
  } else {
    return;
  }
  statfile.open(stat_file, std::ios::app);

  std::string matfile = "";
  if (getenv("TENSOR_FILE")) {
    matfile = getenv("TENSOR_FILE");
    std::cout << "tensor_file: " << matfile << ";\n";
  } else {
    return;
  }

  int64_t dummy_array_size = 2e6;
  int64_t* dummy_array_to_flush_cache = (int64_t*) malloc(dummy_array_size*sizeof(int64_t));

  Tensor<double> B = read(matfile, csf);
  B.setName("B");
  B.pack();
  // write(matfilesrw[matfilenum], B);

  // Generate a random dense matrix and store it in row-major (dense) format. 
  // Matrices correspond to order-2 tensors in taco.
  Tensor<double> C("C", {B.getDimension(2), ldim}, rm);
  for (int i = 0; i < C.getDimension(0); ++i) {
    for (int j = 0; j < C.getDimension(1); ++j) {
      C.insert({i,j}, unif(gen));
    }
  }
  C.pack();

  // Generate another random dense matrix and store it in row-major format.
  Tensor<double> D("D", {ldim, mdim}, rm);
  for (int i = 0; i < D.getDimension(0); ++i) {
    for (int j = 0; j < D.getDimension(1); ++j) {
      D.insert({i,j}, unif(gen));
    }
  }
  D.pack();

  Tensor<double> A({B.getDimension(0), B.getDimension(1), mdim}, custom);
  Tensor<double> ref({B.getDimension(0), B.getDimension(1), mdim}, custom);
  Tensor<double> refn({B.getDimension(0), B.getDimension(1), mdim}, custom);

  // Define the MTTKRP computation using index notation.
  IndexVar i, j, k, l, m;
  IndexVar i0,i1, j0, j1, k0, k1, l0, l1, m0, m1;
  A(i,j,m) = B(i,j,k) * C(k,l) * D(l,m);


  IndexStmt stmt = makeReductionNotation(A.getAssignment());
  stmt = makeConcreteNotation(stmt);
  stmt = reorderLoopsTopologically(stmt);
  stmt = loopFusionOverFission(stmt, A.getAssignment(), "b", 1);
  stmt = stmt.split(i, i0, i1, 16);
  stmt = insertTemporaries(stmt);
  stmt = parallelizeOuterLoop(stmt); 
  printToFile("fusedTTMFinal", stmt);

  A.compile(stmt);
  A.assemble();

  ref(i,j,m) = B(i,j,k) * C(k,l) * D(l,m); // TTM->TTM TACO
  IndexStmt refStmt = makeReductionNotation(ref.getAssignment());
  refStmt = makeConcreteNotation(refStmt);
  refStmt = refStmt
    .split(i, i0, i1, 16);
  refStmt = insertTemporaries(refStmt);
  refStmt = parallelizeOuterLoop(refStmt);
  printToFile("tacoFusedTTM", refStmt);
  ref.compile(refStmt);
  ref.assemble();

  refn(i,j,m) = B(i,j,k) * C(k,l) * D(l,m); // TTM->TTM TACO
  IndexStmt refnStmt = makeReductionNotation(refn.getAssignment());
  refnStmt = makeConcreteNotation(refnStmt);
  refnStmt = refnStmt
    .split(i, i0, i1, 16)
    .reorder({i0, i1, j, k, l, m});
  refnStmt = insertTemporaries(refnStmt);
  refnStmt = parallelizeOuterLoop(refnStmt);
  printToFile("tacoFusedTTM", refnStmt);
  refn.compile(refnStmt);
  refn.assemble();

  Tensor<double> ref1({B.getDimension(0), B.getDimension(1), ldim}, custom);
  ref1(i,j,l) = B(i,j,k) * C(k,l); // TTM1
  IndexStmt ref1Stmt = makeReductionNotation(ref1.getAssignment());
  ref1Stmt = makeConcreteNotation(ref1Stmt);
  ref1Stmt = ref1Stmt.split(i, i0, i1, 16);
  ref1Stmt = insertTemporaries(ref1Stmt);
  ref1Stmt = parallelizeOuterLoop(ref1Stmt);
  ref1.compile(ref1Stmt);
  ref1.assemble();  

  Tensor<double> ref2({B.getDimension(0), B.getDimension(1), mdim}, custom);
  ref2(i,j,m) = ref1(i,j,l) * D(l,m); // TTM2
  IndexStmt ref2Stmt = makeReductionNotation(ref2.getAssignment());
  ref2Stmt = makeConcreteNotation(ref2Stmt);
  ref2Stmt = ref2Stmt.split(i, i0, i1, 16);
  ref2Stmt = insertTemporaries(ref2Stmt);
  ref2Stmt = parallelizeOuterLoop(ref2Stmt);
  ref2.compile(ref2Stmt);
  ref2.assemble();

  Tensor<double> ref3({B.getDimension(2), mdim}, rm);
  ref3(k,m) = C(k,l) * D(l,m); // GeMM
  IndexStmt ref3Stmt = makeReductionNotation(ref3.getAssignment());
  ref3Stmt = makeConcreteNotation(ref3Stmt);
  ref3Stmt = ref3Stmt
    .split(k, k0, k1, 32)
    .split(l, l0, l1, 32)
    .split(m, m0, m1, 32)
    .reorder({k0, l0, m0, k1, l1, m1});
  ref3Stmt = insertTemporaries(ref3Stmt);
  ref3Stmt = parallelizeOuterLoop(ref3Stmt);
  ref3.compile(ref3Stmt);
  ref3.assemble();  

  Tensor<double> ref4({B.getDimension(0), B.getDimension(1), mdim}, custom);
  ref4(i,j,m) = B(i,j,k) * ref3(k,m); // TTM1
  IndexStmt ref4Stmt = makeReductionNotation(ref4.getAssignment());
  ref4Stmt = makeConcreteNotation(ref4Stmt);
  ref4Stmt = ref4Stmt
    .split(i, i0, i1, 16);
  //   // .split(k, k0, k1, 16)
  //   .split(m, m0, m1, 16)
  //   .reorder({i0, i1, j, m0, k, m1});
  ref4Stmt = insertTemporaries(ref4Stmt);
  ref4Stmt = parallelizeOuterLoop(ref4Stmt);
  ref4.compile(ref4Stmt);
  ref4.assemble();

  return;

  std::cout << "compute start\n";
  taco::util::TimeResults timevalue;
  vector<double> timeValues(7);

  int r = rand();
  for (int64_t i=0; i<dummy_array_size; i++) {
    dummy_array_to_flush_cache[i] = r;
  }

  // fused execution
  A.compute(timevalue);
  timeValues[0] = timevalue.mean;

  r = rand();
  for (int64_t i=0; i<dummy_array_size; i++) {
    dummy_array_to_flush_cache[i] = r;
  }

  // reference impl

  ref.compute(timevalue);
  timeValues[1] = timevalue.mean;

  r = rand();
  for (int64_t i=0; i<dummy_array_size; i++) {
    dummy_array_to_flush_cache[i] = r;
  }

  refn.compute(timevalue);
  timeValues[2] = timevalue.mean;

  // schedule 1 ttm -> ttm

  r = rand();
  for (int64_t i=0; i<dummy_array_size; i++) {
    dummy_array_to_flush_cache[i] = r;
  }

  ref1.compute(timevalue);
  timeValues[3] = timevalue.mean;

  r = rand();
  for (int64_t i=0; i<dummy_array_size; i++) {
    dummy_array_to_flush_cache[i] = r;
  }

  ref2.compute(timevalue);
  timeValues[4] = timevalue.mean;

  r = rand();
  for (int64_t i=0; i<dummy_array_size; i++) {
    dummy_array_to_flush_cache[i] = r;
  }

  // schedule 2 gemm, ttm

  ref3.compute(timevalue); // gemm
  timeValues[5] = timevalue.mean;

  r = rand();
  for (int64_t i=0; i<dummy_array_size; i++) {
    dummy_array_to_flush_cache[i] = r;
  }

  ref4.compute(timevalue);
  timeValues[6] = timevalue.mean;

  r = rand();
  bool istrue = false;
  for (int64_t i=0; i<dummy_array_size; i++) {
    if (dummy_array_to_flush_cache[i] != r) {
      istrue = true;
    }
  }
  std::cout << "istrue: " << istrue << std::endl;

  string dataset = matfile.substr(matfile.find_last_of("/\\") + 1);
  if (statfile.is_open()) {
    statfile 
      << dataset.substr(0, dataset.find_first_of(".")) << ", "
      << timeValues[0] << ", "
      << min(timeValues[1], timeValues[2]) << ", "
      << timeValues[3] + timeValues[4] << ", "
      << timeValues[5] + timeValues[6]
      << endl
      ;
  } else { std::cout << " stat file is not open\n"; }

  // double* A_vals = (double*) (A.getTacoTensorT()->vals);
  // double* ref_vals = (double*) (ref.getTacoTensorT()->vals);
  // double* ref2_vals = (double*) (ref2.getTacoTensorT()->vals);
  // double* ref4_vals = (double*) (ref4.getTacoTensorT()->vals);

  // int* A2_pos = (double*) (ref.getTacoTensorT()->vals);

  // for (size_t q=0; q < B.getStorage().getValues().getSize(); q++) {
  //   if ( abs(A_vals[q] - ref_vals[q])/abs(ref_vals[q]) > ERROR_MARGIN) {
  //     std::cout << "error: results don't match i: " << q << ", avals: " << A_vals[q] << " "
  //       << "refvals: " << ref_vals[q] << std::endl;
  //     ASSERT_TRUE(false);
  //   }
  // }

  // std::cout << "our fused vs taco original fused check\n";
  // for (size_t q=0; q < A.getStorage().getValues().getSize(); q++) {
  //   if ( abs(A_vals[q] - ref_vals[q])/abs(ref_vals[q]) > ERROR_MARGIN) {
  //     std::cout << "error: results don't match i: " << q << ", avals: " << A_vals[q] << " "
  //       << "refvals: " << ref_vals[q] << std::endl;
  //     ASSERT_TRUE(false);
  //   }
  // }
  // std::cout << "taco original fused vs TTM1, TTM2 check\n";
  // for (size_t q=0; q < A.getStorage().getValues().getSize(); q++) {
  //   if ( abs(ref_vals[q] - ref2_vals[q])/abs(ref2_vals[q]) > ERROR_MARGIN) {
  //     std::cout << "error: results don't match i: " << q << ", avals: " << ref_vals[q] << " "
  //       << "refvals: " << ref2_vals[q] << std::endl;
  //     ASSERT_TRUE(false);
  //   }
  // }
  // std::cout << "taco original fused vs GeMM, TTM1 check\n";
  // for (size_t q=0; q < A.getStorage().getValues().getSize(); q++) {
  //   if ( abs(ref_vals[q] - ref4_vals[q])/abs(ref4_vals[q]) > ERROR_MARGIN) {
  //     std::cout << "error: results don't match i: " << q << ", avals: " << ref_vals[q] << " "
  //       << "refvals: " << ref4_vals[q] << std::endl;
  //     ASSERT_TRUE(false);
  //   }
  // }

  if (statfile.is_open()) {
    statfile.close();
  }

}


TEST(scheduling_eval, spmmFusedWithSyntheticData) {
  if (should_use_CUDA_codegen()) {
    return;
  }

  taco_set_num_threads(NUM_THREADS_TO_USE);


  std::default_random_engine gen(0);
  std::uniform_real_distribution<double> unif(0.0, 1.0);

  Format csr({dense, sparse});
  Format  rm({dense, dense});
  int ldim = 32;
  int kdim = 64;

  // uncomment this for reading the csr matrix saved in mtx file
  std::cout << "reading B mat mtx\n";

  int NUM_I = 128;
  int NUM_J = 96;
  int NUM_K = 64;
  float SPARSITY = .3;
  Tensor<double> B("B", {NUM_I, NUM_J}, csr);
  srand(75883);
  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        B.insert({i, j}, (double) ((int) (rand_float*3/SPARSITY)));
      }
    }
  }
  B.pack();

  Tensor<double> C("C", {NUM_J, NUM_K}, csr);
  for (int j = 0; j < NUM_J; j++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        B.insert({j, k}, (double) ((int) (rand_float*3/SPARSITY)));
      }
    }
  }
  C.pack();
  // write("/home/min/a/kadhitha/ispc-examples/data/suitesparse/synthetic/synthetic.mtx", B);

  std::cout << "B dim0: " << B.getDimension(0) << ", dim1: " << B.getDimension(1) << std::endl;
  std::cout << "adding c mat\n";
  Tensor<double> D({C.getDimension(1), ldim}, rm);
  for (int i = 0; i < D.getDimension(0); ++i) {
    for (int j = 0; j < D.getDimension(1); ++j) {
      D.insert({i,j}, unif(gen));
    }
  }
  std::cout << "packing C mat\n";
  D.pack();

  // Tensor<double> E({B.getDimension(1), kdim}, rm);
  // for (int i = 0; i < D.getDimension(0); ++i) {
  //   for (int j = 0; j < D.getDimension(1); ++j) {
  //     D.insert({i,j}, unif(gen));
  //   }
  // }
  // std::cout << "packing D mat\n";
  // D.pack();

  // Tensor<double> F({B.getDimension(1), ldim}, rm);
  // for (int i = 0; i < F.getDimension(0); ++i) {
  //   for (int j = 0; j < F.getDimension(1); ++j) {
  //     F.insert({i,j}, unif(gen));
  //   }
  // }
  // std::cout << "packing F mat\n";
  // F.pack();

  Tensor<double> A({B.getDimension(0), ldim}, rm);
  Tensor<double> ref({B.getDimension(0), ldim}, rm);
  IndexVar i, j, k, l;
  A(i,l)=B(i,j)*C(j,k)*D(k,l);

  // IndexStmt stmt = A.getAssignment().concretize();
  IndexStmt stmt = makeReductionNotation(A.getAssignment());
  stmt = makeConcreteNotation(stmt);
  printToFile("fusedMMConcrete", stmt);
  
  stmt = reorderLoopsTopologically(stmt);
  printToFile("fusedMMOrdered", stmt);
  
  stmt = loopFusionOverFission(stmt, A.getAssignment(), "b", 1);
  printToFile("fusedMMFused", stmt);

  stmt = insertTemporaries(stmt);
  printToFile("fusedMMWithTemps", stmt);
  stmt = parallelizeOuterLoop(stmt); 
  printToFile("fusedMMFusedPar", stmt);

  A.compile(stmt);
  // We can now call the functions taco generated to assemble the indices of the
  // output matrix and then actually compute the MTTKRP.
  A.assemble();


  ref(i,l)=B(i,j)*C(j,k)*D(k,l);
  IndexStmt refStmt = makeReductionNotation(ref.getAssignment());
  refStmt = makeConcreteNotation(refStmt);
  refStmt = insertTemporaries(refStmt);
  refStmt = parallelizeOuterLoop(refStmt);
  ref.compile(refStmt);
  ref.assemble();

  // Tensor<double> ref1({B.getDimension(0), B.getDimension(1)}, csr);
  // Tensor<double> ref2({B.getDimension(0), ldim}, rm);
  // ref1(i,j)=B(i,j)*C(i,k)*D(j,k);
  // ref2(i,l)=ref1(i,j)*F(j,l);

  // IndexStmt ref1Stmt = makeReductionNotation(ref1.getAssignment());
  // ref1Stmt = makeConcreteNotation(ref1Stmt);
  // ref1Stmt = insertTemporaries(ref1Stmt);
  // ref1Stmt = parallelizeOuterLoop(ref1Stmt);
  // ref1.compile(ref1Stmt);
  // ref1.assemble();

  // IndexStmt ref2Stmt = makeReductionNotation(ref2.getAssignment());
  // ref2Stmt = makeConcreteNotation(ref2Stmt);
  // ref2Stmt = insertTemporaries(ref2Stmt);
  // ref2Stmt = parallelizeOuterLoop(ref2Stmt);
  // ref2.compile(ref2Stmt);
  // ref2.assemble();

  std::cout << "compute start\n";
  taco::util::TimeResults timevalue;
  bool time                = true;
  TOOL_BENCHMARK_TIMER(ref.compute(), "\n\nReference Kernel: ", timevalue);
  TOOL_BENCHMARK_TIMER(A.compute(), "\n\nFused Kernel: ", timevalue);

  // check results
  for (int q = 0; q < A.getDimension(0); ++q) {
    for (int w = 0; w < A.getDimension(1); ++w) {
      if ( abs(A(q,w) - ref(q,w))/abs(ref(q,w)) > ERROR_MARGIN) {
        std::cout << "error: results don't match A("<< q << "," << w << "): " 
          << A(q,w) << ", ref: " << ref(q,w) << std::endl;
        ASSERT_TRUE(false);
      }
    }
  }
  // // ASSERT_TENSOR_EQ(A, ref);
  // TOOL_BENCHMARK_TIMER(ref1.compute(), "\n\nSDDMM Kernel: ", timevalue);
  // TOOL_BENCHMARK_TIMER(ref2.compute(), "\n\nSpMM Kernel: ", timevalue);

  // for (int q = 0; q < ref2.getDimension(0); ++q) {
  //   for (int w = 0; w < ref2.getDimension(1); ++w) {
  //     if ( abs(ref2(q,w) - ref(q,w))/abs(ref(q,w)) > ERROR_MARGIN) {
  //       std::cout << "error: results don't match A("<< q << "," << w << "): " 
  //         << ref2(q,w) << ", ref: " << ref(q,w) << std::endl;
  //       ASSERT_TRUE(false);
  //     }
  //   }
  // }

}


TEST(scheduling_eval, spmmFused) {
  if (should_use_CUDA_codegen()) {
    return;
  }

  taco_set_num_threads(NUM_THREADS_TO_USE);

  std::default_random_engine gen(0);
  std::uniform_real_distribution<double> unif(0.0, 1.0);

  Format csr({dense, sparse});
  Format rm({dense, dense});
  int kdim = 128;
  int ldim = 64;

  ofstream statfile;
  std::string stat_file = "";
  if (getenv("STAT_FILE")) {
    stat_file = getenv("STAT_FILE");
    std::cout << "stat file: " << stat_file << " is defined\n";
  } else {
    return;
  }
  statfile.open(stat_file, std::ios::app);

  std::string matfile = "";
  if (getenv("TENSOR_FILE")) {
    matfile = getenv("TENSOR_FILE");
    std::cout << "tensor_file: " << matfile << ";\n";
  } else {
    return;
  }

  std::cout << "reading B mat mtx\n";
  Tensor<double> B = read(matfile, csr);
  std::cout << "packing B mat\n";
  B.pack();

  std::cout << "adding C mat\n";
  Tensor<double> C("C", {B.getDimension(1), kdim}, rm);
  for (int i = 0; i < C.getDimension(0); ++i) {
    for (int j = 0; j < C.getDimension(1); ++j) {
      C.insert({i,j}, unif(gen));
    }
  }
  std::cout << "packing C mat\n";
  C.pack();

  std::cout << "adding D mat\n";
  Tensor<double> D({C.getDimension(1), ldim}, rm);
  for (int i = 0; i < D.getDimension(0); ++i) {
    for (int j = 0; j < D.getDimension(1); ++j) {
      D.insert({i,j}, unif(gen));
    }
  }
  std::cout << "packing D mat\n";
  D.pack();


  Tensor<double> A({B.getDimension(0), ldim}, rm);
  Tensor<double> ref({B.getDimension(0), ldim}, rm);
  Tensor<double> refn({B.getDimension(0), ldim}, rm);
  IndexVar i, j, k, l, jpos, jpos0, jpos1;
  IndexVar i0, i1, j0, j1, k0, k1, l0, l1;

  A(i,l)=B(i,j)*C(j,k)*D(k,l);

  // IndexStmt stmt = A.getAssignment().concretize();
  IndexStmt stmt = makeReductionNotation(A.getAssignment());
  stmt = makeConcreteNotation(stmt);
  stmt = reorderLoopsTopologically(stmt);
  stmt = loopFusionOverFission(stmt, A.getAssignment(), "b", 1);
  stmt = stmt.split(i, i0, i1, 16);
  stmt = insertTemporaries(stmt);
  stmt = parallelizeOuterLoop(stmt);

  A.compile(stmt);
  A.assemble();


  ref(i,l)=B(i,j)*C(j,k)*D(k,l);
  refn(i,l)=B(i,j)*C(j,k)*D(k,l);
  // IndexStmt refStmt = ref.getAssignment().concretize();

  // ref1Stmt = ref1Stmt.split(i, i0, i1, 16);
          // .pos(j, jpos, B(i,j));
          // .split(k, k0, k1, 8);
          // .reorder({i0, i1, jpos0, k, jpos1});
          // .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
          // .parallelize(jpos1, ParallelUnit::CPUVector, OutputRaceStrategy::ParallelReduction);
  IndexStmt refStmt = makeReductionNotation(ref.getAssignment());
  refStmt = makeConcreteNotation(refStmt);
  refStmt = refStmt
    .split(i, i0, i1, 16)
    .split(k, k0, k1, 32)
    .split(l, l0, l1, 32)
    .reorder({i0, i1, j, k0, l0, k1, l1});
  refStmt = insertTemporaries(refStmt);
  refStmt = parallelizeOuterLoop(refStmt);
  ref.compile(refStmt);
  ref.assemble();

  IndexStmt refnStmt = makeReductionNotation(refn.getAssignment());
  refnStmt = makeConcreteNotation(refnStmt);
  refnStmt = refnStmt
    .split(i, i0, i1, 16);
  refnStmt = insertTemporaries(refnStmt);
  refnStmt = parallelizeOuterLoop(refnStmt);
  refn.compile(refnStmt);
  refn.assemble();

  // SpMM , GEMM

  Tensor<double> ref1({B.getDimension(0), kdim}, rm);
  Tensor<double> ref1_2({B.getDimension(0), kdim}, rm);
  Tensor<double> ref2({B.getDimension(0), ldim}, rm);
  Tensor<double> ref2_2({B.getDimension(0), ldim}, rm);
  
  ref1(i,k)=B(i,j)*C(j,k);
  ref1_2(i,k)=B(i,j)*C(j,k);
  ref2(i,l)=ref1(i,k)*D(k,l);
  ref2_2(i,l)=ref1(i,k)*D(k,l);

  IndexStmt ref1Stmt = makeReductionNotation(ref1.getAssignment());
  ref1Stmt = makeConcreteNotation(ref1Stmt);
  ref1Stmt = insertTemporaries(ref1Stmt);
  ref1Stmt = parallelizeOuterLoop(ref1Stmt);
  ref1.compile(ref1Stmt);
  ref1.assemble();

  IndexStmt ref1Stmt2 = makeReductionNotation(ref1_2.getAssignment());
  ref1Stmt2 = makeConcreteNotation(ref1Stmt2);
  ref1Stmt2 = ref1Stmt2
    .split(i, i0, i1, 32)
    .pos(j, jpos, B(i,j))
    .split(jpos, jpos0, jpos1, 4)
    .reorder({i0, i1, jpos0, k, jpos1})
    .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
    .parallelize(k, ParallelUnit::CPUVector, OutputRaceStrategy::IgnoreRaces);
    ;
  // ref1Stmt2 = insertTemporaries(ref1Stmt2);
  ref1_2.compile(ref1Stmt2);
  ref1_2.assemble();

  IndexStmt ref2Stmt = makeReductionNotation(ref2.getAssignment());
  ref2Stmt = makeConcreteNotation(ref2Stmt);
  ref2Stmt = insertTemporaries(ref2Stmt);
  ref2Stmt = ref2Stmt
    .split(i, i0, i1, 16);
  ref2Stmt = parallelizeOuterLoop(ref2Stmt);
  ref2.compile(ref2Stmt);
  ref2.assemble();

  IndexStmt ref2Stmt2 = makeReductionNotation(ref2_2.getAssignment());
  ref2Stmt2 = makeConcreteNotation(ref2Stmt2);
  ref2Stmt2 = ref2Stmt2
    .split(i, i0, i1, 32)
    .split(k, k0, k1, 32)
    .split(l, l0, l1, 32)
    .reorder({i0, k0, l0, i1, k1, l1})
    .parallelize(j0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
  // ref2Stmt2 = insertTemporaries(ref2Stmt2);
  // ref2Stmt2 = parallelizeOuterLoop(ref2Stmt2);
  ref2_2.compile(ref2Stmt2);
  ref2_2.assemble();


  // -------------- GeMM and SpMM 

  Tensor<double> ref3({C.getDimension(0), ldim}, rm);
  Tensor<double> ref4({C.getDimension(0), ldim}, rm);
  ref3(j,l)=C(j,k)*D(k,l); // GEMM
  ref4(i,l) = B(i,j)*ref3(j,l); // SpMM

  IndexStmt ref3Stmt = ref3.getAssignment().concretize();
  ref3Stmt = ref3Stmt
    .split(j, j0, j1, 32) // changed to 32
    .split(k, k0, k1, 32)
    .split(l, l0, l1, 32)
    .reorder({j0, k0, l0, j1, k1, l1})
    .parallelize(j0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
  ref2Stmt2 = insertTemporaries(ref2Stmt2);
  ref3.compile(ref3Stmt);
  ref3.assemble();
  
  IndexStmt ref4Stmt = makeReductionNotation(ref4.getAssignment()); // SpMM operation
  ref4Stmt = makeConcreteNotation(ref4Stmt);
  ref4Stmt = ref4Stmt
    .split(i, i0, i1, 32)
    .pos(j, jpos, B(i,j))
    .split(jpos, jpos0, jpos1, 4)
    .reorder({i0, i1, jpos0, l, jpos1})
    .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
    .parallelize(l, ParallelUnit::CPUVector, OutputRaceStrategy::IgnoreRaces)
    ;
  // ref4Stmt = insertTemporaries(ref4Stmt);
  ref4.compile(ref4Stmt);
  ref4.assemble();

  std::cout << "compute start\n";
  taco::util::TimeResults timevalue;
  vector<double> timeValues(9);
  
  // spmm -> gemm
  ref1.compute(timevalue); // spmm computation
  timeValues[0] = timevalue.mean;

  ref1_2.compute(timevalue); // spmm computation
  timeValues[1] = timevalue.mean;
    
  ref2.compute(timevalue); // gemm computation
  timeValues[2] = timevalue.mean;

  ref2_2.compute(timevalue); // gemm computation  
  timeValues[3] = timevalue.mean;

  // gemm -> spmm
  ref3.compute(timevalue); // gemm computation
  timeValues[4] = timevalue.mean;

  ref4.compute(timevalue); // spmm computation
  timeValues[5] = timevalue.mean;

  // reference
  ref.compute(timevalue);     
  timeValues[6] = timevalue.mean;

  refn.compute(timevalue);   
  timeValues[7] = timevalue.mean;

  // fused
  A.compute(timevalue);
  timeValues[8] = timevalue.mean;

  string dataset = matfile.substr(matfile.find_last_of("/\\") + 1);
  if (statfile.is_open()) {
    statfile 
      << dataset.substr(0, dataset.find_first_of(".")) << ", "
      << timeValues[8] << ", "
      << min(
          min(timeValues[0], timeValues[1]) + min(timeValues[2], timeValues[3]),
          timeValues[4] + timeValues[5]) << ", "
      << min(timeValues[6], timeValues[7])
      << endl
      ;
  } else { std::cout << " stat file is not open\n"; }

  double* A_vals = (double*) (A.getTacoTensorT()->vals);
  double* ref_vals = (double*) (ref.getTacoTensorT()->vals);
  double* ref2_vals = (double*) (ref2.getTacoTensorT()->vals);
  double* ref4_vals = (double*) (ref2.getTacoTensorT()->vals);

  // int* A2_pos = (double*) (ref.getTacoTensorT()->vals);

  // for (size_t q=0; q < B.getStorage().getValues().getSize(); q++) {
  //   if ( abs(A_vals[q] - ref_vals[q])/abs(ref_vals[q]) > ERROR_MARGIN) {
  //     std::cout << "error: results don't match i: " << q << ", avals: " << A_vals[q] << " "
  //       << "refvals: " << ref_vals[q] << std::endl;
  //     ASSERT_TRUE(false);
  //   }
  // }

  for (int q=0; q < A.getDimension(0)* A.getDimension(1); q++) {
    if ( abs(A_vals[q] - ref_vals[q])/abs(ref_vals[q]) > ERROR_MARGIN) {
      std::cout << "error: results don't match i: " << q << ", avals: " << A_vals[q] << " "
        << "refvals: " << ref_vals[q] << std::endl;
      ASSERT_TRUE(false);
    }
  }
  for (int q=0; q < A.getDimension(0)* A.getDimension(1); q++) {
    if ( abs(A_vals[q] - ref2_vals[q])/abs(ref2_vals[q]) > ERROR_MARGIN) {
      std::cout << "error: results don't match i: " << q << ", avals: " << A_vals[q] << " "
        << "refvals: " << ref2_vals[q] << std::endl;
      ASSERT_TRUE(false);
    }
  }
  for (int q=0; q < A.getDimension(0)* A.getDimension(1); q++) {
    if ( abs(A_vals[q] - ref4_vals[q])/abs(ref4_vals[q]) > ERROR_MARGIN) {
      std::cout << "error: results don't match i: " << q << ", avals: " << A_vals[q] << " "
        << "refvals: " << ref4_vals[q] << std::endl;
      ASSERT_TRUE(false);
    }
  }

  if (statfile.is_open()) {
    statfile.close();
  }

}


TEST(scheduling_eval, sddmmSpmmFused) {
  if (should_use_CUDA_codegen()) {
    return;
  }

  taco_set_num_threads(NUM_THREADS_TO_USE);

  std::default_random_engine gen(0);
  std::uniform_real_distribution<double> unif(0.0, 1.0);

  Format csr({dense, sparse});
  Format rm({dense, dense});

  int kdim = 64;
  int ldim = 64;
  int mdim = 64;

  ofstream statfile;
  std::string stat_file = "";
  if (getenv("STAT_FILE")) {
    stat_file = getenv("STAT_FILE");
    std::cout << "stat file: " << stat_file << " is defined\n";
  } else {
    return;
  }
  statfile.open(stat_file, std::ios::app);

  std::string matfile = "";
  if (getenv("TENSOR_FILE")) {
    matfile = getenv("TENSOR_FILE");
    std::cout << "tensor_file: " << matfile << ";\n";
  } else {
    return;
  }

  std::cout << "reading B mat mtx\n";
  Tensor<double> B = read(matfile, csr, true);
  B.setName("B");
  B.pack();

  std::cout << "B dim0: " << B.getDimension(0) << ", dim1: " << B.getDimension(1) << std::endl;
  std::cout << "adding C mat\n";
  Tensor<double> C({B.getDimension(0), kdim}, rm);
  for (int i = 0; i < C.getDimension(0); ++i) {
    for (int j = 0; j < C.getDimension(1); ++j) {
      C.insert({i,j}, unif(gen));
    }
  }
  std::cout << "packing C mat\n";
  C.pack();

  std::cout << "adding D mat\n";
  Tensor<double> D({B.getDimension(1), kdim}, rm);
  for (int i = 0; i < D.getDimension(0); ++i) {
    for (int j = 0; j < D.getDimension(1); ++j) {
      D.insert({i,j}, unif(gen));
    }
  }
  std::cout << "packing D mat\n";
  D.pack();

  std::cout << "adding F mat\n";
  Tensor<double> F({B.getDimension(1), ldim}, rm);
  for (int i = 0; i < F.getDimension(0); ++i) {
    for (int j = 0; j < F.getDimension(1); ++j) {
      F.insert({i,j}, unif(gen));
    }
  }
  std::cout << "packing F mat\n";
  F.pack();

  std::cout << "adding G mat\n";
  Tensor<double> G({ldim, mdim}, rm);
  for (int i = 0; i < G.getDimension(0); ++i) {
    for (int j = 0; j < G.getDimension(1); ++j) {
      G.insert({i,j}, unif(gen));
    }
  }
  std::cout << "packing G mat\n";
  G.pack();

  Tensor<double> A({B.getDimension(0), mdim}, rm);
  Tensor<double> ref({B.getDimension(0), mdim}, rm);
  IndexVar i, j, k, l, m;
  IndexVar i0("i0"), i1("i1"), jpos("jpos"), jpos0("jpos0"), jpos1("jpos1"), k0("k0"), k1("k1");
  IndexVar l0("l0"), l1("l1"), m0("m0"), m1("m1");
  
  A(i,m)=B(i,j)*C(i,k)*D(j,k)*F(j,l)*G(l,m);

  // IndexStmt stmt = A.getAssignment().concretize();
  IndexStmt stmt = makeReductionNotation(A.getAssignment());
  stmt = makeConcreteNotation(stmt);
  stmt = reorderLoopsTopologically(stmt);
  stmt = loopFusionOverFission(stmt, A.getAssignment(), "b", 2);
  stmt = stmt.split(i, i0, i1, 16);

  stmt = insertTemporaries(stmt);
  stmt = parallelizeOuterLoop(stmt); 
  printToFile("sddmmSpMMGeMM", stmt);

  A.compile(stmt);
  A.assemble();


  ref(i,m)=B(i,j)*C(i,k)*D(j,k)*F(j,l)*G(l,m);
  IndexStmt refStmt = makeReductionNotation(ref.getAssignment());
  refStmt = makeConcreteNotation(refStmt);
  refStmt = refStmt.split(i, i0, i1, 16);
  refStmt = insertTemporaries(refStmt);
  refStmt = parallelizeOuterLoop(refStmt);
  ref.compile(refStmt);
  ref.assemble();

  Tensor<double> ref1({B.getDimension(0), B.getDimension(1)}, csr);
  Tensor<double> ref2({B.getDimension(0), ldim}, rm);
  Tensor<double> ref3({B.getDimension(0), mdim}, rm);
  ref1(i,j)=B(i,j)*C(i,k)*D(j,k);
  ref2(i,l)=ref1(i,j)*F(j,l);
  ref3(i,m)=ref2(i,l)*G(l,m);

  IndexStmt ref1Stmt = ref1.getAssignment().concretize();
  
  ref1Stmt = ref1Stmt.split(i, i0, i1, 16);
  //         // .pos(j, jpos, B(i,j));
  //         // .split(k, k0, k1, 8);
  //         // .reorder({i0, i1, jpos0, k, jpos1});
  //         // .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
  //         // .parallelize(jpos1, ParallelUnit::CPUVector, OutputRaceStrategy::ParallelReduction);
  // // ref1Stmt.split(i, );
  // // stmt = scheduleSDDMMCPU_forfuse(ref1Stmt, B);
  // IndexStmt ref1Stmt = makeReductionNotation(ref1.getAssignment());
  // ref1Stmt = makeConcreteNotation(ref1Stmt);
  ref1Stmt = insertTemporaries(ref1Stmt);
  ref1Stmt = parallelizeOuterLoop(ref1Stmt);
  ref1.compile(ref1Stmt);
  ref1.assemble();

  ref2(i,l)=ref1(i,j)*F(j,l);
  IndexStmt ref2Stmt = makeReductionNotation(ref2.getAssignment());
  ref2Stmt = makeConcreteNotation(ref2Stmt);
  ref2Stmt = ref2Stmt
    .split(i, i0, i1, 32)
    .pos(j, jpos, ref1(i,j))
    .split(jpos, jpos0, jpos1, 4)
    .reorder({i0, i1, jpos0, l, jpos1})
    .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
    .parallelize(l, ParallelUnit::CPUVector, OutputRaceStrategy::IgnoreRaces);
    ;
  // ref1Stmt2 = insertTemporaries(ref1Stmt2);
  // ref2Stmt = parallelizeOuterLoop(ref2Stmt);
  ref2.compile(ref2Stmt);
  ref2.assemble();

  // ref3(i,m)=ref2(i,l)*G(l,m);
  IndexStmt ref3Stmt = makeReductionNotation(ref3.getAssignment());
  ref3Stmt = makeConcreteNotation(ref3Stmt);
  ref3Stmt = ref3Stmt
    .split(i, i0, i1, 32)
    .split(l, l0, l1, 32)
    .split(m, m0, m1, 32)
    .reorder({i0, l0, m0, i1, l1, m1});
  ref3Stmt = insertTemporaries(ref3Stmt);
  ref3Stmt = parallelizeOuterLoop(ref3Stmt);
  ref3.compile(ref3Stmt);
  ref3.assemble();

  std::cout << "compute start\n";
  taco::util::TimeResults timevalue;
  // fused, sddmm, sddmm_ryan, spmm_ryan, gemm, reference
  vector<double> timeValues(5); 
  
  A.compute(timevalue);
  timeValues[0] = timevalue.mean;
  
  ref1.compute(timevalue);
  timeValues[1] = timevalue.mean;
  
  ref2.compute(timevalue);
  timeValues[2] = timevalue.mean;

  ref3.compute(timevalue);
  timeValues[3] = timevalue.mean;

  ref.compute(timevalue);
  timeValues[4] = timevalue.mean;

  string dataset = matfile.substr(matfile.find_last_of("/\\") + 1);
  if (statfile.is_open()) {
    statfile 
      << dataset.substr(0, dataset.find_first_of(".")) << ", "
      << timeValues[0] << ", "
      << timeValues[1] + timeValues[2] + timeValues[3] << ", "
      << timeValues[4]
      << endl
      ;
  } else { std::cout << " stat file is not open\n"; }

  double* A_vals = (double*) (A.getTacoTensorT()->vals);
  double* ref_vals = (double*) (ref.getTacoTensorT()->vals);
  double* ref3_vals = (double*) (ref3.getTacoTensorT()->vals);

  // int* A2_pos = (double*) (ref.getTacoTensorT()->vals);

  for (int q=0; q < A.getDimension(0)* A.getDimension(1); q++) {
    if ( abs(A_vals[q] - ref_vals[q])/abs(ref_vals[q]) > ERROR_MARGIN) {
      std::cout << "error: results don't match i: " << q << ", avals: " << A_vals[q] << " "
        << "refvals: " << ref_vals[q] << std::endl;
      ASSERT_TRUE(false);
    }
  }

  for (int q=0; q < A.getDimension(0)* A.getDimension(1); q++) {
    if ( abs(ref3_vals[q] - ref_vals[q])/abs(ref_vals[q]) > ERROR_MARGIN) {
      std::cout << "error: results don't match i: " << q << ", avals: " << ref3_vals[q] << " "
        << "refvals: " << ref_vals[q] << std::endl;
      ASSERT_TRUE(false);
    }
  }

  if (statfile.is_open()) {
    statfile.close();
  }

}