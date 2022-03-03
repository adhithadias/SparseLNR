#include "taco/cuda.h"
#include "taco/tensor.h"
#include "test.h"
#include "util.h"
#include <climits>
#include "gtest/gtest.h"
#include <cstdint>
#include <papi.h>

// #define NUM_THREADS_TO_USE 64
#define NUM_THREADS_TO_USE 32

void handle_error (int retval)
{
     printf("PAPI error %d: %s\n", retval, PAPI_strerror(retval));
     exit(1);
}

TEST(scheduling_eval, spmvFusedWithSyntheticData) {
  if (should_use_CUDA_codegen() || should_use_ISPC_codegen()) {
    return;
  }
  taco_set_num_threads(NUM_THREADS_TO_USE);

  std::default_random_engine gen(0);
  std::uniform_real_distribution<double> unif(0.0, 1.0);

  Format csr({dense, sparse});
  Format  rm({dense});

  // uncomment this for reading the csr matrix saved in mtx file
  std::cout << "reading B mat mtx\n";

  int NUM_I = 5; // 1021/10;
  int NUM_J = 5; // 1039/10;
  int NUM_K = 8;
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
  Tensor<double> C("C", {NUM_J, NUM_K}, csr);
  for (int i = 0; i < C.getDimension(0); ++i) {
    for (int j = 0; j < C.getDimension(1); ++j) {
      C.insert({i,j}, unif(gen));
    }
  }
  std::cout << "packing C mat\n";
  C.pack();

  Tensor<double> v("v", {NUM_K}, rm);
  for (int i = 0; i < v.getDimension(0); ++i) {
      v.insert({i}, unif(gen));
  }
  std::cout << "packing D mat\n";
  v.pack();

  Tensor<double> A("A", {NUM_I}, rm);
  Tensor<double> ref("ref", {NUM_I}, rm);
  IndexVar i, j, k, l, m;
  A(i) = B(i,j) * C(j,k) * v(k);

  // IndexStmt stmt = A.getAssignment().concretize();
  IndexStmt stmt = makeReductionNotation(A.getAssignment());
  stmt = makeConcreteNotation(stmt);
  printToFile("SpMVfused", stmt);
  stmt = reorderLoopsTopologically(stmt);
  stmt = loopFusionOverFission(stmt, A.getAssignment(), "f", 1);
  stmt = insertTemporaries(stmt);
  stmt = parallelizeOuterLoop(stmt);

  A.compile(stmt);
  // We can now call the functions taco generated to assemble the indices of the
  // output matrix and then actually compute the MTTKRP.
  A.assemble();


  // ref(i) = B(i,j) * C(j,k) * v(k);
  // IndexStmt refStmt = makeReductionNotation(ref.getAssignment());
  // refStmt = makeConcreteNotation(refStmt);
  // refStmt = insertTemporaries(refStmt);
  // refStmt = parallelizeOuterLoop(refStmt);
  // ref.compile(refStmt);
  // ref.assemble();

  // Tensor<double> ref1({NUM_J}, rm);
  // Tensor<double> ref2({NUM_I}, rm);
  // ref1(j) = C(j,k) * v(k);
  // ref2(i) = B(i,j) * ref1(j);

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
  // TOOL_BENCHMARK_TIMER(ref.compute(), "\n\nReference Kernel: ", timevalue);
  TOOL_BENCHMARK_TIMER(A.compute(), "\n\nFused Kernel: ", timevalue);
  // ASSERT_TENSOR_EQ(ref, A);

  // // check results
  // for (int q = 0; q < A.getDimension(0); ++q) {
  //   if ( abs(A(q) - ref(q))/abs(ref(q)) > ERROR_MARGIN) {
  //     std::cout << "error: results don't match A("<< q << "): " 
  //       << A(q) << ", ref: " << ref(q) << std::endl;
  //     ASSERT_TRUE(false);
  //   }
  // }
  // // ASSERT_TENSOR_EQ(A, ref);
  // TOOL_BENCHMARK_TIMER(ref1.compute(), "\n\nSDDMM Kernel: ", timevalue);
  // TOOL_BENCHMARK_TIMER(ref2.compute(), "\n\nSpMM Kernel: ", timevalue);
  // ASSERT_TENSOR_EQ(ref, ref2);

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

TEST(scheduling_eval, spmvFused) {
  if (should_use_CUDA_codegen() || should_use_ISPC_codegen()) {
    return;
  }

  ofstream statfile;
  statfile.open(
    "/home/min/a/kadhitha/workspace/my_taco/taco/test/stats/spmv-spmv.txt", std::ios::app);
  if (statfile.is_open()) {
    statfile << "\nspmv-spmv execution\n";
    statfile << "\n-----------------------------------------\n";
  }
  taco_set_num_threads(NUM_THREADS_TO_USE);

  std::default_random_engine gen(0);
  std::uniform_real_distribution<double> unif(0.0, 1.0);

  Format csr({dense, sparse});
  Format  rm({dense});



  int filenum = 1;

  std::vector<std::string> matfiles = {
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/synthetic/synthetic.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/cage3/cage3.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/bcsstk17/bcsstk17.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/pdb1HYS/pdb1HYS.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rma10/rma10.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/cant/cant.mtx", // 5
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/consph/consph.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/cop20k_A/cop20k_A.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/shipsec1/shipsec1.mtx", // 8
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/scircuit/scircuit.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/mac_econ_fwd500/mac_econ_fwd500.mtx", // 10
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/wtk/pwtk.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/ufl/webbase-1M/webbase-1M.mtx", // 12
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/wiki-Talk/wiki-Talk.mtx", // 13
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/com-Orkut/com-Orkut.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/circuit5M/circuit5M.mtx", // 15
    "/home/min/a/kadhitha/workspace/my_taco/FusedMM/dataset/harvard.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/twitter7/twitter7.mtx"
  };
  std::vector<std::string> matfilesrw = {
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/synthetic.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/cage3.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/bcsstk17.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/pdb1HYS.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/rma10.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/cant.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/consph.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/cop20k_A.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/shipsec1.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/scircuit.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/mac_econ_fwd500/mac_econ_fwd500.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/wtk/pwtk.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/webbase-1M.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/wiki-Talk.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/com-Orkut.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/circuit5M.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/harvard.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/twitter7.mtx"
  };

  // uncomment this for reading the csr matrix saved in mtx file
  std::cout << "reading B mat mtx\n";


  int kDim = 8;
  float SPARSITY = .3;
  std::string matfile = matfiles[filenum];
  std::cout << "reading B mat mtx\n";
  Tensor<double> B = read(matfile, csr, true);
  B.setName("B");
  B.pack();

  std::cout << "B dim0: " << B.getDimension(0) << ", dim1: " << B.getDimension(1) << std::endl;
  std::cout << "adding c mat\n";

  std::cout << "reading B mat mtx\n";
  Tensor<double> C = read(matfile, csr, true);
  C.setName("C");
  C.pack();


  Tensor<double> v("v", {C.getDimension(1)}, rm);
  for (int i = 0; i < v.getDimension(0); ++i) {
      v.insert({i}, unif(gen));
  }
  std::cout << "packing D mat\n";
  v.pack();

  if (statfile.is_open()) {
    statfile 
      << "A(i) = B(i,j) * C(j,k) * v(k);" << std::endl
      << "B1_dimension: " << B.getDimension(0) << ", B2_dimension: " << B.getDimension(1) << ", vals: " << B.getStorage().getValues().getSize() << std::endl
      << "C1_dimension: " << C.getDimension(0) << ", C2_dimension: " << C.getDimension(1) << ", vals: " << C.getStorage().getValues().getSize() << std::endl
      << "D1_dimension: " << v.getDimension(0) << ", vals: " << v.getStorage().getValues().getSize() << std::endl
      << std::endl;
  }

  Tensor<double> A("A", {B.getDimension(0)}, rm);
  Tensor<double> ref("ref", {B.getDimension(0)}, rm);
  IndexVar i, j, k, l, m;
  A(i) = B(i,j) * C(j,k) * v(k);

  ref(i) = B(i,j) * C(j,k) * v(k);
  IndexStmt refStmt = makeReductionNotation(ref.getAssignment());
  refStmt = makeConcreteNotation(refStmt);
  refStmt = insertTemporaries(refStmt);
  refStmt = parallelizeOuterLoop(refStmt);
  ref.compile(refStmt);
  ref.assemble();

  // IndexStmt stmt = A.getAssignment().concretize();
  IndexStmt stmt = makeReductionNotation(A.getAssignment());
  stmt = makeConcreteNotation(stmt);
  printToFile("SpMVfused", stmt);
  stmt = reorderLoopsTopologically(stmt);
  stmt = loopFusionOverFission(stmt, A.getAssignment(), "f", 1);
  stmt = insertTemporaries(stmt);
  stmt = parallelizeOuterLoop(stmt);
  A.compile(stmt);
  A.assemble();


  // Tensor<double> ref1({NUM_J}, rm);
  // Tensor<double> ref2({NUM_I}, rm);
  // ref1(j) = C(j,k) * v(k);
  // ref2(i) = B(i,j) * ref1(j);

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
    std::string sofused = "/home/min/a/kadhitha/workspace/my_taco/taco/test/kernels/spmv_spmv/spmv_fused.so";

  TOOL_BENCHMARK_TIMER(ref.compute(statfile, sofused), "\n\nReference Kernel: ", timevalue);

  
  std::cout << "b1 dim: " << B.getTacoTensorT()->dimensions[1] << std::endl;
  // TOOL_BENCHMARK_TIMER(ref.compute(statfile, sofused), "\n\nFused Kernel: ", timevalue);
  // ASSERT_TENSOR_EQ(ref, A);

  // // check results
  // for (int q = 0; q < A.getDimension(0); ++q) {
  //   if ( abs(A(q) - ref(q))/abs(ref(q)) > ERROR_MARGIN) {
  //     std::cout << "error: results don't match A("<< q << "): " 
  //       << A(q) << ", ref: " << ref(q) << std::endl;
  //     ASSERT_TRUE(false);
  //   }
  // }
  // // ASSERT_TENSOR_EQ(A, ref);
  // TOOL_BENCHMARK_TIMER(ref1.compute(), "\n\nSDDMM Kernel: ", timevalue);
  // TOOL_BENCHMARK_TIMER(ref2.compute(), "\n\nSpMM Kernel: ", timevalue);
  // ASSERT_TENSOR_EQ(ref, ref2);

  // for (int q = 0; q < ref2.getDimension(0); ++q) {
  //   for (int w = 0; w < ref2.getDimension(1); ++w) {
  //     if ( abs(ref2(q,w) - ref(q,w))/abs(ref(q,w)) > ERROR_MARGIN) {
  //       std::cout << "error: results don't match A("<< q << "," << w << "): " 
  //         << ref2(q,w) << ", ref: " << ref(q,w) << std::endl;
  //       ASSERT_TRUE(false);
  //     }
  //   }
  // }

  if (statfile.is_open()) {
    statfile.close();
  }

}

TEST(scheduling_eval, sddmmFusedWithSyntheticData) {
  if (should_use_CUDA_codegen() || should_use_ISPC_codegen()) {
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
  write("/home/min/a/kadhitha/ispc-examples/data/suitesparse/synthetic/synthetic.mtx", B);

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
  if (should_use_CUDA_codegen() || should_use_ISPC_codegen()) {
    return;
  }

  taco_set_num_threads(NUM_THREADS_TO_USE);

  ofstream statfile;
  statfile.open(
    "/home/min/a/kadhitha/workspace/my_taco/taco/test/stats/sddmm-spmm.txt", std::ios::app);
  if (statfile.is_open()) {
    statfile << "\nsddmm-spmm execution\n";
    statfile << "\n-----------------------------------------\n";
  }

  std::default_random_engine gen(0);
  std::uniform_real_distribution<double> unif(0.0, 1.0);

  Format csr({dense, sparse});
  Format rm({dense, dense});
  int ldim = 128;
  int kdim = 128;

  // vector<int> filenums = {2,3,4,5,6,7,8,9,10,12,15};

  vector<int> filenums = {1};

  for (auto filenum : filenums) {

  // int filenum = 5;

  std::vector<std::string> matfiles = {
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/synthetic/synthetic.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/cage3/cage3.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/bcsstk17/bcsstk17.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/pdb1HYS/pdb1HYS.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rma10/rma10.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/cant/cant.mtx", // 5
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/consph/consph.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/cop20k_A/cop20k_A.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/shipsec1/shipsec1.mtx", // 8
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/scircuit/scircuit.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/mac_econ_fwd500/mac_econ_fwd500.mtx", // 10
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/wtk/pwtk.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/ufl/webbase-1M/webbase-1M.mtx", // 12
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/wiki-Talk/wiki-Talk.mtx", // 13
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/com-Orkut/com-Orkut.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/circuit5M/circuit5M.mtx", // 15
    "/home/min/a/kadhitha/workspace/my_taco/FusedMM/dataset/harvard.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/twitter7/twitter7.mtx"
  };
  std::vector<std::string> matfilesrw = {
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/synthetic.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/cage3.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/bcsstk17.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/pdb1HYS.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/rma10.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/cant.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/consph.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/cop20k_A.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/shipsec1.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/scircuit.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/mac_econ_fwd500/mac_econ_fwd500.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/wtk/pwtk.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/webbase-1M.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/wiki-Talk.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/com-Orkut.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/circuit5M.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/harvard.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/twitter7.mtx"
  };

  std::string matfile = matfiles[filenum];
  std::cout << "reading B mat mtx\n";
  Tensor<double> B = read(matfile, csr, true);
  B.setName("B");
  B.pack();
  // write(matfilesrw[filenum], B);

  if (statfile.is_open()) {
    statfile << matfile << std::endl;
  }

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
  IndexVar i, j, k, l, m;
  IndexVar i0("i0"), i1("i1"), jpos("jpos"), jpos0("jpos0"), jpos1("jpos1"), k0("k0"), k1("k1");
  A(i,l)=B(i,j)*C(i,k)*D(j,k)*F(j,l);
  if (statfile.is_open()) {
    statfile 
      << "ref(i,l)=B(i,j)*C(i,k)*D(j,k)*F(j,l);" << std::endl
      << "B1_dimension: " << B.getDimension(0) << ", B2_dimension: " << B.getDimension(1) << ", vals: " << B.getStorage().getValues().getSize() << std::endl
      << "C1_dimension: " << C.getDimension(0) << ", C2_dimension: " << C.getDimension(1) << ", vals: " << C.getStorage().getValues().getSize() << std::endl
      << "D1_dimension: " << D.getDimension(0) << ", D2_dimension: " << D.getDimension(1) << ", vals: " << D.getStorage().getValues().getSize() << std::endl
      << "E1_dimension: " << F.getDimension(0) << ", E2_dimension: " << F.getDimension(1) << ", vals: " << F.getStorage().getValues().getSize() << std::endl
      << std::endl;
  }

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
    .split(i, i0, i1, 16)
    .reorder({i0, i1, j, k, l});
  stmt = insertTemporaries(stmt);
  refStmt = parallelizeOuterLoop(refStmt);
  ref.compile(refStmt);
  ref.assemble();

  Tensor<double> ref1({B.getDimension(0), B.getDimension(1)}, csr);
  Tensor<double> ref2({B.getDimension(0), ldim}, rm);
  ref1(i,j)=B(i,j)*C(i,k)*D(j,k);
  ref2(i,l)=ref1(i,j)*F(j,l);

  IndexStmt ref1Stmt = ref1.getAssignment().concretize(); // anyway Ryan's kernel is used here
  
  ref1Stmt = ref1Stmt.split(i, i0, i1, 16);
          // .pos(j, jpos, B(i,j));
          // .split(k, k0, k1, 8);
          // .reorder({i0, i1, jpos0, k, jpos1});
          // .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
          // .parallelize(jpos1, ParallelUnit::CPUVector, OutputRaceStrategy::ParallelReduction);
  // ref1Stmt.split(i, );
  // stmt = scheduleSDDMMCPU_forfuse(ref1Stmt, B);
  // IndexStmt ref1Stmt = makeReductionNotation(ref1.getAssignment());
  // ref1Stmt = makeConcreteNotation(ref1Stmt);
  ref1Stmt = insertTemporaries(ref1Stmt);
  ref1Stmt = parallelizeOuterLoop(ref1Stmt);
  ref1.compile(ref1Stmt);
  ref1.assemble();

  IndexStmt ref2Stmt = makeReductionNotation(ref2.getAssignment()); // Ryan's SpMM kernel is used here
  ref2Stmt = makeConcreteNotation(ref2Stmt);
  ref2Stmt = insertTemporaries(ref2Stmt);
  ref2Stmt = parallelizeOuterLoop(ref2Stmt);
  ref2.compile(ref2Stmt);
  ref2.assemble();

  std::cout << "compute start\n";
  taco::util::TimeResults timevalue;
  bool time                = true;
  
  std::string sofile_fused = "/home/min/a/kadhitha/workspace/my_taco/taco/test/kernels/sddmm_spmm/fused_kernel.so";
  TOOL_BENCHMARK_TIMER(A.compute(statfile), "\n\nFused Kernel: ", timevalue);
  if (statfile.is_open()) {
    statfile << "fused time: ";
    statfile << timevalue.mean << std::endl;
  } else { std::cout << " stat file is not open\n"; }

  statfile << "\nseparate execution\n";
  
  // // std::string sofile_sddmm = "/home/min/a/kadhitha/workspace/my_taco/taco/test/kernels/sddmm_spmm/csr_dense_spmm.so";
  // std::string sofile_sddmm = "/home/min/a/kadhitha/workspace/my_taco/taco/test/kernels/sddmm_spmm/csr_dense_dense_sddmm.so";
  // TOOL_BENCHMARK_TIMER(ref1.compute(statfile, sofile_sddmm), "\n\nSDDMM Kernel: ", timevalue);
  // if (statfile.is_open()) {
  //   statfile << "sddmm time: ";
  //   statfile << timevalue.mean << std::endl;
  // } else { std::cout << " stat file is not open\n"; }

  // std::string sofile_sddmm_ryan = "/home/min/a/kadhitha/workspace/my_taco/taco/test/kernels/sddmm_spmm/sddmm_ryan.so";
  // TOOL_BENCHMARK_TIMER(ref1.compute(statfile, sofile_sddmm_ryan), "\n\nSDDMM Kernel: ", timevalue);
  // if (statfile.is_open()) {
  //   statfile << "sddmm time: ";
  //   statfile << timevalue.mean << std::endl;
  // } else { std::cout << " stat file is not open\n"; }
  
  // std::string sofile_spmm = "/home/min/a/kadhitha/workspace/my_taco/taco/test/kernels/sddmm_spmm/csr_dense_spmm.so";
  // TOOL_BENCHMARK_TIMER(ref2.compute(statfile, sofile_spmm), "\n\nSpMM Kernel: ", timevalue);
  // if (statfile.is_open()) {
  //   statfile << "spmm time: ";
  //   statfile << timevalue.mean << std::endl;
  // } else { std::cout << " stat file is not open\n"; }

  // statfile << "\nreference execution \n";

  // std::string sofile_original = "/home/min/a/kadhitha/workspace/my_taco/taco/test/kernels/sddmm_spmm/taco_original.so";
  // TOOL_BENCHMARK_TIMER(ref.compute(statfile, sofile_original), "\n\nReference Kernel: ", timevalue);
  // if (statfile.is_open()) {
  //   statfile << "taco reference time: ";
  //   statfile << timevalue << std::endl;
  // } else { std::cout << " stat file is not open\n"; }

  // double* A_vals = (double*) (A.getTacoTensorT()->vals);
  // double* ref_vals = (double*) (ref.getTacoTensorT()->vals);
  // double* ref2_vals = (double*) (ref2.getTacoTensorT()->vals);

  // // int* A2_pos = (double*) (ref.getTacoTensorT()->vals);

  // // for (size_t q=0; q < B.getStorage().getValues().getSize(); q++) {
  // //   if ( abs(A_vals[q] - ref_vals[q])/abs(ref_vals[q]) > ERROR_MARGIN) {
  // //     std::cout << "error: results don't match i: " << q << ", avals: " << A_vals[q] << " "
  // //       << "refvals: " << ref_vals[q] << std::endl;
  // //     ASSERT_TRUE(false);
  // //   }
  // // }

  // for (size_t q=0; q < A.getDimension(0)* A.getDimension(1); q++) {
  //   if ( abs(A_vals[q] - ref_vals[q])/abs(ref_vals[q]) > ERROR_MARGIN) {
  //     std::cout << "error: results don't match i: " << q << ", avals: " << A_vals[q] << " "
  //       << "refvals: " << ref_vals[q] << std::endl;
  //     ASSERT_TRUE(false);
  //   }
  // }
  // for (size_t q=0; q < A.getDimension(0)* A.getDimension(1); q++) {
  //   if ( abs(A_vals[q] - ref2_vals[q])/abs(ref2_vals[q]) > ERROR_MARGIN) {
  //     std::cout << "error: results don't match i: " << q << ", avals: " << A_vals[q] << " "
  //       << "refvals: " << ref2_vals[q] << std::endl;
  //     ASSERT_TRUE(false);
  //   }
  // }
  // for (int q= 0; q< A_vals
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

  } // end of for loop


  if (statfile.is_open()) {
    statfile.close();
  }
}




TEST(scheduling_eval, hadamardFused) {
  if (should_use_CUDA_codegen() || should_use_ISPC_codegen()) {
    return;
  }

  ofstream statfile;
  statfile.open(
    "/home/min/a/kadhitha/workspace/my_taco/taco/test/stats/hadamard-gemm.txt", std::ios::app);
  if (statfile.is_open()) {
    statfile << "\nsddmm-spmm execution\n";
    statfile << "\n-----------------------------------------\n";
  }

  std::default_random_engine gen(0);
  std::uniform_real_distribution<double> unif(0.0, 1.0);

  Format csr({dense, sparse});
  Format rm({dense, dense});
  int kdim = 128;
  int ldim = 128;

  vector<int> filenums = {2,3,4,5,6,7,8,9,10,12,15};
  // vector<int> filenums = {8,9,10,12};

  for (auto filenum : filenums) {

  // int filenum = 15;

  std::vector<std::string> matfiles = {
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/synthetic/synthetic.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/cage3/cage3.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/bcsstk17/bcsstk17.mtx", // 2
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/pdb1HYS/pdb1HYS.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rma10/rma10.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/cant/cant.mtx", // 5
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/consph/consph.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/cop20k_A/cop20k_A.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/shipsec1/shipsec1.mtx", // 8
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/scircuit/scircuit.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/mac_econ_fwd500/mac_econ_fwd500.mtx", // 10
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/wtk/pwtk.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/ufl/webbase-1M/webbase-1M.mtx", // 12
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/wiki-Talk/wiki-Talk.mtx", // 13
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/com-Orkut/com-Orkut.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/circuit5M/circuit5M.mtx", // 15
    "/home/min/a/kadhitha/workspace/my_taco/FusedMM/dataset/harvard.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/twitter7/twitter7.mtx"
  };
  std::vector<std::string> matfilesrw = {
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/synthetic.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/cage3.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/bcsstk17.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/pdb1HYS.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/rma10.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/cant.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/consph.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/cop20k_A.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/shipsec1.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/scircuit.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/mac_econ_fwd500/mac_econ_fwd500.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/wtk/pwtk.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/webbase-1M.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/wiki-Talk.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/com-Orkut.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/circuit5M.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/harvard.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/twitter7.mtx"
  };

  std::string matfile = matfiles[filenum];
  std::cout << "reading B mat mtx\n";
  Tensor<double> B = read(matfile, csr, true);
  B.setName("B");
  B.pack();
  // write(matfilesrw[filenum], B);

  if (statfile.is_open()) {
    statfile << matfile << std::endl;
  }

  std::cout << "B dim0: " << B.getDimension(0) << ", dim1: " << B.getDimension(1) << std::endl;
  std::cout << "adding c mat\n";
  Tensor<double> C({B.getDimension(1), kdim}, rm);
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
  if (statfile.is_open()) {
    statfile 
      << "ref(i,l)=B(i,j)*C(i,k)*D(j,k)*F(j,l);" << std::endl
      << "B1_dimension: " << B.getDimension(0) << ", B2_dimension: " << B.getDimension(1) << ", vals: " << B.getStorage().getValues().getSize() << std::endl
      << "C1_dimension: " << C.getDimension(0) << ", C2_dimension: " << C.getDimension(1) << ", vals: " << C.getStorage().getValues().getSize() << std::endl
      << "D1_dimension: " << D.getDimension(0) << ", D2_dimension: " << D.getDimension(1) << ", vals: " << D.getStorage().getValues().getSize() << std::endl
      << "E1_dimension: " << F.getDimension(0) << ", E2_dimension: " << F.getDimension(1) << ", vals: " << F.getStorage().getValues().getSize() << std::endl
      << std::endl;
  }

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
  bool time                = true;
  
  TOOL_BENCHMARK_TIMER(A.compute(statfile), "\n\nFused Kernel: ", timevalue);
  if (statfile.is_open()) {
    statfile << "fused time: ";
    statfile << timevalue.mean << std::endl;
  } else { std::cout << " stat file is not open\n"; }
  
  // // std::string sofile_sddmm = "/home/min/a/kadhitha/workspace/my_taco/taco/test/kernels/sddmm_spmm/csr_dense_spmm.so";
  // std::string sofile_sddmm = "/home/min/a/kadhitha/workspace/my_taco/taco/test/kernels/sddmm_spmm/csr_dense_dense_sddmm.so";
  TOOL_BENCHMARK_TIMER(ref1.compute(statfile), "\n\nHadamard Kernel: ", timevalue);
  if (statfile.is_open()) {
    statfile << "hadamard time: ";
    statfile << timevalue.mean << std::endl;
  } else { std::cout << " stat file is not open\n"; }

  // std::string sofile_sddmm_ryan = "/home/min/a/kadhitha/workspace/my_taco/taco/test/kernels/sddmm_spmm/sddmm_ryan.so";
  // TOOL_BENCHMARK_TIMER(ref1.compute(statfile, sofile_sddmm_ryan), "\n\nSDDMM Kernel: ", timevalue);
  // if (statfile.is_open()) {
  //   statfile << "sddmm time: ";
  //   statfile << timevalue.mean << std::endl;
  // } else { std::cout << " stat file is not open\n"; }
  
  // std::string sofile_spmm = "/home/min/a/kadhitha/workspace/my_taco/taco/test/kernels/sddmm_spmm/csr_dense_spmm.so";
  TOOL_BENCHMARK_TIMER(ref2.compute(statfile), "\n\nGeMM Kernel: ", timevalue);
  if (statfile.is_open()) {
    statfile << "gemm time: ";
    statfile << timevalue.mean << std::endl;
  } else { std::cout << " stat file is not open\n"; }

  // std::string sofile_original = "/home/min/a/kadhitha/workspace/my_taco/taco/test/kernels/sddmm_spmm/taco_original.so";
  TOOL_BENCHMARK_TIMER(ref.compute(statfile), "\n\nReference Kernel: ", timevalue);
  if (statfile.is_open()) {
    statfile << "taco reference time: ";
    statfile << timevalue << std::endl;
  } else { std::cout << " stat file is not open\n"; }

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

  } // end of for loop

  if (statfile.is_open()) {
    statfile.close();
  }

}






TEST(scheduling_eval, mttkrpFusedWithSyntheticData) {
  if (should_use_CUDA_codegen() || should_use_ISPC_codegen()) {
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

  int NUM_I = 1021/20;
  int NUM_J = 1039/20;
  int NUM_K = 1057/20;
  int NUM_L = 1232/20;
  int NUM_M = 1231/20;
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
  write("/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/synthetic.tns", B);

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
  if (should_use_CUDA_codegen() || should_use_ISPC_codegen()) {
    return;
  }

  taco_set_num_threads(NUM_THREADS_TO_USE);

  ofstream statfile;
  statfile.open(
    "/home/min/a/kadhitha/workspace/my_taco/taco/test/stats/mttkrp-spmm.txt", std::ios::app);
  if (statfile.is_open()) {
    statfile << "\nmttkrp-spmm execution\n";
    statfile << "\n-----------------------------------------\n";
  }

  std::default_random_engine gen(0);
  std::uniform_real_distribution<double> unif(0.0, 1.0);
  // Predeclare the storage formats that the inputs and output will be stored as.
  // To define a format, you must specify whether each dimension is dense or 
  // sparse and (optionally) the order in which dimensions should be stored. The 
  // formats declared below correspond to compressed sparse fiber (csf) and 
  // row-major dense (rm).
  Format csf({Dense,Sparse,Sparse});
  Format rm({Dense,Dense});
  Format sd({Dense,Dense});
  int jDim = 32;
  int mDim = 64;

  int matfilenum = 3;

  // Load a sparse order-3 tensor from file (stored in the FROSTT format) and 
  // store it as a compressed sparse fiber tensor. The tensor in this example 
  // can be download from: http://frostt.io/tensors/nell-2/
  std::vector<std::string> matfiles = {
    "/home/min/a/kadhitha/ispc-examples/data/tns/matmul_5-5-5.tns",
    "/home/min/a/kadhitha/ispc-examples/data/tns/delicious-3d.tns", 
    "/home/min/a/kadhitha/ispc-examples/data/tns/flickr-3d.tns", // 2
    "/home/min/a/kadhitha/ispc-examples/data/tns/nell-2.tns", // 3
    "/home/min/a/kadhitha/ispc-examples/data/tns/nell-1.tns", // 4
    "/home/min/a/kadhitha/ispc-examples/data/tns/vast-2015-mc1-3d.tns", // 5
    "/home/min/a/kadhitha/ispc-examples/data/tns/darpa1998.tns", // 6
    "/home/min/a/kadhitha/ispc-examples/data/tns/freebase_music.tns",
    "/home/min/a/kadhitha/ispc-examples/data/tns/freebase_sampled.tns" // 8
  };
  std::vector<std::string> matfilesrw = {
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/matmul_5-5-5.tns",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/delicious-3d.tns",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/flickr-3d.tns", // 2 
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/nell-2.tns", //  3
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/nell-1.tns", //   4
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/vast-2015-mc1-3d.tns", // 5
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/darpa1998.tns",  // 6
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/freebase_music.tns",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/freebase_sampled.tns"
  };
  std::string matfile = matfiles[matfilenum];
  Tensor<double> B = read(matfile, csf, true);
  // write(matfilesrw[matfilenum], B);

  // Generate a random dense matrix and store it in row-major (dense) format. 
  // Matrices correspond to order-2 tensors in taco.
  Tensor<double> C({B.getDimension(1), jDim}, rm);
  for (int i = 0; i < C.getDimension(0); ++i) {
    for (int j = 0; j < C.getDimension(1); ++j) {
      C.insert({i,j}, unif(gen));
    }
  }
  C.pack();

  // Generate another random dense matrix and store it in row-major format.
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

  if (statfile.is_open()) {
    statfile 
      << matfile << std::endl
      << "A(i,m) = B(i,k,l) * D(l,j) * C(k,j) * E(j, m)" << std::endl
      << "B1_dimension: " << B.getDimension(0) << ", B2_dimension: " << B.getDimension(1) << ", B3_dimension: " << B.getDimension(0) << ", vals: " << B.getStorage().getValues().getSize() << std::endl
      << "C1_dimension: " << C.getDimension(0) << ", C2_dimension: " << C.getDimension(1) << ", vals: " << C.getStorage().getValues().getSize() << std::endl
      << "D1_dimension: " << D.getDimension(0) << ", D2_dimension: " << D.getDimension(1) << ", vals: " << D.getStorage().getValues().getSize() << std::endl
      << "E1_dimension: " << E.getDimension(0) << ", E2_dimension: " << E.getDimension(1) << ", vals: " << E.getStorage().getValues().getSize() << std::endl
      << std::endl;
  }

    // Declare the output matrix to be a dense matrix with 25 columns and the same
  // number of rows as the number of slices along the first dimension of input
  // tensor B, to be also stored as a row-major dense matrix.
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
  stmt = stmt.split(i, i1, i2, 16);
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
  bool time                = true;

  TOOL_BENCHMARK_TIMER(ref2.compute(statfile), "\n\nDefault MTTKRP: ", timevalue);
  if (statfile.is_open()) {
    statfile << "default mttkrp time: ";
    statfile << timevalue.mean << std::endl;
  } else { std::cout << " stat file is not open\n"; }

  TOOL_BENCHMARK_TIMER(ref2_ryan.compute(statfile), "\n\nRyan MTTKRP workspace: ", timevalue);
  if (statfile.is_open()) {
    statfile << "ryan mttkrp workspace time: ";
    statfile << timevalue.mean << std::endl;
  } else { std::cout << " stat file is not open\n"; }

  double* ref2_vals = (double*) (ref2.getTacoTensorT()->vals);
  double* ref2_ryan_vals = (double*) (ref2_ryan.getTacoTensorT()->vals);
  for (int q=0; q < B.getDimension(0)* jDim; q++) {
    if ( abs(ref2_vals[q] - ref2_ryan_vals[q])/abs(ref2_ryan_vals[q]) > ERROR_MARGIN) {
      std::cout << "error: results don't match i: " << q << ", avals: " << ref2_vals[q] << " "
        << "refvals: " << ref2_ryan_vals[q] << std::endl;
      ASSERT_TRUE(false);
    }
  }

  TOOL_BENCHMARK_TIMER(ref3.compute(statfile), "\n\nGeMM time: ", timevalue);
  if (statfile.is_open()) {
    statfile << "GeMM time: ";
    statfile << timevalue.mean << std::endl;
  } else { std::cout << " stat file is not open\n"; }


  TOOL_BENCHMARK_TIMER(ref.compute(statfile), "\n\nReference MTTKRP+GEMM: ", timevalue);
  if (statfile.is_open()) {
    statfile << "reference asymptotic blowup time: ";
    statfile << timevalue.mean << std::endl;
  } else { std::cout << " stat file is not open\n"; }

  double* ref3_vals = (double*) (ref3.getTacoTensorT()->vals);
  double* ref_vals = (double*) (ref.getTacoTensorT()->vals);
  for (int q=0; q < B.getDimension(0)* mDim; q++) {
    if ( abs(ref3_vals[q] - ref_vals[q])/abs(ref_vals[q]) > ERROR_MARGIN) {
      std::cout << "error: results don't match i: " << q << ", avals: " << ref3_vals[q] << " "
        << "refvals: " << ref_vals[q] << std::endl;
      ASSERT_TRUE(false);
    }
  }

  TOOL_BENCHMARK_TIMER(A.compute(statfile), "\n\nFused MTTKRP+GEMM: ", timevalue);
  if (statfile.is_open()) {
    statfile << "fused mttkrp+gemm time: ";
    statfile << timevalue.mean << std::endl;
  } else { std::cout << " stat file is not open\n"; }

  if (statfile.is_open()) {
    statfile.close();
  }

  double* A_vals = (double*) (A.getTacoTensorT()->vals);
  for (int q=0; q < B.getDimension(0)* mDim; q++) {
    if ( abs(A_vals[q] - ref_vals[q])/abs(ref_vals[q]) > ERROR_MARGIN) {
      std::cout << "error: results don't match i: " << q << ", avals: " << A_vals[q] << " "
        << "refvals: " << ref_vals[q] << std::endl;
      ASSERT_TRUE(false);
    }
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
  int NUM_M = 1024;
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
  write("/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/synthetic.tns", B);

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

  int retval, EventSet = PAPI_NULL;
  retval = PAPI_hl_region_begin("dummy");
  if ( retval != PAPI_OK ) handle_error(1);

  retval = PAPI_hl_region_end("dummy");
  if ( retval != PAPI_OK ) handle_error(1);

  taco_set_num_threads(NUM_THREADS_TO_USE);

  ofstream statfile;
  statfile.open(
    "/home/min/a/kadhitha/workspace/my_taco/taco/test/stats/ttm-ttm.txt", std::ios::app);
  if (statfile.is_open()) {
    statfile << "\nttm-ttm execution\n";
    statfile << "\n-----------------------------------------\n";
  }

  std::default_random_engine gen(0);
  std::uniform_real_distribution<double> unif(0.0, 1.0);
  Format csf({Dense,Sparse,Sparse});
  Format custom({Dense,Sparse,Dense});
  Format rm({Dense,Dense});
  int ldim = 32;
  int mdim = 64;

  int64_t dummy_array_size = 2e6;
  int64_t* dummy_array_to_flush_cache = (int64_t*) malloc(dummy_array_size*sizeof(int64_t));

  vector<int> matfilenums = {5};

  for (auto matfilenum : matfilenums) {

    // int matfilenum = 0;

    

    // Load a sparse order-3 tensor from file (stored in the FROSTT format) and 
    // store it as a compressed sparse fiber tensor. The tensor in this example 
    // can be download from: http://frostt.io/tensors/nell-2/
    std::vector<std::string> matfiles = {
      "/home/min/a/kadhitha/ispc-examples/data/tns/matmul_5-5-5.tns",
      "/home/min/a/kadhitha/ispc-examples/data/tns/delicious-3d.tns",
      "/home/min/a/kadhitha/ispc-examples/data/tns/flickr-3d.tns", // 2
      "/home/min/a/kadhitha/ispc-examples/data/tns/nell-2.tns", // 3
      "/home/min/a/kadhitha/ispc-examples/data/tns/nell-1.tns", // 4
      "/home/min/a/kadhitha/workspace/my_taco/tns/vast-2015-mc1-3d.tns", // 5 
      "/home/min/a/kadhitha/workspace/my_taco/tns/darpa1998.tns", // 6
      "/home/min/a/kadhitha/ispc-examples/data/tns/freebase_music.tns",
      "/home/min/a/kadhitha/ispc-examples/data/tns/freebase_sampled.tns"
    };
    std::vector<std::string> matfilesrw = {
      "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/matmul_5-5-5.tns",
      "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/delicious-3d.tns",
      "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/flickr-3d.tns",
      "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/nell-2.tns",
      "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/nell-1.tns",
      "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/vast-2015-mc1-3d.tns",
      "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/darpa1998.tns",
      "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/freebase_music.tns",
      "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/freebase_sampled.tns"
    };
    statfile << "\nfile: " << matfiles[matfilenum] << std::endl;
    statfile << "----------------------------------------------------------------\n";

    std::string matfile = matfiles[matfilenum];
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

    if (statfile.is_open()) {
      statfile 
        << matfile << std::endl
        << "A(i,j,m) = B(i,j,k) * C(k,l) * D(l,m)" << std::endl
        << "B1_dimension: " << B.getDimension(0) << ", B2_dimension: " << B.getDimension(1) << ", B3_dimension: " << B.getDimension(2) << ", vals: " << B.getStorage().getValues().getSize() << std::endl
        << "C1_dimension: " << C.getDimension(0) << ", C2_dimension: " << C.getDimension(1) << ", vals: " << C.getStorage().getValues().getSize() << std::endl
        << "D1_dimension: " << D.getDimension(0) << ", D2_dimension: " << D.getDimension(1) << ", vals: " << D.getStorage().getValues().getSize() << std::endl
        << std::endl;
    }

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
    // ref1Stmt = ref1Stmt.split(i, i0, i1, 16);
    ref1Stmt = insertTemporaries(ref1Stmt);
    ref1Stmt = parallelizeOuterLoop(ref1Stmt);
    ref1.compile(ref1Stmt);
    ref1.assemble();  

    Tensor<double> ref2({B.getDimension(0), B.getDimension(1), mdim}, custom);
    ref2(i,j,m) = ref1(i,j,l) * D(l,m); // TTM2
    IndexStmt ref2Stmt = makeReductionNotation(ref2.getAssignment());
    ref2Stmt = makeConcreteNotation(ref2Stmt);
    // ref2Stmt = ref2Stmt.split(i, i0, i1, 16);
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
    // ref4Stmt = ref4Stmt
    //   .split(i, i0, i1, 16);
    //   // .split(k, k0, k1, 16)
    //   .split(m, m0, m1, 16)
    //   .reorder({i0, i1, j, m0, k, m1});
    ref4Stmt = insertTemporaries(ref4Stmt);
    ref4Stmt = parallelizeOuterLoop(ref4Stmt);
    ref4.compile(ref4Stmt);
    ref4.assemble();

    std::cout << "compute start\n";
    taco::util::TimeResults timevalue;
    bool time                = true;

    int r = rand();
    for (int64_t i=0; i<dummy_array_size; i++) {
      dummy_array_to_flush_cache[i] = r;
    }

    // TOOL_BENCHMARK_TIMER(ref.compute(), "\n\nReference ISPC: ", timevalue);
    std::string sofile_fused = "/home/min/a/kadhitha/workspace/my_taco/taco/test/kernels/ttm_ttm/fused.so";
    retval = PAPI_hl_region_begin("fusedTTM"); if ( retval != PAPI_OK ) handle_error(1);
    TOOL_BENCHMARK_TIMER(A.compute(statfile, sofile_fused), "\n\nFused TTM->TTM: ", timevalue);
    retval = PAPI_hl_region_end("fusedTTM"); if ( retval != PAPI_OK ) handle_error(1);
    if (statfile.is_open()) {
      statfile << "fused time: ";
      statfile << timevalue.mean << std::endl;
    } else { std::cout << " stat file is not open\n"; }

    r = rand();
    for (int64_t i=0; i<dummy_array_size; i++) {
      dummy_array_to_flush_cache[i] = r;
    }

    statfile << "\nreference impl time \n";

    std::string sofile_original = "/home/min/a/kadhitha/workspace/my_taco/taco/test/kernels/ttm_ttm/ttm_original.so";
    retval = PAPI_hl_region_begin("referenceTTM"); if ( retval != PAPI_OK ) handle_error(1);
    TOOL_BENCHMARK_TIMER(ref.compute(statfile, sofile_original), "\n\nReference TTM->TTM: ", timevalue);
    retval = PAPI_hl_region_end("referenceTTM"); if ( retval != PAPI_OK ) handle_error(1);
    if (statfile.is_open()) {
      statfile << "reference time: ";
      statfile << timevalue.mean << std::endl;
    } else { std::cout << " stat file is not open\n"; }

    r = rand();
    for (int64_t i=0; i<dummy_array_size; i++) {
      dummy_array_to_flush_cache[i] = r;
    }

    std::string sofile_original2 = "/home/min/a/kadhitha/workspace/my_taco/taco/test/kernels/ttm_ttm/ttm_original2.so";
    retval = PAPI_hl_region_begin("ref2TTM"); if ( retval != PAPI_OK ) handle_error(1);
    TOOL_BENCHMARK_TIMER(refn.compute(statfile, sofile_original2), "\n\nReference new TTM->TTM: ", timevalue);
    retval = PAPI_hl_region_end("ref2TTM"); if ( retval != PAPI_OK ) handle_error(1);
    if (statfile.is_open()) {
      statfile << "reference new time: ";
      statfile << timevalue.mean << std::endl;
    } else { std::cout << " stat file is not open\n"; }

    statfile << "\nschedule 1\n";

    r = rand();
    for (int64_t i=0; i<dummy_array_size; i++) {
      dummy_array_to_flush_cache[i] = r;
    }

    std::string sofile_ttm11 = "/home/min/a/kadhitha/workspace/my_taco/taco/test/kernels/ttm_ttm/ttm1_1.so";
    retval = PAPI_hl_region_begin("ttm1_1"); if ( retval != PAPI_OK ) handle_error(1);
    TOOL_BENCHMARK_TIMER(ref1.compute(statfile, sofile_ttm11), "\n\nTTM1: ", timevalue);
    retval = PAPI_hl_region_end("ttm1_1"); if ( retval != PAPI_OK ) handle_error(1);
    if (statfile.is_open()) {
      statfile << "TTM1: ";
      statfile << timevalue.mean << std::endl;
    } else { std::cout << " stat file is not open\n"; }

    r = rand();
    for (int64_t i=0; i<dummy_array_size; i++) {
      dummy_array_to_flush_cache[i] = r;
    }

    std::string sofile_ttm2 = "/home/min/a/kadhitha/workspace/my_taco/taco/test/kernels/ttm_ttm/ttm2.so";
    retval = PAPI_hl_region_begin("ttm2"); if ( retval != PAPI_OK ) handle_error(1);
    TOOL_BENCHMARK_TIMER(ref2.compute(statfile, sofile_ttm2), "\n\nTTM2: ", timevalue);
    retval = PAPI_hl_region_end("ttm2"); if ( retval != PAPI_OK ) handle_error(1);
    if (statfile.is_open()) {
      statfile << "TTM2: ";
      statfile << timevalue.mean << std::endl;
    } else { std::cout << " stat file is not open\n"; }

    r = rand();
    for (int64_t i=0; i<dummy_array_size; i++) {
      dummy_array_to_flush_cache[i] = r;
    }

    statfile << "\nschedule 2\n";

    retval = PAPI_hl_region_begin("gemm"); if ( retval != PAPI_OK ) handle_error(1);
    TOOL_BENCHMARK_TIMER(ref3.compute(statfile), "\n\ndense: ", timevalue);
    retval = PAPI_hl_region_end("gemm"); if ( retval != PAPI_OK ) handle_error(1);
    if (statfile.is_open()) {
      statfile << "dense: ";
      statfile << timevalue.mean << std::endl;
    } else { std::cout << " stat file is not open\n"; }

    r = rand();
    for (int64_t i=0; i<dummy_array_size; i++) {
      dummy_array_to_flush_cache[i] = r;
    }

    std::string sofile_ttm12 = "/home/min/a/kadhitha/workspace/my_taco/taco/test/kernels/ttm_ttm/ttm1_2.so";
    retval = PAPI_hl_region_begin("ttm1_2"); if ( retval != PAPI_OK ) handle_error(1);
    TOOL_BENCHMARK_TIMER(ref4.compute(statfile, sofile_ttm12), "\n\nTTM after dense: ", timevalue);
    retval = PAPI_hl_region_end("ttm1_2"); if ( retval != PAPI_OK ) handle_error(1);
    if (statfile.is_open()) {
      statfile << "TTM after dense: ";
      statfile << timevalue.mean << std::endl;
    } else { std::cout << " stat file is not open\n"; }

    r = rand();
    bool istrue = false;
    for (int64_t i=0; i<dummy_array_size; i++) {
      if (dummy_array_to_flush_cache[i] != r) {
        istrue = true;
      }
    }
    std::cout << "istrue: " << istrue << std::endl;


    double* A_vals = (double*) (A.getTacoTensorT()->vals);
    double* ref_vals = (double*) (ref.getTacoTensorT()->vals);
    double* ref2_vals = (double*) (ref2.getTacoTensorT()->vals);
    double* ref4_vals = (double*) (ref4.getTacoTensorT()->vals);

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

  } // end of forloop

  if (statfile.is_open()) {
    statfile.close();
  }

}




TEST(scheduling_eval, spmmFusedWithSyntheticData) {
  if (should_use_CUDA_codegen() || should_use_ISPC_codegen()) {
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
  if (should_use_CUDA_codegen() || should_use_ISPC_codegen()) {
    return;
  }

  int retval, EventSet = PAPI_NULL;
  retval = PAPI_hl_region_begin("dummy");
  if ( retval != PAPI_OK ) handle_error(1);

  /* Do some computation */

  retval = PAPI_hl_region_end("dummy");
  if ( retval != PAPI_OK ) handle_error(1);

  taco_set_num_threads(NUM_THREADS_TO_USE);

  ofstream statfile;
  statfile.open(
    "/home/min/a/kadhitha/workspace/my_taco/taco/test/stats/spmm-spmm.txt", std::ios::app);
  if (statfile.is_open()) {
    statfile << "\nspmm-spmm execution\n";
    statfile << "\n-----------------------------------------\n";
  }

  std::default_random_engine gen(0);
  std::uniform_real_distribution<double> unif(0.0, 1.0);

  Format csr({dense, sparse});
  Format rm({dense, dense});
  int kdim = 128;
  int ldim = 64;

  // vector<int> filenums = {2,3,4,5,6,7,8,9,10,12,15};
  vector<int> filenums = {3};

  for (auto filenum : filenums) {


    statfile << "filenum: " << filenum << std::endl;
    statfile << "---------------------------------\n";
    // int filenum = 7;

    std::vector<std::string> matfiles = {
      "/home/min/a/kadhitha/ispc-examples/data/suitesparse/synthetic/synthetic.mtx",
      "/home/min/a/kadhitha/ispc-examples/data/suitesparse/cage3/cage3.mtx",
      "/home/min/a/kadhitha/ispc-examples/data/suitesparse/bcsstk17/bcsstk17.mtx", // 2
      "/home/min/a/kadhitha/ispc-examples/data/suitesparse/pdb1HYS/pdb1HYS.mtx",
      "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rma10/rma10.mtx",
      "/home/min/a/kadhitha/ispc-examples/data/suitesparse/cant/cant.mtx",
      "/home/min/a/kadhitha/ispc-examples/data/suitesparse/consph/consph.mtx",
      "/home/min/a/kadhitha/ispc-examples/data/suitesparse/cop20k_A/cop20k_A.mtx",
      "/home/min/a/kadhitha/ispc-examples/data/suitesparse/shipsec1/shipsec1.mtx",
      "/home/min/a/kadhitha/ispc-examples/data/suitesparse/scircuit/scircuit.mtx",
      "/home/min/a/kadhitha/ispc-examples/data/suitesparse/mac_econ_fwd500/mac_econ_fwd500.mtx", // 10
      "/home/min/a/kadhitha/ispc-examples/data/suitesparse/wtk/pwtk.mtx",
      "/home/min/a/kadhitha/ispc-examples/data/ufl/webbase-1M/webbase-1M.mtx", // 12
      "/home/min/a/kadhitha/ispc-examples/data/suitesparse/wiki-Talk/wiki-Talk.mtx",
      "/home/min/a/kadhitha/ispc-examples/data/suitesparse/com-Orkut/com-Orkut.mtx",
      "/home/min/a/kadhitha/ispc-examples/data/suitesparse/circuit5M/circuit5M.mtx", // 15
      "/home/min/a/kadhitha/workspace/my_taco/FusedMM/dataset/harvard.mtx",
      "/home/min/a/kadhitha/ispc-examples/data/suitesparse/twitter7/twitter7.mtx",
      "/home/min/a/kadhitha/ispc-examples/data/suitesparse/cop20k_A/cop20k.mtx",
    };
    std::vector<std::string> matfilesrw = {
      "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/synthetic.mtx",
      "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/cage3.mtx",
      "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/bcsstk17.mtx",
      "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/pdb1HYS.mtx",
      "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/rma10.mtx",
      "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/cant.mtx",
      "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/consph.mtx",
      "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/cop20k_A.mtx",
      "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/shipsec1.mtx",
      "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/scircuit.mtx",
      "/home/min/a/kadhitha/ispc-examples/data/suitesparse/mac_econ_fwd500/mac_econ_fwd500.mtx",
      "/home/min/a/kadhitha/ispc-examples/data/suitesparse/wtk/pwtk.mtx",
      "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/webbase-1M.mtx",
      "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/wiki-Talk.mtx",
      "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/com-Orkut.mtx",
      "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/circuit5M.mtx",
      "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/harvard.mtx",
      "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/twitter7.mtx"
    };

    std::string matfile = matfiles[filenum];
    std::cout << "reading B mat mtx\n";
    Tensor<double> B = read(matfile, csr);
    B.pack();
    // write(matfilesrw[filenum], B);

    if (statfile.is_open()) {
      statfile << matfile << std::endl;
    }

    std::cout << "B dim0: " << B.getDimension(0) << ", dim1: " << B.getDimension(1) << std::endl;
    std::cout << "adding c mat\n";
    // Tensor<double> C = read(matfiles2[filenum], csr, true);
    // std::cout << "packing C mat\n";

    std::cout << "B dim0: " << B.getDimension(0) << ", dim1: " << B.getDimension(1) << std::endl;
    std::cout << "adding c mat\n";
    Tensor<double> C("C", {B.getDimension(1), kdim}, rm);
    for (int i = 0; i < C.getDimension(0); ++i) {
      for (int j = 0; j < C.getDimension(1); ++j) {
        C.insert({i,j}, unif(gen));
      }
    }
    std::cout << "packing C mat\n";
    C.pack();

    Tensor<double> D({C.getDimension(1), ldim}, rm);
    for (int i = 0; i < D.getDimension(0); ++i) {
      for (int j = 0; j < D.getDimension(1); ++j) {
        D.insert({i,j}, unif(gen));
      }
    }
    std::cout << "packing D mat\n";
    D.pack();

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
    Tensor<double> refn({B.getDimension(0), ldim}, rm);
    IndexVar i, j, k, l;
    IndexVar i0, i1, j0, j1, k0, k1, l0, l1;

    A(i,l)=B(i,j)*C(j,k)*D(k,l);
    if (statfile.is_open()) {
      statfile 
        << "ref(i,l)=B(i,j)*C(i,k)*D(j,k);" << std::endl
        << "B1_dimension: " << B.getDimension(0) << ", B2_dimension: " << B.getDimension(1) << ", vals: " << B.getStorage().getValues().getSize() << std::endl
        << "C1_dimension: " << C.getDimension(0) << ", C2_dimension: " << C.getDimension(1) << ", vals: " << C.getStorage().getValues().getSize() << std::endl
        << "D1_dimension: " << D.getDimension(0) << ", D2_dimension: " << D.getDimension(1) << ", vals: " << D.getStorage().getValues().getSize() << std::endl
        // << "E1_dimension: " << F.getDimension(0) << ", E2_dimension: " << F.getDimension(1) << ", vals: " << F.getStorage().getValues().getSize() << std::endl
        << std::endl;
    }

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
    Tensor<double> ref2({B.getDimension(0), ldim}, rm);
    Tensor<double> ref2_2({B.getDimension(0), ldim}, rm);
    
    ref1(i,k)=B(i,j)*C(j,k);
    ref2(i,l)=ref1(i,k)*D(k,l);
    ref2_2(i,l)=ref1(i,k)*D(k,l);

    IndexStmt ref1Stmt = makeReductionNotation(ref1.getAssignment());
    ref1Stmt = makeConcreteNotation(ref1Stmt);
    ref1Stmt = insertTemporaries(ref1Stmt);
    ref1Stmt = parallelizeOuterLoop(ref1Stmt);
    ref1.compile(ref1Stmt);
    ref1.assemble();

    IndexStmt ref2Stmt = makeReductionNotation(ref2.getAssignment());
    ref2Stmt = makeConcreteNotation(ref2Stmt);
    ref2Stmt = insertTemporaries(ref2Stmt);
    ref2Stmt = ref2Stmt.split(i, i0, i1, 16);
    ref2Stmt = parallelizeOuterLoop(ref2Stmt);
    ref2.compile(ref2Stmt);
    ref2.assemble();

    IndexStmt ref2Stmt2 = makeReductionNotation(ref2_2.getAssignment());
    ref2Stmt2 = makeConcreteNotation(ref2Stmt2);
    ref2Stmt2 = ref2Stmt2
      .split(i, i0, i1, 32)
      .split(k,k0,k1, 32)
      .split(l, l0, l1, 32)
      .reorder({i0, k0, l0, i1, k1, l1})
      .parallelize(j0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
    ref2Stmt2 = insertTemporaries(ref2Stmt2);
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
    ref4Stmt = ref4Stmt.split(i, i0, i1, 16);
    ref4Stmt = insertTemporaries(ref4Stmt);
    ref4Stmt = parallelizeOuterLoop(ref4Stmt);
    ref4.compile(ref4Stmt);
    ref4.assemble();


    std::cout << "compute start\n";
    taco::util::TimeResults timevalue;
    bool time                = true;

    statfile << "\n--------- 1st pattern computation TTM, GEMM\n";
    
    retval = PAPI_hl_region_begin("spmm");
    if ( retval != PAPI_OK ) handle_error(1);
    TOOL_BENCHMARK_TIMER(ref1.compute(statfile), "\n\nSpMM Kernel: ", timevalue);
    retval = PAPI_hl_region_end("spmm");
    if ( retval != PAPI_OK ) handle_error(1);
    if (statfile.is_open()) {
      statfile << "SpMM time: ";
      statfile << timevalue.mean << std::endl;
    } else { std::cout << " stat file is not open\n"; }

    std::string sofile_spmm_template = "/home/min/a/kadhitha/workspace/my_taco/taco/test/kernels/sddmm_spmm/csr_dense_spmm.so";
    retval = PAPI_hl_region_begin("spmmtemplate");
    if ( retval != PAPI_OK ) handle_error(1);   
    TOOL_BENCHMARK_TIMER(ref1.compute(statfile, sofile_spmm_template), "\n\nSpMM template Kernel: ", timevalue);
    retval = PAPI_hl_region_end("spmmtemplate");
    if ( retval != PAPI_OK ) handle_error(1);
    if (statfile.is_open()) {
      statfile << "SpMM template time: ";
      statfile << timevalue.mean << std::endl;
    } else { std::cout << " stat file is not open\n"; }
    
    retval = PAPI_hl_region_begin("gemm");
    if ( retval != PAPI_OK ) handle_error(1); 
    TOOL_BENCHMARK_TIMER(ref2.compute(statfile), "\n\nGeMM Kernel: ", timevalue);
    retval = PAPI_hl_region_end("gemm");
    if ( retval != PAPI_OK ) handle_error(1);
    if (statfile.is_open()) {
      statfile << "GeMM time: ";
      statfile << timevalue.mean << std::endl;
    } else { std::cout << " stat file is not open\n"; }

    retval = PAPI_hl_region_begin("gemmtemplate");
    if ( retval != PAPI_OK ) handle_error(1);
    TOOL_BENCHMARK_TIMER(ref2_2.compute(statfile), "\n\nref GeMM template Kernel: ", timevalue);
    retval = PAPI_hl_region_end("gemmtemplate");
    if ( retval != PAPI_OK ) handle_error(1);    
    if (statfile.is_open()) {
      statfile << "ref 2 GeMM template time: ";
      statfile << timevalue.mean << std::endl;
    } else { std::cout << " stat file is not open\n"; }

    // std::string sofile_gemm_template = "/home/min/a/kadhitha/workspace/my_taco/taco/test/kernels/spmm_gemm/spmm_template.so";
    statfile << "\n--------- 2nd pattern computation GEMM, SpMM\n";
    retval = PAPI_hl_region_begin("gemmtemplate2");
    if ( retval != PAPI_OK ) handle_error(1);
    TOOL_BENCHMARK_TIMER(ref3.compute(statfile), "\n\nGeMM template ref3 Kernel: ", timevalue);
    retval = PAPI_hl_region_end("gemmtemplate2");
    if ( retval != PAPI_OK ) handle_error(1);  
    if (statfile.is_open()) {
      statfile << "ref3 GeMM template time: ";
      statfile << timevalue.mean << std::endl;
    } else { std::cout << " stat file is not open\n"; }

    retval = PAPI_hl_region_begin("spmm2");
    if ( retval != PAPI_OK ) handle_error(1);
    TOOL_BENCHMARK_TIMER(ref4.compute(statfile, sofile_spmm_template), "\n\nSpMM template Kernel ref4: ", timevalue);
    retval = PAPI_hl_region_end("spmm2");
    if ( retval != PAPI_OK ) handle_error(1);  
    if (statfile.is_open()) {
      statfile << "SpMM template time ref4: ";
      statfile << timevalue.mean << std::endl;
    } else { std::cout << " stat file is not open\n"; }


    statfile << "\n-------- reference pattern computation\n";

    retval = PAPI_hl_region_begin("ref");
    if ( retval != PAPI_OK ) handle_error(1);
    TOOL_BENCHMARK_TIMER(ref.compute(statfile), "\n\nReference Kernel: ", timevalue);
    retval = PAPI_hl_region_end("ref");
    if ( retval != PAPI_OK ) handle_error(1);     
    if (statfile.is_open()) {
      statfile << "taco reference time: ";
      statfile << timevalue << std::endl;
    } else { std::cout << " stat file is not open\n"; }

    retval = PAPI_hl_region_begin("refnew");
    if ( retval != PAPI_OK ) handle_error(1);
    TOOL_BENCHMARK_TIMER(refn.compute(statfile), "\n\nReference new Kernel: ", timevalue);
    retval = PAPI_hl_region_end("refnew");
    if ( retval != PAPI_OK ) handle_error(1);     
    if (statfile.is_open()) {
      statfile << "taco reference new time: ";
      statfile << timevalue << std::endl;
    } else { std::cout << " stat file is not open\n"; }


    retval = PAPI_hl_region_begin("sparselnr");
    if ( retval != PAPI_OK ) handle_error(1);
    TOOL_BENCHMARK_TIMER(A.compute(statfile), "\n\nFused Kernel: ", timevalue);
    retval = PAPI_hl_region_end("sparselnr");
    if ( retval != PAPI_OK ) handle_error(1);
    if (statfile.is_open()) {
      statfile << "fused time: ";
      statfile << timevalue.mean << std::endl;
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

  } // end of file num for loop

  if (statfile.is_open()) {
    statfile.close();
  }

  
  // unsigned int native = 0x0;

  // retval = PAPI_library_init(PAPI_VER_CURRENT);

  // if (retval != PAPI_VER_CURRENT) {
  //   printf("PAPI library init error!\n");
  //   exit(1);
  // } else {
  //   printf("PAPI library init success\n");
  // }

  // if (PAPI_create_eventset(&EventSet) != PAPI_OK) {
  //   handle_error(1);
  // }

  // /* Add the native event */
  // native = ()

    retval = PAPI_hl_region_begin("computation1");
    if ( retval != PAPI_OK )
        handle_error(1);

    /* Do some computation */

    retval = PAPI_hl_region_end("computation1");
    if ( retval != PAPI_OK )
        handle_error(1);

    retval = PAPI_hl_region_begin("computation2");
    if ( retval != PAPI_OK )
        handle_error(1);

    /* Do some computation */

    retval = PAPI_hl_region_end("computation2");
    if ( retval != PAPI_OK )
        handle_error(1);
}






TEST(scheduling_eval, sddmmspmmFused) {
  if (should_use_CUDA_codegen() || should_use_ISPC_codegen()) {
    return;
  }

  taco_set_num_threads(NUM_THREADS_TO_USE);

  ofstream statfile;
  statfile.open(
    "/home/min/a/kadhitha/workspace/my_taco/taco/test/stats/sddmm-spmm-gemm.txt", std::ios::app);
  if (statfile.is_open()) {
    statfile << "\nsddmm-spmm-gemm execution\n";
    statfile << "\n-----------------------------------------\n";
  }

  std::default_random_engine gen(0);
  std::uniform_real_distribution<double> unif(0.0, 1.0);

  Format csr({dense, sparse});
  Format rm({dense, dense});

  int kdim = 64;
  int ldim = 64;
  int mdim = 64;

  vector<int> filenums{2, 3,4,5,6,7,8,9,10,12,15};

  for (auto filenum : filenums) {


  std::vector<std::string> matfiles = {
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/synthetic/synthetic.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/cage3/cage3.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/bcsstk17/bcsstk17.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/pdb1HYS/pdb1HYS.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rma10/rma10.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/cant/cant.mtx", // 5
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/consph/consph.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/cop20k_A/cop20k_A.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/shipsec1/shipsec1.mtx", // 8
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/scircuit/scircuit.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/mac_econ_fwd500/mac_econ_fwd500.mtx", // 10
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/wtk/pwtk.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/ufl/webbase-1M/webbase-1M.mtx", // 12
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/wiki-Talk/wiki-Talk.mtx", // 13
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/com-Orkut/com-Orkut.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/circuit5M/circuit5M.mtx", // 15
    "/home/min/a/kadhitha/workspace/my_taco/FusedMM/dataset/harvard.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/twitter7/twitter7.mtx"
  };
  std::vector<std::string> matfilesrw = {
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/synthetic.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/cage3.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/bcsstk17.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/pdb1HYS.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/rma10.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/cant.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/consph.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/cop20k_A.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/shipsec1.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/scircuit.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/mac_econ_fwd500/mac_econ_fwd500.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/wtk/pwtk.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/webbase-1M.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/wiki-Talk.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/com-Orkut.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/circuit5M.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/harvard.mtx",
    "/home/min/a/kadhitha/ispc-examples/data/suitesparse/rw/twitter7.mtx"
  };

  std::string matfile = matfiles[filenum];
  std::cout << "reading B mat mtx\n";
  Tensor<double> B = read(matfile, csr, true);
  B.setName("B");
  B.pack();
  // write(matfilesrw[filenum], B);

  if (statfile.is_open()) {
    statfile << matfile << std::endl;
  }

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

  Tensor<double> G({ldim, mdim}, rm);
  for (int i = 0; i < G.getDimension(0); ++i) {
    for (int j = 0; j < G.getDimension(1); ++j) {
      G.insert({i,j}, unif(gen));
    }
  }
  std::cout << "packing F mat\n";
  G.pack();

  Tensor<double> A({B.getDimension(0), mdim}, rm);
  Tensor<double> ref({B.getDimension(0), mdim}, rm);
  IndexVar i, j, k, l, m;
  IndexVar i0("i0"), i1("i1"), jpos("jpos"), jpos0("jpos0"), jpos1("jpos1"), k0("k0"), k1("k1");
  IndexVar l0("l0"), l1("l1"), m0("m0"), m1("m1");
  
  A(i,m)=B(i,j)*C(i,k)*D(j,k)*F(j,l)*G(l,m);
  
  if (statfile.is_open()) {
    statfile 
      << "ref(i,m)=B(i,j)*C(i,k)*D(j,k)*F(j,l)*G(l,m);" << std::endl
      << "B1_dimension: " << B.getDimension(0) << ", B2_dimension: " << B.getDimension(1) << ", vals: " << B.getStorage().getValues().getSize() << std::endl
      << "C1_dimension: " << C.getDimension(0) << ", C2_dimension: " << C.getDimension(1) << ", vals: " << C.getStorage().getValues().getSize() << std::endl
      << "D1_dimension: " << D.getDimension(0) << ", D2_dimension: " << D.getDimension(1) << ", vals: " << D.getStorage().getValues().getSize() << std::endl
      << "E1_dimension: " << F.getDimension(0) << ", E2_dimension: " << F.getDimension(1) << ", vals: " << F.getStorage().getValues().getSize() << std::endl
      << "G1_dimension: " << F.getDimension(0) << ", G2_dimension: " << G.getDimension(1) << ", vals: " << G.getStorage().getValues().getSize() << std::endl
      << std::endl;
  }

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

  IndexStmt ref2Stmt = makeReductionNotation(ref2.getAssignment());
  ref2Stmt = makeConcreteNotation(ref2Stmt);
  ref2Stmt = insertTemporaries(ref2Stmt);
  ref2Stmt = parallelizeOuterLoop(ref2Stmt);
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
  bool time                = true;
  
  // std::string sofile_fused = "/home/min/a/kadhitha/workspace/my_taco/taco/test/kernels/sddmm_spmm/fused_kernel.so";
  TOOL_BENCHMARK_TIMER(A.compute(statfile), "\n\nFused Kernel: ", timevalue);
  if (statfile.is_open()) {
    statfile << "fused time: ";
    statfile << timevalue.mean << std::endl;
  } else { std::cout << " stat file is not open\n"; }
  
  // std::string sofile_sddmm = "/home/min/a/kadhitha/workspace/my_taco/taco/test/kernels/sddmm_spmm/csr_dense_spmm.so";
  std::string sofile_sddmm = "/home/min/a/kadhitha/workspace/my_taco/taco/test/kernels/sddmm_spmm/csr_dense_dense_sddmm.so";
  TOOL_BENCHMARK_TIMER(ref1.compute(statfile, sofile_sddmm), "\n\nSDDMM Kernel: ", timevalue);
  if (statfile.is_open()) {
    statfile << "sddmm time: ";
    statfile << timevalue.mean << std::endl;
  } else { std::cout << " stat file is not open\n"; }

  std::string sofile_sddmm_ryan = "/home/min/a/kadhitha/workspace/my_taco/taco/test/kernels/sddmm_spmm/sddmm_ryan.so";
  TOOL_BENCHMARK_TIMER(ref1.compute(statfile, sofile_sddmm_ryan), "\n\nSDDMM ryan Kernel: ", timevalue);
  if (statfile.is_open()) {
    statfile << "sddmm ryan time: ";
    statfile << timevalue.mean << std::endl;
  } else { std::cout << " stat file is not open\n"; }
  
  std::string sofile_spmm = "/home/min/a/kadhitha/workspace/my_taco/taco/test/kernels/sddmm_spmm/csr_dense_spmm.so";
  TOOL_BENCHMARK_TIMER(ref2.compute(statfile, sofile_spmm), "\n\nSpMM ryan Kernel: ", timevalue);
  if (statfile.is_open()) {
    statfile << "spmm ryan time: ";
    statfile << timevalue.mean << std::endl;
  } else { std::cout << " stat file is not open\n"; }

  // std::string sofile_spmm = "/home/min/a/kadhitha/workspace/my_taco/taco/test/kernels/sddmm_spmm/csr_dense_spmm.so";
  TOOL_BENCHMARK_TIMER(ref3.compute(statfile), "\n\nGeMM Kernel: ", timevalue);
  if (statfile.is_open()) {
    statfile << "gemm time: ";
    statfile << timevalue.mean << std::endl;
  } else { std::cout << " stat file is not open\n"; }

  // std::string sofile_original = "/home/min/a/kadhitha/workspace/my_taco/taco/test/kernels/sddmm_spmm/taco_original.so";
  TOOL_BENCHMARK_TIMER(ref.compute(statfile), "\n\nReference Kernel: ", timevalue);
  if (statfile.is_open()) {
    statfile << "taco reference time: ";
    statfile << timevalue << std::endl;
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



  }

  // int filenum = 3;

  
  // for (size_t q=0; q < A.getDimension(0)* A.getDimension(1); q++) {
  //   if ( abs(A_vals[q] - ref3_vals[q])/abs(ref3_vals[q]) > ERROR_MARGIN) {
  //     std::cout << "error: results don't match i: " << q << ", avals: " << A_vals[q] << " "
  //       << "refvals: " << ref3_vals[q] << std::endl;
  //     ASSERT_TRUE(false);
  //   }
  // }
  // for (int q= 0; q< A_vals
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