#include <bits/types/clock_t.h>
#include <cstdio>
#include <taco/index_notation/transformations.h>
#include <codegen/codegen_c.h>
#include <codegen/codegen_cuda.h>
#include <fstream>
#include "test.h"
#include "test_tensors.h"
#include "taco/tensor.h"
#include "taco/index_notation/index_notation.h"
#include "codegen/codegen.h"
#include "taco/lower/lower.h"
#include "taco/util/env.h"
#include "time.h"
#include "omp.h"

using namespace taco;

void printCodeToFile(string filename, IndexStmt stmt) {
  stringstream source;

  string file_path = "eval_generated/";
  mkdir(file_path.c_str(), 0777);

  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
  ir::Stmt compute = lower(stmt, "compute",  true, true);
  codegen->compile(compute, true);

  ofstream source_file;
  string file_ending = should_use_CUDA_codegen() ? ".cu" : ".c";
  source_file.open(file_path + filename + file_ending);
  source_file << source.str();
  source_file.close();
}

TEST(workspaces, tile_vecElemMul_NoTail) {
  
  Tensor<double> A("A", {16}, Format{Dense});
  Tensor<double> B("B", {16}, Format{Dense});
  Tensor<double> C("C", {16}, Format{Dense});

  for (int i = 0; i < 16; i++) {
      A.insert({i}, (double) i);
      B.insert({i}, (double) i);
  }

  A.pack();
  B.pack();

  IndexVar i("i");
  IndexVar i_bounded("i_bounded");
  IndexVar i0("i0"), i1("i1");
  IndexExpr precomputedExpr = B(i) * C(i);
  A(i) = precomputedExpr;

  IndexStmt stmt = A.getAssignment().concretize();
  TensorVar precomputed("precomputed", Type(Float64, {Dimension(i1)}), taco::dense);
  stmt = stmt.bound(i, i_bounded, 16, BoundType::MaxExact)
             .split(i_bounded, i0, i1, 4)
             .precompute(precomputedExpr, i1, i1, precomputed);
   
  printCodeToFile("tile_vecElemMul_NoTail", stmt);
  A.compile(stmt);
  A.assemble();
  A.compute();

  Tensor<double> expected("expected", {16}, Format{Dense});
  expected(i) = B(i) * C(i);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, A);
}

TEST(workspaces, tile_vecElemMul_Tail1) {
  
  Tensor<double> A("A", {16}, Format{Dense});
  Tensor<double> B("B", {16}, Format{Dense});
  Tensor<double> C("C", {16}, Format{Dense});

  for (int i = 0; i < 16; i++) {
      A.insert({i}, (double) i);
      B.insert({i}, (double) i);
  }

  A.pack();
  B.pack();

  IndexVar i("i");
  IndexVar i_bounded("i_bounded");
  IndexVar i0("i0"), i1("i1");
  IndexExpr precomputedExpr = B(i) * C(i);
  A(i) = precomputedExpr;

  IndexStmt stmt = A.getAssignment().concretize();
  TensorVar precomputed("precomputed", Type(Float64, {Dimension(i1)}), taco::dense);
  stmt = stmt.bound(i, i_bounded, 16, BoundType::MaxExact)
             .split(i_bounded, i0, i1, 5)
             .precompute(precomputedExpr, i1, i1, precomputed);
  printCodeToFile("tile_vecElemMul_Tail1", stmt);
   
  A.compile(stmt.concretize());
  A.assemble();
  A.compute();

  Tensor<double> expected("expected", {16}, Format{Dense});
  expected(i) = B(i) * C(i);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, A);
}

TEST(workspaces, tile_vecElemMul_Tail2) {
  
  Tensor<double> A("A", {17}, Format{Dense});
  Tensor<double> B("B", {17}, Format{Dense});
  Tensor<double> C("C", {17}, Format{Dense});

  for (int i = 0; i < 17; i++) {
      A.insert({i}, (double) i);
      B.insert({i}, (double) i);
  }

  A.pack();
  B.pack();

  IndexVar i("i");
  IndexVar i_bounded("i_bounded");
  IndexVar i0("i0"), i1("i1");
  IndexExpr precomputedExpr = B(i) * C(i);
  A(i) = precomputedExpr;

  IndexStmt stmt = A.getAssignment().concretize();
  TensorVar precomputed("precomputed", Type(Float64, {Dimension(i1)}), taco::dense);
  stmt = stmt.bound(i, i_bounded, 17, BoundType::MaxExact)
             .split(i_bounded, i0, i1, 4)
             .precompute(precomputedExpr, i1, i1, precomputed);
  printCodeToFile("tile_vecElemMul_Tail2", stmt);           
   
  A.compile(stmt.concretize());
  A.assemble();
  A.compute();

  Tensor<double> expected("expected", {17}, Format{Dense});
  expected(i) = B(i) * C(i);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, A);

//  ir::IRPrinter irp = ir::IRPrinter(cout);
//    
//  cout << stmt << endl;
//
//  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
//  ir::Stmt compute = lower(stmt, "compute",  false, true);
//  
//  irp.print(compute);
//  cout << endl;
//  codegen->compile(compute, false);
}

TEST(workspaces, tile_denseMatMul) {
  
  Tensor<double> A("A", {16}, Format{Dense});
  Tensor<double> B("B", {16}, Format{Dense});
  Tensor<double> C("C", {16}, Format{Dense});

  for (int i = 0; i < 16; i++) {
      B.insert({i}, (double) i);
      C.insert({i}, (double) i);
  }

  A.pack();
  B.pack();

  IndexVar i("i");
  IndexVar i_bounded("i_bounded");
  IndexVar i0("i0"), i1("i1");
  IndexExpr precomputedExpr = B(i) * C(i);
  A(i) = precomputedExpr;

  IndexStmt stmt = A.getAssignment().concretize();
  TensorVar precomputed("precomputed", Type(Float64, {Dimension(i1)}), taco::dense);
  stmt = stmt.bound(i, i_bounded, 16, BoundType::MaxExact)
             .split(i_bounded, i0, i1, 4);

  stmt = stmt.precompute(precomputedExpr, i1, i1, precomputed);
  printCodeToFile("tile_denseMatMul", stmt); 

  A.compile(stmt.concretize());
  A.assemble();
  A.compute();

  Tensor<double> expected("expected", {16}, Format{Dense});
  expected(i) = B(i) * C(i);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, A);

//  ir::IRPrinter irp = ir::IRPrinter(cout);
//    
//  cout << stmt << endl;
//
//  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
//  ir::Stmt compute = lower(stmt, "compute",  false, true);
//  
//  irp.print(compute);
//  cout << endl;
//  codegen->compile(compute, false);
  
}

TEST(workspaces, precompute2D_add) {
  int N = 16;
  Tensor<double> A("A", {N, N}, Format{Dense, Dense});
  Tensor<double> B("B", {N, N}, Format{Dense, Dense});
  Tensor<double> C("C", {N, N}, Format{Dense, Dense});
  Tensor<double> D("D", {N, N}, Format{Dense, Dense});

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      B.insert({i, j}, (double) i);
      C.insert({i, j}, (double) j);
      D.insert({i, j}, (double) i*j);
    }
  }

  IndexVar i("i"), j("j");
  IndexExpr precomputedExpr = B(i, j) + C(i, j);
  A(i, j) = precomputedExpr + D(i, j);
    
  IndexStmt stmt = A.getAssignment().concretize();
  TensorVar ws("ws", Type(Float64, {(size_t)N, (size_t)N}), Format{Dense, Dense});
  stmt = stmt.precompute(precomputedExpr, {i, j}, {i, j}, ws);

  std::cout << stmt << endl;
  printCodeToFile("precompute2D_ad", stmt);

  A.compile(stmt.concretize());
  A.assemble();
  A.compute();

  Tensor<double> expected("expected", {N, N}, Format{Dense, Dense});
  expected(i, j) = B(i, j) + C(i, j) + D(i, j);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, A);

}

TEST(workspaces, precompute4D_add) {
  int N = 16;
  Tensor<double> A("A", {N, N, N, N}, Format{Dense, Dense, Dense, Dense});
  Tensor<double> B("B", {N, N, N, N}, Format{Dense, Dense, Dense, Dense});
  Tensor<double> C("C", {N, N, N, N}, Format{Dense, Dense, Dense, Dense});
  Tensor<double> D("D", {N, N, N, N}, Format{Dense, Dense, Dense, Dense});

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        for (int l = 0; l < N; l++) {
          B.insert({i, j, k, l}, (double) i + j);
          C.insert({i, j, k, l}, (double) j * k);
          D.insert({i, j, k, l}, (double) k * l);
        }
      }
    }
  }

  IndexVar i("i"), j("j"), k("k"), l("l");
  IndexExpr precomputedExpr = B(i, j, k, l) + C(i, j, k, l);
  A(i, j, k, l) = precomputedExpr + D(i, j, k, l);


  IndexStmt stmt = A.getAssignment().concretize();
  TensorVar ws1("ws1", Type(Float64, {(size_t)N, (size_t)N, (size_t)N, (size_t)N}), 
                Format{Dense, Dense, Dense, Dense});
  TensorVar ws2("ws2", Type(Float64, {(size_t)N, (size_t)N, (size_t)N, (size_t)N}), 
                Format{Dense, Dense, Dense, Dense});
  stmt = stmt.precompute(precomputedExpr, {i, j, k, l}, {i, j, k, l}, ws1)
    .precompute(ws1(i, j, k, l) + D(i, j, k, l), {i, j, k, l}, {i, j, k ,l}, ws2);
  std::cout << stmt << endl;
  printCodeToFile("precompute4D_add", stmt);

  A.compile(stmt.concretize());
  A.assemble();
  A.compute();

  Tensor<double> expected("expected", {N, N, N, N}, Format{Dense, Dense, Dense, Dense});
  expected(i, j, k, l) = B(i, j, k, l) + C(i, j, k, l) + D(i, j, k, l);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, A);
}

TEST(workspaces, precompute4D_multireduce) {
  int N = 16;
  Tensor<double> A("A", {N, N}, Format{Dense, Dense});
  Tensor<double> B("B", {N, N, N, N}, Format{Dense, Dense, Dense, Dense});
  Tensor<double> C("C", {N, N, N}, Format{Dense, Dense, Dense});
  Tensor<double> D("D", {N, N}, Format{Dense, Dense});

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        for (int l = 0; l < N; l++) {
          B.insert({i, j, k, l}, (double) k*l);
          C.insert({i, j, k}, (double) j * k);
          D.insert({i, j}, (double) i+j);
        }
      }
    }
  }

  IndexVar i("i"), j("j"), k("k"), l("l"), m("m"), n("n");
  IndexExpr precomputedExpr = B(i, j, k, l) * C(k, l, m);
  A(i, j) = precomputedExpr * D(m, n);


  IndexStmt stmt = A.getAssignment().concretize();
  TensorVar ws1("ws1", Type(Float64, {(size_t)N, (size_t)N, (size_t)N}), Format{Dense, Dense, Dense});
  TensorVar ws2("ws2", Type(Float64, {(size_t)N, (size_t)N}), Format{Dense, Dense});
  stmt = stmt.precompute(precomputedExpr, {i, j, m}, {i, j, m}, ws1)
    .precompute(ws1(i, j, m) * D(m, n), {i, j}, {i, j}, ws2);
  
  std::cout << stmt << endl;
  printCodeToFile("precompute4D_multireduce", stmt);

  A.compile(stmt.concretize());
  A.assemble();
  A.compute();

  Tensor<double> expected("expected", {N, N}, Format{Dense, Dense});
  expected(i, j) = B(i, j, k, l) * C(k, l, m) * D(m, n);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, A);
}

TEST(workspaces, precompute3D_TspV) {
  int N = 16;
  Tensor<double> A("A", {N, N}, Format{Dense, Dense});
  Tensor<double> B("B", {N, N, N, N}, Format{Dense, Dense, Dense, Dense});
  Tensor<double> c("c", {N}, Format{Sparse});

  for (int i = 0; i < N; i++) {
    c.insert({i}, (double) i);
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        for (int l = 0; l < N; l++) {
          B.insert({i, j, k, l}, (double) i + j);
        }
      }
    }
  }

  IndexVar i("i"), j("j"), k("k"), l("l");
  IndexExpr precomputedExpr = B(i, j, k, l) * c(l);
  A(i, j) = precomputedExpr * c(k);


  IndexStmt stmt = A.getAssignment().concretize();
  TensorVar ws("ws", Type(Float64, {(size_t)N, (size_t)N, (size_t)N}), Format{Dense, Dense, Dense});
  stmt = stmt.precompute(precomputedExpr, {i, j, k}, {i, j, k}, ws);
  stmt = stmt.concretize();

  std::cout << stmt << endl;
  printCodeToFile("precompute3D_TspV", stmt);

  A.compile(stmt);
  A.assemble();
  A.compute();

  Tensor<double> expected("expected", {N, N}, Format{Dense, Dense});
  expected(i, j) = (B(i, j, k, l) * c(l)) * c(k);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, A);

}

TEST(workspaces, precompute3D_multipleWS) {
  int N = 16;
  Tensor<double> A("A", {N, N}, Format{Dense, Dense});
  Tensor<double> B("B", {N, N, N, N}, Format{Dense, Dense, Dense, Dense});
  Tensor<double> c("c", {N}, Format{Sparse});

  for (int i = 0; i < N; i++) {
    c.insert({i}, (double) i);
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        for (int l = 0; l < N; l++) {
          B.insert({i, j, k, l}, (double) i + j);
        }
      }
    }
  }

  IndexVar i("i"), j("j"), k("k"), l("l");
  IndexExpr precomputedExpr = B(i, j, k, l) * c(l);
  IndexExpr precomputedExpr2 = precomputedExpr * c(k);
  A(i, j) = precomputedExpr2;


  IndexStmt stmt = A.getAssignment().concretize();
  TensorVar ws("ws", Type(Float64, {(size_t)N, (size_t)N, (size_t)N}), Format{Dense, Dense, Dense});
  TensorVar t("t", Type(Float64, {(size_t) N, (size_t)N}), Format{Dense, Dense});
  stmt = stmt.precompute(precomputedExpr, {i, j, k}, {i, j, k}, ws);

  stmt = stmt.precompute(ws(i, j, k) * c(k), {i, j}, {i, j}, t);
  stmt = stmt.concretize();

  std::cout << stmt << endl;
  printCodeToFile("precompute3D_multipleWS", stmt);

  A.compile(stmt);
  A.assemble();
  A.compute();

  Tensor<double> expected("expected", {N, N}, Format{Dense, Dense});
  expected(i, j) = (B(i, j, k, l) * c(l)) * c(k);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, A);

}

TEST(workspaces, precompute3D_renamedIVars_TspV) {
  int N = 16;
  Tensor<double> A("A", {N, N}, Format{Dense, Dense});
  Tensor<double> B("B", {N, N, N, N}, Format{Dense, Dense, Dense, Dense});
  Tensor<double> c("c", {N}, Format{Sparse});

  for (int i = 0; i < N; i++) {
    c.insert({i}, (double) i);
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        for (int l = 0; l < N; l++) {
          B.insert({i, j, k, l}, (double) i + j);
        }
      }
    }
  }

  IndexVar i("i"), j("j"), k("k"), l("l");
  IndexExpr precomputedExpr = B(i, j, k, l) * c(l);
  A(i, j) = precomputedExpr * c(k);


  IndexStmt stmt = A.getAssignment().concretize();
  TensorVar ws("ws", Type(Float64, {(size_t)N, (size_t)N, (size_t)N}),
               Format{Dense, Dense, Dense});

  IndexVar iw("iw"), jw("jw"), kw("kw");
  stmt = stmt.precompute(precomputedExpr, {i, j, k}, {iw, jw, kw}, ws);
  stmt = stmt.concretize();

  std::cout << stmt << endl;
  printCodeToFile("precompute3D_renamedIVars_TspV", stmt);

  A.compile(stmt);
  A.assemble();
  A.compute();

  Tensor<double> expected("expected", {N, N}, Format{Dense, Dense});
  expected(i, j) = (B(i, j, k, l) * c(l)) * c(k);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, A);

}

TEST(workspaces, tile_dotProduct_1) {
  // Test that precompute algorithm correctly decides the reduction operator of C_new(i1) = C(i) and B_new(i1) = B(i).
  // Current indexStmt is:
  // where(forall(i1, A += precomputed(i1)), forall(i0, where(where(forall(i1, precomputed(i1) += B_new(i1) * C_new(i1))
  // ,forall(i1, C_new(i1) = C(i))), forall(i1, B_new(i1) = B(i)))))

  int N = 1024;
  Tensor<double> A("A");
  Tensor<double> B("B", {N}, Format({Dense}));
  Tensor<double> C("C", {N}, Format({Dense}));

  for (int i = 0; i < N; i++) {
    B.insert({i}, (double) i);
    C.insert({i}, (double) i);
  }

  B.pack();
  C.pack();

  IndexVar i("i");
  IndexVar i_bounded("i_bounded");
  IndexVar i0("i0"), i1("i1");
  IndexExpr BExpr = B(i);
  IndexExpr CExpr = C(i);
  IndexExpr precomputedExpr = (BExpr) * (CExpr);
  A() = precomputedExpr;

  IndexStmt stmt = A.getAssignment().concretize();
  TensorVar B_new("B_new", Type(Float64, {(size_t)N}), taco::dense);
  TensorVar C_new("C_new", Type(Float64, {(size_t)N}), taco::dense);
  TensorVar precomputed("precomputed", Type(Float64, {(size_t)N}), taco::dense);

  stmt = stmt.bound(i, i_bounded, (size_t)N, BoundType::MaxExact)
             .split(i_bounded, i0, i1, 32);
  stmt = stmt.precompute(precomputedExpr, i1, i1, precomputed);
  stmt = stmt.precompute(BExpr, i1, i1, B_new)
    .precompute(CExpr, i1, i1, C_new);
  
  stmt = stmt.concretize();

  std::cout << stmt << endl;
  printCodeToFile("tile_dotProduct_1", stmt);

  A.compile(stmt);
  A.assemble();
  A.compute();

  ir::IRPrinter irp = ir::IRPrinter(cout);
    
  cout << stmt << endl;

  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
  ir::Stmt compute = lower(stmt, "compute",  false, true);
  
  irp.print(compute);
  cout << endl;
  codegen->compile(compute, false);

  Tensor<double> expected("expected");
  expected() = B(i) * C(i);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, A);
}

TEST(workspaces, tile_dotProduct_2) {
  // Split on the ALL INSTANCES of an indexVar.
  // Test the wsaccel function that can disable the acceleration.

  int N = 1024;
  Tensor<double> A("A");
  Tensor<double> B("B", {N}, Format({Dense}));
  Tensor<double> C("C", {N}, Format({Dense}));

  for (int i = 0; i < N; i++) {
    B.insert({i}, (double) i / N);
    C.insert({i}, (double) i / N);
  }

  B.pack();
  C.pack();

  IndexVar i("i");
  IndexVar i_bounded("i_bounded");
  IndexVar i0("i0"), i1("i1");
  IndexExpr BExpr = B(i);
  IndexExpr CExpr = C(i);
  IndexExpr precomputedExpr = (BExpr) * (CExpr);
  A() = precomputedExpr;

  IndexStmt stmt = A.getAssignment().concretize();
  TensorVar B_new("B_new", Type(Float64, {(size_t)N}), taco::dense);
  TensorVar C_new("C_new", Type(Float64, {(size_t)N}), taco::dense);
  TensorVar precomputed("precomputed", Type(Float64, {(size_t)N}), taco::dense);

  stmt = stmt.precompute(precomputedExpr, i, i, precomputed);
    
  stmt = stmt.precompute(BExpr, i, i, B_new) 
          .precompute(CExpr, i, i, C_new);

  stmt = stmt.bound(i, i_bounded, (size_t)N, BoundType::MaxExact)
             .split(i_bounded, i0, i1, 32);

  stmt = stmt.concretize();

  std::cout << stmt << endl;
  printCodeToFile("tile_dotProduct_2", stmt);

  stmt = stmt.wsaccel(precomputed, false);
  A.compile(stmt);
  A.assemble();
  A.compute();

  Tensor<double> expected("expected");
  expected() = B(i) * C(i);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, A);
}

TEST(workspaces, tile_dotProduct_3) {
  int N = 1024;
  Tensor<double> A("A");
  Tensor<double> B("B", {N}, Format({Dense}));
  Tensor<double> C("C", {N}, Format({Dense}));

  for (int i = 0; i < N; i++) {
    B.insert({i}, (double) i);
    C.insert({i}, (double) i);
  }

  B.pack();
  C.pack();

  IndexVar i("i");
  IndexVar i_bounded("i_bounded");
  IndexVar i0("i0"), i1("i1");
  IndexExpr BExpr = B(i);
  IndexExpr CExpr = C(i);
  IndexExpr precomputedExpr = (BExpr) * (CExpr);
  A() = precomputedExpr;

  IndexStmt stmt = A.getAssignment().concretize();
  TensorVar B_new("B_new", Type(Float64, {(size_t)N}), taco::dense);
  TensorVar C_new("C_new", Type(Float64, {(size_t)N}), taco::dense);
  TensorVar precomputed("precomputed", Type(Float64, {(size_t)N}), taco::dense);

  stmt = stmt.bound(i, i_bounded, (size_t)N, BoundType::MaxExact)
    .split(i_bounded, i0, i1, 32);
  stmt = stmt.precompute(precomputedExpr, i0, i0, precomputed);

  stmt = stmt.precompute(BExpr, i1, i1, B_new)
    .precompute(CExpr, i1, i1, C_new);


  stmt = stmt.concretize();

  std::cout << stmt << endl;
  printCodeToFile("tile_dotProduct_3", stmt);

  A.compile(stmt);
  A.assemble();
  A.compute();

  Tensor<double> expected("expected");
  expected() = B(i) * C(i);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, A);
}


TEST(workspaces, loopfuse) {
  int N = 16;
  float SPARSITY = 0.3;
  Tensor<double> A("A", {N, N}, Format{Dense, Dense});
  Tensor<double> B("B", {N, N}, Format{Dense, Sparse});
  Tensor<double> C("C", {N, N}, Format{Dense, Dense});
  Tensor<double> D("D", {N, N}, Format{Dense, Dense});
  Tensor<double> E("E", {N, N}, Format{Dense, Dense});

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      float rand_float = (float) rand() / (float) RAND_MAX;
      if (rand_float < SPARSITY)
        B.insert({i, j}, (double) i);
      C.insert({i, j}, (double) j);
      E.insert({i, j}, (double) i*j);
      D.insert({i, j}, (double) i*j);
    }
  }
  B.pack();

  IndexVar i("i"), j("j"), k("k"), l("l"), m("m");
  A(i,m) = B(i,j) * C(j,k) * D(k,l) * E(l,m);

  IndexStmt stmt = A.getAssignment().concretize();
  // TensorVar ws("ws", Type(Float64, {(size_t)N, (size_t)N}), Format{Dense, Dense});
  // TensorVar t("t", Type(Float64, {(size_t)N, (size_t)N}), Format{Dense, Dense});

  std::cout << stmt << endl;
  vector<int> path0;
  vector<int> path1 = {1};
  vector<int> path2 = {1, 0};
  //
  stmt = stmt
    .reorder({i, l, j, k, m})
    .loopfuse(1, true, path0);

  std::cout << "inter: " << stmt << std::endl;

  stmt = stmt
    .reorder(path1, {l, j})
    .loopfuse(2, false, path1)
    // .loopfuse(1, false, path2)
    ;
  // stmt = stmt
  //   .parallelize(i, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
  //   ;
  // 

  stmt = stmt.concretize();
  cout << "final stmt: " << stmt << endl;
  printCodeToFile("loopfuse", stmt);

  A.compile(stmt);
  A.assemble();

  clock_t begin = clock();
  A.compute(stmt);
  clock_t end = clock();
  double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

  std::cout << "executed\n";

  Tensor<double> expected("expected", {N, N}, Format{Dense, Dense});
  expected(i,m) = B(i,j) * C(j,k) * D(k,l) * E(l,m);
  expected.compile();
  expected.assemble();
  begin = clock();
  expected.compute();
  end = clock();
  double elapsed_secs_ref = double(end - begin) / CLOCKS_PER_SEC;
  ASSERT_TENSOR_EQ(expected, A);

  std::cout << elapsed_secs << std::endl;
  std::cout << elapsed_secs_ref << std::endl;
}

TEST(workspaces, sddmm_spmm) {
  int N = 16;
  float SPARSITY = 0.3;
  Tensor<double> A("A", {N, N}, Format{Dense, Dense});
  Tensor<double> B("B", {N, N}, Format{Dense, Sparse});
  Tensor<double> C("C", {N, N}, Format{Dense, Dense});
  Tensor<double> D("D", {N, N}, Format{Dense, Dense});
  Tensor<double> E("E", {N, N}, Format{Dense, Dense});

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      float rand_float = (float) rand() / (float) RAND_MAX;
      if (rand_float < SPARSITY)
        B.insert({i, j}, (double) i);
      C.insert({i, j}, (double) j);
      E.insert({i, j}, (double) i*j);
      D.insert({i, j}, (double) i*j);
    }
  }
  B.pack();



  // 3 -> A(i,l) = B(i,j) * C(i,k) * D(j,k) * E(j,l) - <SDDMM, SpMM>
  IndexVar i("i"), j("j"), k("k"), l("l");
  A(i,l) = B(i,j) * C(i,k) * D(j,k) * E(j,l);

  IndexStmt stmt = A.getAssignment().concretize();
  // TensorVar ws("ws", Type(Float64, {(size_t)N, (size_t)N}), Format{Dense, Dense});
  // TensorVar t("t", Type(Float64, {(size_t)N, (size_t)N}), Format{Dense, Dense});

  std::cout << "original sddmm_spmm stmt: " << stmt << endl;

	/* BEGIN sddmm_spmm TEST */
	vector<int> path0;
	stmt = stmt
		.reorder({i, j, k, l})
		.loopfuse(3, true, path0)
		;
	/* END sddmm_spmm TEST */

  stmt = stmt.concretize();
  cout << "final stmt: " << stmt << endl;
  printCodeToFile("sddmm_spmm", stmt);

  A.compile(stmt);
  A.assemble();

  Tensor<double> expected("expected", {N, N}, Format{Dense, Dense});
  expected(i,l) = B(i,j) * C(i,k) * D(j,k) * E(j,l);
  IndexStmt exp = makeReductionNotation(expected.getAssignment());
  exp = insertTemporaries(exp);
  exp = exp.concretize();
  expected.compile(exp);
  expected.assemble();

  clock_t begin;
  clock_t end;

  for (int i = 0; i< 10; i++) {
    begin = clock();
    A.compute(stmt);
    end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    begin = clock();
    expected.compute();
    end = clock();
    double elapsed_secs_ref = double(end - begin) / CLOCKS_PER_SEC;
    // ASSERT_TENSOR_EQ(expected, A);

    std::cout << elapsed_secs << std::endl;
    std::cout << elapsed_secs_ref << std::endl;
  }

}

TEST(workspaces, sddmm_spmm2) {
  int N = 16;
  float SPARSITY = 0.3;
  Tensor<double> A("A", {N, N}, Format{Dense, Dense});
  Tensor<double> B("B", {N, N}, Format{Dense, Sparse});
  Tensor<double> C("C", {N, N}, Format{Dense, Dense});
  Tensor<double> D("D", {N, N}, Format{Dense, Dense});
  Tensor<double> E("E", {N, N}, Format{Dense, Dense});

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      float rand_float = (float) rand() / (float) RAND_MAX;
      if (rand_float < SPARSITY)
        B.insert({i, j}, (double) i);
      C.insert({i, j}, (double) j);
      E.insert({i, j}, (double) i*j);
      D.insert({i, j}, (double) i*j);
    }
  }
  B.pack();



  // 3 -> A(i,l) = B(i,j) * C(i,k) * D(j,k) * E(j,l) - <SDDMM, SpMM>
  IndexVar i("i"), j("j"), k("k"), l("l");
  A(i,l) = B(i,j) * C(i,k) * D(j,k) * E(j,l);

  IndexStmt stmt = A.getAssignment().concretize();
  // TensorVar ws("ws", Type(Float64, {(size_t)N, (size_t)N}), Format{Dense, Dense});
  // TensorVar t("t", Type(Float64, {(size_t)N, (size_t)N}), Format{Dense, Dense});

  std::cout << "original sddmm_spmm stmt: " << stmt << endl;

	/* BEGIN sddmm_spmm TEST */
	vector<int> path0;
	stmt = stmt
		.reorder({i, l, j, k})
		.loopfuse(3, true, path0)
		;
	/* END sddmm_spmm TEST */

  stmt = stmt.concretize();
  cout << "final stmt: " << stmt << endl;
  printCodeToFile("sddmm_spmm", stmt);

  A.compile(stmt);
  A.assemble();

  Tensor<double> expected("expected", {N, N}, Format{Dense, Dense});
  expected(i,l) = B(i,j) * C(i,k) * D(j,k) * E(j,l);
  IndexStmt exp = makeReductionNotation(expected.getAssignment());
  exp = insertTemporaries(exp);
  exp = exp.concretize();
  expected.compile(exp);
  expected.assemble();

  clock_t begin;
  clock_t end;

  for (int i = 0; i< 10; i++) {
    begin = clock();
    A.compute(stmt);
    end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    begin = clock();
    expected.compute();
    end = clock();
    double elapsed_secs_ref = double(end - begin) / CLOCKS_PER_SEC;
    // ASSERT_TENSOR_EQ(expected, A);

    std::cout << elapsed_secs << std::endl;
    std::cout << elapsed_secs_ref << std::endl;
  }

}

TEST(workspaces, sddmm_spmm_gemm) {
  int N = 16;
  float SPARSITY = 0.3;
  Tensor<double> A("A", {N, N}, Format{Dense, Dense});
  Tensor<double> B("B", {N, N}, Format{Dense, Sparse});
  Tensor<double> C("C", {N, N}, Format{Dense, Dense});
  Tensor<double> D("D", {N, N}, Format{Dense, Dense});
  Tensor<double> E("E", {N, N}, Format{Dense, Dense});
  Tensor<double> F("F", {N, N}, Format{Dense, Dense});

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      float rand_float = (float) rand() / (float) RAND_MAX;
      if (rand_float < SPARSITY)
        B.insert({i, j}, (double) i);
      C.insert({i, j}, (double) j);
      E.insert({i, j}, (double) i*j);
      D.insert({i, j}, (double) i*j);
      F.insert({i, j}, (double) i*j);
    }
  }
  B.pack();

  // 3 -> A(i,l) = B(i,j) * C(i,k) * D(j,k) * E(j,l) - <SDDMM, SpMM>
  IndexVar i("i"), j("j"), k("k"), l("l"), m("m");
  A(i,m) = B(i,j) * C(i,k) * D(j,k) * E(j,l) * F(l,m);

  IndexStmt stmt = A.getAssignment().concretize();

  std::cout << "original assignment: " << stmt << endl;

	/* BEGIN sddmm_spmm_gemm TEST */
	vector<int> path0;
	vector<int> path1 = {1};
	vector<int> path2 = {1, 0};
	vector<int> path3 = {1, 0, 0};
	stmt = stmt
		.reorder({i, k, j, l, m})
		.loopfuse(1, true, path0)
		.loopfuse(4, true, path1)
		.loopfuse(3, true, path2)
		.loopfuse(1, false, path3)
		;
	/* END sddmm_spmm_gemm TEST */

  stmt = stmt.concretize();
  cout << "final stmt: " << stmt << endl;
  printCodeToFile("sddmm_spmm_gemm", stmt);

  // return;
  A.compile(stmt);

  // return;
  A.assemble();

  Tensor<double> expected("expected", {N, N}, Format{Dense, Dense});
  expected(i,m) = B(i,j) * C(i,k) * D(j,k) * E(j,l) * F(l,m);
  IndexStmt exp = makeReductionNotation(expected.getAssignment());
  exp = insertTemporaries(exp);
  exp = exp.concretize();
  expected.compile(exp);
  expected.assemble();

  clock_t begin;
  clock_t end;

  for (int i = 0; i< 11; i++) {
    begin = clock();
    A.compute(stmt);
    end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    begin = clock();
    expected.compute();
    end = clock();
    double elapsed_secs_ref = double(end - begin) / CLOCKS_PER_SEC;
    // ASSERT_TENSOR_EQ(expected, A);

    std::cout << elapsed_secs << std::endl;
    std::cout << elapsed_secs_ref << std::endl;
  }
}

TEST(workspaces, sddmm_spmm_gemm_real) {

  int K = 16;
  int L = 16; 
  int M = 16;

  // for parallel execution
  int nthreads = std::stoi(util::getFromEnv("OMP_NUM_THREADS", "1"));
  taco_set_num_threads(nthreads);
  taco_set_parallel_schedule(ParallelSchedule::Static, 64);

  std::string mat_file = util::getFromEnv("TENSOR_FILE", "");
  int iterations = std::stoi(util::getFromEnv("ITERATIONS", "0"));

  if (mat_file == "") {
    std::cout << "No tensor file specified!\n";
    return;
  }

  Tensor<double> B = read(mat_file, Format({Dense, Sparse}), true);
  B.setName("B");
  B.pack();

  Tensor<double> C("C", {B.getDimension(0), K}, Format{Dense, Dense});
  for (int i=0; i<B.getDimension(0); i++) {
    for (int l=0; l<K; l++) {
      C.insert({i, l}, (double) i);
    }
  }
  C.pack();
  Tensor<double> D("D", {B.getDimension(1), K}, Format{Dense, Dense});
  for (int j=0; j<B.getDimension(1); j++) {
    for (int m=0; m<K; m++) {
      D.insert({j, m}, (double) j);
    }
  }
  D.pack();
  Tensor<double> E("E", {B.getDimension(1), L}, Format{Dense, Dense});
  for (int j=0; j<B.getDimension(1); j++) {
    for (int m=0; m<L; m++) {
      E.insert({j, m}, (double) j);
    }
  }
  E.pack();
  Tensor<double> F("F", {L, M}, Format{Dense, Dense});
  for (int j=0; j<L; j++) {
    for (int m=0; m<M; m++) {
      E.insert({j, m}, (double) j);
    }
  }
  E.pack();

  Tensor<double> A("A", {B.getDimension(0), M}, Format{Dense, Dense});

  // 3 -> A(i,l) = B(i,j) * C(i,k) * D(j,k) * E(j,l) * F(l,m) - <SDDMM, SpMM>
  IndexVar i("i"), j("j"), k("k"), l("l"), m("m");

	/* BEGIN sddmm_spmm_gemm_real TEST */

	vector<int> path_ = {};
	vector<int> path_0 = {0};
	vector<int> path_1 = {1};

	A(i, m) = B(i, j) * C(i, k) * D(j, k) * E(j, l) * F(l, m);
	IndexStmt stmt = A.getAssignment().concretize();
	std::cout << stmt << endl;
	stmt = stmt
		.reorder(path_, {i,j,k,l,m})
		.loopfuse(4, true, path_)
		.reorder(path_0, {j,k,l})
		.loopfuse(3, true, path_0)
		.reorder(path_1, {l,m})
		.parallelize(i, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
		;
	/* END sddmm_spmm_gemm_real TEST */

  stmt = insertTemporaries(stmt);
  stmt = stmt.concretize();
  std::cout << "final stmt: " << stmt << endl;
  printCodeToFile("sddmm_spmm_gemm_real", stmt);

  A.compile(stmt);
  A.assemble();

  // Tensor<double> expected("expected", {B.getDimension(0), M}, Format{Dense, Dense});
  // expected(i,m) = B(i,j) * C(i,k) * D(j,k) * E(j,l) * F(l,m);
  // IndexStmt exp = makeReductionNotation(expected.getAssignment());
  // exp = insertTemporaries(exp);
  // exp = exp.concretize();
  // expected.compile(exp);
  // expected.assemble();

  // IndexStmt stmt2 = expected.getAssignment().concretize();
  // printCodeToFile("reference_sddmm_spmm_gemm_real", stmt2);

  std::chrono::time_point<std::chrono::system_clock> begin, end;
  std::chrono::duration<double> elapsed_secs;
  double elapsed_mills;

  for (int i = 0; i < iterations; i++) {
    begin = std::chrono::system_clock::now();
    A.compute(stmt);
    end = std::chrono::system_clock::now();
    elapsed_secs = end - begin;
    elapsed_mills = elapsed_secs.count() * 1000;
    // begin = clock();
    // expected.compute();
    // end = clock();
    // double elapsed_secs_ref = double(end - begin) / CLOCKS_PER_SEC * 1000;
    // ASSERT_TENSOR_EQ(expected, A);

    std::cout << elapsed_mills << std::endl;
    // std::cout << elapsed_secs_ref << std::endl;
  }

  std::cout << "workspaces, sddmm_spmm_gemm -> execution completed for matrix: " << mat_file << std::endl;
}

TEST(workspaces, default_sddmm_spmm_gemm_real) {

  int K = 16;
  int L = 16; 
  int M = 16;

  // for parallel execution
  int nthreads = std::stoi(util::getFromEnv("OMP_NUM_THREADS", "1"));
  taco_set_num_threads(nthreads);
  taco_set_parallel_schedule(ParallelSchedule::Static, 64);

  std::string mat_file = util::getFromEnv("TENSOR_FILE", "");
  int iterations = std::stoi(util::getFromEnv("ITERATIONS", "0"));

  if (mat_file == "") {
    std::cout << "No tensor file specified!\n";
    return;
  }

  Tensor<double> B = read(mat_file, Format({Dense, Sparse}), true);
  B.setName("B");
  B.pack();

  Tensor<double> C("C", {B.getDimension(0), K}, Format{Dense, Dense});
  for (int i=0; i<B.getDimension(0); i++) {
    for (int l=0; l<K; l++) {
      C.insert({i, l}, (double) i);
    }
  }
  C.pack();
  Tensor<double> D("D", {B.getDimension(1), K}, Format{Dense, Dense});
  for (int j=0; j<B.getDimension(1); j++) {
    for (int m=0; m<K; m++) {
      D.insert({j, m}, (double) j);
    }
  }
  D.pack();
  Tensor<double> E("E", {B.getDimension(1), L}, Format{Dense, Dense});
  for (int j=0; j<B.getDimension(1); j++) {
    for (int m=0; m<L; m++) {
      E.insert({j, m}, (double) j);
    }
  }
  E.pack();
  Tensor<double> F("F", {L, M}, Format{Dense, Dense});
  for (int j=0; j<L; j++) {
    for (int m=0; m<M; m++) {
      E.insert({j, m}, (double) j);
    }
  }
  E.pack();

  Tensor<double> A("A", {B.getDimension(0), M}, Format{Dense, Dense});

  // 3 -> A(i,l) = B(i,j) * C(i,k) * D(j,k) * E(j,l) * F(l,m) - <SDDMM, SpMM>
  IndexVar i("i"), j("j"), k("k"), l("l"), m("m");

  Tensor<double> expected("expected", {B.getDimension(0), M}, Format{Dense, Dense});
  expected(i,m) = B(i,j) * C(i,k) * D(j,k) * E(j,l) * F(l,m);
  IndexStmt exp = makeReductionNotation(expected.getAssignment());
  exp = insertTemporaries(exp);
  exp = exp.concretize();
  exp = exp.parallelize(i, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
  expected.compile(exp);
  expected.assemble();

  std::cout << "reference stmt: " << exp << endl;
  std::cout << "reference stmt: " << exp << endl;
  printCodeToFile("default_sddmm_spmm_gemm_real", exp);

  std::chrono::time_point<std::chrono::system_clock> begin, end;
  std::chrono::duration<double> elapsed_secs;
  double elapsed_secs_ref;

  for (int i = 0; i < iterations; i++) {
    begin = std::chrono::system_clock::now();
    expected.compute();
    end = std::chrono::system_clock::now();
    elapsed_secs = end - begin;
    elapsed_secs_ref = elapsed_secs.count() * 1000;

    std::cout << elapsed_secs_ref << std::endl;
  }

  std::cout << "workspaces, sddmm_spmm_gemm -> execution completed for matrix: " << mat_file << std::endl;
}


TEST(workspaces, sddmm_spmm_real) {
  int K = 16;
  int L = 16;

  // for parallel execution
  int nthreads = std::stoi(util::getFromEnv("OMP_NUM_THREADS", "1"));
  taco_set_num_threads(nthreads);
  taco_set_parallel_schedule(ParallelSchedule::Static, 64);

  std::string mat_file = util::getFromEnv("TENSOR_FILE", "");
  int iterations = std::stoi(util::getFromEnv("ITERATIONS", "0"));

  Tensor<double> B = read(mat_file, Format({Dense, Sparse}), true);
  B.setName("B");
  B.pack();

  auto I = B.getDimension(0);
  auto J = B.getDimension(1);

  if (mat_file == "") {
    std::cout << "No tensor file specified!\n";
    return;
  }

  Tensor<double> C("C", {I, K}, Format{Dense, Dense});
  for (int i=0; i<I; i++) {
    for (int l=0; l<K; l++) {
      C.insert({i, l}, (double) i);
    }
  }
  C.pack();
  Tensor<double> D("D", {J, K}, Format{Dense, Dense});
  for (int j=0; j<J; j++) {
    for (int m=0; m<K; m++) {
      D.insert({j, m}, (double) j);
    }
  }
  D.pack();
  Tensor<double> E("E", {J, L}, Format{Dense, Dense});
  for (int j=0; j<J; j++) {
    for (int m=0; m<L; m++) {
      E.insert({j, m}, (double) j);
    }
  }
  E.pack();

  Tensor<double> A("A", {B.getDimension(0), L}, Format{Dense, Dense});

  // 3 -> A(i,l) = B(i,j) * C(i,k) * D(j,k) * E(j,l) - <SDDMM, SpMM>
  IndexVar i("i"), j("j"), k("k"), l("l");

  /* BEGIN sddmm_spmm_real TEST */

	vector<int> path_ = {};

	A(i, l) = B(i, j) * C(i, k) * D(j, k) * E(j, l);
	IndexStmt stmt = A.getAssignment().concretize();
	std::cout << stmt << endl;
	stmt = stmt
		.reorder(path_, {i,j,k,l})
		.loopfuse(3, true, path_)
		.parallelize(i, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
		;
	/* END sddmm_spmm_real TEST */

  stmt = stmt.concretize();
  cout << "final stmt: " << stmt << endl;
  printCodeToFile("sddmm_spmm_real", stmt);

  A.compile(stmt);
  A.assemble();

  // Tensor<double> expected("expected", {B.getDimension(0), L}, Format{Dense, Dense});
  // expected(i,l) = B(i,j) * C(i,k) * D(j,k) * E(j,l);
  // IndexStmt exp = makeReductionNotation(expected.getAssignment());
  // exp = insertTemporaries(exp);
  // exp = exp.concretize();
  // expected.compile(exp);
  // expected.assemble();

  std::chrono::time_point<std::chrono::system_clock> begin, end;
  std::chrono::duration<double> elapsed_secs;
  double elapsed_mills; 

  for (int i = 0; i < iterations; i++) {
    begin = std::chrono::system_clock::now();
    A.compute(stmt);
    end = std::chrono::system_clock::now();
    elapsed_secs = end - begin;
    elapsed_mills = elapsed_secs.count() * 1000;
    // begin = clock();
    // expected.compute();
    // end = clock();
    // double elapsed_secs_ref = double(end - begin) / CLOCKS_PER_SEC * 1000;
    // // ASSERT_TENSOR_EQ(expected, A);

    std::cout << elapsed_mills << std::endl;
    // std::cout << elapsed_secs_ref << std::endl;
  }

  std::cout << "workspaces, sddmm_spmm -> execution completed for matrix: " << mat_file 
    << ", for number of threads: " << nthreads << std::endl;

}

TEST(workspaces, sddmm_spmm_willow) {
  int K = 16;
  int L = 16;

  int nthreads = std::stoi(util::getFromEnv("OMP_NUM_THREADS", "1"));
  taco_set_num_threads(nthreads);
  taco_set_parallel_schedule(ParallelSchedule::Static, 64);

  std::string mat_file = util::getFromEnv("TENSOR_FILE", "");
  int iterations = std::stoi(util::getFromEnv("ITERATIONS", "0"));

  Tensor<double> B = read(mat_file, Format({Dense, Sparse}), true);
  B.setName("B");
  B.pack();

  auto I = B.getDimension(0);
  auto J = B.getDimension(1);

  if (mat_file == "") {
    std::cout << "No tensor file specified!\n";
    return;
  }

  Tensor<double> C("C", {I, K}, Format{Dense, Dense});
  for (int i=0; i<I; i++) {
    for (int l=0; l<K; l++) {
      C.insert({i, l}, (double) i);
    }
  }
  C.pack();
  Tensor<double> D("D", {J, K}, Format{Dense, Dense});
  for (int j=0; j<J; j++) {
    for (int m=0; m<K; m++) {
      D.insert({j, m}, (double) j);
    }
  }
  D.pack();
  Tensor<double> E("E", {J, L}, Format{Dense, Dense});
  for (int j=0; j<J; j++) {
    for (int m=0; m<L; m++) {
      E.insert({j, m}, (double) j);
    }
  }
  E.pack();

  Tensor<double> A("A", {B.getDimension(0), L}, Format{Dense, Dense});

  // 3 -> A(i,l) = B(i,j) * C(i,k) * D(j,k) * E(j,l) - <SDDMM, SpMM>
  IndexVar i("i"), j("j"), k("k"), l("l");

  /* BEGIN sddmm_spmm_willow TEST */

	vector<int> path_ = {};

	A(i, l) = C(i, k) * D(j, k) * B(i, j) * E(j, l);
	IndexStmt stmt = A.getAssignment().concretize();
	std::cout << stmt << endl;
	stmt = stmt
		.reorder(path_, {i,j,k,l})
		.loopfuse(2, true, path_)
		// .parallelize(i, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
		;
	/* END sddmm_spmm_willow TEST */

  stmt = stmt.concretize();
  cout << "final stmt: " << stmt << endl;
  printCodeToFile("sddmm_spmm_willow", stmt);

  A.compile(stmt);
  A.assemble();

  // Tensor<double> expected("expected", {B.getDimension(0), L}, Format{Dense, Dense});
  // expected(i,l) = B(i,j) * C(i,k) * D(j,k) * E(j,l);
  // IndexStmt exp = makeReductionNotation(expected.getAssignment());
  // exp = insertTemporaries(exp);
  // exp = exp.concretize();
  // expected.compile(exp);
  // expected.assemble();

  std::chrono::time_point<std::chrono::system_clock> begin, end;
  std::chrono::duration<double> elapsed_secs;
  double elapsed_mills; 

  for (int i = 0; i < iterations; i++) {
    begin = std::chrono::system_clock::now();
    A.compute(stmt);
    end = std::chrono::system_clock::now();
    elapsed_secs = end - begin;
    elapsed_mills = elapsed_secs.count() * 1000;
    // begin = clock();
    // expected.compute();
    // end = clock();
    // double elapsed_secs_ref = double(end - begin) / CLOCKS_PER_SEC * 1000;
    // // ASSERT_TENSOR_EQ(expected, A);

    std::cout << elapsed_mills << std::endl;
    // std::cout << elapsed_secs_ref << std::endl;
  }

  std::cout << "workspaces, sddmm_spmm_willow -> execution completed for matrix: " << mat_file << std::endl;

}

TEST(workspaces, default_sddmm_spmm_real) {
  int K = 16;
  int L = 16;

  // for parallel execution
  int nthreads = std::stoi(util::getFromEnv("OMP_NUM_THREADS", "1"));
  taco_set_num_threads(nthreads);
  taco_set_parallel_schedule(ParallelSchedule::Static, 64);

  std::string mat_file = util::getFromEnv("TENSOR_FILE", "");
  int iterations = std::stoi(util::getFromEnv("ITERATIONS", "0"));

  Tensor<double> B = read(mat_file, Format({Dense, Sparse}), true);
  B.setName("B");
  B.pack();

  if (mat_file == "") {
    std::cout << "No tensor file specified!\n";
    return;
  }

  Tensor<double> C("C", {B.getDimension(0), K}, Format{Dense, Dense});
  for (int i=0; i<B.getDimension(0); i++) {
    for (int l=0; l<K; l++) {
      C.insert({i, l}, (double) i);
    }
  }
  C.pack();
  Tensor<double> D("D", {B.getDimension(1), K}, Format{Dense, Dense});
  for (int j=0; j<B.getDimension(1); j++) {
    for (int m=0; m<K; m++) {
      D.insert({j, m}, (double) j);
    }
  }
  D.pack();
  Tensor<double> E("E", {B.getDimension(1), L}, Format{Dense, Dense});
  for (int j=0; j<B.getDimension(1); j++) {
    for (int m=0; m<L; m++) {
      E.insert({j, m}, (double) j);
    }
  }
  E.pack();

  // 3 -> A(i,l) = B(i,j) * C(i,k) * D(j,k) * E(j,l) - <SDDMM, SpMM>
  IndexVar i("i"), j("j"), k("k"), l("l");

  Tensor<double> expected("expected", {B.getDimension(0), L}, Format{Dense, Dense});
  expected(i,l) = B(i,j) * C(i,k) * D(j,k) * E(j,l);
  IndexStmt exp = makeReductionNotation(expected.getAssignment());
  exp = insertTemporaries(exp);
  exp = exp.concretize();
  exp = exp.parallelize(i, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
  expected.compile(exp);
  expected.assemble();

  cout << "default stmt: " << exp << endl;
  cout << "default stmt: " << exp << endl;
  printCodeToFile("default_sddmm_spmm_real", exp);

  // double begin;
  // double end;
  std::chrono::time_point<std::chrono::system_clock> begin, end;
  std::chrono::duration<double> elapsed_seconds;
  double elapsed_mills = 0;

  for (int i = 0; i< iterations; i++) {
    begin = std::chrono::system_clock::now();
    // begin = omp_get_wtime();
    expected.compute();
    // end = omp_get_wtime();
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - begin;
    elapsed_mills = elapsed_seconds.count() * 1000;

    std::cout << elapsed_mills << std::endl;
  }

  std::cout << "workspaces, sddmm_spmm -> execution completed for matrix: " << mat_file << std::endl;
}

TEST(workspaces, loopreversefuse) {
  int N = 16;
  float SPARSITY = 0.3;
  Tensor<double> A("A", {N, N}, Format{Dense, Dense});
  Tensor<double> B("B", {N, N}, Format{Dense, Sparse});
  Tensor<double> C("C", {N, N}, Format{Dense, Dense});
  Tensor<double> D("D", {N, N}, Format{Dense, Dense});
  Tensor<double> E("E", {N, N}, Format{Dense, Dense});

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      float rand_float = (float) rand() / (float) RAND_MAX;
      if (rand_float < SPARSITY) 
        B.insert({i, j}, (double) rand_float);
      C.insert({i, j}, (double) j);
      E.insert({i, j}, (double) i*j);
      D.insert({i, j}, (double) i*j);
    }
  }
  B.pack();

  IndexVar i("i"), j("j"), k("k"), l("l"), m("m");
  A(i,m) = B(i,j) * C(j,k) * D(k,l) * E(l,m);

  IndexStmt stmt = A.getAssignment().concretize();

  std::cout << stmt << endl;
  vector<int> path1;
  stmt = stmt
    .reorder({m,i,l,k,j})
    .loopfuse(3, false, path1)
    ;
  // stmt = stmt
  //   .parallelize(m, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
  //   ;

  stmt = stmt.concretize();
  cout << "final stmt: " << stmt << endl;
  printCodeToFile("loopreversefuse", stmt);

  A.compile(stmt);
  B.pack();
  A.assemble();
  A.compute(stmt);

  Tensor<double> expected("expected", {N, N}, Format{Dense, Dense});
  expected(i,m) = B(i,j) * C(j,k) * D(k,l) * E(l,m);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, A);
}

TEST(workspaces, loopcontractfuse) {

// [jpos = 23,
//  j = 2048,
//  n = 10,
//  i = 3,
//  l = 53,
//  k = 1022,
//  kpos = 649,
//  m = 221]

  // loop 5 is lowest in this configuration
  // int L = 53; int M = 221; int N = 10;
  // int I = 3; int J = 2048; int K = 1022;
  // float JPOS = 23; float KPOS = 649;

  // loop 6 is the lowest in this configuration
  // int L = 256; int M = 200; int N = 196;
  // int I = 1; int J = 200; int K = 4000;
  // float JPOS = 16; float KPOS = 100;

  // // loop 4 is the lowest in this configuration
  // int L = 100; int M = 16; int N = 10;
  // int I = 1800; int J = 800; int K = 1000;
  // float JPOS = 16; float KPOS = 400;

  // // loop 4 is the lowest in this configuration
  // int L = 100; int M = 16; int N = 10;
  // int I = 1800; int J = 800; int K = 1000;
  // float JPOS = 16; float KPOS = 400;

  // loop 5 is the lowest in this configuration
  int L = 10; int M = 10; int N = 10;
  int I = 100; int J = 100; int K = 100;
  float JPOS = 5; float KPOS = 5;

  // int N = 16;
  float jk = (JPOS * KPOS);
  float jkr = (float) (J * K);
  float SPARSITY = jk / jkr;
  // std::cout << "sparsity: " << SPARSITY << std::endl;
  Tensor<double> A("A", {L, M, N}, Format{Dense, Dense, Dense});
  Tensor<double> B("B", {I, J, K}, Format{Dense, Sparse, Sparse});
  Tensor<double> C("C", {I, L}, Format{Dense, Dense});
  Tensor<double> D("D", {J, M}, Format{Dense, Dense});
  Tensor<double> E("E", {K, N}, Format{Dense, Dense});

  int count = 0;

  for (int i = 0; i < I; i++) {
    // std::cout << "i: " << i << std::endl;
    for (int j = 0; j < J; j++) {
      for (int k = 0; k < K; k++) {
        float rnd = (float) rand();
        float rnd_max = (float) RAND_MAX;
        float rand_float = rnd / rnd_max;
        if (rand_float < SPARSITY) {
          B.insert({i, j, k}, (double) i);
          count++;
          // if (count % 1000) std::cout << "count: " << count << std::endl;
        }
      }
    }
  }
  B.pack();
  // write("/home/min/a/kadhitha/workspace/my_taco/tensor-schedules/downloads/265_1207_479_0033.tns", B);
  // return;

  for (int i = 0; i < I; i++) {
    for (int j = 0; j < L; j++) {
      C.insert({i, j}, (double) j);
    }
  }
  // C.pack();

  for (int i = 0; i < J; i++) {
    for (int j = 0; j < M; j++) {
      D.insert({i, j}, (double) i*j);
    }
  }
  // D.pack();

  for (int i = 0; i < K; i++) {
    for (int j = 0; j < N; j++) {
      E.insert({i, j}, (double) i*j);
    }
  }
  // E.pack();

  IndexVar i("i"), j("j"), k("k"), l("l"), m("m"), n("n");
	A(l, m, n) = B(i, j, k) * C(i, l) * D(j, m) * E(k, n);
	
	IndexStmt stmt = A.getAssignment().concretize();
	std::cout << stmt << endl;
	
	vector<int> path0;
	vector<int> path1 = {0};
  vector<int> path2 = {1};
	stmt = stmt
		.reorder({l,m,n,i,j,k})
		.loopfuse(2, true, path0);
  cout << "stmt: " << stmt << endl;
  stmt = stmt  .reorder(path2, {m,k,n,j});
  cout << "stmt: " << stmt << endl;
	stmt = stmt	.loopfuse(2, true, path2)
		;
	/* END loopcontractfuse TEST */


  stmt = stmt.concretize();
  cout << "final stmt: " << stmt << endl;
  printCodeToFile("loopcontractfuse", stmt);

  A.compile(stmt.concretize());
  A.assemble();

  Tensor<double> expected("expected", {L, M, N}, Format{Dense, Dense, Dense});
  expected(l,m,n) = B(i,j,k) * C(i,l) * D(j,m) * E(k,n);
  expected.compile();
  expected.assemble();

  clock_t begin;
  clock_t end;

  for (int i=0; i<11; i++) {
    begin = clock();
    A.compute(stmt);
    end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC * 1000;

    begin = clock();
    expected.compute();
    end = clock();
    double elapsed_secs_ref = double(end - begin) / CLOCKS_PER_SEC * 1000;
    // ASSERT_TENSOR_EQ(expected, A);

    std::cout << elapsed_secs << std::endl;
    std::cout << elapsed_secs_ref << std::endl;
  }

}

TEST(workspaces, loopcontractfuse_real) {
  // Tensor<double> B("B", {N, N, N}, Format{Dense, Sparse, Sparse});
  // Tensor<double> C("C", {N, N}, Format{Dense, Dense});
  // Tensor<double> D("D", {N, N}, Format{Dense, Dense});
  // Tensor<double> E("E", {N, N}, Format{Dense, Dense});

  // for parallel execution
  int nthreads = std::stoi(util::getFromEnv("OMP_NUM_THREADS", "1"));
  taco_set_num_threads(nthreads);
  taco_set_parallel_schedule(ParallelSchedule::Static, 64);

  std::string mat_file = util::getFromEnv("TENSOR_FILE", "");
  int iterations = std::stoi(util::getFromEnv("ITERATIONS", "0"));
  int L = std::stoi(util::getFromEnv("L", "16"));
  int M = std::stoi(util::getFromEnv("M", "16"));
  int N = std::stoi(util::getFromEnv("N", "16"));

  Tensor<double> A("A", {L, M, N}, Format{Dense, Dense, Dense});

  if (mat_file == "") {
    std::cout << "No tensor file specified!\n";
    return;
  }

  Tensor<double> B = read(mat_file, Format({Dense, Sparse, Sparse}), true);
  B.setName("B");
  B.pack();

  // std::cout << "B tensor successfully read and packed!\n";
  // return;

  Tensor<double> C("C", {B.getDimension(0), L}, Format{Dense, Dense});
  for (int i=0; i<B.getDimension(0); i++) {
    for (int l=0; l<L; l++) {
      C.insert({i, l}, (double) i);
    }
  }
  C.pack();
  Tensor<double> D("D", {B.getDimension(1), M}, Format{Dense, Dense});
  for (int j=0; j<B.getDimension(1); j++) {
    for (int m=0; m<M; m++) {
      D.insert({j, m}, (double) j);
    }
  }
  D.pack();
  Tensor<double> E("E", {B.getDimension(2), N}, Format{Dense, Dense});
  for (int k=0; k<B.getDimension(2); k++) {
    for (int n=0; n<N; n++) {
      E.insert({k, n}, (double) k);
    }
  }
  E.pack();

  IndexVar i("i"), j("j"), k("k"), l("l"), m("m"), n("n");
  // A(l,m,n) = B(i,j,k) * C(i,l) * D(j,m) * E(k,n);
  // IndexStmt stmt = A.getAssignment().concretize();
  // std::cout << stmt << endl;

	/* BEGIN loopcontractfuse_real TEST */

	A(l, m, n) = B(i, j, k) * E(k, n) * D(j, m) * C(i, l);
	
	IndexStmt stmt = A.getAssignment().concretize();
	std::cout << stmt << endl;
	
	vector<int> path0;
	vector<int> path1 = {0};
    vector<int> path2 = {1};
	stmt = stmt
    .reorder({i, n, j, k, l, m})
    .loopfuse(3, true, path0)
    .loopfuse(2, true, path1)
		;
    if (nthreads > 1) {
        stmt = stmt.parallelize(i, ParallelUnit::CPUThread, OutputRaceStrategy::Atomics);
    }

	/* END loopcontractfuse_real TEST */

  //
	// vector<int> path0;
	// vector<int> path1 = {0};
	// stmt = stmt
	// 	.reorder({i, n, j, k, l, m})
	// 	.loopfuse(3, true, path0)
	// 	.loopfuse(2, true, path1)
	// 	;

  // // config 1 - loop depth 4
	// stmt = stmt
  //   .reorder({l, i, j, k, m, n})
  //   .loopfuse(2, true, path0)
  //   .reorder(path1, {m, k, j})
  //   .loopfuse(2, true, path1)
	// 	;

  // // config 2 - loop depth 5
  // stmt = stmt
  //   .reorder({l, m, i, j, k, n})
  //   .loopfuse(3, true, path0)
  //   .reorder(path1, {n, k})
  //   ;

  // // config 3 - loop depth 5
  // stmt = stmt
  //   .reorder({l, m, i, j, k, n})
  //   .loopfuse(3, true, path0)
  //   ;

  // // config 4 - loop depth 5
  // stmt = stmt
  //   .reorder({m, l, i, j, k, n})
  //   .loopfuse(3, true, path0)
  //    ;

  // // config 5 - loop depth 4
  // stmt = stmt
  //   .reorder({l, i, j, k, m, n})
  //   .loopfuse(2, true, path0)
  //   .reorder(path1, {k, m, j})
  //   .loopfuse(2, true, path1)
  //  ;

  // // config 6 - loop depth 5
  // stmt = stmt
  //   .reorder({m, l, i, j, k, n})
  //   .loopfuse(3, true, path0)
  //   .reorder(path1, {n, k})
  //    ;

  stmt = insertTemporaries(stmt);
  // stmt = stmt.concretize();
  cout << "final stmt: " << stmt << endl;
  printCodeToFile("loopcontractfuse_real", stmt);

  A.compile(stmt.concretize());
  A.assemble();

  // return;

  // Tensor<double> expected("expected", {N, N, N}, Format{Dense, Dense, Dense});
  // expected(l,m,n) = B(i,j,k) * C(i,l) * D(j,m) * E(k,n);
  // expected.compile();
  // expected.assemble();

  // IndexStmt stmt2 = expected.getAssignment().concretize();
  // printCodeToFile("reference_loopcontractfuse_real", stmt2);

  std::chrono::time_point<std::chrono::system_clock> begin, end;
  std::chrono::duration<double> elapsed_secs;
  double elapsed_mills;

  for (int i=0; i < iterations; i++) {
    begin = std::chrono::system_clock::now();
    A.compute(stmt);
    end = std::chrono::system_clock::now();
    elapsed_secs = end - begin;
    elapsed_mills = elapsed_secs.count() * 1000;

    // begin = clock();
    // if (iteration == 0) expected.compute();
    // end = clock();
    // double elapsed_secs_ref = double(end - begin) / CLOCKS_PER_SEC * 1000;
    // ASSERT_TENSOR_EQ(expected, A);

    std::cout << elapsed_mills << std::endl;
    // std::cout << elapsed_secs_ref << std::endl;
  }

std::cout << "workspaces, loopcontractfuse -> execution completed for matrix: " << mat_file << std::endl;

}

TEST(workspaces, spttn_cyclops_loopcontractfuse_real) {
  int L = std::stoi(util::getFromEnv("L", "16"));
  int M = std::stoi(util::getFromEnv("M", "16"));
  int N = std::stoi(util::getFromEnv("N", "16"));

  Tensor<double> A("A", {L, M, N}, Format{Dense, Dense, Dense});

  std::string mat_file = util::getFromEnv("TENSOR_FILE", "");
  int iterations = std::stoi(util::getFromEnv("ITERATIONS", "0"));

  if (mat_file == "") {
    std::cout << "No tensor file specified!\n";
    return;
  }

  Tensor<double> B = read(mat_file, Format({Dense, Sparse, Sparse}), true);
  B.setName("B");
  B.pack();

  // std::cout << "B tensor successfully read and packed!\n";
  // return;

  Tensor<double> C("C", {B.getDimension(0), L}, Format{Dense, Dense});
  for (int i=0; i<B.getDimension(0); i++) {
    for (int l=0; l<L; l++) {
      C.insert({i, l}, (double) i);
    }
  }
  C.pack();
  Tensor<double> D("D", {B.getDimension(1), M}, Format{Dense, Dense});
  for (int j=0; j<B.getDimension(1); j++) {
    for (int m=0; m<M; m++) {
      D.insert({j, m}, (double) j);
    }
  }
  D.pack();
  Tensor<double> E("E", {B.getDimension(2), N}, Format{Dense, Dense});
  for (int k=0; k<B.getDimension(2); k++) {
    for (int n=0; n<N; n++) {
      E.insert({k, n}, (double) k);
    }
  }
  E.pack();

  IndexVar i("i"), j("j"), k("k"), l("l"), m("m"), n("n");
  // A(l,m,n) = B(i,j,k) * C(i,l) * D(j,m) * E(k,n);
  // IndexStmt stmt = A.getAssignment().concretize();
  // std::cout << stmt << endl;

	/* BEGIN spttn_cyclops_loopcontractfuse_real TEST */

	A(l, m, n) = B(i, j, k) * E(k, n) * D(j, m) * C(i, l);
	
	IndexStmt stmt = A.getAssignment().concretize();
	std::cout << stmt << endl;
	
	vector<int> path0;
	vector<int> path1 = {0};
	stmt = stmt
		.reorder({i, j, k, l, m, n})
		.loopfuse(2, true, path0)
		.loopfuse(2, true, path1)
		;

	/* END spttn_cyclops_loopcontractfuse_real TEST */

  stmt = insertTemporaries(stmt);
  // stmt = stmt.concretize();
  cout << "final stmt: " << stmt << endl;
  printCodeToFile("spttn_cyclops_loopcontractfuse_real", stmt);

  A.compile(stmt.concretize());
  A.assemble();

  // return;

  // Tensor<double> expected("expected", {N, N, N}, Format{Dense, Dense, Dense});
  // expected(l,m,n) = B(i,j,k) * C(i,l) * D(j,m) * E(k,n);
  // expected.compile();
  // expected.assemble();

  // IndexStmt stmt2 = expected.getAssignment().concretize();
  // printCodeToFile("reference_spttn_cyclops_loopcontractfuse_real", stmt2);

  clock_t begin;
  clock_t end;

  for (int i=0; i < iterations; i++) {
    begin = clock();
    A.compute(stmt);
    end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC * 1000;

    // begin = clock();
    // if (iteration == 0) expected.compute();
    // end = clock();
    // double elapsed_secs_ref = double(end - begin) / CLOCKS_PER_SEC * 1000;
    // ASSERT_TENSOR_EQ(expected, A);

    std::cout << elapsed_secs << std::endl;
    // std::cout << elapsed_secs_ref << std::endl;
  }

std::cout << "workspaces, loopcontractfuse -> execution completed for matrix: " << mat_file << std::endl;

}

TEST(workspaces, default_loopcontractfuse_real) {
  int L = 16;
  int M = 16;
  int N = 16;

  // for parallel execution
  int nthreads = std::stoi(util::getFromEnv("OMP_NUM_THREADS", "1"));
  taco_set_num_threads(nthreads);
  taco_set_parallel_schedule(ParallelSchedule::Static, 64);

  std::string mat_file = util::getFromEnv("TENSOR_FILE", "");
  int iterations = std::stoi(util::getFromEnv("ITERATIONS", "0"));

  // std::cout << mat_file << std::endl;

  if (mat_file == "") {
    std::cout << "No tensor file specified!\n";
    return;
  }

  Tensor<double> B = read(mat_file, Format({Dense, Sparse, Sparse}), true);
  B.setName("B");
  B.pack();

  Tensor<double> C("C", {B.getDimension(0), L}, Format{Dense, Dense});
  for (int i=0; i<B.getDimension(0); i++) {
    for (int l=0; l<L; l++) {
      C.insert({i, l}, (double) i);
    }
  }
  C.pack();
  Tensor<double> D("D", {B.getDimension(1), M}, Format{Dense, Dense});
  for (int j=0; j<B.getDimension(1); j++) {
    for (int m=0; m<M; m++) {
      D.insert({j, m}, (double) j);
    }
  }
  D.pack();
  Tensor<double> E("E", {B.getDimension(2), N}, Format{Dense, Dense});
  for (int k=0; k<B.getDimension(2); k++) {
    for (int n=0; n<N; n++) {
      E.insert({k, n}, (double) k);
    }
  }
  E.pack();

  IndexVar i("i"), j("j"), k("k"), l("l"), m("m"), n("n");

  Tensor<double> expected("expected", {N, N, N}, Format{Dense, Dense, Dense});
  expected(l,m,n) = B(i,j,k) * C(i,l) * D(j,m) * E(k,n);
  IndexStmt stmt2 = expected.getAssignment().concretize();
  stmt2 = insertTemporaries(stmt2);
  stmt2 = stmt2.reorder({i, l, j, m, k, n});
  if (nthreads > 1) {
    stmt2 = stmt2.parallelize(i, ParallelUnit::CPUThread, OutputRaceStrategy::Atomics);
  }
  expected.compile(stmt2);
  expected.assemble();
  
  std::cout << "reference stmt: " << stmt2 << endl;
  std::cout << "reference stmt: " << stmt2 << endl;
  printCodeToFile("default_loopcontractfuse_real", stmt2);

  std::chrono::time_point<std::chrono::system_clock> begin, end;
  std::chrono::duration<double> elapsed_seconds;
  double elapsed_mills = 0;

  for (int i = 0; i < iterations; i++) {
    begin = std::chrono::system_clock::now();
    expected.compute();
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - begin;
    elapsed_mills = elapsed_seconds.count() * 1000;
    // ASSERT_TENSOR_EQ(expected, A);

    // std::cout << elapsed_secs << std::endl;
    std::cout << elapsed_mills << std::endl;
  }

  std::cout << "workspaces, reference_loopcontractfuse -> execution completed for matrix: " << mat_file << std::endl;

}


TEST(workspaces, mttkrp_gemm_real) {
  int J = 32;
  int M = 64;

  // for parallel execution
  int nthreads = std::stoi(util::getFromEnv("OMP_NUM_THREADS", "1"));
  taco_set_num_threads(nthreads);
  taco_set_parallel_schedule(ParallelSchedule::Static, 64);

  std::string mat_file = util::getFromEnv("TENSOR_FILE", "");
  int iterations = std::stoi(util::getFromEnv("ITERATIONS", "0"));

  if (mat_file == "") {
    std::cout << "No tensor file specified!\n";
    return;
  }

  Tensor<double> B = read(mat_file, Format({Dense, Sparse, Sparse}), true);
  B.setName("B");
  B.pack();

  // std::cout << "B tensor successfully read and packed!\n";
  // return;
  // std::cout << "0 dim: " << B.getDimension(0) << std::endl;
  //   std::cout << "0 dim: " << B.getDimension(1) << std::endl;
  Tensor<double> C("C", {B.getDimension(2), J}, Format{Dense, Dense});
  for (int i=0; i<B.getDimension(2); i++) {
    for (int l=0; l<J; l++) {
      C.insert({i, l}, (double) i);
    }
  }
  C.pack();
  Tensor<double> D("D", {B.getDimension(1), J}, Format{Dense, Dense});
  for (int j=0; j<B.getDimension(1); j++) {
    for (int m=0; m<J; m++) {
      D.insert({j, m}, (double) j);
    }
  }
  D.pack();
  Tensor<double> E("E", {J, M}, Format{Dense, Dense});
  for (int k=0; k<J; k++) {
    for (int n=0; n<M; n++) {
      E.insert({k, n}, (double) k);
    }
  }
  E.pack();

  IndexVar i("i"), j("j"), k("k"), l("l"), m("m");
  // A(l,m,n) = B(i,j,k) * C(i,l) * D(j,m) * E(k,n);
  // IndexStmt stmt = A.getAssignment().concretize();
  // std::cout << stmt << endl;
  Tensor<double> A("A", {B.getDimension(0), M}, Format{Dense, Dense});


	/* BEGIN mttkrp_gemm_real TEST */

	vector<int> path_ = {};
	vector<int> path_0 = {0};

	A(i, m) = B(i, k, l) * C(l, j) * D(k, j) * E(j, m);
	IndexStmt stmt = A.getAssignment().concretize();
	std::cout << stmt << endl;
	stmt = stmt
		.reorder(path_, {i,j,k,l,m})
		.loopfuse(3, true, path_)
		.reorder(path_0, {k,l})
		.parallelize(i, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
		;
	/* END mttkrp_gemm_real TEST */

  stmt = insertTemporaries(stmt);
  // stmt = stmt.concretize();
  cout << "final stmt: " << stmt << endl;
  printCodeToFile("mttkrp_gemm_real", stmt);

  A.compile(stmt.concretize());
  A.assemble();

  // return;

  // Tensor<double> expected("expected", {N, N, N}, Format{Dense, Dense, Dense});
  // expected(l,m,n) = B(i,j,k) * C(i,l) * D(j,m) * E(k,n);
  // expected.compile();
  // expected.assemble();

  // IndexStmt stmt2 = expected.getAssignment().concretize();
  // printCodeToFile("reference_mttkrp_gemm_real_real", stmt2);

  std::chrono::time_point<std::chrono::system_clock> begin, end;
  std::chrono::duration<double> elapsed_seconds;
  double elapsed_mills = 0;

  for (int i=0; i < iterations; i++) {
    begin = std::chrono::system_clock::now();
    A.compute(stmt);
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - begin;
    elapsed_mills = elapsed_seconds.count() * 1000;

    // begin = clock();
    // if (iteration == 0) expected.compute();
    // end = clock();
    // double elapsed_secs_ref = double(end - begin) / CLOCKS_PER_SEC * 1000;
    // ASSERT_TENSOR_EQ(expected, A);

    std::cout << elapsed_mills << std::endl;
    // std::cout << elapsed_secs_ref << std::endl;
  }

  std::cout << "workspaces, mttkrp-gemm -> execution completed for matrix: " << mat_file << std::endl;

}

TEST(workspaces, default_mttkrp_gemm_real) {
  int J = 32;
  int M = 64;

  // for parallel execution
  int nthreads = std::stoi(util::getFromEnv("OMP_NUM_THREADS", "1"));
  taco_set_num_threads(nthreads);
  taco_set_parallel_schedule(ParallelSchedule::Static, 64);

  std::string mat_file = util::getFromEnv("TENSOR_FILE", "");
  int iterations = std::stoi(util::getFromEnv("ITERATIONS", "0"));

  if (mat_file == "") {
    std::cout << "No tensor file specified!\n";
    return;
  }

  Tensor<double> B = read(mat_file, Format({Dense, Sparse, Sparse}), true);
  B.setName("B");
  B.pack();

  // std::cout << "B tensor successfully read and packed!\n";
  // return;
  // std::cout << "0 dim: " << B.getDimension(0) << std::endl;
  //   std::cout << "0 dim: " << B.getDimension(1) << std::endl;
  Tensor<double> C("C", {B.getDimension(2), J}, Format{Dense, Dense});
  for (int i=0; i<B.getDimension(2); i++) {
    for (int l=0; l<J; l++) {
      C.insert({i, l}, (double) i);
    }
  }
  C.pack();
  Tensor<double> D("D", {B.getDimension(1), J}, Format{Dense, Dense});
  for (int j=0; j<B.getDimension(1); j++) {
    for (int m=0; m<J; m++) {
      D.insert({j, m}, (double) j);
    }
  }
  D.pack();
  Tensor<double> E("E", {J, M}, Format{Dense, Dense});
  for (int k=0; k<J; k++) {
    for (int n=0; n<M; n++) {
      E.insert({k, n}, (double) k);
    }
  }
  E.pack();

  IndexVar i("i"), j("j"), k("k"), l("l"), m("m");
  // A(l,m,n) = B(i,j,k) * C(i,l) * D(j,m) * E(k,n);
  // IndexStmt stmt = A.getAssignment().concretize();
  // std::cout << stmt << endl;
  Tensor<double> A("A", {B.getDimension(0), M}, Format{Dense, Dense});

	A(i,m) = B(i, k, l) * C(l, j) * D(k, j) * E(j, m);
	
	IndexStmt stmt = A.getAssignment().concretize();
	std::cout << "default statement: " << stmt << endl;

  stmt = stmt.parallelize(i, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
  stmt = insertTemporaries(stmt);
  // stmt = stmt.concretize();
  cout << "final stmt: " << stmt << endl;
  printCodeToFile("default_mttkrp_gemm_real", stmt);

  A.compile(stmt.concretize());
  A.assemble();

  std::chrono::time_point<std::chrono::system_clock> begin, end;
  std::chrono::duration<double> elapsed_seconds;
  double elapsed_mills = 0;

  for (int i=0; i < iterations; i++) {
    begin = std::chrono::system_clock::now();
    A.compute(stmt);
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - begin;
    elapsed_mills = elapsed_seconds.count() * 1000;

    std::cout << elapsed_mills << std::endl;
  }

  std::cout << "workspaces, mttkrp-gemm -> execution completed for matrix: " << mat_file << std::endl;

}


TEST(workspaces, spttm_ttm) {
  int N = 16;
  Tensor<double> A("A", {N, N, N}, Format{Dense, Dense, Dense});
  Tensor<double> B("B", {N, N, N}, Format{Dense, Sparse, Sparse});
  Tensor<double> C("C", {N, N}, Format{Dense, Dense});
  Tensor<double> D("D", {N, N}, Format{Dense, Dense});

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        B.insert({i, j, k}, (double) i);
      }
      C.insert({i, j}, (double) j);
      D.insert({i, j}, (double) i*j);
    }
  }

  // 5 -> A(i,l,m) = B(i,j,k) * C(j,l) * D(k,m) - <SpTTM, TTM>
  IndexVar i("i"), j("j"), k("k"), l("l"), m("m"), n("n");
  A(i,l,m) = B(i,j,k) * C(j,l) * D(k,m);

  IndexStmt stmt = A.getAssignment().concretize();

  std::cout << stmt << endl;

	/* BEGIN spttm_ttm TEST */
	vector<int> path0;
	vector<int> path1 = {1};
	stmt = stmt
		.reorder({l, i, j, k, m})
		.loopfuse(2, true, path0)
		.reorder(path1, {m, k})
		;
	/* END spttm_ttm TEST */


  stmt = stmt.concretize();
  cout << "final stmt: " << stmt << endl;
  printCodeToFile("spttm_ttm", stmt);

  A.compile(stmt.concretize());
  A.assemble();

  Tensor<double> expected("expected", {N, N, N}, Format{Dense, Dense, Dense});
  expected(i,l,m) = B(i,j,k) * C(j,l) * D(k,m);
  expected.compile();
  expected.assemble();

  clock_t begin;
  clock_t end;

  for (int i=0; i<4; i++) {
    begin = clock();
    A.compute(stmt);
    end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC * 1000;

    begin = clock();
    expected.compute();
    end = clock();
    double elapsed_secs_ref = double(end - begin) / CLOCKS_PER_SEC * 1000;
    // ASSERT_TENSOR_EQ(expected, A);

    std::cout << elapsed_secs << std::endl;
    std::cout << elapsed_secs_ref << std::endl;
  }

}

TEST(workspaces, spttm_spttm) {
  int N = 16;
  Tensor<double> A("A", {N, N, N}, Format{Dense, Sparse, Dense});
  Tensor<double> B("B", {N, N, N}, Format{Dense, Sparse, Sparse});
  Tensor<double> C("C", {N, N}, Format{Dense, Dense});
  Tensor<double> D("D", {N, N}, Format{Dense, Dense});

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        B.insert({i, j, k}, (double) i);
      }
      C.insert({i, j}, (double) j);
      D.insert({i, j}, (double) i*j);
    }
  }

  // 5 -> A(i,l,m) = B(i,j,k) * C(j,l) * D(k,m) - <SpTTM, TTM>
  IndexVar i("i"), j("j"), k("k"), l("l"), m("m"), n("n");
  // A(i,j,m) = B(i,j,k) * C(k,l) * D(l,m);
  // IndexStmt stmt = A.getAssignment().concretize();
  // std::cout << "stmt: " << stmt << endl;

	/* BEGIN spttm_ttm TEST */
	A(i, j, m) = B(i, j, k) * C(k, l) * D(l, m);
	
	IndexStmt stmt = A.getAssignment().concretize();
	std::cout << stmt << endl;
	
	vector<int> path0;
	stmt = stmt
		.reorder({i, j, l, k, m})
		.loopfuse(2, true, path0)
		;
	/* END spttm_ttm TEST */


  // stmt = stmt.concretize();
  cout << "final stmt: " << stmt << endl;
  printCodeToFile("spttm_spttm", stmt);

  A.compile(stmt.concretize());
  A.assemble();

  Tensor<double> expected("expected", {N, N, N}, Format{Dense, Sparse, Dense});
  expected(i,j,m) = B(i,j,k) * C(k,l) * D(l,m);
  expected.compile();
  expected.assemble();

  IndexStmt expectedStmt = expected.getAssignment().concretize();
  printCodeToFile("reference_spttm_spttm", expectedStmt);

  clock_t begin;
  clock_t end;

  for (int i=0; i<10; i++) {
    begin = clock();
    A.compute(stmt);
    end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC * 1000;

    begin = clock();
    expected.compute();
    end = clock();
    double elapsed_secs_ref = double(end - begin) / CLOCKS_PER_SEC * 1000;
    // ASSERT_TENSOR_EQ(expected, A);

    std::cout << elapsed_secs << std::endl;
    std::cout << elapsed_secs_ref << std::endl;
  }

}

TEST(workspaces, spttm_ttm_real) {
  int L = 16;
  int M = 16;

  // for parallel execution
  int nthreads = std::stoi(util::getFromEnv("OMP_NUM_THREADS", "1"));
  taco_set_num_threads(nthreads);
  taco_set_parallel_schedule(ParallelSchedule::Static, 64);

  std::string mat_file = util::getFromEnv("TENSOR_FILE", "");
  int iterations = std::stoi(util::getFromEnv("ITERATIONS", "0"));

  if (mat_file == "") {
    std::cout << "No tensor file specified!\n";
    return;
  }

  Tensor<double> B = read(mat_file, Format({Dense, Sparse, Sparse}), true);
  B.setName("B");
  B.pack();

  Tensor<double> C("C", {B.getDimension(1), L}, Format{Dense, Dense});
  for (int i=0; i<B.getDimension(1); i++) {
    for (int l=0; l<L; l++) {
      C.insert({i, l}, (double) i);
    }
  }
  C.pack();
  Tensor<double> D("D", {B.getDimension(2), M}, Format{Dense, Dense});
  for (int j=0; j<B.getDimension(2); j++) {
    for (int m=0; m<M; m++) {
      D.insert({j, m}, (double) j);
    }
  }
  D.pack();

  Tensor<double> A("A", {B.getDimension(0), L, M}, Format{Dense, Dense, Dense});

  // 5 -> A(i,l,m) = B(i,j,k) * C(j,l) * D(k,m) - <SpTTM, TTM>
  IndexVar i("i"), j("j"), k("k"), l("l"), m("m"), n("n");

  // A(i,l,m) = B(i,j,k) * C(j,l) * D(k,m);
  // IndexStmt stmt = A.getAssignment().concretize();
  // std::cout << stmt << endl;

	/* BEGIN spttm_ttm_real TEST */

	vector<int> path_ = {};

	A(i, l, m) = B(i, j, k) * D(k, m) * C(j, l);
	IndexStmt stmt = A.getAssignment().concretize();
	std::cout << stmt << endl;
	stmt = stmt
		.reorder(path_, {i,m,j,k,l})
		.loopfuse(2, true, path_)
		.parallelize(i, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
		;
	/* END spttm_ttm_real TEST */


  stmt = stmt.concretize();
  cout << "final stmt: " << stmt << endl;
  printCodeToFile("spttm_ttm_real", stmt);

  A.compile(stmt.concretize());
  A.assemble();

  // Tensor<double> expected("expected", {B.getDimension(0), L, M}, Format{Dense, Dense, Dense});
  // expected(i,l,m) = B(i,j,k) * C(j,l) * D(k,m);
  // expected.compile();
  // expected.assemble();

  // IndexStmt stmt2 = expected.getAssignment().concretize();
  // printCodeToFile("reference_spttm_ttm_real", stmt2);

  std::chrono::time_point<std::chrono::system_clock> begin, end;
  std::chrono::duration<double> elapsed_seconds;
  double elapsed_mills = 0;

  for (int i=0; i < iterations; i++) {
    begin = std::chrono::system_clock::now();
    A.compute(stmt);
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - begin;
    elapsed_mills = elapsed_seconds.count() * 1000;

    // begin = clock();
    // expected.compute();
    // end = clock();
    // double elapsed_secs_ref = double(end - begin) / CLOCKS_PER_SEC * 1000;
    // ASSERT_TENSOR_EQ(expected, A);

    std::cout << elapsed_mills << std::endl;
    // std::cout << elapsed_secs_ref << std::endl;
  }

}

TEST(workspaces, default_spttm_ttm_real) {
  int L = 16;
  int M = 16;

  int nthreads = std::stoi(util::getFromEnv("OMP_NUM_THREADS", "1"));
  taco_set_num_threads(nthreads);
  taco_set_parallel_schedule(ParallelSchedule::Static, 64);

  std::string mat_file = util::getFromEnv("TENSOR_FILE", "");
  int iterations = std::stoi(util::getFromEnv("ITERATIONS", "0"));

  if (mat_file == "") {
    std::cout << "No tensor file specified!\n";
    return;
  }

  Tensor<double> B = read(mat_file, Format({Dense, Sparse, Sparse}), true);
  B.setName("B");
  B.pack();

  Tensor<double> C("C", {B.getDimension(1), L}, Format{Dense, Dense});
  for (int i=0; i<B.getDimension(1); i++) {
    for (int l=0; l<L; l++) {
      C.insert({i, l}, (double) i);
    }
  }
  C.pack();
  Tensor<double> D("D", {B.getDimension(2), M}, Format{Dense, Dense});
  for (int j=0; j<B.getDimension(2); j++) {
    for (int m=0; m<M; m++) {
      D.insert({j, m}, (double) j);
    }
  }
  D.pack();

  // 5 -> A(i,l,m) = B(i,j,k) * C(j,l) * D(k,m) - <SpTTM, TTM>
  IndexVar i("i"), j("j"), k("k"), l("l"), m("m"), n("n");

  Tensor<double> expected("expected", {B.getDimension(0), L, M}, Format{Dense, Dense, Dense});
  expected(i,l,m) = B(i,j,k) * C(j,l) * D(k,m);
  IndexStmt stmt2 = expected.getAssignment().concretize();
  stmt2 = stmt2.parallelize(i, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
  expected.compile();
  expected.assemble();
  
  std::cout << "reference stmt: " << stmt2 << endl;
  std::cout << "reference stmt: " << stmt2 << endl;
  printCodeToFile("default_spttm_ttm_real", stmt2);

  std::chrono::time_point<std::chrono::system_clock> begin, end;
  std::chrono::duration<double> elapsed_seconds;
  double elapsed_mills = 0;

  for (int i=0; i < iterations; i++) {
    begin = std::chrono::system_clock::now();
    expected.compute();
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - begin;
    elapsed_mills = elapsed_seconds.count() * 1000;
    // ASSERT_TENSOR_EQ(expected, A);

    std::cout << elapsed_mills << std::endl;
  }

  std::cout << "default spttm-ttm real test execution finished\n";

}

TEST(workspaces, spttm_spttm_real) {
  int L = 16;
  int M = 16;

  int nthreads = std::stoi(util::getFromEnv("OMP_NUM_THREADS", "1"));
  taco_set_num_threads(nthreads);
  taco_set_parallel_schedule(ParallelSchedule::Static, 64);

  std::string mat_file = util::getFromEnv("TENSOR_FILE", "");
  int iterations = std::stoi(util::getFromEnv("ITERATIONS", "0"));

  if (mat_file == "") {
    std::cout << "No tensor file specified!\n";
    return;
  }

  Tensor<double> B = read(mat_file, Format({Dense, Sparse, Sparse}), true);
  B.setName("B");
  B.pack();

  // A(i, j, m) = B(i, j, k) * C(k, l) * D(l, m);
  Tensor<double> C("C", {B.getDimension(2), L}, Format{Dense, Dense});
  for (int i=0; i<B.getDimension(2); i++) {
    for (int l=0; l<L; l++) {
      C.insert({i, l}, (double) i);
    }
  }
  C.pack();
  Tensor<double> D("D", {L, M}, Format{Dense, Dense});
  for (int j=0; j<L; j++) {
    for (int m=0; m<M; m++) {
      D.insert({j, m}, (double) j);
    }
  }
  D.pack();

  Tensor<double> A("A", {B.getDimension(0), B.getDimension(1), M}, Format{Dense, Sparse, Dense});

  // 5 -> A(i,l,m) = B(i,j,k) * C(j,l) * D(k,m) - <SpTTM, TTM>
  IndexVar i("i"), j("j"), k("k"), l("l"), m("m"), n("n");

  // A(i,l,m) = B(i,j,k) * C(j,l) * D(k,m);
  // IndexStmt stmt = A.getAssignment().concretize();
  // std::cout << stmt << endl;

	/* BEGIN spttm_spttm_real TEST */

	vector<int> path_ = {};

	A(i, j, m) = B(i, j, k) * C(k, l) * D(l, m);
	IndexStmt stmt = A.getAssignment().concretize();
	std::cout << stmt << endl;
	stmt = stmt
		.reorder(path_, {i,j,l,k,m})
		.loopfuse(2, true, path_)
		;
	/* END spttm_spttm_real TEST */

  stmt = stmt.concretize();
  cout << "final stmt: " << stmt << endl;
  printCodeToFile("spttm_spttm_real", stmt);

  A.compile(stmt.concretize());
  A.assemble();

  Tensor<double> expected("expected", {B.getDimension(0), B.getDimension(1), M}, Format{Dense, Sparse, Dense});
  expected(i,j,m) = B(i,j,k) * C(k,l) * D(l,m);
  expected.compile();
  expected.assemble();

  IndexStmt stmt2 = expected.getAssignment().concretize();
  printCodeToFile("reference_spttm_spttm_real", stmt2);

  std::chrono::time_point<std::chrono::system_clock> begin, end;
  std::chrono::duration<double> elapsed_seconds;
  double elapsed_mills = 0;

  for (int i=0; i < iterations; i++) {
    begin = std::chrono::system_clock::now();
    A.compute(stmt);
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - begin;
    elapsed_mills = elapsed_seconds.count() * 1000;

    // begin = clock();
    // expected.compute();
    // end = clock();
    // double elapsed_secs_ref = double(end - begin) / CLOCKS_PER_SEC * 1000;
    // ASSERT_TENSOR_EQ(expected, A);

    std::cout << elapsed_mills << std::endl;
    // std::cout << elapsed_secs_ref << std::endl;
  }

}

TEST(workspaces, default_spttm_spttm_real) {
  int L = 16;
  int M = 16;

  int nthreads = std::stoi(util::getFromEnv("OMP_NUM_THREADS", "1"));
  taco_set_num_threads(nthreads);
  taco_set_parallel_schedule(ParallelSchedule::Static, 64);

  std::string mat_file = util::getFromEnv("TENSOR_FILE", "");
  int iterations = std::stoi(util::getFromEnv("ITERATIONS", "0"));

  if (mat_file == "") {
    std::cout << "No tensor file specified!\n";
    return;
  }

  Tensor<double> B = read(mat_file, Format({Dense, Sparse, Sparse}), true);
  B.setName("B");
  B.pack();

  // A(i, j, m) = B(i, j, k) * C(k, l) * D(l, m);
  Tensor<double> C("C", {B.getDimension(2), L}, Format{Dense, Dense});
  for (int i=0; i<B.getDimension(2); i++) {
    for (int l=0; l<L; l++) {
      C.insert({i, l}, (double) i);
    }
  }
  C.pack();
  Tensor<double> D("D", {L, M}, Format{Dense, Dense});
  for (int j=0; j<L; j++) {
    for (int m=0; m<M; m++) {
      D.insert({j, m}, (double) j);
    }
  }
  D.pack();

  // 5 -> A(i,l,m) = B(i,j,k) * C(j,l) * D(k,m) - <SpTTM, TTM>
  IndexVar i("i"), j("j"), k("k"), l("l"), m("m"), n("n");

  Tensor<double> expected("expected", {B.getDimension(0), B.getDimension(1), M}, Format{Dense, Sparse, Dense});
  expected(i,j,m) = B(i,j,k) * C(k,l) * D(l,m);
  IndexStmt stmt2 = expected.getAssignment().concretize();
  // stmt2 = stmt2.parallelize(i, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
  expected.compile();
  expected.assemble();

  std::cout << "reference stmt: " << stmt2 << endl;
  std::cout << "reference stmt: " << stmt2 << endl;
  printCodeToFile("default_spttm_spttm_real", stmt2);

  std::chrono::time_point<std::chrono::system_clock> begin, end;
  std::chrono::duration<double> elapsed_seconds;
  double elapsed_mills = 0;

  for (int i=0; i < iterations; i++) {
    begin = std::chrono::system_clock::now();
    expected.compute();
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - begin;
    elapsed_mills = elapsed_seconds.count() * 1000;
    // ASSERT_TENSOR_EQ(expected, A);

    std::cout << elapsed_mills << std::endl;
  }

  std::cout << "workspaces, reference_spttm_spttm_real -> execution completed for matrix: " << mat_file << std::endl;

}

TEST(workspaces, spmmh_gemm_real) {
  int J = 64;
  int L = 64;

  int nthreads = std::stoi(util::getFromEnv("OMP_NUM_THREADS", "1"));
  taco_set_num_threads(nthreads);
  taco_set_parallel_schedule(ParallelSchedule::Static, 64);

  std::string mat_file = util::getFromEnv("TENSOR_FILE", "");
  int iterations = std::stoi(util::getFromEnv("ITERATIONS", "0"));

  Tensor<double> B = read(mat_file, Format({Dense, Sparse}), true);
  B.setName("B");
  B.pack();

  if (mat_file == "") {
    std::cout << "No tensor file specified!\n";
    return;
  }

  Tensor<double> C("C", {B.getDimension(1), J}, Format{Dense, Dense});
  Tensor<double> D("D", {B.getDimension(1), J}, Format{Dense, Dense});
  for (int k=0; k<B.getDimension(0); k++) {
    for (int j=0; j<J; j++) {
      C.insert({k, j}, (double) j);
      D.insert({k, j}, (double) j);
    }
  }
  C.pack();
  D.pack();
  Tensor<double> E("E", {J, L}, Format{Dense, Dense});
  for (int j=0; j<J; j++) {
    for (int l=0; l<L; l++) {
      E.insert({j, l}, (double) l);
    }
  }
  E.pack();

  Tensor<double> A("A", {B.getDimension(0), L}, Format{Dense, Dense});

  // 3 -> A(i,l) = B(i,k) * C(k,j) * D(k,j) * E(j,l) - <SpMMH, GeMM>
  IndexVar i("i"), j("j"), k("k"), l("l");

	/* BEGIN spmmh_gemm_real TEST */

	vector<int> path_ = {};

	A(i, l) = B(i, k) * C(k, j) * D(k, j) * E(j, l);
	IndexStmt stmt = A.getAssignment().concretize();
	std::cout << stmt << endl;
	stmt = stmt
		.reorder(path_, {i,j,k,l})
		.loopfuse(3, true, path_)
		.parallelize(i, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
		;
	/* END spmmh_gemm_real TEST */

  stmt = stmt.concretize();
  cout << "final stmt: " << stmt << endl;
  printCodeToFile("spmmh_gemm_real", stmt);

  A.compile(stmt);
  A.assemble();

  // Tensor<double> expected("expected", {B.getDimension(0), L}, Format{Dense, Dense});
  // expected(i,l) = B(i,j) * C(i,k) * D(j,k) * E(j,l);
  // IndexStmt exp = makeReductionNotation(expected.getAssignment());
  // exp = insertTemporaries(exp);
  // exp = exp.concretize();
  // expected.compile(exp);
  // expected.assemble();

  std::chrono::time_point<std::chrono::system_clock> begin, end;
  std::chrono::duration<double> elapsed_seconds;
  double elapsed_mills = 0;

  for (int i = 0; i < iterations; i++) {
    begin = std::chrono::system_clock::now();
    A.compute(stmt);
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - begin;
    elapsed_mills = elapsed_seconds.count() * 1000;
    // begin = clock();
    // expected.compute();
    // end = clock();
    // double elapsed_secs_ref = double(end - begin) / CLOCKS_PER_SEC * 1000;
    // // ASSERT_TENSOR_EQ(expected, A);

    std::cout << elapsed_mills << std::endl;
    // std::cout << elapsed_secs_ref << std::endl;
  }

  std::cout << "workspaces, spmmh_gemm -> execution completed for matrix: " << mat_file << std::endl;

}

TEST(workspaces, default_spmmh_gemm_real) {
  int J = 64;
  int L = 64;

  int nthreads = std::stoi(util::getFromEnv("OMP_NUM_THREADS", "1"));
  taco_set_num_threads(nthreads);
  taco_set_parallel_schedule(ParallelSchedule::Static, 64);

  std::string mat_file = util::getFromEnv("TENSOR_FILE", "");
  int iterations = std::stoi(util::getFromEnv("ITERATIONS", "0"));

  Tensor<double> B = read(mat_file, Format({Dense, Sparse}), true);
  B.setName("B");
  B.pack();

  if (mat_file == "") {
    std::cout << "No tensor file specified!\n";
    return;
  }

  Tensor<double> C("C", {B.getDimension(1), J}, Format{Dense, Dense});
  Tensor<double> D("D", {B.getDimension(1), J}, Format{Dense, Dense});
  for (int k=0; k<B.getDimension(1); k++) {
    for (int j=0; j<J; j++) {
      C.insert({k, j}, (double) j);
      D.insert({k, j}, (double) j);
    }
  }
  C.pack();
  D.pack();
  Tensor<double> E("E", {J, L}, Format{Dense, Dense});
  for (int j=0; j<J; j++) {
    for (int l=0; l<L; l++) {
      E.insert({j, l}, (double) l);
    }
  }
  E.pack();

  // 3 -> A(i,l) = B(i,k) * C(k,j) * D(k,j) * E(j,l) - <SpMMH, GEMM>
  IndexVar i("i"), j("j"), k("k"), l("l");

  Tensor<double> expected("expected", {B.getDimension(0), L}, Format{Dense, Dense});
  expected(i,l) = B(i,k) * C(k,j) * D(k,j) * E(j,l);
  IndexStmt exp = makeReductionNotation(expected.getAssignment());
  exp = insertTemporaries(exp);
  exp = exp.concretize();
  exp = exp.parallelize(i, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
  expected.compile(exp);
  expected.assemble();

  cout << "default stmt: " << exp << endl;
  cout << "default stmt: " << exp << endl;
  printCodeToFile("default_spmmh_gemm_real", exp);

  std::chrono::time_point<std::chrono::system_clock> begin, end;
  std::chrono::duration<double> elapsed_seconds;
  double elapsed_mills = 0;

  for (int i = 0; i< iterations; i++) {
    begin = std::chrono::system_clock::now();
    expected.compute();
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - begin;
    elapsed_mills = elapsed_seconds.count() * 1000;

    std::cout << elapsed_mills << std::endl;
  }

  std::cout << "workspaces, reference_spmmh_gemm -> execution completed for matrix: " << mat_file << std::endl;
}

TEST(workspaces, default_gemm_real) {
  int K = 64;
  int L = 64;

  int nthreads = std::stoi(util::getFromEnv("OMP_NUM_THREADS", "1"));
  taco_set_num_threads(nthreads);
  taco_set_parallel_schedule(ParallelSchedule::Static, 64);

  std::string mat_file = util::getFromEnv("TENSOR_FILE", "");
  int iterations = std::stoi(util::getFromEnv("ITERATIONS", "0"));

  Tensor<double> B = read(mat_file, Format({Dense, Sparse}), true);
  B.setName("B");
  B.pack();

  auto I = B.getDimension(0);
  auto J = B.getDimension(1);

  if (mat_file == "") {
    std::cout << "No tensor file specified!\n";
    return;
  }

  Tensor<double> C("C", {J, K}, Format{Dense, Dense});
  for (int j=0; j<J; j++) {
    for (int k=0; k<K; k++) {
      C.insert({j, k}, (double) k);
    }
  }
  C.pack();
  Tensor<double> D("D", {K, L}, Format{Dense, Dense});
  for (int k=0; k<K; k++) {
    for (int l=0; l<L; l++) {
      D.insert({k, l}, (double) l);
    }
  }
  D.pack();

  Tensor<double> A("A", {I, L}, Format{Dense, Dense});

  // 3 -> A(i,l) = B(i,j) * C(j,k) * D(k,l) - <SpMM, GeMM>
  IndexVar i("i"), j("j"), k("k"), l("l");



	vector<int> path_ = {};

	A(i, l) = B(i, j) * C(j, k) * D(k, l);
	IndexStmt stmt = A.getAssignment().concretize();
	std::cout << stmt << endl;

  stmt = stmt.concretize();
  cout << "final stmt: " << stmt << endl;
  printCodeToFile("spmm_gemm_real", stmt);

  A.compile(stmt);
  A.assemble();

  // Tensor<double> expected("expected", {B.getDimension(0), L}, Format{Dense, Dense});
  // expected(i,l) = B(i,j) * C(i,k) * D(j,k) * E(j,l);
  // IndexStmt exp = makeReductionNotation(expected.getAssignment());
  // exp = insertTemporaries(exp);
  // exp = exp.concretize();
  // expected.compile(exp);
  // expected.assemble();

  std::chrono::time_point<std::chrono::system_clock> begin, end;
  std::chrono::duration<double> elapsed_seconds;
  double elapsed_mills = 0;

  for (int i = 0; i < iterations; i++) {
    begin = std::chrono::system_clock::now();
		A.compute(stmt);
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - begin;
    elapsed_mills = elapsed_seconds.count() * 1000;
    // begin = clock();
    // expected.compute();
    // end = clock();
    // double elapsed_secs_ref = double(end - begin) / CLOCKS_PER_SEC * 1000;
    // // ASSERT_TENSOR_EQ(expected, A);

    std::cout << elapsed_mills << std::endl;
    // std::cout << elapsed_secs_ref << std::endl;
  }

  std::cout << "workspaces, spmm_gemm_willow -> execution completed for matrix: " << mat_file << std::endl;

}

TEST(workspaces, default_spmm_gemm_real) {
  int K = std::stoi(util::getFromEnv("K", "64"));
  int L = std::stoi(util::getFromEnv("L", "64"));

  int nthreads = std::stoi(util::getFromEnv("OMP_NUM_THREADS", "1"));
  taco_set_num_threads(nthreads);
  taco_set_parallel_schedule(ParallelSchedule::Static, 64);

  std::string mat_file = util::getFromEnv("TENSOR_FILE", "");
  int iterations = std::stoi(util::getFromEnv("ITERATIONS", "0"));

  Tensor<double> B = read(mat_file, Format({Dense, Sparse}), true);
  B.setName("B");
  B.pack();

  auto I = B.getDimension(0);
  auto J = B.getDimension(1);

  if (mat_file == "") {
    std::cout << "No tensor file specified!\n";
    return;
  }

  Tensor<double> C("C", {J, K}, Format{Dense, Dense});
  for (int j=0; j<J; j++) {
    for (int k=0; k<K; k++) {
      C.insert({j, k}, (double) k);
    }
  }
  C.pack();
  Tensor<double> D("D", {K, L}, Format{Dense, Dense});
  for (int k=0; k<K; k++) {
    for (int l=0; l<L; l++) {
      D.insert({k, l}, (double) l);
    }
  }
  D.pack();

  Tensor<double> A("A", {I, L}, Format{Dense, Dense});

  // 3 -> A(i,l) = B(i,j) * C(j,k) * D(k,l) - <SpMM, GeMM>
  IndexVar i("i"), j("j"), k("k"), l("l");

	vector<int> path_ = {};

	A(i, l) = B(i, j) * C(j, k) * D(k, l);
	IndexStmt stmt = A.getAssignment().concretize();
	std::cout << stmt << endl;

  stmt = stmt.concretize();
  cout << "final stmt: " << stmt << endl;
  printCodeToFile("spmm_gemm_real", stmt);

  A.compile(stmt);
  A.assemble();

  // Tensor<double> expected("expected", {B.getDimension(0), L}, Format{Dense, Dense});
  // expected(i,l) = B(i,j) * C(i,k) * D(j,k) * E(j,l);
  // IndexStmt exp = makeReductionNotation(expected.getAssignment());
  // exp = insertTemporaries(exp);
  // exp = exp.concretize();
  // expected.compile(exp);
  // expected.assemble();

  std::chrono::time_point<std::chrono::system_clock> begin, end;
  std::chrono::duration<double> elapsed_seconds;
  double elapsed_mills = 0;

  for (int i = 0; i < iterations; i++) {
    begin = std::chrono::system_clock::now();
		A.compute(stmt);
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - begin;
    elapsed_mills = elapsed_seconds.count() * 1000;
    // begin = clock();
    // expected.compute();
    // end = clock();
    // double elapsed_secs_ref = double(end - begin) / CLOCKS_PER_SEC * 1000;
    // // ASSERT_TENSOR_EQ(expected, A);

    std::cout << elapsed_mills << std::endl;
    // std::cout << elapsed_secs_ref << std::endl;
  }

  std::cout << "K=" << K << ", L=" << L 
    << ", workspaces, default_spmm_gemm_real -> execution completed for matrix: " << mat_file << std::endl;

}

TEST(workspaces, spmm_gemm_real) {
  int K = std::stoi(util::getFromEnv("K", "64"));
  int L = std::stoi(util::getFromEnv("L", "64"));

  int nthreads = std::stoi(util::getFromEnv("OMP_NUM_THREADS", "1"));
  taco_set_num_threads(nthreads);
  taco_set_parallel_schedule(ParallelSchedule::Static, 64);

  std::string mat_file = util::getFromEnv("TENSOR_FILE", "");
  int iterations = std::stoi(util::getFromEnv("ITERATIONS", "0"));

  Tensor<double> B = read(mat_file, Format({Dense, Sparse}), true);
  B.setName("B");
  B.pack();

  auto I = B.getDimension(0);
  auto J = B.getDimension(1);

  if (mat_file == "") {
    std::cout << "No tensor file specified!\n";
    return;
  }

  Tensor<double> C("C", {J, K}, Format{Dense, Dense});
  for (int j=0; j<J; j++) {
    for (int k=0; k<K; k++) {
      C.insert({j, k}, (double) k);
    }
  }
  C.pack();
  Tensor<double> D("D", {K, L}, Format{Dense, Dense});
  for (int k=0; k<K; k++) {
    for (int l=0; l<L; l++) {
      D.insert({k, l}, (double) l);
    }
  }
  D.pack();

  Tensor<double> A("A", {I, L}, Format{Dense, Dense});

  // 3 -> A(i,l) = B(i,j) * C(j,k) * D(k,l) - <SpMM, GeMM>
  IndexVar i("i"), j("j"), k("k"), l("l");

/* BEGIN spmm_gemm_real TEST */

	vector<int> path_ = {};

	Tensor<double> _A("_A", {I, K}, Format{Dense, Dense});
	_A(i, k) = B(i, j) * C(j, k);
	IndexStmt stmt__A = _A.getAssignment().concretize();
	stmt__A = stmt__A
		.reorder(path_, {i,j,k})
		;
	stmt__A = stmt__A.concretize();
	_A.compile(stmt__A);
	_A.assemble();

	A(i, l) = _A(i, k) * D(k, l);
	IndexStmt stmt = A.getAssignment().concretize();
	std::cout << stmt << endl;
	stmt = stmt
		.reorder(path_, {i,l,k})
		;
	/* END spmm_gemm_real TEST */

  stmt = stmt.concretize();
  cout << "final stmt: " << stmt << endl;
  printCodeToFile("spmm_gemm_real", stmt);

  A.compile(stmt);
  A.assemble();

  // Tensor<double> expected("expected", {B.getDimension(0), L}, Format{Dense, Dense});
  // expected(i,l) = B(i,j) * C(i,k) * D(j,k) * E(j,l);
  // IndexStmt exp = makeReductionNotation(expected.getAssignment());
  // exp = insertTemporaries(exp);
  // exp = exp.concretize();
  // expected.compile(exp);
  // expected.assemble();

  std::chrono::time_point<std::chrono::system_clock> begin, end;
  std::chrono::duration<double> elapsed_seconds;
  double elapsed_mills = 0;

  for (int i = 0; i < iterations; i++) {
    begin = std::chrono::system_clock::now();
    /* BEGIN spmm_gemm_real_execute TEST */
		_A.compute(stmt__A);
		A.compute(stmt);
		/* END spmm_gemm_real_execute TEST */
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - begin;
    elapsed_mills = elapsed_seconds.count() * 1000;
    // begin = clock();
    // expected.compute();
    // end = clock();
    // double elapsed_secs_ref = double(end - begin) / CLOCKS_PER_SEC * 1000;
    // // ASSERT_TENSOR_EQ(expected, A);

    std::cout << elapsed_mills << std::endl;
    // std::cout << elapsed_secs_ref << std::endl;
  }

  std::cout << "K=" << K << ", L=" << L 
    << ", workspaces, spmm_gemm_real -> execution completed for matrix: " << mat_file << std::endl;

}

TEST(workspaces, spmm_gemm_willow) {
  int K = 64;
  int L = 64;

  int nthreads = std::stoi(util::getFromEnv("OMP_NUM_THREADS", "1"));
  taco_set_num_threads(nthreads);
  taco_set_parallel_schedule(ParallelSchedule::Static, 64);

  std::string mat_file = util::getFromEnv("TENSOR_FILE", "");
  int iterations = std::stoi(util::getFromEnv("ITERATIONS", "0"));

  Tensor<double> B = read(mat_file, Format({Dense, Sparse}), true);
  B.setName("B");
  B.pack();

  auto I = B.getDimension(0);
  auto J = B.getDimension(1);

  if (mat_file == "") {
    std::cout << "No tensor file specified!\n";
    return;
  }

  Tensor<double> C("C", {J, K}, Format{Dense, Dense});
  for (int j=0; j<J; j++) {
    for (int k=0; k<K; k++) {
      C.insert({j, k}, (double) k);
    }
  }
  C.pack();
  Tensor<double> D("D", {K, L}, Format{Dense, Dense});
  for (int k=0; k<K; k++) {
    for (int l=0; l<L; l++) {
      D.insert({k, l}, (double) l);
    }
  }
  D.pack();

  Tensor<double> A("A", {I, L}, Format{Dense, Dense});

  // 3 -> A(i,l) = B(i,j) * C(j,k) * D(k,l) - <SpMM, GeMM>
  IndexVar i("i"), j("j"), k("k"), l("l");

  /* BEGIN spmm_gemm_willow TEST */

	vector<int> path_ = {};
    vector<int> path1_ = {1};

	A(i, l) = B(i, j) * C(j, k) * D(k, l);
	IndexStmt stmt = A.getAssignment().concretize();
	std::cout << stmt << endl;

	stmt = stmt
		.reorder(path_, {i,l,k,j})
		.loopfuse(2, true, path_)
        .reorder(path1_, {l,k})
		;

  /* END spmm_gemm_willow TEST */

  stmt = stmt.concretize();
  cout << "final stmt: " << stmt << endl;
  printCodeToFile("spmm_gemm_real", stmt);

  A.compile(stmt);
  A.assemble();

  // Tensor<double> expected("expected", {B.getDimension(0), L}, Format{Dense, Dense});
  // expected(i,l) = B(i,j) * C(i,k) * D(j,k) * E(j,l);
  // IndexStmt exp = makeReductionNotation(expected.getAssignment());
  // exp = insertTemporaries(exp);
  // exp = exp.concretize();
  // expected.compile(exp);
  // expected.assemble();

  std::chrono::time_point<std::chrono::system_clock> begin, end;
  std::chrono::duration<double> elapsed_seconds;
  double elapsed_mills = 0;

  for (int i = 0; i < iterations; i++) {
    begin = std::chrono::system_clock::now();
		A.compute(stmt);
    
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - begin;
    elapsed_mills = elapsed_seconds.count() * 1000;
    // begin = clock();
    // expected.compute();
    // end = clock();
    // double elapsed_secs_ref = double(end - begin) / CLOCKS_PER_SEC * 1000;
    // // ASSERT_TENSOR_EQ(expected, A);

    std::cout << elapsed_mills << std::endl;
    // std::cout << elapsed_secs_ref << std::endl;
  }

  std::cout << "workspaces, spmm_gemm_willow -> execution completed for matrix: " << mat_file << std::endl;

}

TEST(workspaces, spttm_spttm_willow) {
  int L = 16;
  int M = 16;

  int nthreads = std::stoi(util::getFromEnv("OMP_NUM_THREADS", "1"));
  taco_set_num_threads(nthreads);
  taco_set_parallel_schedule(ParallelSchedule::Static, 64);

  std::string mat_file = util::getFromEnv("TENSOR_FILE", "");
  int iterations = std::stoi(util::getFromEnv("ITERATIONS", "0"));

  if (mat_file == "") {
    std::cout << "No tensor file specified!\n";
    return;
  }

  Tensor<double> B = read(mat_file, Format({Dense, Sparse, Sparse}), true);
  B.setName("B");
  B.pack();

  // A(i, j, m) = B(i, j, k) * C(k, l) * D(l, m);
  Tensor<double> C("C", {B.getDimension(2), L}, Format{Dense, Dense});
  for (int i=0; i<B.getDimension(2); i++) {
    for (int l=0; l<L; l++) {
      C.insert({i, l}, (double) i);
    }
  }
  C.pack();
  Tensor<double> D("D", {L, M}, Format{Dense, Dense});
  for (int j=0; j<L; j++) {
    for (int m=0; m<M; m++) {
      D.insert({j, m}, (double) j);
    }
  }
  D.pack();

  Tensor<double> A("A", {B.getDimension(0), B.getDimension(1), M}, Format{Dense, Sparse, Dense});

  // 5 -> A(i,l,m) = B(i,j,k) * C(j,l) * D(k,m) - <SpTTM, TTM>
  IndexVar i("i"), j("j"), k("k"), l("l"), m("m"), n("n");

  // A(i,l,m) = B(i,j,k) * C(j,l) * D(k,m);
  // IndexStmt stmt = A.getAssignment().concretize();
  // std::cout << stmt << endl;

	/* BEGIN spttm_spttm_willow TEST */

	A(i, j, m) = B(i, j, k) * C(k, l) * D(l, m);
	
	IndexStmt stmt = A.getAssignment().concretize();
	std::cout << stmt << endl;
	
	vector<int> path0;
    vector<int> path1 = {0};
	stmt = stmt
		.reorder({i, j, k, l, m})
		.loopfuse(2, true, path0)
        .reorder(path1, {k, l})
		;

	/* END spttm_spttm_willow TEST */

  stmt = stmt.concretize();
  cout << "final stmt: " << stmt << endl;
  printCodeToFile("spttm_spttm_willow", stmt);

  A.compile(stmt.concretize());
  A.assemble();

  Tensor<double> expected("expected", {B.getDimension(0), B.getDimension(1), M}, Format{Dense, Sparse, Dense});
  expected(i,j,m) = B(i,j,k) * C(k,l) * D(l,m);
  expected.compile();
  expected.assemble();

  IndexStmt stmt2 = expected.getAssignment().concretize();
  printCodeToFile("reference_spttm_spttm_real", stmt2);

  std::chrono::time_point<std::chrono::system_clock> begin, end;
  std::chrono::duration<double> elapsed_seconds;
  double elapsed_mills = 0;

  for (int i=0; i < iterations; i++) {
    begin = std::chrono::system_clock::now();
    A.compute(stmt);
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - begin;
    elapsed_mills = elapsed_seconds.count() * 1000;
    // begin = clock();
    // expected.compute();
    // end = clock();
    // double elapsed_secs_ref = double(end - begin) / CLOCKS_PER_SEC * 1000;
    // // ASSERT_TENSOR_EQ(expected, A);

    std::cout << elapsed_mills << std::endl;
    // std::cout << elapsed_secs_ref << std::endl;
  }

  std::cout << "workspaces, spttm_spttm_willow -> execution completed for matrix: " << mat_file << std::endl;

}