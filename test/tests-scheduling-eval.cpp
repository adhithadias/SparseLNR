#include "util.h"

const IndexVar i("i"), j("j"), k("k"), l("l"), m("m"), n("n");
int WARP_SIZE = 32;

IndexStmt scheduleSpMVCPU(IndexStmt stmt, int CHUNK_SIZE=16) {
  IndexVar i0("i0"), i1("i1"), kpos("kpos"), kpos0("kpos0"), kpos1("kpos1");
  return stmt.split(i, i0, i1, CHUNK_SIZE)
          .reorder({i0, i1, j})
          .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
}

IndexStmt scheduleSpMVISPC(IndexStmt stmt, int CHUNK_SIZE=16) {
  IndexVar i0("i0"), i1("i1"), kpos("kpos"), kpos0("kpos0"), kpos1("kpos1");
  // return stmt;
  return stmt.split(i, i0, i1, CHUNK_SIZE)
          .reorder({i0, i1, j})
          .parallelize(i0, ParallelUnit::CPUSimd, OutputRaceStrategy::NoRaces);
}

IndexStmt scheduleSpMMCPU(IndexStmt stmt, Tensor<double> A, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  IndexVar i0("i0"), i1("i1"), kbounded("kbounded"), k0("k0"), k1("k1"), jpos("jpos"), jpos0("jpos0"), jpos1("jpos1");
  return stmt.split(i, i0, i1, CHUNK_SIZE)
          .pos(j, jpos, A(i,j))
          .split(jpos, jpos0, jpos1, UNROLL_FACTOR)
          .reorder({i0, i1, jpos0, k, jpos1})
          .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
          .parallelize(k, ParallelUnit::CPUVector, OutputRaceStrategy::IgnoreRaces);
}

IndexStmt scheduleSpMMISPC1(IndexStmt stmt, Tensor<double> A, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  IndexVar i0("i0"), i1("i1"), kbounded("kbounded"), k0("k0"), k1("k1"), jpos("jpos"), jpos0("jpos0"), jpos1("jpos1");
  return stmt.split(i, i0, i1, CHUNK_SIZE)
          .pos(j, jpos, A(i,j))
          .split(jpos, jpos0, jpos1, UNROLL_FACTOR)
          .reorder({i0, i1, jpos0, k, jpos1})
          // .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
          .parallelize(k, ParallelUnit::CPUSimd, OutputRaceStrategy::IgnoreRaces);
}

IndexStmt scheduleSpMMISPCOMP1(IndexStmt stmt, Tensor<double> A, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  IndexVar i0("i0"), i1("i1"), kbounded("kbounded"), k0("k0"), k1("k1"), jpos("jpos"), jpos0("jpos0"), jpos1("jpos1");
  return stmt.split(i, i0, i1, CHUNK_SIZE)
          .pos(j, jpos, A(i,j))
          .split(jpos, jpos0, jpos1, UNROLL_FACTOR)
          .reorder({i0, i1, jpos0, k, jpos1})
          .parallelize(i0, ParallelUnit::CPUSpmd, OutputRaceStrategy::NoRaces)
          .parallelize(k, ParallelUnit::CPUSimd, OutputRaceStrategy::IgnoreRaces);
}

IndexStmt scheduleSpMMISPC1_2(IndexStmt stmt, Tensor<double> A, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  IndexVar i0("i0"), i1("i1"), kbounded("kbounded"), k0("k0"), k1("k1"), jpos("jpos"), jpos0("jpos0"), jpos1("jpos1");
  return stmt.split(i, i0, i1, CHUNK_SIZE)
          .pos(j, jpos, A(i,j))
          .split(jpos, jpos0, jpos1, UNROLL_FACTOR)
          .reorder({i0, i1, jpos0, k, jpos1})
          // .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
          .parallelize(i0, ParallelUnit::CPUSimd, OutputRaceStrategy::IgnoreRaces);
}

IndexStmt scheduleSpMMISPC1_3(IndexStmt stmt, Tensor<double> A, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  IndexVar i0("i0"), i1("i1"), kbounded("kbounded"), k0("k0"), k1("k1"), jpos("jpos"), jpos0("jpos0"), jpos1("jpos1");
  return stmt.split(i, i0, i1, CHUNK_SIZE)
          .pos(j, jpos, A(i,j))
          .split(jpos, jpos0, jpos1, UNROLL_FACTOR)
          .reorder({i0, i1, jpos0, k, jpos1})
          // .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
          .parallelize(i1, ParallelUnit::CPUSimd, OutputRaceStrategy::IgnoreRaces);
}

IndexStmt scheduleSpMMISPC2(IndexStmt stmt, Tensor<double> A, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  IndexVar i0("i0"), i1("i1"), kbounded("kbounded"), k0("k0"), k1("k1"), jpos("jpos"), jpos0("jpos0"), jpos1("jpos1");
  return stmt
          .parallelize(k, ParallelUnit::CPUSimd, OutputRaceStrategy::IgnoreRaces);
}

IndexStmt scheduleSpMMISPC2_2(IndexStmt stmt, Tensor<double> A, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  IndexVar i0("i0"), i1("i1"), kbounded("kbounded"), k0("k0"), k1("k1"), jpos("jpos"), jpos0("jpos0"), jpos1("jpos1");
  return stmt
          .parallelize(i, ParallelUnit::CPUSimd, OutputRaceStrategy::IgnoreRaces);
}

IndexStmt scheduleSpMMISPC3(IndexStmt stmt, Tensor<double> A, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  IndexVar i0("i0"), i1("i1"), kbounded("kbounded"), k0("k0"), k1("k1"), jpos("jpos"), jpos0("jpos0"), jpos1("jpos1");
  return stmt
          // .split(i, i0, i1, CHUNK_SIZE)
          // .pos(j, jpos, A(i,j))
          // .split(jpos, jpos0, jpos1, UNROLL_FACTOR)
          .reorder({j, k})
          // .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
          .parallelize(k, ParallelUnit::CPUSimd, OutputRaceStrategy::IgnoreRaces);
}

IndexStmt scheduleSpMMISPC3_2(IndexStmt stmt, Tensor<double> A, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  IndexVar i0("i0"), i1("i1"), kbounded("kbounded"), k0("k0"), k1("k1"), jpos("jpos"), jpos0("jpos0"), jpos1("jpos1");
  return stmt
          // .split(i, i0, i1, CHUNK_SIZE)
          // .pos(j, jpos, A(i,j))
          // .split(jpos, jpos0, jpos1, UNROLL_FACTOR)
          .reorder({j, k})
          // .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
          .parallelize(i, ParallelUnit::CPUSimd, OutputRaceStrategy::IgnoreRaces);
}

IndexStmt scheduleSpGEMMCPU(IndexStmt stmt, bool doPrecompute) {
  Assignment assign = stmt.as<Forall>().getStmt().as<Forall>().getStmt()
                          .as<Forall>().getStmt().as<Assignment>();
  TensorVar result = assign.getLhs().getTensorVar();

  stmt = reorderLoopsTopologically(stmt);
  if (doPrecompute) {
    IndexVar j = assign.getLhs().getIndexVars()[1];
    TensorVar w("w", Type(result.getType().getDataType(), 
                {result.getType().getShape().getDimension(1)}), taco::dense);
    stmt = stmt.precompute(assign.getRhs(), j, j, w);
  }
  stmt = stmt.assemble(result, AssembleStrategy::Insert, true);

  IndexVar qi = stmt.as<Assemble>().getQueries().as<Forall>().getIndexVar();
  stmt = stmt.parallelize(i, ParallelUnit::CPUThread,
                          OutputRaceStrategy::NoRaces)
             .parallelize(qi, ParallelUnit::CPUThread,
                          OutputRaceStrategy::NoRaces);

  return stmt;
}

IndexStmt scheduleSpAddCPU(IndexStmt stmt) {
  IndexStmt body = stmt.as<Forall>().getStmt().as<Forall>().getStmt();
  if (isa<Forall>(body)) {
    body = body.as<Forall>().getStmt();
  }
  Assignment assign = body.as<Assignment>();
  TensorVar result = assign.getLhs().getTensorVar();

  stmt = reorderLoopsTopologically(stmt);
  stmt = stmt.assemble(result, AssembleStrategy::Insert, true);

  IndexVar qi = stmt.as<Assemble>().getQueries().as<Forall>().getIndexVar();
  stmt = stmt.parallelize(i, ParallelUnit::CPUThread,
                          OutputRaceStrategy::NoRaces)
             .parallelize(qi, ParallelUnit::CPUThread,
                          OutputRaceStrategy::NoRaces);

  return stmt;
}

IndexStmt scheduleSDDMMCPU(IndexStmt stmt, Tensor<double> B, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  IndexVar i0("i0"), i1("i1"), kpos("kpos"), kpos0("kpos0"), kpos1("kpos1");
  return stmt.split(i, i0, i1, CHUNK_SIZE)
          .pos(k, kpos, B(i,k))
          .split(kpos, kpos0, kpos1, UNROLL_FACTOR)
          .reorder({i0, i1, kpos0, j, kpos1})
          .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
          .parallelize(kpos1, ParallelUnit::CPUVector, OutputRaceStrategy::ParallelReduction);
}

IndexStmt scheduleSDDMMCSRCPU(IndexStmt stmt, Tensor<double> B, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  IndexVar i0("i0"), i1("i1"), kpos("kpos"), kpos0("kpos0"), kpos1("kpos1");
  return stmt;
  // return stmt.split(i, i0, i1, CHUNK_SIZE)
  //         .pos(k, kpos, B(i,k))
  //         .split(kpos, kpos0, kpos1, UNROLL_FACTOR)
  //         .reorder({i0, i1, kpos0, j, kpos1});
          // .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
          // .parallelize(k, ParallelUnit::CPUVector, OutputRaceStrategy::IgnoreRaces);
}

IndexStmt scheduleSDDMM2CPU(IndexStmt stmt, Tensor<double> B, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  IndexVar i0("i0"), i1("i1"), jpos("jpos"), jpos0("jpos0"), jpos1("jpos1");
  return stmt.split(i, i0, i1, CHUNK_SIZE)
          .pos(j, jpos, B(i,j))
          .split(jpos, jpos0, jpos1, UNROLL_FACTOR)
          .reorder({i0, i1, jpos0, k, jpos1})
          .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
          .parallelize(jpos1, ParallelUnit::CPUVector, OutputRaceStrategy::ParallelReduction);
}

IndexStmt scheduleSDDMMISPC(IndexStmt stmt, Tensor<double> B, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  IndexVar i0("i0"), i1("i1"), kpos("kpos"), kpos0("kpos0"), kpos1("kpos1");
  return stmt.split(i, i0, i1, CHUNK_SIZE)
          .pos(k, kpos, B(i,k))
          .split(kpos, kpos0, kpos1, UNROLL_FACTOR)
          .reorder({i0, i1, kpos0, j, kpos1})
          // .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
          .parallelize(kpos1, ParallelUnit::CPUSimd, OutputRaceStrategy::ParallelReduction);
}

IndexStmt scheduleSDDMM2ISPC(IndexStmt stmt, Tensor<double> B, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  IndexVar i0("i0"), i1("i1"), jpos("jpos"), jpos0("jpos0"), jpos1("jpos1");
  return stmt.split(i, i0, i1, CHUNK_SIZE)
          .pos(j, jpos, B(i,j))
          .split(jpos, jpos0, jpos1, UNROLL_FACTOR)
          .reorder({i0, i1, jpos0, k, jpos1})
          // .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
          .parallelize(jpos1, ParallelUnit::CPUSimd, OutputRaceStrategy::ParallelReduction);
}

IndexStmt scheduleSDDMMISPC1(IndexStmt stmt, Tensor<double> B, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  IndexVar i0("i0"), i1("i1"), kpos("kpos"), kpos0("kpos0"), kpos1("kpos1");
  return stmt.split(i, i0, i1, CHUNK_SIZE)
          .pos(k, kpos, B(i,k))
          .split(kpos, kpos0, kpos1, UNROLL_FACTOR)
          .reorder({i0, i1, kpos0, j, kpos1})
          .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
          .parallelize(kpos1, ParallelUnit::CPUSimd, OutputRaceStrategy::ParallelReduction);
}

IndexStmt scheduleSDDMMISPC2(IndexStmt stmt, Tensor<double> B, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  IndexVar i0("i0"), i1("i1"), kpos("kpos"), kpos0("kpos0"), kpos1("kpos1");
  return stmt;
          // .split(i, i0, i1, CHUNK_SIZE)
          // .pos(k, kpos, B(i,k))
          // .split(kpos, kpos0, kpos1, UNROLL_FACTOR)
          // .reorder({i0, i1, kpos0, j, kpos1})
          // .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
          // .parallelize(kpos1, ParallelUnit::CPUSimd, OutputRaceStrategy::ParallelReduction);
}

IndexStmt scheduleTTVCPU(IndexStmt stmt, Tensor<double> B, int CHUNK_SIZE=16) {
  IndexVar f("f"), fpos("fpos"), chunk("chunk"), fpos2("fpos2");
  return stmt.fuse(i, j, f)
          .pos(f, fpos, B(i,j,k))
          .split(fpos, chunk, fpos2, CHUNK_SIZE)
          .reorder({chunk, fpos2, k})
          .parallelize(chunk, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
}

IndexStmt scheduleTTVISPC(IndexStmt stmt, Tensor<double> B, int CHUNK_SIZE=16) {
  IndexVar f("f"), fpos("fpos"), chunk("chunk"), fpos2("fpos2");
  // return stmt;
  return stmt.fuse(i, j, f)
          .pos(f, fpos, B(i,j,k))
          .split(fpos, chunk, fpos2, CHUNK_SIZE)
          .reorder({chunk, fpos2, k})
          .parallelize(chunk, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
}

IndexStmt scheduleTTVCPUCSR(IndexStmt stmt) {
  TensorVar result = stmt.as<Forall>().getStmt().as<Forall>().getStmt()
                         .as<Forall>().getStmt().as<Assignment>().getLhs()
                         .getTensorVar();
  return stmt.assemble(result, AssembleStrategy::Insert)
             .parallelize(i, ParallelUnit::CPUThread, 
                          OutputRaceStrategy::NoRaces);
}

IndexStmt scheduleTTVCPUCSR_ST(IndexStmt stmt) {
  TensorVar result = stmt.as<Forall>().getStmt().as<Forall>().getStmt()
                         .as<Forall>().getStmt().as<Assignment>().getLhs()
                         .getTensorVar();
  return stmt.assemble(result, AssembleStrategy::Insert);
}

IndexStmt scheduleTTVISPCCSR(IndexStmt stmt) {
  TensorVar result = stmt.as<Forall>().getStmt().as<Forall>().getStmt()
                         .as<Forall>().getStmt().as<Assignment>().getLhs()
                         .getTensorVar();
  return stmt.assemble(result, AssembleStrategy::Insert)
             .parallelize(i, ParallelUnit::CPUSimd, OutputRaceStrategy::NoRaces);
}

IndexStmt scheduleTTVISPCCSR2(IndexStmt stmt) {
  return stmt;
}

IndexStmt scheduleTTMCPU(IndexStmt stmt, Tensor<double> B, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  IndexVar f("f"), fpos("fpos"), chunk("chunk"), fpos2("fpos2"), kpos("kpos"), kpos1("kpos1"), kpos2("kpos2");
  return stmt.fuse(i, j, f)
          .pos(f, fpos, B(i,j,k))
          .split(fpos, chunk, fpos2, CHUNK_SIZE)
          .pos(k, kpos, B(i,j,k))
          .split(kpos, kpos1, kpos2, UNROLL_FACTOR)
          .reorder({chunk, fpos2, kpos1, l, kpos2})
          .parallelize(chunk, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
          .parallelize(kpos2, ParallelUnit::CPUVector, OutputRaceStrategy::ParallelReduction);;
}

IndexStmt scheduleMTTKRPCPU(IndexStmt stmt, Tensor<double> B, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  IndexVar i1("i1"), i2("i2");
  IndexExpr precomputeExpr = stmt.as<Forall>().getStmt().as<Forall>().getStmt()
                                 .as<Forall>().getStmt().as<Forall>().getStmt()
                                 .as<Assignment>().getRhs().as<Mul>().getA();
  TensorVar w("w", Type(Float64, {Dimension(j)}), taco::dense);
  return stmt.split(i, i1, i2, CHUNK_SIZE)
          .reorder({i1, i2, k, l, j})
          .precompute(precomputeExpr, j, j, w)
          .parallelize(i1, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
}

IndexStmt scheduleMTTKRPCPU_ST(IndexStmt stmt, Tensor<double> B, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  IndexVar i1("i1"), i2("i2");
  IndexExpr precomputeExpr = stmt.as<Forall>().getStmt().as<Forall>().getStmt()
                                 .as<Forall>().getStmt().as<Forall>().getStmt()
                                 .as<Assignment>().getRhs().as<Mul>().getA();
  TensorVar w("w", Type(Float64, {Dimension(j)}), taco::dense);
  return stmt.split(i, i1, i2, CHUNK_SIZE)
          .reorder({i1, i2, k, l, j})
          .precompute(precomputeExpr, j, j, w);
          // .parallelize(j, ParallelUnit::CPUVector, OutputRaceStrategy::Atomics); // gives error when lowering for IgnoreRaces, NoRaces and Atomics
          // .parallelize(i1, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
}

IndexStmt scheduleMTTKRPISPC(IndexStmt stmt, Tensor<double> B, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  IndexVar i1("i1"), i2("i2");
  IndexExpr precomputeExpr = stmt.as<Forall>().getStmt().as<Forall>().getStmt()
                                 .as<Forall>().getStmt().as<Forall>().getStmt()
                                 .as<Assignment>().getRhs().as<Mul>().getA();
  TensorVar w("w", Type(Float64, {Dimension(j)}), taco::dense);
  return stmt.split(i, i1, i2, CHUNK_SIZE)
          .reorder({i1, i2, k, l, j})
          .precompute(precomputeExpr, j, j, w)
          .parallelize(j, ParallelUnit::CPUSimd, OutputRaceStrategy::NoRaces);
}

IndexStmt scheduleMTTKRPPrecomputedCPU(IndexStmt stmt, Tensor<double> B, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  IndexVar i1("i1"), i2("i2"), j_pre("j_pre");
  return stmt.split(i, i1, i2, CHUNK_SIZE)
          .parallelize(i1, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
}

IndexStmt scheduleMTTKRPPrecomputedCPU_ST(IndexStmt stmt, Tensor<double> B, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  IndexVar i1("i1"), i2("i2"), j_pre("j_pre");
  return stmt.split(i, i1, i2, CHUNK_SIZE);
}

IndexStmt scheduleMTTKRPPrecomputedISPC_ST(IndexStmt stmt, Tensor<double> B, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  IndexVar i1("i1"), i2("i2"), j_pre("j_pre");
  return stmt.parallelize(j, ParallelUnit::CPUSimd, OutputRaceStrategy::NoRaces);
}

IndexStmt scheduleMTTKRP4CPU(IndexStmt stmt, Tensor<double> B, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  IndexVar i1("i1"), i2("i2");
  return stmt.split(i, i1, i2, CHUNK_SIZE)
          .reorder({i1, i2, k, l, m, j})
          .parallelize(i1, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
}

IndexStmt scheduleMTTKRP4CPU_ST(IndexStmt stmt, Tensor<double> B, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  IndexVar i1("i1"), i2("i2");
  return stmt.split(i, i1, i2, CHUNK_SIZE)
          .reorder({i1, i2, k, l, m, j});
}

IndexStmt scheduleMTTKRP4ISPC_ST(IndexStmt stmt, Tensor<double> B, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  IndexVar i1("i1"), i2("i2");
  return stmt.split(i, i1, i2, CHUNK_SIZE)
          .reorder({i1, i2, k, l, m, j})
          .parallelize(j, ParallelUnit::CPUSimd, OutputRaceStrategy::NoRaces);
}

IndexStmt scheduleMTTKRP5CPU(IndexStmt stmt, Tensor<double> B, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  IndexVar i1("i1"), i2("i2");
  return stmt.split(i, i1, i2, CHUNK_SIZE)
          .reorder({i1, i2, k, l, m, n, j})
          .parallelize(i1, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
}

IndexStmt scheduleSpMVGPU(IndexStmt stmt, Tensor<double> A, IndexExpr precomputedExpr, int NNZ_PER_THREAD=8, int BLOCK_SIZE=256) {
  int NNZ_PER_WARP = NNZ_PER_THREAD * WARP_SIZE;
  int NNZ_PER_TB = NNZ_PER_THREAD * BLOCK_SIZE;
  IndexVar f("f"), fpos("fpos"), fpos1("fpos1"), fpos2("fpos2"), block("block"), warp("warp"), thread("thread"), thread_nz("thread_nz"), thread_nz_pre("thread_nz_pre");
  TensorVar precomputed("precomputed", Type(Float64, {Dimension(thread_nz)}), taco::dense);
  return stmt.fuse(i, j, f)
          .pos(f, fpos, A(i, j))
          .split(fpos, block, fpos1, NNZ_PER_TB)
          .split(fpos1, warp, fpos2, NNZ_PER_WARP)
          .split(fpos2, thread, thread_nz, NNZ_PER_THREAD)
          .reorder({block, warp, thread, thread_nz})
          .precompute(precomputedExpr, thread_nz, thread_nz_pre, precomputed)
          .unroll(thread_nz_pre, NNZ_PER_THREAD)
          .parallelize(block, ParallelUnit::GPUBlock, OutputRaceStrategy::IgnoreRaces)
          .parallelize(warp, ParallelUnit::GPUWarp, OutputRaceStrategy::IgnoreRaces)
          .parallelize(thread, ParallelUnit::GPUThread, OutputRaceStrategy::Atomics);
}

IndexStmt scheduleSpMVRowsGPU(IndexStmt stmt, Tensor<double> A, IndexExpr precomputedExpr, int ROWS_PER_WARP=1, int BLOCK_SIZE=256) {
  int ROWS_PER_TB = ROWS_PER_WARP * BLOCK_SIZE;
  IndexVar block("block"), warp("warp"), thread("thread"), thread_nz("thread_nz"), i1("i1"), jpos("jpos"), block_row("block_row"), warp_row("warp_row");
  TensorVar precomputed("precomputed", Type(Float64, {Dimension(thread_nz)}), taco::dense);
  return stmt.split(i, block, block_row, ROWS_PER_TB)
          .split(block_row, warp_row, warp, BLOCK_SIZE / WARP_SIZE)
          .pos(j, jpos, A(i, j))
          .split(jpos, thread_nz, thread, WARP_SIZE)
          .reorder({block, warp, warp_row, thread, thread_nz})
          .parallelize(block, ParallelUnit::GPUBlock, OutputRaceStrategy::IgnoreRaces)
          .parallelize(warp, ParallelUnit::GPUWarp, OutputRaceStrategy::IgnoreRaces)
          .parallelize(thread, ParallelUnit::GPUThread, OutputRaceStrategy::Temporary);
}

IndexStmt scheduleSpMVThreadPerRowGPU(IndexStmt stmt, Tensor<double> A, IndexExpr precomputedExpr, int BLOCK_SIZE=256) {
  int ROWS_PER_TB = BLOCK_SIZE;
  IndexVar block("block"), warp("warp"), thread("thread"), thread_nz("thread_nz"), i1("i1"), jpos("jpos"), block_row("block_row"), warp_row("warp_row");
  return stmt.split(i, block, thread, ROWS_PER_TB)
          .parallelize(block, ParallelUnit::GPUBlock, OutputRaceStrategy::NoRaces)
          .parallelize(thread, ParallelUnit::GPUThread, OutputRaceStrategy::NoRaces);
}

IndexStmt scheduleSpMVSplitPosGPU(IndexStmt stmt, Tensor<double> A, IndexExpr precomputedExpr, int NNZ_PER_THREAD=8, int BLOCK_SIZE=256) {
  int NNZ_PER_WARP = NNZ_PER_THREAD * WARP_SIZE;
  int NNZ_PER_TB = NNZ_PER_THREAD * BLOCK_SIZE;
  IndexVar f("f"), fpos("fpos"), fpos1("fpos1"), fpos2("fpos2"), block("block"), warp("warp"), thread("thread"), thread_nz("thread_nz"), thread_nz_pre("thread_nz_pre");
  TensorVar precomputed("precomputed", Type(Float64, {Dimension(thread_nz)}), taco::dense);
  return stmt.fuse(i, j, f)
          .pos(f, fpos, A(i, j))
          .split(fpos, block, fpos1, NNZ_PER_TB)
          .split(fpos1, warp, fpos2, NNZ_PER_WARP)
          .split(fpos2, thread, thread_nz, NNZ_PER_THREAD)
          .reorder({block, warp, thread, thread_nz})
          .parallelize(block, ParallelUnit::GPUBlock, OutputRaceStrategy::IgnoreRaces)
          .parallelize(warp, ParallelUnit::GPUWarp, OutputRaceStrategy::IgnoreRaces)
          .parallelize(thread, ParallelUnit::GPUThread, OutputRaceStrategy::Atomics);
}

IndexStmt scheduleSpMMGPU(IndexStmt stmt, Tensor<double> A, IndexExpr precomputedExpr, int NNZ_PER_WARP=8, int BLOCK_SIZE=256) {
  int NNZ_PER_TB = NNZ_PER_WARP * (BLOCK_SIZE / WARP_SIZE);
  IndexVar f("f"), fpos("fpos"), block("block"), fpos1("fpos1"), warp("warp"), nnz("nnz"), nnz_pre("nnz_pre");
  IndexVar dense_val_unbounded("dense_val_unbounded"), dense_val("dense_val"), thread("thread");
  IndexVar thread_nz("thread_nz");
  TensorVar precomputed("precomputed", Type(Float64, {Dimension(nnz)}), taco::dense);
  return stmt.reorder({i, j, k})
          .fuse(i, j, f)
          .pos(f, fpos, A(i, j))
          .split(fpos, block, fpos1, NNZ_PER_TB)
          .split(fpos1, warp, nnz, NNZ_PER_WARP)
          .split(k, dense_val_unbounded, thread, WARP_SIZE)
          .reorder({block, warp, thread, dense_val_unbounded, nnz})
          //.precompute(precomputedExpr, nnz, nnz, precomputed)
          .bound(dense_val_unbounded, dense_val, 4, BoundType::MaxExact)
          //.unroll(dense_val, 4)
          .parallelize(block, ParallelUnit::GPUBlock, OutputRaceStrategy::IgnoreRaces)
          .parallelize(warp, ParallelUnit::GPUWarp, OutputRaceStrategy::IgnoreRaces)
          .parallelize(thread, ParallelUnit::GPUThread, OutputRaceStrategy::Atomics);
}

IndexStmt scheduleSDDMMGPU(IndexStmt stmt, Tensor<double> B, int NNZ_PER_WARP=8*32, int BLOCK_SIZE=256, int CO_FACTOR=4) {
  int NNZ_PER_TB = NNZ_PER_WARP * (BLOCK_SIZE / WARP_SIZE);
  IndexVar f("f"), fpos("fpos"), block("block"), fpos1("fpos1"), warp("warp"), nnz("nnz");
  IndexVar dense_val_unbounded("dense_val_unbounded"), dense_val("dense_val"), thread("thread");
  IndexVar thread_nz("thread_nz");
  return stmt.reorder({i, k, j})
          .fuse(i, k, f)
          .pos(f, fpos, B(i,k))
          .split(fpos, block, fpos1, NNZ_PER_TB)
          .split(fpos1, warp, nnz, NNZ_PER_WARP)
          .split(j, dense_val_unbounded, thread, WARP_SIZE)
          .bound(dense_val_unbounded, dense_val, CO_FACTOR, BoundType::MaxExact)
          .reorder({block, warp, nnz, thread, dense_val})
          .unroll(dense_val, CO_FACTOR)
          .parallelize(block, ParallelUnit::GPUBlock, OutputRaceStrategy::IgnoreRaces)
          .parallelize(warp, ParallelUnit::GPUWarp, OutputRaceStrategy::Atomics)
          .parallelize(thread, ParallelUnit::GPUThread, OutputRaceStrategy::ParallelReduction);
}

IndexStmt scheduleTTMGPU(IndexStmt stmt, Tensor<double> B, int NNZ_PER_WARP=8*32, int BLOCK_SIZE=256, int CO_FACTOR=4) {
  int NNZ_PER_TB = NNZ_PER_WARP * (BLOCK_SIZE / WARP_SIZE);
  IndexVar jk("jk"), f("f"), fpos("fpos"), block("block"), fpos1("fpos1"), warp("warp"), nnz("nnz"), dense_val_unbounded("dense_val_unbounded"), dense_val("dense_val"), thread("thread");

  return stmt.reorder({i, j, k, l})
          .fuse(j, k, jk)
          .fuse(i, jk, f)
          .pos(f, fpos, B(i, j, k))
          .split(fpos, block, fpos1, NNZ_PER_TB)
          .split(fpos1, warp, nnz, NNZ_PER_WARP)
          .split(l, dense_val_unbounded, thread, WARP_SIZE)
          .bound(dense_val_unbounded, dense_val, CO_FACTOR, BoundType::MaxExact)
          .reorder({block, warp, nnz, thread, dense_val})
          .unroll(dense_val, CO_FACTOR)
          .parallelize(block, ParallelUnit::GPUBlock, OutputRaceStrategy::IgnoreRaces)
          .parallelize(warp, ParallelUnit::GPUWarp, OutputRaceStrategy::IgnoreRaces)
          .parallelize(thread, ParallelUnit::GPUThread, OutputRaceStrategy::Atomics);
}

IndexStmt scheduleTTVGPU(IndexStmt stmt, Tensor<double> B, IndexExpr precomputedExpr, int NNZ_PER_WARP=8*32, int BLOCK_SIZE=256) {
  int NNZ_PER_TB = NNZ_PER_WARP * (BLOCK_SIZE / WARP_SIZE);
  IndexVar jk("jk"), f("f"), fpos("fpos"), block("block"), fpos1("fpos1"), warp("warp"), fpos2("fpos2"), thread("thread"), thread_nz("thread_nz"), thread_nz_pre("thread_nz_pre");
  TensorVar precomputed("precomputed", Type(Float64, {Dimension(thread_nz)}), taco::dense);

  return stmt.fuse(j, k, jk)
          .fuse(i, jk, f)
          .pos(f, fpos, B(i,j,k))
          .split(fpos, block, fpos1, NNZ_PER_TB)
          .split(fpos1, warp, fpos2, NNZ_PER_WARP)
          .split(fpos2, thread, thread_nz, NNZ_PER_WARP/WARP_SIZE)
          .reorder({block, warp, thread, thread_nz})
          .precompute(precomputedExpr, thread_nz, thread_nz_pre, precomputed)
          .unroll(thread_nz_pre, NNZ_PER_WARP/WARP_SIZE)
          .parallelize(block, ParallelUnit::GPUBlock, OutputRaceStrategy::IgnoreRaces)
          .parallelize(warp, ParallelUnit::GPUWarp, OutputRaceStrategy::IgnoreRaces)
          .parallelize(thread, ParallelUnit::GPUThread, OutputRaceStrategy::Atomics);
}

IndexStmt scheduleMTTKRPGPU(IndexStmt stmt, Tensor<double> B, int NNZ_PER_WARP=16, int BLOCK_SIZE=256) {
  int NNZ_PER_TB = NNZ_PER_WARP * (BLOCK_SIZE / WARP_SIZE);
  IndexVar kl("kl"), f("f"), fpos("fpos"), block("block"), fpos1("fpos1"), warp("warp"), nnz("nnz"), dense_val_unbounded("dense_val_unbounded"), dense_val("dense_val"), thread("thread");
  return stmt.reorder({i,k,l,j})
          .fuse(k, l, kl)
          .fuse(i, kl, f)
          .pos(f, fpos, B(i, k, l))
          .split(fpos, block, fpos1, NNZ_PER_TB)
          .split(fpos1, warp, nnz, NNZ_PER_WARP)
          .split(j, dense_val_unbounded, thread, WARP_SIZE)
          .bound(dense_val_unbounded, dense_val, 1, BoundType::MaxExact)
          .reorder({block, warp, dense_val, thread, nnz})
          .parallelize(block, ParallelUnit::GPUBlock, OutputRaceStrategy::IgnoreRaces)
          .parallelize(warp, ParallelUnit::GPUWarp, OutputRaceStrategy::IgnoreRaces)
          .parallelize(thread, ParallelUnit::GPUThread, OutputRaceStrategy::Atomics);
}

// splits so same number of rows per warp and then each thread in warp gets 1/32 of the columns space
IndexStmt scheduleSpMMRowsGPU(IndexStmt stmt, Tensor<double> A, int ROWS_PER_WARP=4, int BLOCK_SIZE=256) {
  int ROWS_PER_TB = ROWS_PER_WARP * (BLOCK_SIZE / WARP_SIZE);
  IndexVar i1("i1"), block("block"), warp("warp"), warp_row("warp_row"), thread("thread"), thread_col("thread_col");
  return stmt.split(i, block, i1, ROWS_PER_TB)
          .split(i1, warp, warp_row, ROWS_PER_WARP)
          .split(k, thread, thread_col, 32)
          .reorder({block, warp, warp_row, thread, thread_col, j})
          .parallelize(block, ParallelUnit::GPUBlock, OutputRaceStrategy::IgnoreRaces)
          .parallelize(warp, ParallelUnit::GPUWarp, OutputRaceStrategy::IgnoreRaces)
          .parallelize(thread, ParallelUnit::GPUThread, OutputRaceStrategy::Atomics);
}

// splits so same number of nonzero rows per warp and then each thread in warp gets 1/32 of the columns space (no search needed)
IndexStmt scheduleSpMMNZRowsGPU(IndexStmt stmt, Tensor<double> A, int NZ_ROWS_PER_WARP=4, int BLOCK_SIZE=256) {
  int NZ_ROWS_PER_TB = NZ_ROWS_PER_WARP * (BLOCK_SIZE / WARP_SIZE);
  IndexVar ip("ip"), ip1("ip1"), block("block"), warp("warp"), warp_row("warp_row"), thread("thread"), thread_col("thread_col");
  return stmt.pos(i, ip, A(i, j))
          .split(ip, block, ip1, NZ_ROWS_PER_TB)
          .split(ip1, warp, warp_row, NZ_ROWS_PER_WARP)
          .split(k, thread, thread_col, 32)
          .reorder({block, warp, warp_row, thread, thread_col, j})
          .parallelize(block, ParallelUnit::GPUBlock, OutputRaceStrategy::IgnoreRaces)
          .parallelize(warp, ParallelUnit::GPUWarp, OutputRaceStrategy::IgnoreRaces)
          .parallelize(thread, ParallelUnit::GPUThread, OutputRaceStrategy::Atomics);
}


IndexStmt scheduleSpMMCPUNoVec(IndexStmt stmt, Tensor<double> A, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  IndexVar i0("i0"), i1("i1"), kbounded("kbounded"), k0("k0"), k1("k1"), jpos("jpos"), jpos0("jpos0"), jpos1("jpos1");
  return stmt.split(i, i0, i1, CHUNK_SIZE)
          .pos(j, jpos, A(i,j))
          .split(jpos, jpos0, jpos1, UNROLL_FACTOR)
          .reorder({i0, i1, jpos0, k, jpos1})
          .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
}

IndexStmt exampleScheduleSPMVUntiled(IndexStmt stmt, Tensor<double> A) {
  return stmt;
}

IndexStmt exampleScheduleSPMVCPURowTiling(IndexStmt stmt, Tensor<double> A) {
  IndexVar i1("i1"), i2("i2");
  int ROWS_PER_TILE = 4;
  return stmt.split(i, i1, i2, ROWS_PER_TILE);
}

IndexStmt exampleScheduleSPMVPosIteration(IndexStmt stmt, Tensor<double> A) {
  IndexVar f("f"), p("p");
  return stmt.fuse(i, j, f)
             .pos(f, p, A(i, j));
}

TEST(scheduling_eval, test_spmvCPU_temp) {
  if (should_use_CUDA_codegen()) {
    return;
  }
  int NUM_I = 1021/10;
  int NUM_J = 1039/10;
  float SPARSITY = .3;
  Tensor<double> A("A", {NUM_I, NUM_J}, CSR);
  Tensor<double> x("x", {NUM_J}, Format({Dense}));
  Tensor<double> y("y", {NUM_I}, Format({Dense}));

  srand(4353);
  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        A.insert({i, j}, (double) ((int) (rand_float*3/SPARSITY)));
      }
    }
  }

  for (int j = 0; j < NUM_J; j++) {
    float rand_float = (float)rand()/(float)(RAND_MAX);
    x.insert({j}, (double) ((int) (rand_float*3/SPARSITY)));
  }

  x.pack();
  A.pack();


  y(i) = A(i, j) * x(j);
  IndexStmt stmt = y.getAssignment().concretize();
  stmt = stmt.parallelize(i, ParallelUnit::CPUThread, OutputRaceStrategy::Atomics);

  //printToFile("test_spmvCPU_temp", stmt);

  y.compile(stmt);
  y.assemble();
  y.compute();

  Tensor<double> expected("expected", {NUM_I}, Format({Dense}));
  expected(i) = A(i, j) * x(j);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, y);
}

TEST(scheduling_eval, test_sptvCPU_temp) {
  if (should_use_CUDA_codegen()) {
    return;
  }
  int NUM_I = 1021/10;
  int NUM_J = 1039/10;
  int NUM_K = 1049/10;
  float SPARSITY = .01;
  Tensor<double> A("A", {NUM_I, NUM_J, NUM_K}, Format({Sparse, Sparse, Sparse}));
  Tensor<double> x("x", {NUM_K}, Format({Dense}));
  Tensor<double> y("y", {NUM_J}, Format({Dense}));

  srand(4357);
  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      for (int k = 0; k < NUM_K; k++) {
        float rand_float = (float)rand()/(float)(RAND_MAX);
        if (rand_float < SPARSITY) {
          A.insert({i, j, k}, (double) ((int) (rand_float*3/SPARSITY)));
        }
      }
    }
  }

  for (int k = 0; k < NUM_K; k++) {
    float rand_float = (float)rand()/(float)(RAND_MAX);
    x.insert({k}, (double) ((int) (rand_float*3/SPARSITY)));
  }

  x.pack();
  A.pack();


  y(j) = A(i, j, k) * x(k);
  IndexStmt stmt = y.getAssignment().concretize();
  stmt = stmt.reorder({i,j,k}).parallelize(j, ParallelUnit::CPUThread, OutputRaceStrategy::Atomics);

  //printToFile("test_sptvCPU_temp", stmt);

  y.compile(stmt);
  y.assemble();
  y.compute();

  Tensor<double> expected("expected", {NUM_J}, Format({Dense}));
  expected(j) = A(i, j, k) * x(k);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, y);
}

TEST(scheduling_eval, example_spmvCPU_splitpos) {
  if (should_use_CUDA_codegen()) {
    return;
  }
  int NUM_I = 1021/10;
  int NUM_J = 1039/10;
  float SPARSITY = .3;
  int CHUNK_SIZE = 16;
  Tensor<double> A("A", {NUM_I, NUM_J}, CSR);
  Tensor<double> x("x", {NUM_J}, Format({Dense}));
  Tensor<double> y("y", {NUM_I}, Format({Dense}));

  srand(53535);
  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        A.insert({i, j}, (double) ((int) (rand_float*3/SPARSITY)));
      }
    }
  }

  for (int j = 0; j < NUM_J; j++) {
    float rand_float = (float)rand()/(float)(RAND_MAX);
    x.insert({j}, (double) ((int) (rand_float*3/SPARSITY)));
  }

  x.pack();
  A.pack();

  IndexVar i0("i0"), i1("i1"), kpos("kpos"), kpos0("kpos0"), kpos1("kpos1");
  y(i) = A(i, j) * x(j);

  IndexStmt stmt = y.getAssignment().concretize();
  stmt = stmt.fuse(i, j, k)
          .pos(k, kpos, A(i, j))
          .split(kpos, kpos0, kpos1, CHUNK_SIZE)
          .parallelize(kpos0, ParallelUnit::CPUThread, OutputRaceStrategy::Atomics);

  //printToFile("example_spmv_cpu_splitpos", stmt);

  y.compile(stmt);
  y.assemble();
  y.compute();

  Tensor<double> expected("expected", {NUM_I}, Format({Dense}));
  expected(i) = A(i, j) * x(j);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, y);
}

TEST(scheduling_eval, spmmCPU) {
  if (should_use_CUDA_codegen()) {
    return;
  }
  int NUM_I = 1021/10;
  int NUM_J = 1039/10;
  int NUM_K = 128;
  float SPARSITY = .3;
  Tensor<double> A("A", {NUM_I, NUM_J}, CSR);
  Tensor<double> B("B", {NUM_J, NUM_K}, {Dense, Dense});
  Tensor<double> C("C", {NUM_I, NUM_K}, {Dense, Dense});

  srand(75883);
  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        A.insert({i, j}, (double) ((int) (rand_float*3/SPARSITY)));
      }
    }
  }

  for (int j = 0; j < NUM_J; j++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      B.insert({j, k}, (double) ((int) (rand_float*3/SPARSITY)));
    }
  }

  A.pack();
  B.pack();

  C(i, k) = A(i, j) * B(j, k);

  IndexStmt stmt = C.getAssignment().concretize();
  stmt = scheduleSpMMCPU(stmt, A);

  //printToFile("spmm_cpu", stmt);

  C.compile(stmt);
  C.assemble();
  C.compute();

  Tensor<double> expected("expected", {NUM_I, NUM_K}, {Dense, Dense});
  expected(i, k) = A(i, j) * B(j, k);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, C);
}

TEST(scheduling_eval, spmmISPC) {
  taco::util::TimeResults timevalue;
  bool time                = true;

  set_ISPC_codegen_enabled(false);
  set_CUDA_codegen_enabled(false);
  
  int NUM_I = 1021/10;
  int NUM_J = 1039/10;
  int NUM_K = 128;
  float SPARSITY = .1;
  Tensor<double> A("A", {NUM_I, NUM_J}, CSR);
  Tensor<double> B("B", {NUM_J, NUM_K}, {Dense, Dense});
  Tensor<double> C("C", {NUM_I, NUM_K}, {Dense, Dense});

  srand(75883);
  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        A.insert({i, j}, (double) ((int) (rand_float*3/SPARSITY)));
      }
    }
  }

  for (int j = 0; j < NUM_J; j++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      B.insert({j, k}, (double) ((int) (rand_float*3/SPARSITY)));
    }
  }

  A.pack();
  B.pack();

  set_ISPC_codegen_enabled(true);
  C(i, k) = A(i, j) * B(j, k);

  IndexStmt stmt = C.getAssignment().concretize();
  // stmt = scheduleSpMMISPC1(stmt, A);
  // stmt = scheduleSpMMISPC1_2(stmt, A);
  stmt = scheduleSpMMISPC1_3(stmt, A);
  
  // stmt = scheduleSpMMISPC2(stmt, A);
  // stmt = scheduleSpMMISPC2_2(stmt, A);
  
  // stmt = scheduleSpMMISPC3(stmt, A);
  // stmt = scheduleSpMMISPC3_2(stmt, A);

  //printToFile("spmm_cpu", stmt);

  C.compile(stmt);
  C.assemble();
  C.compute();

  set_ISPC_codegen_enabled(false);
  Tensor<double> expected("expected", {NUM_I, NUM_K}, {Dense, Dense});
  expected(i, k) = A(i, j) * B(j, k);
  IndexStmt stmt_taco = expected.getAssignment().concretize();
  stmt_taco = scheduleSpMMCPU(stmt_taco, A);

  expected.compile(stmt_taco);
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, C);

  // float ERROR_MARGIN = 0.01;
  // ASSERT_TENSOR_VAL(expected, y);
  for (int i = 0; i < NUM_I; i++) {
    for (int k = 0; k < NUM_K; k++) {
      if (expected(i,k) <= C(i,k) + ERROR_MARGIN && expected(i,k) >= C(i,k) - ERROR_MARGIN) {
        // std::cout << "matched values: expected -> " << expected(j) << " == " << y(j) << " <- actual\n";
      }
      else {
        std::cout << "unmatched values: expected -> " << expected(i,k) << " != " << C(i,k) << " <- actual\n";
        ASSERT_TRUE(false);
      };
    }
  }

  for (int i=0; i<10; i++) {
    TOOL_BENCHMARK_TIMER(C.compute(), "Compute ISPC: ", timevalue);
    TOOL_BENCHMARK_TIMER(expected.compute(), "Compute TACO: ", timevalue);
  }
}

struct spgemm : public TestWithParam<std::tuple<Format,Format,bool>> {};

TEST_P(spgemm, scheduling_eval) {
  if (should_use_CUDA_codegen()) {
    return;
  }

  Format aFormat, bFormat;
  bool doPrecompute;
  std::tie(aFormat, bFormat, doPrecompute) = GetParam();

  int NUM_I = 100;
  int NUM_J = 100;
  int NUM_K = 100;
  float SPARSITY = .03;
  Tensor<double> A("A", {NUM_I, NUM_J}, aFormat);
  Tensor<double> B("B", {NUM_J, NUM_K}, bFormat);
  Tensor<double> C("C", {NUM_I, NUM_K}, CSR);

  srand(75883);
  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        A.insert({i, j}, (double) ((int) (rand_float*3/SPARSITY)));
      }
    }
  }

  for (int j = 0; j < NUM_J; j++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        B.insert({j, k}, (double) ((int) (rand_float*3/SPARSITY)));
      }
    }
  }

  A.pack();
  B.pack();

  C(i, k) = A(i, j) * B(j, k);
  IndexStmt stmt = C.getAssignment().concretize();
  stmt = scheduleSpGEMMCPU(stmt, doPrecompute);

  C.compile(stmt);
  C.assemble();
  C.compute();

  Tensor<double> expected("expected", {NUM_I, NUM_K}, {Dense, Dense});
  expected(i, k) = A(i, j) * B(j, k);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, C);
}

INSTANTIATE_TEST_CASE_P(spgemm, spgemm,
                        Values(std::make_tuple(CSR, CSR, true),
                               std::make_tuple(DCSR, CSR, true),
                               std::make_tuple(DCSR, DCSR, true),
                               std::make_tuple(CSR, CSC, false),
                               std::make_tuple(DCSR, DCSC, false)));

TEST(scheduling_eval, spmataddCPU) {
  if (should_use_CUDA_codegen()) {
    return;
  }

  int NUM_I = 1000;
  int NUM_J = 10;
  float SPARSITY = .15;
  Tensor<double> A("A", {NUM_I, NUM_J}, CSR);
  Tensor<double> B("B", {NUM_I, NUM_J}, CSR);
  Tensor<double> C("C", {NUM_I, NUM_J}, CSR);
  Tensor<double> eA("eA", {NUM_I, NUM_J}, Dense);
  Tensor<double> eB("eB", {NUM_I, NUM_J}, Dense);
  Tensor<double> eC("eC", {NUM_I, NUM_J}, Dense);

  srand(75883);
  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        double val = (double)((int)(rand_float*3/SPARSITY));
        A.insert({i, j}, val);
        eA.insert({i, j}, val);
      }
    }
  }

  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        double val = (double)((int)(rand_float*3/SPARSITY));
        B.insert({i, j}, val);
        eB.insert({i, j}, val);
      }
    }
  }

  A.pack();
  B.pack();

  C(i, j) = A(i, j) + B(i, j);
  IndexStmt stmt = C.getAssignment().concretize();
  stmt = scheduleSpAddCPU(stmt);

  C.compile(stmt);
  C.assemble();
  C.compute();

  eC(i, j) = eA(i, j) + eB(i, j);
  eC.compile();
  eC.assemble();
  eC.compute();
  ASSERT_TENSOR_EQ(eC, C);
}

TEST(scheduling_eval, sptenaddCPU) {
  if (should_use_CUDA_codegen()) {
    return;
  }

  int NUM_I = 100;
  int NUM_J = 10;
  int NUM_K = 10;
  float SPARSITY = .02;
  Format ecsr({Dense, Compressed(ModeFormat::NOT_UNIQUE), 
               Singleton(ModeFormat::UNIQUE)});
  Tensor<double> A("A", {NUM_I, NUM_J, NUM_K}, ecsr);
  Tensor<double> B("B", {NUM_I, NUM_J, NUM_K}, ecsr);
  Tensor<double> C("C", {NUM_I, NUM_J, NUM_K}, ecsr);
  Tensor<double> eA("eA", {NUM_I, NUM_J, NUM_K}, Dense);
  Tensor<double> eB("eB", {NUM_I, NUM_J, NUM_K}, Dense);
  Tensor<double> eC("eC", {NUM_I, NUM_J, NUM_K}, Dense);

  srand(75883);
  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      for (int k = 0; k < NUM_K; k++) {
        float rand_float = (float)rand()/(float)(RAND_MAX);
        if (rand_float < SPARSITY) {
          double val = (double)((int)(rand_float*3/SPARSITY));
          A.insert({i, j, k}, val);
          eA.insert({i, j, k}, val);
        }
      }
    }
  }

  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      for (int k = 0; k < NUM_K; k++) {
        float rand_float = (float)rand()/(float)(RAND_MAX);
        if (rand_float < SPARSITY) {
          double val = (double)((int)(rand_float*3/SPARSITY));
          B.insert({i, j, k}, val);
          eB.insert({i, j, k}, val);
        }
      }
    }
  }

  A.pack();
  B.pack();

  C(i, j, k) = A(i, j, k) + B(i, j, k);
  IndexStmt stmt = C.getAssignment().concretize();
  stmt = scheduleSpAddCPU(stmt);

  C.compile(stmt);
  C.assemble();
  C.compute();

  eC(i, j, k) = eA(i, j, k) + eB(i, j, k);
  eC.compile();
  eC.assemble();
  eC.compute();
  ASSERT_TENSOR_EQ(eC, C);
}

TEST(scheduling_eval, sddmmCPU) {
  if (should_use_CUDA_codegen()) {
    return;
  }
  int NUM_I = 1021/10;
  int NUM_J = 1039/10;
  int NUM_K = 1057/10;
  float SPARSITY = .3;
  Tensor<double> A("A", {NUM_I, NUM_K}, {Dense, Dense});
  Tensor<double> B("B", {NUM_I, NUM_K}, CSR);
  Tensor<double> C("C", {NUM_I, NUM_J}, {Dense, Dense});
  Tensor<double> D("D", {NUM_J, NUM_K}, {Dense, Dense});

  srand(268238);
  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      C.insert({i, j}, (double) ((int) (rand_float*3/SPARSITY)));
    }
  }

  for (int i = 0; i < NUM_I; i++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        B.insert({i, k}, (double) ((int) (rand_float*3/SPARSITY)));
      }
    }
  }

  for (int j = 0; j < NUM_J; j++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      D.insert({j, k}, (double) ((int) (rand_float*3/SPARSITY)));
    }
  }

  B.pack();
  C.pack();
  D.pack();

  A(i,k) = B(i,k) * C(i,j) * D(j,k);

  IndexStmt stmt = A.getAssignment().concretize();
  stmt = scheduleSDDMMCPU(stmt, B);

  printToFile("sddmm_cpu_ryan2", stmt);

  A.compile(stmt);
  A.assemble();
  A.compute();

  Tensor<double> expected("expected", {NUM_I, NUM_K}, {Dense, Dense});
  expected(i,k) = B(i,k) * C(i,j) * D(j,k);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, A);
}

TEST(scheduling_eval, sddmmSPMMFusedCPU) {
  if (should_use_CUDA_codegen() || should_use_ISPC_codegen()) {
    return;
  }

  int NUM_I = 1021/10;
  int NUM_J = 1039/10;
  int NUM_K = 1057/10;
  float SPARSITY = .3;
  Tensor<double> A("A", {NUM_I, NUM_K}, {Dense, Dense});
  Tensor<double> B("B", {NUM_I, NUM_K}, CSR);
  Tensor<double> C("C", {NUM_I, NUM_J}, {Dense, Dense});
  Tensor<double> D("D", {NUM_J, NUM_K}, {Dense, Dense});

  srand(268238);
  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      C.insert({i, j}, (double) ((int) (rand_float*3/SPARSITY)));
    }
  }

  for (int i = 0; i < NUM_I; i++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        B.insert({i, k}, (double) ((int) (rand_float*3/SPARSITY)));
      }
    }
  }

  for (int j = 0; j < NUM_J; j++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      D.insert({j, k}, (double) ((int) (rand_float*3/SPARSITY)));
    }
  }

  B.pack();
  C.pack();
  D.pack();

  A(i,k) = B(i,k) * C(i,j) * D(j,k);

  IndexStmt stmt = A.getAssignment().concretize();
  stmt = scheduleSDDMMCPU(stmt, B);

  printToFile("sddmm_cpu_ryan2", stmt);

  A.compile(stmt);
  A.assemble();
  A.compute();

  Tensor<double> expected("expected", {NUM_I, NUM_K}, {Dense, Dense});
  expected(i,k) = B(i,k) * C(i,j) * D(j,k);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, A);
}


TEST(scheduling_eval, sddmmcsrCPU) {
  if (should_use_CUDA_codegen()) {
    return;
  }
  int NUM_I = 1021/10;
  int NUM_J = 1039/10;
  int NUM_K = 1057/10;
  float SPARSITY = .3;
  Tensor<double> A("A", {NUM_I, NUM_K}, CSR);
  Tensor<double> B("B", {NUM_I, NUM_K}, CSR);
  Tensor<double> C("C", {NUM_I, NUM_J}, {Dense, Dense});
  Tensor<double> D("D", {NUM_J, NUM_K}, {Dense, Dense});

  srand(268238);
  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      C.insert({i, j}, (double) ((int) (rand_float*3/SPARSITY)));
    }
  }

  for (int i = 0; i < NUM_I; i++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        B.insert({i, k}, (double) ((int) (rand_float*3/SPARSITY)));
      }
    }
  }

  for (int j = 0; j < NUM_J; j++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      D.insert({j, k}, (double) ((int) (rand_float*3/SPARSITY)));
    }
  }

  B.pack();
  C.pack();
  D.pack();

  A(i,k) = B(i,k) * C(i,j) * D(j,k);

  IndexStmt stmt = A.getAssignment().concretize();
  stmt = scheduleSDDMMCSRCPU(stmt, B);

  printToFile("sddmm_cpu", stmt);

  A.compile(stmt);
  A.assemble();
  A.compute();

  Tensor<double> expected("expected", {NUM_I, NUM_K}, CSR);
  expected(i,k) = B(i,k) * C(i,j) * D(j,k);
  
  IndexStmt stmt_ref = expected.getAssignment().concretize();
  printToFile("sddmm_cpu_ref", stmt_ref);

  expected.compile(stmt_ref);
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, A);
}


TEST(scheduling_eval, sddmm2CPU) {
  if (should_use_CUDA_codegen()) {
    return;
  }
  int NUM_I = 1021/10;
  int NUM_J = 1021/10;
  int NUM_K = 18;
  float SPARSITY = .3;
  Tensor<double> Y("Y", {NUM_I, NUM_J}, {Dense, Compressed(ModeFormat::UNIQUE)});
  Tensor<double> A("A", {NUM_I, NUM_J}, {Dense, Compressed(ModeFormat::UNIQUE)});
  Tensor<double> X("X", {NUM_I, NUM_K}, {Dense, Dense});

  srand(268238);

  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        A.insert({i, j}, (double) ((int) (rand_float*3/SPARSITY)));
      }
    }
  }

  for (int i = 0; i < NUM_J; i++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      X.insert({i, k}, (double) ((int) (rand_float*3/SPARSITY)));
    }
  }

  A.pack();
  X.pack();

  Y(i,j) = A(i,j) * X(i,k) * X(k,j);

  // IndexStmt stmt = A.getAssignment().concretize();
  // // stmt = scheduleSDDMMCPU(stmt, A);

  // printToFile("sddmm2_cpu", stmt);

  // A.compile(stmt);
  // A.assemble();
  // A.compute();

  // Tensor<double> expected("expected", {NUM_I, NUM_J}, {Dense, Dense});
  // expected(i,j) = A(i,j) * X(i,k) * X(j,k);
  // expected.compile();
  // expected.assemble();
  // expected.compute();
  // ASSERT_TENSOR_EQ(expected, A);
}



// bin/taco-test --gtest_filter=scheduling_eval.sddmmISPC
TEST(scheduling_eval, sddmmISPC) {

  taco::util::TimeResults timevalue;
  bool time                = true;

  set_CUDA_codegen_enabled(false);
  set_ISPC_codegen_enabled(false);

  int NUM_I = 1021/10;
  int NUM_J = 1039/10;
  int NUM_K = 1057/10;
  float SPARSITY = .3;
  Tensor<double> A("A", {NUM_I, NUM_K}, {Dense, Dense});
  Tensor<double> B("B", {NUM_I, NUM_K}, CSR);
  Tensor<double> C("C", {NUM_I, NUM_J}, {Dense, Dense});
  Tensor<double> D("D", {NUM_J, NUM_K}, {Dense, Dense});

  srand(268238);
  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      C.insert({i, j}, (double) ((int) (rand_float*3/SPARSITY)));
    }
  }

  for (int i = 0; i < NUM_I; i++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        B.insert({i, k}, (double) ((int) (rand_float*3/SPARSITY)));
      }
    }
  }

  for (int j = 0; j < NUM_J; j++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      D.insert({j, k}, (double) ((int) (rand_float*3/SPARSITY)));
    }
  }

  B.pack();
  C.pack();
  D.pack();

  set_ISPC_codegen_enabled(true);
  A(i,k) = B(i,k) * C(i,j) * D(j,k);

  IndexStmt stmt = A.getAssignment().concretize();
  stmt = scheduleSDDMMISPC(stmt, B);

  //printToFile("sddmm_cpu", stmt);

  A.compile(stmt);
  A.assemble();
  // A.compute();

  set_ISPC_codegen_enabled(false);
  Tensor<double> expected("expected", {NUM_I, NUM_K}, {Dense, Dense});
  expected(i,k) = B(i,k) * C(i,j) * D(j,k);
  IndexStmt stmt_taco = A.getAssignment().concretize();
  stmt_taco = scheduleSDDMMCPU(stmt_taco, B);
  expected.compile(stmt_taco);
  expected.assemble();
  // expected.compute();

  TOOL_BENCHMARK_TIMER(A.compute(), "Compute ISPC: ", timevalue);
  TOOL_BENCHMARK_TIMER(expected.compute(), "Compute TACO: ", timevalue);

  ASSERT_TENSOR_EQ(expected, A);


  // float ERROR_MARGIN = 0.01;
  // ASSERT_TENSOR_VAL(expected, y);
  for (int i = 0; i < NUM_I; i++) {
    for (int k = 0; k < NUM_K; k++) {
      if (expected(i,k) <= A(i,k) + ERROR_MARGIN && expected(i,k) >= A(i,k) - ERROR_MARGIN) {
        // std::cout << "matched values: expected -> " << expected(j) << " == " << y(j) << " <- actual\n";
      }
      else {
        std::cout << "unmatched values: expected -> " << expected(i,k) << " != " << A(i,k) << " <- actual\n";
        ASSERT_TRUE(false);
      };
    }
  }
  std::cout << "test scheduling_eval.sddmmISPC passed\n";

}


// bin/taco-test --gtest_filter=scheduling_eval.sddmmISPC
TEST(scheduling_eval, sddmm2ISPC) {

  taco::util::TimeResults timevalue;
  bool time                = true;

  set_CUDA_codegen_enabled(false);
  set_ISPC_codegen_enabled(false);

  int NUM_I = 1021/10;
  int NUM_K = 1039/10;
  int NUM_J = 1021/10;
  float SPARSITY = .3;
  Tensor<double> A("A", {NUM_I, NUM_J}, {Dense, Dense});
  Tensor<double> B("B", {NUM_I, NUM_J}, CSR);
  Tensor<double> C("C", {NUM_I, NUM_K}, {Dense, Dense});

  srand(268238);
  for (int i = 0; i < NUM_I; i++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      C.insert({i, k}, (double) ((int) (rand_float*3/SPARSITY)));
    }
  }

  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        B.insert({i, j}, (double) ((int) (rand_float*3/SPARSITY)));
      }
    }
  }

  B.pack();
  C.pack();

  set_ISPC_codegen_enabled(true);
  A(i,j) = B(i,j) * C(i,k) * C(j,k);

  IndexStmt stmt = A.getAssignment().concretize();
  stmt = scheduleSDDMM2ISPC(stmt, B);

  //printToFile("sddmm_cpu", stmt);

  A.compile(stmt);
  A.assemble();
  // A.compute();

  set_ISPC_codegen_enabled(false);
  Tensor<double> expected("expected", {NUM_I, NUM_J}, {Dense, Dense});
  expected(i,j) = B(i,j) * C(i,k) * C(j,k);
  IndexStmt stmt_taco = A.getAssignment().concretize();
  stmt_taco = scheduleSDDMM2CPU(stmt_taco, B);
  expected.compile(stmt_taco);
  expected.assemble();
  // expected.compute();

  TOOL_BENCHMARK_TIMER(A.compute(), "Compute ISPC: ", timevalue);
  TOOL_BENCHMARK_TIMER(expected.compute(), "Compute TACO: ", timevalue);

  ASSERT_TENSOR_EQ(expected, A);


  // float ERROR_MARGIN = 0.01;
  // ASSERT_TENSOR_VAL(expected, y);
  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      if (expected(i,j) <= A(i,j) + ERROR_MARGIN && expected(i,j) >= A(i,j) - ERROR_MARGIN) {
        // std::cout << "matched values: expected -> " << expected(j) << " == " << y(j) << " <- actual\n";
      }
      else {
        std::cout << "unmatched values: expected -> " << expected(i,j) << " != " << A(i,j) << " <- actual\n";
        ASSERT_TRUE(false);
      };
    }
  }
  std::cout << "test scheduling_eval.sddmmISPC passed\n";

}


TEST(scheduling_eval, spmvCPU) {
  if (should_use_CUDA_codegen()) {
    return;
  }
  int NUM_I = 1021/10;
  int NUM_J = 1039/10;
  float SPARSITY = .3;
  Tensor<double> A("A", {NUM_I, NUM_J}, CSR);
  Tensor<double> x("x", {NUM_J}, Format({Dense}));
  Tensor<double> y("y", {NUM_I}, Format({Dense}));

  srand(120);
  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        A.insert({i, j}, (double) ((int) (rand_float * 3 / SPARSITY)));
      }
    }
  }

  for (int j = 0; j < NUM_J; j++) {
    float rand_float = (float)rand()/(float)(RAND_MAX);
    x.insert({j}, (double) ((int) (rand_float*3/SPARSITY)));
  }

  x.pack();
  A.pack();

  y(i) = A(i, j) * x(j);

  IndexStmt stmt = y.getAssignment().concretize();
  stmt = scheduleSpMVCPU(stmt);

  //printToFile("spmv_cpu", stmt);

  y.compile(stmt);
  y.assemble();
  y.compute();

  Tensor<double> expected("expected", {NUM_I}, Format({Dense}));
  expected(i) = A(i, j) * x(j);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, y);
}


TEST(scheduling_eval, spmvISPC) {

  taco::util::TimeResults timevalue;
  bool time                = true;

  set_ISPC_codegen_enabled(false);
  set_CUDA_codegen_enabled(false);
  
  int NUM_I = 200021/10;
  int NUM_J = 200039/10;
  float SPARSITY = .2;
  Tensor<double> A("A", {NUM_I, NUM_J}, CSR);
  Tensor<double> x("x", {NUM_J}, Format({Dense}));
  Tensor<double> y("y", {NUM_I}, Format({Dense}));

  srand(120);
  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        A.insert({i, j}, (double) ((int) (rand_float * 3 / SPARSITY)));
      }
    }
  }

  for (int j = 0; j < NUM_J; j++) {
    float rand_float = (float)rand()/(float)(RAND_MAX);
    x.insert({j}, (double) ((int) (rand_float*3/SPARSITY)));
  }

  x.pack();
  A.pack();

  set_ISPC_codegen_enabled(true);

  y(i) = A(i, j) * x(j);

  IndexStmt stmt = y.getAssignment().concretize();
  // stmt = scheduleSpMVISPC(stmt);

  printToFile("spmv_cpu", stmt);

  y.compile(stmt);
  y.assemble();
  // y.compile();

  set_ISPC_codegen_enabled(false);

  // Tensor<double> expected("expected", {NUM_I}, Format({Dense}));
  // expected(i) = A(i, j) * x(j);
  // expected.compile();
  // expected.assemble();
  // expected.compute();


  Tensor<double> expected("expected", {NUM_I}, Format({Dense}));
  expected(i) = A(i, j) * x(j);
  IndexStmt stmt_taco = expected.getAssignment().concretize();
  stmt_taco = scheduleSpMVCPU(stmt_taco);
  
  expected.compile(stmt_taco);
  expected.assemble();
  // expected.compile();


  TOOL_BENCHMARK_TIMER(y.compute(), "Compute ISPC: ", timevalue);
  TOOL_BENCHMARK_TIMER(expected.compute(), "Compute TACO: ", timevalue);
  

  ASSERT_TENSOR_EQ(expected, y);

  // float ERROR_MARGIN = 0.01;
  // ASSERT_TENSOR_VAL(expected, y);
  for (int j = 0; j < NUM_J; j++) {
    if (expected(j) <= y(j) + ERROR_MARGIN && expected(j) >= y(j) - ERROR_MARGIN) {
      // std::cout << "matched values: expected -> " << expected(j) << " == " << y(j) << " <- actual\n";
    }
    else {
      std::cout << "unmatched values: expected -> " << expected(j) << " != " << y(j) << " <- actual\n";
      ASSERT_TRUE(false);
    };
  }

  std::cout << "test scheduling_eval.spmvISPC passed\n";

  for (int i=0; i<10; i++) {
    TOOL_BENCHMARK_TIMER(y.compute(), "Compute ISPC: ", timevalue);
    TOOL_BENCHMARK_TIMER(expected.compute(), "Compute TACO: ", timevalue);
  }


}

TEST(scheduling_eval, ttvCPU) {
  if (should_use_CUDA_codegen()) {
    return;
  }
  int NUM_I = 1021/10;
  int NUM_J = 1039/10;
  int NUM_K = 1057/10;
  float SPARSITY = .3;
  Tensor<double> A("A", {NUM_I, NUM_J}, {Dense, Dense}); // TODO: change to sparse outputs
  Tensor<double> B("B", {NUM_I, NUM_J, NUM_K}, {Sparse, Sparse, Sparse});
  Tensor<double> c("c", {NUM_K}, Format({Dense}));

  srand(9536);
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

  for (int k = 0; k < NUM_K; k++) {
    float rand_float = (float)rand()/(float)(RAND_MAX);
    c.insert({k}, (double) ((int) (rand_float*3)));
  }

  B.pack();
  c.pack();

  A(i,j) = B(i,j,k) * c(k);

  IndexStmt stmt = A.getAssignment().concretize();
  stmt = scheduleTTVCPU(stmt, B);

  printToFile("ttv_cpu", stmt);

  A.compile(stmt);
  A.assemble();
  A.compute();

  Tensor<double> expected("expected", {NUM_I, NUM_J}, {Dense, Dense});
  expected(i,j) = B(i,j,k) * c(k);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, A);
}


TEST(scheduling_eval, ttvISPC) {
  if (should_use_CUDA_codegen()) {
    return;
  }
  set_CUDA_codegen_enabled(false);
  set_ISPC_codegen_enabled(false);
  int NUM_I = 1021/10;
  int NUM_J = 1039/10;
  int NUM_K = 1057/10;
  float SPARSITY = .3;
  Tensor<double> A("A", {NUM_I, NUM_J}, {Dense, Dense}); // TODO: change to sparse outputs
  Tensor<double> B("B", {NUM_I, NUM_J, NUM_K}, {Sparse, Sparse, Sparse});
  Tensor<double> c("c", {NUM_K}, Format({Dense}));

  srand(9536);
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

  for (int k = 0; k < NUM_K; k++) {
    float rand_float = (float)rand()/(float)(RAND_MAX);
    c.insert({k}, (double) ((int) (rand_float*3)));
  }

  B.pack();
  c.pack();

  set_ISPC_codegen_enabled(true);
  A(i,j) = B(i,j,k) * c(k);

  IndexStmt stmt = A.getAssignment().concretize();
  stmt = scheduleTTVISPC(stmt, B);

  printToFile("ttv_ispc", "__ttv_ispc", stmt);

  A.compile(stmt);
  A.assemble();
  A.compute();

  set_ISPC_codegen_enabled(false);
  Tensor<double> expected("expected", {NUM_I, NUM_J}, {Dense, Dense});
  expected(i,j) = B(i,j,k) * c(k);
  IndexStmt stmt_taco = expected.getAssignment().concretize();
  stmt_taco = scheduleTTVCPU(stmt_taco, B);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, A);
}


TEST(scheduling_eval, ttvCPU_CSR) {
  if (should_use_CUDA_codegen()) {
    return;
  }

  int NUM_I = 1021/10;
  int NUM_J = 1039/10;
  int NUM_K = 1057/10;
  float SPARSITY = .3;
  Tensor<double> A("A", {NUM_I, NUM_J}, {Dense, Sparse});
  Tensor<double> B("B", {NUM_I, NUM_J, NUM_K}, {Dense, Sparse, Sparse});
  Tensor<double> c("c", {NUM_K}, Format({Dense}));

  srand(9536);
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

  for (int k = 0; k < NUM_K; k++) {
    float rand_float = (float)rand()/(float)(RAND_MAX);
    c.insert({k}, (double) ((int) (rand_float*3)));
  }

  B.pack();
  c.pack();

  A(i,j) = B(i,j,k) * c(k);

  IndexStmt stmt = A.getAssignment().concretize();
  stmt = scheduleTTVCPUCSR(stmt);

  printToFile("ttv_cpu_csr", stmt);

  A.compile(stmt);
  A.assemble();
  A.compute();

  Tensor<double> expected("expected", {NUM_I, NUM_J}, {Dense, Sparse});
  expected(i,j) = B(i,j,k) * c(k);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, A);
}

TEST(scheduling_eval, ttvISPC_CSR) {
  if (should_use_CUDA_codegen()) {
    return;
  }

  int NUM_I = 10000;
  int NUM_J = 1039/10;
  int NUM_K = 128;
  float SPARSITY = .3;
  Tensor<double> A("A", {NUM_I, NUM_J}, {Dense, Sparse});
  Tensor<double> B("B", {NUM_I, NUM_J, NUM_K}, {Dense, Sparse, Sparse});
  Tensor<double> c("c", {NUM_K}, Format({Dense}));

  srand(9536);
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

  for (int k = 0; k < NUM_K; k++) {
    float rand_float = (float)rand()/(float)(RAND_MAX);
    c.insert({k}, (double) ((int) (rand_float*3)));
  }

  B.pack();
  c.pack();

  set_ISPC_codegen_enabled(true);
  A(i,j) = B(i,j,k) * c(k);

  IndexStmt stmt = A.getAssignment().concretize();
  stmt = scheduleTTVISPCCSR(stmt);
  printToFile("ttv_ispc_csr", "__ttv_ispc_csr", stmt);

  A.compile(stmt);
  A.assemble();
  A.compute();

  set_ISPC_codegen_enabled(false);
  Tensor<double> expected("expected", {NUM_I, NUM_J}, {Dense, Sparse});
  expected(i,j) = B(i,j,k) * c(k);
  IndexStmt taco_stmt = expected.getAssignment().concretize();
  taco_stmt = scheduleTTVCPUCSR_ST(taco_stmt);
  expected.compile(taco_stmt);
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, A);

  Tensor<double> A2("A2", {NUM_I, NUM_J}, {Dense, Sparse});
  set_ISPC_codegen_enabled(true);
  A2(i,j) = B(i,j,k) * c(k);

  IndexStmt stmt2 = A2.getAssignment().concretize();

  A2.compile(stmt2);
  A2.assemble();
  A2.compute();

  taco::util::TimeResults timevalue;
  bool time                = true;

  for (int i=0; i<3; i++) {
    TOOL_BENCHMARK_TIMER(expected.compute(), "Compute TACO1: ", timevalue);
    TOOL_BENCHMARK_TIMER(A.compute(), "Compute ISPC1: ", timevalue);
    TOOL_BENCHMARK_TIMER(A2.compute(), "Compute ISPC2: ", timevalue);
  }

  
}

TEST(scheduling_eval, ttmCPU) {
  if (should_use_CUDA_codegen()) {
    return;
  }
  int NUM_I = 1021/40;
  int NUM_J = 1039/40;
  int NUM_K = 1057/40;
  int NUM_L = 1232/40;
  float SPARSITY = .1;
  Tensor<double> A("A", {NUM_I, NUM_J, NUM_L}, {Dense, Dense, Dense}); // TODO: change to sparse outputs
  Tensor<double> B("B", {NUM_I, NUM_J, NUM_K}, {Sparse, Sparse, Sparse});
  Tensor<double> C("C", {NUM_K, NUM_L}, {Dense, Dense});

  srand(935);
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

  for (int k = 0; k < NUM_K; k++) {
    for (int l = 0; l < NUM_L; l++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      C.insert({k, l}, (double) ((int) (rand_float*3)));
    }
  }

  B.pack();
  C.pack();

  A(i,j,l) = B(i,j,k) * C(k,l);

  IndexStmt stmt = A.getAssignment().concretize();
  stmt = scheduleTTMCPU(stmt, B);

  //printToFile("ttm_cpu", stmt);

  A.compile(stmt);
  A.assemble();
  A.compute();

  Tensor<double> expected("expected", {NUM_I, NUM_J, NUM_L}, {Dense, Dense, Dense});
  expected(i,j,l) = B(i,j,k) * C(k,l);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, A);
}

TEST(scheduling_eval, ttmISPC) {
  if (should_use_CUDA_codegen()) {
    return;
  }
  int NUM_I = 1021/40;
  int NUM_J = 1039/40;
  int NUM_K = 1057/40;
  int NUM_L = 1232/40;
  float SPARSITY = .1;
  Tensor<double> A("A", {NUM_I, NUM_J, NUM_L}, {Dense, Dense, Dense}); // TODO: change to sparse outputs
  Tensor<double> B("B", {NUM_I, NUM_J, NUM_K}, {Sparse, Sparse, Sparse});
  Tensor<double> C("C", {NUM_K, NUM_L}, {Dense, Dense});

  srand(935);
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

  for (int k = 0; k < NUM_K; k++) {
    for (int l = 0; l < NUM_L; l++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      C.insert({k, l}, (double) ((int) (rand_float*3)));
    }
  }

  B.pack();
  C.pack();

  A(i,j,l) = B(i,j,k) * C(k,l);

  IndexStmt stmt = A.getAssignment().concretize();
  stmt = scheduleTTMCPU(stmt, B);

  //printToFile("ttm_cpu", stmt);

  A.compile(stmt);
  A.assemble();
  A.compute();

  Tensor<double> expected("expected", {NUM_I, NUM_J, NUM_L}, {Dense, Dense, Dense});
  expected(i,j,l) = B(i,j,k) * C(k,l);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, A);
}

TEST(scheduling_eval, mttkrpCPU) {
  if (should_use_CUDA_codegen()) {
    return;
  }
  int NUM_I = 1021/20;
  int NUM_J = 1039/20;
  int NUM_K = 1057/20;
  int NUM_L = 1232/20;
  float SPARSITY = .1;
  Tensor<double> A("A", {NUM_I, NUM_J}, {Dense, Dense});
  Tensor<double> B("B", {NUM_I, NUM_K, NUM_L}, {Dense, Sparse, Sparse});
  Tensor<double> C("C", {NUM_K, NUM_J}, {Dense, Dense});
  Tensor<double> D("D", {NUM_L, NUM_J}, {Dense, Dense});

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

  for (int k = 0; k < NUM_K; k++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      C.insert({k, j}, (double) ((int) (rand_float*3)));
    }
  }

  for (int l = 0; l < NUM_L; l++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      D.insert({l, j}, (double) ((int) (rand_float*3)));
    }
  }

  B.pack();
  C.pack();
  D.pack();

  A(i,j) = B(i,k,l) * C(k,j) * D(l,j);

  IndexStmt stmt = A.getAssignment().concretize();
  stmt = scheduleMTTKRPCPU(stmt, B);
  //printToFile("mttkrp_cpu", stmt);

  A.compile(stmt);
  A.assemble();
  A.compute();

  Tensor<double> expected("expected", {NUM_I, NUM_J}, {Dense, Dense});
  expected(i,j) = B(i,k,l) * C(k,j) * D(l,j);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, A);
}

TEST(scheduling_eval, temp) {
  if (should_use_CUDA_codegen() || should_use_ISPC_codegen()) {
    return;
  }
  std::default_random_engine gen(0);
  std::uniform_real_distribution<double> unif(0.0, 1.0);
  // Predeclare the storage formats that the inputs and output will be stored as.
  // To define a format, you must specify whether each dimension is dense or sparse
  // and (optionally) the order in which dimensions should be stored. The formats
  // declared below correspond to doubly compressed sparse row (dcsr), row-major
  // dense (rm), and column-major dense (dm).
  Format dcsr({Sparse,Sparse});
  Format   rm({Dense,Dense});
  Format   cm({Dense,Dense}, {1,0});

  // Load a sparse matrix from file (stored in the Matrix Market format) and
  // store it as a doubly compressed sparse row matrix. Matrices correspond to
  // order-2 tensors in taco. The matrix in this example can be download from:
  // https://www.cise.ufl.edu/research/sparse/MM/Williams/webbase-1M.tar.gz
  Tensor<double> B = read("/home/min/a/kadhitha/ispc-examples/data/ufl/webbase-1M/webbase-1M.mtx", dcsr);
  // Generate a random dense matrix and store it in row-major (dense) format.
  Tensor<double> C({B.getDimension(0), 1000}, rm);
  for (int i = 0; i < C.getDimension(0); ++i) {
    for (int j = 0; j < C.getDimension(1); ++j) {
      C.insert({i,j}, unif(gen));
    }
  }
  C.pack();

  // Generate another random dense matrix and store it in column-major format.
  Tensor<double> D({1000, B.getDimension(1)}, cm);
  for (int i = 0; i < D.getDimension(0); ++i) {
    for (int j = 0; j < D.getDimension(1); ++j) {
      D.insert({i,j}, unif(gen));
    }
  }
  D.pack();

  // Declare the output matrix to be a sparse matrix with the same dimensions as
  // input matrix B, to be also stored as a doubly compressed sparse row matrix.
  Tensor<double> A(B.getDimensions(), dcsr);

  // Define the SDDMM computation using index notation.
  IndexVar i, j, k;
  A(i,j) = B(i,j) * C(i,k) * D(k,j);

  // At this point, we have defined how entries in the output matrix should be
  // computed from entries in the input matrices but have not actually performed
  // the computation yet. To do so, we must first tell taco to generate code that
  // can be executed to compute the SDDMM operation.
  A.compile();
  // We can now call the functions taco generated to assemble the indices of the
  // output matrix and then actually compute the SDDMM.
  A.assemble();
  A.compute();
  // Write the output of the computation to file (stored in the Matrix Market format).
  write("A.mtx", A);
}

TEST(scheduling_eval, mttkrpISPC) {
  if (should_use_CUDA_codegen()) {
    return;
  }
  set_ISPC_codegen_enabled(false);
  set_CUDA_codegen_enabled(false);
  int NUM_I = 10000; // 1021/20;
  int NUM_J = 256;
  int NUM_K = 1057/20;
  int NUM_L = 1232/20;
  float SPARSITY = .1;
  Tensor<double> B("B", {NUM_I, NUM_K, NUM_L}, {Dense, Sparse, Sparse});
  Tensor<double> C("C", {NUM_K, NUM_J}, {Dense, Dense});
  Tensor<double> D("D", {NUM_L, NUM_J}, {Dense, Dense});

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

  for (int k = 0; k < NUM_K; k++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      C.insert({k, j}, (double) ((int) (rand_float*3)));
    }
  }

  for (int l = 0; l < NUM_L; l++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      D.insert({l, j}, (double) ((int) (rand_float*3)));
    }
  }

  B.pack();
  C.pack();
  D.pack();

  set_ISPC_codegen_enabled(true);

  Tensor<double> A1("A1", {NUM_I, NUM_J}, {Dense, Dense});
  A1(i,j) = B(i,k,l) * C(k,j) * D(l,j);
  IndexStmt stmt1 = A1.getAssignment().concretize();
  stmt1 = scheduleMTTKRPISPC(stmt1, B);
  // printToFile("mttkrp1_cpu_ispc", stmt1);
  A1.compile(stmt1);
  A1.assemble();
  A1.compute();

  set_ISPC_codegen_enabled(false);
  Tensor<double> expected1("expected1", {NUM_I, NUM_J}, {Dense, Dense});
  expected1(i,j) = B(i,k,l) * C(k,j) * D(l,j);
  IndexStmt taco_stmt1 = expected1.getAssignment().concretize();
  taco_stmt1 = scheduleMTTKRPCPU(taco_stmt1, B);
  expected1.compile(taco_stmt1);
  expected1.assemble();
  expected1.compute();
  ASSERT_TENSOR_EQ(expected1, A1);

  set_ISPC_codegen_enabled(true);
  Tensor<double> A2("A2", {NUM_I, NUM_J}, {Dense, Dense});
  A2(i,j) = B(i,k,l) * C(k,j) * D(l,j);
  IndexStmt stmt2 = A1.getAssignment().concretize();
  stmt2 = scheduleMTTKRPPrecomputedISPC_ST(stmt2, B);
  // printToFile("mttkrp_cpu_ispc", stmt);
  A2.compile(stmt2);
  A2.assemble();
  A2.compute();
  ASSERT_TENSOR_EQ(expected1, A2);
  
  set_ISPC_codegen_enabled(false);
  Tensor<double> expected2("expected2", {NUM_I, NUM_J}, {Dense, Dense});
  expected2(i,j) = B(i,k,l) * C(k,j) * D(l,j);
  IndexStmt taco_stmt2 = expected2.getAssignment().concretize();
  taco_stmt2 = scheduleMTTKRPPrecomputedCPU_ST(taco_stmt2, B);
  expected2.compile(taco_stmt2);
  expected2.assemble();
  expected2.compute();
  ASSERT_TENSOR_EQ(expected1, expected2);

  taco::util::TimeResults timevalue;
  bool time                = true;

  for (int i=0; i<3; i++) {
    TOOL_BENCHMARK_TIMER(expected1.compute(), "Compute TACO1: ", timevalue);
    TOOL_BENCHMARK_TIMER(A1.compute(), "Compute ISPC1: ", timevalue);
    TOOL_BENCHMARK_TIMER(expected2.compute(), "Compute TACO2: ", timevalue);
    TOOL_BENCHMARK_TIMER(A2.compute(), "Compute ISPC2: ", timevalue);
  }
}


TEST(scheduling_eval, mttkrp4ISPC) {
  if (should_use_CUDA_codegen()) {
    return;
  }
  set_ISPC_codegen_enabled(false);
  set_CUDA_codegen_enabled(false);
  int NUM_I = 1000; // 1021/20;
  int NUM_J = 16;
  int NUM_K = 1057/20;
  int NUM_L = 1232/20;
  int NUM_M = 1124/20;
  float SPARSITY = .1;
  Tensor<double> B("B", {NUM_I, NUM_K, NUM_L, NUM_M}, {Dense, Sparse, Sparse, Sparse});
  Tensor<double> C("C", {NUM_K, NUM_J}, {Dense, Dense});
  Tensor<double> D("D", {NUM_L, NUM_J}, {Dense, Dense});
  Tensor<double> E("E", {NUM_M, NUM_J}, {Dense, Dense});

  srand(549694);
  for (int i = 0; i < NUM_I; i++) {
    for (int k = 0; k < NUM_K; k++) {
      for (int l = 0; l < NUM_L; l++) {
        for (int m = 0; m < NUM_M; m++) {
          float rand_float = (float) rand() / (float) (RAND_MAX);
          if (rand_float < SPARSITY) {
            B.insert({i, k, l, m}, (double) ((int) (rand_float * 3 / SPARSITY)));
          }
        }
      }
    }
  }

  for (int k = 0; k < NUM_K; k++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      C.insert({k, j}, (double) ((int) (rand_float*3)));
    }
  }

  for (int l = 0; l < NUM_L; l++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      D.insert({l, j}, (double) ((int) (rand_float*3)));
    }
  }

  for (int m = 0; m < NUM_M; m++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      E.insert({m, j}, (double) ((int) (rand_float*3)));
    }
  }

  B.pack();
  C.pack();
  D.pack();
  E.pack();

  set_ISPC_codegen_enabled(true);
  Tensor<double> A1("A1", {NUM_I, NUM_J}, {Dense, Dense});
  A1(i,j) = B(i,k,l,m) * C(k,j) * D(l,j) * E(m,j);
  IndexStmt stmt1 = A1.getAssignment().concretize();
  stmt1 = scheduleMTTKRP4ISPC_ST(stmt1, B);
  // printToFile("mttkrp1_cpu_ispc", stmt1);
  A1.compile(stmt1);
  A1.assemble();
  A1.compute();

  set_ISPC_codegen_enabled(false);
  Tensor<double> expected1("expected1", {NUM_I, NUM_J}, {Dense, Dense});
  expected1(i,j) = B(i,k,l,m) * C(k,j) * D(l,j) * E(m,j);
  IndexStmt taco_stmt1 = expected1.getAssignment().concretize();
  taco_stmt1 = scheduleMTTKRP4CPU_ST(taco_stmt1, B);
  expected1.compile(taco_stmt1);
  expected1.assemble();
  expected1.compute();
  ASSERT_TENSOR_EQ(expected1, A1);

  // set_ISPC_codegen_enabled(true);
  // Tensor<double> A2("A2", {NUM_I, NUM_J}, {Dense, Dense});
  // A2(i,j) = B(i,k,l) * C(k,j) * D(l,j);
  // IndexStmt stmt2 = A1.getAssignment().concretize();
  // stmt2 = scheduleMTTKRPPrecomputedISPC_ST(stmt2, B);
  // // printToFile("mttkrp_cpu_ispc", stmt);
  // A2.compile(stmt2);
  // A2.assemble();
  // A2.compute();
  // ASSERT_TENSOR_EQ(expected1, A2);
  
  set_ISPC_codegen_enabled(false);
  Tensor<double> expected2("expected2", {NUM_I, NUM_J}, {Dense, Dense});
  expected2(i,j) = B(i,k,l,m) * C(k,j) * D(l,j) * E(m,j);

  IndexExpr BE = B(i,k,l,m) * E(m,j);
  IndexExpr BDE = BE * D(l, j);
  expected2(i,j) = BDE * C(k,j);
  IndexStmt taco_stmt2 = expected2.getAssignment().concretize();
  TensorVar BE_workspace("BE_workspace", Type(Float64, {Dimension(j)}), taco::dense);
  TensorVar BDE_workspace("BDE_workspace", Type(Float64, {Dimension(j)}), taco::dense);

  IndexStmt precomputed_stmt = forall(i, forall(k,
          where(forall(j, expected2(i,j) += BDE_workspace(j) * C(k,j)),
            forall(l, where(forall(j, BDE_workspace(j) += BE_workspace(j) * D(l,j)),
                forall(m, forall(j, BE_workspace(j) += B(i,k,l,m) * E(m,j))))))));

  // IndexStmt scheduled2 = scheduleMTTKRPPrecomputedCPU(precomputed_stmt, B, 64);
  // expected2.compile(scheduled2);
  // expected2.assemble();
  // expected2.compute();
  // ASSERT_TENSOR_EQ(expected1, expected2);

  taco::util::TimeResults timevalue;
  bool time                = true;

  for (int i=0; i<3; i++) {
    TOOL_BENCHMARK_TIMER(expected1.compute(), "Compute TACO1: ", timevalue);
    TOOL_BENCHMARK_TIMER(A1.compute(), "Compute ISPC1: ", timevalue);
    // TOOL_BENCHMARK_TIMER(expected2.compute(), "Compute TACO2: ", timevalue);
    // TOOL_BENCHMARK_TIMER(A2.compute(), "Compute ISPC2: ", timevalue);
  }
}



TEST(scheduling_eval, spmvGPU) {
  if (!should_use_CUDA_codegen()) {
    return;
  }
  int NUM_I = 1021/10;
  int NUM_J = 1039/10;
  float SPARSITY = .01;
  Tensor<double> A("A", {NUM_I, NUM_J}, CSR);
  Tensor<double> x("x", {NUM_J}, Format({Dense}));
  Tensor<double> y("y", {NUM_I}, Format({Dense}));

  srand(94353);
  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        A.insert({i, j}, (double) ((int) (rand_float * 3 / SPARSITY)));
      }
    }
  }

  for (int j = 0; j < NUM_J; j++) {
    float rand_float = (float)rand()/(float)(RAND_MAX);
    x.insert({j}, (double) ((int) (rand_float*3/SPARSITY)));
  }

  x.pack();
  A.pack();
  IndexExpr precomputed = A(i, j) * x(j);
  y(i) = precomputed;

  IndexStmt stmt = y.getAssignment().concretize();
  stmt = scheduleSpMVGPU(stmt, A, precomputed);

  //printToFile("spmv_gpu", stmt);

  y.compile(stmt);
  y.assemble();
  y.compute();

  Tensor<double> expected("expected", {NUM_I}, Format({Dense}));
  expected(i) = A(i, j) * x(j);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, y);
}

TEST(scheduling_eval, spmmGPU) {
  if (!should_use_CUDA_codegen()) {
    return;
  }
  int NUM_I = 1021/10;
  int NUM_J = 1039/10;
  int NUM_K = 128;
  float SPARSITY = .3;
  Tensor<double> A("A", {NUM_I, NUM_J}, CSR);
  Tensor<double> B("B", {NUM_J, NUM_K}, {Dense, Dense});
  Tensor<double> C("C", {NUM_I, NUM_K}, Format({{Dense, Dense}, {1, 0}}));

  srand(434321);
  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        A.insert({i, j}, (double) ((int) (rand_float*3/SPARSITY)));
      }
    }
  }

  for (int j = 0; j < NUM_J; j++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      B.insert({j, k}, (double) ((int) (rand_float*3/SPARSITY)));
    }
  }

  A.pack();
  B.pack();
  IndexExpr precomputed = A(i, j);
  C(i, k) = B(j, k) * precomputed;

  IndexStmt stmt = C.getAssignment().concretize();
  stmt = scheduleSpMMGPU(stmt, A, precomputed);

  //printToFile("spmm_gpu", stmt);

  C.compile(stmt);
  C.assemble();
  C.compute();

  Tensor<double> expected("expected", {NUM_I, NUM_K}, Format({{Dense, Dense}, {1, 0}}));
  expected(i, k) = A(i, j) * B(j, k);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, C);
}

TEST(scheduling_eval, spmmDCSRGPU) {
  if (!should_use_CUDA_codegen()) {
    return;
  }
  int NUM_I = 1021/10;
  int NUM_J = 1039/10;
  int NUM_K = 128;
  float SPARSITY = .3;
  Tensor<double> A("A", {NUM_I, NUM_J}, {Sparse, Sparse});
  Tensor<double> B("B", {NUM_J, NUM_K}, {Dense, Dense});
  Tensor<double> C("C", {NUM_I, NUM_K}, {Dense, Dense});

  srand(25643);
  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        A.insert({i, j}, (double) ((int) (rand_float*3/SPARSITY)));
      }
    }
  }

  for (int j = 0; j < NUM_J; j++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      B.insert({j, k}, (double) ((int) (rand_float*3/SPARSITY)));
    }
  }

  A.pack();
  B.pack();

  C(i, k) = A(i, j) * B(j, k);

  IndexStmt stmt = C.getAssignment().concretize();
  stmt = scheduleSpMMNZRowsGPU(stmt, A);

  //printToFile("spmm_dcsr_gpu", stmt);

  C.compile(stmt);
  C.assemble();
  C.compute();

  Tensor<double> expected("expected", {NUM_I, NUM_K}, {Dense, Dense});
  expected(i, k) = A(i, j) * B(j, k);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, C);
}

TEST(scheduling_eval, sddmmGPU) {
  if (!should_use_CUDA_codegen()) {
    return;
  }
  int NUM_I = 1021/10;
  int NUM_K = 1039/10;
  int NUM_J = 128;
  float SPARSITY = .3;
  Tensor<double> A("A", {NUM_I, NUM_K}, {Dense, Dense});
  Tensor<double> B("B", {NUM_I, NUM_K}, CSR);
  Tensor<double> C("C", {NUM_I, NUM_J}, {Dense, Dense});
  Tensor<double> D("D", {NUM_J, NUM_K}, {Dense, Dense});

  srand(535366);
  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      C.insert({i, j}, (double) ((int) (rand_float*3/SPARSITY)));
    }
  }

  for (int i = 0; i < NUM_I; i++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        B.insert({i, k}, (double) ((int) (rand_float*3/SPARSITY)));
      }
    }
  }

  for (int j = 0; j < NUM_J; j++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      D.insert({j, k}, (double) ((int) (rand_float*3/SPARSITY)));
    }
  }

  B.pack();
  C.pack();
  D.pack();

  A(i,k) = B(i,k) * C(i,j) * D(j,k);

  IndexStmt stmt = A.getAssignment().concretize();
  stmt = scheduleSDDMMGPU(stmt, B);

  //printToFile("sddmm_gpu", stmt);

  A.compile(stmt);
  A.assemble();
  A.compute();

  Tensor<double> expected("expected", {NUM_I, NUM_K}, {Dense, Dense});
  expected(i,k) = B(i,k) * C(i,j) * D(j,k);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, A);
}

TEST(scheduling_eval, ttmGPU) {
  if (!should_use_CUDA_codegen()) {
    return;
  }
  int NUM_I = 1021/40;
  int NUM_J = 1039/40;
  int NUM_K = 1232/40;
  int NUM_L = 128;
  float SPARSITY = .1;
  Tensor<double> A("A", {NUM_I, NUM_J, NUM_L}, {Dense, Dense, Dense}); // TODO: change to sparse outputs
  Tensor<double> B("B", {NUM_I, NUM_J, NUM_K}, {Sparse, Sparse, Sparse});
  Tensor<double> C("C", {NUM_K, NUM_L}, {Dense, Dense});

  srand(34644);
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

  for (int k = 0; k < NUM_K; k++) {
    for (int l = 0; l < NUM_L; l++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      C.insert({k, l}, (double) ((int) (rand_float*3)));
    }
  }

  B.pack();
  C.pack();

  A(i,j,l) = B(i,j,k) * C(k,l);

  IndexStmt stmt = A.getAssignment().concretize();
  stmt = scheduleTTMGPU(stmt, B);

  //printToFile("ttm_gpu", stmt);

  A.compile(stmt);
  A.assemble();
  A.compute();

  Tensor<double> expected("expected", {NUM_I, NUM_J, NUM_L}, {Dense, Dense, Dense});
  expected(i,j,l) = B(i,j,k) * C(k,l);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, A);
}

TEST(scheduling_eval, ttvGPU) {
  if (!should_use_CUDA_codegen()) {
    return;
  }
  int NUM_I = 1021/10;
  int NUM_J = 1039/10;
  int NUM_K = 1057/10;
  float SPARSITY = .3;
  Tensor<double> A("A", {NUM_I, NUM_J}, {Dense, Dense}); // TODO: change to sparse outputs
  Tensor<double> B("B", {NUM_I, NUM_J, NUM_K}, {Sparse, Sparse, Sparse});
  Tensor<double> c("c", {NUM_K}, Format({Dense}));

  srand(35325);
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

  for (int k = 0; k < NUM_K; k++) {
    float rand_float = (float)rand()/(float)(RAND_MAX);
    c.insert({k}, (double) ((int) (rand_float*3)));
  }

  B.pack();
  c.pack();

  IndexExpr precomputedExpr = B(i,j,k) * c(k);
  A(i,j) = precomputedExpr;

  IndexStmt stmt = A.getAssignment().concretize();
  stmt = scheduleTTVGPU(stmt, B, precomputedExpr);

  //printToFile("ttv_gpu", stmt);

  A.compile(stmt);
  A.assemble();
  A.compute();

  Tensor<double> expected("expected", {NUM_I, NUM_J}, {Dense, Dense});
  expected(i,j) = B(i,j,k) * c(k);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, A);
}

TEST(scheduling_eval, mttkrpGPU) {
  if (!should_use_CUDA_codegen()) {
    return;
  }
  int NUM_I = 1021/40;
  int NUM_J = 32;
  int NUM_K = 1039/40;
  int NUM_L = 1232/40;
  float SPARSITY = .1;
  Tensor<double> A("A", {NUM_I, NUM_J}, {Dense, Dense});
  Tensor<double> B("B", {NUM_I, NUM_K, NUM_L}, {Sparse, Sparse, Sparse});
  Tensor<double> C("C", {NUM_K, NUM_J}, {Dense, Dense});
  Tensor<double> D("D", {NUM_L, NUM_J}, {Dense, Dense});

  srand(5464164);
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

  for (int k = 0; k < NUM_K; k++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      C.insert({k, j}, (double) ((int) (rand_float*3)));
    }
  }

  for (int l = 0; l < NUM_L; l++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      D.insert({l, j}, (double) ((int) (rand_float*3)));
    }
  }

  B.pack();
  C.pack();
  D.pack();

  A(i,j) = B(i,k,l) * C(k,j) * D(l,j);

  IndexStmt stmt = A.getAssignment().concretize();
  stmt = scheduleMTTKRPGPU(stmt, B);

  //printToFile("mttkrp_gpu", stmt);

  A.compile(stmt);
  A.assemble();
  A.compute();

  Tensor<double> expected("expected", {NUM_I, NUM_J}, {Dense, Dense});
  expected(i,j) = B(i,k,l) * C(k,j) * D(l,j);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, A);
}

TEST(generate_evaluation_files, ispc) {
  std::cout << "Hi Adhitha!\n" << std::endl ;
  set_CUDA_codegen_enabled(false);
  set_ISPC_codegen_enabled(true);

  vector<vector<int>> spmv_parameters = {{32}};
  vector<vector<int>> spmspv_parameters = {{8}};

  // 4 to 512 and 4, 8, 16
  vector<vector<int>> spmm_dcsr_parameters = {{16, 8}};
  vector<vector<int>> spmm_parameters = {{16,4}};

  vector<vector<int>> mttkrp_parameters = {};
  mttkrp_parameters.push_back({64,0});

  vector<vector<int>> sddmm_parameters = {{8, 8}};
  vector<vector<int>> ttv_parameters = {{32}};

  int NUM_I = 100;
  int NUM_J = 100;
  int NUM_K = 100;
  int NUM_L = 100;

  string c_file_ending = ".h";
  string file_ending = ".ispc";
  string file_path = "eval_prepared_ispc/";
  mkdir(file_path.c_str(), 0777);

  // spmv
  {
    stringstream source1;
    stringstream source2;
    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source1, source2, ir::CodeGen::ImplementationGen);
    Tensor<double> A("A", {NUM_I, NUM_J}, CSR);
    Tensor<double> x("x", {NUM_J}, {Dense});
    Tensor<double> y("y", {NUM_I}, {Dense});
    y(i) = A(i, j) * x(j);
    std::cout << "concretizing the assignment statement\n";
    IndexStmt stmt = y.getAssignment().concretize();
    std::cout << "Printing the original IndexStmt: " << stmt << std::endl;

    for (auto paramSet : spmv_parameters) {
      std::cout << "param set: " << paramSet[0] << std::endl;
      IndexStmt scheduled = scheduleSpMVISPC(stmt, paramSet[0]);
      std::cout << "scheduled IndexStmt: " << scheduled << std::endl;
      ir::Stmt compute = lower(scheduled, string("compute_") + util::join(paramSet, "_"),  false, true);
      std::cout << "computed statement: \n" << compute << std::endl;
      codegen->compile(compute, false);
    }
    ofstream source_file;
    source_file.open(file_path + "spmv_csr_ispc_taco" + c_file_ending);
    source_file << source1.str();
    source_file.close();

    ofstream ispc_source_file;
    ispc_source_file.open(file_path + "__spmv_csr_ispc_taco" + file_ending);
    ispc_source_file << source2.str();
    ispc_source_file.close();
    
  }

  // spmm
  {
    stringstream source1;
    stringstream source2;
    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source1, source2, ir::CodeGen::ImplementationGen);
    Tensor<double> A("A", {NUM_I, NUM_J}, CSR);
    Tensor<double> X("X", {NUM_J, NUM_K}, {Dense, Dense});
    Tensor<double> Y("Y", {NUM_I, NUM_K}, {Dense, Dense});
    Y(i, k) = A(i, j) * X(j, k);
    IndexStmt stmt = Y.getAssignment().concretize();
    bool isFirst = true;
    for (auto paramSet : spmm_parameters) {
      IndexStmt scheduled = scheduleSpMMISPC1(stmt, A, paramSet[0], paramSet[1]);
      ir::Stmt compute = lower(scheduled, string("compute1_") + util::join(paramSet, "_"),  false, true);
      codegen->compile(compute, isFirst);
      isFirst = false;
    }
    ofstream source_file;
    source_file.open(file_path + "spmm_csr_ispc_taco1" + c_file_ending);
    source_file << source1.str();
    source_file.close();

    ofstream ispc_source_file;
    ispc_source_file.open(file_path + "__spmm_csr_ispc_taco1" + file_ending);
    ispc_source_file << source2.str();
    ispc_source_file.close();
  }

  // spmm omp
  {
    stringstream source1;
    stringstream source2;
    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source1, source2, ir::CodeGen::ImplementationGen);
    Tensor<double> A("A", {NUM_I, NUM_J}, CSR);
    Tensor<double> X("X", {NUM_J, NUM_K}, {Dense, Dense});
    Tensor<double> Y("Y", {NUM_I, NUM_K}, {Dense, Dense});
    Y(i, k) = A(i, j) * X(j, k);
    IndexStmt stmt = Y.getAssignment().concretize();
    bool isFirst = true;
    for (auto paramSet : spmm_parameters) {
      IndexStmt scheduled = scheduleSpMMISPCOMP1(stmt, A, paramSet[0], paramSet[1]);
      ir::Stmt compute = lower(scheduled, string("compute1_") + util::join(paramSet, "_"),  false, true);
      codegen->compile(compute, isFirst);
      isFirst = false;
    }
    ofstream source_file;
    source_file.open(file_path + "spmm_omp_ispc_taco1" + c_file_ending);
    source_file << source1.str();
    source_file.close();

    ofstream ispc_source_file;
    ispc_source_file.open(file_path + "__spmm_omp_ispc_taco1" + file_ending);
    ispc_source_file << source2.str();
    ispc_source_file.close();
  }

  // spmm2
  {
    stringstream source1;
    stringstream source2;
    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source1, source2, ir::CodeGen::ImplementationGen);
    Tensor<double> A("A", {NUM_I, NUM_J}, CSR);
    Tensor<double> X("X", {NUM_J, NUM_K}, {Dense, Dense});
    Tensor<double> Y("Y", {NUM_I, NUM_K}, {Dense, Dense});
    Y(i, k) = A(i, j) * X(j, k);
    IndexStmt stmt = Y.getAssignment().concretize();
    bool isFirst = true;
    for (auto paramSet : spmm_parameters) {
      IndexStmt scheduled = scheduleSpMMISPC2(stmt, A, paramSet[0], paramSet[1]);
      ir::Stmt compute = lower(scheduled, string("compute2_") + util::join(paramSet, "_"),  false, true);
      codegen->compile(compute, isFirst);
      isFirst = false;
    }
    ofstream source_file;
    source_file.open(file_path + "spmm_csr_ispc_taco2" + c_file_ending);
    source_file << source1.str();
    source_file.close();

    ofstream ispc_source_file;
    ispc_source_file.open(file_path + "__spmm_csr_ispc_taco2" + file_ending);
    ispc_source_file << source2.str();
    ispc_source_file.close();
  }

  // spmm
  {
    stringstream source1;
    stringstream source2;
    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source1, source2, ir::CodeGen::ImplementationGen);
    Tensor<double> A("A", {NUM_I, NUM_J}, CSR);
    Tensor<double> X("X", {NUM_J, NUM_K}, {Dense, Dense});
    Tensor<double> Y("Y", {NUM_I, NUM_K}, {Dense, Dense});
    Y(i, k) = A(i, j) * X(j, k);
    IndexStmt stmt = Y.getAssignment().concretize();
    bool isFirst = true;
    for (auto paramSet : spmm_parameters) {
      IndexStmt scheduled = scheduleSpMMISPC3(stmt, A, paramSet[0], paramSet[1]);
      ir::Stmt compute = lower(scheduled, string("compute3_") + util::join(paramSet, "_"),  false, true);
      codegen->compile(compute, isFirst);
      isFirst = false;
    }
    ofstream source_file;
    source_file.open(file_path + "spmm_csr_ispc_taco3" + c_file_ending);
    source_file << source1.str();
    source_file.close();

    ofstream ispc_source_file;
    ispc_source_file.open(file_path + "__spmm_csr_ispc_taco3" + file_ending);
    ispc_source_file << source2.str();
    ispc_source_file.close();
  }

  // ttv
  {
    stringstream source;
    stringstream source2;
    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, source2, ir::CodeGen::ImplementationGen);
    Tensor<double> A("A", {NUM_I, NUM_J}, {Dense, Dense}); // TODO: change to sparse outputs
    Tensor<double> B("B", {NUM_I, NUM_J, NUM_K}, {Sparse, Sparse, Sparse});
    Tensor<double> c("c", {NUM_K}, Format({Dense}));
    A(i,j) = B(i,j,k) * c(k);
    IndexStmt stmt = A.getAssignment().concretize();
    bool isFirst = true;
    for (auto paramSet : ttv_parameters) {
      IndexStmt scheduled = scheduleTTVCPU(stmt, B, paramSet[0]);
      ir::Stmt compute = lower(scheduled, string("compute_") + util::join(paramSet, "_"),  false, true);
      codegen->compile(compute, isFirst);
      isFirst = false;
    }
    ofstream source_file;
    source_file.open(file_path + "ttv_cpu" + c_file_ending);
    source_file << source.str();
    source_file.close();

    ofstream ispc_source_file;
    ispc_source_file.open(file_path + "__ttv_cpu" + file_ending);
    ispc_source_file << source2.str();
    ispc_source_file.close();
  }


  // mttkrp3
  {
    stringstream source;
    stringstream source2;
    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, source2, ir::CodeGen::ImplementationGen);
    Tensor<double> A("A", {NUM_I, NUM_J}, {Dense, Dense});
    Tensor<double> B("B", {NUM_I, NUM_K, NUM_L}, {Dense, Sparse, Sparse});
    Tensor<double> C("C", {NUM_K, NUM_J}, {Dense, Dense});
    Tensor<double> D("D", {NUM_L, NUM_J}, {Dense, Dense});
    A(i,j) = B(i,k,l) * C(k,j) * D(l,j);
    IndexStmt stmt = A.getAssignment().concretize();
    bool isFirst = true;
    for (auto paramSet : mttkrp_parameters) {
      IndexStmt scheduled = scheduleMTTKRPCPU(stmt, B, paramSet[0], paramSet[1]);
      ir::Stmt compute = lower(scheduled, string("compute_") + util::join(paramSet, "_"),  false, true);
      codegen->compile(compute, isFirst);
      isFirst = false;
    }
    ofstream source_file;
    source_file.open(file_path + "mttkrp3_cpu" + c_file_ending);
    source_file << source.str();
    source_file.close();

    ofstream ispc_source_file;
    ispc_source_file.open(file_path + "__mttkrp3_cpu" + file_ending);
    ispc_source_file << source2.str();
    ispc_source_file.close();
  }


  return;
}



TEST(generate_ispc_sddmm_evaluation_files, ispc) {
  std::cout << "Hi Adhitha!\n" << std::endl ;
  set_CUDA_codegen_enabled(false);
  set_ISPC_codegen_enabled(true);

  vector<vector<int>> spmv_parameters = {{32}};
  vector<vector<int>> spmspv_parameters = {{8}};

  // 4 to 512 and 4, 8, 16
  vector<vector<int>> spmm_dcsr_parameters = {{16, 8}};
  vector<vector<int>> spmm_parameters = {{16,4}};

  vector<vector<int>> mttkrp_parameters = {};
  mttkrp_parameters.push_back({64,0});

  vector<vector<int>> sddmm_parameters = {{8, 8}};
  vector<vector<int>> ttv_parameters = {{32}};

  int NUM_I = 100;
  int NUM_J = 100;
  int NUM_K = 100;

  string c_file_ending = ".h";
  string file_ending = ".ispc";
  string file_path = "eval_prepared_ispc/sddmm/";
  mkdir(file_path.c_str(), 0777);

  // sddmm
  {
    stringstream source1;
    stringstream source2;
    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source1, source2, ir::CodeGen::ImplementationGen);
    Tensor<double> A("A", {NUM_I, NUM_K}, {Dense, Dense});
    Tensor<double> B("B", {NUM_I, NUM_K}, CSR);
    Tensor<double> C("C", {NUM_I, NUM_J}, {Dense, Dense});
    Tensor<double> D("D", {NUM_J, NUM_K}, {Dense, Dense});
    A(i,k) = B(i,k) * C(i,j) * D(j,k);
    IndexStmt stmt = A.getAssignment().concretize();
    bool isFirst = true;
    for (auto paramSet : sddmm_parameters) {
      IndexStmt scheduled = scheduleSDDMMISPC1(stmt, B, paramSet[0], paramSet[1]);
      ir::Stmt compute = lower(scheduled, string("compute1_") + util::join(paramSet, "_"),  false, true);
      codegen->compile(compute, isFirst);
      isFirst = false;
    }
    ofstream source_file;
    source_file.open(file_path + "sddmm_cpu_ispc_taco1" + file_ending);
    source_file << source1.str();
    source_file.close();

    ofstream ispc_source_file;
    ispc_source_file.open(file_path + "__sddmm_cpu_ispc_taco1" + file_ending);
    ispc_source_file << source2.str();
    ispc_source_file.close();
  }


  // sddmm
  {
    stringstream source1;
    stringstream source2;
    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source1, source2, ir::CodeGen::ImplementationGen);
    Tensor<double> Y("Y", {NUM_I, NUM_K}, {Dense, Dense});
    Tensor<double> A("A", {NUM_I, NUM_K}, CSR);
    Tensor<double> X("X", {NUM_I, NUM_J}, {Dense, Dense});
    Y(i,j) = A(i,j) * X(i,k) * X(j,k);
    IndexStmt stmt = Y.getAssignment().concretize();
    bool isFirst = true;
    for (auto paramSet : sddmm_parameters) {
      IndexStmt scheduled = scheduleSDDMMISPC2(stmt, A, paramSet[0], paramSet[1]);
      ir::Stmt compute = lower(scheduled, string("compute2_") + util::join(paramSet, "_"),  false, true);
      codegen->compile(compute, isFirst);
      isFirst = false;
    }
    ofstream source_file;
    source_file.open(file_path + "sddmm_cpu_ispc_taco2" + file_ending);
    source_file << source1.str();
    source_file.close();

    ofstream ispc_source_file;
    ispc_source_file.open(file_path + "__sddmm_cpu_ispc_taco2" + file_ending);
    ispc_source_file << source2.str();
    ispc_source_file.close();
  }


  return;
}




TEST(generate_evaluation_files, cpu) {
  if (should_use_CUDA_codegen()) {
    return;
  }
  vector<vector<int>> spmv_parameters = {{8}, {16}, {32}};

  // 4 to 512 and 4, 8, 16
  vector<vector<int>> spmm_dcsr_parameters = {{16, 8}};
  vector<vector<int>> spmm_parameters = {};

  for (int i = 4; i <= 512; i *= 2) {
    for (int j = 4; j <= 16; j *= 2) {
      spmm_parameters.push_back({i,j});
    }
  }

  vector<vector<int>> mttkrp_parameters = {};
  for (int i = 1; i <= 64; i *= 2) {
    mttkrp_parameters.push_back({i,0});

  }
  vector<vector<int>> sddmm_parameters = {{16, 8}, {8, 8}};
  vector<vector<int>> ttv_parameters = {{16}, {8}, {32}};
  vector<vector<int>> ttm_parameters = {{16, 8}, {8, 8}};

  int NUM_I = 100;
  int NUM_J = 100;
  int NUM_K = 100;
  int NUM_L = 100;
  int NUM_M = 100;
  int NUM_N = 100;

  string file_ending = should_use_CUDA_codegen() ? ".cu" : ".c";
  string file_path = "eval_prepared_cpu/";
  mkdir(file_path.c_str(), 0777);

  // spmv
  {
    stringstream source;
    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
    Tensor<double> A("A", {NUM_I, NUM_J}, CSR);
    Tensor<double> x("x", {NUM_J}, Format({Dense}));
    Tensor<double> y("y", {NUM_I}, Format({Dense}));
    y(i) = A(i, j) * x(j);
    IndexStmt stmt = y.getAssignment().concretize();
    bool isFirst = true;
    for (auto paramSet : spmv_parameters) {
      IndexStmt scheduled = scheduleSpMVCPU(stmt, paramSet[0]);
      ir::Stmt compute = lower(scheduled, string("compute_") + util::join(paramSet, "_"),  false, true);
      codegen->compile(compute, isFirst);
      isFirst = false;
    }
    ofstream source_file;
    source_file.open(file_path + "spmv_cpu" + file_ending);
    source_file << source.str();
    source_file.close();
  }

  // spmm
  {
    stringstream source;
    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
    Tensor<double> A("A", {NUM_I, NUM_J}, CSR);
    Tensor<double> B("B", {NUM_J, NUM_K}, {Dense, Dense});
    Tensor<double> C("C", {NUM_I, NUM_K}, {Dense, Dense});
    C(i, k) = A(i, j) * B(j, k);
    IndexStmt stmt = C.getAssignment().concretize();
    bool isFirst = true;
    for (auto paramSet : spmm_parameters) {
      IndexStmt scheduled = scheduleSpMMCPU(stmt, A, paramSet[0], paramSet[1]);
      ir::Stmt compute = lower(scheduled, string("compute_") + util::join(paramSet, "_"),  false, true);
      codegen->compile(compute, isFirst);
      isFirst = false;
    }
    ofstream source_file;
    source_file.open(file_path + "spmm_cpu" + file_ending);
    source_file << source.str();
    source_file.close();
  }

  // sddmm
  {
    stringstream source;
    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
    Tensor<double> A("A", {NUM_I, NUM_K}, {Dense, Dense});
    Tensor<double> B("B", {NUM_I, NUM_K}, CSR);
    Tensor<double> C("C", {NUM_I, NUM_J}, {Dense, Dense});
    Tensor<double> D("D", {NUM_J, NUM_K}, {Dense, Dense});
    A(i,k) = B(i,k) * C(i,j) * D(j,k);
    IndexStmt stmt = A.getAssignment().concretize();
    bool isFirst = true;
    for (auto paramSet : sddmm_parameters) {
      IndexStmt scheduled = scheduleSDDMMCPU(stmt, B, paramSet[0], paramSet[1]);
      ir::Stmt compute = lower(scheduled, string("compute_") + util::join(paramSet, "_"),  false, true);
      codegen->compile(compute, isFirst);
      isFirst = false;
    }
    ofstream source_file;
    source_file.open(file_path + "sddmm_cpu" + file_ending);
    source_file << source.str();
    source_file.close();
  }

  // ttv
  {
    stringstream source;
    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
    Tensor<double> A("A", {NUM_I, NUM_J}, {Dense, Dense}); // TODO: change to sparse outputs
    Tensor<double> B("B", {NUM_I, NUM_J, NUM_K}, {Sparse, Sparse, Sparse});
    Tensor<double> c("c", {NUM_K}, Format({Dense}));
    A(i,j) = B(i,j,k) * c(k);
    IndexStmt stmt = A.getAssignment().concretize();
    bool isFirst = true;
    for (auto paramSet : ttv_parameters) {
      IndexStmt scheduled = scheduleTTVCPU(stmt, B, paramSet[0]);
      ir::Stmt compute = lower(scheduled, string("compute_") + util::join(paramSet, "_"),  false, true);
      codegen->compile(compute, isFirst);
      isFirst = false;
    }
    ofstream source_file;
    source_file.open(file_path + "ttv_cpu" + file_ending);
    source_file << source.str();
    source_file.close();
  }

  // ttm
  {
    stringstream source;
    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
    Tensor<double> A("A", {NUM_I, NUM_J, NUM_L}, {Dense, Dense, Dense}); // TODO: change to sparse outputs
    Tensor<double> B("B", {NUM_I, NUM_J, NUM_K}, {Sparse, Sparse, Sparse});
    Tensor<double> C("C", {NUM_K, NUM_L}, {Dense, Dense});
    A(i,j,l) = B(i,j,k) * C(k,l);
    IndexStmt stmt = A.getAssignment().concretize();
    bool isFirst = true;
    for (auto paramSet : ttm_parameters) {
      IndexStmt scheduled = scheduleTTMCPU(stmt, B, paramSet[0], paramSet[1]);
      ir::Stmt compute = lower(scheduled, string("compute_") + util::join(paramSet, "_"),  false, true);
      codegen->compile(compute, isFirst);
      isFirst = false;
    }
    ofstream source_file;
    source_file.open(file_path + "ttm_cpu" + file_ending);
    source_file << source.str();
    source_file.close();
  }

  // mttkrp3
  {
    stringstream source;
    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
    Tensor<double> A("A", {NUM_I, NUM_J}, {Dense, Dense});
    Tensor<double> B("B", {NUM_I, NUM_K, NUM_L}, {Dense, Sparse, Sparse});
    Tensor<double> C("C", {NUM_K, NUM_J}, {Dense, Dense});
    Tensor<double> D("D", {NUM_L, NUM_J}, {Dense, Dense});
    A(i,j) = B(i,k,l) * C(k,j) * D(l,j);
    IndexStmt stmt = A.getAssignment().concretize();
    bool isFirst = true;
    for (auto paramSet : mttkrp_parameters) {
      IndexStmt scheduled = scheduleMTTKRPCPU(stmt, B, paramSet[0], paramSet[1]);
      ir::Stmt compute = lower(scheduled, string("compute_") + util::join(paramSet, "_"),  false, true);
      codegen->compile(compute, isFirst);
      isFirst = false;
    }
    ofstream source_file;
    source_file.open(file_path + "mttkrp3_cpu" + file_ending);
    source_file << source.str();
    source_file.close();
  }

  // mttkrp3 workspace
  {
    stringstream source;
    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
    Tensor<double> A("A", {NUM_I, NUM_J}, {Dense, Dense});
    Tensor<double> B("B", {NUM_I, NUM_K, NUM_L}, {Dense, Sparse, Sparse});
    Tensor<double> C("C", {NUM_K, NUM_J}, {Dense, Dense});
    Tensor<double> D("D", {NUM_L, NUM_J}, {Dense, Dense});
    IndexExpr precomputedExpr = B(i,k,l) * D(l,j);
    A(i,j) = precomputedExpr * C(k,j);
    IndexStmt stmt = A.getAssignment().concretize();
    TensorVar precomputed("precomputed", Type(Float64, {Dimension(j)}), taco::dense);

    IndexStmt precomputed_stmt = forall(i, forall(k,
                      where(forall(j, A(i,j) += precomputed(j) * C(k,j)),
                            forall(l, forall(j, precomputed(j) += B(i,k,l) * D(l,j))))));
    IndexStmt scheduled = scheduleMTTKRPPrecomputedCPU(precomputed_stmt, B, 64);
    ir::Stmt compute = lower(scheduled, string("mttkrp3_workspace"),  false, true);
    codegen->compile(compute, true);

    ofstream source_file;
    source_file.open(file_path + "mttkrp3_cpu_workspace" + file_ending);
    source_file << source.str();
    source_file.close();
  }

  // mttkrp4
  {
    stringstream source;
    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
    Tensor<double> A("A", {NUM_I, NUM_J}, {Dense, Dense});
    Tensor<double> B("B", {NUM_I, NUM_K, NUM_L, NUM_M}, {Dense, Sparse, Sparse, Sparse});
    Tensor<double> C("C", {NUM_K, NUM_J}, {Dense, Dense});
    Tensor<double> D("D", {NUM_L, NUM_J}, {Dense, Dense});
    Tensor<double> E("E", {NUM_M, NUM_J}, {Dense, Dense});
    A(i,j) = B(i,k,l,m) * C(k,j) * D(l,j) * E(m,j);
    IndexStmt stmt = A.getAssignment().concretize();
    bool isFirst = true;
    for (auto paramSet : mttkrp_parameters) {
      IndexStmt scheduled = scheduleMTTKRP4CPU(stmt, B, paramSet[0], paramSet[1]);
      ir::Stmt compute = lower(scheduled, string("compute_") + util::join(paramSet, "_"),  false, true);
      codegen->compile(compute, isFirst);
      isFirst = false;
    }
    ofstream source_file;
    source_file.open(file_path + "mttkrp4_cpu" + file_ending);
    source_file << source.str();
    source_file.close();
  }

  // mttkrp4 workspace
  {
    stringstream source;
    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
    Tensor<double> A("A", {NUM_I, NUM_J}, {Dense, Dense});
    Tensor<double> B("B", {NUM_I, NUM_K, NUM_L, NUM_M}, {Dense, Sparse, Sparse, Sparse});
    Tensor<double> C("C", {NUM_K, NUM_J}, {Dense, Dense});
    Tensor<double> D("D", {NUM_L, NUM_J}, {Dense, Dense});
    Tensor<double> E("E", {NUM_M, NUM_J}, {Dense, Dense});
    A(i,j) = B(i,k,l,m) * C(k,j) * D(l,j) * E(m,j);

    IndexExpr BE = B(i,k,l,m) * E(m,j);
    IndexExpr BDE = BE * D(l, j);
    A(i,j) = BDE * C(k,j);
    IndexStmt stmt = A.getAssignment().concretize();
    TensorVar BE_workspace("BE_workspace", Type(Float64, {Dimension(j)}), taco::dense);
    TensorVar BDE_workspace("BDE_workspace", Type(Float64, {Dimension(j)}), taco::dense);

    IndexStmt precomputed_stmt = forall(i, forall(k,
            where(forall(j, A(i,j) += BDE_workspace(j) * C(k,j)),
              forall(l, where(forall(j, BDE_workspace(j) += BE_workspace(j) * D(l,j)),
                  forall(m, forall(j, BE_workspace(j) += B(i,k,l,m) * E(m,j))))))));

    IndexStmt scheduled = scheduleMTTKRPPrecomputedCPU(precomputed_stmt, B, 64);
    ir::Stmt compute = lower(scheduled, string("mttkrp4_workspace"),  false, true);
    codegen->compile(compute, true);

    ofstream source_file;
    source_file.open(file_path + "mttkrp4_cpu_workspace" + file_ending);
    source_file << source.str();
    source_file.close();
  }

  // mttkrp5
  {
    stringstream source;
    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
    Tensor<double> A("A", {NUM_I, NUM_J}, {Dense, Dense});
    Tensor<double> B("B", {NUM_I, NUM_K, NUM_L, NUM_M, NUM_N}, {Dense, Sparse, Sparse, Sparse, Sparse});
    Tensor<double> C("C", {NUM_K, NUM_J}, {Dense, Dense});
    Tensor<double> D("D", {NUM_L, NUM_J}, {Dense, Dense});
    Tensor<double> E("E", {NUM_M, NUM_J}, {Dense, Dense});
    Tensor<double> F("F", {NUM_N, NUM_J}, {Dense, Dense});
    A(i,j) = B(i,k,l,m,n) * C(k,j) * D(l,j) * E(m,j) * F(n,j);
    IndexStmt stmt = A.getAssignment().concretize();
    bool isFirst = true;
    for (auto paramSet : mttkrp_parameters) {
      IndexStmt scheduled = scheduleMTTKRP5CPU(stmt, B, paramSet[0], paramSet[1]);
      ir::Stmt compute = lower(scheduled, string("compute_") + util::join(paramSet, "_"),  false, true);
      codegen->compile(compute, isFirst);
      isFirst = false;
    }
    ofstream source_file;
    source_file.open(file_path + "mttkrp5_cpu" + file_ending);
    source_file << source.str();
    source_file.close();
  }

  // mttkrp5 workspace
  {
    stringstream source;
    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
    Tensor<double> A("A", {NUM_I, NUM_J}, {Dense, Dense});
    Tensor<double> B("B", {NUM_I, NUM_K, NUM_L, NUM_M, NUM_N}, {Dense, Sparse, Sparse, Sparse, Sparse});
    Tensor<double> C("C", {NUM_K, NUM_J}, {Dense, Dense});
    Tensor<double> D("D", {NUM_L, NUM_J}, {Dense, Dense});
    Tensor<double> E("E", {NUM_M, NUM_J}, {Dense, Dense});
    Tensor<double> F("F", {NUM_N, NUM_J}, {Dense, Dense});
    A(i,j) = B(i,k,l,m,n) * C(k,j) * D(l,j) * E(m,j) * F(n,j);
    IndexStmt stmt = A.getAssignment().concretize();

    IndexExpr BF = B(i,k,l,m,n) * F(n,j);
    IndexExpr BEF = BF * E(m,j);
    IndexExpr BDEF = BEF * D(l, j);
    A(i,j) = BDEF * C(k,j);
    TensorVar BF_workspace("BF_workspace", Type(Float64, {Dimension(j)}), taco::dense);
    TensorVar BEF_workspace("BEF_workspace", Type(Float64, {Dimension(j)}), taco::dense);
    TensorVar BDEF_workspace("BDEF_workspace", Type(Float64, {Dimension(j)}), taco::dense);

    IndexStmt precomputed_stmt = forall(i, forall(k,
            where(forall(j, A(i,j) += BDEF_workspace(j) * C(k,j)),
               forall(l, where(forall(j, BDEF_workspace(j) += BEF_workspace(j) * D(l,j)),
                 forall(m, where(forall(j, BEF_workspace(j) += BF_workspace(j) * E(m,j)),
                   forall(n, forall(j, BF_workspace(j) += B(i,k,l,m,n)*F(n,j))))))))));

    IndexStmt scheduled = scheduleMTTKRPPrecomputedCPU(precomputed_stmt, B, 64);
    ir::Stmt compute = lower(scheduled, string("mttkrp5_workspace"),  false, true);
    codegen->compile(compute, true);

    ofstream source_file;
    source_file.open(file_path + "mttkrp5_cpu_workspace" + file_ending);
    source_file << source.str();
    source_file.close();
  }
}

TEST(generate_evaluation_files, spmv_ispc) {
  set_CUDA_codegen_enabled(false);
  set_ISPC_codegen_enabled(true);

  std::cout << "executing generate_evaluation_file.ispc\n";

  int NUM_I = 100;
  int NUM_J = 100;

  vector<vector<int>> spmv_parameters = {}; // {NNZ_PER_THREAD, BLOCK_SIZE}
  for (int i = 3; i <= 20; i++) {
    spmv_parameters.push_back({i, 512});
  }

  string file_ending_c = ".c";
  string file_ending_ispc = ".ispc";
  string file_path = "eval_prepared_ispc/spmv/";
  mkdir(file_path.c_str(), 0777);

    // spmv
  {
    stringstream source1;
    stringstream source2;
    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source1, source2, ir::CodeGen::ImplementationGen);
    Tensor<double> A("A", {NUM_I, NUM_J}, CSR);
    Tensor<double> x("x", {NUM_J}, Format({Dense}));
    Tensor<double> y("y", {NUM_I}, Format({Dense}));
    IndexExpr precomputed = A(i, j) * x(j);
    y(i) = precomputed;
    IndexStmt stmt = y.getAssignment().concretize();
    bool isFirst = true;
    for (auto paramSet : spmv_parameters) {
      IndexStmt scheduled = scheduleSpMVCPU(stmt);
      ir::Stmt compute = lower(scheduled, string("compute_") + util::join(paramSet, "_"),  false, true);
      codegen->compile(compute, isFirst);
      isFirst = false;
    }
    ofstream source_file1;
    source_file1.open(file_path + "spmv_ispc" + file_ending_c);
    source_file1 << source1.str();
    source_file1.close();

    ofstream source_file2;
    source_file2.open(file_path + "__spmv_ispc" + file_ending_ispc);
    source_file2 << source2.str();
    source_file2.close();
  }
}

TEST(generate_evaluation_files, gpu) {
  // if (!should_use_CUDA_codegen()) {
  //   return;
  // }
  set_CUDA_codegen_enabled(true);
  set_ISPC_codegen_enabled(false);

  std::cout << "executing generate_evaluation_file.gpu\n";

  vector<vector<int>> spmv_parameters = {}; // {NNZ_PER_THREAD, BLOCK_SIZE}
  for (int i = 3; i <= 20; i++) {
    spmv_parameters.push_back({i, 512});
  }
  vector<vector<int>> spmm_parameters = {}; // {NNZ_PER_WARP, BLOCK_SIZE, CO_FACTOR}

  // 4, 8, ... 32 for NNZ_PER_WARP 512 block size
  for (int i = 4; i <= 32; i += 4) {
    spmm_parameters.push_back({i,512});
  }

  vector<vector<int>> mttkrp_parameters = spmm_parameters; // {NNZ_PER_WARP, BLOCK_SIZE, CO_FACTOR}

  vector<vector<int>> spmm_dcsr_parameters = {{4, 256, 4}}; // {NNZ_PER_WARP, BLOCK_SIZE, CO_FACTOR}
  vector<vector<int>> sddmm_parameters = {{8*32, 256, 4}, {4*32, 512, 4}}; // {NNZ_PER_WARP, BLOCK_SIZE, CO_FACTOR}
  vector<vector<int>> ttv_parameters = {{8*32, 256}, {4*32, 512}}; // {NNZ_PER_WARP, BLOCK_SIZE}
  vector<vector<int>> ttm_parameters = {{8*32, 256, 4}, {4*32, 512, 8}}; // {NNZ_PER_WARP, BLOCK_SIZE, CO_FACTOR}

  int NUM_I = 100;
  int NUM_J = 100;
  int NUM_K = 100;
  int NUM_L = 100;

  string file_ending = should_use_CUDA_codegen() ? ".cu" : ".c";
  string file_path = "eval_prepared_gpu/";
  mkdir(file_path.c_str(), 0777);

  // spmv load-balance
  {
    stringstream source;
    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
    Tensor<double> A("A", {NUM_I, NUM_J}, CSR);
    Tensor<double> x("x", {NUM_J}, Format({Dense}));
    Tensor<double> y("y", {NUM_I}, Format({Dense}));
    IndexExpr precomputed = A(i, j) * x(j);
    y(i) = precomputed;
    IndexStmt stmt = y.getAssignment().concretize();
    bool isFirst = true;

    IndexStmt scheduled = scheduleSpMVRowsGPU(stmt, A, precomputed);
    ir::Stmt compute = lower(scheduled, string("compute_warp_row"),  false, true);
    codegen->compile(compute, isFirst);
    isFirst = false;

    scheduled = scheduleSpMVThreadPerRowGPU(stmt, A, precomputed);
    compute = lower(scheduled, string("compute_thread_row"),  false, true);
    codegen->compile(compute, isFirst);


    ofstream source_file;
    source_file.open(file_path + "spmv_gpu_warp_vs_thread" + file_ending);
    source_file << source.str();
    source_file.close();
  }

  // spmv
  {
    stringstream source;
    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
    Tensor<double> A("A", {NUM_I, NUM_J}, CSR);
    Tensor<double> x("x", {NUM_J}, Format({Dense}));
    Tensor<double> y("y", {NUM_I}, Format({Dense}));
    IndexExpr precomputed = A(i, j) * x(j);
    y(i) = precomputed;
    IndexStmt stmt = y.getAssignment().concretize();
    bool isFirst = true;
    for (auto paramSet : spmv_parameters) {
      IndexStmt scheduled = scheduleSpMVGPU(stmt, A, precomputed, paramSet[0], paramSet[1]);
      ir::Stmt compute = lower(scheduled, string("compute_") + util::join(paramSet, "_"),  false, true);
      codegen->compile(compute, isFirst);
      isFirst = false;
    }
    ofstream source_file;
    source_file.open(file_path + "spmv_gpu" + file_ending);
    source_file << source.str();
    source_file.close();
  }

  // spmm
  {
    stringstream source;
    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
    Tensor<double> A("A", {NUM_I, NUM_J}, CSR);
    bool isFirst = true;
    for (auto paramSet : spmm_parameters) {
      int NUM_K = 128;
      Tensor<double> B("B", {NUM_J, NUM_K}, {Dense, Dense});
      Tensor<double> C("C", {NUM_I, NUM_K}, Format({{Dense, Dense}, {1, 0}}));
      IndexExpr precomputed = A(i, j);
      C(i, k) = precomputed * B(j, k);
      IndexStmt stmt = C.getAssignment().concretize();
      IndexStmt scheduled = scheduleSpMMGPU(stmt, A, precomputed, paramSet[0], paramSet[1]);
      ir::Stmt compute = lower(scheduled, string("compute_") + util::join(paramSet, "_"),  false, true);
      codegen->compile(compute, isFirst);
      isFirst = false;
    }
    ofstream source_file;
    source_file.open(file_path + "spmm_gpu" + file_ending);
    source_file << source.str();
    source_file.close();
  }

  // sddmm
  {
    stringstream source;

    Tensor<double> C("C", {NUM_I, NUM_J}, {Dense, Dense});
    bool isFirst = true;
    for (auto paramSet : sddmm_parameters) {
      int NUM_K = paramSet[2] * WARP_SIZE;
      std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
      Tensor<double> A("A", {NUM_I, NUM_K}, {Dense, Dense});
      Tensor<double> B("B", {NUM_I, NUM_K}, CSR);
      Tensor<double> D("D", {NUM_J, NUM_K}, {Dense, Dense});
      A(i,k) = B(i,k) * C(i,j) * D(j,k);
      IndexStmt stmt = A.getAssignment().concretize();
      IndexStmt scheduled = scheduleSDDMMGPU(stmt, B, paramSet[0], paramSet[1], paramSet[2]);
      ir::Stmt compute = lower(scheduled, string("compute_") + util::join(paramSet, "_"),  false, true);
      codegen->compile(compute, isFirst);
      isFirst = false;
    }
    ofstream source_file;
    source_file.open(file_path + "sddmm_gpu" + file_ending);
    source_file << source.str();
    source_file.close();
  }

  // ttv
  {
    stringstream source;
    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
    Tensor<double> A("A", {NUM_I, NUM_J}, {Dense, Dense}); // TODO: change to sparse outputs
    Tensor<double> B("B", {NUM_I, NUM_J, NUM_K}, {Sparse, Sparse, Sparse});
    Tensor<double> c("c", {NUM_K}, Format({Dense}));
    IndexExpr precomputedExpr = B(i,j,k) * c(k);
    A(i,j) = precomputedExpr;
    IndexStmt stmt = A.getAssignment().concretize();
    bool isFirst = true;
    for (auto paramSet : ttv_parameters) {
      IndexStmt scheduled = scheduleTTVGPU(stmt, B, precomputedExpr, paramSet[0], paramSet[1]);
      ir::Stmt compute = lower(scheduled, string("compute_") + util::join(paramSet, "_"),  false, true);
      codegen->compile(compute, isFirst);
      isFirst = false;
    }
    ofstream source_file;
    source_file.open(file_path + "ttv_gpu" + file_ending);
    source_file << source.str();
    source_file.close();
  }

  // ttm
  {
    stringstream source;
    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
    Tensor<double> A("A", {NUM_I, NUM_J, NUM_L}, {Dense, Dense, Dense}); // TODO: change to sparse outputs
    bool isFirst = true;
    for (auto paramSet : ttm_parameters) {
      int NUM_K = paramSet[2] * WARP_SIZE;
      Tensor<double> B("B", {NUM_I, NUM_J, NUM_K}, {Sparse, Sparse, Sparse});
      Tensor<double> C("C", {NUM_K, NUM_L}, {Dense, Dense});
      A(i,j,l) = B(i,j,k) * C(k,l);
      IndexStmt stmt = A.getAssignment().concretize();
      IndexStmt scheduled = scheduleTTMGPU(stmt, B, paramSet[0], paramSet[1], paramSet[2]);
      ir::Stmt compute = lower(scheduled, string("compute_") + util::join(paramSet, "_"),  false, true);
      codegen->compile(compute, isFirst);
      isFirst = false;
    }
    ofstream source_file;
    source_file.open(file_path + "ttm_gpu" + file_ending);
    source_file << source.str();
    source_file.close();
  }

  // mttkrp
  {
    stringstream source;
    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
    Tensor<double> B("B", {NUM_I, NUM_K, NUM_L}, {Sparse, Sparse, Sparse});

    bool isFirst = true;
    for (auto paramSet : mttkrp_parameters) {
      int NUM_J = WARP_SIZE;
      Tensor<double> A("A", {NUM_I, NUM_J}, {Dense, Dense});
      Tensor<double> C("C", {NUM_K, NUM_J}, {Dense, Dense});
      Tensor<double> D("D", {NUM_L, NUM_J}, {Dense, Dense});
      A(i,j) = B(i,k,l) * C(k,j) * D(l,j);
      IndexStmt stmt = A.getAssignment().concretize();
      IndexStmt scheduled = scheduleMTTKRPGPU(stmt, B, paramSet[0], paramSet[1]);
      ir::Stmt compute = lower(scheduled, string("compute_") + util::join(paramSet, "_"),  false, true);
      codegen->compile(compute, isFirst);
      isFirst = false;
    }
    ofstream source_file;
    source_file.open(file_path + "mttkrp_gpu" + file_ending);
    source_file << source.str();
    source_file.close();
  }
}

TEST(generate_figures, DISABLED_cpu) {
  if (should_use_CUDA_codegen()) {
    return;
  }

  int NUM_I = 100;
  int NUM_J = 100;

  string file_ending = should_use_CUDA_codegen() ? ".cu" : ".c";
  string file_path = "figures_cpu/";
  mkdir(file_path.c_str(), 0777);

  // spmv
  {
    stringstream source;
    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
    Tensor<double> A("A", {NUM_I, NUM_J}, CSR);
    Tensor<double> x("x", {NUM_J}, Format({Dense}));
    Tensor<double> y("y", {NUM_I}, Format({Dense}));
    y(i) = A(i, j) * x(j);
    IndexStmt stmt = y.getAssignment().concretize();
    bool isFirst = true;
    string functionNames[] = {"spmv_unscheduled", "spmv_row_tiled", "spmv_pos_iteration"};
    IndexStmt  (* schedulingFunctions [])(IndexStmt, Tensor<double>) = {&exampleScheduleSPMVUntiled, &exampleScheduleSPMVCPURowTiling, &exampleScheduleSPMVPosIteration};

    int ii = 0;
    for (auto schedulingFunction : schedulingFunctions) {
      IndexStmt scheduled = schedulingFunction(stmt, A);
      ir::Stmt compute = lower(scheduled, functionNames[ii], false, true);
      codegen->compile(compute, isFirst);
      isFirst = false;
      ii++;
    }
    ofstream source_file;
    source_file.open(file_path + "fig_spmv" + file_ending);
    source_file << source.str();
    source_file.close();
  }
}
