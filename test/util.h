#ifndef __SCHEDULE_UTIL_HH__
#define __SCHEDULE_UTIL_HH__

#include <iostream>
#include <taco/index_notation/transformations.h>
#include <codegen/codegen_c.h>
#include <codegen/codegen_ispc.h>
#include <codegen/codegen_cuda.h>
#include <fstream>
#include <memory>
#include <random>
#include "taco/cuda.h"
#include "test.h"
#include "test_tensors.h"
#include "taco/tensor.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/transformations.h"
#include "codegen/codegen.h"
#include "taco/lower/lower.h"
#include "taco/util/timers.h"

using namespace taco;

#define ERROR_MARGIN (1.0e-2)

#define TOOL_BENCHMARK_TIMER(CODE,NAME,TIMER) {                  \
    if (time) {                                                  \
      taco::util::Timer timer;                                   \
      timer.start();                                             \
      CODE;                                                      \
      timer.stop();                                              \
      taco::util::TimeResults result = timer.getResult();        \
      cout << NAME << " " << result << " ms" << endl;            \
      TIMER=result;                                              \
    }                                                            \
    else {                                                       \
      CODE;                                                      \
    }                                                            \
}

#define TOOL_BENCHMARK_TIMER2(CODE,NAME,TIMER) {                  \
    if (time) {                                                  \
      taco::util::Timer timer;                                   \
      timer.start();                                             \
      CODE;                                                      \
      timer.stop();                                              \
      taco::util::TimeResults result = timer.getResult();        \
      if (statfile.is_open()) {                                  \
        statfile << NAME << " " << result << " ms" << endl;      \
      } else {                                                   \
        cout << NAME << " " << result << " ms" << endl;          \
      }                                                          \
      TIMER=result;                                              \
    }                                                            \
    else {                                                       \
      CODE;                                                      \
    }                                                            \
}

static void printToCout(IndexStmt stmt);
static void printToFile(string filename, IndexStmt stmt);
static void printToFile(string filename, string additional_filename, IndexStmt stmt);


static void printToCout(IndexStmt stmt) {
  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
  ir::Stmt compute = lower(stmt, "compute", false, true);
  codegen->compile(compute, true);
}

void printToFile(string filename, IndexStmt stmt) {
  stringstream source;

  string file_path = "eval_generated/";
  mkdir(file_path.c_str(), 0777);

  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
  ir::Stmt compute = lower(stmt, "compute",  false, true);
  codegen->compile(compute, true);

  ofstream source_file;
  string file_ending = should_use_CUDA_codegen() ? ".cu" : ".c";
  source_file.open(file_path + filename + file_ending);
  source_file << source.str();
  source_file.close();
}

void printToFile(string filename, string additional_filename, IndexStmt stmt) {
  stringstream source1;
  stringstream source2;

  string file_path = "eval_generated/";
  mkdir(file_path.c_str(), 0777);

  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source1, source2, ir::CodeGen::ImplementationGen);
  ir::Stmt compute = lower(stmt, "compute", false, true);
  codegen->compile(compute, true);

  ofstream source_file;
  string file_ending = should_use_CUDA_codegen() ? ".cu" : ".c";
  source_file.open(file_path+filename+file_ending);
  source_file << source1.str();
  source_file.close();

  ofstream additional_source_file;
  string additional_file_ending = ".ispc";
  additional_source_file.open(file_path+additional_filename+additional_file_ending);
  additional_source_file << source2.str();
  additional_source_file.close();

}

#endif // __SCHEDULE_UTIL_HH__