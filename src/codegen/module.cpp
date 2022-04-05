#include "taco/codegen/module.h"

#include <iostream>
#include <fstream>
#include <dlfcn.h>
#include <unistd.h>
#if USE_OPENMP
#include <omp.h>
#endif

#include "taco/tensor.h"
#include "taco/error.h"
#include "taco/util/strings.h"
#include "taco/util/env.h"
#include "codegen/codegen_c.h"
#include "codegen/codegen_ispc.h"
#include "codegen/codegen_cuda.h"
#include "taco/cuda.h"

using namespace std;

namespace taco {
namespace ir {

std::string Module::chars = "abcdefghijkmnpqrstuvwxyz0123456789";
std::default_random_engine Module::gen = std::default_random_engine();
std::uniform_int_distribution<int> Module::randint =
    std::uniform_int_distribution<int>(0, chars.length() - 1);

void Module::setJITTmpdir() {
  tmpdir = util::getTmpdir();
}

void Module::setJITLibname() {
  libname.resize(12);
  for (int i=0; i<12; i++)
    libname[i] = chars[randint(gen)];
}

void Module::addFunction(Stmt func) {
  funcs.push_back(func);
}

void Module::compileToSource(string path, string prefix) {
  if (!moduleFromUserSource) {
    std::cout << "module not from user source\n";
  
    // create a codegen instance and add all the funcs
    bool didGenRuntime = false;
    
    header.str("");
    header.clear();
    source.str("");
    source.clear();
    additional_source.str("");
    additional_source.clear();

    taco_tassert(target.arch == Target::C99) <<
        "Only C99 codegen supported currently";
    std::shared_ptr<CodeGen> sourcegen =
        CodeGen::init_default(source, additional_source, CodeGen::ImplementationGen);
    std::shared_ptr<CodeGen> headergen =
            CodeGen::init_default(header, CodeGen::HeaderGen);

    for (auto func: funcs) {
      sourcegen->compile(func, !didGenRuntime);
      headergen->compile(func, !didGenRuntime);
      didGenRuntime = true;
    }
  }

  ofstream source_file;
  string file_ending = should_use_CUDA_codegen() ? ".cu" : ".c";
  source_file.open(path+prefix+file_ending);
  if (should_use_ISPC_codegen()) {
    source_file << "#include \"" << path+prefix+"_ispc.h\"\n";
  }
  source_file << source.str();
  source_file.close();

  ofstream additional_source_file;
  string file_ending2 = ".ispc";
  additional_source_file.open(path+prefix+file_ending2);
  additional_source_file << additional_source.str();
  additional_source_file.close();
  
  ofstream header_file;
  header_file.open(path+prefix+".h");
  header_file << header.str();
  header_file.close();
}

void Module::compileToStaticLibrary(string path, string prefix) {
  taco_tassert(false) << "Compiling to a static library is not supported";
}
  
namespace {

void writeShims(vector<Stmt> funcs, string path, string prefix) {
  stringstream shims;
  for (auto func: funcs) {
    if (should_use_CUDA_codegen()) {
      CodeGen_CUDA::generateShim(func, shims);
    }
    // else if (should_use_ISPC_codegen()) {
    //   CodeGen_ISPC::generateShim(func, shims);
    // }
    else {
      CodeGen_C::generateShim(func, shims);
    }
  }
  
  ofstream shims_file;
  if (should_use_CUDA_codegen()) {
    shims_file.open(path+prefix+"_shims.cpp");
  }
  // else if (should_use_ISPC_codegen()) {
  //   shims_file.open(path+prefix+".c", ios::app);
  // }
  else {
    shims_file.open(path+prefix+".c", ios::app);
  }
  shims_file << "#include \"" << path << prefix << ".h\"\n";
  shims_file << shims.str();
  shims_file.close();
}

} // anonymous namespace

string Module::compile() {
  std::cout << "Module::compile\n";
  string prefix = tmpdir+libname;
  string fullpath = prefix + ".so";
  
  string cc;
  string cflags;
  string file_ending;
  string shims_file;
  if (should_use_CUDA_codegen()) {
    cc = util::getFromEnv("TACO_NVCC", "nvcc");
    cflags = util::getFromEnv("TACO_NVCCFLAGS",
    get_default_CUDA_compiler_flags());
    file_ending = ".cu";
    shims_file = prefix + "_shims.cpp";
  }
  // else if (should_use_ISPC_codegen()) {
  //   cc = util::getFromEnv("TACO_ISPC", "ispc");
  //   cflags = util::getFromEnv("TACO_ISPC_FLAGS",
  //   " --target=sse2-i32x4,sse4-i32x8,avx1-i32x8,avx2-i32x8,avx512knl-i32x16,avx512skx-i32x16 --pic -O3 --addressing=64 --arch=x86-64"
  //   ) + " ";

  // }
  else {
    cc = util::getFromEnv(target.compiler_env, target.compiler);
    cflags = util::getFromEnv("TACO_CFLAGS",
    "-O3 -ffast-math -std=c99") + " -shared -fPIC";
#if USE_OPENMP
    cflags += " -fopenmp";
#endif
    file_ending = ".c";
    shims_file = "";
  }
  
  string cmd = cc + " " + cflags + " " +
    prefix + file_ending + " " + shims_file + " " + 
    "-o " + fullpath + " -lm";
  std::cout << "--------------------------------------------------------------------------------tmpdir: " << tmpdir << std::endl;
  std::cout << "--------------------------------------------------------------------------------libname: " << libname << std::endl;
  std::cout << "--------------------------------------------------------------------------------prefix: " << prefix << std::endl;
  std::cout << "--------------------------------------------------------------------------------fullpath: " << fullpath << std::endl;
  std::cout << "--------------------------------------------------------------------------------cmd: " << cmd << std::endl;

  // open the output file & write out the source
  compileToSource(tmpdir, libname);

  
  // write out the shims
  writeShims(funcs, tmpdir, libname);
  for (auto &statement : funcs) {
    std::cout << "----- statement --------" << std::endl;
    // std::cout << statement;
    std::cout << std::endl;
  }
  std::cout << tmpdir << std::endl << libname << std::endl;
  
  if (should_use_ISPC_codegen()) {
    string ispc = util::getFromEnv("TACO_ISPC", "ispc");
    string ispcflags = util::getFromEnv("TACO_ISPC_FLAGS",
    " --target=sse2-i32x4,sse4-i32x8,avx1-i32x8,avx2-i32x8,avx512knl-i32x16,avx512skx-i32x16 --pic -O3 --addressing=64 --arch=x86-64"
    ) + " ";
    string cmd = ispc + " " + ispcflags + " -o " + prefix + ".ispc.o " + " --emit-obj " + prefix + ".ispc " + "-h " + prefix + "_ispc.h";

    // now compile the ispc file to generate the object file and the ispc header file
    std::cout << "--------------------------------------------------------------------------------cmd: " << cmd << std::endl;
    int err = system(cmd.data());
    taco_uassert(err == 0) << "Compilation command failed:\n" << cmd
      << "\nreturned " << err;

    string ispc_object_file = " " + prefix + ".ispc.o ";
    string ispc_object_files_for_diff_targets = " " + prefix + ".ispc_* ";
    cmd = cc + " " + cflags + " " +
      prefix + file_ending + " " + ispc_object_file + ispc_object_files_for_diff_targets + shims_file + " " + 
      "-o " + fullpath + " -lm -lrt ";

    // now compile the c file linking the ispc object file. ispc header is added to the top of the c file
    std::cout << "--------------------------------------------------------------------------------cmd: " << cmd << std::endl;
    err = system(cmd.data());
    taco_uassert(err == 0) << "Compilation command failed:\n" << cmd
      << "\nreturned " << err;
  } else {
    // now compile it
    int err = system(cmd.data());
    taco_uassert(err == 0) << "Compilation command failed:\n" << cmd
      << "\nreturned " << err;
  }

  // use dlsym() to open the compiled library
  if (lib_handle) {
    dlclose(lib_handle);
  }
  lib_handle = dlopen(fullpath.data(), RTLD_NOW | RTLD_LOCAL);
  taco_uassert(lib_handle) << "Failed to load generated code, error is: " << dlerror();

  return fullpath;
}

void Module::setSource(string source) {
  this->source << source;
  moduleFromUserSource = true;
}

string Module::getSource() {
  return source.str();
}

void* Module::getFuncPtr(std::string name) {
  return dlsym(lib_handle, name.data());
}

int Module::callFuncPackedRaw(std::string name, void** args) {
  typedef int (*fnptr_t)(void**);
  static_assert(sizeof(void*) == sizeof(fnptr_t),
    "Unable to cast dlsym() returned void pointer to function pointer");
  void* v_func_ptr = getFuncPtr(name);
  fnptr_t func_ptr;
  *reinterpret_cast<void**>(&func_ptr) = v_func_ptr;

#if USE_OPENMP
  omp_sched_t existingSched;
  ParallelSchedule tacoSched;
  int existingChunkSize, tacoChunkSize;
  int existingNumThreads = omp_get_max_threads();
  omp_get_schedule(&existingSched, &existingChunkSize);
  taco_get_parallel_schedule(&tacoSched, &tacoChunkSize);
  switch (tacoSched) {
    case ParallelSchedule::Static:
      omp_set_schedule(omp_sched_static, tacoChunkSize);
      break;
    case ParallelSchedule::Dynamic:
      omp_set_schedule(omp_sched_dynamic, tacoChunkSize);
      break;
    default:
      break;
  }
  omp_set_num_threads(taco_get_num_threads());
#endif

  int ret = func_ptr(args);

#if USE_OPENMP
  omp_set_schedule(existingSched, existingChunkSize);
  omp_set_num_threads(existingNumThreads);
#endif

  return ret;
}

} // namespace ir
} // namespace taco
