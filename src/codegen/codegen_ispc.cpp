#include <iostream>
#include <fstream>
#include <dlfcn.h>
#include <algorithm>
#include <unordered_set>
#include <taco.h>

#include "taco/cuda.h"
#include "taco/ir/ir_printer.h"
#include "taco/ir/ir_visitor.h"
#include "taco/ir/ir_rewriter.h"
#include "taco/ir/simplify.h"

#include "codegen_c.h"
#include "codegen_ispc.h"
#include "taco/error.h"
#include "taco/util/strings.h"
#include "taco/util/collections.h"

using namespace std;

namespace taco {
namespace ir {

// Some helper functions
namespace {

// Include stdio.h for printf
// stdlib.h for malloc/realloc
// math.h for sqrt
// MIN preprocessor macro
// This *must* be kept in sync with taco_tensor_t.h
const string cHeaders =
  "#ifndef TACO_C_HEADERS\n"
  "#define TACO_C_HEADERS\n"
  "#include <stdio.h>\n"
  "#include <stdlib.h>\n"
  "#include <stdint.h>\n"
  "#include <stdbool.h>\n"
  "#include <math.h>\n"
  "#include <complex.h>\n"
  "#include <string.h>\n"
  "#if _OPENMP\n"
  "#include <omp.h>\n"
  "#endif\n"
  "#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))\n"
  "#define TACO_MAX(_a,_b) ((_a) > (_b) ? (_a) : (_b))\n"
  "#define TACO_DEREF(_a) (((___context___*)(*__ctx__))->_a)\n"
  "#ifndef TACO_TENSOR_T_DEFINED\n"
  "#define TACO_TENSOR_T_DEFINED\n"
  "typedef enum { taco_mode_dense, taco_mode_sparse } taco_mode_t;\n"
  "typedef struct {\n"
  "  int32_t      order;         // tensor order (number of modes)\n"
  "  int32_t*     dimensions;    // tensor dimensions\n"
  "  int32_t      csize;         // component size\n"
  "  int32_t*     mode_ordering; // mode storage ordering\n"
  "  taco_mode_t* mode_types;    // mode storage types\n"
  "  uint8_t***   indices;       // tensor index data (per mode)\n"
  "  uint8_t*     vals;          // tensor values\n"
  "  int32_t      vals_size;     // values array size\n"
  "} taco_tensor_t;\n"
  "#endif\n"
  "#if !_OPENMP\n"
  "int omp_get_thread_num() { return 0; }\n"
  "int omp_get_max_threads() { return 1; }\n"
  "#endif\n"
  "int cmp(const void *a, const void *b) {\n"
  "  return *((const int*)a) - *((const int*)b);\n"
  "}\n"
  "int taco_binarySearchAfter(int *array, int arrayStart, int arrayEnd, int target) {\n"
  "  if (array[arrayStart] >= target) {\n"
  "    return arrayStart;\n"
  "  }\n"
  "  int lowerBound = arrayStart; // always < target\n"
  "  int upperBound = arrayEnd; // always >= target\n"
  "  while (upperBound - lowerBound > 1) {\n"
  "    int mid = (upperBound + lowerBound) / 2;\n"
  "    int midValue = array[mid];\n"
  "    if (midValue < target) {\n"
  "      lowerBound = mid;\n"
  "    }\n"
  "    else if (midValue > target) {\n"
  "      upperBound = mid;\n"
  "    }\n"
  "    else {\n"
  "      return mid;\n"
  "    }\n"
  "  }\n"
  "  return upperBound;\n"
  "}\n"
  "int taco_binarySearchBefore(int *array, int arrayStart, int arrayEnd, int target) {\n"
  "  if (array[arrayEnd] <= target) {\n"
  "    return arrayEnd;\n"
  "  }\n"
  "  int lowerBound = arrayStart; // always <= target\n"
  "  int upperBound = arrayEnd; // always > target\n"
  "  while (upperBound - lowerBound > 1) {\n"
  "    int mid = (upperBound + lowerBound) / 2;\n"
  "    int midValue = array[mid];\n"
  "    if (midValue < target) {\n"
  "      lowerBound = mid;\n"
  "    }\n"
  "    else if (midValue > target) {\n"
  "      upperBound = mid;\n"
  "    }\n"
  "    else {\n"
  "      return mid;\n"
  "    }\n"
  "  }\n"
  "  return lowerBound;\n"
  "}\n"
  "taco_tensor_t* init_taco_tensor_t(int32_t order, int32_t csize,\n"
  "                                  int32_t* dimensions, int32_t* mode_ordering,\n"
  "                                  taco_mode_t* mode_types) {\n"
  "  taco_tensor_t* t = (taco_tensor_t *) malloc(sizeof(taco_tensor_t));\n"
  "  t->order         = order;\n"
  "  t->dimensions    = (int32_t *) malloc(order * sizeof(int32_t));\n"
  "  t->mode_ordering = (int32_t *) malloc(order * sizeof(int32_t));\n"
  "  t->mode_types    = (taco_mode_t *) malloc(order * sizeof(taco_mode_t));\n"
  "  t->indices       = (uint8_t ***) malloc(order * sizeof(uint8_t***));\n"
  "  t->csize         = csize;\n"
  "  for (int32_t i = 0; i < order; i++) {\n"
  "    t->dimensions[i]    = dimensions[i];\n"
  "    t->mode_ordering[i] = mode_ordering[i];\n"
  "    t->mode_types[i]    = mode_types[i];\n"
  "    switch (t->mode_types[i]) {\n"
  "      case taco_mode_dense:\n"
  "        t->indices[i] = (uint8_t **) malloc(1 * sizeof(uint8_t **));\n"
  "        break;\n"
  "      case taco_mode_sparse:\n"
  "        t->indices[i] = (uint8_t **) malloc(2 * sizeof(uint8_t **));\n"
  "        break;\n"
  "    }\n"
  "  }\n"
  "  return t;\n"
  "}\n"
  "void deinit_taco_tensor_t(taco_tensor_t* t) {\n"
  "  for (int i = 0; i < t->order; i++) {\n"
  "    free(t->indices[i]);\n"
  "  }\n"
  "  free(t->indices);\n"
  "  free(t->dimensions);\n"
  "  free(t->mode_ordering);\n"
  "  free(t->mode_types);\n"
  "  free(t);\n"
  "}\n"
  "#endif\n";

const string ispcHeaders = 
  "#define __TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))\n"
  "#define __TACO_MAX(_a,_b) ((_a) > (_b) ? (_a) : (_b))\n"
  "#define __TACO_DEREF(_a) (((___context___*)(*__ctx__))->_a)\n"
  "int __cmp(const void *a, const void *b) {\n"
  "  return *((const int*)a) - *((const int*)b);\n"
  "}\n"
  "int __taco_binarySearchAfter(int *array, int arrayStart, int arrayEnd, int target) {\n"
  "  if (array[arrayStart] >= target) {\n"
  "    return arrayStart;\n"
  "  }\n"
  "  int lowerBound = arrayStart; // always < target\n"
  "  int upperBound = arrayEnd; // always >= target\n"
  "  while (upperBound - lowerBound > 1) {\n"
  "    int mid = (upperBound + lowerBound) / 2;\n"
  "    int midValue = array[mid];\n"
  "    if (midValue < target) {\n"
  "      lowerBound = mid;\n"
  "    }\n"
  "    else if (midValue > target) {\n"
  "      upperBound = mid;\n"
  "    }\n"
  "    else {\n"
  "      return mid;\n"
  "    }\n"
  "  }\n"
  "  return upperBound;\n"
  "}\n"
  "int __taco_binarySearchBefore(int *array, int arrayStart, int arrayEnd, int target) {\n"
  "  if (array[arrayEnd] <= target) {\n"
  "    return arrayEnd;\n"
  "  }\n"
  "  int lowerBound = arrayStart; // always <= target\n"
  "  int upperBound = arrayEnd; // always > target\n"
  "  while (upperBound - lowerBound > 1) {\n"
  "    int mid = (upperBound + lowerBound) / 2;\n"
  "    int midValue = array[mid];\n"
  "    if (midValue < target) {\n"
  "      lowerBound = mid;\n"
  "    }\n"
  "    else if (midValue > target) {\n"
  "      upperBound = mid;\n"
  "    }\n"
  "    else {\n"
  "      return mid;\n"
  "    }\n"
  "  }\n"
  "  return lowerBound;\n"
  "}\n\n\n";

} // anonymous namespace



// find variables for generating declarations
// generates a single var for each GetProperty
class CodeGen_ISPC::FindVars : public IRVisitor {
public:
  map<Expr, string, ExprCompare> varMap;

  // the variables for which we need to add declarations
  map<Expr, string, ExprCompare> varDecls;

  vector<Expr> localVars;

  // this maps from tensor, property, mode, index to the unique var
  map<tuple<Expr, TensorProperty, int, int>, string> canonicalPropertyVar;

  // this is for convenience, recording just the properties unpacked
  // from the output tensor so we can re-save them at the end
  map<tuple<Expr, TensorProperty, int, int>, string> outputProperties;

  // TODO: should replace this with an unordered set
  vector<Expr> outputTensors;
  vector<Expr> inputTensors;

  CodeGen_ISPC *codeGen;

  // copy inputs and outputs into the map
  FindVars(vector<Expr> inputs, vector<Expr> outputs, CodeGen_ISPC *codeGen)
  : codeGen(codeGen) {
    for (auto v: inputs) {
      auto var = v.as<Var>();
      taco_iassert(var) << "Inputs must be vars in codegen";
      taco_iassert(varMap.count(var)==0) << "Duplicate input found in codegen";
      inputTensors.push_back(v);
      varMap[var] = var->name;
    }
    for (auto v: outputs) {
      auto var = v.as<Var>();
      taco_iassert(var) << "Outputs must be vars in codegen";
      taco_iassert(varMap.count(var)==0) << "Duplicate output found in codegen";
      outputTensors.push_back(v);
      varMap[var] = var->name;
    }
  }

protected:
  using IRVisitor::visit;

  virtual void visit(const Var *op) {
    if (varMap.count(op) == 0) {
      varMap[op] = op->is_ptr? op->name : codeGen->genUniqueName(op->name);
    }
  }

  virtual void visit(const VarDecl *op) {
    if (!util::contains(localVars, op->var)) {
      localVars.push_back(op->var);
    }
    op->var.accept(this);
    op->rhs.accept(this);
  }

  virtual void visit(const For *op) {
    if (!util::contains(localVars, op->var)) {
      localVars.push_back(op->var);
    }
    op->var.accept(this);
    op->start.accept(this);
    op->end.accept(this);
    op->increment.accept(this);
    op->contents.accept(this);
  }

  virtual void visit(const GetProperty *op) {
    if (!util::contains(inputTensors, op->tensor) &&
        !util::contains(outputTensors, op->tensor)) {
      // Don't create header unpacking code for temporaries
      return;
    }

    if (varMap.count(op) == 0) {
      auto key =
              tuple<Expr,TensorProperty,int,int>(op->tensor,op->property,
                                                 (size_t)op->mode,
                                                 (size_t)op->index);
      if (canonicalPropertyVar.count(key) > 0) {
        varMap[op] = canonicalPropertyVar[key];
      } else {
        auto unique_name = codeGen->genUniqueName(op->name);
        canonicalPropertyVar[key] = unique_name;
        varMap[op] = unique_name;
        varDecls[op] = unique_name;
        if (util::contains(outputTensors, op->tensor)) {
          outputProperties[key] = unique_name;
        }
      }
    }
  }
};


// Finds all for loops tagged with accelerator and adds statements to deviceFunctions
// Also tracks scope of when device function is called and
// tracks which variables must be passed to function.
class CodeGen_ISPC::FunctionCollector : public IRVisitor {
public:
  vector<Stmt> threadFors; // contents is device function
  vector<Stmt> initFors;  // for loops to initialize statements
  map<Expr, string, ExprCompare> scopeMap;

  // the variables to pass to each device function
  vector<vector<pair<string, Expr>>> functionParameters;
  vector<pair<string, Expr>> currentParameters; // keep as vector so code generation is deterministic
  set<Expr> currentParameterSet;

  set<Expr> variablesDeclaredInKernel;

  vector<pair<string, Expr>> threadIDVars;
  vector<pair<string, Expr>> blockIDVars;
  vector<pair<string, Expr>> warpIDVars;
  vector<Expr> numThreads;
  vector<Expr> numWarps;

  CodeGen_ISPC *codeGen;
  // copy inputs and outputs into the map
  FunctionCollector(vector<Expr> inputs, vector<Expr> outputs, CodeGen_ISPC *codeGen) : codeGen(codeGen)  {
    inDeviceFunction = false;
    for (auto v: inputs) {
      auto var = v.as<Var>();
      taco_iassert(var) << "Inputs must be vars in codegen";
      taco_iassert(scopeMap.count(var) == 0) <<
                                             "Duplicate input found in codegen";
      scopeMap[var] = var->name;
    }
    for (auto v: outputs) {
      auto var = v.as<Var>();
      taco_iassert(var) << "Outputs must be vars in codegen";
      taco_iassert(scopeMap.count(var) == 0) <<
                                             "Duplicate output found in codegen";

      scopeMap[var] = var->name;
    }
  }

protected:
  bool inDeviceFunction;
  using IRVisitor::visit;

  virtual void visit(const For *op) {
    if (op->parallel_unit == ParallelUnit::CPUSpmd) {
      std::cout << "ParallelUnit::CPUSpmd directive found\n";

      inDeviceFunction = false;
      op->var.accept(this);
      inDeviceFunction = true;

      threadFors.push_back(op);
      std::cout << "scopeMap: [" << scopeMap[op->var] << "], varExpr: [" << op->var << "]\n";
      threadIDVars.push_back(pair<string, Expr>(scopeMap[op->var], op->var));
      Expr blockSize = ir::simplify(ir::Div::make(ir::Sub::make(op->end, op->start), op->increment));
      numThreads.push_back(blockSize);

    }
    else if (op->parallel_unit == ParallelUnit::CPUSimd) {
      std::cout << "************************************************************************** CPUSimd For node\n";
    }
    else if (op->kind == LoopKind::Init) {
      std::cout << "************************************************************************* Init loop kind found\n";
      initFors.push_back(op);
    }
    else{
      op->var.accept(this);
    }
    op->start.accept(this);
    op->end.accept(this);
    op->increment.accept(this);
    op->contents.accept(this);
  }

  virtual void visit(const Var *op) {
    if (scopeMap.count(op) == 0) {
      string name = codeGen->genUniqueName(op->name);
      if (!inDeviceFunction) {
        scopeMap[op] = name;
      }
    }
    else if (scopeMap.count(op) == 1 && inDeviceFunction && currentParameterSet.count(op) == 0
            && (threadIDVars.empty() || op != threadIDVars.back().second)
            && !variablesDeclaredInKernel.count(op)) {
      currentParameters.push_back(pair<string, Expr>(scopeMap[op], op));
      currentParameterSet.insert(op);
    }
  }

  virtual void visit(const VarDecl *op) {
    if (inDeviceFunction) {
      variablesDeclaredInKernel.insert(op->var);
    }
    op->var.accept(this);
    op->rhs.accept(this);
  }

  virtual void visit(const GetProperty *op) {
    if (scopeMap.count(op->tensor) == 0 && !inDeviceFunction) {
      auto key =
              tuple<Expr,TensorProperty,int,int>(op->tensor,op->property,
                                                 (size_t)op->mode,
                                                 (size_t)op->index);
      auto unique_name = codeGen->genUniqueName(op->name);
      scopeMap[op->tensor] = unique_name;
    }
    else if (scopeMap.count(op->tensor) == 1 && inDeviceFunction && currentParameterSet.count(op->tensor) == 0) {
      currentParameters.push_back(pair<string, Expr>(op->tensor.as<Var>()->name, op->tensor));
      currentParameterSet.insert(op->tensor);
    }
  }
};


CodeGen_ISPC::CodeGen_ISPC(std::ostream &dest, OutputKind outputKind, bool simplify)
    : CodeGen_C(dest, dest, outputKind, simplify) {}

CodeGen_ISPC::CodeGen_ISPC(std::ostream &dest, std::ostream &dest2, OutputKind outputKind, bool simplify)
    : CodeGen_C(dest, dest2, outputKind, simplify) {}

CodeGen_ISPC::~CodeGen_ISPC() {}

void CodeGen_ISPC::compile(Stmt stmt, bool isFirst) {
  varMap = {};
  localVars = {};

  if (isFirst) {
    // output the headers
    out << cHeaders;

    if (&out != &out2) {
      out2 << ispcHeaders;
    }
  }
  out << endl;
  // generate code for the Stmt
  std::cout << "Compiling the code\n";
  stmt.accept(this);
}



string CodeGen_ISPC::printCallISPCFunc(const std::string& funcName, map<Expr, string, ExprCompare> varMap,
                                  vector<const GetProperty*> &sortedProps) {
  std::stringstream ret;
  ret << "  ";
  unordered_set<string> propsAlreadyGenerated;

  ret << "__" << funcName << "(";


  for (unsigned long i=0; i < sortedProps.size(); i++) {
    ret << varMap[sortedProps[i]];
    if (i != sortedProps.size()-1) {
      ret << ", ";
    }
    propsAlreadyGenerated.insert(varMap[sortedProps[i]]);
  }

  ret << ");\n";
  return ret.str();
}

// varMap is already sorted <- make sure to pass the sorted varMap
void CodeGen_ISPC::printISPCFunc(const Function *func, map<Expr, string, ExprCompare> varMap,
                                  vector<const GetProperty*> &sortedProps) {

  FunctionCollector functionCollector(func->inputs, func->outputs, this);
  func->body.accept(&functionCollector);

  vector<Expr> inputs = func->inputs;
  vector<Expr> outputs = func->outputs;
  unordered_set<string> propsAlreadyGenerated;

  for (unsigned long i=0; i < sortedProps.size(); i++) {
    auto prop = sortedProps[i];
    bool isOutputProp = (find(outputs.begin(), outputs.end(),
                              prop->tensor) != outputs.end());
    
    auto var = prop->tensor.as<Var>();
    if (var->is_parameter) {
      if (isOutputProp) {
        funcVariables << "  " << printTensorProperty(varMap[prop], prop, false) << ";" << endl;
      } else {
        break; 
      }
    } else {
      funcVariables << getUnpackedTensorArgument(varMap[prop], prop, isOutputProp);
    }
    propsAlreadyGenerated.insert(varMap[prop]);

    if (i!=sortedProps.size()-1) {
      funcVariables << ", ";
    }
    if (i%2==0) {
      funcVariables << "\n\t";
    }
  }

  resetUniqueNameCounters();

  // threadFors code generation
  for (size_t i = 0; i < functionCollector.threadFors.size(); i++) {

    const For *threadloop = to<For>(functionCollector.threadFors[i]);
    taco_iassert(threadloop->parallel_unit == ParallelUnit::CPUSpmd);
    Stmt function = threadloop->contents;
    std::cout << "threadloop function: " << function << std::endl;

    out2 << "\nstatic task void __" << func->name << "__ (";
    out2 << funcVariables.str();
    out2 << "\n) {\n\n";

    indent++;
    // output body of the threadloop
    taskCode = true;
    print(threadloop);
    indent--;
    out2 << "}\n\n";  

  }

  taskCode = false;
  out2 << "export void __" << func->name << " (";
  out2 << funcVariables.str();
  out2 << "\n) {\n\n";

  indent++;
  // output body
  print(func->body);
  indent--;
  out2 << "}\n";
  
}

void CodeGen_ISPC::sendToStream(std::stringstream &stream) {
  if (is_ISPC_code_stream_enabled()) {
    this->out2 << stream.str();
  }
  else {
    CodeGen_C::sendToStream(stream);
  }
}

void CodeGen_ISPC::visit(const Function* func) {
  set_ISPC_code_stream_enabled(false);

  // if generating a header, protect the function declaration with a guard
  if (func->name == "assemble") {
    if (outputKind == HeaderGen) {
      out << "#ifndef TACO_GENERATED_" << func->name << "\n";
      out << "#define TACO_GENERATED_" << func->name << "\n";
    }

    int numYields = countYields(func);
    emittingCoroutine = (numYields > 0);
    funcName = func->name;
    labelCount = 0;

    resetUniqueNameCounters();
    FindVars inputVarFinder(func->inputs, {}, this);
    func->body.accept(&inputVarFinder);
    FindVars outputVarFinder({}, func->outputs, this);
    func->body.accept(&outputVarFinder);

    // output function declaration
    doIndent();
    out << printFuncName(func, inputVarFinder.varDecls, outputVarFinder.varDecls);

    // if we're just generating a header, this is all we need to do
    if (outputKind == HeaderGen) {
      out << ";\n";
      out << "#endif\n";
      return;
    }

    out << " {\n";

    indent++;

    // find all the vars that are not inputs or outputs and declare them
    resetUniqueNameCounters();
    FindVars varFinder(func->inputs, func->outputs, this);
    func->body.accept(&varFinder);
    varMap = varFinder.varMap;
    localVars = varFinder.localVars;

    // Print variable declarations
    out << printDecls(varFinder.varDecls, func->inputs, func->outputs) << endl;

    if (emittingCoroutine) {
      out << printContextDeclAndInit(varMap, localVars, numYields, func->name)
          << endl;
    }

    // output body
    print(func->body);

    // output repack only if we allocated memory
    if (checkForAlloc(func))
      out << endl << printPack(varFinder.outputProperties, func->outputs);

    if (emittingCoroutine) {
      out << printCoroutineFinish(numYields, funcName);
    }

    doIndent();
    out << "return 0;\n";
    indent--;

    doIndent();
    out << "}\n";
    return;

  }


  if (outputKind == HeaderGen) {
    out << "#ifndef TACO_GENERATED_" << func->name << "\n";
    out << "#define TACO_GENERATED_" << func->name << "\n";
  }

  int numYields = countYields(func);
  emittingCoroutine = (numYields > 0);
  funcName = func->name;
  labelCount = 0;

  resetUniqueNameCounters();
  FindVars inputVarFinder(func->inputs, {}, this);
  func->body.accept(&inputVarFinder);
  FindVars outputVarFinder({}, func->outputs, this);
  func->body.accept(&outputVarFinder);

  // output function declaration
  doIndent();
  out << printFuncName(func, inputVarFinder.varDecls, outputVarFinder.varDecls);

  // if we're just generating a header, this is all we need to do
  if (outputKind == HeaderGen) {
    out << ";\n";
    out << "#endif\n";
    return;
  }

  out << " {\n";

  indent++;

  // find all the vars that are not inputs or outputs and declare them
  resetUniqueNameCounters();
  FindVars varFinder(func->inputs, func->outputs, this);
  func->body.accept(&varFinder);
  varMap = varFinder.varMap;
  localVars = varFinder.localVars;

  // Print variable declarations
  out << printDecls(varFinder.varDecls, func->inputs, func->outputs) << endl;

  sortedProps = {};
  vector<Expr> inputs = func->inputs;
  vector<Expr> outputs = func->outputs;
  getSortedProps(varFinder.varDecls, sortedProps, inputs, outputs);
  out << printCallISPCFunc(func->name, varFinder.varDecls, sortedProps);

  if (emittingCoroutine) {
    out << printContextDeclAndInit(varMap, localVars, numYields, func->name)
        << endl;
  }

  // output repack only if we allocated memory
  if (checkForAlloc(func))
    out << endl << printPack(varFinder.outputProperties, func->outputs);

  if (emittingCoroutine) {
    out << printCoroutineFinish(numYields, funcName);
  }

  doIndent();
  out << "return 0;\n";
  indent--;

  doIndent();
  out << "}\n\n";

  set_ISPC_code_stream_enabled(true);
  printISPCFunc(func, varFinder.varDecls, sortedProps);
  set_ISPC_code_stream_enabled(false);

}

void CodeGen_ISPC::visit(const VarDecl* op) {
  // std::stringstream stream;
  if (is_ISPC_code_stream_enabled()) {
    if (emittingCoroutine) {
      doIndent();
      op->var.accept(this);
      parentPrecedence = Precedence::TOP;
      stream2 << " = ";
      op->rhs.accept(this);
      stream2 << ";";
      stream2 << endl;
    } else {
      IRPrinter::visit(op);
    }
  }
  else {
    CodeGen_C::visit(op);   
  }

  // sendToStream(stream);
}

void CodeGen_ISPC::visit(const Yield* op) {
  printYield(op, localVars, varMap, labelCount, funcName);
}

// For Vars, we replace their names with the generated name,
// since we match by reference (not name)
void CodeGen_ISPC::visit(const Var* op) {
  if (is_ISPC_code_stream_enabled()) {
    taco_iassert(varMap.count(op) > 0) <<
        "Var " << op->name << " not found in varMap";
    if (emittingCoroutine) {
  //    out << "TACO_DEREF(";
    }
    out2 << varMap[op];
    if (emittingCoroutine) {
  //    out << ")";
    }
  }
  else {
    CodeGen_C::visit(op);
  }
}

static string genVectorizePragma(int width) {
  stringstream ret;
  ret << "#pragma clang loop interleave(enable) ";
  if (!width)
    ret << "vectorize(enable)";
  else
    ret << "vectorize_width(" << width << ")";

  return ret.str();
}

// static string getParallelizePragma(LoopKind kind) {
//   stringstream ret;
//   ret << "#pragma omp parallel for schedule";
//   switch (kind) {
//     case LoopKind::Static:
//       ret << "(static, 1)";
//       break;
//     case LoopKind::Dynamic:
//       ret << "(dynamic, 1)";
//       break;
//     case LoopKind::Runtime:
//       ret << "(runtime)";
//       break;
//     case LoopKind::Static_Chunked:
//       ret << "(static)";
//       break;
//     default:
//       break;
//   }
//   return ret.str();
// }

// static string getUnrollPragma(size_t unrollFactor) {
//   return "#pragma unroll " + std::to_string(unrollFactor);
// }

static string getAtomicPragma() {
  return "#pragma omp atomic";
}

// The next two need to output the correct pragmas depending
// on the loop kind (Serial, Static, Dynamic, Vectorized)
//
// Docs for vectorization pragmas:
// http://clang.llvm.org/docs/LanguageExtensions.html#extensions-for-loop-hint-optimizations
void CodeGen_ISPC::visit(const For* op) {
  if (!is_ISPC_code_stream_enabled()) {
    CodeGen_C::visit(op);
    return;
  }
  doIndent();

  if (op->kind == LoopKind::Mul_Thread) {
    if (!taskCode) {
      out2 << "launch[4] " << printCallISPCFunc(funcName+"__", varMap, sortedProps) << "\n";
      return;
    }
    stream2 << "uniform unsigned int chunk_size = (";
    op->end.accept(this);
    stream2 << " - ";
    op->start.accept(this);
    stream2 << ") / taskCount;\n";
    stream2 << "  uniform unsigned int modulo = (";
    op->end.accept(this);
    stream2 << " - ";
    op->start.accept(this);
    stream2 << ") % taskCount;\n";

    stream2 << "  uniform unsigned int start = ";
    op->start.accept(this);
    stream2 << " + chunk_size * taskIndex;\n";

    stream2 << "  if (taskIndex != 0) {\n";
    stream2 << "    start += modulo;\n";
    stream2 << "  }\n";
    
    stream2 << "  uniform unsigned int end = start + chunk_size;\n";
    stream2 << "  if (taskIndex == 0) {\n";
    stream2 << "    end += modulo;\n";
    stream2 << "  }\n\n";
        
    stream2 << keywordString("  for") << " (";
    if (!emittingCoroutine) {
      if (op->var.type() == Int32) {
          stream2 << "int32 ";
      }
      else if (op->var.type() == Int64) {
          stream2 << "int64 ";
      }
      
    }
    op->var.accept(this);
    stream2 << " = ";
    stream2 << "start";
    // op->start.accept(this);
    stream2 << keywordString("; ");
    op->var.accept(this);
    stream2 << " < ";
    parentPrecedence = BOTTOM;
    stream2 << "end";
    // op->end.accept(this);
    stream2 << keywordString("; ");
    op->var.accept(this);

    auto lit = op->increment.as<Literal>();
    if (lit != nullptr && ((lit->type.isInt()  && lit->equalsScalar(1)) ||
                          (lit->type.isUInt() && lit->equalsScalar(1)))) {
      stream2 << "++";
    }
    else {
      stream2 << " += ";
      op->increment.accept(this);
    }

  }

  else if (op->kind == LoopKind::Foreach) {
    stream2 << keywordString("foreach") << " (";

    op->var.accept(this);
    stream2 << " = ";
    op->start.accept(this);
    stream2 << keywordString(" ... ");
    op->end.accept(this);

  } else {
    stream2 << keywordString("for") << " (";
    if (!emittingCoroutine) {
      if (op->var.type() == Int32) {
          stream2 << "int32 ";
      }
      else if (op->var.type() == Int64) {
          stream2 << "int64 ";
      }
      
    }
    op->var.accept(this);
    stream2 << " = ";
    op->start.accept(this);
    stream2 << keywordString("; ");
    op->var.accept(this);
    stream2 << " < ";
    parentPrecedence = BOTTOM;
    op->end.accept(this);
    stream2 << keywordString("; ");
    op->var.accept(this);

    auto lit = op->increment.as<Literal>();
    if (lit != nullptr && ((lit->type.isInt()  && lit->equalsScalar(1)) ||
                          (lit->type.isUInt() && lit->equalsScalar(1)))) {
      stream2 << "++";
    }
    else {
      stream2 << " += ";
      op->increment.accept(this);
    }
    
  }

  stream2 << ") {\n";
  op->contents.accept(this);
  doIndent();
  stream2 << "}";
  stream2 << endl;

}

void CodeGen_ISPC::visit(const While* op) {
  // it's not clear from documentation that clang will vectorize
  // while loops
  // however, we'll output the pragmas anyway
  if (op->kind == LoopKind::Vectorized) {
    doIndent();
    out << genVectorizePragma(op->vec_width);
    out << "\n";
  }

  CodeGen_C::visit(op);
}

void CodeGen_ISPC::visit(const GetProperty* op) {
  taco_iassert(varMap.count(op) > 0) <<
      "Property " << Expr(op) << " of " << op->tensor << " not found in varMap";
  if (is_ISPC_code_stream_enabled()) {
    out2 << varMap[op];
  }
  else {
    out << varMap[op];
  }

}

void CodeGen_ISPC::visit(const Min* op) {
  if (op->operands.size() == 1) {
    op->operands[0].accept(this);
    return;
  }
  for (size_t i=0; i<op->operands.size()-1; i++) {
    stream << "TACO_MIN(";
    op->operands[i].accept(this);
    stream << ",";
  }
  op->operands.back().accept(this);
  for (size_t i=0; i<op->operands.size()-1; i++) {
    stream << ")";
  }
}

void CodeGen_ISPC::visit(const Max* op) {
  if (op->operands.size() == 1) {
    op->operands[0].accept(this);
    return;
  }
  for (size_t i=0; i<op->operands.size()-1; i++) {
    stream << "TACO_MAX(";
    op->operands[i].accept(this);
    stream << ",";
  }
  op->operands.back().accept(this);
  for (size_t i=0; i<op->operands.size()-1; i++) {
    stream << ")";
  }
}

void CodeGen_ISPC::visit(const Allocate* op) {


  if (is_ISPC_code_stream_enabled()) {
    string elementType = printCType(op->var.type(), false);
    doIndent();

    op->var.accept(this);
    stream2 << " = ";
    // stream2 << " = (";
    // stream2 << elementType << "*";
    // stream2 << ")";
    if (op->is_realloc) {
      stream2 << "realloc(";
      op->var.accept(this);
      stream2 << ", ";
    }
    else {
      // If the allocation was requested to clear the allocated memory,
      // use calloc instead of malloc.
      if (op->clear) {
        stream2 << "calloc(1, ";
      } else {
        stream2 << "new ";
      }
    }
    stream2 << elementType << "[";
    parentPrecedence = MUL;
    op->num_elements.accept(this);
    parentPrecedence = TOP;
    stream2 << "];";
    stream2 << endl;


  } else {
    CodeGen_C::visit(op);

  }


}

void CodeGen_ISPC::visit(const Sqrt* op) {
  taco_tassert(op->type.isFloat() && op->type.getNumBits() == 64) <<
      "Codegen doesn't currently support non-double sqrt";
  stream << "sqrt(";
  op->a.accept(this);
  stream << ")";
}

void CodeGen_ISPC::visit(const Assign* op) {
  if (is_ISPC_code_stream_enabled()) {
    doIndent();
    op->lhs.accept(this);
    parentPrecedence = Precedence::TOP;
    bool printed = false;
    if (simplify) {
      if (isa<ir::Add>(op->rhs)) {
        auto add = to<Add>(op->rhs);
        if (add->a == op->lhs) {
          const Literal* lit = add->b.as<Literal>();
          if (lit != nullptr && ((lit->type.isInt()  && lit->equalsScalar(1)) ||
                                (lit->type.isUInt() && lit->equalsScalar(1)))) {
            stream2 << "++";
          }
          else {
            if (op->use_atomics) {
              stream2 << " += reduce_add(";
              add->b.accept(this);
              stream2 << ")";
            }
            else {
              stream2 << " += ";
              add->b.accept(this);
            }
          }
          printed = true;
        }
      }
      else if (isa<Mul>(op->rhs)) {
        auto mul = to<Mul>(op->rhs);
        if (mul->a == op->lhs) {
          stream2 << " *= ";
          mul->b.accept(this);
          printed = true;
        }
      }
      else if (isa<BitOr>(op->rhs)) {
        auto bitOr = to<BitOr>(op->rhs);
        if (bitOr->a == op->lhs) {
          stream2 << " |= ";
          bitOr->b.accept(this);
          printed = true;
        }
      }
    }
    if (!printed) {
      stream2 << " = ";
      op->rhs.accept(this);
    }

    stream2 << ";";
    stream2 << endl;

    IRPrinter::visit(op);
  }
  else {
    CodeGen_C::visit(op);
  
  }

  
}

void CodeGen_ISPC::visit(const Store* op) {
  if (is_ISPC_code_stream_enabled()) {
    if (op->use_atomics) {
      doIndent();
      stream2 << getAtomicPragma() << endl;
    }
  }
  else {
    if (op->use_atomics) {
      doIndent();
      stream << getAtomicPragma() << endl;
    }    
  }
  IRPrinter::visit(op);
}

}
}
