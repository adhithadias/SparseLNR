#ifndef TACO_BACKEND_ISPC_H
#define TACO_BACKEND_ISPC_H
#include <map>
#include <vector>
#include <stdbool.h>

#include "taco/ir/ir.h"
#include "taco/ir/ir_printer.h"
#include "codegen_c.h"

namespace taco {
namespace ir {


class CodeGen_ISPC : public CodeGen {
public:
  /// Initialize a code generator that generates code to an
  /// output stream.
  CodeGen_ISPC(std::ostream &dest, OutputKind outputKind, bool simplify=true);
  CodeGen_ISPC(std::ostream &dest, std::ostream &dest2, OutputKind outputKind, bool simplify=true);
  ~CodeGen_ISPC();

  /// Compile a lowered function
  void compile(Stmt stmt, bool isFirst=false);

  /// Generate shims that unpack an array of pointers representing
  /// a mix of taco_tensor_t* and scalars into a function call
  static void generateShim(const Stmt& func, std::stringstream &stream);

protected:
  using IRPrinter::visit;

  void visit(const Function*);
  void visit(const VarDecl*);
  void visit(const Yield*);
  void visit(const Var*);
  void visit(const For*);
  void visit(const While*);
  void visit(const GetProperty*);
  void visit(const Min*);
  void visit(const Max*);
  void visit(const Allocate*);
  void visit(const Sqrt*);
  void visit(const Store*);
  void visit(const Assign*);

  Stmt simplifyFunctionBodies(Stmt stmt);
  std::string printCallISPCFunc(const std::string& funcName, std::map<Expr, std::string, ExprCompare> varMap,
                                std::vector<const GetProperty*> &sortedProps);
  void printISPCFunc(const Function *func, std::map<Expr, std::string, ExprCompare> varMap,
                                  std::vector<const GetProperty*> &sortedProps);

  std::map<Expr, std::string, ExprCompare> varMap;
  std::vector<Expr> localVars;
  bool taskCode = false;
  std::ostream &out;
  std::ostream &out2;
  
  OutputKind outputKind;

  std::string funcName;
  std::stringstream funcVariables;
  std::vector<const GetProperty*> sortedProps;
  int labelCount;
  bool emittingCoroutine;

  class FindVars;
  class FunctionCollector;

private:
  virtual std::string restrictKeyword() const { return "restrict"; }
  void sendToStream(std::stringstream &stream);
};

} // namespace ir
} // namespace taco
#endif
