#ifndef TACO_BACKEND_C_H
#define TACO_BACKEND_C_H
#include <map>
#include <vector>

#include "taco/ir/ir.h"
#include "taco/ir/ir_printer.h"
#include "codegen.h"

namespace taco {
namespace ir {


class CodeGen_C : public CodeGen {
public:
  /// Initialize a code generator that generates code to an
  /// output stream.
  CodeGen_C(std::ostream &dest, OutputKind outputKind, bool simplify=true);
  CodeGen_C(std::ostream &dest, std::ostream &dest2, OutputKind outputKind, bool simplify=true);
  ~CodeGen_C();

  /// Compile a lowered function
  void compile(Stmt stmt, bool isFirst=false);

  /// Generate shims that unpack an array of pointers representing
  /// a mix of taco_tensor_t* and scalars into a function call
  static void generateShim(const Stmt& func, std::stringstream &stream);

protected:
  using IRPrinter::visit;

  virtual void visit(const Function*);
  virtual void visit(const VarDecl*);
  virtual void visit(const Yield*);
  virtual void visit(const Var*);
  virtual void visit(const For*);
  virtual void visit(const While*);
  virtual void visit(const GetProperty*);
  virtual void visit(const Min*);
  virtual void visit(const Max*);
  virtual void visit(const Allocate*);
  virtual void visit(const Sqrt*);
  virtual void visit(const Store*);
  virtual void visit(const Assign*);

  std::map<Expr, std::string, ExprCompare> varMap;
  std::vector<Expr> localVars;
  std::ostream &out;
  std::ostream &out2;
  
  OutputKind outputKind;

  std::string funcName;
  int labelCount;
  bool emittingCoroutine;

  class FindVars;

private:
  virtual std::string restrictKeyword() const { return "restrict"; }
};

} // namespace ir
} // namespace taco
#endif
