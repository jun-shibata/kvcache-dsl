#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"


#include "kv/KVDialect.h"

// Forward declare your pass
std::unique_ptr<mlir::Pass> createKVToVectorPass();
std::unique_ptr<mlir::Pass> createKVLayoutAnalysisPass();
std::unique_ptr<mlir::Pass> createKVAutoReorderPass();

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<
    kv::KVDialect,
    mlir::vector::VectorDialect,
    mlir::memref::MemRefDialect,
    mlir::arith::ArithDialect
  >();

  // Register pass
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createKVToVectorPass();
  });

  mlir::registerPass([] -> std::unique_ptr<mlir::Pass> {
    return createKVLayoutAnalysisPass();
  });
  
  mlir::registerPass([] -> std::unique_ptr<mlir::Pass> {
    return createKVAutoReorderPass();
  });

  return mlir::asMainReturnCode(
    mlir::MlirOptMain(argc, argv, "KVCache DSL optimizer\n", registry));
}