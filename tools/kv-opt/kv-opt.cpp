#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "kv/KVDialect.h"

// Forward declare your pass
std::unique_ptr<mlir::Pass> createKVToVectorPass();

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
    return cretateKVAutoReorderPass();
  });

  return mlir::asMainReturnCode(
    mlir::MlirOptMain(argc, argv, "KVCache DSL optimizer\n", registry));
}