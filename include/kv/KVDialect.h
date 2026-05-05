#pragma once
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#define GET_DIALECT_DECL
#include "KVDialect.h.inc"

/*
namespace kv {
class KVDialect : public mlir::Dialect {
  public:
   explicit KVDialect(mlir::MLIRContext *ctx);
   void initialize();
};

static llvm::StringRef getDialectNamespace() { return "kv"; }
}
*/

/*
namespace kv {
  class KVDialect : public mlir::Dialect {
    public:
      explicit KVDialect(mlir::MLIRContext *ctx);

      static llvm::StringRef getDialectNamespace() { return "kv"; }
    };
}
*/