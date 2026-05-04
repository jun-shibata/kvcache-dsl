#pragma once
#include "mlir/IR/Dialect.h"

namespace kv {
  class KVDialect : public mlir::Dialect {
    public:
      explicit KVDialect(mlir::MLIRContext *ctx);
  };
}