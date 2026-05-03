#pragma once
#include "mlir/Dialect/MemRef/IR/MemRef.h"

inline bool isUnitStrideInnerDim(mlir::MemRefType type, int64_t dim) {
  int64_t offset;
  llvm::SmallVector<int64_t> strides;

  if (failed(getStridesAndOffset(type, strides, offset))) {
    return false;
  }

  if (dim < 0 || dim >= (int64_t)strides.size()) {
    return false;
  }

  return strides[dim] == 1;
}