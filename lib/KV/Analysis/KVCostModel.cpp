#include <cstdint>
#include "mlir/Dialect/Affine/IR/AffineOps.h"

struct KVVectorCost {
  int64_t stride = 1;
  int64_t vectorizationCost = 0;
  bool contiguous = false;
};

static int64_t getLinearStride(AffineMap layoutMap,
                               ArrayRef<int64_t> shape,
                               int64_t logicalDim) {
  // Assumption:
  // layoutMap maps logical dims -> physical order.
  // Example:
  //   (s, h, d) -> (s, h, d)
  // physical dims: [s, h, d]
  //
  // Row-major stride:
  //   stride(physical i) = product of sizes after i.

  unsigned rank = shape.size();

  int64_t physicalPos = -1;
  for (unsigned i = 0; i < rank; ++i) {
    auto dimExpr = layoutMap.getResult(i).dyn_cast<AffineDimExpr>();
    if (!dimExpr)
      return std::numeric_limits<int64_t>::max();
    if ((int64_t)dimExpr.getPosition() == logicalDim) {
      physicalPos = i;
      break;
    }
  }

  if (physicalPos < 0)
    return std::numeric_limits<int64_t>::max();
  
  int64_t stride = 1;
  for (unsigned i = physicalPos + 1; i < rank; ++i) {
    auto dimExpr = layoutMap.getResult(i).dyn_cast<AffineDimExpr>();
    if (!dimExpr)
      return std::numeric_limits<int64_t>::max();
    
    unsigned logicalPos = dimExpr.getPosition();
    stride *= shape[logicalPos];
  }

  return stride;
}

static KVVectorCost estimateVectorLoadCost(AffineMap layoutMap,
                                           ArrayRef<int64_t> shape,
                                           int64_t dim,
                                           int64_t vectorWidth) {
  KVVectorCost cost;
  cost.stride = getLinearStride(layoutMap, shape, dim);
  cost.contiguous = cost.stride == 1;

  if (cost.contiguous) {
    cost.vectorizationCost = 1;
  } else {
    cost.vectorizationCost = cost.stride * vectorWidth;
  }

  return cost;
}