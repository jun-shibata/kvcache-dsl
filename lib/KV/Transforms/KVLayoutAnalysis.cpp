#include "kv/KVOps.h"
#include "kv/KVDialect.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

static bool isUnitStrideInnerDim(MemRefType type, int64_t dim) {
  int64_t offset;
  SmallVector<int64_t> strides;
  if (failed(getStridesAndOffset(type, strides, offset))) {
    return false;
  }
  if (dim < 0 || dim >= (int64_t)strides.size()) {
    return false;
  }
  return strides[dim] == 1;
}

struct KVLayoutAnalysisPass
  : public PassWrapper<KVLayoutAnalysisPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    getOperation().walk([&](kv::KVVectorLoadOp op) {
      // Expect the cache to be already lowered (or wrapped) as memref.
      // If not, skip (you can extend later to read #kv.layout).
      Value cache = op.getCache();
      auto memrefTy = cache.getType().dyn_cast<MemRefType>();
      if (!memrefTy) {
        return;
      }
      int64_t dim = op.getDim();
      int64_t width = op.getWidth();
      bool unit = isUnitStrideInnerDim(memrefTy, dim);
      
      // Very simple heuristic for now
      int32_t cost = unit ? 0 : 10;
      op->setAttr("vectorization_cost",
                  IntegerAttr::get(IntegerType::get(ctx, 32), cost));
      op->setAttr("kv.vec_status",
                  StringAttr::get(ctx, unit ? "ok" : "strided"));
      // Optional: stash requested width (useful for later passes)
      op->setAttr("kv.vec_width",
                  IntegerAttr::get(IntegerType::get(ctx, 32), width));
    });
  }
};

} //namespace

// Registration hook
std::unique_ptr<Pass> createKVLayoutAnalysisPass() {
  return std::make_unique<KVLayoutAnalysisPass>();
}