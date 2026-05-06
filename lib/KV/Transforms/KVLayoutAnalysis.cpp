#include "kv/KVOps.h"
#include "kv/KVDialect.h"
#include "kv/KVAttributes.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

// Returns true if logical `dim` is the innermost (unit-stride) position
// in the layout's affine map. For a permutation map, "innermost" means
// dim appears as the last result expression.

static bool isInnermostDim(kv::LayoutAttr layout, int64_t dim, MLIRContext *ctx) {
  AffineMap map = layout.getMap().getValue();
  if (map.getNumResults() == 0) {
    return false;
  }
  AffineExpr lastResult = map.getResult(map.getNumResults() - 1);
  return lastResult == getAffineDimExpr(dim, ctx);
}

struct KVLayoutAnalysisPass
  : public PassWrapper<KVLayoutAnalysisPass, OperationPass<ModuleOp>> {
  
  StringRef getArgument() const final { return "kv-layout-analysis"; }

  StringRef getDescription() const final {
    return "Analyze KV cache layout for vectorization";
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    getOperation().walk([&](kv::VectorLoadOp op) {
      // Expect the cache to be already lowered (or wrapped) as memref.
      // If not, skip (you can extend later to read #kv.layout).
      Value cache = op.getCache();

      if (!mlir::isa<kv::CacheType>(cache.getType())) { return; }

      int64_t dim = op.getDim();
      int64_t width = op.getWidth();

      // Trace back to the op that carries the layout attributes.
      kv::LayoutAttr layout;
      if (auto allocOp = cache.getDefiningOp<kv::AllocOp>()) {
        layout = allocOp.getLayout();
      }
      else if (auto reorderOp = cache.getDefiningOp<kv::ReorderOp>()) {
        layout = reorderOp.getTargetLayout();
      }

      if (!layout) return;

      bool unit = isInnermostDim(layout, dim, ctx);
      int32_t cost = unit ? 0 : 10;
     
      op->setAttr("vectorization_cost",
                  IntegerAttr::get(IntegerType::get(ctx, 32), cost));
      
      op->setAttr("kv.vec_status",
                  StringAttr::get(ctx, unit ? "ok" : "strided"));
      
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