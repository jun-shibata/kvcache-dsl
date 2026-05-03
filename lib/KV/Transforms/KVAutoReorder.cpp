#include "kv/KVOps.h"
#include "kv/KVDialect.h"
#include "kv/KVAttributes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

// Build an affine map that puts `dim` as the innermost dimension.
// For rank-3 (s, h, d), if dim=2 -> identity; if dim=0 -> (h, d, s); etc/
static AffineMap makeInnermostMap(MLIRContext *ctx, int64_t rank, int64_t dim) {
  SmallVector<AffineExpr> exprs;
  exprs.reserve(rank);

  SmallVector<unsigned> order;
  order.reserve(rank);

  // keep all dims except `dim` first
  for (unsigned i = 0; i < (unsigned)rank; ++i) {
    if ((int64_t)i != dim) {
      order.push_back(i);
    }
  }

  // push target dim last (innermost)
  order.push_back((unsigned)dim);

  for (unsigned i : order) {
    exprs.push_back(getAffineDimExpr(i, ctx));
  }
  return AffineMap::get(rank, 0, exprs, ctx);
}

struct KVAutoReorderPass
  : public PassWrapper<KVAutoReorderPass, OperationPass<ModuleOp>> {

  Option<int> Threshold {
    *this, "threshold",
    llvm::cl::desc("Reorder when vectorization_cost > threshold"),
    llvm::cl::init(5)
  };

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    getOperation().walk([&](kv::KVVectorLoadOp op) {
      auto costAttr = op->getAttrOfType<IntegerAttr>("vectorization_cost");
      if (!costAttr) { return; }
      if (costAttr.getInt() <= Threshold) { return; }
      Value cache = op.getCache();
      auto cacheTy = cache.getType().dyn_cast<kv::CacheType>();
      if (!cacheTy) { return; } // Only handle KV cache values

      // Assume rank = 3 for now (S,H,D)
      int64_t rank = 3;
      int64_t dim = op.getDim();

      // Build new layout: make `dim` innermost
      AffineMap newMap = makeInnermostMap(ctx, rank, dim);

      // Wrap into your LayoutAttr
      auto layoutAttr = kv::LayoutAttr::get(
        ctx, newMap,
        /*alignment=*/128
      );

      OpBuilder builder(op);

      // Insert reorder before op
      auto reordered = builder.create<kv::KVReorderOp>(
        op.getLoc(),
        cacheTy,
        cache,
        layoutAttr
      );

      // Rewire vector_load to use reordered cache
      op.getCacheMutable().assign(reordered);

      // (Optional) clear stale analysis attrs to force recompute if rerun
      op->removeAttr("vectorization_cost");
      op->removeAttr("kv.vec_status");
    });
  }
};

} // namespace

// Registration hook
std::unique_ptr<Pass> createKVAutoReorderPass() {
  return std::make_unique<KVAutoReorderPass>();
}