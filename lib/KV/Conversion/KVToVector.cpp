#include "kv/KVAttributes.h"
#include "kv/KVOps.h"
#include "kv/KVTypes.h"

#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/APFloat.h"

using namespace mlir;

struct KVVectorLoadLowering
  : public OpConversionPattern<kv::VectorLoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
    kv::VectorLoadOp op,
    typename kv::VectorLoadOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();

    auto cacheTy = mlir::dyn_cast<kv::CacheType>(op.getCache().getType());
    if (!cacheTy)
      return failure();

    // Build a flat row-major MemRefType from the cache shape/element type.
    SmallVector<int64_t> shape(cacheTy.getShape().asArrayRef());
    auto memrefType = MemRefType::get(shape, cacheTy.getElementType());

    // Bridge kv.cache -> memref; a later pass can concretize this.
    auto castOp = UnrealizedConversionCastOp::create(
        rewriter, loc, TypeRange{memrefType}, ValueRange{adaptor.getCache()});
    Value memref = castOp.getResult(0);

    int64_t dim = op.getDim();
    int64_t width = op.getWidth();

    // Determine if this load is unit-stride: prefer the pre-computed annotation
    // from -kv-layout-analysis; fall back to inspecting the layout's affine map
    // directly so that -kv-to-vector works without running -kv-layout-analysis.
    bool canVectorize = false;
    if (auto costAttr = op->getAttrOfType<IntegerAttr>("vectorization_cost")) {
      canVectorize = (costAttr.getInt() == 0);
    } else {
      Value cache = op.getCache();
      kv::LayoutAttr layout;
      if (auto allocOp = cache.getDefiningOp<kv::AllocOp>())
        layout = allocOp.getLayout();
      else if (auto reorderOp = cache.getDefiningOp<kv::ReorderOp>())
        layout = reorderOp.getTargetLayout();

      if (layout) {
        AffineMap map = layout.getMap().getValue();
        if (map.getNumResults() > 0) {
          AffineExpr lastResult = map.getResult(map.getNumResults() - 1);
          canVectorize = (lastResult == getAffineDimExpr(dim, rewriter.getContext()));
        }
      }
    }

    // Build result types
    auto elemType = memrefType.getElementType();
    auto vecType = VectorType::get({width}, elemType);

    // Status: 0=ok, 1=strided
    Value status;

    // --- vector.transfer_read path ---
    SmallVector<Value> indices;
    for (int i = 0; i < memrefType.getRank(); ++i) {
      indices.push_back(arith::ConstantIndexOp::create(rewriter, loc, 0).getResult());
    }

    if (canVectorize) {
      // Map from memref rank -> vector rank 1, selecting the vectorized dim.
      auto map = AffineMap::get(memrefType.getRank(), 0,
          {getAffineDimExpr(dim, rewriter.getContext())},
          rewriter.getContext());

      auto floatType = mlir::cast<FloatType>(elemType);
      Value padding = arith::ConstantFloatOp::create(
          rewriter, loc, floatType,
          APFloat::getZero(floatType.getFloatSemantics())).getResult();

      SmallVector<bool> inBounds(vecType.getRank(), false);
      auto read = vector::TransferReadOp::create(
        rewriter, loc, vecType, memref, indices,
        padding,
        map,
        ArrayRef<bool>(inBounds)
      );

      status = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
      rewriter.replaceOp(op, {read, status});
      return success();
    }

    // --- fallback path (scalar load) ---
    auto scalar = memref::LoadOp::create(rewriter, loc, memref, indices);

    // Expand scalar -> vector (broadcast)
    auto vec = vector::BroadcastOp::create(rewriter, loc, vecType, scalar);

    status = arith::ConstantIntOp::create(rewriter, loc, 1, 32);
    
    rewriter.replaceOp(op, {vec, status});
    return success();
  }
};

struct KVToVectorPass : public PassWrapper<KVToVectorPass, OperationPass<mlir::ModuleOp>> {
  StringRef getArgument() const final { return "kv-to-vector"; }
  
  StringRef getDescription() const final {
    return "Lower KV dialect to vector dialect";
  }

  void runOnOperation() override {
    // Ensure target dialects are loaded before any ops are created in patterns.
    // test.mlir contains only kv.* ops so these dialects may not be loaded yet.
    getContext().loadDialect<
        mlir::arith::ArithDialect,
        mlir::memref::MemRefDialect,
        mlir::vector::VectorDialect>();

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<KVVectorLoadLowering>(&getContext());

    ConversionTarget target(getContext());
    
    target.addLegalDialect<
      mlir::vector::VectorDialect,
      mlir::memref::MemRefDialect,
      mlir::arith::ArithDialect>();

    target.addLegalOp<mlir::UnrealizedConversionCastOp>();
    target.addIllegalOp<kv::VectorLoadOp>();

    if (failed(mlir::applyPartialConversion(
      getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> createKVToVectorPass() {
  return std::make_unique<KVToVectorPass>();
}