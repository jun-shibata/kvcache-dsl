#include "kv/KVOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

static bool isUnitStrideInnerDim(MemRefType type, int64_t dim) {
  int64_t offset;
  SmallVector<int64_t> strides;

  if (failed(getStridesAndOffset(type, strides, offset)))
    return false;

  if (dim < 0 || dim >= (int64_t)strides.size())
    return false;
  
  return strides[dim] == 1;
}

struct KVToVectorPass : public PassWrapper<KVToVectorPass, OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    populateKVToVectorPatterns(patterns);

    mlir::ConversionTarget target(getContext());

    target.addLegalDialect<
      mlir::vector::VectorDialect,
      mlir::vector::MemRefDialect,
      mlir::arith::ArithDialect>();

    target.addIllegalOp<kv::KVVectorLoadOp>();

    if (failed(mlir::applyPartialConversion(
      getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

struct KVVectorLoadLowering
  : public OpConversionPattern<kv::KVVectorLoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
  kv::KVVectorLoadOp op,
  OpAdaptor adaptor,
  ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    
    // Assume cache already lowered to memref
    Value memref = adaptor.getCache();

    auto memrefType = memref.getType().dyn_cast<MemRefType>();
    if (!memrefType) {
      return failure();
    }

    int64_t dim = op.getDim();
    int64_t width = op.getWidth();

    bool canVectorize = isUnitStrideInnerDim(memrefType, dim);

    // Build result types
    auto elemType = memrefType.getElementType();
    auto vecType = VectorType::get({width}, elemType);

    // Status: 0=ok, 1=strided
    Value status;

    if (canVectorize) {
      // --- vector.transfer_read path ---

      SmallVector<Value> indices;
      for (int i = 0; i < memrefType.getRank(); ++i) {
        indices.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
      }

      auto read = rewriter.create<vector::TransferReadOp>(
        loc,
        vecType,
        memref,
        indices,
        Value(), // padding
        Value(), // mask
        ArrayAttr() // in_bounds
      );

      status = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);

      rewriter.replaceOp(op, {read, status});
      return success();
    }

    // --- fallback path (scalar load) ---
    SmallVector<Value> indices;
    for (int i = 0; i < memrefType.getRank(); ++i) {
      indices.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
    }

    auto scalar = rewriter.create<memref::LoadOp>(loc, memref, indices);

    // Expand scalar -> vector (broadcast)
    auto vec = rewriter.create<vector::BroadcastOp>(
      loc, vecType, scalar
    );

    status = rewriter.create<arith::ConstantIntOp>(loc, 1, 32);
    
    rewriter.replaceOp(op, {vec, status});
    return success();
  }
}
