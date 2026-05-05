#include "kv/KVOps.h"

#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

static bool isUnitStrideInnerDim(MemRefType type, int64_t dim) {
  int64_t offset;
  SmallVector<int64_t> strides;

  if (failed(type.getStridesAndOffset(strides, offset)))
    return false;

  if (dim < 0 || dim >= (int64_t)strides.size())
    return false;
  
  return strides[dim] == 1;
}

struct KVVectorLoadLowering
  : public OpConversionPattern<kv::VectorLoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
    kv::VectorLoadOp op,
    typename kv::VectorLoadOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    
    // Assume cache already lowered to memref
    Value memref = adaptor.getCache();

    auto memrefType = mlir::dyn_cast<MemRefType>(memref.getType());
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

    // --- vector.transfer_read path ---
    SmallVector<Value> indices;
    for (int i = 0; i < memrefType.getRank(); ++i) {
      indices.push_back(arith::ConstantIndexOp::create(rewriter, loc, 0));
    }

    if (canVectorize) {
      auto map = AffineMap::getMultiDimIdentityMap(
        memrefType.getRank(), rewriter.getContext());

      auto read = vector::TransferReadOp::create(
        rewriter, loc, vecType, memref, indices,
        /*padding=*/Value(),
        /*permutation_map=*/map,
        /*in_bounds=*/ArrayRef<bool>{}
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
  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<KVVectorLoadLowering>(&getContext());

    ConversionTarget target(getContext());
    
    target.addLegalDialect<
      mlir::vector::VectorDialect,
      mlir::memref::MemRefDialect,
      mlir::arith::ArithDialect>();

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