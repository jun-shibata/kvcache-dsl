include "mlir/IR/AttrTypeBase.td"

def KV_LayoutAttr : AttrDef<KVDialect, "Layout"> {
  let summary = "Layout as affine map + alignment";
  let parameters = (ins
    "mlir::AffineMap":$map,
    "int64_t":$alignment
  );

  let assemblyFormat = "`<` $map `,` `alignment` `=` $alignment `>`";
}

def KV_AccessAttr : AttrDef<KVDialect, "Access"> {
  let summary = "Access pattern";
  let parameters = (ins
    "mlir::StringRef":$pattern  // "seq_major", etc.
  );

  let assemblyFormat = "`<` $pattern `>`";
}

def KV_ScheduleAttr : AttrDef<KVDialect, "Schedule"> {
  let summary = "Loop schedule";
  let parameters = (ins
    "mlir::ArrayRef<int64_t>":$loopOrder,
    "mlir::ArrayRef<int64_t>":$tile
  );
}

