include "mlir/IR/AttrTypeBase.td"
include "kv/KVDialect.td"

def KV_LayoutAttr : AttrDef<KVDialect, "Layout"> {
  let summary = "Layout as affine map + alignment";
  let parameters = (ins
    "mlir::AffineMapAttr":$map,
    "mlir::IntegerAttr":$alignment
  );

  let mnemonic = "layout";

  let assemblyFormat = "`<` $map `,` `alignment` `=` $alignment `>`";
}

def KV_AccessAttr : AttrDef<KVDialect, "Access"> {
  let summary = "Access pattern";
  let parameters = (ins
    "mlir::StringAttr":$pattern  // "seq_major", etc.
  );

  let mnemonic = "access";

  let assemblyFormat = "`<` $pattern `>`";
}

def KV_ScheduleAttr : AttrDef<KVDialect, "Schedule"> {
  let summary = "Loop schedule";
  let parameters = (ins
    "mlir::StringAttr":$pattern,
    "mlir::ArrayAttr":$loopOrder,
    "mlir::ArrayAttr":$tile
  );

  let mnemonic = "schedule";

  let assemblyFormat = "`<` $loopOrder `,` $tile `>`";
}

