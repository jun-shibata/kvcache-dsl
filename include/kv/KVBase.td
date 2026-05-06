include "mlir/IR/AttrTypeBase.td"
include "kv/KVDialect.td"

// --- Attributes ---

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
    "mlir::DenseI64ArrayAttr":$loopOrder,
    "mlir::DenseI64ArrayAttr":$tile
  );

  let mnemonic = "schedule";

  let assemblyFormat = "`<` $loopOrder `,` $tile `>`";
}

// --- Types ---

def KV_CacheType : TypeDef<KVDialect, "Cache"> {
  let summary = "KV Cache logical type";
  let parameters = (ins
    "mlir::Type":$elementType,
    "mlir::DenseI64ArrayAttr":$shape, // (S, H, D)
    "mlir::StringAttr":$role  // "key" | "value"
  );

  let mnemonic = "cache";
  let assemblyFormat = "`<` $elementType `,` $shape `,` $role `>`";
}
