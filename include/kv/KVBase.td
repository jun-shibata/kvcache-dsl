include "mlir/IR/AttrTypeBase.td"
include "kv/KVDialect.td"

// --- Attributes ---

def KV_LayoutAttr : AttrDef<KVDialect, "Layout"> {
  let summary = "Layout as affine map + alignment";
  let parameters = (ins
    "mlir::AffineMap":$map,
    "int64_t":$alignment
  );

  let mnemonic = "layout";

  let assemblyFormat = "`<` $map `,` `alignment` `=` $alignment `>`";
}

def KV_AccessAttr : AttrDef<KVDialect, "Access"> {
  let summary = "Access pattern";
  let parameters = (ins
    "mlir::StringRef":$pattern  // "seq_major", etc.
  );

  let mnemonic = "access";

  let assemblyFormat = "`<` $pattern `>`";
}

def KV_ScheduleAttr : AttrDef<KVDialect, "Schedule"> {
  let summary = "Loop schedule";
  let parameters = (ins
    "mlir::ArrayRef<int64_t>":$loopOrder,
    "mlir::ArrayRef<int64_t>":$tile
  );

  let mnemonic = "schedule";

  let assemblyFormat = "`<` $loopOrder `,` $tile `>`";
}

// --- Types ---

def KV_CacheType : TypeDef<KVDialect, "Cache"> {
  let summary = "KV Cache logical type";
  let parameters = (ins
    "mlir::Type":$elementType,
    "mlir::ArrayRef<int64_t>":$shape, // (S, H, D)
    "mlir::StringRef":$role           // "key" | "value"
  );

  let mnemonic = "cache";
  let assemblyFormat = "`<` $elementType `,` $shape `,` $role `>`";
}
