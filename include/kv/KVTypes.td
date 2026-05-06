include "mlir/IR/AttrTypeBase.td"
include "kv/KVDialect.td"

def KV_CacheType : TypeDef<KVDialect, "Cache"> {
  let summary = "KV Cache logical type";
  let parameters = (ins
    "mlir::Type":$elementType,
    "mlir::ArrayAttr":$shape, // (S, H, D)
    "mlir::StringAttr":$role  // "key" | "value"
  );

  let mnemonic = "cache";
  let assemblyFormat = "`<` $elementType `,` attr-dict `>`";
}