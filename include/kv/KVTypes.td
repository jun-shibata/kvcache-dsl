include "mlir/IR/AttrTypeBase.td"

def KV_CacheType : TypeDef<KVDialect, "Cache"> {
  let summary = "KV Cache logical type";
  let parameters = (ins
    "mlir::Type":$elementType,
    "mlir::ArrayRef<int64_t>":$shape, // (S, H, D)
    "mlir::StringRef":$role           // "key" | "value"
  );
  let assemblyFormat = "`<` $elementType `,` $shape `,` $role `>`";
}