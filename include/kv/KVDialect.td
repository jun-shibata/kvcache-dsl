include "mlir/IR/DialectBase.td"

def KVDialect : Dialect {
  let name = "kv";
  let cppNamespace = "::kv";
  let summary = "KV Cache DSL Dialect";
  let description = [{
    "Dialect for representing KV cache layout, access atterns,
    and vectorization-aware transformations."
  }];
}