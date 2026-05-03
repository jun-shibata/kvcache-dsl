include "mlir/IR/OpBase.td"
include "kv/KVTypes.td"
include "kv/KVAttributes.td"

def KV_AllocOp : Op<KVDialect, "alloc"> {
  let summary = "Allocate KV cache";
  let arguments = (ins
    KV_LayoutAttr:$layout,
    KV_ScheduleAttr:$schedule
  );

  let results = (outs KV_CacheType:$result);
  let assemblyFormat = "`{` $layout `,` $schedule `}` attr-dict `:` type($result)";
}

// --- append ---
def KV_AppendOp : Op<KVDialect, "append"> {
  let summary = "Append to KV cache";

  let arguments = (ins
    KV_CacheType:$cache,
    Index:$start,
    Index:$end
  );

  let results = (outs KV_CacheType:$view);
}

// --- load ---
def KV_LoadOp : Op<KVDialect, "load"> {
  let summary = "Structured load";

  let arguments = (ins
    KV_CacheType:$view,
    KV_AccessAttr:$access
  );

  let results = (outs AnyTensor:$result);
}

// --- gather ---
def KV_GatherOp : Op<KVDialect, "gather"> {
  let summary = "Irregular gather";
  
  let arguments = (ins
    KV_CacheType:$view
  );

  let results = (outs AnyTensor:$result);
}

// --- reorder ---
def KV_ReorderOp : Op<KVDialect, "reorder"> {
  let summary = "Change layout";

  let arguments = (ins
    KV_CacheType:$cache,
    KV_LayoutAttr:$targetLayout
  );

  let results = (outs KV_CacheType:$result);
}

// --- vector load ---
def KV_VectorLoadOp : Op<KVDialect, "vector_load"> {
  let summary = "vectorized load fallback semantics";

  let arguments = (ins
    KV_CacheType:$cache,
    I64Attr:$dim,
    I64Attr:$width
  );

  let results = (outs
    AnyType:$vector,
    I32:$status       // 0=ok, 1=strided, 2=misaligned
  );
}