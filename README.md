# kvcache-dsl
An MLIR-based DSL for making memory layout, access patterns, and vectorization jointly analyzable and transformable, targeting KV cache optimization in LLM inference.

## Motivation
Efficient KV cache access is critical for high-performance transformer inference (e.g., attention mechanisms). However:
- Memory layout (e.g., SHD vs HSD) significantly impacts performance
- Vectorization requires careful alignment of data layout and access patterns
- These optimizations are typically implemented manually and are error-prone

KVCache-DSL addresses this by:

- Separating logical layout from physical layout
- Allowing the compiler to automatically choose transformations
- Lowering high-level operations into efficient vectorized code

## Features
- High-level KV cache abstraction (kv.alloc, kv.vector_load)
- Layout specification via affine maps
- Automatic layout analysis and reordering
- Vectorization-aware lowering to MLIR vector dialect
- Extensible MLIR-based design

## Compilation Pipeline
KVCache-DSL transformations are applied using:

```
kv-opt input.mlir \
  -kv-layout-analysis \
  -kv-auto-reorder \
  -kv-to-vector
```

## Passes
- **kv-layout-analysis**  
Materializes and normalizes layout information
- **kv-auto-reorder**  
Reorders layout and schedule to enable efficient vector access
- **kv-to-vector**  
Lowers KVCache-DSL operations into standard MLIR (memref, vector)

## Example
### Input (KVCache-DSL)

```mlir
module {
  %cache = "kv.alloc"() {
    layout = #kv.layout<affine_map<(s,h,d)->(h,s,d)>, alignment = 128>,
    schedule = #kv.schedule<[0,1,2],[1,1,128]>
  }:() -> !kv.cache<f16, [2048,16,128], "key">

  %v, %status = "kv.vector_load"(%cache) { dim = 2, width = 16 }
    : (!kv.cache<f16, [2048,16,128], "key">) -> (vector<16xf16>, i32)
}
```

### Output (after lowering)

```mlir
#map = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
module {
  %0 = kv.alloc{<#map, alignment = 128 : i64>, <[0, 1, 2], [1, 1, 128]>}
    : <f16, [2048, 16, 128], "key">

  %1 = unrealized_conversion_cast %0
    : !kv.cache<f16, [2048, 16, 128], "key">
    to memref<2048x16x128xf16>

  %c0 = arith.constant 0 : index
  %c0_0 = arith.constant 0 : index
  %c0_1 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f16

  %2 = vector.transfer_read %1[%c0, %c0_0, %c0_1], %cst
    : memref<2048x16x128xf16>, vector<16xf16>

  %c0_i32 = arith.constant 0 : i32
}
```

### What This Transformation Does

This pipeline converts a high-level KV cache operation into a concrete vectorized memory access.

Key transformations:
1. **Layout materialization**  
```mlir
affine_map<(s,h,d)->(h,s,d)>
→
affine_map<(d0,d1,d2)->(d1,d0,d2)>
```
- Makes memory layout explicit
- Enables downstream optimizations
2. **KV cache → memref**  
```mlir
!kv.cache<...>
→
memref<2048x16x128xf16>  
```
- Converts abstract storage into a concrete memory buffer
3. **Abstract load → vector load**  
```mlir
kv.vector_load(dim=2, width=16)
→
vector.transfer_read  
```
- Translates logical access into hardware-friendly vector operations
- Uses contiguous dimension (d) for efficient access
4. **Explicit indexing**  
- Implicit access → explicit indices ([0,0,0])
- Padding value added for safe vector reads

## Key Insight

By specifying:

```mlir
(s,h,d) → (h,s,d)
```

the DSL ensures that:

- The d dimension becomes contiguous in memory
- Vector loads (width = 16) become efficient and aligned

This is critical for high-performance attention kernels.

## Why This Matters

KVCache-DSL enables:

- Layout-aware optimization without manual tuning
- Portable performance across hardware targets
- Cleaner separation between algorithm and memory layout

## Future Work
- Loop-level lowering for full attention kernels
- Support for FlashAttention-style tiling
- GPU-specific optimizations
- Async and prefetch support