module {
  // Transform layout from logical (s,h,d)
  // into physical (d,s,h).
  //
  // This is intended to test whether KVCache-DSL
  // can eliminate strided vector accesses by changing
  // the memory layout so that the vectorized dimension
  // becomes contiguous for the generated vector loop.

  %cache = "kv.alloc"() {
    layout = #kv.layout<
      affine_map<(s,h,d)->(d,s,h)>,
      alignment = 128
    >,
    schedule = #kv.schedule<[2,0,1],[16,1,1]>
  } : () -> !kv.cache<f16, [2048,16,128], "key">

  // Vectorize along dimension d.
  //
  // In the original SHD layout this access would be strided,
  // but after transforming to DSH the vector loop can become
  // contiguous.
  %v, %status = "kv.vector_load"(%cache) {
    dim = 2,
    width = 16
  } : (!kv.cache<f16, [2048,16,128], "key">)
    -> (vector<16xf16>, i32)
}
