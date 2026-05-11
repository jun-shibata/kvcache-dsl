// RUN: kv-opt %s -kv-layout-analysis -kv-auto-reorder | FileCheck %s

module {
  %cache = "kv.alloc"() {
    layout = #kv.layout<
      affine_map<(s,h,d)->(d,s,h)>,
      alignment = 128
    >,
    schedule = #kv.schedule<[2,0,1],[16,1,1]>
  } : () -> !kv.cache<f16, [2048,16,128], "key">

  %v, %status = "kv.vector_load"(%cache) {
    dim = 2,
    width = 16
  } : (!kv.cache<f16, [2048,16,128], "key">)
    -> (vector<16xf16>, i32)
}

// CHECK-NOT: kv.reorder
// CHECK: targetLayout