// RUN: /Users/jun/Work/pj/kvcache-dsl-workspace/build/tools/kv-opt/kv-opt %s -kv-to-vector | FileCheck %s

module {
  %cache = "kv.alloc"() {
    layout = #kv.layout<
      affine_map<(s,h,d)->(s,h,d)>,
      alignment = 128
    >,
    schedule = #kv.schedule<[0,1,2],[1,1,16]>
  } : () -> !kv.cache<f16, [2048,16,128], "key">

  %v, %status = "kv.vector_load"(%cache) {
    dim = 2,
    width = 16
  } : (!kv.cache<f16, [2048,16,128], "key">)
    -> (vector<16xf16>, i32)
}

// CHECK-NOT: vector.broadcast
// CHECK: vector.transfer_read