#include "KVDialect.h"
#include "KVOps.h"
#include "KVTypes.h"

using namespace mlir;
using namespace kv;

KVDialect::KVDialect(MLIRContext *ctx)
  : Dialect("kv", ctx, TypeID::get<KVDialect>()) {
    addTypes<
    #define GET_TYPEDEF_LIST
    #include "KVTypes.cpp.inc"
    >();

    addAttributes<
    #define GET_ATTRDEF_LIST
    #include "KVAttributes.cpp.inc"
    >();

    addOperations<
    #define GET_OP_LIST
    #include "KVOps.cpp.inc"
    >();
  }