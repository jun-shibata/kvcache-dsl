#include "kv/KVDialect.h"
#include "kv/KVOps.h"
#include "kv/KVTypes.h"
#include "kv/KVAttributes.h"

#include "mlir/IR/DialectImplementation.h"

// --- Types ---
#define GET_TYPEDEF_CLASSES
#include "KVTypes.cpp.inc"

// --- Attributes ---
#define GET_ATTRDEF_CLASSES
#include "KVAttributes.cpp.inc"

using namespace mlir;
using namespace kv;

MLIR_DEFINE_EXPLICIT_TYPE_ID(kv::KVDialect)

KVDialect::~KVDialect() = default;

KVDialect::KVDialect(mlir::MLIRContext *ctx)
  : Dialect(getDialectNamespace(), ctx, TypeID::get<KVDialect>()) {
    initialize();
}

void KVDialect::initialize() {
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