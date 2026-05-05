#include "kv/KVTypes.h"
#include "kv/KVDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

#include "llvm/ADT/TypeSwitch.h"

#define GET_TYPEDEF_CLASSES
#include "KVTypes.cpp.inc"
