#include "kv/KVAttributes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/TypeSwitch.h"

#define GET_ATTRDEF_CLASSES
#include "KVAttributes.cpp.inc"