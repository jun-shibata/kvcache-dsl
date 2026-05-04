#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

#include "kv/KVDialect.h"
#include "kv/KVOps.h"
#include "kv/KVTypes.h"
#include "kv/KVAttributes.h"

#define GET_OP_CLASSES
#include "KVOps.cpp.inc"
