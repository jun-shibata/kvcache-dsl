#pragma once

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/TypeSwitch.h"

#include "kv/KVDialect.h"
#include "kv/KVTypes.h"
#include "kv/KVAttributes.h"

// Generated op declarations
#define GET_OP_CLASSES
#include "KVOps.h.inc"
