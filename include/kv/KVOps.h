#pragma once

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/DialectImplementation.h"

#include "llvm/ADT/TypeSwitch.h"

#include "kv/KVTypes.h"
#include "kv/KVAttributes.h"
#include "kv/KVDialect.h"

#define GET_OP_CLASSES
#include "KVOps.h.inc"