# --- Dialect ---
set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/kv/KVDialect.td)

mlir_tablegen(KVDialect.h.inc -gen-dialect-decls -dialect=kv)
mlir_tablegen(KVDialect.cpp.inc -gen-dialect-defs -dialect=kv)

add_public_tablegen_target(KVDialectIncGen)

# --- Ops ---
set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/kv/KVOps.td)

mlir_tablegen(KVOps.h.inc -gen-op-decls)
mlir_tablegen(KVOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(KVOpsIncGen)

# --- Types ---
set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/kv/KVBase.td)

mlir_tablegen(KVTypes.h.inc -gen-typedef-decls -typedefs-dialect=kv)
mlir_tablegen(KVTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=kv)

add_public_tablegen_target(KVTypesIncGen)

# --- Attributes ---
set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/kv/KVBase.td)

mlir_tablegen(KVAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=kv)
mlir_tablegen(KVAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=kv)

add_public_tablegen_target(KVAttributesIncGen)

# --- Dialect library ---
add_mlir_dialect_library(KVCacheDSL
  KVDialect.cpp
  KVTypes.cpp
  KVAttributes.cpp
  KVOps.cpp

  Transforms/KVLayoutAnalysis.cpp
  Transforms/KVAutoReorder.cpp

  Conversion/KVToVector.cpp

  DEPENDS
  KVDialectIncGen
  KVOpsIncGen
  KVTypesIncGen
  KVAttributesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRAffineDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRVectorDialect
  MLIRSCFDialect
  MLIRArithDialect
)

target_include_directories(KVCacheDSL PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}
)