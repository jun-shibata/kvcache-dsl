# --- Ops ---
set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/kv/KVOps.td)

mlir_tablegen(KVOps.h.inc -gen-op-decls)
mlir_tablegen(KVOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(KVOpsIncGen)

# --- Types ---
set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/kv/KVTypes.td)

mlir_tablegen(KVTypes.h.inc -gen-typedef-decls)
mlir_tablegen(KVTypes.cpp.inc -gen-typedef-defs)

add_public_tablegen_target(KVTypesIncGen)

# --- Attributes ---
set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/kv/KVAttributes.td)

mlir_tablegen(KVAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(KVAttributes.cpp.inc -gen-attrdef-defs)

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