import os
import lit.formats

from lit.llvm import llvm_config

# Test suite name
config.name = 'KVCacheDSL'

# Use ShTest format
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# File extensions treated as tests
config.suffixes = ['.mlir']

# Root directories
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = config.kvcache_obj_root + '/test'

# Add tool substitutions
llvm_config.use_default_substitutions()

# PATH
llvm_config.with_environment(
    'PATH',
    config.llvm_tools_dir,
    append_path=True
)

# Tools available in RUN lines
tool_dirs = [
    config.kvcache_tools_dir,
    config.llvm_tools_dir
]

tools = [
    'kv-opt',
    'FileCheck'
]

llvm_config.add_tool_substitutions(tools, tool_dirs)