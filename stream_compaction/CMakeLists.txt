set(headers
    "common.h"
    "cpu.h"
    "naive.h"
    "efficient.h"
    "thrust.h"
    "radix.h"
    "sharedmem.h"
    )

set(sources
    "common.cu"
    "cpu.cu"
    "naive.cu"
    "efficient.cu"
    "thrust.cu"
    "radix.cu"
    "sharedmem.cu"
    )

list(SORT headers)
list(SORT sources)

source_group(Headers FILES ${headers})
source_group(Sources FILES ${sources})

add_library(stream_compaction ${sources} ${headers})
if(CMAKE_VERSION VERSION_LESS "3.23.0")
    set_target_properties(stream_compaction} PROPERTIES CUDA_ARCHITECTURES OFF)
elseif(CMAKE_VERSION VERSION_LESS "3.24.0")
    set_target_properties(stream_compaction PROPERTIES CUDA_ARCHITECTURES all-major)
else()
    set_target_properties(stream_compaction PROPERTIES CUDA_ARCHITECTURES native)
endif()
target_compile_options(stream_compaction PRIVATE "$<$<AND:$<CONFIG:Debug,RelWithDebInfo>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
target_compile_options(stream_compaction PRIVATE "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo;-src-in-ptx>")
if(MSVC)
    # CUDA 13.x's CCCL/Thrust headers require MSVC's conforming preprocessor.
    target_compile_options(stream_compaction PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>")
endif()

# Overridable at configure time (-DBLOCK_SIZE=256) to sweep block sizes.
set(BLOCK_SIZE 128 CACHE STRING "CUDA thread block size used by naive/efficient/radix/sharedmem")
target_compile_definitions(stream_compaction PRIVATE BLOCK_SIZE=${BLOCK_SIZE})
