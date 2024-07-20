enable_language(CUDA)

set(headers
    "common.h"
    "cpu.h"
    "naive.h"
    "efficient.h"
    "thrust.h"
    "hac.h"
    )

set(sources
    "common.cu"
    "cpu.cu"
    "naive.cu"
    "efficient.cu"
    "thrust.cu"
    "hac.cu"
    )

list(SORT headers)
list(SORT sources)

source_group(Headers FILES ${headers})
source_group(Sources FILES ${sources})

cuda_add_library(stream_compaction ${sources} ${headers})
