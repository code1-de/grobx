// c-suite-library-path's test code: a libgrobxc.so other than the build's,
// which test 1 puts first on LD_LIBRARY_PATH. It defines the functions
// linkage.cpp calls, and its version is no crate's.
#include "grobx.h"

grobx_params grobx_params_new(uint32_t order, size_t threads, uint64_t seed) {
    grobx_params p = {
        .order = order,
        .threads = threads,
        .seed = seed,
        .deadline = GROBX_NONE,
        .memory = GROBX_NONE,
    };
    return p;
}

const char *grobx_version(void) {
    return "stale";
}
