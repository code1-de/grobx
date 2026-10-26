// c-interface's test code: grobx.h included in C++ as it stands, and its
// functions linked by their C names (c-interface.22). Prints the version.
#include <cstdio>

#include "grobx.h"

int main() {
    grobx_params p = grobx_params_new(GROBX_GREVLEX, 1, 1);
    if (p.deadline != GROBX_NONE) {
        return 1;
    }
    std::printf("%s\n", grobx_version());
    return 0;
}
