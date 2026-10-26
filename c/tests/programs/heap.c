/* page-fault-time's test code: the main heap before and after a call of
 * grobx_groebner from a program's main thread.
 *
 * heap <instance file> <out dir>
 *
 * Before any call the program takes blocks of 64 KiB until
 * mallinfo2().arena grows, and writes the growth; then the excess of a
 * 1 MiB block, malloc_usable_size less the request. It calls
 * grobx_groebner on the file's text, grevlex, one thread, seed 1, and
 * writes the basis text to the file `basis` in the out dir. Then it takes
 * blocks of 64 KiB until the arena grows again and writes that growth,
 * and the excess of a 1 MiB and of a 64 MiB block. Standard output takes
 * the lines `before <bytes>`, `excess-before-1m <bytes>`, `status <n>`,
 * `after <bytes>`, `excess-1m <bytes>`, and `excess-64m <bytes>`. Exits 0,
 * or 99 where an argument or the file cannot be read, an allocation
 * fails, the arena does not grow within the blocks taken, or the basis
 * cannot be written.
 */
#define _GNU_SOURCE

#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "grobx.h"

/* The blocks the probe takes before it gives up on the arena's growth:
 * 2 GiB of 64 KiB blocks. */
#define PROBE_BLOCKS 32768
#define PROBE_BLOCK ((size_t)64 << 10)

static void fail(const char *what) {
    fprintf(stderr, "heap: %s\n", what);
    exit(99);
}

static char *slurp(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fail(path);
    }
    size_t size = 0, cap = 4096;
    char *buf = malloc(cap);
    size_t n;
    while (buf && (n = fread(buf + size, 1, cap - size - 1, f)) > 0) {
        size += n;
        if (cap - size - 1 == 0) {
            cap *= 2;
            buf = realloc(buf, cap);
        }
    }
    if (!buf || ferror(f)) {
        fail(path);
    }
    fclose(f);
    buf[size] = '\0';
    return buf;
}

/* Takes blocks of PROBE_BLOCK until the arena grows, frees them, and
 * returns the growth. */
static size_t growth(void) {
    static void *blocks[PROBE_BLOCKS];
    size_t start = mallinfo2().arena;
    size_t grown = 0;
    int taken = 0;
    while (taken < PROBE_BLOCKS && grown == 0) {
        blocks[taken] = malloc(PROBE_BLOCK);
        if (!blocks[taken]) {
            fail("malloc");
        }
        /* The block is written, as a call writes its blocks. */
        memset(blocks[taken], 1, PROBE_BLOCK);
        taken++;
        grown = mallinfo2().arena - start;
    }
    for (int i = 0; i < taken; i++) {
        free(blocks[i]);
    }
    if (grown == 0) {
        fail("the arena did not grow");
    }
    return grown;
}

/* The excess of a block of `size` bytes: malloc_usable_size less `size`. */
static size_t excess(size_t size) {
    void *p = malloc(size);
    if (!p) {
        fail("malloc");
    }
    size_t usable = malloc_usable_size(p);
    free(p);
    return usable - size;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fail("usage: heap <instance file> <out dir>");
    }
    char *text = slurp(argv[1]);
    printf("before %zu\n", growth());
    printf("excess-before-1m %zu\n", excess((size_t)1 << 20));
    grobx_params p = grobx_params_new(GROBX_GREVLEX, 1, 1);
    grobx_result r;
    int status = grobx_groebner(text, &p, &r);
    printf("status %d\n", status);
    if (r.basis) {
        size_t len = strlen(argv[2]) + sizeof "/basis";
        char *path = malloc(len);
        if (!path) {
            fail("malloc");
        }
        snprintf(path, len, "%s/basis", argv[2]);
        FILE *f = fopen(path, "wb");
        if (!f || fputs(r.basis, f) == EOF || fclose(f) != 0) {
            fail(path);
        }
        free(path);
    }
    grobx_result_free(&r);
    printf("after %zu\n", growth());
    printf("excess-1m %zu\n", excess((size_t)1 << 20));
    printf("excess-64m %zu\n", excess((size_t)64 << 20));
    free(text);
    return 0;
}
