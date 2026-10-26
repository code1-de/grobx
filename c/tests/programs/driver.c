/* c-interface's test code: one call of grobx_groebner from C.
 *
 * driver <instance file> <order> <blocks -|k1,...,km> <threads> <seed>
 *        <deadline|-> <memory|-> <checker -|exact|points> <cofactors 0|1>
 *        <cancel -|never|set|ms> <out dir>
 *
 * The order is its GROBX_* value. The text is the file's bytes and a NUL. `never` passes a cancel flag
 * that nothing raises, `set` raises it before the call, and a number of
 * milliseconds raises it from a thread of its own after that time. Each non-NULL text of the result goes to the
 * file of its name in the out dir, and standard output takes the status,
 * stopped, certified, the basis text's digest, and the version, a line
 * each. The result is freed twice. Exits 0, or 99 where the arguments or
 * the file cannot be read, an allocation fails, a text cannot be written
 * to the out dir, or pthread_create or grobx_digest fails.
 */
#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "grobx.h"

static void fail(const char *what) {
    fprintf(stderr, "driver: %s\n", what);
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

static uint64_t number(const char *s) {
    char *end;
    errno = 0;
    unsigned long long v = strtoull(s, &end, 10);
    if (errno || *end || !*s) {
        fail(s);
    }
    return (uint64_t)v;
}

/* The comma-separated sizes in `s`, their count in `*n`; NULL for `-`. */
static size_t *sizes(const char *s, size_t *n) {
    *n = 0;
    if (!strcmp(s, "-")) {
        return NULL;
    }
    size_t cap = 1;
    for (const char *c = s; *c; c++) {
        cap += *c == ',';
    }
    size_t *out = malloc(cap * sizeof *out);
    char *copy = strdup(s);
    if (!out || !copy) {
        fail("malloc");
    }
    char *save;
    for (char *k = strtok_r(copy, ",", &save); k; k = strtok_r(NULL, ",", &save)) {
        out[(*n)++] = (size_t)number(k);
    }
    free(copy);
    return out;
}

static void put(const char *dir, const char *name, const char *text) {
    if (!text) {
        return;
    }
    char path[4096];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *f = fopen(path, "wb");
    if (!f || fputs(text, f) == EOF || fclose(f)) {
        fail(path);
    }
}

struct later {
    grobx_cancel *cancel;
    uint64_t ms;
};

static void *raise_later(void *arg) {
    struct later *l = arg;
    struct timespec t = {(time_t)(l->ms / 1000), (long)(l->ms % 1000) * 1000000L};
    nanosleep(&t, NULL);
    grobx_cancel_set(l->cancel);
    return NULL;
}

int main(int argc, char **argv) {
    if (argc != 12) {
        fail("eleven arguments");
    }
    char *text = slurp(argv[1]);
    grobx_params p = grobx_params_new((uint32_t)number(argv[2]), (size_t)number(argv[4]), number(argv[5]));
    size_t *blocks = sizes(argv[3], &p.nblocks);
    p.blocks = blocks;
    if (strcmp(argv[6], "-")) {
        p.deadline = number(argv[6]);
    }
    if (strcmp(argv[7], "-")) {
        p.memory = number(argv[7]);
    }
    if (!strcmp(argv[8], "exact")) {
        p.checker = GROBX_CHECKER_EXACT;
    } else if (strcmp(argv[8], "-")) {
        p.checker = GROBX_CHECKER_CHEAP;
        p.points = (uint32_t)number(argv[8]);
    }
    p.cofactors = !strcmp(argv[9], "1");
    pthread_t thread;
    struct later later = {NULL, 0};
    int threaded = 0;
    if (strcmp(argv[10], "-")) {
        p.cancel = grobx_cancel_new();
        if (!strcmp(argv[10], "set")) {
            grobx_cancel_set(p.cancel);
        } else if (strcmp(argv[10], "never")) {
            later.cancel = p.cancel;
            later.ms = number(argv[10]);
            if (pthread_create(&thread, NULL, raise_later, &later)) {
                fail("pthread_create");
            }
            threaded = 1;
        }
    }
    grobx_result r;
    int status = grobx_groebner(text, &p, &r);
    if (threaded) {
        pthread_join(thread, NULL);
    }
    char digest[65] = "-";
    if (r.basis && grobx_digest(r.basis, digest) != GROBX_OK) {
        fail("grobx_digest");
    }
    put(argv[11], "basis", r.basis);
    put(argv[11], "cofactors", r.cofactors);
    put(argv[11], "flaw", r.flaw);
    put(argv[11], "error", r.error);
    printf("status %d\nstopped %d\ncertified %d\ndigest %s\nversion %s\n", status, r.stopped,
           r.certified, digest, grobx_version());
    grobx_result_free(&r);
    grobx_result_free(&r);
    grobx_result_free(NULL);
    grobx_cancel_free(p.cancel);
    free(blocks);
    free(text);
    return 0;
}
