/* c-interface's test code: what the C interface refuses, and its small
 * functions. Each case is a line on standard output: its name, the status,
 * and the error's message or `-`; the digests, the flags, and the fields of
 * grobx_params_new's result print their values instead.
 */
#include <stdio.h>
#include <string.h>

#include "grobx.h"

static const char *TINY = "x,y\n7\nx^2+y,\nx*y-1\n";

/* Calls grobx_groebner, prints the case, and frees the result. */
static void call(const char *name, const char *text, const grobx_params *p) {
    grobx_result r;
    int status = grobx_groebner(text, p, &r);
    printf("%s %d %s\n", name, status, r.error ? r.error : "-");
    grobx_result_free(&r);
}

static void digest(const char *name, const char *text, int out) {
    char d[65] = "-";
    int status = grobx_digest(text, out ? d : NULL);
    printf("%s %d %s\n", name, status, d);
}

int main(void) {
    grobx_params p = grobx_params_new(GROBX_GREVLEX, 1, 1);
    call("null-text", NULL, &p);
    call("null-params", TINY, NULL);
    printf("null-result %d -\n", grobx_groebner(TINY, &p, NULL));
    grobx_params q = p;
    q.order = 3;
    call("order-3", TINY, &q);
    q.order = GROBX_BLOCK;
    q.nblocks = 1;
    call("null-blocks", TINY, &q);
    q.nblocks = 0;
    call("block-none", TINY, &q);
    static const size_t zero[] = {0};
    q.blocks = zero;
    q.nblocks = 1;
    call("block-0", TINY, &q);
    call("text-not-utf8", "\xff\xfe", &p);
    call("empty-text", "", &p);
    q = p;
    q.checker = 3;
    call("checker-3", TINY, &q);
    q.checker = GROBX_CHECKER_CHEAP;
    q.points = 0;
    call("cheap-0", TINY, &q);
    digest("digest-null-text", NULL, 1);
    digest("digest-null-out", "abc", 0);
    digest("digest-not-utf8", "\xff", 1);
    digest("digest-abc", "abc", 1);

    grobx_cancel_set(NULL);
    grobx_cancel_free(NULL);
    printf("cancel-null %d\n", grobx_cancel_is_set(NULL));
    grobx_cancel *c = grobx_cancel_new();
    int before = grobx_cancel_is_set(c);
    grobx_cancel_set(c);
    printf("cancel %d %d\n", before, grobx_cancel_is_set(c));
    grobx_cancel_free(c);

    grobx_params n = grobx_params_new(GROBX_LEX, 7, 11);
    printf("params %u %d %zu %zu %llu %llu %d %llu %u %u %d\n", n.order, n.blocks == NULL, n.nblocks,
           n.threads, (unsigned long long)n.seed, (unsigned long long)n.deadline, n.cancel == NULL,
           (unsigned long long)n.memory, n.checker, n.points, n.cofactors);
    grobx_params d = grobx_params_default();
    grobx_params e = grobx_params_default();
    printf("default %u %d %zu %zu %d %llu %d %llu %u %u %d\n", d.order, d.blocks == NULL, d.nblocks,
           d.threads, d.seed != e.seed, (unsigned long long)d.deadline, d.cancel == NULL,
           (unsigned long long)d.memory, d.checker, d.points, d.cofactors);
    return 0;
}
