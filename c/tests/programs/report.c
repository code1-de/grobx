/* call-report's test code through the C interface: the header's sizes and
 * the offsets of the fields `report`, a line each as `<name> <value>`;
 * then calls on the instance in the file argv[1], and the error call on a
 * text whose exponent overflows, each a line on standard output, the report
 * text written to the file argv[2].
 *
 *   default <report>               grobx_params_new's report field
 *   off <status> <report null>     a call without the report
 *   on <status> <basis null> <report null> <basis bytes>
 *   freed <basis null> <report null>
 *   error <status> <report null>   a call that fails, with the report
 */
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "grobx.h"

#define SHOW(name, value) printf("%s %lld\n", name, (long long)(value))

/* The whole file `path`, NUL-terminated, or NULL. */
static char *slurp(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        return NULL;
    }
    size_t cap = 1 << 16, len = 0;
    char *buf = malloc(cap);
    size_t n;
    while (buf && (n = fread(buf + len, 1, cap - len - 1, f)) > 0) {
        len += n;
        if (cap - len - 1 == 0) {
            cap *= 2;
            buf = realloc(buf, cap);
        }
    }
    fclose(f);
    if (buf) {
        buf[len] = '\0';
    }
    return buf;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: report <instance> <report file>\n");
        return 2;
    }
    SHOW("sizeof grobx_params", sizeof(grobx_params));
    SHOW("grobx_params.report", offsetof(grobx_params, report));
    SHOW("sizeof grobx_result", sizeof(grobx_result));
    SHOW("grobx_result.report", offsetof(grobx_result, report));

    char *text = slurp(argv[1]);
    if (!text) {
        fprintf(stderr, "%s: cannot read\n", argv[1]);
        return 2;
    }
    grobx_params p = grobx_params_new(GROBX_GREVLEX, 1, 11);
    printf("default %d\n", p.report ? 1 : 0);

    grobx_result r;
    int status = grobx_groebner(text, &p, &r);
    printf("off %d %d\n", status, r.report == NULL);
    grobx_result_free(&r);

    p.report = true;
    status = grobx_groebner(text, &p, &r);
    printf("on %d %d %d %zu\n", status, r.basis == NULL, r.report == NULL,
           r.basis ? strlen(r.basis) : (size_t)0);
    if (r.report) {
        FILE *out = fopen(argv[2], "wb");
        if (!out || fputs(r.report, out) == EOF || fclose(out) != 0) {
            fprintf(stderr, "%s: cannot write\n", argv[2]);
            return 2;
        }
    }
    grobx_result_free(&r);
    printf("freed %d %d\n", r.basis == NULL, r.report == NULL);

    status = grobx_groebner("x\n7\nx^2147483648\n", &p, &r);
    printf("error %d %d\n", status, r.report == NULL);
    grobx_result_free(&r);
    free(text);
    return 0;
}
