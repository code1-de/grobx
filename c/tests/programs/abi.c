/* c-interface's test code: the header's sizes, offsets, and constants, a
 * line each as `<name> <value>`, which abi.rs compares with the library's.
 */
#include <stddef.h>
#include <stdio.h>

#include "grobx.h"

#define SHOW(name, value) printf("%s %lld\n", name, (long long)(value))
#define FIELD(type, field) SHOW(#type "." #field, offsetof(type, field))

int main(void) {
    SHOW("sizeof grobx_params", sizeof(grobx_params));
    FIELD(grobx_params, order);
    FIELD(grobx_params, blocks);
    FIELD(grobx_params, nblocks);
    FIELD(grobx_params, threads);
    FIELD(grobx_params, seed);
    FIELD(grobx_params, deadline);
    FIELD(grobx_params, cancel);
    FIELD(grobx_params, memory);
    FIELD(grobx_params, checker);
    FIELD(grobx_params, points);
    FIELD(grobx_params, cofactors);
    SHOW("sizeof grobx_result", sizeof(grobx_result));
    FIELD(grobx_result, basis);
    FIELD(grobx_result, cofactors);
    FIELD(grobx_result, stopped);
    FIELD(grobx_result, certified);
    FIELD(grobx_result, flaw);
    FIELD(grobx_result, error);
    SHOW("GROBX_OK", GROBX_OK);
    SHOW("GROBX_SYNTAX", GROBX_SYNTAX);
    SHOW("GROBX_UNKNOWN_VARIABLE", GROBX_UNKNOWN_VARIABLE);
    SHOW("GROBX_REPEATED_VARIABLE", GROBX_REPEATED_VARIABLE);
    SHOW("GROBX_CHARACTERISTIC", GROBX_CHARACTERISTIC);
    SHOW("GROBX_OVERFLOW", GROBX_OVERFLOW);
    SHOW("GROBX_NO_INVERSE", GROBX_NO_INVERSE);
    SHOW("GROBX_ORDER", GROBX_ORDER);
    SHOW("GROBX_MEMORY", GROBX_MEMORY);
    SHOW("GROBX_ERROR", GROBX_ERROR);
    SHOW("GROBX_ARGUMENT", GROBX_ARGUMENT);
    SHOW("GROBX_PANIC", GROBX_PANIC);
    SHOW("GROBX_SIZE", GROBX_SIZE);
    SHOW("GROBX_DROPS", GROBX_DROPS);
    /* GROBX_NONE is UINT64_MAX, which a long long does not hold. */
    printf("GROBX_NONE %llu\n", (unsigned long long)GROBX_NONE);
    SHOW("GROBX_GREVLEX", GROBX_GREVLEX);
    SHOW("GROBX_LEX", GROBX_LEX);
    SHOW("GROBX_BLOCK", GROBX_BLOCK);
    SHOW("GROBX_CHECKER_OFF", GROBX_CHECKER_OFF);
    SHOW("GROBX_CHECKER_EXACT", GROBX_CHECKER_EXACT);
    SHOW("GROBX_CHECKER_CHEAP", GROBX_CHECKER_CHEAP);
    SHOW("GROBX_STOP_NONE", GROBX_STOP_NONE);
    SHOW("GROBX_STOP_DEADLINE", GROBX_STOP_DEADLINE);
    SHOW("GROBX_STOP_CANCEL", GROBX_STOP_CANCEL);
    SHOW("GROBX_STOP_MEMORY", GROBX_STOP_MEMORY);
    SHOW("GROBX_UNCHECKED", GROBX_UNCHECKED);
    SHOW("GROBX_REJECTED", GROBX_REJECTED);
    SHOW("GROBX_CERTIFIED", GROBX_CERTIFIED);
    return 0;
}
