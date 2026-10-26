#!/bin/sh
# maple-run.sh SCRIPT LOG -- runs SCRIPT under $MAPLE/bin/maple -q, writes
# its output to LOG and to stdout, and exits 0 only where the run succeeded.
#
# Maple's command line goes on after an error and exits 0, even under -e 2,
# and a syntax error in the script itself prints no "Error," line. A run
# fails on its status, on a line that starts with "Error,", and on "syntax
# error".
set -u
if [ $# -ne 2 ]; then
    echo "usage: maple-run.sh SCRIPT LOG" >&2
    exit 2
fi
: "${MAPLE:?MAPLE must name the Maple installation}"
"$MAPLE/bin/maple" -q "$1" > "$2" 2>&1
status=$?
cat "$2"
[ "$status" -eq 0 ] && ! grep -q -e '^Error,' -e 'syntax error' "$2"
