# Makefile -- `make test` runs test.mpl under Maple on the program grobx
# that `cargo build --release -p grobx-cli` in grobx/ writes.

MAPLE ?= $(HOME)/maple2026
GROBX ?= $(abspath ../target/release/grobx)
export MAPLE GROBX

test:
	./maple-run.sh test.mpl test.log && grep -qx 'test.mpl: all passed' test.log

clean:
	rm -f test.log

.PHONY: test clean
