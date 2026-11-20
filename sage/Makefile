# make dist -- builds grobx-sage's sdist and wheel into dist/.
# make test -- installs that wheel and grobx's wheel, built from this tree, under Sage in the image
# SAGE_IMAGE, and runs test_grobx_sage.py against them.

SAGE_IMAGE ?= sagemath/sagemath:10.10
TARGET = ../target
GROBX_WHEEL = grobx-wheel

.PHONY: dist grobx-wheel test

dist:
	rm -rf dist
	uv build --out-dir dist

# Zig against glibc 2.28, as user/pypi-release.py builds the Linux wheel: a build host's glibc may be newer
# than the image's.
grobx-wheel:
	rm -rf $(TARGET)/$(GROBX_WHEEL)
	uvx --from 'maturin>=1.9,<2' maturin build --release -i 3.12 --manifest-path ../python/Cargo.toml \
	  --target x86_64-unknown-linux-gnu --zig --compatibility manylinux_2_28 --out $(TARGET)/$(GROBX_WHEEL)

# The files reach the container as a tar on stdin: its user sage cannot read a bind mount of a home
# directory that is closed to others. The test runs where no grobx_sage.py lies, so it imports the
# wheel's. --no-index and --no-deps keep PyPI's grobx out: the test is of this tree's, whatever version
# the wheel's name carries.
test: dist grobx-wheel
	tar -c dist/*.whl test_grobx_sage.py -C $(TARGET) $(GROBX_WHEEL) | \
	docker run --rm -i --entrypoint bash $(SAGE_IMAGE) -c \
	  'set -e; cd "$$(mktemp -d)"; tar -x; sage -pip install -q --no-index --no-deps $(GROBX_WHEEL)/*.whl dist/*.whl; \
	   sage -python -m pytest -q -p no:cacheprovider test_grobx_sage.py'
