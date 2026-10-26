//! Links the extension module as an interpreter loads it.

// port.9 design -- on macOS the build script leaves the interpreter's symbols to the loader by pyo3-build-config's -undefined dynamic_lookup, which maturin passes as well, not a cargo configuration a build outside maturin must name (not measured)
fn main() {
    pyo3_build_config::add_extension_module_link_args();
}
