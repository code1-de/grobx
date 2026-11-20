//! grobx-c's integration tests, one test binary (test-build-time.1). Each
//! other `.rs` file of this directory is a module of it, declared below.
//! A test file added later goes here too (test-build-time.6).

#![allow(
    clippy::expect_used,
    clippy::let_underscore_must_use,
    clippy::let_underscore_untyped,
    clippy::panic,
    clippy::unreachable,
    clippy::unwrap_used,
    clippy::wildcard_enum_match_arm,
    reason = "test code, where these lints carry no weight"
)]

// test-build-time.1 design-checked -- the package's integration tests are one test binary, each test file a module of it, not a test binary per file (log 0019-1)
// test-build-time.2 design-checked -- the test files lie in tests/suite/ and cargo discovers the test targets, not autotests = false with the suite named as the one [[test]] target (not measured)

mod abi;
mod c_nul_removal;
mod call_report;
mod calls;
mod checker_points;
mod common;
mod deglex_order;
// c-suite-library-path.7 design-checked -- the item's test file is compiled on Linux alone, whose loader reads LD_LIBRARY_PATH, not on every unix (not measured)
#[cfg(target_os = "linux")]
mod library_path;
mod misuse;
mod q_zero_denominator;
#[path = "../../../tests/common/systems.rs"]
mod systems;
mod unreached_test_paths;
