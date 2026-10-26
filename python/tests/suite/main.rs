//! grobx-python's integration tests, one test binary (test-build-time.1). Each
//! other `.rs` file of this directory is a module of it, declared below.
//! A test file added later goes here too (test-build-time.6).

#![allow(
    clippy::expect_used,
    clippy::format_push_string,
    clippy::panic,
    clippy::unreachable,
    clippy::unwrap_used,
    reason = "test code, where these lints carry no weight"
)]

// test-build-time.1 design -- the package's integration tests are one test binary, each test file a module of it, not a test binary per file (log 0019-1)
// test-build-time.2 design -- the test files lie in tests/suite/ and cargo discovers the test targets, not autotests = false with the suite named as the one [[test]] target (not measured)
// test-build-time.4 design -- every test file is a module here, those that run calls with a deadline among them, not kept as test binaries of their own (log 0019-1)

mod call_report;
mod checker_points;
mod large_primes;
mod layout;
mod orders;
mod q;
mod q_zero_denominator;
mod small_block_time;
#[path = "../../../tests/common/systems.rs"]
mod systems;
mod text_copy_peak;
mod wrapper;
