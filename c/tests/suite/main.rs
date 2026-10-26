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

// test-build-time.1 design -- the package's integration tests are one test binary, each test file a module of it, not a test binary per file (log 0019-1)
// test-build-time.2 design -- the test files lie in tests/suite/ and cargo discovers the test targets, not autotests = false with the suite named as the one [[test]] target (not measured)

mod abi;
mod call_report;
mod calls;
mod checker_points;
mod common;
#[path = "../../../tests/common/heap.rs"]
mod heap;
mod misuse;
mod page_fault_time;
mod q_zero_denominator;
mod small_block_time;
#[path = "../../../tests/common/systems.rs"]
mod systems;
