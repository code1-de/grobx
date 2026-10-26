//! grobx-checker's integration tests, one test binary (test-build-time.1). Each
//! other `.rs` file of this directory is a module of it, declared below.
//! A test file added later goes here too (test-build-time.6).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]
#![allow(
    clippy::expect_used,
    clippy::let_underscore_must_use,
    clippy::let_underscore_untyped,
    clippy::maybe_infinite_iter,
    clippy::panic,
    clippy::unreachable,
    clippy::unwrap_used,
    clippy::wildcard_enum_match_arm,
    reason = "test code, where these lints carry no weight"
)]

// test-build-time.1 design -- the package's integration tests are one test binary, each test file a module of it, not a test binary per file (log 0019-1)
// test-build-time.2 design -- the test files lie in tests/suite/ and cargo discovers the test targets, not autotests = false with the suite named as the one [[test]] target (not measured)
// test-build-time.4 design -- every test file is a module here, those that run calls with a deadline among them, not kept as test binaries of their own (log 0019-1)

// test-build-time.3 design -- the shared modules stay where they were, declared here by #[path], not moved into tests/suite/ (not measured)
#[path = "../agree/mod.rs"]
mod agree;
#[path = "../common/mod.rs"]
mod common;
#[path = "../orders/mod.rs"]
mod orders;

mod agree_edge;
mod agree_random;
mod characteristic;
mod cheap_batches;
mod cheap_verdicts;
mod edge_gfp;
mod edge_q;
mod grammar;
mod grammar_q;
mod instance_reading;
mod layout;
mod orders_edge;
mod orders_random;
mod orders_verdicts;
mod poll;
mod poll_units;
mod random_gfp;
mod random_q;
mod verdicts;
mod verdicts_q;
