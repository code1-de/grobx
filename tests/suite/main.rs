//! grobx's integration tests, one test binary (test-build-time.1). Each
//! other `.rs` file of this directory is a module of it, declared below.
//! A test file added later goes here too (test-build-time.6).

#![allow(
    clippy::cast_possible_truncation,
    reason = "test code: its casts narrow small indices, counts, and draws"
)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::let_underscore_must_use,
    clippy::let_underscore_untyped,
    clippy::many_single_char_names,
    clippy::needless_pass_by_value,
    clippy::panic,
    clippy::too_many_lines,
    clippy::unreachable,
    clippy::unwrap_used,
    clippy::wildcard_enum_match_arm,
    reason = "test code, where these lints carry no weight"
)]

// test-build-time.1 design -- the package's integration tests are one test binary, each test file a module of it, not a test binary per file (log 0019-1)
// test-build-time.2 design -- the test files lie in tests/suite/ and cargo discovers the test targets, not autotests = false with the suite named as the one [[test]] target (not measured)
// test-build-time.4 design -- every test file is a module here, those that run calls with a deadline among them, not kept as test binaries of their own (log 0019-1)

// small-block-time.4 design -- the integration suite sets the crate's allocator as its global allocator, not the crate setting it for every program that links it (not measured)
#[global_allocator]
static GLOBAL: grobx::Allocator = grobx::Allocator;

// test-build-time.3 design -- the shared modules stay where they were, declared here by #[path], not moved into tests/suite/ (not measured)
#[path = "../common/mod.rs"]
mod common;
#[path = "../common/heap.rs"]
mod heap;
#[path = "../large/mod.rs"]
mod large;
#[path = "../orders/mod.rs"]
mod orders;
#[path = "../q/mod.rs"]
mod q;
#[path = "../report/mod.rs"]
mod report;

mod accept;
mod accept_tuning;
mod api;
mod budget;
mod call_report;
mod call_report_q;
mod call_report_tuning;
mod checker_points_abort;
mod checker_stop_latency;
mod cofactor_stop_latency;
mod cofactors;
mod core_field;
mod core_monomial;
mod core_text;
mod cyclic_gap;
mod divisibility_time;
mod early_handover_time;
mod engine;
mod fglm_edge;
mod fglm_random;
mod fglm_replay;
mod fglm_tuning;
mod formats;
mod handover_time;
mod input_redundancy;
mod instance_grammar;
mod large_field;
mod large_primes;
mod layout;
mod learn;
mod many_variables_time;
mod mask_scan_time;
mod matrix_build_time;
mod matrix_slack;
mod matrix_sort_time;
mod matrix_stop_latency;
mod monomial_time;
mod orders_edge;
mod orders_names;
mod orders_random;
mod orders_replay;
mod orders_tuning;
mod outgrown_round_size;
mod page_fault_time;
mod pair_work_handover;
mod pair_work_ratio;
mod plain;
mod plain_reduction_time;
mod plain_update_time;
mod plain_zero_time;
mod product_lookup_time;
mod q_api;
mod q_basis_text_time;
mod q_crt;
mod q_edge;
mod q_later_runs_time;
mod q_learn_time;
mod q_lift_rest_time;
mod q_modulus_time;
mod q_random;
mod q_rational;
mod q_reconstruct;
mod q_replay;
mod q_tuning;
mod q_zero_denominator;
mod reduction_time;
mod replay;
mod result_time;
mod round_memory;
mod row_scan_time;
mod row_traffic_time;
mod rule_breaches;
mod rule_breaches_allows;
mod rule_breaches_engine;
mod rule_breaches_names;
mod rule_breaches_suite;
mod run_generators_time;
mod signature_pairs;
mod signature_zero_time;
mod signatures;
mod small_block_time;
mod sources_scan;
mod table_read_time;
mod test_mq_draws;
mod test_mq_suite;
mod test_mq_systems;
mod text_stop_latency;
mod text_time;
mod trace;
mod unsolved;
mod zero_rows_time;
