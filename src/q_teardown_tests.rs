//! q-teardown-stop-latency's test code, the item's: over ℚ the texts of calls with and without
//! cofactors are those the code wrote before the item, the checker certifies them, a stop forced
//! at each of the lift's sites with cofactors stops the call there, and a lift that drops its
//! record certifies its cofactors (`iterations/0140-q-teardown-stop-latency/plan.md`, _Test code_,
//! `q_teardown_tests.rs`).

use std::num::NonZeroUsize;

use grobx_checker::{Mode, Order, check};

use crate::budget::Site;
use crate::call_stop_sites_tests::{Case, stops_at};
use crate::lift::{Report, lift_text};
use crate::shared_tests::{L1, U1, U2};
use crate::split_piece_time_tests::EDGE;
use crate::systems::{dense, random_square, random_system};
use crate::{MonomialOrder, Outcome, Params, Threads, digest};

/// The seed of every call.
const SEED: u64 = 1;

/// The digits of an integer past 2^128, which some basis text and some cofactor text of the cases
/// hold.
const WIDE_DIGITS: usize = 40;

/// Per case of `cases`, in order: its name, its order, and the digests of its basis text and of
/// its cofactor text, with cofactors, seed 1, as the code wrote them at `0.114.0` before the item
/// (`iterations/0140-q-teardown-stop-latency/instrument/golden.txt`).
const DIGESTS: [(&str, &str, &str, &str); 40] = [
    (
        "random_system(0, 0)",
        "grevlex",
        "1b49f75bd2b39c6732cc6b8516027d056a44632011c8ba3c836b1c5cce261a03",
        "820140f18703f7f5a6e9ca54290885d37304554c69550b8a8cfef3f28a93125b",
    ),
    (
        "random_system(0, 0)",
        "lex",
        "4d0734c0b3657a9638672ed30c5d3d4e4b2e0f69de7b6f08a206ae844e9a0fc1",
        "820140f18703f7f5a6e9ca54290885d37304554c69550b8a8cfef3f28a93125b",
    ),
    (
        "random_square(0, 0)",
        "grevlex",
        "ec11b50a575c6ec7bd87a059cea6d0cbdfc8d32537ea6df17989adb7b7009547",
        "a743fc7d9cd7eea7af5f783b633474facf37bc6ab6fa5393ef3c63c26ec786b0",
    ),
    (
        "random_square(0, 0)",
        "lex",
        "b735d78b8c4c5e342b2343112ee997a2742595cb43b41b0ae3300f7ad9698885",
        "e504cafa9c6b80aec058bd695ff0f25eab0cf187808aef49270b9c8d13e9ea92",
    ),
    (
        "random_system(0, 1)",
        "grevlex",
        "aa68c47605a55b90b2157ead4a3416c07935cf50a3201111038de23742e769a4",
        "1ae009d66ea7e3dbc7dd49cdd54181753cf5d7cdf6e0aa71f49a60149405c90d",
    ),
    (
        "random_system(0, 1)",
        "lex",
        "aa68c47605a55b90b2157ead4a3416c07935cf50a3201111038de23742e769a4",
        "1fb5be6db26653db64ffd283cecfb63d69dfef5dd6ac3add96f39204ac3bb2b3",
    ),
    (
        "random_square(0, 1)",
        "grevlex",
        "1e87e525b95b3a0adf2e133cddb5214b9856ee094484603eaad78453135bcc7f",
        "152dae22ff9ef22068808d59e82939cedfd057eb0d870f0627c2c121766ed333",
    ),
    (
        "random_square(0, 1)",
        "lex",
        "dd586453ba9f818ceb5164a560fc46bb9eb6793a191de4a770bb56992c6b6dee",
        "6b5bcbed4d3b981a37f395136c47cbe038ea67aa2d833875a9b672b7b2eda388",
    ),
    (
        "random_system(0, 2)",
        "grevlex",
        "aa68c47605a55b90b2157ead4a3416c07935cf50a3201111038de23742e769a4",
        "c21f53ff501a81cfe57be705d4a908a388c0efccdbb04d7cf24eddf99dafb103",
    ),
    (
        "random_system(0, 2)",
        "lex",
        "aa68c47605a55b90b2157ead4a3416c07935cf50a3201111038de23742e769a4",
        "c21f53ff501a81cfe57be705d4a908a388c0efccdbb04d7cf24eddf99dafb103",
    ),
    (
        "random_square(0, 2)",
        "grevlex",
        "1b26a44b3ce7bb0aaef88cc66c36dd685e93d18934626b26e0e8c3bb50670e59",
        "97740541ba62c65454ba27381f0e3538ca58afa24fe9a0e7b4c46a1d12d2ab7f",
    ),
    (
        "random_square(0, 2)",
        "lex",
        "fb4afe925df126eaf2c6d272f42ce4b3bc26e4e499a7e12a16428a367eeae72a",
        "14f6ce184bbb80d1ae276e3b1d5010f64ec38fe0a2ff6342a4c7dc3e5476858d",
    ),
    (
        "random_system(0, 3)",
        "grevlex",
        "ae490a59d6fabb360cf40a5f1264af0b8b2d594852fc84abbd4d01e920dcacdb",
        "6ce311005010aeb2952611dd7ce84280950b0c6031e9e7e57ca646a20a41c0d2",
    ),
    (
        "random_system(0, 3)",
        "lex",
        "ae490a59d6fabb360cf40a5f1264af0b8b2d594852fc84abbd4d01e920dcacdb",
        "6ce311005010aeb2952611dd7ce84280950b0c6031e9e7e57ca646a20a41c0d2",
    ),
    (
        "random_square(0, 3)",
        "grevlex",
        "a353566918389587857589d21aa56cdeae9f833c13e59a7b677c647942aee08b",
        "03a095713b62b89a347558415ecaa8a95f3f0eac6d1903d155562f8b16dbf640",
    ),
    (
        "random_square(0, 3)",
        "lex",
        "2658041645537e2f3f1699ef51d8acd62d400c8fe3cdd977f3ff3ba501692a9a",
        "983a8ee0f7dbc10931928096ef8e03da22213eb90889cdbeb163afa338ad6f32",
    ),
    (
        "random_system(0, 4)",
        "grevlex",
        "aa68c47605a55b90b2157ead4a3416c07935cf50a3201111038de23742e769a4",
        "6d7488520a1b953f37e1f3659dca20296ae7bbb1d77781b798e159883b53f791",
    ),
    (
        "random_system(0, 4)",
        "lex",
        "aa68c47605a55b90b2157ead4a3416c07935cf50a3201111038de23742e769a4",
        "6d7488520a1b953f37e1f3659dca20296ae7bbb1d77781b798e159883b53f791",
    ),
    (
        "random_square(0, 4)",
        "grevlex",
        "a363a35a594d75007b8e06c737059feb20381ed19e11eb19887369e78dea3056",
        "03254c23d7fe9c86e9cda6e440add6bffa77ce4545df4fedb1df0f155c3258c9",
    ),
    (
        "random_square(0, 4)",
        "lex",
        "e4821f72301156cc180b5f3485d88c17d986451142f1992878c15ff6d8190f47",
        "a08922ff198c0e472ed4e168eaa43e371ea65fe417d90f9d038f3fdf085ac74d",
    ),
    (
        "random_system(0, 5)",
        "grevlex",
        "aa68c47605a55b90b2157ead4a3416c07935cf50a3201111038de23742e769a4",
        "84f045ced78c90d40a1d966380970e4e40d2dfee7cd1a8a588fdc85161cbe334",
    ),
    (
        "random_system(0, 5)",
        "lex",
        "aa68c47605a55b90b2157ead4a3416c07935cf50a3201111038de23742e769a4",
        "6589ecae45526ad967ed663fedf7fa584ff118222a82000a34aa767c273c1579",
    ),
    (
        "random_square(0, 5)",
        "grevlex",
        "87d6b089cb00bf4fb0cdf48b58b82918b7d90cf1a4dc682d93110d9b6cf51a3f",
        "64b0abe482f1015f9f8c399e683b0d8fd5c342b6479393be6aeccac1ddb5f2d8",
    ),
    (
        "random_square(0, 5)",
        "lex",
        "87f247759f065a516f1ccc53ea30ade5c96151a9d635a4c7da634318e3c14032",
        "53388131788f76111dea113def9961f550bf7cf6ea1b5e2dbffd46e662ed8609",
    ),
    (
        "dense(3, 3, 0, 0)",
        "grevlex",
        "66a86813345767d3a6a966ec01d062cd508dd0663e75960ed6438b63c7e07a40",
        "565500bc8da23a1e8f900a736fead47720f0e533d35493d0629543f860cd7810",
    ),
    (
        "dense(3, 3, 0, 1)",
        "grevlex",
        "7f43bd040272665dda99ea15982d6110b1d57afd0ead2dd533d3d9891ea6a7b8",
        "aa45573f8fa429aea91a6d7d67446b737659d28cceb5202dd9bc57b11761ae45",
    ),
    (
        "EDGE[0]",
        "grevlex",
        "59353431d7a2d9d652f065b8767fc7d62ccfa7081e0ec13a2bbe939b7bbdee3c",
        "59353431d7a2d9d652f065b8767fc7d62ccfa7081e0ec13a2bbe939b7bbdee3c",
    ),
    (
        "EDGE[0]",
        "lex",
        "59353431d7a2d9d652f065b8767fc7d62ccfa7081e0ec13a2bbe939b7bbdee3c",
        "59353431d7a2d9d652f065b8767fc7d62ccfa7081e0ec13a2bbe939b7bbdee3c",
    ),
    (
        "EDGE[1]",
        "grevlex",
        "59353431d7a2d9d652f065b8767fc7d62ccfa7081e0ec13a2bbe939b7bbdee3c",
        "59353431d7a2d9d652f065b8767fc7d62ccfa7081e0ec13a2bbe939b7bbdee3c",
    ),
    (
        "EDGE[1]",
        "lex",
        "59353431d7a2d9d652f065b8767fc7d62ccfa7081e0ec13a2bbe939b7bbdee3c",
        "59353431d7a2d9d652f065b8767fc7d62ccfa7081e0ec13a2bbe939b7bbdee3c",
    ),
    (
        "EDGE[2]",
        "grevlex",
        "59353431d7a2d9d652f065b8767fc7d62ccfa7081e0ec13a2bbe939b7bbdee3c",
        "59353431d7a2d9d652f065b8767fc7d62ccfa7081e0ec13a2bbe939b7bbdee3c",
    ),
    (
        "EDGE[2]",
        "lex",
        "59353431d7a2d9d652f065b8767fc7d62ccfa7081e0ec13a2bbe939b7bbdee3c",
        "59353431d7a2d9d652f065b8767fc7d62ccfa7081e0ec13a2bbe939b7bbdee3c",
    ),
    (
        "EDGE[3]",
        "grevlex",
        "aa68c47605a55b90b2157ead4a3416c07935cf50a3201111038de23742e769a4",
        "186570e27bf4aaf6aad8a725b7292f13176d1322e867d782ed26a1905f6fb983",
    ),
    (
        "EDGE[3]",
        "lex",
        "aa68c47605a55b90b2157ead4a3416c07935cf50a3201111038de23742e769a4",
        "186570e27bf4aaf6aad8a725b7292f13176d1322e867d782ed26a1905f6fb983",
    ),
    (
        "EDGE[4]",
        "grevlex",
        "61714b3102a316e866e0d2b43949a04bb3c2490475379d1f1f8c3a478ee28b07",
        "6c4b825f6026762125672a2201b9700a076992101a144e1d7e0b454be56c2871",
    ),
    (
        "EDGE[4]",
        "lex",
        "dc7333b87ac2f781e2838a6a770700fe04c5ab6c801455407ab70640f2096409",
        "9bec579102c1d8f20473b21793d7e1511e49e1a5b940c7137224faff5e9304c3",
    ),
    (
        "EDGE[5]",
        "grevlex",
        "99ccd2f74da83c82fb8f927ac142f8864ba1a52808d96effa588b1f4989ac1fe",
        "d50ea46f1d15fc418ce96e2bb709a74ba0303dd7b1069911e5bb2260d765307c",
    ),
    (
        "EDGE[5]",
        "lex",
        "d8cb9ab91bd703d0da2900d7dba55e1dfec484fdd0027d728ee123406b6df7f9",
        "74c89f976f907d5810fcc93f4a236415ead0bdea815f09657f98971b00790a74",
    ),
    (
        "EDGE[6]",
        "grevlex",
        "357c0d38f5965c13302fb46fd93dad887672e92090e266e4d9438932f9048ef0",
        "f81cd1a3ace0c762a4e9086adee783b44b6186c3ffa77b6009c16be3603a16c3",
    ),
    (
        "EDGE[6]",
        "lex",
        "04ad472124d453ba28ad1d2f44203a837110906d491e7d8c0c71963b0a51d1ad",
        "05ad88ffb19542950f48571e0d18cd7113dd5586c8e5d2cb4fdbdae93b2b628a",
    ),
];

/// The cases, each its name, its instance text, and its order's name: over ℚ, random systems and
/// random square systems, seeds 0 to 5, and the edge cases, in grevlex and lex, and dense cubics on
/// three variables, seeds 0 and 1, in grevlex, in the order of `DIGESTS`.
fn cases() -> Vec<(String, String, &'static str)> {
    let mut systems = Vec::new();
    for seed in 0..6 {
        systems.push((
            format!("random_system(0, {seed})"),
            random_system(0, seed),
            true,
        ));
        systems.push((
            format!("random_square(0, {seed})"),
            random_square(0, seed),
            true,
        ));
    }
    for seed in 0..2 {
        systems.push((
            format!("dense(3, 3, 0, {seed})"),
            dense(3, 3, 0, seed),
            false,
        ));
    }
    for (i, gens) in EDGE.iter().enumerate() {
        systems.push((format!("EDGE[{i}]"), format!("x,y,z\n0\n{gens}\n"), true));
    }
    let mut out = Vec::new();
    for (name, text, lex) in systems {
        out.push((name.clone(), text.clone(), "grevlex"));
        if lex {
            out.push((name, text, "lex"));
        }
    }
    out
}

/// The parameters of a case's call in the order named, with cofactors where `cofactors`.
fn params(order: &str, cofactors: bool) -> Params {
    let order: MonomialOrder = order.parse().expect("a name of the family");
    let mut params = Params::new(order, Threads::Count(NonZeroUsize::new(1).unwrap()), SEED);
    params.cofactors = cofactors;
    params
}

/// The outcome of `text`'s call in the order named, with cofactors where `cofactors`, which must
/// hold a basis.
fn outcome(name: &str, text: &str, order: &str, cofactors: bool) -> Outcome {
    let outcome = crate::groebner(text, &params(order, cofactors))
        .unwrap_or_else(|e| panic!("{name} in {order}: the call failed: {e}"));
    assert!(
        outcome.basis.is_some() && outcome.stopped.is_none(),
        "{name} in {order}: no basis: {outcome:?}"
    );
    outcome
}

/// The most decimal digits of an integer in `text`.
fn longest_integer(text: &str) -> usize {
    text.split(|c: char| !c.is_ascii_digit())
        .map(str::len)
        .max()
        .unwrap_or(0)
}

/// Whether the exact checker certifies `outcome` of `text` in grevlex: Err with why not.
fn certified(text: &str, outcome: &Outcome) -> Result<(), String> {
    let (Some(basis), Some(cofactors)) = (outcome.basis.as_deref(), outcome.cofactors.as_deref())
    else {
        return Err(format!("no basis or no cofactors: {outcome:?}"));
    };
    let v = check(text, basis, cofactors, &Order::Grevlex, Mode::Exact)
        .map_err(|e| format!("the checker cannot read the result: {e:?}"))?;
    if v.groebner.is_ok() && v.reduced.is_ok() {
        Ok(())
    } else {
        Err(format!(
            "the checker rejects the result: {v:?}\n{basis}\n{cofactors}"
        ))
    }
}

/// Test 1: with cofactors each case's basis text and cofactor text are those the code wrote
/// before the item, byte for byte by their digests, and without cofactors its basis text is the
/// same; some basis text and some cofactor text hold an integer past 2^128
/// (q-teardown-stop-latency.1, .8, .9).
#[test]
#[cfg_attr(
    not(feature = "cofactors"),
    ignore = "the build leaves out the cofactors"
)]
fn texts_unchanged() {
    let cases = cases();
    assert_eq!(cases.len(), DIGESTS.len(), "a digest per case");
    let (mut wide_basis, mut wide_cofactors) = (false, false);
    for ((name, text, order), (want_name, want_order, basis_digest, cofactor_digest)) in
        cases.iter().zip(DIGESTS)
    {
        assert_eq!(
            (name.as_str(), *order),
            (want_name, want_order),
            "the cases' order"
        );
        let with = outcome(name, text, order, true);
        let basis = with.basis.as_deref().unwrap_or_default();
        let cofactors = with
            .cofactors
            .as_deref()
            .unwrap_or_else(|| panic!("{name} in {order}: no cofactor text"));
        assert_eq!(
            digest(basis),
            basis_digest,
            "{name} in {order}: the basis text\n{basis}"
        );
        assert_eq!(
            digest(cofactors),
            cofactor_digest,
            "{name} in {order}: the cofactor text\n{cofactors}"
        );
        let without = outcome(name, text, order, false);
        assert_eq!(
            without.basis.as_deref(),
            Some(basis),
            "{name} in {order}: the basis text without cofactors"
        );
        wide_basis |= longest_integer(basis) >= WIDE_DIGITS;
        wide_cofactors |= longest_integer(cofactors) >= WIDE_DIGITS;
    }
    assert!(
        wide_basis && wide_cofactors,
        "an integer of {WIDE_DIGITS} digits in a basis text {wide_basis}, in a cofactor text {wide_cofactors}"
    );
}

/// Test 2: the exact checker certifies every case's call with cofactors in grevlex.
#[test]
#[cfg_attr(
    not(feature = "cofactors"),
    ignore = "the build leaves out the cofactors"
)]
fn the_checker_certifies() {
    let mut failures = Vec::new();
    for (name, text, order) in cases().iter().filter(|(_, _, order)| *order == "grevlex") {
        if let Err(e) = certified(text, &outcome(name, text, order, true)) {
            failures.push(format!("{name}: {e}"));
        }
    }
    let joined = failures.join("\n");
    assert!(failures.is_empty(), "{joined}");
}

/// Test 3: over ℚ with cofactors, each of the lift's sites stops a call from inside it, after
/// which the lifting is dropped, the call's result unchanged with every spend checking
/// (q-teardown-stop-latency.4, .5).
#[test]
#[cfg_attr(
    not(feature = "cofactors"),
    ignore = "the build leaves out the cofactors"
)]
fn stops_inside_the_lift_with_cofactors() {
    let mut failures = Vec::new();
    for site in [
        Site::Text,
        Site::Finish,
        Site::Verify,
        Site::Merge,
        Site::Reconstruct,
        Site::Cofactors,
    ] {
        let case = Case::new(random_system(0, 11), "grevlex", true);
        if let Err(e) = stops_at(site, &case) {
            failures.push(e);
        }
    }
    let joined = failures.join("\n");
    assert!(failures.is_empty(), "{joined}");
}

/// Test 4: with cofactors, lifts whose forced primes make the lift learn afresh or a replay
/// misfit, which drops the record, and the lifting with it, return `lift_tests`' bases, and the
/// checker certifies their cofactors (q-lift.19, .20).
#[test]
#[cfg_attr(
    not(feature = "cofactors"),
    ignore = "the build leaves out the cofactors"
)]
fn a_dropped_lifting_leaves_the_result() {
    let n = U1 * U2;
    let square = u128::from(U1) * u128::from(U1);
    let cases: [(String, Vec<u64>, String); 2] = [
        (
            format!("x,y\n0\n{U1}*x+y,\nx^2-1\n"),
            vec![U1],
            format!("[{U1}*x+y,\ny^2-{square}]:\n"),
        ),
        (
            format!("x,y\n0\nx+{n},\nx*y\n"),
            vec![L1, U1],
            format!("[y,\nx+{n}]:\n"),
        ),
    ];
    for (text, forced, want) in cases {
        let mut report = Report::default();
        let got = lift_text(&text, &params("grevlex", true), forced.clone(), &mut report)
            .unwrap_or_else(|e| panic!("{forced:?}: the lift failed: {e:?}\non\n{text}"));
        assert_eq!(
            got.basis.as_deref(),
            Some(want.as_str()),
            "{forced:?}: {report:?}"
        );
        assert!(
            report.learned.len() > 1 || !report.misfit.is_empty(),
            "{forced:?}: no record dropped and no misfit: {report:?}"
        );
        if let Err(e) = certified(&text, &got) {
            panic!("{forced:?}: {e}");
        }
    }
}
