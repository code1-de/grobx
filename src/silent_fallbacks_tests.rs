//! silent-fallbacks, the crate's test code: a pair sugar past `u32::MAX`
//! is an overflow of the pair update, which interns no lcm of it
//! (`iterations/0094-silent-fallbacks/plan.md`, _Test code_, 4).

use crate::divisibility_tests::Engine;
use crate::monomial::Dense16;
use crate::order::{Block, Grevlex, Lex, Order};
use crate::table::Limit;

/// The leading monomial of the first element, x², and of the second,
/// x·y, whose pair has the lcm x²·y of degree 3.
const FIRST: [u32; 2] = [2, 0];
const SECOND: [u32; 2] = [1, 1];

/// The second element's sugar, its leading monomial's degree.
const SECOND_SUGAR: u32 = 2;

// hygiene.15 design-checked -- the degree of this pair's lcm takes a name of its own beside `cooper_rest_gap_tests.rs`' `LCM_DEGREE`, the degree of another system's lcms, whose tokens agree with it, not one constant for both (not measured)
/// The degree of the pair's lcm, x²·y.
const PAIR_LCM_DEGREE: u32 = 3;

/// x² with sugar `sugar`, then x·y, in `order`: the update's result for
/// x·y, the ids the table gained in it, and the sugars of the pairs.
fn update<O: Order<Dense16>>(order: &O, sugar: u32) -> (Result<(), Limit>, usize, Vec<u32>) {
    let mut e: Engine<Dense16> = Engine::new(2, 0x5f5f);
    e.enter(order, &FIRST, sugar).0.unwrap();
    let (got, gained) = e.enter(order, &SECOND, SECOND_SUGAR);
    let sugars = e.written().iter().map(|p| p.sugar).collect();
    (got, gained.len(), sugars)
}

/// In an order that is not graded, the pair's sugar is x²'s sugar less
/// its degree, 2, plus the lcm's degree, 3: `u32::MAX` from a sugar of
/// `u32::MAX` − 1, and past it from `u32::MAX`.
fn check_sugars<O: Order<Dense16>>(name: &str, order: &O) {
    let (got, _, sugars) = update(order, u32::MAX - 1);
    assert_eq!(got, Ok(()), "{name}: a sugar of u32::MAX");
    assert_eq!(sugars, vec![u32::MAX], "{name}");
    let (got, gained, _) = update(order, u32::MAX);
    assert_eq!(got, Err(Limit::Overflow), "{name}: a sugar past u32::MAX");
    assert_eq!(gained, 0, "{name}: the update interned the pair's lcm");
}

/// Test 4: silent-fallbacks.8, in lex and `block:1`; in grevlex the pair's
/// sugar is its degree plus one, whatever x²'s fall (cooper-rest-gap.1).
#[test]
fn a_sugar_past_u32_is_an_overflow() {
    check_sugars("lex", &Lex);
    check_sugars(
        "block:1",
        &Block::new(&[1], 2).expect("block:1 on two variables"),
    );
    let (got, _, sugars) = update(&Grevlex, u32::MAX);
    assert_eq!(got, Ok(()), "grevlex");
    assert_eq!(
        sugars,
        vec![PAIR_LCM_DEGREE + 1],
        "grevlex: the lcm's degree plus one"
    );
}
