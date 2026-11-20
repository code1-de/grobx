//! Numbers on 64-bit limbs, the least significant first, with no zero limb
//! at the top.

use std::cmp::Ordering;

use num_bigint::BigUint;

use crate::narrow::Narrow;

/// x against y, each on limbs, least significant first, with no zero limb
/// at the top.
pub(crate) fn cmp(x: &[u64], y: &[u64]) -> Ordering {
    x.len()
        .cmp(&y.len())
        .then_with(|| x.iter().rev().cmp(y.iter().rev()))
}

/// x's i-th limb, 0 past its top.
pub(crate) fn limb(x: &[u64], i: usize) -> u64 {
    x.get(i).copied().unwrap_or(0)
}

/// x on limbs as num-bigint's: from a u128 where x has at most two limbs.
pub(crate) fn big(x: &[u64]) -> BigUint {
    if x.len() <= 2 {
        return BigUint::from(u128::from(limb(x, 0)) | u128::from(limb(x, 1)) << 64);
    }
    BigUint::new(
        x.iter()
            .flat_map(|&l| -> [u32; 2] { [(l & u64::from(u32::MAX)).narrow(), (l >> 32).narrow()] })
            .collect(),
    )
}

/// L(x), the bit length: 0 for 0.
pub(crate) fn bits(x: &[u64]) -> u64 {
    x.last()
        .map_or(0, |&l| 64 * x.len() as u64 - u64::from(l.leading_zeros()))
}
