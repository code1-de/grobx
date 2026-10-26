//! Narrowing integer conversions whose fit the caller guarantees: checked
//! under debug assertions, the `checked` profile the tests run on, and a
//! bare cast on the release profile. The checker's casts that keep the low
//! bits on purpose, its hash slots (`divisors.rs`), take them by an `as`
//! cast and a mask.
//!
//! grobx holds its own copy, since the checker shares no code with it
//! (`OBJECTIVES.md`, _Checker_).

pub(crate) trait Narrow<T> {
    fn narrow(self) -> T;
}

macro_rules! narrow {
    ($($from:ty => $to:ty),* $(,)?) => {$(
        impl Narrow<$to> for $from {
            #[inline]
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "the fit is the caller's guarantee, asserted under debug assertions"
            )]
            fn narrow(self) -> $to {
                debug_assert!(
                    <$to>::try_from(self).is_ok(),
                    "{self} does not fit in {to}",
                    to = stringify!($to)
                );
                self as $to
            }
        }
    )*};
}

narrow!(
    i128 => u64,
    u128 => u64,
    u64 => u32,
    usize => u32,
);
