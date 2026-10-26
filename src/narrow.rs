//! Narrowing integer conversions whose fit the caller guarantees: checked
//! under debug assertions, the `checked` profile the tests run on, and a
//! bare cast on the release profile. A cast that keeps the low bits on
//! purpose masks them first, so that they fit:
//! `(x & u128::from(u64::MAX)).narrow()`. A count whose fit nobody
//! guarantees goes through `id`, which fails where it does not fit.
//!
//! The checker holds its own copy, since it shares no code with the rest of
//! grobx (`OBJECTIVES.md`, _Checker_).

pub(crate) trait Narrow<T> {
    fn narrow(self) -> T;
}

macro_rules! narrow {
    ($($from:ty => $to:ty),* $(,)?) => {$(
        impl Narrow<$to> for $from {
            #[inline]
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_possible_wrap,
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
    i64 => u32,
    i128 => u64,
    u128 => u32,
    u128 => u64,
    u64 => i64,
    u64 => u32,
    u64 => u8,
    usize => u32,
    usize => u8,
);

/// A count past what a u32 id numbers (`id`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Size;

/// The id of the item a collection gains when it holds `n`: `n` itself.
/// Monomials, rows, terms, columns, and elements are numbered in u32, and
/// `u32::MAX` marks no item. A count from `u32::MAX` up would wrap into an
/// id that names another item, and the result would be wrong without a
/// sign, so it is `Size`, which the call returns as `Error::Size`. An index
/// below a count `id` passed is narrowed with `Narrow`.
pub(crate) fn id(n: usize) -> Result<u32, Size> {
    match u32::try_from(n) {
        Ok(i) if i < u32::MAX => Ok(i),
        Ok(_) | Err(_) => Err(Size),
    }
}

#[cfg(test)]
mod tests {
    use super::{Size, id};

    #[test]
    fn an_id_stops_below_the_mark_of_no_item() {
        let top = u32::MAX as usize;
        let got = [
            id(0),
            id(1),
            id(top - 1),
            id(top),
            id(top + 1),
            id(usize::MAX),
        ];
        let want = [
            Ok(0),
            Ok(1),
            Ok(u32::MAX - 1),
            Err(Size),
            Err(Size),
            Err(Size),
        ];
        assert_eq!(got, want);
    }
}
