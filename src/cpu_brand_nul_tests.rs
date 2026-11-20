//! cpu-brand-nul, the item's test code: the call report's `cpu` field holds
//! the processor's brand up to its first NUL byte, and every call line names
//! the brand of this machine (`iterations/0150-cpu-brand-nul/plan.md`, _Test
//! code_, 1 and 2).

use crate::MonomialOrder;
use crate::rebuilt_reducer_tests::{call, reported};
use crate::report::{BRAND_BYTES, Or, brand, cpu};
use crate::split_piece_time_tests::EDGE;

/// Letters, `BRAND_BYTES` of them, none a NUL.
const LETTERS: &[u8; BRAND_BYTES] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUV";

/// The brand leaves' bytes of `text` and NUL bytes after it.
fn padded(text: &[u8]) -> [u8; BRAND_BYTES] {
    let mut bytes = [0; BRAND_BYTES];
    bytes[..text.len()].copy_from_slice(text);
    bytes
}

/// Test 1 (cpu-brand-nul.1, .2, .3): the field is the bytes before the first
/// NUL as `field_value` makes them, all of them where none is NUL, `None`
/// where they hold no word; a byte before the NUL that is no UTF-8 stands as
/// U+FFFD, and the bytes after the NUL, whatever they are, are no part of it.
#[test]
fn a_brand_ends_at_its_first_nul() {
    let all = String::from_utf8(LETTERS.to_vec()).unwrap();
    let cases: [(&str, [u8; BRAND_BYTES], Option<&str>); 12] = [
        (
            "NUL bytes at the end alone",
            padded(b"Intel(R) Core(TM) i7-14700K"),
            Some("Intel(R)_Core(TM)_i7-14700K"),
        ),
        (
            "words after a NUL",
            padded(b"AMD Ryzen 9\0 7950X 16-Core"),
            Some("AMD_Ryzen_9"),
        ),
        (
            "several NULs, bytes between them",
            padded(b"abc\0def\0ghi"),
            Some("abc"),
        ),
        (
            "a NUL at the last byte alone",
            padded(&LETTERS[..BRAND_BYTES - 1]),
            Some(&all[..BRAND_BYTES - 1]),
        ),
        ("no NUL", *LETTERS, Some(&all)),
        (
            "spaces before the brand, as a right-justified brand has",
            padded(b"         Intel(R) Pentium(R) 4 CPU 1.50GHz"),
            Some("Intel(R)_Pentium(R)_4_CPU_1.50GHz"),
        ),
        (
            "a NUL at the first byte, words after it",
            padded(b"\0Intel(R) Core(TM)"),
            None,
        ),
        ("every byte NUL", [0; BRAND_BYTES], None),
        ("spaces alone before the NUL", padded(b"   \0abc"), None),
        (
            "an = and a tab before the NUL",
            padded(b"a=b\tc\0d e"),
            Some("a_b_c"),
        ),
        (
            "a byte that is no UTF-8 before the NUL",
            padded(b"\xffx\0y"),
            Some("\u{fffd}x"),
        ),
        (
            "bytes that are no UTF-8 after the NUL",
            padded(b"x y\0\xff\xfe"),
            Some("x_y"),
        ),
    ];
    for (case, bytes, want) in cases {
        assert_eq!(brand(&bytes).as_deref(), want, "{case}: {bytes:?}");
    }
}

/// Test 2: over the edge cases of `OBJECTIVES.md` over GF(2), GF(65521), and
/// ℚ, the `call` line, the report's first, names the brand of this machine,
/// or `none`.
#[test]
fn every_call_line_names_the_brand() {
    let want = Or(cpu(), "none").to_string();
    for p in [2u64, 65521, 0] {
        for gens in EDGE {
            let text = format!("x,y,z\n{p}\n{gens}\n");
            let (_, lines) = call(&text, &reported(MonomialOrder::Grevlex));
            let first = lines.first().expect("a line");
            assert_eq!(
                (first.kind.as_str(), first.get("cpu")),
                ("call", Some(want.as_str())),
                "{text}"
            );
        }
    }
}
