//! text-copy-peak, the crate's test code: `text::hand_over` hands every
//! byte of a text once, in order, in chunks of at most its chunk, at every
//! length around a page and a chunk, and a byte past ASCII ends it
//! (`iterations/0081-text-copy-peak/plan.md`, _Test code_, 1 and 2).

use std::io;
use std::num::NonZeroUsize;

use rand_core::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::budget::page_size;
use crate::text::hand_over;

/// A text of `len` drawn ASCII bytes, NUL and the control bytes among
/// them.
fn ascii(len: usize, seed: u64) -> String {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    (0..len)
        .map(|_| char::from((rng.next_u32() % 128) as u8))
        .collect()
}

/// The chunks `hand_over` hands of `text` at `chunk`, each its offset and
/// its bytes; the hand-over's error where it ends with one.
fn chunks_of(text: String, chunk: usize) -> (Vec<(usize, Vec<u8>)>, io::Result<()>) {
    let mut got = Vec::new();
    let chunk = NonZeroUsize::new(chunk).unwrap();
    let ended = hand_over(text, chunk, |from, part| got.push((from, part.to_vec())));
    (got, ended)
}

/// Test 1, text-copy-peak.1, .4, and .9: at lengths 0 to 2 and around one
/// and three pages and a chunk of 2^18 bytes, and at chunks of 1 byte
/// where the text is short, of 7 bytes, and around a page, 2^16, the text,
/// and past it, `hand_over` hands the text's bytes once, in order, each
/// chunk at its offset and of the chunk's length but the last, and returns
/// no error. A text of 8 MiB at chunks of 2^18 bytes as well.
#[test]
fn every_byte_is_handed_once_in_order() {
    let page = usize::try_from(page_size().unwrap()).unwrap();
    let lengths = [
        0,
        1,
        2,
        page - 1,
        page,
        page + 1,
        3 * page + 5,
        (1 << 18) - 1,
        1 << 18,
        (1 << 18) + 1,
        (1 << 20) + 3,
    ];
    for (k, &len) in lengths.iter().enumerate() {
        let mut chunks = vec![7, page - 1, page, page + 1, 1 << 16, len.max(1), len + 1];
        if len <= 3 * page + 5 {
            chunks.push(1);
        }
        for chunk in chunks {
            let text = ascii(len, k as u64);
            let (got, ended) = chunks_of(text.clone(), chunk);
            assert!(ended.is_ok(), "{len} bytes at chunks of {chunk}: {ended:?}");
            let mut next = 0;
            let mut whole = Vec::with_capacity(len);
            for (i, (from, part)) in got.iter().enumerate() {
                assert_eq!(*from, next, "{len} bytes at chunks of {chunk}: chunk {i}");
                assert!(!part.is_empty() && part.len() <= chunk);
                if i + 1 < got.len() {
                    assert_eq!(part.len(), chunk, "{len} bytes: chunk {i} short");
                }
                next += part.len();
                whole.extend_from_slice(part);
            }
            assert_eq!(next, len, "{len} bytes at chunks of {chunk}");
            assert!(whole == text.as_bytes(), "{len} bytes at chunks of {chunk}");
        }
    }
    let text = ascii(8 << 20, 99);
    let (got, ended) = chunks_of(text.clone(), 1 << 18);
    assert!(ended.is_ok(), "{ended:?}");
    assert_eq!(got.len(), 32);
    let whole: Vec<u8> = got.into_iter().flat_map(|(_, part)| part).collect();
    assert_eq!(whole, text.as_bytes());
}

/// Test 2, text-copy-peak.6: a text whose byte at k lies past ASCII hands
/// the chunks before the one that holds it, and no more, and returns
/// `InvalidData`; at the first byte, inside the first chunk, at a chunk's
/// last and first byte, and at the text's last.
#[test]
fn a_byte_past_ascii_ends_the_hand_over() {
    let chunk = 4096;
    let len = 5 * chunk + 17;
    for at in [0, 1, chunk - 1, chunk, 3 * chunk + 9, len - 2] {
        let mut text = ascii(len, at as u64);
        // "é" is two bytes past ASCII, at `at` and at + 1.
        text.replace_range(at..at + 2, "é");
        let (got, ended) = chunks_of(text.clone(), chunk);
        let err = ended.unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData, "byte {at}: {err}");
        assert_eq!(got.len(), at / chunk, "byte {at}: the chunks handed");
        for (i, (from, part)) in got.iter().enumerate() {
            assert_eq!(*from, i * chunk);
            assert_eq!(part[..], text.as_bytes()[i * chunk..(i + 1) * chunk]);
        }
    }
}
