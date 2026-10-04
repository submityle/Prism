//! Canonical, length-limited Huffman machinery for the dynamic-block
//! (`BTYPE=10`) `DEFLATE` encoder.
//!
//! This module is pure numeric table construction with no bit `I/O`: it turns
//! symbol frequencies into optimal, length-bounded canonical Huffman code
//! lengths, assigns the canonical codes those lengths imply (`RFC 1951`
//! §3.2.2), and run-length-encodes a combined code-length sequence into the
//! code-length alphabet (`RFC 1951` §3.2.7). The bit emission that consumes
//! these tables lives in [`super::deflate_encode`], and the decode side that
//! must agree with them lives in [`super::deflate`].
//!
//! Optimal length-limited lengths are produced by the Larmore–Hirschberg
//! package-merge algorithm, which minimises `sum(freq[i] * length[i])` subject
//! to `length[i] <= max_bits`, so the dynamic block is never larger than it has
//! to be for a given `LZ77` token stream while always staying inside the
//! 15-bit (`literal/length`, `distance`) and 7-bit (`code-length`) ceilings
//! `RFC 1951` imposes.

use alloc::vec;
use alloc::vec::Vec;

/// One run-length-encoded code-length symbol for the dynamic header
/// (`RFC 1951` §3.2.7): a code-length-alphabet `symbol` plus any trailing
/// `extra_value` payload (`extra_bits` wide, written `LSB`-first).
pub(crate) struct ClToken {
    /// Code-length-alphabet symbol (`0..=18`).
    pub symbol: u8,
    /// Payload for repeat symbols `16/17/18` (unused when `extra_bits == 0`).
    pub extra_value: u16,
    /// Number of payload bits the symbol carries (`0`, `2`, `3`, or `7`).
    pub extra_bits: u8,
}

/// A package-merge work item: a coalesced coin whose numismatic `weight` is the
/// sum of its member leaf frequencies, tracking which leaves it contains.
struct Package {
    weight: u64,
    members: Vec<u16>,
}

impl Clone for Package {
    fn clone(&self) -> Self {
        Self {
            weight: self.weight,
            members: self.members.clone(),
        }
    }
}

/// Computes optimal canonical Huffman code lengths for `freq`, guaranteeing no
/// code exceeds `max_bits`, via the Larmore–Hirschberg package-merge algorithm.
///
/// Unused symbols (`freq == 0`) get length `0`. A single used symbol gets
/// length `1` (an intentionally incomplete but unambiguously decodable code,
/// which the inflate core accepts). The result otherwise forms a complete code
/// whose Kraft sum is exactly `1`.
#[must_use]
pub(crate) fn length_limited_lengths(freq: &[u32], max_bits: u32) -> Vec<u8> {
    let mut lengths = vec![0u8; freq.len()];
    let used: Vec<usize> = (0..freq.len()).filter(|&i| freq[i] > 0).collect();
    let count = used.len();
    if count == 0 {
        return lengths;
    }
    if count == 1 {
        lengths[used[0]] = 1;
        return lengths;
    }
    debug_assert!(
        (1u64 << max_bits) >= count as u64,
        "max_bits too small to code every symbol"
    );

    // Leaf coins sorted by ascending weight; a stable sort keeps the result
    // deterministic for equal frequencies.
    let mut leaves: Vec<Package> = used
        .iter()
        .map(|&symbol| Package {
            weight: u64::from(freq[symbol]),
            members: vec![symbol as u16],
        })
        .collect();
    leaves.sort_by_key(|package| package.weight);

    // Package-merge: start from the leaf list and, for each of the remaining
    // `max_bits - 1` levels, pair adjacent items of the previous list and merge
    // those packages back with the original leaves (ascending by weight).
    let mut packages = leaves.clone();
    for _ in 1..max_bits {
        let mut paired: Vec<Package> = Vec::with_capacity(packages.len() / 2);
        let mut i = 0;
        while i + 1 < packages.len() {
            let mut members = packages[i].members.clone();
            members.extend_from_slice(&packages[i + 1].members);
            paired.push(Package {
                weight: packages[i].weight + packages[i + 1].weight,
                members,
            });
            i += 2;
        }
        packages = merge_sorted(&leaves, &paired);
    }

    // The first `2 * count - 2` packages of the final list select the coins; a
    // symbol's code length is how many selected packages contain it.
    let take = 2 * count - 2;
    for package in packages.iter().take(take) {
        for &symbol in &package.members {
            lengths[symbol as usize] += 1;
        }
    }
    lengths
}

/// Merges two ascending-by-weight package lists into one, keeping `a`'s items
/// ahead of `b`'s on ties so the merge is deterministic.
fn merge_sorted(a: &[Package], b: &[Package]) -> Vec<Package> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        if a[i].weight <= b[j].weight {
            out.push(a[i].clone());
            i += 1;
        } else {
            out.push(b[j].clone());
            j += 1;
        }
    }
    out.extend(a[i..].iter().cloned());
    out.extend(b[j..].iter().cloned());
    out
}

/// Assigns canonical Huffman codes (`RFC 1951` §3.2.2) for the given per-symbol
/// `lengths`, ordering by increasing length then increasing symbol value so the
/// codes match the inflate core's canonical layout. Unused symbols map to `0`.
#[must_use]
pub(crate) fn canonical_codes(lengths: &[u8], max_bits: u32) -> Vec<u16> {
    let mut bl_count = vec![0u16; (max_bits + 1) as usize];
    for &len in lengths {
        if len > 0 {
            bl_count[len as usize] += 1;
        }
    }
    let mut next_code = vec![0u16; (max_bits + 2) as usize];
    let mut code = 0u16;
    for bits in 1..=max_bits as usize {
        code = (code + bl_count[bits - 1]) << 1;
        next_code[bits] = code;
    }
    let mut codes = vec![0u16; lengths.len()];
    for (symbol, &len) in lengths.iter().enumerate() {
        if len > 0 {
            codes[symbol] = next_code[len as usize];
            next_code[len as usize] += 1;
        }
    }
    codes
}

/// Run-length-encodes a combined `literal/length` + `distance` code-length
/// sequence into code-length-alphabet symbols (`RFC 1951` §3.2.7), using repeat
/// code `16` (copy previous `3..=6` times), `17` (zero run `3..=10`), and `18`
/// (zero run `11..=138`).
#[must_use]
pub(crate) fn run_length_encode(lengths: &[u8]) -> Vec<ClToken> {
    let mut out = Vec::new();
    let total = lengths.len();
    let mut i = 0;
    while i < total {
        let value = lengths[i];
        let mut run_end = i + 1;
        while run_end < total && lengths[run_end] == value {
            run_end += 1;
        }
        let mut run = run_end - i;
        i = run_end;

        if value == 0 {
            while run >= 11 {
                let span = run.min(138);
                out.push(ClToken {
                    symbol: 18,
                    extra_value: (span - 11) as u16,
                    extra_bits: 7,
                });
                run -= span;
            }
            while run >= 3 {
                let span = run.min(10);
                out.push(ClToken {
                    symbol: 17,
                    extra_value: (span - 3) as u16,
                    extra_bits: 3,
                });
                run -= span;
            }
            for _ in 0..run {
                out.push(literal_cl(0));
            }
        } else {
            // The repeat code `16` copies the previous length, so emit the
            // value literally once before collapsing the remainder.
            out.push(literal_cl(value));
            run -= 1;
            while run >= 3 {
                let span = run.min(6);
                out.push(ClToken {
                    symbol: 16,
                    extra_value: (span - 3) as u16,
                    extra_bits: 2,
                });
                run -= span;
            }
            for _ in 0..run {
                out.push(literal_cl(value));
            }
        }
    }
    out
}

/// A plain code-length symbol (`0..=15`) with no extra payload.
fn literal_cl(symbol: u8) -> ClToken {
    ClToken {
        symbol,
        extra_value: 0,
        extra_bits: 0,
    }
}
