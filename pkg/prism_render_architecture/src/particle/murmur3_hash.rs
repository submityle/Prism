//! `MurmurHash3` `x86_32` non-cryptographic hash: a pure-integer, block-mixing
//! hash for fast in-memory hashing of small keys such as resource names,
//! `GPU` pipeline-state descriptors, spatial-hash cell coordinates, and
//! string-interning tables (design § content-addressing helpers).
//!
//! `MurmurHash3` is Austin Appleby's public-domain successor to `MurmurHash2`.
//! This module implements the `x86_32` variant, which consumes the input as a
//! little-endian stream of `32`-bit blocks and produces a `32`-bit digest. Each
//! `4`-byte block `k` is mixed by multiplying by the constant `C1`, rotating
//! left by `15`, and multiplying by the constant `C2`; the running hash then
//! folds that mixed block in with an exclusive-or, a left rotate of `13`, and
//! the affine step `hash = hash * 5 + 0xe6546b64`:
//!
//! ```text
//! for each 4-byte little-endian block k:
//!     k    = k * C1
//!     k    = rotl(k, 15)
//!     k    = k * C2
//!     hash = hash XOR k
//!     hash = rotl(hash, 13)
//!     hash = hash * 5 + 0xe6546b64      (all wrapping / modulo 2^32)
//! ```
//!
//! A trailing run of `1`, `2`, or `3` leftover bytes is assembled little-endian
//! into a partial block, mixed through the same `C1` / rotate-`15` / `C2`
//! sequence, and folded into the hash without the trailing rotate-and-affine
//! step. The digest is then finalized by mixing in the input length and running
//! the `fmix32` finalizer, an avalanche cascade of two shift-`xor`-multiply
//! rounds (constants `0x85ebca6b` and `0xc2b2ae35`) that scrambles every input
//! bit across the whole `32`-bit word so that flipping one input bit flips
//! roughly half of the output bits:
//!
//! ```text
//! hash = hash XOR len
//! hash = fmix32(hash)
//!
//! fmix32(h):
//!     h = h XOR (h >> 16)
//!     h = h * 0x85ebca6b
//!     h = h XOR (h >> 13)
//!     h = h * 0xc2b2ae35
//!     h = h XOR (h >> 16)
//! ```
//!
//! Every operation is a wrapping multiply, a bit rotate, a right shift, or an
//! exclusive-or; there are no floating-point, transcendental, or table-driven
//! operations, and the running time is linear in the input length, `O(n)`.
//!
//! Three surfaces are exposed. [`murmur3_32`] hashes a byte slice under a caller
//! chosen `seed`; the empty slice under `seed == 0` hashes to `0`, matching the
//! canonical reference implementation. [`murmur3_32_str`] hashes the `UTF-8`
//! bytes of a string slice, and the `const` [`fmix32`] exposes the bare
//! finalizer for callers that already hold a `32`-bit value they want to
//! avalanche (for example finalizing an externally accumulated integer key).
//!
//! Scope: this is a fast hash for hashing and lookup, *not* a cryptographic hash
//! and *not* a message authentication code. `MurmurHash3` is not collision
//! resistant against an adversary — seed-independent multicollisions are known —
//! and must never be used to authenticate data or guard against a malicious
//! party; for those uses pick a real cryptographic hash. This module is
//! deliberately disjoint from the `crc32` module (a polynomial-division
//! checksum) and the `fnv1a_hash` module (a byte-at-a-time `xor`-then-multiply
//! hash): `MurmurHash3` instead mixes whole `32`-bit blocks and finalizes with
//! `fmix32`.

/// First block-mixing multiplier constant (`0xcc9e2d51`).
const C1: u32 = 0xcc9e_2d51;

/// Second block-mixing multiplier constant (`0x1b873593`).
const C2: u32 = 0x1b87_3593;

/// The affine multiplier applied to the running hash after each block (`5`).
const HASH_MUL: u32 = 5;

/// The affine addend applied to the running hash after each block
/// (`0xe6546b64`).
const HASH_ADD: u32 = 0xe654_6b64;

/// Left-rotate amount applied to a mixed block (`15`).
const BLOCK_ROTL: u32 = 15;

/// Left-rotate amount applied to the running hash after each block (`13`).
const HASH_ROTL: u32 = 13;

/// First `fmix32` avalanche multiplier (`0x85ebca6b`).
const FMIX_MUL_1: u32 = 0x85eb_ca6b;

/// Second `fmix32` avalanche multiplier (`0xc2b2ae35`).
const FMIX_MUL_2: u32 = 0xc2b2_ae35;

/// Applies the `MurmurHash3` `fmix32` finalizer to a `32`-bit value.
///
/// `fmix32` is the avalanche cascade that scrambles every bit of `h` across the
/// whole word: an exclusive-or with the high `16` bits, a wrapping multiply, an
/// exclusive-or with the high `13` bits, another wrapping multiply, and a final
/// exclusive-or with the high `16` bits. It is exposed as a `const fn` so it can
/// be used to avalanche any externally accumulated `32`-bit key, not just the
/// output of [`murmur3_32`]. `fmix32(0) == 0`.
#[must_use]
pub const fn fmix32(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(FMIX_MUL_1);
    h ^= h >> 13;
    h = h.wrapping_mul(FMIX_MUL_2);
    h ^= h >> 16;
    h
}

/// Computes the `MurmurHash3` `x86_32` digest of `bytes` under `seed`.
///
/// Full `4`-byte blocks are read little-endian and folded through the
/// `C1` / rotate-`15` / `C2` block mix followed by the running-hash
/// rotate-and-affine step; a trailing `1`-to-`3`-byte remainder is assembled
/// little-endian and folded through the block mix only. The digest is finalized
/// by mixing in the input length and applying [`fmix32`]. The empty slice under
/// `seed == 0` hashes to `0`.
#[must_use]
pub fn murmur3_32(bytes: &[u8], seed: u32) -> u32 {
    let mut hash = seed;

    // Body: consume every full 4-byte little-endian block.
    let mut blocks = bytes.chunks_exact(4);
    for block in blocks.by_ref() {
        let mut k = u32::from_le_bytes([block[0], block[1], block[2], block[3]]);
        k = k.wrapping_mul(C1);
        k = k.rotate_left(BLOCK_ROTL);
        k = k.wrapping_mul(C2);

        hash ^= k;
        hash = hash.rotate_left(HASH_ROTL);
        hash = hash.wrapping_mul(HASH_MUL).wrapping_add(HASH_ADD);
    }

    // Tail: 1..=3 leftover bytes assembled little-endian, block-mixed only.
    let tail = blocks.remainder();
    if !tail.is_empty() {
        let mut k: u32 = 0;
        if tail.len() >= 3 {
            k ^= u32::from(tail[2]) << 16;
        }
        if tail.len() >= 2 {
            k ^= u32::from(tail[1]) << 8;
        }
        k ^= u32::from(tail[0]);
        k = k.wrapping_mul(C1);
        k = k.rotate_left(BLOCK_ROTL);
        k = k.wrapping_mul(C2);
        hash ^= k;
    }

    // Finalize: mix in the length, then avalanche.
    hash ^= bytes.len() as u32;
    fmix32(hash)
}

/// Computes the `MurmurHash3` `x86_32` digest of the `UTF-8` bytes of `s`.
///
/// Equivalent to `murmur3_32(s.as_bytes(), seed)`.
#[must_use]
pub fn murmur3_32_str(s: &str, seed: u32) -> u32 {
    murmur3_32(s.as_bytes(), seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny `LCG`-style generator for deterministic pseudo-random bytes.
    struct Lcg {
        state: u64,
    }

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_u8(&mut self) -> u8 {
            self.state = self
                .state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.state >> 56) as u8
        }

        fn bytes(&mut self, len: usize) -> alloc::vec::Vec<u8> {
            (0..len).map(|_| self.next_u8()).collect()
        }
    }

    // --- Canonical reference vectors (seed 0 unless noted) ---

    #[test]
    fn empty_seed_zero_is_zero() {
        assert_eq!(murmur3_32(b"", 0), 0x0000_0000);
    }

    #[test]
    fn empty_seed_one_is_known() {
        assert_eq!(murmur3_32(b"", 1), 0x514e_28b7);
    }

    #[test]
    fn test_word_is_known() {
        assert_eq!(murmur3_32(b"test", 0), 0xba6b_d213);
    }

    #[test]
    fn hello_world_is_known() {
        assert_eq!(murmur3_32(b"Hello, world!", 0), 0xc036_3e43);
    }

    #[test]
    fn quick_brown_fox_is_known() {
        assert_eq!(
            murmur3_32(b"The quick brown fox jumps over the lazy dog", 0),
            0x2e4f_f723
        );
    }

    #[test]
    fn check_string_123456789_is_known() {
        assert_eq!(murmur3_32(b"123456789", 0), 0xb4fe_f382);
    }

    // --- Tail lengths: 1, 2, 3 leftover bytes each ---

    #[test]
    fn tail_one_byte_is_known() {
        assert_eq!(murmur3_32(b"a", 0), 0x3c25_69b2);
    }

    #[test]
    fn tail_two_bytes_is_known() {
        assert_eq!(murmur3_32(b"ab", 0), 0x9bbf_d75f);
    }

    #[test]
    fn tail_three_bytes_is_known() {
        assert_eq!(murmur3_32(b"abc", 0), 0xb3dd_93fa);
    }

    #[test]
    fn exactly_one_block_is_known() {
        assert_eq!(murmur3_32(b"abcd", 0), 0x43ed_676a);
    }

    // --- Lengths 4/5/6/7/8 straddling the block boundary ---

    #[test]
    fn length_four_is_known() {
        assert_eq!(murmur3_32(b"1234", 0), 0x721c_5dc3);
    }

    #[test]
    fn length_five_is_known() {
        assert_eq!(murmur3_32(b"12345", 0), 0x13a5_1193);
    }

    #[test]
    fn length_six_is_known() {
        assert_eq!(murmur3_32(b"123456", 0), 0xbf60_eab8);
    }

    #[test]
    fn length_seven_is_known() {
        assert_eq!(murmur3_32(b"1234567", 0), 0xb7ef_82f7);
    }

    #[test]
    fn length_eight_is_known() {
        assert_eq!(murmur3_32(b"12345678", 0), 0x91b3_13ce);
    }

    // --- Seed influence ---

    #[test]
    fn seed_changes_fox_digest() {
        let a = murmur3_32(b"The quick brown fox jumps over the lazy dog", 0);
        let b = murmur3_32(b"The quick brown fox jumps over the lazy dog", 42);
        assert_ne!(a, b);
        assert_eq!(b, 0x347c_a102);
    }

    #[test]
    fn seed_changes_hello_digest() {
        assert_ne!(
            murmur3_32(b"Hello, world!", 0),
            murmur3_32(b"Hello, world!", 1)
        );
        assert_eq!(murmur3_32(b"Hello, world!", 1), 0xaa5d_c85b);
    }

    #[test]
    fn empty_input_equals_fmix_of_seed() {
        // For an empty slice the body and tail are skipped and len == 0, so the
        // digest reduces to fmix32(seed).
        for seed in [0u32, 1, 7, 0x9747_b28c, 0xffff_ffff] {
            assert_eq!(murmur3_32(b"", seed), fmix32(seed));
        }
    }

    // --- fmix32 known points ---

    #[test]
    fn fmix32_zero_is_zero() {
        assert_eq!(fmix32(0), 0);
    }

    #[test]
    fn fmix32_one_is_known() {
        assert_eq!(fmix32(1), 0x514e_28b7);
    }

    #[test]
    fn fmix32_all_ones_is_known() {
        assert_eq!(fmix32(0xffff_ffff), 0x81f1_6f39);
    }

    #[test]
    fn fmix32_is_injective_on_sample() {
        // fmix32 is a bijection; a small sample must map to distinct outputs.
        let mut seen = alloc::collections::BTreeSet::new();
        for i in 0u32..1000 {
            assert!(seen.insert(fmix32(i)));
        }
    }

    // --- Avalanche: a single input bit flip alters the digest ---

    #[test]
    fn avalanche_single_bit_flip_differs() {
        let base = murmur3_32(b"avalanche", 0);
        let mut flipped = *b"avalanche";
        flipped[0] ^= 0x01;
        assert_ne!(base, murmur3_32(&flipped, 0));
    }

    #[test]
    fn avalanche_across_every_bit_position() {
        let data = b"murmur-avalanche-probe";
        let base = murmur3_32(data, 0);
        for byte_index in 0..data.len() {
            for bit in 0..8u32 {
                let mut probe = data.to_vec();
                probe[byte_index] ^= 1u8 << bit;
                assert_ne!(base, murmur3_32(&probe, 0));
            }
        }
    }

    #[test]
    fn seed_single_bit_flip_differs() {
        let base = murmur3_32(b"seed-avalanche", 0);
        for bit in 0..32u32 {
            assert_ne!(base, murmur3_32(b"seed-avalanche", 1u32 << bit));
        }
    }

    // --- Order and content sensitivity ---

    #[test]
    fn order_sensitivity() {
        assert_ne!(murmur3_32(b"ab", 0), murmur3_32(b"ba", 0));
    }

    #[test]
    fn appending_zero_changes_result() {
        assert_ne!(murmur3_32(b"data", 0), murmur3_32(b"data\0", 0));
    }

    #[test]
    fn distinct_short_inputs_differ() {
        assert_ne!(murmur3_32(b"aaaa", 0), murmur3_32(b"aaab", 0));
        assert_ne!(murmur3_32(b"aaaa", 0), murmur3_32(b"baaa", 0));
    }

    // --- str convenience wrapper ---

    #[test]
    fn str_wrapper_matches_bytes() {
        assert_eq!(murmur3_32_str("test", 0), murmur3_32(b"test", 0));
        assert_eq!(murmur3_32_str("", 0), 0);
    }

    #[test]
    fn str_wrapper_handles_unicode() {
        let s = "café ❤😀";
        assert_eq!(murmur3_32_str(s, 0), murmur3_32(s.as_bytes(), 0));
        assert_eq!(murmur3_32_str(s, 7), murmur3_32(s.as_bytes(), 7));
    }

    // --- Purity / determinism ---

    #[test]
    fn repeated_calls_are_pure() {
        assert_eq!(murmur3_32(b"idempotent", 3), murmur3_32(b"idempotent", 3));
        assert_eq!(fmix32(0xdead_beef), fmix32(0xdead_beef));
    }

    #[test]
    fn multi_block_input_is_stable() {
        let data = alloc::vec![0x5Au8; 4096];
        assert_eq!(murmur3_32(&data, 0), murmur3_32(&data, 0));
    }

    // --- Collision resistance on random inputs ---

    #[test]
    fn many_random_inputs_have_no_collisions() {
        let mut lcg = Lcg::new(0x0F0F_0F0F_1234_5678);
        let mut seen = alloc::collections::BTreeSet::new();
        for _ in 0..512 {
            let data = lcg.bytes(16);
            seen.insert(murmur3_32(&data, 0));
        }
        assert_eq!(seen.len(), 512);
    }

    #[test]
    fn incrementing_keys_have_no_collisions() {
        let mut seen = alloc::collections::BTreeSet::new();
        for i in 0u32..2048 {
            seen.insert(murmur3_32(&i.to_le_bytes(), 0));
        }
        assert_eq!(seen.len(), 2048);
    }
}
