//! `FNV-1a` non-cryptographic hash (`Fowler`–`Noll`–`Vo` alternate),
//! `32`- and `64`-bit, for fast in-memory hashing of small keys such as
//! resource names, `GPU` pipeline-state descriptors, and string interning tables
//! (design § content-addressing helpers).
//!
//! `FNV-1a` is a multiply-then-xor hash. It keeps a single accumulator seeded
//! with a fixed *offset basis* and, for every input byte, first mixes the byte
//! into the accumulator with an exclusive-or and then multiplies the
//! accumulator by a fixed *`FNV` prime*:
//!
//! ```text
//! hash = offset_basis
//! for each byte:
//!     hash = hash XOR byte
//!     hash = hash * prime      (wrapping / modulo 2^width)
//! ```
//!
//! The `FNV-1a` variant xors *before* multiplying (plain `FNV-1` multiplies
//! first); the alternate ordering gives markedly better avalanche on the low
//! bits, so `FNV-1a` is the form used in practice. This module implements only
//! `FNV-1a`, never plain `FNV-1`.
//!
//! The parameters are the canonical, public `FNV` constants. For the `32`-bit
//! width the offset basis is `0x811c9dc5` and the prime is `0x0100_0193`
//! (`16_777_619`). For the `64`-bit width the offset basis is
//! `0xcbf29ce4_84222325` and the prime is `0x0000_0100_0000_01b3`
//! (`1_099_511_628_211`). All arithmetic is done with wrapping (modular)
//! multiplication and exclusive-or; there are no floating-point, transcendental,
//! or table-driven operations, and the running time is linear in the input
//! length, `O(n)`.
//!
//! Two surfaces are exposed per width. The one-shot [`fnv1a_32`] and
//! [`fnv1a_64`] hash a whole slice in a single call. The streaming state
//! [`Fnv1a32`] / [`Fnv1a64`] (built with [`fnv1a32_new`] / [`fnv1a64_new`])
//! folds data in arbitrarily many chunks via `update` and yields the digest with
//! `finish`; by construction folding a byte stream in one call or across several
//! `update` calls produces the same digest. Convenience wrappers
//! [`fnv1a_32_str`] / [`fnv1a_64_str`] hash the `UTF-8` bytes of a string slice,
//! and [`fnv1a_32_combine`] / [`fnv1a_64_combine`] advance an accumulator by a
//! single byte for callers that drive the loop themselves.
//!
//! Scope: this is a fast hash for hashing and lookup, *not* a cryptographic hash
//! and *not* a message authentication code. `FNV-1a` is not collision resistant
//! against an adversary and must never be used to authenticate data or guard
//! against a malicious party; for those uses pick a real cryptographic hash.
//! This module is deliberately disjoint from the `crc32` module, which computes
//! a cyclic-redundancy checksum by a different (polynomial-division) algorithm.

/// The `32`-bit `FNV` offset basis (`0x811c9dc5`).
pub const FNV1A_32_OFFSET_BASIS: u32 = 0x811c_9dc5;

/// The `32`-bit `FNV` prime (`0x0100_0193`, i.e. `16_777_619`).
pub const FNV1A_32_PRIME: u32 = 0x0100_0193;

/// The `64`-bit `FNV` offset basis (`0xcbf29ce4_84222325`).
pub const FNV1A_64_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;

/// The `64`-bit `FNV` prime (`0x0000_0100_0000_01b3`, i.e.
/// `1_099_511_628_211`).
pub const FNV1A_64_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Advances a `32`-bit `FNV-1a` accumulator by a single byte.
///
/// Mixes `byte` into `hash` with an exclusive-or and then multiplies by the
/// `32`-bit `FNV` prime (wrapping). Starting from [`FNV1A_32_OFFSET_BASIS`]
/// and folding every byte of a slice reproduces [`fnv1a_32`].
#[must_use]
pub const fn fnv1a_32_combine(hash: u32, byte: u8) -> u32 {
    (hash ^ byte as u32).wrapping_mul(FNV1A_32_PRIME)
}

/// Advances a `64`-bit `FNV-1a` accumulator by a single byte.
///
/// Mixes `byte` into `hash` with an exclusive-or and then multiplies by the
/// `64`-bit `FNV` prime (wrapping). Starting from [`FNV1A_64_OFFSET_BASIS`]
/// and folding every byte of a slice reproduces [`fnv1a_64`].
#[must_use]
pub const fn fnv1a_64_combine(hash: u64, byte: u8) -> u64 {
    (hash ^ byte as u64).wrapping_mul(FNV1A_64_PRIME)
}

/// Computes the `32`-bit `FNV-1a` hash of `bytes` in a single call.
///
/// The empty slice yields the offset basis [`FNV1A_32_OFFSET_BASIS`]
/// (`0x811c9dc5`).
#[must_use]
pub fn fnv1a_32(bytes: &[u8]) -> u32 {
    let mut hash = FNV1A_32_OFFSET_BASIS;
    for &byte in bytes {
        hash = fnv1a_32_combine(hash, byte);
    }
    hash
}

/// Computes the `64`-bit `FNV-1a` hash of `bytes` in a single call.
///
/// The empty slice yields the offset basis [`FNV1A_64_OFFSET_BASIS`]
/// (`0xcbf29ce484222325`).
#[must_use]
pub fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash = FNV1A_64_OFFSET_BASIS;
    for &byte in bytes {
        hash = fnv1a_64_combine(hash, byte);
    }
    hash
}

/// Computes the `32`-bit `FNV-1a` hash of the `UTF-8` bytes of `s`.
///
/// Equivalent to `fnv1a_32(s.as_bytes())`.
#[must_use]
pub fn fnv1a_32_str(s: &str) -> u32 {
    fnv1a_32(s.as_bytes())
}

/// Computes the `64`-bit `FNV-1a` hash of the `UTF-8` bytes of `s`.
///
/// Equivalent to `fnv1a_64(s.as_bytes())`.
#[must_use]
pub fn fnv1a_64_str(s: &str) -> u64 {
    fnv1a_64(s.as_bytes())
}

/// A resumable `32`-bit `FNV-1a` hashing state.
///
/// Build one with [`fnv1a32_new`], fold bytes with [`Fnv1a32::update`], and
/// read the digest with [`Fnv1a32::finish`]. Folding a stream across several
/// `update` calls yields the same digest as a single [`fnv1a_32`] call over
/// the concatenation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Fnv1a32 {
    /// The running `32`-bit accumulator (seeded with the offset basis).
    state: u32,
}

/// A resumable `64`-bit `FNV-1a` hashing state.
///
/// Build one with [`fnv1a64_new`], fold bytes with [`Fnv1a64::update`], and
/// read the digest with [`Fnv1a64::finish`]. Folding a stream across several
/// `update` calls yields the same digest as a single [`fnv1a_64`] call over
/// the concatenation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Fnv1a64 {
    /// The running `64`-bit accumulator (seeded with the offset basis).
    state: u64,
}

/// Creates a fresh `32`-bit streaming state seeded with the offset basis.
#[must_use]
pub const fn fnv1a32_new() -> Fnv1a32 {
    Fnv1a32 {
        state: FNV1A_32_OFFSET_BASIS,
    }
}

/// Creates a fresh `64`-bit streaming state seeded with the offset basis.
#[must_use]
pub const fn fnv1a64_new() -> Fnv1a64 {
    Fnv1a64 {
        state: FNV1A_64_OFFSET_BASIS,
    }
}

impl Fnv1a32 {
    /// Folds every byte of `bytes` into the running state.
    pub fn update(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.state = fnv1a_32_combine(self.state, byte);
        }
    }

    /// Returns the current digest without consuming the state.
    #[must_use]
    pub const fn finish(&self) -> u32 {
        self.state
    }
}

impl Default for Fnv1a32 {
    fn default() -> Self {
        fnv1a32_new()
    }
}

impl Fnv1a64 {
    /// Folds every byte of `bytes` into the running state.
    pub fn update(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.state = fnv1a_64_combine(self.state, byte);
        }
    }

    /// Returns the current digest without consuming the state.
    #[must_use]
    pub const fn finish(&self) -> u64 {
        self.state
    }
}

impl Default for Fnv1a64 {
    fn default() -> Self {
        fnv1a64_new()
    }
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

    // --- Empty input equals the offset basis ---

    #[test]
    fn empty_input_32_is_offset_basis() {
        assert_eq!(fnv1a_32(b""), FNV1A_32_OFFSET_BASIS);
        assert_eq!(fnv1a_32(b""), 0x811c_9dc5);
    }

    #[test]
    fn empty_input_64_is_offset_basis() {
        assert_eq!(fnv1a_64(b""), FNV1A_64_OFFSET_BASIS);
        assert_eq!(fnv1a_64(b""), 0xcbf2_9ce4_8422_2325);
    }

    // --- Known public FNV-1a test vectors (canonical public algorithm constants) ---

    #[test]
    fn known_vector_32_a() {
        // Public FNV-1a 32-bit vector for "a".
        assert_eq!(fnv1a_32(b"a"), 0xe40c_292c);
    }

    #[test]
    fn known_vector_32_foobar() {
        // Public FNV-1a 32-bit vector for "foobar".
        assert_eq!(fnv1a_32(b"foobar"), 0xbf9c_f968);
    }

    #[test]
    fn known_vector_32_foo() {
        // Public FNV-1a 32-bit vector for "foo".
        assert_eq!(fnv1a_32(b"foo"), 0xa9f3_7ed7);
    }

    #[test]
    fn known_vector_64_a() {
        // Public FNV-1a 64-bit vector for "a".
        assert_eq!(fnv1a_64(b"a"), 0xaf63_dc4c_8601_ec8c);
    }

    #[test]
    fn known_vector_64_foobar() {
        // Public FNV-1a 64-bit vector for "foobar".
        assert_eq!(fnv1a_64(b"foobar"), 0x8594_4171_f739_67e8);
    }

    #[test]
    fn known_vector_64_foo() {
        // Public FNV-1a 64-bit vector for "foo".
        assert_eq!(fnv1a_64(b"foo"), 0xdcb2_7518_fed9_d577);
    }

    // --- Constant sanity ---

    #[test]
    fn constants_have_canonical_values() {
        assert_eq!(FNV1A_32_OFFSET_BASIS, 0x811c_9dc5);
        assert_eq!(FNV1A_32_PRIME, 16_777_619);
        assert_eq!(FNV1A_64_OFFSET_BASIS, 0xcbf2_9ce4_8422_2325);
        assert_eq!(FNV1A_64_PRIME, 1_099_511_628_211);
    }

    // --- Streaming update matches one-shot ---

    #[test]
    fn streaming_32_two_chunks_matches_oneshot() {
        let mut h = fnv1a32_new();
        h.update(b"foo");
        h.update(b"bar");
        assert_eq!(h.finish(), fnv1a_32(b"foobar"));
    }

    #[test]
    fn streaming_64_two_chunks_matches_oneshot() {
        let mut h = fnv1a64_new();
        h.update(b"foo");
        h.update(b"bar");
        assert_eq!(h.finish(), fnv1a_64(b"foobar"));
    }

    #[test]
    fn streaming_32_new_finish_is_offset_basis() {
        assert_eq!(fnv1a32_new().finish(), FNV1A_32_OFFSET_BASIS);
    }

    #[test]
    fn streaming_64_new_finish_is_offset_basis() {
        assert_eq!(fnv1a64_new().finish(), FNV1A_64_OFFSET_BASIS);
    }

    #[test]
    fn streaming_32_byte_by_byte_matches_oneshot() {
        let data = b"streaming byte by byte should match";
        let mut h = fnv1a32_new();
        for &b in data {
            h.update(&[b]);
        }
        assert_eq!(h.finish(), fnv1a_32(data));
    }

    #[test]
    fn streaming_64_byte_by_byte_matches_oneshot() {
        let data = b"streaming byte by byte should match";
        let mut h = fnv1a64_new();
        for &b in data {
            h.update(&[b]);
        }
        assert_eq!(h.finish(), fnv1a_64(data));
    }

    #[test]
    fn streaming_32_empty_update_is_noop() {
        let mut h = fnv1a32_new();
        h.update(b"");
        assert_eq!(h.finish(), FNV1A_32_OFFSET_BASIS);
    }

    #[test]
    fn streaming_64_empty_update_is_noop() {
        let mut h = fnv1a64_new();
        h.update(b"");
        assert_eq!(h.finish(), FNV1A_64_OFFSET_BASIS);
    }

    #[test]
    fn streaming_32_every_split_matches_oneshot() {
        let mut lcg = Lcg::new(0xABCD_1234_5678_9F00);
        let data = lcg.bytes(300);
        let oneshot = fnv1a_32(&data);
        for split in 0..=data.len() {
            let (left, right) = data.split_at(split);
            let mut h = fnv1a32_new();
            h.update(left);
            h.update(right);
            assert_eq!(h.finish(), oneshot);
        }
    }

    #[test]
    fn streaming_64_random_chunking_matches_oneshot() {
        let mut lcg = Lcg::new(0xDEAD_BEEF_CAFE_F00D);
        let data = lcg.bytes(4096);
        let oneshot = fnv1a_64(&data);
        let mut h = fnv1a64_new();
        for chunk in data.chunks(37) {
            h.update(chunk);
        }
        assert_eq!(h.finish(), oneshot);
    }

    #[test]
    fn default_matches_new() {
        assert_eq!(Fnv1a32::default().finish(), fnv1a32_new().finish());
        assert_eq!(Fnv1a64::default().finish(), fnv1a64_new().finish());
    }

    // --- str convenience wrappers ---

    #[test]
    fn str_wrapper_32_matches_bytes() {
        assert_eq!(fnv1a_32_str("foobar"), fnv1a_32(b"foobar"));
        assert_eq!(fnv1a_32_str(""), FNV1A_32_OFFSET_BASIS);
    }

    #[test]
    fn str_wrapper_64_matches_bytes() {
        assert_eq!(fnv1a_64_str("foobar"), fnv1a_64(b"foobar"));
        assert_eq!(fnv1a_64_str(""), FNV1A_64_OFFSET_BASIS);
    }

    #[test]
    fn str_wrapper_handles_unicode() {
        let s = "café ❤😀";
        assert_eq!(fnv1a_32_str(s), fnv1a_32(s.as_bytes()));
        assert_eq!(fnv1a_64_str(s), fnv1a_64(s.as_bytes()));
    }

    // --- combine single-byte advance matches batch ---

    #[test]
    fn combine_32_matches_oneshot() {
        let data = b"combine-check";
        let mut hash = FNV1A_32_OFFSET_BASIS;
        for &b in data {
            hash = fnv1a_32_combine(hash, b);
        }
        assert_eq!(hash, fnv1a_32(data));
    }

    #[test]
    fn combine_64_matches_oneshot() {
        let data = b"combine-check";
        let mut hash = FNV1A_64_OFFSET_BASIS;
        for &b in data {
            hash = fnv1a_64_combine(hash, b);
        }
        assert_eq!(hash, fnv1a_64(data));
    }

    // --- Avalanche: a single bit change alters the digest ---

    #[test]
    fn avalanche_32_single_bit_flip_differs() {
        let base = fnv1a_32(b"avalanche");
        let mut flipped = *b"avalanche";
        flipped[0] ^= 0x01;
        assert_ne!(base, fnv1a_32(&flipped));
    }

    #[test]
    fn avalanche_64_single_bit_flip_differs() {
        let base = fnv1a_64(b"avalanche");
        let mut flipped = *b"avalanche";
        flipped[0] ^= 0x01;
        assert_ne!(base, fnv1a_64(&flipped));
    }

    #[test]
    fn order_sensitivity_32() {
        assert_ne!(fnv1a_32(b"ab"), fnv1a_32(b"ba"));
    }

    #[test]
    fn order_sensitivity_64() {
        assert_ne!(fnv1a_64(b"ab"), fnv1a_64(b"ba"));
    }

    #[test]
    fn appending_zero_changes_result() {
        assert_ne!(fnv1a_32(b"data"), fnv1a_32(b"data\0"));
        assert_ne!(fnv1a_64(b"data"), fnv1a_64(b"data\0"));
    }

    #[test]
    fn distinct_inputs_mostly_differ() {
        let mut lcg = Lcg::new(0x0F0F_0F0F_1234_5678);
        let mut seen32 = alloc::collections::BTreeSet::new();
        let mut seen64 = alloc::collections::BTreeSet::new();
        for _ in 0..512 {
            let data = lcg.bytes(16);
            seen32.insert(fnv1a_32(&data));
            seen64.insert(fnv1a_64(&data));
        }
        // No 64-bit collisions expected, and very few (if any) 32-bit ones.
        assert_eq!(seen64.len(), 512);
        assert!(seen32.len() >= 510);
    }

    #[test]
    fn repeated_calls_are_pure() {
        assert_eq!(fnv1a_32(b"idempotent"), fnv1a_32(b"idempotent"));
        assert_eq!(fnv1a_64(b"idempotent"), fnv1a_64(b"idempotent"));
    }

    #[test]
    fn combine_32_from_basis_single_byte() {
        assert_eq!(
            fnv1a_32_combine(FNV1A_32_OFFSET_BASIS, b'a'),
            fnv1a_32(b"a")
        );
    }

    #[test]
    fn combine_64_from_basis_single_byte() {
        assert_eq!(
            fnv1a_64_combine(FNV1A_64_OFFSET_BASIS, b'a'),
            fnv1a_64(b"a")
        );
    }

    #[test]
    fn long_uniform_data_stream_matches_oneshot() {
        let data = alloc::vec![0x5Au8; 100_000];
        let oneshot = fnv1a_64(&data);
        let mut h = fnv1a64_new();
        for chunk in data.chunks(4096) {
            h.update(chunk);
        }
        assert_eq!(h.finish(), oneshot);
    }
}
