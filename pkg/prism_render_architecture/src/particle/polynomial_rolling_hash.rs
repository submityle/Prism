//! Polynomial rolling hash (`Rabin-Karp` style) over a `Mersenne`-prime
//! modulus, used for cheap content fingerprinting and substring matching of
//! streamed particle payloads (design § integrity and dedup helpers).
//!
//! The hash treats the input bytes as the coefficients of a polynomial
//! evaluated at a fixed base. For bytes `b[0..n]` the value is
//! `sum(b[i] * BASE^(n-1-i)) mod MOD`, computed with `Horner`'s method so the
//! running value only needs one multiply and one add per byte. The modulus is
//! the `Mersenne` prime `2^61 - 1` (exposed as [`MOD`]); every intermediate
//! product is widened to `u128` before the remainder so the `multiply`-then-add
//! step can never overflow, and the reduced result always fits in a `u64` that
//! is strictly less than [`MOD`].
//!
//! The one-shot [`rolling_hash`] matches the streaming [`RollingHash`]
//! accumulator byte for byte. [`RollingHash`] additionally tracks `BASE^len mod
//! MOD` so that a fixed-width window can be advanced with [`RollingHash::roll`]:
//! the oldest byte (which carries the highest power) is subtracted out, the
//! value is shifted up by one base, and the incoming byte is folded in. The
//! modular exponent needed for the subtraction is produced by [`pow_base`],
//! a square-and-multiply fast-power routine using only integer arithmetic.
//!
//! Scope: this is a non-cryptographic fingerprint. Collisions can be
//! constructed on purpose, so it must never be used to authenticate data or
//! guard against a malicious adversary; use a real hash for security.

/// The hash modulus: the `Mersenne` prime `2^61 - 1`.
pub const MOD: u64 = 2_305_843_009_213_693_951;

/// The polynomial evaluation base.
pub const BASE: u64 = 257;

/// Computes the polynomial rolling hash of `data` over [`MOD`].
///
/// Uses `Horner`'s method: each byte folds in as `(hash * BASE + byte) mod
/// MOD`. The empty slice hashes to `0`.
pub fn rolling_hash(data: &[u8]) -> u64 {
    let mut h: u64 = 0;
    for &b in data {
        h = (((h as u128) * (BASE as u128) + (b as u128)) % (MOD as u128)) as u64;
    }
    h
}

/// Returns `BASE^n mod MOD` using square-and-multiply (fast power).
///
/// `pow_base(0)` is `1`; every intermediate product is widened to `u128` so
/// the squaring step cannot overflow.
pub fn pow_base(n: usize) -> u64 {
    let mut result: u64 = 1;
    let mut base: u64 = BASE % MOD;
    let mut exp = n;
    while exp > 0 {
        if (exp & 1) == 1 {
            result = (((result as u128) * (base as u128)) % (MOD as u128)) as u64;
        }
        base = (((base as u128) * (base as u128)) % (MOD as u128)) as u64;
        exp >>= 1;
    }
    result
}

/// Incremental polynomial rolling-hash accumulator over [`MOD`].
///
/// Feeding bytes with [`RollingHash::push`] reproduces [`rolling_hash`] exactly.
/// The accumulator also tracks `BASE^len mod MOD` so a fixed-width window can be
/// advanced with [`RollingHash::roll`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RollingHash {
    hash: u64,
    pow: u64,
    len: usize,
}

impl RollingHash {
    /// Creates an empty accumulator (`hash = 0`, `pow = BASE^0 = 1`, `len = 0`).
    pub fn new() -> Self {
        Self {
            hash: 0,
            pow: 1,
            len: 0,
        }
    }

    /// Folds one byte into the hash: `hash = (hash * BASE + b) mod MOD`.
    ///
    /// Also advances `pow = BASE^len mod MOD` and increments `len`.
    pub fn push(&mut self, b: u8) {
        self.hash = (((self.hash as u128) * (BASE as u128) + (b as u128)) % (MOD as u128)) as u64;
        self.pow = (((self.pow as u128) * (BASE as u128)) % (MOD as u128)) as u64;
        self.len += 1;
    }

    /// Returns the current hash value (always strictly less than [`MOD`]).
    pub fn hash(&self) -> u64 {
        self.hash
    }

    /// Returns the number of bytes folded in so far.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` when no bytes have been folded in yet.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns `BASE^len mod MOD`, the weight of the next incoming byte's slot.
    pub fn pow(&self) -> u64 {
        self.pow
    }

    /// Advances a fixed-width window by dropping `old_byte` (the highest-power
    /// byte) and folding in `new_byte`, keeping `len` unchanged.
    ///
    /// The result equals [`rolling_hash`] of the shifted window. Rolling an
    /// empty window just folds `new_byte` into a still-empty state.
    pub fn roll(&self, old_byte: u8, new_byte: u8) -> RollingHash {
        let modulus = MOD as u128;
        let high = pow_base(self.len.saturating_sub(1)) as u128;
        let drop = ((old_byte as u128) * high) % modulus;
        let without_old = ((self.hash as u128) + modulus - drop) % modulus;
        let hash = ((without_old * (BASE as u128) + (new_byte as u128)) % modulus) as u64;
        RollingHash {
            hash,
            pow: self.pow,
            len: self.len,
        }
    }
}

impl Default for RollingHash {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference helper: naive repeated multiply for pow_base cross-checks.
    #[cfg(test)]
    fn naive_pow(n: usize) -> u64 {
        let mut acc: u64 = 1;
        for _ in 0..n {
            acc = (((acc as u128) * (BASE as u128)) % (MOD as u128)) as u64;
        }
        acc
    }

    #[test]
    fn vector_empty() {
        assert_eq!(rolling_hash(b""), 0);
    }

    #[test]
    fn vector_single_a() {
        assert_eq!(rolling_hash(b"a"), 97);
    }

    #[test]
    fn vector_abc() {
        assert_eq!(rolling_hash(b"abc"), 6432038);
    }

    #[test]
    fn vector_hello() {
        assert_eq!(rolling_hash(b"hello"), 455418516756);
    }

    #[test]
    fn single_byte_equals_byte_value() {
        for b in 0u8..=255 {
            assert_eq!(rolling_hash(&[b]), b as u64);
        }
    }

    #[test]
    fn empty_is_zero_again() {
        let data: [u8; 0] = [];
        assert_eq!(rolling_hash(&data), 0);
    }

    #[test]
    fn result_always_below_mod() {
        let samples: [&[u8]; 6] = [
            b"",
            b"a",
            b"abc",
            b"hello",
            b"the quick brown fox",
            &[255u8; 64],
        ];
        for s in samples {
            assert!(rolling_hash(s) < MOD);
        }
    }

    #[test]
    fn result_below_mod_many_bytes() {
        let big = [200u8; 1000];
        assert!(rolling_hash(&big) < MOD);
    }

    #[test]
    fn deterministic() {
        let data = b"deterministic payload";
        assert_eq!(rolling_hash(data), rolling_hash(data));
    }

    #[test]
    fn deterministic_repeated() {
        let data = b"\x00\x01\x02\x03\xff\xfe";
        let first = rolling_hash(data);
        for _ in 0..32 {
            assert_eq!(rolling_hash(data), first);
        }
    }

    #[test]
    fn different_inputs_differ_sampling() {
        let inputs: [&[u8]; 8] = [
            b"alpha", b"alphb", b"beta", b"gamma", b"", b"a", b"aa", b"aaa",
        ];
        for i in 0..inputs.len() {
            for j in (i + 1)..inputs.len() {
                assert_ne!(
                    rolling_hash(inputs[i]),
                    rolling_hash(inputs[j]),
                    "unexpected collision between samples {i} and {j}"
                );
            }
        }
    }

    #[test]
    fn single_byte_change_differs() {
        assert_ne!(rolling_hash(b"abcd"), rolling_hash(b"abce"));
    }

    #[test]
    fn order_matters() {
        assert_ne!(rolling_hash(b"ab"), rolling_hash(b"ba"));
    }

    #[test]
    fn push_matches_oneshot_empty() {
        let rh = RollingHash::new();
        assert_eq!(rh.hash(), rolling_hash(b""));
    }

    #[test]
    fn push_matches_oneshot_single() {
        let mut rh = RollingHash::new();
        rh.push(b'a');
        assert_eq!(rh.hash(), rolling_hash(b"a"));
    }

    #[test]
    fn push_matches_oneshot_abc() {
        let mut rh = RollingHash::new();
        for &b in b"abc" {
            rh.push(b);
        }
        assert_eq!(rh.hash(), rolling_hash(b"abc"));
    }

    #[test]
    fn push_matches_oneshot_hello() {
        let mut rh = RollingHash::new();
        for &b in b"hello" {
            rh.push(b);
        }
        assert_eq!(rh.hash(), rolling_hash(b"hello"));
    }

    #[test]
    fn push_matches_oneshot_long() {
        let data = b"The quick brown fox jumps over the lazy dog.";
        let mut rh = RollingHash::new();
        for &b in data {
            rh.push(b);
        }
        assert_eq!(rh.hash(), rolling_hash(data));
    }

    #[test]
    fn push_matches_oneshot_all_bytes() {
        let data: [u8; 256] = core::array::from_fn(|i| i as u8);
        let mut rh = RollingHash::new();
        for &b in data.iter() {
            rh.push(b);
        }
        assert_eq!(rh.hash(), rolling_hash(&data));
    }

    #[test]
    fn push_tracks_len() {
        let mut rh = RollingHash::new();
        assert_eq!(rh.len(), 0);
        rh.push(1);
        rh.push(2);
        rh.push(3);
        assert_eq!(rh.len(), 3);
    }

    #[test]
    fn new_is_empty() {
        let rh = RollingHash::new();
        assert!(rh.is_empty());
    }

    #[test]
    fn not_empty_after_push() {
        let mut rh = RollingHash::new();
        rh.push(42);
        assert!(!rh.is_empty());
    }

    #[test]
    fn default_matches_new() {
        assert_eq!(RollingHash::default(), RollingHash::new());
    }

    #[test]
    fn push_hash_always_below_mod() {
        let mut rh = RollingHash::new();
        for b in 0u8..=255 {
            rh.push(b);
            assert!(rh.hash() < MOD);
        }
    }

    #[test]
    fn pow_tracks_powers() {
        let mut rh = RollingHash::new();
        assert_eq!(rh.pow(), pow_base(0));
        rh.push(0);
        assert_eq!(rh.pow(), pow_base(1));
        rh.push(0);
        assert_eq!(rh.pow(), pow_base(2));
        rh.push(0);
        assert_eq!(rh.pow(), pow_base(3));
    }

    #[test]
    fn pow_base_zero() {
        assert_eq!(pow_base(0), 1);
    }

    #[test]
    fn pow_base_one() {
        assert_eq!(pow_base(1), 257);
    }

    #[test]
    fn pow_base_two() {
        assert_eq!(pow_base(2), (257u64 * 257) % MOD);
    }

    #[test]
    fn pow_base_matches_naive() {
        for n in 0..64 {
            assert_eq!(pow_base(n), naive_pow(n), "mismatch at exponent {n}");
        }
    }

    #[test]
    fn pow_base_matches_naive_large() {
        for n in [100usize, 255, 500, 1000, 2048] {
            assert_eq!(pow_base(n), naive_pow(n), "mismatch at exponent {n}");
        }
    }

    #[test]
    fn pow_base_always_below_mod() {
        for n in 0..100 {
            assert!(pow_base(n) < MOD);
        }
    }

    #[test]
    fn roll_window_len2_matches_substrings() {
        let data = b"abcd";
        let window = 2usize;
        let mut rh = RollingHash::new();
        for &b in &data[..window] {
            rh.push(b);
        }
        assert_eq!(rh.hash(), rolling_hash(&data[0..window]));
        for start in 1..=(data.len() - window) {
            rh = rh.roll(data[start - 1], data[start + window - 1]);
            assert_eq!(
                rh.hash(),
                rolling_hash(&data[start..start + window]),
                "window at start {start}"
            );
        }
    }

    #[test]
    fn roll_window_len3_matches_substrings() {
        let data = b"abcdefg";
        let window = 3usize;
        let mut rh = RollingHash::new();
        for &b in &data[..window] {
            rh.push(b);
        }
        assert_eq!(rh.hash(), rolling_hash(&data[0..window]));
        for start in 1..=(data.len() - window) {
            rh = rh.roll(data[start - 1], data[start + window - 1]);
            assert_eq!(
                rh.hash(),
                rolling_hash(&data[start..start + window]),
                "window at start {start}"
            );
        }
    }

    #[test]
    fn roll_window_long_text() {
        let data = b"the quick brown fox jumps over the lazy dog";
        let window = 5usize;
        let mut rh = RollingHash::new();
        for &b in &data[..window] {
            rh.push(b);
        }
        for start in 1..=(data.len() - window) {
            rh = rh.roll(data[start - 1], data[start + window - 1]);
            assert_eq!(
                rh.hash(),
                rolling_hash(&data[start..start + window]),
                "window at start {start}"
            );
        }
    }

    #[test]
    fn roll_preserves_len_and_pow() {
        let data = b"abcde";
        let window = 2usize;
        let mut rh = RollingHash::new();
        for &b in &data[..window] {
            rh.push(b);
        }
        let len_before = rh.len();
        let pow_before = rh.pow();
        let rolled = rh.roll(data[0], data[window]);
        assert_eq!(rolled.len(), len_before);
        assert_eq!(rolled.pow(), pow_before);
    }

    #[test]
    fn roll_result_below_mod() {
        let data = [255u8; 8];
        let window = 4usize;
        let mut rh = RollingHash::new();
        for &b in &data[..window] {
            rh.push(b);
        }
        for start in 1..=(data.len() - window) {
            rh = rh.roll(data[start - 1], data[start + window - 1]);
            assert!(rh.hash() < MOD);
        }
    }

    #[test]
    fn mod_is_mersenne_prime() {
        assert_eq!(MOD, (1u64 << 61) - 1);
    }

    #[test]
    fn base_value() {
        assert_eq!(BASE, 257);
    }

    #[test]
    fn high_bytes_do_not_overflow() {
        // All-0xFF input exercises the widest intermediate products.
        let data = [0xffu8; 128];
        let direct = rolling_hash(&data);
        let mut rh = RollingHash::new();
        for &b in data.iter() {
            rh.push(b);
        }
        assert_eq!(direct, rh.hash());
        assert!(direct < MOD);
    }

    #[test]
    fn window_full_length_equals_full_hash() {
        let data = b"abcd";
        let mut rh = RollingHash::new();
        for &b in data {
            rh.push(b);
        }
        assert_eq!(rh.hash(), rolling_hash(data));
    }
}
