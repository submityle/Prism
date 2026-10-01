//! `CRC-7`/`UMTS`: the `7`-bit cyclic-redundancy check used by `UMTS`/`3GPP`
//! framing (design § integrity checks).
//!
//! `CRC-7`/`UMTS` treats the input bytes as the coefficients of a polynomial
//! over the binary field `GF(2)` and returns the `7`-bit remainder after
//! dividing by the fixed generator polynomial `x^7 + x^6 + x^2 + 1`, written
//! `0x45` in unreflected (most-significant-bit-first) form. The arithmetic is
//! carry-less: "addition" is exclusive-or (`XOR`) and there are no carries, so
//! the whole computation reduces to shifts, masks, and `XOR`s. The register is
//! masked to `7` bits (`0x7F`) after every shift.
//!
//! Parameters (the standard parametric `CRC` model): `width` `7`, `poly`
//! `0x45`, `init` `0x00`, no input reflection (`refin` `false`), no output
//! reflection (`refout` `false`), and final exclusive-or (`XOR` out) `0x00`.
//! The canonical *check* value for the `ASCII` string `123456789` is `0x61`
//! (exposed as [`CHECK`]); this is asserted directly in the tests.
//!
//! This module exposes a one-shot [`crc7_umts`] function and an incremental
//! [`Crc7Umts`] register for folding data in chunks. Both compute the identical
//! `7`-bit value, so a host `CPU`, a `GPU` upload path, or an asset-`hex` tool
//! can share either entry point.
//!
//! Scope and boundaries. This is a `7`-bit `CRC`: a `GF(2)` polynomial-modulo
//! error-detection code over a `7`-bit register. It is deliberately independent
//! of the neighbouring integrity primitives in this crate and does not
//! reference them. Wider `CRC`s (for example `CRC-16`, `CRC-32`, `CRC-64`) use
//! different generator polynomials and register widths; they share the same
//! carry-less `GF(2)` algebra but are separate codes, and a `7`-bit remainder
//! is weaker, suiting only the short control words `UMTS` framing uses.
//!
//! None of these are cryptographic. A `CRC-7` is trivially invertible and
//! collisions are easy to craft on purpose, so it must only guard against
//! accidental corruption, never against a malicious adversary.

/// Register width in bits.
const WIDTH: u32 = 7;

/// Generator polynomial `0x45` (`x^7 + x^6 + x^2 + 1`), most-significant-bit-first
/// (unreflected) form, with the implicit `x^7` term dropped.
const POLY: u32 = 0x45;

/// Initial register value before any bytes are folded in.
const INIT: u32 = 0x00;

/// Final exclusive-or (`XOR` out) applied to the register.
const XOROUT: u32 = 0x00;

/// Low-`7`-bit mask applied to the register after every shift.
const MASK: u32 = 0x7F;

/// Number of bits in one input byte.
const BITS_PER_BYTE: u32 = 8;

/// The canonical *check* value: `crc7_umts(b"123456789")`.
pub const CHECK: u8 = 0x61;

/// Folds a single byte into the running register, most-significant-bit-first.
///
/// Shared by the one-shot [`crc7_umts`] and the incremental [`Crc7Umts`] so both
/// paths provably produce the same `7`-bit value. Every step is an integer
/// shift, mask, or exclusive-or (`XOR`).
#[must_use]
const fn fold_byte(mut reg: u32, byte: u8) -> u32 {
    let b = byte as u32;
    let mut i = 0u32;
    while i < BITS_PER_BYTE {
        let bit = (b >> (7 - i)) & 1;
        let msb = (reg >> (WIDTH - 1)) & 1;
        reg = (reg << 1) & MASK;
        if (msb ^ bit) != 0 {
            reg ^= POLY;
        }
        i += 1;
    }
    reg
}

/// One-shot `CRC-7`/`UMTS` over `data`, returning the low `7` bits.
///
/// Processes each byte most-significant-bit-first through the bit-by-bit
/// polynomial-division reference. The register starts at `0x00` ([`INIT`]), is
/// masked to `7` bits (`0x7F`) after each shift, and there is no final
/// exclusive-or. The empty slice yields `0x00`.
#[must_use]
pub fn crc7_umts(data: &[u8]) -> u8 {
    let mut reg = INIT & MASK;
    let mut idx = 0usize;
    while idx < data.len() {
        reg = fold_byte(reg, data[idx]);
        idx += 1;
    }
    ((reg ^ XOROUT) & MASK) as u8
}

/// Incremental `CRC-7`/`UMTS` register.
///
/// Folds data in arbitrary chunks and yields the same `7`-bit value as the
/// one-shot [`crc7_umts`] over the concatenation of those chunks. Construct with
/// [`Crc7Umts::new`], feed bytes with [`Crc7Umts::update`], and read the result
/// with [`Crc7Umts::finalize`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Crc7Umts {
    state: u32,
}

impl Crc7Umts {
    /// Creates a fresh register initialised to [`INIT`].
    #[must_use]
    pub const fn new() -> Self {
        Self { state: INIT & MASK }
    }

    /// Folds `data` into the register, most-significant-bit-first.
    pub const fn update(&mut self, data: &[u8]) {
        let mut reg = self.state;
        let mut idx = 0usize;
        while idx < data.len() {
            reg = fold_byte(reg, data[idx]);
            idx += 1;
        }
        self.state = reg;
    }

    /// Returns the current `7`-bit `CRC` value, applying the final exclusive-or.
    #[must_use]
    pub const fn finalize(&self) -> u8 {
        ((self.state ^ XOROUT) & MASK) as u8
    }
}

#[cfg(test)]
mod tests {
    use super::{crc7_umts, Crc7Umts, CHECK, MASK};

    /// Upper bound of a valid `7`-bit result, as a `u8`.
    const SEVEN_BIT_MAX: u8 = MASK as u8;

    /// Length used for the long repeated-`0xA5` reference vector.
    const LONG_LEN: usize = 1000;

    /// Length used for the full-byte-range reference vector.
    const ALL_BYTES_LEN: usize = 256;

    #[test]
    fn vec_empty() {
        assert!(crc7_umts(b"") == 0x00);
    }

    #[test]
    fn vec_z00() {
        assert!(crc7_umts(&[0x00]) == 0x00);
    }

    #[test]
    fn vec_ff() {
        assert!(crc7_umts(&[0xff]) == 0x59);
    }

    #[test]
    fn vec_a() {
        assert!(crc7_umts(b"a") == 0x4a);
    }

    #[test]
    fn vec_b() {
        assert!(crc7_umts(b"b") == 0x40);
    }

    #[test]
    fn vec_ab() {
        assert!(crc7_umts(b"ab") == 0x6f);
    }

    #[test]
    fn vec_abc() {
        assert!(crc7_umts(b"abc") == 0x5f);
    }

    #[test]
    fn vec_two_zeros() {
        assert!(crc7_umts(&[0x00, 0x00]) == 0x00);
    }

    #[test]
    fn vec_01() {
        assert!(crc7_umts(&[0x01]) == 0x45);
    }

    #[test]
    fn vec_02() {
        assert!(crc7_umts(&[0x02]) == 0x4f);
    }

    #[test]
    fn vec_7f() {
        assert!(crc7_umts(&[0x7f]) == 0x0e);
    }

    #[test]
    fn vec_80() {
        assert!(crc7_umts(&[0x80]) == 0x57);
    }

    #[test]
    fn vec_aa_55() {
        assert!(crc7_umts(&[0xaa, 0x55]) == 0x22);
    }

    #[test]
    fn vec_55_aa() {
        assert!(crc7_umts(&[0x55, 0xaa]) == 0x06);
    }

    #[test]
    fn vec_deadbeef() {
        assert!(crc7_umts(&[0xde, 0xad, 0xbe, 0xef]) == 0x77);
    }

    #[test]
    fn vec_hello() {
        assert!(crc7_umts(b"Hello") == 0x6f);
    }

    #[test]
    fn vec_fox() {
        assert!(crc7_umts(b"The quick brown fox") == 0x5e);
    }

    #[test]
    fn vec_four_zeros() {
        assert!(crc7_umts(&[0u8; 4]) == 0x00);
    }

    #[test]
    fn vec_four_ff() {
        assert!(crc7_umts(&[0xffu8; 4]) == 0x53);
    }

    #[test]
    fn vec_0_15() {
        let mut buf = [0u8; 16];
        let mut i = 0usize;
        while i < 16 {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc7_umts(&buf) == 0x6a);
    }

    #[test]
    fn vec_12345678_bytes() {
        assert!(crc7_umts(&[0x12, 0x34, 0x56, 0x78]) == 0x3e);
    }

    #[test]
    fn vec_check_constant() {
        assert!(crc7_umts(b"123456789") == 0x61);
    }

    #[test]
    fn check_matches_exported_constant() {
        assert!(crc7_umts(b"123456789") == CHECK);
    }

    #[test]
    fn vec_a5_1000() {
        let mut buf = [0u8; LONG_LEN];
        let mut i = 0usize;
        while i < LONG_LEN {
            buf[i] = 0xA5;
            i += 1;
        }
        assert!(crc7_umts(&buf) == 0x1b);
    }

    #[test]
    fn vec_all256() {
        let mut buf = [0u8; ALL_BYTES_LEN];
        let mut i = 0usize;
        while i < ALL_BYTES_LEN {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc7_umts(&buf) == 0x22);
    }

    #[test]
    fn determinism_same_input_twice() {
        let input = b"The quick brown fox";
        assert!(crc7_umts(input) == crc7_umts(input));
    }

    #[test]
    fn a_differs_from_b() {
        assert!(crc7_umts(b"a") != crc7_umts(b"b"));
    }

    #[test]
    fn prefix_differs_abc_vs_ab() {
        assert!(crc7_umts(b"abc") != crc7_umts(b"ab"));
    }

    #[test]
    fn length_sensitive_repeated_byte() {
        assert!(crc7_umts(&[0x01]) != crc7_umts(&[0x01, 0x01]));
    }

    #[test]
    fn single_byte_samples_pairwise_distinct() {
        let samples: [u8; 8] = [0x00, 0x01, 0x02, 0x80, 0x7f, 0xff, 0x55, 0xaa];
        let mut i = 0usize;
        while i < samples.len() {
            let a = crc7_umts(&[samples[i]]);
            let mut j = i + 1;
            while j < samples.len() {
                let b = crc7_umts(&[samples[j]]);
                assert!(a != b);
                j += 1;
            }
            i += 1;
        }
    }

    #[test]
    fn result_always_within_7_bits() {
        let mut byte = 0u32;
        while byte < 256 {
            let r = crc7_umts(&[byte as u8]);
            assert!((0u8..=SEVEN_BIT_MAX).contains(&r));
            byte += 1;
        }
    }

    #[test]
    fn empty_register_finalizes_to_zero() {
        let reg = Crc7Umts::new();
        assert!(reg.finalize() == 0x00);
    }

    #[test]
    fn incremental_matches_oneshot_whole() {
        let input = b"The quick brown fox";
        let mut reg = Crc7Umts::new();
        reg.update(input);
        assert!(reg.finalize() == crc7_umts(input));
    }

    #[test]
    fn incremental_split_matches_oneshot() {
        let input = b"123456789";
        let mut reg = Crc7Umts::new();
        reg.update(&input[0..4]);
        reg.update(&input[4..9]);
        assert!(reg.finalize() == crc7_umts(input));
    }

    #[test]
    fn incremental_byte_by_byte_matches_oneshot() {
        let input = b"The quick brown fox";
        let mut reg = Crc7Umts::new();
        let mut i = 0usize;
        while i < input.len() {
            reg.update(&input[i..i + 1]);
            i += 1;
        }
        assert!(reg.finalize() == crc7_umts(input));
    }

    #[test]
    fn incremental_empty_updates_are_noops() {
        let input = b"abc";
        let mut reg = Crc7Umts::new();
        reg.update(b"");
        reg.update(input);
        reg.update(b"");
        assert!(reg.finalize() == crc7_umts(input));
    }

    #[test]
    fn default_register_equals_new() {
        assert!(Crc7Umts::default() == Crc7Umts::new());
    }

    #[test]
    fn check_constant_equals_spec_value() {
        assert!(CHECK == 0x61);
    }
}
