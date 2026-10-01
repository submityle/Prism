//! `CRC-8` cyclic-redundancy checks in several standard parameterisations:
//! pure-integer, bit-by-bit (plus optional table-driven) polynomial division
//! over `GF(2)` for guarding single bytes, framing nibbles, and tiny
//! `SMBUS`/1-Wire style records (design § integrity checks).
//!
//! A `CRC-8` treats the input bytes as the coefficients of a polynomial over
//! the binary field `GF(2)` and returns the `8`-bit remainder after dividing by a
//! fixed generator polynomial. The arithmetic is carry-less: "addition" is
//! exclusive-or and there are no carries, so the whole computation is just
//! shifts, masks, exclusive-ors, and bit reversals. This module implements four
//! widely used `8`-bit variants plus a generic bit-by-bit driver underneath:
//!
//! - `CRC-8`/`SMBUS`: `poly` `0x07`, `init` `0x00`, no input/output
//!   reflection, final exclusive-or `0x00`.
//! - `CRC-8`/`MAXIM-DOW` (Dallas/1-Wire): `poly` `0x31` (reflected form
//!   `0x8C`), `init` `0x00`, reflected input and output, final exclusive-or
//!   `0x00`.
//! - `CRC-8`/`BLUETOOTH`: `poly` `0xA7` (reflected form `0xE5`), `init`
//!   `0x00`, reflected input and output, final exclusive-or `0x00`.
//! - `CRC-8`/`DVB-S2`: `poly` `0xD5`, `init` `0x00`, no reflection, final
//!   exclusive-or `0x00`.
//!
//! The canonical *check* values for the `ASCII` string `123456789` are
//! `SMBUS` `0xF4`, `MAXIM-DOW` `0xA1`, `BLUETOOTH` `0x26`, and `DVB-S2`
//! `0xBC`; these are asserted directly in the tests.
//!
//! Scope and boundaries. This is an `8`-bit `CRC`: a `GF(2)` polynomial-modulo
//! error-detection code over a single-byte register. It is deliberately
//! distinct from the neighbouring integrity primitives in this crate, which
//! differ in both bit width and the underlying algebra:
//!
//! - `crc16_ccitt` is a `16`-bit `CRC` and `crc32` is a `32`-bit `CRC`:
//!   wider registers and different generator polynomials, though the same
//!   carry-less `GF(2)` polynomial-division algebra. An `8`-bit remainder is
//!   much weaker and only suits very short messages.
//! - `fletcher_checksum` and `adler32` are *not* `CRC`s at all: they are
//!   position-weighted sums computed with ordinary integer modular addition
//!   (running sums reduced modulo a constant), not carry-less polynomial
//!   division. Different algebra, different error profile.
//!
//! None of these are cryptographic. A `CRC-8` is trivially invertible and
//! collisions are easy to craft on purpose, so it must only guard against
//! accidental corruption, never against a malicious adversary.

/// Generator polynomial `0x07` (`SMBUS`), most-significant-bit-first
/// (unreflected) form.
const POLY_07: u8 = 0x07;

/// Generator polynomial `0xD5` (`DVB-S2`), unreflected form.
const POLY_D5: u8 = 0xD5;

/// Generator polynomial `0x31` (`MAXIM-DOW`), unreflected form. Used by the
/// generic bit-by-bit driver with input/output reflection enabled.
#[cfg(test)]
const POLY_31: u8 = 0x31;

/// Generator polynomial `0xA7` (`BLUETOOTH`), unreflected form. Used by the
/// generic bit-by-bit driver with input/output reflection enabled.
#[cfg(test)]
const POLY_A7: u8 = 0xA7;

/// Reflected form of polynomial `0x31`, i.e. `0x8C`, used by the
/// least-significant-bit-first (reflected) table driver.
const POLY_31_REFLECTED: u8 = 0x8C;

/// Reflected form of polynomial `0xA7`, i.e. `0xE5`, used by the
/// least-significant-bit-first (reflected) table driver.
const POLY_A7_REFLECTED: u8 = 0xE5;

/// Generic bit-by-bit `CRC-8` driver (the low-level primitive).
///
/// Computes the `8`-bit remainder for arbitrary parameters: generator `poly`
/// in unreflected form, initial register `init`, input reflection `refin`,
/// output reflection `refout`, and final exclusive-or `xorout`. When `refin`
/// is set each input byte is bit-reversed before being folded in; when `refout`
/// is set the final register is bit-reversed before the exclusive-or. This
/// matches the standard parametric `CRC` model and is the authority the
/// table-driven variants are checked against.
pub fn crc8(data: &[u8], poly: u8, init: u8, refin: bool, refout: bool, xorout: u8) -> u8 {
    let mut crc = init;
    for &byte in data {
        let folded = if refin { byte.reverse_bits() } else { byte };
        crc ^= folded;
        for _ in 0..8 {
            if (crc & 0x80) != 0 {
                crc = (crc << 1) ^ poly;
            } else {
                crc <<= 1;
            }
        }
    }
    if refout {
        crc = crc.reverse_bits();
    }
    crc ^ xorout
}

/// Builds the `256`-entry most-significant-bit-first lookup table for an
/// unreflected generator `poly`.
///
/// Entry `i` holds the register value produced by folding the byte `i` into an
/// otherwise-zero register.
fn build_table_msb(poly: u8) -> [u8; 256] {
    let mut table = [0u8; 256];
    for (i, slot) in table.iter_mut().enumerate() {
        let mut crc = i as u8;
        for _ in 0..8 {
            if (crc & 0x80) != 0 {
                crc = (crc << 1) ^ poly;
            } else {
                crc <<= 1;
            }
        }
        *slot = crc;
    }
    table
}

/// Builds the `256`-entry least-significant-bit-first lookup table for a
/// reflected generator `poly_reflected` (for example `0x8C`).
///
/// Entry `i` holds the register value produced by folding the byte `i` into an
/// otherwise-zero register, processing bits from the least significant end.
fn build_table_lsb(poly_reflected: u8) -> [u8; 256] {
    let mut table = [0u8; 256];
    for (i, slot) in table.iter_mut().enumerate() {
        let mut crc = i as u8;
        for _ in 0..8 {
            if (crc & 1) != 0 {
                crc = (crc >> 1) ^ poly_reflected;
            } else {
                crc >>= 1;
            }
        }
        *slot = crc;
    }
    table
}

/// Table-driven most-significant-bit-first `CRC-8` (no reflection).
///
/// Used by the unreflected variants (`SMBUS`, `DVB-S2`). The generator `poly`
/// is given in unreflected form. Because the register is exactly one byte wide,
/// each step simply indexes the table by the exclusive-or of the register and
/// the next byte.
fn crc8_table_msb(data: &[u8], poly: u8, init: u8, xorout: u8) -> u8 {
    let table = build_table_msb(poly);
    let mut crc = init;
    for &byte in data {
        let index = crc ^ byte;
        crc = table[index as usize];
    }
    crc ^ xorout
}

/// Table-driven least-significant-bit-first `CRC-8` (reflected input/output).
///
/// Used by the reflected variants (`MAXIM-DOW`, `BLUETOOTH`). The generator
/// `poly_reflected` is given in reflected form (for example `0x8C`); input and
/// output reflection are intrinsic to this least-significant-bit-first driver,
/// so no explicit bit-reversal is needed.
fn crc8_table_lsb(data: &[u8], poly_reflected: u8, init: u8, xorout: u8) -> u8 {
    let table = build_table_lsb(poly_reflected);
    let mut crc = init;
    for &byte in data {
        let index = crc ^ byte;
        crc = table[index as usize];
    }
    crc ^ xorout
}

/// `CRC-8`/`SMBUS`: `poly` `0x07`, `init` `0x00`, no reflection, final
/// exclusive-or `0x00`. Check value for `123456789` is `0xF4`.
pub fn crc8_smbus(data: &[u8]) -> u8 {
    crc8_table_msb(data, POLY_07, 0x00, 0x00)
}

/// `CRC-8`/`MAXIM-DOW` (Dallas/1-Wire): `poly` `0x31` (reflected `0x8C`),
/// `init` `0x00`, reflected input and output. Check value for `123456789` is
/// `0xA1`.
pub fn crc8_maxim(data: &[u8]) -> u8 {
    crc8_table_lsb(data, POLY_31_REFLECTED, 0x00, 0x00)
}

/// `CRC-8`/`BLUETOOTH`: `poly` `0xA7` (reflected `0xE5`), `init` `0x00`,
/// reflected input and output. Check value for `123456789` is `0x26`.
pub fn crc8_bluetooth(data: &[u8]) -> u8 {
    crc8_table_lsb(data, POLY_A7_REFLECTED, 0x00, 0x00)
}

/// `CRC-8`/`DVB-S2`: `poly` `0xD5`, `init` `0x00`, no reflection, final
/// exclusive-or `0x00`. Check value for `123456789` is `0xBC`.
pub fn crc8_dvb_s2(data: &[u8]) -> u8 {
    crc8_table_msb(data, POLY_D5, 0x00, 0x00)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The canonical `CRC` check string.
    const CHECK: &[u8] = b"123456789";

    // --- Authoritative check values for `123456789` (table-driven) ---

    #[test]
    fn smbus_check_vector() {
        assert_eq!(crc8_smbus(CHECK), 0xF4);
    }

    #[test]
    fn maxim_check_vector() {
        assert_eq!(crc8_maxim(CHECK), 0xA1);
    }

    #[test]
    fn bluetooth_check_vector() {
        assert_eq!(crc8_bluetooth(CHECK), 0x26);
    }

    #[test]
    fn dvb_s2_check_vector() {
        assert_eq!(crc8_dvb_s2(CHECK), 0xBC);
    }

    // --- Same check values via the generic bit-by-bit driver ---

    #[test]
    fn smbus_check_vector_bitwise() {
        assert_eq!(crc8(CHECK, POLY_07, 0x00, false, false, 0x00), 0xF4);
    }

    #[test]
    fn maxim_check_vector_bitwise() {
        assert_eq!(crc8(CHECK, POLY_31, 0x00, true, true, 0x00), 0xA1);
    }

    #[test]
    fn bluetooth_check_vector_bitwise() {
        assert_eq!(crc8(CHECK, POLY_A7, 0x00, true, true, 0x00), 0x26);
    }

    #[test]
    fn dvb_s2_check_vector_bitwise() {
        assert_eq!(crc8(CHECK, POLY_D5, 0x00, false, false, 0x00), 0xBC);
    }

    // --- Empty input returns the (post-reflection) initial register ---

    #[test]
    fn smbus_empty() {
        assert_eq!(crc8_smbus(&[]), 0x00);
    }

    #[test]
    fn maxim_empty() {
        assert_eq!(crc8_maxim(&[]), 0x00);
    }

    #[test]
    fn bluetooth_empty() {
        assert_eq!(crc8_bluetooth(&[]), 0x00);
    }

    #[test]
    fn dvb_s2_empty() {
        assert_eq!(crc8_dvb_s2(&[]), 0x00);
    }

    #[test]
    fn empty_matches_bitwise() {
        assert_eq!(crc8(&[], POLY_07, 0x00, false, false, 0x00), 0x00);
        assert_eq!(crc8(&[], POLY_31, 0x00, true, true, 0x00), 0x00);
    }

    // --- Single-byte inputs ---

    #[test]
    fn smbus_single_zero() {
        assert_eq!(crc8_smbus(&[0x00]), 0x00);
    }

    #[test]
    fn maxim_single_zero() {
        assert_eq!(crc8_maxim(&[0x00]), 0x00);
    }

    #[test]
    fn smbus_single_ff() {
        assert_eq!(crc8_smbus(&[0xFF]), 0xF3);
    }

    #[test]
    fn maxim_single_ff() {
        assert_eq!(crc8_maxim(&[0xFF]), 0x35);
    }

    #[test]
    fn bluetooth_single_ff() {
        assert_eq!(crc8_bluetooth(&[0xFF]), 0x9F);
    }

    #[test]
    fn dvb_s2_single_ff() {
        assert_eq!(crc8_dvb_s2(&[0xFF]), 0xF9);
    }

    #[test]
    fn smbus_single_a() {
        assert_eq!(crc8_smbus(&[0x41]), 0xC0);
    }

    #[test]
    fn maxim_single_a() {
        assert_eq!(crc8_maxim(&[0x41]), 0x18);
    }

    #[test]
    fn bluetooth_single_a() {
        assert_eq!(crc8_bluetooth(&[0x41]), 0xFC);
    }

    #[test]
    fn dvb_s2_single_a() {
        assert_eq!(crc8_dvb_s2(&[0x41]), 0x48);
    }

    // --- All-zero and all-`0xFF` payloads ---

    #[test]
    fn smbus_all_zero_is_zero() {
        assert_eq!(crc8_smbus(&[0x00, 0x00, 0x00, 0x00]), 0x00);
    }

    #[test]
    fn maxim_all_zero_is_zero() {
        assert_eq!(crc8_maxim(&[0x00, 0x00, 0x00, 0x00]), 0x00);
    }

    #[test]
    fn dvb_s2_all_zero_is_zero() {
        assert_eq!(crc8_dvb_s2(&[0x00, 0x00, 0x00, 0x00]), 0x00);
    }

    #[test]
    fn smbus_all_ff() {
        assert_eq!(crc8_smbus(&[0xFF, 0xFF, 0xFF, 0xFF]), 0xDE);
    }

    #[test]
    fn maxim_all_ff() {
        assert_eq!(crc8_maxim(&[0xFF, 0xFF, 0xFF, 0xFF]), 0x8D);
    }

    #[test]
    fn bluetooth_all_ff() {
        assert_eq!(crc8_bluetooth(&[0xFF, 0xFF, 0xFF, 0xFF]), 0x50);
    }

    #[test]
    fn dvb_s2_all_ff() {
        assert_eq!(crc8_dvb_s2(&[0xFF, 0xFF, 0xFF, 0xFF]), 0x21);
    }

    // --- Short mixed payloads ---

    #[test]
    fn smbus_abc() {
        assert_eq!(crc8_smbus(b"abc"), 0x5F);
    }

    #[test]
    fn maxim_abc() {
        assert_eq!(crc8_maxim(b"abc"), 0x42);
    }

    #[test]
    fn bluetooth_abc() {
        assert_eq!(crc8_bluetooth(b"abc"), 0xAA);
    }

    #[test]
    fn dvb_s2_abc() {
        assert_eq!(crc8_dvb_s2(b"abc"), 0x5A);
    }

    #[test]
    fn smbus_two_bytes() {
        assert_eq!(crc8_smbus(&[0x00, 0x01]), 0x07);
    }

    #[test]
    fn dvb_s2_two_bytes() {
        assert_eq!(crc8_dvb_s2(&[0x00, 0x01]), 0xD5);
    }

    #[test]
    fn maxim_deadbeef() {
        assert_eq!(crc8_maxim(&[0xDE, 0xAD, 0xBE, 0xEF]), 0x84);
    }

    #[test]
    fn bluetooth_deadbeef() {
        assert_eq!(crc8_bluetooth(&[0xDE, 0xAD, 0xBE, 0xEF]), 0x74);
    }

    // --- Longer payload ---

    #[test]
    fn smbus_long_sequence() {
        let data: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        assert_eq!(crc8_smbus(&data), 0x41);
    }

    #[test]
    fn maxim_long_sequence() {
        let data: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        assert_eq!(crc8_maxim(&data), 0x3C);
    }

    #[test]
    fn dvb_s2_long_sequence() {
        let data: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        assert_eq!(crc8_dvb_s2(&data), 0x77);
    }

    // --- Reflected polynomial relationships ---

    #[test]
    fn maxim_reflected_poly_relation() {
        assert_eq!(POLY_31_REFLECTED, POLY_31.reverse_bits());
        assert_eq!(0x8C, 0x31u8.reverse_bits());
    }

    #[test]
    fn bluetooth_reflected_poly_relation() {
        assert_eq!(POLY_A7_REFLECTED, POLY_A7.reverse_bits());
        assert_eq!(0xE5, 0xA7u8.reverse_bits());
    }

    // --- Table-driven versus bit-by-bit consistency across many inputs ---

    #[test]
    fn smbus_table_matches_bitwise() {
        for len in 0..40u8 {
            let data: [u8; 40] =
                core::array::from_fn(|i| (i as u8).wrapping_mul(37).wrapping_add(len));
            let slice = &data[..len as usize];
            assert_eq!(
                crc8_smbus(slice),
                crc8(slice, POLY_07, 0x00, false, false, 0x00)
            );
        }
    }

    #[test]
    fn dvb_s2_table_matches_bitwise() {
        for len in 0..40u8 {
            let data: [u8; 40] =
                core::array::from_fn(|i| (i as u8).wrapping_mul(37).wrapping_add(len));
            let slice = &data[..len as usize];
            assert_eq!(
                crc8_dvb_s2(slice),
                crc8(slice, POLY_D5, 0x00, false, false, 0x00)
            );
        }
    }

    #[test]
    fn maxim_table_matches_bitwise() {
        for len in 0..40u8 {
            let data: [u8; 40] =
                core::array::from_fn(|i| (i as u8).wrapping_mul(37).wrapping_add(len));
            let slice = &data[..len as usize];
            assert_eq!(
                crc8_maxim(slice),
                crc8(slice, POLY_31, 0x00, true, true, 0x00)
            );
        }
    }

    #[test]
    fn bluetooth_table_matches_bitwise() {
        for len in 0..40u8 {
            let data: [u8; 40] =
                core::array::from_fn(|i| (i as u8).wrapping_mul(37).wrapping_add(len));
            let slice = &data[..len as usize];
            assert_eq!(
                crc8_bluetooth(slice),
                crc8(slice, POLY_A7, 0x00, true, true, 0x00)
            );
        }
    }

    // --- Cross-variant distinction on the same input ---

    #[test]
    fn variants_differ_on_check_string() {
        let s = crc8_smbus(CHECK);
        let m = crc8_maxim(CHECK);
        let b = crc8_bluetooth(CHECK);
        let d = crc8_dvb_s2(CHECK);
        assert!(s != m);
        assert!(s != b);
        assert!(s != d);
        assert!(m != b);
        assert!(m != d);
        assert!(b != d);
    }

    #[test]
    fn variants_differ_on_short_input() {
        let input = &[0x12, 0x34, 0x56];
        assert!(crc8_smbus(input) != crc8_dvb_s2(input));
        assert!(crc8_maxim(input) != crc8_bluetooth(input));
    }

    // --- A single-bit change perturbs the remainder (error detection) ---

    #[test]
    fn smbus_detects_single_bit_flip() {
        let clean = crc8_smbus(CHECK);
        let flipped = crc8_smbus(b"133456789");
        assert!(clean != flipped);
    }
}
