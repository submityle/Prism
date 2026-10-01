//! `CRC-16` cyclic-redundancy checks in several standard parameterisations
//! (including the `CCITT` family): pure-integer, table-driven polynomial
//! division over `GF(2)` for verifying short headers, framing bytes, and small
//! `GPU` resource descriptors (design § integrity checks).
//!
//! A `CRC-16` treats the input bytes as the coefficients of a polynomial over
//! the binary field `GF(2)` and returns the `16`-bit remainder after dividing
//! by a fixed generator polynomial. The arithmetic is carry-less: "addition" is
//! exclusive-or and there are no carries, so the whole computation is just
//! shifts, masks, and exclusive-ors. This module implements four widely used
//! `16`-bit variants plus a generic bit-by-bit driver underneath them:
//!
//! - `CRC-16`/`CCITT-FALSE`: `poly` `0x1021`, `init` `0xFFFF`, no input/output
//!   reflection, final exclusive-or `0x0000`.
//! - `CRC-16`/`XMODEM`: `poly` `0x1021`, `init` `0x0000`, no reflection, final
//!   exclusive-or `0x0000`.
//! - `CRC-16`/`ARC` (also called `IBM`): `poly` `0x8005` (reflected form
//!   `0xA001`), `init` `0x0000`, reflected input and output.
//! - `CRC-16`/`MODBUS`: `poly` `0x8005` (reflected `0xA001`), `init` `0xFFFF`,
//!   reflected input and output.
//!
//! The canonical *check* values for the `ASCII` string `123456789` are
//! `CCITT-FALSE` `0x29B1`, `XMODEM` `0x31C3`, `ARC` `0xBB3D`, and `MODBUS`
//! `0x4B37`; these are asserted directly in the tests.
//!
//! Scope and boundaries. This is a `16`-bit `CRC`: a `GF(2)` polynomial-modulo
//! error-detection code. It is deliberately distinct from the neighbouring
//! integrity primitives in this crate, which differ in both bit width and the
//! underlying algebra:
//!
//! - `crc32` is a `32`-bit `CRC` (wider register, different generator) but the
//!   same `GF(2)` polynomial-division algebra.
//! - `fletcher_checksum` and `adler32` are *not* `CRC`s at all: they are
//!   position-weighted sums computed with ordinary integer modular addition
//!   (`Fletcher`/`Adler` running sums reduced modulo a constant), not carry-less
//!   polynomial division. Different algebra, different error profile.
//!
//! None of these are cryptographic. A `CRC-16` is trivially invertible and
//! collisions are easy to craft on purpose, so it must only guard against
//! accidental corruption, never against a malicious adversary.

/// Generator polynomial `0x1021` (`CCITT`/`XMODEM` family), most-significant-bit
/// first (unreflected) form.
const POLY_1021: u16 = 0x1021;

/// Generator polynomial `0x8005` (`ARC`/`MODBUS` family), unreflected form.
#[cfg(test)]
const POLY_8005: u16 = 0x8005;

/// Reflected form of polynomial `0x8005`, i.e. `0xA001`, used by the
/// least-significant-bit-first (reflected) table driver.
const POLY_8005_REFLECTED: u16 = 0xA001;

/// Generic bit-by-bit `CRC-16` driver (the low-level primitive).
///
/// Computes the `16`-bit remainder for arbitrary parameters: generator `poly`
/// in unreflected form, initial register `init`, input reflection `refin`,
/// output reflection `refout`, and final exclusive-or `xorout`. When `refin` is
/// set each input byte is bit-reversed before being folded in; when `refout` is
/// set the final register is bit-reversed before the exclusive-or. This matches
/// the standard parametric `CRC` model and is the authority the table-driven
/// variants are checked against.
pub fn crc16(data: &[u8], poly: u16, init: u16, refin: bool, refout: bool, xorout: u16) -> u16 {
    let mut crc = init;
    for &byte in data {
        let folded = if refin { byte.reverse_bits() } else { byte };
        crc ^= (folded as u16) << 8;
        for _ in 0..8 {
            if (crc & 0x8000) != 0 {
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
/// Entry `i` holds the register contribution of folding the byte `i` into an
/// otherwise-zero register, processing the byte in the high `8` bits.
fn build_table_msb(poly: u16) -> [u16; 256] {
    let mut table = [0u16; 256];
    for (i, slot) in table.iter_mut().enumerate() {
        let mut crc = (i as u16) << 8;
        for _ in 0..8 {
            if (crc & 0x8000) != 0 {
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
/// reflected generator `poly_reflected` (for example `0xA001`).
///
/// Entry `i` holds the register contribution of folding the byte `i` into an
/// otherwise-zero register, processing the byte in the low `8` bits.
fn build_table_lsb(poly_reflected: u16) -> [u16; 256] {
    let mut table = [0u16; 256];
    for (i, slot) in table.iter_mut().enumerate() {
        let mut crc = i as u16;
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

/// Table-driven most-significant-bit-first `CRC-16` (no reflection).
///
/// Used by the unreflected variants (`CCITT-FALSE`, `XMODEM`). The generator
/// `poly` is given in unreflected form.
fn crc16_table_msb(data: &[u8], poly: u16, init: u16, xorout: u16) -> u16 {
    let table = build_table_msb(poly);
    let mut crc = init;
    for &byte in data {
        let index = ((crc >> 8) ^ (byte as u16)) & 0xFF;
        crc = (crc << 8) ^ table[index as usize];
    }
    crc ^ xorout
}

/// Table-driven least-significant-bit-first `CRC-16` (reflected input/output).
///
/// Used by the reflected variants (`ARC`, `MODBUS`). The generator
/// `poly_reflected` is given in reflected form (for example `0xA001`); input and
/// output reflection are intrinsic to this least-significant-bit-first driver,
/// so no explicit bit-reversal is needed.
fn crc16_table_lsb(data: &[u8], poly_reflected: u16, init: u16, xorout: u16) -> u16 {
    let table = build_table_lsb(poly_reflected);
    let mut crc = init;
    for &byte in data {
        let index = (crc ^ (byte as u16)) & 0xFF;
        crc = (crc >> 8) ^ table[index as usize];
    }
    crc ^ xorout
}

/// `CRC-16`/`CCITT-FALSE`: `poly` `0x1021`, `init` `0xFFFF`, no reflection,
/// final exclusive-or `0x0000`. Check value for `123456789` is `0x29B1`.
pub fn crc16_ccitt_false(data: &[u8]) -> u16 {
    crc16_table_msb(data, POLY_1021, 0xFFFF, 0x0000)
}

/// `CRC-16`/`XMODEM`: `poly` `0x1021`, `init` `0x0000`, no reflection, final
/// exclusive-or `0x0000`. Check value for `123456789` is `0x31C3`.
pub fn crc16_xmodem(data: &[u8]) -> u16 {
    crc16_table_msb(data, POLY_1021, 0x0000, 0x0000)
}

/// `CRC-16`/`ARC` (also `IBM`): `poly` `0x8005` (reflected `0xA001`), `init`
/// `0x0000`, reflected input and output. Check value for `123456789` is
/// `0xBB3D`.
pub fn crc16_arc(data: &[u8]) -> u16 {
    crc16_table_lsb(data, POLY_8005_REFLECTED, 0x0000, 0x0000)
}

/// `CRC-16`/`MODBUS`: `poly` `0x8005` (reflected `0xA001`), `init` `0xFFFF`,
/// reflected input and output. Check value for `123456789` is `0x4B37`.
pub fn crc16_modbus(data: &[u8]) -> u16 {
    crc16_table_lsb(data, POLY_8005_REFLECTED, 0xFFFF, 0x0000)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The canonical `CRC` check string.
    const CHECK: &[u8] = b"123456789";

    // --- Authoritative check values for `123456789` (table-driven) ---

    #[test]
    fn ccitt_false_check_vector() {
        assert_eq!(crc16_ccitt_false(CHECK), 0x29B1);
    }

    #[test]
    fn xmodem_check_vector() {
        assert_eq!(crc16_xmodem(CHECK), 0x31C3);
    }

    #[test]
    fn arc_check_vector() {
        assert_eq!(crc16_arc(CHECK), 0xBB3D);
    }

    #[test]
    fn modbus_check_vector() {
        assert_eq!(crc16_modbus(CHECK), 0x4B37);
    }

    // --- Same check values via the generic bit-by-bit driver ---

    #[test]
    fn ccitt_false_check_vector_bitwise() {
        assert_eq!(
            crc16(CHECK, POLY_1021, 0xFFFF, false, false, 0x0000),
            0x29B1
        );
    }

    #[test]
    fn xmodem_check_vector_bitwise() {
        assert_eq!(
            crc16(CHECK, POLY_1021, 0x0000, false, false, 0x0000),
            0x31C3
        );
    }

    #[test]
    fn arc_check_vector_bitwise() {
        assert_eq!(crc16(CHECK, POLY_8005, 0x0000, true, true, 0x0000), 0xBB3D);
    }

    #[test]
    fn modbus_check_vector_bitwise() {
        assert_eq!(crc16(CHECK, POLY_8005, 0xFFFF, true, true, 0x0000), 0x4B37);
    }

    // --- Empty input returns `init` (after any reflection/exclusive-or) ---

    #[test]
    fn ccitt_false_empty() {
        assert_eq!(crc16_ccitt_false(&[]), 0xFFFF);
    }

    #[test]
    fn xmodem_empty() {
        assert_eq!(crc16_xmodem(&[]), 0x0000);
    }

    #[test]
    fn arc_empty() {
        assert_eq!(crc16_arc(&[]), 0x0000);
    }

    #[test]
    fn modbus_empty() {
        assert_eq!(crc16_modbus(&[]), 0xFFFF);
    }

    #[test]
    fn empty_matches_bitwise() {
        assert_eq!(
            crc16_ccitt_false(&[]),
            crc16(&[], POLY_1021, 0xFFFF, false, false, 0x0000)
        );
        assert_eq!(
            crc16_xmodem(&[]),
            crc16(&[], POLY_1021, 0x0000, false, false, 0x0000)
        );
        assert_eq!(
            crc16_arc(&[]),
            crc16(&[], POLY_8005, 0x0000, true, true, 0x0000)
        );
        assert_eq!(
            crc16_modbus(&[]),
            crc16(&[], POLY_8005, 0xFFFF, true, true, 0x0000)
        );
    }

    // --- Reflected polynomial relationship ---

    #[test]
    fn reflected_poly_is_bit_reverse() {
        assert_eq!(POLY_8005_REFLECTED, POLY_8005.reverse_bits());
    }

    // --- Cross-variant distinctness on the check string ---

    #[test]
    fn ccitt_false_differs_from_xmodem() {
        assert_ne!(crc16_ccitt_false(CHECK), crc16_xmodem(CHECK));
    }

    #[test]
    fn ccitt_false_differs_from_arc() {
        assert_ne!(crc16_ccitt_false(CHECK), crc16_arc(CHECK));
    }

    #[test]
    fn ccitt_false_differs_from_modbus() {
        assert_ne!(crc16_ccitt_false(CHECK), crc16_modbus(CHECK));
    }

    #[test]
    fn xmodem_differs_from_arc() {
        assert_ne!(crc16_xmodem(CHECK), crc16_arc(CHECK));
    }

    #[test]
    fn xmodem_differs_from_modbus() {
        assert_ne!(crc16_xmodem(CHECK), crc16_modbus(CHECK));
    }

    #[test]
    fn arc_differs_from_modbus() {
        assert_ne!(crc16_arc(CHECK), crc16_modbus(CHECK));
    }

    // --- Single-byte: table-driven agrees with the bit-by-bit driver ---

    #[test]
    fn single_byte_ccitt_false_matches_bitwise() {
        let data = [0x5A];
        assert_eq!(
            crc16_ccitt_false(&data),
            crc16(&data, POLY_1021, 0xFFFF, false, false, 0x0000)
        );
    }

    #[test]
    fn single_byte_xmodem_matches_bitwise() {
        let data = [0x5A];
        assert_eq!(
            crc16_xmodem(&data),
            crc16(&data, POLY_1021, 0x0000, false, false, 0x0000)
        );
    }

    #[test]
    fn single_byte_arc_matches_bitwise() {
        let data = [0x5A];
        assert_eq!(
            crc16_arc(&data),
            crc16(&data, POLY_8005, 0x0000, true, true, 0x0000)
        );
    }

    #[test]
    fn single_byte_modbus_matches_bitwise() {
        let data = [0x5A];
        assert_eq!(
            crc16_modbus(&data),
            crc16(&data, POLY_8005, 0xFFFF, true, true, 0x0000)
        );
    }

    // --- All-zero input: table-driven agrees with the bit-by-bit driver ---

    #[test]
    fn all_zero_ccitt_false_matches_bitwise() {
        let data = [0u8; 32];
        assert_eq!(
            crc16_ccitt_false(&data),
            crc16(&data, POLY_1021, 0xFFFF, false, false, 0x0000)
        );
    }

    #[test]
    fn all_zero_xmodem_matches_bitwise() {
        let data = [0u8; 32];
        assert_eq!(
            crc16_xmodem(&data),
            crc16(&data, POLY_1021, 0x0000, false, false, 0x0000)
        );
    }

    #[test]
    fn all_zero_arc_matches_bitwise() {
        let data = [0u8; 32];
        assert_eq!(
            crc16_arc(&data),
            crc16(&data, POLY_8005, 0x0000, true, true, 0x0000)
        );
    }

    #[test]
    fn all_zero_modbus_matches_bitwise() {
        let data = [0u8; 32];
        assert_eq!(
            crc16_modbus(&data),
            crc16(&data, POLY_8005, 0xFFFF, true, true, 0x0000)
        );
    }

    // --- All-`0xFF` input: table-driven agrees with the bit-by-bit driver ---

    #[test]
    fn all_ff_ccitt_false_matches_bitwise() {
        let data = [0xFFu8; 32];
        assert_eq!(
            crc16_ccitt_false(&data),
            crc16(&data, POLY_1021, 0xFFFF, false, false, 0x0000)
        );
    }

    #[test]
    fn all_ff_xmodem_matches_bitwise() {
        let data = [0xFFu8; 32];
        assert_eq!(
            crc16_xmodem(&data),
            crc16(&data, POLY_1021, 0x0000, false, false, 0x0000)
        );
    }

    #[test]
    fn all_ff_arc_matches_bitwise() {
        let data = [0xFFu8; 32];
        assert_eq!(
            crc16_arc(&data),
            crc16(&data, POLY_8005, 0x0000, true, true, 0x0000)
        );
    }

    #[test]
    fn all_ff_modbus_matches_bitwise() {
        let data = [0xFFu8; 32];
        assert_eq!(
            crc16_modbus(&data),
            crc16(&data, POLY_8005, 0xFFFF, true, true, 0x0000)
        );
    }

    // --- Long input: table-driven agrees with the bit-by-bit driver ---

    #[test]
    fn long_ccitt_false_matches_bitwise() {
        let data = [0xABu8; 1024];
        assert_eq!(
            crc16_ccitt_false(&data),
            crc16(&data, POLY_1021, 0xFFFF, false, false, 0x0000)
        );
    }

    #[test]
    fn long_xmodem_matches_bitwise() {
        let data = [0xABu8; 1024];
        assert_eq!(
            crc16_xmodem(&data),
            crc16(&data, POLY_1021, 0x0000, false, false, 0x0000)
        );
    }

    #[test]
    fn long_arc_matches_bitwise() {
        let data = [0xABu8; 1024];
        assert_eq!(
            crc16_arc(&data),
            crc16(&data, POLY_8005, 0x0000, true, true, 0x0000)
        );
    }

    #[test]
    fn long_modbus_matches_bitwise() {
        let data = [0xABu8; 1024];
        assert_eq!(
            crc16_modbus(&data),
            crc16(&data, POLY_8005, 0xFFFF, true, true, 0x0000)
        );
    }

    // --- Mixed `ASCII` payload: table-driven agrees with the bit-by-bit driver ---

    #[test]
    fn ascii_ccitt_false_matches_bitwise() {
        let data = b"The quick brown fox";
        assert_eq!(
            crc16_ccitt_false(data),
            crc16(data, POLY_1021, 0xFFFF, false, false, 0x0000)
        );
    }

    #[test]
    fn ascii_xmodem_matches_bitwise() {
        let data = b"The quick brown fox";
        assert_eq!(
            crc16_xmodem(data),
            crc16(data, POLY_1021, 0x0000, false, false, 0x0000)
        );
    }

    #[test]
    fn ascii_arc_matches_bitwise() {
        let data = b"The quick brown fox";
        assert_eq!(
            crc16_arc(data),
            crc16(data, POLY_8005, 0x0000, true, true, 0x0000)
        );
    }

    #[test]
    fn ascii_modbus_matches_bitwise() {
        let data = b"The quick brown fox";
        assert_eq!(
            crc16_modbus(data),
            crc16(data, POLY_8005, 0xFFFF, true, true, 0x0000)
        );
    }

    // --- Check string reproduced through the generic driver and the tables ---

    #[test]
    fn check_vector_table_matches_bitwise_all_variants() {
        assert_eq!(
            crc16_ccitt_false(CHECK),
            crc16(CHECK, POLY_1021, 0xFFFF, false, false, 0x0000)
        );
        assert_eq!(
            crc16_xmodem(CHECK),
            crc16(CHECK, POLY_1021, 0x0000, false, false, 0x0000)
        );
        assert_eq!(
            crc16_arc(CHECK),
            crc16(CHECK, POLY_8005, 0x0000, true, true, 0x0000)
        );
        assert_eq!(
            crc16_modbus(CHECK),
            crc16(CHECK, POLY_8005, 0xFFFF, true, true, 0x0000)
        );
    }

    // --- Reflected vs unreflected tables are genuinely different ---

    #[test]
    fn msb_and_lsb_tables_differ() {
        let msb = build_table_msb(POLY_8005);
        let lsb = build_table_lsb(POLY_8005_REFLECTED);
        assert_ne!(msb[1], lsb[1]);
    }

    #[test]
    fn msb_table_entry_zero_is_zero() {
        let table = build_table_msb(POLY_1021);
        assert_eq!(table[0], 0x0000);
    }

    #[test]
    fn lsb_table_entry_zero_is_zero() {
        let table = build_table_lsb(POLY_8005_REFLECTED);
        assert_eq!(table[0], 0x0000);
    }
}
