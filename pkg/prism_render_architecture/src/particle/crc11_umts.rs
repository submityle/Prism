//! `CRC-11`/`UMTS` checksum, implemented bit-at-a-time (`MSB`-first) with pure integers.
//!
//! Parameters: width=11, poly=0x307, init=0x000, refin=false, refout=false, xorout=0x000.
//!
//! This is a `CPU`-verifiable contract module: the running register is held in a
//! `u16` whose low 11 bits are significant. `MASK` keeps the register inside 11
//! bits and `TOP` selects the most-significant bit (bit 10) that is compared
//! against each incoming message bit via an `XOR`. No reflection is applied on
//! input or output, so bytes are processed `MSB`-first and the register is read
//! out directly.

/// Low 11-bit mask applied to the register after every shift.
const MASK: u16 = 0x7FF;
/// Most-significant bit (bit 10) of the 11-bit register.
const TOP: u16 = 0x400;
/// Generator polynomial (`CRC-11`/`UMTS`).
const POLY: u16 = 0x307;
/// Initial register value.
const INIT: u16 = 0x000;
/// Final `XOR` value applied to the register.
const XOROUT: u16 = 0x000;
/// Number of bits folded in per input byte.
const BITS_PER_BYTE: u32 = 8;

/// Compute the `CRC-11`/`UMTS` checksum of `data` in a single call.
///
/// Returns the 11-bit checksum in the low bits of a `u16`; the upper 5 bits are
/// always zero.
#[must_use]
pub fn crc11_umts(data: &[u8]) -> u16 {
    let mut crc = INIT;
    for &byte in data {
        crc = update_byte(crc, byte);
    }
    crc ^ XOROUT
}

/// Fold a single input byte into the running register (`MSB`-first, refin=false).
#[must_use]
fn update_byte(mut crc: u16, byte: u8) -> u16 {
    for i in (0..BITS_PER_BYTE).rev() {
        let inbit = u16::from((byte >> i) & 1);
        let msb = u16::from((crc & TOP) != 0);
        crc = (crc << 1) & MASK;
        if (msb ^ inbit) != 0 {
            crc ^= POLY;
        }
    }
    crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_empty() {
        assert!(crc11_umts(b"") == 0x000);
    }

    #[test]
    fn vector_z00() {
        assert!(crc11_umts(&[0x00]) == 0x000);
    }

    #[test]
    fn vector_ff() {
        assert!(crc11_umts(&[0xff]) == 0x137);
    }

    #[test]
    fn vector_a() {
        assert!(crc11_umts(b"a") == 0x459);
    }

    #[test]
    fn vector_b() {
        assert!(crc11_umts(b"b") == 0x150);
    }

    #[test]
    fn vector_ab() {
        assert!(crc11_umts(b"ab") == 0x047);
    }

    #[test]
    fn vector_abc() {
        assert!(crc11_umts(b"abc") == 0x066);
    }

    #[test]
    fn vector_two_zeros() {
        assert!(crc11_umts(&[0x00, 0x00]) == 0x000);
    }

    #[test]
    fn vector_01() {
        assert!(crc11_umts(&[0x01]) == 0x307);
    }

    #[test]
    fn vector_02() {
        assert!(crc11_umts(&[0x02]) == 0x60e);
    }

    #[test]
    fn vector_7f() {
        assert!(crc11_umts(&[0x7f]) == 0x118);
    }

    #[test]
    fn vector_80() {
        assert!(crc11_umts(&[0x80]) == 0x02f);
    }

    #[test]
    fn vector_aa_55() {
        assert!(crc11_umts(&[0xaa, 0x55]) == 0x17a);
    }

    #[test]
    fn vector_55_aa() {
        assert!(crc11_umts(&[0x55, 0xaa]) == 0x492);
    }

    #[test]
    fn vector_deadbeef() {
        assert!(crc11_umts(&[0xde, 0xad, 0xbe, 0xef]) == 0x70d);
    }

    #[test]
    fn vector_hello() {
        assert!(crc11_umts(b"Hello") == 0x5c6);
    }

    #[test]
    fn vector_fox() {
        assert!(crc11_umts(b"The quick brown fox") == 0x43f);
    }

    #[test]
    fn vector_four_zeros() {
        assert!(crc11_umts(&[0u8; 4]) == 0x000);
    }

    #[test]
    fn vector_four_ff() {
        assert!(crc11_umts(&[0xffu8; 4]) == 0x005);
    }

    #[test]
    fn vector_0_15() {
        let mut buf = [0u8; 16];
        let mut i = 0usize;
        while i < buf.len() {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc11_umts(&buf) == 0x05f);
    }

    #[test]
    fn vector_12345678() {
        assert!(crc11_umts(&[0x12, 0x34, 0x56, 0x78]) == 0x051);
    }

    #[test]
    fn vector_check() {
        assert!(crc11_umts(b"123456789") == 0x061);
    }

    #[test]
    fn vector_a5_1000() {
        let mut buf = [0u8; 1000];
        let mut i = 0usize;
        while i < buf.len() {
            buf[i] = 0xA5;
            i += 1;
        }
        assert!(crc11_umts(&buf) == 0x084);
    }

    #[test]
    fn vector_all256() {
        let mut buf = [0u8; 256];
        let mut i = 0usize;
        while i < buf.len() {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc11_umts(&buf) == 0x3a3);
    }

    #[test]
    fn check_constant_is_0x061() {
        // The canonical `CRC` check value is the result over the ASCII digits
        // b"123456789".
        let check = crc11_umts(b"123456789");
        assert!(check == 0x061);
    }

    #[test]
    fn determinism_repeated_calls_match() {
        let input = b"The quick brown fox";
        let first = crc11_umts(input);
        let second = crc11_umts(input);
        assert!(first == second);
    }

    #[test]
    fn a_differs_from_b() {
        assert!(crc11_umts(b"a") != crc11_umts(b"b"));
    }

    #[test]
    fn prefix_differs() {
        // A shorter prefix must not collide with the longer message.
        assert!(crc11_umts(b"ab") != crc11_umts(b"abc"));
        assert!(crc11_umts(b"a") != crc11_umts(b"ab"));
    }

    #[test]
    fn length_sensitive_same_byte() {
        // Repeating the same byte changes the checksum with length.
        assert!(crc11_umts(&[0xff]) != crc11_umts(&[0xffu8; 4]));
    }

    #[test]
    fn single_byte_pairwise_distinct() {
        let samples: [u8; 5] = [0x01, 0x02, 0x7f, 0x80, 0xff];
        let mut i = 0usize;
        while i < samples.len() {
            let mut j = i + 1;
            while j < samples.len() {
                let left = crc11_umts(&[samples[i]]);
                let right = crc11_umts(&[samples[j]]);
                assert!(left != right);
                j += 1;
            }
            i += 1;
        }
    }

    #[test]
    fn output_within_11_bits() {
        let inputs: [&[u8]; 6] = [
            b"",
            b"a",
            b"abc",
            b"Hello",
            &[0xde, 0xad, 0xbe, 0xef],
            b"123456789",
        ];
        let mut i = 0usize;
        while i < inputs.len() {
            let value = crc11_umts(inputs[i]);
            assert!((0..=MASK).contains(&value));
            i += 1;
        }
    }

    #[test]
    fn empty_equals_init_without_xorout() {
        // With init=0 and xorout=0 the empty message folds to zero.
        assert!(crc11_umts(b"") == 0x000);
    }

    #[test]
    fn high_bits_are_cleared() {
        // The checksum never sets bits above the 11-bit register width.
        let value = crc11_umts(&[0xffu8; 8]);
        assert!((value & !MASK) == 0);
    }

    #[test]
    fn single_0x01_equals_poly() {
        // Folding 0x01 drives the top message bit through the register and
        // leaves exactly the polynomial.
        assert!(crc11_umts(&[0x01]) == POLY);
    }

    #[test]
    fn poly_fits_in_register() {
        assert!((0..=MASK).contains(&POLY));
    }

    #[test]
    fn update_byte_matches_single_call() {
        let folded = update_byte(INIT, 0x80) ^ XOROUT;
        assert!(folded == crc11_umts(&[0x80]));
    }

    #[test]
    fn byte_folding_matches_stepwise() {
        let combined = crc11_umts(&[0xaa, 0x55]);
        let stepwise = update_byte(update_byte(INIT, 0xaa), 0x55) ^ XOROUT;
        assert!(combined == stepwise);
    }
}
