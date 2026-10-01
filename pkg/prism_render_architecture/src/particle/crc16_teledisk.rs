//! `CRC-16/TELEDISK`: the 16-bit cyclic redundancy check used by the Teledisk
//! floppy-image format, provided here as a cheap integrity check for particle
//! resource blobs and streamed payloads (design § integrity checks).
//!
//! The register algorithm is the textbook bit-at-a-time definition. The
//! parameters are fixed: width `16`, polynomial `0xA097`, initial register
//! `0x0`, no input reflection, no output reflection, and final `XOR` `0x0`.
//! Because neither reflection is applied, each input byte is fed most
//! significant bit (`MSB`) first and the shift register runs in its natural
//! left-shifting direction.
//!
//! For every bit the top bit of the register is compared with the incoming
//! message bit; the register is shifted left by one (masked to 16 bits) and,
//! when that comparison bit is set, the polynomial is mixed in with `XOR`. The
//! empty input therefore yields the initial value `0x0`, and the canonical
//! check string `b"123456789"` yields `0x0FB3`.
//!
//! Scope: this is an error-detection code, not a hash and not a message
//! authentication code. A `CRC` is linear and trivial to forge on purpose, so
//! it must never be used to authenticate data or defend against a malicious
//! adversary. It is meant only for catching accidental corruption in transit or
//! storage.

/// The width of the `CRC` register, in bits.
const WIDTH: u32 = 16;

/// The generator polynomial (`Koopman`/normal form, implicit high term).
const POLY: u32 = 0xA097;

/// The initial register value loaded before processing any input.
const INIT: u32 = 0x0;

/// The value `XOR`ed into the register to produce the final result.
const XOROUT: u32 = 0x0;

/// Whether each input byte is reflected (least significant bit first).
const REFLECT_IN: bool = false;

/// Whether the final register is reflected before the output `XOR`.
const REFLECT_OUT: bool = false;

/// Mask that keeps the register within [`WIDTH`] bits.
const MASK: u32 = 0xffff;

/// The most significant bit (`MSB`) of the register.
const TOPBIT: u32 = 1u32 << (WIDTH - 1);

/// Reverses the low `bits` bits of `value`, leaving the rest zero.
///
/// Used for the (here unused) input and output reflection steps so the routine
/// stays faithful to the general `CRC` parameterisation.
const fn reflect(value: u32, bits: u32) -> u32 {
    let mut out = 0u32;
    let mut i = 0u32;
    while i < bits {
        if ((value >> i) & 1) != 0 {
            out |= 1u32 << (bits - 1 - i);
        }
        i += 1;
    }
    out
}

/// Computes the `CRC-16/TELEDISK` checksum of `data`.
///
/// The empty slice yields `0x0000` and `crc16_teledisk(b"123456789")` is the
/// canonical check value `0x0FB3`. The bit-at-a-time loop produces exactly the
/// value defined by the Teledisk parameters.
#[must_use]
pub fn crc16_teledisk(data: &[u8]) -> u16 {
    let mut reg = INIT & MASK;
    let mut idx = 0usize;
    while idx < data.len() {
        let byte = data[idx];
        let b = if REFLECT_IN {
            reflect(u32::from(byte), 8)
        } else {
            u32::from(byte)
        };
        let mut i = 0u32;
        while i < 8 {
            let bit = (b >> (7 - i)) & 1;
            let msb = u32::from((reg & TOPBIT) != 0);
            reg = (reg << 1) & MASK;
            if (msb ^ bit) != 0 {
                reg ^= POLY;
            }
            i += 1;
        }
        idx += 1;
    }
    if REFLECT_OUT {
        reg = reflect(reg, WIDTH);
    }
    ((reg ^ XOROUT) & MASK) as u16
}

#[cfg(test)]
mod tests {
    use super::crc16_teledisk;

    #[test]
    fn vector_empty() {
        assert!(crc16_teledisk(b"") == 0x0000);
    }

    #[test]
    fn vector_z00() {
        assert!(crc16_teledisk(&[0x00]) == 0x0000);
    }

    #[test]
    fn vector_ff() {
        assert!(crc16_teledisk(&[0xff]) == 0xf087);
    }

    #[test]
    fn vector_a() {
        assert!(crc16_teledisk(b"a") == 0x429d);
    }

    #[test]
    fn vector_b() {
        assert!(crc16_teledisk(b"b") == 0x03b3);
    }

    #[test]
    fn vector_ab() {
        assert!(crc16_teledisk(b"ab") == 0xc306);
    }

    #[test]
    fn vector_abc() {
        assert!(crc16_teledisk(b"abc") == 0x8089);
    }

    #[test]
    fn vector_two_zeros() {
        assert!(crc16_teledisk(&[0x00, 0x00]) == 0x0000);
    }

    #[test]
    fn vector_01() {
        assert!(crc16_teledisk(&[0x01]) == 0xa097);
    }

    #[test]
    fn vector_02() {
        assert!(crc16_teledisk(&[0x02]) == 0xe1b9);
    }

    #[test]
    fn vector_7f() {
        assert!(crc16_teledisk(&[0x7f]) == 0x2808);
    }

    #[test]
    fn vector_80() {
        assert!(crc16_teledisk(&[0x80]) == 0xd88f);
    }

    #[test]
    fn vector_aa_55() {
        assert!(crc16_teledisk(&[0xaa, 0x55]) == 0x2cf4);
    }

    #[test]
    fn vector_55_aa() {
        assert!(crc16_teledisk(&[0x55, 0xaa]) == 0x4ef5);
    }

    #[test]
    fn vector_deadbeef() {
        assert!(crc16_teledisk(&[0xde, 0xad, 0xbe, 0xef]) == 0x520d);
    }

    #[test]
    fn vector_hello() {
        assert!(crc16_teledisk(b"Hello") == 0xfe02);
    }

    #[test]
    fn vector_fox() {
        assert!(crc16_teledisk(b"The quick brown fox") == 0x3697);
    }

    #[test]
    fn vector_four_zeros() {
        assert!(crc16_teledisk(&[0x00; 4]) == 0x0000);
    }

    #[test]
    fn vector_four_ff() {
        assert!(crc16_teledisk(&[0xff; 4]) == 0x30b8);
    }

    #[test]
    fn vector_0_15() {
        let mut data = [0u8; 16];
        let mut i = 0usize;
        while i < data.len() {
            data[i] = i as u8;
            i += 1;
        }
        assert!(crc16_teledisk(&data) == 0x98f1);
    }

    #[test]
    fn vector_12345678() {
        assert!(crc16_teledisk(&[0x12, 0x34, 0x56, 0x78]) == 0x9e28);
    }

    #[test]
    fn vector_check() {
        assert!(crc16_teledisk(b"123456789") == 0x0fb3);
    }

    #[test]
    fn vector_a5_1000() {
        let data = [0xA5u8; 1000];
        assert!(crc16_teledisk(&data) == 0xd0f2);
    }

    #[test]
    fn vector_all256() {
        let mut data = [0u8; 256];
        let mut i = 0usize;
        while i < data.len() {
            data[i] = i as u8;
            i += 1;
        }
        assert!(crc16_teledisk(&data) == 0xe0b5);
    }

    #[test]
    fn check_constant_matches_specification() {
        // The documented `CRC-16/TELEDISK` check value for b"123456789".
        assert!(crc16_teledisk(b"123456789") == 0x0fb3);
    }

    #[test]
    fn determinism_repeated_calls_agree() {
        let input = b"determinism sample payload";
        let first = crc16_teledisk(input);
        let second = crc16_teledisk(input);
        assert!(first == second);
    }

    #[test]
    fn order_sensitive_swapped_bytes_differ() {
        assert!(crc16_teledisk(&[0xaa, 0x55]) != crc16_teledisk(&[0x55, 0xaa]));
    }

    #[test]
    fn a_and_b_differ() {
        assert!(crc16_teledisk(b"a") != crc16_teledisk(b"b"));
    }

    #[test]
    fn prefix_differs_from_extended() {
        let prefix = crc16_teledisk(b"abc");
        let extended = crc16_teledisk(b"abcd");
        assert!(prefix != extended);
    }

    #[test]
    fn length_sensitive_zero_runs_differ_from_empty_mix() {
        // Different lengths of distinct content yield distinct checks here.
        assert!(crc16_teledisk(&[0x01]) != crc16_teledisk(&[0x01, 0x01]));
    }

    #[test]
    fn single_byte_values_are_pairwise_distinct() {
        let mut seen = [0u16; 256];
        let mut i = 0usize;
        while i < 256 {
            seen[i] = crc16_teledisk(&[i as u8]);
            i += 1;
        }
        let mut a = 0usize;
        while a < 256 {
            let mut b = a + 1;
            while b < 256 {
                assert!(seen[a] != seen[b]);
                b += 1;
            }
            a += 1;
        }
    }

    #[test]
    fn result_within_u16_range() {
        let samples: [&[u8]; 4] = [b"", b"a", b"123456789", &[0xff, 0x00, 0xaa]];
        let mut i = 0usize;
        while i < samples.len() {
            let v = crc16_teledisk(samples[i]);
            assert!((0..=0xffffu16).contains(&v));
            i += 1;
        }
    }

    #[test]
    fn empty_matches_init_value() {
        assert!(crc16_teledisk(b"") == 0x0000);
    }

    #[test]
    fn appending_zero_byte_changes_result() {
        let base = crc16_teledisk(b"abc");
        let padded = crc16_teledisk(b"abc\x00");
        assert!(base != padded);
    }

    #[test]
    fn two_zeros_matches_single_zero_here() {
        // With init 0 and no xorout, zero bytes keep the register at zero.
        assert!(crc16_teledisk(&[0x00]) == crc16_teledisk(&[0x00, 0x00]));
    }
}
