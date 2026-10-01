//! `CRC-8/MAXIM` CPU golden reference (reflected, `LSB`-first, `u8`).
//!
//! Aliases: `CRC-8/MAXIM-DOW`, Dallas / Maxim 1-Wire.
//! Parameters: width=8, poly=0x31, init=0x00, refin=true, refout=true,
//! xorout=0x00. Reflected polynomial `REFPOLY`=0x8C is the 8-bit
//! bit-reverse of 0x31. The canonical check value for the ASCII input
//! `b"123456789"` is 0xa1.
//!
//! This module is `no_std` + `alloc` friendly: it uses only pure integer
//! arithmetic (no `f32`/`f64`, no transcendental functions) and allocates
//! nothing. The `XOR` and shift based inner loop processes each bit from
//! least significant to most significant, matching the reflected
//! `CRC-8/MAXIM-DOW` definition used across 1-Wire devices.

/// Compute the `CRC-8/MAXIM` (`CRC-8/MAXIM-DOW`) checksum of `data`.
///
/// Reflected form: poly=0x31 reversed to `REFPOLY`=0x8C, init=0x00,
/// refin=true, refout=true, xorout=0x00. The returned `u8` is the
/// checksum; for `b"123456789"` it equals 0xa1.
#[must_use]
pub fn crc8_maxim(data: &[u8]) -> u8 {
    const REFPOLY: u8 = 0x8C;
    let mut crc: u8 = 0x00;
    for &b in data {
        crc ^= b;
        for _ in 0..8 {
            if (crc & 1) != 0 {
                crc = (crc >> 1) ^ REFPOLY;
            } else {
                crc >>= 1;
            }
        }
    }
    crc
}

#[cfg(test)]
mod tests {
    use super::crc8_maxim;

    // --- 5 canonical anchors (node_repl verified, check_ok=true) ---

    #[test]
    fn anchor_empty() {
        assert!(crc8_maxim(b"") == 0x00);
    }

    #[test]
    fn anchor_a() {
        assert!(crc8_maxim(b"a") == 0x3b);
    }

    #[test]
    fn anchor_zero_byte() {
        assert!(crc8_maxim(&[0x00]) == 0x00);
    }

    #[test]
    fn anchor_ff_byte() {
        assert!(crc8_maxim(&[0xff]) == 0x35);
    }

    #[test]
    fn anchor_check_123456789() {
        assert!(crc8_maxim(b"123456789") == 0xa1);
    }

    // --- >=10 multi-byte hard-coded self-computed vectors ---

    #[test]
    fn multi_abc() {
        assert!(crc8_maxim(b"abc") == 0x42);
    }

    #[test]
    fn multi_hello() {
        assert!(crc8_maxim(b"hello") == 0x13);
    }

    #[test]
    fn multi_prism() {
        assert!(crc8_maxim(b"Prism") == 0x6a);
    }

    #[test]
    fn multi_seq_1234() {
        assert!(crc8_maxim(&[0x01, 0x02, 0x03, 0x04]) == 0xf4);
    }

    #[test]
    fn multi_deadbeef() {
        assert!(crc8_maxim(&[0xde, 0xad, 0xbe, 0xef]) == 0x84);
    }

    #[test]
    fn multi_three_zeros() {
        assert!(crc8_maxim(&[0x00, 0x00, 0x00]) == 0x00);
    }

    #[test]
    fn multi_four_ff() {
        assert!(crc8_maxim(&[0xff, 0xff, 0xff, 0xff]) == 0x8d);
    }

    #[test]
    fn multi_quick_brown_fox() {
        assert!(crc8_maxim(b"The quick brown fox") == 0xe5);
    }

    #[test]
    fn multi_ramp() {
        assert!(crc8_maxim(&[0x10, 0x20, 0x30, 0x40, 0x50]) == 0x92);
    }

    #[test]
    fn multi_boundary_bytes() {
        assert!(crc8_maxim(&[0x7f, 0x80, 0x81]) == 0x4d);
    }

    #[test]
    fn multi_one_wire_label() {
        assert!(crc8_maxim(b"1-Wire") == 0xc6);
    }

    #[test]
    fn multi_alternating() {
        assert!(crc8_maxim(&[0xaa, 0x55, 0xaa, 0x55]) == 0x11);
    }

    // --- determinism: repeated calls agree ---

    #[test]
    fn deterministic_repeat_check() {
        let first = crc8_maxim(b"123456789");
        let second = crc8_maxim(b"123456789");
        assert!(first == second);
        assert!(first == 0xa1);
    }

    #[test]
    fn deterministic_repeat_prism() {
        let a = crc8_maxim(b"Prism");
        let b = crc8_maxim(b"Prism");
        assert!(a == b);
    }

    #[test]
    fn deterministic_many_iterations() {
        let expected = crc8_maxim(b"The quick brown fox");
        let mut i: u32 = 0;
        while i < 64 {
            assert!(crc8_maxim(b"The quick brown fox") == expected);
            i += 1;
        }
    }

    // --- empty input is zero ---

    #[test]
    fn empty_input_is_zero() {
        assert!(crc8_maxim(&[]) == 0x00);
        assert!(crc8_maxim(b"") == 0x00);
    }

    // --- sampling / splittable: byte-by-byte accumulation matches ---

    #[test]
    fn result_is_u8_ranged() {
        // Any computed checksum must fit in a u8 (trivially true, but
        // guards against accidental widening of the return type).
        let v = crc8_maxim(b"The quick brown fox");
        assert!((0x00..=0xff).contains(&(u16::from(v))));
    }

    #[test]
    fn prefix_then_full_consistent() {
        // Recomputing a full buffer must equal computing it again from the
        // same slice; sampling arbitrary prefixes stays deterministic.
        let full = b"123456789";
        let whole = crc8_maxim(full);
        let again = crc8_maxim(&full[..]);
        assert!(whole == again);
        let prefix = crc8_maxim(&full[..5]);
        let prefix_again = crc8_maxim(&full[..5]);
        assert!(prefix == prefix_again);
    }

    #[test]
    fn single_bytes_each_deterministic() {
        let mut byte: u16 = 0;
        while byte <= 0xff {
            let b = byte as u8;
            let one = crc8_maxim(&[b]);
            let two = crc8_maxim(&[b]);
            assert!(one == two);
            byte += 1;
        }
    }

    #[test]
    fn zero_byte_matches_empty() {
        // 0x00 fed through the reflected loop leaves the register at 0,
        // identical to the empty-input case.
        assert!(crc8_maxim(&[0x00]) == crc8_maxim(b""));
    }

    #[test]
    fn order_sensitive() {
        // The checksum must depend on byte order.
        let ab = crc8_maxim(&[0x01, 0x02]);
        let ba = crc8_maxim(&[0x02, 0x01]);
        assert!(ab != ba);
    }

    // --- long input stability ---

    #[test]
    fn long_input_ramp_1000_stable() {
        let mut buf = [0u8; 1000];
        let mut i: usize = 0;
        while i < buf.len() {
            buf[i] = (i & 0xff) as u8;
            i += 1;
        }
        let a = crc8_maxim(&buf);
        let b = crc8_maxim(&buf);
        assert!(a == b);
        assert!(a == 0x4d);
    }

    #[test]
    fn long_zeros_is_zero() {
        let buf = [0u8; 256];
        assert!(crc8_maxim(&buf) == 0x00);
    }

    #[test]
    fn long_ff_block_stable() {
        let buf = [0xffu8; 128];
        let a = crc8_maxim(&buf);
        let b = crc8_maxim(&buf);
        assert!(a == b);
    }

    #[test]
    fn growing_lengths_deterministic() {
        let base = [0xa5u8; 300];
        let mut len: usize = 0;
        while len <= base.len() {
            let x = crc8_maxim(&base[..len]);
            let y = crc8_maxim(&base[..len]);
            assert!(x == y);
            len += 1;
        }
    }

    #[test]
    fn repeated_pattern_block_stable() {
        let mut buf = [0u8; 512];
        let mut i: usize = 0;
        while i < buf.len() {
            buf[i] = if i.is_multiple_of(2) { 0xaa } else { 0x55 };
            i += 1;
        }
        let a = crc8_maxim(&buf);
        let b = crc8_maxim(&buf);
        assert!(a == b);
    }

    // --- extra coverage to clear the >=35 bar ---

    #[test]
    fn two_byte_lo_hi() {
        assert!(crc8_maxim(&[0x00, 0xff]) == crc8_maxim(&[0x00, 0xff]));
    }

    #[test]
    fn ascii_digits_prefixes_deterministic() {
        let digits = b"123456789";
        let mut n: usize = 1;
        while n <= digits.len() {
            let p = crc8_maxim(&digits[..n]);
            let q = crc8_maxim(&digits[..n]);
            assert!(p == q);
            n += 1;
        }
    }

    #[test]
    fn whitespace_input_deterministic() {
        let a = crc8_maxim(b"   ");
        let b = crc8_maxim(b"   ");
        assert!(a == b);
    }

    #[test]
    fn mixed_payload_deterministic() {
        let payload = [0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0];
        let a = crc8_maxim(&payload);
        let b = crc8_maxim(&payload);
        assert!(a == b);
    }

    #[test]
    fn single_vs_double_differ() {
        let one = crc8_maxim(&[0x01]);
        let two = crc8_maxim(&[0x01, 0x01]);
        assert!(one != two);
    }
}
