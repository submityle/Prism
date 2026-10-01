//! `CRC-8/WCDMA` CPU golden reference: width=8, poly=0x9B, init=0x00, refin=true, refout=true, xorout=0x00 (reflected, `LSB`-first, `u8`); check(b"123456789")=0x25.

/// Compute the `CRC-8/WCDMA` checksum over `data`.
///
/// This is the reflected (`LSB`-first) form. The reflected polynomial
/// `REFPOLY` = 0xD9 is the 8-bit bit-reverse of poly 0x9B. The register is
/// seeded with init 0x00, each input byte is folded in via `XOR`, and after
/// processing the final `xorout` is 0x00 (a no-op). The result is a `u8`.
#[must_use]
pub fn crc8_wcdma(data: &[u8]) -> u8 {
    const REFPOLY: u8 = 0xD9;
    let mut crc: u8 = 0x00; // init
    for &b in data {
        crc ^= b;
        let mut bit = 0u32;
        while bit < 8 {
            if (crc & 1) != 0 {
                crc = (crc >> 1) ^ REFPOLY;
            } else {
                crc >>= 1;
            }
            bit += 1;
        }
    }
    crc
}

#[cfg(test)]
mod tests {
    use super::crc8_wcdma;

    // ---- 5 anchors (node_repl verified, check_ok=true) ----

    #[test]
    fn anchor_empty_is_zero() {
        assert!(crc8_wcdma(b"") == 0x00);
    }

    #[test]
    fn anchor_a() {
        assert!(crc8_wcdma(b"a") == 0xe6);
    }

    #[test]
    fn anchor_single_zero() {
        assert!(crc8_wcdma(&[0x00]) == 0x00);
    }

    #[test]
    fn anchor_single_ff() {
        assert!(crc8_wcdma(&[0xff]) == 0xde);
    }

    #[test]
    fn anchor_check_string() {
        assert!(crc8_wcdma(b"123456789") == 0x25);
    }

    // ---- >=10 multibyte hardcoded cases ----

    #[test]
    fn multi_1_2_3() {
        assert!(crc8_wcdma(&[0x01, 0x02, 0x03]) == 0xc9);
    }

    #[test]
    fn multi_10_20_30_40() {
        assert!(crc8_wcdma(&[0x10, 0x20, 0x30, 0x40]) == 0xf8);
    }

    #[test]
    fn multi_aa_bb_cc() {
        assert!(crc8_wcdma(&[0xaa, 0xbb, 0xcc]) == 0x4b);
    }

    #[test]
    fn multi_deadbeef() {
        assert!(crc8_wcdma(&[0xde, 0xad, 0xbe, 0xef]) == 0xcc);
    }

    #[test]
    fn multi_four_zeros() {
        assert!(crc8_wcdma(&[0x00, 0x00, 0x00, 0x00]) == 0x00);
    }

    #[test]
    fn multi_ff_ff() {
        assert!(crc8_wcdma(&[0xff, 0xff]) == 0x53);
    }

    #[test]
    fn multi_12_34_56() {
        assert!(crc8_wcdma(&[0x12, 0x34, 0x56]) == 0xa1);
    }

    #[test]
    fn multi_7f_80_81() {
        assert!(crc8_wcdma(&[0x7f, 0x80, 0x81]) == 0x78);
    }

    #[test]
    fn multi_alternating() {
        assert!(crc8_wcdma(&[0x55, 0xaa, 0x55, 0xaa]) == 0x6f);
    }

    #[test]
    fn multi_cafebabe() {
        assert!(crc8_wcdma(&[0xca, 0xfe, 0xba, 0xbe]) == 0xfc);
    }

    #[test]
    fn multi_1_through_10() {
        assert!(crc8_wcdma(&[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a]) == 0x7c);
    }

    #[test]
    fn multi_ab() {
        assert!(crc8_wcdma(b"ab") == 0xff);
    }

    #[test]
    fn multi_abc() {
        assert!(crc8_wcdma(b"abc") == 0x2b);
    }

    #[test]
    fn multi_hello() {
        assert!(crc8_wcdma(b"Hello") == 0xe1);
    }

    #[test]
    fn multi_00_01() {
        assert!(crc8_wcdma(&[0x00, 0x01]) == 0xd0);
    }

    #[test]
    fn multi_ff_00() {
        assert!(crc8_wcdma(&[0xff, 0x00]) == 0x8d);
    }

    #[test]
    fn multi_40_40() {
        assert!(crc8_wcdma(&[0x40, 0x40]) == 0x81);
    }

    #[test]
    fn multi_ramp() {
        assert!(crc8_wcdma(&[0x11, 0x22, 0x33, 0x44, 0x55]) == 0x21);
    }

    #[test]
    fn multi_ab_cd_ef() {
        assert!(crc8_wcdma(&[0xab, 0xcd, 0xef]) == 0x94);
    }

    #[test]
    fn multi_abcdef() {
        assert!(crc8_wcdma(b"abcdef") == 0x0b);
    }

    #[test]
    fn single_01() {
        assert!(crc8_wcdma(&[0x01]) == 0xd0);
    }

    #[test]
    fn single_80() {
        assert!(crc8_wcdma(&[0x80]) == 0xd9);
    }

    #[test]
    fn single_7f() {
        assert!(crc8_wcdma(&[0x7f]) == 0x07);
    }

    // ---- determinism ----

    #[test]
    fn deterministic_repeat() {
        let data = b"deterministic";
        let first = crc8_wcdma(data);
        let second = crc8_wcdma(data);
        assert!(first == second);
    }

    #[test]
    fn deterministic_over_sample() {
        // Calling twice over a sweep of single-byte inputs is stable.
        let mut stable = true;
        let mut cur: u8 = 0;
        loop {
            let buf = [cur];
            if crc8_wcdma(&buf) != crc8_wcdma(&buf) {
                stable = false;
            }
            if cur == 255 {
                break;
            }
            cur = cur.wrapping_add(1);
        }
        assert!(stable);
    }

    // ---- empty input == 0 ----

    #[test]
    fn empty_slice_is_zero() {
        let empty: [u8; 0] = [];
        assert!(crc8_wcdma(&empty) == 0x00);
    }

    // ---- sampling / divisibility property ----

    #[test]
    fn sample_leading_zero_invariant() {
        // init is 0x00, so leading zero bytes do not change the result.
        assert!(crc8_wcdma(&[0x00, 0x01]) == crc8_wcdma(&[0x01]));
        assert!(crc8_wcdma(&[0x00, 0x00, 0xff]) == crc8_wcdma(&[0xff]));
    }

    #[test]
    fn sample_appended_crc_residue_zero() {
        // With xorout 0x00, appending the computed CRC byte yields residue 0x00.
        let msgs: [&[u8]; 5] = [
            &[0x61],
            &[0x31, 0x32, 0x33],
            &[0xde, 0xad],
            &[0x00, 0x00, 0x00],
            &[0xff],
        ];
        let mut all_zero = true;
        let mut idx = 0usize;
        while idx < msgs.len() {
            let m = msgs[idx];
            let c = crc8_wcdma(m);
            // Rebuild message + crc into a fixed buffer (max len here is 3 + 1).
            let mut buf = [0u8; 8];
            let mut j = 0usize;
            while j < m.len() {
                buf[j] = m[j];
                j += 1;
            }
            buf[j] = c;
            if crc8_wcdma(&buf[..=j]) != 0x00 {
                all_zero = false;
            }
            idx += 1;
        }
        assert!(all_zero);
    }

    #[test]
    fn sample_order_sensitive() {
        // Reordering bytes generally changes the checksum.
        assert!(crc8_wcdma(&[0x01, 0x02]) != crc8_wcdma(&[0x02, 0x01]));
    }

    // ---- long input stability ----

    #[test]
    fn long_ramp_256() {
        let mut buf = [0u8; 256];
        let mut idx = 0usize;
        let mut cur: u8 = 0;
        loop {
            buf[idx] = cur;
            if cur == 255 {
                break;
            }
            cur = cur.wrapping_add(1);
            idx += 1;
        }
        assert!(crc8_wcdma(&buf) == 0x59);
    }

    #[test]
    fn long_repeated_5a_512() {
        let buf = [0x5au8; 512];
        assert!(crc8_wcdma(&buf) == 0x55);
    }

    #[test]
    fn long_cycling_1000() {
        let mut buf = [0u8; 1000];
        let mut idx = 0usize;
        let mut cur: u8 = 0;
        while idx < 1000 {
            buf[idx] = cur;
            cur = cur.wrapping_add(1);
            idx += 1;
        }
        assert!(crc8_wcdma(&buf) == 0x88);
    }
}
