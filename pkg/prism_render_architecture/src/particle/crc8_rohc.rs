//! `CRC-8/ROHC` CPU golden reference (reflected, `LSB`-first).
//!
//! Parameters: width=8, poly=0x07, init=0xFF, refin=true, refout=true,
//! xorout=0x00. The reflected polynomial `REFPOLY` is 0xE0 (the 8-bit
//! bit-reverse of 0x07). Because refin and refout are both true and init is
//! symmetric (0xFF), the shift register runs `LSB`-first with an `XOR` of
//! `REFPOLY` on each set low bit. The result is a `u8` with no final `XOR`.
//!
//! check (`b"123456789"`) == 0xd0.

/// Reflected polynomial: the 8-bit bit-reverse of 0x07.
const REFPOLY: u8 = 0xE0;

/// Compute the `CRC-8/ROHC` checksum of `data`.
///
/// Reflected (`LSB`-first) implementation. Returns the final `u8` register
/// value; xorout is 0x00 so no post-`XOR` is applied.
pub fn crc8_rohc(data: &[u8]) -> u8 {
    let mut crc: u8 = 0xFF; // init (symmetric under reflection)
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
    crc // xorout = 0x00
}

#[cfg(test)]
mod tests {
    use super::crc8_rohc;

    // ---- 5 anchors ----
    #[test]
    fn anchor_empty() {
        assert!(crc8_rohc(b"") == 0xff);
    }

    #[test]
    fn anchor_a() {
        assert!(crc8_rohc(b"a") == 0x16);
    }

    #[test]
    fn anchor_zero_byte() {
        assert!(crc8_rohc(&[0x00]) == 0xcf);
    }

    #[test]
    fn anchor_ff_byte() {
        assert!(crc8_rohc(&[0xff]) == 0x00);
    }

    #[test]
    fn anchor_check_value() {
        assert!(crc8_rohc(b"123456789") == 0xd0);
    }

    // ---- single-byte vectors ----
    #[test]
    fn single_0x01() {
        assert!(crc8_rohc(&[0x01]) == 0x5e);
    }

    #[test]
    fn single_0x80() {
        assert!(crc8_rohc(&[0x80]) == 0x2f);
    }

    #[test]
    fn single_0x7f() {
        assert!(crc8_rohc(&[0x7f]) == 0xe0);
    }

    #[test]
    fn single_0x55() {
        assert!(crc8_rohc(&[0x55]) == 0x35);
    }

    #[test]
    fn single_0xaa() {
        assert!(crc8_rohc(&[0xaa]) == 0xfa);
    }

    #[test]
    fn single_0x0f() {
        assert!(crc8_rohc(&[0x0f]) == 0xb4);
    }

    #[test]
    fn single_0xf0() {
        assert!(crc8_rohc(&[0xf0]) == 0x7b);
    }

    // ---- multi-byte hardcoded vectors (>= 10) ----
    #[test]
    fn multi_123() {
        assert!(crc8_rohc(&[1, 2, 3]) == 0xac);
    }

    #[test]
    fn multi_deadbeef() {
        assert!(crc8_rohc(&[0xde, 0xad, 0xbe, 0xef]) == 0xc3);
    }

    #[test]
    fn multi_four_zeros() {
        assert!(crc8_rohc(&[0x00, 0x00, 0x00, 0x00]) == 0x8b);
    }

    #[test]
    fn multi_four_ffs() {
        assert!(crc8_rohc(&[0xff, 0xff, 0xff, 0xff]) == 0xf0);
    }

    #[test]
    fn multi_ramp5() {
        assert!(crc8_rohc(&[0x10, 0x20, 0x30, 0x40, 0x50]) == 0x79);
    }

    #[test]
    fn multi_hello() {
        assert!(crc8_rohc(b"hello") == 0x38);
    }

    #[test]
    fn multi_prism() {
        assert!(crc8_rohc(b"Prism") == 0xfa);
    }

    #[test]
    fn multi_aa55() {
        assert!(crc8_rohc(&[0xaa, 0x55]) == 0xa3);
    }

    #[test]
    fn multi_cafebabe() {
        assert!(crc8_rohc(&[0xca, 0xfe, 0xba, 0xbe]) == 0x65);
    }

    #[test]
    fn multi_quick_fox() {
        assert!(crc8_rohc(b"The quick brown fox") == 0x52);
    }

    #[test]
    fn multi_abc() {
        assert!(crc8_rohc(b"abc") == 0x24);
    }

    #[test]
    fn multi_message_digest() {
        assert!(crc8_rohc(b"message digest") == 0xdc);
    }

    #[test]
    fn multi_crc() {
        assert!(crc8_rohc(b"CRC") == 0x97);
    }

    #[test]
    fn multi_rohc() {
        assert!(crc8_rohc(b"ROHC") == 0x8a);
    }

    #[test]
    fn multi_0x0100() {
        assert!(crc8_rohc(&[0x01, 0x00]) == 0x86);
    }

    #[test]
    fn multi_0x0001() {
        assert!(crc8_rohc(&[0x00, 0x01]) == 0x7a);
    }

    #[test]
    fn multi_ff00() {
        assert!(crc8_rohc(&[0xff, 0x00]) == 0x00);
    }

    #[test]
    fn multi_00ff() {
        assert!(crc8_rohc(&[0x00, 0xff]) == 0x24);
    }

    #[test]
    fn multi_12345678() {
        assert!(crc8_rohc(&[0x12, 0x34, 0x56, 0x78]) == 0x76);
    }

    #[test]
    fn multi_9abcdef0() {
        assert!(crc8_rohc(&[0x9a, 0xbc, 0xde, 0xf0]) == 0xc7);
    }

    // ---- empty input is init (0xff) ----
    #[test]
    fn empty_is_init() {
        assert!(crc8_rohc(&[]) == 0xff);
    }

    // ---- determinism: same input -> same output ----
    #[test]
    fn deterministic_repeat() {
        let data = b"determinism-probe";
        let first = crc8_rohc(data);
        let second = crc8_rohc(data);
        assert!(first == second);
    }

    #[test]
    fn deterministic_across_slices() {
        let a = crc8_rohc(&[0x11, 0x22, 0x33, 0x44]);
        let b = crc8_rohc(&[0x11, 0x22, 0x33, 0x44]);
        let c = crc8_rohc(&[0x11, 0x22, 0x33, 0x44]);
        assert!(a == b);
        assert!(b == c);
    }

    // ---- sampling separability: single-byte map is a bijection ----
    #[test]
    fn single_byte_map_is_bijection() {
        let mut seen = [false; 256];
        let mut i: u16 = 0;
        while i < 256 {
            let crc = crc8_rohc(&[i as u8]);
            let idx = crc as usize;
            assert!(!seen[idx]);
            seen[idx] = true;
            i += 1;
        }
        let mut all = true;
        let mut j = 0usize;
        while j < 256 {
            if !seen[j] {
                all = false;
            }
            j += 1;
        }
        assert!(all);
    }

    #[test]
    fn sampled_distinct_vectors() {
        let va = crc8_rohc(b"sample-A");
        let vb = crc8_rohc(b"sample-B");
        let vc = crc8_rohc(b"sample-C");
        assert!(va != vb);
        assert!(vb != vc);
        assert!(va != vc);
    }

    // ---- long input stability ----
    #[test]
    fn long_ramp_16() {
        let mut buf = [0u8; 16];
        let mut i = 0usize;
        while i < 16 {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc8_rohc(&buf) == 0x5a);
    }

    #[test]
    fn long_ramp_256() {
        let mut buf = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc8_rohc(&buf) == 0x8e);
    }

    #[test]
    fn long_ramp_255() {
        let mut buf = [0u8; 255];
        let mut i = 0usize;
        while i < 255 {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc8_rohc(&buf) == 0x6c);
    }

    #[test]
    fn long_constant_block_stable() {
        let buf = [0x42u8; 100];
        let first = crc8_rohc(&buf);
        let second = crc8_rohc(&buf);
        assert!(first == 0x88);
        assert!(first == second);
    }

    #[test]
    fn long_strided_block() {
        let mut buf = [0u8; 32];
        let mut i = 0usize;
        while i < 32 {
            buf[i] = ((i * 7) & 0xff) as u8;
            i += 1;
        }
        assert!(crc8_rohc(&buf) == 0x98);
    }

    #[test]
    fn long_input_stable_recompute() {
        // A long heterogeneous buffer must hash to a fixed, repeatable value.
        let data = b"abcdefghijklmnopqrstuvwxyz0123456789";
        let first = crc8_rohc(data);
        let second = crc8_rohc(data);
        assert!(first == 0x58);
        assert!(first == second);
    }
}
