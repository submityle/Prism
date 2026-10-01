//! `CRC`-16/EN-13757 contract module (single-file, `no_std` + `alloc` friendly).
//!
//! Parameters: width=16, poly=`0x3D65`, init=`0x0`, refin=false, refout=false,
//! xorout=`0xFFFF`, check=`0xc2b7`.
//!
//! The implementation is a plain bit-wise register algorithm. For every input
//! bit we test the register `MSB`, shift the register left, and conditionally
//! fold in the polynomial. Because this parameter set does not reflect input or
//! output, there is no `LSB`-first processing and no final reflection. The
//! `XOR`-out stage inverts all bits of the final register. The algorithm uses
//! only integer operations so it is portable across any `CPU`.

const WIDTH: u32 = 16;
const POLY: u32 = 0x3D65;
const INIT: u32 = 0x0;
const XOROUT: u32 = 0xFFFF;
const REFLECT_IN: bool = false;
const REFLECT_OUT: bool = false;
const MASK: u32 = 0xffff;
const TOPBIT: u32 = 1u32 << (WIDTH - 1);

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

/// Compute the `CRC`-16/EN-13757 checksum of `data`.
pub fn crc16_en13757(data: &[u8]) -> u16 {
    let mut reg = INIT & MASK;
    let mut idx = 0usize;
    while idx < data.len() {
        let byte = data[idx];
        let b = if REFLECT_IN {
            reflect(byte as u32, 8)
        } else {
            byte as u32
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
    use super::crc16_en13757;

    // ---- 24 golden anchors (hard-coded truths from /tmp/b42_vectors.txt) ----

    #[test]
    fn anchor_empty() {
        assert!(crc16_en13757(b"") == 0xffff);
    }

    #[test]
    fn anchor_z00() {
        assert!(crc16_en13757(&[0x00]) == 0xffff);
    }

    #[test]
    fn anchor_ff() {
        assert!(crc16_en13757(&[0xff]) == 0x53b7);
    }

    #[test]
    fn anchor_a() {
        assert!(crc16_en13757(b"a") == 0xe137);
    }

    #[test]
    fn anchor_b() {
        assert!(crc16_en13757(b"b") == 0xa698);
    }

    #[test]
    fn anchor_ab() {
        assert!(crc16_en13757(b"ab") == 0xa674);
    }

    #[test]
    fn anchor_abc() {
        assert!(crc16_en13757(b"abc") == 0x571c);
    }

    #[test]
    fn anchor_two_zeros() {
        assert!(crc16_en13757(&[0, 0]) == 0xffff);
    }

    #[test]
    fn anchor_01() {
        assert!(crc16_en13757(&[0x01]) == 0xc29a);
    }

    #[test]
    fn anchor_02() {
        assert!(crc16_en13757(&[0x02]) == 0x8535);
    }

    #[test]
    fn anchor_7f() {
        assert!(crc16_en13757(&[0x7f]) == 0x29db);
    }

    #[test]
    fn anchor_80() {
        assert!(crc16_en13757(&[0x80]) == 0x8593);
    }

    #[test]
    fn anchor_aa_55() {
        assert!(crc16_en13757(&[0xaa, 0x55]) == 0x7ad0);
    }

    #[test]
    fn anchor_55_aa() {
        assert!(crc16_en13757(&[0x55, 0xaa]) == 0xd9b6);
    }

    #[test]
    fn anchor_deadbeef() {
        assert!(crc16_en13757(&[0xde, 0xad, 0xbe, 0xef]) == 0x27e4);
    }

    #[test]
    fn anchor_hello() {
        assert!(crc16_en13757(b"Hello") == 0x640d);
    }

    #[test]
    fn anchor_fox() {
        assert!(crc16_en13757(b"The quick brown fox") == 0xed92);
    }

    #[test]
    fn anchor_four_zeros() {
        assert!(crc16_en13757(&[0, 0, 0, 0]) == 0xffff);
    }

    #[test]
    fn anchor_four_ff() {
        assert!(crc16_en13757(&[0xff; 4]) == 0xf15e);
    }

    #[test]
    fn anchor_0_15() {
        let mut arr = [0u8; 16];
        let mut i = 0usize;
        while i < 16 {
            arr[i] = i as u8;
            i += 1;
        }
        assert!(crc16_en13757(&arr) == 0x037e);
    }

    #[test]
    fn anchor_12345678() {
        assert!(crc16_en13757(&[0x12, 0x34, 0x56, 0x78]) == 0x02c9);
    }

    #[test]
    fn anchor_check() {
        assert!(crc16_en13757(b"123456789") == 0xc2b7);
    }

    #[test]
    fn anchor_a5_1000() {
        let data = [0xA5u8; 1000];
        assert!(crc16_en13757(&data) == 0x9c90);
    }

    #[test]
    fn anchor_all256() {
        let mut arr = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            arr[i] = i as u8;
            i += 1;
        }
        assert!(crc16_en13757(&arr) == 0xb50d);
    }

    // ---- Behavioural / property tests ----

    #[test]
    fn determinism() {
        let first = crc16_en13757(b"The quick brown fox");
        let second = crc16_en13757(b"The quick brown fox");
        assert!(first == second);
    }

    #[test]
    fn order_sensitivity() {
        assert!(crc16_en13757(&[0xaa, 0x55]) != crc16_en13757(&[0x55, 0xaa]));
    }

    #[test]
    fn a_differs_from_b() {
        assert!(crc16_en13757(b"a") != crc16_en13757(b"b"));
    }

    #[test]
    fn prefix_differs() {
        assert!(crc16_en13757(b"ab") != crc16_en13757(b"abc"));
    }

    #[test]
    fn length_sensitive() {
        assert!(crc16_en13757(&[0x01]) != crc16_en13757(&[0x01, 0x01]));
    }

    #[test]
    fn single_bytes_pairwise_distinct() {
        let mut out = [0u16; 256];
        let mut i = 0usize;
        while i < 256 {
            out[i] = crc16_en13757(&[i as u8]);
            i += 1;
        }
        let mut a = 0usize;
        while a < 256 {
            let mut b = a + 1;
            while b < 256 {
                assert!(out[a] != out[b]);
                b += 1;
            }
            a += 1;
        }
    }

    #[test]
    fn result_within_u16_range() {
        let v = crc16_en13757(b"123456789");
        assert!((0..=0xffffu16).contains(&v));
    }

    #[test]
    fn check_constant_matches() {
        assert!(crc16_en13757(b"123456789") == 0xc2b7);
    }

    #[test]
    fn empty_equals_init_xor_xorout() {
        // init=0, xorout=0xffff: skeleton yields 0 ^ 0xffff = 0xffff.
        assert!(crc16_en13757(b"") == 0xffff);
    }

    #[test]
    fn empty_matches_leading_zero_runs() {
        // Zero bytes keep the register at 0 under this parameter set.
        assert!(crc16_en13757(b"") == crc16_en13757(&[0, 0]));
        assert!(crc16_en13757(&[0, 0]) == crc16_en13757(&[0, 0, 0, 0]));
    }

    #[test]
    fn nonzero_payload_leaves_default() {
        assert!(crc16_en13757(&[0xde, 0xad, 0xbe, 0xef]) != 0xffff);
    }
}
