//! `CRC`-16/TMS37157 contract module.
//!
//! Bit-wise reference implementation of the `CRC`-16/TMS37157 checksum.
//! Parameters: width=16, poly=0x1021, init=0x89EC, refin=true, refout=true,
//! xorout=0x0, check=0x26b1.
//!
//! The algorithm reflects every input byte (`LSB`-first), runs the classic
//! shift-register loop, reflects the final register (`MSB`/`LSB` swap over the
//! full width), then applies the `XOR` output mask. All arithmetic is integer
//! only; this module performs no floating-point work.

const WIDTH: u32 = 16;
const POLY: u32 = 0x1021;
const INIT: u32 = 0x89EC;
const XOROUT: u32 = 0x0;
const REFLECT_IN: bool = true;
const REFLECT_OUT: bool = true;
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

/// Computes the `CRC`-16/TMS37157 checksum over `data`.
pub fn crc16_tms37157(data: &[u8]) -> u16 {
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
    use super::crc16_tms37157;

    #[test]
    fn vec_empty() {
        assert!(crc16_tms37157(b"") == 0x3791);
    }

    #[test]
    fn vec_z00() {
        assert!(crc16_tms37157(&[0x00]) == 0x8537);
    }

    #[test]
    fn vec_ff() {
        assert!(crc16_tms37157(&[0xff]) == 0x8a4f);
    }

    #[test]
    fn vec_a() {
        assert!(crc16_tms37157(b"a") == 0xf7b8);
    }

    #[test]
    fn vec_b() {
        assert!(crc16_tms37157(b"b") == 0xc523);
    }

    #[test]
    fn vec_ab() {
        assert!(crc16_tms37157(b"ab") == 0x7920);
    }

    #[test]
    fn vec_abc() {
        assert!(crc16_tms37157(b"abc") == 0x70e6);
    }

    #[test]
    fn vec_two_zeros() {
        assert!(crc16_tms37157(&[0, 0]) == 0x45b9);
    }

    #[test]
    fn vec_01() {
        assert!(crc16_tms37157(&[0x01]) == 0x94be);
    }

    #[test]
    fn vec_02() {
        assert!(crc16_tms37157(&[0x02]) == 0xa625);
    }

    #[test]
    fn vec_7f() {
        assert!(crc16_tms37157(&[0x7f]) == 0x0e47);
    }

    #[test]
    fn vec_80() {
        assert!(crc16_tms37157(&[0x80]) == 0x013f);
    }

    #[test]
    fn vec_aa_55() {
        assert!(crc16_tms37157(&[0xaa, 0x55]) == 0x121e);
    }

    #[test]
    fn vec_55_aa() {
        assert!(crc16_tms37157(&[0x55, 0xaa]) == 0xe2a6);
    }

    #[test]
    fn vec_deadbeef() {
        assert!(crc16_tms37157(&[0xde, 0xad, 0xbe, 0xef]) == 0xe1ca);
    }

    #[test]
    fn vec_hello() {
        assert!(crc16_tms37157(b"Hello") == 0xb5d9);
    }

    #[test]
    fn vec_fox() {
        assert!(crc16_tms37157(b"The quick brown fox") == 0x7b7c);
    }

    #[test]
    fn vec_four_zeros() {
        assert!(crc16_tms37157(&[0, 0, 0, 0]) == 0xf8df);
    }

    #[test]
    fn vec_four_ff() {
        assert!(crc16_tms37157(&[0xff; 4]) == 0x0b46);
    }

    #[test]
    fn vec_0_15() {
        let mut buf = [0u8; 16];
        let mut i = 0usize;
        while i < 16 {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc16_tms37157(&buf) == 0xa837);
    }

    #[test]
    fn vec_12345678() {
        assert!(crc16_tms37157(&[0x12, 0x34, 0x56, 0x78]) == 0x9f2f);
    }

    #[test]
    fn vec_check() {
        assert!(crc16_tms37157(b"123456789") == 0x26b1);
    }

    #[test]
    fn vec_a5_1000() {
        let buf = [0xA5u8; 1000];
        assert!(crc16_tms37157(&buf) == 0xfcfc);
    }

    #[test]
    fn vec_all256() {
        let mut buf = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc16_tms37157(&buf) == 0x8920);
    }

    #[test]
    fn determinism() {
        let data = [0x12u8, 0x34, 0x56, 0x78, 0x9a];
        assert!(crc16_tms37157(&data) == crc16_tms37157(&data));
    }

    #[test]
    fn order_sensitivity() {
        assert!(crc16_tms37157(&[0xaa, 0x55]) != crc16_tms37157(&[0x55, 0xaa]));
    }

    #[test]
    fn a_differs_from_b() {
        assert!(crc16_tms37157(b"a") != crc16_tms37157(b"b"));
    }

    #[test]
    fn empty_matches_reflected_init() {
        // refout=true, xorout=0: empty input yields reflect(INIT, WIDTH).
        assert!(crc16_tms37157(b"") == 0x3791);
    }

    #[test]
    fn single_bytes_distinct() {
        let mut seen = [0u16; 256];
        let mut i = 0usize;
        while i < 256 {
            seen[i] = crc16_tms37157(&[i as u8]);
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
    fn prefix_differs() {
        assert!(crc16_tms37157(b"ab") != crc16_tms37157(b"abc"));
    }

    #[test]
    fn length_sensitive() {
        assert!(crc16_tms37157(&[0, 0]) != crc16_tms37157(&[0, 0, 0]));
    }

    #[test]
    fn result_within_range() {
        let v = crc16_tms37157(b"range check");
        assert!((0..=0xffffu16).contains(&v));
    }

    #[test]
    fn check_constant() {
        const CHECK: u16 = 0x26b1;
        assert!(crc16_tms37157(b"123456789") == CHECK);
    }

    #[test]
    fn appended_zero_changes_result() {
        let base = crc16_tms37157(b"abc");
        let extended = crc16_tms37157(&[b'a', b'b', b'c', 0]);
        assert!(base != extended);
    }

    #[test]
    fn repeated_calls_stable_over_vectors() {
        assert!(crc16_tms37157(b"") == 0x3791);
        assert!(crc16_tms37157(&[0x00]) == 0x8537);
        assert!(crc16_tms37157(b"123456789") == 0x26b1);
    }
}
