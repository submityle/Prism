//! `CRC-16/NRSC-5` checksum implementation for particle-engine data integrity.
//!
//! Parameters: width=16, poly=`0x080B`, init=`0xFFFF`, refin=true, refout=true,
//! xorout=`0x0`, check=`0xa066`. This module uses a reflected (`refin`=`refout`=true)
//! bit-wise algorithm so the register and the final value are both reflected.

const WIDTH: u32 = 16;
const POLY: u32 = 0x080B;
const INIT: u32 = 0xFFFF;
const XOROUT: u32 = 0x0;
const REFLECT_IN: bool = true;
const REFLECT_OUT: bool = true;
const MASK: u32 = 0xffff;
const TOPBIT: u32 = 1u32 << (WIDTH - 1);

/// Reflect the low `bits` bits of `value`, reversing their order.
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

/// Compute the `CRC-16/NRSC-5` checksum over `data`.
pub fn crc16_nrsc5(data: &[u8]) -> u16 {
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
    use super::crc16_nrsc5;

    #[test]
    fn vector_empty() {
        assert!(crc16_nrsc5(b"") == 0xffff);
    }

    #[test]
    fn vector_z00() {
        assert!(crc16_nrsc5(&[0x00]) == 0x139c);
    }

    #[test]
    fn vector_ff() {
        assert!(crc16_nrsc5(&[0xff]) == 0x00ff);
    }

    #[test]
    fn vector_a() {
        assert!(crc16_nrsc5(b"a") == 0x7a34);
    }

    #[test]
    fn vector_b() {
        assert!(crc16_nrsc5(b"b") == 0x24d8);
    }

    #[test]
    fn vector_ab() {
        assert!(crc16_nrsc5(b"ab") == 0xcfa8);
    }

    #[test]
    fn vector_abc() {
        assert!(crc16_nrsc5(b"abc") == 0xeb3a);
    }

    #[test]
    fn vector_two_zeros() {
        assert!(crc16_nrsc5(&[0x00, 0x00]) == 0x1190);
    }

    #[test]
    fn vector_01() {
        assert!(crc16_nrsc5(&[0x01]) == 0x2638);
    }

    #[test]
    fn vector_02() {
        assert!(crc16_nrsc5(&[0x02]) == 0x78d4);
    }

    #[test]
    fn vector_7f() {
        assert!(crc16_nrsc5(&[0x7f]) == 0xd0ef);
    }

    #[test]
    fn vector_80() {
        assert!(crc16_nrsc5(&[0x80]) == 0xc38c);
    }

    #[test]
    fn vector_aa_55() {
        assert!(crc16_nrsc5(&[0xaa, 0x55]) == 0x1c13);
    }

    #[test]
    fn vector_55_aa() {
        assert!(crc16_nrsc5(&[0x55, 0xaa]) == 0x0d83);
    }

    #[test]
    fn vector_deadbeef() {
        assert!(crc16_nrsc5(&[0xde, 0xad, 0xbe, 0xef]) == 0xb74d);
    }

    #[test]
    fn vector_hello() {
        assert!(crc16_nrsc5(b"Hello") == 0x4db5);
    }

    #[test]
    fn vector_fox() {
        assert!(crc16_nrsc5(b"The quick brown fox") == 0x86d9);
    }

    #[test]
    fn vector_four_zeros() {
        assert!(crc16_nrsc5(&[0x00; 4]) == 0x5e26);
    }

    #[test]
    fn vector_four_ff() {
        assert!(crc16_nrsc5(&[0xff; 4]) == 0x1190);
    }

    #[test]
    fn vector_0_15() {
        let mut data = [0u8; 16];
        let mut i = 0usize;
        while i < 16 {
            data[i] = i as u8;
            i += 1;
        }
        assert!(crc16_nrsc5(&data) == 0xf290);
    }

    #[test]
    fn vector_12345678() {
        assert!(crc16_nrsc5(&[0x12, 0x34, 0x56, 0x78]) == 0x94a2);
    }

    #[test]
    fn vector_check() {
        assert!(crc16_nrsc5(b"123456789") == 0xa066);
    }

    #[test]
    fn vector_a5_1000() {
        let data = [0xA5u8; 1000];
        assert!(crc16_nrsc5(&data) == 0x6157);
    }

    #[test]
    fn vector_all256() {
        let mut data = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            data[i] = i as u8;
            i += 1;
        }
        assert!(crc16_nrsc5(&data) == 0xc575);
    }

    #[test]
    fn check_constant_matches_spec() {
        const CHECK: u16 = 0xa066;
        assert!(crc16_nrsc5(b"123456789") == CHECK);
    }

    #[test]
    fn determinism_same_input_same_output() {
        let first = crc16_nrsc5(b"The quick brown fox");
        let second = crc16_nrsc5(b"The quick brown fox");
        assert!(first == second);
    }

    #[test]
    fn order_sensitive_aa_55_differs_from_55_aa() {
        let forward = crc16_nrsc5(&[0xaa, 0x55]);
        let reversed = crc16_nrsc5(&[0x55, 0xaa]);
        assert!(forward != reversed);
    }

    #[test]
    fn a_differs_from_b() {
        assert!(crc16_nrsc5(b"a") != crc16_nrsc5(b"b"));
    }

    #[test]
    fn prefix_differs_from_full() {
        let prefix = crc16_nrsc5(b"ab");
        let full = crc16_nrsc5(b"abc");
        assert!(prefix != full);
    }

    #[test]
    fn length_sensitive_zeros() {
        let two = crc16_nrsc5(&[0x00, 0x00]);
        let four = crc16_nrsc5(&[0x00; 4]);
        assert!(two != four);
    }

    #[test]
    fn single_byte_values_pairwise_distinct() {
        let mut seen = [0u16; 256];
        let mut i = 0usize;
        while i < 256 {
            seen[i] = crc16_nrsc5(&[i as u8]);
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
        let value = crc16_nrsc5(b"range check sample");
        assert!((0..=0xffffu16).contains(&value));
    }

    #[test]
    fn empty_differs_from_single_zero() {
        assert!(crc16_nrsc5(b"") != crc16_nrsc5(&[0x00]));
    }

    #[test]
    fn two_distinct_single_bytes_01_02() {
        assert!(crc16_nrsc5(&[0x01]) != crc16_nrsc5(&[0x02]));
    }

    #[test]
    fn appending_byte_changes_result() {
        let base = crc16_nrsc5(b"Hello");
        let extended = crc16_nrsc5(b"Hello!");
        assert!(base != extended);
    }
}
