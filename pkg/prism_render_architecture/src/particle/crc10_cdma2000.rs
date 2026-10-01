//! `CRC-10/CDMA2000` checksum over byte slices.
//!
//! Parameters: width=10, poly=`0x3D9`, init=`0x3FF`, refin=false,
//! refout=false, xorout=`0x000`. The implementation uses a canonical
//! `MSB`-first bit-at-a-time shift register and stays pure-integer so it
//! is suitable for a `no_std` + `alloc` environment without touching any
//! `CPU`/`GPU` floating-point units.

/// Generator polynomial for `CRC-10/CDMA2000` (`x^10 + ... `), low 10 bits.
const POLY: u16 = 0x3D9;
/// Mask selecting the low 10 bits of the register.
const MASK: u16 = 0x03FF;
/// Initial register value (`init`).
const INIT: u16 = 0x3FF;

/// Compute the `CRC-10/CDMA2000` checksum of `data`.
///
/// Returns the 10-bit remainder in the low bits of the `u16`.
pub fn crc10_cdma2000(data: &[u8]) -> u16 {
    let mut reg: u16 = INIT;
    for &byte in data {
        for i in 0..8u32 {
            // refin=false: consume the most significant bit first.
            let bit = ((byte >> (7 - i)) & 1) as u16;
            let hi = (reg >> 9) & 1;
            let fb = hi ^ bit;
            reg = (reg << 1) & MASK;
            if fb != 0 {
                reg ^= POLY;
            }
        }
    }
    // refout=false, xorout=0: no reflection or final xor.
    reg & MASK
}

#[cfg(test)]
mod tests {
    use super::crc10_cdma2000;

    // ----- anchor vectors (ground truth) -----

    #[test]
    fn anchor_empty() {
        assert!(crc10_cdma2000(b"") == 0x3ff);
    }

    #[test]
    fn anchor_a() {
        assert!(crc10_cdma2000(b"a") == 0x334);
    }

    #[test]
    fn anchor_zero_byte() {
        assert!(crc10_cdma2000(&[0x00]) == 0x356);
    }

    #[test]
    fn anchor_ff_byte() {
        assert!(crc10_cdma2000(&[0xff]) == 0x300);
    }

    #[test]
    fn anchor_check_value() {
        assert!(crc10_cdma2000(b"123456789") == 0x233);
    }

    // ----- single-byte computed vectors -----

    #[test]
    fn single_0x01() {
        assert!(crc10_cdma2000(&[0x01]) == 0x08f);
    }

    #[test]
    fn single_0x55() {
        assert!(crc10_cdma2000(&[0x55]) == 0x1d3);
    }

    #[test]
    fn single_0xaa() {
        assert!(crc10_cdma2000(&[0xaa]) == 0x185);
    }

    #[test]
    fn single_0x7f() {
        assert!(crc10_cdma2000(&[0x7f]) == 0x17d);
    }

    #[test]
    fn single_0x80() {
        assert!(crc10_cdma2000(&[0x80]) == 0x12b);
    }

    #[test]
    fn single_z_upper() {
        assert!(crc10_cdma2000(b"Z") == 0x31b);
    }

    #[test]
    fn single_space() {
        assert!(crc10_cdma2000(b" ") == 0x23f);
    }

    #[test]
    fn single_newline() {
        assert!(crc10_cdma2000(b"\n") == 0x291);
    }

    // ----- multi-byte computed vectors -----

    #[test]
    fn multi_abc() {
        assert!(crc10_cdma2000(b"abc") == 0x38e);
    }

    #[test]
    fn multi_ab_upper() {
        assert!(crc10_cdma2000(b"AB") == 0x1f8);
    }

    #[test]
    fn multi_hello() {
        assert!(crc10_cdma2000(b"hello") == 0x19f);
    }

    #[test]
    fn multi_00_01() {
        assert!(crc10_cdma2000(&[0x00, 0x01]) == 0x121);
    }

    #[test]
    fn multi_ff_ff() {
        assert!(crc10_cdma2000(&[0xff, 0xff]) == 0x0f9);
    }

    #[test]
    fn multi_deadbeef() {
        assert!(crc10_cdma2000(&[0xde, 0xad, 0xbe, 0xef]) == 0x018);
    }

    #[test]
    fn multi_counting() {
        assert!(crc10_cdma2000(&[1, 2, 3, 4, 5]) == 0x3d5);
    }

    #[test]
    fn multi_zeros4() {
        assert!(crc10_cdma2000(&[0, 0, 0, 0]) == 0x3ee);
    }

    #[test]
    fn multi_ones4() {
        assert!(crc10_cdma2000(&[0xff, 0xff, 0xff, 0xff]) == 0x1d0);
    }

    #[test]
    fn multi_sentence() {
        assert!(crc10_cdma2000(b"The quick brown fox") == 0x247);
    }

    #[test]
    fn multi_12() {
        assert!(crc10_cdma2000(b"12") == 0x186);
    }

    #[test]
    fn multi_123() {
        assert!(crc10_cdma2000(b"123") == 0x3e1);
    }

    // ----- range / bound properties -----

    #[test]
    fn result_within_10_bits_empty() {
        let v = crc10_cdma2000(b"");
        assert!((0x000..=0x3ff).contains(&v));
    }

    #[test]
    fn result_within_10_bits_check() {
        let v = crc10_cdma2000(b"123456789");
        assert!((0x000..=0x3ff).contains(&v));
    }

    #[test]
    fn result_high_bits_clear() {
        let v = crc10_cdma2000(&[0xde, 0xad, 0xbe, 0xef]);
        assert!((v & !MASK_LOCAL) == 0);
    }

    const MASK_LOCAL: u16 = 0x03FF;

    #[test]
    fn result_bits_fit_many() {
        let samples: &[&[u8]] = &[b"", b"a", b"abc", b"123456789", &[0xff, 0x00]];
        let mut all_fit = true;
        for s in samples {
            if (crc10_cdma2000(s) & !MASK_LOCAL) != 0 {
                all_fit = false;
            }
        }
        assert!(all_fit);
    }

    // ----- determinism -----

    #[test]
    fn deterministic_check() {
        let a = crc10_cdma2000(b"123456789");
        let b = crc10_cdma2000(b"123456789");
        assert!(a == b);
    }

    #[test]
    fn deterministic_empty() {
        assert!(crc10_cdma2000(b"") == crc10_cdma2000(&[]));
    }

    #[test]
    fn deterministic_repeat_runs() {
        let first = crc10_cdma2000(b"deadbeef");
        let mut same = true;
        for _ in 0..16u32 {
            if crc10_cdma2000(b"deadbeef") != first {
                same = false;
            }
        }
        assert!(same);
    }

    // ----- incremental / structural -----

    #[test]
    fn incremental_prefix_differs() {
        let p1 = crc10_cdma2000(b"12");
        let p2 = crc10_cdma2000(b"123");
        assert!(p1 != p2);
    }

    #[test]
    fn incremental_growing_lengths() {
        let s0 = crc10_cdma2000(b"1");
        let s1 = crc10_cdma2000(b"12");
        let s2 = crc10_cdma2000(b"123");
        assert!(s0 != s1 && s1 != s2);
    }

    #[test]
    fn length_sensitivity_zero_padding() {
        let one = crc10_cdma2000(&[0x00]);
        let two = crc10_cdma2000(&[0x00, 0x00]);
        assert!(one != two);
    }

    #[test]
    fn order_sensitivity() {
        let ab = crc10_cdma2000(&[0x01, 0x02]);
        let ba = crc10_cdma2000(&[0x02, 0x01]);
        assert!(ab != ba);
    }

    #[test]
    fn single_bit_change_detected() {
        let a = crc10_cdma2000(&[0x00]);
        let b = crc10_cdma2000(&[0x01]);
        assert!(a != b);
    }

    #[test]
    fn empty_equals_init() {
        // With no input the register keeps its init value.
        assert!(crc10_cdma2000(b"") == 0x3ff);
    }

    #[test]
    fn case_sensitivity() {
        let lower = crc10_cdma2000(b"abc");
        let upper = crc10_cdma2000(b"ABC");
        assert!(lower != upper);
    }

    #[test]
    fn repeated_byte_distinct() {
        let one = crc10_cdma2000(&[0xaa]);
        let two = crc10_cdma2000(&[0xaa, 0xaa]);
        let three = crc10_cdma2000(&[0xaa, 0xaa, 0xaa]);
        assert!(one != two && two != three && one != three);
    }

    #[test]
    fn slice_matches_array() {
        let arr = [0x12u8, 0x34, 0x56];
        let via_slice = crc10_cdma2000(&arr[..]);
        let via_ref = crc10_cdma2000(&arr);
        assert!(via_slice == via_ref);
    }

    #[test]
    fn subslice_consistency() {
        let data = [1u8, 2, 3, 4, 5];
        let whole = crc10_cdma2000(&data);
        let part = crc10_cdma2000(&data[0..5]);
        assert!(whole == part);
    }

    #[test]
    fn differs_from_empty_for_zero_byte() {
        assert!(crc10_cdma2000(&[0x00]) != crc10_cdma2000(b""));
    }
}
