//! `CRC-7/ROHC` checksum over byte slices (particle contract primitive).
//!
//! Catalog parameters: width=7, poly=`0x4F`, init=`0x7F`, refin=true,
//! refout=true, xorout=`0x00`. Because `refin == refout == true`, the
//! implementation runs least-significant-bit-first using the reflected
//! polynomial `bitreverse(0x4F, 7) == 0x79`. The reflected init value
//! `0x7F` (all seven bits set) is its own reflection, so it is reused
//! directly. Every step is masked back to seven bits because the width is
//! smaller than one byte. The design is `no_std` + `alloc` friendly, uses
//! only integer arithmetic, and allocates nothing.

/// Compute the `CRC-7/ROHC` checksum of `data`.
///
/// Returns a seven-bit value (always in the range `0x00..=0x7F`). The
/// `ROHC` catalog `check` value `crc7_rohc(b"123456789") == 0x53` is
/// covered by the test anchors.
pub fn crc7_rohc(data: &[u8]) -> u8 {
    const REFPOLY: u8 = 0x79; // bit-reverse of 0x4F over 7 bits
    const MASK: u8 = 0x7F;
    let mut crc: u8 = 0x7F; // reflected init 0x7F (seven bits all set)
    for &b in data {
        crc ^= b; // high bit (if any) is cleared by the per-step mask below
        for _ in 0..8 {
            if (crc & 1) != 0 {
                crc = ((crc >> 1) ^ REFPOLY) & MASK;
            } else {
                crc = (crc >> 1) & MASK;
            }
        }
        crc &= MASK;
    }
    crc & MASK
}

#[cfg(test)]
mod tests {
    use super::*;

    const MASK: u8 = 0x7F;

    // ---- 5 hard anchor vectors (ground truth) ----
    #[test]
    fn anchor_empty() {
        assert!(crc7_rohc(b"") == 0x7f);
    }
    #[test]
    fn anchor_a() {
        assert!(crc7_rohc(b"a") == 0x18);
    }
    #[test]
    fn anchor_zero_byte() {
        assert!(crc7_rohc(&[0x00]) == 0x46);
    }
    #[test]
    fn anchor_ff_byte() {
        assert!(crc7_rohc(&[0xff]) == 0x79);
    }
    #[test]
    fn anchor_check_value() {
        assert!(crc7_rohc(b"123456789") == 0x53);
    }

    // ---- additional known-answer vectors (computed from this definition) ----
    #[test]
    fn kat_abc() {
        assert!(crc7_rohc(b"abc") == 0x4a);
    }
    #[test]
    fn kat_upper_a() {
        assert!(crc7_rohc(b"A") == 0x43);
    }
    #[test]
    fn kat_hello() {
        assert!(crc7_rohc(b"hello") == 0x0f);
    }
    #[test]
    fn kat_hello_world() {
        assert!(crc7_rohc(b"Hello, World!") == 0x57);
    }
    #[test]
    fn kat_two_zeros() {
        assert!(crc7_rohc(&[0x00, 0x00]) == 0x23);
    }
    #[test]
    fn kat_three_zeros() {
        assert!(crc7_rohc(&[0x00, 0x00, 0x00]) == 0x68);
    }
    #[test]
    fn kat_four_zeros() {
        assert!(crc7_rohc(&[0x00, 0x00, 0x00, 0x00]) == 0x34);
    }
    #[test]
    fn kat_two_ff() {
        assert!(crc7_rohc(&[0xff, 0xff]) == 0x1f);
    }
    #[test]
    fn kat_seq_1234() {
        assert!(crc7_rohc(&[1, 2, 3, 4]) == 0x69);
    }
    #[test]
    fn kat_byte_0x80() {
        assert!(crc7_rohc(&[0x80]) == 0x3f);
    }
    #[test]
    fn kat_byte_0x01() {
        assert!(crc7_rohc(&[0x01]) == 0x06);
    }
    #[test]
    fn kat_byte_0x7f() {
        assert!(crc7_rohc(&[0x7f]) == 0x00);
    }
    #[test]
    fn kat_inc8() {
        let data: [u8; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
        assert!(crc7_rohc(&data) == 0x2a);
    }
    #[test]
    fn kat_inc16() {
        let mut data = [0u8; 16];
        let mut i = 0usize;
        while i < 16 {
            data[i] = i as u8;
            i += 1;
        }
        assert!(crc7_rohc(&data) == 0x15);
    }
    #[test]
    fn kat_inc256() {
        let mut data = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            data[i] = i as u8;
            i += 1;
        }
        assert!(crc7_rohc(&data) == 0x4c);
    }
    #[test]
    fn kat_quick_fox() {
        assert!(crc7_rohc(b"The quick brown fox jumps over the lazy dog") == 0x63);
    }
    #[test]
    fn kat_aa() {
        assert!(crc7_rohc(b"aa") == 0x20);
    }
    #[test]
    fn kat_aaa() {
        assert!(crc7_rohc(b"aaa") == 0x05);
    }
    #[test]
    fn kat_deadbeef() {
        assert!(crc7_rohc(&[0xde, 0xad, 0xbe, 0xef]) == 0x31);
    }
    #[test]
    fn kat_all_55() {
        assert!(crc7_rohc(&[0x55, 0x55, 0x55, 0x55]) == 0x25);
    }
    #[test]
    fn kat_all_aa() {
        assert!(crc7_rohc(&[0xAA, 0xAA, 0xAA, 0xAA]) == 0x16);
    }
    #[test]
    fn kat_str_12() {
        assert!(crc7_rohc(b"12") == 0x16);
    }
    #[test]
    fn kat_str_1() {
        assert!(crc7_rohc(b"1") == 0x09);
    }

    // ---- incremental / growing-prefix behaviour ----
    #[test]
    fn incremental_check_prefixes() {
        // Prefixes of the catalog "123456789" string must land on the
        // established per-length values and finish at the check value.
        assert!(crc7_rohc(b"1") == 0x09);
        assert!(crc7_rohc(b"12") == 0x16);
        assert!(crc7_rohc(b"123456789") == 0x53);
    }
    #[test]
    fn growing_zero_run_changes() {
        let a = crc7_rohc(&[0x00]);
        let b = crc7_rohc(&[0x00, 0x00]);
        let c = crc7_rohc(&[0x00, 0x00, 0x00]);
        assert!(a != b);
        assert!(b != c);
        assert!(a != c);
    }

    // ---- boundary / range invariants ----
    #[test]
    fn output_within_seven_bits_single() {
        let mut b: u16 = 0;
        while b <= 0xFF {
            let out = crc7_rohc(&[b as u8]);
            assert!(out <= MASK);
            b += 1;
        }
    }
    #[test]
    fn output_within_seven_bits_pairs() {
        let samples: [[u8; 2]; 6] = [
            [0x00, 0x00],
            [0xFF, 0xFF],
            [0x00, 0xFF],
            [0xFF, 0x00],
            [0x55, 0xAA],
            [0x7F, 0x80],
        ];
        for pair in &samples {
            assert!(crc7_rohc(pair) <= MASK);
        }
    }
    #[test]
    fn empty_equals_reflected_init() {
        // init 0x7F reflected over 7 bits is still 0x7F.
        assert!(crc7_rohc(&[]) == 0x7F);
    }
    #[test]
    fn high_bit_input_is_masked_consistently() {
        // 0x80 and 0x00 differ only in the eighth (discarded on shift) bit,
        // yet still produce distinct seven-bit results per the definition.
        assert!(crc7_rohc(&[0x80]) == 0x3f);
        assert!(crc7_rohc(&[0x00]) == 0x46);
        assert!(crc7_rohc(&[0x80]) != crc7_rohc(&[0x00]));
    }

    // ---- determinism ----
    #[test]
    fn deterministic_repeat_check() {
        assert!(crc7_rohc(b"123456789") == crc7_rohc(b"123456789"));
    }
    #[test]
    fn deterministic_repeat_empty() {
        assert!(crc7_rohc(b"") == crc7_rohc(b""));
    }
    #[test]
    fn deterministic_across_buffers() {
        let one: [u8; 3] = [b'a', b'b', b'c'];
        let two: [u8; 3] = [0x61, 0x62, 0x63];
        assert!(crc7_rohc(&one) == crc7_rohc(&two));
    }
    #[test]
    fn deterministic_many_iterations() {
        let data: [u8; 5] = [0xDE, 0xAD, 0xBE, 0xEF, 0x00];
        let first = crc7_rohc(&data);
        for _ in 0..64 {
            assert!(crc7_rohc(&data) == first);
        }
    }

    // ---- distinctness / sensitivity ----
    #[test]
    fn distinct_single_bytes_mostly_differ() {
        // Specific neighbouring bytes must not collide here.
        assert!(crc7_rohc(&[0x01]) != crc7_rohc(&[0x02]));
        assert!(crc7_rohc(&[0x00]) != crc7_rohc(&[0x01]));
    }
    #[test]
    fn order_sensitivity() {
        assert!(crc7_rohc(&[0x01, 0x02]) != crc7_rohc(&[0x02, 0x01]));
    }
    #[test]
    fn empty_differs_from_zero_byte() {
        assert!(crc7_rohc(&[]) != crc7_rohc(&[0x00]));
    }
}
