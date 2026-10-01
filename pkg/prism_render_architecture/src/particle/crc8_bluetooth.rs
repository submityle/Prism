//! `CRC`-8/BLUETOOTH checksum implementation.
//!
//! Parameters: width = 8, poly = `0xA7`, init = `0x00`, refin = true,
//! refout = true, xorout = `0x00`.
//!
//! Because refin and refout are both true, this uses a reflected,
//! `LSB`-first algorithm with the bit-reversed polynomial
//! `bitreverse(0xA7, 8) == 0xE5`. The reflected form folds refout into the
//! running state, and xorout is zero, so no final transform is required.
//!
//! This module targets a `no_std` + `alloc` crate, uses only integer
//! arithmetic, and avoids `Vec`/`String`/`format!` in both the
//! implementation and tests.

/// Computes the `CRC`-8/BLUETOOTH checksum over `data`.
///
/// The algorithm processes each byte `LSB`-first using the reflected
/// polynomial `0xE5` (the bit-reverse of `0xA7`). Both reflection of the
/// input and reflection of the output are handled implicitly by the
/// reflected form, and the final xor value is zero.
pub fn crc8_bluetooth(data: &[u8]) -> u8 {
    const REFPOLY: u8 = 0xE5; // bit-reverse of 0xA7
    let mut crc: u8 = 0x00; // init
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
    crc // refout already folded in, xorout = 0
}

#[cfg(test)]
mod tests {
    use super::crc8_bluetooth;

    // ---- Anchor vectors (ground truth) ----

    #[test]
    fn anchor_empty() {
        assert!(crc8_bluetooth(b"") == 0x00);
    }

    #[test]
    fn anchor_a() {
        assert!(crc8_bluetooth(b"a") == 0x52);
    }

    #[test]
    fn anchor_zero_byte() {
        assert!(crc8_bluetooth(&[0x00]) == 0x00);
    }

    #[test]
    fn anchor_ff_byte() {
        assert!(crc8_bluetooth(&[0xff]) == 0x9f);
    }

    #[test]
    fn anchor_check_value() {
        assert!(crc8_bluetooth(b"123456789") == 0x26);
    }

    // ---- Single-byte vectors ----

    #[test]
    fn single_byte_01() {
        assert!(crc8_bluetooth(&[0x01]) == 0x6b);
    }

    #[test]
    fn single_byte_02() {
        assert!(crc8_bluetooth(&[0x02]) == 0xd6);
    }

    #[test]
    fn single_byte_03() {
        assert!(crc8_bluetooth(&[0x03]) == 0xbd);
    }

    #[test]
    fn single_byte_04() {
        assert!(crc8_bluetooth(&[0x04]) == 0x67);
    }

    #[test]
    fn single_byte_05() {
        assert!(crc8_bluetooth(&[0x05]) == 0x0c);
    }

    #[test]
    fn single_byte_06() {
        assert!(crc8_bluetooth(&[0x06]) == 0xb1);
    }

    #[test]
    fn single_byte_07() {
        assert!(crc8_bluetooth(&[0x07]) == 0xda);
    }

    #[test]
    fn single_byte_08() {
        assert!(crc8_bluetooth(&[0x08]) == 0xce);
    }

    #[test]
    fn single_byte_09() {
        assert!(crc8_bluetooth(&[0x09]) == 0xa5);
    }

    #[test]
    fn single_byte_55() {
        assert!(crc8_bluetooth(&[0x55]) == 0xcc);
    }

    #[test]
    fn single_byte_aa() {
        assert!(crc8_bluetooth(&[0xaa]) == 0x53);
    }

    #[test]
    fn single_byte_80() {
        assert!(crc8_bluetooth(&[0x80]) == 0xe5);
    }

    #[test]
    fn single_byte_7f() {
        assert!(crc8_bluetooth(&[0x7f]) == 0x7a);
    }

    #[test]
    fn single_byte_fe() {
        assert!(crc8_bluetooth(&[0xfe]) == 0xf4);
    }

    // ---- Multi-byte vectors ----

    #[test]
    fn multi_abc() {
        assert!(crc8_bluetooth(b"abc") == 0xaa);
    }

    #[test]
    fn multi_ab() {
        assert!(crc8_bluetooth(b"ab") == 0xf9);
    }

    #[test]
    fn multi_01_02() {
        assert!(crc8_bluetooth(&[0x01, 0x02]) == 0x9c);
    }

    #[test]
    fn multi_01_02_03() {
        assert!(crc8_bluetooth(&[0x01, 0x02, 0x03]) == 0xa6);
    }

    #[test]
    fn multi_deadbeef() {
        assert!(crc8_bluetooth(&[0xde, 0xad, 0xbe, 0xef]) == 0x74);
    }

    #[test]
    fn multi_hello() {
        assert!(crc8_bluetooth(b"hello") == 0x5d);
    }

    #[test]
    fn multi_prism() {
        assert!(crc8_bluetooth(b"prism") == 0x32);
    }

    #[test]
    fn multi_upper_a() {
        assert!(crc8_bluetooth(b"A") == 0xfc);
    }

    #[test]
    fn multi_upper_z() {
        assert!(crc8_bluetooth(b"Z") == 0xd8);
    }

    #[test]
    fn multi_space() {
        assert!(crc8_bluetooth(b" ") == 0xae);
    }

    #[test]
    fn multi_two_a() {
        assert!(crc8_bluetooth(b"aa") == 0x44);
    }

    #[test]
    fn multi_three_a() {
        assert!(crc8_bluetooth(b"aaa") == 0xa2);
    }

    #[test]
    fn multi_seq_0_15() {
        let input: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        assert!(crc8_bluetooth(&input) == 0xdb);
    }

    // ---- Boundary / bulk vectors ----

    #[test]
    fn bulk_four_zeros() {
        assert!(crc8_bluetooth(&[0x00, 0x00, 0x00, 0x00]) == 0x00);
    }

    #[test]
    fn bulk_four_ff() {
        assert!(crc8_bluetooth(&[0xff, 0xff, 0xff, 0xff]) == 0x50);
    }

    #[test]
    fn bulk_thirty_two_zeros() {
        let input: [u8; 32] = [0x00; 32];
        assert!(crc8_bluetooth(&input) == 0x00);
    }

    #[test]
    fn bulk_thirty_two_ff() {
        let input: [u8; 32] = [0xff; 32];
        assert!(crc8_bluetooth(&input) == 0x72);
    }

    #[test]
    fn bulk_seq_0_255() {
        let mut input: [u8; 256] = [0u8; 256];
        let mut i: usize = 0;
        while i < 256 {
            input[i] = i as u8;
            i += 1;
        }
        assert!(crc8_bluetooth(&input) == 0xc9);
    }

    // ---- Property / determinism / incremental ----

    #[test]
    fn determinism_repeated_calls() {
        let a = crc8_bluetooth(b"123456789");
        let b = crc8_bluetooth(b"123456789");
        assert!(a == b);
    }

    #[test]
    fn determinism_slice_vs_literal() {
        let data: [u8; 3] = [b'a', b'b', b'c'];
        assert!(crc8_bluetooth(&data) == crc8_bluetooth(b"abc"));
    }

    #[test]
    fn incremental_prefix_differs_from_full() {
        // Appending a byte changes the state; the two results differ here.
        let short = crc8_bluetooth(b"ab");
        let long = crc8_bluetooth(b"abc");
        assert!(short != long);
    }

    #[test]
    fn empty_prefix_identity() {
        // Prepending nothing leaves the result unchanged.
        let empty: [u8; 0] = [];
        assert!(crc8_bluetooth(&empty) == 0x00);
        assert!(crc8_bluetooth(b"abc") == crc8_bluetooth(b"abc"));
    }

    #[test]
    fn single_vs_double_a_differ() {
        assert!(crc8_bluetooth(b"a") != crc8_bluetooth(b"aa"));
    }

    #[test]
    fn order_sensitivity() {
        // Byte order affects the checksum, so swapped inputs differ here.
        let forward = crc8_bluetooth(&[0x01, 0x02]);
        let reversed = crc8_bluetooth(&[0x02, 0x01]);
        assert!(forward != reversed);
    }
}
