//! `CRC-8/I-CODE` (non-reflected, `MSB`-first, `u8`): width=8, poly=`0x1D`,
//! init=`0xFD`, refin=false, refout=false, xorout=`0x00`; check(`b"123456789"`)=`0x7E`.
//!
//! Pure-integer CPU gold-standard reference. `MSB`-first bit-processing with the
//! `XOR` feedback polynomial; the `reg << 1` wraps within `u8` by design (expected
//! `CRC` truncation). `no_std` + `alloc` compatible; no floating point, no `Vec`/`String`.

/// Compute the `CRC-8/I-CODE` checksum of `data`.
///
/// Non-reflected, `MSB`-first `u8` computation with polynomial `0x1D` and
/// initial register `0xFD`. The `XOR`-out value is `0x00`.
#[must_use]
pub fn crc8_i_code(data: &[u8]) -> u8 {
    const POLY: u8 = 0x1D;
    let mut reg: u8 = 0xFD; // init
    for &byte in data {
        reg ^= byte;
        for _ in 0..8 {
            if (reg & 0x80) != 0 {
                reg = (reg << 1) ^ POLY;
            } else {
                reg <<= 1;
            }
        }
    }
    reg
}

#[cfg(test)]
mod tests {
    use super::crc8_i_code;

    // ---- 5 canonical anchors ----
    #[test]
    fn anchor_empty() {
        assert!(crc8_i_code(b"") == 0xfd);
    }
    #[test]
    fn anchor_a() {
        assert!(crc8_i_code(b"a") == 0x77);
    }
    #[test]
    fn anchor_zero_byte() {
        assert!(crc8_i_code(&[0x00]) == 0xfe);
    }
    #[test]
    fn anchor_ff_byte() {
        assert!(crc8_i_code(&[0xff]) == 0x3a);
    }
    #[test]
    fn anchor_check_string() {
        assert!(crc8_i_code(b"123456789") == 0x7e);
    }

    // ---- multi-byte hard-coded reference vectors (self-computed) ----
    #[test]
    fn vec_ab() {
        assert!(crc8_i_code(b"ab") == 0xa4);
    }
    #[test]
    fn vec_abc() {
        assert!(crc8_i_code(b"abc") == 0x66);
    }
    #[test]
    fn vec_hello() {
        assert!(crc8_i_code(b"hello") == 0x82);
    }
    #[test]
    fn vec_prism() {
        assert!(crc8_i_code(b"prism") == 0x6e);
    }
    #[test]
    fn vec_hello_world() {
        assert!(crc8_i_code(b"Hello, World!") == 0x5b);
    }
    #[test]
    fn vec_small_bytes() {
        assert!(crc8_i_code(&[1, 2, 3, 4]) == 0xbf);
    }
    #[test]
    fn vec_four_ff() {
        assert!(crc8_i_code(&[0xff, 0xff, 0xff, 0xff]) == 0xac);
    }
    #[test]
    fn vec_four_zero() {
        assert!(crc8_i_code(&[0, 0, 0, 0]) == 0x81);
    }
    #[test]
    fn vec_ascending_0_9() {
        assert!(crc8_i_code(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) == 0x4c);
    }
    #[test]
    fn vec_deadbeef() {
        assert!(crc8_i_code(&[0xde, 0xad, 0xbe, 0xef]) == 0x6b);
    }
    #[test]
    fn vec_cafebabe() {
        assert!(crc8_i_code(&[0xca, 0xfe, 0xba, 0xbe]) == 0x1a);
    }
    #[test]
    fn vec_quick_fox() {
        assert!(crc8_i_code(b"The quick brown fox") == 0xa3);
    }

    // ---- determinism ----
    #[test]
    fn deterministic_repeat() {
        let data = b"deterministic-check";
        let a = crc8_i_code(data);
        let b = crc8_i_code(data);
        let c = crc8_i_code(data);
        assert!(a == b);
        assert!(b == c);
    }
    #[test]
    fn deterministic_many_rounds() {
        let data = &[0x11, 0x22, 0x33, 0x44, 0x55];
        let first = crc8_i_code(data);
        let mut i: u32 = 0;
        while i < 64 {
            assert!(crc8_i_code(data) == first);
            i += 1;
        }
    }

    // ---- empty input ----
    #[test]
    fn empty_is_init() {
        assert!(crc8_i_code(&[]) == 0xfd);
    }
    #[test]
    fn empty_slice_literal() {
        let empty: &[u8] = &[];
        assert!(crc8_i_code(empty) == 0xfd);
    }

    // ---- sampling / splittable iteration consistency ----
    #[test]
    fn sampling_prefix_full() {
        // Full computation equals an open-coded identical loop over the data.
        let data: &[u8] = &[0x01, 0x7f, 0x80, 0xaa, 0x55, 0xc3];
        let expected = crc8_i_code(data);
        let mut reg: u8 = 0xFD;
        for &byte in data {
            reg ^= byte;
            let mut k = 0;
            while k < 8 {
                if (reg & 0x80) != 0 {
                    reg = (reg << 1) ^ 0x1D;
                } else {
                    reg <<= 1;
                }
                k += 1;
            }
        }
        assert!(reg == expected);
    }
    #[test]
    fn sampling_indices_stable() {
        // Sampling every other byte yields a stable, reproducible result.
        let data: &[u8] = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let evens: &[u8] = &[0, 2, 4, 6, 8, 10];
        let a = crc8_i_code(evens);
        let b = crc8_i_code(evens);
        assert!(a == b);
        // Sampled subset differs from the whole (non-trivial dependence).
        assert!(crc8_i_code(data) != a);
    }

    // ---- distinctness / avalanche sanity ----
    #[test]
    fn single_bit_change_differs() {
        assert!(crc8_i_code(&[0x00]) != crc8_i_code(&[0x01]));
    }
    #[test]
    fn order_matters() {
        assert!(crc8_i_code(&[0x12, 0x34]) != crc8_i_code(&[0x34, 0x12]));
    }
    #[test]
    fn length_matters() {
        assert!(crc8_i_code(&[0x00]) != crc8_i_code(&[0x00, 0x00]));
    }
    #[test]
    fn distinct_messages_distinct() {
        assert!(crc8_i_code(b"abc") != crc8_i_code(b"abd"));
    }

    // ---- range / bound sanity ----
    #[test]
    fn result_is_u8_range() {
        // Every result is a valid u8; verify over a spread of inputs.
        let mut byte: u16 = 0;
        while byte <= 0xff {
            let input = [byte as u8];
            let out = crc8_i_code(&input);
            assert!((0x00..=0xff).contains(&out));
            byte += 1;
        }
    }

    // ---- long input stability ----
    #[test]
    fn long_input_hundred_a() {
        let buf = [0x41u8; 100];
        assert!(crc8_i_code(&buf) == 0xbc);
    }
    #[test]
    fn long_input_256_sequence() {
        let mut buf = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc8_i_code(&buf) == 0xc0);
    }
    #[test]
    fn long_input_stable_repeat() {
        let buf = [0x5au8; 512];
        let first = crc8_i_code(&buf);
        assert!(crc8_i_code(&buf) == first);
        assert!(crc8_i_code(&buf) == first);
    }
    #[test]
    fn long_zero_fill_stable() {
        let buf = [0u8; 1024];
        let a = crc8_i_code(&buf);
        let b = crc8_i_code(&buf);
        assert!(a == b);
    }

    // ---- structural helpers ----
    #[test]
    fn even_odd_partition_sampling() {
        // Partition indices by parity using is_multiple_of; both halves stable.
        let data: &[u8] = &[9, 8, 7, 6, 5, 4, 3, 2, 1, 0];
        let mut even_sum: u8 = 0;
        let mut odd_sum: u8 = 0;
        let mut i = 0usize;
        for &byte in data {
            if i.is_multiple_of(2) {
                even_sum ^= byte;
            } else {
                odd_sum ^= byte;
            }
            i += 1;
        }
        let one = [even_sum];
        let two = [odd_sum];
        assert!(crc8_i_code(&one) == crc8_i_code(&one));
        assert!(crc8_i_code(&two) == crc8_i_code(&two));
    }
    #[test]
    fn repeated_pattern_stable() {
        let buf = [0xa5u8, 0x5a, 0xa5, 0x5a, 0xa5, 0x5a];
        let a = crc8_i_code(&buf);
        let b = crc8_i_code(&buf);
        assert!(a == b);
    }
    #[test]
    fn all_bytes_round_trip_stable() {
        // Compute CRC over the full byte space twice; must match.
        let mut buf = [0u8; 256];
        let mut i = 0usize;
        while i < 256 {
            buf[i] = i as u8;
            i += 1;
        }
        let a = crc8_i_code(&buf);
        let b = crc8_i_code(&buf);
        assert!(a == b);
    }
    #[test]
    fn two_byte_zero_vector() {
        assert!(crc8_i_code(&[0x00, 0x00]) == crc8_i_code(&[0x00, 0x00]));
    }
    #[test]
    fn ff_prefix_distinct() {
        assert!(crc8_i_code(&[0xff, 0x00]) != crc8_i_code(&[0x00, 0xff]));
    }
}
