//! \`CRC-16/DDS-110\`: width=16, poly=\`0x8005\`, init=\`0x800D\`, refin=false, refout=false, xorout=\`0x0000\`, check=\`0x9ecf\`.
//!
//! Non-reflected (\`MSB\`-first) bit-serial \`CRC\` over a byte slice, returning a
//! \`u16\` register value. The feedback uses \`XOR\` with the generator polynomial
//! and performs no final \`XOR\` (\`xorout\` is zero). The module is pure integer
//! arithmetic and `no_std` + `alloc` compatible.

/// Computes the \`CRC-16/DDS-110\` checksum of \`data\`.
///
/// Processes each byte most-significant-bit first through a 16-bit shift
/// register, applying the \`XOR\` feedback of polynomial \`0x8005\` whenever the
/// outgoing high bit differs from the incoming data bit. Returns the final
/// register as a \`u16\`.
pub fn crc16_dds110(data: &[u8]) -> u16 {
    const POLY: u32 = 0x8005;
    const MASK: u32 = 0xFFFF;
    let mut reg: u32 = 0x800D; // init
    for &byte in data {
        for i in 0..8u32 {
            let bit = u32::from((byte >> (7 - i)) & 1);
            let hi = (reg >> 15) & 1;
            let fb = hi ^ bit;
            reg = (reg << 1) & MASK;
            if fb != 0 {
                reg ^= POLY;
            }
        }
    }
    (reg & MASK) as u16
}

#[cfg(test)]
mod tests {
    use super::crc16_dds110;

    // ---- 5 published anchors ----

    #[test]
    fn anchor_empty() {
        assert!(crc16_dds110(b"") == 0x800d);
    }

    #[test]
    fn anchor_a() {
        assert!(crc16_dds110(b"a") == 0x0f46);
    }

    #[test]
    fn anchor_zero_byte() {
        assert!(crc16_dds110(&[0x00]) == 0x8e03);
    }

    #[test]
    fn anchor_ff_byte() {
        assert!(crc16_dds110(&[0xff]) == 0x8c01);
    }

    #[test]
    fn anchor_check() {
        assert!(crc16_dds110(b"123456789") == 0x9ecf);
    }

    // ---- hardcoded multi-byte vectors (self-computed) ----

    #[test]
    fn vec_ab() {
        assert!(crc16_dds110(b"ab") == 0xc76d);
    }

    #[test]
    fn vec_abc() {
        assert!(crc16_dds110(b"abc") == 0xeedb);
    }

    #[test]
    fn vec_hello() {
        assert!(crc16_dds110(b"hello") == 0xe0c5);
    }

    #[test]
    fn vec_prism() {
        assert!(crc16_dds110(b"Prism") == 0x85cb);
    }

    #[test]
    fn vec_two_zero() {
        assert!(crc16_dds110(&[0x00, 0x00]) == 0x0024);
    }

    #[test]
    fn vec_two_ff() {
        assert!(crc16_dds110(&[0xff, 0xff]) == 0x8029);
    }

    #[test]
    fn vec_one_two_three() {
        assert!(crc16_dds110(&[0x01, 0x02, 0x03]) == 0x281e);
    }

    #[test]
    fn vec_deadbeef() {
        assert!(crc16_dds110(&[0xde, 0xad, 0xbe, 0xef]) == 0x96f3);
    }

    #[test]
    fn vec_alternating() {
        assert!(crc16_dds110(&[0x00, 0xff, 0x00, 0xff]) == 0x0ed6);
    }

    #[test]
    fn vec_init_pair_to_zero() {
        assert!(crc16_dds110(&[0x80, 0x0d]) == 0x0000);
    }

    #[test]
    fn vec_zero_to_nine() {
        assert!(crc16_dds110(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) == 0x2adb);
    }

    #[test]
    fn vec_aa_quad() {
        assert!(crc16_dds110(&[0xaa, 0xaa, 0xaa, 0xaa]) == 0x7f15);
    }

    #[test]
    fn vec_mixed_five() {
        assert!(crc16_dds110(&[0x12, 0x34, 0x56, 0x78, 0x9a]) == 0x5818);
    }

    #[test]
    fn vec_one_two() {
        assert!(crc16_dds110(&[0x01, 0x02]) == 0x0628);
    }

    #[test]
    fn vec_two_one() {
        assert!(crc16_dds110(&[0x02, 0x01]) == 0x0c22);
    }

    #[test]
    fn vec_single_7f() {
        assert!(crc16_dds110(&[0x7f]) == 0x0f02);
    }

    #[test]
    fn vec_single_01() {
        assert!(crc16_dds110(&[0x01]) == 0x0e06);
    }

    #[test]
    fn vec_single_55() {
        assert!(crc16_dds110(&[0x55]) == 0x8ffd);
    }

    #[test]
    fn vec_55_aa() {
        assert!(crc16_dds110(&[0x55, 0xaa]) == 0x7ddd);
    }

    #[test]
    fn vec_cafe() {
        assert!(crc16_dds110(&[0xca, 0xfe]) == 0xbe2c);
    }

    #[test]
    fn vec_babe() {
        assert!(crc16_dds110(&[0xba, 0xbe]) == 0x1fa9);
    }

    // ---- structural / behavioral properties ----

    #[test]
    fn empty_equals_init() {
        // Empty input leaves the register at its init value 0x800D.
        assert!(crc16_dds110(&[]) == 0x800d);
    }

    #[test]
    fn determinism_repeated() {
        let data = [0x12u8, 0x34, 0x56, 0x78, 0x9a];
        let first = crc16_dds110(&data);
        let second = crc16_dds110(&data);
        assert!(first == second);
    }

    #[test]
    fn byte_literal_matches_text() {
        // "ab" and the raw bytes 0x61,0x62 must agree.
        assert!(crc16_dds110(b"ab") == crc16_dds110(&[0x61, 0x62]));
    }

    #[test]
    fn order_matters() {
        assert!(crc16_dds110(&[0x01, 0x02]) != crc16_dds110(&[0x02, 0x01]));
    }

    #[test]
    fn prefix_extension_changes() {
        assert!(crc16_dds110(&[0x01, 0x02]) != crc16_dds110(&[0x01, 0x02, 0x03]));
    }

    #[test]
    fn result_within_expected_band() {
        // 0x800D for the empty message falls in the high band 0x8000..=0x8FFF.
        assert!((0x8000u16..=0x8fffu16).contains(&crc16_dds110(b"")));
    }

    // ---- divisibility: appending the CRC (MSB-first) yields residue 0 ----

    #[test]
    fn divisible_one_two_three() {
        assert!(crc16_dds110(&[0x01, 0x02, 0x03, 0x28, 0x1e]) == 0x0000);
    }

    #[test]
    fn divisible_deadbeef() {
        assert!(crc16_dds110(&[0xde, 0xad, 0xbe, 0xef, 0x96, 0xf3]) == 0x0000);
    }

    #[test]
    fn divisible_check_message() {
        assert!(
            crc16_dds110(&[0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x9e, 0xcf,])
                == 0x0000
        );
    }

    #[test]
    fn divisible_55_aa() {
        assert!(crc16_dds110(&[0x55, 0xaa, 0x7d, 0xdd]) == 0x0000);
    }

    #[test]
    fn divisible_sample_set() {
        // A sample of CRC-appended messages must all reduce to the zero residue.
        let samples: [&[u8]; 4] = [
            &[0x01, 0x02, 0x03, 0x28, 0x1e],
            &[0xde, 0xad, 0xbe, 0xef, 0x96, 0xf3],
            &[0x55, 0xaa, 0x7d, 0xdd],
            &[0x80, 0x0d],
        ];
        for s in samples {
            assert!(crc16_dds110(s) == 0x0000);
        }
    }

    #[test]
    fn sampling_parity_split() {
        // Partition indices by parity using is_multiple_of; both halves stay
        // deterministic against their recomputation.
        let data: [u8; 16] = [
            0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd,
            0xee, 0xff,
        ];
        let mut even = [0u8; 8];
        let mut odd = [0u8; 8];
        let mut ei = 0usize;
        let mut oi = 0usize;
        for (i, &b) in data.iter().enumerate() {
            if i.is_multiple_of(2) {
                even[ei] = b;
                ei += 1;
            } else {
                odd[oi] = b;
                oi += 1;
            }
        }
        assert!(crc16_dds110(&even) == crc16_dds110(&even));
        assert!(crc16_dds110(&odd) == crc16_dds110(&odd));
        assert!(crc16_dds110(&data) == crc16_dds110(&data));
    }

    // ---- long input stability ----

    #[test]
    fn long_counter_block_stable() {
        let mut buf = [0u8; 1000];
        for (i, b) in buf.iter_mut().enumerate() {
            *b = (i & 0xff) as u8;
        }
        assert!(crc16_dds110(&buf) == 0xd88d);
    }

    #[test]
    fn long_counter_block_deterministic() {
        let mut buf = [0u8; 1000];
        for (i, b) in buf.iter_mut().enumerate() {
            *b = (i & 0xff) as u8;
        }
        assert!(crc16_dds110(&buf) == crc16_dds110(&buf));
    }

    #[test]
    fn long_uniform_block_stable() {
        let buf = [0x41u8; 256];
        assert!(crc16_dds110(&buf) == 0xc38e);
    }
}
