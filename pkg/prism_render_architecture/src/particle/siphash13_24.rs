//! `SipHash`: Jean-Philippe Aumasson and Daniel J. Bernstein's fast, keyed
//! 64-bit pseudo-random function (`PRF`) built from an add-rotate-`xor` (`ARX`)
//! core. This module ships the two canonical variants, `SipHash-2-4` (the
//! widely deployed default, 2 compression rounds and 4 finalization rounds) and
//! the faster `SipHash-1-3` (1 compression round, 3 finalization rounds).
//!
//! Within Prism these helpers are a cheap, well-distributed keyed fingerprint
//! for hash-table bucketing, cache keys, and streamed particle payloads where a
//! per-process key is desirable to make collisions hard to force. Because the
//! core is pure integer `ARX`, both `CPU` and `GPU` paths can agree bit-for-bit.
//!
//! # Algorithm
//! The 16-byte key is split into two little-endian 64-bit halves `k0 = LE(key
//! [0..8])` and `k1 = LE(key[8..16])`. Four internal `u64` words are seeded by
//! `xor`-ing the key halves into the fixed `ASCII` constants
//! `"somepseu"`/`"dorandom"`/`"lygenera"`/`"tedbytes"`:
//!
//! ```text
//! v0 = 0x736f6d6570736575 ^ k0;
//! v1 = 0x646f72616e646f6d ^ k1;
//! v2 = 0x6c7967656e657261 ^ k0;
//! v3 = 0x7465646279746573 ^ k1;
//! ```
//!
//! The message is consumed in 8-byte little-endian blocks. Each block `m` is
//! absorbed with `v3 ^= m`, then `C` `SipRound`s, then `v0 ^= m`. The final
//! (partial) block packs the low byte of the length into the top byte,
//! `b = (len & 0xff) << 56`, with the trailing `len % 8` bytes placed in the
//! low positions, and is absorbed the same way. Finalization sets `v2 ^= 0xff`,
//! applies `D` `SipRound`s, and returns `v0 ^ v1 ^ v2 ^ v3`.
//!
//! Every operation is `u64` `wrapping_add`, `rotate_left`, and `xor`; there is
//! no floating point and no table lookup. All reads are little-endian and are
//! assembled by hand from bytes so the module is `no_std` self-contained.
//!
//! # Security boundary
//! `SipHash` is a keyed `PRF` designed to resist hash-flooding, not a general
//! cryptographic hash. With a secret key it provides good resistance to
//! adversarial collision generation in hash tables, but it offers no second
//! pre-image or collision guarantees once the key is known, and the 64-bit
//! output is far too short for signatures or message authentication at scale.
//! Do not use it for security tokens, content signing, or anti-tamper.

/// Initialization constant `xor`-ed into `v0` (`ASCII` `"somepseu"`).
const INIT_V0: u64 = 0x736f_6d65_7073_6575;
/// Initialization constant `xor`-ed into `v1` (`ASCII` `"dorandom"`).
const INIT_V1: u64 = 0x646f_7261_6e64_6f6d;
/// Initialization constant `xor`-ed into `v2` (`ASCII` `"lygenera"`).
const INIT_V2: u64 = 0x6c79_6765_6e65_7261;
/// Initialization constant `xor`-ed into `v3` (`ASCII` `"tedbytes"`).
const INIT_V3: u64 = 0x7465_6462_7974_6573;

/// Finalization constant `xor`-ed into `v2` before the finishing rounds.
const FINAL_XOR: u64 = 0xff;

/// Compression rounds for `SipHash-2-4`.
const C_SIPHASH24: usize = 2;
/// Finalization rounds for `SipHash-2-4`.
const D_SIPHASH24: usize = 4;
/// Compression rounds for `SipHash-1-3`.
const C_SIPHASH13: usize = 1;
/// Finalization rounds for `SipHash-1-3`.
const D_SIPHASH13: usize = 3;

/// The four-word `ARX` state used during compression and finalization.
struct SipState {
    v0: u64,
    v1: u64,
    v2: u64,
    v3: u64,
}

impl SipState {
    /// Seeds the state from the 16-byte key split into two little-endian halves.
    fn new(k0: u64, k1: u64) -> Self {
        Self {
            v0: INIT_V0 ^ k0,
            v1: INIT_V1 ^ k1,
            v2: INIT_V2 ^ k0,
            v3: INIT_V3 ^ k1,
        }
    }

    /// One `SipRound`: the `ARX` mixing permutation on the four state words.
    ///
    /// Every addition is `wrapping` and every rotation is the `u64::rotate_left`
    /// intrinsic; there is no floating point and no table lookup.
    fn round(&mut self) {
        self.v0 = self.v0.wrapping_add(self.v1);
        self.v1 = self.v1.rotate_left(13);
        self.v1 ^= self.v0;
        self.v0 = self.v0.rotate_left(32);

        self.v2 = self.v2.wrapping_add(self.v3);
        self.v3 = self.v3.rotate_left(16);
        self.v3 ^= self.v2;

        self.v0 = self.v0.wrapping_add(self.v3);
        self.v3 = self.v3.rotate_left(21);
        self.v3 ^= self.v0;

        self.v2 = self.v2.wrapping_add(self.v1);
        self.v1 = self.v1.rotate_left(17);
        self.v1 ^= self.v2;
        self.v2 = self.v2.rotate_left(32);
    }

    /// Applies `n` consecutive `SipRound`s.
    fn rounds(&mut self, n: usize) {
        for _ in 0..n {
            self.round();
        }
    }

    /// Absorbs one 64-bit message word: `v3 ^= m`, `c` rounds, then `v0 ^= m`.
    fn absorb(&mut self, m: u64, c: usize) {
        self.v3 ^= m;
        self.rounds(c);
        self.v0 ^= m;
    }
}

/// Reads a little-endian `u64` from the key by hand.
///
/// The slice must have at least `offset + 8` valid bytes; the only callers pass
/// a 16-byte key with `offset` of `0` or `8`.
fn read_key_u64_le(key: &[u8; 16], offset: usize) -> u64 {
    let mut acc = 0u64;
    let mut i = 0usize;
    while i < 8 {
        acc |= (key[offset + i] as u64) << (8 * i);
        i += 1;
    }
    acc
}

/// Core `SipHash` over `data` with `c` compression rounds and `d`
/// finalization rounds.
///
/// All arithmetic is `u64` `wrapping_add`, `rotate_left`, and `xor`; message
/// bytes are read little-endian and assembled by hand so the routine stays
/// `no_std` self-contained.
fn siphash(key: &[u8; 16], data: &[u8], c: usize, d: usize) -> u64 {
    let k0 = read_key_u64_le(key, 0);
    let k1 = read_key_u64_le(key, 8);
    let mut state = SipState::new(k0, k1);

    // Full 8-byte little-endian blocks.
    let mut chunks = data.chunks_exact(8);
    for block in chunks.by_ref() {
        let m = block
            .iter()
            .enumerate()
            .fold(0u64, |acc, (i, &b)| acc | ((b as u64) << (8 * i)));
        state.absorb(m, c);
    }

    // Trailing block: the length's low byte sits in the top byte, with the
    // remaining `len % 8` bytes packed into the low positions.
    let remainder = chunks.remainder();
    let mut b = (data.len() as u64 & 0xff) << 56;
    b |= remainder
        .iter()
        .enumerate()
        .fold(0u64, |acc, (i, &byte)| acc | ((byte as u64) << (8 * i)));
    state.absorb(b, c);

    // Finalization.
    state.v2 ^= FINAL_XOR;
    state.rounds(d);
    state.v0 ^ state.v1 ^ state.v2 ^ state.v3
}

/// Computes `SipHash-2-4` of `data` under the 16-byte `key`.
///
/// This is the canonical, widely deployed variant (2 compression rounds, 4
/// finalization rounds). It is a keyed `PRF`, not a cryptographic hash; see the
/// module header for the security boundary.
pub fn siphash24(key: &[u8; 16], data: &[u8]) -> u64 {
    siphash(key, data, C_SIPHASH24, D_SIPHASH24)
}

/// Computes `SipHash-1-3` of `data` under the 16-byte `key`.
///
/// This is the faster variant (1 compression round, 3 finalization rounds)
/// used where throughput matters more than the extra mixing margin. It is a
/// keyed `PRF`, not a cryptographic hash; see the module header for the
/// security boundary.
pub fn siphash13(key: &[u8; 16], data: &[u8]) -> u64 {
    siphash(key, data, C_SIPHASH13, D_SIPHASH13)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The canonical reference key: `key[i] = i` for `i` in `0..16`.
    fn ref_key() -> [u8; 16] {
        let mut key = [0u8; 16];
        let mut i = 0usize;
        while i < 16 {
            key[i] = i as u8;
            i += 1;
        }
        key
    }

    /// An incrementing 64-byte buffer with `buf[i] = i`; take a prefix slice to
    /// obtain a message of any length in `0..=64`.
    fn seq64() -> [u8; 64] {
        let mut buf = [0u8; 64];
        let mut i = 0usize;
        while i < 64 {
            buf[i] = i as u8;
            i += 1;
        }
        buf
    }

    /// Official `SipHash-2-4` reference digests for input length `i` with the
    /// reference key and `data[j] = j`. Index `i` corresponds to a message of
    /// `i` bytes, so index `0` is the empty message.
    const SIPHASH24_VECTORS: [u64; 64] = [
        0x726f_db47_dd0e_0e31,
        0x74f8_39c5_93dc_67fd,
        0x0d6c_8009_d9a9_4f5a,
        0x8567_6696_d7fb_7e2d,
        0xcf27_94e0_2771_87b7,
        0x1876_5564_cd99_a68d,
        0xcbc9_466e_58fe_e3ce,
        0xab02_00f5_8b01_d137,
        0x93f5_f579_9a93_2462,
        0x9e00_82df_0ba9_e4b0,
        0x7a5d_bbc5_94dd_b9f3,
        0xf4b3_2f46_226b_ada7,
        0x751e_8fbc_860e_e5fb,
        0x14ea_5627_c084_3d90,
        0xf723_ca90_8e7a_f2ee,
        0xa129_ca61_49be_45e5,
        0x3f2a_cc7f_57c2_9bdb,
        0x699a_e9f5_2cbe_4794,
        0x4bc1_b3f0_968d_d39c,
        0xbb6d_c91d_a779_61bd,
        0xbed6_5cf2_1aa2_ee98,
        0xd0f2_cbb0_2e3b_67c7,
        0x9353_6795_e3a3_3e88,
        0xa80c_038c_cd5c_cec8,
        0xb8ad_50c6_f649_af94,
        0xbce1_92de_8a85_b8ea,
        0x17d8_35b8_5bbb_15f3,
        0x2f2e_6163_076b_cfad,
        0xde4d_aaac_a71d_c9a5,
        0xa6a2_5066_8795_6571,
        0xad87_a353_5c49_ef28,
        0x32d8_92fa_d841_c342,
        0x7127_512f_72f2_7cce,
        0xa7f3_2346_f959_78e3,
        0x12e0_b01a_bb05_1238,
        0x15e0_34d4_0fa1_97ae,
        0x314d_ffbe_0815_a3b4,
        0x0279_90f0_2962_3981,
        0xcadc_d4e5_9ef4_0c4d,
        0x9abf_d876_6a33_735c,
        0x0e3e_a96b_5304_a7d0,
        0xad0c_42d6_fc58_5992,
        0x1873_06c8_9bc2_15a9,
        0xd4a6_0abc_f379_2b95,
        0xf935_451d_e4f2_1df2,
        0xa953_8f04_1975_5787,
        0xdb9a_cddf_f56c_a510,
        0xd06c_98cd_5c09_75eb,
        0xe612_a3cb_9ecb_a951,
        0xc766_e62c_fcad_af96,
        0xee64_435a_9752_fe72,
        0xa192_d576_b245_165a,
        0x0a87_87bf_8ecb_74b2,
        0x81b3_e73d_20b4_9b6f,
        0x7fa8_220b_a3b2_ecea,
        0x2457_31c1_3ca4_2499,
        0xb78d_bfaf_3a8d_83bd,
        0xea1a_d565_322a_1a0b,
        0x60e6_1c23_a379_5013,
        0x6606_d7e4_4628_2b93,
        0x6ca4_ecb1_5c5f_91e1,
        0x9f62_6da1_5c96_25f3,
        0xe51b_3860_8ef2_5f57,
        0x958a_324c_eb06_4572,
    ];

    // --- Hard reference vectors (siphash24) -------------------------------

    #[test]
    fn siphash24_empty_reference_vector() {
        assert_eq!(siphash24(&ref_key(), &[]), 0x726f_db47_dd0e_0e31);
    }

    #[test]
    fn siphash24_fifteen_byte_reference_vector() {
        let msg: [u8; 15] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];
        assert_eq!(siphash24(&ref_key(), &msg), 0xa129_ca61_49be_45e5);
    }

    // --- Hard reference vectors (siphash13) -------------------------------

    #[test]
    fn siphash13_empty_reference_vector() {
        assert_eq!(siphash13(&ref_key(), &[]), 0xabac_0158_050f_c4dc);
    }

    #[test]
    fn siphash13_fifteen_byte_reference_vector() {
        let msg: [u8; 15] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];
        assert_eq!(siphash13(&ref_key(), &msg), 0xd320_d86d_2a51_9956);
    }

    // --- Official siphash24 0..N vectors ----------------------------------

    #[test]
    fn siphash24_all_official_vectors() {
        let key = ref_key();
        let buf = seq64();
        for (len, &expected) in SIPHASH24_VECTORS.iter().enumerate() {
            assert_eq!(siphash24(&key, &buf[..len]), expected, "len {len}");
        }
    }

    #[test]
    fn siphash24_vector_len_0() {
        assert_eq!(siphash24(&ref_key(), &seq64()[..0]), SIPHASH24_VECTORS[0]);
    }

    #[test]
    fn siphash24_vector_len_1() {
        assert_eq!(siphash24(&ref_key(), &seq64()[..1]), SIPHASH24_VECTORS[1]);
    }

    #[test]
    fn siphash24_vector_len_2() {
        assert_eq!(siphash24(&ref_key(), &seq64()[..2]), SIPHASH24_VECTORS[2]);
    }

    #[test]
    fn siphash24_vector_len_3() {
        assert_eq!(siphash24(&ref_key(), &seq64()[..3]), SIPHASH24_VECTORS[3]);
    }

    #[test]
    fn siphash24_vector_len_4() {
        assert_eq!(siphash24(&ref_key(), &seq64()[..4]), SIPHASH24_VECTORS[4]);
    }

    #[test]
    fn siphash24_vector_len_5() {
        assert_eq!(siphash24(&ref_key(), &seq64()[..5]), SIPHASH24_VECTORS[5]);
    }

    #[test]
    fn siphash24_vector_len_6() {
        assert_eq!(siphash24(&ref_key(), &seq64()[..6]), SIPHASH24_VECTORS[6]);
    }

    #[test]
    fn siphash24_vector_len_7() {
        assert_eq!(siphash24(&ref_key(), &seq64()[..7]), SIPHASH24_VECTORS[7]);
    }

    #[test]
    fn siphash24_vector_len_8() {
        assert_eq!(siphash24(&ref_key(), &seq64()[..8]), SIPHASH24_VECTORS[8]);
    }

    #[test]
    fn siphash24_vector_len_9() {
        assert_eq!(siphash24(&ref_key(), &seq64()[..9]), SIPHASH24_VECTORS[9]);
    }

    #[test]
    fn siphash24_vector_len_16() {
        assert_eq!(siphash24(&ref_key(), &seq64()[..16]), SIPHASH24_VECTORS[16]);
    }

    #[test]
    fn siphash24_vector_len_32() {
        assert_eq!(siphash24(&ref_key(), &seq64()[..32]), SIPHASH24_VECTORS[32]);
    }

    #[test]
    fn siphash24_vector_len_63() {
        assert_eq!(siphash24(&ref_key(), &seq64()[..63]), SIPHASH24_VECTORS[63]);
    }

    // --- 8-byte boundary coverage -----------------------------------------

    #[test]
    fn siphash24_boundary_len_7() {
        // One short of a full block exercises the partial-tail path.
        let out = siphash24(&ref_key(), &seq64()[..7]);
        assert_eq!(out, SIPHASH24_VECTORS[7]);
    }

    #[test]
    fn siphash24_boundary_len_8() {
        // Exactly one block plus an empty-but-length-tagged tail block.
        let out = siphash24(&ref_key(), &seq64()[..8]);
        assert_eq!(out, SIPHASH24_VECTORS[8]);
    }

    #[test]
    fn siphash24_boundary_len_9() {
        // One full block followed by a single trailing byte.
        let out = siphash24(&ref_key(), &seq64()[..9]);
        assert_eq!(out, SIPHASH24_VECTORS[9]);
    }

    #[test]
    fn siphash13_boundary_len_7_8_9_distinct() {
        let key = ref_key();
        let buf = seq64();
        let a = siphash13(&key, &buf[..7]);
        let b = siphash13(&key, &buf[..8]);
        let c = siphash13(&key, &buf[..9]);
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_ne!(a, c);
    }

    // --- Determinism ------------------------------------------------------

    #[test]
    fn siphash24_is_deterministic() {
        let key = ref_key();
        let buf = seq64();
        for len in 0..=64 {
            assert_eq!(siphash24(&key, &buf[..len]), siphash24(&key, &buf[..len]));
        }
    }

    #[test]
    fn siphash13_is_deterministic() {
        let key = ref_key();
        let buf = seq64();
        for len in 0..=64 {
            assert_eq!(siphash13(&key, &buf[..len]), siphash13(&key, &buf[..len]));
        }
    }

    #[test]
    fn siphash24_deterministic_repeated_empty() {
        let key = ref_key();
        assert_eq!(siphash24(&key, &[]), siphash24(&key, &[]));
    }

    #[test]
    fn siphash13_deterministic_repeated_empty() {
        let key = ref_key();
        assert_eq!(siphash13(&key, &[]), siphash13(&key, &[]));
    }

    // --- Key avalanche ----------------------------------------------------

    #[test]
    fn siphash24_key_bit_flip_changes_output() {
        let base = ref_key();
        let data = seq64();
        let out_base = siphash24(&base, &data[..32]);
        for byte in 0..16 {
            for bit in 0..8 {
                let mut key = base;
                key[byte] ^= 1 << bit;
                assert_ne!(
                    siphash24(&key, &data[..32]),
                    out_base,
                    "key byte {byte} bit {bit}"
                );
            }
        }
    }

    #[test]
    fn siphash13_key_bit_flip_changes_output() {
        let base = ref_key();
        let data = seq64();
        let out_base = siphash13(&base, &data[..32]);
        for byte in 0..16 {
            for bit in 0..8 {
                let mut key = base;
                key[byte] ^= 1 << bit;
                assert_ne!(
                    siphash13(&key, &data[..32]),
                    out_base,
                    "key byte {byte} bit {bit}"
                );
            }
        }
    }

    #[test]
    fn siphash24_zero_key_differs_from_ref_key() {
        let zero = [0u8; 16];
        assert_ne!(siphash24(&zero, &[]), siphash24(&ref_key(), &[]));
    }

    #[test]
    fn siphash13_zero_key_differs_from_ref_key() {
        let zero = [0u8; 16];
        assert_ne!(siphash13(&zero, &[]), siphash13(&ref_key(), &[]));
    }

    // --- Message avalanche ------------------------------------------------

    #[test]
    fn siphash24_message_bit_flip_changes_output() {
        let key = ref_key();
        let base = seq64();
        let out_base = siphash24(&key, &base[..32]);
        for byte in 0..32 {
            for bit in 0..8 {
                let mut msg = base;
                msg[byte] ^= 1 << bit;
                assert_ne!(
                    siphash24(&key, &msg[..32]),
                    out_base,
                    "msg byte {byte} bit {bit}"
                );
            }
        }
    }

    #[test]
    fn siphash13_message_bit_flip_changes_output() {
        let key = ref_key();
        let base = seq64();
        let out_base = siphash13(&key, &base[..32]);
        for byte in 0..32 {
            for bit in 0..8 {
                let mut msg = base;
                msg[byte] ^= 1 << bit;
                assert_ne!(
                    siphash13(&key, &msg[..32]),
                    out_base,
                    "msg byte {byte} bit {bit}"
                );
            }
        }
    }

    #[test]
    fn siphash24_single_byte_tail_values_differ() {
        let key = ref_key();
        let out_a = siphash24(&key, &[0x00]);
        let out_b = siphash24(&key, &[0x01]);
        assert_ne!(out_a, out_b);
    }

    #[test]
    fn siphash13_single_byte_tail_values_differ() {
        let key = ref_key();
        let out_a = siphash13(&key, &[0x00]);
        let out_b = siphash13(&key, &[0x01]);
        assert_ne!(out_a, out_b);
    }

    // --- Variant separation & length tagging ------------------------------

    #[test]
    fn siphash13_differs_from_siphash24_empty() {
        let key = ref_key();
        assert_ne!(siphash13(&key, &[]), siphash24(&key, &[]));
    }

    #[test]
    fn siphash13_differs_from_siphash24_fifteen() {
        let key = ref_key();
        let msg: [u8; 15] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];
        assert_ne!(siphash13(&key, &msg), siphash24(&key, &msg));
    }

    #[test]
    fn siphash24_length_tag_distinguishes_zero_padding() {
        // A length-tagged tail means more trailing zero bytes still differ.
        let key = ref_key();
        let short = [0u8; 1];
        let long = [0u8; 2];
        assert_ne!(siphash24(&key, &short), siphash24(&key, &long));
    }

    #[test]
    fn siphash13_length_tag_distinguishes_zero_padding() {
        let key = ref_key();
        let short = [0u8; 1];
        let long = [0u8; 2];
        assert_ne!(siphash13(&key, &short), siphash13(&key, &long));
    }

    #[test]
    fn siphash24_eight_zero_bytes_differs_from_empty() {
        // len 0 and len 8 share no message bytes but differ via the length tag.
        let key = ref_key();
        let eight = [0u8; 8];
        assert_ne!(siphash24(&key, &eight), siphash24(&key, &[]));
    }

    #[test]
    fn key_halves_are_little_endian() {
        // k0 = LE(key[0..8]); build a key whose first half is 1 and confirm it
        // changes the digest relative to an all-zero key.
        let mut key = [0u8; 16];
        key[0] = 1;
        assert_ne!(siphash24(&key, &[]), siphash24(&[0u8; 16], &[]));
    }

    #[test]
    fn key_second_half_affects_output() {
        let mut key = [0u8; 16];
        key[8] = 1;
        assert_ne!(siphash24(&key, &[]), siphash24(&[0u8; 16], &[]));
        assert_ne!(siphash13(&key, &[]), siphash13(&[0u8; 16], &[]));
    }

    #[test]
    fn siphash24_distinct_lengths_mostly_distinct() {
        // The 64 reference-key digests over lengths 0..64 should all differ.
        let key = ref_key();
        let buf = seq64();
        let mut seen = [0u64; 64];
        for len in 0..64 {
            seen[len] = siphash24(&key, &buf[..len]);
        }
        for i in 0..64 {
            for j in (i + 1)..64 {
                assert_ne!(seen[i], seen[j], "collision between len {i} and {j}");
            }
        }
    }

    #[test]
    fn siphash13_distinct_lengths_mostly_distinct() {
        let key = ref_key();
        let buf = seq64();
        let mut seen = [0u64; 64];
        for len in 0..64 {
            seen[len] = siphash13(&key, &buf[..len]);
        }
        for i in 0..64 {
            for j in (i + 1)..64 {
                assert_ne!(seen[i], seen[j], "collision between len {i} and {j}");
            }
        }
    }

    #[test]
    fn siphash24_two_blocks_absorb_order() {
        // 16 bytes exercise two full blocks and the empty length-tagged tail.
        assert_eq!(siphash24(&ref_key(), &seq64()[..16]), SIPHASH24_VECTORS[16]);
    }

    #[test]
    fn siphash24_reordered_bytes_differ() {
        let key = ref_key();
        let a = [1u8, 2, 3, 4];
        let b = [4u8, 3, 2, 1];
        assert_ne!(siphash24(&key, &a), siphash24(&key, &b));
    }

    #[test]
    fn siphash13_reordered_bytes_differ() {
        let key = ref_key();
        let a = [1u8, 2, 3, 4];
        let b = [4u8, 3, 2, 1];
        assert_ne!(siphash13(&key, &a), siphash13(&key, &b));
    }
}
