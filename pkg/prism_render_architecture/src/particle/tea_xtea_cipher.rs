//! `TEA` and `XTEA` block ciphers: tiny 64-bit-block, 128-bit-key Feistel
//! ciphers used here only as cheap, deterministic bit-mixing primitives for
//! particle/`GPU` payload scrambling and reproducible pseudo-random streams
//! (design § lightweight obfuscation), *not* as a security guarantee.
//!
//! Each cipher operates on a 64-bit block expressed as two 32-bit words
//! `[v0, v1]` under a 128-bit key `[u32; 4]`. Both run 32 cycles (64 Feistel
//! rounds) driven by a running sum of the golden-ratio constant [`DELTA`]
//! (`0x9E37_79B9`). Encryption accumulates `DELTA` into `sum`; decryption walks
//! the schedule in reverse starting from `DELTA * 32` (which equals the
//! constant `0xC6EF_3720`). Every operation is a 32-bit wrapping add/subtract,
//! shift, or xor; there are no floating-point or transcendental operations.
//!
//! `TEA` has well-known weaknesses (equivalent keys, related-key attacks);
//! `XTEA` repairs the key schedule but neither is a modern, audited cipher.
//! Treat both as deterministic mixing functions only; never use them to protect
//! secrets against an adversary. The `CPU` reference here mirrors what a `GPU`
//! shader can recompute bit-for-bit.

/// The `TEA`/`XTEA` round constant: `floor(2^32 / golden_ratio)`.
pub const DELTA: u32 = 0x9E37_79B9;

/// Encrypts one 64-bit block with the classic `TEA` cipher (32 cycles).
///
/// `tea_encrypt([0, 0], &[0, 0, 0, 0])` is the reference block
/// `[0x41EA_3A0A, 0x94BA_A940]`.
#[must_use]
pub fn tea_encrypt(block: [u32; 2], key: &[u32; 4]) -> [u32; 2] {
    let k = key;
    let mut sum: u32 = 0;
    let (mut v0, mut v1) = (block[0], block[1]);
    for _ in 0..32 {
        sum = sum.wrapping_add(DELTA);
        v0 = v0.wrapping_add(
            ((v1 << 4).wrapping_add(k[0])) ^ v1.wrapping_add(sum) ^ ((v1 >> 5).wrapping_add(k[1])),
        );
        v1 = v1.wrapping_add(
            ((v0 << 4).wrapping_add(k[2])) ^ v0.wrapping_add(sum) ^ ((v0 >> 5).wrapping_add(k[3])),
        );
    }
    [v0, v1]
}

/// Decrypts one 64-bit block produced by [`tea_encrypt`] under the same key.
///
/// The round schedule is walked in reverse, starting from `DELTA * 32`.
#[must_use]
pub fn tea_decrypt(block: [u32; 2], key: &[u32; 4]) -> [u32; 2] {
    let k = key;
    let (mut v0, mut v1) = (block[0], block[1]);
    let mut sum: u32 = DELTA.wrapping_mul(32);
    for _ in 0..32 {
        v1 = v1.wrapping_sub(
            ((v0 << 4).wrapping_add(k[2])) ^ v0.wrapping_add(sum) ^ ((v0 >> 5).wrapping_add(k[3])),
        );
        v0 = v0.wrapping_sub(
            ((v1 << 4).wrapping_add(k[0])) ^ v1.wrapping_add(sum) ^ ((v1 >> 5).wrapping_add(k[1])),
        );
        sum = sum.wrapping_sub(DELTA);
    }
    [v0, v1]
}

/// Encrypts one 64-bit block with the `XTEA` cipher (32 cycles).
///
/// `XTEA` strengthens `TEA`'s key schedule by selecting a key word per round
/// from `sum`. `xtea_encrypt([0, 0], &[0, 0, 0, 0])` is the reference block
/// `[0xDEE9_D4D8, 0xF713_1ED9]`.
#[must_use]
pub fn xtea_encrypt(block: [u32; 2], key: &[u32; 4]) -> [u32; 2] {
    let k = key;
    let (mut v0, mut v1) = (block[0], block[1]);
    let mut sum: u32 = 0;
    for _ in 0..32 {
        v0 = v0.wrapping_add(
            (((v1 << 4) ^ (v1 >> 5)).wrapping_add(v1)) ^ sum.wrapping_add(k[(sum & 3) as usize]),
        );
        sum = sum.wrapping_add(DELTA);
        v1 = v1.wrapping_add(
            (((v0 << 4) ^ (v0 >> 5)).wrapping_add(v0))
                ^ sum.wrapping_add(k[((sum >> 11) & 3) as usize]),
        );
    }
    [v0, v1]
}

/// Decrypts one 64-bit block produced by [`xtea_encrypt`] under the same key.
///
/// The round schedule is walked in reverse, starting from `DELTA * 32`.
#[must_use]
pub fn xtea_decrypt(block: [u32; 2], key: &[u32; 4]) -> [u32; 2] {
    let k = key;
    let (mut v0, mut v1) = (block[0], block[1]);
    let mut sum: u32 = DELTA.wrapping_mul(32);
    for _ in 0..32 {
        v1 = v1.wrapping_sub(
            (((v0 << 4) ^ (v0 >> 5)).wrapping_add(v0))
                ^ sum.wrapping_add(k[((sum >> 11) & 3) as usize]),
        );
        sum = sum.wrapping_sub(DELTA);
        v0 = v0.wrapping_sub(
            (((v1 << 4) ^ (v1 >> 5)).wrapping_add(v1)) ^ sum.wrapping_add(k[(sum & 3) as usize]),
        );
    }
    [v0, v1]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Counts how many bits differ between two 64-bit blocks.
    #[cfg(test)]
    fn bit_diff(a: [u32; 2], b: [u32; 2]) -> u32 {
        (a[0] ^ b[0]).count_ones() + (a[1] ^ b[1]).count_ones()
    }

    #[test]
    fn delta_constant_value() {
        assert_eq!(DELTA, 0x9E37_79B9);
    }

    #[test]
    fn delta_times_32_is_known_constant() {
        assert_eq!(DELTA.wrapping_mul(32), 0xC6EF_3720);
    }

    // ---- hard reference vectors ----

    #[test]
    fn tea_encrypt_reference_vector() {
        assert_eq!(
            tea_encrypt([0, 0], &[0, 0, 0, 0]),
            [0x41EA_3A0A, 0x94BA_A940]
        );
    }

    #[test]
    fn tea_decrypt_reference_vector() {
        assert_eq!(
            tea_decrypt([0x41EA_3A0A, 0x94BA_A940], &[0, 0, 0, 0]),
            [0, 0]
        );
    }

    #[test]
    fn xtea_encrypt_reference_vector() {
        assert_eq!(
            xtea_encrypt([0, 0], &[0, 0, 0, 0]),
            [0xDEE9_D4D8, 0xF713_1ED9]
        );
    }

    #[test]
    fn xtea_decrypt_reference_vector() {
        assert_eq!(
            xtea_decrypt([0xDEE9_D4D8, 0xF713_1ED9], &[0, 0, 0, 0]),
            [0, 0]
        );
    }

    // ---- TEA round trips ----

    #[test]
    fn tea_roundtrip_zero_block_zero_key() {
        let k = [0, 0, 0, 0];
        assert_eq!(tea_decrypt(tea_encrypt([0, 0], &k), &k), [0, 0]);
    }

    #[test]
    fn tea_roundtrip_simple_block() {
        let k = [1, 2, 3, 4];
        let b = [0x1234_5678, 0x9ABC_DEF0];
        assert_eq!(tea_decrypt(tea_encrypt(b, &k), &k), b);
    }

    #[test]
    fn tea_roundtrip_block_a() {
        let k = [0xDEAD_BEEF, 0xCAFE_BABE, 0x0BAD_F00D, 0xFEED_FACE];
        let b = [0x0000_0001, 0x8000_0000];
        assert_eq!(tea_decrypt(tea_encrypt(b, &k), &k), b);
    }

    #[test]
    fn tea_roundtrip_block_b() {
        let k = [0x1111_1111, 0x2222_2222, 0x3333_3333, 0x4444_4444];
        let b = [0xA5A5_A5A5, 0x5A5A_5A5A];
        assert_eq!(tea_decrypt(tea_encrypt(b, &k), &k), b);
    }

    #[test]
    fn tea_roundtrip_block_c() {
        let k = [0xFFFF_0000, 0x0000_FFFF, 0x00FF_00FF, 0xFF00_FF00];
        let b = [0x0123_4567, 0x89AB_CDEF];
        assert_eq!(tea_decrypt(tea_encrypt(b, &k), &k), b);
    }

    #[test]
    fn tea_roundtrip_block_d() {
        let k = [7, 13, 1009, 65537];
        let b = [42, 0xFFFF_FFFF];
        assert_eq!(tea_decrypt(tea_encrypt(b, &k), &k), b);
    }

    #[test]
    fn tea_roundtrip_block_e() {
        let k = [0x9E37_79B9, 0xC6EF_3720, 0x1357_9BDF, 0x2468_ACE0];
        let b = [0xDEAD_BEEF, 0xFEED_FACE];
        assert_eq!(tea_decrypt(tea_encrypt(b, &k), &k), b);
    }

    #[test]
    fn tea_roundtrip_many_pseudorandom() {
        // Simple LCG to generate many block/key pairs deterministically.
        let mut state: u64 = 0x1234_5678_9ABC_DEF0;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 32) as u32
        };
        for _ in 0..64 {
            let b = [next(), next()];
            let k = [next(), next(), next(), next()];
            assert_eq!(tea_decrypt(tea_encrypt(b, &k), &k), b);
        }
    }

    // ---- XTEA round trips ----

    #[test]
    fn xtea_roundtrip_zero_block_zero_key() {
        let k = [0, 0, 0, 0];
        assert_eq!(xtea_decrypt(xtea_encrypt([0, 0], &k), &k), [0, 0]);
    }

    #[test]
    fn xtea_roundtrip_simple_block() {
        let k = [1, 2, 3, 4];
        let b = [0x1234_5678, 0x9ABC_DEF0];
        assert_eq!(xtea_decrypt(xtea_encrypt(b, &k), &k), b);
    }

    #[test]
    fn xtea_roundtrip_block_a() {
        let k = [0xDEAD_BEEF, 0xCAFE_BABE, 0x0BAD_F00D, 0xFEED_FACE];
        let b = [0x0000_0001, 0x8000_0000];
        assert_eq!(xtea_decrypt(xtea_encrypt(b, &k), &k), b);
    }

    #[test]
    fn xtea_roundtrip_block_b() {
        let k = [0x1111_1111, 0x2222_2222, 0x3333_3333, 0x4444_4444];
        let b = [0xA5A5_A5A5, 0x5A5A_5A5A];
        assert_eq!(xtea_decrypt(xtea_encrypt(b, &k), &k), b);
    }

    #[test]
    fn xtea_roundtrip_block_c() {
        let k = [0xFFFF_0000, 0x0000_FFFF, 0x00FF_00FF, 0xFF00_FF00];
        let b = [0x0123_4567, 0x89AB_CDEF];
        assert_eq!(xtea_decrypt(xtea_encrypt(b, &k), &k), b);
    }

    #[test]
    fn xtea_roundtrip_block_d() {
        let k = [7, 13, 1009, 65537];
        let b = [42, 0xFFFF_FFFF];
        assert_eq!(xtea_decrypt(xtea_encrypt(b, &k), &k), b);
    }

    #[test]
    fn xtea_roundtrip_block_e() {
        let k = [0x9E37_79B9, 0xC6EF_3720, 0x1357_9BDF, 0x2468_ACE0];
        let b = [0xDEAD_BEEF, 0xFEED_FACE];
        assert_eq!(xtea_decrypt(xtea_encrypt(b, &k), &k), b);
    }

    #[test]
    fn xtea_roundtrip_many_pseudorandom() {
        let mut state: u64 = 0x0FED_CBA9_8765_4321;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 32) as u32
        };
        for _ in 0..64 {
            let b = [next(), next()];
            let k = [next(), next(), next(), next()];
            assert_eq!(xtea_decrypt(xtea_encrypt(b, &k), &k), b);
        }
    }

    // ---- boundary blocks/keys ----

    #[test]
    fn tea_roundtrip_all_ones_block_zero_key() {
        let k = [0, 0, 0, 0];
        let b = [0xFFFF_FFFF, 0xFFFF_FFFF];
        assert_eq!(tea_decrypt(tea_encrypt(b, &k), &k), b);
    }

    #[test]
    fn tea_roundtrip_all_ones_block_all_ones_key() {
        let k = [0xFFFF_FFFF; 4];
        let b = [0xFFFF_FFFF, 0xFFFF_FFFF];
        assert_eq!(tea_decrypt(tea_encrypt(b, &k), &k), b);
    }

    #[test]
    fn tea_roundtrip_zero_block_all_ones_key() {
        let k = [0xFFFF_FFFF; 4];
        let b = [0, 0];
        assert_eq!(tea_decrypt(tea_encrypt(b, &k), &k), b);
    }

    #[test]
    fn xtea_roundtrip_all_ones_block_zero_key() {
        let k = [0, 0, 0, 0];
        let b = [0xFFFF_FFFF, 0xFFFF_FFFF];
        assert_eq!(xtea_decrypt(xtea_encrypt(b, &k), &k), b);
    }

    #[test]
    fn xtea_roundtrip_all_ones_block_all_ones_key() {
        let k = [0xFFFF_FFFF; 4];
        let b = [0xFFFF_FFFF, 0xFFFF_FFFF];
        assert_eq!(xtea_decrypt(xtea_encrypt(b, &k), &k), b);
    }

    #[test]
    fn xtea_roundtrip_zero_block_all_ones_key() {
        let k = [0xFFFF_FFFF; 4];
        let b = [0, 0];
        assert_eq!(xtea_decrypt(xtea_encrypt(b, &k), &k), b);
    }

    // ---- avalanche ----

    #[test]
    fn tea_avalanche_plaintext_bit() {
        let k = [0x1234_5678, 0x9ABC_DEF0, 0x0F1E_2D3C, 0x4B5A_6978];
        let b = [0x0000_0000, 0x0000_0000];
        let b2 = [0x0000_0001, 0x0000_0000];
        let diff = bit_diff(tea_encrypt(b, &k), tea_encrypt(b2, &k));
        assert!(diff > 20, "weak TEA plaintext avalanche: {diff}");
    }

    #[test]
    fn tea_avalanche_key_bit() {
        let k = [0, 0, 0, 0];
        let k2 = [1, 0, 0, 0];
        let b = [0xDEAD_BEEF, 0xFEED_FACE];
        let diff = bit_diff(tea_encrypt(b, &k), tea_encrypt(b, &k2));
        assert!(diff > 20, "weak TEA key avalanche: {diff}");
    }

    #[test]
    fn tea_avalanche_high_bit() {
        let k = [0xA, 0xB, 0xC, 0xD];
        let b = [0x0000_0000, 0x0000_0000];
        let b2 = [0x0000_0000, 0x8000_0000];
        let diff = bit_diff(tea_encrypt(b, &k), tea_encrypt(b2, &k));
        assert!(diff > 20, "weak TEA high-bit avalanche: {diff}");
    }

    #[test]
    fn xtea_avalanche_plaintext_bit() {
        let k = [0x1234_5678, 0x9ABC_DEF0, 0x0F1E_2D3C, 0x4B5A_6978];
        let b = [0x0000_0000, 0x0000_0000];
        let b2 = [0x0000_0001, 0x0000_0000];
        let diff = bit_diff(xtea_encrypt(b, &k), xtea_encrypt(b2, &k));
        assert!(diff > 20, "weak XTEA plaintext avalanche: {diff}");
    }

    #[test]
    fn xtea_avalanche_key_bit() {
        let k = [0, 0, 0, 0];
        let k2 = [0, 0, 0, 1];
        let b = [0xDEAD_BEEF, 0xFEED_FACE];
        let diff = bit_diff(xtea_encrypt(b, &k), xtea_encrypt(b, &k2));
        assert!(diff > 20, "weak XTEA key avalanche: {diff}");
    }

    #[test]
    fn xtea_avalanche_high_bit() {
        let k = [0xA, 0xB, 0xC, 0xD];
        let b = [0x0000_0000, 0x0000_0000];
        let b2 = [0x0000_0000, 0x8000_0000];
        let diff = bit_diff(xtea_encrypt(b, &k), xtea_encrypt(b2, &k));
        assert!(diff > 20, "weak XTEA high-bit avalanche: {diff}");
    }

    // ---- distinctness / sanity ----

    #[test]
    fn tea_and_xtea_differ_on_same_input() {
        let k = [1, 2, 3, 4];
        let b = [0x1234_5678, 0x9ABC_DEF0];
        assert_ne!(tea_encrypt(b, &k), xtea_encrypt(b, &k));
    }

    #[test]
    fn tea_encrypt_changes_zero_block() {
        assert_ne!(tea_encrypt([0, 0], &[0, 0, 0, 0]), [0, 0]);
    }

    #[test]
    fn xtea_encrypt_changes_zero_block() {
        assert_ne!(xtea_encrypt([0, 0], &[0, 0, 0, 0]), [0, 0]);
    }

    #[test]
    fn tea_different_keys_give_different_ciphertext() {
        let b = [0x5555_5555, 0xAAAA_AAAA];
        let c1 = tea_encrypt(b, &[1, 0, 0, 0]);
        let c2 = tea_encrypt(b, &[2, 0, 0, 0]);
        assert_ne!(c1, c2);
    }

    #[test]
    fn xtea_different_keys_give_different_ciphertext() {
        let b = [0x5555_5555, 0xAAAA_AAAA];
        let c1 = xtea_encrypt(b, &[1, 0, 0, 0]);
        let c2 = xtea_encrypt(b, &[2, 0, 0, 0]);
        assert_ne!(c1, c2);
    }

    #[test]
    fn tea_distinct_blocks_distinct_ciphertext() {
        let k = [9, 8, 7, 6];
        let c1 = tea_encrypt([1, 1], &k);
        let c2 = tea_encrypt([2, 2], &k);
        assert_ne!(c1, c2);
    }

    #[test]
    fn xtea_distinct_blocks_distinct_ciphertext() {
        let k = [9, 8, 7, 6];
        let c1 = xtea_encrypt([1, 1], &k);
        let c2 = xtea_encrypt([2, 2], &k);
        assert_ne!(c1, c2);
    }

    #[test]
    fn tea_roundtrip_swapped_words() {
        let k = [0xABCD_1234, 0x5678_90EF, 0xFEDC_BA98, 0x7654_3210];
        let b = [0x9ABC_DEF0, 0x1234_5678];
        assert_eq!(tea_decrypt(tea_encrypt(b, &k), &k), b);
    }

    #[test]
    fn xtea_roundtrip_swapped_words() {
        let k = [0xABCD_1234, 0x5678_90EF, 0xFEDC_BA98, 0x7654_3210];
        let b = [0x9ABC_DEF0, 0x1234_5678];
        assert_eq!(xtea_decrypt(xtea_encrypt(b, &k), &k), b);
    }

    #[test]
    fn tea_encrypt_is_deterministic() {
        let k = [3, 1, 4, 1];
        let b = [0x2718_2818, 0x3141_5926];
        assert_eq!(tea_encrypt(b, &k), tea_encrypt(b, &k));
    }

    #[test]
    fn xtea_encrypt_is_deterministic() {
        let k = [3, 1, 4, 1];
        let b = [0x2718_2818, 0x3141_5926];
        assert_eq!(xtea_encrypt(b, &k), xtea_encrypt(b, &k));
    }
}
