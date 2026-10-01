//! `MurmurHash3` 64-bit finalizer (`fmix64`).
//!
//! Pure integer (`u64`) avalanche mixing used to derive deterministic,
//! well-distributed per-particle seeds on both `CPU` and `GPU` paths. No
//! floating point is involved, so the result is bit-for-bit reproducible
//! across platforms. All multiplications use `wrapping_mul`, and every shift
//! is parenthesized to make precedence explicit.

/// Applies the `MurmurHash3` 64-bit finalizer to `x`.
///
/// This is the standard `fmix64` avalanche step: alternating xor-shift and
/// `wrapping_mul` operations that scramble input bits so that flipping any
/// single input bit changes roughly half of the output bits.
#[must_use]
pub fn fmix64(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^= x >> 33;
    x
}

#[cfg(test)]
mod tests {
    use super::fmix64;

    // ----- anchor reference vectors (ground truth) -----

    #[test]
    fn anchor_zero() {
        assert!(fmix64(0) == 0x0000000000000000);
    }

    #[test]
    fn anchor_one() {
        assert!(fmix64(1) == 0xb456bcfc34c2cb2c);
    }

    #[test]
    fn anchor_two() {
        assert!(fmix64(2) == 0x3abf2a20650683e7);
    }

    #[test]
    fn anchor_mixed_pattern() {
        assert!(fmix64(0x0123456789abcdef) == 0x87cbfbfe89022cea);
    }

    #[test]
    fn anchor_all_ones() {
        assert!(fmix64(0xffffffffffffffff) == 0x64b5720b4b825f21);
    }

    // ----- additional exact-value points -----

    #[test]
    fn exact_three() {
        assert!(fmix64(3) == 0x0b5181c509f8d8ce);
    }

    #[test]
    fn exact_four() {
        assert!(fmix64(4) == 0x47900468a8f01875);
    }

    #[test]
    fn exact_five() {
        assert!(fmix64(5) == 0xd66ad737d54c5575);
    }

    #[test]
    fn exact_ten() {
        assert!(fmix64(10) == 0x646172442548d30d);
    }

    #[test]
    fn exact_hundred() {
        assert!(fmix64(100) == 0xe6be2c94a54a2c62);
    }

    #[test]
    fn exact_byte_max() {
        assert!(fmix64(0xff) == 0x1200a2a61d248b28);
    }

    #[test]
    fn exact_two_fifty_six() {
        assert!(fmix64(0x100) == 0xd5bbc1f38c9b893b);
    }

    #[test]
    fn exact_thousand() {
        assert!(fmix64(1000) == 0xaf002c114878da41);
    }

    #[test]
    fn exact_deadbeef() {
        assert!(fmix64(0xdeadbeef) == 0xd24bd59f862a1dac);
    }

    #[test]
    fn exact_all_ones_minus_one() {
        assert!(fmix64(0xfffffffffffffffe) == 0x3a8593886c55a02b);
    }

    #[test]
    fn exact_high_bit_only() {
        assert!(fmix64(0x8000000000000000) == 0x8f780810af31a493);
    }

    #[test]
    fn exact_wide_mixed() {
        assert!(fmix64(0x1234567890abcdef) == 0x0cae996fee6bd396);
    }

    #[test]
    fn exact_forty_two() {
        assert!(fmix64(42) == 0x810879608e4259cc);
    }

    #[test]
    fn exact_cafebabe() {
        assert!(fmix64(0xcafebabe) == 0x0db5e5ac7fb03886);
    }

    #[test]
    fn exact_alternating_high() {
        assert!(fmix64(0xaaaaaaaaaaaaaaaa) == 0xdf6f9107dbf4372b);
    }

    #[test]
    fn exact_alternating_low() {
        assert!(fmix64(0x5555555555555555) == 0xbfa76d135217973d);
    }

    #[test]
    fn exact_signed_max() {
        assert!(fmix64(0x7fffffffffffffff) == 0xabb93df0a930edea);
    }

    #[test]
    fn exact_nibble_pattern() {
        assert!(fmix64(0xf0f0f0f0f0f0f0f0) == 0xc0ca6ad7cf1cbc84);
    }

    // ----- table-driven exact values -----

    #[test]
    fn table_exact_values() {
        let cases: [(u64, u64); 8] = [
            (0, 0x0000000000000000),
            (1, 0xb456bcfc34c2cb2c),
            (2, 0x3abf2a20650683e7),
            (3, 0x0b5181c509f8d8ce),
            (0x0123456789abcdef, 0x87cbfbfe89022cea),
            (0xffffffffffffffff, 0x64b5720b4b825f21),
            (0x8000000000000000, 0x8f780810af31a493),
            (42, 0x810879608e4259cc),
        ];
        let mut i = 0;
        while i < cases.len() {
            let (input, expected) = cases[i];
            assert!(fmix64(input) == expected);
            i += 1;
        }
    }

    // ----- determinism -----

    #[test]
    fn determinism_repeated_calls() {
        let a = fmix64(0x0123456789abcdef);
        let b = fmix64(0x0123456789abcdef);
        assert!(a == b);
    }

    #[test]
    fn determinism_across_inputs() {
        let probes: [u64; 6] = [0, 1, 7, 0xdeadbeef, 0x100, 0xffffffffffffffff];
        let mut i = 0;
        while i < probes.len() {
            let first = fmix64(probes[i]);
            let second = fmix64(probes[i]);
            assert!(first == second);
            i += 1;
        }
    }

    // ----- zero is a fixed point -----

    #[test]
    fn zero_is_fixed_point() {
        assert!(fmix64(0) == 0);
    }

    #[test]
    fn only_zero_maps_to_zero_in_sample() {
        let mut x: u64 = 1;
        while x < 2048 {
            assert!(fmix64(x) != 0);
            x += 1;
        }
    }

    // ----- bit sensitivity (single-bit flips) -----

    #[test]
    fn single_bit_flip_changes_output_from_zero() {
        let base = fmix64(0);
        let mut bit = 0;
        while bit < 64 {
            let flipped = fmix64(1u64 << bit);
            assert!(flipped != base);
            bit += 1;
        }
    }

    #[test]
    fn single_bit_flip_changes_output_from_pattern() {
        let seed: u64 = 0x0123456789abcdef;
        let base = fmix64(seed);
        let mut bit = 0;
        while bit < 64 {
            let flipped = fmix64(seed ^ (1u64 << bit));
            assert!(flipped != base);
            bit += 1;
        }
    }

    #[test]
    fn single_bit_flip_strong_avalanche() {
        // Each single-bit input flip should change at least 8 output bits.
        let seed: u64 = 0xcafebabe12345678;
        let base = fmix64(seed);
        let mut bit = 0;
        while bit < 64 {
            let flipped = fmix64(seed ^ (1u64 << bit));
            let changed = (flipped ^ base).count_ones();
            assert!(changed >= 8);
            bit += 1;
        }
    }

    // ----- injectivity sampling -----

    #[test]
    fn injective_on_small_contiguous_range() {
        // fmix64 is a bijection; a sampled window must have no collisions.
        let mut outputs: [u64; 256] = [0; 256];
        let mut i: usize = 0;
        while i < outputs.len() {
            outputs[i] = fmix64(i as u64);
            i += 1;
        }
        let mut a = 0;
        while a < outputs.len() {
            let mut b = a + 1;
            while b < outputs.len() {
                assert!(outputs[a] != outputs[b]);
                b += 1;
            }
            a += 1;
        }
    }

    #[test]
    fn injective_on_strided_sample() {
        let mut outputs: [u64; 64] = [0; 64];
        let mut i: usize = 0;
        while i < outputs.len() {
            outputs[i] = fmix64((i as u64).wrapping_mul(0x9e3779b97f4a7c15));
            i += 1;
        }
        let mut a = 0;
        while a < outputs.len() {
            let mut b = a + 1;
            while b < outputs.len() {
                assert!(outputs[a] != outputs[b]);
                b += 1;
            }
            a += 1;
        }
    }

    #[test]
    fn injective_on_high_bit_sample() {
        let mut outputs: [u64; 64] = [0; 64];
        let mut i: usize = 0;
        while i < outputs.len() {
            outputs[i] = fmix64(0x8000000000000000 ^ (i as u64));
            i += 1;
        }
        let mut a = 0;
        while a < outputs.len() {
            let mut b = a + 1;
            while b < outputs.len() {
                assert!(outputs[a] != outputs[b]);
                b += 1;
            }
            a += 1;
        }
    }

    // ----- structural / distinctness checks -----

    #[test]
    fn distinct_outputs_for_distinct_small_inputs() {
        assert!(fmix64(1) != fmix64(2));
        assert!(fmix64(2) != fmix64(3));
        assert!(fmix64(3) != fmix64(4));
    }

    #[test]
    fn adjacent_inputs_differ() {
        let mut x: u64 = 0;
        while x < 512 {
            assert!(fmix64(x) != fmix64(x + 1));
            x += 1;
        }
    }

    #[test]
    fn nonzero_outputs_in_range() {
        let mut x: u64 = 1;
        while x < 1024 {
            let y = fmix64(x);
            assert!(y != 0);
            x += 1;
        }
    }

    #[test]
    fn high_and_low_halves_can_both_be_set() {
        // Sanity: outputs are not trivially confined to one half.
        let a = fmix64(0x0123456789abcdef);
        assert!((a >> 32) != 0);
        assert!((a & 0xffffffff) != 0);
    }

    #[test]
    fn complementary_inputs_distinct() {
        let a = fmix64(0x0f0f0f0f0f0f0f0f);
        let b = fmix64(0xf0f0f0f0f0f0f0f0);
        assert!(a != b);
    }

    #[test]
    fn shift_variants_distinct() {
        let a = fmix64(0x0000000000000001);
        let b = fmix64(0x0000000000010000);
        let c = fmix64(0x0001000000000000);
        assert!(a != b);
        assert!(b != c);
        assert!(a != c);
    }

    #[test]
    fn count_ones_varies_across_sample() {
        // The population count of outputs should not be constant.
        let first = fmix64(1).count_ones();
        let mut saw_different = false;
        let mut x: u64 = 2;
        while x < 64 {
            if fmix64(x).count_ones() != first {
                saw_different = true;
            }
            x += 1;
        }
        assert!(saw_different);
    }
}
