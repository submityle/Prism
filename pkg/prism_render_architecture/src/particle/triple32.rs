//! `triple32` integer `hash` (Chris Wellons, 32-bit avalanche mixer).
//!
//! Pure-integer, deterministic, bijective 32-bit `hash`. Suitable for a
//! `no_std` + `alloc` context: it uses only `u32` arithmetic (wrapping
//! multiplies and parenthesized shifts), never floating point or any
//! transcendental function, and allocates nothing.

/// `triple32` avalanche `hash` over the full 32-bit domain.
///
/// Each stage is an xor-shift followed by a wrapping multiply by an odd
/// constant, which keeps the mapping a bijection on `u32`.
pub fn triple32(mut x: u32) -> u32 {
    x ^= x >> 17;
    x = x.wrapping_mul(0xed5a_d4bb);
    x ^= x >> 11;
    x = x.wrapping_mul(0xac4c_1b51);
    x ^= x >> 15;
    x = x.wrapping_mul(0x3184_8bab);
    x ^= x >> 14;
    x
}

#[cfg(test)]
mod tests {
    use super::triple32;

    // --- Anchors: hard ground-truth reference vectors. ---

    #[test]
    fn anchor_zero() {
        let x = triple32(0);
        assert!(x == 0x0000_0000);
    }

    #[test]
    fn anchor_one() {
        let x = triple32(1);
        assert!(x == 0x0427_41d6);
    }

    #[test]
    fn anchor_two() {
        let x = triple32(2);
        assert!(x == 0xf1df_e8e9);
    }

    #[test]
    fn anchor_deadbeef() {
        let x = triple32(0xdead_beef);
        assert!(x == 0x0921_725e);
    }

    #[test]
    fn anchor_all_ones() {
        let x = triple32(0xffff_ffff);
        assert!(x == 0x127f_588f);
    }

    // --- Additional exact-value points. ---

    #[test]
    fn exact_three() {
        let x = triple32(3);
        assert!(x == 0xc0f0_b547);
    }

    #[test]
    fn exact_four() {
        let x = triple32(4);
        assert!(x == 0xd3a1_5f95);
    }

    #[test]
    fn exact_five() {
        let x = triple32(5);
        assert!(x == 0xe33d_e521);
    }

    #[test]
    fn exact_ten() {
        let x = triple32(10);
        assert!(x == 0x082c_8e1a);
    }

    #[test]
    fn exact_hundred() {
        let x = triple32(100);
        assert!(x == 0x8ea8_3b8a);
    }

    #[test]
    fn exact_two_five_five() {
        let x = triple32(255);
        assert!(x == 0xe4f7_8f5d);
    }

    #[test]
    fn exact_two_five_six() {
        let x = triple32(256);
        assert!(x == 0x462d_98bd);
    }

    #[test]
    fn exact_thousand() {
        let x = triple32(1000);
        assert!(x == 0x17e0_3a9e);
    }

    #[test]
    fn exact_forty_two() {
        let x = triple32(42);
        assert!(x == 0x9a67_5f94);
    }

    #[test]
    fn exact_mixed_a() {
        let x = triple32(0x1234_5678);
        assert!(x == 0xfac9_70ff);
    }

    #[test]
    fn exact_mixed_b() {
        let x = triple32(0xabcd_ef01);
        assert!(x == 0x7256_92ce);
    }

    #[test]
    fn exact_msb_only() {
        let x = triple32(0x8000_0000);
        assert!(x == 0x3972_6c96);
    }

    #[test]
    fn exact_all_but_msb() {
        let x = triple32(0x7fff_ffff);
        assert!(x == 0x383b_512a);
    }

    #[test]
    fn exact_low_half() {
        let x = triple32(0x0000_ffff);
        assert!(x == 0x03fc_b5cd);
    }

    #[test]
    fn exact_high_half() {
        let x = triple32(0xffff_0000);
        assert!(x == 0xe617_07a4);
    }

    #[test]
    fn exact_cafebabe() {
        let x = triple32(0xcafe_babe);
        assert!(x == 0xd519_2c13);
    }

    #[test]
    fn exact_a5_pattern() {
        let x = triple32(0xa5a5_a5a5);
        assert!(x == 0xe1b4_8b91);
    }

    #[test]
    fn exact_5a_pattern() {
        let x = triple32(0x5a5a_5a5a);
        assert!(x == 0xc085_76c9);
    }

    #[test]
    fn exact_ones_nibble() {
        let x = triple32(0x1111_1111);
        assert!(x == 0x0c0b_d73f);
    }

    #[test]
    fn exact_f0_pattern() {
        let x = triple32(0xf0f0_f0f0);
        assert!(x == 0xf3f5_8dc5);
    }

    // --- Table-driven exact check over a fixed slice. ---

    #[test]
    fn exact_table_matches() {
        let cases: &[(u32, u32)] = &[
            (0, 0x0000_0000),
            (1, 0x0427_41d6),
            (2, 0xf1df_e8e9),
            (3, 0xc0f0_b547),
            (4, 0xd3a1_5f95),
            (5, 0xe33d_e521),
            (0xdead_beef, 0x0921_725e),
            (0xffff_ffff, 0x127f_588f),
        ];
        let mut i = 0;
        while i < cases.len() {
            let (input, expected) = cases[i];
            let got = triple32(input);
            assert!(got == expected);
            i += 1;
        }
    }

    // --- Determinism. ---

    #[test]
    fn deterministic_single() {
        let a = triple32(0x1357_9bdf);
        let b = triple32(0x1357_9bdf);
        assert!(a == b);
    }

    #[test]
    fn deterministic_over_range() {
        let mut i: u32 = 0;
        while i < 512 {
            let a = triple32(i);
            let b = triple32(i);
            assert!(a == b);
            i += 1;
        }
    }

    #[test]
    fn deterministic_sparse_points() {
        let points: &[u32] = &[
            0x0000_0001,
            0x1000_0000,
            0x0f0f_0f0f,
            0x8000_0001,
            0x7777_7777,
            0xaaaa_aaaa,
        ];
        let mut i = 0;
        while i < points.len() {
            let p = points[i];
            assert!(triple32(p) == triple32(p));
            i += 1;
        }
    }

    // --- Fixed point: zero maps to zero, and nothing else does. ---

    #[test]
    fn zero_is_fixed_point() {
        assert!(triple32(0) == 0);
    }

    #[test]
    fn only_zero_maps_to_zero_small_range() {
        let mut i: u32 = 1;
        while i < 4096 {
            assert!(triple32(i) != 0);
            i += 1;
        }
    }

    #[test]
    fn nonzero_inputs_sampled_are_nonzero() {
        let samples: &[u32] = &[
            0x0000_0001,
            0x0001_0000,
            0x1000_0000,
            0x8000_0000,
            0xffff_ffff,
            0xdead_beef,
        ];
        let mut i = 0;
        while i < samples.len() {
            assert!(triple32(samples[i]) != 0);
            i += 1;
        }
    }

    // --- Bit sensitivity (avalanche): a single-bit flip changes output. ---

    #[test]
    fn single_bit_flip_changes_output_from_zero() {
        let base = triple32(0);
        let mut bit = 0u32;
        while bit < 32 {
            let flipped = triple32(1u32 << bit);
            assert!(flipped != base);
            bit += 1;
        }
    }

    #[test]
    fn single_bit_flip_changes_output_from_base() {
        let base_input: u32 = 0x1234_5678;
        let base = triple32(base_input);
        let mut bit = 0u32;
        while bit < 32 {
            let flipped = triple32(base_input ^ (1u32 << bit));
            assert!(flipped != base);
            bit += 1;
        }
    }

    #[test]
    fn single_bit_flip_changes_many_bits() {
        // Avalanche: flipping one input bit should flip several output bits.
        let base_input: u32 = 0x0f1e_2d3c;
        let base = triple32(base_input);
        let mut bit = 0u32;
        while bit < 32 {
            let flipped = triple32(base_input ^ (1u32 << bit));
            let diff = (base ^ flipped).count_ones();
            assert!(diff >= 3);
            bit += 1;
        }
    }

    // --- Injectivity sampling (triple32 is a bijection). ---

    #[test]
    fn injective_over_small_range() {
        let n: u32 = 1500;
        let mut i: u32 = 0;
        while i < n {
            let hi = triple32(i);
            let mut j = i + 1;
            while j < n {
                assert!(hi != triple32(j));
                j += 1;
            }
            i += 1;
        }
    }

    #[test]
    fn injective_over_sparse_points() {
        let points: &[u32] = &[
            0x0000_0000,
            0x0000_0001,
            0x0000_00ff,
            0x0000_ff00,
            0x00ff_0000,
            0xff00_0000,
            0x1234_5678,
            0x8765_4321,
            0xdead_beef,
            0xcafe_babe,
            0xffff_ffff,
            0x7fff_ffff,
            0x8000_0000,
            0xaaaa_aaaa,
            0x5555_5555,
            0x0f0f_0f0f,
        ];
        let mut i = 0;
        while i < points.len() {
            let hi = triple32(points[i]);
            let mut j = i + 1;
            while j < points.len() {
                assert!(hi != triple32(points[j]));
                j += 1;
            }
            i += 1;
        }
    }

    #[test]
    fn injective_high_input_window() {
        let start: u32 = 0xffff_f000;
        let mut collisions = 0u32;
        let mut i = start;
        loop {
            let hi = triple32(i);
            let mut j = i.wrapping_add(1);
            while j != 0 {
                if triple32(j) == hi {
                    collisions += 1;
                }
                j = j.wrapping_add(1);
            }
            if i == 0xffff_ffff {
                break;
            }
            i += 1;
        }
        assert!(collisions == 0);
    }

    // --- Adjacent and structural distinctness. ---

    #[test]
    fn adjacent_inputs_differ() {
        let mut i: u32 = 0;
        while i < 2048 {
            assert!(triple32(i) != triple32(i + 1));
            i += 1;
        }
    }

    #[test]
    fn complementary_inputs_differ() {
        let samples: &[u32] = &[0, 1, 0x1234_5678, 0xdead_beef, 0x0f0f_0f0f];
        let mut i = 0;
        while i < samples.len() {
            let v = samples[i];
            assert!(triple32(v) != triple32(!v));
            i += 1;
        }
    }

    #[test]
    fn byte_swapped_inputs_differ() {
        let samples: &[u32] = &[0x0000_0001, 0x1234_5678, 0x00ab_cd00];
        let mut i = 0;
        while i < samples.len() {
            let v = samples[i];
            let swapped = v.swap_bytes();
            if swapped != v {
                assert!(triple32(v) != triple32(swapped));
            }
            i += 1;
        }
    }

    #[test]
    fn not_identity_mapping() {
        // The mixer should move almost every point off its input.
        let mut fixed = 0u32;
        let mut i: u32 = 0;
        while i < 4096 {
            if triple32(i) == i {
                fixed += 1;
            }
            i += 1;
        }
        // Only the true fixed point (0) is expected in this window.
        assert!(fixed == 1);
    }

    #[test]
    fn output_bits_are_balanced_sample() {
        // Over a window, the top output bit should not be stuck.
        let mut set_count = 0u32;
        let mut i: u32 = 0;
        while i < 1024 {
            if (triple32(i) >> 31) & 1 == 1 {
                set_count += 1;
            }
            i += 1;
        }
        assert!(set_count > 0);
        assert!(set_count < 1024);
    }

    #[test]
    fn low_output_bit_is_balanced_sample() {
        let mut set_count = 0u32;
        let mut i: u32 = 0;
        while i < 1024 {
            set_count += triple32(i) & 1;
            i += 1;
        }
        assert!(set_count > 0);
        assert!(set_count < 1024);
    }

    #[test]
    fn double_application_is_not_identity_sample() {
        let samples: &[u32] = &[1, 2, 3, 0x1234_5678, 0xdead_beef];
        let mut i = 0;
        while i < samples.len() {
            let v = samples[i];
            assert!(triple32(triple32(v)) != v);
            i += 1;
        }
    }

    #[test]
    fn powers_of_two_are_distinct() {
        let mut bit_a = 0u32;
        while bit_a < 32 {
            let ha = triple32(1u32 << bit_a);
            let mut bit_b = bit_a + 1;
            while bit_b < 32 {
                assert!(ha != triple32(1u32 << bit_b));
                bit_b += 1;
            }
            bit_a += 1;
        }
    }
}
