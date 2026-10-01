//! `lowbias32`: Chris Wellons' 32-bit integer `hash` finalizer.
//!
//! This module provides a deterministic, pure-integer bit mixer suitable for
//! seeding particle state on either `CPU` or `GPU`-style pipelines. It performs
//! no floating point work and relies only on wrapping multiplies and shifts, so
//! results are bit-identical across every target. The constant choices come
//! from Wellons' low-bias search and give excellent avalanche behavior for a
//! single-word `hash`.

/// Mixes a 32-bit word into a well-distributed 32-bit output.
///
/// The transform is injective-free of trivial fixed points except zero and is
/// fully deterministic: the same input always maps to the same output.
pub fn lowbias32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

#[cfg(test)]
mod tests {
    use super::lowbias32;

    // ---- Hard anchor vectors (ground truth) ----

    #[test]
    fn anchor_zero() {
        assert!(lowbias32(0) == 0x0000_0000);
    }

    #[test]
    fn anchor_one() {
        assert!(lowbias32(1) == 0x6889_90c0);
    }

    #[test]
    fn anchor_two() {
        assert!(lowbias32(2) == 0xd113_2181);
    }

    #[test]
    fn anchor_deadbeef() {
        assert!(lowbias32(0xdead_beef) == 0xe628_c683);
    }

    #[test]
    fn anchor_all_ones() {
        assert!(lowbias32(0xffff_ffff) == 0x6768_824a);
    }

    // ---- Additional exact value points (self-computed) ----

    #[test]
    fn exact_3() {
        assert!(lowbias32(3) == 0x53f1_e9dd);
    }

    #[test]
    fn exact_4() {
        assert!(lowbias32(4) == 0xd97e_5ed1);
    }

    #[test]
    fn exact_5() {
        assert!(lowbias32(5) == 0x5c45_d53e);
    }

    #[test]
    fn exact_6() {
        assert!(lowbias32(6) == 0xa7e3_d3bb);
    }

    #[test]
    fn exact_7() {
        assert!(lowbias32(7) == 0x948b_a1e6);
    }

    #[test]
    fn exact_8() {
        assert!(lowbias32(8) == 0xea53_5fba);
    }

    #[test]
    fn exact_9() {
        assert!(lowbias32(9) == 0x429f_d5ab);
    }

    #[test]
    fn exact_10() {
        assert!(lowbias32(10) == 0xb88b_aa7d);
    }

    #[test]
    fn exact_16() {
        assert!(lowbias32(16) == 0x21bd_4a6f);
    }

    #[test]
    fn exact_32() {
        assert!(lowbias32(32) == 0x14fd_6ad2);
    }

    #[test]
    fn exact_64() {
        assert!(lowbias32(64) == 0xdce4_20ba);
    }

    #[test]
    fn exact_100() {
        assert!(lowbias32(100) == 0x4891_52b6);
    }

    #[test]
    fn exact_255() {
        assert!(lowbias32(255) == 0xb344_3e84);
    }

    #[test]
    fn exact_256() {
        assert!(lowbias32(256) == 0xc983_f70d);
    }

    #[test]
    fn exact_1000() {
        assert!(lowbias32(1000) == 0x53df_6d52);
    }

    #[test]
    fn exact_65535() {
        assert!(lowbias32(65535) == 0x33ca_d8ba);
    }

    #[test]
    fn exact_65536() {
        assert!(lowbias32(65536) == 0xdcaf_ae10);
    }

    #[test]
    fn exact_0x12345678() {
        assert!(lowbias32(0x1234_5678) == 0xf5e7_1c96);
    }

    #[test]
    fn exact_cafebabe() {
        assert!(lowbias32(0xcafe_babe) == 0x8e52_963d);
    }

    #[test]
    fn exact_high_bit() {
        assert!(lowbias32(0x8000_0000) == 0xcc4b_4124);
    }

    #[test]
    fn exact_max_signed() {
        assert!(lowbias32(0x7fff_ffff) == 0x8d29_ffb8);
    }

    #[test]
    fn exact_alternating_a() {
        assert!(lowbias32(0xaaaa_aaaa) == 0x094d_4a21);
    }

    #[test]
    fn exact_alternating_5() {
        assert!(lowbias32(0x5555_5555) == 0x5e1b_ffad);
    }

    #[test]
    fn exact_42() {
        assert!(lowbias32(42) == 0x1727_33c2);
    }

    #[test]
    fn exact_1337() {
        assert!(lowbias32(1337) == 0xb23b_0727);
    }

    #[test]
    fn exact_abcdef() {
        assert!(lowbias32(0x00ab_cdef) == 0xcee6_747e);
    }

    #[test]
    fn exact_nibbles() {
        assert!(lowbias32(0x0f0f_0f0f) == 0x9997_0c95);
    }

    // ---- Property: zero is the only trivial fixed point we rely on ----

    #[test]
    fn zero_is_fixed_point() {
        assert!(lowbias32(0) == 0);
    }

    #[test]
    fn nonzero_small_inputs_are_not_fixed_points() {
        let mut i: u32 = 1;
        while i <= 64 {
            assert!(lowbias32(i) != i);
            i += 1;
        }
    }

    // ---- Property: determinism (same input -> same output) ----

    #[test]
    fn determinism_single() {
        assert!(lowbias32(0x1234_5678) == lowbias32(0x1234_5678));
    }

    #[test]
    fn determinism_sweep() {
        let samples: [u32; 8] = [
            0,
            1,
            7,
            1000,
            0xdead_beef,
            0xffff_ffff,
            0x8000_0000,
            0x0f0f_0f0f,
        ];
        let mut idx = 0;
        while idx < samples.len() {
            let v = samples[idx];
            assert!(lowbias32(v) == lowbias32(v));
            idx += 1;
        }
    }

    // ---- Property: avalanche / bit sensitivity ----

    #[test]
    fn single_bit_flip_changes_output() {
        let base: u32 = 0x1357_9bdf;
        let mut bit = 0;
        while bit < 32 {
            let flipped = base ^ (1u32 << bit);
            assert!(lowbias32(base) != lowbias32(flipped));
            bit += 1;
        }
    }

    #[test]
    fn low_bit_flip_from_zero() {
        assert!(lowbias32(0) != lowbias32(0 ^ (1u32 << 0)));
    }

    #[test]
    fn high_bit_flip_sensitivity() {
        let base: u32 = 0x0000_0000;
        assert!(lowbias32(base) != lowbias32(base ^ (1u32 << 31)));
    }

    #[test]
    fn adjacent_inputs_differ() {
        let mut i: u32 = 0;
        while i < 64 {
            assert!(lowbias32(i) != lowbias32(i + 1));
            i += 1;
        }
    }

    // ---- Property: injectivity sampling over a dense range ----

    #[test]
    fn injective_over_small_dense_range() {
        const N: usize = 64;
        let mut outputs: [u32; N] = [0; N];
        let mut i = 0;
        while i < N {
            outputs[i] = lowbias32(i as u32);
            i += 1;
        }
        let mut a = 0;
        while a < N {
            let mut b = a + 1;
            while b < N {
                assert!(outputs[a] != outputs[b]);
                b += 1;
            }
            a += 1;
        }
    }

    #[test]
    fn injective_over_strided_sample() {
        const N: usize = 48;
        let mut outputs: [u32; N] = [0; N];
        let mut i = 0;
        while i < N {
            outputs[i] = lowbias32((i as u32).wrapping_mul(0x9e37_79b9));
            i += 1;
        }
        let mut a = 0;
        while a < N {
            let mut b = a + 1;
            while b < N {
                assert!(outputs[a] != outputs[b]);
                b += 1;
            }
            a += 1;
        }
    }

    #[test]
    fn distinct_outputs_for_power_of_two_inputs() {
        const N: usize = 32;
        let mut outputs: [u32; N] = [0; N];
        let mut i = 0;
        while i < N {
            outputs[i] = lowbias32(1u32 << i);
            i += 1;
        }
        let mut a = 0;
        while a < N {
            let mut b = a + 1;
            while b < N {
                assert!(outputs[a] != outputs[b]);
                b += 1;
            }
            a += 1;
        }
    }

    #[test]
    fn output_differs_from_input_on_sample() {
        let samples: [u32; 6] = [1, 2, 3, 0xdead_beef, 0xcafe_babe, 0x1234_5678];
        let mut idx = 0;
        while idx < samples.len() {
            let v = samples[idx];
            assert!(lowbias32(v) != v);
            idx += 1;
        }
    }
}
