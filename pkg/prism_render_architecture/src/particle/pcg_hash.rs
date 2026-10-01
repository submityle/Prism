//! `PCG` (`Permuted` `Congruential` `Generator`) *stateless* integer mixing and
//! output-permutation hashes for `GPU`-style procedural generation (design §
//! procedural-noise / per-particle seeding helpers).
//!
//! # What this module is (and is not)
//!
//! This module implements the `PCG` family of hashes as pure `hash(index) ->
//! value` functions: you hand in one (or a small tuple of) integer seed/index
//! value and immediately get back a well-distributed integer. The functions
//! here are **stateless** — they hold no accumulator and advance no stream — so
//! the same inputs always yield the same outputs and callers may evaluate them
//! independently and in parallel (one invocation per particle, per grid cell,
//! per pixel) without any shared mutable state.
//!
//! That scope deliberately keeps this module disjoint from its neighbours:
//!
//! * `fnv1a_hash` and `murmur3_hash` are *byte-stream* hashes: they fold an
//!   arbitrary-length `&[u8]` into a digest. They answer "hash these bytes".
//!   This module never consumes a byte slice; it mixes fixed-width integers.
//! * `xorshift_rng` is a *stateful sequence* `RNG` (`splitmix64` seeding plus an
//!   advancing `xorshift` state machine): you create it, then repeatedly call
//!   `next` to walk a stream. This module has no advancing state and produces no
//!   sequence; every call stands alone.
//! * `spatial_hash` buckets positions into grid cells for neighbour queries;
//!   `alpha_hashed` dithers coverage. Neither is a general integer mixer.
//!
//! In short: byte-stream hashing lives in `fnv1a_hash` / `murmur3_hash`,
//! stream `RNG` lives in `xorshift_rng`, and *stateless integer output
//! permutation* lives here.
//!
//! # Algorithms
//!
//! * [`pcg_xsh_rr_u64_to_u32`] is the classic `PCG` `XSH-RR` ("xorshift high,
//!   random rotate") output permutation that turns a `64`-bit `LCG` state into a
//!   `32`-bit result. The high bits of the state pick a rotation amount, which
//!   is applied to an xorshifted fold of the state; data-dependent rotation is
//!   what gives `PCG` its statistical quality (`O'Neill`, "`PCG`: A Family of
//!   Simple Fast Space-Efficient Statistically Good Algorithms for Random
//!   Number Generation", 2014).
//! * [`pcg_hash_u32`], [`pcg2d`] and [`pcg3d`] are the single- and
//!   multi-dimensional `GPU` hashes from `Jarzynski` & `Olano`, "Hash Functions
//!   for `GPU` Rendering" (`JCGT` 2020). Each applies an `LCG` step, cross-mixes
//!   the lanes, folds high bits into low with an xorshift, and mixes again.
//! * [`pcg_advance_u64`] is a single `64`-bit `LCG` step (the state-transition
//!   half of `PCG`), exposed on its own for callers who want to drive a stepping
//!   loop and then apply an output permutation themselves.
//!
//! All integer arithmetic uses wrapping (modulo `2^width`) multiplication and
//! addition, rotation, shifts and exclusive-or; there are no transcendental
//! functions and no tables. The only floating-point surface is
//! [`pcg_f32_unit`], which maps a hash to the half-open unit interval with a
//! single integer-to-`f32` multiply (keeping the `24`-bit `MSB`s, discarding the
//! low `8` bits) — no transcendental math and no exact float equality.
//!
//! Scope: these are fast, non-cryptographic hashes for procedural generation and
//! seeding. They are not collision resistant against an adversary and must never
//! be used to authenticate data; pick a real cryptographic hash for that.

/// The `64`-bit `LCG` multiplier used by `PCG` (`O'Neill`'s canonical constant).
pub const PCG_LCG_MULTIPLIER_U64: u64 = 6_364_136_223_846_793_005;

/// The `LCG` multiplier used by the `Jarzynski` & `Olano` `GPU` hashes.
pub const PCG_GPU_LCG_MULTIPLIER_U32: u32 = 1_664_525;

/// The `LCG` increment used by the `Jarzynski` & `Olano` `GPU` hashes.
pub const PCG_GPU_LCG_INCREMENT_U32: u32 = 1_013_904_223;

/// The step multiplier of the single-input `GPU` `PCG` hash (`747796405`).
pub const PCG_HASH_MULTIPLIER_U32: u32 = 747_796_405;

/// The step increment of the single-input `GPU` `PCG` hash (`2891336453`).
pub const PCG_HASH_INCREMENT_U32: u32 = 2_891_336_453;

/// The final mixing multiplier of the single-input `GPU` `PCG` hash.
pub const PCG_HASH_MIX_MULTIPLIER_U32: u32 = 277_803_737;

/// Applies the classic `PCG` `XSH-RR` output permutation to a `64`-bit state,
/// yielding a `32`-bit result.
///
/// The permutation xorshifts the high bits of `state` down into a `32`-bit
/// word, then rotates that word right by an amount drawn from the top `5` bits
/// of `state` (the data-dependent rotation that gives `PCG` its quality):
///
/// ```text
/// xorshifted = (((state >> 18) ^ state) >> 27) as u32
/// rot        = (state >> 59) as u32
/// result     = xorshifted.rotate_right(rot)
/// ```
///
/// This is a pure output function: it does *not* advance the state. Pair it with
/// [`pcg_advance_u64`] to build a stepping generator.
#[must_use]
pub const fn pcg_xsh_rr_u64_to_u32(state: u64) -> u32 {
    let xorshifted = (((state >> 18) ^ state) >> 27) as u32;
    let rot = (state >> 59) as u32;
    xorshifted.rotate_right(rot)
}

/// Hashes a single `u32` with the `GPU` `PCG` hash of `Jarzynski` & `Olano`.
///
/// This is the stateless workhorse for per-index seeding: one `LCG` step, an
/// xorshift whose shift distance is itself derived from the high bits, a mixing
/// multiply, and a final xorshift fold.
///
/// ```text
/// state = input * 747796405 + 2891336453
/// word  = (state >> ((state >> 28) + 4)) ^ state
/// word  = word * 277803737
/// result = (word >> 22) ^ word
/// ```
#[must_use]
pub const fn pcg_hash_u32(input: u32) -> u32 {
    let state = input
        .wrapping_mul(PCG_HASH_MULTIPLIER_U32)
        .wrapping_add(PCG_HASH_INCREMENT_U32);
    let shift = (state >> 28).wrapping_add(4);
    let word = (state >> shift) ^ state;
    let word = word.wrapping_mul(PCG_HASH_MIX_MULTIPLIER_U32);
    (word >> 22) ^ word
}

/// Hashes a two-dimensional integer coordinate with the `PCG2D` vector hash
/// (`Jarzynski` & `Olano`).
///
/// Each lane takes an `LCG` step; the lanes then cross-mix twice, with an
/// xorshift fold of high bits into low between the two mixing rounds.
#[must_use]
pub const fn pcg2d(x: u32, y: u32) -> [u32; 2] {
    let mut vx = x
        .wrapping_mul(PCG_GPU_LCG_MULTIPLIER_U32)
        .wrapping_add(PCG_GPU_LCG_INCREMENT_U32);
    let mut vy = y
        .wrapping_mul(PCG_GPU_LCG_MULTIPLIER_U32)
        .wrapping_add(PCG_GPU_LCG_INCREMENT_U32);

    vx = vx.wrapping_add(vy.wrapping_mul(PCG_GPU_LCG_MULTIPLIER_U32));
    vy = vy.wrapping_add(vx.wrapping_mul(PCG_GPU_LCG_MULTIPLIER_U32));

    vx ^= vx >> 16;
    vy ^= vy >> 16;

    vx = vx.wrapping_add(vy.wrapping_mul(PCG_GPU_LCG_MULTIPLIER_U32));
    vy = vy.wrapping_add(vx.wrapping_mul(PCG_GPU_LCG_MULTIPLIER_U32));

    vx ^= vx >> 16;
    vy ^= vy >> 16;

    [vx, vy]
}

/// Hashes a three-dimensional integer coordinate with the `PCG3D` vector hash
/// (`Jarzynski` & `Olano`).
///
/// Each lane takes an `LCG` step; the three lanes then cross-mix (each lane
/// gains the product of the other two) twice, with an xorshift fold between the
/// two mixing rounds.
#[must_use]
pub const fn pcg3d(x: u32, y: u32, z: u32) -> [u32; 3] {
    let mut vx = x
        .wrapping_mul(PCG_GPU_LCG_MULTIPLIER_U32)
        .wrapping_add(PCG_GPU_LCG_INCREMENT_U32);
    let mut vy = y
        .wrapping_mul(PCG_GPU_LCG_MULTIPLIER_U32)
        .wrapping_add(PCG_GPU_LCG_INCREMENT_U32);
    let mut vz = z
        .wrapping_mul(PCG_GPU_LCG_MULTIPLIER_U32)
        .wrapping_add(PCG_GPU_LCG_INCREMENT_U32);

    vx = vx.wrapping_add(vy.wrapping_mul(vz));
    vy = vy.wrapping_add(vz.wrapping_mul(vx));
    vz = vz.wrapping_add(vx.wrapping_mul(vy));

    vx ^= vx >> 16;
    vy ^= vy >> 16;
    vz ^= vz >> 16;

    vx = vx.wrapping_add(vy.wrapping_mul(vz));
    vy = vy.wrapping_add(vz.wrapping_mul(vx));
    vz = vz.wrapping_add(vx.wrapping_mul(vy));

    [vx, vy, vz]
}

/// Advances a `64`-bit `PCG` `LCG` state by one step.
///
/// This is the state-transition half of `PCG`: `state * multiplier + (seed |
/// 1)`. The `seed | 1` forces an odd increment, which the `LCG` requires for a
/// full period. This is *only* the step; apply [`pcg_xsh_rr_u64_to_u32`] to the
/// returned state to obtain a usable output.
#[must_use]
pub const fn pcg_advance_u64(state: u64, seed: u64) -> u64 {
    state
        .wrapping_mul(PCG_LCG_MULTIPLIER_U64)
        .wrapping_add(seed | 1)
}

/// Maps [`pcg_hash_u32`] of `input` into the half-open unit interval
/// `[0.0, 1.0)`.
///
/// The top `24` bits of the hash become the mantissa of an `f32`, scaled by
/// `1 / 2^24`. Keeping only `24` bits guarantees the result is representable
/// exactly and never rounds up to `1.0`, so the output always lies in
/// `[0.0, 1.0)`.
#[must_use]
pub fn pcg_f32_unit(input: u32) -> f32 {
    let h = pcg_hash_u32(input);
    (h >> 8) as f32 * (1.0 / 16_777_216.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Determinism ----------------------------------------------------

    #[test]
    fn xsh_rr_is_deterministic() {
        let s = 0x853c_49e6_748f_ea9b_u64;
        assert_eq!(pcg_xsh_rr_u64_to_u32(s), pcg_xsh_rr_u64_to_u32(s));
    }

    #[test]
    fn pcg_hash_u32_is_deterministic() {
        assert_eq!(pcg_hash_u32(123_456), pcg_hash_u32(123_456));
        assert_eq!(pcg_hash_u32(0), pcg_hash_u32(0));
    }

    #[test]
    fn pcg2d_is_deterministic() {
        assert_eq!(pcg2d(7, 42), pcg2d(7, 42));
    }

    #[test]
    fn pcg3d_is_deterministic() {
        assert_eq!(pcg3d(7, 42, 9), pcg3d(7, 42, 9));
    }

    #[test]
    fn advance_is_deterministic() {
        let st = 0x1234_5678_9abc_def0_u64;
        assert_eq!(pcg_advance_u64(st, 1), pcg_advance_u64(st, 1));
    }

    #[test]
    fn f32_unit_is_deterministic() {
        let a = pcg_f32_unit(99);
        let b = pcg_f32_unit(99);
        assert!((a - b).abs() < f32::EPSILON);
    }

    // ---- Hardcoded reference values -------------------------------------
    // Computed independently (see module doc for sources) and verified stable.

    #[test]
    fn xsh_rr_matches_reference_values() {
        assert_eq!(pcg_xsh_rr_u64_to_u32(0x0000_0000_0000_0000), 0x0000_0000);
        assert_eq!(pcg_xsh_rr_u64_to_u32(0x0000_0000_0000_0001), 0x0000_0000);
        assert_eq!(pcg_xsh_rr_u64_to_u32(0x853c_49e6_748f_ea9b), 0x152c_a78d);
        assert_eq!(pcg_xsh_rr_u64_to_u32(0xffff_ffff_ffff_ffff), 0xfff0_0001);
        assert_eq!(pcg_xsh_rr_u64_to_u32(0x1234_5678_9abc_def0), 0x51a2_97ac);
    }

    #[test]
    fn pcg_hash_u32_matches_reference_values() {
        assert_eq!(pcg_hash_u32(0), 0x07bb_2fe2);
        assert_eq!(pcg_hash_u32(1), 0xa8be_ea3c);
        assert_eq!(pcg_hash_u32(2), 0x7a7e_cc88);
        assert_eq!(pcg_hash_u32(42), 0x48f4_32ff);
        assert_eq!(pcg_hash_u32(0xffff_ffff), 0xe62a_4902);
        assert_eq!(pcg_hash_u32(12_345), 0xf45e_ad0e);
    }

    #[test]
    fn pcg2d_matches_reference_values() {
        assert_eq!(pcg2d(0, 0), [0x18e4_31a7, 0x055d_f4d1]);
        assert_eq!(pcg2d(1, 2), [0x02bb_3f0c, 0x0cc2_73a5]);
        assert_eq!(pcg2d(42, 1337), [0x3336_c200, 0xa656_100b]);
    }

    #[test]
    fn pcg3d_matches_reference_values() {
        assert_eq!(pcg3d(0, 0, 0), [0x9baf_d7c6, 0xa8e8_8a6b, 0x3f15_482c]);
        assert_eq!(pcg3d(1, 2, 3), [0xfa9f_79a6, 0x48f2_f44c, 0x596f_5ab1]);
        assert_eq!(pcg3d(10, 20, 30), [0x64c0_ad54, 0x58c9_b38e, 0x9d6b_57e8]);
    }

    #[test]
    fn advance_matches_reference_values() {
        assert_eq!(pcg_advance_u64(0, 0), 0x0000_0000_0000_0001);
        assert_eq!(
            pcg_advance_u64(0x853c_49e6_748f_ea9b, 0xda3e_39cb_94b9_5bdb),
            0xc830_15ca_079f_7e1a
        );
        assert_eq!(pcg_advance_u64(1, 2), 0x5851_f42d_4c95_7f30);
    }

    // ---- Avalanche ------------------------------------------------------

    #[test]
    fn pcg_hash_u32_single_bit_flip_changes_many_bits() {
        // For every input, the mean Hamming distance across all 32 single-bit
        // flips must clear a conservative avalanche floor (well above the
        // weakest individual flip, which can be as low as five bits).
        let mut i = 0u32;
        while i < 1024 {
            let base = pcg_hash_u32(i);
            let mut sum = 0u32;
            let mut bit = 0u32;
            while bit < 32 {
                let flipped = pcg_hash_u32(i ^ (1 << bit));
                let d = (base ^ flipped).count_ones();
                // No flip may be a no-op, and the mean must stay high.
                assert!(d > 0, "fixed point at input {i} bit {bit}");
                sum += d;
                bit += 1;
            }
            let mean = f64::from(sum) / 32.0;
            assert!(mean > 10.0, "weak mean avalanche {mean} at input {i}");
            i += 1;
        }
    }

    #[test]
    fn pcg_hash_u32_average_avalanche_is_near_half() {
        // Over a bit-0 flip across many inputs the mean Hamming distance should
        // sit near 16 of 32 bits.
        let mut total = 0u64;
        let mut i = 0u32;
        while i < 4096 {
            let a = pcg_hash_u32(i);
            let b = pcg_hash_u32(i ^ 1);
            total += u64::from((a ^ b).count_ones());
            i += 1;
        }
        let avg = total as f64 / 4096.0;
        assert!(
            (12.0..20.0).contains(&avg),
            "avalanche mean {avg} outside expected band"
        );
    }

    #[test]
    fn pcg2d_single_lane_flip_changes_output() {
        let base = pcg2d(100, 200);
        let flipped = pcg2d(100 ^ 1, 200);
        let bits = (base[0] ^ flipped[0]).count_ones() + (base[1] ^ flipped[1]).count_ones();
        assert!(bits > 5, "pcg2d avalanche too weak: {bits}");
    }

    #[test]
    fn pcg3d_single_lane_flip_changes_output() {
        let base = pcg3d(100, 200, 300);
        let flipped = pcg3d(100, 200, 300 ^ 1);
        let bits = (base[0] ^ flipped[0]).count_ones()
            + (base[1] ^ flipped[1]).count_ones()
            + (base[2] ^ flipped[2]).count_ones();
        assert!(bits > 5, "pcg3d avalanche too weak: {bits}");
    }

    #[test]
    fn pcg_hash_u32_never_fixed_under_bit_flip() {
        // No single-bit flip should leave the output completely unchanged.
        let mut i = 0u32;
        while i < 2048 {
            let base = pcg_hash_u32(i);
            let mut bit = 0u32;
            while bit < 32 {
                assert_ne!(base, pcg_hash_u32(i ^ (1 << bit)));
                bit += 1;
            }
            i += 1;
        }
    }

    // ---- Zero collisions ------------------------------------------------

    #[test]
    fn pcg_hash_u32_has_no_collisions_over_4096_inputs() {
        let mut outputs: Vec<u32> = (0u32..=4095).map(pcg_hash_u32).collect();
        outputs.sort_unstable();
        assert!(outputs.windows(2).all(|w| w[0] != w[1]));
    }

    #[test]
    fn pcg_hash_u32_has_no_collisions_over_high_block() {
        let mut outputs: Vec<u32> = (1_000_000u32..1_004_096).map(pcg_hash_u32).collect();
        outputs.sort_unstable();
        assert!(outputs.windows(2).all(|w| w[0] != w[1]));
    }

    #[test]
    fn pcg2d_has_no_collisions_over_grid() {
        let mut outputs: Vec<[u32; 2]> = Vec::new();
        let mut x = 0u32;
        while x < 64 {
            let mut y = 0u32;
            while y < 64 {
                outputs.push(pcg2d(x, y));
                y += 1;
            }
            x += 1;
        }
        outputs.sort_unstable();
        assert!(outputs.windows(2).all(|w| w[0] != w[1]));
    }

    #[test]
    fn pcg3d_has_no_collisions_over_grid() {
        let mut outputs: Vec<[u32; 3]> = Vec::new();
        let mut x = 0u32;
        while x < 16 {
            let mut y = 0u32;
            while y < 16 {
                let mut z = 0u32;
                while z < 16 {
                    outputs.push(pcg3d(x, y, z));
                    z += 1;
                }
                y += 1;
            }
            x += 1;
        }
        outputs.sort_unstable();
        assert!(outputs.windows(2).all(|w| w[0] != w[1]));
    }

    #[test]
    fn xsh_rr_has_no_collisions_over_sequential_states() {
        // Walk the LCG and permute each state; outputs should be distinct over
        // this short window.
        let mut outputs: Vec<u32> = Vec::new();
        let mut state = 0x4d59_5df4_d0f3_3173_u64;
        let mut n = 0u32;
        while n < 2048 {
            outputs.push(pcg_xsh_rr_u64_to_u32(state));
            state = pcg_advance_u64(state, 0xda3e_39cb_94b9_5bdb);
            n += 1;
        }
        outputs.sort_unstable();
        assert!(outputs.windows(2).all(|w| w[0] != w[1]));
    }

    // ---- Unit-interval mapping ------------------------------------------

    #[test]
    fn f32_unit_stays_in_half_open_unit_range() {
        let mut i = 0u32;
        while i < 50_000 {
            let v = pcg_f32_unit(i);
            assert!((0.0..1.0).contains(&v), "value {v} out of range at {i}");
            i += 1;
        }
    }

    #[test]
    fn f32_unit_covers_low_and_high_bands() {
        // Across many inputs we should see both small and large fractions.
        let mut saw_low = false;
        let mut saw_high = false;
        let mut i = 0u32;
        while i < 10_000 {
            let v = pcg_f32_unit(i);
            if (0.0..0.1).contains(&v) {
                saw_low = true;
            }
            if (0.9..1.0).contains(&v) {
                saw_high = true;
            }
            i += 1;
        }
        assert!(saw_low && saw_high);
    }

    // ---- LCG stepping / bulk statistics ---------------------------------

    #[test]
    fn advance_odd_increment_breaks_zero_fixed_point() {
        // seed | 1 keeps the increment odd, so a zero state does not stay zero.
        assert_ne!(pcg_advance_u64(0, 0), 0);
        assert_ne!(pcg_advance_u64(0, 2), 0);
    }

    #[test]
    fn advance_then_permute_runs_without_panic() {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut acc = 0u32;
        let mut n = 0u32;
        while n < 100_000 {
            state = pcg_advance_u64(state, 0x1234_5678_9abc_def1);
            acc ^= pcg_xsh_rr_u64_to_u32(state);
            n += 1;
        }
        // acc is data-dependent; the point is the loop completes.
        let _ = acc;
    }

    #[test]
    fn advance_produces_distinct_consecutive_states() {
        let mut state = 42u64;
        let mut n = 0u32;
        while n < 10_000 {
            let next = pcg_advance_u64(state, 1);
            assert_ne!(state, next);
            state = next;
            n += 1;
        }
    }

    #[test]
    fn pcg_hash_u32_output_bits_are_balanced() {
        // Each of the 32 output positions should be set on a healthy fraction
        // of a large input sweep (never stuck at all-zero or all-one).
        let mut ones = [0u32; 32];
        let mut i = 0u32;
        while i < 8192 {
            let h = pcg_hash_u32(i);
            let mut bit = 0u32;
            while bit < 32 {
                if (h >> bit) & 1 == 1 {
                    ones[bit as usize] += 1;
                }
                bit += 1;
            }
            i += 1;
        }
        assert!(ones.iter().all(|&c| (1024..7168).contains(&c)));
    }

    #[test]
    fn pcg2d_lanes_are_not_identical() {
        // The two lanes mix differently, so they should not be equal in bulk.
        let mut equal = 0u32;
        let mut x = 0u32;
        while x < 256 {
            let v = pcg2d(x, x);
            if v[0] == v[1] {
                equal += 1;
            }
            x += 1;
        }
        assert!(equal < 8, "lanes collapse too often: {equal}");
    }

    #[test]
    fn pcg3d_lanes_are_not_identical() {
        let mut equal = 0u32;
        let mut x = 0u32;
        while x < 256 {
            let v = pcg3d(x, x, x);
            if v[0] == v[1] && v[1] == v[2] {
                equal += 1;
            }
            x += 1;
        }
        assert!(equal < 4, "lanes collapse too often: {equal}");
    }

    #[test]
    fn constants_match_documented_values() {
        assert_eq!(PCG_LCG_MULTIPLIER_U64, 6_364_136_223_846_793_005);
        assert_eq!(PCG_GPU_LCG_MULTIPLIER_U32, 1_664_525);
        assert_eq!(PCG_GPU_LCG_INCREMENT_U32, 1_013_904_223);
        assert_eq!(PCG_HASH_MULTIPLIER_U32, 747_796_405);
        assert_eq!(PCG_HASH_INCREMENT_U32, 2_891_336_453);
        assert_eq!(PCG_HASH_MIX_MULTIPLIER_U32, 277_803_737);
    }

    #[test]
    fn const_fn_usable_in_const_context() {
        const A: u32 = pcg_hash_u32(7);
        const B: [u32; 2] = pcg2d(3, 5);
        const C: [u32; 3] = pcg3d(3, 5, 7);
        const D: u32 = pcg_xsh_rr_u64_to_u32(0xdead_beef_cafe_f00d);
        const E: u64 = pcg_advance_u64(1, 1);
        assert_eq!(A, pcg_hash_u32(7));
        assert_eq!(B, pcg2d(3, 5));
        assert_eq!(C, pcg3d(3, 5, 7));
        assert_eq!(D, pcg_xsh_rr_u64_to_u32(0xdead_beef_cafe_f00d));
        assert_eq!(E, pcg_advance_u64(1, 1));
    }

    #[test]
    fn f32_unit_matches_manual_mapping() {
        let i = 777u32;
        let expected = (pcg_hash_u32(i) >> 8) as f32 * (1.0 / 16_777_216.0);
        assert!((pcg_f32_unit(i) - expected).abs() < f32::EPSILON);
    }
}
