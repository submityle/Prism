//! Weighted reservoir sampling (`WRS`) and the `ReSTIR` reservoir primitive for
//! the particle engine's stochastic light / candidate resampling passes.
//!
//! A *reservoir* keeps a single running sample chosen from a stream of weighted
//! candidates without ever storing the stream: each candidate is accepted into
//! the reservoir with probability `weight / w_sum`, so after the whole stream
//! has been observed the held sample is distributed proportionally to its
//! weight. This is the classic one-pass weighted reservoir algorithm; `ReSTIR`
//! layers two extensions on top of it — a `merge` that fuses two independent
//! reservoirs while staying unbiased (spatial / temporal reuse), and an
//! unbiased contribution weight `W = w_sum / (M * target_pdf)` that rescales the
//! held sample so a Monte-Carlo estimator using it stays unbiased.
//!
//! * [`Reservoir::update`] folds one weighted candidate into the stream,
//!   returning whether it replaced the held sample.
//! * [`Reservoir::merge`] combines another reservoir as if its aggregate
//!   `w_sum` were a single super-candidate, summing the sample counts `M`.
//! * [`Reservoir::finalize_w`] computes the unbiased contribution weight `W`
//!   from the target `pdf`, clamping to zero when the `pdf` vanishes.
//!
//! All acceptance decisions consume a caller-supplied uniform random number in
//! `[0, 1)`; when a caller has no source of its own, the tiny [`Rng`] here is a
//! self-contained `splitmix32` integer hash. This module deliberately does not
//! import [`super::determinism`] (the stateless `RngKey` / `StreamId` hash `RNG`
//! infrastructure) nor [`super::halton_sequence`] (low-discrepancy sequences):
//! it owns only the reservoir arithmetic, keeping the `CPU` reference free of
//! external state so a future `GPU` kernel can match it bit for bit. No
//! transcendental functions appear — only plain multiply / divide comparisons
//! and integer hashing.

/// Below this the target `pdf` is treated as zero, so the unbiased contribution
/// weight `W` is forced to zero rather than dividing by a vanishing density.
const FINALIZE_PDF_EPS: f32 = 1e-6;

/// `2^24`, the reciprocal used to normalize a 24-bit mantissa word into the
/// unit interval; `2^24` is represented exactly in an `f32` mantissa.
const INV_2_POW_24: f32 = 1.0 / 16_777_216.0;

/// Widens a sample count to `f32` for the unbiased `W` ratio.
#[expect(
    clippy::cast_precision_loss,
    reason = "sample counts stay well within the f32 exact-integer range for particle streams"
)]
fn count_to_f32(m: u32) -> f32 {
    m as f32
}

/// Widens a 24-bit word to `f32`; the value is below `2^24` so it is exact.
#[expect(
    clippy::cast_precision_loss,
    reason = "a 24-bit word is represented exactly in the f32 mantissa"
)]
fn u24_to_f32(bits: u32) -> f32 {
    bits as f32
}

/// A weighted reservoir holding a single running sample drawn from a stream of
/// weighted candidates (`WRS` / `ReSTIR`).
///
/// The fields are public so `GPU`-facing packing code can read them directly.
/// `sample` is meaningful only once `m > 0`; an empty reservoir reports a zero
/// sample and zero weights.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Reservoir {
    /// The identifier of the currently held candidate (e.g. a light index).
    pub sample: u32,
    /// The running sum of all candidate weights observed so far.
    pub w_sum: f32,
    /// The number of candidates folded into this reservoir.
    pub m: u32,
    /// The unbiased contribution weight, set by [`Reservoir::finalize_w`].
    pub w: f32,
}

impl Reservoir {
    /// Returns an empty reservoir: no sample, zero weight sum, zero count.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            sample: 0,
            w_sum: 0.0,
            m: 0,
            w: 0.0,
        }
    }

    /// Folds one weighted candidate into the stream.
    ///
    /// Adds `weight` to `w_sum`, increments the count `M`, then replaces the
    /// held sample with `candidate` with probability `weight / w_sum` using the
    /// caller-supplied uniform `rand_u01` in `[0, 1)`. Returns `true` when the
    /// held sample was replaced. The comparison is done as `rand_u01 * w_sum <
    /// weight` so no division is needed and a zero `w_sum` never replaces.
    pub fn update(&mut self, candidate: u32, weight: f32, rand_u01: f32) -> bool {
        self.w_sum += weight;
        self.m += 1;
        let replace = rand_u01 * self.w_sum < weight;
        if replace {
            self.sample = candidate;
        }
        replace
    }

    /// Merges another reservoir into this one, staying unbiased.
    ///
    /// The other reservoir is treated as a single super-candidate whose weight
    /// is its aggregate `w_sum`: this reservoir adopts the other's sample with
    /// probability `other.w_sum / (self.w_sum + other.w_sum)`, and the counts
    /// `M` are summed so the combined estimator remains unbiased.
    pub fn merge(&mut self, other: &Reservoir, rand_u01: f32) {
        let combined_m = self.m + other.m;
        self.w_sum += other.w_sum;
        if rand_u01 * self.w_sum < other.w_sum {
            self.sample = other.sample;
        }
        self.m = combined_m;
    }

    /// Computes the unbiased contribution weight `W = w_sum / (M * target_pdf)`.
    ///
    /// When `target_pdf` is at or below [`FINALIZE_PDF_EPS`] (or the reservoir
    /// is empty) the weight is forced to zero, guarding against a division by a
    /// vanishing density.
    pub fn finalize_w(&mut self, target_pdf: f32) {
        if target_pdf <= FINALIZE_PDF_EPS || self.m == 0 {
            self.w = 0.0;
            return;
        }
        let denom = count_to_f32(self.m) * target_pdf;
        self.w = self.w_sum / denom;
    }
}

/// A minimal self-contained `splitmix32` integer hash `RNG`.
///
/// This is intentionally tiny and stateless-by-seed so a caller with no random
/// source of its own can drive [`Reservoir::update`] / [`Reservoir::merge`]. It
/// is *not* the engine's shared determinism `RNG`; it exists only so this
/// module is self-contained.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rng {
    state: u32,
}

impl Rng {
    /// Creates a generator seeded with `seed`.
    #[must_use]
    pub const fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    /// Advances the generator and returns the next 32-bit hash word.
    #[must_use]
    pub fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_add(0x9E37_79B9);
        let mut z = self.state;
        z = (z ^ (z >> 16)).wrapping_mul(0x21F0_AAAD);
        z = (z ^ (z >> 15)).wrapping_mul(0x735A_2D97);
        z ^ (z >> 15)
    }

    /// Returns the next uniform sample in `[0, 1)` using the top 24 bits.
    #[must_use]
    pub fn next_u01(&mut self) -> f32 {
        let bits = self.next_u32() >> 8;
        u24_to_f32(bits) * INV_2_POW_24
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CMP_EPS: f32 = 1e-6;

    fn approx(a: f32, b: f32) -> bool {
        let d = a - b;
        let d = if d < 0.0 { -d } else { d };
        d <= CMP_EPS
    }

    #[test]
    fn empty_reservoir_is_zeroed() {
        let r = Reservoir::empty();
        assert_eq!(r.sample, 0);
        assert!(approx(r.w_sum, 0.0));
        assert_eq!(r.m, 0);
        assert!(approx(r.w, 0.0));
    }

    #[test]
    fn single_candidate_is_always_selected() {
        let mut r = Reservoir::empty();
        let replaced = r.update(42, 2.5, 0.5);
        assert!(replaced);
        assert_eq!(r.sample, 42);
        assert_eq!(r.m, 1);
        assert!(approx(r.w_sum, 2.5));
    }

    #[test]
    fn single_candidate_selected_even_at_rand_zero() {
        let mut r = Reservoir::empty();
        assert!(r.update(7, 1.0, 0.0));
        assert_eq!(r.sample, 7);
    }

    #[test]
    fn update_rand_zero_always_replaces() {
        let mut r = Reservoir::empty();
        r.update(1, 1.0, 0.5);
        let replaced = r.update(2, 3.0, 0.0);
        assert!(replaced);
        assert_eq!(r.sample, 2);
    }

    #[test]
    fn update_rand_near_one_keeps_existing() {
        let mut r = Reservoir::empty();
        r.update(1, 1.0, 0.5);
        // 0.9 * (1 + 1) = 1.8, not < 1.0 -> keep the first sample.
        let replaced = r.update(2, 1.0, 0.9);
        assert!(!replaced);
        assert_eq!(r.sample, 1);
    }

    #[test]
    fn update_returns_false_when_not_replaced() {
        let mut r = Reservoir::empty();
        r.update(10, 5.0, 0.5);
        // Tiny weight against a large w_sum: 0.5 * (5 + 0.001) < 0.001 is false.
        let replaced = r.update(11, 0.001, 0.5);
        assert!(!replaced);
        assert_eq!(r.sample, 10);
    }

    #[test]
    fn w_sum_accumulates_across_updates() {
        let mut r = Reservoir::empty();
        r.update(1, 1.5, 0.5);
        r.update(2, 2.0, 0.5);
        r.update(3, 0.5, 0.5);
        assert!(approx(r.w_sum, 4.0));
    }

    #[test]
    fn m_increments_per_update() {
        let mut r = Reservoir::empty();
        for i in 0..5 {
            r.update(i, 1.0, 0.5);
        }
        assert_eq!(r.m, 5);
    }

    #[test]
    fn zero_weight_candidate_never_replaces_but_counts() {
        let mut r = Reservoir::empty();
        r.update(1, 2.0, 0.5);
        let replaced = r.update(2, 0.0, 0.0);
        assert!(!replaced);
        assert_eq!(r.sample, 1);
        assert_eq!(r.m, 2);
        assert!(approx(r.w_sum, 2.0));
    }

    #[test]
    fn merge_count_is_sum_of_counts() {
        let a = Reservoir {
            sample: 1,
            w_sum: 2.0,
            m: 3,
            w: 0.0,
        };
        let b = Reservoir {
            sample: 2,
            w_sum: 4.0,
            m: 5,
            w: 0.0,
        };
        let mut merged = a;
        merged.merge(&b, 0.5);
        assert_eq!(merged.m, 8);
    }

    #[test]
    fn merge_w_sum_accumulates() {
        let a = Reservoir {
            sample: 1,
            w_sum: 2.0,
            m: 1,
            w: 0.0,
        };
        let b = Reservoir {
            sample: 2,
            w_sum: 4.5,
            m: 1,
            w: 0.0,
        };
        let mut merged = a;
        merged.merge(&b, 0.5);
        assert!(approx(merged.w_sum, 6.5));
    }

    #[test]
    fn merge_rand_zero_takes_other_sample() {
        let a = Reservoir {
            sample: 1,
            w_sum: 3.0,
            m: 1,
            w: 0.0,
        };
        let b = Reservoir {
            sample: 2,
            w_sum: 1.0,
            m: 1,
            w: 0.0,
        };
        let mut merged = a;
        merged.merge(&b, 0.0);
        assert_eq!(merged.sample, 2);
    }

    #[test]
    fn merge_rand_near_one_keeps_self_sample() {
        let a = Reservoir {
            sample: 1,
            w_sum: 3.0,
            m: 1,
            w: 0.0,
        };
        let b = Reservoir {
            sample: 2,
            w_sum: 1.0,
            m: 1,
            w: 0.0,
        };
        let mut merged = a;
        // 0.99 * 4 = 3.96, not < 1.0 -> keep self.
        merged.merge(&b, 0.99);
        assert_eq!(merged.sample, 1);
    }

    #[test]
    fn merge_with_empty_other_is_a_noop_on_sample() {
        let a = Reservoir {
            sample: 9,
            w_sum: 2.0,
            m: 4,
            w: 0.0,
        };
        let empty = Reservoir::empty();
        let mut merged = a;
        merged.merge(&empty, 0.0);
        assert_eq!(merged.sample, 9);
        assert_eq!(merged.m, 4);
        assert!(approx(merged.w_sum, 2.0));
    }

    #[test]
    fn finalize_w_matches_formula() {
        let mut r = Reservoir {
            sample: 1,
            w_sum: 4.0,
            m: 2,
            w: 0.0,
        };
        r.finalize_w(2.0);
        // 4 / (2 * 2) = 1.0
        assert!(approx(r.w, 1.0));
    }

    #[test]
    fn finalize_w_target_pdf_zero_yields_zero() {
        let mut r = Reservoir {
            sample: 1,
            w_sum: 4.0,
            m: 2,
            w: 9.0,
        };
        r.finalize_w(0.0);
        assert!(approx(r.w, 0.0));
    }

    #[test]
    fn finalize_w_below_eps_yields_zero() {
        let mut r = Reservoir {
            sample: 1,
            w_sum: 4.0,
            m: 2,
            w: 9.0,
        };
        r.finalize_w(FINALIZE_PDF_EPS * 0.5);
        assert!(approx(r.w, 0.0));
    }

    #[test]
    fn finalize_w_empty_reservoir_yields_zero() {
        let mut r = Reservoir::empty();
        r.finalize_w(1.0);
        assert!(approx(r.w, 0.0));
    }

    #[test]
    fn update_sequence_is_deterministic() {
        let weights = [1.0_f32, 2.0, 0.5, 3.0, 1.5];
        let rands = [0.1_f32, 0.7, 0.3, 0.05, 0.9];
        let run = || {
            let mut r = Reservoir::empty();
            for i in 0..weights.len() {
                r.update(u32::try_from(i).unwrap(), weights[i], rands[i]);
            }
            r
        };
        let a = run();
        let b = run();
        assert_eq!(a.sample, b.sample);
        assert_eq!(a.m, b.m);
        assert!(approx(a.w_sum, b.w_sum));
    }

    #[test]
    fn selection_frequency_matches_weight_ratio() {
        // Candidate 0 has weight 1, candidate 1 has weight 3, so the held
        // sample should be candidate 1 about 3/4 of the time.
        let trials: u32 = 40_000;
        let mut rng = Rng::new(0xABCD_1234);
        let mut chose_b: u32 = 0;
        for _ in 0..trials {
            let mut r = Reservoir::empty();
            r.update(0, 1.0, rng.next_u01());
            r.update(1, 3.0, rng.next_u01());
            if r.sample == 1 {
                chose_b += 1;
            }
        }
        let ratio = count_to_f32(chose_b) / count_to_f32(trials);
        assert!(ratio > 0.72 && ratio < 0.78, "ratio was {ratio}");
    }

    #[test]
    fn merge_frequency_matches_w_sum_ratio() {
        let base_a = Reservoir {
            sample: 10,
            w_sum: 1.0,
            m: 1,
            w: 0.0,
        };
        let base_b = Reservoir {
            sample: 20,
            w_sum: 3.0,
            m: 1,
            w: 0.0,
        };
        let trials: u32 = 40_000;
        let mut rng = Rng::new(0x0BAD_F00D);
        let mut chose_b: u32 = 0;
        for _ in 0..trials {
            let mut merged = base_a;
            merged.merge(&base_b, rng.next_u01());
            if merged.sample == 20 {
                chose_b += 1;
            }
        }
        let ratio = count_to_f32(chose_b) / count_to_f32(trials);
        assert!(ratio > 0.72 && ratio < 0.78, "ratio was {ratio}");
    }

    #[test]
    fn rng_is_deterministic_per_seed() {
        let mut a = Rng::new(12345);
        let mut b = Rng::new(12345);
        for _ in 0..16 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn rng_u01_stays_in_unit_interval() {
        let mut rng = Rng::new(999);
        for _ in 0..1000 {
            let x = rng.next_u01();
            assert!((0.0..1.0).contains(&x), "sample out of range: {x}");
        }
    }

    #[test]
    fn rng_produces_distinct_words() {
        let mut rng = Rng::new(1);
        let first = rng.next_u32();
        let second = rng.next_u32();
        assert_ne!(first, second);
    }
}
