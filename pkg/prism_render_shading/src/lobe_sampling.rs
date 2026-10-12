//! Deterministic stochastic single-lobe sampling for expensive shading lobes.
//!
//! A material can contain several lobes whose shade-time cost is very
//! different: diffuse/specular are cheap closed forms, while separable SSS,
//! translucency, multiple scattering, and other measured/BTF lobes may each
//! evaluate an integral or march rays.  Summing all expensive lobes at every
//! pixel therefore scales linearly with material complexity.  This module
//! implements the §17.4 alternative: keep the lobe set intact, but select one
//! expensive lobe with probability
//!
//! ```text
//! p_i = w_i / sum_j w_j
//! ```
//!
//! and return the unbiased one-sample estimator
//!
//! ```text
//! F_hat = f_i / p_i,       E[F_hat] = sum_i f_i.
//! ```
//!
//! The estimator is "降采样质量", not lobe collapse: temporal accumulation
//! converges to the same deterministic full sum while each frame evaluates one
//! lobe instead of all of them.
//!
//! # Determinism and blue-noise-like sequence
//! Selection uses the R2 low-discrepancy sequence (Roberts 2018, *The
//! Unreasonable Effectiveness of Quasirandom Sequences*) with a per-pixel
//! Cranley–Patterson rotation and a per-pixel odd temporal stride.  The R2
//! point set is highly uniform and visually blue-noise-like without requiring
//! a precomputed STBN texture; the rotation decorrelates neighbouring pixels,
//! and the odd stride makes successive frames visit the sequence rather than
//! marching in lock-step.  The sequence period is `2^16`, which is exactly
//! representable in `f32`; after that period the sequence repeats, well beyond
//! the range where TAA history has any influence.
//!
//! Every function is a pure, allocation-free function of its inputs.  The same
//! `(pixel, frame, sample, seed)` always produces the same lobe choice; no
//! hidden RNG state exists.  The WESL twin mirrors integer arithmetic, float
//! operation order, clamps, and constants so the CPU golden and GPU path can be
//! compared bit-for-bit once the rendering pass is wired.
//!
//! # Quality policy
//! [`LobeSamplingPolicy`] links §17.4 to the P1-e [`ShadingQuality`] and
//! [`LobeSampleBudget`] tiers:
//!
//! * `Low`: stochastic single-lobe once there is more than one expensive lobe.
//! * `Medium`: keep up to two expensive lobes deterministic, then stochastic.
//! * `High` / `Ultra`: full deterministic sum by default.
//! * If a P1-e budget is scaled below its canonical tier (for example by a
//!   dynamic-resolution controller), the policy becomes more aggressive and
//!   falls to a single stochastic lobe.  This makes the existing continuous
//!   `with_global_scale` knob useful for P3-e without a new specialization key.
//!
//! Cheap diffuse/specular lobes are not part of this module: callers add their
//! deterministic sum to [`lobe_sample_quality_sum`].
//!
//! # Safety invariants
//! * Non-finite, negative, or overflowing weights are treated as zero.
//! * If every supplied weight is unusable, probabilities fall back to uniform
//!   over the supplied lobes, preserving unbiasedness instead of dividing by
//!   zero.
//! * Empty inputs return an inert [`LobeSample`] and estimate zero.
//! * Every returned estimate is finite; adversarial inputs can reduce quality
//!   but cannot produce `NaN` or `Inf`.

use bevy_math::ops;

use crate::quality::{LobeSampleBudget, ShadingQuality};

/// Salt mixed into every pixel seed so an all-zero pixel coordinate does not
/// correlate with an all-zero hash stream.
pub const LOBE_SAMPLING_SEED_SALT: u32 = 0x53a7_1e5d;

/// Golden-ratio odd stride used to advance a pixel's R2 sequence per frame.
pub const LOBE_SAMPLING_GOLDEN_STRIDE: u32 = 0x9e37_79b9;

/// Salt used to derive a per-pixel odd temporal stride.
pub const LOBE_SAMPLING_TEMPORAL_SALT: u32 = 0x51ed_270b;

/// R2 sequence period.  `2^16` is exact in `f32` and gives more than enough
/// temporal states for TAA history.
pub const LOBE_SAMPLING_SEQUENCE_PERIOD: u32 = 1 << 16;

/// Mask selecting the R2 sequence index.
pub const LOBE_SAMPLING_SEQUENCE_MASK: u32 = LOBE_SAMPLING_SEQUENCE_PERIOD - 1;

/// First R2 additive stride `1/p`, where `p` is the plastic constant.
pub const LOBE_SAMPLING_R2_A1: f32 = 0.754_877_7;

/// Second R2 additive stride `1/p^2`.
pub const LOBE_SAMPLING_R2_A2: f32 = 0.569_840_3;

/// Smallest weight treated as a usable probability mass.
pub const LOBE_SAMPLING_MIN_WEIGHT: f32 = f32::MIN_POSITIVE;

/// Largest generated unit value, `1 - 2^-24`.  Keeping the target below one
/// guarantees the CDF walk selects a lobe even when accumulated weights round
/// to the total.
pub const LOBE_SAMPLING_ONE_MINUS_EPSILON: f32 = 0.999_999_94;

/// Sentinel index returned when no lobe can be selected.
pub const LOBE_SAMPLING_NO_LOBE: u32 = u32::MAX;

/// Whether an expensive-lobe group is evaluated in full or by one stochastic
/// representative this frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LobeSamplingMode {
    /// Deterministically sum every supplied expensive lobe.
    Full,
    /// Select one lobe with weighted probability and use the `1 / p` estimator.
    Stochastic,
}

/// Quality-driven plan for P3-e stochastic single-lobe sampling.
///
/// The plan carries the exact P1-e budget snapshot it was derived from, so
/// telemetry can distinguish "full quality" from "full because a budget was
/// scaled below its canonical tier" without re-querying global state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LobeSamplingPolicy {
    quality: ShadingQuality,
    budget: LobeSampleBudget,
    mode: LobeSamplingMode,
    max_full_lobes: u32,
    temporal_accumulation_frames: u32,
}

impl LobeSamplingPolicy {
    /// Builds the canonical policy for a quality tier using its canonical
    /// P1-e budget.
    #[inline]
    pub const fn for_quality(quality: ShadingQuality) -> Self {
        Self::for_budget(quality, quality.budget())
    }

    /// Builds a policy from an explicit P1-e budget snapshot.
    ///
    /// Unscaled `Low`/`Medium` deliberately keep the tier's small deterministic
    /// prefix.  Unscaled `High`/`Ultra` always evaluate the full lobe set.  A
    /// budget below its canonical [`ShadingQuality::budget`] total means the
    /// host has shed samples (for example via
    /// [`LobeSampleBudget::with_global_scale`]); in that case even High/Ultra
    /// fall to a single stochastic lobe.
    #[inline]
    pub const fn for_budget(quality: ShadingQuality, budget: LobeSampleBudget) -> Self {
        let budget = budget.clamped();
        let canonical = quality.budget();
        let downscaled = budget.total_samples() < canonical.total_samples();

        let base_max_full_lobes = match quality {
            ShadingQuality::Low => 1,
            ShadingQuality::Medium => 2,
            ShadingQuality::High | ShadingQuality::Ultra => LOBE_SAMPLING_NO_LOBE,
        };
        let max_full_lobes = if downscaled {
            1
        } else {
            base_max_full_lobes
        };

        let temporal_accumulation_frames = if downscaled {
            8
        } else {
            match quality {
                ShadingQuality::Low => 8,
                ShadingQuality::Medium => 6,
                ShadingQuality::High | ShadingQuality::Ultra => 1,
            }
        };
        let mode = if max_full_lobes == LOBE_SAMPLING_NO_LOBE {
            LobeSamplingMode::Full
        } else {
            LobeSamplingMode::Stochastic
        };

        Self {
            quality,
            budget,
            mode,
            max_full_lobes,
            temporal_accumulation_frames,
        }
    }

    /// Quality tier this policy was derived from.
    #[inline]
    pub const fn quality(self) -> ShadingQuality {
        self.quality
    }

    /// P1-e budget snapshot used by this policy.
    #[inline]
    pub const fn budget(self) -> LobeSampleBudget {
        self.budget
    }

    /// Whether the group is evaluated in full or stochastically.
    #[inline]
    pub const fn mode(self) -> LobeSamplingMode {
        self.mode
    }

    /// Maximum number of expensive lobes that can be summed deterministically
    /// before stochastic sampling starts.  [`LOBE_SAMPLING_NO_LOBE`] means no
    /// limit (full quality).
    #[inline]
    pub const fn max_full_lobes(self) -> u32 {
        self.max_full_lobes
    }

    /// Minimum number of TAA frames recommended to attenuate the one-sample
    /// noise to the level of the corresponding quality tier.  This is a
    /// convergence guide, not a hard history clamp.
    #[inline]
    pub const fn temporal_accumulation_frames(self) -> u32 {
        self.temporal_accumulation_frames
    }

    /// Whether `lobe_count` exceeds the deterministic prefix and must use
    /// stochastic single-lobe sampling.
    #[inline]
    pub const fn should_use_stochastic(self, lobe_count: usize) -> bool {
        lobe_count > 0 && (lobe_count as u64) > self.max_full_lobes as u64
    }

    /// Number of expensive-lobe evaluations per pixel this frame: the full
    /// count below the deterministic prefix, otherwise one.
    #[inline]
    pub const fn evaluations_per_pixel(self, lobe_count: usize) -> usize {
        if lobe_count == 0 {
            0
        } else if self.should_use_stochastic(lobe_count) {
            1
        } else {
            lobe_count
        }
    }
}

/// One stochastic lobe selection and its unbiased estimator factor.
///
/// `index` is [`LOBE_SAMPLING_NO_LOBE`] when the input has no selectable lobe.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LobeSample {
    /// Index into the caller's expensive-lobe arrays.
    pub index: u32,
    /// Probability with which this lobe was selected, in `(0, 1]`.
    pub probability: f32,
    /// Unbiased estimator factor `1 / probability`.
    pub estimator_weight: f32,
}

impl LobeSample {
    /// Inert sample used for empty input and deterministic full-sum fallbacks.
    pub const NONE: Self = Self {
        index: LOBE_SAMPLING_NO_LOBE,
        probability: 0.0,
        estimator_weight: 0.0,
    };

    /// Whether this sample refers to a real lobe.
    #[inline]
    pub fn is_valid(self) -> bool {
        self.index != LOBE_SAMPLING_NO_LOBE
            && lobe_is_finite(self.probability)
            && self.probability > 0.0
            && lobe_is_finite(self.estimator_weight)
            && self.estimator_weight > 0.0
    }

    /// Applies the unbiased `1 / p` estimator to one scalar lobe contribution.
    ///
    /// Non-finite contributions or an overflowing product collapse to zero, so
    /// an adversarial material cannot inject `NaN` into the TAA history.
    #[inline]
    pub fn estimate(self, contribution: f32) -> f32 {
        if !self.is_valid() || !lobe_is_finite(contribution) {
            return 0.0;
        }
        let estimate = contribution * self.estimator_weight;
        if lobe_is_finite(estimate) {
            estimate
        } else {
            0.0
        }
    }
}

/// Scalar finite predicate with the same `(x - x) == 0` semantics as WESL.
#[inline]
pub fn lobe_is_finite(x: f32) -> bool {
    x.is_finite()
}

/// `triple32` integer avalanche hash (Chris Wellons), reproduced by WGSL's
/// wrapping `u32` arithmetic.
#[inline]
pub fn lobe_hash_u32(mut x: u32) -> u32 {
    x ^= x >> 17;
    x = x.wrapping_mul(0xed5a_d4bb);
    x ^= x >> 11;
    x = x.wrapping_mul(0xac4c_1b51);
    x ^= x >> 15;
    x = x.wrapping_mul(0x3184_8bab);
    x ^= x >> 14;
    x
}

/// Order-sensitive hash combine.
#[inline]
pub fn lobe_hash_combine(seed: u32, value: u32) -> u32 {
    lobe_hash_u32(
        seed.wrapping_add(0x9e37_79b9)
            .wrapping_add(lobe_hash_u32(value)),
    )
}

/// Maps a hash to an exactly representable `f32` in `[0, 1)`.
#[inline]
pub fn lobe_uniform_from_hash(h: u32) -> f32 {
    (h >> 8) as f32 * (1.0 / 16_777_216.0)
}

/// Counter-based uniform draw from a stream seeded by `seed`.
#[inline]
pub fn lobe_uniform_at(seed: u32, counter: u32) -> f32 {
    lobe_uniform_from_hash(lobe_hash_combine(seed, counter))
}

/// Sanitises a weight to a finite non-negative value.
#[inline]
pub fn lobe_sanitize_weight(weight: f32) -> f32 {
    if lobe_is_finite(weight) && weight > 0.0 {
        weight
    } else {
        0.0
    }
}

/// Clamps a sample coordinate to `[0, 1 - 2^-24]`; non-finite input maps to 0.
#[inline]
pub fn lobe_clamp_unit(u: f32) -> f32 {
    if lobe_is_finite(u) {
        u.clamp(0.0, LOBE_SAMPLING_ONE_MINUS_EPSILON)
    } else {
        0.0
    }
}

/// R2 point for a sequence index and additive Cranley–Patterson rotation.
///
/// Both coordinates are returned in `[0, 1)`; non-finite rotations and
/// indices beyond the exact `f32` range are handled without propagating
/// non-finite values.
#[inline]
pub fn lobe_r2_sample_2d(index: u32, rotation_x: f32, rotation_y: f32) -> (f32, f32) {
    let n = (index & LOBE_SAMPLING_SEQUENCE_MASK) as f32;
    let rx = if lobe_is_finite(rotation_x) {
        rotation_x
    } else {
        0.0
    };
    let ry = if lobe_is_finite(rotation_y) {
        rotation_y
    } else {
        0.0
    };
    let x = ops::fract(0.5 + LOBE_SAMPLING_R2_A1 * n + rx);
    let y = ops::fract(0.5 + LOBE_SAMPLING_R2_A2 * n + ry);
    (
        if lobe_is_finite(x) { x } else { 0.0 },
        if lobe_is_finite(y) { y } else { 0.0 },
    )
}

/// Per-pixel, per-frame, per-sample blue-noise-like R2 point.
///
/// The pixel seed controls the Cranley–Patterson rotation and an odd temporal
/// stride.  The odd stride guarantees that frames visit different residue
/// classes of the 2^16 sequence, while the rotation gives neighbouring pixels
/// independent phase.
#[inline]
pub fn lobe_blue_noise_2d(
    pixel_x: u32,
    pixel_y: u32,
    frame: u32,
    sample: u32,
    seed: u32,
) -> (f32, f32) {
    let pixel_seed = lobe_pixel_seed(pixel_x, pixel_y, seed);
    let rotation_x = lobe_uniform_from_hash(pixel_seed);
    let rotation_y = lobe_uniform_from_hash(lobe_rotate_left_16(pixel_seed));
    let temporal_stride = lobe_temporal_stride(pixel_seed);
    let sequence_index = sample
        .wrapping_add(frame.wrapping_mul(temporal_stride))
        & LOBE_SAMPLING_SEQUENCE_MASK;
    lobe_r2_sample_2d(sequence_index, rotation_x, rotation_y)
}

/// Per-pixel seed controlling the Cranley–Patterson rotation and temporal
/// stride.  Exposed separately so CPU/GPU parity tests can freeze every
/// intermediate of the sequence.
#[inline]
pub fn lobe_pixel_seed(pixel_x: u32, pixel_y: u32, seed: u32) -> u32 {
    lobe_hash_combine(
        lobe_hash_combine(lobe_hash_combine(seed, LOBE_SAMPLING_SEED_SALT), pixel_x),
        pixel_y,
    )
}

/// Rotate-left by 16, matching the WESL twin's `(x << 16) | (x >> 16)`.
#[inline]
pub fn lobe_rotate_left_16(x: u32) -> u32 {
    x.rotate_left(16)
}

/// Per-pixel odd temporal stride.
#[inline]
pub fn lobe_temporal_stride(pixel_seed: u32) -> u32 {
    (lobe_hash_combine(pixel_seed, LOBE_SAMPLING_TEMPORAL_SALT)
        ^ LOBE_SAMPLING_GOLDEN_STRIDE)
        | 1
}

/// One-dimensional convenience form of [`lobe_blue_noise_2d`].
#[inline]
pub fn lobe_blue_noise_1d(
    pixel_x: u32,
    pixel_y: u32,
    frame: u32,
    sample: u32,
    seed: u32,
) -> f32 {
    lobe_blue_noise_2d(pixel_x, pixel_y, frame, sample, seed).0
}

/// Writes normalised probabilities for `weights`.
///
/// Returns the number of entries written.  The function writes nothing and
/// returns zero for an empty input or an output slice shorter than the input.
/// If all weights are unusable, probabilities are uniform over the supplied
/// entries, which keeps the estimator unbiased for a fully unweighted group.
#[inline]
pub fn lobe_write_probabilities(weights: &[f32], probabilities: &mut [f32]) -> usize {
    let count = weights.len();
    if count == 0 || probabilities.len() < count {
        return 0;
    }

    let mut total = 0.0f32;
    for &weight in weights {
        total += lobe_sanitize_weight(weight);
    }
    if lobe_is_finite(total) && total > LOBE_SAMPLING_MIN_WEIGHT {
        for (slot, &weight) in probabilities[..count].iter_mut().zip(weights) {
            let probability = lobe_sanitize_weight(weight) / total;
            *slot = if lobe_is_finite(probability) {
                probability.clamp(0.0, 1.0)
            } else {
                0.0
            };
        }
    } else {
        let uniform = 1.0 / count as f32;
        for slot in &mut probabilities[..count] {
            *slot = uniform;
        }
    }
    count
}

/// Probability of one sanitised weight given the sanitised total.
#[inline]
pub fn lobe_weight_probability(weight: f32, total_weight: f32) -> f32 {
    let total = if lobe_is_finite(total_weight) && total_weight > LOBE_SAMPLING_MIN_WEIGHT {
        total_weight
    } else {
        return 0.0;
    };
    let probability = lobe_sanitize_weight(weight) / total;
    if lobe_is_finite(probability) {
        probability.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Returns the unbiased estimator factor `1 / p` for a selection probability.
#[inline]
pub fn lobe_estimator_weight(probability: f32) -> f32 {
    if !lobe_is_finite(probability) || probability <= 0.0 {
        return 0.0;
    }
    let inverse = 1.0 / probability.clamp(LOBE_SAMPLING_MIN_WEIGHT, 1.0);
    if lobe_is_finite(inverse) {
        inverse
    } else {
        0.0
    }
}

/// Builds a valid sample for an explicitly selected lobe and probability.
#[inline]
pub fn lobe_make_sample(index: u32, probability: f32) -> LobeSample {
    if index == LOBE_SAMPLING_NO_LOBE {
        return LobeSample::NONE;
    }
    let probability = if lobe_is_finite(probability) {
        probability.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let estimator_weight = lobe_estimator_weight(probability);
    if probability <= 0.0 || estimator_weight <= 0.0 {
        LobeSample::NONE
    } else {
        LobeSample {
            index,
            probability,
            estimator_weight,
        }
    }
}

/// Selects one lobe by a sanitised, clamped uniform variate `u`.
///
/// Non-positive weights are omitted when at least one finite positive weight
/// exists.  If every weight is unusable, selection falls back to a uniform
/// bucket over the supplied entries.  Empty input returns [`LobeSample::NONE`].
#[inline]
pub fn lobe_select(weights: &[f32], u: f32) -> LobeSample {
    let count = weights.len();
    if count == 0 {
        return LobeSample::NONE;
    }
    let u = lobe_clamp_unit(u);

    let mut total = 0.0f32;
    for &weight in weights {
        total += lobe_sanitize_weight(weight);
    }

    if !lobe_is_finite(total) || total <= LOBE_SAMPLING_MIN_WEIGHT {
        let index = ops::floor(u * count as f32) as u32;
        let index = index.min((count - 1) as u32);
        return lobe_make_sample(index, 1.0 / count as f32);
    }

    let target = u * total;
    let mut cumulative = 0.0f32;
    let mut last_index = LOBE_SAMPLING_NO_LOBE;
    let mut last_weight = 0.0f32;
    for (index, &raw_weight) in weights.iter().enumerate() {
        let weight = lobe_sanitize_weight(raw_weight);
        if weight <= 0.0 {
            continue;
        }
        last_index = index as u32;
        last_weight = weight;
        cumulative += weight;
        if target < cumulative {
            return lobe_make_sample(last_index, weight / total);
        }
    }

    if last_index != LOBE_SAMPLING_NO_LOBE {
        lobe_make_sample(last_index, last_weight / total)
    } else {
        LobeSample::NONE
    }
}

/// Selects a lobe using the deterministic blue-noise-like stream for this
/// pixel, frame, and intra-frame sample.
#[inline]
pub fn lobe_select_blue_noise(
    weights: &[f32],
    pixel_x: u32,
    pixel_y: u32,
    frame: u32,
    sample: u32,
    seed: u32,
) -> LobeSample {
    let u = lobe_blue_noise_1d(pixel_x, pixel_y, frame, sample, seed);
    lobe_select(weights, u)
}

/// Unbiased scalar estimate for one selected lobe.
#[inline]
pub fn lobe_estimate(sample: LobeSample, contribution: f32) -> f32 {
    sample.estimate(contribution)
}

/// Deterministic full sum used as the P3-e reference.
///
/// When at least one positive weight exists, only positive-weight lobes belong
/// to the weighted group and zero-weight contributions are excluded.  When all
/// weights are unusable, the selection fallback is uniform, so every supplied
/// contribution belongs to the reference sum.
#[inline]
pub fn lobe_full_sum(weights: &[f32], contributions: &[f32]) -> f32 {
    let count = weights.len().min(contributions.len());
    if count == 0 {
        return 0.0;
    }

    let mut total = 0.0f32;
    for &weight in &weights[..count] {
        total += lobe_sanitize_weight(weight);
    }
    let active_mask = lobe_is_finite(total) && total > LOBE_SAMPLING_MIN_WEIGHT;

    let mut sum = 0.0f32;
    for index in 0..count {
        let weight = lobe_sanitize_weight(weights[index]);
        if active_mask && weight <= 0.0 {
            continue;
        }
        let contribution = contributions[index];
        if lobe_is_finite(contribution) {
            sum += contribution;
        }
    }
    if lobe_is_finite(sum) {
        sum
    } else {
        0.0
    }
}

/// Evaluates an expensive-lobe group according to a P1-e-linked policy.
///
/// Returns `(value, sample)`.  A deterministic full sum returns
/// [`LobeSample::NONE`] as the sample; stochastic mode returns the selected
/// lobe and its estimator factor.  Cheap lobes must be added by the caller.
#[inline]
pub fn lobe_sample_quality_sum(
    policy: LobeSamplingPolicy,
    weights: &[f32],
    contributions: &[f32],
    pixel_x: u32,
    pixel_y: u32,
    frame: u32,
    sample: u32,
    seed: u32,
) -> (f32, LobeSample) {
    let count = weights.len().min(contributions.len());
    if count == 0 {
        return (0.0, LobeSample::NONE);
    }

    let weights = &weights[..count];
    let contributions = &contributions[..count];
    if !policy.should_use_stochastic(count) {
        return (lobe_full_sum(weights, contributions), LobeSample::NONE);
    }

    let selected = lobe_select_blue_noise(weights, pixel_x, pixel_y, frame, sample, seed);
    let value = if selected.is_valid() {
        selected.estimate(contributions[selected.index as usize])
    } else {
        0.0
    };
    (value, selected)
}

/// The resolve's single-lobe sampling state for one pixel (§17.4, P3-e).
///
/// `policy` comes from the host's quality tier (P1-e budget applied); `frame`
/// and `pixel` seed the blue-noise draw. The default is `High` — the full
/// evaluation, i.e. an exact identity — so a caller that does not opt into the
/// amortised path shades bit-for-bit as before.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LobeSamplingSettings {
    /// Quality/budget plan derived by the host.
    pub policy: LobeSamplingPolicy,
    /// Frame index driving the temporal decorrelation (TAA accumulates it).
    pub frame_index: u32,
    /// The pixel's integer coordinate (`x`, `y`).
    pub pixel: [u32; 2],
}

impl Default for LobeSamplingSettings {
    fn default() -> Self {
        Self {
            policy: LobeSamplingPolicy::for_quality(crate::ShadingQuality::High),
            frame_index: 0,
            pixel: [0, 0],
        }
    }
}

/// Per-pixel factors for the two *shapes* of expensive lobe (§17.4).
///
/// The shapes need different estimators, and the difference is a property of the
/// lobe, not a taste:
///
/// * **Blend-shaped** (glint: `mix(smooth, discrete, weight)`) — the weight *is*
///   the mixture probability, so the sampler draws the **state** and applies no
///   scaling. Scaling a signed delta by `1/p` would drive the blended NDF
///   negative, i.e. negative radiance.
/// * **Additive** (measured BRDF: `surface = analytic + w·table`) — the classic
///   single-sample estimator: pick one lobe with probability `p` and scale its
///   (non-negative) response by `1/p`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExpensiveLobeFactors {
    /// `GLINT_STATE_AUTHORED` when the pixel did not sample glint, else the
    /// drawn mixture state in `[0, 1]`.
    pub glint_state: f32,
    /// Estimator factor for the additive lobe group: `1/p` when this pixel
    /// evaluates the measured lobe, `0` when it does not.
    pub measured: f32,
}

impl ExpensiveLobeFactors {
    /// The exact, un-amortised factors: the authored blend and the full additive
    /// term. Every golden baseline is this value.
    pub const FULL: Self = Self {
        glint_state: crate::lighting::GLINT_STATE_AUTHORED,
        measured: 1.0,
    };
}

/// Draw this pixel's expensive-lobe factors from the sampling policy.
///
/// `weights` are the authored lobe weights (`[glint, measured]`); an inert lobe
/// is never selected. With the policy in [`LobeSamplingMode::Full`] — or with
/// nothing expensive live — this returns [`ExpensiveLobeFactors::FULL`], so the
/// full-quality path is an exact identity.
///
/// `pixel` is the pixel's integer coordinate, `frame` the frame index and
/// `sample` a per-pixel sequence index (a second sample per frame is available
/// if a host wants to derive two decisions); the blue-noise draw is spatially
/// decorrelated and changes every frame, which is what lets TAA converge the
/// estimator at `LobeSamplingPolicy::temporal_accumulation_frames`.
#[must_use]
pub fn select_expensive_lobe_factors(
    policy: LobeSamplingPolicy,
    weights: [f32; 2],
    pixel: [u32; 2],
    frame: u32,
    sample: u32,
) -> ExpensiveLobeFactors {
    let glint_live = lobe_sanitize_weight(weights[0]) > LOBE_SAMPLING_MIN_WEIGHT;
    let measured_live = lobe_sanitize_weight(weights[1]) > LOBE_SAMPLING_MIN_WEIGHT;
    let live = usize::from(glint_live) + usize::from(measured_live);
    if !policy.should_use_stochastic(live) {
        return ExpensiveLobeFactors::FULL;
    }

    let mut factors = ExpensiveLobeFactors::FULL;

    // Blend-shaped: draw the state, never scale it.
    if glint_live {
        let u = lobe_blue_noise_1d(
            pixel[0],
            pixel[1],
            frame,
            sample,
            LOBE_SAMPLING_SEED_SALT,
        );
        factors.glint_state = f32::from(lobe_sanitize_weight(weights[0]) > u);
    }

    // Additive: pick one with probability proportional to its weight and use the
    // `1/p` estimator. With a single live additive lobe the selection is
    // degenerate (`p == 1`), which is the honest state of the catalogue today.
    if measured_live {
        let sanitized = [lobe_sanitize_weight(weights[1])];
        let u = lobe_blue_noise_1d(
            pixel[0],
            pixel[1],
            frame,
            sample.wrapping_add(1),
            LOBE_SAMPLING_SEED_SALT,
        );
        let picked = lobe_select(&sanitized, u);
        factors.measured = if picked.is_valid() {
            lobe_estimator_weight(picked.probability)
        } else {
            1.0
        };
    }

    factors
}

#[cfg(test)]
mod tests {
    use super::*;

    const WEIGHTS: [f32; 3] = [1.0, 2.0, 3.0];
    const CONTRIBUTIONS: [f32; 3] = [0.25, 0.5, 0.75];
    const FULL_SUM: f32 = 1.5;

    fn full_sum_f64(weights: &[f32], contributions: &[f32]) -> f64 {
        let active = weights
            .iter()
            .copied()
            .map(lobe_sanitize_weight)
            .sum::<f32>()
            > LOBE_SAMPLING_MIN_WEIGHT;
        weights
            .iter()
            .copied()
            .zip(contributions.iter().copied())
            .filter(|(weight, _)| !active || lobe_sanitize_weight(*weight) > 0.0)
            .map(|(_, contribution)| contribution as f64)
            .sum()
    }

    #[test]
    fn policy_aggressiveness_tracks_quality() {
        assert_eq!(
            LobeSamplingPolicy::for_quality(ShadingQuality::Low).max_full_lobes(),
            1
        );
        assert_eq!(
            LobeSamplingPolicy::for_quality(ShadingQuality::Medium).max_full_lobes(),
            2
        );
        assert_eq!(
            LobeSamplingPolicy::for_quality(ShadingQuality::High).mode(),
            LobeSamplingMode::Full
        );
        assert_eq!(
            LobeSamplingPolicy::for_quality(ShadingQuality::Ultra).mode(),
            LobeSamplingMode::Full
        );
        assert!(!LobeSamplingPolicy::for_quality(ShadingQuality::High)
            .should_use_stochastic(64));
    }

    #[test]
    fn medium_keeps_two_lobes_and_low_keeps_one_deterministic() {
        let low = LobeSamplingPolicy::for_quality(ShadingQuality::Low);
        let medium = LobeSamplingPolicy::for_quality(ShadingQuality::Medium);
        assert_eq!(low.evaluations_per_pixel(1), 1);
        assert_eq!(low.evaluations_per_pixel(2), 1);
        assert_eq!(medium.evaluations_per_pixel(2), 2);
        assert_eq!(medium.evaluations_per_pixel(3), 1);
        assert!(!medium.should_use_stochastic(2));
        assert!(medium.should_use_stochastic(3));
    }

    #[test]
    fn downscaled_budget_makes_high_stochastic() {
        let scaled = ShadingQuality::High.budget().with_global_scale(0.5);
        let policy = LobeSamplingPolicy::for_budget(ShadingQuality::High, scaled);
        assert_eq!(policy.mode(), LobeSamplingMode::Stochastic);
        assert_eq!(policy.max_full_lobes(), 1);
        assert_eq!(policy.evaluations_per_pixel(4), 1);
        assert_eq!(policy.temporal_accumulation_frames(), 8);
    }

    #[test]
    fn high_and_ultra_are_exact_full_sum_fallbacks() {
        for quality in [ShadingQuality::High, ShadingQuality::Ultra] {
            let policy = LobeSamplingPolicy::for_quality(quality);
            let (value, sample) = lobe_sample_quality_sum(
                policy, &WEIGHTS, &CONTRIBUTIONS, 17, 29, 123, 0, 0xdead_beef,
            );
            assert_eq!(sample, LobeSample::NONE);
            assert_eq!(value, FULL_SUM);
            assert_eq!(policy.evaluations_per_pixel(WEIGHTS.len()), WEIGHTS.len());
        }
    }

    #[test]
    fn probabilities_are_normalized_and_ordered() {
        let mut probabilities = [0.0f32; 3];
        assert_eq!(
            lobe_write_probabilities(&WEIGHTS, &mut probabilities),
            WEIGHTS.len()
        );
        assert!((probabilities[0] - 1.0 / 6.0).abs() < 1.0e-7);
        assert!((probabilities[1] - 2.0 / 6.0).abs() < 1.0e-7);
        assert!((probabilities[2] - 3.0 / 6.0).abs() < 1.0e-7);
        let sum: f32 = probabilities.iter().sum();
        assert!((sum - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn all_zero_weights_fall_back_to_uniform_probability() {
        let mut probabilities = [0.0f32; 3];
        assert_eq!(
            lobe_write_probabilities(&[0.0, -1.0, f32::NAN], &mut probabilities),
            3
        );
        for probability in probabilities {
            assert!((probability - 1.0 / 3.0).abs() < 1.0e-7);
        }
        let sum: f32 = probabilities.iter().sum();
        assert!((sum - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn zero_weight_lobes_are_never_selected() {
        let weights = [1.0, 0.0, 2.0];
        for i in 0..4096u32 {
            let u = (i as f32 + 0.5) / 4096.0;
            let sample = lobe_select(&weights, u);
            assert!(sample.is_valid());
            assert_ne!(sample.index, 1);
        }
    }

    #[test]
    fn single_lobe_selection_is_an_exact_identity() {
        let sample = lobe_select(&[7.0], 0.37);
        assert_eq!(sample.index, 0);
        assert!((sample.probability - 1.0).abs() < 1.0e-7);
        assert!((sample.estimator_weight - 1.0).abs() < 1.0e-7);
        assert!((lobe_estimate(sample, 0.42) - 0.42).abs() < 1.0e-7);
    }

    #[test]
    fn empty_inputs_are_inert() {
        let empty: [f32; 0] = [];
        assert_eq!(lobe_select(&empty, 0.5), LobeSample::NONE);
        assert_eq!(
            lobe_select_blue_noise(&empty, 0, 0, 0, 0, 0),
            LobeSample::NONE
        );
        assert_eq!(lobe_full_sum(&empty, &empty), 0.0);
        let (value, sample) = lobe_sample_quality_sum(
            LobeSamplingPolicy::for_quality(ShadingQuality::Low),
            &empty,
            &empty,
            0,
            0,
            0,
            0,
            0,
        );
        assert_eq!(value, 0.0);
        assert_eq!(sample, LobeSample::NONE);
    }

    #[test]
    fn estimator_weight_is_inverse_probability() {
        for probability in [1.0f32, 0.5, 0.25, 0.125, 1.0 / 64.0] {
            let inverse = lobe_estimator_weight(probability);
            assert!((inverse * probability - 1.0).abs() < 1.0e-5);
        }
        assert_eq!(lobe_estimator_weight(0.0), 0.0);
        assert_eq!(lobe_estimator_weight(f32::NAN), 0.0);
    }

    #[test]
    fn same_coordinates_are_bit_deterministic() {
        let a = lobe_select_blue_noise(&WEIGHTS, 137, 901, 42, 0, 0x1234_5678);
        let b = lobe_select_blue_noise(&WEIGHTS, 137, 901, 42, 0, 0x1234_5678);
        assert_eq!(a, b);
        assert_eq!(
            lobe_blue_noise_2d(137, 901, 42, 0, 0x1234_5678),
            lobe_blue_noise_2d(137, 901, 42, 0, 0x1234_5678)
        );
    }

    #[test]
    fn frozen_parity_fixture_matches_the_wesl_inputs() {
        let pixel_seed = lobe_pixel_seed(137, 901, 0x1234_5678);
        assert_eq!(pixel_seed, 0x41e9_7335);
        assert_eq!(lobe_rotate_left_16(pixel_seed), 0x7335_41e9);

        let temporal_stride = lobe_temporal_stride(pixel_seed);
        assert_eq!(temporal_stride, 0xef7c_f4a7);
        assert_eq!((42u32.wrapping_mul(temporal_stride)) & 0xffff, 9062);

        let (u, v) = lobe_blue_noise_2d(137, 901, 42, 0, 0x1234_5678);
        assert_eq!(u.to_bits(), 0x3eeb_0000);
        assert_eq!(v.to_bits(), 0x3f57_e000);

        let sample = lobe_select(&WEIGHTS, u);
        assert_eq!(sample.index, 1);
        assert_eq!(sample.probability.to_bits(), 0x3eaa_aaab);
        assert_eq!(sample.estimator_weight.to_bits(), 0x4040_0000);
        assert_eq!(lobe_estimate(sample, CONTRIBUTIONS[1]).to_bits(), 0x3fc0_0000);
    }

    #[test]
    fn successive_frames_decorrelate() {
        let mut changed = 0u32;
        for frame in 0..256u32 {
            let a = lobe_select_blue_noise(&WEIGHTS, 31, 47, frame, 0, 7);
            let b = lobe_select_blue_noise(&WEIGHTS, 31, 47, frame + 1, 0, 7);
            changed += u32::from(a.index != b.index);
        }
        assert!(changed > 100, "only {changed}/256 frame pairs changed");
    }

    #[test]
    fn neighbouring_pixels_decorrelate() {
        let mut counts = [0u32; 3];
        for y in 0..16u32 {
            for x in 0..16u32 {
                let sample = lobe_select_blue_noise(&WEIGHTS, x, y, 0, 0, 19);
                counts[sample.index as usize] += 1;
            }
        }
        assert!(counts.iter().all(|&count| count > 20), "{counts:?}");
    }

    #[test]
    fn blue_noise_is_in_unit_interval_and_well_centred() {
        let mut sum = 0.0f64;
        let count = 16_384u32;
        for frame in 0..count {
            let u = lobe_blue_noise_1d(73, 41, frame, 0, 0xfeed_face);
            assert!((0.0..1.0).contains(&u), "u = {u}");
            sum += u as f64;
        }
        let mean = sum / count as f64;
        assert!((mean - 0.5).abs() < 0.002, "mean = {mean}");
    }

    #[test]
    fn r2_point_set_is_well_spread() {
        let mut points = Vec::with_capacity(256);
        for index in 0..256u32 {
            points.push(lobe_r2_sample_2d(index, 0.0, 0.0));
        }
        let mut minimum_distance = f32::INFINITY;
        for i in 0..points.len() {
            for j in (i + 1)..points.len() {
                let dx = points[i].0 - points[j].0;
                let dy = points[i].1 - points[j].1;
                minimum_distance = minimum_distance.min((dx * dx + dy * dy).sqrt());
            }
        }
        assert!(
            minimum_distance > 0.5 / (256.0f32).sqrt(),
            "minimum distance = {minimum_distance}"
        );
    }

    #[test]
    fn stochastic_estimator_converges_to_full_sum() {
        let mut sum = 0.0f64;
        let count = 32_768u32;
        for frame in 0..count {
            let (value, sample) = lobe_sample_quality_sum(
                LobeSamplingPolicy::for_quality(ShadingQuality::Low),
                &WEIGHTS,
                &CONTRIBUTIONS,
                0x1234,
                0x5678,
                frame,
                0,
                0x0bad_f00d,
            );
            assert!(sample.is_valid());
            sum += value as f64;
        }
        let reference = full_sum_f64(&WEIGHTS, &CONTRIBUTIONS);
        let mean = sum / count as f64;
        let relative_error = (mean - reference).abs() / reference.abs().max(1.0e-12);
        assert!(
            relative_error < 0.01,
            "mean = {mean}, reference = {reference}, relative error = {relative_error}"
        );
    }

    #[test]
    fn all_zero_weight_estimator_is_still_unbiased() {
        let weights = [0.0f32, 0.0, 0.0];
        let contributions = [0.25f32, 0.5, 0.75];
        let mut sum = 0.0f64;
        let count = 16_384u32;
        for frame in 0..count {
            let (value, sample) = lobe_sample_quality_sum(
                LobeSamplingPolicy::for_quality(ShadingQuality::Low),
                &weights,
                &contributions,
                91,
                17,
                frame,
                0,
                0xabcd_1234,
            );
            assert!(sample.is_valid());
            sum += value as f64;
        }
        let reference = full_sum_f64(&weights, &contributions);
        let mean = sum / count as f64;
        let relative_error = (mean - reference).abs() / reference.abs().max(1.0e-12);
        assert!(
            relative_error < 0.01,
            "mean = {mean}, reference = {reference}, relative error = {relative_error}"
        );
    }

    #[test]
    fn full_sum_respects_the_weight_mask() {
        let weights = [1.0f32, 0.0, 2.0];
        let contributions = [0.25f32, 100.0, 0.75];
        assert_eq!(lobe_full_sum(&weights, &contributions), 1.0);
        let all_zero = [0.0f32, 0.0, 0.0];
        assert_eq!(lobe_full_sum(&all_zero, &contributions), 101.0);
    }

    #[test]
    fn non_finite_adversarial_inputs_stay_finite() {
        let weights = [f32::NAN, f32::INFINITY, -3.0, 0.0];
        for i in 0..4096u32 {
            let u = (i as f32 + 0.5) / 4096.0;
            let sample = lobe_select(&weights, u);
            assert!(sample.is_valid());
            assert!(lobe_is_finite(sample.estimator_weight));
            let estimate = sample.estimate(f32::NAN);
            assert_eq!(estimate, 0.0);
        }
        let sample = lobe_select(&WEIGHTS, f32::NAN);
        assert!(sample.is_valid());
        assert!(lobe_is_finite(lobe_estimate(sample, 1.0e30)));
    }

    #[test]
    fn probability_writer_is_bounds_safe() {
        let mut too_small = [0.0f32; 2];
        assert_eq!(lobe_write_probabilities(&WEIGHTS, &mut too_small), 0);
        let empty: [f32; 0] = [];
        let mut output = [0.0f32; 0];
        assert_eq!(lobe_write_probabilities(&empty, &mut output), 0);
    }


    #[test]
    fn full_quality_is_an_exact_identity() {
        // §17.4: the amortised path must never perturb a pixel that did not opt
        // into it. `High` evaluates everything, so the factors are the sentinel
        // plus the full additive weight.
        for quality in [ShadingQuality::High, ShadingQuality::Ultra] {
            let policy = LobeSamplingPolicy::for_quality(quality);
            let factors =
                select_expensive_lobe_factors(policy, [0.9, 0.7], [3, 4], 11, 0);
            assert_eq!(factors, ExpensiveLobeFactors::FULL, "{quality:?}");
        }
        // …and an inert material is never sampled, whatever the tier says.
        let policy = LobeSamplingPolicy::for_quality(ShadingQuality::Low);
        assert_eq!(
            select_expensive_lobe_factors(policy, [0.0, 0.0], [0, 0], 0, 0),
            ExpensiveLobeFactors::FULL
        );
    }

    #[test]
    fn the_glint_state_draws_with_the_authored_weight() {
        // The mixture identity in action: over pixels and frames the drawn state
        // is `1` exactly as often as the authored weight, and it is never scaled.
        // Both expensive lobes live, so the tier's one-lobe budget makes the
        // group stochastic. With a *single* live lobe the canonical tiers stay
        // on the deterministic path — that case is covered separately.
        let policy = LobeSamplingPolicy::for_quality(ShadingQuality::Low);
        let weight = 0.35_f32;
        let mut ones = 0u32;
        let mut total = 0u32;
        for frame in 0..64u32 {
            for y in 0..8u32 {
                for x in 0..8u32 {
                    let factors =
                        select_expensive_lobe_factors(policy, [weight, 0.5], [x, y], frame, 0);
                    assert!(
                        factors.glint_state == 0.0 || factors.glint_state == 1.0,
                        "the state must be a draw, not a factor: {}",
                        factors.glint_state
                    );
                    ones += u32::from(factors.glint_state == 1.0);
                    total += 1;
                }
            }
        }
        let fraction = ones as f32 / total as f32;
        assert!(
            (fraction - weight).abs() < 0.03,
            "drawn state fraction {fraction} vs weight {weight}"
        );
    }

    #[test]
    fn a_single_live_lobe_stays_deterministic_at_the_canonical_tiers() {
        // The catalogue carries one additive heavy lobe (measured) and one
        // blend-shaped one (glint); `Low`'s budget is one lobe, so a material
        // with only one of them is never sampled. Nothing is scaled down, and
        // neither lobe needs a special case to say so.
        let policy = LobeSamplingPolicy::for_quality(ShadingQuality::Low);
        for frame in 0..16u32 {
            let only_measured =
                select_expensive_lobe_factors(policy, [0.0, 0.8], [1, 2], frame, 0);
            assert_eq!(only_measured, ExpensiveLobeFactors::FULL);
            let only_glint =
                select_expensive_lobe_factors(policy, [0.6, 0.0], [1, 2], frame, 0);
            assert_eq!(only_glint, ExpensiveLobeFactors::FULL);
        }
    }

    #[test]
    fn the_additive_group_uses_the_inverse_probability_estimator() {
        // Drive the estimator with an explicit two-lobe group: equal weights, so
        // the selected lobe is weighted by 2 and the ensemble mean of
        // `factor · response` is the plain sum.
        let weights = [1.0_f32, 1.0_f32];
        let mut sums = [0.0_f32; 2];
        let mut picks = [0u32; 2];
        let draws = 4096u32;
        for index in 0..draws {
            let pixel = [index % 64, index / 64];
            let u = lobe_blue_noise_1d(
                pixel[0],
                pixel[1],
                0,
                1,
                LOBE_SAMPLING_SEED_SALT,
            );
            let picked = lobe_select(&weights, u);
            assert!(picked.is_valid());
            let factor = lobe_estimator_weight(picked.probability);
            picks[picked.index as usize] += 1;
            sums[picked.index as usize] += factor * weights[picked.index as usize];
        }
        assert!(picks[0] > 0 && picks[1] > 0, "both lobes must be picked");
        // `E[1_{pick i}/p_i · w_i] = w_i`, so each lobe's accumulated estimate is
        // its weight times the number of draws.
        for lobe in 0..2 {
            let mean = sums[lobe] / draws as f32;
            assert!(
                (mean - weights[lobe]).abs() < 0.05,
                "lobe {lobe}: {mean} vs {}",
                weights[lobe]
            );
        }
    }


    #[test]
    fn a_downscaled_budget_samples_where_full_quality_would_not() {
        // `LobeSampleBudget::with_global_scale` is the host's "shed samples"
        // lever: below the tier's canonical total, `High` drops from "evaluate
        // everything" to the one-lobe budget, so a two-heavy-lobe material is
        // sampled even on the highest tier.
        let policy = LobeSamplingPolicy::for_quality(ShadingQuality::High);
        assert_eq!(policy.mode(), LobeSamplingMode::Full);
        assert_eq!(
            select_expensive_lobe_factors(policy, [0.5, 0.5], [0, 0], 0, 0),
            ExpensiveLobeFactors::FULL
        );

        let budget = ShadingQuality::High.budget().with_global_scale(0.5);
        let downscaled = LobeSamplingPolicy::for_budget(ShadingQuality::High, budget);
        assert_eq!(downscaled.mode(), LobeSamplingMode::Stochastic);
        let mut states = [0u32; 2];
        for frame in 0..32u32 {
            let factors = select_expensive_lobe_factors(
                downscaled,
                [0.5, 0.5],
                [frame % 7, frame / 7],
                frame,
                0,
            );
            assert!(factors.glint_state == 0.0 || factors.glint_state == 1.0);
            states[usize::from(factors.glint_state == 1.0)] += 1;
        }
        assert!(states[0] > 0 && states[1] > 0, "both states must occur");
    }

    #[test]
    fn quality_policy_exposes_its_budget_snapshot() {
        let budget = ShadingQuality::Low.budget().with_global_scale(0.5);
        let policy = LobeSamplingPolicy::for_budget(ShadingQuality::Low, budget);
        assert_eq!(policy.quality(), ShadingQuality::Low);
        assert_eq!(policy.budget(), budget.clamped());
        assert_eq!(policy.temporal_accumulation_frames(), 8);
    }
}
