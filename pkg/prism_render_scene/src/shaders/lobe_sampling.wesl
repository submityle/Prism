//! Deterministic stochastic single-lobe sampling helpers for expensive lobes;
//! no bindings or entry points.
//!
//! GPU twin of `pkg/prism_render_shading/src/lobe_sampling.rs` (§17.4). Keep a
//! material's complete lobe set, select one expensive lobe with probability
//! `p_i = w_i / sum(w)`, and return `f_i / p_i`. The expected value is the
//! deterministic full sum (`E[f_i / p_i] = sum(f)`), so TAA converges without
//! lobe collapse.
//!
//! Selection uses the R2 low-discrepancy sequence with a per-pixel
//! Cranley-Patterson rotation and a per-pixel odd temporal stride. The R2 point
//! set is blue-noise-like over time, neighbouring pixels do not share phase,
//! successive frames advance to different residue classes, and the 2^16 period
//! remains exactly representable in f32. All arithmetic, constants, clamps, and
//! operation order mirror the CPU golden; callers iterate their sparse lobe
//! arrays and use `ls_select_bucket` or `ls_make_sample`.

/// Salt mixed into every pixel seed.
const LS_SEED_SALT: u32 = 0x53a71e5du;
/// Golden-ratio odd stride used by the sequence.
const LS_GOLDEN_STRIDE: u32 = 0x9e3779b9u;
/// Salt used to derive a per-pixel temporal stride.
const LS_TEMPORAL_SALT: u32 = 0x51ed270bu;
/// R2 sequence period, exact in f32.
const LS_SEQUENCE_PERIOD: u32 = 65536u;
/// Mask selecting the sequence index.
const LS_SEQUENCE_MASK: u32 = 65535u;
/// First R2 additive stride.
const LS_R2_A1: f32 = 0.7548776662466928;
/// Second R2 additive stride.
const LS_R2_A2: f32 = 0.5698402909980403;
/// Smallest usable probability mass.
const LS_MIN_WEIGHT: f32 = 1.1754944e-38;
/// Largest generated unit value, `1 - 2^-24`.
const LS_ONE_MINUS_EPSILON: f32 = 0.99999994;
/// Sentinel index used when no lobe can be selected.
const LS_NO_LOBE: u32 = 4294967295u;

/// One stochastic lobe selection and its unbiased estimator factor.
struct LobeSample {
    /// Index into the caller's expensive-lobe arrays.
    index: u32,
    /// Probability with which this lobe was selected.
    probability: f32,
    /// Unbiased estimator factor `1 / probability`.
    estimator_weight: f32,
}

/// Scalar finite predicate, matching the CPU golden's `(x - x) == 0` idiom.
fn ls_is_finite(x: f32) -> bool {
    return (x - x) == 0.0;
}

/// `triple32` integer avalanche hash.
fn ls_hash_u32(x_in: u32) -> u32 {
    var x = x_in;
    x = x ^ (x >> 17u);
    x = x * 0xed5ad4bbu;
    x = x ^ (x >> 11u);
    x = x * 0xac4c1b51u;
    x = x ^ (x >> 15u);
    x = x * 0x31848babu;
    x = x ^ (x >> 14u);
    return x;
}

/// Order-sensitive hash combine.
fn ls_hash_combine(seed: u32, value: u32) -> u32 {
    return ls_hash_u32(seed + 0x9e3779b9u + ls_hash_u32(value));
}

/// Maps a hash to an exactly representable f32 in [0, 1).
fn ls_uniform_from_hash(h: u32) -> f32 {
    return f32(h >> 8u) * (1.0 / 16777216.0);
}

/// Counter-based uniform draw from a stream seeded by `seed`.
fn ls_uniform_at(seed: u32, counter: u32) -> f32 {
    return ls_uniform_from_hash(ls_hash_combine(seed, counter));
}

/// Sanitises a weight to a finite non-negative value.
fn ls_sanitize_weight(weight: f32) -> f32 {
    if (ls_is_finite(weight) && weight > 0.0) {
        return weight;
    }
    return 0.0;
}

/// Clamps a sample coordinate to [0, 1 - 2^-24].
fn ls_clamp_unit(u: f32) -> f32 {
    if (ls_is_finite(u)) {
        return clamp(u, 0.0, LS_ONE_MINUS_EPSILON);
    }
    return 0.0;
}

/// Manual rotate-left by 16, matching Rust's `u32::rotate_left(16)`.
fn ls_rotate_left_16(x: u32) -> u32 {
    return (x << 16u) | (x >> 16u);
}

/// R2 point for a sequence index and additive rotation.
fn ls_r2_sample_2d(index: u32, rotation_x: f32, rotation_y: f32) -> vec2<f32> {
    let n = f32(index & LS_SEQUENCE_MASK);
    var rx = rotation_x;
    var ry = rotation_y;
    if (!ls_is_finite(rx)) {
        rx = 0.0;
    }
    if (!ls_is_finite(ry)) {
        ry = 0.0;
    }
    let x = fract(0.5 + LS_R2_A1 * n + rx);
    let y = fract(0.5 + LS_R2_A2 * n + ry);
    if (!ls_is_finite(x)) {
        return vec2<f32>(0.0, select(0.0, y, ls_is_finite(y)));
    }
    return vec2<f32>(x, select(0.0, y, ls_is_finite(y)));
}

/// Per-pixel seed controlling sequence rotation and temporal stride.
fn ls_pixel_seed(pixel_x: u32, pixel_y: u32, seed: u32) -> u32 {
    return ls_hash_combine(
        ls_hash_combine(ls_hash_combine(seed, LS_SEED_SALT), pixel_x),
        pixel_y,
    );
}

/// Per-pixel odd temporal stride.
fn ls_temporal_stride(pixel_seed: u32) -> u32 {
    return (ls_hash_combine(pixel_seed, LS_TEMPORAL_SALT) ^ LS_GOLDEN_STRIDE) | 1u;
}

/// Per-pixel, per-frame, per-sample blue-noise-like R2 point.
fn ls_blue_noise_2d(
    pixel_x: u32,
    pixel_y: u32,
    frame: u32,
    sample: u32,
    seed: u32,
) -> vec2<f32> {
    let pixel_seed = ls_pixel_seed(pixel_x, pixel_y, seed);
    let rotation_x = ls_uniform_from_hash(pixel_seed);
    let rotation_y = ls_uniform_from_hash(ls_rotate_left_16(pixel_seed));
    let sequence_index =
        (sample + frame * ls_temporal_stride(pixel_seed)) & LS_SEQUENCE_MASK;
    return ls_r2_sample_2d(sequence_index, rotation_x, rotation_y);
}

/// One-dimensional convenience form of `ls_blue_noise_2d`.
fn ls_blue_noise_1d(
    pixel_x: u32,
    pixel_y: u32,
    frame: u32,
    sample: u32,
    seed: u32,
) -> f32 {
    return ls_blue_noise_2d(pixel_x, pixel_y, frame, sample, seed).x;
}

/// Maximum deterministic lobe prefix for a quality tier.
fn ls_max_full_lobes(quality: u32) -> u32 {
    if (quality == 0u) {
        return 1u;
    }
    if (quality == 1u) {
        return 2u;
    }
    return LS_NO_LOBE;
}

/// Whether a lobe count must use stochastic single-lobe sampling.
fn ls_should_use_stochastic(max_full_lobes: u32, lobe_count: u32) -> bool {
    return lobe_count > 0u && lobe_count > max_full_lobes;
}

/// Quality-tier convenience form of `ls_should_use_stochastic`.
fn ls_should_use_stochastic_for_quality(quality: u32, lobe_count: u32) -> bool {
    return ls_should_use_stochastic(ls_max_full_lobes(quality), lobe_count);
}

/// Number of lobe evaluations per pixel under a deterministic-prefix limit.
fn ls_full_evaluations(max_full_lobes: u32, lobe_count: u32) -> u32 {
    if (lobe_count == 0u) {
        return 0u;
    }
    if (ls_should_use_stochastic(max_full_lobes, lobe_count)) {
        return 1u;
    }
    return lobe_count;
}

/// Probability of one sanitised weight given the sanitised total.
fn ls_weight_probability(weight: f32, total_weight: f32) -> f32 {
    if (!ls_is_finite(total_weight) || total_weight <= LS_MIN_WEIGHT) {
        return 0.0;
    }
    let probability = ls_sanitize_weight(weight) / total_weight;
    if (ls_is_finite(probability)) {
        return clamp(probability, 0.0, 1.0);
    }
    return 0.0;
}

/// Returns the unbiased estimator factor `1 / p` for a probability.
fn ls_estimator_weight(probability: f32) -> f32 {
    if (!ls_is_finite(probability) || probability <= 0.0) {
        return 0.0;
    }
    let inverse = 1.0 / clamp(probability, LS_MIN_WEIGHT, 1.0);
    if (ls_is_finite(inverse)) {
        return inverse;
    }
    return 0.0;
}

/// Whether a sample refers to a real lobe.
fn ls_sample_is_valid(sample: LobeSample) -> bool {
    return sample.index != LS_NO_LOBE
        && ls_is_finite(sample.probability)
        && sample.probability > 0.0
        && ls_is_finite(sample.estimator_weight)
        && sample.estimator_weight > 0.0;
}

/// Builds a valid sample for an explicitly selected lobe and probability.
fn ls_make_sample(index: u32, probability: f32) -> LobeSample {
    var out: LobeSample;
    out.index = index;
    let p = select(probability, 0.0, ls_is_finite(probability));
    out.probability = clamp(p, 0.0, 1.0);
    out.estimator_weight = ls_estimator_weight(out.probability);
    if (index == LS_NO_LOBE || !ls_sample_is_valid(out)) {
        out.index = LS_NO_LOBE;
        out.probability = 0.0;
        out.estimator_weight = 0.0;
    }
    return out;
}

/// Unbiased scalar estimate for one selected lobe.
fn ls_estimate_scalar(contribution: f32, sample: LobeSample) -> f32 {
    if (!ls_sample_is_valid(sample) || !ls_is_finite(contribution)) {
        return 0.0;
    }
    let estimate = contribution * sample.estimator_weight;
    if (ls_is_finite(estimate)) {
        return estimate;
    }
    return 0.0;
}

/// CDF target for a clamped unit variate and sanitised weight total.
fn ls_cdf_target(u: f32, total_weight: f32) -> f32 {
    if (!ls_is_finite(total_weight) || total_weight <= 0.0) {
        return 0.0;
    }
    return ls_clamp_unit(u) * total_weight;
}

/// Whether a CDF bucket is selected; callers pass the cumulative sum including
/// the current lobe. A strict comparison plus the clamped target reproduces the
/// CPU golden's interval convention.
fn ls_select_bucket(cdf_target: f32, cumulative_after: f32) -> bool {
    return cdf_target < cumulative_after;
}

/// Uniform fallback index used when every supplied weight is unusable.
fn ls_uniform_fallback_index(u: f32, lobe_count: u32) -> u32 {
    if (lobe_count == 0u) {
        return LS_NO_LOBE;
    }
    let index = u32(floor(ls_clamp_unit(u) * f32(lobe_count)));
    return min(index, lobe_count - 1u);
}
