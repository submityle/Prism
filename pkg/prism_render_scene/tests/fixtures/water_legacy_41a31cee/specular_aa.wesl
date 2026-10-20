//! Shared classical specular anti-aliasing helpers; no bindings or entry points.
//!
//! GPU twins of `gi/specular_aa/{normal_variance,toksvig}.rs` and the
//! `gi/spec_gi/ggx_lobe.rs` roughness-to-alpha mapping. Keep arithmetic order,
//! sanitization, floors and fallback branches aligned with those CPU goldens.
//! Perceptual roughness r maps to GGX width alpha = r*r; variance is alpha*alpha.
//!
//! Geometric AA uses caller-supplied, same-frame normal differences per pixel
//! (fragment derivatives or compute finite differences in a consistent space).
//! On geometric normals it captures curvature/silhouette aliasing, NOT hidden
//! sub-texel normal-map variance. Toksvig instead captures normal-map minification
//! variance using the length of the filtered normal BEFORE renormalization.
//! Neither path reconstructs detail absent from its input.
//!
//! The additive Toksvig variance is a high-gloss approximation; it must not be
//! substituted for the exact shininess-domain factor in `toksvig_roughness`.

/// GGX width floor shared with the CPU GGX lobe.
const SPEC_AA_MIN_ALPHA: f32 = 1.0e-3;
/// Tokuyoshi-Kaplanyan projected normal-variance scale.
const SPEC_AA_SIGMA2: f32 = 0.15915494;
/// Default cap on additional geometric GGX variance.
const SPEC_AA_KAPPA_MAX: f32 = 0.18;
/// Minimum averaged-normal length; prevents division by zero.
const TOKSVIG_MIN_LEN: f32 = 1.0e-4;
/// Maximum Blinn-Phong shininess used by the CPU golden.
const TOKSVIG_MAX_SHININESS: f32 = 1.0e6;

/// Scalar finite predicate, matching the inline SSR mirror's isfinite idiom.
fn spec_aa_is_finite(x: f32) -> bool {
    return (x - x) == 0.0;
}

/// Perceptual roughness -> GGX alpha, not alpha squared; mirrors ggx_lobe.rs.
fn spec_aa_roughness_to_alpha(roughness: f32) -> f32 {
    let r = clamp(roughness, 0.0, 1.0);
    return max(r * r, SPEC_AA_MIN_ALPHA);
}

/// Squared normal-difference length. Any non-finite component discards the
/// entire vector, exactly as the CPU's Vec3::is_finite guard does.
fn spec_aa_safe_sq_len(v: vec3<f32>) -> f32 {
    let finite = spec_aa_is_finite(v.x) && spec_aa_is_finite(v.y) && spec_aa_is_finite(v.z);
    if (!finite) {
        return 0.0;
    }
    return max(dot(v, v), 0.0);
}

/// Projected screen-space normal variance: scale * (|ddx|^2 + |ddy|^2).
/// Non-finite scale or final variance contributes zero; negative scale is zero.
fn spec_aa_screen_space_variance(ddx_normal: vec3<f32>, ddy_normal: vec3<f32>, scale: f32) -> f32 {
    var s = 0.0;
    if (spec_aa_is_finite(scale)) {
        s = max(scale, 0.0);
    }
    let sum = spec_aa_safe_sq_len(ddx_normal) + spec_aa_safe_sq_len(ddy_normal);
    let variance = s * sum;
    if (spec_aa_is_finite(variance)) {
        return max(variance, 0.0);
    }
    return 0.0;
}

/// Additional geometric GGX variance min(2*variance, max_kernel).
/// Both input variance and cap are sanitized and floored non-negative.
fn spec_aa_kernel_roughness_sq(variance: f32, max_kernel: f32) -> f32 {
    var v = 0.0;
    if (spec_aa_is_finite(variance)) {
        v = max(variance, 0.0);
    }
    var cap = 0.0;
    if (spec_aa_is_finite(max_kernel)) {
        cap = max(max_kernel, 0.0);
    }
    return min(2.0 * v, cap);
}

/// Input and output are alpha squared. Adds capped geometric variance and
/// clamps to [MIN_ALPHA^2, 1]; overflow falls back to the clamped base.
fn spec_aa_filter_alpha_sq(base_alpha_sq: f32, variance: f32, max_kernel: f32) -> f32 {
    var base = 0.0;
    if (spec_aa_is_finite(base_alpha_sq)) {
        base = max(base_alpha_sq, 0.0);
    }
    let eff = base + spec_aa_kernel_roughness_sq(variance, max_kernel);
    let floor = SPEC_AA_MIN_ALPHA * SPEC_AA_MIN_ALPHA;
    if (spec_aa_is_finite(eff)) {
        return clamp(eff, floor, 1.0);
    }
    return clamp(base, floor, 1.0);
}

/// Geometric delta-alpha-squared from normal derivatives, ready to add to a
/// caller's base variance; scale and max_kernel are in the domains above.
fn spec_aa_delta_alpha_sq_from_derivatives(ddx_normal: vec3<f32>, ddy_normal: vec3<f32>, scale: f32, max_kernel: f32) -> f32 {
    let variance = spec_aa_screen_space_variance(ddx_normal, ddy_normal, scale);
    return spec_aa_kernel_roughness_sq(variance, max_kernel);
}

/// Geometric AA in perceptual roughness space. The fourth root converts
/// filtered alpha squared back to roughness; defaults are SIGMA2/KAPPA_MAX.
fn spec_aa_geometric_specular_aa_roughness(base_roughness: f32, ddx_normal: vec3<f32>, ddy_normal: vec3<f32>, scale: f32, max_kernel: f32) -> f32 {
    let alpha = spec_aa_roughness_to_alpha(base_roughness);
    let variance = spec_aa_screen_space_variance(ddx_normal, ddy_normal, scale);
    let eff_alpha_sq = spec_aa_filter_alpha_sq(alpha * alpha, variance, max_kernel);
    let roughness = sqrt(sqrt(max(eff_alpha_sq, 0.0)));
    return clamp(roughness, 0.0, 1.0);
}

/// Averaged-normal length -> [MIN_LEN, 1]; non-finite means maximally noisy.
fn toksvig_sanitize_len(avg_normal_len: f32) -> f32 {
    if (spec_aa_is_finite(avg_normal_len)) {
        return clamp(avg_normal_len, TOKSVIG_MIN_LEN, 1.0);
    }
    return TOKSVIG_MIN_LEN;
}

/// Blinn-Phong shininess -> [0, MAX_SHININESS]; non-finite becomes zero.
fn toksvig_sanitize_shininess(shininess: f32) -> f32 {
    if (spec_aa_is_finite(shininess)) {
        return clamp(shininess, 0.0, TOKSVIG_MAX_SHININESS);
    }
    return 0.0;
}

/// Exact Toksvig factor ft = len / (len + s*(1-len)), in [0, 1].
/// The length must come from a mip-averaged normal before normalization.
fn toksvig_factor(avg_normal_len: f32, shininess: f32) -> f32 {
    let len = toksvig_sanitize_len(avg_normal_len);
    let s = toksvig_sanitize_shininess(shininess);
    let denom = max(len + s * (1.0 - len), TOKSVIG_MIN_LEN);
    let ft = len / denom;
    if (spec_aa_is_finite(ft)) {
        return clamp(ft, 0.0, 1.0);
    }
    return 1.0;
}

/// Reduced Blinn-Phong shininess s_eff = ft*s, not a GGX width.
fn toksvig_effective_shininess(base_shininess: f32, avg_normal_len: f32) -> f32 {
    let s = toksvig_sanitize_shininess(base_shininess);
    let ft = toksvig_factor(avg_normal_len, s);
    return clamp(ft * s, 0.0, TOKSVIG_MAX_SHININESS);
}

/// GGX alpha -> shininess s = 2/alpha^2 - 2 (slope-variance bridge).
/// Preserve the CPU alpha floor and non-finite-result fallback, not a new
/// finite-input guard: positive infinite alpha maps to zero shininess.
fn toksvig_shininess_from_alpha(alpha: f32) -> f32 {
    let a = max(alpha, SPEC_AA_MIN_ALPHA);
    let s = 2.0 / (a * a) - 2.0;
    if (spec_aa_is_finite(s)) {
        return clamp(s, 0.0, TOKSVIG_MAX_SHININESS);
    }
    return TOKSVIG_MAX_SHININESS;
}

/// Sanitized shininess -> GGX alpha = sqrt(2/(s+2)), floored at MIN_ALPHA.
fn toksvig_alpha_from_shininess(shininess: f32) -> f32 {
    let s = toksvig_sanitize_shininess(shininess);
    let a = sqrt(max(2.0 / (s + 2.0), 0.0));
    return max(a, SPEC_AA_MIN_ALPHA);
}

/// High-gloss approximate additive variance delta-alpha^2 = 2*(1-len)/len.
/// This is not the exact factor outside the high-shininess limit.
fn toksvig_delta_alpha_sq(avg_normal_len: f32) -> f32 {
    let len = toksvig_sanitize_len(avg_normal_len);
    let delta = 2.0 * (1.0 - len) / len;
    if (spec_aa_is_finite(delta)) {
        return max(delta, 0.0);
    }
    return 0.0;
}

/// Input is base alpha (NOT alpha squared); output is filtered alpha squared:
/// alpha_eff^2 = alpha^2 + 2*(1-len)/len. Finite sums clamp to [MIN_ALPHA^2, 1].
/// The golden returns base_sq unchanged on overflow, even when non-finite;
/// deliberately preserve that fallback rather than promising finite output.
fn toksvig_effective_alpha_sq_from_len(base_alpha: f32, avg_normal_len: f32) -> f32 {
    let a = max(base_alpha, SPEC_AA_MIN_ALPHA);
    let base_sq = a * a;
    let eff = base_sq + toksvig_delta_alpha_sq(avg_normal_len);
    let floor = SPEC_AA_MIN_ALPHA * SPEC_AA_MIN_ALPHA;
    if (spec_aa_is_finite(eff)) {
        return clamp(eff, floor, 1.0);
    }
    return base_sq;
}

/// Perceptual roughness -> alpha -> s -> ft*s -> alpha -> roughness.
/// Uses the exact shininess-domain factor, not the additive approximation.
fn toksvig_roughness(base_roughness: f32, avg_normal_len: f32) -> f32 {
    let alpha = spec_aa_roughness_to_alpha(base_roughness);
    let shininess = toksvig_shininess_from_alpha(alpha);
    let eff_shininess = toksvig_effective_shininess(shininess, avg_normal_len);
    let eff_alpha = toksvig_alpha_from_shininess(eff_shininess);
    let roughness = sqrt(max(eff_alpha, 0.0));
    return clamp(roughness, 0.0, 1.0);
}

/// Coarser of two perceptual roughnesses; clamp each to [0, 1] and replace
/// non-finite inputs with zero before comparing. Mirrors combine_roughness.
fn toksvig_combine_roughness(a: f32, b: f32) -> f32 {
    var ca = 0.0;
    if (spec_aa_is_finite(a)) {
        ca = clamp(a, 0.0, 1.0);
    }
    var cb = 0.0;
    if (spec_aa_is_finite(b)) {
        cb = clamp(b, 0.0, 1.0);
    }
    return max(ca, cb);
}
