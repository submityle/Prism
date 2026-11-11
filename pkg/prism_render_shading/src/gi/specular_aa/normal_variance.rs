//! Geometric specular anti-aliasing from screen-space normal derivatives —
//! CPU golden reference (Kaplanyan 2016 / Tokuyoshi–Kaplanyan 2019).
//!
//! A curved or bumpy surface packs many micro-orientations into a single
//! shaded pixel.  Under motion this sub-pixel normal distribution is sampled
//! at ever-shifting points, so a tight specular lobe flickers between frames.
//! The geometric specular AA filter estimates the *screen-space normal
//! variance* `σ²` directly from the per-pixel partial derivatives of the
//! shading normal and widens the GGX lobe just enough to cover it, trading a
//! slightly softer highlight for a stable one.
//!
//! Given the screen-space derivatives `∂n/∂x` and `∂n/∂y` of the (interpolated,
//! renormalised) shading normal, the projected variance estimate is
//!
//! ```text
//! σ² = SIGMA2 · ( ⟨∂n/∂x, ∂n/∂x⟩ + ⟨∂n/∂y, ∂n/∂y⟩ ),
//! ```
//!
//! the kernel roughness is `κ = 2σ²`, and the filtered GGX width follows the
//! additive-variance rule
//!
//! ```text
//! alpha'² = clamp( alpha² + min(2σ², KAPPA_MAX),  0, 1 ).
//! ```
//!
//! The `min(·, KAPPA_MAX)` clamp (Tokuyoshi–Kaplanyan's `κ` cap) bounds how
//! much a silhouette or normal-map discontinuity may blur the lobe, preventing
//! grazing edges from turning mirror-like materials fully rough.  The default
//! constants `SIGMA2 = 0.15915494` and `KAPPA_MAX = 0.18` are the values
//! recommended by Tokuyoshi & Kaplanyan 2019.
//!
//! # Conventions
//! * `no_std`: allocation-free; only [`prism_math::Vec3`] (the normal-derivative
//!   vectors) is imported.  No transcendental functions are required, so
//!   [`prism_math::ops`] is *not* imported; the inherent `f32::sqrt` suffices.
//! * `∂n/∂x`, `∂n/∂y` are world- or view-space normal differences per pixel;
//!   only their squared lengths enter the estimate, so the frame choice is
//!   irrelevant as long as it is consistent.
//! * Perceptual `roughness ∈ [0, 1]` maps to the GGX width `alpha = roughness²`
//!   through [`roughness_to_alpha`]; this module never re-derives it.
//! * All inputs are sanitised: non-finite derivatives contribute zero variance,
//!   the scale and cap are floored non-negative, and every result is finite and
//!   clamped to a physical range (never `NaN`/`inf`).
//!
//! # References
//! * Anton Kaplanyan et al. 2016, *Filtering Distributions of Normals for
//!   Shading Antialiasing* (HPG) — the screen-space normal-variance filter.
//! * Yusuke Tokuyoshi & Anton Kaplanyan 2019, *Improved Geometric Specular
//!   Antialiasing* (I3D) — the projected-variance estimate and the `κ` cap used
//!   here, including the recommended `SIGMA2`/`KAPPA_MAX` constants.

use crate::gi::spec_gi::ggx_lobe::{roughness_to_alpha, MIN_ALPHA};
use prism_math::Vec3;

/// Default projection constant `SIGMA2` scaling the summed squared normal
/// derivatives into a variance (Tokuyoshi–Kaplanyan 2019, ≈ `1/(2π)`).
pub const DEFAULT_SIGMA2: f32 = 0.159_154_94;

/// Default kernel-roughness cap `KAPPA_MAX` bounding `min(2σ², κ)`
/// (Tokuyoshi–Kaplanyan 2019).
pub const DEFAULT_KAPPA_MAX: f32 = 0.18;

/// Explicit normal RGB + projected-variance alpha texture semantic. Ordinary
/// normal records (3) never interpret alpha. Mirrors `material_sample.wesl`;
/// this is a semantic identity, not a texture/sampler heap slot.
pub const SEMANTIC_NORMAL_VARIANCE: u32 = 11;

/// Whether a record supplies the base tangent-space normal (last record wins).
pub const fn is_normal_semantic(semantic: u32) -> bool {
    semantic == crate::SEMANTIC_NORMAL || semantic == SEMANTIC_NORMAL_VARIANCE
}

/// Decode an explicitly tagged, linear alpha payload. Missing/invalid payloads
/// contribute zero. The bound matches the WESL finite guard, including on Metal.
pub fn normal_variance_from_texel(semantic: u32, texel: [f32; 4]) -> f32 {
    let variance = texel[3];
    if semantic == SEMANTIC_NORMAL_VARIANCE && variance >= 0.0 && variance < 1.0e37 {
        variance
    } else {
        0.0
    }
}

/// Produce one linear RGBA normal/variance texel from the source footprint.
///
/// RGB is the caller's representative normal at this mip. Alpha stores the
/// projected second moment `0.5 * mean(|n_i - n_center|²)`, after decoding and
/// applying the material's actual `normal_scale` to every normal. Thus arbitrary
/// scale is supported without approximating variance by `scale²`. The caller
/// must use that same scale at runtime, and must not bake this correction into
/// roughness as well. Footprints may cover any number of source texels, but must
/// be supplied from the original detail (not renormalized previous mip RGB).
///
/// Publish with [`SEMANTIC_NORMAL_VARIANCE`] in `GpuMaterialTexture.semantic`;
/// sample RGB and alpha together, with identical UV/LOD/filtering. Use a linear
/// format retaining alpha (floating-point when values exceed UNORM range), not
/// BC5 or sRGB. No heap slot, texture dimensions or mip count is prescribed.
pub fn bake_normal_variance_texel(
    center: [f32; 3],
    footprint: &[[f32; 3]],
    normal_scale: f32,
) -> Option<[f32; 4]> {
    fn decode(texel: [f32; 3], scale: f32) -> Option<Vec3> {
        if !texel
            .into_iter()
            .all(|v| v.is_finite() && (0.0..=1.0).contains(&v))
        {
            return None;
        }
        let raw = Vec3::new(
            (texel[0] * 2.0 - 1.0) * scale,
            (texel[1] * 2.0 - 1.0) * scale,
            texel[2] * 2.0 - 1.0,
        );
        if !raw.is_finite() || !raw.length_squared().is_finite() || raw.length_squared() <= 1.0e-12
        {
            return None;
        }
        Some(Vec3::from_array(crate::decode_tangent_normal(texel, scale)))
    }
    if !normal_scale.is_finite() || footprint.is_empty() {
        return None;
    }
    let center_normal = decode(center, normal_scale)?;
    // Average incrementally to avoid a sum overflowing for large footprints.
    let mut moment = 0.0;
    for (index, texel) in footprint.iter().enumerate() {
        let difference = decode(*texel, normal_scale)? - center_normal;
        moment += (0.5 * difference.length_squared() - moment) / (index + 1) as f32;
    }
    Some([center[0], center[1], center[2], moment])
}

/// Explicit offline fallback for one sampled normal-map binding.
///
/// The variance is projected normal variance `σ²`, in the same units as
/// [`screen_space_variance`], NOT kernel roughness `Δα² = 2σ²`. The producer
/// must supply a value for the same UV/LOD, normal scale and normal transform
/// as the effective normal sample. Do not also bake this correction into the
/// material's roughness: that would apply the variance twice.
///
/// `binding_index` addresses the CPU resolve's sampled texture slice (not a
/// global texture-heap slot). Explicit association prevents a fallback for an
/// overwritten normal record from affecting the final normal. No texture alpha
/// channel is implicitly interpreted as variance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OfflineNormalVariance {
    binding_index: usize,
    projected_variance: f32,
}

impl OfflineNormalVariance {
    /// Rejects malformed offline data instead of turning it into a valid zero.
    pub fn new(binding_index: usize, projected_variance: f32) -> Option<Self> {
        if !projected_variance.is_finite() || projected_variance < 0.0 {
            return None;
        }
        Some(Self {
            binding_index,
            projected_variance,
        })
    }

    pub fn binding_index(self) -> usize {
        self.binding_index
    }

    pub fn projected_variance(self) -> f32 {
        self.projected_variance
    }
}

/// Returns the squared length of a normal-derivative vector, treating any
/// non-finite component as a zero contribution.
#[inline]
fn safe_sq_len(v: Vec3) -> f32 {
    if v.is_finite() {
        v.dot(v).max(0.0)
    } else {
        0.0
    }
}

/// Projected screen-space normal variance
/// `σ² = scale · (⟨∂n/∂x, ∂n/∂x⟩ + ⟨∂n/∂y, ∂n/∂y⟩)`.
///
/// `scale` is the `SIGMA2` projection constant (see [`DEFAULT_SIGMA2`]); it is
/// floored non-negative.  This is the Tokuyoshi 2019 "screen-space variance"
/// estimate: it rises with surface curvature and normal-map detail and is `0`
/// on a locally flat, static surface.
#[inline]
pub fn screen_space_variance(ddx_normal: Vec3, ddy_normal: Vec3, scale: f32) -> f32 {
    let s = if scale.is_finite() {
        scale.max(0.0)
    } else {
        0.0
    };
    let sum = safe_sq_len(ddx_normal) + safe_sq_len(ddy_normal);
    let variance = s * sum;
    if variance.is_finite() {
        variance.max(0.0)
    } else {
        0.0
    }
}

/// Kernel roughness `κ = min(2σ², max_kernel)` — the extra GGX variance the
/// filter is allowed to inject.
///
/// `max_kernel` is the `KAPPA_MAX` cap (see [`DEFAULT_KAPPA_MAX`]); it is
/// floored non-negative.  The `min` prevents discontinuities from blurring the
/// lobe without bound.
#[inline]
pub fn kernel_roughness_sq(variance: f32, max_kernel: f32) -> f32 {
    let v = if variance.is_finite() {
        variance.max(0.0)
    } else {
        0.0
    };
    let cap = if max_kernel.is_finite() {
        max_kernel.max(0.0)
    } else {
        0.0
    };
    (2.0 * v).min(cap)
}

/// Adds the capped kernel variance to a base GGX variance:
/// `alpha'² = clamp(alpha² + min(2σ², max_kernel), MIN_ALPHA², 1)`.
///
/// Returns the filtered `alpha'²` as a valid, finite GGX width that is never
/// sharper than the base.
#[inline]
pub fn filter_alpha_sq(base_alpha_sq: f32, variance: f32, max_kernel: f32) -> f32 {
    let base = if base_alpha_sq.is_finite() {
        base_alpha_sq.max(0.0)
    } else {
        0.0
    };
    let eff = base + kernel_roughness_sq(variance, max_kernel);
    let floor = MIN_ALPHA * MIN_ALPHA;
    if eff.is_finite() {
        eff.clamp(floor, 1.0)
    } else {
        base.clamp(floor, 1.0)
    }
}

/// Closed-form additional GGX variance `Δα² = min(2σ², max_kernel)` contributed
/// by the screen-space normal derivatives.
///
/// Convenience wrapper that chains [`screen_space_variance`] and
/// [`kernel_roughness_sq`] so callers can add the result to their own `alpha²`.
#[inline]
pub fn delta_alpha_sq_from_derivatives(
    ddx_normal: Vec3,
    ddy_normal: Vec3,
    scale: f32,
    max_kernel: f32,
) -> f32 {
    let variance = screen_space_variance(ddx_normal, ddy_normal, scale);
    kernel_roughness_sq(variance, max_kernel)
}

/// High-level helper: filters a perceptual `base_roughness` using the
/// screen-space normal derivatives, returning the anti-aliased perceptual
/// roughness in `[0, 1]`.
///
/// Pipeline: `roughness → alpha² → + min(2σ², max_kernel) → alpha'² →
/// roughness`.  Because the extra variance is non-negative, the output is
/// always `≥` the input (the filter only softens).
#[inline]
pub fn geometric_specular_aa_roughness(
    base_roughness: f32,
    ddx_normal: Vec3,
    ddy_normal: Vec3,
    scale: f32,
    max_kernel: f32,
) -> f32 {
    let alpha = roughness_to_alpha(base_roughness);
    let variance = screen_space_variance(ddx_normal, ddy_normal, scale);
    let eff_alpha_sq = filter_alpha_sq(alpha * alpha, variance, max_kernel);
    // roughness = (α²)^(1/4) since alpha = roughness² and alpha² is the input.
    let roughness = eff_alpha_sq.max(0.0).sqrt().sqrt();
    roughness.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1.0e-6;

    #[test]
    fn offline_normal_variance_baker_uses_the_actual_scaled_footprint() {
        let center = [0.5, 0.5, 1.0];
        let footprint = [[0.8, 0.5, 0.9], [0.2, 0.5, 0.9]];
        for scale in [0.0, 0.5, 1.0, 2.0] {
            let packed = bake_normal_variance_texel(center, &footprint, scale).unwrap();
            assert_eq!(&packed[..3], &center);
            let n = Vec3::from_array(crate::decode_tangent_normal(footprint[0], scale));
            let expected = 0.5 * (n - Vec3::Z).length_squared();
            assert!((packed[3] - expected).abs() < EPS);
            assert_eq!(
                normal_variance_from_texel(SEMANTIC_NORMAL_VARIANCE, packed),
                packed[3]
            );
            assert_eq!(
                normal_variance_from_texel(crate::SEMANTIC_NORMAL, packed),
                0.0
            );
            // σ² reaches the roughness kernel exactly once, not squared or
            // multiplied by the derivative projection constant a second time.
            assert_eq!(
                kernel_roughness_sq(packed[3], DEFAULT_KAPPA_MAX),
                (2.0 * expected).min(DEFAULT_KAPPA_MAX)
            );
        }
        assert_eq!(
            bake_normal_variance_texel(center, &[center; 4], 2.0).unwrap()[3],
            0.0
        );
    }

    #[test]
    fn offline_normal_variance_baker_and_payload_reject_invalid_input() {
        let center = [0.5, 0.5, 1.0];
        assert!(bake_normal_variance_texel(center, &[], 1.0).is_none());
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(bake_normal_variance_texel(center, &[center], invalid).is_none());
            assert!(bake_normal_variance_texel([invalid, 0.5, 1.0], &[center], 1.0).is_none());
            assert!(bake_normal_variance_texel(center, &[[0.5, invalid, 1.0]], 1.0).is_none());
        }
        assert!(bake_normal_variance_texel(center, &[[0.5; 3]], 1.0).is_none());
        assert!(bake_normal_variance_texel(center, &[[1.1, 0.5, 1.0]], 1.0).is_none());
        for alpha in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.01, 1.0e37] {
            assert_eq!(
                normal_variance_from_texel(SEMANTIC_NORMAL_VARIANCE, [0.5, 0.5, 1.0, alpha]),
                0.0
            );
        }
    }

    #[test]
    fn offline_normal_variance_production_sampler_contract_validates() {
        use alloc::{format, string::String};
        // Validate exact production structs/helpers, not a hand-reimplemented
        // shader twin. Resource declarations are deliberately excluded: this
        // is a source/type gate, not a pipeline-layout or device execution gate.
        const SAMPLE: &str = prism_render_material::shaders::MATERIAL_SAMPLE_SHADER_SOURCE;
        const MATERIAL: &str = prism_render_material::shaders::MATERIAL_SHADER_SOURCE;
        fn declaration<'a>(source: &'a str, prefix: &str) -> &'a str {
            let start = source.find(prefix).unwrap();
            let open = start + source[start..].find('{').unwrap();
            let mut depth = 0;
            for (offset, c) in source[open..].char_indices() {
                match c {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            return &source[start..open + offset + 1];
                        }
                    }
                    _ => {}
                }
            }
            panic!("unterminated declaration {prefix}");
        }
        assert!(SAMPLE.contains(&format!(
            "const SEMANTIC_NORMAL_VARIANCE: u32 = {SEMANTIC_NORMAL_VARIANCE}u;"
        )));
        let mut source = String::new();
        for line in SAMPLE.lines().filter(|line| {
            line.starts_with("const SEMANTIC_") || line.starts_with("const NORMAL_EPSILON")
        }) {
            source.push_str(line);
            source.push('\n');
        }
        for name in [
            "struct PrismMaterialTexture ",
            "struct PrismSurfaceParameters ",
        ] {
            source.push_str(declaration(MATERIAL, name));
            source.push('\n');
        }
        source.push_str(declaration(SAMPLE, "struct SampledMaterial "));
        for name in [
            "srgb_channel_to_linear",
            "srgb_to_linear",
            "decode_tangent_normal",
            "sampled_material_defaults",
            "material_default_texture_texel",
            "material_is_normal_semantic",
            "material_offline_normal_variance",
            "fail_closed_sampled_material",
            "fold_material_texel",
        ] {
            source.push_str(declaration(SAMPLE, &format!("fn {name}(")));
            source.push('\n');
        }
        source.push_str(r#"
@group(0) @binding(0) var<storage, read_write> result: array<vec4<f32>>;
@compute @workgroup_size(1) fn check_contract() {
    var parameters: PrismSurfaceParameters;
    parameters.normal_scale = 1.0;
    var record: PrismMaterialTexture;
    record.semantic = SEMANTIC_NORMAL_VARIANCE;
    let tagged = fold_material_texel(record, vec4<f32>(0.5, 0.5, 1.0, 0.02), 1.0, sampled_material_defaults(parameters));
    record.semantic = SEMANTIC_NORMAL;
    let reset = fold_material_texel(record, vec4<f32>(0.5, 0.5, 1.0, 1.0), 1.0, tagged);
    result[0] = vec4<f32>(tagged.offline_normal_variance, reset.offline_normal_variance, material_default_texture_texel(SEMANTIC_NORMAL_VARIANCE).w, select(0.0, 1.0, material_is_normal_semantic(SEMANTIC_NORMAL_VARIANCE)));
}
"#);
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("production offline-variance sampler must type-check");
    }

    #[test]
    fn offline_normal_variance_rejects_malformed_data_and_preserves_units() {
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.01] {
            assert!(OfflineNormalVariance::new(3, invalid).is_none());
        }
        let zero = OfflineNormalVariance::new(3, 0.0).unwrap();
        assert_eq!(zero.projected_variance(), 0.0);
        let offline = OfflineNormalVariance::new(3, 0.02).unwrap();
        assert_eq!(offline.binding_index(), 3);
        assert_eq!(offline.projected_variance(), 0.02);
        assert_eq!(
            kernel_roughness_sq(offline.projected_variance(), DEFAULT_KAPPA_MAX),
            0.04
        );
        let large = OfflineNormalVariance::new(3, f32::MAX).unwrap();
        assert_eq!(
            kernel_roughness_sq(large.projected_variance(), DEFAULT_KAPPA_MAX),
            DEFAULT_KAPPA_MAX
        );
    }

    #[test]
    fn flat_surface_adds_no_variance() {
        let v = screen_space_variance(Vec3::ZERO, Vec3::ZERO, DEFAULT_SIGMA2);
        assert!(v.abs() < EPS);
        let r = geometric_specular_aa_roughness(
            0.2,
            Vec3::ZERO,
            Vec3::ZERO,
            DEFAULT_SIGMA2,
            DEFAULT_KAPPA_MAX,
        );
        assert!((r - 0.2).abs() < 1.0e-4, "r = {r}");
    }

    #[test]
    fn variance_scales_with_derivative_magnitude() {
        let small = screen_space_variance(
            Vec3::new(0.1, 0.0, 0.0),
            Vec3::new(0.0, 0.1, 0.0),
            DEFAULT_SIGMA2,
        );
        let large = screen_space_variance(
            Vec3::new(0.4, 0.0, 0.0),
            Vec3::new(0.0, 0.4, 0.0),
            DEFAULT_SIGMA2,
        );
        assert!(large > small);
        assert!(small > 0.0);
    }

    #[test]
    fn kappa_cap_bounds_kernel() {
        // Enormous variance must still be capped at max_kernel.
        let k = kernel_roughness_sq(1.0e3, DEFAULT_KAPPA_MAX);
        assert!((k - DEFAULT_KAPPA_MAX).abs() < EPS, "k = {k}");
    }

    #[test]
    fn uncapped_region_is_two_sigma_sq() {
        let v = 0.01_f32;
        let k = kernel_roughness_sq(v, DEFAULT_KAPPA_MAX);
        assert!((k - 2.0 * v).abs() < EPS, "k = {k}");
    }

    #[test]
    fn roughness_only_increases_and_is_bounded() {
        let ddx = Vec3::new(0.3, 0.1, 0.0);
        let ddy = Vec3::new(0.1, 0.3, 0.0);
        for &r in &[0.0_f32, 0.05, 0.2, 0.5, 0.9, 1.0] {
            let out =
                geometric_specular_aa_roughness(r, ddx, ddy, DEFAULT_SIGMA2, DEFAULT_KAPPA_MAX);
            assert!(out >= r - 1.0e-4, "r {r} out {out}");
            assert!((0.0..=1.0).contains(&out));
        }
    }

    #[test]
    fn filter_alpha_sq_clamped_to_unit() {
        // Base already at the ceiling plus kernel must saturate at 1, not blow up.
        let out = filter_alpha_sq(0.95, 10.0, DEFAULT_KAPPA_MAX);
        assert!((0.0..=1.0).contains(&out));
    }

    #[test]
    fn delta_matches_pipeline() {
        let ddx = Vec3::new(0.2, 0.05, 0.01);
        let ddy = Vec3::new(0.03, 0.22, 0.0);
        let variance = screen_space_variance(ddx, ddy, DEFAULT_SIGMA2);
        let direct = delta_alpha_sq_from_derivatives(ddx, ddy, DEFAULT_SIGMA2, DEFAULT_KAPPA_MAX);
        let staged = kernel_roughness_sq(variance, DEFAULT_KAPPA_MAX);
        assert!((direct - staged).abs() < EPS);
    }

    #[test]
    fn defends_against_garbage() {
        let nan = Vec3::new(f32::NAN, 0.0, 0.0);
        let inf = Vec3::splat(f32::INFINITY);
        assert!(screen_space_variance(nan, inf, DEFAULT_SIGMA2).is_finite());
        assert!(kernel_roughness_sq(f32::NAN, f32::NAN).is_finite());
        let out = geometric_specular_aa_roughness(f32::NAN, nan, inf, f32::NAN, f32::NAN);
        assert!((0.0..=1.0).contains(&out));
    }

    #[test]
    fn negative_scale_is_floored() {
        let v = screen_space_variance(Vec3::ONE, Vec3::ONE, -5.0);
        assert!(v.abs() < EPS);
    }
}
