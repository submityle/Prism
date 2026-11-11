//! Backend-neutral texture-sampling math for the bindless material resolve.
//!
//! The GPU compute resolve (`material_sample.wesl`) samples each material's
//! bound textures from a bindless `binding_array` and folds the texels into the
//! analytic surface before shading.  The sandbox/container has no GPU, so the
//! numeric transforms that turn raw texels into shading inputs -- the sRGB
//! electro-optical decode, the tangent-space normal reconstruction and the
//! channel-wise modulation of the authored `PrismSurfaceParameters` -- live here
//! as a deterministic CPU golden.  `material_sample.wesl` mirrors every function
//! below builtin-for-builtin (`pow`/`dot`/`inverseSqrt`/`textureSampleLevel`),
//! so a green golden here plus a green WESL compile test pins both sides of the
//! contract; GPU numeric parity itself must be confirmed on real hardware.
//!
//! Texel color-space conventions match glTF 2.0 / `StandardMaterial`:
//!
//! * base-color and emissive textures are authored in sRGB and decoded to
//!   linear before modulating the (already linear) authored factors;
//! * the metallic-roughness texture is linear with roughness in G and metallic
//!   in B; occlusion is linear in R (either a dedicated occlusion texture or the
//!   R channel of a packed ORM texture);
//! * normal maps are linear, store a unit tangent-space normal remapped to
//!   `[0, 1]`, and are decoded with `n * 2 - 1`, XY scaled by `normal_scale`.

use crate::gi::specular_aa::normal_variance::normal_variance_from_texel;
use prism_math::ops;

/// Texture semantics, matching `TextureSemantic` (`#[repr(u32)]`) in
/// `prism_render_material::bevy_bridge` and the `SEMANTIC_*` constants in
/// `material_sample.wesl`.
pub const SEMANTIC_BASE_COLOR: u32 = 0;
/// Emissive color, sRGB-encoded.
pub const SEMANTIC_EMISSIVE: u32 = 1;
/// Packed metallic (B) / roughness (G), linear.
pub const SEMANTIC_METALLIC_ROUGHNESS: u32 = 2;
/// Tangent-space normal map, linear, `[0, 1]`-remapped.
pub const SEMANTIC_NORMAL: u32 = 3;
/// Linear tangent-space normal RGB with offline projected variance `σ²` in alpha.
/// The producer must bake for the runtime normal scale, without also baking
/// this correction into roughness. Shares the specular-AA semantic identity.
pub use crate::gi::specular_aa::normal_variance::SEMANTIC_NORMAL_VARIANCE;
/// Ambient occlusion, linear, in the R channel.
pub const SEMANTIC_OCCLUSION: u32 = 4;
/// Clear-coat strength, linear, in the R channel.
pub const SEMANTIC_CLEAR_COAT: u32 = 5;
/// Clear-coat roughness, linear, in the G channel.
pub const SEMANTIC_CLEAR_COAT_ROUGHNESS: u32 = 6;
/// Clear-coat tangent-space normal map, linear, `[0, 1]`-remapped.
pub const SEMANTIC_CLEAR_COAT_NORMAL: u32 = 7;
/// Dedicated toon face-shadow SDF consumer; it is not a surface-factor fold.
pub const SEMANTIC_FACE_SDF: u32 = prism_render_material::FACE_SHADOW_SDF_SEMANTIC;
/// Measured (tabulated) BRDF/BTF table (§17.6).
///
/// Not a fold input like the maps above: the row carries a whole *table*, which
/// the measured lobe reads as an atlas (width = `theta_h`, height = `theta_d` ×
/// `phi_d`) with integer texel loads. The tag exists so a material can bind the
/// table through the ordinary texture list and both lines find it the same way.
pub const SEMANTIC_MEASURED_BRDF: u32 = 9;
/// Dedicated region-illumination selector consumer; it is not a surface-factor fold.
pub const SEMANTIC_ILLUMINATION_MASK: u32 = prism_render_material::ILLUMINATION_MASK_SEMANTIC;

// The semantic bytes are an ABI shared with the material crate and every
// shader; pin the one the measured lobe scans for, so a drift is a build error.
const _: () = assert!(SEMANTIC_MEASURED_BRDF == prism_render_material::MEASURED_BRDF_SEMANTIC);
const _: () = assert!(SEMANTIC_FACE_SDF == prism_render_material::FACE_SHADOW_SDF_SEMANTIC);
const _: () =
    assert!(SEMANTIC_ILLUMINATION_MASK == prism_render_material::ILLUMINATION_MASK_SEMANTIC);

/// Explicit route for every semantic that can enter a material texture row.
/// Dedicated consumers are intentionally not folded here; an unknown value is
/// an ABI error and must fail closed rather than disappear into an identity
/// operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TextureSemanticRoute {
    SurfaceFold,
    DedicatedConsumer,
    Unsupported,
}

#[must_use]
pub const fn texture_semantic_route(semantic: u32) -> TextureSemanticRoute {
    match semantic {
        SEMANTIC_BASE_COLOR
        | SEMANTIC_EMISSIVE
        | SEMANTIC_METALLIC_ROUGHNESS
        | SEMANTIC_NORMAL
        | SEMANTIC_NORMAL_VARIANCE
        | SEMANTIC_OCCLUSION
        | SEMANTIC_CLEAR_COAT
        | SEMANTIC_CLEAR_COAT_ROUGHNESS
        | SEMANTIC_CLEAR_COAT_NORMAL => TextureSemanticRoute::SurfaceFold,
        SEMANTIC_FACE_SDF | SEMANTIC_MEASURED_BRDF | SEMANTIC_ILLUMINATION_MASK => {
            TextureSemanticRoute::DedicatedConsumer
        }
        _ => TextureSemanticRoute::Unsupported,
    }
}

/// An unknown semantic cannot be represented by the current CPU/WESL surface
/// accumulator. Callers that need a hard error can use
/// [`fold_material_texel_checked`]; the compatibility wrapper below converts
/// it to an explicit fail-closed sample.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UnsupportedTextureSemantic {
    pub semantic: u32,
}

/// Squared-length floor below which a decoded normal is treated as degenerate
/// and replaced by the geometric `+Z` tangent-space normal.  Mirrors the
/// `normalize_or` epsilon used across the shading reference and `1e-12` in the
/// WESL twin.
const NORMAL_EPSILON: f32 = 1e-12;

/// Decodes one sRGB-encoded channel to linear using the exact IEC 61966-2-1
/// piecewise transfer function (not the `pow(c, 2.2)` approximation).  Mirrors
/// `srgb_channel_to_linear` in `material_sample.wesl`.
#[must_use]
pub fn srgb_channel_to_linear(c: f32) -> f32 {
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ops::powf((c + 0.055) / 1.055, 2.4)
    }
}

/// Decodes an sRGB-encoded RGB triple to linear, channel by channel.  Mirrors
/// `srgb_to_linear` in `material_sample.wesl`.
#[must_use]
pub fn srgb_to_linear(c: [f32; 3]) -> [f32; 3] {
    [
        srgb_channel_to_linear(c[0]),
        srgb_channel_to_linear(c[1]),
        srgb_channel_to_linear(c[2]),
    ]
}

/// Decodes a `[0, 1]`-remapped tangent-space normal-map texel into a unit
/// tangent-space normal, scaling the XY (tangent/bitangent) components by
/// `normal_scale` before renormalizing.  A degenerate result collapses to the
/// geometric `+Z` normal.  Mirrors `decode_tangent_normal` in
/// `material_sample.wesl`.
#[must_use]
pub fn decode_tangent_normal(texel: [f32; 3], normal_scale: f32) -> [f32; 3] {
    let x = (texel[0] * 2.0 - 1.0) * normal_scale;
    let y = (texel[1] * 2.0 - 1.0) * normal_scale;
    let z = texel[2] * 2.0 - 1.0;
    let len_sq = x * x + y * y + z * z;
    if len_sq > NORMAL_EPSILON {
        let inv = len_sq.sqrt().recip();
        [x * inv, y * inv, z * inv]
    } else {
        [0.0, 0.0, 1.0]
    }
}

/// Authored, texture-independent surface factors, mirroring the subset of
/// `PrismSurfaceParameters` the bindless resolve modulates.  Values are already
/// in their shading space (base color / emissive linear; scalars in `[0, 1]`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaterialModulationParams {
    /// Linear base-color factor with alpha.
    pub base_color: [f32; 4],
    /// Linear emissive factor.
    pub emissive: [f32; 3],
    /// Metallic factor.
    pub metallic: f32,
    /// Perceptual (artist) roughness factor.
    pub perceptual_roughness: f32,
    /// Ambient-occlusion factor.
    pub ambient_occlusion: f32,
    /// Tangent-space normal-map XY scale.
    pub normal_scale: f32,
    /// Clear-coat strength factor.
    pub clearcoat: f32,
    /// Clear-coat perceptual roughness factor.
    pub clearcoat_roughness: f32,
}

impl Default for MaterialModulationParams {
    fn default() -> Self {
        Self {
            base_color: [1.0, 1.0, 1.0, 1.0],
            emissive: [0.0, 0.0, 0.0],
            metallic: 1.0,
            perceptual_roughness: 1.0,
            ambient_occlusion: 1.0,
            normal_scale: 1.0,
            clearcoat: 0.0,
            clearcoat_roughness: 0.0,
        }
    }
}

/// Texture-modulated surface inputs handed to the analytic BSDF.  Produced by
/// folding every bound texel over [`MaterialModulationParams`].  Mirrors
/// `SampledMaterial` in `material_sample.wesl` field for field.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampledMaterial {
    /// Linear base color after texture modulation.
    pub base_color: [f32; 3],
    /// Base-color alpha after texture modulation.
    pub base_alpha: f32,
    /// Linear emissive after texture modulation.
    pub emissive: [f32; 3],
    /// Metallic after texture modulation.
    pub metallic: f32,
    /// Perceptual roughness after texture modulation.
    pub perceptual_roughness: f32,
    /// Ambient occlusion after texture modulation.
    pub occlusion: f32,
    /// Decoded tangent-space normal; `+Z` when no normal map is bound.
    pub normal_tangent: [f32; 3],
    /// Whether a base-layer normal map contributed to `normal_tangent`.
    pub has_normal_map: bool,
    /// False only after an unsupported semantic was encountered. The wrapper
    /// also zeros the visible response so an invalid row cannot shade as an
    /// apparently valid material.
    pub texture_contract_valid: bool,
    /// Offline projected variance `σ²` from the last base-layer normal record.
    /// Zero without a tagged payload, or when its alpha is outside `[0, 1e37)`.
    /// Already baked for `normal_scale`; not kernel roughness or an extra term
    /// to add to a valid realtime normal-variance estimate.
    pub offline_normal_variance: f32,
    /// Clear-coat strength after texture modulation.
    pub clearcoat: f32,
    /// Clear-coat perceptual roughness after texture modulation.
    pub clearcoat_roughness: f32,
    /// Decoded clear-coat tangent-space normal; `+Z` when none is bound.
    pub clearcoat_normal_tangent: [f32; 3],
    /// Whether a clear-coat normal map contributed.
    pub has_clearcoat_normal: bool,
}

/// Seeds a [`SampledMaterial`] from the authored factors alone, before any
/// texture is folded in.  A material that binds no textures shades with exactly
/// these values, matching the identity behaviour of the bindless heap's default
/// white / flat-normal slots on the GPU.  Mirrors `sampled_material_defaults`.
#[must_use]
pub fn sampled_material_defaults(params: &MaterialModulationParams) -> SampledMaterial {
    SampledMaterial {
        base_color: [
            params.base_color[0],
            params.base_color[1],
            params.base_color[2],
        ],
        base_alpha: params.base_color[3],
        emissive: params.emissive,
        metallic: params.metallic,
        perceptual_roughness: params.perceptual_roughness,
        occlusion: params.ambient_occlusion,
        normal_tangent: [0.0, 0.0, 1.0],
        has_normal_map: false,
        texture_contract_valid: true,
        offline_normal_variance: 0.0,
        clearcoat: params.clearcoat,
        clearcoat_roughness: params.clearcoat_roughness,
        clearcoat_normal_tangent: [0.0, 0.0, 1.0],
        has_clearcoat_normal: false,
    }
}

/// Folds a single sampled texel into `sampled` according to its `semantic`.
/// Dedicated-consumer semantics are explicitly preserved for their own pass;
/// unknown semantics return an error instead of being silently dropped.
#[must_use]
pub fn fold_material_texel_checked(
    semantic: u32,
    texel: [f32; 4],
    normal_scale: f32,
    mut sampled: SampledMaterial,
) -> Result<SampledMaterial, UnsupportedTextureSemantic> {
    if !sampled.texture_contract_valid {
        return Ok(sampled);
    }
    let rgb = [texel[0], texel[1], texel[2]];
    match texture_semantic_route(semantic) {
        TextureSemanticRoute::SurfaceFold => match semantic {
            SEMANTIC_BASE_COLOR => {
                let linear = srgb_to_linear(rgb);
                sampled.base_color = [
                    sampled.base_color[0] * linear[0],
                    sampled.base_color[1] * linear[1],
                    sampled.base_color[2] * linear[2],
                ];
                sampled.base_alpha *= texel[3];
            }
            SEMANTIC_EMISSIVE => {
                let linear = srgb_to_linear(rgb);
                sampled.emissive = [
                    sampled.emissive[0] * linear[0],
                    sampled.emissive[1] * linear[1],
                    sampled.emissive[2] * linear[2],
                ];
            }
            SEMANTIC_METALLIC_ROUGHNESS => {
                sampled.perceptual_roughness *= texel[1];
                sampled.metallic *= texel[2];
            }
            SEMANTIC_OCCLUSION => {
                sampled.occlusion *= texel[0];
            }
            SEMANTIC_NORMAL | SEMANTIC_NORMAL_VARIANCE => {
                sampled.normal_tangent = decode_tangent_normal(rgb, normal_scale);
                sampled.has_normal_map = true;
                // Last base-normal record wins, including clearing an earlier
                // payload when an ordinary normal record replaces it.
                sampled.offline_normal_variance = normal_variance_from_texel(semantic, texel);
            }
            SEMANTIC_CLEAR_COAT => {
                sampled.clearcoat *= texel[0];
            }
            SEMANTIC_CLEAR_COAT_ROUGHNESS => {
                sampled.clearcoat_roughness *= texel[1];
            }
            SEMANTIC_CLEAR_COAT_NORMAL => {
                sampled.clearcoat_normal_tangent = decode_tangent_normal(rgb, normal_scale);
                sampled.has_clearcoat_normal = true;
            }
            _ => unreachable!("surface-fold route omitted a known semantic"),
        },
        TextureSemanticRoute::DedicatedConsumer => {}
        TextureSemanticRoute::Unsupported => {
            return Err(UnsupportedTextureSemantic { semantic });
        }
    }
    Ok(sampled)
}

fn fail_closed_sampled_material(mut sampled: SampledMaterial) -> SampledMaterial {
    sampled.base_color = [0.0, 0.0, 0.0];
    sampled.base_alpha = 0.0;
    sampled.emissive = [0.0, 0.0, 0.0];
    sampled.metallic = 0.0;
    sampled.perceptual_roughness = 1.0;
    sampled.occlusion = 0.0;
    sampled.normal_tangent = [0.0, 0.0, 1.0];
    sampled.has_normal_map = false;
    sampled.offline_normal_variance = 0.0;
    sampled.clearcoat = 0.0;
    sampled.clearcoat_roughness = 1.0;
    sampled.clearcoat_normal_tangent = [0.0, 0.0, 1.0];
    sampled.has_clearcoat_normal = false;
    sampled.texture_contract_valid = false;
    sampled
}

/// Compatibility entry point used by existing CPU resolve callers. It keeps
/// the stable return shape while making an unsupported row visibly invalid.
#[must_use]
pub fn fold_material_texel(
    semantic: u32,
    texel: [f32; 4],
    normal_scale: f32,
    sampled: SampledMaterial,
) -> SampledMaterial {
    match fold_material_texel_checked(semantic, texel, normal_scale, sampled) {
        Ok(sampled) => sampled,
        Err(_) => fail_closed_sampled_material(sampled),
    }
}

/// One bound texture's sampled result: its [`TextureSemantic`] discriminant and
/// the RGBA texel sampled at the surface UV.  On the GPU the texel comes from
/// `textureSampleLevel(material_texture_heap[record.index], ...)`; the CPU
/// golden accepts the already-sampled texel directly so the fold arithmetic is
/// literally the same code path on both sides.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampledTextureBinding {
    /// `TextureSemantic` discriminant (`SEMANTIC_*`).
    pub semantic: u32,
    /// RGBA texel sampled at the caller-resolved LOD in the texture's native
    /// encoding (sRGB for base-color/emissive, linear otherwise).
    pub texel: [f32; 4],
    /// Optional forward samples at the same texture record and LOD as
    /// [`Self::texel`], ordered as `[uv + dUV/dx, uv + dUV/dy]`.
    ///
    /// Only [`SEMANTIC_NORMAL`] consumes these samples. Keeping the footprint
    /// on the binding makes it impossible to pair a normal-map centre sample
    /// with neighbours from a different record or mip. `None` means the CPU
    /// caller cannot provide a valid one-pixel footprint, so variance is zero.
    pub normal_map_forward_texels: Option<[[f32; 4]; 2]>,
}

/// Folds every bound texture over the authored factors in order, returning the
/// modulated [`SampledMaterial`].  Seeds [`sampled_material_defaults`] then
/// applies [`fold_material_texel`] for each binding using the shared
/// `normal_scale`, mirroring the `texture_offset..texture_offset + texture_count`
/// loop in `shading_resolve.wesl` iteration-for-iteration.
#[must_use]
pub fn sample_material(
    params: &MaterialModulationParams,
    bindings: &[SampledTextureBinding],
) -> SampledMaterial {
    match sample_material_checked(params, bindings) {
        Ok(sampled) => sampled,
        Err(_) => fail_closed_sampled_material(sampled_material_defaults(params)),
    }
}

/// Checked form of [`sample_material`]. Unknown texture semantics are an ABI
/// error; dedicated-consumer semantics remain present for their own pass and
/// are intentionally not folded into the generic surface accumulator.
#[must_use]
pub fn sample_material_checked(
    params: &MaterialModulationParams,
    bindings: &[SampledTextureBinding],
) -> Result<SampledMaterial, UnsupportedTextureSemantic> {
    let mut sampled = sampled_material_defaults(params);
    for binding in bindings {
        sampled = fold_material_texel_checked(
            binding.semantic,
            binding.texel,
            params.normal_scale,
            sampled,
        )?;
    }
    Ok(sampled)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-5 + 1e-4 * b.abs()
    }

    fn approx3(a: [f32; 3], b: [f32; 3]) -> bool {
        approx(a[0], b[0]) && approx(a[1], b[1]) && approx(a[2], b[2])
    }

    #[test]
    fn srgb_decode_matches_piecewise_reference() {
        // Below the linear knee: pure division by 12.92.
        assert!(approx(srgb_channel_to_linear(0.0), 0.0));
        assert!(approx(srgb_channel_to_linear(0.04045), 0.04045 / 12.92));
        // Above the knee: the gamma segment. 0.5 sRGB -> ~0.2140 linear.
        assert!(approx(srgb_channel_to_linear(0.5), 0.214_041_14));
        // White is a fixed point.
        assert!(approx(srgb_channel_to_linear(1.0), 1.0));
    }

    #[test]
    fn srgb_decode_is_continuous_at_the_knee() {
        // The two branches must agree at the 0.04045 boundary or a normal map's
        // neighbouring texels would jump; parity with the WESL twin depends on
        // both using the same threshold.
        let below = 0.040_45_f32 / 12.92;
        let above = ops::powf((0.040_45 + 0.055) / 1.055, 2.4);
        assert!((below - above).abs() < 1e-4);
    }

    #[test]
    fn flat_normal_texel_decodes_to_plus_z() {
        // The heap's FLAT_NORMAL default slot is (0.5, 0.5, 1.0).
        assert!(approx3(
            decode_tangent_normal([0.5, 0.5, 1.0], 1.0),
            [0.0, 0.0, 1.0]
        ));
    }

    #[test]
    fn normal_decode_is_unit_length_and_scaled() {
        let n = decode_tangent_normal([1.0, 0.5, 0.75], 1.0);
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        assert!(approx(len, 1.0));
        // Full-scale +X in tangent space tips toward +X.
        assert!(n[0] > 0.0);
    }

    #[test]
    fn normal_scale_flattens_the_tangent_plane() {
        // normal_scale = 0 removes all XY perturbation, leaving +Z.
        assert!(approx3(
            decode_tangent_normal([1.0, 0.0, 0.5], 0.0),
            [0.0, 0.0, 1.0]
        ));
    }

    #[test]
    fn degenerate_normal_texel_falls_back_to_plus_z() {
        // A (0.5, 0.5, 0.5) texel decodes to the zero vector before renormalize.
        assert!(approx3(
            decode_tangent_normal([0.5, 0.5, 0.5], 1.0),
            [0.0, 0.0, 1.0]
        ));
    }

    #[test]
    fn offline_normal_variance_defaults_and_neutral_fallback_are_zero() {
        let params = MaterialModulationParams::default();
        let defaults = sampled_material_defaults(&params);
        assert_eq!(sample_material(&params, &[]), defaults);
        assert_eq!(defaults.offline_normal_variance, 0.0);
        assert!(!defaults.has_normal_map);

        // WESL's missing-resource fallback for semantic 11 has alpha zero,
        // unlike the legacy flat-normal slot, whose alpha is one.
        let sampled = fold_material_texel(
            SEMANTIC_NORMAL_VARIANCE,
            [0.5, 0.5, 1.0, 0.0],
            params.normal_scale,
            defaults,
        );
        assert_eq!(
            sampled,
            SampledMaterial {
                has_normal_map: true,
                ..defaults
            }
        );
    }

    #[test]
    fn tagged_normal_decodes_rgb_and_preserves_projected_variance_units() {
        let defaults = sampled_material_defaults(&MaterialModulationParams::default());
        // At scale 0.5, this RGB decodes to the unit normal (0.6, 0, 0.8).
        let sampled = fold_material_texel(
            SEMANTIC_NORMAL_VARIANCE,
            [0.875, 0.5, 0.75, 0.125],
            0.5,
            defaults,
        );
        assert!(approx3(sampled.normal_tangent, [0.6, 0.0, 0.8]));
        assert_eq!(sampled.offline_normal_variance, 0.125);
        assert!(sampled.has_normal_map);
        assert_eq!(sampled.perceptual_roughness, defaults.perceptual_roughness);

        // The guard is [0, 1e37), not UNORM, and alpha is already baked:
        // neither runtime normal_scale nor the kernel's factor of two applies.
        let below_limit = f32::from_bits(1.0e37_f32.to_bits() - 1);
        for variance in [0.0, -0.0, 0.125, 2.0, below_limit] {
            for scale in [0.0, 0.5, 1.0, 2.0] {
                let texel = [0.875, 0.5, 0.75, variance];
                let tagged = fold_material_texel(SEMANTIC_NORMAL_VARIANCE, texel, scale, defaults);
                let normal = fold_material_texel(SEMANTIC_NORMAL, texel, scale, defaults);
                assert_eq!(
                    tagged,
                    SampledMaterial {
                        offline_normal_variance: variance,
                        ..normal
                    }
                );
            }
        }
    }

    #[test]
    fn tagged_degenerate_normal_keeps_valid_variance_like_wesl() {
        let sampled = fold_material_texel(
            SEMANTIC_NORMAL_VARIANCE,
            [0.5, 0.5, 0.5, 0.125],
            1.0,
            sampled_material_defaults(&MaterialModulationParams::default()),
        );
        assert_eq!(sampled.normal_tangent, [0.0, 0.0, 1.0]);
        assert!(sampled.has_normal_map);
        assert_eq!(sampled.offline_normal_variance, 0.125);
    }

    #[test]
    fn tagged_normal_sanitizes_invalid_variance_and_replaces_previous_payload() {
        let before = fold_material_texel(
            SEMANTIC_NORMAL_VARIANCE,
            [0.5, 0.5, 1.0, 0.25],
            1.0,
            sampled_material_defaults(&MaterialModulationParams::default()),
        );
        for alpha in [
            f32::NAN,
            -0.25,
            -f32::MIN_POSITIVE,
            f32::NEG_INFINITY,
            f32::INFINITY,
            1.0e37,
            f32::MAX,
        ] {
            let sampled = fold_material_texel(
                SEMANTIC_NORMAL_VARIANCE,
                [0.875, 0.5, 0.75, alpha],
                0.5,
                before,
            );
            assert_eq!(sampled.offline_normal_variance, 0.0, "alpha={alpha:?}");
            assert!(approx3(sampled.normal_tangent, [0.6, 0.0, 0.8]));
            assert!(sampled.has_normal_map);
        }
    }

    #[test]
    fn ordinary_normal_ignores_alpha_and_clears_offline_variance() {
        let before = fold_material_texel(
            SEMANTIC_NORMAL_VARIANCE,
            [0.5, 0.5, 1.0, 0.25],
            1.0,
            sampled_material_defaults(&MaterialModulationParams::default()),
        );
        for alpha in [0.0, 0.125, 1.0, -0.25, f32::NAN, f32::INFINITY] {
            let sampled =
                fold_material_texel(SEMANTIC_NORMAL, [0.875, 0.5, 0.75, alpha], 0.5, before);
            assert_eq!(sampled.offline_normal_variance, 0.0);
            assert!(approx3(sampled.normal_tangent, [0.6, 0.0, 0.8]));
            assert!(sampled.has_normal_map);
        }
    }

    #[test]
    fn sample_material_last_base_normal_wins_for_normal_and_variance() {
        let params = MaterialModulationParams {
            normal_scale: 0.5,
            ..Default::default()
        };
        let bindings = [
            SampledTextureBinding {
                semantic: SEMANTIC_NORMAL_VARIANCE,
                texel: [0.875, 0.5, 0.75, 0.25],
                normal_map_forward_texels: None,
            },
            SampledTextureBinding {
                semantic: SEMANTIC_NORMAL,
                texel: [0.5, 0.5, 1.0, 1.0],
                normal_map_forward_texels: None,
            },
            SampledTextureBinding {
                semantic: SEMANTIC_NORMAL_VARIANCE,
                texel: [0.5, 0.875, 0.75, 0.125],
                normal_map_forward_texels: None,
            },
        ];
        let first = sample_material(&params, &bindings[..1]);
        assert!(approx3(first.normal_tangent, [0.6, 0.0, 0.8]));
        assert_eq!(first.offline_normal_variance, 0.25);
        let replaced = sample_material(&params, &bindings[..2]);
        assert_eq!(replaced.normal_tangent, [0.0, 0.0, 1.0]);
        assert_eq!(replaced.offline_normal_variance, 0.0);
        let last = sample_material(&params, &bindings);
        assert!(approx3(last.normal_tangent, [0.0, 0.6, 0.8]));
        assert_eq!(last.offline_normal_variance, 0.125);
        assert!(last.has_normal_map);
    }

    #[test]
    fn known_surface_semantics_preserve_base_normal_and_offline_variance() {
        let before = fold_material_texel(
            SEMANTIC_NORMAL_VARIANCE,
            [0.875, 0.5, 0.75, 0.125],
            0.5,
            sampled_material_defaults(&MaterialModulationParams::default()),
        );
        for semantic in [
            SEMANTIC_BASE_COLOR,
            SEMANTIC_EMISSIVE,
            SEMANTIC_METALLIC_ROUGHNESS,
            SEMANTIC_OCCLUSION,
            SEMANTIC_CLEAR_COAT,
            SEMANTIC_CLEAR_COAT_ROUGHNESS,
            SEMANTIC_CLEAR_COAT_NORMAL,
            SEMANTIC_MEASURED_BRDF,
        ] {
            let sampled = fold_material_texel(semantic, [0.5, 0.875, 0.75, 0.25], 0.5, before);
            assert_eq!(sampled.normal_tangent, before.normal_tangent);
            assert!(sampled.has_normal_map);
            assert_eq!(sampled.offline_normal_variance, 0.125);
        }
    }

    #[test]
    fn dedicated_consumer_semantics_are_explicitly_not_surface_folds() {
        let before = sampled_material_defaults(&MaterialModulationParams::default());
        for semantic in [
            SEMANTIC_FACE_SDF,
            SEMANTIC_MEASURED_BRDF,
            SEMANTIC_ILLUMINATION_MASK,
        ] {
            let after = fold_material_texel_checked(semantic, [0.1, 0.2, 0.3, 0.4], 1.0, before)
                .expect("dedicated consumers are valid ABI semantics");
            assert_eq!(after, before);
            assert_eq!(
                texture_semantic_route(semantic),
                TextureSemanticRoute::DedicatedConsumer
            );
        }
    }

    #[test]
    fn unknown_semantics_return_an_error_and_compatibility_path_fails_closed() {
        let params = MaterialModulationParams {
            base_color: [0.8, 0.4, 0.2, 0.9],
            emissive: [1.0, 2.0, 3.0],
            metallic: 0.3,
            perceptual_roughness: 0.6,
            ambient_occlusion: 0.7,
            clearcoat: 0.5,
            clearcoat_roughness: 0.25,
            ..Default::default()
        };
        let before = sampled_material_defaults(&params);
        let error = fold_material_texel_checked(4242, [0.1, 0.2, 0.3, 0.4], 1.0, before)
            .expect_err("unknown semantics must not be silently dropped");
        assert_eq!(error, UnsupportedTextureSemantic { semantic: 4242 });
        let after = fold_material_texel(4242, [0.1, 0.2, 0.3, 0.4], 1.0, before);
        assert!(!after.texture_contract_valid);
        assert_eq!(after.base_color, [0.0, 0.0, 0.0]);
        assert_eq!(after.base_alpha, 0.0);
        assert_eq!(after.occlusion, 0.0);
        assert_eq!(after.perceptual_roughness, 1.0);
        assert_eq!(after.clearcoat_roughness, 1.0);
    }

    #[test]
    fn cpu_route_and_wesl_source_share_all_texture_contract_semantics() {
        let source = prism_render_material::shaders::MATERIAL_SAMPLE_SHADER_SOURCE;
        for (name, semantic) in [
            ("SEMANTIC_BASE_COLOR", SEMANTIC_BASE_COLOR),
            ("SEMANTIC_EMISSIVE", SEMANTIC_EMISSIVE),
            ("SEMANTIC_METALLIC_ROUGHNESS", SEMANTIC_METALLIC_ROUGHNESS),
            ("SEMANTIC_NORMAL", SEMANTIC_NORMAL),
            ("SEMANTIC_NORMAL_VARIANCE", SEMANTIC_NORMAL_VARIANCE),
            ("SEMANTIC_OCCLUSION", SEMANTIC_OCCLUSION),
            ("SEMANTIC_CLEAR_COAT", SEMANTIC_CLEAR_COAT),
            (
                "SEMANTIC_CLEAR_COAT_ROUGHNESS",
                SEMANTIC_CLEAR_COAT_ROUGHNESS,
            ),
            ("SEMANTIC_CLEAR_COAT_NORMAL", SEMANTIC_CLEAR_COAT_NORMAL),
            ("SEMANTIC_FACE_SDF", SEMANTIC_FACE_SDF),
            ("SEMANTIC_MEASURED_BRDF", SEMANTIC_MEASURED_BRDF),
            ("SEMANTIC_ILLUMINATION_MASK", SEMANTIC_ILLUMINATION_MASK),
        ] {
            assert!(source.contains(&format!("const {name}: u32 = {semantic}u;")));
        }
        assert!(source.contains(
            "case SEMANTIC_FACE_SDF, SEMANTIC_MEASURED_BRDF, SEMANTIC_ILLUMINATION_MASK:"
        ));
        assert!(source.contains("out.texture_contract_valid = false;"));
        assert_eq!(
            texture_semantic_route(4242),
            TextureSemanticRoute::Unsupported
        );
    }

    #[test]
    fn white_texels_leave_authored_factors_untouched() {
        // Every default slot is white / flat-normal, so folding them must be the
        // identity over the authored parameters (matching the no-texture path).
        let params = MaterialModulationParams {
            base_color: [0.8, 0.4, 0.2, 0.9],
            emissive: [1.0, 2.0, 3.0],
            metallic: 0.3,
            perceptual_roughness: 0.6,
            ambient_occlusion: 0.7,
            normal_scale: 1.0,
            clearcoat: 0.5,
            clearcoat_roughness: 0.25,
        };
        let mut sampled = sampled_material_defaults(&params);
        // sRGB(1.0) == 1.0, so a white base-color / emissive texel is identity.
        sampled = fold_material_texel(SEMANTIC_BASE_COLOR, [1.0, 1.0, 1.0, 1.0], 1.0, sampled);
        sampled = fold_material_texel(SEMANTIC_EMISSIVE, [1.0, 1.0, 1.0, 1.0], 1.0, sampled);
        sampled = fold_material_texel(
            SEMANTIC_METALLIC_ROUGHNESS,
            [1.0, 1.0, 1.0, 1.0],
            1.0,
            sampled,
        );
        sampled = fold_material_texel(SEMANTIC_OCCLUSION, [1.0, 1.0, 1.0, 1.0], 1.0, sampled);
        sampled = fold_material_texel(SEMANTIC_NORMAL, [0.5, 0.5, 1.0, 1.0], 1.0, sampled);

        assert!(approx3(sampled.base_color, [0.8, 0.4, 0.2]));
        assert!(approx(sampled.base_alpha, 0.9));
        assert!(approx3(sampled.emissive, [1.0, 2.0, 3.0]));
        assert!(approx(sampled.metallic, 0.3));
        assert!(approx(sampled.perceptual_roughness, 0.6));
        assert!(approx(sampled.occlusion, 0.7));
        assert!(approx3(sampled.normal_tangent, [0.0, 0.0, 1.0]));
        assert!(sampled.has_normal_map);
    }

    #[test]
    fn base_color_texel_is_srgb_decoded_then_modulated() {
        let params = MaterialModulationParams {
            base_color: [0.5, 0.5, 0.5, 1.0],
            ..Default::default()
        };
        let sampled = fold_material_texel(
            SEMANTIC_BASE_COLOR,
            [0.5, 0.5, 0.5, 0.5],
            1.0,
            sampled_material_defaults(&params),
        );
        let linear = srgb_channel_to_linear(0.5);
        assert!(approx3(
            sampled.base_color,
            [0.5 * linear, 0.5 * linear, 0.5 * linear]
        ));
        assert!(approx(sampled.base_alpha, 0.5));
    }

    #[test]
    fn metallic_roughness_reads_gltf_bg_channels() {
        let params = MaterialModulationParams {
            metallic: 1.0,
            perceptual_roughness: 1.0,
            ..Default::default()
        };
        // Roughness in G, metallic in B; R must be ignored here.
        let sampled = fold_material_texel(
            SEMANTIC_METALLIC_ROUGHNESS,
            [0.123, 0.4, 0.8, 1.0],
            1.0,
            sampled_material_defaults(&params),
        );
        assert!(approx(sampled.perceptual_roughness, 0.4));
        assert!(approx(sampled.metallic, 0.8));
    }

    #[test]
    fn occlusion_reads_red_channel() {
        let params = MaterialModulationParams {
            ambient_occlusion: 1.0,
            ..Default::default()
        };
        let sampled = fold_material_texel(
            SEMANTIC_OCCLUSION,
            [0.25, 0.9, 0.9, 1.0],
            1.0,
            sampled_material_defaults(&params),
        );
        assert!(approx(sampled.occlusion, 0.25));
    }

    #[test]
    fn clearcoat_channels_modulate_authored_factors() {
        let params = MaterialModulationParams {
            clearcoat: 0.8,
            clearcoat_roughness: 0.5,
            ..Default::default()
        };
        let mut sampled = sampled_material_defaults(&params);
        sampled = fold_material_texel(SEMANTIC_CLEAR_COAT, [0.5, 0.0, 0.0, 1.0], 1.0, sampled);
        sampled = fold_material_texel(
            SEMANTIC_CLEAR_COAT_ROUGHNESS,
            [0.0, 0.4, 0.0, 1.0],
            1.0,
            sampled,
        );
        sampled = fold_material_texel(
            SEMANTIC_CLEAR_COAT_NORMAL,
            [1.0, 0.5, 0.5, 1.0],
            1.0,
            sampled,
        );
        assert!(approx(sampled.clearcoat, 0.4));
        assert!(approx(sampled.clearcoat_roughness, 0.2));
        assert!(sampled.has_clearcoat_normal);
        assert!(sampled.clearcoat_normal_tangent[0] > 0.0);
    }
}
