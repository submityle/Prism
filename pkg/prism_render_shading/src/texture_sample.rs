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

use bevy_math::ops;

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
/// Ambient occlusion, linear, in the R channel.
pub const SEMANTIC_OCCLUSION: u32 = 4;
/// Clear-coat strength, linear, in the R channel.
pub const SEMANTIC_CLEAR_COAT: u32 = 5;
/// Clear-coat roughness, linear, in the G channel.
pub const SEMANTIC_CLEAR_COAT_ROUGHNESS: u32 = 6;
/// Clear-coat tangent-space normal map, linear, `[0, 1]`-remapped.
pub const SEMANTIC_CLEAR_COAT_NORMAL: u32 = 7;

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
        base_color: [params.base_color[0], params.base_color[1], params.base_color[2]],
        base_alpha: params.base_color[3],
        emissive: params.emissive,
        metallic: params.metallic,
        perceptual_roughness: params.perceptual_roughness,
        occlusion: params.ambient_occlusion,
        normal_tangent: [0.0, 0.0, 1.0],
        has_normal_map: false,
        clearcoat: params.clearcoat,
        clearcoat_roughness: params.clearcoat_roughness,
        clearcoat_normal_tangent: [0.0, 0.0, 1.0],
        has_clearcoat_normal: false,
    }
}

/// Folds a single sampled texel into `sampled` according to its `semantic`,
/// returning the updated accumulator.  Unknown semantics pass through
/// unchanged.  Mirrors `fold_material_texel` in `material_sample.wesl`
/// arm-for-arm.
#[must_use]
pub fn fold_material_texel(
    semantic: u32,
    texel: [f32; 4],
    normal_scale: f32,
    mut sampled: SampledMaterial,
) -> SampledMaterial {
    let rgb = [texel[0], texel[1], texel[2]];
    match semantic {
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
        SEMANTIC_NORMAL => {
            sampled.normal_tangent = decode_tangent_normal(rgb, normal_scale);
            sampled.has_normal_map = true;
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
        _ => {}
    }
    sampled
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
        assert!(approx3(decode_tangent_normal([0.5, 0.5, 1.0], 1.0), [0.0, 0.0, 1.0]));
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
        assert!(approx3(decode_tangent_normal([1.0, 0.0, 0.5], 0.0), [0.0, 0.0, 1.0]));
    }

    #[test]
    fn degenerate_normal_texel_falls_back_to_plus_z() {
        // A (0.5, 0.5, 0.5) texel decodes to the zero vector before renormalize.
        assert!(approx3(decode_tangent_normal([0.5, 0.5, 0.5], 1.0), [0.0, 0.0, 1.0]));
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
        sampled = fold_material_texel(SEMANTIC_METALLIC_ROUGHNESS, [1.0, 1.0, 1.0, 1.0], 1.0, sampled);
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
        assert!(approx3(sampled.base_color, [0.5 * linear, 0.5 * linear, 0.5 * linear]));
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
        sampled = fold_material_texel(SEMANTIC_CLEAR_COAT_ROUGHNESS, [0.0, 0.4, 0.0, 1.0], 1.0, sampled);
        sampled = fold_material_texel(SEMANTIC_CLEAR_COAT_NORMAL, [1.0, 0.5, 0.5, 1.0], 1.0, sampled);
        assert!(approx(sampled.clearcoat, 0.4));
        assert!(approx(sampled.clearcoat_roughness, 0.2));
        assert!(sampled.has_clearcoat_normal);
        assert!(sampled.clearcoat_normal_tangent[0] > 0.0);
    }

    #[test]
    fn unknown_semantic_is_ignored() {
        let params = MaterialModulationParams::default();
        let before = sampled_material_defaults(&params);
        let after = fold_material_texel(4242, [0.1, 0.2, 0.3, 0.4], 1.0, before);
        assert_eq!(before, after);
    }
}
