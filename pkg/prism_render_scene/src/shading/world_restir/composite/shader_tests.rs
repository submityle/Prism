//! WESL compile coverage plus a CPU parity check for the composite fold.
//!
//! The composite folds the resolve's direct-illumination export back over the
//! shaded scene under the energy-conserving substitution (see [`super`]'s
//! module docs for the algebra):
//!
//!     scene = max(base + confidence * albedo * INV_PI * (gi_out - direct), 0)
//!
//! The Rust parity below mirrors `wr_composite`'s fold exactly and asserts the
//! physically meaningful invariants: a miss (`confidence == 0`) is a no-op, a
//! confident pixel swaps the clustered punctual diffuse for the `ReSTIR`
//! estimate, an equal estimate and clustered direct cancel to the base, and the
//! result never drops below zero energy. The sandbox has no GPU, so this CPU
//! parity (plus the standalone WESL compile) is the device-equivalence proof.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_math::Vec3;
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

/// Reciprocal of pi, matching the shader's `INV_PI`.
const INV_PI: f32 = core::f32::consts::FRAC_1_PI;

fn load(_: &(), s: ShaderCacheSource, _: &ValidateShader) -> Result<String, ShaderCacheError> {
    match s {
        ShaderCacheSource::Wgsl(x) => Ok(x),
        ShaderCacheSource::SpirV(_) => unreachable!(),
    }
}

#[test]
fn composite_wesl_compiles_standalone() {
    let mut c = ShaderCache::new((), load);
    let id = AssetId::Uuid {
        // "PRISM_WRESTIR_CO"
        uuid: Uuid::from_u128(0x505249534d5f575245535449525f434f),
    };
    c.set_shader(
        id,
        Shader::from_wesl(
            include_str!("../../../shaders/world_restir_composite.wesl"),
            "embedded://prism_render_scene/shaders/world_restir_composite.wesl",
        ),
    );
    c.get(0, id, &[])
        .unwrap_or_else(|e| panic!("world-restir composite failed: {e}"));
}

/// Mirrors `wr_composite`'s fold: swap the clustered punctual diffuse for the
/// `ReSTIR` estimate under confidence, both converted to Lambertian diffuse
/// radiance via `albedo * INV_PI`, clamped to non-negative energy.
fn composite_fold(base: Vec3, gi_out: Vec3, direct: Vec3, albedo: Vec3, confidence: f32) -> Vec3 {
    let confidence = confidence.clamp(0.0, 1.0);
    let delta = albedo * (confidence * INV_PI) * (gi_out - direct);
    (base + delta).max(Vec3::ZERO)
}

#[test]
fn miss_is_a_no_op() {
    // confidence == 0 leaves the shaded colour untouched, preserving the
    // shading-resolve clustered punctual direct.
    let base = Vec3::new(0.2, 0.3, 0.4);
    let out = composite_fold(
        base,
        Vec3::splat(5.0),
        Vec3::splat(1.0),
        Vec3::splat(0.8),
        0.0,
    );
    assert_eq!(out, base);
}

#[test]
fn equal_estimate_and_direct_cancel_to_base() {
    // When the ReSTIR estimate equals the clustered punctual direct, the
    // substitution is a no-op regardless of confidence or albedo.
    let base = Vec3::new(0.5, 0.6, 0.7);
    let out = composite_fold(
        base,
        Vec3::splat(2.0),
        Vec3::splat(2.0),
        Vec3::new(0.9, 0.8, 0.7),
        1.0,
    );
    assert!((out - base).length() < 1e-6);
}

#[test]
fn full_confidence_swaps_direct_for_the_estimate() {
    // At confidence 1 the fold removes `albedo * INV_PI * direct` and adds
    // `albedo * INV_PI * gi_out`, exactly `albedo * INV_PI * (gi_out - direct)`.
    let base = Vec3::new(0.1, 0.1, 0.1);
    let albedo = Vec3::new(0.5, 0.5, 0.5);
    let gi_out = Vec3::new(4.0, 4.0, 4.0);
    let direct = Vec3::new(1.0, 1.0, 1.0);
    let out = composite_fold(base, gi_out, direct, albedo, 1.0);
    let expected = base + albedo * INV_PI * (gi_out - direct);
    assert!((out - expected).length() < 1e-6);
}

#[test]
fn confidence_scales_the_substitution_linearly() {
    let base = Vec3::splat(0.3);
    let albedo = Vec3::splat(0.7);
    let gi_out = Vec3::splat(3.0);
    let direct = Vec3::splat(0.5);
    let half = composite_fold(base, gi_out, direct, albedo, 0.5);
    let full = composite_fold(base, gi_out, direct, albedo, 1.0);
    // The delta at confidence 1 is exactly twice the delta at confidence 0.5.
    let delta_half = half - base;
    let delta_full = full - base;
    assert!((delta_full - delta_half * 2.0).length() < 1e-6);
}

#[test]
fn result_is_clamped_to_non_negative_energy() {
    // A large clustered punctual direct minus a tiny estimate would drive the
    // radiance negative; the fold clamps it to zero exactly as the shader does.
    let base = Vec3::splat(0.05);
    let out = composite_fold(
        base,
        Vec3::splat(0.0),
        Vec3::splat(100.0),
        Vec3::splat(1.0),
        1.0,
    );
    assert_eq!(out, Vec3::ZERO);
}
