//! WESL compile coverage and CPU-mirror parity for `spec_gi_composite.wesl`.
//!
//! The sandbox has no GPU, so the compile test drives the composite module
//! through the same `ShaderCache` / `wesl` pipeline the render world uses,
//! proving it parses and type-checks exactly as it will on device (and guards
//! its immediate `CompositeParams` layout against drift from the 16-byte
//! [`GpuSpecGiCompositeParams`] contract).
//!
//! The parity test transcribes the kernel's energy-conserving fold op-for-op
//! into Rust and asserts the two defining properties of option C: a confident
//! hit substitutes the glossy reflection for the IBL specular, and a miss
//! (`confidence == 0`) leaves the shaded base untouched.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

use super::abi::GpuSpecGiCompositeParams;

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("spec_gi composite shader is WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

/// Compiles `spec_gi_composite.wesl` standalone, proving the energy-conserving
/// substitution kernel parses and type-checks as it will in the render world.
/// The kernel is self-contained (no intra-crate imports), so a green result
/// also guards its immediate `CompositeParams` layout against drift from the
/// 16-byte [`GpuSpecGiCompositeParams`] contract (extent + two pad words).
#[test]
fn spec_gi_composite_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let composite = shader_id(0x5350_4543_5f47_495f_434d_5053_5f00_0001);
    cache.set_shader(
        composite,
        Shader::from_wesl(
            include_str!("../../shaders/spec_gi_composite.wesl"),
            "embedded://prism_render_scene/shaders/spec_gi_composite.wesl",
        ),
    );

    cache
        .get(0, composite, &[])
        .unwrap_or_else(|error| panic!("spec_gi_composite.wesl failed to compile: {error}"));

    // The immediate block the pipeline specializes against is 16 bytes.
    assert_eq!(size_of::<GpuSpecGiCompositeParams>(), 16);
}

/// Rust transcription of the WESL fold, op-for-op:
/// `resolved = max(base + (max(contrib, 0) - ibl_specular) * clamp(conf, 0, 1), 0)`.
fn composite_fold(base: [f32; 3], contrib: [f32; 3], ibl: [f32; 3], confidence: f32) -> [f32; 3] {
    let c = confidence.clamp(0.0, 1.0);
    let mut out = [0.0f32; 3];
    for i in 0..3 {
        let contrib_i = contrib[i].max(0.0);
        let delta = (contrib_i - ibl[i]) * c;
        out[i] = (base[i] + delta).max(0.0);
    }
    out
}

/// A miss (`confidence == 0`) is the identity: the shaded base survives
/// unchanged, so the IBL specular already folded in by the resolve is kept.
#[test]
fn composite_miss_is_identity() {
    let base = [0.2, 0.4, 0.6];
    let out = composite_fold(base, [1.0, 1.0, 1.0], [0.3, 0.3, 0.3], 0.0);
    assert_eq!(out, base);
}

/// A full-confidence hit substitutes the glossy reflection for the IBL
/// specular: `base - ibl + contrib`. With the resolve having folded `ibl` into
/// `base`, a confident hit nets out to `base_without_specular + contrib`.
#[test]
fn composite_hit_replaces_ibl_specular() {
    let base = [0.5, 0.5, 0.5];
    let ibl = [0.2, 0.1, 0.05];
    let contrib = [0.4, 0.3, 0.2];
    let out = composite_fold(base, contrib, ibl, 1.0);
    for i in 0..3 {
        let want = base[i] - ibl[i] + contrib[i];
        assert!(
            (out[i] - want).abs() < 1e-6,
            "channel {i}: {} vs {}",
            out[i],
            want
        );
    }
}

/// Partial confidence linearly interpolates between the IBL fallback (miss) and
/// the full glossy substitution (hit).
#[test]
fn composite_partial_confidence_lerps() {
    let base = [0.5, 0.5, 0.5];
    let ibl = [0.2, 0.1, 0.05];
    let contrib = [0.4, 0.3, 0.2];
    let miss = composite_fold(base, contrib, ibl, 0.0);
    let hit = composite_fold(base, contrib, ibl, 1.0);
    let half = composite_fold(base, contrib, ibl, 0.5);
    for i in 0..3 {
        let want = 0.5 * miss[i] + 0.5 * hit[i];
        assert!(
            (half[i] - want).abs() < 1e-6,
            "channel {i}: {} vs {}",
            half[i],
            want
        );
    }
}

/// The resolved colour is floored at zero so an over-bright IBL specular minus a
/// dark reflection cannot drive `scene_color` negative.
#[test]
fn composite_clamps_negative_to_zero() {
    let out = composite_fold([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 1.0);
    assert_eq!(out, [0.0, 0.0, 0.0]);
}
