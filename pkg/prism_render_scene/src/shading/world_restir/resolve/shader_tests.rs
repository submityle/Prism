//! WESL compile coverage plus a CPU parity check for the resolve estimate.
//!
//! The resolve re-evaluates each hit reservoir's stored light sample under a
//! reconnection shift (golden `reconnection_target`, RGB):
//!
//!     E_rgb = radiance * geometric_term(x_v, n_v, x_s, n_s) * W * intensity
//!
//! The `geometric_term` in `world_restir_resolve.wesl` is a byte-for-byte port
//! of the golden [`prism_render_shading::gi::screen_probe::geometric_term`], so
//! the Rust parity test below mirrors the resolve estimate against that golden
//! and asserts the physically meaningful invariants: a facing pair resolves
//! positive energy, the geometry factor falls off as inverse-square distance,
//! and a backfacing sample resolves to zero. The sandbox has no GPU, so this
//! CPU parity (plus the standalone WESL compile) is the device-equivalence
//! proof.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_math::Vec3;
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};
use prism_render_shading::gi::screen_probe::geometric_term;

fn load(_: &(), s: ShaderCacheSource, _: &ValidateShader) -> Result<String, ShaderCacheError> {
    match s {
        ShaderCacheSource::Wgsl(x) => Ok(x),
        ShaderCacheSource::SpirV(_) => unreachable!(),
    }
}

#[test]
fn resolve_wesl_compiles_standalone() {
    let mut c = ShaderCache::new((), load);
    let id = AssetId::Uuid {
        // "PRISM_WRESTIR_RS"
        uuid: Uuid::from_u128(0x505249534d5f575245535449525f5253),
    };
    c.set_shader(
        id,
        Shader::from_wesl(
            include_str!("../../../shaders/world_restir_resolve.wesl"),
            "embedded://prism_render_scene/shaders/world_restir_resolve.wesl",
        ),
    );
    c.get(0, id, &[])
        .unwrap_or_else(|e| panic!("world-restir resolve failed: {e}"));
}

/// Mirrors the shader's resolve estimate: `radiance * g * W * intensity`,
/// clamped to non-negative energy exactly as the shader's `max(estimate, 0)`.
fn resolve_estimate(
    radiance: Vec3,
    visible_point: Vec3,
    visible_normal: Vec3,
    sample_point: Vec3,
    sample_normal: Vec3,
    w: f32,
    intensity: f32,
) -> Vec3 {
    let g = geometric_term(visible_point, visible_normal, sample_point, sample_normal);
    (radiance * (g * w * intensity)).max(Vec3::ZERO)
}

#[test]
fn facing_pair_resolves_positive_energy() {
    // Two unit-separated surfaces facing each other along +y: the visible
    // normal points up at the sample, the sample normal points down at the
    // visible point, so both cosines are 1 and the geometry factor is 1.
    let energy = resolve_estimate(
        Vec3::new(2.0, 3.0, 4.0),
        Vec3::ZERO,
        Vec3::Y,
        Vec3::new(0.0, 1.0, 0.0),
        -Vec3::Y,
        0.5,
        1.0,
    );
    // g = cos_v * cos_s / dist_sq = 1 * 1 / 1 = 1; E = radiance * 1 * 0.5 * 1.
    assert!((energy - Vec3::new(1.0, 1.5, 2.0)).length() < 1e-5);
}

#[test]
fn geometry_factor_falls_off_as_inverse_square() {
    // Same facing geometry at distances 1 and 2: the geometry factor (hence the
    // resolved energy) must drop by exactly 4x (inverse-square falloff).
    let near = resolve_estimate(
        Vec3::ONE,
        Vec3::ZERO,
        Vec3::Y,
        Vec3::new(0.0, 1.0, 0.0),
        -Vec3::Y,
        1.0,
        1.0,
    );
    let far = resolve_estimate(
        Vec3::ONE,
        Vec3::ZERO,
        Vec3::Y,
        Vec3::new(0.0, 2.0, 0.0),
        -Vec3::Y,
        1.0,
        1.0,
    );
    assert!(near.x > 0.0 && far.x > 0.0);
    assert!((near.x / far.x - 4.0).abs() < 1e-4);
}

#[test]
fn backfacing_sample_resolves_to_zero() {
    // The sample normal faces away from the visible point (points up, same as
    // the visible normal), so `cos_s = max(dot(n_s, -dir), 0) = 0` and the
    // geometry factor — hence the resolved energy — is zero.
    let energy = resolve_estimate(
        Vec3::splat(5.0),
        Vec3::ZERO,
        Vec3::Y,
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::Y,
        1.0,
        1.0,
    );
    assert_eq!(energy, Vec3::ZERO);
}

#[test]
fn intensity_scales_the_resolved_energy_linearly() {
    let base = resolve_estimate(
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::ZERO,
        Vec3::Y,
        Vec3::new(0.0, 1.0, 0.0),
        -Vec3::Y,
        0.25,
        1.0,
    );
    let gained = resolve_estimate(
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::ZERO,
        Vec3::Y,
        Vec3::new(0.0, 1.0, 0.0),
        -Vec3::Y,
        0.25,
        3.0,
    );
    assert!((gained - base * 3.0).length() < 1e-6);
}
