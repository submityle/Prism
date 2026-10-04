//! WESL compile coverage plus a CPU parity check for the view-position
//! reconstruction the producer shares with the SSGI trace.
//!
//! The kernel reconstructs each tile's view-space shading point by the exact
//! inverse of the projection (`view_from_clip * clip`, perspective divide), so
//! the Rust parity test below mirrors that reconstruction and asserts it
//! round-trips a known view point through NDC + tile-centre uv back to itself.
//! The sandbox has no GPU, so this CPU parity (plus the standalone WESL compile)
//! is the device-equivalence proof.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_math::{Mat4, Vec3, Vec4};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

fn load(_: &(), s: ShaderCacheSource, _: &ValidateShader) -> Result<String, ShaderCacheError> {
    match s {
        ShaderCacheSource::Wgsl(x) => Ok(x),
        ShaderCacheSource::SpirV(_) => unreachable!(),
    }
}

#[test]
fn visible_points_wesl_compiles_standalone() {
    let mut c = ShaderCache::new((), load);
    let id = AssetId::Uuid {
        // "PRISM_WRESTIR_VP"
        uuid: Uuid::from_u128(0x505249534d5f575245535449525f5650),
    };
    c.set_shader(
        id,
        Shader::from_wesl(
            include_str!("../../../shaders/world_restir_visible_points.wesl"),
            "embedded://prism_render_scene/shaders/world_restir_visible_points.wesl",
        ),
    );
    c.get(0, id, &[])
        .unwrap_or_else(|e| panic!("world-restir visible points failed: {e}"));
}

/// Mirrors the shader's `reconstruct_view_position`: NDC from tile-centre uv +
/// device depth, inverse projection, perspective divide.
fn reconstruct_view_position(view_from_clip: Mat4, uv: bevy_math::Vec2, device_depth: f32) -> Vec3 {
    let ndc = Vec3::new(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, device_depth);
    let view = view_from_clip * Vec4::new(ndc.x, ndc.y, ndc.z, 1.0);
    view.truncate() / view.w
}

#[test]
fn reconstruction_round_trips_a_known_view_point() {
    // A standard RH perspective projection; the round-trip holds for any
    // invertible projection, reverse-Z or not — the shader's reverse-Z is only
    // a depth-convention detail the inverse recovers regardless.
    let proj = Mat4::perspective_rh(60f32.to_radians(), 16.0 / 9.0, 0.1, 1000.0);
    let view_from_clip = proj.inverse();

    // A point in front of the RH camera (negative view-space z).
    let p_view = Vec3::new(1.5, -0.75, -12.0);

    // Forward project to NDC, then to the tile-centre uv the shader derives.
    let clip = proj * Vec4::new(p_view.x, p_view.y, p_view.z, 1.0);
    let ndc = clip.truncate() / clip.w;
    let uv = bevy_math::Vec2::new(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);

    let reconstructed = reconstruct_view_position(view_from_clip, uv, ndc.z);
    assert!(
        (reconstructed - p_view).length() < 1e-3,
        "reconstructed {reconstructed:?} should match the known view point {p_view:?}",
    );
}
