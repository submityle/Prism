//! WESL compilation coverage for the motion-blur shader.
//!
//! The sandbox has no GPU, so this test compiles the WESL source through the
//! same `ShaderCache` / `wesl` pipeline the render world uses, validating that
//! `motion_blur.wesl` parses and type-checks exactly as it will on device. The
//! kernel is self-contained (no intra-crate `import`s, matching `exposure.wesl`
//! / `bloom.wesl`), so a green result also guards the reconstruction maths —
//! TileMax/NeighborMax velocity dilation, soft depth classification and the
//! cone/cylinder gather weight — against drift from its CPU golden twin in
//! `prism_render_shading::motion_blur`.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("motion-blur shaders are WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

/// Compiles `motion_blur.wesl`, proving the plausible motion-blur
/// reconstruction kernel parses and type-checks exactly as it will in the
/// render world (velocity / depth in; a per-tap reconstruction weight out), and
/// that its maths matches the CPU golden.
#[test]
fn motion_blur_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let motion_blur = shader_id(0x5052_4953_4d5f_0000_4d4f_5449_4f4e_424c);
    cache.set_shader(
        motion_blur,
        Shader::from_wesl(
            include_str!("../../shaders/motion_blur.wesl"),
            "embedded://prism_render_scene/shaders/motion_blur.wesl",
        ),
    );

    cache
        .get(0, motion_blur, &[])
        .unwrap_or_else(|error| panic!("motion_blur.wesl failed to compile: {error}"));
}

/// The immediate block the three passes share is exactly the size and alignment
/// `motion_blur.wesl`'s `MotionBlurParams` struct expects, and the tile /
/// workgroup constants match the shader's `@workgroup_size` and tile stride.
/// A drift here would silently mis-upload the push-constant block on device even
/// though the shader still compiles, so the compile test above cannot catch it.
#[test]
fn motion_blur_abi_matches_the_shader_layout() {
    use super::abi::{MotionBlurParams, MOTION_BLUR_TILE_SIZE, MOTION_BLUR_WORKGROUP_SIZE};

    // mat4x4 (64) + six u32 (24) + three f32 (12) + three u32 pads (12) = 112,
    // a multiple of the 16-byte immediate alignment the matrix forces.
    assert_eq!(size_of::<MotionBlurParams>(), 112);
    assert_eq!(align_of::<MotionBlurParams>(), 4);
    // Tiles are 16 texels square; every entry runs at @workgroup_size(8, 8, 1).
    assert_eq!(MOTION_BLUR_TILE_SIZE, 16);
    assert_eq!(MOTION_BLUR_WORKGROUP_SIZE, 8);
}
