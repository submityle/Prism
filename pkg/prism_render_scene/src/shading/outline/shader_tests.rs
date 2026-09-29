//! WESL compilation coverage for the outline shader.
//!
//! The sandbox has no GPU, so this test compiles the WESL source through the
//! same `ShaderCache` / `wesl` pipeline the render world uses, validating that
//! `outline.wesl` parses and type-checks exactly as it will on device. The
//! kernel is self-contained (no intra-crate `import`s, matching `ssgi.wesl` /
//! `gtao.wesl`), so a green result also guards the shared immediate
//! `OutlineParams` layout and the id/depth/normal edge reductions against drift
//! from their CPU golden twin in `prism_render_shading::outline`.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("outline shaders are WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

/// Compiles `outline.wesl`, proving the NPR silhouette/crease outline kernel
/// parses and type-checks exactly as it will in the render world (vis-buffer
/// material id, linear depth and packed normal over a four-neighbour cross in;
/// a single `[0, 1]` outline coverage out), and that the `OutlineParams`
/// immediate layout matches the CPU golden.
#[test]
fn outline_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let outline = shader_id(0x5052_4953_4d5f_4f55_544c_494e_455f_0001);
    cache.set_shader(
        outline,
        Shader::from_wesl(
            include_str!("../../shaders/outline.wesl"),
            "embedded://prism_render_scene/shaders/outline.wesl",
        ),
    );

    cache
        .get(0, outline, &[])
        .unwrap_or_else(|error| panic!("outline.wesl failed to compile: {error}"));
}

/// Guards the Rust immediate-block ABI against drift from the WESL struct: the
/// single `GpuOutlineParams` block is the 112-byte matrix-led block matching
/// `outline.wesl`'s one `var<immediate>` global, and the workgroup constant
/// matches `@workgroup_size(8, 8, 1)`.
#[test]
fn outline_abi_matches_the_shader_layout() {
    use super::abi::{GpuOutlineParams, OUTLINE_WORKGROUP_SIZE};
    assert_eq!(size_of::<GpuOutlineParams>(), 112);
    assert_eq!(align_of::<GpuOutlineParams>(), 4);
    assert_eq!(OUTLINE_WORKGROUP_SIZE, 8);
}
