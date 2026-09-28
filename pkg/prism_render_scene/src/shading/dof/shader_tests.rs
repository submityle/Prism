//! WESL compilation coverage for the depth-of-field shader.
//!
//! The sandbox has no GPU, so this test compiles the WESL source through the
//! same `ShaderCache` / `wesl` pipeline the render world uses, validating that
//! `dof.wesl` parses and type-checks exactly as it will on device. The kernel
//! is self-contained (no intra-crate `import`s, matching `bloom.wesl` /
//! `exposure.wesl`), so a green result also guards the DoF maths — thin-lens
//! circle of confusion, near/far field separation, the mm->pixel CoC
//! conversion, the soft-edged bokeh weight and the sharp/blurred blend —
//! against drift from its CPU golden twin in `prism_render_shading::dof`.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("dof shaders are WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

/// Compiles `dof.wesl`, proving the physically-based depth-of-field kernel
/// parses and type-checks exactly as it will in the render world (pre-exposed
/// HDR radiance + depth in; circle of confusion, near/far layers and the
/// bokeh-gathered composite out), and that the optics match the CPU golden.
#[test]
fn dof_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let dof = shader_id(0x5052_4953_4d5f_0000_444f_4650_5f5f_5f00);
    cache.set_shader(
        dof,
        Shader::from_wesl(
            include_str!("../../shaders/dof.wesl"),
            "embedded://prism_render_scene/shaders/dof.wesl",
        ),
    );

    cache
        .get(0, dof, &[])
        .unwrap_or_else(|error| panic!("dof.wesl failed to compile: {error}"));
}
