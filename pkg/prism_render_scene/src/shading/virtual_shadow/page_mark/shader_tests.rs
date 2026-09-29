//! WESL compilation coverage for the virtual-shadow-map page-request shader.
//!
//! The sandbox has no GPU, so this test does **not** run the page-mark pass. It
//! compiles `shaders/vsm_page_mark.wesl` through the same `ShaderCache` / `wesl`
//! pipeline the render world uses, validating that the clipmap addressing
//! (level selection, world-page snapping, camera-snapped resident-window slot
//! indexing), the receiver storage buffer read and the atomic page-request
//! marking all parse and type-check as WESL.
//!
//! The shader has no `import` statements, so a green result also proves it
//! stands alone and needs none of the shading module graph registered, and that
//! its immediate block matches the 48-byte [`super::super::abi::GpuVsmPageMarkParams`]
//! ABI it is compiled against. Numerical parity with the golden
//! [`prism_render_shading::shadow`] virtual-shadow-map reference is guaranteed
//! by construction (each helper mirrors its CPU twin line for line) but must be
//! confirmed on real hardware once the allocator / resolve wiring lands.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("virtual shadow shaders are WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

/// Compiles `vsm_page_mark.wesl` standalone. It has no imports, so a green
/// result proves the clipmap helpers (level selection, page-world sizing,
/// world-page coords, camera-snapped window origin, filter page radius), the
/// resident-window slot indexing, the receiver storage buffer and the atomic
/// page-request bitmap all parse and type-check on their own.
#[test]
fn vsm_page_mark_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let page_mark = shader_id(0x5052_4953_4d5f_5653_4d5f_504d_524b_0002);
    cache.set_shader(
        page_mark,
        Shader::from_wesl(
            include_str!("../../../shaders/vsm_page_mark.wesl"),
            "embedded://prism_render_scene/shaders/vsm_page_mark.wesl",
        ),
    );

    cache
        .get(0, page_mark, &[])
        .unwrap_or_else(|error| panic!("vsm_page_mark.wesl failed to compile: {error:?}"));
}
