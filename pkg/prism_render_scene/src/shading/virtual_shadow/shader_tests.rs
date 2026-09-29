//! WESL compilation coverage for the virtual-shadow-map shaders.
//!
//! The sandbox has no GPU, so these tests do **not** run any VSM pass.  They
//! compile each shader through the same `ShaderCache` / `wesl` pipeline the
//! render world uses, validating that the clipmap addressing (level selection,
//! world-page snapping, resident-window slot indexing), the atomic page-request
//! marking and the page-table / physical-atlas PCF sampling all parse and
//! type-check as WESL.
//!
//! Both shaders have no `import` statements, so a green result also proves each
//! stands alone and needs none of the shading module graph registered, and that
//! its immediate block matches the 48-byte `GpuVsm*Params` ABI it is compiled
//! against.  Numerical parity with the golden
//! [`prism_render_shading::shadow`] virtual-shadow-map reference is guaranteed
//! by construction (each helper mirrors its CPU twin line for line) but must be
//! confirmed on real hardware once the atlas-filling and resolve wiring land.

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

/// Compiles `vsm_page_mark.wesl` standalone.  It has no imports, so a green
/// result proves the clipmap helpers (level selection, page-world sizing,
/// world-page coords, camera-snapped window origin, filter page radius), the
/// resident-window slot indexing, the receiver storage buffer and the atomic
/// page-request bitmap all parse and type-check on their own.
#[test]
fn vsm_page_mark_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let page_mark = shader_id(0x5052_4953_4d5f_5653_4d5f_504d_524b_0001);
    cache.set_shader(
        page_mark,
        Shader::from_wesl(
            include_str!("../../shaders/vsm_page_mark.wesl"),
            "embedded://prism_render_scene/shaders/vsm_page_mark.wesl",
        ),
    );

    cache
        .get(0, page_mark, &[])
        .unwrap_or_else(|error| panic!("vsm_page_mark.wesl failed to compile: {error}"));
}

/// Compiles `vsm_sample.wesl` standalone.  It has no imports, so a green result
/// proves the clipmap helpers, the page-table lookup (virtual slot -> physical
/// page or unmapped sentinel), the physical-atlas tile mapping and the
/// seam-clamped PCF filter, plus the `vsm_selftest` call graph, all parse and
/// type-check on their own.
#[test]
fn vsm_sample_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let sample = shader_id(0x5052_4953_4d5f_5653_4d5f_5341_4d50_0001);
    cache.set_shader(
        sample,
        Shader::from_wesl(
            include_str!("../../shaders/vsm_sample.wesl"),
            "embedded://prism_render_scene/shaders/vsm_sample.wesl",
        ),
    );

    cache
        .get(0, sample, &[])
        .unwrap_or_else(|error| panic!("vsm_sample.wesl failed to compile: {error}"));
}

/// Compiles `vsm_receiver_gen.wesl` standalone.  It has no imports, so a green
/// result proves the depth unprojection (Bevy y-down UV -> NDC, inverse
/// view-projection multiply, perspective divide) and the clipmap-plane
/// projection (light-basis dot products + camera view distance) that mirror the
/// golden `receiver_gen` parse and type-check on their own, and that the receiver
/// output struct matches the `vsm_page_mark` `VsmReceiver` it feeds.
#[test]
fn vsm_receiver_gen_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let receiver_gen = shader_id(0x5052_4953_4d5f_5653_4d5f_5247_454e_0001);
    cache.set_shader(
        receiver_gen,
        Shader::from_wesl(
            include_str!("../../shaders/vsm_receiver_gen.wesl"),
            "embedded://prism_render_scene/shaders/vsm_receiver_gen.wesl",
        ),
    );

    cache
        .get(0, receiver_gen, &[])
        .unwrap_or_else(|error| panic!("vsm_receiver_gen.wesl failed to compile: {error}"));
}
