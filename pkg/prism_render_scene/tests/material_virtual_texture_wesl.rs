//! Standalone WESL link coverage for the material virtual-texture resolver.
//!
//! The fixture storage binding below exists only to instantiate the helper's
//! `GpuPageTable` pointer parameter while linking a compute entry point. The
//! production helper declares no group/binding and makes no feedback or upload
//! ordering claim; those remain caller-owned contracts.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("material VT helper is WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

const LINK_TEST_SOURCE: &str = r#"
import prism_render_scene::shaders::material_virtual_texture::{
    MATERIAL_VT_MISS,
    MATERIAL_VT_NON_VIRTUAL_TEXTURE,
    MaterialVirtualTextureConfig,
    MaterialVirtualTextureResolution,
    material_vt_compare_words,
    material_vt_exact_slot,
    material_vt_shift_page,
    material_vt_coarse_page_uv,
    material_vt_atlas_uv,
    material_vt_empty_resolution,
    material_vt_resolve,
};

// Test fixture only; the helper intentionally does not own a production binding.
@group(0) @binding(0) var<storage, read> fixture_page_table: array<u32>;
@group(0) @binding(1) var<storage, read_write> link_output: array<vec4<u32>>;

@compute @workgroup_size(1)
fn material_virtual_texture_link_test(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x != 0u { return; }

    let order = material_vt_compare_words(3u, 0x01000200u, 0x00040005u, 3u, 0x01000200u, 0x00040006u);
    let slot = material_vt_exact_slot(&fixture_page_table, 0u, 3u, 0x01000200u, 0x00040005u);
    let shifted = material_vt_shift_page(0xffffu, 16u);
    let coarse_uv = material_vt_coarse_page_uv(3u, 2u, vec2<f32>(0.25, 0.75), 1u);
    let atlas_uv = material_vt_atlas_uv(vec2<f32>(0.5, 0.5), 5u, 136u, 8u, 128u, 4u);
    let empty: MaterialVirtualTextureResolution = material_vt_empty_resolution(false);

    var config: MaterialVirtualTextureConfig;
    config.enabled = 0u;
    config.entry_count = 0u;
    config.page_size = 128u;
    config.tile_extent_px = 136u;
    config.tiles_per_row = 8u;
    config.atlas_layers = 1u;
    config.gutter_px = 4u;
    config.coarsest_mip = 8u;
    let disabled = material_vt_resolve(
        &fixture_page_table, config, MATERIAL_VT_NON_VIRTUAL_TEXTURE,
        0u, 0u, 0u, 0u, vec2<f32>(0.5, 0.5),
    );
    config.enabled = 1u;
    config.entry_count = 1u;
    let non_vt = material_vt_resolve(
        &fixture_page_table, config, MATERIAL_VT_NON_VIRTUAL_TEXTURE,
        0u, 0u, 0u, 0u, vec2<f32>(0.5, 0.5),
    );
    let miss = material_vt_resolve(
        &fixture_page_table, config, 7u,
        0u, 0u, 0u, 0u, vec2<f32>(0.5, 0.5),
    );

    link_output[0] = vec4<u32>(
        bitcast<u32>(f32(order)), slot, shifted, MATERIAL_VT_MISS,
    );
    link_output[1] = vec4<u32>(
        bitcast<u32>(coarse_uv.x), bitcast<u32>(coarse_uv.y),
        bitcast<u32>(atlas_uv.x), bitcast<u32>(atlas_uv.y),
    );
    link_output[2] = vec4<u32>(
        select(0u, 1u, empty.hit),
        select(0u, 1u, disabled.vt_enabled),
        select(0u, 1u, non_vt.vt_enabled),
        select(0u, 1u, miss.hit),
    );
    link_output[3] = vec4<u32>(
        select(0u, 1u, miss.vt_enabled),
        miss.slot,
        miss.resolved_mip,
        miss.atlas_layer,
    );
}
"#;

/// Links a real compute entry point that references the page-table search,
/// coarse fallback UV, atlas gutter mapping, and disabled/miss result paths.
#[test]
fn material_virtual_texture_wesl_compiles_and_links_every_helper() {
    let mut cache = ShaderCache::new((), load_source);

    let module = shader_id(0x5052_4953_4d5f_4d41_5456_5458_0000_0001);
    cache.set_shader(
        module,
        Shader::from_wesl(
            include_str!("../src/shaders/material_virtual_texture.wesl"),
            "embedded://prism_render_scene/shaders/material_virtual_texture.wesl",
        ),
    );

    let link_test = shader_id(0x5052_4953_4d5f_4d41_5456_5458_0000_0002);
    cache.set_shader(
        link_test,
        Shader::from_wesl(
            LINK_TEST_SOURCE,
            "embedded://prism_render_scene/tests/material_virtual_texture_link_test.wesl",
        ),
    );

    cache.get(0, link_test, &[]).unwrap_or_else(|error| {
        panic!("material_virtual_texture.wesl failed to compile/link: {error}")
    });
}
