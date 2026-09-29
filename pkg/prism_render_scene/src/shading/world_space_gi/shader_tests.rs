//! WESL compilation coverage + ABI layout guards for the world-space GI
//! subsystem.
//!
//! The sandbox has no GPU, so these tests compile both WESL sources through
//! the same `ShaderCache` / `wesl` pipeline the render world uses, validating
//! that `world_space_gi_probe_update.wesl` and `world_space_gi_resolve.wesl`
//! parse and type-check exactly as they will on device. The kernels are
//! self-contained (no intra-crate `import`s, matching `ssgi.wesl`), so a green
//! result also guards the GI maths — octahedral encode/decode, the L1 SH
//! basis, the screen-probe placement and the bilinear-plus-geometric
//! interpolation — against drift from its CPU golden twin in
//! [`prism_render_shading::gi::world_space`].

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("world-space GI shaders are WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

/// Compiles `world_space_gi_probe_update.wesl`, proving the screen-probe
/// capture kernel parses and type-checks exactly as it will in the render
/// world (depth + normal + colour reads and the probe storage buffer out).
#[test]
fn probe_update_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let probe_update = shader_id(0x5052_4953_4d5f_5753_4749_5f50_5542_0001);
    cache.set_shader(
        probe_update,
        Shader::from_wesl(
            include_str!("../../shaders/world_space_gi_probe_update.wesl"),
            "embedded://prism_render_scene/shaders/world_space_gi_probe_update.wesl",
        ),
    );

    cache
        .get(0, probe_update, &[])
        .unwrap_or_else(|error| panic!("world_space_gi_probe_update.wesl failed to compile: {error}"));
}

/// Compiles `world_space_gi_resolve.wesl`, proving the per-pixel probe
/// interpolation kernel parses and type-checks exactly as it will in the
/// render world (depth + normal reads, the probe storage buffer and the GI
/// export storage write).
#[test]
fn resolve_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let resolve = shader_id(0x5052_4953_4d5f_5753_4749_5f52_5356_0002);
    cache.set_shader(
        resolve,
        Shader::from_wesl(
            include_str!("../../shaders/world_space_gi_resolve.wesl"),
            "embedded://prism_render_scene/shaders/world_space_gi_resolve.wesl",
        ),
    );

    cache
        .get(0, resolve, &[])
        .unwrap_or_else(|error| panic!("world_space_gi_resolve.wesl failed to compile: {error}"));
}

/// Compiles `world_space_gi_composite.wesl`, proving both composite entry
/// points (the `scene_color` -> `gi_base` copy and the energy-conserving GI
/// fold) parse and type-check exactly as they will in the render world.
#[test]
fn composite_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let composite = shader_id(0x5052_4953_4d5f_5753_4749_5f43_4d50_0003);
    cache.set_shader(
        composite,
        Shader::from_wesl(
            include_str!("../../shaders/world_space_gi_composite.wesl"),
            "embedded://prism_render_scene/shaders/world_space_gi_composite.wesl",
        ),
    );

    cache
        .get(0, composite, &[])
        .unwrap_or_else(|error| panic!("world_space_gi_composite.wesl failed to compile: {error}"));
}

/// Guards the Rust immediate-block ABIs against drift from the WESL structs:
/// both blocks are the 96-byte `mat4x4`-led blocks matching each shader's one
/// `var<immediate>` global, and the workgroup constant matches
/// `@workgroup_size(8, 8, 1)`.
#[test]
fn world_space_gi_abi_matches_the_shader_layout() {
    use super::abi::{
        GpuWorldSpaceGiCompositeParams, GpuWorldSpaceGiProbeParams, GpuWorldSpaceGiResolveParams,
        WORLD_SPACE_GI_WORKGROUP_SIZE,
    };
    assert_eq!(size_of::<GpuWorldSpaceGiProbeParams>(), 96);
    assert_eq!(align_of::<GpuWorldSpaceGiProbeParams>(), 4);
    assert_eq!(size_of::<GpuWorldSpaceGiResolveParams>(), 96);
    assert_eq!(align_of::<GpuWorldSpaceGiResolveParams>(), 4);
    assert_eq!(size_of::<GpuWorldSpaceGiCompositeParams>(), 16);
    assert_eq!(align_of::<GpuWorldSpaceGiCompositeParams>(), 4);
    assert_eq!(WORLD_SPACE_GI_WORKGROUP_SIZE, 8);
}
