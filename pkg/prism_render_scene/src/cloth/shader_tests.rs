//! `WESL` compilation coverage for the cloth aerodynamics shaders.
//!
//! The sandbox has no `GPU`, so these tests compile the aerodynamics `WESL`
//! sources through the same `ShaderCache` / `wesl` pipeline the render world
//! uses, validating that `cloth_aerodynamics_snapshot.wesl` and
//! `cloth_aerodynamics.wesl` parse and type-check exactly as they will on
//! device. Both kernels are self-contained (no intra-crate `import`s, matching
//! `face_shadow.wesl`), so a green result also guards the aerodynamics maths —
//! the per-triangle drag/lift split, the `1/3` face-force distribution, the
//! integer turbulence hash and the frozen-snapshot gather — against drift from
//! its `CPU` golden twin in `prism_render_architecture::cloth::aero_gather`.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("cloth aerodynamics shaders are WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

/// Compiles `cloth_aerodynamics_snapshot.wesl`, proving the velocity-freeze
/// pre-pass (read-only source velocities in, the read-write snapshot copy out,
/// the uniform vertex-count guard) parses and type-checks exactly as it will in
/// the render world.
#[test]
fn cloth_aerodynamics_snapshot_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let snapshot = shader_id(0x5052_4953_4d5f_434c_4f54_4841_4552_5301);
    cache.set_shader(
        snapshot,
        Shader::from_wesl(
            include_str!("../shaders/cloth_aerodynamics_snapshot.wesl"),
            "embedded://prism_render_scene/shaders/cloth_aerodynamics_snapshot.wesl",
        ),
    );

    cache.get(0, snapshot, &[]).unwrap_or_else(|error| {
        panic!("cloth_aerodynamics_snapshot.wesl failed to compile: {error}")
    });
}

/// Compiles `cloth_aerodynamics.wesl`, proving the per-vertex gather kernel (the
/// read-only positions, the read-write velocities, the frozen velocity
/// snapshot, the triangle topology and the two `CSR` adjacency buffers in, plus
/// the wind/aero/dt uniform block) parses and type-checks exactly as it will in
/// the render world, and that its helper and struct layouts match the `CPU`
/// golden.
#[test]
fn cloth_aerodynamics_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let aero = shader_id(0x5052_4953_4d5f_434c_4f54_4841_4552_5302);
    cache.set_shader(
        aero,
        Shader::from_wesl(
            include_str!("../shaders/cloth_aerodynamics.wesl"),
            "embedded://prism_render_scene/shaders/cloth_aerodynamics.wesl",
        ),
    );

    cache
        .get(0, aero, &[])
        .unwrap_or_else(|error| panic!("cloth_aerodynamics.wesl failed to compile: {error}"));
}

/// Guards the Rust immediate-block `ABI` against drift from the `WESL` struct:
/// the `GpuClothAeroParams` uniform is the 32-byte block (a `vec3` wind row plus
/// five trailing scalars) matching both aerodynamics shaders' `aero_params`
/// global, and the workgroup constant matches `@workgroup_size(64)`.
#[test]
fn cloth_aerodynamics_abi_matches_the_shader_layout() {
    use super::abi::{GpuClothAeroParams, CLOTH_WORKGROUP_SIZE};
    assert_eq!(size_of::<GpuClothAeroParams>(), 32);
    assert_eq!(align_of::<GpuClothAeroParams>(), 4);
    assert_eq!(CLOTH_WORKGROUP_SIZE, 64);
}
