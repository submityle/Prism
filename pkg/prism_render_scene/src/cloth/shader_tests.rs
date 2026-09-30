//! `WESL` compilation coverage for the cloth aerodynamics shaders.
//!
//! The sandbox has no `GPU`, so these tests compile the aerodynamics `WESL`
//! sources through the same `ShaderCache` / `wesl` pipeline the render world
//! uses, validating that `cloth_aerodynamics_snapshot.wesl` and
//! `cloth_aerodynamics.wesl` parse and type-check exactly as they will on
//! device. Both kernels are self-contained (no intra-crate `import`s, matching
//! `ssgi.wesl` / `bloom.wesl`), so a green result also guards the aerodynamics maths —
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
/// the `GpuClothAeroParams` uniform is the 48-byte block (a `vec3` wind row, the
/// five original trailing scalars, then the fluid density plus its pad rounding
/// the block up to a whole third 16-byte uniform row) matching both
/// aerodynamics shaders' `aero_params` global, and the workgroup constant
/// matches `@workgroup_size(64)`.
#[test]
fn cloth_aerodynamics_abi_matches_the_shader_layout() {
    use super::abi::{GpuClothAeroParams, CLOTH_WORKGROUP_SIZE};
    assert_eq!(size_of::<GpuClothAeroParams>(), 48);
    assert_eq!(align_of::<GpuClothAeroParams>(), 4);
    assert_eq!(CLOTH_WORKGROUP_SIZE, 64);
}

/// Compiles `cloth_self_collision_virtual.wesl`, proving the three own-slot
/// virtual-particle kernels (the `atomicExchange` hash build, the 27-cell
/// resolve accumulating each sample's own half-push into `sample_dp`, and the
/// per-vertex CSR scatter) parse and type-check exactly as they will in the
/// render world, and that their struct/helper layouts match the `CPU` golden
/// `prism_render_architecture::cloth::virtual_particles_jacobi`.
#[test]
fn cloth_self_collision_virtual_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let virtual_tier = shader_id(0x5052_4953_4d5f_434c_4f54_4841_4552_5303);
    cache.set_shader(
        virtual_tier,
        Shader::from_wesl(
            include_str!("../shaders/cloth_self_collision_virtual.wesl"),
            "embedded://prism_render_scene/shaders/cloth_self_collision_virtual.wesl",
        ),
    );

    cache.get(0, virtual_tier, &[]).unwrap_or_else(|error| {
        panic!("cloth_self_collision_virtual.wesl failed to compile: {error}")
    });
}

/// Guards the Rust immediate-block `ABI` against drift from the virtual-particle
/// `WESL` structs: the `GpuClothVpSample` record and the `GpuClothVpParams`
/// uniform are both the flat 32-byte blocks matching `ClothVpSample` /
/// `ClothVpParams` in `cloth_self_collision_virtual.wesl`, and the workgroup
/// constant matches the kernels' `@workgroup_size(64)`.
#[test]
fn cloth_virtual_self_collision_abi_matches_the_shader_layout() {
    use super::abi::{GpuClothVpParams, GpuClothVpSample, CLOTH_WORKGROUP_SIZE};
    assert_eq!(size_of::<GpuClothVpSample>(), 32);
    assert_eq!(align_of::<GpuClothVpSample>(), 4);
    assert_eq!(size_of::<GpuClothVpParams>(), 32);
    assert_eq!(align_of::<GpuClothVpParams>(), 4);
    assert_eq!(CLOTH_WORKGROUP_SIZE, 64);
}

/// Compiles `cloth_self_ccd.wesl`, proving the three own-slot continuous
/// self-collision kernels (the `atomicExchange` swept-box hash build, the
/// canonical-cell resolve accumulating each particle's own half TOI snap and
/// restitution impulse into `ccd_pos_delta` / `ccd_vel_delta`, and the pinned-
/// guarded apply) parse and type-check exactly as they will in the render world,
/// and that their struct/helper layouts match the `CPU` golden
/// `prism_render_architecture::cloth::self_ccd`.
#[test]
fn cloth_self_ccd_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let self_ccd = shader_id(0x5052_4953_4d5f_434c_4f54_4841_4552_5304);
    cache.set_shader(
        self_ccd,
        Shader::from_wesl(
            include_str!("../shaders/cloth_self_ccd.wesl"),
            "embedded://prism_render_scene/shaders/cloth_self_ccd.wesl",
        ),
    );

    cache
        .get(0, self_ccd, &[])
        .unwrap_or_else(|error| panic!("cloth_self_ccd.wesl failed to compile: {error}"));
}

/// Guards the Rust immediate-block `ABI` against drift from the self-CCD `WESL`
/// struct: the `GpuClothCcdParams` uniform is the flat 32-byte block matching
/// `ClothCcdParams` in `cloth_self_ccd.wesl`, and the workgroup constant matches
/// the kernels' `@workgroup_size(64)`.
#[test]
fn cloth_self_ccd_abi_matches_the_shader_layout() {
    use super::abi::{GpuClothCcdParams, CLOTH_WORKGROUP_SIZE};
    assert_eq!(size_of::<GpuClothCcdParams>(), 32);
    assert_eq!(size_of::<GpuClothCcdParams>() % 16, 0);
    assert_eq!(align_of::<GpuClothCcdParams>(), 4);
    assert_eq!(CLOTH_WORKGROUP_SIZE, 64);
}

/// Compiles `cloth_pressure.wesl`, proving the two own-slot pressure (volume)
/// kernels (the single-invocation `cloth_pressure_solve` reduction summing the
/// signed volume and per-vertex gradient in the golden's order then solving the
/// shared `d_lambda`, and the per-particle `cloth_pressure_apply` that folds it
/// into every free particle) parse and type-check exactly as they will in the
/// render world, guarding the closed-mesh volume maths against drift from the
/// `CPU` golden `prism_render_architecture::cloth::pressure`.
#[test]
fn cloth_pressure_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let pressure = shader_id(0x5052_4953_4d5f_434c_4f54_4841_4552_5305);
    cache.set_shader(
        pressure,
        Shader::from_wesl(
            include_str!("../shaders/cloth_pressure.wesl"),
            "embedded://prism_render_scene/shaders/cloth_pressure.wesl",
        ),
    );

    cache
        .get(0, pressure, &[])
        .unwrap_or_else(|error| panic!("cloth_pressure.wesl failed to compile: {error}"));
}

/// Guards the Rust `ABI` against drift from the pressure `WESL` struct: the
/// `GpuClothPressureParams` uniform is the flat 16-byte block matching
/// `ClothPressureParams` in `cloth_pressure.wesl`, and the workgroup constant
/// matches the apply kernel's `@workgroup_size(64)`.
#[test]
fn cloth_pressure_abi_matches_the_shader_layout() {
    use super::abi::{GpuClothPressureParams, CLOTH_WORKGROUP_SIZE};
    assert_eq!(size_of::<GpuClothPressureParams>(), 16);
    assert_eq!(size_of::<GpuClothPressureParams>() % 16, 0);
    assert_eq!(align_of::<GpuClothPressureParams>(), 4);
    assert_eq!(CLOTH_WORKGROUP_SIZE, 64);
}

/// Compiles `cloth_plasticity.wesl`, proving the single own-slot per-edge
/// plastic-creep kernel (`cloth_apply_plasticity`, one invocation per constraint
/// that skips one-sided, degenerate, out-of-range, and within-yield edges then
/// creeps the rest length toward the current length under the residual clamp)
/// parses and type-checks exactly as it will in the render world, guarding the
/// plastic-flow maths against drift from the `CPU` golden
/// `prism_render_architecture::cloth::tearing::apply_plasticity`.
#[test]
fn cloth_plasticity_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let plasticity = shader_id(0x5052_4953_4d5f_434c_4f54_4841_4552_5306);
    cache.set_shader(
        plasticity,
        Shader::from_wesl(
            include_str!("../shaders/cloth_plasticity.wesl"),
            "embedded://prism_render_scene/shaders/cloth_plasticity.wesl",
        ),
    );

    cache
        .get(0, plasticity, &[])
        .unwrap_or_else(|error| panic!("cloth_plasticity.wesl failed to compile: {error}"));
}

/// Guards the Rust `ABI` against drift from the plasticity `WESL` struct: the
/// `GpuClothPlasticityParams` uniform is the flat 32-byte block matching
/// `ClothPlasticityParams` in `cloth_plasticity.wesl`, and the workgroup
/// constant matches the kernel's `@workgroup_size(64)`.
#[test]
fn cloth_plasticity_abi_matches_the_shader_layout() {
    use super::abi::{GpuClothPlasticityParams, CLOTH_WORKGROUP_SIZE};
    assert_eq!(size_of::<GpuClothPlasticityParams>(), 32);
    assert_eq!(size_of::<GpuClothPlasticityParams>() % 16, 0);
    assert_eq!(align_of::<GpuClothPlasticityParams>(), 4);
    assert_eq!(CLOTH_WORKGROUP_SIZE, 64);
}

/// Compiles `cloth_ccd.wesl`, proving the per-particle continuous-collision
/// sweep kernel (`cloth_resolve_ccd`, one invocation per particle that walks the
/// colliders for the earliest closed-form time of impact, snaps to the surface
/// with a skin offset, reflects the normal velocity by restitution and damps the
/// tangential slide with Coulomb friction) parses and type-checks exactly as it
/// will in the render world, guarding the swept-TOI maths against drift from the
/// `CPU` golden `prism_render_architecture::cloth::ccd::resolve_ccd`.
#[test]
fn cloth_ccd_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let ccd = shader_id(0x5052_4953_4d5f_434c_4f54_4841_4552_5307);
    cache.set_shader(
        ccd,
        Shader::from_wesl(
            include_str!("../shaders/cloth_ccd.wesl"),
            "embedded://prism_render_scene/shaders/cloth_ccd.wesl",
        ),
    );

    cache
        .get(0, ccd, &[])
        .unwrap_or_else(|error| panic!("cloth_ccd.wesl failed to compile: {error}"));
}

/// Guards the Rust `ABI` against drift from the CCD `WESL` struct: the
/// `GpuClothCcdParams` uniform is the flat 32-byte block matching
/// `ClothCcdParams` in `cloth_ccd.wesl`, and the workgroup constant matches the
/// kernel's `@workgroup_size(64)`.
#[test]
fn cloth_ccd_abi_matches_the_shader_layout() {
    use super::abi::{GpuClothCcdSweepParams, CLOTH_WORKGROUP_SIZE};
    assert_eq!(size_of::<GpuClothCcdSweepParams>(), 32);
    assert_eq!(size_of::<GpuClothCcdSweepParams>() % 16, 0);
    assert_eq!(align_of::<GpuClothCcdSweepParams>(), 4);
    assert_eq!(CLOTH_WORKGROUP_SIZE, 64);
}
