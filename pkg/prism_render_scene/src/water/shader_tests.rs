//! `WESL` compilation coverage for the water subsystem shaders.
//!
//! These tests compile the water `WESL` sources through the same
//! [`ShaderCache`] / `wesl` pipeline the render world uses, so a green result
//! proves every source parses and type-checks exactly as it will on device and
//! that the intra-crate `import prism_render_scene::shaders::...` statements
//! resolve against the crate's embedded-asset module paths
//! (`embedded://prism_render_scene/shaders/<name>.wesl`, byte-identical to what
//! `load_shader_library!` produces from `lib.rs`).
//!
//! The five compute shaders (`water_ocean`, `water_flip`, `water_pbf`,
//! `water_surface`, `water_render_fx`) are self-contained, so compiling them
//! also guards the solver maths - the `Tessendorf` spectrum `IFFT`, the analytic
//! `Gerstner` fan, the `FLIP`/`APIC` `P2G`/`G2P` transfer and pressure
//! projection, the `PBF` density constraint and crest-spray emitter, the `SWE`
//! step, the foam advection, the waterline mask, and the caustics / dispersion /
//! underwater / wetness / coupling render passes - against drift from their
//! `CPU` golden twins in `prism_render_architecture::water`. The
//! `water.wesl` single-layer BSDF lobe additionally resolves the shared
//! `prism_render_scene::shaders::{brdf, lighting}` helpers, matching the render
//! world's `SingleLayerWater` closure.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("water shaders are WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

/// Compiles a single self-contained water compute shader, panicking with the
/// shader name on any parse / type-check failure.
fn compile_standalone(source: &'static str, path: &'static str, tag: u128) {
    let mut cache = ShaderCache::new((), load_source);
    let id = shader_id(tag);
    cache.set_shader(id, Shader::from_wesl(source, path));
    cache
        .get(0, id, &[])
        .unwrap_or_else(|error| panic!("{path} failed to compile: {error}"));
}

/// The spectral inverse-`FFT` (`Tessendorf`) and analytic `Gerstner`
/// superposition kernels, sharing one nine-binding `@group(0)`.
#[test]
fn water_ocean_wesl_compiles_standalone() {
    compile_standalone(
        include_str!("../shaders/water_ocean.wesl"),
        "embedded://prism_render_scene/shaders/water_ocean.wesl",
        0x5052_4953_4d5f_5741_5445_524f_4345_4e01,
    );
}

/// The three `FLIP`/`APIC` passes (`P2G` scatter, pressure projection, `G2P`
/// gather) and the screen-space surface reconstruction.
#[test]
fn water_flip_wesl_compiles_standalone() {
    compile_standalone(
        include_str!("../shaders/water_flip.wesl"),
        "embedded://prism_render_scene/shaders/water_flip.wesl",
        0x5052_4953_4d5f_5741_5445_5246_4c49_5001,
    );
}

/// The `PBF` density solve and the crest-spray emitter (two distinct
/// `@group(0)` resource sets in one file).
#[test]
fn water_pbf_wesl_compiles_standalone() {
    compile_standalone(
        include_str!("../shaders/water_pbf.wesl"),
        "embedded://prism_render_scene/shaders/water_pbf.wesl",
        0x5052_4953_4d5f_5741_5445_5250_4246_0001,
    );
}

/// The `SWE` step (`@group(0)`), foam advection (`@group(1)`) and waterline
/// mask (`@group(2)`), each on its own group index.
#[test]
fn water_surface_wesl_compiles_standalone() {
    compile_standalone(
        include_str!("../shaders/water_surface.wesl"),
        "embedded://prism_render_scene/shaders/water_surface.wesl",
        0x5052_4953_4d5f_5741_5445_5253_5552_0001,
    );
}

/// The caustics projection, spectral dispersion refract, underwater volume,
/// wetness step and coupling readback render passes (`@group(0..=4)`).
#[test]
fn water_render_fx_wesl_compiles_standalone() {
    compile_standalone(
        include_str!("../shaders/water_render_fx.wesl"),
        "embedded://prism_render_scene/shaders/water_render_fx.wesl",
        0x5052_4953_4d5f_5741_5445_5246_5800_0001,
    );
}

/// The standalone ping-pong butterfly `FFT` (`water_fft_bitrev` /
/// `water_fft_stage` / `water_fft_normalize`), the real-device twin of the
/// `CPU` golden `prism_render_architecture::water::fft::ifft2`.
#[test]
fn water_butterfly_wesl_compiles_standalone() {
    compile_standalone(
        include_str!("../shaders/water_butterfly.wesl"),
        "embedded://prism_render_scene/shaders/water_butterfly.wesl",
        0x5052_4953_4d5f_5741_5445_5242_5546_0001,
    );
}

/// Registers `lighting.wesl` and `brdf.wesl` under their canonical module paths
/// and compiles `water.wesl`, forcing the importer to resolve the
/// `prism_render_scene::shaders::{brdf, lighting}::{...}` imports the
/// single-layer water BSDF lobe depends on.
#[test]
fn water_bsdf_wesl_compiles_and_resolves_imports() {
    let mut cache = ShaderCache::new((), load_source);

    let lighting = shader_id(0x5052_4953_4d5f_4c49_4748_5449_4e47_00a1);
    cache.set_shader(
        lighting,
        Shader::from_wesl(
            include_str!("../shaders/lighting.wesl"),
            "embedded://prism_render_scene/shaders/lighting.wesl",
        ),
    );

    let brdf = shader_id(0x5052_4953_4d5f_4252_4446_0000_0000_00a1);
    cache.set_shader(
        brdf,
        Shader::from_wesl(
            include_str!("../shaders/brdf.wesl"),
            "embedded://prism_render_scene/shaders/brdf.wesl",
        ),
    );

    let water = shader_id(0x5052_4953_4d5f_5741_5445_5242_5344_0001);
    cache.set_shader(
        water,
        Shader::from_wesl(
            include_str!("../shaders/water.wesl"),
            "embedded://prism_render_scene/shaders/water.wesl",
        ),
    );

    cache
        .get(0, water, &[])
        .unwrap_or_else(|error| panic!("water.wesl failed to compile/resolve imports: {error}"));
}
