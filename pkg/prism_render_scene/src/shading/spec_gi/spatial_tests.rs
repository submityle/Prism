//! WESL compile + import-resolution coverage for the glossy-specular ReSTIR
//! **spatial** reuse pass (`shaders/spec_gi_spatial.wesl`) and host-parity
//! coverage for its one piece of new numeric logic — the frame-jittered
//! Fibonacci-spiral neighbour-sampling kernel.
//!
//! The sandbox has no GPU, so the compile test drives `spec_gi_spatial.wesl`
//! through the same `ShaderCache` / `wesl` pipeline the render world uses, with
//! its sole dependency (`spec_gi_reservoir.wesl`) registered under the canonical
//! module path the crate's embedded-asset registration produces. A green result
//! proves the spatial kernel parses and type-checks, every
//! `import prism_render_scene::shaders::spec_gi_reservoir::*` resolves against
//! the committed, parity-tested reservoir module (so the glossy reservoir maths
//! — already proven equal to the CPU golden in [`super::reservoir_tests`] — is
//! *reused*, not duplicated), and the reservoir module's test-only probe anchor
//! tree-shakes out so it does not clash with the spatial kernel's own group-0
//! bindings.
//!
//! The reservoir merge itself (`specr_merge_glossy` / `specr_finalize_glossy` /
//! `specr_glossy_contribution`) is numerically identical between the temporal
//! and spatial passes and is already proven equal to the golden
//! `prism_render_shading::gi::spec_gi::glossy_reservoir` sequence by
//! [`super::reservoir_tests`]; the only logic unique to this pass is the
//! neighbour-tap generation, which the parity test below transcribes op-for-op
//! and checks for the determinism, bounded radius and area-uniform radial
//! growth the kernel relies on.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("spec_gi spatial shader is WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

/// Registers `spec_gi_reservoir.wesl` under its canonical module path and
/// compiles `spec_gi_spatial.wesl`, forcing every
/// `import prism_render_scene::shaders::spec_gi_reservoir::*` to resolve exactly
/// as it will in the render world and proving the spatial kernel type-checks
/// against the committed reservoir maths rather than a local copy.
#[test]
fn spec_gi_spatial_wesl_compiles_and_resolves_imports() {
    let mut cache = ShaderCache::new((), load_source);

    // The spatial kernel's single dependency; itself self-contained (no further
    // imports), registered under the path its `import` statement names.
    let reservoir = shader_id(0x5350_4543_5f47_495f_5245_5356_5f00_0002);
    cache.set_shader(
        reservoir,
        Shader::from_wesl(
            include_str!("../../shaders/spec_gi_reservoir.wesl"),
            "embedded://prism_render_scene/shaders/spec_gi_reservoir.wesl",
        ),
    );

    let spatial = shader_id(0x5350_4543_5f47_495f_5350_4154_5f00_0003);
    cache.set_shader(
        spatial,
        Shader::from_wesl(
            include_str!("../../shaders/spec_gi_spatial.wesl"),
            "embedded://prism_render_scene/shaders/spec_gi_spatial.wesl",
        ),
    );

    cache
        .get(0, spatial, &[])
        .unwrap_or_else(|error| panic!("spec_gi_spatial.wesl failed to compile: {error}"));
}

// --- Host transcription of the WESL neighbour-sampling kernel (op-for-op) ----
// Mirrors `spec_gi_spatial.wesl::{sgs_hash_u32, sgs_rng, sgs_neighbour_offset}`
// so the Fibonacci-spiral disc the GPU kernel pools over is verified for the
// determinism, bounded radius and area-uniform radial growth the resolve relies
// on, without a GPU.

/// 2*pi * (1 - 1/phi); mirrors `SGS_GOLDEN_ANGLE`.
const GOLDEN_ANGLE: f32 = 2.399_963_229_728_653;

/// Rust twin of `spec_gi_spatial.wesl::sgs_hash_u32` (PCG-style integer hash).
fn sgs_hash_u32(x_in: u32) -> u32 {
    let mut x = x_in;
    x = x.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    x = ((x >> ((x >> 28).wrapping_add(4))) ^ x).wrapping_mul(277_803_737);
    (x >> 22) ^ x
}

/// Rust twin of `spec_gi_spatial.wesl::sgs_rng` → uniform float in [0, 1).
fn sgs_rng(pixel: (u32, u32), salt: u32) -> f32 {
    let h = sgs_hash_u32(
        pixel
            .0
            .wrapping_mul(1973)
            .wrapping_add(pixel.1.wrapping_mul(9277))
            .wrapping_add(salt.wrapping_mul(26699))
            .wrapping_add(1),
    );
    (h as f32) * (1.0 / 4_294_967_296.0)
}

/// Rust twin of `spec_gi_spatial.wesl::sgs_neighbour_offset`.
#[allow(clippy::disallowed_methods)]
fn sgs_neighbour_offset(
    pixel: (u32, u32),
    k: u32,
    sample_count: u32,
    radius: f32,
    frame: u32,
) -> (i32, i32) {
    let count = sample_count.max(1);
    let jitter = sgs_rng(pixel, frame.wrapping_mul(977).wrapping_add(7));
    let frac = (k as f32 + 0.5) / count as f32;
    let r = radius * frac.sqrt();
    let angle = (k as f32 + jitter) * GOLDEN_ANGLE;
    let ox = r * angle.cos();
    let oy = r * angle.sin();
    (ox.round() as i32, oy.round() as i32)
}

#[test]
fn neighbour_rng_is_deterministic_and_unit_ranged() {
    for &(px, py) in &[(0u32, 0u32), (640, 360), (1919, 1079)] {
        for salt in 0u32..16 {
            let a = sgs_rng((px, py), salt);
            let b = sgs_rng((px, py), salt);
            assert_eq!(a, b, "RNG must be deterministic for a fixed pixel/salt");
            assert!((0.0..1.0).contains(&a), "RNG {a} out of [0,1)");
        }
    }
}

#[test]
#[allow(clippy::disallowed_methods)]
fn neighbour_offsets_stay_within_the_sampling_disc() {
    // Every tap must land inside the configured pixel radius (after rounding to
    // the integer texel grid, allow the <=0.5-texel rounding slack per axis).
    let radius = 16.0_f32;
    let count = 8u32;
    for &(px, py) in &[(10u32, 10u32), (123, 456), (1900, 1050)] {
        for frame in 0u32..4 {
            for k in 0..count {
                let (ox, oy) = sgs_neighbour_offset((px, py), k, count, radius, frame);
                let dist = ((ox as f32).powi(2) + (oy as f32).powi(2)).sqrt();
                assert!(
                    dist <= radius + 0.75,
                    "tap k={k} at frame {frame} fell outside the disc: dist={dist} radius={radius}"
                );
            }
        }
    }
}

#[test]
#[allow(clippy::disallowed_methods)]
fn neighbour_radius_grows_area_uniformly_with_tap_index() {
    // The pre-round spiral radius is `radius * sqrt((k+0.5)/count)`, strictly
    // increasing in `k`, so taps spread from the centre outward area-uniformly
    // (no clustering). Verify the analytic radius is monotonic in k.
    let radius = 32.0_f32;
    let count = 6u32;
    let mut prev = -1.0_f32;
    for k in 0..count {
        let frac = (k as f32 + 0.5) / count as f32;
        let r = radius * frac.sqrt();
        assert!(r > prev, "spiral radius must grow with tap index k");
        assert!(
            r <= radius,
            "spiral radius must stay within the disc radius"
        );
        prev = r;
    }
}

#[test]
fn per_frame_jitter_decorrelates_the_pattern() {
    // The angular jitter is salted by the frame counter, so two different
    // frames must produce a different neighbour pattern for the same pixel
    // (temporal decorrelation the denoiser + temporal reuse resolve).
    let radius = 16.0_f32;
    let count = 8u32;
    let pixel = (200u32, 300u32);
    let frame_a: Vec<_> = (0..count)
        .map(|k| sgs_neighbour_offset(pixel, k, count, radius, 0))
        .collect();
    let frame_b: Vec<_> = (0..count)
        .map(|k| sgs_neighbour_offset(pixel, k, count, radius, 1))
        .collect();
    assert_ne!(
        frame_a, frame_b,
        "a frame advance must jitter the neighbour pattern"
    );
}
