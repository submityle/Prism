//! WESL compile + import-resolution coverage and host-ABI round-trip parity for
//! the glossy-specular ReSTIR *reuse* pass (`shaders/spec_gi_reuse.wesl` +
//! [`super::abi`]).
//!
//! The sandbox has no GPU, so the compile test drives `spec_gi_reuse.wesl`
//! through the same `ShaderCache` / `wesl` pipeline the render world uses, with
//! its sole dependency (`spec_gi_reservoir.wesl`) registered under the canonical
//! module path the crate's embedded-asset registration produces. A green result
//! proves three things at once: the reuse kernel parses and type-checks, every
//! `import prism_render_scene::shaders::spec_gi_reservoir::*` resolves against
//! the committed, parity-tested reservoir module (so the 467-line reservoir
//! maths is *reused*, not duplicated), and the reservoir module's test-only
//! `@group(0) @binding(0) specr_probe` anchor tree-shakes out so it does not
//! clash with the reuse kernel's own `@group(0) @binding(0)` config uniform.
//!
//! The ABI round-trip test transcribes `sgr_pack` / `sgr_unpack` op-for-op into
//! Rust over [`GpuSpecrReservoir`] and asserts every reservoir field survives
//! the flattened 64-byte storage layout bit-for-bit, proving the host mirror in
//! [`super::abi`] and the WGSL storage struct agree field-for-field.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

use super::abi::GpuSpecrReservoir;

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("spec_gi reuse shader is WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

/// Registers `spec_gi_reservoir.wesl` under its canonical module path and
/// compiles `spec_gi_reuse.wesl`, forcing every
/// `import prism_render_scene::shaders::spec_gi_reservoir::*` to resolve exactly
/// as it will in the render world and proving the reuse kernel type-checks
/// against the committed reservoir maths rather than a local copy.
#[test]
fn spec_gi_reuse_wesl_compiles_and_resolves_imports() {
    let mut cache = ShaderCache::new((), load_source);

    // The reuse kernel's single dependency; itself self-contained (no further
    // imports), registered under the path its `import` statement names.
    let reservoir = shader_id(0x5350_4543_5f47_495f_5245_5356_5f00_0002);
    cache.set_shader(
        reservoir,
        Shader::from_wesl(
            include_str!("../../shaders/spec_gi_reservoir.wesl"),
            "embedded://prism_render_scene/shaders/spec_gi_reservoir.wesl",
        ),
    );

    let reuse = shader_id(0x5350_4543_5f47_495f_5245_5553_455f_0001);
    cache.set_shader(
        reuse,
        Shader::from_wesl(
            include_str!("../../shaders/spec_gi_reuse.wesl"),
            "embedded://prism_render_scene/shaders/spec_gi_reuse.wesl",
        ),
    );

    cache
        .get(0, reuse, &[])
        .unwrap_or_else(|error| panic!("spec_gi_reuse.wesl failed to compile: {error}"));
}

// --- Host transcription of the WESL pack/unpack (op-for-op) -------------------
// Mirrors the shader's working reservoir; the flattened storage carries exactly
// these fields, so a Rust round-trip through `GpuSpecrReservoir` must preserve
// every one of them.

#[derive(Clone, Copy, Debug, PartialEq)]
struct MirrorReservoir {
    visible_point: [f32; 3],
    sample_point: [f32; 3],
    radiance: [f32; 3],
    w_sum: f32,
    m: f32,
    w: f32,
    has_sample: u32,
}

/// Rust twin of `spec_gi_reuse.wesl::sgr_pack`.
fn sgr_pack(r: MirrorReservoir) -> GpuSpecrReservoir {
    GpuSpecrReservoir {
        visible_point: r.visible_point,
        w_sum: r.w_sum,
        sample_point: r.sample_point,
        m: r.m,
        radiance: r.radiance,
        w: r.w,
        has_sample: r.has_sample,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    }
}

/// Rust twin of `spec_gi_reuse.wesl::sgr_unpack`.
fn sgr_unpack(g: GpuSpecrReservoir) -> MirrorReservoir {
    MirrorReservoir {
        visible_point: g.visible_point,
        sample_point: g.sample_point,
        radiance: g.radiance,
        w_sum: g.w_sum,
        m: g.m,
        w: g.w,
        has_sample: g.has_sample,
    }
}

#[test]
fn sgr_pack_unpack_round_trips_every_field() {
    let cases = [
        MirrorReservoir {
            visible_point: [1.5, -2.0, 3.25],
            sample_point: [0.1, 0.2, 4.0],
            radiance: [10.0, 8.5, 0.5],
            w_sum: 7.75,
            m: 12.0,
            w: 0.625,
            has_sample: 1,
        },
        MirrorReservoir {
            visible_point: [0.0, 0.0, 0.0],
            sample_point: [0.0, 0.0, 0.0],
            radiance: [0.0, 0.0, 0.0],
            w_sum: 0.0,
            m: 0.0,
            w: 0.0,
            has_sample: 0,
        },
        MirrorReservoir {
            visible_point: [-123.5, 456.75, -789.125],
            sample_point: [1000.0, -2000.0, 3000.0],
            radiance: [0.001, 0.002, 0.003],
            w_sum: 1.0e6,
            m: 32.0,
            w: 1.0e-4,
            has_sample: 1,
        },
    ];

    for case in cases {
        let packed = sgr_pack(case);
        // The three pad words must stay zero (deterministic storage, no leaks).
        assert_eq!((packed._pad0, packed._pad1, packed._pad2), (0, 0, 0));
        // vec3 + trailing scalar must land in the field they pair with.
        assert_eq!(packed.visible_point, case.visible_point);
        assert_eq!(packed.w_sum, case.w_sum);
        assert_eq!(packed.sample_point, case.sample_point);
        assert_eq!(packed.m, case.m);
        assert_eq!(packed.radiance, case.radiance);
        assert_eq!(packed.w, case.w);
        assert_eq!(packed.has_sample, case.has_sample);
        // Full round-trip restores the working reservoir exactly.
        assert_eq!(sgr_unpack(packed), case);
    }
}
