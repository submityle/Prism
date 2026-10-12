//! Device-free WGSL parse gate for every single-sourced shader fragment.
//!
//! The `*_parity` tests skip silently when [`GpuContext::try_headless`] returns
//! `None`, so on a device-less host a fragment that does not even compile would
//! still report green. This test closes that gap: it composes each kernel's WGSL
//! exactly as its module does and parses it with the same front end `wgpu` uses
//! ([`wgpu::naga::front::wgsl`]), so a syntax error, an unknown builtin (e.g. a
//! WGSL-Spec name the pinned Naga does not implement), or an unresolved
//! identifier fails **without any GPU adapter**.
//!
//! Parsing alone does not prove runtime parity — the `*_parity` suites still own
//! that on real hardware — but it guarantees every fragment is at least a valid
//! module on the pinned toolchain.

use wgpu::naga::valid::{Capabilities, ValidationFlags, Validator};

use prism_math::shader_mirror::{
    WGSL_AABB_OPS, WGSL_DUAL, WGSL_DUAL_QUAT_ALG, WGSL_DUAL_QUAT_SKIN, WGSL_EASING, WGSL_F16,
    WGSL_FRACTAL, WGSL_FRUSTUM_CULL, WGSL_HILBERT2, WGSL_HILBERT3, WGSL_HSL, WGSL_I16F16,
    WGSL_INTERVAL, WGSL_LOOK_AT_RH, WGSL_MORTON, WGSL_OCTAHEDRAL, WGSL_OKLAB, WGSL_OVERLAP,
    WGSL_PACK16, WGSL_PACK8, WGSL_PCG32, WGSL_PERLIN, WGSL_PLANE, WGSL_PROJECTION_RH,
    WGSL_QUAT_ALG, WGSL_QUAT_INTERP, WGSL_QUAT_ROTATE, WGSL_RAYCAST, WGSL_RAYTRI, WGSL_SEGMENT,
    WGSL_SH3_EVAL, WGSL_SH_COSINE, WGSL_SIMPLEX, WGSL_SPLINE, WGSL_SPLITMIX64, WGSL_SRGB,
    WGSL_SURFACE, WGSL_TEMPERATURE, WGSL_U64, WGSL_XOSHIRO256SS, WGSL_XYZ,
};

/// The two-line base-noise forwarding alias the fractal host prepends (inlined
/// here because the module keeps it private).
const ALIAS_PERLIN: &str = "\
fn prism_base_sample2(x: f32, y: f32) -> f32 { return prism_perlin_get2(x, y); }\n\
fn prism_base_sample3(x: f32, y: f32, z: f32) -> f32 { return prism_perlin_get3(x, y, z); }\n";

/// The two-line base-noise forwarding alias for the simplex source.
const ALIAS_SIMPLEX: &str = "\
fn prism_base_sample2(x: f32, y: f32) -> f32 { return prism_simplex_get2(x, y); }\n\
fn prism_base_sample3(x: f32, y: f32, z: f32) -> f32 { return prism_simplex_get3(x, y, z); }\n";

/// Every composed kernel source, labelled by the module that builds it.
fn composed_sources() -> Vec<(&'static str, String)> {
    let join = |fragments: &[&str]| fragments.join("\n");
    vec![
        ("quat", WGSL_QUAT_ROTATE.into()),
        ("quat_alg", WGSL_QUAT_ALG.into()),
        ("dual_quat_alg", join(&[WGSL_QUAT_ALG, WGSL_DUAL_QUAT_ALG])),
        (
            "skinning",
            join(&[WGSL_QUAT_ROTATE, WGSL_QUAT_ALG, WGSL_DUAL_QUAT_SKIN]),
        ),
        ("quat_interp", WGSL_QUAT_INTERP.into()),
        ("projection", WGSL_PROJECTION_RH.into()),
        ("view", WGSL_LOOK_AT_RH.into()),
        ("segment", WGSL_SEGMENT.into()),
        ("plane", WGSL_PLANE.into()),
        ("aabb_ops", WGSL_AABB_OPS.into()),
        ("dual", WGSL_DUAL.into()),
        ("interval", WGSL_INTERVAL.into()),
        ("i16f16", format!("{WGSL_U64}\n{WGSL_I16F16}")),
        ("pcg", join(&[WGSL_U64, WGSL_PCG32])),
        ("rng64_splitmix", join(&[WGSL_U64, WGSL_SPLITMIX64])),
        (
            "rng64_xoshiro",
            join(&[WGSL_U64, WGSL_SPLITMIX64, WGSL_XOSHIRO256SS]),
        ),
        ("sh_cosine", WGSL_SH_COSINE.into()),
        ("sh3_eval", WGSL_SH3_EVAL.into()),
        ("raycast", WGSL_RAYCAST.into()),
        ("raytri", WGSL_RAYTRI.into()),
        ("overlap", WGSL_OVERLAP.into()),
        ("frustum_cull", WGSL_FRUSTUM_CULL.into()),
        ("octahedral", WGSL_OCTAHEDRAL.into()),
        ("morton", WGSL_MORTON.into()),
        ("hilbert2", WGSL_HILBERT2.into()),
        ("hilbert3", WGSL_HILBERT3.into()),
        ("f16", WGSL_F16.into()),
        ("srgb", WGSL_SRGB.into()),
        ("pack8", WGSL_PACK8.into()),
        ("pack16", WGSL_PACK16.into()),
        ("oklab", WGSL_OKLAB.into()),
        ("xyz", WGSL_XYZ.into()),
        ("hsl", WGSL_HSL.into()),
        ("temperature", WGSL_TEMPERATURE.into()),
        ("easing", WGSL_EASING.into()),
        ("spline", WGSL_SPLINE.into()),
        ("surface", join(&[WGSL_SPLINE, WGSL_SURFACE])),
        ("perlin", WGSL_PERLIN.into()),
        ("simplex", WGSL_SIMPLEX.into()),
        (
            "fractal_perlin",
            join(&[WGSL_PERLIN, ALIAS_PERLIN, WGSL_FRACTAL]),
        ),
        (
            "fractal_simplex",
            join(&[WGSL_SIMPLEX, ALIAS_SIMPLEX, WGSL_FRACTAL]),
        ),
    ]
}

#[test]
fn every_fragment_parses_on_the_pinned_naga() {
    let sources = composed_sources();
    assert_eq!(sources.len(), 41, "update the count when adding a fragment");
    for (label, source) in sources {
        match wgpu::naga::front::wgsl::parse_str(&source) {
            Err(err) => {
                panic!("WGSL fragment `{label}` failed to parse: {err}\n--- source ---\n{source}")
            }
            Ok(module) => {
                // The f16 fragment uses `pack2x16float`, which Naga gates behind
                // this capability; every other fragment is core WGSL.
                let mut validator = Validator::new(
                    ValidationFlags::all(),
                    Capabilities::SHADER_FLOAT16_IN_FLOAT32,
                );
                if let Err(err) = validator.validate(&module) {
                    panic!(
                        "WGSL fragment `{label}` failed validation: {err:?}\n--- source ---\n{source}"
                    );
                }
            }
        }
    }
}

#[test]
fn batch_dispatch_regressions_remain_guarded() {
    // These wrappers read and write storage arrays directly. Keep both sides
    // of the guard: the host normally allocates matching buffers, but a
    // mismatched binding must fail closed instead of becoming an out-of-range
    // shader access.
    for (module, names) in [
        ("affine", ["src", "dst"]),
        ("mat4", ["src", "dst"]),
        ("octahedral", ["src", "dst"]),
        ("sphere", ["src", "dst"]),
        ("frustum_cull", ["spheres", "result"]),
    ] {
        let source = match module {
            "affine" => include_str!("../src/affine/mod.rs"),
            "mat4" => include_str!("../src/mat4/mod.rs"),
            "octahedral" => include_str!("../src/octahedral/mod.rs"),
            "sphere" => include_str!("../src/sphere/mod.rs"),
            "frustum_cull" => include_str!("../src/frustum_cull/mod.rs"),
            _ => unreachable!(),
        };
        for name in names {
            assert!(
                source.contains(&format!("arrayLength(&{name})")),
                "{module} wrapper lost its {name} storage-length guard"
            );
        }
    }

    let frustum = include_str!("../src/frustum_cull/mod.rs");
    assert!(frustum.contains("planes: [[f32; 4]; 6],\n    count: u32,\n    _pad: [u32; 3]"));

    // Every standard batch dispatch must use the checked host conversion;
    // this catches a future reintroduction of silent usize -> u32 wrapping.
    for source in [
        include_str!("../src/affine/mod.rs"),
        include_str!("../src/aabb_ops/mod.rs"),
        include_str!("../src/dual/mod.rs"),
        include_str!("../src/dual_quat_alg/mod.rs"),
        include_str!("../src/easing/mod.rs"),
        include_str!("../src/f16/mod.rs"),
        include_str!("../src/fractal/mod.rs"),
        include_str!("../src/frustum_cull/mod.rs"),
        include_str!("../src/hilbert/mod.rs"),
        include_str!("../src/hsl/mod.rs"),
        include_str!("../src/i16f16/mod.rs"),
        include_str!("../src/interval/mod.rs"),
        include_str!("../src/mat4/mod.rs"),
        include_str!("../src/morton/mod.rs"),
        include_str!("../src/octahedral/mod.rs"),
        include_str!("../src/oklab/mod.rs"),
        include_str!("../src/overlap/mod.rs"),
        include_str!("../src/pack16/mod.rs"),
        include_str!("../src/pack8/mod.rs"),
        include_str!("../src/pcg/mod.rs"),
        include_str!("../src/perlin/mod.rs"),
        include_str!("../src/plane/mod.rs"),
        include_str!("../src/quat/mod.rs"),
        include_str!("../src/quat_alg/mod.rs"),
        include_str!("../src/quat_interp/mod.rs"),
        include_str!("../src/raycast/mod.rs"),
        include_str!("../src/raytri/mod.rs"),
        include_str!("../src/rng64/mod.rs"),
        include_str!("../src/segment/mod.rs"),
        include_str!("../src/sh_batch/mod.rs"),
        include_str!("../src/sh_cosine/mod.rs"),
        include_str!("../src/simplex/mod.rs"),
        include_str!("../src/sphere/mod.rs"),
        include_str!("../src/spline/mod.rs"),
        include_str!("../src/srgb/mod.rs"),
        include_str!("../src/surface/mod.rs"),
        include_str!("../src/temperature/mod.rs"),
        include_str!("../src/xyz/mod.rs"),
    ] {
        assert!(
            source.contains("crate::dispatch::checked_workgroup_count"),
            "batch module lost checked workgroup dispatch"
        );
    }
}
