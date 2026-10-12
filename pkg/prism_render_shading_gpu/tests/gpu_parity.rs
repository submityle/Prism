//! GPU-vs-CPU parity for the PBR surface mirror.
//!
//! The harness builds one list of [`GpuSurfaceParams`] and drives *both* sides
//! from it: the same struct is uploaded as the shader's uniform batch, and it is
//! projected back into the CPU golden's `SurfaceSample` / `ShadingFrame` /
//! `DirectLightSample`. Because the two evaluations read identical inputs, any
//! disagreement is a real difference in the BRDF math rather than a harness
//! bookkeeping mismatch.
//!
//! Without an adapter the test returns early, so the suite still passes on a
//! GPU-less host; with `PRISM_REQUIRE_GPU` set, a missing adapter fails instead,
//! so a run meant to exercise hardware cannot silently degrade into a no-op.

use prism_render_driver::{CommandEncoder, RenderQueue};
use prism_render_driver_wgpu::{WgpuDevice, WgpuInstance, WgpuQueue};
use prism_render_shading::{
    evaluate_principled_direct, DirectLightSample, ShadingFrame, SurfaceSample,
};
use prism_render_shading_gpu::{GpuSurfaceParams, SurfacePass, MAX_BATCH, RESULT_STRIDE};

/// The relative part of the tolerance: `abs_error <= ABS + REL * |cpu|`.
///
/// The mirror performs the same operations in the same order as the CPU golden,
/// so the only divergence is floating-point rounding: the CPU is a `f32` scalar
/// pipeline, while the GPU may contract `a*b + c` into a fused multiply-add and
/// evaluates the one transcendental in the path (`exp2`, inside the Karis
/// environment-BRDF fit that drives the multiple-scattering compensation) with
/// its own implementation rather than the CPU's `libm` one. Both sources are
/// bounded by a few ulp.
///
/// The gate is set from measurement, not assumption. On Apple M2 / Metal the
/// worst channel over the 111-case grid is `4.7e-6` relative, so `5.0e-5` leaves
/// roughly a 10x margin for a different driver's `exp2` and FMA choices while
/// staying far below any genuine formula error: the one real defect this test
/// has caught (a swapped coefficient in the `env_brdf_approx` fit) moved
/// channels by `2e-4` to `8e-3` relative, at least 4x above the gate.
const TOLERANCE_REL: f32 = 5.0e-5;

/// The absolute part of the tolerance, for channels whose value is near zero
/// (where a relative bound is meaningless). It sits above the `f32`
/// representation error of a small positive result produced by cancellation.
///
/// Absolute error is not the primary gate here: a near-mirror metal head-on
/// reaches channel values in the thousands, so its worst absolute error is
/// `~7.8e-3` while still being only `~7.8e-6` relative.
const TOLERANCE_ABS: f32 = 1.0e-6;

/// Brings up a device/queue pair, or `None` when no GPU is available.
fn device_or_skip() -> Option<(WgpuDevice, WgpuQueue)> {
    WgpuInstance::new().request_default_device().ok()
}

/// When `PRISM_REQUIRE_GPU` is set, a missing adapter fails the test.
fn require_gpu() -> bool {
    std::env::var_os("PRISM_REQUIRE_GPU").is_some()
}

/// One shading geometry: the frame plus the light, in the GPU struct's own field
/// form so the CPU golden is built from the same numbers.
#[derive(Clone, Copy, Debug)]
struct Geometry {
    normal: [f32; 3],
    view: [f32; 3],
    direction: [f32; 3],
    tangent: [f32; 3],
    bitangent: [f32; 3],
}

// Every direction below is an unnormalised integer vector: the shader and the
// CPU golden each run it through their own `normalize_or`, so the two agree
// exactly while the harness avoids hand-rounded decimal literals.

/// Head-on: the half-vector sits on the normal, so the specular lobe is at its
/// peak and the Fresnel is at normal incidence.
const HEAD_ON: Geometry = Geometry {
    normal: [0.0, 1.0, 0.0],
    view: [0.0, 1.0, 0.0],
    direction: [0.0, 1.0, 0.0],
    tangent: [1.0, 0.0, 0.0],
    bitangent: [0.0, 0.0, -1.0],
};

/// A 45-degree view against a 45-degree light on the opposite azimuth, so the
/// half-vector is tilted well off the normal.
const FORTY_FIVE: Geometry = Geometry {
    normal: [0.0, 1.0, 0.0],
    view: [1.0, 1.0, 0.0],
    direction: [-1.0, 1.0, 0.0],
    tangent: [1.0, 0.0, 0.0],
    bitangent: [0.0, 0.0, -1.0],
};

/// A steep grazing view (about 10 degrees off the horizon, `n.v ~ 0.16`)
/// against a nearly overhead light: the regime where the exact dielectric
/// Fresnel and the metal `F82`-tint differ most from a plain Schlick curve.
const GRAZING: Geometry = Geometry {
    normal: [0.0, 1.0, 0.0],
    view: [6.0, 1.0, 0.0],
    direction: [-1.0, 8.0, 0.0],
    tangent: [1.0, 0.0, 0.0],
    bitangent: [0.0, 0.0, -1.0],
};

/// A near-normal view against a 30-degree light: a mid, non-degenerate angle.
const NARROW: Geometry = Geometry {
    normal: [0.0, 1.0, 0.0],
    view: [1.0, 3.0, 0.0],
    direction: [1.0, 2.0, 0.0],
    tangent: [1.0, 0.0, 0.0],
    bitangent: [0.0, 0.0, -1.0],
};

/// The light is behind the surface, so `n.l <= 0` and the shader must return the
/// emissive term alone (exercising the early-out).
const BEHIND: Geometry = Geometry {
    normal: [0.0, 1.0, 0.0],
    view: [0.0, 1.0, 0.0],
    direction: [0.0, -1.0, 0.0],
    tangent: [1.0, 0.0, 0.0],
    bitangent: [0.0, 0.0, -1.0],
};

/// A tilted normal with its own orthogonal tangent frame, so the mirror's
/// `orthonormal_basis` / re-orthonormalisation path runs away from the trivial
/// `+Y` axis.
const TILTED: Geometry = Geometry {
    normal: [3.0, 4.0, 0.0],
    view: [1.0, 2.0, 0.0],
    direction: [2.0, 1.0, 0.0],
    tangent: [4.0, -3.0, 0.0],
    bitangent: [0.0, 0.0, -1.0],
};

/// Applies a geometry to a material, producing the struct both sides consume.
fn with_geometry(mut material: GpuSurfaceParams, geometry: Geometry) -> GpuSurfaceParams {
    material.normal = geometry.normal;
    material.view = geometry.view;
    material.direction = geometry.direction;
    material.tangent = geometry.tangent;
    material.bitangent = geometry.bitangent;
    material
}

/// The grid of configurations the parity test evaluates.
///
/// It spans the regimes the brief names — dielectric vs metal, a roughness
/// sweep, grazing angles, and `n.l <= 0` (emissive only) — plus the base-lobe
/// controls that would otherwise go untested: the `normal_variance` width
/// filter, anisotropy, the specular IOR/weight/colour group, `diffuse_roughness`
/// (Oren-Nayar vs Lambert), `base_weight`, and shadow visibility.
fn cases() -> Vec<GpuSurfaceParams> {
    let mut out = Vec::new();
    let geometries = [HEAD_ON, FORTY_FIVE, GRAZING, NARROW];

    // Core sweep: metalness x roughness x geometry.
    for metallic in [0.0_f32, 0.35, 1.0] {
        for perceptual_roughness in [0.045_f32, 0.1, 0.25, 0.5, 0.75, 1.0] {
            for geometry in geometries {
                out.push(with_geometry(
                    GpuSurfaceParams {
                        metallic,
                        perceptual_roughness,
                        ..Default::default()
                    },
                    geometry,
                ));
            }
        }
    }

    // `n.l <= 0`: emissive only, with and without a real emissive term.
    out.push(with_geometry(
        GpuSurfaceParams {
            emissive: [0.25, 0.5, 0.125],
            ..Default::default()
        },
        BEHIND,
    ));
    out.push(with_geometry(
        GpuSurfaceParams {
            metallic: 1.0,
            perceptual_roughness: 0.2,
            ..Default::default()
        },
        BEHIND,
    ));

    // The GGX-width filter from `normal_variance`.
    for normal_variance in [1.0e-4_f32, 1.0e-2, 0.1] {
        out.push(with_geometry(
            GpuSurfaceParams {
                metallic: 1.0,
                perceptual_roughness: 0.08,
                normal_variance,
                ..Default::default()
            },
            HEAD_ON,
        ));
    }

    // Anisotropy, at zero rotation so the libm/GPU `cos`/`sin` do not enter.
    for anisotropy in [0.3_f32, 0.8] {
        for geometry in [HEAD_ON, FORTY_FIVE, GRAZING] {
            out.push(with_geometry(
                GpuSurfaceParams {
                    anisotropy,
                    perceptual_roughness: 0.4,
                    ..Default::default()
                },
                geometry,
            ));
        }
    }

    // The dielectric specular group: IOR x weight.
    for specular_ior in [1.2_f32, 2.0, 3.0] {
        for specular_weight in [0.0_f32, 0.5] {
            out.push(with_geometry(
                GpuSurfaceParams {
                    specular_ior,
                    specular_weight,
                    ..Default::default()
                },
                GRAZING,
            ));
        }
    }

    // `specular_color` tint, dielectric and metal.
    for specular_color in [[1.0_f32, 0.4, 0.2], [0.2, 0.6, 1.0]] {
        out.push(with_geometry(
            GpuSurfaceParams {
                specular_color,
                ..Default::default()
            },
            FORTY_FIVE,
        ));
        out.push(with_geometry(
            GpuSurfaceParams {
                specular_color,
                metallic: 1.0,
                ..Default::default()
            },
            GRAZING,
        ));
    }

    // Oren-Nayar vs Lambert across the roughness range.
    for (diffuse_roughness, geometry) in [
        (0.25_f32, FORTY_FIVE),
        (0.5, GRAZING),
        (1.0, GRAZING),
        (1.0, NARROW),
    ] {
        out.push(with_geometry(
            GpuSurfaceParams {
                diffuse_roughness,
                ..Default::default()
            },
            geometry,
        ));
    }

    // `base_weight` scaling with an emissive term present.
    for base_weight in [0.0_f32, 0.5, 1.0] {
        out.push(with_geometry(
            GpuSurfaceParams {
                base_weight,
                emissive: [0.01, 0.02, 0.03],
                ..Default::default()
            },
            FORTY_FIVE,
        ));
    }

    // The multiple-scattering compensation in isolation: a black base with the
    // specular layer switched off makes the diffuse and the single-scatter lobe
    // exactly zero, so the whole result is the Kulla-Conty/Turquin term and the
    // `env_brdf_approx` fit it drives. This is the case that catches a
    // coefficient slip in that fit.
    for perceptual_roughness in [0.045_f32, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0] {
        out.push(GpuSurfaceParams {
            base_color: [0.0; 3],
            specular_weight: 0.0,
            perceptual_roughness,
            ..Default::default()
        });
    }

    // Shadow visibility.
    for visibility in [0.0_f32, 0.5, 1.0] {
        out.push(with_geometry(
            GpuSurfaceParams {
                visibility,
                ..Default::default()
            },
            NARROW,
        ));
    }

    // A tinted mixed dielectric/metal on the tilted frame.
    out.push(with_geometry(
        GpuSurfaceParams {
            base_color: [0.2, 0.6, 0.9],
            metallic: 0.5,
            perceptual_roughness: 0.35,
            ..Default::default()
        },
        TILTED,
    ));

    assert!(
        out.len() <= MAX_BATCH as usize,
        "the case grid ({}) must fit one uniform batch ({MAX_BATCH})",
        out.len(),
    );
    out
}

/// Projects the GPU struct back into the CPU golden's input types.
///
/// Every mirrored field is copied verbatim; the CPU fields the mirror does not
/// cover stay at their identity defaults (coat, transmission, iridescence = 0),
/// which is exactly the scope the shader documents.
fn cpu_surface(params: &GpuSurfaceParams) -> SurfaceSample {
    SurfaceSample {
        base_color: params.base_color,
        base_weight: params.base_weight,
        diffuse_roughness: params.diffuse_roughness,
        metallic: params.metallic,
        perceptual_roughness: params.perceptual_roughness,
        specular_weight: params.specular_weight,
        specular_color: params.specular_color,
        specular_ior: params.specular_ior,
        normal_variance: params.normal_variance,
        anisotropy: params.anisotropy,
        anisotropy_rotation: params.anisotropy_rotation,
        emissive: params.emissive,
        ..Default::default()
    }
}

/// The CPU golden for one configuration.
fn cpu_reference(params: &GpuSurfaceParams) -> [f32; 3] {
    evaluate_principled_direct(
        cpu_surface(params),
        ShadingFrame {
            normal: params.normal,
            view: params.view,
            tangent: params.tangent,
            bitangent: params.bitangent,
        },
        DirectLightSample {
            direction: params.direction,
            illuminance: params.illuminance,
            visibility: params.visibility,
        },
    )
}

/// Decodes the read-back buffer into one `[f32; 3]` per evaluated surface.
fn decode_results(bytes: &[u8], count: usize) -> Vec<[f32; 3]> {
    assert_eq!(bytes.len() as u64, RESULT_STRIDE * count as u64);
    bytes
        .chunks_exact(RESULT_STRIDE as usize)
        .map(|record| {
            let channel = |index: usize| {
                let start = index * 4;
                f32::from_le_bytes([
                    record[start],
                    record[start + 1],
                    record[start + 2],
                    record[start + 3],
                ])
            };
            [channel(0), channel(1), channel(2)]
        })
        .collect()
}

/// The end-to-end parity check: shade the grid on the GPU, read it back, and
/// diff every channel against the CPU golden within the documented budget.
#[test]
fn gpu_base_lobe_matches_the_cpu_golden() {
    let Some((device, queue)) = device_or_skip() else {
        assert!(
            !require_gpu(),
            "PRISM_REQUIRE_GPU is set but no adapter was found",
        );
        return;
    };

    let grid = cases();
    let pass = SurfacePass::new(&device);
    let evaluation = pass.evaluate(&device, &queue, &grid);

    let mut encoder = CommandEncoder::labeled("prism_render_shading_gpu parity");
    encoder.push_compute_pass(Some("evaluate_surfaces".into()), pass.record(evaluation));
    queue.submit(&[encoder.finish()]);

    let bytes = queue.read_buffer(evaluation.results, evaluation.result_bytes());
    let gpu = decode_results(&bytes, grid.len());
    assert_eq!(gpu.len(), grid.len());

    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    for (index, (params, gpu_value)) in grid.iter().zip(gpu.iter()).enumerate() {
        let cpu_value = cpu_reference(params);
        for channel in 0..3 {
            let expected = cpu_value[channel];
            let actual = gpu_value[channel];
            assert!(
                actual.is_finite(),
                "case {index} channel {channel} is not finite: {actual}",
            );
            assert!(
                actual >= 0.0,
                "case {index} channel {channel} went negative: {actual}",
            );
            let abs_error = (actual - expected).abs();
            let relative = abs_error / expected.abs().max(actual.abs()).max(TOLERANCE_ABS);
            max_abs = max_abs.max(abs_error);
            max_rel = max_rel.max(relative);
            let budget = TOLERANCE_ABS + TOLERANCE_REL * expected.abs();
            assert!(
                abs_error <= budget,
                "case {index} channel {channel}: cpu={expected} gpu={actual} \
                 abs_error={abs_error} exceeds budget={budget} \
                 (REL={TOLERANCE_REL}, ABS={TOLERANCE_ABS})",
            );
        }
        // The alpha slot is a fixed 1.0 sentinel, not a shaded value.
        let alpha_start = index * RESULT_STRIDE as usize + 12;
        let alpha = f32::from_le_bytes([
            bytes[alpha_start],
            bytes[alpha_start + 1],
            bytes[alpha_start + 2],
            bytes[alpha_start + 3],
        ]);
        assert_eq!(alpha, 1.0, "case {index} alpha sentinel");
    }

    #[expect(
        clippy::print_stderr,
        reason = "test diagnostic: report the observed parity budget on hardware"
    )]
    {
        eprintln!(
            "surface parity over {} cases: max_abs={max_abs:.3e} max_rel={max_rel:.3e} \
             (budget REL={TOLERANCE_REL:.1e} ABS={TOLERANCE_ABS:.1e})",
            grid.len(),
        );
    }

    pass.destroy(&device);
}
