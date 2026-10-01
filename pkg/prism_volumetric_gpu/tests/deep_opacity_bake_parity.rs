//! Real-device parity for the `deep opacity map` baker twin:
//! [`GpuDeepOpacityBake`] must reproduce the `CPU` golden
//! [`DeepOpacityRecorder::bake_depth_profile`](prism_render_architecture::particle::deep_opacity_bake::DeepOpacityRecorder::bake_depth_profile)
//! for every layer across several extinction profiles and recorder
//! configurations.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable
//! core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Both sides run the identical algebraic step-opacity recurrence
//! `transmittance *= 1 - clamp(sigma * sub_length, 0, 1)` over the identical
//! `sigma` samples — the `GPU` reads exactly the values the reference samples
//! at the [`march_centers`] depths — with no reorderable summation. Each layer
//! is asserted to within `abs_diff < 1e-6` or `rel_diff < 1e-5`, far tighter
//! than any physically meaningful transmittance difference and enough to fail a
//! wrong port (a dropped clamp, a swapped span, a misaligned sample window).
//! The scenes also assert the physical shape — an empty medium stays fully lit,
//! a constant absorber decreases monotonically, and layer depths ascend — so a
//! degenerate constant kernel could not pass.
//!
//! Provenance: standard Lokovic-Veach / Yuksel-Keyser deep-opacity recording;
//! no Unreal Engine source or derived code.

use prism_render_architecture::particle::deep_opacity_bake::DeepOpacityRecorder;
use prism_render_architecture::particle::volumetrics::DeepOpacityLayer;
use prism_volumetric_gpu::deep_opacity_bake::{march_centers, GpuDeepOpacityBake};
use prism_volumetric_gpu::GpuContext;

/// Shared absolute tolerance floor for the relative-difference guard.
const REL_FLOOR: f32 = 1e-6;

/// Samples an extinction function at the recorder's march sub-step centers,
/// producing the per-profile sample slice the baker consumes. Feeding these
/// exact samples makes the `GPU` read the same `sigma` the `CPU` golden samples
/// internally, so the two run the identical recurrence.
fn samples_at<F>(recorder: &DeepOpacityRecorder, mut extinction_at: F) -> Vec<f32>
where
    F: FnMut(f32) -> f32,
{
    march_centers(recorder)
        .into_iter()
        .map(&mut extinction_at)
        .collect()
}

/// Asserts the `gpu` curve matches the `cpu` golden curve layer-for-layer to
/// within the documented tolerance.
fn assert_curve_parity(cpu: &[DeepOpacityLayer], gpu: &[DeepOpacityLayer]) {
    assert_eq!(gpu.len(), cpu.len(), "one layer per recorded depth");
    for (i, (got, exp)) in gpu.iter().zip(cpu.iter()).enumerate() {
        for (got_v, exp_v, name) in [
            (got.depth, exp.depth, "depth"),
            (got.transmittance, exp.transmittance, "transmittance"),
        ] {
            let abs_diff = (got_v - exp_v).abs();
            let rel_diff = abs_diff / exp_v.abs().max(REL_FLOOR);
            assert!(
                abs_diff < 1e-6 || rel_diff < 1e-5,
                "{name} mismatch at layer {i}: gpu {got_v}, cpu {exp_v} (abs {abs_diff}, rel {rel_diff})"
            );
        }
    }
}

/// The canonical recorder shared by several scenes: `5` layers across depth
/// `0..=4` with a `0.25` sub-step, matching the `CPU` golden's own fixture.
fn recorder() -> DeepOpacityRecorder {
    DeepOpacityRecorder::new(5, 0.0, 4.0, 0.25).expect("valid configuration")
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_bake_matches_cpu_golden_across_densities() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping deep-opacity-bake parity: no wgpu adapter on this host");
        return;
    };
    let baker = GpuDeepOpacityBake::new(&ctx);
    let rec = recorder();

    // A battery of extinction-versus-depth profiles exercised in one batched
    // dispatch (one thread per profile): empty, constant, depth-ramped, a
    // clamped-negative sample and a slab that only absorbs mid-ray.
    let closures: [fn(f32) -> f32; 5] = [
        |_depth| 0.0,
        |_depth| 0.5,
        |depth| 0.2 + 0.1 * depth,
        |depth| depth - 2.0, // negative near the start; the baker floors it to 0.
        |depth| {
            if depth >= 1.0 && depth <= 3.0 {
                0.8
            } else {
                0.0
            }
        },
    ];

    let profiles: Vec<Vec<f32>> = closures.iter().map(|f| samples_at(&rec, f)).collect();
    let gpu = baker.bake(&ctx, &rec, &profiles);
    assert_eq!(gpu.len(), closures.len(), "one curve per profile");

    for (f, gpu_curve) in closures.iter().zip(gpu.iter()) {
        let cpu_curve = rec.bake_depth_profile(f);
        assert_curve_parity(&cpu_curve, gpu_curve);
    }
}

#[test]
fn gpu_bake_matches_cpu_golden_across_recorders() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let baker = GpuDeepOpacityBake::new(&ctx);

    // Several distinct configurations: a single sub-step per layer, a long span
    // with many sub-steps, and an offset near plane.
    let recorders = [
        DeepOpacityRecorder::new(5, 0.0, 4.0, 4.0).expect("valid configuration"),
        DeepOpacityRecorder::new(8, 0.0, 10.0, 0.1).expect("valid configuration"),
        DeepOpacityRecorder::new(3, 2.0, 7.0, 0.3).expect("valid configuration"),
    ];

    for rec in &recorders {
        let samples = samples_at(rec, |depth| 0.15 + 0.05 * depth);
        let gpu = baker.bake(&ctx, rec, std::slice::from_ref(&samples));
        let cpu = rec.bake_depth_profile(|depth| 0.15 + 0.05 * depth);
        assert_curve_parity(&cpu, &gpu[0]);
    }
}

#[test]
fn empty_medium_is_fully_lit() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let baker = GpuDeepOpacityBake::new(&ctx);
    let rec = recorder();

    let samples = samples_at(&rec, |_depth| 0.0);
    let gpu = baker.bake(&ctx, &rec, std::slice::from_ref(&samples));
    let curve = &gpu[0];
    assert_eq!(curve.len(), 5, "the recorder bakes five layers");
    for layer in curve {
        // Zero extinction never attenuates: transmittance stays at unity.
        let diff = (layer.transmittance - 1.0).abs();
        assert!(
            diff < 1e-6,
            "empty medium must stay fully lit, got {}",
            layer.transmittance
        );
    }
}

#[test]
fn constant_density_decreases_monotonically() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let baker = GpuDeepOpacityBake::new(&ctx);
    let rec = recorder();

    let samples = samples_at(&rec, |_depth| 0.5);
    let gpu = baker.bake(&ctx, &rec, std::slice::from_ref(&samples));
    let curve = &gpu[0];

    // The near layer is still the fully-lit side.
    let near_diff = (curve[0].transmittance - 1.0).abs();
    assert!(near_diff < 1e-6, "layer 0 is the fully-lit near side");
    // A uniform absorber: each deeper layer survives strictly less than the
    // last, and every layer stays physically bounded in `0..=1`.
    for pair in curve.windows(2) {
        assert!(
            pair[1].transmittance < pair[0].transmittance,
            "a uniform absorber must strictly attenuate deeper layers"
        );
    }
    for layer in curve {
        assert!(
            layer.transmittance >= 0.0 && layer.transmittance <= 1.0,
            "transmittance must stay in 0..=1, got {}",
            layer.transmittance
        );
    }
}

#[test]
fn layers_are_depth_ordered() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let baker = GpuDeepOpacityBake::new(&ctx);
    let rec = recorder();

    let samples = samples_at(&rec, |_depth| 0.3);
    let gpu = baker.bake(&ctx, &rec, std::slice::from_ref(&samples));
    let curve = &gpu[0];

    for pair in curve.windows(2) {
        assert!(
            pair[1].depth > pair[0].depth,
            "recorded depths must ascend: {} then {}",
            pair[0].depth,
            pair[1].depth
        );
    }
    // The first and last depths match the configured planes.
    assert!(
        (curve[0].depth - 0.0).abs() < 1e-6,
        "layer 0 sits at the near plane"
    );
    assert!(
        (curve[curve.len() - 1].depth - 4.0).abs() < 1e-6,
        "the last layer sits at the far plane"
    );
}

#[test]
fn empty_profiles_yields_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let baker = GpuDeepOpacityBake::new(&ctx);
    let rec = recorder();
    let out = baker.bake(&ctx, &rec, &[]);
    assert!(out.is_empty(), "an empty profile slice yields no curves");
}
