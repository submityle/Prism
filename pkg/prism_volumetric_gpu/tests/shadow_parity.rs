//! Real-device parity for the cloud self-shadow twin: [`GpuShadow`] must
//! reproduce the `CPU` golden
//! [`accumulate_shadow`](prism_render_architecture::volumetric::shadow::accumulate_shadow)
//! across a spread of density profiles and march steps.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The kernel mirrors the reference's hand-rolled `exp_approx` (base-two range
//! reduction, a fractional polynomial times an integer power assembled from the
//! float exponent field) rather than the device-native `exp`, so `CPU` and
//! `GPU` evaluate the same closed-form algebra. The only slack is a legal
//! multiply-add contraction of a few `ULP` in the optical-depth sum and the
//! polynomial, so values are asserted to within `abs_diff < 1e-6` or
//! `rel_diff < 1e-5` — tight enough to fail a wrong port (a dropped `max(0)`
//! clamp, a mis-scaled step, a swapped polynomial coefficient). The scenes also
//! assert the documented monotonicity (denser or longer rays never raise the
//! transmittance) and the `[0, 1]` range, so a degenerate kernel could not
//! pass.
//!
//! Provenance: standard Beer-Lambert volumetric shadowing; no Unreal Engine
//! source or derived code.

use prism_render_architecture::volumetric::shadow::accumulate_shadow;
use prism_volumetric_gpu::{GpuContext, GpuShadow, ShadowRay};

/// Asserts every `gpu` value matches the `CPU` golden to within the documented
/// tolerance and stays inside the unit range.
fn assert_parity(rays: &[ShadowRay], gpu: &[f32]) {
    assert_eq!(gpu.len(), rays.len(), "one transmittance per ray");
    for (i, r) in rays.iter().enumerate() {
        let exp = accumulate_shadow(&r.densities, r.step);
        let got = gpu[i];
        let abs_diff = (got - exp).abs();
        let rel_diff = abs_diff / exp.abs().max(1e-6);
        assert!(
            abs_diff < 1e-6 || rel_diff < 1e-5,
            "shadow mismatch for ray {i} ({r:?}): gpu {got}, cpu {exp} \
             (abs {abs_diff}, rel {rel_diff})"
        );
        assert!(
            (0.0..=1.0).contains(&got),
            "gpu transmittance must stay in the unit range: {got}"
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_shadow_matches_cpu_golden_across_profiles() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping shadow parity: no wgpu adapter on this host");
        return;
    };
    let gpu_shadow = GpuShadow::new(&ctx);

    // A deterministic spread of density profiles: clear air, a thin haze, a
    // dense core, a ramp, negatives that must clamp to zero, and a long march
    // that accumulates a large optical depth (transmittance near zero).
    let mut rays: Vec<ShadowRay> = vec![
        ShadowRay {
            densities: vec![],
            step: 10.0,
        },
        ShadowRay {
            densities: vec![0.02, 0.03, 0.05, 0.04],
            step: 5.0,
        },
        ShadowRay {
            densities: vec![0.9, 1.2, 0.7],
            step: 8.0,
        },
        ShadowRay {
            densities: vec![-0.5, 0.1, -0.2, 0.3],
            step: 4.0,
        },
        ShadowRay {
            densities: vec![0.0; 64],
            step: 12.0,
        },
    ];
    // A deterministic long ramp with a fine step.
    let ramp: Vec<f32> = (0..128).map(|k| 0.01 + (k as f32) * 0.002).collect();
    rays.push(ShadowRay {
        densities: ramp,
        step: 2.5,
    });
    // A zero-step ray: optical depth stays zero, transmittance is one.
    rays.push(ShadowRay {
        densities: vec![1.0, 2.0, 3.0],
        step: 0.0,
    });

    let gpu = gpu_shadow.eval(&ctx, &rays);
    assert_parity(&rays, &gpu);

    // Empty ray and zero-step ray must fully transmit; the dense core must
    // attenuate well below one, proving the extinction actually ran.
    assert!((gpu[0] - 1.0).abs() < 1e-6, "clear air must fully transmit");
    assert!(
        gpu[2] < 0.5,
        "a dense core must attenuate strongly: {}",
        gpu[2]
    );
}

#[test]
fn gpu_shadow_is_monotone_non_increasing() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_shadow = GpuShadow::new(&ctx);

    // Growing the optical depth — by lengthening the ray or raising a sample —
    // may only lower the surviving transmittance. Build a family where each ray
    // strictly dominates the previous one's density.
    let step = 6.0f32;
    let mut rays: Vec<ShadowRay> = Vec::new();
    for n in 0..=32 {
        let densities: Vec<f32> = (0..n).map(|_| 0.05).collect();
        rays.push(ShadowRay { densities, step });
    }

    let gpu = gpu_shadow.eval(&ctx, &rays);
    assert_parity(&rays, &gpu);

    for w in gpu.windows(2) {
        assert!(
            w[1] <= w[0] + 1e-6,
            "transmittance must be monotone non-increasing in optical depth: \
             {} then {}",
            w[0],
            w[1]
        );
    }
    assert!(
        gpu[0] > gpu[gpu.len() - 1] + 1e-3,
        "the longest ray must transmit strictly less than the empty ray"
    );
}

#[test]
fn empty_rays_yields_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_shadow = GpuShadow::new(&ctx);
    let out = gpu_shadow.eval(&ctx, &[]);
    assert!(out.is_empty(), "an empty ray slice yields no values");
}
