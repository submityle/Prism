//! Real-device parity for the Nubis powder dark-edge twin: [`GpuPowder`] must
//! reproduce the `CPU` golden
//! [`powder`](prism_render_architecture::volumetric::scatter::powder) across a
//! spread of optical depths and strengths.
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
//! multiply-add contraction of a few `ULP` in the polynomial, so values are
//! asserted to within `abs_diff < 1e-6` or `rel_diff < 1e-5` — tight enough to
//! fail a wrong port (a dropped `max(0)` clamp, a mis-scaled `-2` factor, a
//! swapped polynomial coefficient). The scenes also assert the documented
//! monotonicity (denser rays raise the powder term toward one), the `[0, 1]`
//! range and `density = 0 -> 0`, so a degenerate kernel could not pass.
//!
//! Provenance: standard Nubis-style powder approximation; no Unreal Engine
//! source or derived code.

use prism_render_architecture::volumetric::scatter::powder;
use prism_volumetric_gpu::{GpuContext, GpuPowder, PowderQuery};

/// Asserts every `gpu` value matches the `CPU` golden to within the documented
/// tolerance and stays inside the unit range.
fn assert_parity(queries: &[PowderQuery], gpu: &[f32]) {
    assert_eq!(gpu.len(), queries.len(), "one powder value per query");
    for (i, q) in queries.iter().enumerate() {
        let exp = powder(q.density, q.strength);
        let got = gpu[i];
        let abs_diff = (got - exp).abs();
        let rel_diff = abs_diff / exp.abs().max(1e-6);
        assert!(
            abs_diff < 1e-6 || rel_diff < 1e-5,
            "powder mismatch for query {i} ({q:?}): gpu {got}, cpu {exp} \
             (abs {abs_diff}, rel {rel_diff})"
        );
        assert!(
            (0.0..=1.0).contains(&got),
            "gpu powder must stay in the unit range: {got}"
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_powder_matches_cpu_golden_across_queries() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping powder parity: no wgpu adapter on this host");
        return;
    };
    let gpu_powder = GpuPowder::new(&ctx);

    // A deterministic spread: clear air, a faint edge, a moderate edge, a dense
    // core near saturation, negative inputs that must clamp to zero, and a
    // strength sweep that should track the CPU curve exactly.
    let mut queries: Vec<PowderQuery> = vec![
        PowderQuery {
            density: 0.0,
            strength: 1.0,
        },
        PowderQuery {
            density: 0.1,
            strength: 0.5,
        },
        PowderQuery {
            density: 0.75,
            strength: 1.0,
        },
        PowderQuery {
            density: 4.0,
            strength: 2.0,
        },
        PowderQuery {
            density: -1.0,
            strength: 1.0,
        },
        PowderQuery {
            density: 0.5,
            strength: -3.0,
        },
        PowderQuery {
            density: 0.5,
            strength: 0.0,
        },
    ];
    // A deterministic density ramp at fixed strength.
    for k in 0..96 {
        queries.push(PowderQuery {
            density: (k as f32) * 0.05,
            strength: 0.8,
        });
    }

    let gpu = gpu_powder.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);

    // density = 0 -> 0; a dense core must approach one, proving the extinction
    // curve actually ran; the negative inputs must clamp to zero powder.
    assert!(gpu[0].abs() < 1e-6, "zero density yields no powder");
    assert!(
        gpu[3] > 0.9,
        "a dense edge must approach full powder: {}",
        gpu[3]
    );
    assert!(
        gpu[4].abs() < 1e-6,
        "negative density clamps to zero powder"
    );
    assert!(gpu[6].abs() < 1e-6, "zero strength yields no powder");
}

#[test]
fn gpu_powder_is_monotone_non_decreasing_in_density() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_powder = GpuPowder::new(&ctx);

    // At fixed strength the powder term rises monotonically from zero toward
    // one as the optical depth grows.
    let strength = 1.25f32;
    let queries: Vec<PowderQuery> = (0..=64)
        .map(|k| PowderQuery {
            density: (k as f32) * 0.06,
            strength,
        })
        .collect();

    let gpu = gpu_powder.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);

    for w in gpu.windows(2) {
        assert!(
            w[1] >= w[0] - 1e-6,
            "powder must be monotone non-decreasing in density: {} then {}",
            w[0],
            w[1]
        );
    }
    assert!(
        gpu[gpu.len() - 1] > gpu[0] + 1e-3,
        "the densest query must yield strictly more powder than the clear one"
    );
}

#[test]
fn empty_queries_yields_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_powder = GpuPowder::new(&ctx);
    let out = gpu_powder.eval(&ctx, &[]);
    assert!(out.is_empty(), "an empty query slice yields no values");
}
