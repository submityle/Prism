//! Real-device parity for the aerial-perspective fade twin: [`GpuAerialPerspective`]
//! must reproduce the `CPU` golden
//! [`aerial_perspective_weight`](prism_render_architecture::volumetric::atmosphere::aerial_perspective_weight)
//! across a distance sweep, degenerate far distances and the far-plane edge.
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
//! fail a wrong port (a dropped `saturate`, a mis-scaled extinction, a swapped
//! numerator/denominator, a missing degenerate branch). The scenes also assert
//! the documented monotonicity (farther fragments fade more), the `[0, 1]`
//! range, `distance = 0 -> 0`, `distance >= max_distance -> 1` and the
//! non-positive `max_distance` degeneracy, so a degenerate kernel could not
//! pass.
//!
//! Provenance: standard normalised exponential aerial-perspective fade; no
//! Unreal Engine source or derived code.

use prism_render_architecture::volumetric::atmosphere::aerial_perspective_weight;
use prism_volumetric_gpu::{AerialQuery, GpuAerialPerspective, GpuContext};

/// Asserts every `gpu` value matches the `CPU` golden to within the documented
/// tolerance and stays inside the unit range.
fn assert_parity(queries: &[AerialQuery], gpu: &[f32]) {
    assert_eq!(gpu.len(), queries.len(), "one aerial value per query");
    for (i, q) in queries.iter().enumerate() {
        let exp = aerial_perspective_weight(q.distance, q.max_distance);
        let got = gpu[i];
        let abs_diff = (got - exp).abs();
        let rel_diff = abs_diff / exp.abs().max(1e-6);
        assert!(
            abs_diff < 1e-6 || rel_diff < 1e-5,
            "aerial mismatch for query {i} ({q:?}): gpu {got}, cpu {exp} \
             (abs {abs_diff}, rel {rel_diff})"
        );
        assert!(
            (0.0..=1.0).contains(&got),
            "gpu aerial weight must stay in the unit range: {got}"
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_aerial_matches_cpu_golden_across_queries() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping aerial parity: no wgpu adapter on this host");
        return;
    };
    let gpu_aerial = GpuAerialPerspective::new(&ctx);

    // A deterministic spread: the camera itself, a near fragment, the mid
    // range, the far plane exactly, a fragment past the far plane (must clamp
    // to one), and the two degenerate far-distance branches.
    let mut queries: Vec<AerialQuery> = vec![
        AerialQuery {
            distance: 0.0,
            max_distance: 1000.0,
        },
        AerialQuery {
            distance: 100.0,
            max_distance: 1000.0,
        },
        AerialQuery {
            distance: 500.0,
            max_distance: 1000.0,
        },
        AerialQuery {
            distance: 1000.0,
            max_distance: 1000.0,
        },
        AerialQuery {
            distance: 5000.0,
            max_distance: 1000.0,
        },
        // Degenerate far distance: any positive distance maps to one.
        AerialQuery {
            distance: 5.0,
            max_distance: 0.0,
        },
        // Degenerate far distance with a non-positive distance maps to zero.
        AerialQuery {
            distance: -5.0,
            max_distance: -3.0,
        },
    ];
    // A deterministic distance ramp at a fixed far plane.
    for k in 0..96 {
        queries.push(AerialQuery {
            distance: (k as f32) * 20.0,
            max_distance: 1200.0,
        });
    }

    let gpu = gpu_aerial.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);

    // distance = 0 -> 0; the far plane exactly -> 1; past the far plane -> 1;
    // the degenerate positive-distance branch -> 1, the non-positive one -> 0.
    assert!(gpu[0].abs() < 1e-6, "zero distance yields no fade");
    assert!(
        (gpu[3] - 1.0).abs() < 1e-6,
        "the far plane must reach full opacity: {}",
        gpu[3]
    );
    assert!(
        (gpu[4] - 1.0).abs() < 1e-6,
        "past the far plane must clamp to full opacity: {}",
        gpu[4]
    );
    assert!(
        (gpu[5] - 1.0).abs() < 1e-6,
        "a positive distance with a degenerate far plane maps to one: {}",
        gpu[5]
    );
    assert!(
        gpu[6].abs() < 1e-6,
        "a non-positive distance with a degenerate far plane maps to zero: {}",
        gpu[6]
    );
}

#[test]
fn gpu_aerial_is_monotone_non_decreasing_in_distance() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_aerial = GpuAerialPerspective::new(&ctx);

    // At a fixed far plane the fade rises monotonically from zero at the camera
    // toward one at the far plane as the fragment recedes.
    let max_distance = 2000.0f32;
    let queries: Vec<AerialQuery> = (0..=100)
        .map(|k| AerialQuery {
            distance: (k as f32) * 20.0,
            max_distance,
        })
        .collect();

    let gpu = gpu_aerial.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);

    for w in gpu.windows(2) {
        assert!(
            w[1] >= w[0] - 1e-6,
            "aerial weight must be monotone non-decreasing in distance: {} then {}",
            w[0],
            w[1]
        );
    }
    assert!(
        gpu[gpu.len() - 1] > gpu[0] + 1e-3,
        "the farthest query must fade strictly more than the camera one"
    );
}

#[test]
fn empty_queries_yields_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_aerial = GpuAerialPerspective::new(&ctx);
    let out = gpu_aerial.eval(&ctx, &[]);
    assert!(out.is_empty(), "an empty query slice yields no values");
}
