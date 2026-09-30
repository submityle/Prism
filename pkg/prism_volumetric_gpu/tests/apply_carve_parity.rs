//! Real-device parity for the carve-application choke point:
//! [`GpuApplyCarve`] must reproduce the `CPU` golden
//! [`apply_carve`](prism_render_architecture::volumetric::coupling::apply_carve)
//! across a spread of base densities and signed carve deltas.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The kernel contains no transcendental call — one add and a saturating clamp
//! — so `CPU` and `GPU` evaluate the same closed-form algebra. The only slack
//! is a legal multiply-add contraction of a few `ULP`, so values are asserted
//! to within `abs_diff < 1e-6` or `rel_diff < 1e-5` — tight enough to fail a
//! wrong port (a dropped saturate, a swapped sign). The scenes also assert the
//! documented `[0, 1]` range, that an over-carve (a large negative delta)
//! clamps to `0` — the key safety invariant that keeps carving from producing
//! a negative density — and that an over-dense base clamps to `1`, so a
//! degenerate kernel could not pass.
//!
//! Provenance: standard saturating density accumulation; no Unreal Engine
//! source or derived code.

use prism_render_architecture::volumetric::coupling::apply_carve;
use prism_volumetric_gpu::{ApplyCarveQuery, GpuApplyCarve, GpuContext};

/// Asserts every `gpu` value matches the `CPU` golden to within the documented
/// tolerance and stays inside the unit range.
fn assert_parity(queries: &[ApplyCarveQuery], gpu: &[f32]) {
    assert_eq!(gpu.len(), queries.len(), "one carved density per query");
    for (i, q) in queries.iter().enumerate() {
        let exp = apply_carve(q.base_density, q.delta);
        let got = gpu[i];
        let abs_diff = (got - exp).abs();
        let rel_diff = abs_diff / exp.abs().max(1e-6);
        assert!(
            abs_diff < 1e-6 || rel_diff < 1e-5,
            "apply carve mismatch for query {i} ({q:?}): gpu {got}, cpu {exp} \
             (abs {abs_diff}, rel {rel_diff})"
        );
        assert!(
            (0.0..=1.0).contains(&got),
            "gpu carved density must stay in the unit range: {got}"
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_apply_carve_matches_cpu_golden_across_queries() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping apply carve parity: no wgpu adapter on this host");
        return;
    };
    let gpu_carve = GpuApplyCarve::new(&ctx);

    // A deterministic spread: a mild carve into a dense cloud, a carve that
    // exactly cancels the base, an over-carve that must clamp to zero, an
    // over-dense base with no carve that must clamp to one, and a positive
    // delta that fills density back in.
    let mut queries: Vec<ApplyCarveQuery> = vec![
        ApplyCarveQuery {
            base_density: 0.8,
            delta: -0.3,
        },
        ApplyCarveQuery {
            base_density: 0.5,
            delta: -0.5,
        },
        ApplyCarveQuery {
            base_density: 0.4,
            delta: -1.0,
        },
        ApplyCarveQuery {
            base_density: 1.5,
            delta: 0.0,
        },
        ApplyCarveQuery {
            base_density: 0.2,
            delta: 0.5,
        },
        ApplyCarveQuery {
            base_density: 0.0,
            delta: -0.9,
        },
        ApplyCarveQuery {
            base_density: 1.0,
            delta: 1.0,
        },
    ];
    // A deterministic delta ramp at fixed base, plus a base ramp at fixed
    // (negative) carve delta.
    for k in 0..48 {
        queries.push(ApplyCarveQuery {
            base_density: 0.6,
            delta: -1.0 + (k as f32) / 24.0,
        });
    }
    for k in 0..48 {
        queries.push(ApplyCarveQuery {
            base_density: (k as f32) / 48.0,
            delta: -0.35,
        });
    }

    let gpu = gpu_carve.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);

    // A mild carve subtracts cleanly; an exact cancel and an over-carve both
    // clamp to zero; an over-dense base clamps to one.
    assert!(
        (gpu[0] - 0.5).abs() < 1e-6,
        "mild carve subtracts cleanly: {}",
        gpu[0]
    );
    assert!(
        gpu[1].abs() < 1e-6,
        "an exact cancel drives density to zero: {}",
        gpu[1]
    );
    assert!(
        gpu[2].abs() < 1e-6,
        "an over-carve clamps to zero (never negative): {}",
        gpu[2]
    );
    assert!(
        (gpu[3] - 1.0).abs() < 1e-6,
        "an over-dense base clamps to one: {}",
        gpu[3]
    );
    assert!(
        (gpu[4] - 0.7).abs() < 1e-6,
        "a positive delta fills density back in: {}",
        gpu[4]
    );
    assert!(
        gpu[5].abs() < 1e-6,
        "carving empty space stays at zero: {}",
        gpu[5]
    );
}

#[test]
fn gpu_apply_carve_is_monotone_in_delta() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_carve = GpuApplyCarve::new(&ctx);

    // At fixed base the carved density rises monotonically from zero toward one
    // as the carve delta grows from a strong negative carve to a strong fill.
    let base = 0.5f32;
    let queries: Vec<ApplyCarveQuery> = (0..=100)
        .map(|k| ApplyCarveQuery {
            base_density: base,
            delta: -1.0 + (k as f32) / 50.0,
        })
        .collect();

    let gpu = gpu_carve.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);

    for w in gpu.windows(2) {
        assert!(
            w[1] >= w[0] - 1e-6,
            "carved density must be monotone non-decreasing in delta: {} then {}",
            w[0],
            w[1]
        );
    }
    assert!(
        gpu[0].abs() < 1e-6,
        "a strong negative carve drives density to zero"
    );
    assert!(
        (gpu[gpu.len() - 1] - 1.0).abs() < 1e-6,
        "a strong fill saturates density to one"
    );
}

#[test]
fn empty_queries_yields_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_carve = GpuApplyCarve::new(&ctx);
    let out = gpu_carve.eval(&ctx, &[]);
    assert!(out.is_empty(), "an empty query slice yields no values");
}
