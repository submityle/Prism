//! Real-device parity for the carve-brush density-delta twin:
//! [`GpuDensityDelta`] must reproduce the `CPU` golden
//! [`density_delta`](prism_render_architecture::volumetric::coupling::density_delta)
//! across a spread of positions and brush parameters.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The kernel mirrors the reference's `Vec3::distance` (`sqrt` of the squared
//! component sum) and its hand-rolled Hermite `smoothstep` (the collapsed-edge
//! `EPS` guard and the `t*t*(3-2t)` polynomial), and it contains no
//! transcendental call, so `CPU` and `GPU` evaluate the same closed-form
//! algebra. The only slack is a legal multiply-add contraction of a few `ULP`,
//! so values are asserted to within `abs_diff < 1e-6` or `rel_diff < 1e-5` —
//! tight enough to fail a wrong port (a dropped floor, a swapped sign, a missing
//! saturate). The scenes also assert the documented `-1..=0` range, the deepest
//! carve at the centre, the vanishing delta beyond the radius, and bounded
//! degenerate brushes, so a degenerate kernel could not pass.
//!
//! Provenance: standard spherical-falloff density carving; no Unreal Engine
//! source or derived code.

use prism_render_architecture::volumetric::coupling::{density_delta, CarveBrush as CpuBrush};
use prism_render_architecture::volumetric::Vec3;
use prism_volumetric_gpu::{CarveBrush, DensityDeltaQuery, GpuContext, GpuDensityDelta};

/// Maps a GPU brush onto the `CPU` golden brush so both evaluate identically.
fn cpu_brush(b: CarveBrush) -> CpuBrush {
    CpuBrush {
        center: b.center,
        radius: b.radius,
        strength: b.strength,
    }
}

/// Asserts every `gpu` value matches the `CPU` golden to within the documented
/// tolerance and stays inside the `-1..=0` range.
fn assert_parity(queries: &[DensityDeltaQuery], gpu: &[f32]) {
    assert_eq!(gpu.len(), queries.len(), "one delta value per query");
    for (i, q) in queries.iter().enumerate() {
        let exp = density_delta(q.pos, cpu_brush(q.brush));
        let got = gpu[i];
        let abs_diff = (got - exp).abs();
        let rel_diff = abs_diff / exp.abs().max(1e-6);
        assert!(
            abs_diff < 1e-6 || rel_diff < 1e-5,
            "density delta mismatch for query {i} ({q:?}): gpu {got}, cpu {exp} \
             (abs {abs_diff}, rel {rel_diff})"
        );
        assert!(
            (-1.0..=0.0).contains(&got),
            "gpu density delta must stay in -1..=0: {got}"
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_density_delta_matches_cpu_golden_across_queries() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping density delta parity: no wgpu adapter on this host");
        return;
    };
    let gpu_delta = GpuDensityDelta::new(&ctx);

    let brush = CarveBrush {
        center: Vec3::new(1.0, 2.0, 3.0),
        radius: 4.0,
        strength: 0.75,
    };
    // A deterministic spread: brush centre (deepest carve), inside the radius,
    // at the radius, well outside (zero delta), plus a degenerate brush.
    let degenerate = CarveBrush {
        center: Vec3::ZERO,
        radius: 0.0,
        strength: 5.0,
    };
    let mut queries: Vec<DensityDeltaQuery> = vec![
        DensityDeltaQuery {
            pos: brush.center,
            brush,
        },
        DensityDeltaQuery {
            pos: Vec3::new(3.0, 2.0, 3.0),
            brush,
        },
        DensityDeltaQuery {
            pos: Vec3::new(5.0, 2.0, 3.0),
            brush,
        },
        DensityDeltaQuery {
            pos: Vec3::new(100.0, 2.0, 3.0),
            brush,
        },
        DensityDeltaQuery {
            pos: Vec3::ZERO,
            brush: degenerate,
        },
        DensityDeltaQuery {
            pos: Vec3::new(1000.0, 0.0, 0.0),
            brush: degenerate,
        },
    ];
    // A deterministic radial ramp out through and past the brush radius.
    for k in 0..96 {
        let t = (k as f32) * 0.1;
        queries.push(DensityDeltaQuery {
            pos: Vec3::new(1.0 + t, 2.0, 3.0),
            brush,
        });
    }

    let gpu = gpu_delta.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);

    // Deepest carve at the centre; vanishing beyond the radius.
    assert!(
        (gpu[0] + 0.75).abs() < 1e-6,
        "centre carves the full strength: {}",
        gpu[0]
    );
    assert!(
        gpu[3].abs() < 1e-6,
        "far outside the radius carves nothing: {}",
        gpu[3]
    );
    assert!(
        gpu[5].abs() < 1e-6,
        "the degenerate brush does not reach far points: {}",
        gpu[5]
    );
    assert!(
        gpu[0] <= gpu[1] + 1e-6 && gpu[1] <= gpu[2] + 1e-6,
        "the carve must ease from the centre toward the radius"
    );
}

#[test]
fn gpu_density_delta_is_monotone_non_decreasing_with_distance() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_delta = GpuDensityDelta::new(&ctx);

    // At fixed brush the (negative) delta rises monotonically toward zero as the
    // sample moves away from the centre.
    let brush = CarveBrush {
        center: Vec3::new(-5.0, 10.0, 2.0),
        radius: 6.0,
        strength: 0.9,
    };
    let queries: Vec<DensityDeltaQuery> = (0..=120)
        .map(|k| DensityDeltaQuery {
            pos: Vec3::new(-5.0 + (k as f32) * 0.1, 10.0, 2.0),
            brush,
        })
        .collect();

    let gpu = gpu_delta.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);

    for w in gpu.windows(2) {
        assert!(
            w[1] >= w[0] - 1e-6,
            "density delta must be monotone non-decreasing with distance: {} then {}",
            w[0],
            w[1]
        );
    }
    assert!(
        gpu[0] < gpu[gpu.len() - 1] - 1e-3,
        "the centre must carve strictly deeper than the far sample"
    );
}

#[test]
fn empty_queries_yields_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_delta = GpuDensityDelta::new(&ctx);
    let out = gpu_delta.eval(&ctx, &[]);
    assert!(out.is_empty(), "an empty query slice yields no values");
}
