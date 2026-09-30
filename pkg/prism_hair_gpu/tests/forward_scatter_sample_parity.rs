//! Real-device parity for the forward-scatter decode twin:
//! [`GpuHairForwardScatterSample`] must reproduce the `CPU` golden
//! [`sample_forward_scatter`](prism_render_architecture::hair::dual_scattering::sample_forward_scatter)
//! for a batch of receiver queries against packed
//! [`ForwardScatterLayers`](prism_render_architecture::hair::dual_scattering::ForwardScatterLayers)
//! curves, covering the front-of-curve zero read, the beyond-deepest last-count
//! read, the interior linear interpolation between bracketing boundaries, an
//! exact-boundary probe, the empty (no-occluder) curve, a degenerate zero-span
//! window, a multi-curve batch whose queries fan out to different curves, an
//! out-of-range curve guard, a batch that crosses the `64`-wide workgroup
//! boundary, and a build-then-sample round-trip off the golden
//! [`build_forward_scatter`](prism_render_architecture::hair::dual_scattering::build_forward_scatter).
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The decode is a closed-form bracket-and-interpolate with no transcendental
//! call, so the `CPU` and `GPU` evaluate the same expressions and diverge only
//! through legal fused-multiply-add contraction on the interpolation. Parity is
//! asserted to within `abs_diff < 1e-4` or `rel_diff < 1e-3` — tight enough to
//! fail a genuinely wrong port (a wrong bracket, a missing interpolation, a
//! swapped boundary), loose enough to admit the contraction. Curves are built
//! from explicit depth/opacity literals (never `sin`/`cos`), and the shaping
//! cases assert a non-trivial crossing count so a no-op kernel could not pass.
//!
//! Provenance: standard dual-scattering forward-scatter curve decode plus
//! `wgpu` compute dispatch; no Unreal Engine source or derived code.

use prism_hair_gpu::forward_scatter_sample::{GpuHairForwardScatterSample, ScatterQuery};
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::deep_transmittance::TransmittanceSample;
use prism_render_architecture::hair::dual_scattering::{
    build_forward_scatter, sample_forward_scatter, ForwardScatterLayers,
};

/// Asserts a single value matches within the documented tolerance.
fn assert_close(got: f32, expected: f32, label: &str) {
    let abs_diff = (got - expected).abs();
    let rel_diff = abs_diff / expected.abs().max(1e-6);
    assert!(
        abs_diff < 1e-4 || rel_diff < 1e-3,
        "{label}: gpu {got} vs cpu {expected} (abs {abs_diff}, rel {rel_diff})",
    );
}

/// Builds a curve directly from parallel `(depth, crossing)` pairs.
fn curve_from(pairs: &[(f32, f32)]) -> ForwardScatterLayers {
    ForwardScatterLayers {
        layer_depths: pairs.iter().map(|&(d, _)| d).collect(),
        layer_crossings: pairs.iter().map(|&(_, c)| c).collect(),
    }
}

/// Runs the decode twin and the golden on the same curves/queries and asserts
/// every decoded crossing count matches the golden for its named curve.
fn assert_batch_parity(
    ctx: &GpuContext,
    decoder: &GpuHairForwardScatterSample,
    curves: &[ForwardScatterLayers],
    queries: &[ScatterQuery],
) -> Vec<f32> {
    let gpu = decoder.eval(ctx, curves, queries);
    assert_eq!(gpu.len(), queries.len(), "one output per query");
    for (i, q) in queries.iter().enumerate() {
        let cpu = sample_forward_scatter(&curves[q.curve as usize], q.depth);
        assert_close(
            gpu[i],
            cpu,
            &format!("query {i} (curve {} d {})", q.curve, q.depth),
        );
    }
    gpu
}

/// A receiver in front of the frontmost boundary reads exactly `0` — nothing
/// has been crossed yet.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_front_reads_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping forward scatter sample parity: no wgpu adapter on this host");
        return;
    };
    let decoder = GpuHairForwardScatterSample::new(&ctx);
    let curves = [curve_from(&[(1.0, 0.5), (2.0, 1.0), (3.0, 1.5)])];
    let queries = [
        ScatterQuery {
            curve: 0,
            depth: 0.0,
        },
        ScatterQuery {
            curve: 0,
            depth: 1.0,
        },
    ];
    let gpu = assert_batch_parity(&ctx, &decoder, &curves, &queries);
    assert_eq!(gpu[0], 0.0, "well in front reads zero exactly");
    assert_eq!(gpu[1], 0.0, "at the front boundary reads zero exactly");
}

/// A receiver at or beyond the deepest boundary takes the last (largest)
/// crossing count.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_beyond_reads_last() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping forward scatter sample parity: no wgpu adapter on this host");
        return;
    };
    let decoder = GpuHairForwardScatterSample::new(&ctx);
    let curves = [curve_from(&[(1.0, 0.5), (2.0, 1.0), (3.0, 1.75)])];
    let queries = [
        ScatterQuery {
            curve: 0,
            depth: 3.0,
        },
        ScatterQuery {
            curve: 0,
            depth: 9.0,
        },
    ];
    let gpu = assert_batch_parity(&ctx, &decoder, &curves, &queries);
    assert_close(gpu[0], 1.75, "at deepest boundary reads last count");
    assert_close(
        gpu[1],
        1.75,
        "beyond deepest boundary saturates at last count",
    );
}

/// An interior receiver linearly interpolates between the two bracketing
/// boundaries.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_interior_interpolates() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping forward scatter sample parity: no wgpu adapter on this host");
        return;
    };
    let decoder = GpuHairForwardScatterSample::new(&ctx);
    // Boundaries at 1,2,3 with counts 0.4, 1.0, 1.6.
    let curves = [curve_from(&[(1.0, 0.4), (2.0, 1.0), (3.0, 1.6)])];
    let queries = [
        ScatterQuery {
            curve: 0,
            depth: 1.5,
        }, // midway in first span: 0.4 + 0.3 = 0.7
        ScatterQuery {
            curve: 0,
            depth: 2.25,
        }, // quarter into second span: 1.0 + 0.15 = 1.15
    ];
    let gpu = assert_batch_parity(&ctx, &decoder, &curves, &queries);
    assert_close(gpu[0], 0.7, "midway in first span");
    assert_close(gpu[1], 1.15, "quarter into second span");
}

/// A receiver exactly on an interior boundary reads that boundary's count.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_exact_boundary() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping forward scatter sample parity: no wgpu adapter on this host");
        return;
    };
    let decoder = GpuHairForwardScatterSample::new(&ctx);
    let curves = [curve_from(&[(1.0, 0.4), (2.0, 1.0), (3.0, 1.6)])];
    let queries = [ScatterQuery {
        curve: 0,
        depth: 2.0,
    }];
    let gpu = assert_batch_parity(&ctx, &decoder, &curves, &queries);
    assert_close(gpu[0], 1.0, "exact interior boundary reads its count");
}

/// An empty (no-occluder) curve decodes to `0` at every depth.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_empty_curve_reads_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping forward scatter sample parity: no wgpu adapter on this host");
        return;
    };
    let decoder = GpuHairForwardScatterSample::new(&ctx);
    let curves = [ForwardScatterLayers::default()];
    let queries = [
        ScatterQuery {
            curve: 0,
            depth: -1.0,
        },
        ScatterQuery {
            curve: 0,
            depth: 0.0,
        },
        ScatterQuery {
            curve: 0,
            depth: 5.0,
        },
    ];
    let gpu = assert_batch_parity(&ctx, &decoder, &curves, &queries);
    for (i, v) in gpu.iter().enumerate() {
        assert_eq!(*v, 0.0, "empty curve query {i} reads zero exactly");
    }
}

/// A degenerate window with a zero-width span uses the `t = 0` branch, reading
/// the near boundary's count rather than dividing by zero.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_degenerate_span() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping forward scatter sample parity: no wgpu adapter on this host");
        return;
    };
    let decoder = GpuHairForwardScatterSample::new(&ctx);
    // Two boundaries share depth 2.0 (zero-width middle span); a query landing
    // in the (1.0, 2.0] window interpolates, but a probe at exactly 2.0 hits the
    // first `depth <= d1` window and reads its near count.
    let curves = [curve_from(&[
        (1.0, 0.5),
        (2.0, 1.0),
        (2.0, 1.4),
        (3.0, 2.0),
    ])];
    let queries = [
        ScatterQuery {
            curve: 0,
            depth: 2.0,
        },
        ScatterQuery {
            curve: 0,
            depth: 2.5,
        },
    ];
    let gpu = assert_batch_parity(&ctx, &decoder, &curves, &queries);
    // Golden hits the first window (1.0,2.0] with t=1.0 -> 1.0, matching the CPU;
    // the parity assert already guarantees agreement, this pins the shape.
    assert_close(
        gpu[0],
        1.0,
        "exact-at-duplicate-boundary reads first window end",
    );
    assert_close(
        gpu[1],
        1.7,
        "midway in the (2.0,3.0] span after the duplicate",
    );
}

/// A multi-curve batch: each query fans out to its own curve, and an
/// out-of-range curve index decodes to `0` without panicking.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_multi_curve_batch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping forward scatter sample parity: no wgpu adapter on this host");
        return;
    };
    let decoder = GpuHairForwardScatterSample::new(&ctx);
    let curves = [
        curve_from(&[(1.0, 0.5), (2.0, 1.0)]),
        ForwardScatterLayers::default(),
        curve_from(&[(0.5, 0.2), (1.5, 0.9), (4.0, 2.4)]),
    ];
    let queries = [
        ScatterQuery {
            curve: 0,
            depth: 1.5,
        },
        ScatterQuery {
            curve: 2,
            depth: 3.0,
        },
        ScatterQuery {
            curve: 1,
            depth: 2.0,
        },
        ScatterQuery {
            curve: 2,
            depth: 0.75,
        },
        ScatterQuery {
            curve: 0,
            depth: 9.0,
        },
    ];
    assert_batch_parity(&ctx, &decoder, &curves, &queries);

    // Out-of-range curve index has no golden equivalent; the guard must decode
    // it to zero rather than panic or read garbage.
    let guarded = [ScatterQuery {
        curve: 7,
        depth: 1.5,
    }];
    let gpu = decoder.eval(&ctx, &curves, &guarded);
    assert_eq!(gpu.len(), 1, "one output for the guarded query");
    assert_eq!(gpu[0], 0.0, "out-of-range curve decodes to zero");
}

/// A batch larger than one `64`-wide workgroup: every query in the second group
/// must decode correctly, proving the dispatch covers all invocations.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_large_batch_crosses_workgroup() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping forward scatter sample parity: no wgpu adapter on this host");
        return;
    };
    let decoder = GpuHairForwardScatterSample::new(&ctx);
    let curves = [curve_from(&[
        (0.0, 0.25),
        (2.0, 0.75),
        (4.0, 1.5),
        (6.0, 2.0),
    ])];
    // 150 queries sweeping the depth range, well past the 64-wide workgroup.
    let queries: Vec<ScatterQuery> = (0..150)
        .map(|i| ScatterQuery {
            curve: 0,
            depth: i as f32 * 0.05,
        })
        .collect();
    assert_batch_parity(&ctx, &decoder, &curves, &queries);
}

/// Build a curve with the golden `build_forward_scatter`, then decode a depth
/// sweep on the `GPU` and compare to the golden decode: the packing and the
/// decode must agree end-to-end.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_build_then_sample_roundtrip() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping forward scatter sample parity: no wgpu adapter on this host");
        return;
    };
    let decoder = GpuHairForwardScatterSample::new(&ctx);
    let samples = [
        TransmittanceSample::new(0.5, 0.4),
        TransmittanceSample::new(1.5, 0.6),
        TransmittanceSample::new(3.0, 0.8),
        TransmittanceSample::new(5.0, 0.5),
        TransmittanceSample::new(7.5, 0.9),
    ];
    let curve = build_forward_scatter(&samples, 4, 0.0);
    let curves = [curve];
    let queries: Vec<ScatterQuery> = (0..40)
        .map(|i| ScatterQuery {
            curve: 0,
            depth: i as f32 * 0.25,
        })
        .collect();
    assert_batch_parity(&ctx, &decoder, &curves, &queries);
}
