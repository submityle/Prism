//! Real-device parity for the deep-opacity transmittance decode twin:
//! [`GpuHairDeepTransmittanceSample`] must reproduce the `CPU` golden
//! [`sample_transmittance`](prism_render_architecture::hair::deep_transmittance::sample_transmittance)
//! for a batch of receiver queries against packed
//! [`DeepOpacityLayers`](prism_render_architecture::hair::deep_transmittance::DeepOpacityLayers)
//! curves, covering the front-of-curve fully-lit read, the beyond-deepest
//! last-transmittance read, the interior linear interpolation between
//! bracketing boundaries, an exact-boundary probe, the empty (no-occluder)
//! curve, a degenerate zero-span window, a multi-curve batch whose queries fan
//! out to different curves, an out-of-range curve guard, a batch that crosses
//! the `64`-wide workgroup boundary, and a build-then-sample round-trip off the
//! golden
//! [`build_deep_opacity`](prism_render_architecture::hair::deep_transmittance::build_deep_opacity).
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
//! from explicit depth/transmittance literals (never `sin`/`cos`), and the
//! empty/front reads are asserted exactly so a no-op kernel could not pass.
//!
//! This is the **multiplicative dual** of the additive `forward_scatter_sample`
//! twin: the same layered bracket-and-interpolate, but the packed curve is the
//! monotonically non-increasing transmittance `T` in `0..=1`, and the
//! out-of-band reads are fully transmissive (`1.0`) rather than `0`.
//!
//! Provenance: standard deep-opacity transmittance curve decode plus `wgpu`
//! compute dispatch; no Unreal Engine source or derived code.

use prism_hair_gpu::deep_transmittance_sample::{
    GpuHairDeepTransmittanceSample, TransmittanceQuery,
};
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::deep_transmittance::{
    build_deep_opacity, sample_transmittance, DeepOpacityLayers, TransmittanceSample,
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

/// Builds a curve directly from parallel `(depth, transmittance)` pairs.
fn curve_from(pairs: &[(f32, f32)]) -> DeepOpacityLayers {
    DeepOpacityLayers {
        layer_depths: pairs.iter().map(|&(d, _)| d).collect(),
        layer_transmittance: pairs.iter().map(|&(_, t)| t).collect(),
    }
}

/// Runs the decode twin and the golden on the same curves/queries and asserts
/// every decoded transmittance matches the golden for its named curve.
fn assert_batch_parity(
    ctx: &GpuContext,
    decoder: &GpuHairDeepTransmittanceSample,
    curves: &[DeepOpacityLayers],
    queries: &[TransmittanceQuery],
) -> Vec<f32> {
    let gpu = decoder.eval(ctx, curves, queries);
    assert_eq!(gpu.len(), queries.len(), "one output per query");
    for (i, q) in queries.iter().enumerate() {
        let cpu = sample_transmittance(&curves[q.curve as usize], q.depth);
        assert_close(
            gpu[i],
            cpu,
            &format!("query {i} (curve {} d {})", q.curve, q.depth),
        );
    }
    gpu
}

/// A receiver in front of the frontmost boundary reads exactly `1.0` — nothing
/// occludes it yet, so it is fully lit.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_front_reads_one() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping deep transmittance sample parity: no wgpu adapter on this host");
        return;
    };
    let decoder = GpuHairDeepTransmittanceSample::new(&ctx);
    let curves = [curve_from(&[(1.0, 0.8), (2.0, 0.5), (3.0, 0.2)])];
    let queries = [
        TransmittanceQuery {
            curve: 0,
            depth: 0.0,
        },
        TransmittanceQuery {
            curve: 0,
            depth: 1.0,
        },
    ];
    let gpu = assert_batch_parity(&ctx, &decoder, &curves, &queries);
    assert_eq!(gpu[0], 1.0, "well in front reads one exactly");
    assert_eq!(gpu[1], 1.0, "at the front boundary reads one exactly");
}

/// A receiver at or beyond the deepest boundary takes the last (most occluded)
/// transmittance.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_beyond_reads_last() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping deep transmittance sample parity: no wgpu adapter on this host");
        return;
    };
    let decoder = GpuHairDeepTransmittanceSample::new(&ctx);
    let curves = [curve_from(&[(1.0, 0.8), (2.0, 0.5), (3.0, 0.2)])];
    let queries = [
        TransmittanceQuery {
            curve: 0,
            depth: 3.0,
        },
        TransmittanceQuery {
            curve: 0,
            depth: 9.0,
        },
    ];
    let gpu = assert_batch_parity(&ctx, &decoder, &curves, &queries);
    assert_close(gpu[0], 0.2, "at deepest boundary reads last transmittance");
    assert_close(
        gpu[1],
        0.2,
        "beyond deepest boundary saturates at last transmittance",
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
        eprintln!("skipping deep transmittance sample parity: no wgpu adapter on this host");
        return;
    };
    let decoder = GpuHairDeepTransmittanceSample::new(&ctx);
    // Boundaries at 1,2,3 with transmittance 0.8, 0.5, 0.2.
    let curves = [curve_from(&[(1.0, 0.8), (2.0, 0.5), (3.0, 0.2)])];
    let queries = [
        TransmittanceQuery {
            curve: 0,
            depth: 1.5,
        }, // midway in first span: 0.8 + (0.5 - 0.8) * 0.5 = 0.65
        TransmittanceQuery {
            curve: 0,
            depth: 2.25,
        }, // quarter into second span: 0.5 + (0.2 - 0.5) * 0.25 = 0.425
    ];
    let gpu = assert_batch_parity(&ctx, &decoder, &curves, &queries);
    assert_close(gpu[0], 0.65, "midway in first span");
    assert_close(gpu[1], 0.425, "quarter into second span");
}

/// A receiver exactly on an interior boundary reads that boundary's
/// transmittance (the first `depth <= d1` window ends there).
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_exact_boundary() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping deep transmittance sample parity: no wgpu adapter on this host");
        return;
    };
    let decoder = GpuHairDeepTransmittanceSample::new(&ctx);
    let curves = [curve_from(&[(1.0, 0.8), (2.0, 0.5), (3.0, 0.2)])];
    let queries = [TransmittanceQuery {
        curve: 0,
        depth: 2.0,
    }];
    let gpu = assert_batch_parity(&ctx, &decoder, &curves, &queries);
    assert_close(
        gpu[0],
        0.5,
        "exact interior boundary reads its transmittance",
    );
}

/// An empty (no-occluder) curve decodes to `1.0` (fully lit) at every depth.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_empty_curve_reads_one() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping deep transmittance sample parity: no wgpu adapter on this host");
        return;
    };
    let decoder = GpuHairDeepTransmittanceSample::new(&ctx);
    let curves = [DeepOpacityLayers::default()];
    let queries = [
        TransmittanceQuery {
            curve: 0,
            depth: -1.0,
        },
        TransmittanceQuery {
            curve: 0,
            depth: 0.0,
        },
        TransmittanceQuery {
            curve: 0,
            depth: 5.0,
        },
    ];
    let gpu = assert_batch_parity(&ctx, &decoder, &curves, &queries);
    for (i, v) in gpu.iter().enumerate() {
        assert_eq!(*v, 1.0, "empty curve query {i} reads one exactly");
    }
}

/// A degenerate window with a zero-width span uses the `f = 0` branch, reading
/// the near boundary's transmittance rather than dividing by zero.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_degenerate_span() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping deep transmittance sample parity: no wgpu adapter on this host");
        return;
    };
    let decoder = GpuHairDeepTransmittanceSample::new(&ctx);
    // Two boundaries share depth 2.0 (zero-width middle span); a probe at
    // exactly 2.0 hits the first `depth <= d1` window and reads its far value,
    // while a query in the (2.0, 3.0] window after the duplicate interpolates.
    let curves = [curve_from(&[
        (1.0, 0.8),
        (2.0, 0.5),
        (2.0, 0.4),
        (3.0, 0.2),
    ])];
    let queries = [
        TransmittanceQuery {
            curve: 0,
            depth: 2.0,
        },
        TransmittanceQuery {
            curve: 0,
            depth: 2.5,
        },
    ];
    let gpu = assert_batch_parity(&ctx, &decoder, &curves, &queries);
    // Golden hits the first window (1.0, 2.0] with t = 1.0 -> 0.5; the parity
    // assert already guarantees agreement, this pins the shape.
    assert_close(
        gpu[0],
        0.5,
        "exact-at-duplicate-boundary reads first window end",
    );
    // Midway in (2.0, 3.0] after the duplicate: 0.4 + (0.2 - 0.4) * 0.5 = 0.3.
    assert_close(
        gpu[1],
        0.3,
        "midway in the (2.0,3.0] span after the duplicate",
    );
}

/// A multi-curve batch: each query fans out to its own curve, and an
/// out-of-range curve index decodes to `1.0` without panicking.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_multi_curve_batch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping deep transmittance sample parity: no wgpu adapter on this host");
        return;
    };
    let decoder = GpuHairDeepTransmittanceSample::new(&ctx);
    let curves = [
        curve_from(&[(1.0, 0.7), (2.0, 0.4)]),
        DeepOpacityLayers::default(),
        curve_from(&[(0.5, 0.9), (1.5, 0.6), (4.0, 0.1)]),
    ];
    let queries = [
        TransmittanceQuery {
            curve: 0,
            depth: 1.5,
        },
        TransmittanceQuery {
            curve: 2,
            depth: 3.0,
        },
        TransmittanceQuery {
            curve: 1,
            depth: 2.0,
        },
        TransmittanceQuery {
            curve: 2,
            depth: 0.75,
        },
        TransmittanceQuery {
            curve: 0,
            depth: 9.0,
        },
    ];
    assert_batch_parity(&ctx, &decoder, &curves, &queries);

    // Out-of-range curve index has no golden equivalent; the guard must decode
    // it to one (fully transmissive) rather than panic or read garbage.
    let guarded = [TransmittanceQuery {
        curve: 7,
        depth: 1.5,
    }];
    let gpu = decoder.eval(&ctx, &curves, &guarded);
    assert_eq!(gpu.len(), 1, "one output for the guarded query");
    assert_eq!(gpu[0], 1.0, "out-of-range curve decodes to one");
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
        eprintln!("skipping deep transmittance sample parity: no wgpu adapter on this host");
        return;
    };
    let decoder = GpuHairDeepTransmittanceSample::new(&ctx);
    let curves = [curve_from(&[
        (0.0, 0.95),
        (2.0, 0.6),
        (4.0, 0.3),
        (6.0, 0.05),
    ])];
    // 150 queries sweeping the depth range, well past the 64-wide workgroup.
    let queries: Vec<TransmittanceQuery> = (0..150)
        .map(|i| TransmittanceQuery {
            curve: 0,
            depth: i as f32 * 0.05,
        })
        .collect();
    assert_batch_parity(&ctx, &decoder, &curves, &queries);
}

/// Build a curve with the golden `build_deep_opacity`, then decode a depth
/// sweep on the `GPU` and compare to the golden decode: the packing and the
/// decode must agree end-to-end.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_build_then_sample_roundtrip() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping deep transmittance sample parity: no wgpu adapter on this host");
        return;
    };
    let decoder = GpuHairDeepTransmittanceSample::new(&ctx);
    let samples = [
        TransmittanceSample::new(0.5, 0.4),
        TransmittanceSample::new(1.5, 0.6),
        TransmittanceSample::new(3.0, 0.8),
        TransmittanceSample::new(5.0, 0.5),
        TransmittanceSample::new(7.5, 0.9),
    ];
    let curve = build_deep_opacity(&samples, 4, 0.0);
    let curves = [curve];
    let queries: Vec<TransmittanceQuery> = (0..40)
        .map(|i| TransmittanceQuery {
            curve: 0,
            depth: i as f32 * 0.25,
        })
        .collect();
    assert_batch_parity(&ctx, &decoder, &curves, &queries);
}
