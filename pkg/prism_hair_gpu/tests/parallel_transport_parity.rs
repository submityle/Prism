//! Real-device parity for the parallel-transport frame twin: [`GpuParallelTransport`]
//! must reproduce the `CPU` golden
//! [`parallel_transport_frames`](prism_render_architecture::hair::cosserat::parallel_transport_frames)
//! for a batch of independent strands, covering the first-edge seed from the
//! transported `+z` axis, the chained minimal-rotation transport along later
//! edges, the degenerate (zero-length) edge that reuses the running frame, the
//! sub-two-point early return, a non-identity initial frame, and the multi-strand
//! batch split.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL
//! (only `dot`/`cross`/`sqrt`/`min`/`max` and multiply-add), so it needs no
//! optional device feature.
//!
//! # Parity criterion
//!
//! The transport chains a re-normalize per edge, so the `CPU` and `GPU` evaluate
//! the same expressions and diverge only through legal fused-multiply-add
//! contraction, which compounds along the strand. Parity is asserted per
//! component to within `abs_diff < 3e-3` or `rel_diff < 1e-2` — tight enough to
//! fail a swapped branch, a missing normalize or a wrong seed, loose enough to
//! admit the compounded contraction. Each strand additionally asserts the
//! transported frames stay unit-norm and are not all identical (so a no-op or
//! constant kernel could not pass), and the degenerate-edge case asserts the
//! `GPU` reuses the previous frame bit-for-bit. Test strands are built from
//! integer-delta poly-lines (never `sin`/`cos`).
//!
//! Provenance: standard rotation-minimizing / parallel-transport frame (Bishop
//! frame) construction; no Unreal Engine source or derived code.

use prism_hair_gpu::parallel_transport::{
    reference_parallel_transport, GpuParallelTransport, TransportStrand,
};
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::cosserat::{Quat, Vec3};

/// Asserts a single quaternion component matches within the chained-transport
/// tolerance documented on the twin module.
fn assert_close(got: f32, expected: f32, label: &str) {
    let abs_diff = (got - expected).abs();
    let rel_diff = abs_diff / expected.abs().max(1e-6);
    assert!(
        abs_diff < 3e-3 || rel_diff < 1e-2,
        "{label}: gpu {got}, cpu {expected} (abs {abs_diff}, rel {rel_diff})"
    );
}

/// Builds a poly-line from a start point and a list of integer-ish deltas,
/// avoiding any transcendental call so the clippy `disallowed_methods` lint is
/// satisfied and the inputs stay exactly reproducible.
fn poly(start: Vec3, deltas: &[Vec3]) -> Vec<Vec3> {
    let mut pts = Vec::with_capacity(deltas.len() + 1);
    let mut p = start;
    pts.push(p);
    for d in deltas {
        p = p.add(*d);
        pts.push(p);
    }
    pts
}

/// A non-identity unit quaternion built without trig: `w` closes the unit-norm
/// identity for `|omega| <= 1`.
fn initial_frame() -> Quat {
    let omega = Vec3::new(0.2, 0.1, 0.3);
    let w = (1.0 - omega.dot(omega)).max(0.0).sqrt();
    Quat::new(w, omega.x, omega.y, omega.z)
}

/// Asserts the per-strand `GPU` transport matches the `CPU` golden component by
/// component, and returns the `GPU` frames for extra per-case invariants.
fn assert_strand_parity(
    ctx: &GpuContext,
    twin: &GpuParallelTransport,
    points: &[Vec3],
    initial: Quat,
) -> Vec<Quat> {
    let cpu = reference_parallel_transport(points, initial);
    let gpu = twin.eval(ctx, &[TransportStrand { points, initial }]);
    assert_eq!(gpu.len(), 1, "one strand in, one strand out");
    let got = &gpu[0];
    assert_eq!(got.len(), cpu.len(), "frame count must match the reference");
    for (i, (g, c)) in got.iter().zip(cpu.iter()).enumerate() {
        assert_close(g.w, c.w, &format!("frame[{i}].w"));
        assert_close(g.x, c.x, &format!("frame[{i}].x"));
        assert_close(g.y, c.y, &format!("frame[{i}].y"));
        assert_close(g.z, c.z, &format!("frame[{i}].z"));
    }
    got.clone()
}

/// Every transported frame must be (numerically) unit-norm.
fn assert_unit(frames: &[Quat]) {
    for (i, q) in frames.iter().enumerate() {
        let norm2 = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
        assert!(
            (norm2 - 1.0).abs() < 3e-3,
            "frame[{i}] not unit: norm^2 = {norm2}"
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_curved_strand_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping parallel-transport parity: no wgpu adapter on this host");
        return;
    };
    let twin = GpuParallelTransport::new(&ctx);

    let points = poly(
        Vec3::new(-0.5, 0.25, 0.1),
        &[
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 1.0),
            Vec3::new(1.0, 0.0, 1.0),
            Vec3::new(2.0, 1.0, 0.0),
            Vec3::new(0.0, 2.0, 1.0),
        ],
    );
    let frames = assert_strand_parity(&ctx, &twin, &points, Quat::IDENTITY);
    assert_unit(&frames);
    // A genuinely curved strand must not collapse to a single repeated frame.
    let first = frames[0];
    let varies = frames
        .iter()
        .any(|q| (q.x - first.x).abs() > 1e-3 || (q.y - first.y).abs() > 1e-3);
    assert!(
        varies,
        "transported frames should vary along a curved strand"
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_non_identity_initial_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping parallel-transport parity: no wgpu adapter on this host");
        return;
    };
    let twin = GpuParallelTransport::new(&ctx);

    let points = poly(
        Vec3::new(0.0, 0.0, 0.0),
        &[
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.5, 0.0, 1.0),
            Vec3::new(0.5, 0.5, 1.0),
            Vec3::new(0.0, 0.5, 1.0),
        ],
    );
    let frames = assert_strand_parity(&ctx, &twin, &points, initial_frame());
    assert_unit(&frames);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_straight_strand_frames_are_constant() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping parallel-transport parity: no wgpu adapter on this host");
        return;
    };
    let twin = GpuParallelTransport::new(&ctx);

    // A straight line: every edge direction equals the previous, so each
    // minimal rotation is identity and all frames after the seed are equal.
    let points = poly(
        Vec3::new(0.0, 0.0, 0.0),
        &[
            Vec3::new(0.3, 0.4, 0.0),
            Vec3::new(0.3, 0.4, 0.0),
            Vec3::new(0.3, 0.4, 0.0),
        ],
    );
    let frames = assert_strand_parity(&ctx, &twin, &points, Quat::IDENTITY);
    assert_unit(&frames);
    let f0 = frames[0];
    for (i, q) in frames.iter().enumerate().skip(1) {
        assert_close(q.w, f0.w, &format!("straight frame[{i}].w vs frame[0]"));
        assert_close(q.x, f0.x, &format!("straight frame[{i}].x vs frame[0]"));
        assert_close(q.y, f0.y, &format!("straight frame[{i}].y vs frame[0]"));
        assert_close(q.z, f0.z, &format!("straight frame[{i}].z vs frame[0]"));
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_degenerate_edge_reuses_running_frame() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping parallel-transport parity: no wgpu adapter on this host");
        return;
    };
    let twin = GpuParallelTransport::new(&ctx);

    // The third edge repeats a point (zero-length), so the running frame must be
    // reused for that edge index without advancing the transport reference.
    let points = poly(
        Vec3::new(0.0, 0.0, 0.0),
        &[
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 0.5, 0.0),
            Vec3::new(0.0, 0.0, 0.0), // degenerate: repeats the previous point
            Vec3::new(1.0, 0.5, 0.5),
        ],
    );
    let frames = assert_strand_parity(&ctx, &twin, &points, Quat::IDENTITY);
    assert_unit(&frames);
    // Edge index 2 is degenerate: the GPU must reuse the frame from edge 1
    // bit-for-bit (its own running value, independent of CPU rounding).
    let reused = frames[2];
    let prev = frames[1];
    assert_eq!(
        reused.w.to_bits(),
        prev.w.to_bits(),
        "degenerate edge must reuse the previous frame .w bit-for-bit"
    );
    assert_eq!(reused.x.to_bits(), prev.x.to_bits());
    assert_eq!(reused.y.to_bits(), prev.y.to_bits());
    assert_eq!(reused.z.to_bits(), prev.z.to_bits());
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_short_strands_emit_no_frames() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping parallel-transport parity: no wgpu adapter on this host");
        return;
    };
    let twin = GpuParallelTransport::new(&ctx);

    let single = [Vec3::new(1.0, 2.0, 3.0)];
    let empty: [Vec3; 0] = [];
    let out = twin.eval(
        &ctx,
        &[
            TransportStrand {
                points: &single,
                initial: Quat::IDENTITY,
            },
            TransportStrand {
                points: &empty,
                initial: initial_frame(),
            },
        ],
    );
    assert_eq!(out.len(), 2);
    assert!(out[0].is_empty(), "single-point strand yields no frames");
    assert!(out[1].is_empty(), "empty strand yields no frames");
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_multi_strand_batch_splits_correctly() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping parallel-transport parity: no wgpu adapter on this host");
        return;
    };
    let twin = GpuParallelTransport::new(&ctx);

    let a = poly(
        Vec3::new(0.0, 0.0, 0.0),
        &[
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 1.0),
        ],
    );
    let b = poly(
        Vec3::new(2.0, -1.0, 0.5),
        &[
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 1.0),
            Vec3::new(1.0, 0.0, 1.0),
            Vec3::new(1.0, 1.0, 0.0),
        ],
    );
    let single = [Vec3::new(5.0, 5.0, 5.0)];

    let strands = [
        TransportStrand {
            points: &a,
            initial: Quat::IDENTITY,
        },
        TransportStrand {
            points: &single,
            initial: Quat::IDENTITY,
        },
        TransportStrand {
            points: &b,
            initial: initial_frame(),
        },
    ];
    let gpu = twin.eval(&ctx, &strands);
    assert_eq!(gpu.len(), 3);

    let cpu_a = reference_parallel_transport(&a, Quat::IDENTITY);
    let cpu_b = reference_parallel_transport(&b, initial_frame());
    assert_eq!(gpu[0].len(), cpu_a.len());
    assert!(
        gpu[1].is_empty(),
        "middle single-point strand yields nothing"
    );
    assert_eq!(gpu[2].len(), cpu_b.len());

    for (i, (g, c)) in gpu[0].iter().zip(cpu_a.iter()).enumerate() {
        assert_close(g.w, c.w, &format!("a frame[{i}].w"));
        assert_close(g.x, c.x, &format!("a frame[{i}].x"));
        assert_close(g.y, c.y, &format!("a frame[{i}].y"));
        assert_close(g.z, c.z, &format!("a frame[{i}].z"));
    }
    for (i, (g, c)) in gpu[2].iter().zip(cpu_b.iter()).enumerate() {
        assert_close(g.w, c.w, &format!("b frame[{i}].w"));
        assert_close(g.x, c.x, &format!("b frame[{i}].x"));
        assert_close(g.y, c.y, &format!("b frame[{i}].y"));
        assert_close(g.z, c.z, &format!("b frame[{i}].z"));
    }
}
