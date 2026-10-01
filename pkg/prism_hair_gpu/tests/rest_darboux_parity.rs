//! Real-device parity for the per-edge rest `Darboux` twin:
//! [`GpuRestDarboux`] must reproduce the `CPU` golden
//! [`rest_darboux_from_frames`](prism_render_architecture::hair::cosserat::rest_darboux_from_frames)
//! for a batch of material-frame chains, emitting one rest `Darboux` vector per
//! adjacent frame pair (`frames.len() - 1` per strand, empty under two frames).
//!
//! # Parity criterion
//!
//! The rest `Darboux` vector is the imaginary part of the single Hamilton
//! product `conjugate(a) * b` — no `normalize`, no `sanitize` — so the only
//! `CPU` vs `GPU` divergence is legal fused-multiply-add contraction. Each
//! component is asserted to within `abs_diff < 1e-4` or `rel_diff < 1e-3`. An
//! identity frame pair rotates nothing, so its rest `Darboux` vector is a
//! bit-exact zero (asserted on raw `f32` bit patterns): a kernel that fused in
//! a stray term could not pass. Non-identity strands are also asserted to carry
//! at least one non-zero component, so a zero-writing no-op kernel could not
//! pass either.
//!
//! The suite drives a single multi-frame strand, a mixed multi-strand batch, an
//! all-identity strand (bit-exact zero), a sub-two-frame strand (empty), a
//! wholly degenerate batch (every strand under two frames), and a >64-edge
//! strand that crosses the 64-wide dispatch boundary.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-`WGSL`,
//! so it needs no optional device feature. Frames are built by parallel
//! transport along integer/affine poly-lines or by direct normalized
//! quaternions — never `f32::sin`/`cos` — so test data stays deterministic
//! without introducing transcendental divergence.
//!
//! Provenance: standard `Cosserat`/`Kirchhoff` rod rest-`Darboux` construction
//! plus a `wgpu` compute dispatch; no Unreal Engine source or derived code.

use prism_hair_gpu::rest_darboux::{reference_rest_darboux, DarbouxStrand, GpuRestDarboux};
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::cosserat::{parallel_transport_frames, Quat, Vec3};

/// Acquires a headless context, or `None` (with a skip notice) when the host has
/// no `wgpu` adapter so the suite stays green off-device.
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn context_or_skip(label: &str) -> Option<GpuContext> {
    match GpuContext::try_headless() {
        Some(ctx) => Some(ctx),
        None => {
            eprintln!("skipping {label}: no wgpu adapter on this host");
            None
        }
    }
}

/// True when `got` matches `want` within the fused-multiply-add tolerance.
fn close(got: f32, want: f32) -> bool {
    let diff = (got - want).abs();
    diff < 1.0e-4 || diff <= 1.0e-3 * want.abs()
}

/// Asserts every strand's rest `Darboux` chain matches the `CPU` golden term
/// for term.
fn assert_batch_matches(gpu: &[Vec<Vec3>], strands: &[&[Quat]]) {
    assert_eq!(gpu.len(), strands.len(), "one output chain per strand");
    for (s, frames) in strands.iter().enumerate() {
        let want = reference_rest_darboux(frames);
        assert_eq!(
            gpu[s].len(),
            want.len(),
            "strand {s} rest Darboux count mismatch",
        );
        for (j, (g, w)) in gpu[s].iter().zip(&want).enumerate() {
            assert!(
                close(g.x, w.x) && close(g.y, w.y) && close(g.z, w.z),
                "strand {s} edge {j}: gpu ({}, {}, {}) vs cpu ({}, {}, {})",
                g.x,
                g.y,
                g.z,
                w.x,
                w.y,
                w.z,
            );
        }
    }
}

/// A deterministic bent poly-line whose transported frames twist and bend, so
/// its rest `Darboux` vectors are non-trivial. Uses only integer coordinates —
/// no transcendental calls.
fn bent_frames(len: usize) -> Vec<Quat> {
    let mut points = Vec::with_capacity(len + 1);
    for i in 0..=len {
        let t = i as f32;
        // A space curve from pure integer/affine arithmetic (no sin/cos).
        let x = t;
        let y = (i % 3) as f32 - 1.0;
        let z = (i % 2) as f32;
        points.push(Vec3::new(x, y, z));
    }
    parallel_transport_frames(&points, Quat::IDENTITY)
}

#[test]
fn gpu_single_strand_matches_cpu() {
    let Some(ctx) = context_or_skip("gpu_single_strand_matches_cpu") else {
        return;
    };
    let frames = bent_frames(7);
    assert_eq!(frames.len(), 7, "8 points transport to 7 frames");
    let twin = GpuRestDarboux::new(&ctx);
    let strands = [DarbouxStrand { frames: &frames }];
    let gpu = twin.eval(&ctx, &strands);
    assert_batch_matches(&gpu, &[frames.as_slice()]);
    assert_eq!(
        gpu[0].len(),
        6,
        "a 7-frame strand must yield 6 rest Darboux vectors",
    );
    // A bent/twisted strand must carry at least one non-zero component so a
    // zero-writing no-op kernel could not pass.
    let any_nonzero = gpu[0]
        .iter()
        .any(|d| d.x.abs() > 1.0e-4 || d.y.abs() > 1.0e-4 || d.z.abs() > 1.0e-4);
    assert!(any_nonzero, "bent strand rest Darboux must be non-trivial");
}

#[test]
fn gpu_multi_strand_batch_matches_cpu() {
    let Some(ctx) = context_or_skip("gpu_multi_strand_batch_matches_cpu") else {
        return;
    };
    let a = bent_frames(4);
    let b = bent_frames(9);
    let c = vec![
        Quat::new(1.0, 0.5, -0.25, 0.75).normalize(),
        Quat::new(0.25, 1.0, 0.5, -0.5).normalize(),
        Quat::new(-0.5, 0.25, 1.0, 0.5).normalize(),
    ];
    let twin = GpuRestDarboux::new(&ctx);
    let strands = [
        DarbouxStrand { frames: &a },
        DarbouxStrand { frames: &b },
        DarbouxStrand { frames: &c },
    ];
    let gpu = twin.eval(&ctx, &strands);
    assert_batch_matches(&gpu, &[a.as_slice(), b.as_slice(), c.as_slice()]);
}

#[test]
fn gpu_identity_frames_are_bit_exact_zero() {
    let Some(ctx) = context_or_skip("gpu_identity_frames_are_bit_exact_zero") else {
        return;
    };
    // conjugate(I) * I == I, whose imaginary part is exactly (0, 0, 0).
    let frames = vec![Quat::IDENTITY; 5];
    let twin = GpuRestDarboux::new(&ctx);
    let strands = [DarbouxStrand { frames: &frames }];
    let gpu = twin.eval(&ctx, &strands);
    assert_eq!(gpu.len(), 1);
    assert_eq!(gpu[0].len(), 4, "5 identity frames yield 4 edges");
    for d in &gpu[0] {
        assert_eq!(d.x.to_bits(), 0.0f32.to_bits(), "identity bend1 must be +0");
        assert_eq!(d.y.to_bits(), 0.0f32.to_bits(), "identity bend2 must be +0");
        assert_eq!(d.z.to_bits(), 0.0f32.to_bits(), "identity twist must be +0");
    }
}

#[test]
fn gpu_sub_two_frame_strand_is_empty() {
    let Some(ctx) = context_or_skip("gpu_sub_two_frame_strand_is_empty") else {
        return;
    };
    let single = vec![Quat::IDENTITY];
    let twin = GpuRestDarboux::new(&ctx);
    // A one-frame strand batched with a real strand: the short strand yields no
    // edges, the real one is unaffected.
    let real = bent_frames(5);
    let strands = [
        DarbouxStrand { frames: &single },
        DarbouxStrand { frames: &real },
    ];
    let gpu = twin.eval(&ctx, &strands);
    assert_eq!(gpu.len(), 2);
    assert!(gpu[0].is_empty(), "a one-frame strand yields no edges");
    assert_batch_matches(&gpu, &[single.as_slice(), real.as_slice()]);
}

#[test]
fn gpu_wholly_degenerate_batch_is_empty() {
    let Some(ctx) = context_or_skip("gpu_wholly_degenerate_batch_is_empty") else {
        return;
    };
    // Every strand under two frames: no dispatch, every output chain empty.
    let empty: Vec<Quat> = Vec::new();
    let one = vec![Quat::IDENTITY];
    let twin = GpuRestDarboux::new(&ctx);
    let strands = [
        DarbouxStrand { frames: &empty },
        DarbouxStrand { frames: &one },
    ];
    let gpu = twin.eval(&ctx, &strands);
    assert_eq!(gpu.len(), 2);
    assert!(gpu[0].is_empty() && gpu[1].is_empty());
    // The empty top-level batch is likewise handled without a dispatch.
    let none = twin.eval(&ctx, &[]);
    assert!(none.is_empty(), "empty batch yields no output chains");
}

#[test]
fn gpu_cross_workgroup_batch_matches_cpu() {
    let Some(ctx) = context_or_skip("gpu_cross_workgroup_batch_matches_cpu") else {
        return;
    };
    // 200 frames -> 199 edges, well past the 64-wide dispatch boundary.
    let frames = bent_frames(200);
    assert_eq!(frames.len(), 200, "201 points transport to 200 frames");
    let twin = GpuRestDarboux::new(&ctx);
    let strands = [DarbouxStrand { frames: &frames }];
    let gpu = twin.eval(&ctx, &strands);
    assert_eq!(gpu[0].len(), 199, "199 edges across three workgroups");
    assert_batch_matches(&gpu, &[frames.as_slice()]);
}
