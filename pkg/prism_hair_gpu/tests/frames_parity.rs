//! Real-device parity for the rotation-minimizing strand-frame twin:
//! [`GpuStrandFrames`] must reproduce the `CPU` golden
//! [`build_strand_frames`](prism_render_architecture::hair::frames::build_strand_frames)
//! for every control point of every strand, including the single-point and
//! coincident-point edge cases the reference guards, and it must return an
//! orthonormal right-handed basis at each point.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The transport is closed-form geometry (the reference restricts itself to
//! `sqrt`), so the `CPU` and `GPU` evaluate the same expression and diverge only
//! through legal fused-multiply-add contraction. Each frame component is
//! asserted to within `abs_diff < 1e-4` or `rel_diff < 1e-3` — tight enough to
//! fail a genuinely wrong port (a swapped reflection, a missing
//! re-orthonormalization, a sign error), loose enough to admit fma contraction.
//! The smooth sweeps also assert orthonormality (the axes are unit, mutually
//! perpendicular and right-handed) so a degenerate kernel could not pass. Test
//! inputs stay well away from the degenerate `len_sq <= f32::EPSILON` thresholds
//! so `CPU` and `GPU` never diverge on a branch; the degenerate cases use
//! exactly coincident points so both sides take the fallback branch bit-for-bit.
//!
//! Provenance: standard double-reflection rotation-minimizing frame; no Unreal
//! Engine source or derived code.

use prism_hair_gpu::frames::{GpuStrandFrame, GpuStrandFrames};
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::frames::build_strand_frames;
use prism_render_architecture::hair::interpolation::Vec3;

/// Converts a flat `[f32; 3]` control point to the golden's `Vec3`.
fn to_vec3(p: [f32; 3]) -> Vec3 {
    Vec3::new(p[0], p[1], p[2])
}

/// Dot product of two `[f32; 3]` axes.
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Cross product of two `[f32; 3]` axes.
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Asserts a single component matches within the documented fma tolerance.
fn assert_close(got: f32, expected: f32, label: &str) {
    let abs_diff = (got - expected).abs();
    let rel_diff = abs_diff / expected.abs().max(1e-6);
    assert!(
        abs_diff < 1e-4 || rel_diff < 1e-3,
        "{label}: gpu {got}, cpu {expected} (abs {abs_diff}, rel {rel_diff})"
    );
}

/// Asserts every `GPU` frame of `strand` equals the `CPU` golden per component.
fn assert_strand_parity(strand: &[[f32; 3]], gpu: &[GpuStrandFrame]) {
    let points: Vec<Vec3> = strand.iter().map(|p| to_vec3(*p)).collect();
    let cpu = build_strand_frames(&points);
    assert_eq!(gpu.len(), cpu.len(), "one frame per control point");
    for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        assert_close(g.tangent[0], c.tangent.x, &format!("frame {i} tangent.x"));
        assert_close(g.tangent[1], c.tangent.y, &format!("frame {i} tangent.y"));
        assert_close(g.tangent[2], c.tangent.z, &format!("frame {i} tangent.z"));
        assert_close(g.normal[0], c.normal.x, &format!("frame {i} normal.x"));
        assert_close(g.normal[1], c.normal.y, &format!("frame {i} normal.y"));
        assert_close(g.normal[2], c.normal.z, &format!("frame {i} normal.z"));
        assert_close(
            g.bitangent[0],
            c.bitangent.x,
            &format!("frame {i} bitangent.x"),
        );
        assert_close(
            g.bitangent[1],
            c.bitangent.y,
            &format!("frame {i} bitangent.y"),
        );
        assert_close(
            g.bitangent[2],
            c.bitangent.z,
            &format!("frame {i} bitangent.z"),
        );
    }
}

/// Asserts each frame is an orthonormal right-handed basis.
fn assert_orthonormal(gpu: &[GpuStrandFrame]) {
    for (i, f) in gpu.iter().enumerate() {
        let tt = dot(f.tangent, f.tangent);
        let nn = dot(f.normal, f.normal);
        let bb = dot(f.bitangent, f.bitangent);
        assert!((tt - 1.0).abs() < 1e-3, "frame {i} tangent not unit: {tt}");
        assert!((nn - 1.0).abs() < 1e-3, "frame {i} normal not unit: {nn}");
        assert!(
            (bb - 1.0).abs() < 1e-3,
            "frame {i} bitangent not unit: {bb}"
        );
        assert!(dot(f.tangent, f.normal).abs() < 1e-3, "frame {i} t.n != 0");
        assert!(
            dot(f.tangent, f.bitangent).abs() < 1e-3,
            "frame {i} t.b != 0"
        );
        assert!(
            dot(f.normal, f.bitangent).abs() < 1e-3,
            "frame {i} n.b != 0"
        );
        // Right-handed: bitangent == tangent x normal.
        let expected_b = cross(f.tangent, f.normal);
        assert!(
            (f.bitangent[0] - expected_b[0]).abs() < 1e-3
                && (f.bitangent[1] - expected_b[1]).abs() < 1e-3
                && (f.bitangent[2] - expected_b[2]).abs() < 1e-3,
            "frame {i} not right-handed"
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_smooth_curves_match_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping frames parity: no wgpu adapter on this host");
        return;
    };

    // Two smooth polynomial space curves (genuine 3-D curvature and torsion, so
    // the transported frame actually turns) with clearly non-zero segments, so
    // every tangent/reflection stays well clear of the degenerate guards. Built
    // from arithmetic only — no `f32::sin`/`cos`, which the workspace disallows
    // for determinism.
    let mut curve_a: Vec<[f32; 3]> = Vec::new();
    for k in 0..40u32 {
        let t = k as f32 * 0.3;
        curve_a.push([t, 0.5 * t - 0.08 * t * t, 0.03 * t * t - 0.004 * t * t * t]);
    }
    let mut curve_b: Vec<[f32; 3]> = Vec::new();
    for k in 0..32u32 {
        let t = k as f32 * 0.25;
        curve_b.push([
            0.3 * t + 0.02 * t * t,
            -0.4 * t + 0.05 * t * t,
            0.6 * t - 0.03 * t * t + 0.002 * t * t * t,
        ]);
    }

    let strands: [&[[f32; 3]]; 2] = [&curve_a, &curve_b];
    let out = GpuStrandFrames::new(&ctx).eval(&ctx, &strands);
    assert_eq!(out.len(), 2, "one frame vec per strand");

    assert_strand_parity(&curve_a, &out[0]);
    assert_strand_parity(&curve_b, &out[1]);
    assert_orthonormal(&out[0]);
    assert_orthonormal(&out[1]);

    // Non-degenerate shape: on a curved strand the normal must actually turn,
    // so a constant-returning kernel could not pass.
    let first_n = out[0][0].normal;
    let last_n = out[0][out[0].len() - 1].normal;
    let turned = (first_n[0] - last_n[0]).abs()
        + (first_n[1] - last_n[1]).abs()
        + (first_n[2] - last_n[2]).abs();
    assert!(
        turned > 1e-3,
        "frame normal should vary along a curved strand"
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_degenerate_strands_match_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping frames parity: no wgpu adapter on this host");
        return;
    };

    // Single point: one frame around the fallback tangent.
    let single: Vec<[f32; 3]> = vec![[2.0, -1.0, 0.5]];
    // Coincident interior points (exact equality on both sides -> fallback
    // branch bit-for-bit) mixed with real segments.
    let coincident: Vec<[f32; 3]> = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0], // duplicate of the previous point
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
    ];
    // A short straight segment: constant tangent, constant transported normal.
    let straight: Vec<[f32; 3]> = vec![
        [-1.0, 0.5, 0.0],
        [0.0, 0.5, 0.0],
        [1.0, 0.5, 0.0],
        [2.0, 0.5, 0.0],
    ];

    let strands: [&[[f32; 3]]; 3] = [&single, &coincident, &straight];
    let out = GpuStrandFrames::new(&ctx).eval(&ctx, &strands);
    assert_eq!(out.len(), 3, "one frame vec per strand");

    assert_strand_parity(&single, &out[0]);
    assert_strand_parity(&coincident, &out[1]);
    assert_strand_parity(&straight, &out[2]);

    assert_eq!(out[0].len(), 1, "single point yields one frame");
    assert_close(out[0][0].tangent[0], 0.0, "single fallback tangent.x");
    assert_close(out[0][0].tangent[1], 1.0, "single fallback tangent.y");
    assert_close(out[0][0].tangent[2], 0.0, "single fallback tangent.z");

    assert_orthonormal(&out[0]);
    assert_orthonormal(&out[2]);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_empty_batches_yield_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping frames parity: no wgpu adapter on this host");
        return;
    };
    let kernel = GpuStrandFrames::new(&ctx);

    // No strands at all: empty outer vec, no dispatch.
    let none: [&[[f32; 3]]; 0] = [];
    assert!(kernel.eval(&ctx, &none).is_empty(), "no strands -> empty");

    // Every strand empty: one empty inner vec per strand, still no dispatch
    // (storage buffers cannot be zero-sized).
    let empty0: [[f32; 3]; 0] = [];
    let empty1: [[f32; 3]; 0] = [];
    let all_empty: [&[[f32; 3]]; 2] = [&empty0, &empty1];
    let out = kernel.eval(&ctx, &all_empty);
    assert_eq!(out.len(), 2, "one inner vec per (empty) strand");
    assert!(
        out[0].is_empty() && out[1].is_empty(),
        "empty strands -> empty"
    );
}
