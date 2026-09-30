//! Real-device parity for the strand-to-ribbon (`Cards` LOD proxy) twin:
//! [`GpuRibbon`] must reproduce the `CPU` golden
//! [`build_ribbon`](prism_render_architecture::hair::ribbon::build_ribbon) for
//! every ribbon vertex of every strand — the `±radius` edge offsets along the
//! rotation-minimizing bitangent, the arc-length `v` coordinate (including the
//! zero-length fallback), the carried tangent, and the deterministic triangle
//! winding — while reproducing the reference's guards (a strand with fewer than
//! two points, or with mismatched attribute lengths, yields an empty mesh).
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The meshing is closed-form geometry (the reference restricts itself to
//! `sqrt` for the arc length), so the `CPU` and `GPU` evaluate the same
//! expression and diverge only through legal fused-multiply-add contraction and
//! a possible reassociation of the arc-length sum. Each position, tangent and
//! UV component is asserted to within `abs_diff < 1e-4` or `rel_diff < 1e-3` —
//! tight enough to fail a genuinely wrong port (a swapped edge, a missing
//! radius scale, a wrong `v`), loose enough to admit fma contraction. Indices
//! are a pure function of the point count and carry no float error, so they are
//! asserted for exact equality. The smooth sweeps also assert ribbon geometry
//! (left/right edges symmetric about the centerline, width `≈ 2·radius`,
//! monotone `v` root→tip, `u ∈ {0, 1}`) so a degenerate kernel could not pass.
//! The zero-length case uses exactly coincident points so both sides take the
//! even-spacing fallback branch bit-for-bit.
//!
//! Provenance: standard view-independent ribbon/card meshing; no Unreal Engine
//! source or derived code.

use prism_hair_gpu::ribbon::{GpuRibbon, RibbonStrandInput};
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::frames::build_strand_frames;
use prism_render_architecture::hair::interpolation::Vec3;
use prism_render_architecture::hair::ribbon::{build_ribbon, RibbonMesh};

/// Converts a flat `[f32; 3]` control point to the golden's `Vec3`.
fn to_vec3(p: [f32; 3]) -> Vec3 {
    Vec3::new(p[0], p[1], p[2])
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

/// Builds the `CPU` golden mesh from raw control points, deriving the frames
/// exactly as the render pipeline does before meshing.
fn cpu_golden(points: &[[f32; 3]], radii: &[f32]) -> RibbonMesh {
    let pts: Vec<Vec3> = points.iter().map(|p| to_vec3(*p)).collect();
    let frames = build_strand_frames(&pts);
    build_ribbon(&pts, &frames, radii)
}

/// Builds a [`RibbonStrandInput`] view over the derived tangents/bitangents and
/// the given radii; borrows must outlive the returned inputs, so the frame
/// arrays are returned alongside.
fn frame_axes(points: &[[f32; 3]]) -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
    let pts: Vec<Vec3> = points.iter().map(|p| to_vec3(*p)).collect();
    let frames = build_strand_frames(&pts);
    let tangents = frames
        .iter()
        .map(|f| [f.tangent.x, f.tangent.y, f.tangent.z])
        .collect();
    let bitangents = frames
        .iter()
        .map(|f| [f.bitangent.x, f.bitangent.y, f.bitangent.z])
        .collect();
    (tangents, bitangents)
}

/// Asserts the `GPU` ribbon for one strand matches the `CPU` golden per vertex.
fn assert_ribbon_parity(gpu: &prism_hair_gpu::ribbon::GpuRibbonMesh, cpu: &RibbonMesh) {
    assert_eq!(
        gpu.positions.len(),
        cpu.positions.len(),
        "vertex count must match golden"
    );
    assert_eq!(gpu.tangents.len(), cpu.tangents.len(), "tangent count");
    assert_eq!(gpu.uvs.len(), cpu.uvs.len(), "uv count");
    for (i, (g, c)) in gpu.positions.iter().zip(cpu.positions.iter()).enumerate() {
        assert_close(g[0], c.x, &format!("vertex {i} position.x"));
        assert_close(g[1], c.y, &format!("vertex {i} position.y"));
        assert_close(g[2], c.z, &format!("vertex {i} position.z"));
    }
    for (i, (g, c)) in gpu.tangents.iter().zip(cpu.tangents.iter()).enumerate() {
        assert_close(g[0], c.x, &format!("vertex {i} tangent.x"));
        assert_close(g[1], c.y, &format!("vertex {i} tangent.y"));
        assert_close(g[2], c.z, &format!("vertex {i} tangent.z"));
    }
    for (i, (g, c)) in gpu.uvs.iter().zip(cpu.uvs.iter()).enumerate() {
        assert_close(g[0], c[0], &format!("vertex {i} u"));
        assert_close(g[1], c[1], &format!("vertex {i} v"));
    }
    // Indices are a pure function of the point count: exact equality.
    assert_eq!(
        gpu.indices, cpu.indices,
        "triangle winding must match golden"
    );
}

/// Asserts the intrinsic ribbon geometry: left/right edges are symmetric about
/// the centerline, the width is `≈ 2·radius`, `v` is monotone non-decreasing
/// root→tip and `u` alternates `0`/`1`.
fn assert_ribbon_shape(
    mesh: &prism_hair_gpu::ribbon::GpuRibbonMesh,
    points: &[[f32; 3]],
    radii: &[f32],
) {
    let n = points.len();
    let mut prev_v = -1.0_f32;
    for i in 0..n {
        let left = mesh.positions[2 * i];
        let right = mesh.positions[2 * i + 1];
        // Midpoint of the two edges is the centerline control point.
        let mid = [
            0.5 * (left[0] + right[0]),
            0.5 * (left[1] + right[1]),
            0.5 * (left[2] + right[2]),
        ];
        assert_close(mid[0], points[i][0], &format!("point {i} centerline.x"));
        assert_close(mid[1], points[i][1], &format!("point {i} centerline.y"));
        assert_close(mid[2], points[i][2], &format!("point {i} centerline.z"));
        // Width equals 2 * radius.
        let dx = right[0] - left[0];
        let dy = right[1] - left[1];
        let dz = right[2] - left[2];
        let width = (dx * dx + dy * dy + dz * dz).sqrt();
        assert_close(width, 2.0 * radii[i], &format!("point {i} width"));
        // u is 0 on the left, 1 on the right.
        assert_close(mesh.uvs[2 * i][0], 0.0, &format!("point {i} left u"));
        assert_close(mesh.uvs[2 * i + 1][0], 1.0, &format!("point {i} right u"));
        // v is monotone non-decreasing and shared by both edges.
        let v = mesh.uvs[2 * i][1];
        assert_close(mesh.uvs[2 * i + 1][1], v, &format!("point {i} v shared"));
        assert!(
            v >= prev_v - 1e-4,
            "point {i} v should be monotone non-decreasing: {v} < {prev_v}"
        );
        prev_v = v;
    }
    // Root v is 0, tip v is 1 on a non-degenerate strand.
    assert_close(mesh.uvs[0][1], 0.0, "root v");
    assert_close(mesh.uvs[2 * (n - 1)][1], 1.0, "tip v");
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_smooth_ribbons_match_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping ribbon parity: no wgpu adapter on this host");
        return;
    };

    // Two smooth polynomial space curves (genuine 3-D curvature so the frame
    // bitangent actually turns) with clearly non-zero segments. Built from
    // arithmetic only — no `f32::sin`/`cos`, which the workspace disallows.
    let mut curve_a: Vec<[f32; 3]> = Vec::new();
    let mut radii_a: Vec<f32> = Vec::new();
    for k in 0..40u32 {
        let t = k as f32 * 0.3;
        curve_a.push([t, 0.5 * t - 0.08 * t * t, 0.03 * t * t - 0.004 * t * t * t]);
        // Taper the radius root -> tip so widths vary along the strand.
        radii_a.push(0.05 + 0.002 * (39 - k as i32) as f32);
    }
    let mut curve_b: Vec<[f32; 3]> = Vec::new();
    let mut radii_b: Vec<f32> = Vec::new();
    for k in 0..32u32 {
        let t = k as f32 * 0.25;
        curve_b.push([
            0.3 * t + 0.02 * t * t,
            -0.4 * t + 0.05 * t * t,
            0.6 * t - 0.03 * t * t + 0.002 * t * t * t,
        ]);
        radii_b.push(0.03 + 0.0015 * k as f32);
    }

    let (tan_a, bit_a) = frame_axes(&curve_a);
    let (tan_b, bit_b) = frame_axes(&curve_b);
    let inputs = [
        RibbonStrandInput {
            points: &curve_a,
            tangents: &tan_a,
            bitangents: &bit_a,
            radii: &radii_a,
        },
        RibbonStrandInput {
            points: &curve_b,
            tangents: &tan_b,
            bitangents: &bit_b,
            radii: &radii_b,
        },
    ];

    let out = GpuRibbon::new(&ctx).eval(&ctx, &inputs);
    assert_eq!(out.len(), 2, "one mesh per strand");

    assert_ribbon_parity(&out[0], &cpu_golden(&curve_a, &radii_a));
    assert_ribbon_parity(&out[1], &cpu_golden(&curve_b, &radii_b));
    assert_ribbon_shape(&out[0], &curve_a, &radii_a);
    assert_ribbon_shape(&out[1], &curve_b, &radii_b);

    // Non-degenerate shape: on a curved strand the ribbon must actually bend,
    // so the tangent at root and tip differ — a constant-returning kernel could
    // not pass.
    let first_t = out[0].tangents[0];
    let last_t = out[0].tangents[out[0].tangents.len() - 1];
    let turned = (first_t[0] - last_t[0]).abs()
        + (first_t[1] - last_t[1]).abs()
        + (first_t[2] - last_t[2]).abs();
    assert!(
        turned > 1e-3,
        "ribbon tangent should vary along a curved strand"
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_zero_length_strand_falls_back_to_even_v() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping ribbon parity: no wgpu adapter on this host");
        return;
    };

    // Exactly coincident control points (zero total arc length): both sides
    // take the even-parameter-spacing fallback bit-for-bit, so v is 0, 0.5, 1.
    let coincident: Vec<[f32; 3]> = vec![[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]];
    let radii: Vec<f32> = vec![0.1, 0.1, 0.1];
    let (tan, bit) = frame_axes(&coincident);
    let inputs = [RibbonStrandInput {
        points: &coincident,
        tangents: &tan,
        bitangents: &bit,
        radii: &radii,
    }];

    let out = GpuRibbon::new(&ctx).eval(&ctx, &inputs);
    assert_eq!(out.len(), 1, "one mesh per strand");
    assert_ribbon_parity(&out[0], &cpu_golden(&coincident, &radii));

    // Even spacing: v runs 0, 0.5, 1 across the three coincident points.
    assert_close(out[0].uvs[0][1], 0.0, "coincident v[0]");
    assert_close(out[0].uvs[2][1], 0.5, "coincident v[1]");
    assert_close(out[0].uvs[4][1], 1.0, "coincident v[2]");
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_degenerate_strands_yield_empty_meshes() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping ribbon parity: no wgpu adapter on this host");
        return;
    };

    // A single valid strand plus two invalid strands (one too short, one with a
    // mismatched attribute length) all in one batch: the valid strand meshes,
    // the invalid ones reconstruct as empty meshes, matching the golden guards.
    let good: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.5, 0.0]];
    let good_radii: Vec<f32> = vec![0.08, 0.06, 0.04];
    let (good_tan, good_bit) = frame_axes(&good);

    // Single point: below the two-point minimum.
    let single: Vec<[f32; 3]> = vec![[5.0, 5.0, 5.0]];
    let single_radii: Vec<f32> = vec![0.1];
    let (single_tan, single_bit) = frame_axes(&single);

    // Mismatched: two points but only one radius.
    let mismatched_pts: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]];
    let mismatched_radii: Vec<f32> = vec![0.1];
    let (mm_tan, mm_bit) = frame_axes(&mismatched_pts);

    let inputs = [
        RibbonStrandInput {
            points: &good,
            tangents: &good_tan,
            bitangents: &good_bit,
            radii: &good_radii,
        },
        RibbonStrandInput {
            points: &single,
            tangents: &single_tan,
            bitangents: &single_bit,
            radii: &single_radii,
        },
        RibbonStrandInput {
            points: &mismatched_pts,
            tangents: &mm_tan,
            bitangents: &mm_bit,
            radii: &mismatched_radii,
        },
    ];

    let out = GpuRibbon::new(&ctx).eval(&ctx, &inputs);
    assert_eq!(out.len(), 3, "one mesh per strand, valid or not");

    assert_ribbon_parity(&out[0], &cpu_golden(&good, &good_radii));
    assert!(!out[0].positions.is_empty(), "valid strand should mesh");
    assert!(out[1].positions.is_empty(), "single-point strand -> empty");
    assert!(
        out[1].indices.is_empty(),
        "single-point strand -> no indices"
    );
    assert!(out[2].positions.is_empty(), "mismatched strand -> empty");
    assert!(out[2].indices.is_empty(), "mismatched strand -> no indices");
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_empty_batches_yield_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping ribbon parity: no wgpu adapter on this host");
        return;
    };
    let kernel = GpuRibbon::new(&ctx);

    // No strands at all: empty outer vec, no dispatch.
    let none: [RibbonStrandInput; 0] = [];
    assert!(kernel.eval(&ctx, &none).is_empty(), "no strands -> empty");

    // Every strand invalid (all below the two-point minimum): one empty mesh
    // per strand, still no dispatch (storage buffers cannot be zero-sized).
    let empty_pts: Vec<[f32; 3]> = Vec::new();
    let empty_tan: Vec<[f32; 3]> = Vec::new();
    let empty_bit: Vec<[f32; 3]> = Vec::new();
    let empty_radii: Vec<f32> = Vec::new();
    let all_invalid = [
        RibbonStrandInput {
            points: &empty_pts,
            tangents: &empty_tan,
            bitangents: &empty_bit,
            radii: &empty_radii,
        },
        RibbonStrandInput {
            points: &empty_pts,
            tangents: &empty_tan,
            bitangents: &empty_bit,
            radii: &empty_radii,
        },
    ];
    let out = kernel.eval(&ctx, &all_invalid);
    assert_eq!(out.len(), 2, "one mesh per (invalid) strand");
    assert!(
        out[0].positions.is_empty() && out[1].positions.is_empty(),
        "invalid strands -> empty meshes"
    );
}
