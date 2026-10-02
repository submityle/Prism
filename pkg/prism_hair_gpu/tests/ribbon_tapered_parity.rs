//! Real-device parity for the tapered strand-to-ribbon (`Cards` LOD proxy) twin:
//! [`GpuRibbonTapered`] must reproduce the `CPU` golden
//! [`build_ribbon_tapered`](prism_render_architecture::hair::ribbon::build_ribbon_tapered)
//! for every ribbon vertex of every strand — the authored root→tip radius taper
//! (`radius_at(i / (n - 1))`), the `±radius` edge offsets along the
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
//! expression and diverge only through legal fused-multiply-add contraction (the
//! radius lerp and the edge offset are prime candidates) and a possible
//! reassociation of the arc-length sum. Each position, tangent and UV component
//! is asserted to within `abs_diff < 1e-4` or `rel_diff < 1e-3` — tight enough
//! to fail a genuinely wrong port (a swapped edge, a missing taper, a wrong
//! `v`), loose enough to admit fma contraction. Indices are a pure function of
//! the point count and carry no float error, so they are asserted for exact
//! equality. The smooth sweeps also assert the taper is live (the tip half-width
//! differs from the root, and the width shrinks monotonically when the groom
//! narrows) so a degenerate kernel that ignored the taper could not pass. The
//! zero-length case uses exactly coincident points so both sides take the
//! even-spacing fallback branch bit-for-bit.
//!
//! Provenance: standard view-independent ribbon/card meshing with an authored
//! linear radius taper; no Unreal Engine source or derived code.

use prism_hair_gpu::ribbon::GpuRibbonMesh;
use prism_hair_gpu::ribbon_tapered::{GpuRibbonTapered, TaperedRibbonStrandInput};
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::frames::build_strand_frames;
use prism_render_architecture::hair::groom_import::StrandAttributes;
use prism_render_architecture::hair::interpolation::Vec3;
use prism_render_architecture::hair::ribbon::{build_ribbon_tapered, RibbonMesh};

/// Converts a flat `[f32; 3]` control point to the golden's `Vec3`.
fn to_vec3(p: [f32; 3]) -> Vec3 {
    Vec3::new(p[0], p[1], p[2])
}

/// Euclidean width between a ribbon's left/right edge vertices.
fn edge_width(l: [f32; 3], r: [f32; 3]) -> f32 {
    let dx = r[0] - l[0];
    let dy = r[1] - l[1];
    let dz = r[2] - l[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
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

/// Builds the `CPU` golden mesh from raw control points and authored radii,
/// deriving the frames exactly as the render pipeline does before meshing.
fn cpu_golden(points: &[[f32; 3]], attrs: &StrandAttributes) -> RibbonMesh {
    let pts: Vec<Vec3> = points.iter().map(|p| to_vec3(*p)).collect();
    let frames = build_strand_frames(&pts);
    build_ribbon_tapered(&pts, &frames, attrs)
}

/// Derives the per-point tangent/bitangent axes the way the render pipeline does
/// before meshing; borrows must outlive the returned inputs, so the arrays are
/// returned to the caller to own.
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
fn assert_ribbon_parity(gpu: &GpuRibbonMesh, cpu: &RibbonMesh) {
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
    assert_eq!(
        gpu.indices, cpu.indices,
        "indices are exact (no float error)"
    );
}

/// A smooth bent strand with four control points.
fn bent_strand() -> Vec<[f32; 3]> {
    vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.2, 0.0],
        [2.0, 0.6, 0.1],
        [3.0, 1.2, 0.3],
    ]
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_tapered_ribbon_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping tapered ribbon parity: no wgpu adapter on this host");
        return;
    };

    let points = bent_strand();
    // A groom that narrows from a thick root to a thin tip.
    let attrs = StrandAttributes::new(0.12, 0.03, [0.0, 0.0], 7);
    let (tan, bit) = frame_axes(&points);
    let inputs = [TaperedRibbonStrandInput {
        points: &points,
        tangents: &tan,
        bitangents: &bit,
        root_radius: attrs.root_radius,
        tip_radius: attrs.tip_radius,
    }];

    let out = GpuRibbonTapered::new(&ctx).eval(&ctx, &inputs);
    assert_eq!(out.len(), 1, "one mesh per strand");
    let cpu = cpu_golden(&points, &attrs);
    assert_ribbon_parity(&out[0], &cpu);

    // The taper must be live: the root ribbon is wider than the tip ribbon.
    let n = points.len();
    let root_width = {
        let l = out[0].positions[0];
        let r = out[0].positions[1];
        edge_width(l, r)
    };
    let tip_width = {
        let l = out[0].positions[2 * (n - 1)];
        let r = out[0].positions[2 * (n - 1) + 1];
        edge_width(l, r)
    };
    assert_close(root_width, 2.0 * 0.12, "root width = 2*root_radius");
    assert_close(tip_width, 2.0 * 0.03, "tip width = 2*tip_radius");
    assert!(
        root_width > tip_width + 1e-3,
        "taper must narrow root→tip: root {root_width}, tip {tip_width}"
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_uniform_radii_match_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping tapered ribbon parity: no wgpu adapter on this host");
        return;
    };

    // Equal root/tip radii degenerate to a uniform-width ribbon; still exercises
    // the taper lerp (which collapses to a constant) and must match the golden.
    let points = bent_strand();
    let attrs = StrandAttributes::new(0.08, 0.08, [0.25, 0.5], 3);
    let (tan, bit) = frame_axes(&points);
    let inputs = [TaperedRibbonStrandInput {
        points: &points,
        tangents: &tan,
        bitangents: &bit,
        root_radius: attrs.root_radius,
        tip_radius: attrs.tip_radius,
    }];

    let out = GpuRibbonTapered::new(&ctx).eval(&ctx, &inputs);
    assert_eq!(out.len(), 1, "one mesh per strand");
    assert_ribbon_parity(&out[0], &cpu_golden(&points, &attrs));

    // Every control point's width is the same constant 2*radius.
    for i in 0..points.len() {
        let l = out[0].positions[2 * i];
        let r = out[0].positions[2 * i + 1];
        let w = edge_width(l, r);
        assert_close(w, 2.0 * 0.08, &format!("uniform width at point {i}"));
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_inverted_taper_widens_root_to_tip() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping tapered ribbon parity: no wgpu adapter on this host");
        return;
    };

    // A thin root growing to a thick tip (reverse taper): verifies the lerp
    // direction is correct, not just that two widths differ.
    let points = bent_strand();
    let attrs = StrandAttributes::new(0.02, 0.1, [0.0, 0.0], 11);
    let (tan, bit) = frame_axes(&points);
    let inputs = [TaperedRibbonStrandInput {
        points: &points,
        tangents: &tan,
        bitangents: &bit,
        root_radius: attrs.root_radius,
        tip_radius: attrs.tip_radius,
    }];

    let out = GpuRibbonTapered::new(&ctx).eval(&ctx, &inputs);
    assert_ribbon_parity(&out[0], &cpu_golden(&points, &attrs));

    let n = points.len();
    let width_at = |i: usize| {
        let l = out[0].positions[2 * i];
        let r = out[0].positions[2 * i + 1];
        edge_width(l, r)
    };
    assert!(
        width_at(n - 1) > width_at(0) + 1e-3,
        "reverse taper must widen root→tip"
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_zero_length_strand_falls_back_to_even_v() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping tapered ribbon parity: no wgpu adapter on this host");
        return;
    };

    // Exactly coincident control points (zero total arc length): both sides take
    // the even-parameter-spacing fallback bit-for-bit, so v is 0, 0.5, 1.
    let coincident: Vec<[f32; 3]> = vec![[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]];
    let attrs = StrandAttributes::new(0.1, 0.05, [0.0, 0.0], 1);
    let (tan, bit) = frame_axes(&coincident);
    let inputs = [TaperedRibbonStrandInput {
        points: &coincident,
        tangents: &tan,
        bitangents: &bit,
        root_radius: attrs.root_radius,
        tip_radius: attrs.tip_radius,
    }];

    let out = GpuRibbonTapered::new(&ctx).eval(&ctx, &inputs);
    assert_eq!(out.len(), 1, "one mesh per strand");
    assert_ribbon_parity(&out[0], &cpu_golden(&coincident, &attrs));

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
fn gpu_mixed_batch_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping tapered ribbon parity: no wgpu adapter on this host");
        return;
    };

    // Several strands of differing lengths and tapers in one dispatch: each must
    // reconstruct from its own disjoint vertex slice and match its golden.
    let a = bent_strand();
    let a_attrs = StrandAttributes::new(0.1, 0.02, [0.0, 0.0], 2);
    let (a_tan, a_bit) = frame_axes(&a);

    let b: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let b_attrs = StrandAttributes::new(0.05, 0.05, [0.0, 0.0], 4);
    let (b_tan, b_bit) = frame_axes(&b);

    let c: Vec<[f32; 3]> = vec![
        [5.0, 0.0, 0.0],
        [5.0, 0.0, 1.0],
        [5.0, 0.0, 2.0],
        [5.0, 0.0, 3.0],
        [5.0, 0.0, 4.0],
    ];
    let c_attrs = StrandAttributes::new(0.03, 0.09, [0.0, 0.0], 6);
    let (c_tan, c_bit) = frame_axes(&c);

    let inputs = [
        TaperedRibbonStrandInput {
            points: &a,
            tangents: &a_tan,
            bitangents: &a_bit,
            root_radius: a_attrs.root_radius,
            tip_radius: a_attrs.tip_radius,
        },
        TaperedRibbonStrandInput {
            points: &b,
            tangents: &b_tan,
            bitangents: &b_bit,
            root_radius: b_attrs.root_radius,
            tip_radius: b_attrs.tip_radius,
        },
        TaperedRibbonStrandInput {
            points: &c,
            tangents: &c_tan,
            bitangents: &c_bit,
            root_radius: c_attrs.root_radius,
            tip_radius: c_attrs.tip_radius,
        },
    ];

    let out = GpuRibbonTapered::new(&ctx).eval(&ctx, &inputs);
    assert_eq!(out.len(), 3, "one mesh per strand");
    assert_ribbon_parity(&out[0], &cpu_golden(&a, &a_attrs));
    assert_ribbon_parity(&out[1], &cpu_golden(&b, &b_attrs));
    assert_ribbon_parity(&out[2], &cpu_golden(&c, &c_attrs));
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_degenerate_strands_yield_empty_meshes() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping tapered ribbon parity: no wgpu adapter on this host");
        return;
    };

    let good = bent_strand();
    let good_attrs = StrandAttributes::new(0.08, 0.04, [0.0, 0.0], 9);
    let (good_tan, good_bit) = frame_axes(&good);

    // Single point: below the two-point minimum.
    let single: Vec<[f32; 3]> = vec![[5.0, 5.0, 5.0]];
    let single_attrs = StrandAttributes::new(0.1, 0.1, [0.0, 0.0], 0);
    let (single_tan, single_bit) = frame_axes(&single);

    // Mismatched: two points but only one bitangent.
    let mismatched_pts: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]];
    let mismatched_attrs = StrandAttributes::new(0.1, 0.1, [0.0, 0.0], 0);
    let (mm_tan, _mm_bit) = frame_axes(&mismatched_pts);
    let mm_bit_short: Vec<[f32; 3]> = vec![[0.0, 1.0, 0.0]];

    let inputs = [
        TaperedRibbonStrandInput {
            points: &good,
            tangents: &good_tan,
            bitangents: &good_bit,
            root_radius: good_attrs.root_radius,
            tip_radius: good_attrs.tip_radius,
        },
        TaperedRibbonStrandInput {
            points: &single,
            tangents: &single_tan,
            bitangents: &single_bit,
            root_radius: single_attrs.root_radius,
            tip_radius: single_attrs.tip_radius,
        },
        TaperedRibbonStrandInput {
            points: &mismatched_pts,
            tangents: &mm_tan,
            bitangents: &mm_bit_short,
            root_radius: mismatched_attrs.root_radius,
            tip_radius: mismatched_attrs.tip_radius,
        },
    ];

    let out = GpuRibbonTapered::new(&ctx).eval(&ctx, &inputs);
    assert_eq!(out.len(), 3, "one mesh per strand, valid or not");

    assert_ribbon_parity(&out[0], &cpu_golden(&good, &good_attrs));
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
        eprintln!("skipping tapered ribbon parity: no wgpu adapter on this host");
        return;
    };
    let kernel = GpuRibbonTapered::new(&ctx);

    // No strands at all: empty outer vec, no dispatch.
    let none: [TaperedRibbonStrandInput; 0] = [];
    assert!(kernel.eval(&ctx, &none).is_empty(), "no strands -> empty");

    // Every strand invalid (all below the two-point minimum): one empty mesh
    // per strand, still no dispatch (storage buffers cannot be zero-sized).
    let empty_pts: Vec<[f32; 3]> = Vec::new();
    let empty_axis: Vec<[f32; 3]> = Vec::new();
    let all_invalid = [
        TaperedRibbonStrandInput {
            points: &empty_pts,
            tangents: &empty_axis,
            bitangents: &empty_axis,
            root_radius: 0.1,
            tip_radius: 0.05,
        },
        TaperedRibbonStrandInput {
            points: &empty_pts,
            tangents: &empty_axis,
            bitangents: &empty_axis,
            root_radius: 0.1,
            tip_radius: 0.05,
        },
    ];
    let out = kernel.eval(&ctx, &all_invalid);
    assert_eq!(out.len(), 2, "one mesh per (invalid) strand");
    assert!(
        out[0].positions.is_empty() && out[1].positions.is_empty(),
        "invalid strands -> empty meshes"
    );
}
