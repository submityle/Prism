//! Real-device parity for the tapered strand-to-shell (`Mesh` LOD proxy) twin:
//! [`GpuMeshShellTapered`] must reproduce the `CPU` golden
//! [`build_shell_tapered`](prism_render_architecture::hair::mesh_shell::build_shell_tapered)
//! for every shell vertex of every strand. Unlike the plain shell twin, the
//! cross-section is not uniform: at ring `i` the square section's half-width and
//! half-thickness are both
//! [`StrandAttributes::radius_at`](prism_render_architecture::hair::groom_import::StrandAttributes::radius_at)
//! evaluated at the ring's normalized arc-length position `t = i / (n - 1)`, a
//! clamped linear ramp `root_radius + (tip_radius - root_radius) * t`. The test
//! checks the four tapering rectangle-corner positions, the diagonal outward
//! corner normals (with the `normalize_or` fallback), the quarter-perimeter `u`
//! and arc-length `v` UVs (including the zero-length fallback), and the
//! deterministic triangle winding, while reproducing the reference's guards (a
//! strand with fewer than two points, or with mismatched attribute lengths,
//! yields an empty mesh).
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The meshing is closed-form geometry (the reference restricts itself to
//! `sqrt` for the arc length and the corner-normal normalize, plus a linear
//! radius ramp), so the `CPU` and `GPU` evaluate the same expression and diverge
//! only through legal fused-multiply-add contraction and a possible
//! reassociation of the arc-length sum. Each position, normal and UV component
//! is asserted to within `abs_diff < 1e-4` or `rel_diff < 1e-3` — tight enough
//! to fail a genuinely wrong port (a swapped corner, a missing taper, a wrong
//! `v`), loose enough to admit fma contraction. Indices are a pure function of
//! the point count and carry no float error, so they are asserted for exact
//! equality. The smooth sweeps also assert the taper directly (the ring
//! half-extent recovered from opposite corners equals `radius_at(t)` at the
//! root, the tip and an interior ring, so a plain non-tapering kernel could not
//! pass), unit normals, monotone `v` root→tip, and `u ∈ {0, 0.25, 0.5, 0.75}`.
//! The zero-length case uses exactly coincident points so both sides take the
//! even-spacing fallback branch bit-for-bit.
//!
//! Provenance: standard rectangular shell-tube meshing with an authored linear
//! radius taper; no Unreal Engine source or derived code.

use prism_hair_gpu::mesh_shell::GpuShellMesh;
use prism_hair_gpu::mesh_shell_tapered::{GpuMeshShellTapered, TaperedShellStrandInput};
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::frames::build_strand_frames;
use prism_render_architecture::hair::groom_import::StrandAttributes;
use prism_render_architecture::hair::interpolation::Vec3;
use prism_render_architecture::hair::mesh_shell::{build_shell_tapered, ShellMesh};

/// Converts a flat `[f32; 3]` control point to the golden's `Vec3`.
fn to_vec3(p: [f32; 3]) -> Vec3 {
    Vec3::new(p[0], p[1], p[2])
}

/// Authored attributes carrying the taper the section narrows between. Built as
/// a struct literal (not `StrandAttributes::new`, which clamps) so the exact
/// authored radii reach both the golden and the `GPU` input; the tests only use
/// non-negative radii, so the two constructions agree.
fn attrs(root_radius: f32, tip_radius: f32) -> StrandAttributes {
    StrandAttributes {
        root_radius,
        tip_radius,
        root_uv: [0.0, 0.0],
        seed: 0,
    }
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
fn cpu_golden(points: &[[f32; 3]], root_radius: f32, tip_radius: f32) -> ShellMesh {
    let pts: Vec<Vec3> = points.iter().map(|p| to_vec3(*p)).collect();
    let frames = build_strand_frames(&pts);
    build_shell_tapered(&pts, &frames, &attrs(root_radius, tip_radius))
}

/// Derives the per-point bitangent and normal frame axes the shell builder
/// consumes; borrows must outlive the returned inputs, so the arrays are
/// returned to the caller to hold.
fn frame_axes(points: &[[f32; 3]]) -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
    let pts: Vec<Vec3> = points.iter().map(|p| to_vec3(*p)).collect();
    let frames = build_strand_frames(&pts);
    let bitangents = frames
        .iter()
        .map(|f| [f.bitangent.x, f.bitangent.y, f.bitangent.z])
        .collect();
    let normals = frames
        .iter()
        .map(|f| [f.normal.x, f.normal.y, f.normal.z])
        .collect();
    (bitangents, normals)
}

/// Asserts the `GPU` shell for one strand matches the `CPU` golden per vertex.
fn assert_shell_parity(gpu: &GpuShellMesh, cpu: &ShellMesh) {
    assert_eq!(
        gpu.positions.len(),
        cpu.positions.len(),
        "vertex count must match golden"
    );
    assert_eq!(gpu.normals.len(), cpu.normals.len(), "normal count");
    assert_eq!(gpu.uvs.len(), cpu.uvs.len(), "uv count");
    for (i, (g, c)) in gpu.positions.iter().zip(cpu.positions.iter()).enumerate() {
        assert_close(g[0], c.x, &format!("vertex {i} position.x"));
        assert_close(g[1], c.y, &format!("vertex {i} position.y"));
        assert_close(g[2], c.z, &format!("vertex {i} position.z"));
    }
    for (i, (g, c)) in gpu.normals.iter().zip(cpu.normals.iter()).enumerate() {
        assert_close(g[0], c.x, &format!("vertex {i} normal.x"));
        assert_close(g[1], c.y, &format!("vertex {i} normal.y"));
        assert_close(g[2], c.z, &format!("vertex {i} normal.z"));
    }
    for (i, (g, c)) in gpu.uvs.iter().zip(cpu.uvs.iter()).enumerate() {
        assert_close(g[0], c[0], &format!("vertex {i} u"));
        assert_close(g[1], c[1], &format!("vertex {i} v"));
    }
    // Indices are a pure integer function of the ring count: exact.
    assert_eq!(
        gpu.indices, cpu.indices,
        "indices must match golden exactly"
    );
}

/// Recovers the ring's square half-extent (the `radius_at` value the section
/// tapered to) from opposite corners 0 and 2. For an orthonormal frame,
/// `corner0 - corner2 = 2 r (bitangent + normal)`, whose length is
/// `2 r sqrt(2)`, so `r = |corner0 - corner2| / (2 sqrt(2))`.
fn ring_radius(mesh: &GpuShellMesh, ring: usize) -> f32 {
    let c0 = mesh.positions[4 * ring];
    let c2 = mesh.positions[4 * ring + 2];
    let dx = c0[0] - c2[0];
    let dy = c0[1] - c2[1];
    let dz = c0[2] - c2[2];
    let diag = (dx * dx + dy * dy + dz * dz).sqrt();
    diag / (2.0 * 2.0_f32.sqrt())
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_straight_tapered_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping mesh_shell_tapered parity: no wgpu adapter on this host");
        return;
    };

    // Four evenly spaced rings, t = 0, 1/3, 2/3, 1; radius ramps 0.12 -> 0.03.
    let points: Vec<[f32; 3]> = vec![
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 3.0, 0.0],
    ];
    let root_radius = 0.12;
    let tip_radius = 0.03;
    let (bit, nrm) = frame_axes(&points);
    let inputs = [TaperedShellStrandInput {
        points: &points,
        bitangents: &bit,
        normals: &nrm,
        root_radius,
        tip_radius,
    }];

    let out = GpuMeshShellTapered::new(&ctx).eval(&ctx, &inputs);
    assert_eq!(out.len(), 1, "one mesh per strand");
    let cpu = cpu_golden(&points, root_radius, tip_radius);
    assert_shell_parity(&out[0], &cpu);

    // Non-trivial structure: four rings of four corners, 8*(n-1)+4 triangles.
    assert_eq!(out[0].positions.len(), 16, "four corners × four rings");
    assert_eq!(
        out[0].indices.len(),
        3 * (8 * 3 + 4),
        "side + cap triangles"
    );

    // The taper is real: the recovered half-extent equals radius_at(t) at the
    // root (0.12), an interior ring (t = 1/3 -> 0.09) and the tip (0.03). A
    // plain non-tapering kernel would report a single radius on all rings.
    assert_close(ring_radius(&out[0], 0), 0.12, "root ring radius");
    assert_close(ring_radius(&out[0], 1), 0.09, "interior ring radius");
    assert_close(ring_radius(&out[0], 3), 0.03, "tip ring radius");

    // Opposite corners (0/2) average back to the centerline at every ring.
    for (i, p) in points.iter().enumerate() {
        let c0 = out[0].positions[4 * i];
        let c2 = out[0].positions[4 * i + 2];
        assert_close(0.5 * (c0[0] + c2[0]), p[0], &format!("ring {i} center x"));
        assert_close(0.5 * (c0[1] + c2[1]), p[1], &format!("ring {i} center y"));
        assert_close(0.5 * (c0[2] + c2[2]), p[2], &format!("ring {i} center z"));
    }
    // Every emitted normal is unit length.
    for (i, n) in out[0].normals.iter().enumerate() {
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        assert_close(len, 1.0, &format!("vertex {i} normal length"));
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_curved_tapered_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping mesh_shell_tapered parity: no wgpu adapter on this host");
        return;
    };

    // A bending strand exercises the rotation-minimizing frame transport and the
    // arc-length accumulation together with the taper (five rings, t across the
    // full 0..=1 range so radius_at spans root -> tip).
    let points: Vec<[f32; 3]> = vec![
        [0.0, 0.0, 0.0],
        [0.4, 0.9, 0.1],
        [1.0, 1.6, 0.5],
        [1.8, 2.0, 1.2],
        [2.7, 2.1, 2.0],
    ];
    let root_radius = 0.15;
    let tip_radius = 0.04;
    let (bit, nrm) = frame_axes(&points);
    let inputs = [TaperedShellStrandInput {
        points: &points,
        bitangents: &bit,
        normals: &nrm,
        root_radius,
        tip_radius,
    }];

    let out = GpuMeshShellTapered::new(&ctx).eval(&ctx, &inputs);
    assert_eq!(out.len(), 1, "one mesh per strand");
    let cpu = cpu_golden(&points, root_radius, tip_radius);
    assert_shell_parity(&out[0], &cpu);

    // Taper at the endpoints and a strictly decreasing interior radius.
    assert_close(ring_radius(&out[0], 0), root_radius, "root ring radius");
    assert_close(ring_radius(&out[0], 4), tip_radius, "tip ring radius");
    let mut prev = f32::INFINITY;
    for i in 0..points.len() {
        let r = ring_radius(&out[0], i);
        assert!(r <= prev + 1e-4, "ring {i} radius must not grow root->tip");
        prev = r;
    }

    // u walks the perimeter in quarter steps; v is monotone non-decreasing and
    // spans exactly 0 → 1 root to tip.
    for (i, _) in points.iter().enumerate() {
        for j in 0..4 {
            assert_close(
                out[0].uvs[4 * i + j][0],
                j as f32 * 0.25,
                &format!("ring {i} corner {j} u"),
            );
        }
    }
    assert_close(out[0].uvs[0][1], 0.0, "root v");
    let last = out[0].uvs.len() - 1;
    assert_close(out[0].uvs[last][1], 1.0, "tip v");
    let mut prev_v = -1.0_f32;
    for uv in &out[0].uvs {
        assert!(uv[1] >= prev_v - 1e-4, "v must be non-decreasing");
        prev_v = uv[1];
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_multi_strand_batch_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping mesh_shell_tapered parity: no wgpu adapter on this host");
        return;
    };

    // Three strands with distinct point counts and distinct tapers in one batch:
    // the flat attribute packing and per-strand vertex offsets must keep them
    // independent, each matching its own golden.
    let a: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]];
    let b: Vec<[f32; 3]> = vec![
        [1.0, 0.0, 0.0],
        [1.3, 0.8, 0.2],
        [1.7, 1.5, 0.6],
        [2.2, 2.0, 1.1],
    ];
    let c: Vec<[f32; 3]> = vec![[-1.0, 0.0, 0.0], [-1.0, 0.5, 0.4]];

    let (a_bit, a_nrm) = frame_axes(&a);
    let (b_bit, b_nrm) = frame_axes(&b);
    let (c_bit, c_nrm) = frame_axes(&c);

    let inputs = [
        TaperedShellStrandInput {
            points: &a,
            bitangents: &a_bit,
            normals: &a_nrm,
            root_radius: 0.10,
            tip_radius: 0.02,
        },
        TaperedShellStrandInput {
            points: &b,
            bitangents: &b_bit,
            normals: &b_nrm,
            root_radius: 0.18,
            tip_radius: 0.05,
        },
        TaperedShellStrandInput {
            points: &c,
            bitangents: &c_bit,
            normals: &c_nrm,
            root_radius: 0.07,
            tip_radius: 0.07,
        },
    ];

    let out = GpuMeshShellTapered::new(&ctx).eval(&ctx, &inputs);
    assert_eq!(out.len(), 3, "one mesh per strand");
    assert_shell_parity(&out[0], &cpu_golden(&a, 0.10, 0.02));
    assert_shell_parity(&out[1], &cpu_golden(&b, 0.18, 0.05));
    assert_shell_parity(&out[2], &cpu_golden(&c, 0.07, 0.07));

    // Third strand has equal root/tip radii: a genuine uniform section falls out
    // as a special case of the taper (every ring radius equals 0.07).
    assert_close(ring_radius(&out[2], 0), 0.07, "uniform root radius");
    assert_close(ring_radius(&out[2], 1), 0.07, "uniform tip radius");
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_zero_length_strand_falls_back_to_even_v() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping mesh_shell_tapered parity: no wgpu adapter on this host");
        return;
    };

    // Exactly coincident control points (zero total arc length): both sides take
    // the even-parameter-spacing fallback bit-for-bit, so v is 0, 0.5, 1. The
    // taper still applies by ring index (radius_at at t = 0, 0.5, 1).
    let coincident: Vec<[f32; 3]> = vec![[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]];
    let (bit, nrm) = frame_axes(&coincident);
    let inputs = [TaperedShellStrandInput {
        points: &coincident,
        bitangents: &bit,
        normals: &nrm,
        root_radius: 0.10,
        tip_radius: 0.04,
    }];

    let out = GpuMeshShellTapered::new(&ctx).eval(&ctx, &inputs);
    assert_eq!(out.len(), 1, "one mesh per strand");
    assert_shell_parity(&out[0], &cpu_golden(&coincident, 0.10, 0.04));

    // Even spacing: v runs 0, 0.5, 1 across the three coincident rings.
    assert_close(out[0].uvs[0][1], 0.0, "coincident v[0]");
    assert_close(out[0].uvs[4][1], 0.5, "coincident v[1]");
    assert_close(out[0].uvs[8][1], 1.0, "coincident v[2]");
    // The taper follows ring index: root 0.10, midpoint 0.07, tip 0.04.
    assert_close(ring_radius(&out[0], 0), 0.10, "coincident root radius");
    assert_close(ring_radius(&out[0], 1), 0.07, "coincident mid radius");
    assert_close(ring_radius(&out[0], 2), 0.04, "coincident tip radius");
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_degenerate_strands_yield_empty_meshes() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping mesh_shell_tapered parity: no wgpu adapter on this host");
        return;
    };

    // A valid strand plus two invalid ones (one too short, one with a mismatched
    // attribute length) in one batch: the valid strand meshes, the invalid ones
    // reconstruct as empty meshes, matching the golden guards.
    let good: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.5, 0.0]];
    let (good_bit, good_nrm) = frame_axes(&good);

    let single: Vec<[f32; 3]> = vec![[5.0, 5.0, 5.0]];
    let (single_bit, single_nrm) = frame_axes(&single);

    // Two points but only one bitangent: mismatched attribute length.
    let mm_pts: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]];
    let mm_bit: Vec<[f32; 3]> = vec![[0.0, 0.0, 1.0]];
    let mm_nrm: Vec<[f32; 3]> = vec![[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]];

    let inputs = [
        TaperedShellStrandInput {
            points: &good,
            bitangents: &good_bit,
            normals: &good_nrm,
            root_radius: 0.08,
            tip_radius: 0.02,
        },
        TaperedShellStrandInput {
            points: &single,
            bitangents: &single_bit,
            normals: &single_nrm,
            root_radius: 0.1,
            tip_radius: 0.1,
        },
        TaperedShellStrandInput {
            points: &mm_pts,
            bitangents: &mm_bit,
            normals: &mm_nrm,
            root_radius: 0.1,
            tip_radius: 0.1,
        },
    ];

    let out = GpuMeshShellTapered::new(&ctx).eval(&ctx, &inputs);
    assert_eq!(out.len(), 3, "one mesh per strand, valid or not");
    assert_shell_parity(&out[0], &cpu_golden(&good, 0.08, 0.02));
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
        eprintln!("skipping mesh_shell_tapered parity: no wgpu adapter on this host");
        return;
    };
    let kernel = GpuMeshShellTapered::new(&ctx);

    // No strands at all: empty outer vec, no dispatch.
    let none: [TaperedShellStrandInput; 0] = [];
    assert!(kernel.eval(&ctx, &none).is_empty(), "no strands -> empty");

    // Every strand invalid (below the two-point minimum): one empty mesh per
    // strand, still no dispatch (storage buffers cannot be zero-sized).
    let empty_pts: Vec<[f32; 3]> = Vec::new();
    let empty_bit: Vec<[f32; 3]> = Vec::new();
    let empty_nrm: Vec<[f32; 3]> = Vec::new();
    let all_invalid = [
        TaperedShellStrandInput {
            points: &empty_pts,
            bitangents: &empty_bit,
            normals: &empty_nrm,
            root_radius: 0.1,
            tip_radius: 0.1,
        },
        TaperedShellStrandInput {
            points: &empty_pts,
            bitangents: &empty_bit,
            normals: &empty_nrm,
            root_radius: 0.1,
            tip_radius: 0.1,
        },
    ];
    let out = kernel.eval(&ctx, &all_invalid);
    assert_eq!(out.len(), 2, "one mesh per (invalid) strand");
    assert!(
        out[0].positions.is_empty() && out[1].positions.is_empty(),
        "invalid strands -> empty meshes"
    );
}
