//! Real-device parity for the strand-to-shell (`Mesh` LOD proxy) twin:
//! [`GpuMeshShell`] must reproduce the `CPU` golden
//! [`build_shell`](prism_render_architecture::hair::mesh_shell::build_shell) for
//! every shell vertex of every strand — the four rectangle-corner positions
//! offset along the bitangent (`±half_width`) and normal (`±half_thickness`), the
//! diagonal outward corner normals (with the `normalize_or` fallback), the
//! quarter-perimeter `u` and arc-length `v` UVs (including the zero-length
//! fallback), and the deterministic triangle winding — while reproducing the
//! reference's guards (a strand with fewer than two points, or with mismatched
//! attribute lengths, yields an empty mesh).
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The meshing is closed-form geometry (the reference restricts itself to
//! `sqrt` for the arc length and the corner-normal normalize), so the `CPU` and
//! `GPU` evaluate the same expression and diverge only through legal
//! fused-multiply-add contraction and a possible reassociation of the arc-length
//! sum. Each position, normal and UV component is asserted to within
//! `abs_diff < 1e-4` or `rel_diff < 1e-3` — tight enough to fail a genuinely
//! wrong port (a swapped corner, a missing extent scale, a wrong `v`), loose
//! enough to admit fma contraction. Indices are a pure function of the point
//! count and carry no float error, so they are asserted for exact equality. The
//! smooth sweeps also assert shell geometry (opposite corners centered on the
//! centerline, extents matching `half_width`/`half_thickness`, unit normals,
//! monotone `v` root→tip, `u ∈ {0, 0.25, 0.5, 0.75}`) so a degenerate kernel
//! could not pass. The zero-length case uses exactly coincident points so both
//! sides take the even-spacing fallback branch bit-for-bit.
//!
//! Provenance: standard rectangular shell-tube meshing; no Unreal Engine source
//! or derived code.

use prism_hair_gpu::mesh_shell::{GpuMeshShell, GpuShellMesh, ShellStrandInput};
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::frames::build_strand_frames;
use prism_render_architecture::hair::interpolation::Vec3;
use prism_render_architecture::hair::mesh_shell::{build_shell, ShellMesh};

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
fn cpu_golden(points: &[[f32; 3]], half_width: f32, half_thickness: f32) -> ShellMesh {
    let pts: Vec<Vec3> = points.iter().map(|p| to_vec3(*p)).collect();
    let frames = build_strand_frames(&pts);
    build_shell(&pts, &frames, half_width, half_thickness)
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

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_straight_strand_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping mesh_shell parity: no wgpu adapter on this host");
        return;
    };

    let points: Vec<[f32; 3]> = vec![
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 3.0, 0.0],
    ];
    let (bit, nrm) = frame_axes(&points);
    let inputs = [ShellStrandInput {
        points: &points,
        bitangents: &bit,
        normals: &nrm,
        half_width: 0.1,
        half_thickness: 0.05,
    }];

    let out = GpuMeshShell::new(&ctx).eval(&ctx, &inputs);
    assert_eq!(out.len(), 1, "one mesh per strand");
    let cpu = cpu_golden(&points, 0.1, 0.05);
    assert_shell_parity(&out[0], &cpu);

    // Non-trivial structure: four rings of four corners, 8*(n-1)+4 triangles.
    assert_eq!(out[0].positions.len(), 16, "four corners × four rings");
    assert_eq!(
        out[0].indices.len(),
        3 * (8 * 3 + 4),
        "side + cap triangles"
    );

    // Opposite corners (0/2 and 1/3) average back to the centerline.
    for (i, p) in points.iter().enumerate() {
        let c0 = out[0].positions[4 * i];
        let c2 = out[0].positions[4 * i + 2];
        assert_close(
            0.5 * (c0[0] + c2[0]),
            p[0],
            &format!("ring {i} 0/2 center x"),
        );
        assert_close(
            0.5 * (c0[1] + c2[1]),
            p[1],
            &format!("ring {i} 0/2 center y"),
        );
        assert_close(
            0.5 * (c0[2] + c2[2]),
            p[2],
            &format!("ring {i} 0/2 center z"),
        );
    }
    // Every emitted normal is unit length.
    for (i, nrm) in out[0].normals.iter().enumerate() {
        let len = (nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2]).sqrt();
        assert_close(len, 1.0, &format!("vertex {i} normal length"));
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_curved_strand_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping mesh_shell parity: no wgpu adapter on this host");
        return;
    };

    // A bending strand exercises the rotation-minimizing frame transport, the
    // arc-length accumulation, and a non-square section (width ≠ thickness).
    let points: Vec<[f32; 3]> = vec![
        [0.0, 0.0, 0.0],
        [0.4, 0.9, 0.1],
        [1.0, 1.6, 0.5],
        [1.8, 2.0, 1.2],
        [2.7, 2.1, 2.0],
    ];
    let (bit, nrm) = frame_axes(&points);
    let inputs = [ShellStrandInput {
        points: &points,
        bitangents: &bit,
        normals: &nrm,
        half_width: 0.14,
        half_thickness: 0.06,
    }];

    let out = GpuMeshShell::new(&ctx).eval(&ctx, &inputs);
    assert_eq!(out.len(), 1, "one mesh per strand");
    let cpu = cpu_golden(&points, 0.14, 0.06);
    assert_shell_parity(&out[0], &cpu);

    // u walks the perimeter in quarter steps; v is monotone non-decreasing and
    // spans exactly 0 → 1 root to tip.
    for i in 0..points.len() {
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
    let mut prev = -1.0_f32;
    for uv in &out[0].uvs {
        assert!(uv[1] >= prev - 1e-4, "v must be non-decreasing");
        prev = uv[1];
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_batch_of_strands_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping mesh_shell parity: no wgpu adapter on this host");
        return;
    };

    // Two strands of different lengths and sections meshed in one dispatch:
    // exercises the per-strand vertex-offset packing and readback re-split.
    let a: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0], [0.5, 1.0, 0.0], [1.0, 2.0, 0.3]];
    let b: Vec<[f32; 3]> = vec![
        [3.0, 0.0, 0.0],
        [3.0, 0.8, 0.4],
        [3.2, 1.5, 0.9],
        [3.6, 2.0, 1.4],
    ];
    let (a_bit, a_nrm) = frame_axes(&a);
    let (b_bit, b_nrm) = frame_axes(&b);
    let inputs = [
        ShellStrandInput {
            points: &a,
            bitangents: &a_bit,
            normals: &a_nrm,
            half_width: 0.09,
            half_thickness: 0.09,
        },
        ShellStrandInput {
            points: &b,
            bitangents: &b_bit,
            normals: &b_nrm,
            half_width: 0.12,
            half_thickness: 0.03,
        },
    ];

    let out = GpuMeshShell::new(&ctx).eval(&ctx, &inputs);
    assert_eq!(out.len(), 2, "one mesh per strand");
    assert_shell_parity(&out[0], &cpu_golden(&a, 0.09, 0.09));
    assert_shell_parity(&out[1], &cpu_golden(&b, 0.12, 0.03));
    assert_eq!(out[0].positions.len(), 12, "3 rings × 4 corners");
    assert_eq!(out[1].positions.len(), 16, "4 rings × 4 corners");
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_zero_length_strand_falls_back_to_even_v() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping mesh_shell parity: no wgpu adapter on this host");
        return;
    };

    // Exactly coincident control points (zero total arc length): both sides take
    // the even-parameter-spacing fallback bit-for-bit, so v is 0, 0.5, 1.
    let coincident: Vec<[f32; 3]> = vec![[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]];
    let (bit, nrm) = frame_axes(&coincident);
    let inputs = [ShellStrandInput {
        points: &coincident,
        bitangents: &bit,
        normals: &nrm,
        half_width: 0.1,
        half_thickness: 0.1,
    }];

    let out = GpuMeshShell::new(&ctx).eval(&ctx, &inputs);
    assert_eq!(out.len(), 1, "one mesh per strand");
    assert_shell_parity(&out[0], &cpu_golden(&coincident, 0.1, 0.1));

    // Even spacing: v runs 0, 0.5, 1 across the three coincident rings.
    assert_close(out[0].uvs[0][1], 0.0, "coincident v[0]");
    assert_close(out[0].uvs[4][1], 0.5, "coincident v[1]");
    assert_close(out[0].uvs[8][1], 1.0, "coincident v[2]");
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_degenerate_strands_yield_empty_meshes() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping mesh_shell parity: no wgpu adapter on this host");
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
        ShellStrandInput {
            points: &good,
            bitangents: &good_bit,
            normals: &good_nrm,
            half_width: 0.08,
            half_thickness: 0.04,
        },
        ShellStrandInput {
            points: &single,
            bitangents: &single_bit,
            normals: &single_nrm,
            half_width: 0.1,
            half_thickness: 0.1,
        },
        ShellStrandInput {
            points: &mm_pts,
            bitangents: &mm_bit,
            normals: &mm_nrm,
            half_width: 0.1,
            half_thickness: 0.1,
        },
    ];

    let out = GpuMeshShell::new(&ctx).eval(&ctx, &inputs);
    assert_eq!(out.len(), 3, "one mesh per strand, valid or not");
    assert_shell_parity(&out[0], &cpu_golden(&good, 0.08, 0.04));
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
        eprintln!("skipping mesh_shell parity: no wgpu adapter on this host");
        return;
    };
    let kernel = GpuMeshShell::new(&ctx);

    // No strands at all: empty outer vec, no dispatch.
    let none: [ShellStrandInput; 0] = [];
    assert!(kernel.eval(&ctx, &none).is_empty(), "no strands -> empty");

    // Every strand invalid (below the two-point minimum): one empty mesh per
    // strand, still no dispatch (storage buffers cannot be zero-sized).
    let empty_pts: Vec<[f32; 3]> = Vec::new();
    let empty_bit: Vec<[f32; 3]> = Vec::new();
    let empty_nrm: Vec<[f32; 3]> = Vec::new();
    let all_invalid = [
        ShellStrandInput {
            points: &empty_pts,
            bitangents: &empty_bit,
            normals: &empty_nrm,
            half_width: 0.1,
            half_thickness: 0.1,
        },
        ShellStrandInput {
            points: &empty_pts,
            bitangents: &empty_bit,
            normals: &empty_nrm,
            half_width: 0.1,
            half_thickness: 0.1,
        },
    ];
    let out = kernel.eval(&ctx, &all_invalid);
    assert_eq!(out.len(), 2, "one mesh per (invalid) strand");
    assert!(
        out[0].positions.is_empty() && out[1].positions.is_empty(),
        "invalid strands -> empty meshes"
    );
}
