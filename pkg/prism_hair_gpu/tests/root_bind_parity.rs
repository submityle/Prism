//! Real-device parity for the import-time root-binding bake twin:
//! [`GpuHairRootBind`] must reproduce the `CPU` golden
//! [`bind_roots`](prism_render_architecture::hair::binding::bind_roots) for a
//! batch of authored roots. The suite covers an interior-face projection, a
//! vertex-region and an edge-region clamp, nearest-triangle selection among
//! several well-separated faces, unbound degeneracies (empty mesh and an
//! out-of-range face, both exact `UNBOUND`), the empty no-op, a large
//! multi-workgroup batch, and an end-to-end bake -> resolve round trip.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The chosen triangle is an integer selection, so it is asserted *exactly*
//! (`assert_eq!`); the barycentric weights and the signed height are plain IEEE
//! `sqrt`/`dot`/`cross`/multiply-add both sides evaluate identically, differing
//! only where a `GPU` may fuse a multiply-add the scalar reference leaves
//! separate, so they are asserted to within `abs_diff < 1e-4` or
//! `rel_diff < 1e-3`. Unbound roots match exactly (sentinel index, zeroed
//! weights and height). Every test root has an unambiguous nearest triangle so
//! the integer selection can never flip under that fma perturbation. No
//! `sin`/`cos` appears anywhere.
//!
//! Provenance: standard closest-point-on-triangle (Ericson Voronoi-region test)
//! plus a brute-force nearest scan; no Unreal Engine source or derived code.

use prism_hair_gpu::GpuContext;
use prism_hair_gpu::GpuHairRootBind;
use prism_render_architecture::hair::binding::{
    bind_roots, resolve_root_frames, MeshBinding, UNBOUND,
};
use prism_render_architecture::hair::interpolation::Vec3;

/// Asserts a single scalar component matches within the documented fma tolerance.
fn assert_close(got: f32, expected: f32, label: &str) {
    let abs_diff = (got - expected).abs();
    let rel_diff = abs_diff / expected.abs().max(1e-6);
    assert!(
        abs_diff < 1e-4 || rel_diff < 1e-3,
        "{label}: gpu {got}, cpu {expected} (abs {abs_diff}, rel {rel_diff})"
    );
}

/// Asserts a single component is bit-identical (zero difference).
///
/// Avoids `==` on floats (the `float_cmp` lint) by bounding the absolute
/// difference by `0.0`, which only holds when the values match exactly —
/// unbound sentinels must not drift at all.
fn assert_exact(got: f32, expected: f32, label: &str) {
    assert!(
        (got - expected).abs() <= 0.0,
        "{label}: {got} != {expected}"
    );
}

/// Asserts a baked binding matches the golden: triangle index exactly, weights
/// and height within tolerance for a bound root, everything exact for `UNBOUND`.
fn assert_binding(got: &MeshBinding, expected: &MeshBinding, label: &str) {
    assert_eq!(got.triangle, expected.triangle, "{label}: triangle index");
    if expected.is_bound() {
        assert_close(got.bary[0], expected.bary[0], &format!("{label} bary0"));
        assert_close(got.bary[1], expected.bary[1], &format!("{label} bary1"));
        assert_close(got.bary[2], expected.bary[2], &format!("{label} bary2"));
        assert_close(got.height, expected.height, &format!("{label} height"));
    } else {
        assert_exact(got.bary[0], 0.0, &format!("{label} unbound bary0"));
        assert_exact(got.bary[1], 0.0, &format!("{label} unbound bary1"));
        assert_exact(got.bary[2], 0.0, &format!("{label} unbound bary2"));
        assert_exact(got.height, 0.0, &format!("{label} unbound height"));
    }
}

fn v(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3::new(x, y, z)
}

/// A single unit right-triangle in the y = 0 plane; winding [0, 2, 1] makes the
/// outward face normal point +y (a scalp patch facing up).
fn unit_scalp() -> (Vec<Vec3>, Vec<[u32; 3]>) {
    let verts = vec![v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0), v(0.0, 0.0, 1.0)];
    let tris = vec![[0u32, 2, 1]];
    (verts, tris)
}

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

/// Bakes `roots` on both the `CPU` golden and the `GPU`, asserting the full
/// batch matches binding-for-binding.
fn assert_batch_parity(
    ctx: &GpuContext,
    kernel: &GpuHairRootBind,
    roots: &[Vec3],
    vertices: &[Vec3],
    triangles: &[[u32; 3]],
    label: &str,
) {
    let gpu = kernel.eval(ctx, roots, vertices, triangles);
    let cpu = bind_roots(roots, vertices, triangles);
    assert_eq!(gpu.len(), roots.len(), "{label}: one binding per root");
    assert_eq!(cpu.len(), roots.len(), "{label}: golden count");
    for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        assert_binding(g, c, &format!("{label} root {i}"));
    }
}

#[test]
fn binds_root_inside_single_face() {
    let Some(ctx) = context_or_skip("root_bind interior parity") else {
        return;
    };
    let kernel = GpuHairRootBind::new(&ctx);
    let (verts, tris) = unit_scalp();
    // A root above the face interior projects onto an interior point (all three
    // barycentric weights strictly positive) at a positive float-off height.
    let roots = vec![v(0.25, 0.5, 0.25)];
    let cpu = bind_roots(&roots, &verts, &tris);
    assert_eq!(cpu[0].triangle, 0, "interior root binds to the only face");
    assert!(cpu[0].bary.iter().all(|&w| w > 0.0), "interior weights");
    assert!(cpu[0].height > 0.4, "positive float-off height");
    assert_batch_parity(&ctx, &kernel, &roots, &verts, &tris, "interior");
}

#[test]
fn clamps_root_to_vertex_region() {
    let Some(ctx) = context_or_skip("root_bind vertex-region parity") else {
        return;
    };
    let kernel = GpuHairRootBind::new(&ctx);
    let (verts, tris) = unit_scalp();
    // A root off the corner past vertex A clamps to A: bary = [1, 0, 0].
    let roots = vec![v(-1.0, 0.5, -1.0)];
    let cpu = bind_roots(&roots, &verts, &tris);
    assert_close(cpu[0].bary[0], 1.0, "cpu vertex-region bary0");
    assert_close(cpu[0].bary[1], 0.0, "cpu vertex-region bary1");
    assert_close(cpu[0].bary[2], 0.0, "cpu vertex-region bary2");
    assert_batch_parity(&ctx, &kernel, &roots, &verts, &tris, "vertex region");
}

#[test]
fn clamps_root_to_edge_region() {
    let Some(ctx) = context_or_skip("root_bind edge-region parity") else {
        return;
    };
    let kernel = GpuHairRootBind::new(&ctx);
    let (verts, tris) = unit_scalp();
    // A root off an edge clamps to that edge's midpoint. With the [0, 2, 1]
    // winding the triangle corners are a=(0,0,0), b=(0,0,1), c=(1,0,0), so
    // this root clamps onto edge AC at its midpoint: bary = [0.5, 0, 0.5].
    let roots = vec![v(0.5, 0.5, -1.0)];
    let cpu = bind_roots(&roots, &verts, &tris);
    assert_close(cpu[0].bary[0], 0.5, "cpu edge-region bary0");
    assert_close(cpu[0].bary[1], 0.0, "cpu edge-region bary1");
    assert_close(cpu[0].bary[2], 0.5, "cpu edge-region bary2");
    assert_batch_parity(&ctx, &kernel, &roots, &verts, &tris, "edge region");
}

/// Two well-separated scalp patches so every root has an unambiguous nearest
/// face: patch 0 around x in [0, 1], patch 1 around x in [10, 11].
fn two_far_patches() -> (Vec<Vec3>, Vec<[u32; 3]>) {
    let verts = vec![
        v(0.0, 0.0, 0.0),
        v(1.0, 0.0, 0.0),
        v(0.0, 0.0, 1.0),
        v(10.0, 0.0, 0.0),
        v(11.0, 0.0, 0.0),
        v(10.0, 0.0, 1.0),
    ];
    let tris = vec![[0u32, 2, 1], [3u32, 5, 4]];
    (verts, tris)
}

#[test]
fn selects_nearest_among_multiple_faces() {
    let Some(ctx) = context_or_skip("root_bind nearest-face parity") else {
        return;
    };
    let kernel = GpuHairRootBind::new(&ctx);
    let (verts, tris) = two_far_patches();
    // One root clearly over patch 0, one clearly over patch 1.
    let roots = vec![v(0.25, 0.3, 0.25), v(10.25, 0.3, 0.25)];
    let cpu = bind_roots(&roots, &verts, &tris);
    assert_eq!(cpu[0].triangle, 0, "root near patch 0 binds face 0");
    assert_eq!(cpu[1].triangle, 1, "root near patch 1 binds face 1");
    assert_batch_parity(&ctx, &kernel, &roots, &verts, &tris, "nearest face");
}

#[test]
fn empty_mesh_binds_to_exact_unbound() {
    let Some(ctx) = context_or_skip("root_bind empty-mesh parity") else {
        return;
    };
    let kernel = GpuHairRootBind::new(&ctx);
    let verts: Vec<Vec3> = Vec::new();
    let tris: Vec<[u32; 3]> = Vec::new();
    let roots = vec![v(0.2, 0.5, 0.3), v(1.0, 1.0, 1.0)];
    let gpu = kernel.eval(&ctx, &roots, &verts, &tris);
    let cpu = bind_roots(&roots, &verts, &tris);
    assert_eq!(gpu.len(), roots.len());
    for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        assert_eq!(c.triangle, UNBOUND, "golden root {i} unbound");
        assert_binding(g, c, &format!("empty-mesh root {i}"));
    }
}

#[test]
fn out_of_range_face_is_skipped_to_exact_unbound() {
    let Some(ctx) = context_or_skip("root_bind out-of-range parity") else {
        return;
    };
    let kernel = GpuHairRootBind::new(&ctx);
    // The only face references a vertex index past the pool, so it is skipped and
    // the root has no bindable triangle -> UNBOUND.
    let verts = vec![v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0), v(0.0, 0.0, 1.0)];
    let tris = vec![[0u32, 1, 9]];
    let roots = vec![v(0.25, 0.5, 0.25)];
    let cpu = bind_roots(&roots, &verts, &tris);
    assert_eq!(cpu[0].triangle, UNBOUND, "golden skips the bad face");
    let gpu = kernel.eval(&ctx, &roots, &verts, &tris);
    assert_binding(&gpu[0], &cpu[0], "out-of-range face");
}

#[test]
fn empty_roots_is_a_no_op() {
    let Some(ctx) = context_or_skip("root_bind empty no-op") else {
        return;
    };
    let kernel = GpuHairRootBind::new(&ctx);
    let (verts, tris) = unit_scalp();
    let gpu = kernel.eval(&ctx, &[], &verts, &tris);
    assert!(gpu.is_empty(), "empty root list yields no bindings");
}

#[test]
fn large_batch_across_workgroups_matches_golden() {
    let Some(ctx) = context_or_skip("root_bind large-batch parity") else {
        return;
    };
    let kernel = GpuHairRootBind::new(&ctx);
    let (verts, tris) = two_far_patches();
    // 200 roots (> 3 workgroups of 64), each deterministically placed clearly
    // over one of the two far-apart patches so its nearest face is unambiguous.
    let n = 200usize;
    let mut roots = Vec::with_capacity(n);
    for i in 0..n {
        let fx = ((i * 7 + 3) % 100) as f32 / 100.0;
        let fz = ((i * 13 + 5) % 100) as f32 / 100.0;
        let fy = ((i * 3 + 1) % 50) as f32 / 100.0;
        let base_x = if i % 2 == 0 { 0.0 } else { 10.0 };
        roots.push(v(base_x + fx, fy, fz));
    }
    let cpu = bind_roots(&roots, &verts, &tris);
    for (i, b) in cpu.iter().enumerate() {
        let expected = if i % 2 == 0 { 0 } else { 1 };
        assert_eq!(b.triangle, expected, "root {i} nearest face");
    }
    assert_batch_parity(&ctx, &kernel, &roots, &verts, &tris, "large batch");
}

#[test]
fn bake_then_resolve_round_trips_to_authored_roots() {
    let Some(ctx) = context_or_skip("root_bind round-trip parity") else {
        return;
    };
    let kernel = GpuHairRootBind::new(&ctx);
    let (verts, tris) = unit_scalp();
    // Bake on the GPU, then resolve the GPU-baked bindings against the same rest
    // pose on the CPU golden: the resolved positions must reproduce the authored
    // roots (the exact round-trip property the binding module pins).
    let roots = vec![
        v(0.2, 0.0, 0.2),
        v(0.3, 0.5, 0.1),
        v(0.1, 0.9, 0.4),
        v(0.45, 0.0, 0.45),
    ];
    let gpu_bindings = kernel.eval(&ctx, &roots, &verts, &tris);
    assert!(gpu_bindings.iter().all(MeshBinding::is_bound));
    assert_batch_parity(&ctx, &kernel, &roots, &verts, &tris, "round trip");
    let frames = resolve_root_frames(&gpu_bindings, &verts, &tris);
    for (i, (frame, root)) in frames.iter().zip(roots.iter()).enumerate() {
        assert_close(frame.position.x, root.x, &format!("round-trip root {i}.x"));
        assert_close(frame.position.y, root.y, &format!("round-trip root {i}.y"));
        assert_close(frame.position.z, root.z, &format!("round-trip root {i}.z"));
    }
}
