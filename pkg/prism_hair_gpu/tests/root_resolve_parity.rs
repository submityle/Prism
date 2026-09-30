//! Real-device parity for the per-frame root-binding resolve twin:
//! [`GpuHairRootResolve`] must reproduce the `CPU` golden
//! [`resolve_root_frames`](prism_render_architecture::hair::binding::resolve_root_frames)
//! for a batch of scalp attachments. The suite covers an on-surface resolve at
//! zero height, a float-off along the face normal, a batched orthonormal-basis
//! check, unbound / out-of-range / zero-area degeneracies (all exact identity),
//! following a deformed (skinned) scalp, a large multi-workgroup batch and the
//! empty no-op — plus an end-to-end `bind_roots` → resolve round trip.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The barycentric blend, the face normal and the Gram-Schmidt basis are plain
//! IEEE `sqrt`/`dot`/`cross`/multiply-add both sides evaluate identically; they
//! differ only where a `GPU` may fuse a multiply-add the scalar reference leaves
//! separate, so non-identity components are asserted to within
//! `abs_diff < 1e-4` or `rel_diff < 1e-3` — tight enough to fail a genuinely
//! wrong resolve (a swapped bary weight, a dropped height, a mis-built basis),
//! loose enough to admit the fma contraction. Identity frames (unbound /
//! degenerate) match exactly. No `sin`/`cos` appears anywhere.
//!
//! Provenance: standard barycentric mesh-attachment resolve plus a Gram-Schmidt
//! orthonormal basis; no Unreal Engine source or derived code.

use prism_hair_gpu::GpuContext;
use prism_hair_gpu::GpuHairRootResolve;
use prism_render_architecture::hair::binding::{
    bind_roots, resolve_root_frames, MeshBinding, RootFrame, UNBOUND,
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

/// Asserts two vectors match component-wise within tolerance.
fn assert_vec_close(got: Vec3, expected: Vec3, label: &str) {
    assert_close(got.x, expected.x, &format!("{label}.x"));
    assert_close(got.y, expected.y, &format!("{label}.y"));
    assert_close(got.z, expected.z, &format!("{label}.z"));
}

/// Asserts a full frame matches the golden within tolerance.
fn assert_frame_close(got: &RootFrame, expected: &RootFrame, label: &str) {
    assert_vec_close(
        got.position,
        expected.position,
        &format!("{label} position"),
    );
    assert_vec_close(got.tangent, expected.tangent, &format!("{label} tangent"));
    assert_vec_close(got.normal, expected.normal, &format!("{label} normal"));
    assert_vec_close(
        got.bitangent,
        expected.bitangent,
        &format!("{label} bitangent"),
    );
}

/// Asserts two vectors are bit-identical (zero component difference).
///
/// Avoids `==` on floats (the `float_cmp` lint) by bounding the absolute
/// component difference by `0.0`, which only holds when the values match
/// exactly — degenerate bindings must not drift at all.
fn assert_vec_exact(got: Vec3, expected: Vec3, label: &str) {
    assert!(
        (got.x - expected.x).abs() <= 0.0,
        "{label}.x: {got:?} != {expected:?}"
    );
    assert!(
        (got.y - expected.y).abs() <= 0.0,
        "{label}.y: {got:?} != {expected:?}"
    );
    assert!(
        (got.z - expected.z).abs() <= 0.0,
        "{label}.z: {got:?} != {expected:?}"
    );
}

/// Asserts a frame equals the identity frame exactly (degenerate bindings must
/// not drift at all).
fn assert_identity_exact(got: &RootFrame, label: &str) {
    let id = RootFrame::IDENTITY;
    assert_vec_exact(got.position, id.position, &format!("{label} position"));
    assert_vec_exact(got.tangent, id.tangent, &format!("{label} tangent"));
    assert_vec_exact(got.normal, id.normal, &format!("{label} normal"));
    assert_vec_exact(got.bitangent, id.bitangent, &format!("{label} bitangent"));
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

/// Resolves `bindings` on both the `CPU` golden and the `GPU`, asserting the
/// full batch matches: identity frames exactly, live frames within tolerance.
fn assert_batch_parity(
    ctx: &GpuContext,
    kernel: &GpuHairRootResolve,
    bindings: &[MeshBinding],
    vertices: &[Vec3],
    triangles: &[[u32; 3]],
    label: &str,
) {
    let gpu = kernel.eval(ctx, bindings, vertices, triangles);
    let cpu = resolve_root_frames(bindings, vertices, triangles);
    assert_eq!(gpu.len(), bindings.len(), "{label}: one frame per binding");
    assert_eq!(cpu.len(), bindings.len(), "{label}: golden count");
    for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        if bindings[i].is_bound() {
            assert_frame_close(g, c, &format!("{label} binding {i}"));
        } else {
            assert_identity_exact(g, &format!("{label} binding {i} (unbound)"));
        }
    }
}

#[test]
fn resolves_on_surface_at_zero_height() {
    let Some(ctx) = context_or_skip("root_resolve on-surface parity") else {
        return;
    };
    let kernel = GpuHairRootResolve::new(&ctx);
    let (verts, tris) = unit_scalp();
    // A root right on the surface: bind then resolve must land back on the plane.
    let roots = vec![v(0.25, 0.0, 0.25)];
    let bindings = bind_roots(&roots, &verts, &tris);
    assert!(bindings[0].is_bound());
    let gpu = kernel.eval(&ctx, &bindings, &verts, &tris);
    let cpu = resolve_root_frames(&bindings, &verts, &tris);
    assert_frame_close(&gpu[0], &cpu[0], "on-surface frame");
    // The resolved position sits on y = 0 and reproduces the authored root.
    assert_close(gpu[0].position.y, 0.0, "on-surface height");
    assert_vec_close(gpu[0].position, roots[0], "on-surface position");
    assert_vec_close(gpu[0].normal, v(0.0, 1.0, 0.0), "on-surface normal");
}

#[test]
fn floats_root_off_along_face_normal() {
    let Some(ctx) = context_or_skip("root_resolve height parity") else {
        return;
    };
    let kernel = GpuHairRootResolve::new(&ctx);
    let (verts, tris) = unit_scalp();
    // A root floating above the surface keeps its signed height on resolve.
    let roots = vec![v(0.3, 0.5, 0.2)];
    let bindings = bind_roots(&roots, &verts, &tris);
    assert!(
        bindings[0].height > 0.4,
        "expected a positive float-off height"
    );
    assert_batch_parity(&ctx, &kernel, &bindings, &verts, &tris, "float-off");
    let gpu = kernel.eval(&ctx, &bindings, &verts, &tris);
    // Round trip reproduces the authored root position (+y float-off preserved).
    assert_vec_close(gpu[0].position, roots[0], "float-off position");
}

#[test]
fn resolved_basis_is_orthonormal() {
    let Some(ctx) = context_or_skip("root_resolve orthonormal parity") else {
        return;
    };
    let kernel = GpuHairRootResolve::new(&ctx);
    let (verts, tris) = unit_scalp();
    let roots = vec![v(0.25, 0.1, 0.25), v(0.1, 0.3, 0.1), v(0.4, 0.0, 0.4)];
    let bindings = bind_roots(&roots, &verts, &tris);
    assert_batch_parity(&ctx, &kernel, &bindings, &verts, &tris, "orthonormal batch");
    let gpu = kernel.eval(&ctx, &bindings, &verts, &tris);
    for (i, f) in gpu.iter().enumerate() {
        assert_close(f.tangent.length(), 1.0, &format!("frame {i} |tangent|"));
        assert_close(f.normal.length(), 1.0, &format!("frame {i} |normal|"));
        assert_close(f.bitangent.length(), 1.0, &format!("frame {i} |bitangent|"));
        assert!(
            f.tangent.dot(f.normal).abs() < 1e-4,
            "frame {i} tangent·normal not orthogonal"
        );
        assert!(
            f.tangent.dot(f.bitangent).abs() < 1e-4,
            "frame {i} tangent·bitangent not orthogonal"
        );
        assert!(
            f.normal.dot(f.bitangent).abs() < 1e-4,
            "frame {i} normal·bitangent not orthogonal"
        );
    }
}

#[test]
fn unbound_binding_resolves_to_exact_identity() {
    let Some(ctx) = context_or_skip("root_resolve unbound parity") else {
        return;
    };
    let kernel = GpuHairRootResolve::new(&ctx);
    let (verts, tris) = unit_scalp();
    let bindings = vec![MeshBinding {
        triangle: UNBOUND,
        bary: [0.5, 0.25, 0.25],
        height: 3.0,
    }];
    let gpu = kernel.eval(&ctx, &bindings, &verts, &tris);
    assert_eq!(gpu.len(), 1);
    assert_identity_exact(&gpu[0], "unbound");
}

#[test]
fn out_of_range_triangle_resolves_to_exact_identity() {
    let Some(ctx) = context_or_skip("root_resolve out-of-range parity") else {
        return;
    };
    let kernel = GpuHairRootResolve::new(&ctx);
    let (verts, tris) = unit_scalp();
    let bindings = vec![MeshBinding {
        triangle: 99,
        bary: [0.5, 0.25, 0.25],
        height: 1.0,
    }];
    let gpu = kernel.eval(&ctx, &bindings, &verts, &tris);
    assert_eq!(gpu.len(), 1);
    assert_identity_exact(&gpu[0], "out-of-range triangle");
}

#[test]
fn degenerate_zero_area_face_resolves_to_exact_identity() {
    let Some(ctx) = context_or_skip("root_resolve degenerate parity") else {
        return;
    };
    let kernel = GpuHairRootResolve::new(&ctx);
    // All three vertices coincide: a zero-area face.
    let verts = vec![v(1.0, 1.0, 1.0), v(1.0, 1.0, 1.0), v(1.0, 1.0, 1.0)];
    let tris = vec![[0u32, 1, 2]];
    let bindings = vec![MeshBinding {
        triangle: 0,
        bary: [0.33, 0.33, 0.34],
        height: 0.5,
    }];
    let gpu = kernel.eval(&ctx, &bindings, &verts, &tris);
    let cpu = resolve_root_frames(&bindings, &verts, &tris);
    assert_eq!(gpu.len(), 1);
    // The golden collapses a zero-area face to identity; the twin must too.
    assert_identity_exact(&cpu[0], "degenerate golden");
    assert_identity_exact(&gpu[0], "degenerate gpu");
}

#[test]
fn follows_deformed_scalp() {
    let Some(ctx) = context_or_skip("root_resolve deform parity") else {
        return;
    };
    let kernel = GpuHairRootResolve::new(&ctx);
    let (verts, tris) = unit_scalp();
    // Bind against the rest pose, then resolve against a rigidly translated
    // (skinned) scalp: the root must ride the surface by the same offset.
    let roots = vec![v(0.3, 0.4, 0.2), v(0.1, 0.0, 0.1)];
    let bindings = bind_roots(&roots, &verts, &tris);
    let shift = v(2.0, -1.0, 3.0);
    let deformed: Vec<Vec3> = verts.iter().map(|&p| p + shift).collect();
    assert_batch_parity(&ctx, &kernel, &bindings, &deformed, &tris, "deformed");
    let gpu = kernel.eval(&ctx, &bindings, &deformed, &tris);
    for (i, (frame, root)) in gpu.iter().zip(roots.iter()).enumerate() {
        assert_vec_close(frame.position, *root + shift, &format!("deformed root {i}"));
    }
}

#[test]
fn large_batch_across_workgroups_matches_golden() {
    let Some(ctx) = context_or_skip("root_resolve large-batch parity") else {
        return;
    };
    let kernel = GpuHairRootResolve::new(&ctx);
    // Two coplanar triangles so bindings spread over more than one face.
    let verts = vec![
        v(0.0, 0.0, 0.0),
        v(1.0, 0.0, 0.0),
        v(0.0, 0.0, 1.0),
        v(1.0, 0.0, 1.0),
    ];
    let tris = vec![[0u32, 2, 1], [1u32, 2, 3]];
    // 200 roots (> 3 workgroups of 64) scattered above the two faces, with a
    // deterministic sprinkling of unbound sentinels to exercise both branches.
    let n = 200usize;
    let mut roots = Vec::with_capacity(n);
    for i in 0..n {
        let fx = ((i * 7 + 3) % 100) as f32 / 100.0;
        let fz = ((i * 13 + 5) % 100) as f32 / 100.0;
        let fy = ((i * 3 + 1) % 50) as f32 / 100.0;
        roots.push(v(fx, fy, fz));
    }
    let mut bindings = bind_roots(&roots, &verts, &tris);
    for (i, b) in bindings.iter_mut().enumerate() {
        if i % 17 == 0 {
            *b = MeshBinding {
                triangle: UNBOUND,
                bary: [0.0, 0.0, 0.0],
                height: 0.0,
            };
        }
    }
    assert_eq!(bindings.len(), n);
    assert_batch_parity(&ctx, &kernel, &bindings, &verts, &tris, "large batch");
}

#[test]
fn empty_bindings_is_a_no_op() {
    let Some(ctx) = context_or_skip("root_resolve empty no-op") else {
        return;
    };
    let kernel = GpuHairRootResolve::new(&ctx);
    let (verts, tris) = unit_scalp();
    let gpu = kernel.eval(&ctx, &[], &verts, &tris);
    assert!(gpu.is_empty(), "empty binding list yields no frames");
}

#[test]
fn matches_bind_roots_round_trip_batch() {
    let Some(ctx) = context_or_skip("root_resolve round-trip parity") else {
        return;
    };
    let kernel = GpuHairRootResolve::new(&ctx);
    let (verts, tris) = unit_scalp();
    // A general batch of authored roots (some on-surface, some floated off) bound
    // through the real import bake, then resolved on both sides.
    let roots = vec![
        v(0.2, 0.0, 0.2),
        v(0.3, 0.5, 0.1),
        v(0.1, 0.9, 0.4),
        v(0.45, 0.0, 0.45),
    ];
    let bindings = bind_roots(&roots, &verts, &tris);
    assert!(bindings.iter().all(MeshBinding::is_bound));
    assert_batch_parity(&ctx, &kernel, &bindings, &verts, &tris, "round trip");
    let gpu = kernel.eval(&ctx, &bindings, &verts, &tris);
    for (i, (frame, root)) in gpu.iter().zip(roots.iter()).enumerate() {
        assert_vec_close(frame.position, *root, &format!("round-trip root {i}"));
    }
}
