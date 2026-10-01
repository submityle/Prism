//! Real-device parity for the isolated follicle-binding capture twin:
//! [`GpuHairBindFollicle`] must reproduce the `CPU` golden
//! [`reference_bind_follicle`](prism_hair_gpu::bind_follicle::reference_bind_follicle)
//! (which forwards
//! [`bind_follicle`](prism_render_architecture::hair::follicle_bind::bind_follicle))
//! for a batch of roots broadcast against one shared triangle, mapping each root
//! to its [`FollicleBinding`] (sanitised barycentric weights plus signed normal
//! offset) independently and in order. The suite drives vertex recovery (a root
//! on a vertex returns a unit weight and zero offset), an interior root floated
//! above / below the surface (signed offset tracks the normal), an exterior root
//! (weights are clamped on-surface, not left negative), a degenerate collinear
//! triangle and a collapsed triangle (both fall back to the centroid), a
//! non-planar triangle, the empty no-op, and a large multi-workgroup batch that
//! crosses the 64-wide dispatch boundary.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each component is a barycentric solve, a clamp/renormalise, a `sqrt`-based
//! normalise and multiply-adds a `GPU` may fuse, so every value is asserted
//! within `abs_diff < 1e-4` or `rel_diff < 1e-3` rather than bit-for-bit. Every
//! output component is also asserted finite. All inputs are explicit literals or
//! integer-derived fractions and are kept finite so the `CPU` and `GPU` walk the
//! identical branch of the sanitiser; no `sin`/`cos` appears anywhere.
//!
//! Provenance: Ericson "Real-Time Collision Detection" barycentric solve plus
//! standard mesh-skinning attachment; no Unreal Engine source or derived code.

use prism_hair_gpu::bind_follicle::{reference_bind_follicle, GpuHairBindFollicle};
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::follicle_bind::{FollicleBinding, TriangleFrame};

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

/// Asserts two scalars agree within the documented fma tolerance.
fn assert_close(got: f32, want: f32, what: &str) {
    let abs = (got - want).abs();
    let rel = abs / want.abs().max(1.0);
    assert!(
        abs < 1e-4 || rel < 1e-3,
        "{what}: got {got}, want {want} (abs {abs}, rel {rel})"
    );
}

/// Asserts a whole batch of bindings matches the `CPU` golden element by
/// element, component by component, and that every component is finite.
fn assert_batch(got: &[FollicleBinding], roots: &[[f32; 3]], tri: TriangleFrame) {
    assert_eq!(got.len(), roots.len(), "one binding per root");
    for (i, (out, &root)) in got.iter().zip(roots.iter()).enumerate() {
        let reference = reference_bind_follicle(root, tri);
        assert_close(out.bary.u, reference.bary.u, &format!("root {i} u"));
        assert_close(out.bary.v, reference.bary.v, &format!("root {i} v"));
        assert_close(out.bary.w, reference.bary.w, &format!("root {i} w"));
        assert_close(
            out.normal_offset,
            reference.normal_offset,
            &format!("root {i} offset"),
        );
        assert!(
            out.bary.u.is_finite(),
            "root {i} u finite, got {}",
            out.bary.u
        );
        assert!(
            out.bary.v.is_finite(),
            "root {i} v finite, got {}",
            out.bary.v
        );
        assert!(
            out.bary.w.is_finite(),
            "root {i} w finite, got {}",
            out.bary.w
        );
        assert!(
            out.normal_offset.is_finite(),
            "root {i} offset finite, got {}",
            out.normal_offset
        );
    }
}

/// Dispatches one batch through the device twin.
fn run(ctx: &GpuContext, roots: &[[f32; 3]], tri: TriangleFrame) -> Vec<FollicleBinding> {
    GpuHairBindFollicle::new(ctx).eval(ctx, roots, tri)
}

/// A flat right-triangle in the z = 0 plane with uniform `+Z` normals.
fn flat_tri() -> TriangleFrame {
    TriangleFrame {
        positions: [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        normals: [[0.0, 0.0, 1.0]; 3],
        tangents: [[1.0, 0.0, 0.0]; 3],
    }
}

#[test]
fn vertices_recover_unit_weights_zero_offset() {
    let Some(ctx) = context_or_skip("vertices_recover_unit_weights_zero_offset") else {
        return;
    };
    let tri = flat_tri();
    // A root sitting exactly on a vertex binds to that vertex with zero offset.
    let roots = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let got = run(&ctx, &roots, tri);
    assert_batch(&got, &roots, tri);
    assert_close(got[0].bary.u, 1.0, "vtx0 u");
    assert_close(got[0].normal_offset, 0.0, "vtx0 offset");
    assert_close(got[1].bary.v, 1.0, "vtx1 v");
    assert_close(got[2].bary.w, 1.0, "vtx2 w");
}

#[test]
fn interior_root_signed_offset_tracks_normal() {
    let Some(ctx) = context_or_skip("interior_root_signed_offset_tracks_normal") else {
        return;
    };
    let tri = flat_tri();
    // The same interior (x, y) floated above and below the z = 0 surface: the
    // offset is the signed height along +Z.
    let roots = [[0.25, 0.25, 0.75], [0.25, 0.25, -0.5], [0.25, 0.25, 0.0]];
    let got = run(&ctx, &roots, tri);
    assert_batch(&got, &roots, tri);
    assert_close(got[0].bary.u, 0.5, "interior u");
    assert_close(got[0].bary.v, 0.25, "interior v");
    assert_close(got[0].bary.w, 0.25, "interior w");
    assert_close(got[0].normal_offset, 0.75, "above offset");
    assert_close(got[1].normal_offset, -0.5, "below offset");
    assert_close(got[2].normal_offset, 0.0, "on offset");
}

#[test]
fn exterior_root_clamps_on_surface() {
    let Some(ctx) = context_or_skip("exterior_root_clamps_on_surface") else {
        return;
    };
    let tri = flat_tri();
    // A root whose planar projection lands outside the triangle: the raw weights
    // would go negative, but the binding sanitiser clamps and renormalises so the
    // stored weights are non-negative and sum to one.
    let roots = [[-0.5, -0.5, 1.0], [0.75, 0.75, 0.0]];
    let got = run(&ctx, &roots, tri);
    assert_batch(&got, &roots, tri);
    for (i, b) in got.iter().enumerate() {
        assert!(
            b.bary.u >= -1e-6,
            "root {i} u non-negative, got {}",
            b.bary.u
        );
        assert!(
            b.bary.v >= -1e-6,
            "root {i} v non-negative, got {}",
            b.bary.v
        );
        assert!(
            b.bary.w >= -1e-6,
            "root {i} w non-negative, got {}",
            b.bary.w
        );
        let sum = b.bary.u + b.bary.v + b.bary.w;
        assert_close(sum, 1.0, &format!("root {i} weight sum"));
    }
}

#[test]
fn collinear_triangle_falls_back_to_centroid() {
    let Some(ctx) = context_or_skip("collinear_triangle_falls_back_to_centroid") else {
        return;
    };
    // Three collinear vertices: zero area -> centroid weights for every root.
    let tri = TriangleFrame {
        positions: [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 2.0, 0.0]],
        normals: [[0.0, 0.0, 1.0]; 3],
        tangents: [[1.0, 0.0, 0.0]; 3],
    };
    let roots = [[0.5, 0.5, 1.0], [3.0, -1.0, -2.0]];
    let got = run(&ctx, &roots, tri);
    assert_batch(&got, &roots, tri);
    let third = 1.0 / 3.0;
    for (i, b) in got.iter().enumerate() {
        assert_close(b.bary.u, third, &format!("collinear {i} u"));
        assert_close(b.bary.v, third, &format!("collinear {i} v"));
        assert_close(b.bary.w, third, &format!("collinear {i} w"));
    }
}

#[test]
fn collapsed_triangle_falls_back_to_centroid() {
    let Some(ctx) = context_or_skip("collapsed_triangle_falls_back_to_centroid") else {
        return;
    };
    // All three vertices coincident: zero area -> centroid for every root.
    let tri = TriangleFrame {
        positions: [[1.0, 2.0, 3.0]; 3],
        normals: [[0.0, 0.0, 1.0]; 3],
        tangents: [[1.0, 0.0, 0.0]; 3],
    };
    let roots = [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]];
    let got = run(&ctx, &roots, tri);
    assert_batch(&got, &roots, tri);
    let third = 1.0 / 3.0;
    assert_close(got[0].bary.u, third, "collapsed u");
    assert_close(got[1].bary.w, third, "collapsed w");
}

#[test]
fn non_planar_triangle_binds() {
    let Some(ctx) = context_or_skip("non_planar_triangle_binds") else {
        return;
    };
    // A triangle lifted out of any axis plane with non-uniform (unnormalised)
    // per-vertex normals and several roots. Correctness is delegated to the
    // golden via assert_batch; roots are kept near the triangle.
    let tri = TriangleFrame {
        positions: [[1.0, -2.0, 0.5], [-1.5, 0.0, 2.0], [0.25, 3.0, -1.0]],
        normals: [[0.2, 1.0, 0.3], [-0.4, 0.8, 0.1], [0.1, 0.9, -0.5]],
        tangents: [[1.0, 0.0, 0.0]; 3],
    };
    let roots = [
        [0.0, 0.0, 0.0],
        [-0.5, 1.0, 1.0],
        [0.5, -0.5, 0.25],
        [-1.0, 2.0, -0.5],
    ];
    assert_batch(&run(&ctx, &roots, tri), &roots, tri);
}

#[test]
fn empty_batch_is_noop() {
    let Some(ctx) = context_or_skip("empty_batch_is_noop") else {
        return;
    };
    let got = run(&ctx, &[], flat_tri());
    assert!(got.is_empty(), "empty batch yields no bindings");
}

#[test]
fn large_batch_crosses_workgroup_boundary() {
    let Some(ctx) = context_or_skip("large_batch_crosses_workgroup_boundary") else {
        return;
    };
    // 200 roots span four 64-wide workgroups over a deterministic sweep of
    // integer-derived in-plane coordinates and signed heights (a few land outside
    // the triangle to exercise the on-surface clamp across the boundary).
    let tri = TriangleFrame {
        positions: [[0.0, 0.0, 0.0], [4.0, 1.0, 0.0], [1.0, 4.0, 2.0]],
        normals: [[0.0, 0.1, 1.0], [0.1, 0.0, 1.0], [-0.1, -0.1, 1.0]],
        tangents: [[1.0, 0.0, 0.0]; 3],
    };
    let mut roots = Vec::new();
    for k in 0u32..200 {
        let x = (k % 7) as f32 / 2.0 - 1.0;
        let y = (k % 5) as f32 / 2.0 - 1.0;
        let z = (k % 3) as f32 / 4.0 - 0.25;
        roots.push([x, y, z]);
    }
    assert_batch(&run(&ctx, &roots, tri), &roots, tri);
}
