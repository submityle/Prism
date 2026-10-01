//! Real-device parity for the isolated follicle root-transfer twin:
//! [`GpuHairFollicleBind`] must reproduce the `CPU` golden
//! [`reference_transfer_root_map`](prism_hair_gpu::follicle_bind::reference_transfer_root_map)
//! (which forwards
//! [`transfer_root_map`](prism_render_architecture::hair::follicle_bind::transfer_root_map))
//! for a batch of follicle bindings broadcast against one shared deformed
//! triangle, mapping each binding to its transferred world root position
//! independently and in order. The suite drives vertex recovery (a weight set
//! collapsing onto a single vertex), the centroid fallback for an
//! all-non-positive weight set, positive / negative / zero normal offsets, a
//! zero-length interpolated normal falling back to `+Z`, a non-planar triangle
//! with an arbitrary offset, a mixed batch, the empty no-op, and a large
//! multi-workgroup batch that crosses the 64-wide dispatch boundary.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each component is a barycentric fold plus a `sqrt`-based normalise and a
//! multiply-add a `GPU` may fuse, so every value is asserted within
//! `abs_diff < 1e-4` or `rel_diff < 1e-3` rather than bit-for-bit. Every output
//! component is also asserted finite. All inputs are explicit literals or
//! integer-derived fractions and are kept finite so the `CPU` and `GPU` walk the
//! identical branch of the sanitiser; no `sin`/`cos` appears anywhere.
//!
//! Provenance: standard barycentric mesh-attachment transfer plus `wgpu`
//! compute dispatch; no Unreal Engine source or derived code.

use prism_hair_gpu::follicle_bind::{reference_transfer_root_map, GpuHairFollicleBind};
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::follicle_bind::{Barycentric, FollicleBinding, TriangleFrame};

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

/// Asserts a whole batch of transferred root positions matches the `CPU` golden
/// element by element, component by component, and that every component is
/// finite.
fn assert_batch(got: &[[f32; 3]], bindings: &[FollicleBinding], deformed: TriangleFrame) {
    assert_eq!(
        got.len(),
        bindings.len(),
        "one transferred root position per binding"
    );
    let want = reference_transfer_root_map(bindings, deformed);
    assert_eq!(got.len(), want.len(), "golden length must match");
    for (i, (out, reference)) in got.iter().zip(want.iter()).enumerate() {
        for (axis, (&o, &r)) in out.iter().zip(reference.iter()).enumerate() {
            assert_close(o, r, &format!("element {i} axis {axis}"));
            assert!(
                o.is_finite(),
                "element {i} axis {axis} must be finite, got {o}"
            );
        }
    }
}

/// Dispatches one batch through the device twin.
fn run(ctx: &GpuContext, bindings: &[FollicleBinding], deformed: TriangleFrame) -> Vec<[f32; 3]> {
    GpuHairFollicleBind::new(ctx).eval(ctx, bindings, deformed)
}

/// A flat right-triangle in the z = 0 plane with uniform `+Z` normals.
fn flat_tri() -> TriangleFrame {
    TriangleFrame {
        positions: [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        normals: [[0.0, 0.0, 1.0]; 3],
        tangents: [[1.0, 0.0, 0.0]; 3],
    }
}

/// Builds a binding from explicit barycentric weights and a signed offset.
fn binding(u: f32, v: f32, w: f32, normal_offset: f32) -> FollicleBinding {
    FollicleBinding {
        bary: Barycentric { u, v, w },
        normal_offset,
    }
}

#[test]
fn vertex_recovery_with_offset() {
    let Some(ctx) = context_or_skip("vertex_recovery_with_offset") else {
        return;
    };
    let tri = flat_tri();
    // A pure weight on each vertex lands on that vertex, floated +0.5 along +Z.
    let bindings = [
        binding(1.0, 0.0, 0.0, 0.5),
        binding(0.0, 1.0, 0.0, 0.5),
        binding(0.0, 0.0, 1.0, 0.5),
    ];
    let got = run(&ctx, &bindings, tri);
    assert_batch(&got, &bindings, tri);
    assert_close(got[0][0], 0.0, "vtx0 x");
    assert_close(got[0][2], 0.5, "vtx0 z");
    assert_close(got[1][0], 1.0, "vtx1 x");
    assert_close(got[2][1], 1.0, "vtx2 y");
}

#[test]
fn all_non_positive_falls_back_to_centroid() {
    let Some(ctx) = context_or_skip("all_non_positive_falls_back_to_centroid") else {
        return;
    };
    let tri = flat_tri();
    // Every weight non-positive -> sanitiser collapses to the centroid (1/3 each),
    // so the surface point is the triangle centroid, floated along +Z.
    let bindings = [binding(0.0, 0.0, 0.0, 0.0), binding(-1.0, -2.0, -0.5, 0.25)];
    let got = run(&ctx, &bindings, tri);
    assert_batch(&got, &bindings, tri);
    let third = 1.0 / 3.0;
    assert_close(got[0][0], third, "centroid x");
    assert_close(got[0][1], third, "centroid y");
    assert_close(got[0][2], 0.0, "centroid z");
}

#[test]
fn negative_weight_clamps_and_renormalises() {
    let Some(ctx) = context_or_skip("negative_weight_clamps_and_renormalises") else {
        return;
    };
    let tri = flat_tri();
    // u clamps to 0, (v, w) renormalise to (0.25, 0.75): surface = (0.25, 0.75, 0).
    let bindings = [binding(-1.0, 1.0, 3.0, 0.0)];
    let got = run(&ctx, &bindings, tri);
    assert_batch(&got, &bindings, tri);
    // surface = p1 * 0.25 + p2 * 0.75 = (0.25, 0.75, 0).
    assert_close(got[0][0], 0.25, "x");
    assert_close(got[0][1], 0.75, "y");
}

#[test]
fn signed_offsets_track_normal() {
    let Some(ctx) = context_or_skip("signed_offsets_track_normal") else {
        return;
    };
    let tri = flat_tri();
    // Same interior point, three offsets: below, on, and above the surface.
    let bindings = [
        binding(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, -0.75),
        binding(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0),
        binding(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 2.0),
    ];
    let got = run(&ctx, &bindings, tri);
    assert_batch(&got, &bindings, tri);
    assert_close(got[0][2], -0.75, "below z");
    assert_close(got[1][2], 0.0, "on z");
    assert_close(got[2][2], 2.0, "above z");
}

#[test]
fn zero_length_normal_falls_back_to_plus_z() {
    let Some(ctx) = context_or_skip("zero_length_normal_falls_back_to_plus_z") else {
        return;
    };
    // All per-vertex normals are zero: the interpolated normal is zero-length and
    // falls back to the canonical +Z axis, so the offset floats along +Z.
    let tri = TriangleFrame {
        positions: [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        normals: [[0.0, 0.0, 0.0]; 3],
        tangents: [[1.0, 0.0, 0.0]; 3],
    };
    let bindings = [binding(0.5, 0.25, 0.25, 1.5)];
    let got = run(&ctx, &bindings, tri);
    assert_batch(&got, &bindings, tri);
    assert_close(got[0][2], 1.5, "fallback +Z offset");
}

#[test]
fn non_planar_triangle_arbitrary_offset() {
    let Some(ctx) = context_or_skip("non_planar_triangle_arbitrary_offset") else {
        return;
    };
    // A triangle lifted out of any axis plane with non-uniform (unnormalised)
    // per-vertex normals; several interior weights and offsets. Correctness is
    // delegated to the golden via assert_batch.
    let tri = TriangleFrame {
        positions: [[1.0, -2.0, 0.5], [-1.5, 0.0, 2.0], [0.25, 3.0, -1.0]],
        normals: [[0.2, 1.0, 0.3], [-0.4, 0.8, 0.1], [0.1, 0.9, -0.5]],
        tangents: [[1.0, 0.0, 0.0]; 3],
    };
    let bindings = [
        binding(0.6, 0.3, 0.1, 0.4),
        binding(0.1, 0.1, 0.8, -1.2),
        binding(0.33, 0.33, 0.34, 0.0),
        binding(2.0, 0.0, 0.0, 3.0),
    ];
    assert_batch(&run(&ctx, &bindings, tri), &bindings, tri);
}

#[test]
fn empty_batch_is_noop() {
    let Some(ctx) = context_or_skip("empty_batch_is_noop") else {
        return;
    };
    let got = run(&ctx, &[], flat_tri());
    assert!(got.is_empty(), "empty batch yields no positions");
}

#[test]
fn large_batch_crosses_workgroup_boundary() {
    let Some(ctx) = context_or_skip("large_batch_crosses_workgroup_boundary") else {
        return;
    };
    // 130 bindings span three 64-wide workgroups over a deterministic sweep of
    // integer-derived weights and offsets (with every 11th slot forced to an
    // all-zero weight set to exercise the centroid fallback across the boundary).
    let tri = TriangleFrame {
        positions: [[0.0, 0.0, 0.0], [4.0, 1.0, 0.0], [1.0, 4.0, 2.0]],
        normals: [[0.0, 0.1, 1.0], [0.1, 0.0, 1.0], [-0.1, -0.1, 1.0]],
        tangents: [[1.0, 0.0, 0.0]; 3],
    };
    let mut bindings = Vec::new();
    for k in 0u32..130 {
        if k % 11 == 0 {
            bindings.push(binding(0.0, 0.0, 0.0, (k % 5) as f32 / 2.0));
        } else {
            let u = (k % 7) as f32 / 6.0;
            let v = (k % 5) as f32 / 4.0;
            let w = (k % 3) as f32 / 2.0;
            let offset = (k % 9) as f32 / 3.0 - 1.0;
            bindings.push(binding(u, v, w, offset));
        }
    }
    assert_batch(&run(&ctx, &bindings, tri), &bindings, tri);
}
