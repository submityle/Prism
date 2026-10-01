//! Real-device parity for the follicle local-frame transfer twin:
//! [`GpuHairTransferFrame`] must reproduce the `CPU` golden
//! [`reference_transfer_frame`](prism_hair_gpu::transfer_frame::reference_transfer_frame)
//! (which forwards
//! [`transfer_frame`](prism_render_architecture::hair::follicle_bind::transfer_frame))
//! for a batch of bindings broadcast against one shared deformed triangle,
//! rebuilding each binding's orthonormal [`FollicleFrame`] (normal, tangent,
//! bitangent) independently and in order. The suite drives a regular orthonormal
//! triangle, an unnormalised / non-orthogonal input frame (Gram-Schmidt must
//! re-orthonormalise), a normal pointing (near) `+X` so the fallback-tangent
//! branch selected by `|normal.x| >= 0.9` is taken, an all-non-positive weight
//! set (sanitises to the centroid), a zero-length interpolated normal (falls
//! back to `+Z`), a tangent parallel to the normal (falls back to a canonical
//! axis), the empty no-op, and a large multi-workgroup batch crossing the
//! 64-wide dispatch boundary.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each component is a barycentric interpolation fold, a Gram-Schmidt
//! projection and `sqrt`-based normalises a `GPU` may fuse, so every value is
//! asserted within `abs_diff < 1e-4` or `rel_diff < 1e-3` rather than
//! bit-for-bit. Every output component is also asserted finite, and the basis is
//! checked for unit length and mutual orthogonality (`dot(n, t) ~= 0`,
//! `dot(v, v) ~= 1`). All inputs are explicit literals or integer-derived
//! fractions and are kept finite so the `CPU` and `GPU` walk the identical
//! branch of the sanitiser and fallbacks; no `sin`/`cos` appears anywhere.
//!
//! Provenance: standard barycentric mesh-attachment frame transfer plus
//! Gram-Schmidt orthonormalisation; no Unreal Engine source or derived code.

use prism_hair_gpu::transfer_frame::{reference_transfer_frame, GpuHairTransferFrame};
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::follicle_bind::{
    Barycentric, FollicleBinding, FollicleFrame, TriangleFrame,
};

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

/// Builds a binding from raw barycentric weights (the normal offset is unused by
/// the frame transfer, so any finite value is fine).
fn binding(u: f32, v: f32, w: f32) -> FollicleBinding {
    FollicleBinding {
        bary: Barycentric { u, v, w },
        normal_offset: 0.5,
    }
}

/// Asserts a whole batch of frames matches the `CPU` golden element by element,
/// component by component, that every component is finite, and that each basis is
/// (near) orthonormal.
fn assert_batch(got: &[FollicleFrame], bindings: &[FollicleBinding], tri: TriangleFrame) {
    assert_eq!(got.len(), bindings.len(), "one frame per binding");
    for (i, (out, &b)) in got.iter().zip(bindings.iter()).enumerate() {
        let reference = reference_transfer_frame(b, tri);
        for axis in 0..3 {
            assert_close(
                out.normal[axis],
                reference.normal[axis],
                &format!("binding {i} normal[{axis}]"),
            );
            assert_close(
                out.tangent[axis],
                reference.tangent[axis],
                &format!("binding {i} tangent[{axis}]"),
            );
            assert_close(
                out.bitangent[axis],
                reference.bitangent[axis],
                &format!("binding {i} bitangent[{axis}]"),
            );
            assert!(
                out.normal[axis].is_finite()
                    && out.tangent[axis].is_finite()
                    && out.bitangent[axis].is_finite(),
                "binding {i} component {axis} finite"
            );
        }
        // The basis the golden emits is orthonormal; the twin must agree.
        let dot = |a: [f32; 3], c: [f32; 3]| a[0] * c[0] + a[1] * c[1] + a[2] * c[2];
        assert_close(
            dot(out.normal, out.normal),
            1.0,
            &format!("binding {i} |n|"),
        );
        assert_close(
            dot(out.tangent, out.tangent),
            1.0,
            &format!("binding {i} |t|"),
        );
        assert_close(
            dot(out.bitangent, out.bitangent),
            1.0,
            &format!("binding {i} |b|"),
        );
        assert_close(
            dot(out.normal, out.tangent),
            0.0,
            &format!("binding {i} n.t"),
        );
    }
}

/// Dispatches one batch through the device twin.
fn run(ctx: &GpuContext, bindings: &[FollicleBinding], tri: TriangleFrame) -> Vec<FollicleFrame> {
    GpuHairTransferFrame::new(ctx).eval(ctx, bindings, tri)
}

/// A flat right-triangle in the z = 0 plane with uniform `+Z` normals and `+X`
/// tangents (already orthonormal).
fn flat_tri() -> TriangleFrame {
    TriangleFrame {
        positions: [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        normals: [[0.0, 0.0, 1.0]; 3],
        tangents: [[1.0, 0.0, 0.0]; 3],
    }
}

#[test]
fn orthonormal_triangle_transfers_frame() {
    let Some(ctx) = context_or_skip("orthonormal_triangle_transfers_frame") else {
        return;
    };
    let tri = flat_tri();
    let bindings = [
        binding(1.0, 0.0, 0.0),
        binding(0.0, 1.0, 0.0),
        binding(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        binding(0.5, 0.25, 0.25),
    ];
    let got = run(&ctx, &bindings, tri);
    assert_batch(&got, &bindings, tri);
    // On this orthonormal triangle the frame is just (+X, +Y, +Z) regardless of
    // weights.
    assert_close(got[0].normal[2], 1.0, "flat normal z");
    assert_close(got[0].tangent[0], 1.0, "flat tangent x");
    assert_close(got[0].bitangent[1], 1.0, "flat bitangent y");
}

#[test]
fn unnormalised_non_orthogonal_frame_is_reorthonormalised() {
    let Some(ctx) = context_or_skip("unnormalised_non_orthogonal_frame_is_reorthonormalised")
    else {
        return;
    };
    // Per-vertex normals/tangents are neither unit nor orthogonal: Gram-Schmidt
    // must still produce an orthonormal basis matching the golden.
    let tri = TriangleFrame {
        positions: [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        normals: [[0.0, 0.0, 3.0], [0.1, 0.0, 2.5], [-0.1, 0.2, 2.0]],
        tangents: [[2.0, 0.3, 0.5], [1.5, -0.2, 0.1], [1.0, 0.4, -0.2]],
    };
    let bindings = [
        binding(0.6, 0.3, 0.1),
        binding(0.2, 0.2, 0.6),
        binding(0.34, 0.33, 0.33),
    ];
    assert_batch(&run(&ctx, &bindings, tri), &bindings, tri);
}

#[test]
fn normal_near_plus_x_takes_fallback_branch() {
    let Some(ctx) = context_or_skip("normal_near_plus_x_takes_fallback_branch") else {
        return;
    };
    // Normals point along +X and the tangent is parallel to the normal, so the
    // projected tangent is (near) zero and the kernel must take the
    // |normal.x| >= 0.9 fallback branch (cross with +Y).
    let tri = TriangleFrame {
        positions: [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        normals: [[1.0, 0.0, 0.0]; 3],
        tangents: [[1.0, 0.0, 0.0]; 3],
    };
    let bindings = [binding(1.0, 0.0, 0.0), binding(0.3, 0.3, 0.4)];
    let got = run(&ctx, &bindings, tri);
    assert_batch(&got, &bindings, tri);
    assert_close(got[0].normal[0], 1.0, "fallback normal x");
}

#[test]
fn degenerate_weights_fall_back_to_centroid() {
    let Some(ctx) = context_or_skip("degenerate_weights_fall_back_to_centroid") else {
        return;
    };
    let tri = flat_tri();
    // All-non-positive weights sanitise to the centroid; on the flat triangle the
    // frame is still (+X, +Y, +Z).
    let bindings = [binding(0.0, 0.0, 0.0), binding(-1.0, -2.0, -3.0)];
    let got = run(&ctx, &bindings, tri);
    assert_batch(&got, &bindings, tri);
    assert_close(got[0].normal[2], 1.0, "centroid normal z");
}

#[test]
fn zero_length_normal_falls_back_to_plus_z() {
    let Some(ctx) = context_or_skip("zero_length_normal_falls_back_to_plus_z") else {
        return;
    };
    // Per-vertex normals cancel to zero under interpolation, so the normalise
    // falls back to +Z (and the tangent/bitangent follow the golden).
    let tri = TriangleFrame {
        positions: [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        normals: [[0.0, 0.0, 0.0]; 3],
        tangents: [[1.0, 0.0, 0.0]; 3],
    };
    let bindings = [binding(0.5, 0.3, 0.2)];
    let got = run(&ctx, &bindings, tri);
    assert_batch(&got, &bindings, tri);
    assert_close(got[0].normal[2], 1.0, "fallback normal z");
}

#[test]
fn zero_length_tangent_falls_back() {
    let Some(ctx) = context_or_skip("zero_length_tangent_falls_back") else {
        return;
    };
    // The interpolated tangent is zero, so the projected tangent is zero and the
    // kernel must take the canonical-axis fallback (normal is +Z, so |x| < 0.9
    // branch: cross(+Z, +X) = +Y).
    let tri = TriangleFrame {
        positions: [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        normals: [[0.0, 0.0, 1.0]; 3],
        tangents: [[0.0, 0.0, 0.0]; 3],
    };
    let bindings = [binding(0.4, 0.4, 0.2)];
    let got = run(&ctx, &bindings, tri);
    assert_batch(&got, &bindings, tri);
    assert_close(got[0].tangent[1], 1.0, "fallback tangent y");
}

#[test]
fn empty_batch_is_noop() {
    let Some(ctx) = context_or_skip("empty_batch_is_noop") else {
        return;
    };
    let got = run(&ctx, &[], flat_tri());
    assert!(got.is_empty(), "empty batch yields no frames");
}

#[test]
fn large_batch_crosses_workgroup_boundary() {
    let Some(ctx) = context_or_skip("large_batch_crosses_workgroup_boundary") else {
        return;
    };
    // 200 bindings span four 64-wide workgroups over a deterministic sweep of
    // integer-derived weights against a tilted, unnormalised input frame.
    let tri = TriangleFrame {
        positions: [[0.0, 0.0, 0.0], [4.0, 1.0, 0.0], [1.0, 4.0, 2.0]],
        normals: [[0.0, 0.1, 1.0], [0.1, 0.0, 1.0], [-0.1, -0.1, 1.0]],
        tangents: [[1.0, 0.2, 0.0], [0.9, -0.1, 0.1], [1.1, 0.0, -0.1]],
    };
    let mut bindings = Vec::new();
    for k in 0u32..200 {
        let u = (k % 7) as f32 + 1.0;
        let v = (k % 5) as f32 + 1.0;
        let w = (k % 3) as f32 + 1.0;
        bindings.push(binding(u, v, w));
    }
    assert_batch(&run(&ctx, &bindings, tri), &bindings, tri);
}
