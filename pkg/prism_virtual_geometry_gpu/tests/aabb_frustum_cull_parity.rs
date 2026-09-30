//! Real-device parity for the AABB-frustum-cull twin: [`GpuAabbFrustumCull`]
//! must reproduce the CPU golden
//! [`Frustum::intersects_bounds`](prism_render_architecture::virtual_geometry::Frustum::intersects_bounds)
//! for every axis-aligned bounding box.
//!
//! The test skips (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable
//! core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! A cull verdict is a discrete decision derived from sign comparisons, not a
//! continuous value. Every box in the scene either sits deep inside the
//! frustum, reaches across exactly one face by a wide projected-extent margin
//! (the box-specific partial-inclusion case that distinguishes the AABB test
//! from the sphere test), or falls fully outside one face by a large margin.
//! Each box clears its boundary by margins far larger than any
//! float-reassociation error, so the integer verdict is stable under any legal
//! fused-multiply-add contraction and is asserted index-for-index against
//! `intersects_bounds(..) as u32` with no tolerance. The scene exercises both
//! the inside (`1`) and outside (`0`) verdicts, each of the six faces in
//! isolation, the half-extent-reaches-across-a-face partial case, a
//! `>64`-invocation dispatch that spans multiple workgroups, and the empty
//! input.
//!
//! Provenance: standard AABB projected-radius frustum culling; no Unreal Engine
//! source or derived code.

extern crate alloc;

use alloc::collections::BTreeSet;

use prism_render_architecture::virtual_geometry::{Frustum, Plane};
use prism_virtual_geometry_gpu::{bounds_of, AabbQuery, GpuAabbFrustumCull, GpuContext};

/// Axis-aligned box frustum: `-10 <= x <= 10`, `-10 <= y <= 10`,
/// `0 <= z <= 100`, all six planes inward-facing and unit-normal, matching the
/// golden's own test frustum.
fn box_frustum() -> Frustum {
    Frustum::from_planes([
        Plane::new([1.0, 0.0, 0.0], 10.0),
        Plane::new([-1.0, 0.0, 0.0], 10.0),
        Plane::new([0.0, 1.0, 0.0], 10.0),
        Plane::new([0.0, -1.0, 0.0], 10.0),
        Plane::new([0.0, 0.0, 1.0], 0.0),
        Plane::new([0.0, 0.0, -1.0], 100.0),
    ])
}

fn aabb(center: [f32; 3], half_extents: [f32; 3]) -> AabbQuery {
    AabbQuery {
        center,
        half_extents,
    }
}

/// A scene spanning both verdicts. Three inside cases: one deep inside, and two
/// partial cases where the centre sits just outside a face but the projected
/// half-extent reaches in by a wide margin (the box-specific behaviour that a
/// bounding-sphere test cannot express with a single radius). Six outside
/// cases: one box pushed far past each of the six faces so that face rejects in
/// isolation. Every box clears its boundary by a large margin.
fn scene() -> Vec<AabbQuery> {
    vec![
        // Deep inside.
        aabb([0.0, 0.0, 50.0], [1.0, 1.0, 1.0]),
        // Centre just outside the right face (x <= 10) but the x half-extent
        // reaches in: -x plane signed distance = -1, projected extent = 5,
        // -1 >= -5, margin 4.
        aabb([11.0, 0.0, 50.0], [5.0, 1.0, 1.0]),
        // Centre just behind the near face (z >= 0) but the z half-extent
        // reaches in: near plane signed distance = -2, projected extent = 5,
        // -2 >= -5, margin 3.
        aabb([0.0, 0.0, -2.0], [1.0, 1.0, 5.0]),
        // Fully outside each of the six faces (margin ~89).
        aabb([100.0, 0.0, 50.0], [1.0, 1.0, 1.0]),  // past -x face
        aabb([-100.0, 0.0, 50.0], [1.0, 1.0, 1.0]), // past +x face
        aabb([0.0, 100.0, 50.0], [1.0, 1.0, 1.0]),  // past -y face
        aabb([0.0, -100.0, 50.0], [1.0, 1.0, 1.0]), // past +y face
        aabb([0.0, 0.0, -50.0], [1.0, 1.0, 1.0]),   // behind the near face
        aabb([0.0, 0.0, 200.0], [1.0, 1.0, 1.0]),   // past the far face
    ]
}

fn expected(frustum: &Frustum, query: &AabbQuery) -> u32 {
    u32::from(frustum.intersects_bounds(&bounds_of(query)))
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_aabb_frustum_cull_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping aabb-frustum-cull parity: no wgpu adapter on this host");
        return;
    };
    let culler = GpuAabbFrustumCull::new(&ctx);
    let frustum = box_frustum();
    let boxes = scene();

    let gpu = culler.cull(&ctx, &frustum, &boxes);
    assert_eq!(gpu.len(), boxes.len(), "one verdict per box");

    // Track which verdicts actually appear so a degenerate scene that only ever
    // emits one verdict cannot pass vacuously.
    let mut seen: BTreeSet<u32> = BTreeSet::new();
    for (i, b) in boxes.iter().enumerate() {
        let want = expected(&frustum, b);
        assert_eq!(
            gpu[i], want,
            "verdict mismatch for box {b:?}: gpu {}, cpu {want}",
            gpu[i]
        );
        seen.insert(gpu[i]);
    }

    // The scene must exercise both inside (1) and outside (0).
    assert_eq!(
        seen,
        BTreeSet::from([0, 1]),
        "scene must exercise both verdicts, saw {seen:?}"
    );
}

#[test]
fn empty_scene_culls_to_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let culler = GpuAabbFrustumCull::new(&ctx);
    let frustum = box_frustum();
    let out = culler.cull(&ctx, &frustum, &[]);
    assert!(out.is_empty(), "no boxes cull to no verdicts");
}

#[test]
fn each_frustum_face_culls_independently() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let culler = GpuAabbFrustumCull::new(&ctx);
    let frustum = box_frustum();
    // One box pushed well past each face, so every face must reject in
    // isolation; all six are expected outside (0).
    let boxes = vec![
        aabb([100.0, 0.0, 50.0], [1.0, 1.0, 1.0]),
        aabb([-100.0, 0.0, 50.0], [1.0, 1.0, 1.0]),
        aabb([0.0, 100.0, 50.0], [1.0, 1.0, 1.0]),
        aabb([0.0, -100.0, 50.0], [1.0, 1.0, 1.0]),
        aabb([0.0, 0.0, -50.0], [1.0, 1.0, 1.0]),
        aabb([0.0, 0.0, 200.0], [1.0, 1.0, 1.0]),
    ];
    let gpu = culler.cull(&ctx, &frustum, &boxes);
    let want: Vec<u32> = boxes.iter().map(|b| expected(&frustum, b)).collect();
    assert_eq!(gpu, want, "each face must cull identically to the golden");
    assert_eq!(gpu, vec![0, 0, 0, 0, 0, 0], "every face rejects its box");
}

#[test]
fn multi_workgroup_dispatch_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let culler = GpuAabbFrustumCull::new(&ctx);
    let frustum = box_frustum();
    // 130 boxes span three workgroups (workgroup_size 64), alternating a
    // deep-inside box with a far-outside one so tiling and per-thread indexing
    // are both exercised and both verdicts appear across the tiles.
    let mut boxes = Vec::with_capacity(130);
    for i in 0..130u32 {
        if i % 2 == 0 {
            boxes.push(aabb([0.0, 0.0, 50.0], [1.0, 1.0, 1.0])); // inside
        } else {
            boxes.push(aabb([1000.0, 0.0, 50.0], [1.0, 1.0, 1.0])); // outside
        }
    }
    let gpu = culler.cull(&ctx, &frustum, &boxes);
    let want: Vec<u32> = boxes.iter().map(|b| expected(&frustum, b)).collect();
    assert_eq!(gpu.len(), 130, "one verdict per box across workgroups");
    assert_eq!(gpu, want, "tiled dispatch must match the golden");
}
