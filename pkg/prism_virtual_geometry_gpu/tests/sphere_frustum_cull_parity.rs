//! Real-device parity for the sphere-frustum-cull twin: [`GpuSphereFrustumCull`]
//! must reproduce the CPU golden
//! [`Frustum::contains_sphere`](prism_render_architecture::virtual_geometry::Frustum::contains_sphere)
//! for every bounding sphere.
//!
//! The test skips (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable
//! core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! A cull verdict is a discrete decision derived from sign comparisons, not a
//! continuous value. Every sphere in the scene either sits deep inside the
//! frustum, reaches across exactly one face by a wide radius margin (the
//! sphere-specific partial-inclusion case an AABB test cannot express the same
//! way), or falls fully outside one face by a large margin. Each sphere clears
//! its boundary by margins far larger than any float-reassociation error, so
//! the integer verdict is stable under any legal fused-multiply-add
//! contraction and is asserted index-for-index against
//! `contains_sphere(..) as u32` with no tolerance. The scene exercises both the
//! inside (`1`) and outside (`0`) verdicts, each of the six faces in isolation,
//! the radius-reaches-across-a-face partial case, a `>64`-invocation dispatch
//! that spans multiple workgroups, and the empty input.
//!
//! Provenance: standard sphere-frustum half-space culling; no Unreal Engine
//! source or derived code.

use std::collections::BTreeSet;

use prism_render_architecture::virtual_geometry::{Frustum, Plane};
use prism_virtual_geometry_gpu::{GpuContext, GpuSphereFrustumCull, SphereQuery};

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

fn sphere(center: [f32; 3], radius: f32) -> SphereQuery {
    SphereQuery { center, radius }
}

/// A scene spanning both verdicts. Three inside cases: one deep inside, and two
/// partial cases where the centre sits just outside a face but the radius
/// reaches in by a wide margin (the sphere-specific behaviour). Six outside
/// cases: one sphere pushed far past each of the six faces so that face rejects
/// in isolation. Every sphere clears its boundary by a large margin.
fn scene() -> Vec<SphereQuery> {
    vec![
        // Deep inside.
        sphere([0.0, 0.0, 50.0], 1.0),
        // Centre just outside the right face (x <= 10) but radius reaches in:
        // -x plane signed distance = -11 + 10 = -1 >= -5, margin 4.
        sphere([11.0, 0.0, 50.0], 5.0),
        // Centre just behind the near face (z >= 0) but radius reaches in:
        // near plane signed distance = -2 >= -5, margin 3.
        sphere([0.0, 0.0, -2.0], 5.0),
        // Fully outside each of the six faces (margin ~89).
        sphere([100.0, 0.0, 50.0], 1.0),  // past -x face
        sphere([-100.0, 0.0, 50.0], 1.0), // past +x face
        sphere([0.0, 100.0, 50.0], 1.0),  // past -y face
        sphere([0.0, -100.0, 50.0], 1.0), // past +y face
        sphere([0.0, 0.0, -50.0], 1.0),   // behind the near face
        sphere([0.0, 0.0, 200.0], 1.0),   // past the far face
    ]
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_sphere_frustum_cull_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping sphere-frustum-cull parity: no wgpu adapter on this host");
        return;
    };
    let culler = GpuSphereFrustumCull::new(&ctx);
    let frustum = box_frustum();
    let spheres = scene();

    let gpu = culler.cull(&ctx, &frustum, &spheres);
    assert_eq!(gpu.len(), spheres.len(), "one verdict per sphere");

    // Track which verdicts actually appear so a degenerate scene that only ever
    // emits one verdict cannot pass vacuously.
    let mut seen: BTreeSet<u32> = BTreeSet::new();
    for (i, s) in spheres.iter().enumerate() {
        let expected = u32::from(frustum.contains_sphere(s.center, s.radius));
        assert_eq!(
            gpu[i], expected,
            "verdict mismatch for sphere {s:?}: gpu {}, cpu {expected}",
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
    let culler = GpuSphereFrustumCull::new(&ctx);
    let frustum = box_frustum();
    let out = culler.cull(&ctx, &frustum, &[]);
    assert!(out.is_empty(), "no spheres cull to no verdicts");
}

#[test]
fn each_frustum_face_culls_independently() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let culler = GpuSphereFrustumCull::new(&ctx);
    let frustum = box_frustum();
    // One sphere pushed well past each face, so every face must reject in
    // isolation; all six are expected outside (0).
    let spheres = vec![
        sphere([100.0, 0.0, 50.0], 1.0),
        sphere([-100.0, 0.0, 50.0], 1.0),
        sphere([0.0, 100.0, 50.0], 1.0),
        sphere([0.0, -100.0, 50.0], 1.0),
        sphere([0.0, 0.0, -50.0], 1.0),
        sphere([0.0, 0.0, 200.0], 1.0),
    ];
    let gpu = culler.cull(&ctx, &frustum, &spheres);
    let expected: Vec<u32> = spheres
        .iter()
        .map(|s| u32::from(frustum.contains_sphere(s.center, s.radius)))
        .collect();
    assert_eq!(gpu, expected, "each face must cull identically to the golden");
    assert_eq!(gpu, vec![0, 0, 0, 0, 0, 0], "every face rejects its sphere");
}

#[test]
fn multi_workgroup_dispatch_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let culler = GpuSphereFrustumCull::new(&ctx);
    let frustum = box_frustum();
    // 130 spheres span three workgroups (workgroup_size 64), alternating a
    // deep-inside sphere with a far-outside one so tiling and per-thread
    // indexing are both exercised and both verdicts appear across the tiles.
    let mut spheres = Vec::with_capacity(130);
    for i in 0..130u32 {
        if i % 2 == 0 {
            spheres.push(sphere([0.0, 0.0, 50.0], 1.0)); // inside
        } else {
            spheres.push(sphere([1000.0, 0.0, 50.0], 1.0)); // outside
        }
    }
    let gpu = culler.cull(&ctx, &frustum, &spheres);
    let expected: Vec<u32> = spheres
        .iter()
        .map(|s| u32::from(frustum.contains_sphere(s.center, s.radius)))
        .collect();
    assert_eq!(gpu.len(), 130, "one verdict per sphere across workgroups");
    assert_eq!(gpu, expected, "tiled dispatch must match the golden");
}
