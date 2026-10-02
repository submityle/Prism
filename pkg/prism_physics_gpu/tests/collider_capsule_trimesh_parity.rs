//! Real-device parity: the GPU capsule-versus-trimesh collider
//! ([`GpuCapsuleTrimeshCollider`]) must agree with the CPU golden
//! ([`cpu_capsule_trimesh_collide`]) on an actual Metal (or other `wgpu`)
//! adapter.
//!
//! Each test builds a mesh, its `LBVH`, and a capsule batch, runs both the CPU
//! golden and the device collider, and asserts the per-capsule manifolds match:
//! the same capsules contact, the same winning triangle index wins, and the
//! normal, depth, and point agree within the tolerance the narrow phase's
//! square roots and reciprocals impose. The suite is skipped (not failed) when
//! no adapter is available so it is a no-op on headless CI.

use glam::Vec3;
use prism_physics_gpu::{
    cpu_build_lbvh, cpu_capsule_trimesh_collide, Capsule, Contact, GpuCapsuleTrimeshCollider,
    GpuContext, Trimesh,
};

/// Tolerance for the inexact square roots and barycentric reciprocals.
const TOL: f32 = 1.0e-4;

/// Asserts two optional contacts agree: both absent, or both present with the
/// same indices and geometry within tolerance.
fn assert_contact_eq(cpu: Option<Contact>, gpu: Option<Contact>, capsule: usize) {
    match (cpu, gpu) {
        (None, None) => {}
        (Some(c), Some(g)) => {
            assert_eq!(c.a, g.a, "capsule {capsule}: contact.a mismatch");
            assert_eq!(
                c.b, g.b,
                "capsule {capsule}: winning triangle index mismatch"
            );
            assert!(
                (c.normal - g.normal).length() < TOL,
                "capsule {capsule}: normal mismatch cpu={:?} gpu={:?}",
                c.normal,
                g.normal
            );
            assert!(
                (c.depth - g.depth).abs() < TOL,
                "capsule {capsule}: depth mismatch cpu={} gpu={}",
                c.depth,
                g.depth
            );
            assert!(
                (c.point - g.point).length() < TOL,
                "capsule {capsule}: point mismatch cpu={:?} gpu={:?}",
                c.point,
                g.point
            );
        }
        (c, g) => panic!("capsule {capsule}: validity mismatch cpu={c:?} gpu={g:?}"),
    }
}

/// Runs both paths over the same inputs and checks per-capsule parity.
fn check(mesh: &Trimesh, capsules: &[Capsule], capacity: u32) {
    let Some(ctx) = GpuContext::try_headless() else {
        // No adapter on this machine; skip rather than fail, matching the
        // other real-device parity suites.
        return;
    };
    let lbvh = cpu_build_lbvh(&mesh.triangle_aabbs());
    let cpu = cpu_capsule_trimesh_collide(mesh, &lbvh, capsules, capacity)
        .expect("cpu golden must not overflow");
    let collider = GpuCapsuleTrimeshCollider::new(&ctx);
    let gpu = collider
        .collide(&ctx, mesh, &lbvh, capsules, capacity)
        .expect("gpu collider must not overflow");
    assert_eq!(cpu.len(), gpu.len(), "per-capsule count mismatch");
    for (i, (c, g)) in cpu.into_iter().zip(gpu).enumerate() {
        assert_contact_eq(c, g, i);
    }
}

/// A flat two-triangle quad in the z = 0 plane spanning [0,2] x [0,2].
fn quad() -> Trimesh {
    Trimesh::new(
        vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(2.0, 2.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        ],
        vec![[0, 1, 2], [0, 2, 3]],
    )
}

#[test]
fn single_capsule_face_contact() {
    // Axis-horizontal capsule hovering over triangle 0's interior.
    let capsules = vec![Capsule::new(
        Vec3::new(1.2, 0.5, 0.3),
        Vec3::new(1.6, 0.5, 0.3),
        0.5,
    )];
    check(&quad(), &capsules, 16);
}

#[test]
fn single_capsule_misses() {
    let capsules = vec![Capsule::new(
        Vec3::new(1.0, 1.0, 5.0),
        Vec3::new(1.5, 1.0, 5.0),
        0.5,
    )];
    check(&quad(), &capsules, 16);
}

#[test]
fn shared_edge_tie_break() {
    // Capsule straddling the shared diagonal: both triangles penetrate equally,
    // so the reduction must award triangle 0 on both paths.
    let capsules = vec![Capsule::new(
        Vec3::new(0.8, 0.8, 0.3),
        Vec3::new(1.2, 1.2, 0.3),
        0.5,
    )];
    check(&quad(), &capsules, 16);
}

#[test]
fn tilted_capsule_spans_both_triangles() {
    // A capsule tilted in z whose segment crosses the diagonal, exercising the
    // segment-triangle closest-point branch rather than a point query.
    let capsules = vec![Capsule::new(
        Vec3::new(0.4, 0.6, 0.25),
        Vec3::new(1.6, 1.4, 0.45),
        0.5,
    )];
    check(&quad(), &capsules, 16);
}

#[test]
fn mixed_batch_stays_in_order() {
    let capsules = vec![
        Capsule::new(Vec3::new(1.2, 0.5, 0.3), Vec3::new(1.6, 0.5, 0.3), 0.5), // face
        Capsule::new(Vec3::new(1.0, 1.0, 5.0), Vec3::new(1.5, 1.0, 5.0), 0.5), // miss
        Capsule::new(Vec3::new(0.4, 1.5, -0.3), Vec3::new(0.8, 1.5, -0.3), 0.5), // below
        Capsule::new(Vec3::new(0.1, 0.1, 0.2), Vec3::new(0.3, 0.3, 0.2), 0.4), // corner
    ];
    check(&quad(), &capsules, 16);
}

#[test]
fn stacked_triangles_deepest_wins() {
    let mesh = Trimesh::new(
        vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 0.3),
            Vec3::new(1.0, 0.0, 0.3),
            Vec3::new(0.0, 1.0, 0.3),
        ],
        vec![[0, 1, 2], [3, 4, 5]],
    );
    let capsules = vec![Capsule::new(
        Vec3::new(0.2, 0.2, 0.5),
        Vec3::new(0.3, 0.3, 0.5),
        0.6,
    )];
    check(&mesh, &capsules, 16);
}

#[test]
fn larger_grid_many_capsules() {
    // A 4x4 vertex grid (18 triangles) with a scattering of capsules, some
    // contacting faces/edges and some missing, exercising the broad phase's
    // multi-candidate gather against the batched narrow phase.
    let mut vertices = Vec::new();
    for j in 0..4u32 {
        for i in 0..4u32 {
            vertices.push(Vec3::new(i as f32, j as f32, 0.0));
        }
    }
    let mut indices = Vec::new();
    for j in 0..3u32 {
        for i in 0..3u32 {
            let v = j * 4 + i;
            indices.push([v, v + 1, v + 4]);
            indices.push([v + 1, v + 5, v + 4]);
        }
    }
    let mesh = Trimesh::new(vertices, indices);
    let capsules = vec![
        Capsule::new(Vec3::new(0.4, 0.5, 0.2), Vec3::new(0.7, 0.5, 0.2), 0.4),
        Capsule::new(Vec3::new(1.3, 1.4, 0.3), Vec3::new(1.7, 1.6, 0.3), 0.5),
        Capsule::new(Vec3::new(2.4, 0.5, -0.2), Vec3::new(2.6, 0.5, -0.2), 0.4),
        Capsule::new(Vec3::new(0.9, 2.0, 0.25), Vec3::new(1.1, 2.0, 0.25), 0.5),
        Capsule::new(Vec3::new(2.0, 2.0, 3.0), Vec3::new(2.2, 2.2, 3.0), 0.5),
        Capsule::new(Vec3::new(0.0, 3.0, 0.1), Vec3::new(0.2, 2.8, 0.1), 0.3),
    ];
    check(&mesh, &capsules, 32);
}
