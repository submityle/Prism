//! Real-device parity: the GPU sphere-versus-trimesh collider
//! ([`GpuSphereTrimeshCollider`]) must agree with the CPU golden
//! ([`cpu_sphere_trimesh_collide`]) on an actual Metal (or other `wgpu`)
//! adapter.
//!
//! Each test builds a mesh, its `LBVH`, and a sphere batch, runs both the CPU
//! golden and the device collider, and asserts the per-sphere manifolds match:
//! the same spheres contact, the same winning triangle index wins, and the
//! normal, depth, and point agree within the tolerance the narrow phase's
//! square root and reciprocals impose. The suite is skipped (not failed) when
//! no adapter is available so it is a no-op on headless CI.

use glam::Vec3;
use prism_physics_gpu::{
    cpu_build_lbvh, cpu_sphere_trimesh_collide, Contact, GpuContext, GpuSphereTrimeshCollider,
    Particle, Trimesh,
};

/// Tolerance for the inexact square root and barycentric reciprocals.
const TOL: f32 = 1.0e-4;

/// Asserts two optional contacts agree: both absent, or both present with the
/// same indices and geometry within tolerance.
fn assert_contact_eq(cpu: Option<Contact>, gpu: Option<Contact>, sphere: usize) {
    match (cpu, gpu) {
        (None, None) => {}
        (Some(c), Some(g)) => {
            assert_eq!(c.a, g.a, "sphere {sphere}: contact.a mismatch");
            assert_eq!(c.b, g.b, "sphere {sphere}: winning triangle index mismatch");
            assert!(
                (c.normal - g.normal).length() < TOL,
                "sphere {sphere}: normal mismatch cpu={:?} gpu={:?}",
                c.normal,
                g.normal
            );
            assert!(
                (c.depth - g.depth).abs() < TOL,
                "sphere {sphere}: depth mismatch cpu={} gpu={}",
                c.depth,
                g.depth
            );
            assert!(
                (c.point - g.point).length() < TOL,
                "sphere {sphere}: point mismatch cpu={:?} gpu={:?}",
                c.point,
                g.point
            );
        }
        (c, g) => panic!("sphere {sphere}: validity mismatch cpu={c:?} gpu={g:?}"),
    }
}

/// Runs both paths over the same inputs and checks per-sphere parity.
fn check(mesh: &Trimesh, spheres: &[Particle], capacity: u32) {
    let Some(ctx) = GpuContext::try_headless() else {
        // No adapter on this machine; skip rather than fail, matching the
        // other real-device parity suites.
        return;
    };
    let lbvh = cpu_build_lbvh(&mesh.triangle_aabbs());
    let cpu = cpu_sphere_trimesh_collide(mesh, &lbvh, spheres, capacity)
        .expect("cpu golden must not overflow");
    let collider = GpuSphereTrimeshCollider::new(&ctx);
    let gpu = collider
        .collide(&ctx, mesh, &lbvh, spheres, capacity)
        .expect("gpu collider must not overflow");
    assert_eq!(cpu.len(), gpu.len(), "per-sphere count mismatch");
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
fn single_sphere_face_contact() {
    let spheres = vec![Particle::new(Vec3::new(1.5, 0.5, 0.4), 0.5)];
    check(&quad(), &spheres, 16);
}

#[test]
fn single_sphere_misses() {
    let spheres = vec![Particle::new(Vec3::new(1.0, 1.0, 5.0), 0.5)];
    check(&quad(), &spheres, 16);
}

#[test]
fn shared_edge_tie_break() {
    // Sphere on the shared diagonal: both triangles penetrate equally, so the
    // reduction must award triangle 0 on both paths.
    let spheres = vec![Particle::new(Vec3::new(1.0, 1.0, 0.3), 0.5)];
    check(&quad(), &spheres, 16);
}

#[test]
fn mixed_batch_stays_in_order() {
    let spheres = vec![
        Particle::new(Vec3::new(1.5, 0.5, 0.4), 0.5),
        Particle::new(Vec3::new(1.0, 1.0, 5.0), 0.5),
        Particle::new(Vec3::new(0.5, 1.5, -0.3), 0.5),
        Particle::new(Vec3::new(0.2, 0.2, 0.2), 0.4),
    ];
    check(&quad(), &spheres, 16);
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
    let spheres = vec![Particle::new(Vec3::new(0.25, 0.25, 0.5), 0.6)];
    check(&mesh, &spheres, 16);
}

#[test]
fn larger_grid_many_spheres() {
    // A 4x4 vertex grid (18 triangles) with a scattering of spheres, some
    // contacting faces/edges/vertices and some missing, to exercise the broad
    // phase's multi-candidate gather against the batched narrow phase.
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
    let spheres = vec![
        Particle::new(Vec3::new(0.5, 0.5, 0.2), 0.4),
        Particle::new(Vec3::new(1.5, 1.5, 0.3), 0.5),
        Particle::new(Vec3::new(2.5, 0.5, -0.2), 0.4),
        Particle::new(Vec3::new(1.0, 2.0, 0.25), 0.5),
        Particle::new(Vec3::new(2.0, 2.0, 3.0), 0.5),
        Particle::new(Vec3::new(0.0, 3.0, 0.1), 0.3),
    ];
    check(&mesh, &spheres, 32);
}
