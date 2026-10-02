//! Real-device parity: the GPU OBB-versus-trimesh collider
//! ([`GpuObbTrimeshCollider`]) must agree with the CPU golden
//! ([`cpu_obb_trimesh_collide`]) on an actual Metal (or other `wgpu`) adapter.
//!
//! Each test builds a mesh, its `LBVH`, and an oriented-box batch, runs both the
//! CPU golden and the device collider, and asserts the per-box manifolds match:
//! the same boxes contact, the same winning triangle index wins, and the
//! normal, depth, and point agree within the tolerance the narrow phase's
//! square roots and reciprocals impose. The suite is skipped (not failed) when
//! no adapter is available so it is a no-op on headless CI.

use glam::{Quat, Vec3};
use prism_physics_gpu::{
    cpu_build_lbvh, cpu_obb_trimesh_collide, Contact, GpuContext, GpuObbTrimeshCollider, Obb,
    Trimesh,
};

/// Tolerance for the inexact square roots and barycentric reciprocals.
const TOL: f32 = 1.0e-4;

/// Asserts two optional contacts agree: both absent, or both present with the
/// same indices and geometry within tolerance.
fn assert_contact_eq(cpu: Option<Contact>, gpu: Option<Contact>, obb: usize) {
    match (cpu, gpu) {
        (None, None) => {}
        (Some(c), Some(g)) => {
            assert_eq!(c.a, g.a, "box {obb}: contact.a mismatch");
            assert_eq!(c.b, g.b, "box {obb}: winning triangle index mismatch");
            assert!(
                (c.normal - g.normal).length() < TOL,
                "box {obb}: normal mismatch cpu={:?} gpu={:?}",
                c.normal,
                g.normal
            );
            assert!(
                (c.depth - g.depth).abs() < TOL,
                "box {obb}: depth mismatch cpu={} gpu={}",
                c.depth,
                g.depth
            );
            assert!(
                (c.point - g.point).length() < TOL,
                "box {obb}: point mismatch cpu={:?} gpu={:?}",
                c.point,
                g.point
            );
        }
        (c, g) => panic!("box {obb}: validity mismatch cpu={c:?} gpu={g:?}"),
    }
}

/// Runs both paths over the same inputs and checks per-box parity.
fn check(mesh: &Trimesh, boxes: &[Obb], capacity: u32) {
    let Some(ctx) = GpuContext::try_headless() else {
        // No adapter on this machine; skip rather than fail, matching the
        // other real-device parity suites.
        return;
    };
    let lbvh = cpu_build_lbvh(&mesh.triangle_aabbs());
    let cpu =
        cpu_obb_trimesh_collide(mesh, &lbvh, boxes, capacity).expect("cpu golden must not overflow");
    let collider = GpuObbTrimeshCollider::new(&ctx);
    let gpu = collider
        .collide(&ctx, mesh, &lbvh, boxes, capacity)
        .expect("gpu collider must not overflow");
    assert_eq!(cpu.len(), gpu.len(), "per-box count mismatch");
    for (i, (c, g)) in cpu.into_iter().zip(gpu).enumerate() {
        assert_contact_eq(c, g, i);
    }
}

/// The identity axis triple.
fn axes() -> [Vec3; 3] {
    [Vec3::X, Vec3::Y, Vec3::Z]
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
fn single_box_face_contact() {
    let boxes = vec![Obb::new(Vec3::new(1.5, 0.5, 0.4), axes(), Vec3::splat(0.5))];
    check(&quad(), &boxes, 16);
}

#[test]
fn single_box_misses() {
    let boxes = vec![Obb::new(Vec3::new(1.0, 1.0, 5.0), axes(), Vec3::splat(0.5))];
    check(&quad(), &boxes, 16);
}

#[test]
fn shared_edge_tie_break() {
    let boxes = vec![Obb::new(Vec3::new(1.0, 1.0, 0.4), axes(), Vec3::splat(0.5))];
    check(&quad(), &boxes, 16);
}

#[test]
fn rotated_box_about_z() {
    // A box yawed 45 degrees about z: its world AABB grows by sqrt(2), and the
    // edge-edge SAT axes come into play, so this exercises the narrow phase's
    // full axis cascade rather than a pure face query.
    let rot = Quat::from_rotation_z(std::f32::consts::FRAC_PI_4);
    let boxes = vec![Obb::from_quat(
        Vec3::new(1.0, 1.0, 0.4),
        rot,
        Vec3::splat(0.5),
    )];
    check(&quad(), &boxes, 16);
}

#[test]
fn tilted_box_edge_contact() {
    // A box tilted about x so a lower edge dips into the face: the minimal
    // penetration axis is no longer a world axis.
    let rot = Quat::from_rotation_x(0.3);
    let boxes = vec![Obb::from_quat(
        Vec3::new(1.2, 0.7, 0.4),
        rot,
        Vec3::splat(0.5),
    )];
    check(&quad(), &boxes, 16);
}

#[test]
fn mixed_batch_stays_in_order() {
    let boxes = vec![
        Obb::new(Vec3::new(1.5, 0.5, 0.4), axes(), Vec3::splat(0.5)), // face
        Obb::new(Vec3::new(1.0, 1.0, 5.0), axes(), Vec3::splat(0.5)), // miss
        Obb::new(Vec3::new(0.5, 1.5, -0.4), axes(), Vec3::splat(0.5)), // below
        Obb::from_quat(
            Vec3::new(0.4, 0.4, 0.3),
            Quat::from_rotation_z(0.5),
            Vec3::splat(0.4),
        ), // corner, rotated
    ];
    check(&quad(), &boxes, 16);
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
    let boxes = vec![Obb::new(Vec3::new(0.2, 0.2, 0.5), axes(), Vec3::splat(0.4))];
    check(&mesh, &boxes, 16);
}

#[test]
fn larger_grid_many_boxes() {
    // A 4x4 vertex grid (18 triangles) with a scattering of boxes, some
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
    let boxes = vec![
        Obb::new(Vec3::new(0.5, 0.5, 0.2), axes(), Vec3::splat(0.4)),
        Obb::from_quat(
            Vec3::new(1.5, 1.5, 0.3),
            Quat::from_rotation_z(0.4),
            Vec3::splat(0.45),
        ),
        Obb::new(Vec3::new(2.5, 0.5, -0.2), axes(), Vec3::splat(0.4)),
        Obb::from_quat(
            Vec3::new(1.0, 2.0, 0.25),
            Quat::from_rotation_x(0.2),
            Vec3::splat(0.45),
        ),
        Obb::new(Vec3::new(2.0, 2.0, 3.0), axes(), Vec3::splat(0.5)),
        Obb::new(Vec3::new(0.0, 3.0, 0.1), axes(), Vec3::splat(0.3)),
    ];
    check(&mesh, &boxes, 32);
}
