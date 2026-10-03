//! Real-device parity for the `GPU` cloth two-way coupling kernel against its
//! `CPU` golden twin.
//!
//! Every test acquires a headless device with `GpuContext::try_headless()` and
//! skips cleanly when no adapter is available (for example inside a sandbox),
//! so the suite is a no-op rather than a failure off a real `GPU`.
//!
//! Both paths delegate the per-particle contact arithmetic to
//! `prism_physics_core` (the `CPU` twin calls `resolve_two_way_coupling`; the
//! kernel reimplements the same scalar `couple_particle_against_body` in `WGSL`
//! and reduces the per-body contributions on the host in index order). The only
//! divergence is a few `ULP` in `inverseSqrt`/division inside the projection, so
//! parity is checked within a tight tolerance — and the test inputs deliberately
//! avoid exact knife-edge contacts where that rounding would flip a branch.
//!
//! Provenance: the inverse-mass-weighted contact split and the Newton reaction
//! impulse are textbook position-based-dynamics / rigid-body contact mechanics.
//! No Unreal Engine source or derived code.

use glam::Vec3;
use prism_physics_core::{BodyCollider, ConvexProxy, CouplingBody, Plane};
use prism_physics_gpu::context::GpuContext;
use prism_physics_gpu::{cpu_cloth_coupling, GpuClothCoupling};

/// Absolute/relative tolerance for position and impulse parity.
const TOL: f32 = 1.0e-4;

#[expect(
    clippy::print_stderr,
    reason = "the suite is a deliberate no-op when no GPU adapter is present"
)]
fn headless() -> Option<GpuContext> {
    match GpuContext::try_headless() {
        Some(ctx) => Some(ctx),
        None => {
            eprintln!("skipping cloth coupling parity: no GPU adapter available");
            None
        }
    }
}

/// Asserts two vectors agree within [`TOL`] (combined absolute/relative).
#[track_caller]
fn assert_vec_close(a: Vec3, b: Vec3, what: &str) {
    let scale = a.length().max(b.length()).max(1.0);
    assert!(
        a.distance(b) <= TOL * scale,
        "{what}: cpu={a:?} gpu={b:?} (dist {})",
        a.distance(b)
    );
}

/// Runs both paths over the same inputs and asserts full parity of the applied
/// positions and every body's post-pass collider pose and reaction impulse.
#[track_caller]
fn assert_parity(
    ctx: &GpuContext,
    kernel: &GpuClothCoupling,
    positions: &[Vec3],
    inverse_masses: &[f32],
    bodies: &[CouplingBody],
    dt: f32,
) {
    let (cpu_pos, cpu_bodies) = cpu_cloth_coupling(positions, inverse_masses, bodies, dt);
    let (gpu_pos, gpu_bodies) = kernel.solve(ctx, positions, inverse_masses, bodies, dt);

    assert_eq!(cpu_pos.len(), gpu_pos.len(), "position count mismatch");
    for (i, (c, g)) in cpu_pos.iter().zip(&gpu_pos).enumerate() {
        assert_vec_close(*c, *g, &format!("position[{i}]"));
    }

    assert_eq!(cpu_bodies.len(), gpu_bodies.len(), "body count mismatch");
    for (k, (c, g)) in cpu_bodies.iter().zip(&gpu_bodies).enumerate() {
        assert_vec_close(
            c.reaction_impulse,
            g.reaction_impulse,
            &format!("body[{k}].reaction_impulse"),
        );
        match (c.collider, g.collider) {
            (
                BodyCollider::Sphere {
                    center: cc,
                    radius: cr,
                },
                BodyCollider::Sphere {
                    center: gc,
                    radius: gr,
                },
            ) => {
                assert_vec_close(cc, gc, &format!("body[{k}].sphere.center"));
                assert!((cr - gr).abs() <= TOL, "body[{k}].sphere.radius");
            }
            (
                BodyCollider::Capsule {
                    p0: cp0,
                    p1: cp1,
                    radius: cr,
                },
                BodyCollider::Capsule {
                    p0: gp0,
                    p1: gp1,
                    radius: gr,
                },
            ) => {
                assert_vec_close(cp0, gp0, &format!("body[{k}].capsule.p0"));
                assert_vec_close(cp1, gp1, &format!("body[{k}].capsule.p1"));
                assert!((cr - gr).abs() <= TOL, "body[{k}].capsule.radius");
            }
            (
                BodyCollider::HalfSpace {
                    normal: cn,
                    offset: co,
                },
                BodyCollider::HalfSpace {
                    normal: gn,
                    offset: go,
                },
            ) => {
                assert_vec_close(cn, gn, &format!("body[{k}].halfspace.normal"));
                assert!(
                    (co - go).abs() <= TOL * co.abs().max(1.0),
                    "body[{k}].halfspace.offset"
                );
            }
            (
                BodyCollider::Obb {
                    center: cc,
                    orientation: cq,
                    half_extents: che,
                },
                BodyCollider::Obb {
                    center: gc,
                    orientation: gq,
                    half_extents: ghe,
                },
            ) => {
                assert_vec_close(cc, gc, &format!("body[{k}].obb.center"));
                assert_vec_close(che, ghe, &format!("body[{k}].obb.half_extents"));
                assert_vec_close(
                    cq.xyz(),
                    gq.xyz(),
                    &format!("body[{k}].obb.orientation.xyz"),
                );
                assert!((cq.w - gq.w).abs() <= TOL, "body[{k}].obb.orientation.w");
            }
            (BodyCollider::ConvexHull(cp), BodyCollider::ConvexHull(gp)) => {
                assert_vec_close(
                    cp.center(),
                    gp.center(),
                    &format!("body[{k}].convex.center"),
                );
                let cf = cp.planes();
                let gf = gp.planes();
                assert_eq!(cf.len(), gf.len(), "body[{k}] convex face-count mismatch");
                for (f, (cfp, gfp)) in cf.iter().zip(gf).enumerate() {
                    assert_vec_close(
                        cfp.normal,
                        gfp.normal,
                        &format!("body[{k}].convex.face[{f}].normal"),
                    );
                    assert!(
                        (cfp.offset - gfp.offset).abs() <= TOL * cfp.offset.abs().max(1.0),
                        "body[{k}].convex.face[{f}].offset: cpu={} gpu={}",
                        cfp.offset,
                        gfp.offset
                    );
                }
            }
            (c_other, g_other) => {
                panic!("body[{k}] collider kind diverged: cpu={c_other:?} gpu={g_other:?}");
            }
        }
    }
}

#[test]
fn single_sphere_many_particles_matches_cpu() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothCoupling::new(&ctx);
    // A spread of particles, several inside a unit sphere, none on the surface.
    let positions = [
        Vec3::new(0.37, 0.11, -0.08),
        Vec3::new(0.9, 0.6, 0.5),
        Vec3::new(-0.21, 0.44, 0.13),
        Vec3::new(2.0, 0.0, 0.0), // outside, untouched
        Vec3::new(0.05, -0.3, 0.2),
    ];
    let inverse_masses = [1.0, 1.0, 2.0, 1.0, 0.5];
    let bodies = [CouplingBody::new(
        BodyCollider::Sphere {
            center: Vec3::ZERO,
            radius: 1.0,
        },
        0.75,
    )];
    assert_parity(
        &ctx,
        &kernel,
        &positions,
        &inverse_masses,
        &bodies,
        1.0 / 60.0,
    );
}

#[test]
fn capsule_matches_cpu() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothCoupling::new(&ctx);
    let positions = [
        Vec3::new(0.1, 0.33, 0.0),
        Vec3::new(-0.4, 0.2, 0.07),
        Vec3::new(0.6, 0.41, -0.12),
    ];
    let inverse_masses = [1.0, 0.8, 1.3];
    let bodies = [CouplingBody::new(
        BodyCollider::Capsule {
            p0: Vec3::new(-1.0, 0.0, 0.0),
            p1: Vec3::new(1.0, 0.0, 0.0),
            radius: 0.5,
        },
        1.0,
    )];
    assert_parity(
        &ctx,
        &kernel,
        &positions,
        &inverse_masses,
        &bodies,
        1.0 / 90.0,
    );
}

#[test]
fn obb_matches_cpu() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothCoupling::new(&ctx);
    // Particles inside and around an oriented box; two-way coupling moves both
    // the particles and the (dynamic) body, so the correction must match the
    // CPU golden for the `OBB` projection on both ends.
    let orientation = glam::Quat::from_euler(glam::EulerRot::XYZ, 0.2, -0.5, 0.8);
    let positions = [
        Vec3::new(0.1, 0.05, -0.03),
        Vec3::new(-0.2, 0.15, 0.1),
        Vec3::new(0.9, 0.6, 0.5), // outside, untouched
    ];
    let inverse_masses = [1.0, 0.7, 1.2];
    let bodies = [CouplingBody::new(
        BodyCollider::Obb {
            center: Vec3::ZERO,
            orientation,
            half_extents: Vec3::new(0.6, 0.4, 0.5),
        },
        1.0,
    )];
    assert_parity(
        &ctx,
        &kernel,
        &positions,
        &inverse_masses,
        &bodies,
        1.0 / 90.0,
    );
}

#[test]
fn half_space_matches_cpu() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothCoupling::new(&ctx);
    // Plane y = 0, outward normal +Y; a few particles dip below.
    let positions = [
        Vec3::new(0.0, -0.17, 0.0),
        Vec3::new(1.0, -0.4, -0.5),
        Vec3::new(-0.5, 0.3, 0.2), // above plane, untouched
    ];
    let inverse_masses = [1.0, 1.0, 1.0];
    let bodies = [CouplingBody::new(
        BodyCollider::HalfSpace {
            normal: Vec3::Y,
            offset: 0.0,
        },
        0.5,
    )];
    assert_parity(
        &ctx,
        &kernel,
        &positions,
        &inverse_masses,
        &bodies,
        1.0 / 60.0,
    );
}

#[test]
fn multiple_bodies_in_sequence_match_cpu() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothCoupling::new(&ctx);
    // Two overlapping spheres: a particle corrected by the first is then seen
    // by the second, exercising the sequential (slice-order) dependency.
    let positions = [
        Vec3::new(0.3, 0.2, 0.1),
        Vec3::new(0.8, 0.1, -0.2),
        Vec3::new(-0.6, 0.5, 0.3),
    ];
    let inverse_masses = [1.0, 1.5, 0.7];
    let bodies = [
        CouplingBody::new(
            BodyCollider::Sphere {
                center: Vec3::new(-0.3, 0.0, 0.0),
                radius: 1.0,
            },
            0.9,
        ),
        CouplingBody::new(
            BodyCollider::Sphere {
                center: Vec3::new(0.4, 0.0, 0.0),
                radius: 0.9,
            },
            0.6,
        ),
    ];
    assert_parity(
        &ctx,
        &kernel,
        &positions,
        &inverse_masses,
        &bodies,
        1.0 / 120.0,
    );
}

#[test]
fn kinematic_body_matches_cpu() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothCoupling::new(&ctx);
    // Zero inverse mass: the body never moves but still records a reaction.
    let positions = [Vec3::new(0.4, 0.12, -0.06), Vec3::new(0.0, 0.5, 0.3)];
    let inverse_masses = [1.0, 1.0];
    let bodies = [CouplingBody::kinematic(BodyCollider::Sphere {
        center: Vec3::ZERO,
        radius: 1.0,
    })];
    assert_parity(
        &ctx,
        &kernel,
        &positions,
        &inverse_masses,
        &bodies,
        1.0 / 60.0,
    );
}

#[test]
fn pinned_particles_against_kinematic_body_match_cpu() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothCoupling::new(&ctx);
    // Infinite-mass particle (w=0) against a kinematic body: w_sum == 0, a no-op.
    let positions = [Vec3::new(0.3, 0.0, 0.0), Vec3::new(0.5, 0.2, 0.0)];
    let inverse_masses = [0.0, 1.0];
    let bodies = [CouplingBody::kinematic(BodyCollider::Sphere {
        center: Vec3::ZERO,
        radius: 1.0,
    })];
    assert_parity(
        &ctx,
        &kernel,
        &positions,
        &inverse_masses,
        &bodies,
        1.0 / 60.0,
    );
}

#[test]
fn all_particles_outside_leave_body_still() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothCoupling::new(&ctx);
    let positions = [Vec3::new(3.0, 0.0, 0.0), Vec3::new(0.0, 4.0, 0.0)];
    let inverse_masses = [1.0, 1.0];
    let bodies = [CouplingBody::new(
        BodyCollider::Sphere {
            center: Vec3::ZERO,
            radius: 1.0,
        },
        1.0,
    )];
    assert_parity(
        &ctx,
        &kernel,
        &positions,
        &inverse_masses,
        &bodies,
        1.0 / 60.0,
    );
}

#[test]
fn empty_inputs_return_unchanged() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothCoupling::new(&ctx);
    // Empty particles.
    let bodies = [CouplingBody::new(
        BodyCollider::Sphere {
            center: Vec3::ZERO,
            radius: 1.0,
        },
        1.0,
    )];
    let (pos, out) = kernel.solve(&ctx, &[], &[], &bodies, 1.0 / 60.0);
    assert!(pos.is_empty());
    assert_eq!(out.len(), 1);
    assert_eq!(out[0], bodies[0]);
    // Empty bodies.
    let positions = [Vec3::new(0.3, 0.0, 0.0)];
    let (pos2, out2) = kernel.solve(&ctx, &positions, &[1.0], &[], 1.0 / 60.0);
    assert_eq!(pos2, positions.to_vec());
    assert!(out2.is_empty());
}

/// A translated/rotated convex box proxy coupled against the cloth must move
/// (two-way) and project particles identically to the CPU golden, exercising
/// the new `COLLIDER_CONVEX` arm in the coupling kernel end to end.
#[test]
fn convex_box_coupling_matches_cpu() {
    let Some(ctx) = headless() else {
        return;
    };
    let kernel = GpuClothCoupling::new(&ctx);
    let proxy = ConvexProxy::from_box(
        Vec3::new(0.2, -0.1, 0.3),
        glam::Quat::from_rotation_y(0.6) * glam::Quat::from_rotation_x(0.3),
        Vec3::new(0.8, 0.5, 0.6),
    );
    // Several particles driven inside the hull plus one safely outside.
    let positions = [
        Vec3::new(0.2, -0.1, 0.3),
        Vec3::new(0.4, 0.1, 0.1),
        Vec3::new(-0.1, -0.3, 0.5),
        Vec3::new(0.35, -0.25, 0.2),
        Vec3::new(2.5, 2.5, 2.5),
    ];
    let inverse_masses = [1.0, 1.0, 1.5, 0.75, 1.0];
    let bodies = [CouplingBody::new(BodyCollider::ConvexHull(proxy), 0.8)];
    assert_parity(
        &ctx,
        &kernel,
        &positions,
        &inverse_masses,
        &bodies,
        1.0 / 60.0,
    );
}

/// A bevelled (seven-face) convex hull stresses the least-penetration tie-break
/// across more than six faces while the body is also free to recoil.
#[test]
fn convex_bevelled_coupling_matches_cpu() {
    let Some(ctx) = headless() else {
        return;
    };
    let kernel = GpuClothCoupling::new(&ctx);
    let planes = [
        Plane {
            normal: Vec3::X,
            offset: 1.0,
        },
        Plane {
            normal: -Vec3::X,
            offset: 1.0,
        },
        Plane {
            normal: Vec3::Y,
            offset: 1.0,
        },
        Plane {
            normal: -Vec3::Y,
            offset: 1.0,
        },
        Plane {
            normal: Vec3::Z,
            offset: 1.0,
        },
        Plane {
            normal: -Vec3::Z,
            offset: 1.0,
        },
        Plane {
            normal: Vec3::new(1.0, 1.0, 1.0),
            offset: 2.2,
        },
    ];
    let proxy = ConvexProxy::from_planes(Vec3::ZERO, 1.8, &planes)
        .expect("bevelled hull is within the plane budget");
    let positions = [
        Vec3::new(0.1, 0.2, -0.1),
        Vec3::new(-0.3, 0.4, 0.2),
        Vec3::new(0.5, 0.5, 0.5),
        Vec3::new(0.6, 0.6, 0.6),
        Vec3::new(-0.5, -0.5, 0.1),
        Vec3::new(3.0, 0.0, 0.0),
    ];
    let inverse_masses = [1.0, 0.8, 1.2, 1.0, 1.4, 1.0];
    let bodies = [CouplingBody::new(BodyCollider::ConvexHull(proxy), 1.1)];
    assert_parity(
        &ctx,
        &kernel,
        &positions,
        &inverse_masses,
        &bodies,
        1.0 / 90.0,
    );
}

/// A kinematic (zero inverse-mass) convex hull never moves but still projects
/// particles and records a reaction, matching the CPU golden.
#[test]
fn convex_kinematic_coupling_matches_cpu() {
    let Some(ctx) = headless() else {
        return;
    };
    let kernel = GpuClothCoupling::new(&ctx);
    let proxy = ConvexProxy::from_box(
        Vec3::ZERO,
        glam::Quat::from_rotation_z(0.4),
        Vec3::new(0.6, 0.9, 0.5),
    );
    let positions = [
        Vec3::new(0.1, 0.1, 0.1),
        Vec3::new(-0.2, 0.3, -0.1),
        Vec3::new(0.3, -0.4, 0.2),
        Vec3::new(2.0, 2.0, 2.0),
    ];
    let inverse_masses = [1.0, 1.0, 1.0, 1.0];
    let bodies = [CouplingBody::kinematic(BodyCollider::ConvexHull(proxy))];
    assert_parity(
        &ctx,
        &kernel,
        &positions,
        &inverse_masses,
        &bodies,
        1.0 / 60.0,
    );
}
