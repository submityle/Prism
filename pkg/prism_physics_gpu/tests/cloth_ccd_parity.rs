//! Real-device parity for the `GPU` cloth continuous-collision (CCD) sweep
//! against its `CPU` golden twin.
//!
//! Every test acquires a headless device with `GpuContext::try_headless()` and
//! skips cleanly when no adapter is available (for example inside a sandbox),
//! so the suite is a no-op rather than a failure off a real `GPU`.
//!
//! The `CPU` twin ([`cpu_cloth_ccd`]) delegates to `prism_physics_core`'s
//! `resolve_ccd`, while the kernel reimplements the same closed-form TOI
//! solvers, surface snap, restitution reflection, and Coulomb friction in
//! `WGSL`. The only divergence is a few `ULP` in `sqrt`/division, so parity is
//! checked within a tight tolerance — and the inputs deliberately avoid exact
//! knife-edge grazes where that rounding would flip a hit/miss branch.
//!
//! Provenance: the closed-form swept-primitive TOI solvers are standard
//! analytic continuous-collision geometry, and the tangential-friction
//! projection reuses the Macklin et al. (2014) primitive. No Unreal Engine
//! source or derived code.

use glam::Vec3;
use prism_physics_core::{BodyCollider, CcdParams, ConvexProxy, Plane};
use prism_physics_gpu::context::GpuContext;
use prism_physics_gpu::{cpu_cloth_ccd, GpuClothCcd};

/// Absolute/relative tolerance for position and velocity parity.
const TOL: f32 = 1.0e-4;

#[expect(
    clippy::print_stderr,
    reason = "the suite is a deliberate no-op when no GPU adapter is present"
)]
fn headless() -> Option<GpuContext> {
    match GpuContext::try_headless() {
        Some(ctx) => Some(ctx),
        None => {
            eprintln!("skipping cloth ccd parity: no GPU adapter available");
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
/// positions and velocities.
#[track_caller]
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors resolve_ccd's full column/param signature"
)]
fn assert_parity(
    ctx: &GpuContext,
    kernel: &GpuClothCcd,
    positions: &[Vec3],
    prev_positions: &[Vec3],
    velocities: &[Vec3],
    inverse_masses: &[f32],
    colliders: &[BodyCollider],
    params: CcdParams,
    dt: f32,
    friction: f32,
) {
    let (cpu_pos, cpu_vel) = cpu_cloth_ccd(
        positions,
        prev_positions,
        velocities,
        inverse_masses,
        colliders,
        params,
        dt,
        friction,
    );
    let (gpu_pos, gpu_vel) = kernel.solve(
        ctx,
        positions,
        prev_positions,
        velocities,
        inverse_masses,
        colliders,
        params,
        dt,
        friction,
    );

    assert_eq!(cpu_pos.len(), gpu_pos.len(), "position count mismatch");
    assert_eq!(cpu_vel.len(), gpu_vel.len(), "velocity count mismatch");
    for (i, (c, g)) in cpu_pos.iter().zip(&gpu_pos).enumerate() {
        assert_vec_close(*c, *g, &format!("position[{i}]"));
    }
    for (i, (c, g)) in cpu_vel.iter().zip(&gpu_vel).enumerate() {
        assert_vec_close(*c, *g, &format!("velocity[{i}]"));
    }
}

fn sphere(center: Vec3, radius: f32) -> BodyCollider {
    BodyCollider::Sphere { center, radius }
}

fn capsule(p0: Vec3, p1: Vec3, radius: f32) -> BodyCollider {
    BodyCollider::Capsule { p0, p1, radius }
}

fn half_space(normal: Vec3, offset: f32) -> BodyCollider {
    BodyCollider::HalfSpace { normal, offset }
}

fn obb(center: Vec3, orientation: glam::Quat, half_extents: Vec3) -> BodyCollider {
    BodyCollider::Obb {
        center,
        orientation,
        half_extents,
    }
}

const DT: f32 = 1.0 / 60.0;

#[test]
fn tunnelling_through_sphere_is_caught() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothCcd::new(&ctx);
    // Sweeps clean through a unit sphere in one step.
    let positions = [Vec3::new(2.3, 0.1, -0.2)];
    let prev = [Vec3::new(-2.1, 0.1, -0.2)];
    let velocities = [Vec3::ZERO];
    let inv_mass = [1.0];
    assert_parity(
        &ctx,
        &kernel,
        &positions,
        &prev,
        &velocities,
        &inv_mass,
        &[sphere(Vec3::ZERO, 1.0)],
        CcdParams::default(),
        DT,
        0.0,
    );
}

#[test]
fn tunnelling_through_capsule_is_caught() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothCcd::new(&ctx);
    let positions = [Vec3::new(0.07, 1.4, 0.03)];
    let prev = [Vec3::new(0.07, -1.5, 0.03)];
    let velocities = [Vec3::ZERO];
    let inv_mass = [1.0];
    assert_parity(
        &ctx,
        &kernel,
        &positions,
        &prev,
        &velocities,
        &inv_mass,
        &[capsule(
            Vec3::new(-0.6, 0.0, 0.0),
            Vec3::new(0.6, 0.0, 0.0),
            0.5,
        )],
        CcdParams::default(),
        DT,
        0.0,
    );
}

#[test]
fn tunnelling_through_half_space_is_caught() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothCcd::new(&ctx);
    // Crosses the plane y = 0 from above to below.
    let positions = [Vec3::new(0.3, -0.8, -0.1)];
    let prev = [Vec3::new(0.3, 0.9, -0.1)];
    let velocities = [Vec3::ZERO];
    let inv_mass = [1.0];
    assert_parity(
        &ctx,
        &kernel,
        &positions,
        &prev,
        &velocities,
        &inv_mass,
        &[half_space(Vec3::Y, 0.0)],
        CcdParams::default(),
        DT,
        0.0,
    );
}

#[test]
fn tunnelling_through_obb_is_caught() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothCcd::new(&ctx);
    // Sweeps clean through an oriented box in one step; the swept segment must
    // register the earliest slab entry exactly as the CPU golden does.
    let orientation = glam::Quat::from_euler(glam::EulerRot::XYZ, 0.4, 0.9, -0.3);
    let positions = [Vec3::new(2.4, 0.15, -0.1)];
    let prev = [Vec3::new(-2.2, 0.15, -0.1)];
    let velocities = [Vec3::ZERO];
    let inv_mass = [1.0];
    assert_parity(
        &ctx,
        &kernel,
        &positions,
        &prev,
        &velocities,
        &inv_mass,
        &[obb(Vec3::ZERO, orientation, Vec3::new(0.7, 0.5, 0.6))],
        CcdParams::default(),
        DT,
        0.0,
    );
}

#[test]
fn earliest_hit_across_multiple_colliders_wins() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothCcd::new(&ctx);
    // Two spheres on the path; the nearer one must claim the particle.
    let positions = [Vec3::new(6.2, 0.05, 0.0)];
    let prev = [Vec3::new(-6.1, 0.05, 0.0)];
    let velocities = [Vec3::ZERO];
    let inv_mass = [1.0];
    assert_parity(
        &ctx,
        &kernel,
        &positions,
        &prev,
        &velocities,
        &inv_mass,
        &[
            sphere(Vec3::new(3.0, 0.0, 0.0), 1.0),
            sphere(Vec3::new(-3.0, 0.0, 0.0), 1.0),
        ],
        CcdParams::default(),
        DT,
        0.0,
    );
}

#[test]
fn pinned_particle_holds_still() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothCcd::new(&ctx);
    let positions = [Vec3::new(2.2, 0.0, 0.0)];
    let prev = [Vec3::new(-2.1, 0.0, 0.0)];
    let velocities = [Vec3::new(0.5, 0.0, 0.0)];
    let inv_mass = [0.0];
    assert_parity(
        &ctx,
        &kernel,
        &positions,
        &prev,
        &velocities,
        &inv_mass,
        &[sphere(Vec3::ZERO, 1.0)],
        CcdParams::default(),
        DT,
        0.0,
    );
}

#[test]
fn grazing_miss_leaves_particle_alone() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothCcd::new(&ctx);
    // Passes well clear of the sphere (y = 3 against radius 1).
    let positions = [Vec3::new(2.4, 3.0, 0.0)];
    let prev = [Vec3::new(-2.3, 3.0, 0.0)];
    let velocities = [Vec3::new(0.2, 0.0, 0.0)];
    let inv_mass = [1.0];
    assert_parity(
        &ctx,
        &kernel,
        &positions,
        &prev,
        &velocities,
        &inv_mass,
        &[sphere(Vec3::ZERO, 1.0)],
        CcdParams::default(),
        DT,
        0.0,
    );
}

#[test]
fn disabled_params_are_a_no_op() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothCcd::new(&ctx);
    let positions = [Vec3::new(2.2, 0.0, 0.0)];
    let prev = [Vec3::new(-2.1, 0.0, 0.0)];
    let velocities = [Vec3::new(0.3, 0.1, 0.0)];
    let inv_mass = [1.0];
    let params = CcdParams {
        enabled: false,
        ..CcdParams::default()
    };
    assert_parity(
        &ctx,
        &kernel,
        &positions,
        &prev,
        &velocities,
        &inv_mass,
        &[sphere(Vec3::ZERO, 1.0)],
        params,
        DT,
        0.0,
    );
}

#[test]
fn empty_colliders_are_a_no_op() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothCcd::new(&ctx);
    let positions = [Vec3::new(2.2, 0.0, 0.0)];
    let prev = [Vec3::new(-2.1, 0.0, 0.0)];
    let velocities = [Vec3::new(0.3, 0.0, 0.0)];
    let inv_mass = [1.0];
    assert_parity(
        &ctx,
        &kernel,
        &positions,
        &prev,
        &velocities,
        &inv_mass,
        &[],
        CcdParams::default(),
        DT,
        0.0,
    );
}

#[test]
fn restitution_reflects_normal_velocity() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothCcd::new(&ctx);
    // A fast downward crossing of the floor, bouncing with restitution.
    let positions = [Vec3::new(0.2, -1.1, 0.15)];
    let prev = [Vec3::new(0.2, 0.9, 0.15)];
    let velocities = [Vec3::new(0.1, -3.0, 0.0)];
    let inv_mass = [1.0];
    let params = CcdParams {
        skin: 1e-3,
        restitution: 0.6,
        enabled: true,
    };
    assert_parity(
        &ctx,
        &kernel,
        &positions,
        &prev,
        &velocities,
        &inv_mass,
        &[half_space(Vec3::Y, 0.0)],
        params,
        DT,
        0.0,
    );
}

#[test]
fn coulomb_friction_damps_tangential_slide() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothCcd::new(&ctx);
    // Oblique crossing of the floor with a strong tangential component.
    let positions = [Vec3::new(1.3, -0.6, 0.0)];
    let prev = [Vec3::new(0.1, 0.7, 0.0)];
    let velocities = [Vec3::ZERO];
    let inv_mass = [1.0];
    for &mu in &[0.0_f32, 0.35, 1.0] {
        assert_parity(
            &ctx,
            &kernel,
            &positions,
            &prev,
            &velocities,
            &inv_mass,
            &[half_space(Vec3::Y, 0.0)],
            CcdParams::default(),
            DT,
            mu,
        );
    }
}

#[test]
fn zero_dt_disables_velocity_reflection() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothCcd::new(&ctx);
    let positions = [Vec3::new(0.2, -1.1, 0.0)];
    let prev = [Vec3::new(0.2, 0.9, 0.0)];
    let velocities = [Vec3::new(0.0, -2.0, 0.0)];
    let inv_mass = [1.0];
    assert_parity(
        &ctx,
        &kernel,
        &positions,
        &prev,
        &velocities,
        &inv_mass,
        &[half_space(Vec3::Y, 0.0)],
        CcdParams::default(),
        0.0,
        0.0,
    );
}

#[test]
fn mixed_batch_many_particles_and_colliders() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothCcd::new(&ctx);
    // A deterministic spread of hits, misses, and pinned particles over a mix
    // of collider primitives — exercises the full per-particle write set.
    let mut positions = Vec::new();
    let mut prev = Vec::new();
    let mut velocities = Vec::new();
    let mut inv_mass = Vec::new();
    for k in 0..128u32 {
        let f = k as f32;
        // Deterministic, trig-free spread so the batch stays reproducible
        // without pulling in a non-libm transcendental (clippy-disallowed).
        let jitter = ((k % 11) as f32) * 0.05 - 0.25;
        let z = ((k % 7) as f32) * 0.06 - 0.18;
        let x = -3.0 + jitter * 0.4;
        positions.push(Vec3::new(3.0 + jitter * 0.2, 0.2 + z, z));
        prev.push(Vec3::new(x, 0.2 + z, z));
        velocities.push(Vec3::new(
            jitter,
            -0.5 - ((k % 5) as f32) * 0.1,
            0.1 + f * 0.001,
        ));
        // Every seventh particle is pinned.
        inv_mass.push(if k % 7 == 0 { 0.0 } else { 1.0 });
    }
    let colliders = [
        sphere(Vec3::new(0.0, 0.2, 0.0), 0.9),
        capsule(Vec3::new(-0.5, 0.2, -0.5), Vec3::new(0.5, 0.2, 0.5), 0.4),
        half_space(Vec3::Y, -1.5),
    ];
    let params = CcdParams {
        skin: 2e-3,
        restitution: 0.3,
        enabled: true,
    };
    assert_parity(
        &ctx,
        &kernel,
        &positions,
        &prev,
        &velocities,
        &inv_mass,
        &colliders,
        params,
        DT,
        0.4,
    );
}

/// Builds a `ConvexHull` collider from an oriented box.
fn convex_box(center: Vec3, orientation: glam::Quat, half_extents: Vec3) -> BodyCollider {
    BodyCollider::ConvexHull(ConvexProxy::from_box(center, orientation, half_extents))
}

/// A unit box with its +X+Y+Z corner shaved off: six axis faces plus one
/// diagonal bevel, seven live faces total.
fn bevelled_convex() -> BodyCollider {
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
    BodyCollider::ConvexHull(
        ConvexProxy::from_planes(Vec3::ZERO, 1.8, &planes)
            .expect("bevelled hull is within the plane budget"),
    )
}

/// A particle sweeping clean through a convex box in one step must register the
/// earliest slab entry (segment TOI) exactly as the CPU golden, matching the
/// oriented-box twin.
#[test]
fn tunnelling_through_convex_box_is_caught() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothCcd::new(&ctx);
    let orientation = glam::Quat::from_euler(glam::EulerRot::XYZ, 0.4, 0.9, -0.3);
    let positions = [Vec3::new(2.4, 0.15, -0.1)];
    let prev = [Vec3::new(-2.2, 0.15, -0.1)];
    let velocities = [Vec3::ZERO];
    let inv_mass = [1.0];
    assert_parity(
        &ctx,
        &kernel,
        &positions,
        &prev,
        &velocities,
        &inv_mass,
        &[convex_box(
            Vec3::ZERO,
            orientation,
            Vec3::new(0.7, 0.5, 0.6),
        )],
        CcdParams::default(),
        DT,
        0.0,
    );
}

/// Sweeping through a seven-face bevelled hull stresses the slab clip across
/// more faces than a box and the least-penetration surface snap.
#[test]
fn tunnelling_through_bevelled_convex_is_caught() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothCcd::new(&ctx);
    // Aim at the shaved corner so the diagonal bevel is the entry face.
    let positions = [Vec3::new(2.0, 2.0, 2.0)];
    let prev = [Vec3::new(-2.0, -2.0, -2.0)];
    let velocities = [Vec3::ZERO];
    let inv_mass = [1.0];
    assert_parity(
        &ctx,
        &kernel,
        &positions,
        &prev,
        &velocities,
        &inv_mass,
        &[bevelled_convex()],
        CcdParams::default(),
        DT,
        0.0,
    );
}

/// A convex-hull hit with restitution and Coulomb friction exercises the full
/// post-impact response (surface snap + normal reflection + tangential damping)
/// on the convex arm.
#[test]
fn convex_hit_with_restitution_and_friction_matches_cpu() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothCcd::new(&ctx);
    let positions = [Vec3::new(1.6, 0.4, 0.2)];
    let prev = [Vec3::new(-1.4, 0.4, 0.2)];
    // A diagonal inbound velocity so friction has a tangential component.
    let velocities = [Vec3::new(40.0, 6.0, -3.0)];
    let inv_mass = [1.0];
    let params = CcdParams {
        restitution: 0.6,
        ..CcdParams::default()
    };
    assert_parity(
        &ctx,
        &kernel,
        &positions,
        &prev,
        &velocities,
        &inv_mass,
        &[convex_box(
            Vec3::ZERO,
            glam::Quat::from_rotation_y(0.5),
            Vec3::new(0.8, 0.6, 0.7),
        )],
        params,
        DT,
        0.5,
    );
}

/// A mixed scene with two distinct convex hulls plus analytic primitives checks
/// that the host packs each hull's face run at the right plane offset and the
/// shader indexes the correct slice per collider across the swept walk.
#[test]
fn mixed_scene_with_convex_hulls_matches_cpu() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothCcd::new(&ctx);
    let positions = [
        Vec3::new(2.0, 0.0, 0.0),
        Vec3::new(-2.0, 0.1, 0.0),
        Vec3::new(0.0, 2.0, 0.3),
        Vec3::new(0.2, -2.0, -0.1),
    ];
    let prev = [
        Vec3::new(-2.0, 0.0, 0.0),
        Vec3::new(2.0, 0.1, 0.0),
        Vec3::new(0.0, -2.0, 0.3),
        Vec3::new(0.2, 2.0, -0.1),
    ];
    let velocities = [Vec3::ZERO; 4];
    let inv_mass = [1.0; 4];
    let colliders = [
        sphere(Vec3::new(0.0, 0.0, 0.0), 0.4),
        convex_box(
            Vec3::new(-0.6, 0.0, 0.0),
            glam::Quat::IDENTITY,
            Vec3::new(0.4, 0.4, 0.4),
        ),
        half_space(Vec3::Y, -1.5),
        convex_box(
            Vec3::new(0.7, 0.1, -0.2),
            glam::Quat::from_rotation_z(0.9),
            Vec3::new(0.5, 0.3, 0.6),
        ),
    ];
    assert_parity(
        &ctx,
        &kernel,
        &positions,
        &prev,
        &velocities,
        &inv_mass,
        &colliders,
        CcdParams::default(),
        DT,
        0.0,
    );
}
