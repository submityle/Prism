//! Real-device parity for the `GPU` cloth point (vertex-vertex) self-collision
//! pass against its `CPU` golden twin.
//!
//! Every test acquires a headless device with `GpuContext::try_headless()` and
//! skips cleanly when no adapter is available (for example inside a sandbox),
//! so the suite is a no-op rather than a failure off a real `GPU`.
//!
//! The `CPU` twin ([`cpu_cloth_self_collision_point`]) delegates to
//! `prism_physics_core`'s `resolve_self_collision_jacobi` /
//! `resolve_self_collision_with_friction_jacobi`, while the kernel reimplements
//! the same inverse-mass-weighted separation and position-level Coulomb
//! friction in `WGSL` over a host-built candidate set and incidence list. The
//! only divergence is a few `ULP` in `sqrt`/division, so parity is checked
//! within a tight tolerance.
//!
//! Provenance: the inverse-mass-weighted separation is standard position-based
//! dynamics; the tangential-friction projection is the one published by Macklin
//! et al. (2014); the Jacobi own-slot accumulate/apply split is standard
//! parallel position-based dynamics. No Unreal Engine source or derived code.

use glam::Vec3;
use prism_physics_gpu::context::GpuContext;
use prism_physics_gpu::{cpu_cloth_self_collision_point, GpuClothSelfCollisionPoint};

/// Absolute/relative tolerance for position parity.
const TOL: f32 = 1.0e-4;

#[expect(
    clippy::print_stderr,
    reason = "the suite is a deliberate no-op when no GPU adapter is present"
)]
fn headless() -> Option<GpuContext> {
    match GpuContext::try_headless() {
        Some(ctx) => Some(ctx),
        None => {
            eprintln!("skipping cloth point self-collision parity: no GPU adapter available");
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
/// positions.
#[track_caller]
#[expect(clippy::too_many_arguments, reason = "mirrors the solve parameter set")]
fn assert_parity(
    ctx: &GpuContext,
    kernel: &GpuClothSelfCollisionPoint,
    positions: &[Vec3],
    prev_positions: &[Vec3],
    inverse_masses: &[f32],
    cell_size: f32,
    thickness: f32,
    friction: f32,
) {
    let cpu = cpu_cloth_self_collision_point(
        positions,
        prev_positions,
        inverse_masses,
        cell_size,
        thickness,
        friction,
    );
    let gpu = kernel.solve(
        ctx,
        positions,
        prev_positions,
        inverse_masses,
        cell_size,
        thickness,
        friction,
    );
    assert_eq!(cpu.len(), gpu.len(), "position count parity");
    for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert_vec_close(*c, *g, &format!("position[{i}]"));
    }
}

/// Advances both paths for several Jacobi iterations and asserts parity at each
/// step, so accumulated reduction-order drift (if any) would show up.
#[track_caller]
#[expect(clippy::too_many_arguments, reason = "mirrors the solve parameter set")]
fn assert_parity_iterated(
    ctx: &GpuContext,
    kernel: &GpuClothSelfCollisionPoint,
    positions: &[Vec3],
    inverse_masses: &[f32],
    cell_size: f32,
    thickness: f32,
    friction: f32,
    steps: usize,
) {
    let mut cpu = positions.to_vec();
    let mut gpu = positions.to_vec();
    for _ in 0..steps {
        let prev_cpu = cpu.clone();
        let prev_gpu = gpu.clone();
        cpu = cpu_cloth_self_collision_point(
            &cpu,
            &prev_cpu,
            inverse_masses,
            cell_size,
            thickness,
            friction,
        );
        gpu = kernel.solve(
            ctx,
            &gpu,
            &prev_gpu,
            inverse_masses,
            cell_size,
            thickness,
            friction,
        );
        for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
            let scale = c.length().max(g.length()).max(1.0);
            assert!(
                c.distance(*g) <= 1.0e-3 * scale,
                "iterated position[{i}]: cpu={c:?} gpu={g:?}"
            );
        }
    }
}

#[test]
fn plain_penetrating_pair_parity() {
    let Some(ctx) = headless() else {
        return;
    };
    let kernel = GpuClothSelfCollisionPoint::new(&ctx);
    let positions = [Vec3::ZERO, Vec3::new(0.5, 0.0, 0.0)];
    let prev = positions;
    let im = [1.0, 1.0];
    assert_parity(&ctx, &kernel, &positions, &prev, &im, 2.0, 1.0, 0.0);
}

#[test]
fn plain_asymmetric_mass_pair_parity() {
    let Some(ctx) = headless() else {
        return;
    };
    let kernel = GpuClothSelfCollisionPoint::new(&ctx);
    let positions = [Vec3::ZERO, Vec3::new(0.3, 0.1, 0.0)];
    let prev = positions;
    let im = [0.25, 1.0];
    assert_parity(&ctx, &kernel, &positions, &prev, &im, 2.0, 1.0, 0.0);
}

#[test]
fn pinned_partner_parity() {
    let Some(ctx) = headless() else {
        return;
    };
    let kernel = GpuClothSelfCollisionPoint::new(&ctx);
    let positions = [Vec3::ZERO, Vec3::new(0.4, 0.0, 0.0)];
    let prev = positions;
    let im = [0.0, 1.0];
    assert_parity(&ctx, &kernel, &positions, &prev, &im, 2.0, 1.0, 0.0);
}

#[test]
fn coincident_pair_parity() {
    let Some(ctx) = headless() else {
        return;
    };
    let kernel = GpuClothSelfCollisionPoint::new(&ctx);
    // Two coincident free particles must separate along the index-oriented
    // +X / -X axis, identically on both paths.
    let positions = [Vec3::ZERO, Vec3::ZERO];
    let prev = positions;
    let im = [1.0, 1.0];
    assert_parity(&ctx, &kernel, &positions, &prev, &im, 1.0, 2.0, 0.0);
}

#[test]
fn separated_pair_is_untouched_parity() {
    let Some(ctx) = headless() else {
        return;
    };
    let kernel = GpuClothSelfCollisionPoint::new(&ctx);
    // Within a shared neighborhood but farther apart than thickness: both paths
    // must leave the pair exactly where it started.
    let positions = [Vec3::ZERO, Vec3::new(0.8, 0.0, 0.0)];
    let prev = positions;
    let im = [1.0, 1.0];
    assert_parity(&ctx, &kernel, &positions, &prev, &im, 1.0, 0.5, 0.0);
}

#[test]
fn friction_pair_parity() {
    let Some(ctx) = headless() else {
        return;
    };
    let kernel = GpuClothSelfCollisionPoint::new(&ctx);
    // Integer-friendly offsets so clippy's trig lint never fires; a real
    // tangential slide between frame start and end exercises the friction path.
    let positions = [Vec3::ZERO, Vec3::new(0.5, 0.0, 0.0)];
    let prev = [Vec3::new(0.0, -1.0, 0.0), Vec3::new(0.5, 1.0, 0.0)];
    let im = [1.0, 1.0];
    assert_parity(&ctx, &kernel, &positions, &prev, &im, 2.0, 1.0, 0.5);
}

#[test]
fn friction_full_mu_parity() {
    let Some(ctx) = headless() else {
        return;
    };
    let kernel = GpuClothSelfCollisionPoint::new(&ctx);
    let positions = [Vec3::ZERO, Vec3::new(0.5, 0.0, 0.0)];
    let prev = [Vec3::new(0.0, -2.0, 0.0), Vec3::new(0.5, 2.0, 0.0)];
    let im = [1.0, 1.0];
    assert_parity(&ctx, &kernel, &positions, &prev, &im, 2.0, 1.0, 1.0);
}

#[test]
fn cluster_parity() {
    let Some(ctx) = headless() else {
        return;
    };
    let kernel = GpuClothSelfCollisionPoint::new(&ctx);
    // A dense four-particle cluster couples each particle through several pairs,
    // so the per-particle CSR reduction order must match for parity to hold.
    let positions = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(3.0, 1.0, 0.0),
        Vec3::new(1.0, 2.0, 1.0),
        Vec3::new(2.0, 0.0, 3.0),
    ];
    let im = [1.0, 1.0, 1.0, 1.0];
    // Scale down into a penetrating configuration via a large thickness.
    let positions: Vec<Vec3> = positions.iter().map(|p| *p * 0.1).collect();
    let prev = positions.clone();
    assert_parity(&ctx, &kernel, &positions, &prev, &im, 1.0, 0.5, 0.0);
}

#[test]
fn iterated_cluster_parity() {
    let Some(ctx) = headless() else {
        return;
    };
    let kernel = GpuClothSelfCollisionPoint::new(&ctx);
    let positions = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(3.0, 0.0, 0.0),
        Vec3::new(0.0, 3.0, 0.0),
        Vec3::new(3.0, 3.0, 0.0),
    ];
    let positions: Vec<Vec3> = positions.iter().map(|p| *p * 0.1).collect();
    let im = [1.0, 1.0, 1.0, 1.0];
    assert_parity_iterated(&ctx, &kernel, &positions, &im, 1.0, 0.5, 0.0, 16);
}

#[test]
fn empty_candidate_set_is_untouched_parity() {
    let Some(ctx) = headless() else {
        return;
    };
    let kernel = GpuClothSelfCollisionPoint::new(&ctx);
    // Far apart: the host broad phase yields no pairs and both paths no-op.
    let positions = [Vec3::ZERO, Vec3::new(100.0, 0.0, 0.0)];
    let prev = positions;
    let im = [1.0, 1.0];
    assert_parity(&ctx, &kernel, &positions, &prev, &im, 1.0, 0.5, 0.0);
}
