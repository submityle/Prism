//! Real-device parity for the `GPU` body-proxy cloth-collision and backstop
//! kernels against their `CPU` golden twins.
//!
//! Every test acquires a headless device with `GpuContext::try_headless()` and
//! skips cleanly when no adapter is available (for example inside a sandbox),
//! so the suite is a no-op rather than a failure off a real `GPU`.
//!
//! Body collision is *per-particle independent*: one thread owns one particle
//! and walks every collider in slice order (last-collider-to-push wins), the
//! same sequential write set the golden
//! ([`cpu_cloth_body_collision`]) applies. Both delegate the arithmetic to the
//! same `prism_physics_core` projection, so the only divergence is a few `ULP`
//! in `inverseSqrt`/division; parity is checked within a tight relative
//! tolerance rather than bit-for-bit, matching the rest of the solver.
//!
//! Provenance: analytic body-proxy projections and the one-sided backstop are
//! standard position-based collision techniques; the tangential-friction
//! projection is Macklin et al. (2014). No Unreal Engine source or derived code.

use glam::Vec3;
use prism_physics_core::{Backstop, BodyCollider};
use prism_physics_gpu::context::GpuContext;
use prism_physics_gpu::{cpu_cloth_backstops, cpu_cloth_body_collision, GpuClothBodyCollision};

/// A tiny deterministic `xorshift64*` generator so the tests need no external
/// crate for reproducible jitter.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng { state: seed | 1 }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// A float in `[lo, hi)`.
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        let unit = (self.next_u64() >> 40) as f32 / (1_u64 << 24) as f32;
        lo + (hi - lo) * unit
    }
}

/// Relative per-component tolerance: the device computes the radial direction
/// with `inverseSqrt` and the friction scale with a division, so only the low
/// bits can diverge from the sequential golden.
const TOL: f32 = 1.0e-4;

fn close(a: Vec3, b: Vec3) -> bool {
    let scale = a.abs().max(b.abs()).max(Vec3::ONE);
    (a.x - b.x).abs() <= TOL * scale.x
        && (a.y - b.y).abs() <= TOL * scale.y
        && (a.z - b.z).abs() <= TOL * scale.z
}

fn assert_fields_match(cpu: &[Vec3], gpu: &[Vec3]) {
    assert_eq!(cpu.len(), gpu.len(), "field length mismatch");
    for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert!(close(*c, *g), "particle {i}: cpu {c:?} != gpu {g:?}");
    }
}

#[expect(
    clippy::print_stderr,
    reason = "the suite is a deliberate no-op when no GPU adapter is present"
)]
fn headless() -> Option<GpuContext> {
    match GpuContext::try_headless() {
        Some(ctx) => Some(ctx),
        None => {
            eprintln!("skipping cloth body-collision parity: no GPU adapter available");
            None
        }
    }
}

#[test]
fn sphere_projection_matches_cpu() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothBodyCollision::new(&ctx);
    let colliders = [BodyCollider::Sphere {
        center: Vec3::new(0.1, -0.2, 0.3),
        radius: 1.0,
    }];
    let positions = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.4, 0.1, 0.2),
        Vec3::new(2.0, 0.0, 0.0),
    ];
    let inverse_masses = [1.0, 1.0, 1.0];
    let cpu = cpu_cloth_body_collision(&positions, &inverse_masses, &positions, &colliders, 0.0);
    let gpu = kernel.solve(
        &ctx,
        &positions,
        &inverse_masses,
        &positions,
        &colliders,
        0.0,
    );
    assert_fields_match(&cpu, &gpu);
}

#[test]
fn capsule_projection_matches_cpu() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothBodyCollision::new(&ctx);
    let colliders = [BodyCollider::Capsule {
        p0: Vec3::new(-1.0, 0.0, 0.0),
        p1: Vec3::new(1.0, 0.0, 0.0),
        radius: 0.5,
    }];
    let positions = [
        Vec3::new(0.0, 0.2, 0.0),
        Vec3::new(1.5, 0.1, 0.0),
        Vec3::new(-1.3, 0.0, 0.2),
        Vec3::new(0.0, 3.0, 0.0),
    ];
    let inverse_masses = [1.0; 4];
    let cpu = cpu_cloth_body_collision(&positions, &inverse_masses, &positions, &colliders, 0.0);
    let gpu = kernel.solve(
        &ctx,
        &positions,
        &inverse_masses,
        &positions,
        &colliders,
        0.0,
    );
    assert_fields_match(&cpu, &gpu);
}

#[test]
fn half_space_projection_matches_cpu() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothBodyCollision::new(&ctx);
    let colliders = [BodyCollider::HalfSpace {
        normal: Vec3::new(0.0, 2.0, 0.0),
        offset: 0.0,
    }];
    let positions = [
        Vec3::new(0.3, -0.5, 0.2),
        Vec3::new(0.0, 0.5, 0.0),
        Vec3::new(-0.4, -1.2, 0.1),
    ];
    let inverse_masses = [1.0; 3];
    let cpu = cpu_cloth_body_collision(&positions, &inverse_masses, &positions, &colliders, 0.0);
    let gpu = kernel.solve(
        &ctx,
        &positions,
        &inverse_masses,
        &positions,
        &colliders,
        0.0,
    );
    assert_fields_match(&cpu, &gpu);
}

#[test]
fn obb_projection_matches_cpu() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothBodyCollision::new(&ctx);
    let orientation = glam::Quat::from_euler(glam::EulerRot::XYZ, 0.3, -0.7, 1.1);
    let colliders = [BodyCollider::Obb {
        center: Vec3::new(0.2, -0.1, 0.4),
        orientation,
        half_extents: Vec3::new(0.8, 0.4, 0.6),
    }];
    let mut rng = Rng::new(0x0bb_face1);
    // A mix of deep-interior, near-surface, and clearly-outside points so both
    // the push-to-least-penetration face and the outside early-out are hit.
    let positions: Vec<Vec3> = (0..96)
        .map(|_| {
            Vec3::new(
                rng.range(-1.2, 1.2),
                rng.range(-1.2, 1.2),
                rng.range(-1.2, 1.2),
            ) + Vec3::new(0.2, -0.1, 0.4)
        })
        .collect();
    let inverse_masses = vec![1.0f32; positions.len()];
    let cpu = cpu_cloth_body_collision(&positions, &inverse_masses, &positions, &colliders, 0.0);
    let gpu = kernel.solve(
        &ctx,
        &positions,
        &inverse_masses,
        &positions,
        &colliders,
        0.0,
    );
    assert_fields_match(&cpu, &gpu);
}

#[test]
fn multiple_colliders_last_wins_matches_cpu() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothBodyCollision::new(&ctx);
    let colliders = [
        BodyCollider::Sphere {
            center: Vec3::ZERO,
            radius: 1.0,
        },
        BodyCollider::HalfSpace {
            normal: Vec3::Y,
            offset: 0.5,
        },
        BodyCollider::Capsule {
            p0: Vec3::new(-2.0, 0.0, 0.0),
            p1: Vec3::new(2.0, 0.0, 0.0),
            radius: 0.3,
        },
    ];
    let mut rng = Rng::new(0x00dd_cafe);
    let positions: Vec<Vec3> = (0..64)
        .map(|_| {
            Vec3::new(
                rng.range(-1.5, 1.5),
                rng.range(-1.5, 1.5),
                rng.range(-1.5, 1.5),
            )
        })
        .collect();
    let inverse_masses = vec![1.0f32; positions.len()];
    let cpu = cpu_cloth_body_collision(&positions, &inverse_masses, &positions, &colliders, 0.0);
    let gpu = kernel.solve(
        &ctx,
        &positions,
        &inverse_masses,
        &positions,
        &colliders,
        0.0,
    );
    assert_fields_match(&cpu, &gpu);
}

#[test]
fn friction_slide_matches_cpu() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothBodyCollision::new(&ctx);
    let colliders = [BodyCollider::HalfSpace {
        normal: Vec3::Y,
        offset: 0.0,
    }];
    let mut rng = Rng::new(0x5eed_1234);
    // Particles start below the floor (normal push) and have slid tangentially
    // from `prev`, so a non-trivial friction correction is exercised.
    let positions: Vec<Vec3> = (0..48)
        .map(|_| {
            Vec3::new(
                rng.range(-1.0, 1.0),
                rng.range(-0.6, -0.05),
                rng.range(-1.0, 1.0),
            )
        })
        .collect();
    let prev: Vec<Vec3> = positions
        .iter()
        .map(|p| Vec3::new(p.x - rng.range(0.0, 0.4), p.y, p.z - rng.range(0.0, 0.4)))
        .collect();
    let inverse_masses = vec![1.0f32; positions.len()];
    let mu = 0.35;
    let cpu = cpu_cloth_body_collision(&positions, &inverse_masses, &prev, &colliders, mu);
    let gpu = kernel.solve(&ctx, &positions, &inverse_masses, &prev, &colliders, mu);
    assert_fields_match(&cpu, &gpu);
}

#[test]
fn pinned_and_empty_match_cpu() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothBodyCollision::new(&ctx);
    let colliders = [BodyCollider::Sphere {
        center: Vec3::ZERO,
        radius: 1.0,
    }];
    let positions = [Vec3::new(0.3, 0.0, 0.0), Vec3::new(0.0, 0.2, 0.0)];
    // First particle pinned (inverse mass 0): must not move.
    let inverse_masses = [0.0, 1.0];
    let cpu = cpu_cloth_body_collision(&positions, &inverse_masses, &positions, &colliders, 0.0);
    let gpu = kernel.solve(
        &ctx,
        &positions,
        &inverse_masses,
        &positions,
        &colliders,
        0.0,
    );
    assert_fields_match(&cpu, &gpu);
    assert!(
        close(gpu[0], positions[0]),
        "pinned particle moved: {:?}",
        gpu[0]
    );

    // Empty colliders: identity on both sides.
    let gpu_empty = kernel.solve(&ctx, &positions, &inverse_masses, &positions, &[], 0.5);
    assert_fields_match(&positions, &gpu_empty);
}

#[test]
fn backstops_match_cpu() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothBodyCollision::new(&ctx);
    let mut rng = Rng::new(0xbead_f00d);
    let positions: Vec<Vec3> = (0..40)
        .map(|_| {
            Vec3::new(
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
            )
        })
        .collect();
    let backstops: Vec<Backstop> = (0..positions.len())
        .map(|i| Backstop {
            origin: Vec3::new(0.0, 0.0, 0.0),
            normal: Vec3::new(rng.range(-1.0, 1.0), 1.0, rng.range(-1.0, 1.0)),
            distance: 0.1 + 0.2 * (i % 3) as f32,
        })
        .collect();
    // One pinned particle to exercise the skip.
    let mut inverse_masses = vec![1.0f32; positions.len()];
    inverse_masses[5] = 0.0;
    let cpu = cpu_cloth_backstops(&positions, &inverse_masses, &backstops);
    let gpu = kernel.solve_backstops(&ctx, &positions, &inverse_masses, &backstops);
    assert_fields_match(&cpu, &gpu);
}

#[test]
fn backstops_short_slice_matches_cpu() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothBodyCollision::new(&ctx);
    let positions = [
        Vec3::new(0.0, -0.5, 0.0),
        Vec3::new(0.1, -0.5, 0.1),
        Vec3::new(0.2, -0.5, 0.2),
    ];
    // Fewer backstops than particles: trailing particles stay unconstrained.
    let backstops = [Backstop {
        origin: Vec3::ZERO,
        normal: Vec3::Y,
        distance: 0.0,
    }];
    let inverse_masses = [1.0; 3];
    let cpu = cpu_cloth_backstops(&positions, &inverse_masses, &backstops);
    let gpu = kernel.solve_backstops(&ctx, &positions, &inverse_masses, &backstops);
    assert_fields_match(&cpu, &gpu);
}
