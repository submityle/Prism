//! Real-device parity for the particle<->rigid-body two-way-coupling twin:
//! [`GpuTwoWayCoupling`](prism_volumetric_gpu::two_way_coupling::GpuTwoWayCoupling)
//! must reproduce the `CPU` golden
//! [`two_way_coupling`](prism_render_architecture::particle::two_way_coupling)
//! across the `Mat3` primitives (`mul_vec3` / `transpose` / `mul_mat3`), the
//! world inverse inertia tensor `R * diag(principal) * Rᵀ`, the generalized
//! inverse mass, the body surface velocity `v + omega × r`, the contact impulse
//! `j * n`, the resolved (updated) velocities, and the linear momentum.
//!
//! The fixtures cover the shapes the golden unit tests call out: an orthonormal
//! `3-4-5` rotation and the identity for the matrix / inertia work, a clearly
//! *approaching* contact (relative normal velocity well below zero) for the
//! impulse and resolve solves, and a clearly *separating* contact that must
//! collapse to the zero-impulse identity on both sides. Every rotation is built
//! from the exact `3-4-5` columns and every other input is a simple decimal, so
//! the fixtures stay pure and need no external math library and no transcendental
//! method. The random batch draws from a host-side `u64` `LCG` and rejection
//! samples away from the degenerate branches (near-zero normal, grazing contact)
//! so the comparison always exercises the live solve.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every twinned quantity threads through multiplies, adds and one guarded
//! division, so `CPU` and `GPU` are compared under tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`). The
//! zero-impulse identities land on an exact zero on both sides, which the same
//! tolerance accepts.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::two_way_coupling`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::two_way_coupling::{
    body_point_velocity, coupling_impulse, generalized_inverse_mass, inv_inertia_world,
    linear_momentum, resolve_coupling, CouplingBody, CouplingParticle, Mat3,
};
use prism_render_architecture::particle::Vec3;
use prism_volumetric_gpu::two_way_coupling::{
    GpuTwoWayCoupling, TwoWayCouplingQuery, TwoWayCouplingResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for one continuous `f32` lane.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for one continuous `f32` lane.
const REL_EPS: f32 = 1.0e-3;
/// Floor for the relative-tolerance denominator.
const REL_FLOOR: f32 = 1.0e-6;

/// Mixed absolute / relative tolerance comparison for one `f32` lane.
fn approx(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_EPS
}

/// Tolerant comparison of two [`Vec3`] values, lane by lane.
fn approx_vec(a: Vec3, b: Vec3) -> bool {
    approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z)
}

/// Builds a [`Mat3`] from its three column-major columns `[col_x, col_y,
/// col_z]`.
fn mat(columns: [Vec3; 3]) -> Mat3 {
    Mat3::from_columns(columns[0], columns[1], columns[2])
}

/// Unpacks a [`Mat3`] into its three columns `[col_x, col_y, col_z]`.
fn cols(m: Mat3) -> [Vec3; 3] {
    [m.col_x, m.col_y, m.col_z]
}

/// An orthonormal rotation built from an exact `3-4-5` turn in the `xy` plane,
/// so the fixture needs no transcendental method to stay orthonormal.
fn rot_345() -> [Vec3; 3] {
    [
        Vec3::new(0.6, 0.8, 0.0),
        Vec3::new(-0.8, 0.6, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    ]
}

/// The identity rotation, as column-major columns.
fn identity_cols() -> [Vec3; 3] {
    cols(Mat3::IDENTITY)
}

/// Rebuilds the golden `CPU` particle, body, normal and restitution from a
/// `CouplingImpulse` or `ResolveCoupling` query so the reference can be
/// evaluated with the exact same inputs the twin saw.
fn coupling_inputs(q: &TwoWayCouplingQuery) -> (CouplingParticle, CouplingBody, Vec3, f32) {
    match q {
        TwoWayCouplingQuery::CouplingImpulse {
            particle_inv_mass,
            particle_position,
            particle_velocity,
            body_inv_mass,
            body_inv_inertia_world,
            body_center_of_mass,
            body_linear_velocity,
            body_angular_velocity,
            normal,
            restitution,
        }
        | TwoWayCouplingQuery::ResolveCoupling {
            particle_inv_mass,
            particle_position,
            particle_velocity,
            body_inv_mass,
            body_inv_inertia_world,
            body_center_of_mass,
            body_linear_velocity,
            body_angular_velocity,
            normal,
            restitution,
        } => {
            let particle =
                CouplingParticle::new(*particle_inv_mass, *particle_position, *particle_velocity);
            let body = CouplingBody::new(
                *body_inv_mass,
                mat(*body_inv_inertia_world),
                *body_center_of_mass,
                *body_linear_velocity,
                *body_angular_velocity,
            );
            (particle, body, *normal, *restitution)
        }
        _ => panic!("coupling_inputs is only valid for impulse / resolve queries"),
    }
}

/// Asserts a [`TwoWayCouplingResult`] is the expected single vector.
fn assert_vector(got: TwoWayCouplingResult, expected: Vec3) {
    match got {
        TwoWayCouplingResult::Vector(v) => {
            assert!(
                approx_vec(v, expected),
                "vector mismatch: gpu {v:?} vs cpu {expected:?}"
            );
        }
        other => panic!("expected a vector result, got {other:?}"),
    }
}

/// Asserts a [`TwoWayCouplingResult`] is the expected column-major matrix.
fn assert_matrix(got: TwoWayCouplingResult, expected: Mat3) {
    match got {
        TwoWayCouplingResult::Matrix(m) => {
            let e = cols(expected);
            assert!(
                approx_vec(m[0], e[0]) && approx_vec(m[1], e[1]) && approx_vec(m[2], e[2]),
                "matrix mismatch: gpu {m:?} vs cpu {e:?}"
            );
        }
        other => panic!("expected a matrix result, got {other:?}"),
    }
}

/// Asserts the twinned answer for one query matches the `CPU` golden, dispatching
/// on the query variant to pick the reference function and the result shape.
fn assert_parity(gpu: &GpuTwoWayCoupling, ctx: &GpuContext, q: &TwoWayCouplingQuery) {
    let got = gpu.evaluate(ctx, std::slice::from_ref(q));
    assert_eq!(got.len(), 1, "one result per query");
    let g = got[0];
    match q {
        TwoWayCouplingQuery::MatMulVec3 { matrix, vector } => {
            assert_vector(g, mat(*matrix).mul_vec3(*vector));
        }
        TwoWayCouplingQuery::MatTranspose { matrix } => {
            assert_matrix(g, mat(*matrix).transpose());
        }
        TwoWayCouplingQuery::MatMulMat3 { lhs, rhs } => {
            assert_matrix(g, mat(*lhs).mul_mat3(mat(*rhs)));
        }
        TwoWayCouplingQuery::InvInertiaWorld {
            principal_inv_inertia,
            rotation,
        } => {
            assert_matrix(g, inv_inertia_world(*principal_inv_inertia, mat(*rotation)));
        }
        TwoWayCouplingQuery::GeneralizedInverseMass {
            inv_mass,
            inv_inertia_world: tensor,
            lever_arm,
            direction,
        } => {
            let expected =
                generalized_inverse_mass(*inv_mass, mat(*tensor), *lever_arm, *direction);
            match g {
                TwoWayCouplingResult::Scalar(s) => {
                    assert!(
                        approx(s, expected),
                        "scalar mismatch: gpu {s} vs cpu {expected}"
                    );
                }
                other => panic!("expected a scalar result, got {other:?}"),
            }
        }
        TwoWayCouplingQuery::BodyPointVelocity {
            center_of_mass,
            linear_velocity,
            angular_velocity,
            point,
        } => {
            let body = CouplingBody::new(
                0.0,
                Mat3::IDENTITY,
                *center_of_mass,
                *linear_velocity,
                *angular_velocity,
            );
            assert_vector(g, body_point_velocity(&body, *point));
        }
        TwoWayCouplingQuery::CouplingImpulse { .. } => {
            let (particle, body, normal, restitution) = coupling_inputs(q);
            assert_vector(g, coupling_impulse(&particle, &body, normal, restitution));
        }
        TwoWayCouplingQuery::ResolveCoupling { .. } => {
            let (mut particle, mut body, normal, restitution) = coupling_inputs(q);
            let impulse = resolve_coupling(&mut particle, &mut body, normal, restitution);
            match g {
                TwoWayCouplingResult::Velocities {
                    impulse: gi,
                    particle_velocity,
                    body_linear_velocity,
                    body_angular_velocity,
                } => {
                    assert!(
                        approx_vec(gi, impulse),
                        "impulse mismatch: gpu {gi:?} vs cpu {impulse:?}"
                    );
                    assert!(
                        approx_vec(particle_velocity, particle.velocity),
                        "particle velocity mismatch: gpu {particle_velocity:?} vs cpu {:?}",
                        particle.velocity
                    );
                    assert!(
                        approx_vec(body_linear_velocity, body.linear_velocity),
                        "body linear mismatch: gpu {body_linear_velocity:?} vs cpu {:?}",
                        body.linear_velocity
                    );
                    assert!(
                        approx_vec(body_angular_velocity, body.angular_velocity),
                        "body angular mismatch: gpu {body_angular_velocity:?} vs cpu {:?}",
                        body.angular_velocity
                    );
                }
                other => panic!("expected a velocities result, got {other:?}"),
            }
        }
        TwoWayCouplingQuery::LinearMomentum { mass, velocity } => {
            assert_vector(g, linear_momentum(*mass, *velocity));
        }
    }
}

/// A clearly *approaching* contact (relative normal velocity well below zero)
/// built on an orthonormal `3-4-5` world inverse inertia tensor, as a resolve
/// query. The sibling `impulse` builder reuses the same numbers.
fn approaching_resolve() -> TwoWayCouplingQuery {
    let tensor = cols(inv_inertia_world(Vec3::new(0.8, 0.5, 0.9), mat(rot_345())));
    TwoWayCouplingQuery::ResolveCoupling {
        particle_inv_mass: 0.5,
        particle_position: Vec3::new(0.2, 1.0, -0.1),
        particle_velocity: Vec3::new(0.1, -3.0, 0.2),
        body_inv_mass: 0.25,
        body_inv_inertia_world: tensor,
        body_center_of_mass: Vec3::new(0.0, -0.1, 0.0),
        body_linear_velocity: Vec3::new(-0.2, 0.3, 0.1),
        body_angular_velocity: Vec3::new(0.1, -0.2, 0.3),
        normal: Vec3::new(0.1, 0.95, -0.05),
        restitution: 0.5,
    }
}

/// The same approaching contact as a `coupling_impulse` query.
fn approaching_impulse() -> TwoWayCouplingQuery {
    let tensor = cols(inv_inertia_world(Vec3::new(0.8, 0.5, 0.9), mat(rot_345())));
    TwoWayCouplingQuery::CouplingImpulse {
        particle_inv_mass: 0.5,
        particle_position: Vec3::new(0.2, 1.0, -0.1),
        particle_velocity: Vec3::new(0.1, -3.0, 0.2),
        body_inv_mass: 0.25,
        body_inv_inertia_world: tensor,
        body_center_of_mass: Vec3::new(0.0, -0.1, 0.0),
        body_linear_velocity: Vec3::new(-0.2, 0.3, 0.1),
        body_angular_velocity: Vec3::new(0.1, -0.2, 0.3),
        normal: Vec3::new(0.1, 0.95, -0.05),
        restitution: 0.5,
    }
}

/// A clearly *separating* contact (relative normal velocity well above zero) as
/// a `coupling_impulse` query, so both sides take the zero-impulse identity.
fn separating_impulse() -> TwoWayCouplingQuery {
    let tensor = cols(inv_inertia_world(Vec3::new(0.8, 0.5, 0.9), mat(rot_345())));
    TwoWayCouplingQuery::CouplingImpulse {
        particle_inv_mass: 0.5,
        particle_position: Vec3::new(0.2, 1.0, -0.1),
        particle_velocity: Vec3::new(0.1, 3.0, 0.2),
        body_inv_mass: 0.25,
        body_inv_inertia_world: tensor,
        body_center_of_mass: Vec3::new(0.0, -0.1, 0.0),
        body_linear_velocity: Vec3::new(-0.2, 0.3, 0.1),
        body_angular_velocity: Vec3::new(0.1, -0.2, 0.3),
        normal: Vec3::new(0.0, 1.0, 0.0),
        restitution: 0.5,
    }
}

/// A tiny host-side `u64` `LCG`, used only to drive the deterministic random
/// batch; it uses no transcendental method and no external math library.
struct Lcg {
    /// Current generator state.
    state: u64,
}

impl Lcg {
    /// Seeds the generator.
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Advances the state and returns the high `32` bits.
    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.state >> 32) as u32
    }

    /// Returns a sample in `0.0..1.0` from the top `24` mantissa bits.
    fn unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1_u32 << 24) as f32
    }

    /// Returns a signed sample in `-1.0..1.0`.
    fn signed(&mut self) -> f32 {
        self.unit() * 2.0 - 1.0
    }

    /// Returns a signed [`Vec3`] with each lane in `-1.0..1.0`.
    fn vec3(&mut self) -> Vec3 {
        Vec3::new(self.signed(), self.signed(), self.signed())
    }
}

/// Draws a random, non-degenerate resolve query: positive inverse masses, an
/// orthonormal `3-4-5` inertia frame, a normal well clear of zero length, and a
/// contact that is clearly approaching (relative normal velocity below a margin
/// so the fixture never straddles the grazing branch).
fn random_resolve(rng: &mut Lcg) -> TwoWayCouplingQuery {
    loop {
        let principal = Vec3::new(
            0.4 + 0.6 * rng.unit(),
            0.4 + 0.6 * rng.unit(),
            0.4 + 0.6 * rng.unit(),
        );
        let tensor = inv_inertia_world(principal, mat(rot_345()));
        let particle = CouplingParticle::new(0.25 + rng.unit(), rng.vec3(), rng.vec3().scale(2.0));
        let body = CouplingBody::new(
            0.2 + 0.8 * rng.unit(),
            tensor,
            rng.vec3().scale(0.3),
            rng.vec3().scale(0.5),
            rng.vec3().scale(0.5),
        );
        let raw_normal = rng.vec3();
        if raw_normal.length_squared() < 0.25 {
            continue;
        }
        let n = raw_normal.normalize_or_zero();
        let contact = particle.position;
        let vn = particle
            .velocity
            .sub(body_point_velocity(&body, contact))
            .dot(n);
        if vn > -0.3 {
            continue;
        }
        return TwoWayCouplingQuery::ResolveCoupling {
            particle_inv_mass: particle.inv_mass,
            particle_position: particle.position,
            particle_velocity: particle.velocity,
            body_inv_mass: body.inv_mass,
            body_inv_inertia_world: cols(body.inv_inertia_world),
            body_center_of_mass: body.center_of_mass,
            body_linear_velocity: body.linear_velocity,
            body_angular_velocity: body.angular_velocity,
            normal: raw_normal,
            restitution: rng.unit(),
        };
    }
}

#[test]
fn mat3_primitives_match() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTwoWayCoupling::new(&ctx);
    let a = [
        Vec3::new(1.0, 2.0, -1.0),
        Vec3::new(0.5, -0.5, 2.0),
        Vec3::new(-1.5, 1.0, 0.5),
    ];
    let b = rot_345();
    assert_parity(
        &gpu,
        &ctx,
        &TwoWayCouplingQuery::MatMulVec3 {
            matrix: a,
            vector: Vec3::new(0.3, -1.2, 0.7),
        },
    );
    assert_parity(&gpu, &ctx, &TwoWayCouplingQuery::MatTranspose { matrix: a });
    assert_parity(
        &gpu,
        &ctx,
        &TwoWayCouplingQuery::MatMulMat3 { lhs: a, rhs: b },
    );
}

#[test]
fn inv_inertia_world_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTwoWayCoupling::new(&ctx);
    // Identity rotation leaves the tensor diagonal; the 3-4-5 rotation mixes the
    // xy block while leaving z untouched.
    assert_parity(
        &gpu,
        &ctx,
        &TwoWayCouplingQuery::InvInertiaWorld {
            principal_inv_inertia: Vec3::new(2.0, 3.0, 4.0),
            rotation: identity_cols(),
        },
    );
    assert_parity(
        &gpu,
        &ctx,
        &TwoWayCouplingQuery::InvInertiaWorld {
            principal_inv_inertia: Vec3::new(0.8, 0.5, 0.9),
            rotation: rot_345(),
        },
    );
}

#[test]
fn generalized_inverse_mass_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTwoWayCoupling::new(&ctx);
    let tensor = cols(inv_inertia_world(Vec3::new(0.8, 0.5, 0.9), mat(rot_345())));
    assert_parity(
        &gpu,
        &ctx,
        &TwoWayCouplingQuery::GeneralizedInverseMass {
            inv_mass: 0.25,
            inv_inertia_world: tensor,
            lever_arm: Vec3::new(0.2, 1.1, -0.1),
            direction: Vec3::new(0.0, 1.0, 0.0),
        },
    );
}

#[test]
fn body_point_velocity_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTwoWayCoupling::new(&ctx);
    // Spin about +Z at 2 rad/s; a point at +X on the body moves toward +Y.
    assert_parity(
        &gpu,
        &ctx,
        &TwoWayCouplingQuery::BodyPointVelocity {
            center_of_mass: Vec3::ZERO,
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::new(0.0, 0.0, 2.0),
            point: Vec3::new(1.0, 0.0, 0.0),
        },
    );
    assert_parity(
        &gpu,
        &ctx,
        &TwoWayCouplingQuery::BodyPointVelocity {
            center_of_mass: Vec3::new(0.1, -0.2, 0.3),
            linear_velocity: Vec3::new(-0.4, 0.5, 0.2),
            angular_velocity: Vec3::new(0.3, -0.1, 0.6),
            point: Vec3::new(0.7, 0.2, -0.5),
        },
    );
}

#[test]
fn coupling_impulse_matches_and_zeroes_on_separation() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTwoWayCoupling::new(&ctx);
    assert_parity(&gpu, &ctx, &approaching_impulse());
    // A separating contact must collapse to the zero-impulse identity on both
    // sides.
    assert_parity(&gpu, &ctx, &separating_impulse());
}

#[test]
fn resolve_coupling_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTwoWayCoupling::new(&ctx);
    assert_parity(&gpu, &ctx, &approaching_resolve());
}

#[test]
fn linear_momentum_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTwoWayCoupling::new(&ctx);
    assert_parity(
        &gpu,
        &ctx,
        &TwoWayCouplingQuery::LinearMomentum {
            mass: 4.0,
            velocity: Vec3::new(0.2, -1.5, 0.7),
        },
    );
}

#[test]
fn mixed_batch_matches_elementwise() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTwoWayCoupling::new(&ctx);
    // One dispatch mixing every query variant exercises the tagged-union batch
    // and the one-thread-per-query flattening; each result must be independent.
    let a = [
        Vec3::new(1.0, 2.0, -1.0),
        Vec3::new(0.5, -0.5, 2.0),
        Vec3::new(-1.5, 1.0, 0.5),
    ];
    let tensor = cols(inv_inertia_world(Vec3::new(0.8, 0.5, 0.9), mat(rot_345())));
    let batch = [
        TwoWayCouplingQuery::MatMulVec3 {
            matrix: a,
            vector: Vec3::new(0.3, -1.2, 0.7),
        },
        TwoWayCouplingQuery::MatTranspose { matrix: a },
        TwoWayCouplingQuery::MatMulMat3 {
            lhs: a,
            rhs: rot_345(),
        },
        TwoWayCouplingQuery::InvInertiaWorld {
            principal_inv_inertia: Vec3::new(0.8, 0.5, 0.9),
            rotation: rot_345(),
        },
        TwoWayCouplingQuery::GeneralizedInverseMass {
            inv_mass: 0.25,
            inv_inertia_world: tensor,
            lever_arm: Vec3::new(0.2, 1.1, -0.1),
            direction: Vec3::new(0.0, 1.0, 0.0),
        },
        TwoWayCouplingQuery::BodyPointVelocity {
            center_of_mass: Vec3::new(0.1, -0.2, 0.3),
            linear_velocity: Vec3::new(-0.4, 0.5, 0.2),
            angular_velocity: Vec3::new(0.3, -0.1, 0.6),
            point: Vec3::new(0.7, 0.2, -0.5),
        },
        approaching_impulse(),
        separating_impulse(),
        approaching_resolve(),
        TwoWayCouplingQuery::LinearMomentum {
            mass: 4.0,
            velocity: Vec3::new(0.2, -1.5, 0.7),
        },
    ];
    let got = gpu.evaluate(&ctx, &batch);
    assert_eq!(got.len(), batch.len());
    for q in &batch {
        assert_parity(&gpu, &ctx, q);
    }
}

#[test]
fn random_batch_matches_elementwise() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTwoWayCoupling::new(&ctx);
    // A deterministic swarm of non-degenerate resolve contacts, each rejection
    // sampled to be clearly approaching, dispatched as one batch.
    let mut rng = Lcg::new(0x9E37_79B9_7F4A_7C15);
    let batch: Vec<TwoWayCouplingQuery> = (0..24).map(|_| random_resolve(&mut rng)).collect();
    let got = gpu.evaluate(&ctx, &batch);
    assert_eq!(got.len(), batch.len());
    for q in &batch {
        assert_parity(&gpu, &ctx, q);
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTwoWayCoupling::new(&ctx);
    // No dispatch is issued and the result vector is empty.
    assert!(gpu.evaluate(&ctx, &[]).is_empty());
}
