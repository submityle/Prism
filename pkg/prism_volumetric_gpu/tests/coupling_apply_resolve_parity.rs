//! Real-device parity for the particle<->rigid-body coupling-*resolve* twin:
//! [`GpuCouplingApplyResolve`](prism_volumetric_gpu::coupling_apply_resolve::GpuCouplingApplyResolve)
//! must reproduce the `CPU` golden
//! [`resolve_coupling`](prism_render_architecture::particle::two_way_coupling::resolve_coupling)
//! — the contact impulse (branch for branch with its three zero-impulse
//! identities) followed by the in-place
//! [`apply_coupling`](prism_render_architecture::particle::two_way_coupling::apply_coupling)
//! update — across a single live contact, a randomized batch of clearly-
//! approaching live contacts, and the three zero-impulse degenerate identities
//! (a separating contact, a near-zero-length normal, and a jointly immovable
//! pair), plus an empty batch.
//!
//! For every contact the four reproduced values are the returned `impulse`, the
//! updated particle linear velocity, and the updated body linear and angular
//! velocities. Each contact owns a private copy of its particle and body, so
//! the pass is one-thread-per-contact with no cross-contact accumulation.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each contact is a fixed, non-reorderable sequence of multiplies, adds and
//! divides, so `CPU` and `GPU` evaluate the same closed form in the same order.
//! They are not bit-exact: a `GPU` may fuse a multiply-add the scalar reference
//! leaves separate, perturbing the low mantissa bits by a few units in the last
//! place. The comparison therefore allows `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3` on every `f32` field.
//!
//! # Conditioning
//!
//! The live fixtures are rejection sampled well away from every branch tie: the
//! contact normal clears a safe non-degenerate length, the relative normal
//! velocity is clearly negative (an approaching contact), and the effective
//! inverse mass sits far above the immovable floor. The degenerate fixtures hit
//! each branch exactly (a clearly positive relative normal velocity, an exactly
//! zero normal, an exactly zero effective inverse mass), so `CPU` and `GPU`
//! take the same branch regardless of a few units in the last place of slack.
//! When the impulse is the zero vector the apply step leaves every velocity
//! unchanged, which the degenerate tests additionally assert.
//!
//! Provenance: twinned from this repository's
//! [`two_way_coupling`](prism_render_architecture::particle::two_way_coupling);
//! no third-party engine source or derived code.

use prism_render_architecture::particle::two_way_coupling::{
    body_point_velocity, generalized_inverse_mass, resolve_coupling, CouplingBody,
    CouplingParticle, Mat3,
};
use prism_render_architecture::particle::Vec3;
use prism_volumetric_gpu::coupling_apply_resolve::{
    GpuCouplingApplyResolve, GpuCouplingApplyResolveQuery,
};
use prism_volumetric_gpu::coupling_impulse::GpuMat3;
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound. A `GPU` may fuse a multiply-add the scalar reference
/// leaves separate, perturbing the low mantissa bits by a few units in the last
/// place; `1e-4` admits that legal slack while still failing a wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in
/// the last place exceed the absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// Asserts two vectors agree channel-for-channel within the parity bound.
fn close_vec(label: &str, idx: usize, got: Vec3, want: Vec3) {
    assert!(
        close(got.x, want.x) && close(got.y, want.y) && close(got.z, want.z),
        "contact {idx} {label}: gpu ({}, {}, {}) vs cpu ({}, {}, {})",
        got.x,
        got.y,
        got.z,
        want.x,
        want.y,
        want.z
    );
}

/// Returns whether `v` is the zero vector within the absolute parity bound.
fn close_zero(v: Vec3) -> bool {
    close(v.x, 0.0) && close(v.y, 0.0) && close(v.z, 0.0)
}

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears (fixtures need no external math library).
/// Returns a value in `[0, 1)`.
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    (bits & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// A pseudo-random value in `[-span, span)` drawn from `state`.
fn signed(state: &mut u64, span: f32) -> f32 {
    (lcg(state) * 2.0 - 1.0) * span
}

/// A pseudo-random value in `[lo, hi)` drawn from `state`.
fn uniform(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + lcg(state) * (hi - lo)
}

/// A pseudo-random vector with each component in `[-span, span)`.
fn rand_vec(state: &mut u64, span: f32) -> Vec3 {
    Vec3::new(
        signed(state, span),
        signed(state, span),
        signed(state, span),
    )
}

/// Rebuilds the golden `CouplingParticle` and `CouplingBody` a query describes,
/// along with the body's world inverse inertia tensor as a golden `Mat3`.
///
/// The matrix arrives already assembled on the query (as it does on the golden
/// `CouplingBody`), so it is copied column for column rather than rebuilt.
fn golden_pair(q: &GpuCouplingApplyResolveQuery) -> (CouplingParticle, CouplingBody, Mat3) {
    let world = Mat3::from_columns(
        q.body_inv_inertia_world.col_x,
        q.body_inv_inertia_world.col_y,
        q.body_inv_inertia_world.col_z,
    );
    let particle = CouplingParticle::new(
        q.particle_inv_mass,
        q.particle_position,
        q.particle_velocity,
    );
    let body = CouplingBody::new(
        q.body_inv_mass,
        world,
        q.body_center_of_mass,
        q.body_linear_velocity,
        q.body_angular_velocity,
    );
    (particle, body, world)
}

/// Dispatches `queries` on device and asserts each resolved contact equals the
/// golden `resolve_coupling` on an identically-initialized copy, within the
/// parity bound, across all four outputs.
fn check(
    ctx: &GpuContext,
    gpu: &GpuCouplingApplyResolve,
    queries: &[GpuCouplingApplyResolveQuery],
) {
    let results = gpu.eval(ctx, queries);
    assert_eq!(
        results.len(),
        queries.len(),
        "one result per contact must be returned"
    );
    for (idx, (q, got)) in queries.iter().zip(&results).enumerate() {
        let (mut particle, mut body, _world) = golden_pair(q);

        // The golden resolve mutates its own copy in place and returns the
        // impulse; the kernel must match the returned impulse and the state it
        // leaves behind on both bodies.
        let impulse = resolve_coupling(&mut particle, &mut body, q.normal, q.restitution);

        close_vec("impulse", idx, got.impulse, impulse);
        close_vec(
            "particle_velocity",
            idx,
            got.particle_velocity,
            particle.velocity,
        );
        close_vec(
            "body_linear_velocity",
            idx,
            got.body_linear_velocity,
            body.linear_velocity,
        );
        close_vec(
            "body_angular_velocity",
            idx,
            got.body_angular_velocity,
            body.angular_velocity,
        );
    }
}

/// Draws one clearly-approaching live contact by rejection sampling.
///
/// Inverse masses are strictly positive, the world inverse inertia tensor is
/// diagonally dominant (a positive diagonal plus small off-diagonal terms), the
/// contact normal clears a safe non-degenerate length, the relative normal
/// velocity is clearly negative so the contact is unambiguously approaching, and
/// the effective inverse mass clears a safe positive margin — keeping `CPU` and
/// `GPU` on the live-solve side of every branch.
fn live_contact(state: &mut u64) -> GpuCouplingApplyResolveQuery {
    loop {
        let diag_x = uniform(state, 0.3, 1.0);
        let diag_y = uniform(state, 0.3, 1.0);
        let diag_z = uniform(state, 0.3, 1.0);
        let off = 0.08;
        let world = GpuMat3::from_columns(
            Vec3::new(diag_x, signed(state, off), signed(state, off)),
            Vec3::new(signed(state, off), diag_y, signed(state, off)),
            Vec3::new(signed(state, off), signed(state, off), diag_z),
        );

        let candidate = GpuCouplingApplyResolveQuery {
            particle_inv_mass: uniform(state, 0.1, 2.0),
            particle_position: rand_vec(state, 2.0),
            particle_velocity: rand_vec(state, 2.0),
            body_inv_mass: uniform(state, 0.1, 2.0),
            body_inv_inertia_world: world,
            body_center_of_mass: rand_vec(state, 2.0),
            body_linear_velocity: rand_vec(state, 2.0),
            body_angular_velocity: rand_vec(state, 2.0),
            normal: rand_vec(state, 1.5),
            restitution: uniform(state, 0.0, 1.0),
        };

        // Reject a near-degenerate normal so both devices normalize identically.
        if candidate.normal.length_squared() < 0.25 {
            continue;
        }

        let (particle, body, world_mat) = golden_pair(&candidate);
        let n = candidate.normal.normalize_or_zero();

        // Require a clearly-approaching contact: relative normal velocity well
        // below zero so no fixture straddles the separating-contact branch.
        let relative = particle
            .velocity
            .sub(body_point_velocity(&body, particle.position));
        let vn = relative.dot(n);
        if vn > -0.2 {
            continue;
        }

        // Require the effective inverse mass clearly above the immovable floor so
        // no fixture straddles that branch.
        let r = particle.position.sub(body.center_of_mass);
        let effective =
            particle.inv_mass + generalized_inverse_mass(body.inv_mass, world_mat, r, n);
        if effective < 0.05 {
            continue;
        }

        return candidate;
    }
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCouplingApplyResolve::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn single_live_contact_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCouplingApplyResolve::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    let query = live_contact(&mut state);
    check(&ctx, &gpu, &[query]);
}

#[test]
fn random_live_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCouplingApplyResolve::new(&ctx);
    let mut state = 0x0bad_c0ffee_u64 ^ 0xa5a5_5a5a_1234_9999;
    let queries: Vec<GpuCouplingApplyResolveQuery> =
        (0..64).map(|_| live_contact(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}

#[test]
fn separating_contact_yields_zero_impulse() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCouplingApplyResolve::new(&ctx);
    // The particle recedes along +y at the contact (relative normal velocity
    // clearly positive), so the reference returns the zero impulse identity and
    // leaves every velocity unchanged; the kernel must match.
    let query = GpuCouplingApplyResolveQuery {
        particle_inv_mass: 0.5,
        particle_position: Vec3::new(0.3, 0.7, -0.2),
        particle_velocity: Vec3::new(0.1, 1.5, -0.1),
        body_inv_mass: 0.4,
        body_inv_inertia_world: GpuMat3::from_columns(
            Vec3::new(0.6, 0.0, 0.0),
            Vec3::new(0.0, 0.4, 0.0),
            Vec3::new(0.0, 0.0, 0.8),
        ),
        body_center_of_mass: Vec3::new(0.0, -0.5, 0.0),
        body_linear_velocity: Vec3::new(0.0, -0.6, 0.0),
        body_angular_velocity: Vec3::new(0.05, -0.1, 0.07),
        normal: Vec3::new(0.0, 1.0, 0.0),
        restitution: 0.4,
    };
    let got = &gpu.eval(&ctx, &[query])[0];
    assert!(
        close_zero(got.impulse),
        "a separating contact must produce the zero impulse, got ({}, {}, {})",
        got.impulse.x,
        got.impulse.y,
        got.impulse.z
    );
    // A zero impulse leaves every velocity unchanged.
    close_vec(
        "particle_velocity",
        0,
        got.particle_velocity,
        query.particle_velocity,
    );
    close_vec(
        "body_linear_velocity",
        0,
        got.body_linear_velocity,
        query.body_linear_velocity,
    );
    close_vec(
        "body_angular_velocity",
        0,
        got.body_angular_velocity,
        query.body_angular_velocity,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn degenerate_normal_yields_zero_impulse() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCouplingApplyResolve::new(&ctx);
    // An exactly zero-length normal carries no contact direction, so both the
    // reference and the kernel return the zero impulse and leave the state
    // unchanged.
    let query = GpuCouplingApplyResolveQuery {
        particle_inv_mass: 0.7,
        particle_position: Vec3::new(-0.4, 0.2, 0.9),
        particle_velocity: Vec3::new(0.3, -0.8, 0.2),
        body_inv_mass: 0.5,
        body_inv_inertia_world: GpuMat3::from_columns(
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(0.0, 0.9, 0.0),
            Vec3::new(0.0, 0.0, 0.3),
        ),
        body_center_of_mass: Vec3::new(0.1, 0.0, -0.1),
        body_linear_velocity: Vec3::new(-0.1, 0.2, 0.05),
        body_angular_velocity: Vec3::new(0.1, 0.0, -0.2),
        normal: Vec3::ZERO,
        restitution: 0.6,
    };
    let got = &gpu.eval(&ctx, &[query])[0];
    assert!(
        close_zero(got.impulse),
        "a degenerate normal must produce the zero impulse, got ({}, {}, {})",
        got.impulse.x,
        got.impulse.y,
        got.impulse.z
    );
    close_vec(
        "particle_velocity",
        0,
        got.particle_velocity,
        query.particle_velocity,
    );
    close_vec(
        "body_linear_velocity",
        0,
        got.body_linear_velocity,
        query.body_linear_velocity,
    );
    close_vec(
        "body_angular_velocity",
        0,
        got.body_angular_velocity,
        query.body_angular_velocity,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn immovable_pair_yields_zero_impulse() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCouplingApplyResolve::new(&ctx);
    // Both inverse masses and the whole world inverse inertia tensor are exactly
    // zero, so the effective inverse mass is zero (at the immovable floor) and no
    // finite impulse exists; both devices return the zero impulse and leave the
    // state unchanged.
    let query = GpuCouplingApplyResolveQuery {
        particle_inv_mass: 0.0,
        particle_position: Vec3::new(0.5, 0.5, 0.5),
        particle_velocity: Vec3::new(0.2, -1.0, 0.3),
        body_inv_mass: 0.0,
        body_inv_inertia_world: GpuMat3::from_columns(Vec3::ZERO, Vec3::ZERO, Vec3::ZERO),
        body_center_of_mass: Vec3::new(0.0, 0.0, 0.0),
        body_linear_velocity: Vec3::new(0.0, 0.0, 0.0),
        body_angular_velocity: Vec3::new(0.0, 0.0, 0.0),
        normal: Vec3::new(0.0, 1.0, 0.0),
        restitution: 0.5,
    };
    let got = &gpu.eval(&ctx, &[query])[0];
    assert!(
        close_zero(got.impulse),
        "a jointly immovable pair must produce the zero impulse, got ({}, {}, {})",
        got.impulse.x,
        got.impulse.y,
        got.impulse.z
    );
    close_vec(
        "particle_velocity",
        0,
        got.particle_velocity,
        query.particle_velocity,
    );
    close_vec(
        "body_linear_velocity",
        0,
        got.body_linear_velocity,
        query.body_linear_velocity,
    );
    close_vec(
        "body_angular_velocity",
        0,
        got.body_angular_velocity,
        query.body_angular_velocity,
    );
    check(&ctx, &gpu, &[query]);
}
