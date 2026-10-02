//! Real-device parity for the particle<->rigid-body coupling-impulse twin:
//! [`GpuCouplingImpulse`](prism_volumetric_gpu::coupling_impulse::GpuCouplingImpulse)
//! must reproduce the `CPU` golden
//! [`two_way_coupling`](prism_render_architecture::particle::two_way_coupling)
//! per-contact function cluster — the world inverse inertia tensor
//! `R * diag(principal) * Rᵀ`, the body surface velocity at the contact, the
//! generalized inverse mass along the normalized contact normal, the contact
//! impulse vector, and the body linear momentum — across a single live
//! contact, a randomized batch of clearly-approaching live contacts, and the
//! three zero-impulse degenerate identities (a separating contact, a
//! near-zero-length normal, and a jointly immovable pair), plus an
//! identity-rotation sanity contact and an empty batch.
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
//! velocity is clearly negative (an approaching contact), and both inverse
//! masses are strictly positive so the effective inverse mass sits far above
//! the immovable floor. The degenerate fixtures hit each branch exactly (a
//! clearly positive relative normal velocity, an exactly zero normal, an
//! exactly zero effective inverse mass), so `CPU` and `GPU` take the same
//! branch regardless of a few units in the last place of slack.
//!
//! Provenance: twinned from this repository's
//! [`two_way_coupling`](prism_render_architecture::particle::two_way_coupling);
//! no third-party engine source or derived code.

use prism_render_architecture::particle::two_way_coupling::{
    body_point_velocity, coupling_impulse, generalized_inverse_mass, inv_inertia_world,
    linear_momentum, CouplingBody, CouplingParticle, Mat3,
};
use prism_render_architecture::particle::Vec3;
use prism_volumetric_gpu::coupling_impulse::{
    GpuCouplingImpulse, GpuCouplingImpulseQuery, GpuMat3,
};
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

/// Asserts two matrices agree column-for-column within the parity bound.
fn close_mat(idx: usize, got: GpuMat3, want: Mat3) {
    close_vec("inv_inertia_world col_x", idx, got.col_x, want.col_x);
    close_vec("inv_inertia_world col_y", idx, got.col_y, want.col_y);
    close_vec("inv_inertia_world col_z", idx, got.col_z, want.col_z);
}

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears. Returns a value in `[0, 1)`.
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

/// Rebuilds the golden `CouplingParticle` and `CouplingBody` a query describes.
///
/// The body's world inverse inertia tensor is assembled with the golden
/// `inv_inertia_world` from the query's principal inverse inertia and rotation
/// columns, exactly as the kernel does on device.
fn golden_pair(q: &GpuCouplingImpulseQuery) -> (CouplingParticle, CouplingBody, Mat3) {
    let rotation = Mat3::from_columns(q.rotation.col_x, q.rotation.col_y, q.rotation.col_z);
    let world = inv_inertia_world(q.principal_inv_inertia, rotation);
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

/// Dispatches `queries` on device and asserts each result equals the golden
/// per-contact cluster within the parity bound.
fn check(ctx: &GpuContext, gpu: &GpuCouplingImpulse, queries: &[GpuCouplingImpulseQuery]) {
    let results = gpu.eval(ctx, queries);
    assert_eq!(
        results.len(),
        queries.len(),
        "one result per contact must be returned"
    );
    for (idx, (q, got)) in queries.iter().zip(&results).enumerate() {
        let (particle, body, world) = golden_pair(q);

        // World inverse inertia tensor R * diag(principal) * Rᵀ.
        close_mat(idx, got.inv_inertia_world, world);

        // Body surface velocity at the contact point (the particle position).
        let bpv = body_point_velocity(&body, particle.position);
        close_vec("body_point_velocity", idx, got.body_point_velocity, bpv);

        // Generalized inverse mass along the normalized contact normal, with the
        // lever arm r = contact - center_of_mass, exactly as the kernel builds it.
        let n = q.normal.normalize_or_zero();
        let r = particle.position.sub(body.center_of_mass);
        let gim = generalized_inverse_mass(body.inv_mass, world, r, n);
        assert!(
            close(got.generalized_inverse_mass, gim),
            "contact {idx} generalized_inverse_mass: gpu {} vs cpu {}",
            got.generalized_inverse_mass,
            gim
        );

        // Contact impulse vector (the three zero-impulse identities included).
        let imp = coupling_impulse(&particle, &body, q.normal, q.restitution);
        close_vec("impulse", idx, got.impulse, imp);

        // Body linear momentum mass * velocity.
        let mom = linear_momentum(q.body_mass, body.linear_velocity);
        close_vec("body_linear_momentum", idx, got.body_linear_momentum, mom);
    }
}

/// Draws one clearly-approaching live contact by rejection sampling.
///
/// Inverse masses are strictly positive, the principal inverse inertia entries
/// are strictly positive, the contact normal clears a safe non-degenerate
/// length, and the relative normal velocity is clearly negative so the contact
/// is unambiguously approaching — keeping `CPU` and `GPU` on the live-solve side
/// of every branch.
fn live_contact(state: &mut u64) -> GpuCouplingImpulseQuery {
    loop {
        let candidate = GpuCouplingImpulseQuery {
            particle_inv_mass: uniform(state, 0.1, 2.0),
            particle_position: rand_vec(state, 2.0),
            particle_velocity: rand_vec(state, 2.0),
            body_inv_mass: uniform(state, 0.1, 2.0),
            body_mass: uniform(state, 0.5, 4.0),
            body_center_of_mass: rand_vec(state, 2.0),
            body_linear_velocity: rand_vec(state, 2.0),
            body_angular_velocity: rand_vec(state, 2.0),
            principal_inv_inertia: Vec3::new(
                uniform(state, 0.1, 1.0),
                uniform(state, 0.1, 1.0),
                uniform(state, 0.1, 1.0),
            ),
            rotation: GpuMat3::from_columns(
                rand_vec(state, 1.0),
                rand_vec(state, 1.0),
                rand_vec(state, 1.0),
            ),
            normal: rand_vec(state, 1.5),
            restitution: uniform(state, 0.0, 1.0),
        };

        // Reject a near-degenerate normal so both devices normalize identically.
        if candidate.normal.length_squared() < 0.25 {
            continue;
        }

        // Require a clearly-approaching contact: relative normal velocity well
        // below zero so no fixture straddles the separating-contact branch.
        let (particle, body, _) = golden_pair(&candidate);
        let n = candidate.normal.normalize_or_zero();
        let relative = particle
            .velocity
            .sub(body_point_velocity(&body, particle.position));
        let vn = relative.dot(n);
        if vn > -0.2 {
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
    let gpu = GpuCouplingImpulse::new(&ctx);
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
    let gpu = GpuCouplingImpulse::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    let query = live_contact(&mut state);
    check(&ctx, &gpu, &[query]);
}

#[test]
fn random_live_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCouplingImpulse::new(&ctx);
    let mut state = 0x0bad_c0ffee_u64 ^ 0xa5a5_5a5a_1234_9999;
    let queries: Vec<GpuCouplingImpulseQuery> = (0..64).map(|_| live_contact(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}

#[test]
fn separating_contact_yields_zero_impulse() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCouplingImpulse::new(&ctx);
    // The particle recedes along +y at the contact (relative normal velocity
    // clearly positive), so the reference returns the zero impulse identity and
    // the kernel must match. All other readouts are still exercised.
    let query = GpuCouplingImpulseQuery {
        particle_inv_mass: 0.5,
        particle_position: Vec3::new(0.3, 0.7, -0.2),
        particle_velocity: Vec3::new(0.1, 1.5, -0.1),
        body_inv_mass: 0.4,
        body_mass: 2.5,
        body_center_of_mass: Vec3::new(0.0, -0.5, 0.0),
        body_linear_velocity: Vec3::new(0.0, -0.6, 0.0),
        body_angular_velocity: Vec3::new(0.05, -0.1, 0.07),
        principal_inv_inertia: Vec3::new(0.6, 0.4, 0.8),
        rotation: GpuMat3::IDENTITY,
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
    check(&ctx, &gpu, &[query]);
}

#[test]
fn degenerate_normal_yields_zero_impulse() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCouplingImpulse::new(&ctx);
    // An exactly zero-length normal carries no contact direction, so both the
    // reference and the kernel return the zero impulse.
    let query = GpuCouplingImpulseQuery {
        particle_inv_mass: 0.7,
        particle_position: Vec3::new(-0.4, 0.2, 0.9),
        particle_velocity: Vec3::new(0.3, -0.8, 0.2),
        body_inv_mass: 0.5,
        body_mass: 2.0,
        body_center_of_mass: Vec3::new(0.1, 0.0, -0.1),
        body_linear_velocity: Vec3::new(-0.1, 0.2, 0.05),
        body_angular_velocity: Vec3::new(0.1, 0.0, -0.2),
        principal_inv_inertia: Vec3::new(0.5, 0.9, 0.3),
        rotation: GpuMat3::IDENTITY,
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
    check(&ctx, &gpu, &[query]);
}

#[test]
fn immovable_pair_yields_zero_impulse() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCouplingImpulse::new(&ctx);
    // Both inverse masses and the whole principal inverse inertia are exactly
    // zero, so the effective inverse mass is zero (at the immovable floor) and
    // no finite impulse exists; both devices return the zero impulse.
    let query = GpuCouplingImpulseQuery {
        particle_inv_mass: 0.0,
        particle_position: Vec3::new(0.5, 0.5, 0.5),
        particle_velocity: Vec3::new(0.2, -1.0, 0.3),
        body_inv_mass: 0.0,
        body_mass: 0.0,
        body_center_of_mass: Vec3::new(0.0, 0.0, 0.0),
        body_linear_velocity: Vec3::new(0.0, 0.0, 0.0),
        body_angular_velocity: Vec3::new(0.0, 0.0, 0.0),
        principal_inv_inertia: Vec3::ZERO,
        rotation: GpuMat3::IDENTITY,
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
    check(&ctx, &gpu, &[query]);
}

#[test]
fn identity_rotation_tensor_is_diagonal() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCouplingImpulse::new(&ctx);
    // With the identity rotation the world inverse inertia tensor is the
    // principal diagonal; the kernel must reproduce the golden build along with
    // a live approaching contact.
    let query = GpuCouplingImpulseQuery {
        particle_inv_mass: 0.8,
        particle_position: Vec3::new(0.0, 1.0, 0.0),
        particle_velocity: Vec3::new(0.0, -2.0, 0.0),
        body_inv_mass: 0.5,
        body_mass: 2.0,
        body_center_of_mass: Vec3::new(0.0, 0.0, 0.0),
        body_linear_velocity: Vec3::new(0.0, 0.0, 0.0),
        body_angular_velocity: Vec3::new(0.0, 0.0, 0.0),
        principal_inv_inertia: Vec3::new(2.0, 3.0, 4.0),
        rotation: GpuMat3::IDENTITY,
        normal: Vec3::new(0.0, 1.0, 0.0),
        restitution: 0.3,
    };
    let got = &gpu.eval(&ctx, &[query])[0];
    // The world tensor equals the principal diagonal.
    close_vec(
        "diag col_x",
        0,
        got.inv_inertia_world.col_x,
        Vec3::new(2.0, 0.0, 0.0),
    );
    close_vec(
        "diag col_y",
        0,
        got.inv_inertia_world.col_y,
        Vec3::new(0.0, 3.0, 0.0),
    );
    close_vec(
        "diag col_z",
        0,
        got.inv_inertia_world.col_z,
        Vec3::new(0.0, 0.0, 4.0),
    );
    check(&ctx, &gpu, &[query]);
}

/// Returns whether `v` is the zero vector within the absolute parity bound.
fn close_zero(v: Vec3) -> bool {
    close(v.x, 0.0) && close(v.y, 0.0) && close(v.z, 0.0)
}
