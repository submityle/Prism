//! Real-device parity for the isolated barrier-contact twin:
//! [`GpuHairBarrierContact`] must reproduce the `CPU` golden
//! [`reference_resolve`](prism_hair_gpu::barrier_contact::reference_resolve)
//! (built on
//! [`resolve_contact`](prism_render_architecture::hair::barrier_contact::resolve_contact))
//! for a batch of contacts, each resolved independently from its original
//! endpoint state (`Jacobi`). The suite drives the normal-only repulsion, the
//! sticking and slipping `Coulomb` friction branches, a pinned endpoint, the
//! three no-op guards (both pinned, degenerate normal, gap at/beyond `dhat`),
//! the distance-floor clamp, an oblique (non-axis) normal, the empty no-op, and
//! a large multi-workgroup batch that crosses the 64-wide dispatch boundary.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Positions, velocities and both impulses are closed-form but a `GPU` may fuse
//! multiply-adds the scalar reference leaves separate, so each scalar is
//! asserted within `abs_diff < 1e-4` or `rel_diff < 1e-3`. The `slipping` flag
//! is a branch pick, so every case sits well clear of the `Coulomb` cone
//! threshold and the flag is asserted exactly. No `sin`/`cos` appears anywhere;
//! all inputs are explicit literals.
//!
//! Provenance: real-time simplified `C-IPC` (Li 2021) barrier + semi-implicit
//! `Coulomb` friction, plus `wgpu` compute dispatch; no Unreal Engine source or
//! derived code.

use prism_hair_gpu::barrier_contact::{
    reference_resolve, ContactInput, ContactOutput, GpuHairBarrierContact,
};
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::barrier_contact::{BarrierParams, ContactPoint, Vec3};

fn v(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3::new(x, y, z)
}

fn cp(position: Vec3, velocity: Vec3, inv_mass: f32) -> ContactPoint {
    ContactPoint::new(position, velocity, inv_mass)
}

/// The reference tuning: a `2 cm` activation distance, stiff barrier, `1 mm`
/// floor and a `0.5` friction coefficient.
fn params() -> BarrierParams {
    BarrierParams {
        dhat: 0.02,
        stiffness: 50.0,
        d_floor: 0.001,
        friction_mu: 0.5,
    }
}

/// Acquires a headless context, or `None` (with a skip notice) when the host has
/// no `wgpu` adapter so the suite stays green off-device.
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn context_or_skip(label: &str) -> Option<GpuContext> {
    match GpuContext::try_headless() {
        Some(ctx) => Some(ctx),
        None => {
            eprintln!("skipping {label}: no wgpu adapter on this host");
            None
        }
    }
}

/// Asserts one scalar component matches within the documented fma tolerance.
fn assert_close(got: f32, expected: f32, label: &str) {
    let abs_diff = (got - expected).abs();
    let rel_diff = abs_diff / expected.abs().max(1e-6);
    assert!(
        abs_diff < 1e-4 || rel_diff < 1e-3,
        "{label}: gpu {got}, cpu {expected} (abs {abs_diff}, rel {rel_diff})"
    );
}

/// Asserts a resolved contact matches the golden component-wise (positions,
/// velocities, impulses within tolerance; `slipping` exactly).
fn assert_output(got: &ContactOutput, exp: &ContactOutput, label: &str) {
    assert_close(
        got.a.position.x,
        exp.a.position.x,
        &format!("{label} a.pos.x"),
    );
    assert_close(
        got.a.position.y,
        exp.a.position.y,
        &format!("{label} a.pos.y"),
    );
    assert_close(
        got.a.position.z,
        exp.a.position.z,
        &format!("{label} a.pos.z"),
    );
    assert_close(
        got.a.velocity.x,
        exp.a.velocity.x,
        &format!("{label} a.vel.x"),
    );
    assert_close(
        got.a.velocity.y,
        exp.a.velocity.y,
        &format!("{label} a.vel.y"),
    );
    assert_close(
        got.a.velocity.z,
        exp.a.velocity.z,
        &format!("{label} a.vel.z"),
    );
    assert_close(
        got.b.position.x,
        exp.b.position.x,
        &format!("{label} b.pos.x"),
    );
    assert_close(
        got.b.position.y,
        exp.b.position.y,
        &format!("{label} b.pos.y"),
    );
    assert_close(
        got.b.position.z,
        exp.b.position.z,
        &format!("{label} b.pos.z"),
    );
    assert_close(
        got.b.velocity.x,
        exp.b.velocity.x,
        &format!("{label} b.vel.x"),
    );
    assert_close(
        got.b.velocity.y,
        exp.b.velocity.y,
        &format!("{label} b.vel.y"),
    );
    assert_close(
        got.b.velocity.z,
        exp.b.velocity.z,
        &format!("{label} b.vel.z"),
    );
    assert_close(
        got.normal_impulse,
        exp.normal_impulse,
        &format!("{label} normal_impulse"),
    );
    assert_close(
        got.friction_impulse,
        exp.friction_impulse,
        &format!("{label} friction_impulse"),
    );
    assert_eq!(got.slipping, exp.slipping, "{label} slipping");
}

/// Runs one contact on device and compares it to the golden, returning the
/// device output for extra physical assertions.
fn run_one(ctx: &GpuContext, input: ContactInput, p: BarrierParams, label: &str) -> ContactOutput {
    let kernel = GpuHairBarrierContact::new(ctx);
    let got = kernel.eval(ctx, &[input], p);
    assert_eq!(got.len(), 1, "{label}: one output expected");
    let exp = reference_resolve(&input, p);
    assert_output(&got[0], &exp, label);
    got[0]
}

#[test]
fn pure_normal_repulsion_no_friction() {
    let Some(ctx) = context_or_skip("pure_normal_repulsion_no_friction") else {
        return;
    };
    let p = params();
    let input = ContactInput {
        a: cp(v(0.0, 0.005, 0.0), Vec3::ZERO, 1.0),
        b: cp(v(0.0, 0.0, 0.0), Vec3::ZERO, 1.0),
        normal: v(0.0, 1.0, 0.0),
        distance: 0.005,
        // Relative velocity purely along the normal: no tangential slide.
        rel_velocity: v(0.0, -0.1, 0.0),
    };
    let out = run_one(&ctx, input, p, "pure_normal_repulsion_no_friction");
    assert!(out.normal_impulse > 0.0, "expected repulsion");
    assert!(!out.slipping, "no tangential velocity means no slip");
    assert_close(out.friction_impulse, 0.0, "no friction");
    assert!(out.a.position.y > 0.005, "a pushed along +normal");
    assert!(out.b.position.y < 0.0, "b pushed along -normal");
}

#[test]
fn repulsion_with_sticking_friction() {
    let Some(ctx) = context_or_skip("repulsion_with_sticking_friction") else {
        return;
    };
    let p = params();
    let input = ContactInput {
        a: cp(v(0.0, 0.005, 0.0), Vec3::ZERO, 1.0),
        b: cp(v(0.0, 0.0, 0.0), Vec3::ZERO, 1.0),
        normal: v(0.0, 1.0, 0.0),
        distance: 0.005,
        // Tiny tangential slide: far below the friction cone, so it sticks.
        rel_velocity: v(0.001, 0.0, 0.0),
    };
    let out = run_one(&ctx, input, p, "repulsion_with_sticking_friction");
    assert!(out.normal_impulse > 0.0, "expected repulsion");
    assert!(!out.slipping, "small tangential slide should stick");
    assert!(out.friction_impulse > 0.0, "friction opposes the slide");
    // Sticking impulse exactly cancels the tangential slide: jt = speed/w.
    assert_close(
        out.friction_impulse,
        0.0005,
        "stick impulse = speed * inv_w",
    );
}

#[test]
fn repulsion_with_slipping_friction() {
    let Some(ctx) = context_or_skip("repulsion_with_slipping_friction") else {
        return;
    };
    let p = params();
    let input = ContactInput {
        a: cp(v(0.0, 0.005, 0.0), Vec3::ZERO, 1.0),
        b: cp(v(0.0, 0.0, 0.0), Vec3::ZERO, 1.0),
        normal: v(0.0, 1.0, 0.0),
        distance: 0.005,
        // Huge tangential slide: clamped to the `Coulomb` cone, so it slips.
        rel_velocity: v(1000.0, 0.0, 0.0),
    };
    let out = run_one(&ctx, input, p, "repulsion_with_slipping_friction");
    assert!(out.normal_impulse > 0.0, "expected repulsion");
    assert!(out.slipping, "large tangential slide should slip");
    // On the cone the impulse saturates at mu * jn.
    assert_close(
        out.friction_impulse,
        p.friction_mu * out.normal_impulse,
        "slip impulse = mu * normal_impulse",
    );
}

#[test]
fn one_pinned_endpoint() {
    let Some(ctx) = context_or_skip("one_pinned_endpoint") else {
        return;
    };
    let p = params();
    let b_pos = v(0.0, 0.0, 0.0);
    let input = ContactInput {
        a: cp(v(0.0, 0.005, 0.0), v(5.0, 0.0, 0.0), 1.0),
        b: cp(b_pos, Vec3::ZERO, 0.0), // pinned
        normal: v(0.0, 1.0, 0.0),
        distance: 0.005,
        rel_velocity: v(5.0, 0.0, 0.0),
    };
    let out = run_one(&ctx, input, p, "one_pinned_endpoint");
    assert!(out.normal_impulse > 0.0, "expected repulsion");
    assert_close(out.b.position.x, b_pos.x, "pinned b stays put x");
    assert_close(out.b.position.y, b_pos.y, "pinned b stays put y");
    assert_close(out.b.position.z, b_pos.z, "pinned b stays put z");
    assert_close(out.b.velocity.x, 0.0, "pinned b keeps velocity x");
    assert!(out.a.position.y > 0.005, "free a absorbs the whole push");
}

#[test]
fn both_pinned_is_noop() {
    let Some(ctx) = context_or_skip("both_pinned_is_noop") else {
        return;
    };
    let p = params();
    let input = ContactInput {
        a: cp(v(0.0, 0.005, 0.0), Vec3::ZERO, 0.0),
        b: cp(v(0.0, 0.0, 0.0), Vec3::ZERO, 0.0),
        normal: v(0.0, 1.0, 0.0),
        distance: 0.005,
        rel_velocity: v(1.0, 0.0, 0.0),
    };
    let out = run_one(&ctx, input, p, "both_pinned_is_noop");
    assert_close(out.normal_impulse, 0.0, "no impulse when both pinned");
    assert_close(out.friction_impulse, 0.0, "no friction when both pinned");
    assert!(!out.slipping, "no slip when both pinned");
    assert_close(out.a.position.y, 0.005, "a unchanged");
    assert_close(out.b.position.y, 0.0, "b unchanged");
}

#[test]
fn degenerate_normal_is_noop() {
    let Some(ctx) = context_or_skip("degenerate_normal_is_noop") else {
        return;
    };
    let p = params();
    let input = ContactInput {
        a: cp(v(0.0, 0.005, 0.0), Vec3::ZERO, 1.0),
        b: cp(v(0.0, 0.0, 0.0), Vec3::ZERO, 1.0),
        normal: Vec3::ZERO,
        distance: 0.005,
        rel_velocity: v(1.0, 0.0, 0.0),
    };
    let out = run_one(&ctx, input, p, "degenerate_normal_is_noop");
    assert_close(out.normal_impulse, 0.0, "zero normal yields no impulse");
    assert!(!out.slipping, "zero normal yields no slip");
    assert_close(out.a.position.y, 0.005, "a unchanged");
    assert_close(out.b.position.y, 0.0, "b unchanged");
}

#[test]
fn distance_beyond_dhat_is_noop() {
    let Some(ctx) = context_or_skip("distance_beyond_dhat_is_noop") else {
        return;
    };
    let p = params();
    let input = ContactInput {
        a: cp(v(0.0, 0.05, 0.0), Vec3::ZERO, 1.0),
        b: cp(v(0.0, 0.0, 0.0), Vec3::ZERO, 1.0),
        normal: v(0.0, 1.0, 0.0),
        distance: 0.05, // >= dhat
        rel_velocity: v(1.0, 0.0, 0.0),
    };
    let out = run_one(&ctx, input, p, "distance_beyond_dhat_is_noop");
    assert_close(out.normal_impulse, 0.0, "no barrier beyond dhat");
    assert!(!out.slipping, "no slip beyond dhat");
    assert_close(out.a.position.y, 0.05, "a unchanged");
}

#[test]
fn distance_clamped_to_floor() {
    let Some(ctx) = context_or_skip("distance_clamped_to_floor") else {
        return;
    };
    let p = params();
    let input = ContactInput {
        a: cp(v(0.0, 0.0001, 0.0), Vec3::ZERO, 1.0),
        b: cp(v(0.0, 0.0, 0.0), Vec3::ZERO, 1.0),
        normal: v(0.0, 1.0, 0.0),
        distance: 0.0001, // below d_floor: evaluated on the clamped floor
        rel_velocity: v(0.0, -0.1, 0.0),
    };
    let out = run_one(&ctx, input, p, "distance_clamped_to_floor");
    assert!(out.normal_impulse > 0.0, "floor still repels");
    assert!(out.normal_impulse.is_finite(), "clamped force stays finite");
    // The floor force must dominate a mid-window force (monotone barrier).
    let mid = reference_resolve(
        &ContactInput {
            distance: 0.01,
            ..input
        },
        p,
    );
    assert!(
        out.normal_impulse > mid.normal_impulse,
        "closer contact repels harder"
    );
}

#[test]
fn oblique_normal_sticking() {
    let Some(ctx) = context_or_skip("oblique_normal_sticking") else {
        return;
    };
    let p = params();
    let input = ContactInput {
        a: cp(v(0.003, 0.004, 0.0), Vec3::ZERO, 1.0),
        b: cp(v(0.0, 0.0, 0.0), Vec3::ZERO, 1.0),
        // Unit oblique normal (0.6, 0.8, 0): exercises the tangent projection.
        normal: v(0.6, 0.8, 0.0),
        distance: 0.005,
        // Mostly along the normal with a tiny tangential component: sticks.
        rel_velocity: v(0.0006, -0.0008, 0.0),
    };
    let out = run_one(&ctx, input, p, "oblique_normal_sticking");
    assert!(out.normal_impulse > 0.0, "expected repulsion");
    assert!(!out.slipping, "small tangential slide should stick");
}

#[test]
fn large_batch_crosses_workgroup_boundary() {
    let Some(ctx) = context_or_skip("large_batch_crosses_workgroup_boundary") else {
        return;
    };
    let p = params();
    let kernel = GpuHairBarrierContact::new(&ctx);

    let mut inputs = Vec::new();
    for k in 0u32..130 {
        // Distance sweeps the active window, with every 7th contact set beyond
        // `dhat` (a no-op) and every 11th pinned on both ends (a no-op).
        let dist = 0.002 + ((k % 15) as f32) * 0.001;
        let beyond = (k % 7) == 0;
        let distance = if beyond { 0.05 } else { dist };
        let both_pinned = (k % 11) == 5;
        let a_inv = if both_pinned { 0.0 } else { 1.0 };
        // Zero inverse mass marks a pinned endpoint: either the whole contact
        // is pinned on both ends or just this `b` endpoint (every 3rd contact).
        let b_inv = if both_pinned || (k % 3) == 0 {
            0.0
        } else {
            2.0
        };
        // Tangential slide is either tiny (clearly sticks) or huge (clearly
        // slips) so the branch pick is never near the cone threshold.
        let tangential = if (k % 2) == 0 { 0.0001 } else { 1.0e5 };
        let offset = (k as f32) * 0.01;
        inputs.push(ContactInput {
            a: cp(v(offset, 0.005, 0.0), Vec3::ZERO, a_inv),
            b: cp(v(offset, 0.0, 0.0), Vec3::ZERO, b_inv),
            normal: v(0.0, 1.0, 0.0),
            distance,
            rel_velocity: v(tangential, -0.05, 0.0),
        });
    }

    let got = kernel.eval(&ctx, &inputs, p);
    assert_eq!(got.len(), inputs.len(), "one output per contact");
    let mut active = 0usize;
    for (k, input) in inputs.iter().enumerate() {
        let exp = reference_resolve(input, p);
        assert_output(&got[k], &exp, &format!("batch[{k}]"));
        if exp.normal_impulse > 0.0 {
            active += 1;
        }
    }
    assert!(
        active > 0,
        "batch must contain active contacts, not all no-ops"
    );
}

#[test]
fn empty_batch_is_noop() {
    let Some(ctx) = context_or_skip("empty_batch_is_noop") else {
        return;
    };
    let kernel = GpuHairBarrierContact::new(&ctx);
    let out = kernel.eval(&ctx, &[], params());
    assert!(
        out.is_empty(),
        "empty input yields empty output, no dispatch"
    );
}
