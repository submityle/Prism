//! Real-device parity for the capsule-vs-plane contact twin:
//! [`GpuCapsulePlaneContact`](prism_volumetric_gpu::capsule_plane_contact::GpuCapsulePlaneContact)
//! must reproduce the world-space body of the `CPU` golden
//! `prism_physics_core::collide::primitives::capsule_plane`, which seats a
//! contact at each of a capsule segment's two endpoints against a plane
//! half-space.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! the shared manifold normal `normal = -normal_world`, and for each endpoint
//! `e` the signed distance `signed = dot(normal_world, e) - offset_world`, the
//! penetration `pen = radius - signed`, the admissibility test
//! `pen >= -CONTACT_TOLERANCE`, and the witness pair
//! `point_a = e - normal_world * radius`, `point_b = e - normal_world * signed`
//! — written out directly so the test never imports `prism_physics_core` or
//! `prism_render_architecture`, nor `glam`. The closed form has no division.
//!
//! The fixtures cover the branches the kernel must honor: both endpoints
//! penetrating; only one endpoint penetrating; both separated (the golden's
//! `None`, i.e. both `valid` words zero); a grazing contact with the penetration
//! held clear of the knee; a deep penetration; a multi-element batch that
//! validates the `std430` stride; and a `512`-step sweep over random unit plane
//! normals, segments and radii, kept away from the admissibility knee by
//! rejection sampling so no `valid` word flips under `f32` noise. An empty batch
//! is short-circuited on the host with no dispatch.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The manifold normal, witness points and penetrations are continuous `f32`
//! outputs, so parity uses an absolute-or-relative tolerance
//! (`abs <= 1e-4 || rel <= 1e-3`, with a `1e-6` relative floor so near-zero
//! components compare on the absolute leg). The `valid` words are discrete and
//! compared exactly; a non-contacting endpoint reports `valid = 0` with zeroed
//! witness points and penetration.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collide::primitives`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::capsule_plane_contact::{
    CapsulePlaneContactQuery, GpuCapsulePlaneContact,
};
use prism_volumetric_gpu::GpuContext;

/// Contact admissibility tolerance matching the golden `CONTACT_TOLERANCE`.
const CONTACT_TOLERANCE: f32 = 1.0e-4;
/// Absolute tolerance leg for the continuous comparisons.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance leg for the continuous comparisons.
const REL_EPS: f32 = 1.0e-3;
/// Relative-tolerance floor so near-zero components fall back to the absolute
/// leg instead of demanding an impossible relative match.
const REL_FLOOR: f32 = 1.0e-6;

/// Absolute-or-relative closeness for a single `f32` lane.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff <= REL_EPS * scale
}

/// Independent oracle for one query, returning
/// `(normal, [pa0, pa1], [pb0, pb1], [pen0, pen1], [valid0, valid1])`.
///
/// Reproduces the world-space body of `capsule_plane` operator by operator, in
/// the same order as the kernel: a non-contacting endpoint
/// (`pen < -CONTACT_TOLERANCE`) reports `valid = 0` and zeroes its witness
/// points and penetration.
fn oracle(
    q: &CapsulePlaneContactQuery,
) -> ([f32; 3], [[f32; 3]; 2], [[f32; 3]; 2], [f32; 2], [u32; 2]) {
    let normal = [-q.nx, -q.ny, -q.nz];
    let endpoints = [[q.seg0x, q.seg0y, q.seg0z], [q.seg1x, q.seg1y, q.seg1z]];
    let mut pa = [[0.0f32; 3]; 2];
    let mut pb = [[0.0f32; 3]; 2];
    let mut pen = [0.0f32; 2];
    let mut valid = [0u32; 2];
    for (i, e) in endpoints.iter().enumerate() {
        let signed = q.nx * e[0] + q.ny * e[1] + q.nz * e[2] - q.plane_offset;
        let penetration = q.radius - signed;
        if penetration >= -CONTACT_TOLERANCE {
            valid[i] = 1;
            pa[i] = [
                e[0] - q.nx * q.radius,
                e[1] - q.ny * q.radius,
                e[2] - q.nz * q.radius,
            ];
            pb[i] = [
                e[0] - q.nx * signed,
                e[1] - q.ny * signed,
                e[2] - q.nz * signed,
            ];
            pen[i] = penetration;
        }
    }
    (normal, pa, pb, pen, valid)
}

/// Asserts one device result matches the oracle, lane by lane.
fn check(
    q: &CapsulePlaneContactQuery,
    r: &prism_volumetric_gpu::capsule_plane_contact::CapsulePlaneContactResult,
) {
    let (normal, pa, pb, pen, valid) = oracle(q);
    assert!(
        close(r.normal_x, normal[0])
            && close(r.normal_y, normal[1])
            && close(r.normal_z, normal[2]),
        "normal mismatch: query={q:?} gpu=({}, {}, {}) oracle=({}, {}, {})",
        r.normal_x,
        r.normal_y,
        r.normal_z,
        normal[0],
        normal[1],
        normal[2]
    );
    assert_eq!(r.valid0, valid[0], "valid0 mismatch: query={q:?}");
    assert_eq!(r.valid1, valid[1], "valid1 mismatch: query={q:?}");
    assert!(
        close(r.pa0x, pa[0][0]) && close(r.pa0y, pa[0][1]) && close(r.pa0z, pa[0][2]),
        "pa0 mismatch: query={q:?} gpu=({}, {}, {}) oracle=({}, {}, {})",
        r.pa0x,
        r.pa0y,
        r.pa0z,
        pa[0][0],
        pa[0][1],
        pa[0][2]
    );
    assert!(
        close(r.pb0x, pb[0][0]) && close(r.pb0y, pb[0][1]) && close(r.pb0z, pb[0][2]),
        "pb0 mismatch: query={q:?} gpu=({}, {}, {}) oracle=({}, {}, {})",
        r.pb0x,
        r.pb0y,
        r.pb0z,
        pb[0][0],
        pb[0][1],
        pb[0][2]
    );
    assert!(
        close(r.pen0, pen[0]),
        "pen0 mismatch: query={q:?} gpu={} oracle={}",
        r.pen0,
        pen[0]
    );
    assert!(
        close(r.pa1x, pa[1][0]) && close(r.pa1y, pa[1][1]) && close(r.pa1z, pa[1][2]),
        "pa1 mismatch: query={q:?} gpu=({}, {}, {}) oracle=({}, {}, {})",
        r.pa1x,
        r.pa1y,
        r.pa1z,
        pa[1][0],
        pa[1][1],
        pa[1][2]
    );
    assert!(
        close(r.pb1x, pb[1][0]) && close(r.pb1y, pb[1][1]) && close(r.pb1z, pb[1][2]),
        "pb1 mismatch: query={q:?} gpu=({}, {}, {}) oracle=({}, {}, {})",
        r.pb1x,
        r.pb1y,
        r.pb1z,
        pb[1][0],
        pb[1][1],
        pb[1][2]
    );
    assert!(
        close(r.pen1, pen[1]),
        "pen1 mismatch: query={q:?} gpu={} oracle={}",
        r.pen1,
        pen[1]
    );
}

/// Dispatches one query and asserts the device result matches the oracle.
fn assert_parity(ctx: &GpuContext, gpu: &GpuCapsulePlaneContact, q: CapsulePlaneContactQuery) {
    let results = gpu.evaluate(ctx, &[q]);
    assert_eq!(results.len(), 1, "one result per query");
    check(&q, &results[0]);
}

#[test]
fn both_endpoints_penetrate() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapsulePlaneContact::new(&ctx);
    // A capsule lying parallel just below the ground plane (normal +Y, offset 0,
    // radius 0.5). Both endpoints sit at y = 0.2 so signed = 0.2 and
    // pen = 0.3 > 0 for each: both valid = 1.
    assert_parity(
        &ctx,
        &gpu,
        CapsulePlaneContactQuery::new(
            -1.0, 0.2, 0.0, // seg0
            1.0, 0.2, 0.0, // seg1
            0.5, // radius
            0.0, 1.0, 0.0, // plane normal +Y
            0.0, // plane offset
        ),
    );
}

#[test]
fn only_one_endpoint_penetrates() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapsulePlaneContact::new(&ctx);
    // A tilted capsule: seg0 at y = 0.1 (signed 0.1, pen 0.4 > 0 -> valid),
    // seg1 at y = 3.0 (signed 3.0, pen -2.5 < -tol -> not valid). radius 0.5.
    assert_parity(
        &ctx,
        &gpu,
        CapsulePlaneContactQuery::new(
            -1.0, 0.1, 0.0, // seg0 (contact)
            1.0, 3.0, 0.0, // seg1 (separated)
            0.5, 0.0, 1.0, 0.0, 0.0,
        ),
    );
}

#[test]
fn both_separated_is_none() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapsulePlaneContact::new(&ctx);
    // Both endpoints high above the plane: signed = 5.0 and 6.0, pen clearly
    // below -tol, so both valid = 0 (golden returns None).
    assert_parity(
        &ctx,
        &gpu,
        CapsulePlaneContactQuery::new(
            -1.0, 5.0, 0.0, // seg0
            1.0, 6.0, 0.0, // seg1
            0.5, 0.0, 1.0, 0.0, 0.0,
        ),
    );
}

#[test]
fn grazing_contact_clears_knee() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapsulePlaneContact::new(&ctx);
    // seg0 touches so signed = radius exactly -> pen = 0 (>= -tol, valid). seg1
    // sits one millimetre too far: signed = radius + 1e-3 -> pen = -1e-3 which
    // is below -tol, held clear of the knee so the branch is decisive.
    let radius = 0.5f32;
    assert_parity(
        &ctx,
        &gpu,
        CapsulePlaneContactQuery::new(
            -1.0,
            radius,
            0.0, // seg0: pen = 0
            1.0,
            radius + 1.0e-3,
            0.0, // seg1: pen = -1e-3
            radius,
            0.0,
            1.0,
            0.0,
            0.0,
        ),
    );
}

#[test]
fn deep_penetration() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapsulePlaneContact::new(&ctx);
    // A capsule well inside the half-space: signed negative, pen large positive,
    // both endpoints valid, on a non-axis-aligned unit normal.
    let n = {
        let (x, y, z) = (1.0f32, 2.0f32, -0.5f32);
        let inv = 1.0 / (x * x + y * y + z * z).sqrt();
        [x * inv, y * inv, z * inv]
    };
    assert_parity(
        &ctx,
        &gpu,
        CapsulePlaneContactQuery::new(
            -0.5, -1.0, 0.3, // seg0
            0.5, -1.5, -0.3, // seg1
            0.75, n[0], n[1], n[2], 0.5,
        ),
    );
}

#[test]
fn mixed_batch_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapsulePlaneContact::new(&ctx);
    // A multi-element batch exercises the std430 stride: adjacent 19-lane result
    // slots must decode independently and in order, spanning both-contact,
    // one-contact and both-separated cases.
    let both =
        CapsulePlaneContactQuery::new(-1.0, 0.2, 0.0, 1.0, 0.2, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0);
    let one = CapsulePlaneContactQuery::new(-1.0, 0.1, 0.0, 1.0, 3.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0);
    let none =
        CapsulePlaneContactQuery::new(-1.0, 5.0, 0.0, 1.0, 6.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0);
    let queries = [both, one, none];
    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        check(q, r);
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapsulePlaneContact::new(&ctx);
    assert!(
        gpu.evaluate(&ctx, &[]).is_empty(),
        "empty batch returns an empty vector with no dispatch"
    );
}

/// A small deterministic linear-congruential generator so the sweep needs no
/// external randomness. Constants are the Numerical Recipes values.
struct Lcg {
    state: u32,
}

impl Lcg {
    fn new(seed: u32) -> Self {
        Lcg { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        self.state
    }

    /// A `[0, 1)` fraction built from the top bits, keeping the fixture pure
    /// integer host-side with no transcendental call.
    fn next_unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    /// A `[lo, hi)` fraction.
    fn next_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_unit()
    }
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapsulePlaneContact::new(&ctx);
    let mut rng = Lcg::new(0x51_B2_7E_43);
    // Each query uses a random unit plane normal, random segment endpoints and a
    // positive radius. Rejection sampling keeps each endpoint's penetration at
    // least 1e-2 away from the admissibility knee (-CONTACT_TOLERANCE) so no
    // valid word flips under f32 noise.
    const KNEE_MARGIN: f32 = 1.0e-2;
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        let nx = rng.next_range(-1.0, 1.0);
        let ny = rng.next_range(-1.0, 1.0);
        let nz = rng.next_range(-1.0, 1.0);
        let nlen_sq = nx * nx + ny * ny + nz * nz;
        // Reject near-zero vectors so the normalized plane normal is stable.
        if nlen_sq < 0.1 {
            continue;
        }
        let inv = 1.0 / nlen_sq.sqrt();
        let n = [nx * inv, ny * inv, nz * inv];
        let seg0 = [
            rng.next_range(-2.0, 2.0),
            rng.next_range(-2.0, 2.0),
            rng.next_range(-2.0, 2.0),
        ];
        let seg1 = [
            rng.next_range(-2.0, 2.0),
            rng.next_range(-2.0, 2.0),
            rng.next_range(-2.0, 2.0),
        ];
        let radius = rng.next_range(0.1, 1.5);
        let offset = rng.next_range(-1.0, 1.0);
        // Compute both penetrations and reject if either sits in the knee band.
        let signed0 = n[0] * seg0[0] + n[1] * seg0[1] + n[2] * seg0[2] - offset;
        let signed1 = n[0] * seg1[0] + n[1] * seg1[1] + n[2] * seg1[2] - offset;
        let pen0 = radius - signed0;
        let pen1 = radius - signed1;
        if (pen0 - (-CONTACT_TOLERANCE)).abs() < KNEE_MARGIN
            || (pen1 - (-CONTACT_TOLERANCE)).abs() < KNEE_MARGIN
        {
            continue;
        }
        queries.push(CapsulePlaneContactQuery::new(
            seg0[0], seg0[1], seg0[2], seg1[0], seg1[1], seg1[2], radius, n[0], n[1], n[2], offset,
        ));
    }

    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        check(q, r);
    }
}
