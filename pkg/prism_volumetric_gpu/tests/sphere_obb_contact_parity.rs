//! Real-device parity for the sphere-versus-oriented-box contact twin:
//! [`GpuSphereObbContact`](prism_volumetric_gpu::sphere_obb_contact::GpuSphereObbContact)
//! must reproduce the `CPU` golden `point_vs_box` of
//! `prism_physics_core::collide::primitives`.
//!
//! The oracle here is an independent re-implementation of that closed form,
//! written out with scalar `f32` arithmetic so the test never imports
//! `prism_physics_core`, `prism_render_architecture` or `glam`. The sphere
//! centre is projected into the box frame through the three box axes, clamped to
//! the half extents and classified inside or outside. The outside branch builds
//! the closest point, its distance to the centre, the penetration
//! `radius - distance` and rejects when the penetration drops below
//! `-CONTACT_TOLERANCE`; the inside branch escapes along the nearest face by the
//! smallest remaining half-extent slack. Every operator is evaluated in the same
//! order the kernel uses so no extra `f32` error is introduced.
//!
//! The fixtures cover a separated pair (rejected, all-zero `valid = 0`), a
//! shallow face contact, a deep face penetration, an inside-escape along each of
//! the three axes, the box-centre degenerate case (which also escapes inside),
//! the outside fallback normal when the closest point coincides with the centre,
//! a rotated `OBB`, a mixed `>=2`-element batch that validates the `std430`
//! stride end to end, and a `512`-query `LCG` sweep. An empty batch is
//! short-circuited by the host with no dispatch.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! # Parity criterion
//!
//! The normal, witness points and penetration are continuous and checked with an
//! absolute-or-relative tolerance (`abs <= 1e-4 || rel <= 1e-3`,
//! `REL_FLOOR = 1e-6`); `valid` is discrete and compared exactly. The
//! conditioning knees are the closest-point distance crossing `GEOMETRIC_EPS`
//! (fallback normal), the penetration crossing `-CONTACT_TOLERANCE` (accept or
//! reject), the inside/outside boundary (`|local| == he`), the inside-escape
//! face tie and the escape-sign knee at `local == 0`. The fixtures and sweep
//! stay clear of those knees via rejection sampling so host and device make the
//! same discrete choice.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collide::primitives`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::sphere_obb_contact::{
    GpuSphereObbContact, SphereObbContactQuery, SphereObbContactResult,
};
use prism_volumetric_gpu::GpuContext;

/// Golden constants, mirrored from `prism_physics_core::collide::primitives`.
const CONTACT_TOLERANCE: f32 = 1.0e-4;
const GEOMETRIC_EPS: f32 = 1.0e-6;

/// Dot product of two flat 3-vectors.
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Independent host re-implementation of `point_vs_box`, flattened into the
/// `(normal, point_a, point_b, penetration, valid)` record the twin encodes.
/// Each operator mirrors the kernel in evaluation order; a rejected pair reports
/// all-zero payload with `valid = 0`. No `prism_physics_core` / `glam` import.
fn oracle(q: &SphereObbContactQuery) -> SphereObbContactResult {
    let sc = [q.scx, q.scy, q.scz];
    let bc = [q.bcx, q.bcy, q.bcz];
    let a0 = [q.a0x, q.a0y, q.a0z];
    let a1 = [q.a1x, q.a1y, q.a1z];
    let a2 = [q.a2x, q.a2y, q.a2z];
    let he = [q.hex, q.hey, q.hez];
    let radius = q.radius;

    let delta = [sc[0] - bc[0], sc[1] - bc[1], sc[2] - bc[2]];
    let local = [dot3(delta, a0), dot3(delta, a1), dot3(delta, a2)];
    let clamped = [
        local[0].max(-he[0]).min(he[0]),
        local[1].max(-he[1]).min(he[1]),
        local[2].max(-he[2]).min(he[2]),
    ];
    let inside = local[0].abs() <= he[0] && local[1].abs() <= he[1] && local[2].abs() <= he[2];

    let zero = SphereObbContactResult {
        nx: 0.0,
        ny: 0.0,
        nz: 0.0,
        pax: 0.0,
        pay: 0.0,
        paz: 0.0,
        pbx: 0.0,
        pby: 0.0,
        pbz: 0.0,
        pen: 0.0,
        valid: 0,
    };

    if !inside {
        // Outside branch: build the contact from the closest point on the box.
        let closest = [
            bc[0] + a0[0] * clamped[0] + a1[0] * clamped[1] + a2[0] * clamped[2],
            bc[1] + a0[1] * clamped[0] + a1[1] * clamped[1] + a2[1] * clamped[2],
            bc[2] + a0[2] * clamped[0] + a1[2] * clamped[1] + a2[2] * clamped[2],
        ];
        let diff = [closest[0] - sc[0], closest[1] - sc[1], closest[2] - sc[2]];
        let dist = dot3(diff, diff).sqrt();
        let pen = radius - dist;
        if pen < -CONTACT_TOLERANCE {
            return zero;
        }
        let normal = if dist > GEOMETRIC_EPS {
            [diff[0] / dist, diff[1] / dist, diff[2] / dist]
        } else {
            [1.0, 0.0, 0.0]
        };
        let pa = [
            sc[0] + normal[0] * radius,
            sc[1] + normal[1] * radius,
            sc[2] + normal[2] * radius,
        ];
        let pb = closest;
        return SphereObbContactResult {
            nx: normal[0],
            ny: normal[1],
            nz: normal[2],
            pax: pa[0],
            pay: pa[1],
            paz: pa[2],
            pbx: pb[0],
            pby: pb[1],
            pbz: pb[2],
            pen,
            valid: 1,
        };
    }

    // Inside branch: escape along the nearest face.
    let fp = [
        he[0] - local[0].abs(),
        he[1] - local[1].abs(),
        he[2] - local[2].abs(),
    ];
    let mut axis_index = 0usize;
    let mut min_fp = fp[0];
    if fp[1] < min_fp {
        min_fp = fp[1];
        axis_index = 1;
    }
    if fp[2] < min_fp {
        min_fp = fp[2];
        axis_index = 2;
    }
    let lc = local[axis_index];
    let axis_sel = match axis_index {
        0 => a0,
        1 => a1,
        _ => a2,
    };
    let sign_val = if lc >= 0.0 { 1.0 } else { -1.0 };
    let outward = [
        axis_sel[0] * sign_val,
        axis_sel[1] * sign_val,
        axis_sel[2] * sign_val,
    ];
    let normal = [-outward[0], -outward[1], -outward[2]];
    let pen = radius + min_fp;
    let pb = [
        sc[0] + outward[0] * min_fp,
        sc[1] + outward[1] * min_fp,
        sc[2] + outward[2] * min_fp,
    ];
    let pa = [
        sc[0] - outward[0] * radius,
        sc[1] - outward[1] * radius,
        sc[2] - outward[2] * radius,
    ];
    SphereObbContactResult {
        nx: normal[0],
        ny: normal[1],
        nz: normal[2],
        pax: pa[0],
        pay: pa[1],
        paz: pa[2],
        pbx: pb[0],
        pby: pb[1],
        pbz: pb[2],
        pen,
        valid: 1,
    }
}

/// Absolute-or-relative closeness for a continuous channel.
fn close(got: f32, want: f32) -> bool {
    let diff = (got - want).abs();
    if diff <= 1e-4 {
        return true;
    }
    let rel_floor = 1e-6_f32;
    let denom = want.abs().max(got.abs()).max(rel_floor);
    diff / denom <= 1e-3
}

/// Asserts one GPU result matches the oracle. The ten payload components are
/// continuous (tolerance); `valid` is discrete (exact).
fn assert_result(got: SphereObbContactResult, want: SphereObbContactResult, label: &str) {
    assert_eq!(got.valid, want.valid, "valid mismatch: {label}");
    let fields: [(f32, f32, &str); 10] = [
        (got.nx, want.nx, "nx"),
        (got.ny, want.ny, "ny"),
        (got.nz, want.nz, "nz"),
        (got.pax, want.pax, "pax"),
        (got.pay, want.pay, "pay"),
        (got.paz, want.paz, "paz"),
        (got.pbx, want.pbx, "pbx"),
        (got.pby, want.pby, "pby"),
        (got.pbz, want.pbz, "pbz"),
        (got.pen, want.pen, "pen"),
    ];
    for (g, w, name) in fields {
        assert!(close(g, w), "{name} mismatch: {label}: got {g} want {w}");
    }
}

/// Asserts a single-query GPU result matches the oracle.
fn assert_parity(
    ctx: &GpuContext,
    gpu: &GpuSphereObbContact,
    q: SphereObbContactQuery,
    label: &str,
) {
    let got = gpu.evaluate(ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1, "one result per query: {label}");
    assert_result(got[0], oracle(&q), label);
}

/// The canonical axis-aligned basis.
const IDENTITY: ([f32; 3], [f32; 3], [f32; 3]) =
    ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);

/// Builds an orthonormal basis from intrinsic yaw/pitch/roll (host-side
/// transcendental calls are fine; the kernel never computes them). The three
/// returned vectors are the columns the query carries as `axis0/1/2`.
fn basis_from_euler(yaw: f32, pitch: f32, roll: f32) -> ([f32; 3], [f32; 3], [f32; 3]) {
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    let (sr, cr) = roll.sin_cos();
    // Rz(yaw) * Ry(pitch) * Rx(roll), columns extracted.
    let a0 = [cy * cp, sy * cp, -sp];
    let a1 = [cy * sp * sr - sy * cr, sy * sp * sr + cy * cr, cp * sr];
    let a2 = [cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr];
    (a0, a1, a2)
}

/// Places a sphere so its centre sits at box-frame coordinate `local`.
fn center_from_local(
    bc: [f32; 3],
    axes: &([f32; 3], [f32; 3], [f32; 3]),
    local: [f32; 3],
) -> [f32; 3] {
    let (a0, a1, a2) = axes;
    [
        bc[0] + a0[0] * local[0] + a1[0] * local[1] + a2[0] * local[2],
        bc[1] + a0[1] * local[0] + a1[1] * local[1] + a2[1] * local[2],
        bc[2] + a0[2] * local[0] + a1[2] * local[1] + a2[2] * local[2],
    ]
}

#[test]
fn separated_pair_is_rejected() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphereObbContact::new(&ctx);
    // Sphere far outside the box: penetration << -CONTACT_TOLERANCE -> valid 0.
    let (a0, a1, a2) = IDENTITY;
    let q = SphereObbContactQuery::new(
        [5.0, 0.0, 0.0],
        0.5,
        [0.0, 0.0, 0.0],
        a0,
        a1,
        a2,
        [1.0, 1.0, 1.0],
    );
    let want = oracle(&q);
    assert_eq!(want.valid, 0, "fixture sanity: separated pair rejected");
    assert_parity(&ctx, &gpu, q, "separated_pair_is_rejected");
}

#[test]
fn shallow_face_contact() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphereObbContact::new(&ctx);
    // Centre just outside the +x face, closest point on the face, small positive
    // penetration -> valid, normal pointing back toward the box (-x).
    let (a0, a1, a2) = IDENTITY;
    let q = SphereObbContactQuery::new(
        [1.4, 0.0, 0.0],
        0.5,
        [0.0, 0.0, 0.0],
        a0,
        a1,
        a2,
        [1.0, 1.0, 1.0],
    );
    let want = oracle(&q);
    assert_eq!(want.valid, 1, "fixture sanity: shallow contact valid");
    assert!(close(want.pen, 0.1), "fixture sanity: pen ~ 0.1");
    assert!(close(want.nx, -1.0), "fixture sanity: normal toward box");
    assert_parity(&ctx, &gpu, q, "shallow_face_contact");
}

#[test]
fn deep_face_penetration() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphereObbContact::new(&ctx);
    // Centre barely outside the +x face but large radius -> deep penetration.
    let (a0, a1, a2) = IDENTITY;
    let q = SphereObbContactQuery::new(
        [1.1, 0.0, 0.0],
        0.5,
        [0.0, 0.0, 0.0],
        a0,
        a1,
        a2,
        [1.0, 1.0, 1.0],
    );
    let want = oracle(&q);
    assert_eq!(want.valid, 1, "fixture sanity: deep contact valid");
    assert!(close(want.pen, 0.4), "fixture sanity: pen ~ 0.4");
    assert_parity(&ctx, &gpu, q, "deep_face_penetration");
}

#[test]
fn inside_escape_plus_x() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphereObbContact::new(&ctx);
    // Centre inside, nearest face is +x (smallest slack), escapes outward +x.
    let (a0, a1, a2) = IDENTITY;
    let q = SphereObbContactQuery::new(
        [0.5, 0.0, 0.0],
        0.3,
        [0.0, 0.0, 0.0],
        a0,
        a1,
        a2,
        [1.0, 1.0, 1.0],
    );
    let want = oracle(&q);
    assert_eq!(want.valid, 1, "fixture sanity: inside escape valid");
    assert!(
        close(want.nx, -1.0),
        "fixture sanity: normal -x (toward box)"
    );
    assert_parity(&ctx, &gpu, q, "inside_escape_plus_x");
}

#[test]
fn inside_escape_minus_x() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphereObbContact::new(&ctx);
    let (a0, a1, a2) = IDENTITY;
    let q = SphereObbContactQuery::new(
        [-0.5, 0.0, 0.0],
        0.3,
        [0.0, 0.0, 0.0],
        a0,
        a1,
        a2,
        [1.0, 1.0, 1.0],
    );
    let want = oracle(&q);
    assert_eq!(want.valid, 1, "fixture sanity: inside escape valid");
    assert!(
        close(want.nx, 1.0),
        "fixture sanity: normal +x (toward box)"
    );
    assert_parity(&ctx, &gpu, q, "inside_escape_minus_x");
}

#[test]
fn inside_escape_plus_y() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphereObbContact::new(&ctx);
    // Smallest slack on the y axis.
    let (a0, a1, a2) = IDENTITY;
    let q = SphereObbContactQuery::new(
        [0.0, 0.4, 0.0],
        0.3,
        [0.0, 0.0, 0.0],
        a0,
        a1,
        a2,
        [1.0, 1.0, 1.0],
    );
    let want = oracle(&q);
    assert_eq!(want.valid, 1, "fixture sanity: inside escape valid");
    assert!(
        close(want.ny, -1.0),
        "fixture sanity: normal -y (toward box)"
    );
    assert_parity(&ctx, &gpu, q, "inside_escape_plus_y");
}

#[test]
fn inside_escape_plus_z() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphereObbContact::new(&ctx);
    // Smallest slack on the z axis.
    let (a0, a1, a2) = IDENTITY;
    let q = SphereObbContactQuery::new(
        [0.0, 0.0, 0.3],
        0.3,
        [0.0, 0.0, 0.0],
        a0,
        a1,
        a2,
        [1.0, 1.0, 1.0],
    );
    let want = oracle(&q);
    assert_eq!(want.valid, 1, "fixture sanity: inside escape valid");
    assert!(
        close(want.nz, -1.0),
        "fixture sanity: normal -z (toward box)"
    );
    assert_parity(&ctx, &gpu, q, "inside_escape_plus_z");
}

#[test]
fn centre_at_box_centre_escapes_inside() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphereObbContact::new(&ctx);
    // Sphere centre == box centre: local = 0, inside is true, nearest face is the
    // smallest half extent (x here), lc = 0 -> sign +1. Not the outside fallback.
    let (a0, a1, a2) = IDENTITY;
    let q = SphereObbContactQuery::new(
        [0.0, 0.0, 0.0],
        0.3,
        [0.0, 0.0, 0.0],
        a0,
        a1,
        a2,
        [1.0, 2.0, 3.0],
    );
    let want = oracle(&q);
    assert_eq!(want.valid, 1, "fixture sanity: degenerate escapes inside");
    assert!(
        close(want.nx, -1.0),
        "fixture sanity: escape along smallest he (x)"
    );
    assert_parity(&ctx, &gpu, q, "centre_at_box_centre_escapes_inside");
}

#[test]
fn outside_fallback_normal() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphereObbContact::new(&ctx);
    // Centre a hair outside the +x face: closest point ~ centre so dist < EPS and
    // the outside branch takes the (1, 0, 0) fallback normal.
    let (a0, a1, a2) = IDENTITY;
    let q = SphereObbContactQuery::new(
        [1.0 + 3.0e-7, 0.0, 0.0],
        0.5,
        [0.0, 0.0, 0.0],
        a0,
        a1,
        a2,
        [1.0, 1.0, 1.0],
    );
    let want = oracle(&q);
    assert_eq!(want.valid, 1, "fixture sanity: fallback contact valid");
    assert!(
        close(want.nx, 1.0) && close(want.ny, 0.0) && close(want.nz, 0.0),
        "fixture sanity: fallback normal is (1,0,0)"
    );
    assert_parity(&ctx, &gpu, q, "outside_fallback_normal");
}

#[test]
fn rotated_obb_contact() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphereObbContact::new(&ctx);
    // A rotated box, sphere placed outside along axis0 at local (1.3, 0, 0).
    let axes = basis_from_euler(0.5, -0.3, 0.8);
    let bc = [0.2, -0.4, 0.1];
    let sc = center_from_local(bc, &axes, [1.3, 0.0, 0.0]);
    let (a0, a1, a2) = axes;
    let q = SphereObbContactQuery::new(sc, 0.5, bc, a0, a1, a2, [1.0, 1.0, 1.0]);
    let want = oracle(&q);
    assert_eq!(want.valid, 1, "fixture sanity: rotated contact valid");
    assert!(close(want.pen, 0.2), "fixture sanity: pen ~ 0.2");
    assert_parity(&ctx, &gpu, q, "rotated_obb_contact");
}

#[test]
fn mixed_batch_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphereObbContact::new(&ctx);
    // A >=2-element batch mixing a rejected pair, an outside contact, an inside
    // escape and a rotated contact validates the std430 stride end to end.
    let (i0, i1, i2) = IDENTITY;
    let axes = basis_from_euler(-0.7, 0.4, 0.2);
    let bc = [-0.3, 0.5, 0.2];
    let sc = center_from_local(bc, &axes, [0.0, 1.25, 0.0]);
    let (r0, r1, r2) = axes;
    let queries = vec![
        SphereObbContactQuery::new(
            [8.0, 0.0, 0.0],
            0.5,
            [0.0, 0.0, 0.0],
            i0,
            i1,
            i2,
            [1.0, 1.0, 1.0],
        ),
        SphereObbContactQuery::new(
            [1.3, 0.0, 0.0],
            0.6,
            [0.0, 0.0, 0.0],
            i0,
            i1,
            i2,
            [1.0, 1.0, 1.0],
        ),
        SphereObbContactQuery::new(
            [0.0, -0.4, 0.0],
            0.3,
            [0.0, 0.0, 0.0],
            i0,
            i1,
            i2,
            [1.0, 1.0, 1.0],
        ),
        SphereObbContactQuery::new(sc, 0.5, bc, r0, r1, r2, [1.0, 1.0, 1.0]),
    ];
    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (i, (q, r)) in queries.iter().zip(results.iter()).enumerate() {
        assert_result(*r, oracle(q), &format!("mixed batch index {i}"));
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphereObbContact::new(&ctx);
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

    /// A `[0, 1)` fraction built from the top bits.
    fn next_unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    /// A `[lo, hi)` fraction.
    fn next_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_unit()
    }
}

/// Rejects a candidate whose geometry sits on a conditioning knee where host and
/// device could disagree on a discrete choice. Returns `true` to keep it.
fn well_conditioned(q: &SphereObbContactQuery) -> bool {
    let sc = [q.scx, q.scy, q.scz];
    let bc = [q.bcx, q.bcy, q.bcz];
    let a0 = [q.a0x, q.a0y, q.a0z];
    let a1 = [q.a1x, q.a1y, q.a1z];
    let a2 = [q.a2x, q.a2y, q.a2z];
    let he = [q.hex, q.hey, q.hez];
    let delta = [sc[0] - bc[0], sc[1] - bc[1], sc[2] - bc[2]];
    let local = [dot3(delta, a0), dot3(delta, a1), dot3(delta, a2)];

    // Keep clear of the inside/outside boundary on every axis.
    for axis in 0..3 {
        if (local[axis].abs() - he[axis]).abs() < 0.05 {
            return false;
        }
    }
    let inside = local[0].abs() <= he[0] && local[1].abs() <= he[1] && local[2].abs() <= he[2];

    if !inside {
        let clamped = [
            local[0].max(-he[0]).min(he[0]),
            local[1].max(-he[1]).min(he[1]),
            local[2].max(-he[2]).min(he[2]),
        ];
        let closest = [
            bc[0] + a0[0] * clamped[0] + a1[0] * clamped[1] + a2[0] * clamped[2],
            bc[1] + a0[1] * clamped[0] + a1[1] * clamped[1] + a2[1] * clamped[2],
            bc[2] + a0[2] * clamped[0] + a1[2] * clamped[1] + a2[2] * clamped[2],
        ];
        let diff = [closest[0] - sc[0], closest[1] - sc[1], closest[2] - sc[2]];
        let dist = dot3(diff, diff).sqrt();
        // Away from the fallback knee and the accept/reject penetration knee.
        if dist < 0.05 {
            return false;
        }
        let pen = q.radius - dist;
        if (pen + CONTACT_TOLERANCE).abs() < 0.05 {
            return false;
        }
        true
    } else {
        let fp = [
            he[0] - local[0].abs(),
            he[1] - local[1].abs(),
            he[2] - local[2].abs(),
        ];
        // No near-tie between any two face slacks (face-selection knee).
        if (fp[0] - fp[1]).abs() < 0.05
            || (fp[0] - fp[2]).abs() < 0.05
            || (fp[1] - fp[2]).abs() < 0.05
        {
            return false;
        }
        // Pick the minimum axis the same way the kernel does, then stay clear of
        // the escape-sign knee at local == 0.
        let mut axis_index = 0usize;
        let mut min_fp = fp[0];
        if fp[1] < min_fp {
            min_fp = fp[1];
            axis_index = 1;
        }
        if fp[2] < min_fp {
            axis_index = 2;
        }
        if local[axis_index].abs() < 0.05 {
            return false;
        }
        true
    }
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphereObbContact::new(&ctx);
    let mut rng = Lcg::new(0x51_7C_0B_9D);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        // Random orthonormal basis so the sweep exercises rotated OBBs.
        let axes = basis_from_euler(
            rng.next_range(-3.1, 3.1),
            rng.next_range(-1.5, 1.5),
            rng.next_range(-3.1, 3.1),
        );
        let bc = [
            rng.next_range(-1.0, 1.0),
            rng.next_range(-1.0, 1.0),
            rng.next_range(-1.0, 1.0),
        ];
        let he = [
            rng.next_range(0.5, 2.0),
            rng.next_range(0.5, 2.0),
            rng.next_range(0.5, 2.0),
        ];
        let radius = rng.next_range(0.3, 2.0);
        // Build the centre from a target box-frame coordinate so inside/outside
        // and the knees are easy to control.
        let target_local = [
            rng.next_range(-4.0, 4.0),
            rng.next_range(-4.0, 4.0),
            rng.next_range(-4.0, 4.0),
        ];
        let sc = center_from_local(bc, &axes, target_local);
        let (a0, a1, a2) = axes;
        let q = SphereObbContactQuery::new(sc, radius, bc, a0, a1, a2, he);
        if well_conditioned(&q) {
            queries.push(q);
        }
    }

    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (i, (q, r)) in queries.iter().zip(results.iter()).enumerate() {
        assert_result(*r, oracle(q), &format!("sweep index {i}"));
    }
}
