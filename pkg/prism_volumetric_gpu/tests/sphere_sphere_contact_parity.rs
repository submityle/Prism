//! Real-device parity for the sphere-sphere contact twin:
//! [`GpuSphereSphereContact`](prism_volumetric_gpu::sphere_sphere_contact::GpuSphereSphereContact)
//! must reproduce the `CPU` golden `sphere_sphere` of
//! `prism_physics_core::collide::primitives`, which turns two world-space
//! sphere centers and radii into a contact manifold: a normal pointing from
//! `a` toward `b`, two surface witness points, and a penetration depth, or no
//! contact when the pair is separated beyond the contact tolerance.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! the center-to-center delta, its length, the penetration test against the
//! contact tolerance, the guarded normal (with the `(1, 0, 0)` fallback for a
//! coincident pair) and the two surface witness points — written out directly
//! in flat `f32` array math so the test never imports `prism_physics_core`,
//! `prism_render_architecture` or `glam`. It mirrors the reference operation
//! for operation and in the same evaluation order.
//!
//! The fixtures cover the regimes the kernel must honor: a separated pair
//! (rejected, all outputs zeroed), a shallow touch, a deep overlap, a
//! coincident degenerate pair (fallback normal `(1, 0, 0)`), a multi-element
//! mixed batch that validates the `std430` array stride end to end, plus an
//! empty batch the host short-circuits with no dispatch. A sweep over random
//! centers and radii follows, kept well away from the penetration knee so the
//! discrete `valid` flag never sits on a branch knife edge.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The witness points thread through a few multiply-adds, so `CPU` and `GPU`
//! evaluate the same closed form but need not be bit-exact. The continuous
//! comparison is `abs_diff <= 1e-4 || rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`) on
//! the normal, both witness points and the penetration; the discrete `valid`
//! flag is compared exactly. Fixtures and the sweep keep the penetration a
//! comfortable margin from `-CONTACT_TOLERANCE` and the distance away from
//! `GEOMETRIC_EPS`, so no comparison sits on a branch knife edge.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collide::primitives::sphere_sphere`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::sphere_sphere_contact::{
    GpuSphereSphereContact, SphereSphereContactQuery,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous-quantity parity comparison.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the continuous-quantity parity comparison.
const REL_EPS: f32 = 1.0e-3;
/// Floor on the relative-tolerance denominator so values near zero still use a
/// meaningful scale.
const REL_FLOOR: f32 = 1.0e-6;

/// Contact tolerance matching the golden constant: a pair is rejected only when
/// the penetration drops below `-CONTACT_TOLERANCE`.
const CONTACT_TOLERANCE: f32 = 1.0e-4;
/// Geometric epsilon matching the golden constant: below this center distance
/// the normal falls back to `(1, 0, 0)`.
const GEOMETRIC_EPS: f32 = 1.0e-6;

/// Returns `true` when two continuous values agree within the module tolerance.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_EPS
}

/// Returns `true` when two 3-vectors agree component-wise within tolerance.
fn close3(a: [f32; 3], b: [f32; 3]) -> bool {
    (0..3).all(|k| close(a[k], b[k]))
}

/// Independent host re-implementation of the golden `sphere_sphere`, returning
/// the contact normal, both witness points, the penetration depth and the
/// `valid` flag without importing the golden crate or `glam`. The delta, its
/// length, the penetration test, the guarded normal and the witness points are
/// evaluated in the same order as the kernel.
fn oracle(q: &SphereSphereContactQuery) -> ([f32; 3], [f32; 3], [f32; 3], f32, u32) {
    let delta = [
        q.center_b[0] - q.center_a[0],
        q.center_b[1] - q.center_a[1],
        q.center_b[2] - q.center_a[2],
    ];
    let dist = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
    let sum_r = q.radius_a + q.radius_b;
    let pen = sum_r - dist;

    if pen < -CONTACT_TOLERANCE {
        return ([0.0; 3], [0.0; 3], [0.0; 3], 0.0, 0);
    }

    let normal = if dist > GEOMETRIC_EPS {
        [delta[0] / dist, delta[1] / dist, delta[2] / dist]
    } else {
        [1.0, 0.0, 0.0]
    };
    let point_a = [
        q.center_a[0] + normal[0] * q.radius_a,
        q.center_a[1] + normal[1] * q.radius_a,
        q.center_a[2] + normal[2] * q.radius_a,
    ];
    let point_b = [
        q.center_b[0] - normal[0] * q.radius_b,
        q.center_b[1] - normal[1] * q.radius_b,
        q.center_b[2] - normal[2] * q.radius_b,
    ];
    (normal, point_a, point_b, pen, 1)
}

/// Dispatches a single query and asserts the device output matches the oracle.
fn assert_parity(ctx: &GpuContext, gpu: &GpuSphereSphereContact, q: SphereSphereContactQuery) {
    let results = gpu.evaluate(ctx, std::slice::from_ref(&q));
    assert_eq!(results.len(), 1, "one result per query");
    let r = results[0];
    let (normal, point_a, point_b, pen, valid) = oracle(&q);
    assert_eq!(r.valid, valid, "valid mismatch: query={q:?}");
    assert!(
        close3(r.normal, normal),
        "normal mismatch: gpu={:?} cpu={normal:?} query={q:?}",
        r.normal
    );
    assert!(
        close3(r.point_a, point_a),
        "point_a mismatch: gpu={:?} cpu={point_a:?} query={q:?}",
        r.point_a
    );
    assert!(
        close3(r.point_b, point_b),
        "point_b mismatch: gpu={:?} cpu={point_b:?} query={q:?}",
        r.point_b
    );
    assert!(
        close(r.penetration, pen),
        "penetration mismatch: gpu={} cpu={pen} query={q:?}",
        r.penetration
    );
}

#[test]
fn separated_pair_is_rejected() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphereSphereContact::new(&ctx);
    // Centers 5 apart, radii sum to 2: a wide gap, well past the tolerance, so
    // the pair is rejected and every output field is zero.
    let q = SphereSphereContactQuery::new([0.0, 0.0, 0.0], 1.0, [5.0, 0.0, 0.0], 1.0);
    assert_parity(&ctx, &gpu, q);
    let r = gpu.evaluate(&ctx, std::slice::from_ref(&q))[0];
    assert_eq!(r.valid, 0, "a widely separated pair is rejected");
    assert!(close3(r.normal, [0.0; 3]), "rejected normal is zeroed");
    assert!(close3(r.point_a, [0.0; 3]), "rejected point_a is zeroed");
    assert!(close3(r.point_b, [0.0; 3]), "rejected point_b is zeroed");
    assert!(close(r.penetration, 0.0), "rejected penetration is zeroed");
}

#[test]
fn shallow_touch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphereSphereContact::new(&ctx);
    // Centers 1.9 apart along X, radii sum to 2: a shallow overlap of 0.1.
    let q = SphereSphereContactQuery::new([0.0, 0.0, 0.0], 1.0, [1.9, 0.0, 0.0], 1.0);
    assert_parity(&ctx, &gpu, q);
    let r = gpu.evaluate(&ctx, std::slice::from_ref(&q))[0];
    assert_eq!(r.valid, 1, "a shallow overlap forms a contact");
    assert!(
        close3(r.normal, [1.0, 0.0, 0.0]),
        "normal points from a toward b along +X, got {:?}",
        r.normal
    );
    assert!(
        close(r.penetration, 0.1),
        "penetration is the overlap depth, got {}",
        r.penetration
    );
}

#[test]
fn deep_overlap_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphereSphereContact::new(&ctx);
    // Centers only 0.5 apart along Y, radii sum to 3: a deep overlap of 2.5.
    let q = SphereSphereContactQuery::new([0.0, 0.0, 0.0], 1.5, [0.0, 0.5, 0.0], 1.5);
    assert_parity(&ctx, &gpu, q);
    let r = gpu.evaluate(&ctx, std::slice::from_ref(&q))[0];
    assert_eq!(r.valid, 1, "a deep overlap forms a contact");
    assert!(
        close3(r.normal, [0.0, 1.0, 0.0]),
        "normal points from a toward b along +Y, got {:?}",
        r.normal
    );
    assert!(
        close(r.penetration, 2.5),
        "penetration is the overlap depth, got {}",
        r.penetration
    );
}

#[test]
fn concentric_pair_uses_fallback_normal() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphereSphereContact::new(&ctx);
    // Coincident centers: the distance is below GEOMETRIC_EPS, so the normal
    // falls back to (1, 0, 0) and the pair is still a valid contact.
    let q = SphereSphereContactQuery::new([2.0, -1.0, 3.0], 1.0, [2.0, -1.0, 3.0], 2.0);
    assert_parity(&ctx, &gpu, q);
    let r = gpu.evaluate(&ctx, std::slice::from_ref(&q))[0];
    assert_eq!(r.valid, 1, "a coincident pair is still a contact");
    assert!(
        close3(r.normal, [1.0, 0.0, 0.0]),
        "coincident pair uses the (1, 0, 0) fallback normal, got {:?}",
        r.normal
    );
    assert!(
        close(r.penetration, 3.0),
        "penetration is the full radii sum, got {}",
        r.penetration
    );
}

#[test]
fn multi_element_batch_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphereSphereContact::new(&ctx);
    // A mixed batch (separated, shallow, deep, concentric) exercises the std430
    // array stride: every slot must decode at the right byte offset.
    let queries = [
        SphereSphereContactQuery::new([0.0, 0.0, 0.0], 1.0, [5.0, 0.0, 0.0], 1.0),
        SphereSphereContactQuery::new([0.0, 0.0, 0.0], 1.0, [1.9, 0.0, 0.0], 1.0),
        SphereSphereContactQuery::new([0.0, 0.0, 0.0], 1.5, [0.0, 0.5, 0.0], 1.5),
        SphereSphereContactQuery::new([2.0, -1.0, 3.0], 1.0, [2.0, -1.0, 3.0], 2.0),
    ];
    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        let (normal, point_a, point_b, pen, valid) = oracle(q);
        assert_eq!(r.valid, valid, "batch valid mismatch: query={q:?}");
        assert!(
            close3(r.normal, normal),
            "batch normal mismatch: gpu={:?} cpu={normal:?} query={q:?}",
            r.normal
        );
        assert!(
            close3(r.point_a, point_a),
            "batch point_a mismatch: gpu={:?} cpu={point_a:?} query={q:?}",
            r.point_a
        );
        assert!(
            close3(r.point_b, point_b),
            "batch point_b mismatch: gpu={:?} cpu={point_b:?} query={q:?}",
            r.point_b
        );
        assert!(
            close(r.penetration, pen),
            "batch penetration mismatch: gpu={} cpu={pen} query={q:?}",
            r.penetration
        );
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphereSphereContact::new(&ctx);
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
    let gpu = GpuSphereSphereContact::new(&ctx);
    let mut rng = Lcg::new(0x4B_17_9E_55);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        let center_a = [
            rng.next_range(-3.0, 3.0),
            rng.next_range(-3.0, 3.0),
            rng.next_range(-3.0, 3.0),
        ];
        let center_b = [
            rng.next_range(-3.0, 3.0),
            rng.next_range(-3.0, 3.0),
            rng.next_range(-3.0, 3.0),
        ];
        let radius_a = rng.next_range(0.2, 2.0);
        let radius_b = rng.next_range(0.2, 2.0);
        let delta = [
            center_b[0] - center_a[0],
            center_b[1] - center_a[1],
            center_b[2] - center_a[2],
        ];
        let dist = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
        let pen = (radius_a + radius_b) - dist;
        // Reject samples near the penetration knee so the discrete valid flag
        // never sits on a knife edge, and keep the distance well above
        // GEOMETRIC_EPS so the normalizing branch is unambiguous.
        if (pen + CONTACT_TOLERANCE).abs() < 1.0e-2 || dist < 1.0e-2 {
            continue;
        }
        queries.push(SphereSphereContactQuery::new(
            center_a, radius_a, center_b, radius_b,
        ));
    }

    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        let (normal, point_a, point_b, pen, valid) = oracle(q);
        assert_eq!(r.valid, valid, "sweep valid mismatch: query={q:?}");
        assert!(
            close3(r.normal, normal),
            "sweep normal mismatch: gpu={:?} cpu={normal:?} query={q:?}",
            r.normal
        );
        assert!(
            close3(r.point_a, point_a),
            "sweep point_a mismatch: gpu={:?} cpu={point_a:?} query={q:?}",
            r.point_a
        );
        assert!(
            close3(r.point_b, point_b),
            "sweep point_b mismatch: gpu={:?} cpu={point_b:?} query={q:?}",
            r.point_b
        );
        assert!(
            close(r.penetration, pen),
            "sweep penetration mismatch: gpu={} cpu={pen} query={q:?}",
            r.penetration
        );
    }
}
