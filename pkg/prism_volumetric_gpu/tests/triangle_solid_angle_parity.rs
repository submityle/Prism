//! Real-device parity for the signed-solid-angle twin:
//! [`GpuTriangleSolidAngle`](prism_volumetric_gpu::triangle_solid_angle::GpuTriangleSolidAngle)
//! must reproduce the `CPU` golden `signed_solid_angle` of
//! `prism_physics_core::collider::winding_number` for a single triangle seen
//! from a query point.
//!
//! The oracle here is an independent re-implementation of that closed form,
//! written out directly so the test never imports `prism_render_architecture`,
//! `prism_physics_core` or `glam`. It threads through the same operator order
//! as the device kernel: lift the three vertices relative to the query point,
//! form the Van Oosterom-Strackee `denom` and `numer`, and take
//! `2 * atan2(numer, denom)`, forcing a zero contribution when a vertex
//! coincides with the query point so an otherwise undefined `atan2(0, 0)`
//! contributes nothing.
//!
//! The golden accumulates in `f64`, so the oracle also evaluates in `f64`; the
//! device kernel runs in `f32` with `atan2` as the `WGSL` built-in, so the two
//! sides are compared to tolerance rather than bit-exactly.
//!
//! # Parity criterion
//!
//! The continuous arithmetic threads through operators a `GPU` may contract and
//! through the `f32`-vs-`f64` `atan2` gap, so `CPU` and `GPU` evaluate the same
//! closed form but need not be bit-exact. Both continuous scalars are compared
//! with `abs <= 1e-4 || rel <= 1e-3` (`REL_FLOOR = 1e-6`). The `atan2` branch
//! cut lives at `numer -> 0` with `denom < 0`, where the solid angle jumps
//! between `+2*pi` and `-2*pi`; the random fixtures are rejection-sampled to
//! keep every vertex a clear margin from the query point and the magnitude
//! clear of that `+-2*pi` knee, so round-off cannot land the two sides on
//! opposite branches.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::winding_number`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::triangle_solid_angle::{
    GpuTriangleSolidAngle, TriangleSolidAngleQuery, TriangleSolidAngleResult,
};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// Vertex-coincidence degeneracy edge, mirroring the kernel's `EPS0`.
const EPS0: f64 = 1.0e-30;

fn to_f64(a: [f32; 3]) -> [f64; 3] {
    [a[0] as f64, a[1] as f64, a[2] as f64]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn length(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

/// Independent `f64` re-implementation of the golden `signed_solid_angle`,
/// mirroring the device kernel's operator order exactly.
fn oracle(q: &TriangleSolidAngleQuery) -> TriangleSolidAngleResult {
    let point = to_f64(q.point);
    let a = sub(to_f64(q.v0), point);
    let b = sub(to_f64(q.v1), point);
    let c = sub(to_f64(q.v2), point);
    let la = length(a);
    let lb = length(b);
    let lc = length(c);

    let denom = la * lb * lc + dot(a, b) * lc + dot(b, c) * la + dot(c, a) * lb;
    let numer = dot(a, cross(b, c));

    let degenerate = numer.abs() < EPS0 && denom.abs() < EPS0;
    let omega = if degenerate {
        0.0
    } else {
        2.0 * numer.atan2(denom)
    };
    let winding = omega / (4.0 * std::f64::consts::PI);
    TriangleSolidAngleResult {
        solid_angle: omega as f32,
        winding_contribution: winding as f32,
    }
}

/// Mixed absolute-or-relative closeness for a continuous quantity.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= 1.0e-4 {
        return true;
    }
    diff <= 1.0e-3 * a.abs().max(b.abs()).max(REL_FLOOR)
}

/// Asserts both continuous outputs match the independent oracle.
fn assert_full(gpu: &TriangleSolidAngleResult, q: &TriangleSolidAngleQuery, label: &str) {
    let want = oracle(q);
    assert!(
        close(gpu.solid_angle, want.solid_angle),
        "{label}: solid_angle gpu={} oracle={}",
        gpu.solid_angle,
        want.solid_angle
    );
    assert!(
        close(gpu.winding_contribution, want.winding_contribution),
        "{label}: winding gpu={} oracle={}",
        gpu.winding_contribution,
        want.winding_contribution
    );
}

/// A query is well conditioned when every vertex is a clear margin from the
/// query point (so `atan2(0, 0)` cannot trigger) and the solid-angle magnitude
/// is clear of the `+-2*pi` branch cut, so round-off cannot flip the branch.
fn well_conditioned(q: &TriangleSolidAngleQuery) -> bool {
    let point = to_f64(q.point);
    let a = sub(to_f64(q.v0), point);
    let b = sub(to_f64(q.v1), point);
    let c = sub(to_f64(q.v2), point);
    let la = length(a);
    let lb = length(b);
    let lc = length(c);
    if la < 0.05 || lb < 0.05 || lc < 0.05 {
        return false;
    }
    let want = oracle(q);
    let two_pi = 2.0 * std::f32::consts::PI;
    want.solid_angle.abs() <= two_pi - 0.1
}

/// A triangle in the `z = 0` plane; a query above it and the mirrored query
/// below it see solid angles of opposite sign. Dispatched as a two-element
/// batch so the test also exercises the `std430` query stride.
#[test]
fn front_and_back_have_opposite_sign() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTriangleSolidAngle::new(&ctx);
    let v0 = [1.0, -0.5, 0.0];
    let v1 = [0.0, 1.0, 0.0];
    let v2 = [-1.0, -0.5, 0.0];
    let front = TriangleSolidAngleQuery::new([0.0, 0.0, 0.7], v0, v1, v2);
    let back = TriangleSolidAngleQuery::new([0.0, 0.0, -0.7], v0, v1, v2);
    let out = gpu.evaluate(&ctx, &[front, back]);
    assert_eq!(out.len(), 2);
    // Opposite sides of the triangle give opposite-signed solid angles.
    assert!(
        out[0].solid_angle * out[1].solid_angle < 0.0,
        "front {} and back {} should have opposite sign",
        out[0].solid_angle,
        out[1].solid_angle
    );
    assert_full(&out[0], &front, "front");
    assert_full(&out[1], &back, "back");
}

/// A query coplanar with the triangle but outside it has a zero numerator, so
/// the solid angle vanishes.
#[test]
fn coplanar_outside_point_is_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTriangleSolidAngle::new(&ctx);
    let q = TriangleSolidAngleQuery::new(
        [5.0, 5.0, 0.0],
        [1.0, -0.5, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, -0.5, 0.0],
    );
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(
        close(out[0].solid_angle, 0.0),
        "omega {}",
        out[0].solid_angle
    );
    assert_full(&out[0], &q, "coplanar_outside");
}

/// A query coincident with a vertex makes both `numer` and `denom` vanish, so
/// the kernel emits a hard zero rather than an undefined `atan2(0, 0)`.
#[test]
fn vertex_coincident_is_degenerate_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTriangleSolidAngle::new(&ctx);
    let v0 = [0.3, -0.7, 1.2];
    let q = TriangleSolidAngleQuery::new(v0, v0, [0.0, 1.0, 0.0], [-1.0, -0.5, 0.0]);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(
        close(out[0].solid_angle, 0.0),
        "omega {}",
        out[0].solid_angle
    );
    assert!(
        close(out[0].winding_contribution, 0.0),
        "winding {}",
        out[0].winding_contribution
    );
    assert_full(&out[0], &q, "vertex_coincident");
}

/// A mixed batch of a normal front query, a degenerate coincident query and a
/// normal back query validates that each `std430` slot decodes independently.
#[test]
fn batch_mixes_cases_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTriangleSolidAngle::new(&ctx);
    let v0 = [1.0, -0.5, 0.0];
    let v1 = [0.0, 1.0, 0.0];
    let v2 = [-1.0, -0.5, 0.0];
    let normal_front = TriangleSolidAngleQuery::new([0.0, 0.0, 0.6], v0, v1, v2);
    let degenerate = TriangleSolidAngleQuery::new(v1, v0, v1, v2);
    let normal_back = TriangleSolidAngleQuery::new([0.0, 0.0, -0.9], v0, v1, v2);
    let queries = [normal_front, degenerate, normal_back];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), 3);
    assert_full(&out[0], &normal_front, "mix[0]");
    assert!(
        close(out[1].solid_angle, 0.0),
        "mix[1] omega {}",
        out[1].solid_angle
    );
    assert_full(&out[1], &degenerate, "mix[1]");
    assert_full(&out[2], &normal_back, "mix[2]");
}

/// An empty batch short-circuits on the host with no dispatch issued.
#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTriangleSolidAngle::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

/// A `512`-sample rejection-sampled sweep of well-conditioned triangles, each
/// compared in full against the independent oracle.
#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTriangleSolidAngle::new(&ctx);
    let mut lcg = Lcg::new(0x50A1_D4E5);
    let mut queries = Vec::with_capacity(512);
    let mut guard = 0u32;
    while queries.len() < 512 && guard < 5_000_000 {
        guard += 1;
        let q = random_query(&mut lcg);
        if well_conditioned(&q) {
            queries.push(q);
        }
    }
    assert_eq!(
        queries.len(),
        512,
        "could not sample enough well-conditioned triangles"
    );
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_full(res, q, &format!("sweep[{i}]"));
    }
}

/// Builds a random query point and triangle with coordinates in `[-2, 2]^3`.
fn random_query(lcg: &mut Lcg) -> TriangleSolidAngleQuery {
    let mut vertex = || {
        [
            lcg.next_range(-2.0, 2.0),
            lcg.next_range(-2.0, 2.0),
            lcg.next_range(-2.0, 2.0),
        ]
    };
    let point = vertex();
    TriangleSolidAngleQuery::new(point, vertex(), vertex(), vertex())
}

/// A small deterministic linear-congruential generator; the fixture carries no
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
