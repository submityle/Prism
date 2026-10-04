//! Real-device parity for the capsule-contains twin:
//! [`GpuBoundingCapsuleContains`](prism_volumetric_gpu::bounding_capsule_contains::GpuBoundingCapsuleContains)
//! must reproduce the `CPU` golden `BoundingCapsule::contains` of
//! `prism_physics_core::collider::bounding_capsule`.
//!
//! The oracle here is an independent re-implementation of that closed form,
//! written out with scalar `f32` arithmetic so the test never imports
//! `prism_physics_core`, `prism_render_architecture` or `glam`. The point is
//! projected onto the central segment, the squared distance `dsq` is measured,
//! and the inflated radius `r = radius + radius * CONTAIN_EPS + eps` drives the
//! membership test `dsq <= r * r`. Every operator is evaluated in the same order
//! the kernel uses.
//!
//! # Precision note
//!
//! The golden computes `distance_sq_to_segment` in `f64`; both the twin and this
//! oracle use `f32`. The continuous `dsq` channel is compared with an
//! absolute-or-relative tolerance (`abs <= 1e-4 || rel <= 1e-3`,
//! `REL_FLOOR = 1e-6`); the discrete `contains` flag is compared exactly. Since
//! `f32`-vs-`f64` could only disagree in a razor-thin band around
//! `dsq == r * r`, every fixture and every swept query is kept well away from
//! that boundary (`|dsq - r*r|` far larger than the tolerance), so the discrete
//! flag never flips. The one division is guarded, so a zero-length segment
//! (a sphere) poses no conditioning hazard; `valid` is always `1`.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::bounding_capsule`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::bounding_capsule_contains::{
    BoundingCapsuleContainsQuery, BoundingCapsuleContainsResult, GpuBoundingCapsuleContains,
};
use prism_volumetric_gpu::GpuContext;

/// The capsule-contains relative slack, matching the golden `f32` constant.
const CONTAIN_EPS: f32 = 1e-5;

/// Independent host re-implementation of `BoundingCapsule::contains`, flattened
/// into the `(dsq, contains, valid)` record the twin encodes. Each operator
/// mirrors the kernel in evaluation order and stays in `f32`. No
/// `prism_physics_core` / `glam` import.
fn oracle(q: &BoundingCapsuleContainsQuery) -> BoundingCapsuleContainsResult {
    let axis = [q.cbx - q.cax, q.cby - q.cay, q.cbz - q.caz];
    let seg_len_sq = axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2];

    let t = if seg_len_sq > 0.0 {
        let dot = (q.px - q.cax) * axis[0] + (q.py - q.cay) * axis[1] + (q.pz - q.caz) * axis[2];
        (dot / seg_len_sq).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let closest = [
        q.cax + axis[0] * t,
        q.cay + axis[1] * t,
        q.caz + axis[2] * t,
    ];
    let diff = [q.px - closest[0], q.py - closest[1], q.pz - closest[2]];
    let dsq = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];

    let slack = q.radius * CONTAIN_EPS + q.eps;
    let r = q.radius + slack;
    let contains = u32::from(dsq <= r * r);
    BoundingCapsuleContainsResult {
        dsq,
        contains,
        valid: 1,
    }
}

/// The inflated radius `r` for a query, used by fixtures to position points far
/// from the `dsq == r*r` boundary.
fn inflated_radius(radius: f32, eps: f32) -> f32 {
    radius + radius * CONTAIN_EPS + eps
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

/// Asserts one GPU result matches the oracle: `dsq` is continuous (tolerance);
/// `contains` and `valid` are discrete (exact).
fn assert_result(
    got: BoundingCapsuleContainsResult,
    want: BoundingCapsuleContainsResult,
    label: &str,
) {
    assert_eq!(got.valid, want.valid, "valid mismatch: {label}");
    assert_eq!(got.contains, want.contains, "contains mismatch: {label}");
    assert!(
        close(got.dsq, want.dsq),
        "dsq mismatch: {label}: got {} want {}",
        got.dsq,
        want.dsq
    );
}

/// Asserts a single-query GPU result matches the oracle.
fn assert_parity(
    ctx: &GpuContext,
    gpu: &GpuBoundingCapsuleContains,
    q: BoundingCapsuleContainsQuery,
    label: &str,
) {
    let got = gpu.evaluate(ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1, "one result per query: {label}");
    assert_result(got[0], oracle(&q), label);
}

#[test]
fn interior_axis_projection_inside() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingCapsuleContains::new(&ctx);
    // Point projects to the middle of the segment, offset half a radius away.
    let q = BoundingCapsuleContainsQuery::new(
        [0.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
        1.0,
        [0.5, 2.0, 0.0],
        0.0,
    );
    assert_parity(&ctx, &gpu, q, "interior axis projection inside");
}

#[test]
fn endpoint_clamp_low_inside() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingCapsuleContains::new(&ctx);
    // Point beyond center_a: t clamps to 0, measured against the first cap.
    let q = BoundingCapsuleContainsQuery::new(
        [0.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
        1.5,
        [0.0, -1.0, 0.0],
        0.0,
    );
    assert_parity(&ctx, &gpu, q, "endpoint clamp low inside");
}

#[test]
fn endpoint_clamp_high_outside() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingCapsuleContains::new(&ctx);
    // Point well beyond center_b: t clamps to 1, far outside the end cap.
    let q = BoundingCapsuleContainsQuery::new(
        [0.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
        1.0,
        [0.0, 10.0, 0.0],
        0.0,
    );
    assert_parity(&ctx, &gpu, q, "endpoint clamp high outside");
}

#[test]
fn zero_length_sphere_inside() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingCapsuleContains::new(&ctx);
    // center_a == center_b degenerates to a sphere; the point is comfortably in.
    let q = BoundingCapsuleContainsQuery::new(
        [1.0, -2.0, 3.0],
        [1.0, -2.0, 3.0],
        3.0,
        [2.0, -2.0, 3.0],
        0.0,
    );
    assert_parity(&ctx, &gpu, q, "zero-length sphere inside");
}

#[test]
fn zero_length_sphere_outside() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingCapsuleContains::new(&ctx);
    // Same sphere, point far outside.
    let q = BoundingCapsuleContainsQuery::new(
        [1.0, -2.0, 3.0],
        [1.0, -2.0, 3.0],
        1.0,
        [6.0, -2.0, 3.0],
        0.0,
    );
    assert_parity(&ctx, &gpu, q, "zero-length sphere outside");
}

#[test]
fn clearly_inside_small_dsq() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingCapsuleContains::new(&ctx);
    // Point sits almost on the segment: dsq is tiny, far below r*r.
    let q = BoundingCapsuleContainsQuery::new(
        [-2.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        2.0,
        [0.0, 0.1, 0.0],
        0.0,
    );
    assert_parity(&ctx, &gpu, q, "clearly inside small dsq");
}

#[test]
fn clearly_outside_large_dsq() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingCapsuleContains::new(&ctx);
    // Point is many radii away: dsq is huge, far above r*r.
    let q = BoundingCapsuleContainsQuery::new(
        [-2.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        0.5,
        [0.0, 20.0, 0.0],
        0.0,
    );
    assert_parity(&ctx, &gpu, q, "clearly outside large dsq");
}

#[test]
fn eps_slack_pushes_inside() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingCapsuleContains::new(&ctx);
    // Point sits at distance 2.5 from the axis; radius 2 would exclude it, but a
    // generous eps = 1.0 inflates r to 3, placing the point well inside.
    let q = BoundingCapsuleContainsQuery::new(
        [0.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
        2.0,
        [2.5, 2.0, 0.0],
        1.0,
    );
    assert_parity(&ctx, &gpu, q, "eps slack pushes inside");
}

#[test]
fn diagonal_segment_inside() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingCapsuleContains::new(&ctx);
    let q = BoundingCapsuleContainsQuery::new(
        [0.5, -0.5, 1.0],
        [2.5, 1.5, 3.0],
        1.0,
        [1.5, 0.4, 2.0],
        0.0,
    );
    assert_parity(&ctx, &gpu, q, "diagonal segment inside");
}

#[test]
fn mixed_batch_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingCapsuleContains::new(&ctx);
    // A >=2-element batch with distinct shapes and a mix of inside/outside checks
    // the std430 stride end to end: every element must land at its own slot and
    // decode correctly.
    let queries = [
        BoundingCapsuleContainsQuery::new(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            1.0,
            [0.2, 0.0, 0.0],
            0.0,
        ),
        BoundingCapsuleContainsQuery::new(
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            0.5,
            [2.0, 3.0, 0.0],
            0.0,
        ),
        BoundingCapsuleContainsQuery::new(
            [-1.0, 2.0, -3.0],
            [4.0, -2.0, 1.0],
            3.0,
            [1.0, 0.0, -1.0],
            0.0,
        ),
        BoundingCapsuleContainsQuery::new(
            [10.0, 10.0, 10.0],
            [10.0, 13.0, 14.0],
            0.25,
            [10.0, 11.0, 12.0],
            0.1,
        ),
    ];
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (i, q) in queries.iter().enumerate() {
        assert_result(got[i], oracle(q), &format!("mixed batch element {i}"));
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingCapsuleContains::new(&ctx);
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "empty batch returns no results");
}

/// Small LCG for a reproducible pseudo-random sweep.
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

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingCapsuleContains::new(&ctx);
    let mut rng = Lcg::new(0x5EED_B10B);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        let ca = [
            rng.next_range(-10.0, 10.0),
            rng.next_range(-10.0, 10.0),
            rng.next_range(-10.0, 10.0),
        ];
        let cb = [
            rng.next_range(-10.0, 10.0),
            rng.next_range(-10.0, 10.0),
            rng.next_range(-10.0, 10.0),
        ];
        let radius = rng.next_range(0.25, 4.0);
        let eps = rng.next_range(0.0, 0.5);
        let point = [
            rng.next_range(-12.0, 12.0),
            rng.next_range(-12.0, 12.0),
            rng.next_range(-12.0, 12.0),
        ];
        let q = BoundingCapsuleContainsQuery::new(ca, cb, radius, point, eps);
        // Rejection sampling: keep |dsq - r*r| far from the boundary so the
        // discrete `contains` flag can never flip under f32-vs-f64 round-off.
        let want = oracle(&q);
        let r = inflated_radius(radius, eps);
        let r_sq = r * r;
        let margin = (want.dsq - r_sq).abs();
        let scale = r_sq.max(want.dsq).max(1.0);
        if margin < 0.05 * scale {
            continue;
        }
        queries.push(q);
    }
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (i, q) in queries.iter().enumerate() {
        assert_result(got[i], oracle(q), &format!("sweep index {i}"));
    }
}
