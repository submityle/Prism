//! Real-device parity for the closest-point-on-segment projection twin:
//! [`GpuSoftClosestPointOnSegment`](prism_volumetric_gpu::soft_closest_point_on_segment::GpuSoftClosestPointOnSegment)
//! must reproduce the `CPU` golden `closest_point_on_segment` of
//! `prism_physics_core::soft::collision::body`.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! the axis `axis = p1 - p0`, the squared length `len_sq = dot(axis, axis)`,
//! the degenerate guard `len_sq <= EPS_LEN_SQ` returning `p0`, the clamped
//! projection parameter `t = clamp(dot(pos - p0, axis) / len_sq, 0, 1)` and the
//! result `out = p0 + axis * t` — written out directly so the test never
//! imports `prism_physics_core` or `prism_render_architecture`.
//!
//! The fixtures cover a zero-length segment that echoes `p0`, a query whose
//! projection parameter clamps below `0` (snapped to `p0`), one that clamps
//! above `1` (snapped to `p1`), a genuine mid-segment projection, and a mixed
//! batch that validates the `std430` stride end to end. A `512`-query `LCG`
//! sweep follows, keeping samples clear of the `len_sq = EPS_LEN_SQ` degenerate
//! knee and both `t = 0` / `t = 1` clamp knees, and an empty batch the host
//! short-circuits with no dispatch.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! # Parity criterion
//!
//! The closest point (`outx`, `outy`, `outz`) is continuous and checked with an
//! absolute-or-relative tolerance (`abs <= 1e-4 || rel <= 1e-3`,
//! `REL_FLOOR = 1e-6`); `valid` is discrete and compared exactly. The random
//! sweep keeps samples away from the degenerate and both clamp knees so a
//! host/device ordering difference on the guards cannot flip a discrete channel
//! or a clamp decision; dedicated named fixtures pin those boundary cases.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::soft::collision::body`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::soft_closest_point_on_segment::{
    GpuSoftClosestPointOnSegment, SoftClosestPointOnSegmentQuery, SoftClosestPointOnSegmentResult,
};
use prism_volumetric_gpu::GpuContext;

/// Squared-length floor below which the segment is treated as a single point;
/// matches `EPS_LEN_SQ` used by the golden.
const EPS_LEN_SQ: f32 = 1e-12;

/// Independent host re-implementation of `closest_point_on_segment`, flattened
/// into the `(out, valid)` record the twin encodes. The division orders exactly
/// as the device does (`dot(pos - p0, axis) / len_sq`) so no extra `f32` error
/// is introduced. No `prism_physics_core` import.
fn oracle(q: &SoftClosestPointOnSegmentQuery) -> SoftClosestPointOnSegmentResult {
    let ax = q.p1x - q.p0x;
    let ay = q.p1y - q.p0y;
    let az = q.p1z - q.p0z;
    let len_sq = ax * ax + ay * ay + az * az;

    // Ordered guard: a NaN len_sq fails this and routes to the degenerate echo.
    if !(len_sq > EPS_LEN_SQ) {
        return SoftClosestPointOnSegmentResult {
            outx: q.p0x,
            outy: q.p0y,
            outz: q.p0z,
            valid: 0,
        };
    }

    let dpx = q.posx - q.p0x;
    let dpy = q.posy - q.p0y;
    let dpz = q.posz - q.p0z;
    let raw_t = (dpx * ax + dpy * ay + dpz * az) / len_sq;
    let t = raw_t.clamp(0.0, 1.0);

    SoftClosestPointOnSegmentResult {
        outx: q.p0x + ax * t,
        outy: q.p0y + ay * t,
        outz: q.p0z + az * t,
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

/// Asserts one GPU result matches the oracle. The closest point is continuous
/// (tolerance); `valid` is discrete (exact).
fn assert_result(
    got: SoftClosestPointOnSegmentResult,
    want: SoftClosestPointOnSegmentResult,
    label: &str,
) {
    assert_eq!(got.valid, want.valid, "valid mismatch: {label}");
    assert!(
        close(got.outx, want.outx),
        "outx mismatch: {label}: got {} want {}",
        got.outx,
        want.outx
    );
    assert!(
        close(got.outy, want.outy),
        "outy mismatch: {label}: got {} want {}",
        got.outy,
        want.outy
    );
    assert!(
        close(got.outz, want.outz),
        "outz mismatch: {label}: got {} want {}",
        got.outz,
        want.outz
    );
}

/// Asserts a single-query GPU result matches the oracle.
fn assert_parity(
    ctx: &GpuContext,
    gpu: &GpuSoftClosestPointOnSegment,
    q: SoftClosestPointOnSegmentQuery,
    label: &str,
) {
    let got = gpu.evaluate(ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1, "one result per query: {label}");
    assert_result(got[0], oracle(&q), label);
}

#[test]
fn zero_length_segment_echoes_p0() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftClosestPointOnSegment::new(&ctx);
    // A collapsed segment (p0 == p1) has no axis direction: the closest point is
    // p0 and valid is 0, regardless of where the query position lies.
    let q = SoftClosestPointOnSegmentQuery::new([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [5.0, -4.0, 7.0]);
    let want = oracle(&q);
    assert_eq!(want.valid, 0, "fixture sanity: degenerate segment is inert");
    assert_eq!(want.outx, 1.0, "fixture sanity: echoes p0.x");
    assert_eq!(want.outy, 2.0, "fixture sanity: echoes p0.y");
    assert_eq!(want.outz, 3.0, "fixture sanity: echoes p0.z");
    assert_parity(&ctx, &gpu, q, "zero_length_segment_echoes_p0");
}

#[test]
fn projection_clamps_to_p0() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftClosestPointOnSegment::new(&ctx);
    // The query projects behind p0 (parameter < 0): the clamp pins the closest
    // point to p0 while valid stays 1 (the segment has positive length).
    let q = SoftClosestPointOnSegmentQuery::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-2.0, 1.0, 0.0]);
    let want = oracle(&q);
    assert_eq!(want.valid, 1, "fixture sanity: positive-length segment");
    assert!(want.outx.abs() < 1e-4, "fixture sanity: clamped to p0.x");
    assert_parity(&ctx, &gpu, q, "projection_clamps_to_p0");
}

#[test]
fn projection_clamps_to_p1() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftClosestPointOnSegment::new(&ctx);
    // The query projects past p1 (parameter > 1): the clamp pins the closest
    // point to p1.
    let q = SoftClosestPointOnSegmentQuery::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 1.0, 0.0]);
    let want = oracle(&q);
    assert_eq!(want.valid, 1, "fixture sanity: positive-length segment");
    assert!(
        (want.outx - 1.0).abs() < 1e-4,
        "fixture sanity: clamped to p1.x, got {}",
        want.outx
    );
    assert_parity(&ctx, &gpu, q, "projection_clamps_to_p1");
}

#[test]
fn mid_segment_projection() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftClosestPointOnSegment::new(&ctx);
    // The query projects onto the interior of the segment (0 < t < 1): the
    // closest point is the perpendicular foot.
    let q = SoftClosestPointOnSegmentQuery::new([0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 0.0]);
    let want = oracle(&q);
    assert_eq!(want.valid, 1, "fixture sanity: positive-length segment");
    assert!(
        (want.outx - 1.0).abs() < 1e-4,
        "fixture sanity: foot at x = 1, got {}",
        want.outx
    );
    assert_parity(&ctx, &gpu, q, "mid_segment_projection");
}

#[test]
fn mixed_batch_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftClosestPointOnSegment::new(&ctx);
    // A >=2-element batch mixing every branch validates the std430 stride end to
    // end: degenerate echo, both clamp sides and a mid-segment foot.
    let queries = vec![
        SoftClosestPointOnSegmentQuery::new([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [5.0, -4.0, 7.0]),
        SoftClosestPointOnSegmentQuery::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-2.0, 1.0, 0.0]),
        SoftClosestPointOnSegmentQuery::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 1.0, 0.0]),
        SoftClosestPointOnSegmentQuery::new([0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 0.0]),
        SoftClosestPointOnSegmentQuery::new([-1.0, 0.5, 2.0], [3.0, -2.0, 1.0], [0.75, 1.25, -0.5]),
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
    let gpu = GpuSoftClosestPointOnSegment::new(&ctx);
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
    let gpu = GpuSoftClosestPointOnSegment::new(&ctx);
    let mut rng = Lcg::new(0x51_3A_9E_4D);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        let p0 = [
            rng.next_range(-4.0, 4.0),
            rng.next_range(-4.0, 4.0),
            rng.next_range(-4.0, 4.0),
        ];
        let p1 = [
            rng.next_range(-4.0, 4.0),
            rng.next_range(-4.0, 4.0),
            rng.next_range(-4.0, 4.0),
        ];
        let pos = [
            rng.next_range(-4.0, 4.0),
            rng.next_range(-4.0, 4.0),
            rng.next_range(-4.0, 4.0),
        ];

        let ax = p1[0] - p0[0];
        let ay = p1[1] - p0[1];
        let az = p1[2] - p0[2];
        let len_sq = ax * ax + ay * ay + az * az;
        // Keep the main body well clear of the degenerate knee so the discrete
        // valid channel cannot flip on a rounding tie.
        if len_sq < 1e-2 {
            continue;
        }

        let dpx = pos[0] - p0[0];
        let dpy = pos[1] - p0[1];
        let dpz = pos[2] - p0[2];
        let raw_t = (dpx * ax + dpy * ay + dpz * az) / len_sq;
        // Keep samples clear of both clamp knees (t = 0 and t = 1) so a
        // host/device ordering difference cannot flip the clamp decision.
        if (raw_t - 0.0).abs() < 1e-2 || (raw_t - 1.0).abs() < 1e-2 {
            continue;
        }

        queries.push(SoftClosestPointOnSegmentQuery::new(p0, p1, pos));
    }

    // Append a degenerate sample to exercise the valid = 0 echo in the sweep.
    queries.push(SoftClosestPointOnSegmentQuery::new(
        [2.0, -1.0, 0.5],
        [2.0, -1.0, 0.5],
        [-3.0, 4.0, 1.0],
    ));

    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (i, (q, r)) in queries.iter().zip(results.iter()).enumerate() {
        assert_result(*r, oracle(q), &format!("sweep index {i}"));
    }
}
