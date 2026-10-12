//! Real-device parity for the cubic-spline-segment twin:
//! [`GpuCubicSplineSegment`](prism_volumetric_gpu::cubic_spline_segment::GpuCubicSplineSegment)
//! must reproduce the `CPU` golden `prism_math::curve::spline` evaluator family.
//! Each query carries an operation id (`0..=5`), four control vectors and a
//! segment parameter `t`; any id `> 5` is rejected as invalid.
//!
//! The oracle here is an independent re-implementation of all six closed forms
//! over `[f32; 3]`, written out directly so the test never imports `prism_math`,
//! `prism_render_architecture`, `prism_physics_core` or `glam`. Every evaluator
//! is a finite polynomial that the device may contract, so each `value`
//! component is compared with a tolerance while the discrete `valid` word is
//! compared exactly.
//!
//! The fixtures cover hand-verified endpoint properties (Bézier `t = 0 -> p0`,
//! `t = 1 -> p3`; Hermite `t = 0 -> p0`, `t = 1 -> p1`; the Bézier tangent
//! `t = 0 -> 3*(p1 - p0)`), the identity `catmull_rom == hermite(p1, (p2-p0)/2,
//! p2, (p3-p1)/2, t)`, an out-of-range id, a mixed batch of two or more
//! elements that validates the `std430` stride, an empty batch the host
//! short-circuits, and a 512-step sweep over all six ids.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! # Parity criterion
//!
//! Each `value` component is compared with `abs <= 1e-4 || rel <= 1e-3`
//! (`REL_FLOOR = 1e-6`); the discrete `valid` word is compared exactly.
//!
//! Provenance: 孪生自本仓 `prism_math::curve::spline`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::cubic_spline_segment::{
    CubicSplineSegmentQuery, CubicSplineSegmentResult, GpuCubicSplineSegment,
};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// Component-wise vector add.
fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Component-wise vector subtract.
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Scalar multiply.
fn scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Independent host oracle for `hermite`, in golden operator order.
fn hermite(p0: [f32; 3], m0: [f32; 3], p1: [f32; 3], m1: [f32; 3], t: f32) -> [f32; 3] {
    let t2 = t * t;
    let t3 = t2 * t;
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;
    add(
        add(scale(p0, h00), scale(m0, h10)),
        add(scale(p1, h01), scale(m1, h11)),
    )
}

/// Independent host oracle for `hermite_tangent`.
fn hermite_tangent(p0: [f32; 3], m0: [f32; 3], p1: [f32; 3], m1: [f32; 3], t: f32) -> [f32; 3] {
    let t2 = t * t;
    let h00 = 6.0 * t2 - 6.0 * t;
    let h10 = 3.0 * t2 - 4.0 * t + 1.0;
    let h01 = -6.0 * t2 + 6.0 * t;
    let h11 = 3.0 * t2 - 2.0 * t;
    add(
        add(scale(p0, h00), scale(m0, h10)),
        add(scale(p1, h01), scale(m1, h11)),
    )
}

/// Independent host oracle for `catmull_rom`.
fn catmull_rom(p0: [f32; 3], p1: [f32; 3], p2: [f32; 3], p3: [f32; 3], t: f32) -> [f32; 3] {
    let m1 = scale(sub(p2, p0), 0.5);
    let m2 = scale(sub(p3, p1), 0.5);
    hermite(p1, m1, p2, m2, t)
}

/// Independent host oracle for `catmull_rom_tangent`.
fn catmull_rom_tangent(p0: [f32; 3], p1: [f32; 3], p2: [f32; 3], p3: [f32; 3], t: f32) -> [f32; 3] {
    let m1 = scale(sub(p2, p0), 0.5);
    let m2 = scale(sub(p3, p1), 0.5);
    hermite_tangent(p1, m1, p2, m2, t)
}

/// Independent host oracle for `bezier_cubic`.
fn bezier_cubic(p0: [f32; 3], p1: [f32; 3], p2: [f32; 3], p3: [f32; 3], t: f32) -> [f32; 3] {
    let u = 1.0 - t;
    let uu = u * u;
    let tt = t * t;
    let b0 = uu * u;
    let b1 = 3.0 * uu * t;
    let b2 = 3.0 * u * tt;
    let b3 = tt * t;
    add(
        add(scale(p0, b0), scale(p1, b1)),
        add(scale(p2, b2), scale(p3, b3)),
    )
}

/// Independent host oracle for `bezier_cubic_tangent`.
fn bezier_cubic_tangent(
    p0: [f32; 3],
    p1: [f32; 3],
    p2: [f32; 3],
    p3: [f32; 3],
    t: f32,
) -> [f32; 3] {
    let u = 1.0 - t;
    let c0 = 3.0 * u * u;
    let c1 = 6.0 * u * t;
    let c2 = 3.0 * t * t;
    add(
        add(scale(sub(p1, p0), c0), scale(sub(p2, p1), c1)),
        scale(sub(p3, p2), c2),
    )
}

/// Independent host oracle: resolves one query to `(value, valid)`. An id `> 5`
/// yields `valid = 0` with a zero vector.
fn oracle(q: &CubicSplineSegmentQuery) -> ([f32; 3], u32) {
    let value = match q.op_id {
        0 => hermite(q.a0, q.a1, q.a2, q.a3, q.t),
        1 => hermite_tangent(q.a0, q.a1, q.a2, q.a3, q.t),
        2 => catmull_rom(q.a0, q.a1, q.a2, q.a3, q.t),
        3 => catmull_rom_tangent(q.a0, q.a1, q.a2, q.a3, q.t),
        4 => bezier_cubic(q.a0, q.a1, q.a2, q.a3, q.t),
        5 => bezier_cubic_tangent(q.a0, q.a1, q.a2, q.a3, q.t),
        _ => return ([0.0, 0.0, 0.0], 0),
    };
    (value, 1)
}

/// Tolerance predicate: `abs <= 1e-4 || rel <= 1e-3` with a relative floor.
fn close(a: f32, b: f32) -> bool {
    if a.is_nan() || b.is_nan() {
        return false;
    }
    let diff = (a - b).abs();
    if diff <= 1.0e-4 {
        return true;
    }
    diff <= 1.0e-3 * a.abs().max(b.abs()).max(REL_FLOOR)
}

/// Asserts one `GPU` result matches the oracle: `valid` exactly, each `value`
/// component to tolerance when valid.
fn assert_parity(gpu: &CubicSplineSegmentResult, q: &CubicSplineSegmentQuery, label: &str) {
    let (value, valid) = oracle(q);
    assert_eq!(gpu.valid, valid, "{label}: valid flag mismatch");
    if valid == 1 {
        for k in 0..3 {
            assert!(
                close(gpu.value[k], value[k]),
                "{label}: value[{k}] mismatch gpu={} oracle={}",
                gpu.value[k],
                value[k]
            );
        }
    }
}

#[test]
fn bezier_endpoints_hit_control_points() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCubicSplineSegment::new(&ctx);
    let p0 = [1.0, 2.0, 3.0];
    let p1 = [4.0, -1.0, 0.5];
    let p2 = [-2.0, 5.0, 1.0];
    let p3 = [7.0, 7.0, -3.0];
    let q0 = CubicSplineSegmentQuery::new(4, p0, p1, p2, p3, 0.0);
    let q1 = CubicSplineSegmentQuery::new(4, p0, p1, p2, p3, 1.0);
    let out = gpu.evaluate(&ctx, &[q0, q1]);
    assert_eq!(out.len(), 2);
    for k in 0..3 {
        assert!(close(out[0].value[k], p0[k]), "t=0 should land on p0");
        assert!(close(out[1].value[k], p3[k]), "t=1 should land on p3");
    }
    assert_parity(&out[0], &q0, "bezier_t0");
    assert_parity(&out[1], &q1, "bezier_t1");
}

#[test]
fn hermite_endpoints_hit_positions() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCubicSplineSegment::new(&ctx);
    let p0 = [0.0, 1.0, -2.0];
    let m0 = [3.0, 0.0, 1.0];
    let p1 = [5.0, -4.0, 2.0];
    let m1 = [-1.0, 2.0, 0.0];
    let q0 = CubicSplineSegmentQuery::new(0, p0, m0, p1, m1, 0.0);
    let q1 = CubicSplineSegmentQuery::new(0, p0, m0, p1, m1, 1.0);
    let out = gpu.evaluate(&ctx, &[q0, q1]);
    assert_eq!(out.len(), 2);
    for k in 0..3 {
        assert!(close(out[0].value[k], p0[k]), "t=0 should land on p0");
        assert!(close(out[1].value[k], p1[k]), "t=1 should land on p1");
    }
    assert_parity(&out[0], &q0, "hermite_t0");
    assert_parity(&out[1], &q1, "hermite_t1");
}

#[test]
fn bezier_tangent_start_is_three_times_edge() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCubicSplineSegment::new(&ctx);
    let p0 = [1.0, 1.0, 1.0];
    let p1 = [2.5, 3.0, -1.0];
    let p2 = [0.0, 4.0, 2.0];
    let p3 = [6.0, -2.0, 5.0];
    let q = CubicSplineSegmentQuery::new(5, p0, p1, p2, p3, 0.0);
    let out = gpu.evaluate(&ctx, &[q]);
    assert_eq!(out.len(), 1);
    let expected = scale(sub(p1, p0), 3.0);
    for k in 0..3 {
        assert!(
            close(out[0].value[k], expected[k]),
            "bezier tangent at t=0 should be 3*(p1-p0)"
        );
    }
    assert_parity(&out[0], &q, "bezier_tangent_t0");
}

#[test]
fn catmull_rom_matches_hermite_construction() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCubicSplineSegment::new(&ctx);
    let p0 = [-3.0, 2.0, 0.0];
    let p1 = [1.0, 1.0, 4.0];
    let p2 = [5.0, -2.0, 2.0];
    let p3 = [2.0, 6.0, -1.0];
    let t = 0.37;
    let q = CubicSplineSegmentQuery::new(2, p0, p1, p2, p3, t);
    let out = gpu.evaluate(&ctx, &[q]);
    assert_eq!(out.len(), 1);
    // catmull_rom == hermite(p1, (p2-p0)/2, p2, (p3-p1)/2, t).
    let m1 = scale(sub(p2, p0), 0.5);
    let m2 = scale(sub(p3, p1), 0.5);
    let expected = hermite(p1, m1, p2, m2, t);
    for k in 0..3 {
        assert!(
            close(out[0].value[k], expected[k]),
            "catmull_rom should equal its Hermite construction"
        );
    }
    assert_parity(&out[0], &q, "catmull_identity");
}

#[test]
fn out_of_range_op_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCubicSplineSegment::new(&ctx);
    let pts = [1.0, 2.0, 3.0];
    let a = CubicSplineSegmentQuery::new(6, pts, pts, pts, pts, 0.5);
    let b = CubicSplineSegmentQuery::new(99, pts, pts, pts, pts, 0.5);
    let out = gpu.evaluate(&ctx, &[a, b]);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].valid, 0, "op 6 should be invalid");
    assert_eq!(out[1].valid, 0, "op 99 should be invalid");
    for k in 0..3 {
        assert!(close(out[0].value[k], 0.0), "invalid value should be zero");
        assert!(close(out[1].value[k], 0.0), "invalid value should be zero");
    }
    assert_parity(&out[0], &a, "oob6");
    assert_parity(&out[1], &b, "oob99");
}

#[test]
fn mixed_batch_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCubicSplineSegment::new(&ctx);
    let p0 = [1.0, 0.0, -1.0];
    let p1 = [2.0, 3.0, 1.0];
    let p2 = [-1.0, 4.0, 2.0];
    let p3 = [5.0, 1.0, -2.0];
    let queries = vec![
        CubicSplineSegmentQuery::new(0, p0, p1, p2, p3, 0.25),
        CubicSplineSegmentQuery::new(1, p0, p1, p2, p3, 0.5),
        CubicSplineSegmentQuery::new(7, p0, p1, p2, p3, 0.5), // invalid id
        CubicSplineSegmentQuery::new(3, p0, p1, p2, p3, 0.75),
        CubicSplineSegmentQuery::new(4, p0, p1, p2, p3, 0.6),
        CubicSplineSegmentQuery::new(5, p0, p1, p2, p3, 0.1),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    assert_eq!(out[2].valid, 0, "invalid id slot should report valid=0");
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("batch[{i}]"));
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCubicSplineSegment::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCubicSplineSegment::new(&ctx);
    let mut lcg = Lcg::new(0x5C2B_1D77);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        // Cycle through all six ops so each gets ~85 samples.
        let op = (queries.len() % 6) as u32;
        let a0 = [
            lcg.next_range(-10.0, 10.0),
            lcg.next_range(-10.0, 10.0),
            lcg.next_range(-10.0, 10.0),
        ];
        let a1 = [
            lcg.next_range(-10.0, 10.0),
            lcg.next_range(-10.0, 10.0),
            lcg.next_range(-10.0, 10.0),
        ];
        let a2 = [
            lcg.next_range(-10.0, 10.0),
            lcg.next_range(-10.0, 10.0),
            lcg.next_range(-10.0, 10.0),
        ];
        let a3 = [
            lcg.next_range(-10.0, 10.0),
            lcg.next_range(-10.0, 10.0),
            lcg.next_range(-10.0, 10.0),
        ];
        // Keep t away from the exact segment endpoints so no component sits on a
        // basis knee; parity holds across the whole span regardless.
        let t = lcg.next_range(0.05, 0.95);
        queries.push(CubicSplineSegmentQuery::new(op, a0, a1, a2, a3, t));
    }
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("sweep[{i}]"));
    }
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
