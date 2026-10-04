//! Real-device parity for the quaternion-slerp twin:
//! [`GpuQuatSlerp`](prism_volumetric_gpu::quat_slerp::GpuQuatSlerp) must
//! reproduce the `CPU` golden `Quat::slerp` of `prism_math::quat`. Spherical
//! linear interpolation walks the shortest great-circle arc between two
//! orientations: the golden dots the operands, negates the right one when the
//! dot is negative (so the interpolation follows the shortest arc, since a
//! quaternion and its negation encode the same rotation), and falls back to a
//! normalized linear interpolation (`nlerp`) when the operands are nearly
//! colinear (`dot > 0.9995`). One thread resolves one query.
//!
//! The oracle here is an independent re-implementation of `slerp` and its
//! `nlerp` fallback, written out directly in `f32` so the test never imports
//! `prism_math`, `prism_render_architecture`, `prism_physics_core` or `glam`.
//!
//! # Regime split and knee avoidance
//!
//! The `regime` flag is a discrete decision: `0` for the near-colinear
//! normalized-lerp fallback and `1` for the general `sin`-weighted slerp,
//! decided by comparing the shortest-arc dot against `0.9995`. Because that
//! decision (and the `dot < 0` shortest-arc flip) is a float threshold, the
//! sweep keeps the shortest-arc dot a comfortable margin away from both the `0`
//! flip knee and the `0.9995` regime knee, so `CPU` and `GPU` never straddle a
//! branch boundary and the exact `regime` comparison holds.
//!
//! # Fixtures
//!
//! Named fixtures cover a general slerp pair (`regime = 1`), a near-colinear
//! pair that falls back to `nlerp` (`regime = 0`), a reverse pair whose raw dot
//! is negative (triggering the shortest-arc flip), a zero-length operand
//! (`valid = 0`), an empty batch the host short-circuits, and a mixed batch
//! that validates the `std430` stride. A `512`-step LCG sweep follows.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! # Parity criterion
//!
//! The continuous arithmetic threads through operators and transcendentals a
//! `GPU` may contract, so `CPU` and `GPU` evaluate the same closed form but need
//! not be bit-exact. The four output components are compared with
//! `abs <= 1e-4 || rel <= 1e-3` (`REL_FLOOR = 1e-6`); the discrete `regime` and
//! `valid` flags are compared exactly.
//!
//! Provenance: 孪生自本仓 `prism_math::quat`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::quat_slerp::{GpuQuatSlerp, QuatSlerpQuery, QuatSlerpResult};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// The regime knee: operands with (shortest-arc) dot above this fall back to
/// `nlerp`.
const DOT_THRESHOLD: f32 = 0.9995;

/// Squared-length guard: an operand with squared length at or below this is
/// degenerate.
const EPS_LEN_SQ: f32 = 1.0e-12;

/// Four-component dot product.
fn dot4(a: [f32; 4], b: [f32; 4]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

/// Independent host oracle: reproduces `Quat::slerp` in the golden operator
/// order, returning `(out, regime, valid)`. A zero-length operand is invalid
/// with `out = [0; 4]` and `regime = 0`.
fn oracle(q: &QuatSlerpQuery) -> ([f32; 4], u32, u32) {
    let (qa, mut rhs, t) = (q.qa, q.qb, q.t);
    let len_sq_a = dot4(qa, qa);
    let len_sq_b = dot4(rhs, rhs);
    if len_sq_a <= EPS_LEN_SQ || len_sq_b <= EPS_LEN_SQ {
        return ([0.0; 4], 0, 0);
    }
    let mut d = dot4(qa, rhs);
    if d < 0.0 {
        rhs = [-rhs[0], -rhs[1], -rhs[2], -rhs[3]];
        d = -d;
    }
    if d > DOT_THRESHOLD {
        // nlerp fallback: rhs is already same-hemisphere, so normalize the
        // straight-line blend qa + (rhs - qa) * t.
        let l = [
            qa[0] + (rhs[0] - qa[0]) * t,
            qa[1] + (rhs[1] - qa[1]) * t,
            qa[2] + (rhs[2] - qa[2]) * t,
            qa[3] + (rhs[3] - qa[3]) * t,
        ];
        let inv = 1.0 / dot4(l, l).sqrt();
        return ([l[0] * inv, l[1] * inv, l[2] * inv, l[3] * inv], 0, 1);
    }
    let theta = d.clamp(-1.0, 1.0).acos();
    let sin_theta = theta.sin();
    let s0 = ((1.0 - t) * theta).sin() / sin_theta;
    let s1 = (t * theta).sin() / sin_theta;
    (
        [
            qa[0] * s0 + rhs[0] * s1,
            qa[1] * s0 + rhs[1] * s1,
            qa[2] * s0 + rhs[2] * s1,
            qa[3] * s0 + rhs[3] * s1,
        ],
        1,
        1,
    )
}

/// Mixed absolute-or-relative closeness for a continuous quantity.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= 1.0e-4 {
        return true;
    }
    diff <= 1.0e-3 * a.abs().max(b.abs()).max(REL_FLOOR)
}

/// Asserts a single `GPU` result matches the independent oracle: the discrete
/// `valid` and `regime` flags exactly (regime only when valid), and the four
/// output components to tolerance when valid.
fn assert_parity(gpu: &QuatSlerpResult, q: &QuatSlerpQuery, label: &str) {
    let (out, regime, valid) = oracle(q);
    assert_eq!(gpu.valid, valid, "{label}: valid flag mismatch");
    if valid == 1 {
        assert_eq!(
            gpu.regime, regime,
            "{label}: regime mismatch gpu={} oracle={}",
            gpu.regime, regime
        );
        for k in 0..4 {
            assert!(
                close(gpu.out[k], out[k]),
                "{label}: out[{k}] mismatch gpu={} oracle={}",
                gpu.out[k],
                out[k]
            );
        }
    } else {
        for k in 0..4 {
            assert_eq!(gpu.out[k], 0.0, "{label}: invalid out[{k}] should be zero");
        }
    }
}

/// Normalizes a quaternion; panics if degenerate (test fixtures only use
/// non-degenerate quaternions when normalizing).
fn normalize(q: [f32; 4]) -> [f32; 4] {
    let inv = 1.0 / dot4(q, q).sqrt();
    [q[0] * inv, q[1] * inv, q[2] * inv, q[3] * inv]
}

#[test]
fn general_slerp_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuatSlerp::new(&ctx);
    // Identity to a 90-degree rotation about z: a clean general (regime 1) pair.
    let qa = [0.0, 0.0, 0.0, 1.0];
    let qb = normalize([0.0, 0.0, 0.707_106_77, 0.707_106_77]);
    let queries = vec![
        QuatSlerpQuery::new(qa, qb, 0.0),
        QuatSlerpQuery::new(qa, qb, 0.25),
        QuatSlerpQuery::new(qa, qb, 0.5),
        QuatSlerpQuery::new(qa, qb, 0.75),
        QuatSlerpQuery::new(qa, qb, 1.0),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_eq!(res.valid, 1, "general[{i}] should be valid");
        assert_eq!(res.regime, 1, "general[{i}] should be regime 1");
        assert_parity(res, q, &format!("general[{i}]"));
    }
}

#[test]
fn near_colinear_falls_back_to_nlerp() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuatSlerp::new(&ctx);
    // Two nearly identical orientations: dot sits above 0.9995, so the golden
    // falls back to nlerp (regime 0).
    let qa = [0.0, 0.0, 0.0, 1.0];
    let qb = normalize([0.0, 0.0, 0.01, 1.0]);
    let d = dot4(qa, qb);
    assert!(d > DOT_THRESHOLD, "fixture must be near-colinear, dot={d}");
    let queries = vec![
        QuatSlerpQuery::new(qa, qb, 0.0),
        QuatSlerpQuery::new(qa, qb, 0.3),
        QuatSlerpQuery::new(qa, qb, 0.5),
        QuatSlerpQuery::new(qa, qb, 1.0),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_eq!(res.valid, 1, "near[{i}] should be valid");
        assert_eq!(res.regime, 0, "near[{i}] should be regime 0 (nlerp)");
        assert_parity(res, q, &format!("near[{i}]"));
    }
}

#[test]
fn reverse_pair_takes_shortest_arc() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuatSlerp::new(&ctx);
    // qb is a large rotation whose raw dot with qa is negative, forcing the
    // shortest-arc flip; the oracle mirrors the flip, so parity confirms it.
    let qa = [0.0, 0.0, 0.0, 1.0];
    let qb = normalize([0.0, 0.0, 0.965_925_8, -0.258_819_04]);
    let raw = dot4(qa, qb);
    assert!(raw < 0.0, "fixture must have negative raw dot, got {raw}");
    let queries = vec![
        QuatSlerpQuery::new(qa, qb, 0.0),
        QuatSlerpQuery::new(qa, qb, 0.4),
        QuatSlerpQuery::new(qa, qb, 1.0),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_eq!(res.valid, 1, "reverse[{i}] should be valid");
        assert_parity(res, q, &format!("reverse[{i}]"));
    }
}

#[test]
fn degenerate_zero_length_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuatSlerp::new(&ctx);
    let good = [0.0, 0.0, 0.0, 1.0];
    let zero = [0.0, 0.0, 0.0, 0.0];
    let queries = vec![
        QuatSlerpQuery::new(zero, good, 0.5),
        QuatSlerpQuery::new(good, zero, 0.5),
        QuatSlerpQuery::new(zero, zero, 0.5),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_eq!(res.valid, 0, "degenerate[{i}] should be invalid");
        for k in 0..4 {
            assert_eq!(res.out[k], 0.0, "degenerate[{i}] out[{k}] should be zero");
        }
        assert_parity(res, q, &format!("degenerate[{i}]"));
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuatSlerp::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn batch_mixes_regimes_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuatSlerp::new(&ctx);
    let qa = [0.0, 0.0, 0.0, 1.0];
    // regime 1: general pair.
    let general = normalize([0.0, 0.0, 0.707_106_77, 0.707_106_77]);
    // regime 0: near-colinear pair.
    let near = normalize([0.0, 0.0, 0.01, 1.0]);
    // zero-length invalid operand.
    let zero = [0.0, 0.0, 0.0, 0.0];
    let queries = vec![
        QuatSlerpQuery::new(qa, general, 0.5),
        QuatSlerpQuery::new(qa, near, 0.5),
        QuatSlerpQuery::new(qa, zero, 0.5),
        QuatSlerpQuery::new(qa, general, 0.2),
        QuatSlerpQuery::new(qa, near, 0.8),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    assert_eq!(out[0].valid, 1);
    assert_eq!(out[0].regime, 1);
    assert_eq!(out[1].valid, 1);
    assert_eq!(out[1].regime, 0);
    assert_eq!(out[2].valid, 0);
    assert_eq!(out[3].valid, 1);
    assert_eq!(out[3].regime, 1);
    assert_eq!(out[4].valid, 1);
    assert_eq!(out[4].regime, 0);
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("batch[{i}]"));
    }
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuatSlerp::new(&ctx);
    let mut lcg = Lcg::new(0x51E3_0D24);
    let mut queries: Vec<QuatSlerpQuery> = Vec::with_capacity(512);
    while queries.len() < 512 {
        // Alternate between the two regimes so both are well covered.
        let want_near = queries.len() % 3 == 0;
        let qa = random_unit_quat(&mut lcg);
        let t = lcg.next_unit();
        if want_near {
            // Build a near-colinear partner: perturb qa slightly then
            // normalize, rejection-sampling the dot into [0.9997, 0.99999].
            let qb = loop {
                let scale = lcg.next_range(0.001, 0.02);
                let delta = random_unit_quat(&mut lcg);
                let cand = normalize([
                    qa[0] + delta[0] * scale,
                    qa[1] + delta[1] * scale,
                    qa[2] + delta[2] * scale,
                    qa[3] + delta[3] * scale,
                ]);
                let mut d = dot4(qa, cand);
                d = d.abs();
                if (0.9997..=0.99999).contains(&d) {
                    break cand;
                }
            };
            queries.push(QuatSlerpQuery::new(qa, qb, t));
        } else {
            // General pair: rejection-sample so the shortest-arc dot sits in a
            // safe band [0.05, 0.985], away from the 0 flip knee and the
            // 0.9995 regime knee. Raw dot may be negative (flip is exercised).
            let qb = loop {
                let cand = random_unit_quat(&mut lcg);
                let d = dot4(qa, cand).abs();
                if (0.05..=0.985).contains(&d) {
                    break cand;
                }
            };
            queries.push(QuatSlerpQuery::new(qa, qb, t));
        }
    }
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_eq!(res.valid, 1, "sweep[{i}] should be valid");
        assert_parity(res, q, &format!("sweep[{i}]"));
    }
}

/// Builds a random unit quaternion; rejection-samples away from near-zero
/// length so normalization is stable.
fn random_unit_quat(lcg: &mut Lcg) -> [f32; 4] {
    loop {
        let q = [
            lcg.next_range(-1.0, 1.0),
            lcg.next_range(-1.0, 1.0),
            lcg.next_range(-1.0, 1.0),
            lcg.next_range(-1.0, 1.0),
        ];
        let len_sq = dot4(q, q);
        if len_sq > 0.1 {
            return normalize(q);
        }
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
