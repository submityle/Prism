//! Real-device parity for the Planckian-locus twin:
//! [`GpuPlanckianLocus`](prism_volumetric_gpu::planckian_locus::GpuPlanckianLocus)
//! must reproduce the `CPU` golden `planckian_locus_xy` of
//! `prism_math::color::temperature` for one correlated color temperature per
//! thread. A temperature in Kelvin is mapped to a `CIE` 1931 `(x, y)`
//! chromaticity on the Planckian locus using the Kim et al. (2002) cubic-spline
//! approximation.
//!
//! The oracle below is an independent re-implementation of that closed form —
//! clamp to `[1667, 25000]`, the `x` cubic in `1 / t` chosen by the `4000`
//! Kelvin breakpoint, then the `y` cubic in `x` chosen by the `2222` and `4000`
//! Kelvin breakpoints, in the golden operator order — written out directly so
//! the test never imports `prism_render_architecture`, `prism_physics_core`,
//! `prism_math` or `glam`.
//!
//! This is a distinct fit from `color_temperature`, which reproduces the
//! `WhiteBalance` red/blue `RGB`-gain rational pieces; the two share no
//! coefficients.
//!
//! The fixtures cover the two clamp regions, each spline regime, both the
//! `2222` and `4000` Kelvin knees from each side, a mixed batch validating the
//! `std430` stride, an empty batch the host short-circuits, and a `512`-step
//! random sweep that stays clear of the knees so the discrete `regime` cannot
//! flip by round-off.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each continuous output `x` and `y` is compared with
//! `abs <= 1e-4 || rel <= 1e-3` (`REL_FLOOR = 1e-6`); the discrete `regime` and
//! `valid` flags are compared exactly. The sweep keeps samples at least a few
//! Kelvin clear of the `2222` and `4000` breakpoints so the regime decision
//! cannot flip by round-off.
//!
//! Provenance: 孪生自本仓 `prism_math::color::temperature`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::planckian_locus::{
    GpuPlanckianLocus, PlanckianLocusQuery, PlanckianLocusResult,
};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// Independent host oracle: reproduces `planckian_locus_xy` in the golden `f32`
/// operator order, plus the regime encoding the kernel reports.
fn oracle(q: &PlanckianLocusQuery) -> PlanckianLocusResult {
    let t = q.kelvin.clamp(1667.0, 25000.0);
    let inv = 1.0 / t;
    let inv2 = inv * inv;
    let inv3 = inv2 * inv;

    let x = if t <= 4000.0 {
        -0.266_123_9e9 * inv3 - 0.234_358_9e6 * inv2 + 0.877_695_6e3 * inv + 0.179_910
    } else {
        -3.025_846_9e9 * inv3 + 2.107_038e6 * inv2 + 0.222_634_7e3 * inv + 0.240_390
    };

    let x2 = x * x;
    let x3 = x2 * x;

    let y = if t <= 2222.0 {
        -1.106_381_4 * x3 - 1.348_110_2 * x2 + 2.185_558_3 * x - 0.202_196_83
    } else if t <= 4000.0 {
        -0.954_947_6 * x3 - 1.374_185_9 * x2 + 2.091_37 * x - 0.167_488_67
    } else {
        3.081_758 * x3 - 5.873_387 * x2 + 3.751_129_9 * x - 0.370_014_83
    };

    let x_regime: u32 = if t <= 4000.0 { 0 } else { 1 };
    let y_regime: u32 = if t <= 2222.0 {
        0
    } else if t <= 4000.0 {
        1
    } else {
        2
    };

    PlanckianLocusResult {
        x,
        y,
        regime: x_regime * 4 + y_regime,
        valid: 1,
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

/// Asserts a single `GPU` result matches the independent oracle: the discrete
/// `regime` and `valid` flags exactly, and the continuous `x` and `y` scalars
/// to tolerance.
fn assert_parity(gpu: &PlanckianLocusResult, q: &PlanckianLocusQuery, label: &str) {
    let want = oracle(q);
    assert_eq!(gpu.valid, want.valid, "{label}: valid flag mismatch");
    assert_eq!(gpu.regime, want.regime, "{label}: regime mismatch");
    assert!(
        close(gpu.x, want.x),
        "{label}: x gpu={} oracle={}",
        gpu.x,
        want.x
    );
    assert!(
        close(gpu.y, want.y),
        "{label}: y gpu={} oracle={}",
        gpu.y,
        want.y
    );
}

#[test]
fn clamps_below_lower_bound() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPlanckianLocus::new(&ctx);
    // Below 1667 K clamps to the lower bound; 1667 <= 2222 so regime = 0.
    let q = PlanckianLocusQuery::new(1000.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    let r = out[0];
    assert_eq!(r.valid, 1);
    assert_eq!(r.regime, 0);
    // Equal to evaluating at the clamped lower bound exactly.
    let at_bound = gpu.evaluate(&ctx, &[PlanckianLocusQuery::new(1667.0)]);
    assert!(close(r.x, at_bound[0].x));
    assert!(close(r.y, at_bound[0].y));
    assert_parity(&r, &q, "clamp_low");
}

#[test]
fn clamps_above_upper_bound() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPlanckianLocus::new(&ctx);
    // Above 25000 K clamps to the upper bound; t > 4000 so regime = 6.
    let q = PlanckianLocusQuery::new(30000.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    let r = out[0];
    assert_eq!(r.valid, 1);
    assert_eq!(r.regime, 6);
    let at_bound = gpu.evaluate(&ctx, &[PlanckianLocusQuery::new(25000.0)]);
    assert!(close(r.x, at_bound[0].x));
    assert!(close(r.y, at_bound[0].y));
    assert_parity(&r, &q, "clamp_high");
}

#[test]
fn low_temperature_regime() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPlanckianLocus::new(&ctx);
    // 2000 K: t <= 2222 -> x_regime 0, y_regime 0 -> regime 0.
    let q = PlanckianLocusQuery::new(2000.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    let r = out[0];
    assert_eq!(r.valid, 1);
    assert_eq!(r.regime, 0);
    assert_parity(&r, &q, "low_2000");
}

#[test]
fn mid_temperature_regime() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPlanckianLocus::new(&ctx);
    // 3000 K: 2222 < t <= 4000 -> x_regime 0, y_regime 1 -> regime 1.
    let q = PlanckianLocusQuery::new(3000.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    let r = out[0];
    assert_eq!(r.valid, 1);
    assert_eq!(r.regime, 1);
    assert_parity(&r, &q, "mid_3000");
}

#[test]
fn high_temperature_regime() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPlanckianLocus::new(&ctx);
    // 6000 K: t > 4000 -> x_regime 1, y_regime 2 -> regime 6.
    let q = PlanckianLocusQuery::new(6000.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    let r = out[0];
    assert_eq!(r.valid, 1);
    assert_eq!(r.regime, 6);
    assert_parity(&r, &q, "high_6000");
}

#[test]
fn lower_knee_both_sides() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPlanckianLocus::new(&ctx);
    // 2217 K just below the 2222 K knee -> regime 0; 2227 K just above -> 1.
    let below = PlanckianLocusQuery::new(2217.0);
    let above = PlanckianLocusQuery::new(2227.0);
    let out = gpu.evaluate(&ctx, &[below, above]);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].regime, 0, "2217 K below the lower knee");
    assert_eq!(out[1].regime, 1, "2227 K above the lower knee");
    assert_parity(&out[0], &below, "lower_knee_below");
    assert_parity(&out[1], &above, "lower_knee_above");
}

#[test]
fn upper_knee_both_sides() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPlanckianLocus::new(&ctx);
    // 3995 K just below the 4000 K knee -> regime 1 (x_regime 0, y_regime 1);
    // 4005 K just above -> regime 6 (x_regime 1, y_regime 2).
    let below = PlanckianLocusQuery::new(3995.0);
    let above = PlanckianLocusQuery::new(4005.0);
    let out = gpu.evaluate(&ctx, &[below, above]);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].regime, 1, "3995 K below the upper knee");
    assert_eq!(out[1].regime, 6, "4005 K above the upper knee");
    assert_parity(&out[0], &below, "upper_knee_below");
    assert_parity(&out[1], &above, "upper_knee_above");
}

#[test]
fn exactly_four_thousand_is_low_side() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPlanckianLocus::new(&ctx);
    // t == 4000 satisfies the ordered `t <= 4000`, so x_regime 0, y_regime 1.
    let q = PlanckianLocusQuery::new(4000.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    let r = out[0];
    assert_eq!(r.valid, 1);
    assert_eq!(r.regime, 1);
    assert_parity(&r, &q, "exact_4000");
}

#[test]
fn mixed_batch_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPlanckianLocus::new(&ctx);
    let queries = vec![
        PlanckianLocusQuery::new(1000.0),
        PlanckianLocusQuery::new(2000.0),
        PlanckianLocusQuery::new(3000.0),
        PlanckianLocusQuery::new(6000.0),
        PlanckianLocusQuery::new(30000.0),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    assert_eq!(out[0].regime, 0);
    assert_eq!(out[1].regime, 0);
    assert_eq!(out[2].regime, 1);
    assert_eq!(out[3].regime, 6);
    assert_eq!(out[4].regime, 6);
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_eq!(res.valid, 1, "batch[{i}] should be valid");
        assert_parity(res, q, &format!("batch[{i}]"));
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPlanckianLocus::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPlanckianLocus::new(&ctx);
    let mut lcg = Lcg::new(0x5EED_0C75);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        // Cover [1000, 30000] K including the two clamp regions, but reject
        // samples whose clamped temperature lands within 5 K of either knee so
        // the discrete regime cannot flip by round-off.
        let kelvin = lcg.next_range(1000.0, 30000.0);
        let t = kelvin.clamp(1667.0, 25000.0);
        if (t - 2222.0).abs() < 5.0 || (t - 4000.0).abs() < 5.0 {
            continue;
        }
        queries.push(PlanckianLocusQuery::new(kelvin));
    }
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_eq!(res.valid, 1, "sweep[{i}] should be valid");
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
