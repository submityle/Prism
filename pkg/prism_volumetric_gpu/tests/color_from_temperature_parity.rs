//! Real-device parity for the color-from-temperature twin:
//! [`GpuColorFromTemperature`](prism_volumetric_gpu::color_from_temperature::GpuColorFromTemperature)
//! must reproduce the `CPU` golden `prism_math::color::LinearRgba::from_temperature`
//! for one correlated color temperature per thread. A blackbody temperature in
//! Kelvin is mapped to a unit-luminance linear sRGB color: first to a `CIE` 1931
//! `(x, y)` chromaticity on the Planckian locus using the Kim et al. (2002)
//! cubic-spline approximation, then to `XYZ` with luminance normalized to
//! `Y = 1`, then through the `XYZ` to linear-`RGB` matrix, with negative
//! components clamped to `0`.
//!
//! The oracle below is an independent re-implementation of that closed form,
//! written out directly so the test never imports `prism_render_architecture`,
//! `prism_physics_core`, `prism_math` or `glam`.
//!
//! This is the complete `CCT` to linear-`RGB` composite. It is a different
//! function from the `planckian_locus` twin, which stops at the `(x, y)`
//! chromaticity, and from the `color_temperature` twin, which reproduces the
//! `WhiteBalance` red/blue gain rational pieces; the three share no outputs.
//!
//! The fixtures cover the two clamp regions, each spline regime, both the
//! `2222` and `4000` Kelvin knees from each side, non-finite inputs
//! (`NaN`/infinity) that report `valid = 0`, a mixed batch validating the
//! `std430` stride, an empty batch the host short-circuits, and a `512`-step
//! random sweep that stays clear of the knees so the piecewise decision cannot
//! flip by round-off.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each continuous output `r`, `g`, `b` is compared with
//! `abs <= 1e-4 || rel <= 1e-3` (`REL_FLOOR = 1e-6`); the discrete `valid` flag
//! is compared exactly, and a non-finite input must yield a zero color.
//!
//! Provenance: 孪生自本仓 `prism_math::color::temperature` 与
//! `prism_math::color::linear`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::color_from_temperature::{
    ColorFromTemperatureQuery, ColorFromTemperatureResult, GpuColorFromTemperature,
};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// Independent host oracle: reproduces `from_temperature` in the golden `f32`
/// operator order. A non-finite `kelvin` yields a zero color and `valid = 0`.
fn oracle(q: &ColorFromTemperatureQuery) -> ColorFromTemperatureResult {
    let is_finite = q.kelvin.abs() < 3.0e38;
    if !is_finite {
        return ColorFromTemperatureResult {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            valid: 0,
        };
    }

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

    // xyY (Y = 1) -> XYZ.
    let big_x = x / y;
    let big_y = 1.0;
    let big_z = (1.0 - x - y) / y;

    // XYZ -> linear sRGB.
    let r = 3.240_454_2 * big_x - 1.537_138_5 * big_y - 0.498_531_4 * big_z;
    let g = -0.969_266 * big_x + 1.876_010_8 * big_y + 0.041_556_0 * big_z;
    let b = 0.055_643_4 * big_x - 0.204_025_9 * big_y + 1.057_225_2 * big_z;

    ColorFromTemperatureResult {
        r: r.max(0.0),
        g: g.max(0.0),
        b: b.max(0.0),
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
/// `valid` flag exactly, and the continuous `r`, `g`, `b` scalars to tolerance.
/// When the input is non-finite, the color must be exactly zero.
fn assert_parity(gpu: &ColorFromTemperatureResult, q: &ColorFromTemperatureQuery, label: &str) {
    let want = oracle(q);
    assert_eq!(gpu.valid, want.valid, "{label}: valid flag mismatch");
    if want.valid == 0 {
        assert_eq!(gpu.r, 0.0, "{label}: invalid r must be zero");
        assert_eq!(gpu.g, 0.0, "{label}: invalid g must be zero");
        assert_eq!(gpu.b, 0.0, "{label}: invalid b must be zero");
        return;
    }
    assert!(
        close(gpu.r, want.r),
        "{label}: r gpu={} oracle={}",
        gpu.r,
        want.r
    );
    assert!(
        close(gpu.g, want.g),
        "{label}: g gpu={} oracle={}",
        gpu.g,
        want.g
    );
    assert!(
        close(gpu.b, want.b),
        "{label}: b gpu={} oracle={}",
        gpu.b,
        want.b
    );
}

#[test]
fn clamps_below_lower_bound() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuColorFromTemperature::new(&ctx);
    // Below 1667 K clamps to the lower bound and stays valid.
    let q = ColorFromTemperatureQuery::new(1000.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert_parity(&out[0], &q, "clamp_below");
}

#[test]
fn clamps_above_upper_bound() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuColorFromTemperature::new(&ctx);
    // Above 25000 K clamps to the upper bound and stays valid.
    let q = ColorFromTemperatureQuery::new(30000.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert_parity(&out[0], &q, "clamp_above");
}

#[test]
fn low_regime_2000k() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuColorFromTemperature::new(&ctx);
    // 2000 K sits in the lowest spline regime (t <= 2222).
    let q = ColorFromTemperatureQuery::new(2000.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert_parity(&out[0], &q, "low_2000");
}

#[test]
fn mid_regime_3000k() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuColorFromTemperature::new(&ctx);
    // 3000 K sits in the middle spline regime (2222 < t <= 4000).
    let q = ColorFromTemperatureQuery::new(3000.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert_parity(&out[0], &q, "mid_3000");
}

#[test]
fn high_regime_6000k() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuColorFromTemperature::new(&ctx);
    // 6000 K sits in the highest spline regime (t > 4000).
    let q = ColorFromTemperatureQuery::new(6000.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert_parity(&out[0], &q, "high_6000");
}

#[test]
fn lower_knee_both_sides() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuColorFromTemperature::new(&ctx);
    // 2217 K just below and 2227 K just above the 2222 K knee.
    let below = ColorFromTemperatureQuery::new(2217.0);
    let above = ColorFromTemperatureQuery::new(2227.0);
    let out = gpu.evaluate(&ctx, &[below, above]);
    assert_eq!(out.len(), 2);
    assert_parity(&out[0], &below, "lower_knee_below");
    assert_parity(&out[1], &above, "lower_knee_above");
}

#[test]
fn upper_knee_both_sides() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuColorFromTemperature::new(&ctx);
    // 3995 K just below and 4005 K just above the 4000 K knee.
    let below = ColorFromTemperatureQuery::new(3995.0);
    let above = ColorFromTemperatureQuery::new(4005.0);
    let out = gpu.evaluate(&ctx, &[below, above]);
    assert_eq!(out.len(), 2);
    assert_parity(&out[0], &below, "upper_knee_below");
    assert_parity(&out[1], &above, "upper_knee_above");
}

#[test]
fn non_finite_inputs_are_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuColorFromTemperature::new(&ctx);
    let queries = vec![
        ColorFromTemperatureQuery::new(f32::NAN),
        ColorFromTemperatureQuery::new(f32::INFINITY),
        ColorFromTemperatureQuery::new(f32::NEG_INFINITY),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), 3);
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_eq!(res.valid, 0, "non-finite[{i}] should be invalid");
        assert_eq!(res.r, 0.0, "non-finite[{i}] r must be zero");
        assert_eq!(res.g, 0.0, "non-finite[{i}] g must be zero");
        assert_eq!(res.b, 0.0, "non-finite[{i}] b must be zero");
        assert_parity(res, q, &format!("non_finite[{i}]"));
    }
}

#[test]
fn mixed_batch_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuColorFromTemperature::new(&ctx);
    // A finite/non-finite mixture exercises the std430 stride end to end.
    let queries = vec![
        ColorFromTemperatureQuery::new(1000.0),
        ColorFromTemperatureQuery::new(2000.0),
        ColorFromTemperatureQuery::new(f32::NAN),
        ColorFromTemperatureQuery::new(3000.0),
        ColorFromTemperatureQuery::new(6000.0),
        ColorFromTemperatureQuery::new(30000.0),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    assert_eq!(out[2].valid, 0, "batch NaN entry should be invalid");
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("batch[{i}]"));
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuColorFromTemperature::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuColorFromTemperature::new(&ctx);
    let mut lcg = Lcg::new(0x5EED_0C75);
    let mut queries = Vec::with_capacity(512);
    // Track regime coverage so every spline branch is exercised.
    let mut low = 0u32;
    let mut mid = 0u32;
    let mut high = 0u32;
    while queries.len() < 512 {
        // Cover [1000, 30000] K including the two clamp regions, but reject
        // samples whose clamped temperature lands within 5 K of either knee so
        // the discrete regime cannot flip by round-off.
        let kelvin = lcg.next_range(1000.0, 30000.0);
        let t = kelvin.clamp(1667.0, 25000.0);
        if (t - 2222.0).abs() < 5.0 || (t - 4000.0).abs() < 5.0 {
            continue;
        }
        if t <= 2222.0 {
            low += 1;
        } else if t <= 4000.0 {
            mid += 1;
        } else {
            high += 1;
        }
        queries.push(ColorFromTemperatureQuery::new(kelvin));
    }
    assert!(low >= 20, "sweep should cover the low regime, got {low}");
    assert!(mid >= 20, "sweep should cover the mid regime, got {mid}");
    assert!(high >= 20, "sweep should cover the high regime, got {high}");
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
