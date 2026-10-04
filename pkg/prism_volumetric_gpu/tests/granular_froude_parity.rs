//! Real-device parity for the granular Froude-number twin:
//! [`GpuGranularFroude`](prism_volumetric_gpu::granular_froude::GpuGranularFroude)
//! must reproduce the `CPU` golden
//! `prism_physics_core::collider::granular_froude`'s `GranularFroude`
//! constructors `from_flow` and `from_inclined_flow`.
//!
//! The Froude number `Fr = u / sqrt(g_eff * h)` compares the depth-averaged
//! flow speed with the shallow gravity-wave speed, where `g_eff` is `g` on a
//! flat base and `g * cos(theta)` on a chute inclined by `theta`. A query is
//! valid only when every input is finite, with `u >= 0`, `h > 0`, `g > 0` and
//! `g_eff > 0`, the wave speed is finite and strictly positive, and the ratio
//! is finite; the inclined path additionally requires `cos(theta) > 0`. The
//! regime uses the half-open threshold `1.0`: `Fr < 1` is `Subcritical`
//! (regime `0`) and `Fr >= 1` is `Supercritical` (regime `1`).
//!
//! The oracle here is an independent re-implementation of that closed form,
//! written out directly so the test never imports `prism_render_architecture`
//! or `prism_physics_core`. The inclined path mirrors the golden exactly by
//! evaluating the cosine in `f64` (`f64::from(theta).cos() as f32`), while the
//! device kernel evaluates `cos` natively in `f32`.
//!
//! # Parity criterion
//!
//! Because the kernel evaluates `cos` natively in `f32` while the golden uses
//! `f64`, `CPU` and `GPU` are not bit-exact. The valid `froude` scalar is
//! compared with `abs <= 1e-4 || rel <= 1e-3` (`REL_FLOOR = 1e-6`), which
//! absorbs that difference; the discrete `regime` and `valid` flags are
//! compared exactly. The random sweep keeps `Fr` out of a narrow band around
//! the critical knee `Fr = 1`, and all inclined angles well inside
//! `cos(theta) > 0`, so the regime and validity decisions agree on both sides.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::granular_froude`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::granular_froude::{
    GpuGranularFroude, GranularFroudeQuery, GranularFroudeResult,
};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// Finiteness bound mirroring the kernel's ordered `abs(x) < 3.0e38` test.
const FINITE_LIMIT: f32 = 3.0e38;

/// Ordered finiteness test matching the kernel (rejects both infinities and
/// `NaN` without a bare `x == x`).
fn is_finite(x: f32) -> bool {
    x.abs() < FINITE_LIMIT
}

/// Independent host oracle: reproduces the `GranularFroude` constructors in
/// pure `f32`, returning `(froude, regime, valid)`. The inclined path evaluates
/// the cosine in `f64` exactly as the golden does.
fn oracle(q: &GranularFroudeQuery) -> (f32, u32, u32) {
    let u = q.speed;
    let h = q.flow_depth;
    let g = q.gravity;
    let base_ok = is_finite(u) && u >= 0.0 && is_finite(h) && h > 0.0 && is_finite(g) && g > 0.0;

    let (g_eff, path_ok) = if q.inclined != 0 {
        let theta = q.slope_angle;
        if !is_finite(theta) {
            (0.0, false)
        } else {
            let cos_theta = f64::from(theta).cos() as f32;
            if cos_theta > 0.0 {
                (g * cos_theta, true)
            } else {
                (0.0, false)
            }
        }
    } else {
        (g, true)
    };

    let g_eff_ok = base_ok && path_ok && is_finite(g_eff) && g_eff > 0.0;
    if !g_eff_ok {
        return (0.0, 0, 0);
    }
    let radicand = g_eff * h;
    if !is_finite(radicand) || radicand <= 0.0 {
        return (0.0, 0, 0);
    }
    let wave = radicand.sqrt();
    if !is_finite(wave) || wave <= 0.0 {
        return (0.0, 0, 0);
    }
    let froude = u / wave;
    if !is_finite(froude) {
        return (0.0, 0, 0);
    }
    let regime = if froude < 1.0 { 0u32 } else { 1u32 };
    (froude, regime, 1u32)
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
/// `regime` and `valid` flags exactly, and the `froude` scalar to tolerance
/// when valid, else all zero.
fn assert_parity(gpu: &GranularFroudeResult, q: &GranularFroudeQuery, label: &str) {
    let (froude, regime, valid) = oracle(q);
    assert_eq!(gpu.valid, valid, "{label}: valid flag mismatch");
    if valid == 1u32 {
        assert_eq!(gpu.regime, regime, "{label}: regime mismatch");
        assert!(
            close(gpu.froude, froude),
            "{label}: froude mismatch gpu={} oracle={}",
            gpu.froude,
            froude
        );
    } else {
        assert_eq!(gpu.froude, 0.0, "{label}: invalid froude should be zero");
        assert_eq!(gpu.regime, 0u32, "{label}: invalid regime should be zero");
    }
}

#[test]
fn flat_subcritical_fixture() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGranularFroude::new(&ctx);
    // u=1, h=1, g=9.81 -> wave=3.1321, Fr=0.31927 < 1 -> Subcritical.
    let queries = vec![GranularFroudeQuery::from_flow(1.0, 1.0, 9.81)];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    assert_eq!(out[0].valid, 1u32);
    assert_eq!(out[0].regime, 0u32);
    assert!(close(out[0].froude, 0.319_275_6));
    assert_parity(&out[0], &queries[0], "flat_subcritical");
}

#[test]
fn flat_supercritical_fixture() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGranularFroude::new(&ctx);
    // u=10, h=1, g=9.81 -> Fr=3.1928 >= 1 -> Supercritical.
    let queries = vec![GranularFroudeQuery::from_flow(10.0, 1.0, 9.81)];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    assert_eq!(out[0].valid, 1u32);
    assert_eq!(out[0].regime, 1u32);
    assert!(close(out[0].froude, 3.192_755_7));
    assert_parity(&out[0], &queries[0], "flat_supercritical");
}

#[test]
fn zero_speed_is_subcritical() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGranularFroude::new(&ctx);
    // u=0 is valid (u >= 0): Fr = 0, Subcritical.
    let queries = vec![GranularFroudeQuery::from_flow(0.0, 2.0, 9.81)];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out[0].valid, 1u32);
    assert_eq!(out[0].regime, 0u32);
    assert!(close(out[0].froude, 0.0));
    assert_parity(&out[0], &queries[0], "zero_speed");
}

#[test]
fn inclined_zero_angle_matches_flat() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGranularFroude::new(&ctx);
    // theta=0 -> cos=1 -> g_eff=g, so the inclined path matches from_flow.
    let flat = GranularFroudeQuery::from_flow(4.0, 0.5, 9.81);
    let inclined = GranularFroudeQuery::from_inclined(4.0, 0.5, 9.81, 0.0);
    let out = gpu.evaluate(&ctx, &[flat, inclined]);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].valid, 1u32);
    assert_eq!(out[1].valid, 1u32);
    assert!(
        close(out[0].froude, out[1].froude),
        "inclined zero angle should match flat: {} vs {}",
        out[0].froude,
        out[1].froude
    );
    assert_parity(&out[0], &flat, "inclined_zero_flat");
    assert_parity(&out[1], &inclined, "inclined_zero_incl");
}

#[test]
fn inclined_increases_froude() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGranularFroude::new(&ctx);
    // theta=1 rad -> cos~0.5403 -> g_eff smaller -> wave smaller -> Fr larger
    // than the equivalent flat query.
    let flat = GranularFroudeQuery::from_flow(2.0, 1.0, 9.81);
    let inclined = GranularFroudeQuery::from_inclined(2.0, 1.0, 9.81, 1.0);
    let out = gpu.evaluate(&ctx, &[flat, inclined]);
    assert_eq!(out[0].valid, 1u32);
    assert_eq!(out[1].valid, 1u32);
    assert!(
        out[1].froude > out[0].froude,
        "inclined Froude should exceed flat: {} vs {}",
        out[1].froude,
        out[0].froude
    );
    assert_parity(&out[0], &flat, "incl_flat");
    assert_parity(&out[1], &inclined, "incl_incl");
}

#[test]
fn degenerate_queries_are_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGranularFroude::new(&ctx);
    let queries = vec![
        // Negative speed.
        GranularFroudeQuery::from_flow(-1.0, 1.0, 9.81),
        // Non-positive depth.
        GranularFroudeQuery::from_flow(1.0, 0.0, 9.81),
        // Non-positive gravity.
        GranularFroudeQuery::from_flow(1.0, 1.0, -9.81),
        // NaN speed.
        GranularFroudeQuery::from_flow(f32::NAN, 1.0, 9.81),
        // Infinite depth.
        GranularFroudeQuery::from_flow(1.0, f32::INFINITY, 9.81),
        // Inclined past pi/2 -> cos(theta) < 0.
        GranularFroudeQuery::from_inclined(1.0, 1.0, 9.81, 2.0),
        // Inclined exactly pi/2 -> cos ~ 0 (not > 0).
        GranularFroudeQuery::from_inclined(1.0, 1.0, 9.81, std::f32::consts::FRAC_PI_2),
        // Inclined with NaN angle.
        GranularFroudeQuery::from_inclined(1.0, 1.0, 9.81, f32::NAN),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_eq!(res.valid, 0u32, "degenerate[{i}] should be invalid");
        assert_parity(res, q, &format!("degenerate[{i}]"));
    }
}

#[test]
fn batch_mixes_valid_and_invalid_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGranularFroude::new(&ctx);
    let queries = vec![
        GranularFroudeQuery::from_flow(1.0, 1.0, 9.81),
        GranularFroudeQuery::from_flow(-1.0, 1.0, 9.81),
        GranularFroudeQuery::from_inclined(10.0, 1.0, 9.81, 0.6),
        GranularFroudeQuery::from_inclined(1.0, 1.0, 9.81, 2.0),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    assert_eq!(out[0].valid, 1u32);
    assert_eq!(out[1].valid, 0u32);
    assert_eq!(out[2].valid, 1u32);
    assert_eq!(out[3].valid, 0u32);
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("batch[{i}]"));
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGranularFroude::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGranularFroude::new(&ctx);
    let mut lcg = Lcg::new(0x0F20_11D4);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        // Draw every parameter from a strictly positive band so the query is
        // always valid, split the two constructor paths, and reject any sample
        // whose Froude number lands in a narrow band around the critical knee
        // Fr = 1 so the regime decision cannot flip under round-off.
        let u = lcg.next_range(0.1, 12.0);
        let h = lcg.next_range(0.1, 4.0);
        let g = lcg.next_range(1.0, 20.0);
        let inclined = (lcg.next_u32() & 1) == 1;
        let (q, g_eff) = if inclined {
            // Keep theta well inside cos(theta) > 0 so validity is stable.
            let theta = lcg.next_range(-1.2, 1.2);
            let cos_theta = f64::from(theta).cos() as f32;
            (
                GranularFroudeQuery::from_inclined(u, h, g, theta),
                g * cos_theta,
            )
        } else {
            (GranularFroudeQuery::from_flow(u, h, g), g)
        };
        let froude = u / (g_eff * h).sqrt();
        if (froude - 1.0).abs() < 0.02 {
            continue;
        }
        queries.push(q);
    }
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_eq!(res.valid, 1u32, "sweep[{i}] should be valid");
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
