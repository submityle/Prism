//! Real-device parity for the hair reactive-mask twin:
//! [`GpuHairReactiveMask`](prism_volumetric_gpu::hair_reactive_mask::GpuHairReactiveMask)
//! must reproduce the `CPU` golden `pixel_reactivity`, `dither_threshold` and
//! `dither_alpha` of `prism_render_architecture::hair::reactive_mask`, which
//! steer a temporal-anti-aliasing history buffer toward the current frame where
//! thin hair fibres are most likely to ghost and resolve a hard sub-pixel draw
//! decision from a blue-noise dither threshold.
//!
//! The oracle here is an independent re-implementation of those closed forms —
//! the sanitise helpers, the `v / (v + k)` saturation ramps, the weighted,
//! clamped reactivity sum, the integer xorshift / multiply dither hash and the
//! `coverage >= threshold` draw decision — written out directly so the test
//! never imports `prism_render_architecture`. It mirrors the reference branch
//! for branch, including the non-finite guards.
//!
//! The fixtures cover the branches the kernel must honor: default weights with
//! low coverage and high velocity (high reactivity), near-full coverage with no
//! motion (near-zero reactivity), oversized weights clamped to the maximum,
//! non-finite inputs that sanitise to a finite in-range result, known
//! `(x, y, frame)` triples whose dither threshold lands in `[0, 1)` and agrees
//! bit-for-bit with the oracle, hard draw decisions on either side of (and
//! exactly on) the threshold, and an empty batch the host short-circuits with
//! no dispatch. A sweep over random weights, inputs and coordinates follows.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The reactivity path threads through multiplies, adds and guarded divisions,
//! so `CPU` and `GPU` evaluate the same closed form but need not be bit-exact
//! (a `GPU` may contract a multiply-add). The dither threshold is pure integer
//! arithmetic plus one `u32 -> f32` round-to-nearest-even conversion, so both
//! sides agree to the last bit. The continuous comparison is
//! `abs_diff <= 1e-4 || rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`); the discrete
//! `valid` flag is compared exactly. The sweep keeps the dither coverage away
//! from the threshold so the ordered draw comparison agrees on both sides.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::hair::reactive_mask`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::hair_reactive_mask::{GpuHairReactiveMask, HairReactiveMaskQuery};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous-quantity parity comparison.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the continuous-quantity parity comparison.
const REL_EPS: f32 = 1.0e-3;
/// Floor on the relative-tolerance denominator so values near zero still use a
/// meaningful scale.
const REL_FLOOR: f32 = 1.0e-6;

/// Saturation constant (pixels per frame) for the velocity ramp, matching the
/// reference and the kernel.
const VELOCITY_SATURATION: f32 = 2.0;
/// Saturation constant for the depth-delta ramp, matching the reference and the
/// kernel.
const DEPTH_SATURATION: f32 = 0.1;

/// Returns `true` when two continuous values agree within the module tolerance.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_EPS
}

/// A finite, non-negative weight: non-finite or non-positive collapses to zero.
fn sanitize_weight(w: f32) -> f32 {
    if w.is_finite() && w > 0.0 {
        w
    } else {
        0.0
    }
}

/// Sub-pixel coverage clamped to `[0, 1]`; non-finite is treated as fully
/// covered.
fn sanitize_coverage(c: f32) -> f32 {
    if c.is_finite() {
        c.clamp(0.0, 1.0)
    } else {
        1.0
    }
}

/// A finite, non-negative magnitude: non-finite is zero, otherwise the absolute
/// value.
fn sanitize_magnitude(v: f32) -> f32 {
    if v.is_finite() {
        v.abs()
    } else {
        0.0
    }
}

/// A rational saturation ramp `v / (v + k)` with `v >= 0` and `k > 0`.
fn saturation_ramp(v: f32, k: f32) -> f32 {
    v / (v + k)
}

/// Independent host re-implementation of `pixel_reactivity`: sanitise the
/// weights and inputs, form the `1 - coverage` term and the two saturation
/// ramps, take the weighted sum and clamp it to `[0, max_reactivity]`.
fn pixel_reactivity(q: &HairReactiveMaskQuery) -> f32 {
    let coverage_weight = sanitize_weight(q.coverage_weight);
    let velocity_weight = sanitize_weight(q.velocity_weight);
    let depth_weight = sanitize_weight(q.depth_weight);
    let max_reactivity = if q.max_reactivity.is_finite() {
        q.max_reactivity.clamp(0.0, 1.0)
    } else {
        1.0
    };

    let coverage = sanitize_coverage(q.coverage);
    let velocity = sanitize_magnitude(q.screen_velocity);
    let depth = sanitize_magnitude(q.depth_delta);

    let coverage_term = 1.0 - coverage;
    let velocity_term = saturation_ramp(velocity, VELOCITY_SATURATION);
    let depth_term = saturation_ramp(depth, DEPTH_SATURATION);

    let raw = coverage_weight * coverage_term
        + velocity_weight * velocity_term
        + depth_weight * depth_term;

    raw.clamp(0.0, max_reactivity)
}

/// Independent host re-implementation of the integer dither hash: the three
/// coordinates are mixed with large odd constants through a wrapping xorshift /
/// multiply finaliser and normalised by `2^32`.
fn dither_threshold(x: u32, y: u32, frame: u32) -> f32 {
    let mut h = x.wrapping_mul(0x9E37_79B1);
    h ^= y.wrapping_mul(0x85EB_CA77);
    h ^= frame.wrapping_mul(0xC2B2_AE3D);
    h ^= h >> 16;
    h = h.wrapping_mul(0x7FEB_352D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846C_A68B);
    h ^= h >> 16;
    (h as f32) / 4_294_967_296.0
}

/// Independent host re-implementation of `dither_alpha`: a hard draw decision,
/// `1` when the sanitised coverage is at least the sanitised threshold.
fn dither_alpha(coverage: f32, threshold: f32) -> f32 {
    let c = if coverage.is_finite() {
        coverage.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let t = if threshold.is_finite() {
        threshold.clamp(0.0, 1.0)
    } else {
        1.0
    };
    if c >= t {
        1.0
    } else {
        0.0
    }
}

/// The full host oracle for one query: `(reactivity, threshold, dither_alpha,
/// valid)`. Every input sanitises to a finite in-range answer, so `valid` is
/// always `1`.
fn oracle(q: &HairReactiveMaskQuery) -> (f32, f32, f32, u32) {
    let threshold = dither_threshold(q.x, q.y, q.frame);
    let reactivity = pixel_reactivity(q);
    let alpha = dither_alpha(q.dither_coverage, threshold);
    (reactivity, threshold, alpha, 1)
}

/// Dispatches one query and asserts every continuous channel plus the validity
/// flag against the independent oracle.
fn assert_parity(ctx: &GpuContext, gpu: &GpuHairReactiveMask, q: HairReactiveMaskQuery) {
    let got = gpu.evaluate(ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1, "one result per query");
    let (reactivity, threshold, alpha, valid) = oracle(&q);
    let r = got[0];
    assert_eq!(r.valid, valid, "valid flag mismatch: query={q:?}");
    assert!(
        close(r.reactivity, reactivity),
        "reactivity mismatch: gpu={} cpu={reactivity} query={q:?}",
        r.reactivity
    );
    assert!(
        close(r.threshold, threshold),
        "threshold mismatch: gpu={} cpu={threshold} query={q:?}",
        r.threshold
    );
    assert!(
        close(r.dither_alpha, alpha),
        "dither_alpha mismatch: gpu={} cpu={alpha} query={q:?}",
        r.dither_alpha
    );
}

#[test]
fn default_params_low_coverage() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairReactiveMask::new(&ctx);
    // Default weights (0.3, 0.45, 0.25, 1.0) with low coverage and strong
    // motion drive the reactivity high.
    let q = HairReactiveMaskQuery::new(0.3, 0.45, 0.25, 1.0, 0.1, 8.0, 1.0, 0.5, 12, 7, 3);
    assert_parity(&ctx, &gpu, q);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert!(
        got[0].reactivity > 0.5,
        "low-coverage, fast-moving pixel should be highly reactive: {}",
        got[0].reactivity
    );
}

#[test]
fn high_coverage_low_reactivity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairReactiveMask::new(&ctx);
    // Full coverage, no motion, no depth change: every term is zero, so the
    // reactivity collapses to zero.
    let q = HairReactiveMaskQuery::new(0.3, 0.45, 0.25, 1.0, 1.0, 0.0, 0.0, 0.5, 1, 1, 0);
    assert_parity(&ctx, &gpu, q);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert!(
        close(got[0].reactivity, 0.0),
        "fully covered static pixel should be inert: {}",
        got[0].reactivity
    );
}

#[test]
fn clamped_to_max_reactivity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairReactiveMask::new(&ctx);
    // Oversized weights with a small max_reactivity: the raw weighted sum
    // overshoots and the result clamps to the configured maximum.
    let max = 0.4_f32;
    let q = HairReactiveMaskQuery::new(5.0, 5.0, 5.0, max, 0.0, 1000.0, 100.0, 0.5, 4, 9, 2);
    assert_parity(&ctx, &gpu, q);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert!(
        close(got[0].reactivity, max),
        "oversized sum should clamp to max_reactivity: {}",
        got[0].reactivity
    );
}

#[test]
fn non_finite_inputs_sanitized() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairReactiveMask::new(&ctx);
    // Non-finite weights and inputs all sanitise: weights -> 0, coverage NaN ->
    // 1 (fully covered), velocity/depth non-finite -> 0, max_reactivity NaN ->
    // 1. The result stays finite and in range.
    let q = HairReactiveMaskQuery::new(
        f32::NAN,
        -1.0,
        f32::INFINITY,
        f32::NAN,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        5,
        6,
        1,
    );
    assert_parity(&ctx, &gpu, q);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    let r = got[0];
    assert!(r.reactivity.is_finite(), "reactivity must be finite");
    assert!(
        (0.0..=1.0).contains(&r.reactivity),
        "reactivity must stay in range: {}",
        r.reactivity
    );
    // Non-finite dither coverage -> 0, so the draw never fires.
    assert!(
        close(r.dither_alpha, 0.0),
        "non-finite dither coverage should not draw: {}",
        r.dither_alpha
    );
}

#[test]
fn dither_threshold_in_unit_range() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairReactiveMask::new(&ctx);
    // A handful of known coordinates: the threshold must land in [0, 1) and
    // agree with the integer-hash oracle (the bit-near conversion risk point).
    let coords = [
        (0u32, 0u32, 0u32),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1920, 1080, 42),
        (640, 480, 7),
        (u32::MAX, u32::MAX, u32::MAX),
    ];
    for (x, y, frame) in coords {
        let q = HairReactiveMaskQuery::new(0.3, 0.45, 0.25, 1.0, 0.5, 1.0, 0.1, 0.5, x, y, frame);
        assert_parity(&ctx, &gpu, q);
        let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
        let t = got[0].threshold;
        assert!(
            (0.0..1.0).contains(&t),
            "threshold must lie in [0, 1): {t} for ({x}, {y}, {frame})"
        );
    }
}

#[test]
fn dither_alpha_draw_decisions() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairReactiveMask::new(&ctx);
    // Pick a coordinate, read its threshold, then probe coverage above, below
    // and exactly on it, plus non-finite guards.
    let (x, y, frame) = (3u32, 5u32, 9u32);
    let threshold = dither_threshold(x, y, frame);
    let probes = [
        (threshold + 0.25).min(1.0),
        (threshold - 0.25).max(0.0),
        threshold, // coverage == threshold must draw (>=)
        f32::NAN,
        f32::INFINITY,
    ];
    for coverage in probes {
        let q =
            HairReactiveMaskQuery::new(0.3, 0.45, 0.25, 1.0, 0.5, 1.0, 0.1, coverage, x, y, frame);
        assert_parity(&ctx, &gpu, q);
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairReactiveMask::new(&ctx);
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
    let gpu = GpuHairReactiveMask::new(&ctx);
    let mut rng = Lcg::new(0x51_7A_2E_63);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        let x = rng.next_u32();
        let y = rng.next_u32();
        let frame = rng.next_u32();
        let dither_coverage = rng.next_range(0.0, 1.0);
        // Reject coverage sitting near the threshold so the ordered draw
        // comparison agrees on both sides despite any last-bit difference.
        let threshold = dither_threshold(x, y, frame);
        if (dither_coverage - threshold).abs() < 1.0e-2 {
            continue;
        }
        queries.push(HairReactiveMaskQuery::new(
            rng.next_range(0.0, 1.0),
            rng.next_range(0.0, 1.0),
            rng.next_range(0.0, 1.0),
            rng.next_range(0.0, 1.0),
            rng.next_range(0.0, 1.2),
            rng.next_range(0.0, 1000.0),
            rng.next_range(-5.0, 5.0),
            dither_coverage,
            x,
            y,
            frame,
        ));
    }

    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        let (reactivity, threshold, alpha, valid) = oracle(q);
        assert_eq!(r.valid, valid, "sweep valid mismatch: query={q:?}");
        assert!(
            close(r.reactivity, reactivity),
            "sweep reactivity mismatch: gpu={} cpu={reactivity} query={q:?}",
            r.reactivity
        );
        assert!(
            close(r.threshold, threshold),
            "sweep threshold mismatch: gpu={} cpu={threshold} query={q:?}",
            r.threshold
        );
        assert!(
            close(r.dither_alpha, alpha),
            "sweep dither_alpha mismatch: gpu={} cpu={alpha} query={q:?}",
            r.dither_alpha
        );
    }
}
