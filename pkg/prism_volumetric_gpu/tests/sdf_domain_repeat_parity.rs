//! Real-device parity for the signed-distance *domain-operator* twin:
//! [`GpuSdfDomainRepeat`](prism_volumetric_gpu::sdf_domain_repeat::GpuSdfDomainRepeat)
//! must reproduce the `CPU` closed forms of
//! `prism_render_architecture::ray_scene::sdf_domain` — the finite-lattice
//! `limited_repeat`, the mirrored-lattice `mirror_repeat`, the `elongate`
//! displacement and the `elongate_correction` interior scalar — across
//! interior, exterior and saturated points plus a randomized sweep compared
//! query-for-query.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! This wave forbids depending on the golden crate, so the host oracle is an
//! *independent* reimplementation of the same four closed forms (the clamped
//! lattice fold for `limited_repeat`, the odd-cell-reflecting fold for
//! `mirror_repeat`, the clamped-core subtraction for `elongate`, and the
//! non-positive box distance for `elongate_correction`). Because the reference
//! and this oracle are both scalar `f32`, a `GPU == oracle` pass is direct
//! evidence the ported kernel computes the same transforms the reference does.
//!
//! # Parity criterion
//!
//! The lattice folds thread through a divide (`point / period`) and a
//! round-to-nearest-cell, so a `GPU` component may land a few units in the last
//! place from the scalar oracle; each is asserted within `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`, with a `rel_diff` floor of `1e-6` so a near-zero expected
//! component does not inflate the relative error.
//!
//! # Conditioning
//!
//! The cell index is discontinuous at the half-period boundaries
//! `|p / period| = n + 0.5`, where a last-place difference could pick a
//! neighbouring cell (and, for `mirror_repeat`, flip the reflection parity). A
//! last-place disagreement there could pick a different branch, so named
//! fixtures stay a safe margin from those boundaries and the randomized sweep
//! reject-samples every point until every axis is well clear of a half-period
//! boundary. The `elongate` displacement and its correction are `C0`
//! continuous, so they need no rejection.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::sdf_domain`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::sdf_domain_repeat::{
    GpuSdfDomainRepeat, SdfDomainRepeatQuery, SdfDomainRepeatResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute bound on each transformed component. A `GPU` divide may land a few
/// units in the last place from the scalar oracle; `1e-4` admits that legal
/// slack while still failing a wrong port.
const DIST_ABS: f32 = 1.0e-4;

/// Relative bound on each transformed component, applied for larger magnitudes
/// where a few units in the last place exceed the absolute floor.
const DIST_REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected component
/// does not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Shared per-axis lattice period, driving both `limited_repeat` and
/// `mirror_repeat`.
const PERIOD: [f32; 3] = [2.0, 1.5, 3.0];
/// Shared per-axis finite-lattice instance limit for `limited_repeat`.
const LIMIT: [f32; 3] = [2.0, 1.0, 3.0];
/// Shared per-axis elongation half-extent for `elongate` and
/// `elongate_correction`.
const HALF_EXTENT: [f32; 3] = [0.5, 0.4, 0.3];

/// Smallest positive normal `f32`, matching the reference guard that leaves a
/// zero- (or sub-normal-) period axis unchanged.
const MIN_POSITIVE: f32 = f32::MIN_POSITIVE;

/// Returns whether `a` and `b` agree within the given absolute or relative
/// bound (relative error floored at `REL_FLOOR`).
fn close(a: f32, b: f32, abs_eps: f32, rel_eps: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= abs_eps || rel <= rel_eps
}

/// Per-axis finite lattice fold:
/// `p - period * clamp(round(p / period), -limit, limit)`, leaving a zero- (or
/// sub-normal-) period axis unchanged.
fn limited_repeat_axis(p: f32, period: f32, limit: f32) -> f32 {
    if period.abs() > MIN_POSITIVE {
        let cell = (p / period).round().clamp(-limit, limit);
        return p - period * cell;
    }
    p
}

/// Independent reimplementation of the reference `limited_repeat`: fold each
/// axis into the finite lattice `p - period * clamp(round(p / period), -limit,
/// limit)`, leaving a zero- (or sub-normal-) period axis unchanged.
fn limited_repeat_oracle(point: [f32; 3], period: [f32; 3], limit: [f32; 3]) -> [f32; 3] {
    [
        limited_repeat_axis(point[0], period[0], limit[0]),
        limited_repeat_axis(point[1], period[1], limit[1]),
        limited_repeat_axis(point[2], period[2], limit[2]),
    ]
}

/// Per-axis mirrored lattice fold: fold to the cell-local offset and negate the
/// odd cells, leaving a zero- (or sub-normal-) period axis unchanged.
fn mirror_repeat_axis(p: f32, period: f32) -> f32 {
    if period.abs() > MIN_POSITIVE {
        let cell = (p / period).round();
        let mut local = p - period * cell;
        if (cell % 2.0).abs() > 0.5 {
            local = -local;
        }
        return local;
    }
    p
}

/// Independent reimplementation of the reference `mirror_repeat`: fold each
/// axis to its cell-local offset `p - period * round(p / period)` and negate
/// the odd cells, leaving a zero- (or sub-normal-) period axis unchanged.
fn mirror_repeat_oracle(point: [f32; 3], period: [f32; 3]) -> [f32; 3] {
    [
        mirror_repeat_axis(point[0], period[0]),
        mirror_repeat_axis(point[1], period[1]),
        mirror_repeat_axis(point[2], period[2]),
    ]
}

/// Independent reimplementation of the reference `elongate`: subtract the
/// clamped `[-h, h]` core from each axis.
fn elongate_oracle(point: [f32; 3], half_extent: [f32; 3]) -> [f32; 3] {
    [
        point[0] - point[0].clamp(-half_extent[0], half_extent[0]),
        point[1] - point[1].clamp(-half_extent[1], half_extent[1]),
        point[2] - point[2].clamp(-half_extent[2], half_extent[2]),
    ]
}

/// Independent reimplementation of the reference `elongate_correction`: the
/// non-positive signed distance into the inserted elongation box,
/// `min(max(|p.x| - h.x, |p.y| - h.y, |p.z| - h.z), 0)`.
fn elongate_correction_oracle(point: [f32; 3], half_extent: [f32; 3]) -> f32 {
    let qx = point[0].abs() - half_extent[0];
    let qy = point[1].abs() - half_extent[1];
    let qz = point[2].abs() - half_extent[2];
    qx.max(qy).max(qz).min(0.0)
}

/// Computes the expected result from the independent host oracle, the faithful
/// reference the `GPU` is pinned against.
fn oracle(q: &SdfDomainRepeatQuery) -> SdfDomainRepeatResult {
    SdfDomainRepeatResult {
        limited_repeat: limited_repeat_oracle(q.point, q.period, q.limit),
        mirror_repeat: mirror_repeat_oracle(q.point, q.period),
        elongate: elongate_oracle(q.point, q.half_extent),
        elongate_correction: elongate_correction_oracle(q.point, q.half_extent),
    }
}

/// Pins the three components of one transformed point against the oracle.
fn check_vec(idx: usize, name: &str, got: [f32; 3], want: [f32; 3]) {
    for (axis, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            close(*g, *w, DIST_ABS, DIST_REL),
            "query {idx} {name} axis {axis}: gpu {g} vs cpu {w}"
        );
    }
}

/// Pins one `GPU` result against the host oracle under the tolerance.
fn check_one(idx: usize, got: &SdfDomainRepeatResult, want: &SdfDomainRepeatResult) {
    check_vec(
        idx,
        "limited_repeat",
        got.limited_repeat,
        want.limited_repeat,
    );
    check_vec(idx, "mirror_repeat", got.mirror_repeat, want.mirror_repeat);
    check_vec(idx, "elongate", got.elongate, want.elongate);
    assert!(
        close(
            got.elongate_correction,
            want.elongate_correction,
            DIST_ABS,
            DIST_REL
        ),
        "query {idx} elongate_correction: gpu {} vs cpu {}",
        got.elongate_correction,
        want.elongate_correction
    );
}

/// Dispatches every query and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuSdfDomainRepeat, queries: &[SdfDomainRepeatQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        check_one(idx, result, &want);
    }
}

/// A tiny integer linear-congruential generator; only integer work, so no
/// transcendental appears. Returns the raw high bits as a `u32`.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

/// Draws a `f32` in `[lo, hi)` from the generator, using only integer-to-float
/// division (no transcendental).
fn uniform(state: &mut u64, lo: f32, hi: f32) -> f32 {
    let u = lcg(state) as f32 * (1.0 / 4_294_967_296.0);
    lo + (hi - lo) * u
}

/// Builds a query carrying the point plus the shared, well-conditioned lattice
/// and elongation parameters.
fn make_query(point: [f32; 3]) -> SdfDomainRepeatQuery {
    SdfDomainRepeatQuery::new(point, PERIOD, LIMIT, HALF_EXTENT)
}

/// Whether `point` is a safe margin from every axis's half-period boundary, so
/// a last-place `GPU` difference never folds into a different cell than the
/// oracle. The `elongate` displacement and correction are `C0` continuous, so
/// only the lattice folds constrain the sampling.
fn conditioned(point: [f32; 3]) -> bool {
    for (p, period) in point.iter().zip(PERIOD.iter()) {
        let t = p / period;
        let frac = t - t.round();
        if (0.5 - frac.abs()) <= 0.05 {
            return false;
        }
    }
    true
}

/// Draws one well-conditioned random point in `[-6, 6]^3`, reject-sampling
/// until every axis clears its half-period boundary.
fn random_query(state: &mut u64) -> SdfDomainRepeatQuery {
    for _ in 0..4096 {
        let point = [
            uniform(state, -6.0, 6.0),
            uniform(state, -6.0, 6.0),
            uniform(state, -6.0, 6.0),
        ];
        if conditioned(point) {
            return make_query(point);
        }
    }
    // Fallback (practically never reached): a known well-conditioned point.
    make_query([0.1, 0.1, 0.1])
}

/// A fixed battery of named points, each a safe margin from every half-period
/// boundary, spanning interior cells, saturated finite-lattice bands and both
/// signs so all four operators are exercised.
fn fixture_queries() -> Vec<SdfDomainRepeatQuery> {
    vec![
        // Origin cell, deep interior of the elongation box.
        make_query([0.1, 0.1, 0.1]),
        // First positive cell on every axis.
        make_query([2.2, 1.6, 3.3]),
        // `x` saturated past the finite limit; `t = 4.3`, clamped to cell 2.
        make_query([8.6, 0.2, 0.2]),
        // First negative cell on every axis.
        make_query([-2.2, -1.6, -0.2]),
        // `x` saturated on the negative side; `t = -4.3`, clamped to cell -2.
        make_query([-8.6, 1.7, -3.4]),
        // Mixed cell exercising the `x` limit edge and larger `z` cell.
        make_query([4.3, 0.3, 6.4]),
    ]
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping sdf_domain_repeat parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuSdfDomainRepeat::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn limited_repeat_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfDomainRepeat::new(&ctx);
    // Interior cell, and a saturated point past the finite instance limit.
    let interior = make_query([2.2, 1.6, 3.3]);
    let saturated = make_query([8.6, 0.2, 0.2]);
    let got = gpu.evaluate(&ctx, &[interior, saturated]);
    assert_eq!(got.len(), 2);
    check_one(0, &got[0], &oracle(&interior));
    check_one(1, &got[1], &oracle(&saturated));
}

#[test]
fn mirror_repeat_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfDomainRepeat::new(&ctx);
    // Even cell (no reflection) on `x` and odd cell (reflection) on `x`.
    let even_cell = make_query([0.1, 0.1, 0.1]);
    let odd_cell = make_query([2.2, 1.6, 3.3]);
    let got = gpu.evaluate(&ctx, &[even_cell, odd_cell]);
    assert_eq!(got.len(), 2);
    check_one(0, &got[0], &oracle(&even_cell));
    check_one(1, &got[1], &oracle(&odd_cell));
}

#[test]
fn elongate_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfDomainRepeat::new(&ctx);
    // A point inside the elongation box (displacement collapses to near zero)
    // and a point well outside it (displacement grows with the point).
    let inside = make_query([0.1, 0.1, 0.1]);
    let outside = make_query([2.2, 1.6, 3.3]);
    let got = gpu.evaluate(&ctx, &[inside, outside]);
    assert_eq!(got.len(), 2);
    check_one(0, &got[0], &oracle(&inside));
    check_one(1, &got[1], &oracle(&outside));
}

#[test]
fn elongate_correction_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfDomainRepeat::new(&ctx);
    // Interior point (non-positive correction) and exterior point (zero).
    let interior = make_query([0.1, 0.1, 0.1]);
    let exterior = make_query([2.2, 1.6, 3.3]);
    let got = gpu.evaluate(&ctx, &[interior, exterior]);
    assert_eq!(got.len(), 2);
    check_one(0, &got[0], &oracle(&interior));
    check_one(1, &got[1], &oracle(&exterior));
    // Spot-check the documented sign of the correction either side of the box.
    assert!(
        got[0].elongate_correction <= 0.0,
        "an interior point yields a non-positive correction"
    );
    assert!(
        close(got[1].elongate_correction, 0.0, DIST_ABS, DIST_REL),
        "an exterior point yields a zero correction"
    );
}

#[test]
fn fixture_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfDomainRepeat::new(&ctx);
    check(&ctx, &gpu, &fixture_queries());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfDomainRepeat::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    let mut queries = fixture_queries();
    // Several workgroups' worth of random, well-conditioned points pin all four
    // domain operators across a wide span of query positions.
    for _ in 0..512 {
        queries.push(random_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}
