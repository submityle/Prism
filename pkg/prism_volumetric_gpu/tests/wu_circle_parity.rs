//! Real-device parity for the Xiaolin Wu anti-aliased circle per-column twin:
//! [`GpuWuCircle`](prism_volumetric_gpu::wu_circle::GpuWuCircle) must reproduce
//! the numeric core of the `CPU` golden
//! [`wu_circle`](prism_render_architecture::particle::wu_circle) — the
//! sub-pixel coverage split of one first-octant integer column — across a sweep
//! of radii and columns, a within/past-diagonal boundary column, a mixed batch,
//! and a randomized sweep compared column-for-column.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The golden's per-column helpers (`ipart`, `fpart`, the column walk) are
//! private, and its only public entry point
//! [`rasterize`](prism_render_architecture::particle::wu_circle::rasterize) is a
//! variable-length, symmetry-mirrored, de-duplicated, sorted aggregate. The
//! expected per-column values are therefore reconstructed in-host from the
//! golden's exact closed form (`y = sqrt(r^2 - x^2)` guarded to zero,
//! `y_lo = floor(y)`, `frac = y - floor(y)`, `cov_lo = 1 - frac`,
//! `cov_hi = frac`). `grounding_oracle_matches_rasterize_emission` separately
//! proves that closed form is faithful by locating the very same coverage values
//! inside the public `rasterize` output for uniquely-emitted interior columns,
//! so a `GPU == oracle` pass transitively establishes `GPU == golden`.
//!
//! # Parity criterion
//!
//! The column `x`, the lower row `y_lo` and the `within_diagonal` flag follow
//! `floor`-based rounding and magnitude comparisons, so they agree exactly for
//! the chosen fixtures and are asserted with `==`. The coverages `cov_lo` and
//! `cov_hi` thread through a subtract and a `sqrt`, so a `GPU` `sqrt` may land a
//! few units in the last place from the scalar reference; they are asserted
//! within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`.
//!
//! # Conditioning
//!
//! Every fixture is deliberately away from a discrete tie. A column whose ring
//! height `y` is near an integer is rejected (its `floor` could straddle the
//! integer differently once a `GPU` `sqrt` and a `CPU` `sqrt` disagree by a unit
//! in the last place, and its `frac` could cross the drop threshold), and a
//! column near the exact `2 * x^2 == r^2` diagonal is rejected (its
//! `within_diagonal` flag could flip). The surviving columns keep both the ring
//! height and the diagonal margin far beyond the `f32` slack, so `CPU` and `GPU`
//! stay on the same side of every discrete decision.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::wu_circle`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::wu_circle::rasterize;
use prism_volumetric_gpu::wu_circle::{GpuWuCircle, WuCircleQuery, WuCircleResult};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on a coverage. A `GPU` `sqrt` may land a few units in
/// the last place from the scalar reference; `1e-4` admits that legal slack
/// while still failing a wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in the
/// last place exceed the absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// Reconstructs the golden's exact per-column closed form in-host: the faithful
/// oracle the `GPU` is pinned against. Uses only `sqrt` and `floor`, never an
/// `f32` `==`, and no transcendental method, matching the house rules.
fn oracle(r: f32, column: u32) -> WuCircleResult {
    let xf = column as f32;
    let r_sq = r * r;
    let within_diagonal = 2.0 * xf * xf <= r_sq;
    let inside = r_sq - xf * xf;
    let y = if inside <= 0.0 { 0.0 } else { inside.sqrt() };
    let y_lo = y.floor();
    let frac = y - y.floor();
    WuCircleResult {
        x: column as i32,
        y_lo: y_lo as i32,
        cov_lo: 1.0 - frac,
        cov_hi: frac,
        within_diagonal,
    }
}

/// Pins one `GPU` column result against the in-host oracle: coordinates and the
/// flag exactly, both coverages within tolerance.
fn check_column(idx: usize, got: &WuCircleResult, want: &WuCircleResult) {
    assert_eq!(
        got.x, want.x,
        "column {idx} x: gpu {} vs cpu {}",
        got.x, want.x
    );
    assert_eq!(
        got.y_lo, want.y_lo,
        "column {idx} y_lo: gpu {} vs cpu {}",
        got.y_lo, want.y_lo
    );
    assert_eq!(
        got.within_diagonal, want.within_diagonal,
        "column {idx} within_diagonal: gpu {} vs cpu {}",
        got.within_diagonal, want.within_diagonal
    );
    assert!(
        close(got.cov_lo, want.cov_lo),
        "column {idx} cov_lo: gpu {} vs cpu {}",
        got.cov_lo,
        want.cov_lo
    );
    assert!(
        close(got.cov_hi, want.cov_hi),
        "column {idx} cov_hi: gpu {} vs cpu {}",
        got.cov_hi,
        want.cov_hi
    );
}

/// Dispatches every column and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuWuCircle, queries: &[WuCircleQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q.radius, q.column);
        check_column(idx, result, &want);
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

/// Draws a radius in `[2.0, 60.0]` at milli resolution from `state`.
fn radius(state: &mut u64) -> f32 {
    2.0 + (lcg(state) % 58_000) as f32 / 1000.0
}

/// Whether a column is clear of both discrete ties: its ring height `y` is far
/// from an integer (so the `floor` and the drop threshold are stable) and it is
/// far from the exact `2 * x^2 == r^2` diagonal (so the `within_diagonal` flag
/// is stable). Uses only `sqrt`/`floor` and never an `f32` `==`.
fn well_conditioned(r: f32, column: u32) -> bool {
    let xf = column as f32;
    let r_sq = r * r;
    // Diagonal margin: keep `2 x^2` a full unit clear of `r^2`.
    if (r_sq - 2.0 * xf * xf).abs() < 1.0 {
        return false;
    }
    let inside = r_sq - xf * xf;
    let y = if inside <= 0.0 { 0.0 } else { inside.sqrt() };
    let frac = y - y.floor();
    // Ring-height margin: keep `frac` away from 0 and 1.
    (0.08..=0.92).contains(&frac)
}

/// Enumerates every well-conditioned first-octant column of radius `r`.
fn columns_for_radius(r: f32) -> Vec<WuCircleQuery> {
    let mut out = Vec::new();
    let r_sq = r * r;
    let mut x: u32 = 0;
    loop {
        let xf = x as f32;
        if 2.0 * xf * xf > r_sq {
            break;
        }
        if well_conditioned(r, x) {
            out.push(WuCircleQuery::new(r, x));
        }
        x += 1;
    }
    out
}

/// The deterministic radius fixtures, each a non-integer radius whose columns
/// span the first octant.
const FIXTURE_RADII: [f32; 6] = [3.7, 8.3, 12.35, 27.6, 41.17, 59.42];

/// Builds the deterministic column fixtures: every well-conditioned column of
/// each fixture radius.
fn fixture_columns() -> Vec<WuCircleQuery> {
    let mut out = Vec::new();
    for &r in &FIXTURE_RADII {
        out.extend(columns_for_radius(r));
    }
    out
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping wu_circle parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuWuCircle::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn small_radius_columns_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWuCircle::new(&ctx);
    check(&ctx, &gpu, &columns_for_radius(3.7));
}

#[test]
fn mid_radius_columns_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWuCircle::new(&ctx);
    check(&ctx, &gpu, &columns_for_radius(27.6));
}

#[test]
fn large_radius_columns_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWuCircle::new(&ctx);
    check(&ctx, &gpu, &columns_for_radius(59.42));
}

#[test]
fn axis_column_zero_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWuCircle::new(&ctx);
    // The x = 0 column sits on the axis: y = r, so a non-integer radius keeps it
    // well away from a ring-height tie.
    check(&ctx, &gpu, &[WuCircleQuery::new(12.35, 0)]);
}

#[test]
fn past_diagonal_column_reports_outside() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWuCircle::new(&ctx);
    // A column well past the 45-degree diagonal (2 x^2 >> r^2): the radicand is
    // negative, guarded to zero, so y = 0, y_lo = 0, cov_lo = 1, cov_hi = 0, and
    // the within flag is false.
    let r = 10.37;
    let column = 20;
    let got = gpu.evaluate(&ctx, &[WuCircleQuery::new(r, column)]);
    assert_eq!(got.len(), 1);
    let want = oracle(r, column);
    assert!(!want.within_diagonal, "fixture must be past the diagonal");
    check_column(0, &got[0], &want);
}

#[test]
fn mixed_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWuCircle::new(&ctx);
    // Every fixture radius's columns dispatched together so the per-thread
    // indexing and the contiguous output slots are both exercised.
    check(&ctx, &gpu, &fixture_columns());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWuCircle::new(&ctx);
    let mut state = 0x0f1e_2d3c_4b5a_6978_u64;
    let mut queries = fixture_columns();
    // Many random radii (several workgroups' worth of columns) pin every
    // reported column across a wide span of ring sizes.
    for _ in 0..64 {
        let r = radius(&mut state);
        queries.extend(columns_for_radius(r));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn grounding_oracle_matches_rasterize_emission() {
    // Prove the in-host oracle is faithful to the public golden: for an interior
    // column that is emitted into a unique lattice cell (no symmetry collision,
    // no near-zero drop), the oracle's cov_lo and cov_hi must appear verbatim in
    // the `rasterize` output at (x, y_lo) and (x, y_lo + 1). This grounds the
    // transitive GPU == oracle == golden argument.
    let mut state = 0x1234_5678_9abc_def0_u64;
    let mut checked = 0u32;
    for _ in 0..4000 {
        if checked >= 24 {
            break;
        }
        let r = radius(&mut state);
        let r_sq = r * r;
        // Pick an interior column 1 <= x with a comfortable diagonal margin.
        let x = 1 + (lcg(&mut state) % 10);
        let xf = x as f32;
        if 2.0 * xf * xf > r_sq - 2.0 {
            continue;
        }
        if !well_conditioned(r, x) {
            continue;
        }
        let o = oracle(r, x);
        // Uniqueness: the lower cell (x, y_lo) can only collide with a swapped
        // reflection of column y_lo, which is only walked when 2 y_lo^2 <= r^2.
        // Requiring 2 y_lo^2 > r^2 rules that out, so (x, y_lo) and (x, y_lo + 1)
        // are each emitted by exactly this column.
        let ylo = o.y_lo as f32;
        if 2.0 * ylo * ylo <= r_sq {
            continue;
        }
        let samples = rasterize(0, 0, r);
        let lower = samples
            .iter()
            .find(|&&(sx, sy, _)| sx == o.x && sy == o.y_lo)
            .map(|&(_, _, c)| c)
            .expect("lower-row cell must be present in the rasterize output");
        let upper = samples
            .iter()
            .find(|&&(sx, sy, _)| sx == o.x && sy == o.y_lo + 1)
            .map(|&(_, _, c)| c)
            .expect("upper-row cell must be present in the rasterize output");
        assert!(
            close(lower, o.cov_lo),
            "oracle cov_lo {} vs rasterize {} at r = {r}, x = {x}",
            o.cov_lo,
            lower
        );
        assert!(
            close(upper, o.cov_hi),
            "oracle cov_hi {} vs rasterize {} at r = {r}, x = {x}",
            o.cov_hi,
            upper
        );
        checked += 1;
    }
    assert!(
        checked >= 24,
        "expected at least 24 grounding columns, got {checked}"
    );
}
