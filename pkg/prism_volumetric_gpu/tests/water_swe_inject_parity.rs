//! Real-device parity for the shallow-water injection twin:
//! [`GpuWaterSweInject`](prism_volumetric_gpu::water_swe_inject::GpuWaterSweInject)
//! must reproduce the stateless outcome of the `CPU` goldens
//! [`inject_depth`](prism_render_architecture::water::swe::inject_depth) and
//! [`inject_velocity`](prism_render_architecture::water::swe::inject_velocity)
//! over the shared
//! [`SweConfig::index`](prism_render_architecture::water::swe::SweConfig::index)
//! — the range flag, the resolved index and the perturbed cell values — across
//! the in-range, out-of-bounds, index-valid-but-past-cell-count and randomized
//! cases compared query-for-query.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The reference functions are public, so they drive the oracle directly: each
//! query seeds a uniform [`SweState`](prism_render_architecture::water::swe::SweState)
//! of length `cell_count`, calls
//! [`inject_depth`](prism_render_architecture::water::swe::inject_depth) and
//! [`inject_velocity`](prism_render_architecture::water::swe::inject_velocity)
//! on it, and reads back the returned flag and the modified cell. A passing
//! `GPU == oracle` run is direct evidence the kernel computes the same
//! injection.
//!
//! # Parity criterion
//!
//! Both sides evaluate the identical integer range predicate and the identical
//! additive perturbation, so the range flag and the resolved index match
//! exactly and the perturbed values match to within floating-point tolerance
//! (`abs <= 1e-4` or `rel <= 1e-3`, with a `REL_FLOOR` of `1e-6`; being the same
//! add, they match exactly in practice). The one branch decision is a pure
//! integer comparison, so there is no floating-point crossing to flip.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::swe`；无第三方引擎源码或衍生代码。

use prism_render_architecture::water::swe::{inject_depth, inject_velocity, SweConfig, SweState};
use prism_volumetric_gpu::water_swe_inject::{
    GpuWaterSweInject, WaterSweInjectQuery, WaterSweInjectResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute closeness floor for the parity comparison.
const ABS: f32 = 1e-4;
/// Relative closeness bound for the parity comparison.
const REL: f32 = 1e-3;
/// Smallest denominator used in the relative comparison, guarding `0 == 0`.
const REL_FLOOR: f32 = 1e-6;

/// Absolute-or-relative closeness: `true` when `a` and `b` agree to within the
/// shared tolerance.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= ABS || rel <= REL
}

/// Computes the reference response for one query by seeding a uniform state and
/// calling both goldens directly.
fn oracle(q: &WaterSweInjectQuery) -> WaterSweInjectResult {
    let cfg = SweConfig {
        nx: q.nx,
        nz: q.nz,
        dx: 1.0,
        gravity: 9.81,
        damping: 0.0,
    };
    let cell_count = q.cell_count as usize;
    let mut state = SweState {
        h: vec![q.old_depth; cell_count],
        u: vec![q.old_u; cell_count],
        v: vec![q.old_v; cell_count],
    };
    let depth_hit = inject_depth(&mut state, cfg, q.x, q.z, q.delta_depth);
    let vel_hit = inject_velocity(&mut state, cfg, q.x, q.z, q.delta_u, q.delta_v);
    // The three vectors share `cell_count`, so both injections agree on the hit.
    assert_eq!(
        depth_hit, vel_hit,
        "depth and velocity injections must agree"
    );
    if depth_hit {
        let idx = cfg
            .index(q.x, q.z)
            .expect("a hit implies the index resolved");
        WaterSweInjectResult {
            in_range: 1,
            idx: idx as u32,
            new_depth: state.h[idx],
            new_u: state.u[idx],
            new_v: state.v[idx],
        }
    } else {
        WaterSweInjectResult {
            in_range: 0,
            idx: 0,
            new_depth: q.old_depth,
            new_u: q.old_u,
            new_v: q.old_v,
        }
    }
}

/// Pins one `GPU` result against the oracle: the flag and index exactly, the
/// perturbed values within tolerance.
fn check_result(idx: usize, got: &WaterSweInjectResult, want: &WaterSweInjectResult) {
    assert_eq!(
        got.in_range, want.in_range,
        "query {idx} in_range: gpu {} vs cpu {}",
        got.in_range, want.in_range
    );
    assert_eq!(
        got.idx, want.idx,
        "query {idx} idx: gpu {} vs cpu {}",
        got.idx, want.idx
    );
    assert!(
        close(got.new_depth, want.new_depth),
        "query {idx} new_depth: gpu {} vs cpu {}",
        got.new_depth,
        want.new_depth
    );
    assert!(
        close(got.new_u, want.new_u),
        "query {idx} new_u: gpu {} vs cpu {}",
        got.new_u,
        want.new_u
    );
    assert!(
        close(got.new_v, want.new_v),
        "query {idx} new_v: gpu {} vs cpu {}",
        got.new_v,
        want.new_v
    );
}

/// Dispatches every query and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuWaterSweInject, queries: &[WaterSweInjectQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        check_result(idx, result, &want);
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

/// Maps a raw `u32` into `[lo, hi)` as an `f32`, using only integer and
/// floating-point arithmetic (no transcendental), for the random sweep.
fn uniform(raw: u32, lo: f32, hi: f32) -> f32 {
    let unit = (raw as f32) / (u32::MAX as f32);
    lo + unit * (hi - lo)
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping water_swe_inject parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuWaterSweInject::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn in_range_injection_adds_deltas() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterSweInject::new(&ctx);
    // An in-bounds cell on a full 4x4 grid: depth and velocity both perturbed,
    // index z*nx+x = 2*4+1 = 9.
    let queries = [WaterSweInjectQuery::new(
        4, 4, 1, 2, 16, 0.5, -0.25, 0.75, 0.1, 0.2, -0.3,
    )];
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got[0].in_range, 1, "the cell is in range");
    assert_eq!(got[0].idx, 9, "row-major index z*nx+x = 9");
    assert!(close(got[0].new_depth, 0.6), "depth gains delta_depth");
    assert!(close(got[0].new_u, -0.05), "u gains delta_u");
    assert!(close(got[0].new_v, 0.45), "v gains delta_v");
    check(&ctx, &gpu, &queries);
}

#[test]
fn out_of_bounds_leaves_state_untouched() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterSweInject::new(&ctx);
    // x past nx and z past nz both miss; the state and index stay untouched.
    let queries = [
        WaterSweInjectQuery::new(4, 4, 4, 1, 16, 0.5, 0.1, 0.2, 1.0, 1.0, 1.0),
        WaterSweInjectQuery::new(4, 4, 1, 4, 16, 0.5, 0.1, 0.2, 1.0, 1.0, 1.0),
    ];
    let got = gpu.evaluate(&ctx, &queries);
    for r in &got {
        assert_eq!(r.in_range, 0, "an out-of-bounds cell misses");
        assert_eq!(r.idx, 0, "a miss reports index 0");
        assert!(close(r.new_depth, 0.5), "depth is untouched on a miss");
        assert!(close(r.new_u, 0.1), "u is untouched on a miss");
        assert!(close(r.new_v, 0.2), "v is untouched on a miss");
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn index_valid_but_past_cell_count_misses() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterSweInject::new(&ctx);
    // The coordinate is within nx/nz, so the row-major index resolves to
    // 3*4+3 = 15, but the backing vectors only hold 10 cells, so the shared
    // guard rejects it exactly as the goldens do.
    let queries = [WaterSweInjectQuery::new(
        4, 4, 3, 3, 10, 0.5, 0.1, 0.2, 1.0, 1.0, 1.0,
    )];
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(
        got[0].in_range, 0,
        "a resolved index past cell_count still misses"
    );
    assert_eq!(got[0].idx, 0, "a miss reports index 0");
    assert!(close(got[0].new_depth, 0.5), "depth is untouched");
    check(&ctx, &gpu, &queries);
}

#[test]
fn last_in_range_cell_hits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterSweInject::new(&ctx);
    // The final cell of a full 4x4 grid: x=3, z=3, index 15 < 16, a boundary hit.
    let queries = [WaterSweInjectQuery::new(
        4, 4, 3, 3, 16, 2.0, 0.0, 0.0, 0.5, 0.0, 0.0,
    )];
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got[0].in_range, 1, "the last cell is in range");
    assert_eq!(got[0].idx, 15, "the last cell index is 15");
    assert!(close(got[0].new_depth, 2.5), "depth gains the delta");
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterSweInject::new(&ctx);
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    let mut queries = Vec::new();
    // Several workgroups' worth of queries. Grid extents span 1..=16 and the
    // coordinates span 0..=18, so a healthy share of queries miss on the x/z
    // bounds; cell_count is either the full grid or a smaller value, so some
    // in-bounds coordinates resolve to an index past cell_count. All branches
    // are pure integer comparisons, so no reject-sampling is needed.
    while queries.len() < 400 {
        let nx = 1 + lcg(&mut state) % 16;
        let nz = 1 + lcg(&mut state) % 16;
        let x = lcg(&mut state) % 19;
        let z = lcg(&mut state) % 19;
        let full = nx * nz;
        // Half the queries clamp cell_count below the full grid to exercise the
        // index-past-cell_count branch.
        let cell_count = if lcg(&mut state).is_multiple_of(2) {
            full
        } else {
            1 + lcg(&mut state) % full
        };
        let old_depth = uniform(lcg(&mut state), -2.0, 5.0);
        let old_u = uniform(lcg(&mut state), -3.0, 3.0);
        let old_v = uniform(lcg(&mut state), -3.0, 3.0);
        let delta_depth = uniform(lcg(&mut state), -1.0, 1.0);
        let delta_u = uniform(lcg(&mut state), -1.0, 1.0);
        let delta_v = uniform(lcg(&mut state), -1.0, 1.0);
        queries.push(WaterSweInjectQuery::new(
            nx,
            nz,
            x,
            z,
            cell_count,
            old_depth,
            old_u,
            old_v,
            delta_depth,
            delta_u,
            delta_v,
        ));
    }
    check(&ctx, &gpu, &queries);
}
