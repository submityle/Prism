//! Real-device parity for the `multigrid`-level twin:
//! [`GpuWaterMultigridLevel`](prism_volumetric_gpu::water_multigrid_level::GpuWaterMultigridLevel)
//! must reproduce the `CPU` golden
//! [`is_valid_level_size`](prism_render_architecture::water::pressure_multigrid::is_valid_level_size),
//! [`coarse_size`](prism_render_architecture::water::pressure_multigrid::coarse_size)
//! and
//! [`l2_norm`](prism_render_architecture::water::pressure_multigrid::l2_norm)
//! across valid and invalid level sizes, several residual fields, and a
//! randomized sweep.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The expected values are produced by calling the golden `is_valid_level_size`,
//! `coarse_size` and `l2_norm` directly, so the test pins `GPU == golden`, not
//! merely that the shader compiles. `coarse_size` requires `n >= 3`, so the
//! oracle guards it exactly as the kernel does.
//!
//! # Parity criterion
//!
//! `valid` and `coarse` are exact integer results, pinned with equality. `l2`
//! goes through one `sqrt`, so it is asserted within `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::pressure_multigrid`；无第三方引擎源码或衍生代码。

use prism_render_architecture::water::pressure_multigrid::{
    coarse_size, is_valid_level_size, l2_norm,
};
use prism_volumetric_gpu::water_multigrid_level::{
    GpuWaterMultigridLevel, WaterMultigridLevelQuery, WaterMultigridLevelResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound. A `GPU` square root may land a few units in the last
/// place from the scalar reference; `1e-4` admits that legal slack while still
/// failing a wrong port.
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

/// Reconstructs the golden result in-host by calling the reference functions
/// directly. `coarse_size` requires `n >= 3`, so it is guarded to `0` below
/// that, matching the kernel.
fn oracle(q: &WaterMultigridLevelQuery) -> WaterMultigridLevelResult {
    let valid = is_valid_level_size(q.n as usize);
    let coarse = if q.n >= 3 {
        coarse_size(q.n as usize) as u32
    } else {
        0
    };
    let l2 = l2_norm(&q.field[..q.field_count as usize]);
    WaterMultigridLevelResult { valid, coarse, l2 }
}

/// Pins one `GPU` result against the in-host oracle: exact validity and coarse
/// count, and the `L2` norm within tolerance.
fn check_query(idx: usize, got: &WaterMultigridLevelResult, want: &WaterMultigridLevelResult) {
    assert_eq!(
        got.valid, want.valid,
        "query {idx} valid: gpu {} vs cpu {}",
        got.valid, want.valid
    );
    assert_eq!(
        got.coarse, want.coarse,
        "query {idx} coarse: gpu {} vs cpu {}",
        got.coarse, want.coarse
    );
    assert!(
        close(got.l2, want.l2),
        "query {idx} l2: gpu {} vs cpu {}",
        got.l2,
        want.l2
    );
}

/// Dispatches `queries` and checks every result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuWaterMultigridLevel, queries: &[WaterMultigridLevelQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (idx, (q, g)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        check_query(idx, g, &want);
    }
}

/// A small `LCG` for the randomized sweep (host-only; the kernel is portable).
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

/// Maps a raw `u32` to an `f32` in `[lo, hi]` without any transcendental call.
fn uniform(bits: u32, lo: f32, hi: f32) -> f32 {
    let unit = (bits as f32) / (u32::MAX as f32);
    lo + unit * (hi - lo)
}

#[test]
fn empty_batch_produces_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterMultigridLevel::new(&ctx);
    // An empty batch short-circuits on the host (a storage buffer cannot be
    // zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn valid_level_sizes_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterMultigridLevel::new(&ctx);
    // Every valid vertex-centred level size 2^L + 1 up to 65.
    let queries: Vec<WaterMultigridLevelQuery> = [3u32, 5, 9, 17, 33, 65]
        .into_iter()
        .map(|n| WaterMultigridLevelQuery::new(n, &[]))
        .collect();
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got.len(), queries.len());
    for (idx, (q, g)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        assert!(want.valid, "size {} should be valid", q.n);
        check_query(idx, g, &want);
    }
}

#[test]
fn invalid_level_sizes_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterMultigridLevel::new(&ctx);
    // Counts below 3 and non-(2^L + 1) counts are all invalid.
    let queries: Vec<WaterMultigridLevelQuery> = [0u32, 1, 2, 4, 6, 7, 100]
        .into_iter()
        .map(|n| WaterMultigridLevelQuery::new(n, &[]))
        .collect();
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got.len(), queries.len());
    for (idx, (q, g)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        assert!(!want.valid, "size {} should be invalid", q.n);
        check_query(idx, g, &want);
    }
}

#[test]
fn l2_norm_fields_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterMultigridLevel::new(&ctx);
    // All-zero field, a single-element field, and a full 64-element field.
    let all_zero = [0.0f32; 64];
    let single = [2.5f32];
    let mut full = [0.0f32; 64];
    for (i, slot) in full.iter_mut().enumerate() {
        // A mild ramp so the sum of squares is a healthy, non-degenerate value.
        *slot = 0.5 + (i as f32) * 0.125;
    }
    let queries = [
        WaterMultigridLevelQuery::new(17, &all_zero),
        WaterMultigridLevelQuery::new(9, &single),
        WaterMultigridLevelQuery::new(33, &full),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterMultigridLevel::new(&ctx);
    let mut state = 0x2f6b_18d4_c7e0_519au64;
    let mut queries: Vec<WaterMultigridLevelQuery> = Vec::new();
    // Several workgroups' worth of random queries. Node counts span a mix of
    // valid 2^L + 1 sizes and arbitrary invalid ones; field lengths span the
    // whole fixed capacity and field values stay in a healthy magnitude band.
    let valid_sizes = [3u32, 5, 9, 17, 33, 65];
    while queries.len() < 512 {
        // Half the queries use a known valid size, the other half an arbitrary
        // count, so both the valid and invalid branches get heavy coverage.
        let pick = lcg(&mut state);
        let n = if pick & 1 == 0 {
            valid_sizes[(pick >> 1) as usize % valid_sizes.len()]
        } else {
            lcg(&mut state) % 130
        };
        let field_count = (lcg(&mut state) % 65) as usize;
        let mut field = [0.0f32; 64];
        for slot in field.iter_mut().take(field_count) {
            *slot = uniform(lcg(&mut state), -4.0, 4.0);
        }
        queries.push(WaterMultigridLevelQuery::new(n, &field[..field_count]));
    }
    check(&ctx, &gpu, &queries);
}
