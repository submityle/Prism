//! Real-device parity for the alpha-erosion (dissolve) twin:
//! [`GpuAlphaErosion`](prism_volumetric_gpu::alpha_erosion::GpuAlphaErosion)
//! must reproduce the `CPU` golden
//! [`alpha_erosion`](prism_render_architecture::particle::alpha_erosion) across
//! an interior dissolve (partial `alpha`, lit rim glow), a fully-eroded sample
//! (`alpha` `0`, no glow), a fully-opaque sample (`alpha` `1`, no glow), a
//! glow-band centre (peak rim weight), a hard-step cutoff (`edge_width` `0`,
//! both sides of the tie), the age `clamp` seams, a bit-exact hash sweep, and
//! randomized batches compared element-for-element.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The hashed `u32` word is an exact integer mix, so it is compared with `==`.
//! The continuous `noise01`, `alpha` and `glow_rgb` thread through multiplies,
//! adds and one guarded division, so `CPU` and `GPU` are not bit-exact: a `GPU`
//! may fuse a multiply-add the scalar reference leaves separate, perturbing the
//! low mantissa bits. Those fields therefore allow `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`.
//!
//! # Conditioning
//!
//! Every fixture is kept well away from the discrete cracks: the `smoothstep`
//! fixtures keep `edge_width` comfortably above `MIN_EDGE`, the random batch
//! rejects any `n` within a margin of the window endpoints (`threshold` and
//! `threshold + edge_width`), and the age stays off the `0`/`1` `clamp` seams,
//! so `CPU` and `GPU` stay on the same branch regardless of a few units in the
//! last place of slack. The dedicated hard-step fixture uses `edge_width` `0`
//! with `n` far from the `threshold` tie so both devices take the hard cutoff on
//! the same side.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::alpha_erosion`；
//! no third-party engine source or derived code.

use prism_render_architecture::particle::alpha_erosion::threshold_over_age;
use prism_volumetric_gpu::alpha_erosion::{
    golden, AlphaErosionQuery, AlphaErosionResult, GpuAlphaErosion,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound. A `GPU` may fuse a multiply-add the scalar reference
/// leaves separate, perturbing the low mantissa bits by a few units in the last
/// place; `1e-4` admits that legal slack while still failing a wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in
/// the last place exceed the absolute floor.
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

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears. Returns a value in `[0, 1)`.
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    (bits & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// A pseudo-random value in `[lo, hi)` drawn from `state`.
fn range(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + (hi - lo) * lcg(state)
}

/// A pseudo-random `u32` seed drawn from the high bits of `state`.
fn rand_seed(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 32) as u32
}

/// Builds a clearly-conditioned erosion query by rejection sampling: the
/// `edge_width` is kept well above `MIN_EDGE`, the age stays off the `clamp`
/// seams, and `n` is accepted only when it is a comfortable margin away from the
/// two window endpoints (`threshold` and `threshold + edge_width`), so neither
/// `smoothstep` endpoint nor the glow window edge is on a tie.
fn rand_query(state: &mut u64) -> AlphaErosionQuery {
    let edge_width = range(state, 0.15, 0.3);
    let start = range(state, 0.15, 0.35);
    let end = range(state, 0.4, 0.6);
    let age = range(state, 0.15, 0.85);
    let threshold = threshold_over_age(age, start, end);
    let n = loop {
        let candidate = lcg(state);
        let d_lo = (candidate - threshold).abs();
        let d_hi = (candidate - (threshold + edge_width)).abs();
        if d_lo > 0.03 && d_hi > 0.03 {
            break candidate;
        }
    };
    let glow_color = [
        range(state, 0.0, 1.0),
        range(state, 0.0, 1.0),
        range(state, 0.0, 1.0),
    ];
    let glow_intensity = range(state, 0.5, 3.0);
    AlphaErosionQuery::new(
        rand_seed(state),
        age,
        n,
        edge_width,
        glow_color,
        glow_intensity,
        start,
        end,
    )
}

/// Pins one `GPU` result against the `CPU` golden for `query`: the hashed word
/// must match bit-for-bit, and the dissolve `alpha`, the rim `glow_rgb` and the
/// `noise01` must agree within bound.
fn pin(idx: usize, query: &AlphaErosionQuery, got: &AlphaErosionResult) {
    let want = golden(query);
    assert_eq!(
        got.hashed, want.hashed,
        "query {idx} hashed: gpu {} vs cpu {}",
        got.hashed, want.hashed
    );
    assert!(
        close(got.noise01, want.noise01),
        "query {idx} noise01: gpu {} vs cpu {}",
        got.noise01,
        want.noise01
    );
    assert!(
        close(got.alpha, want.alpha),
        "query {idx} alpha: gpu {} vs cpu {}",
        got.alpha,
        want.alpha
    );
    for lane in 0..3 {
        assert!(
            close(got.glow_rgb[lane], want.glow_rgb[lane]),
            "query {idx} glow_rgb[{lane}]: gpu {} vs cpu {}",
            got.glow_rgb[lane],
            want.glow_rgb[lane]
        );
    }
}

/// Dispatches `queries` and pins every result element-for-element.
fn check(ctx: &GpuContext, gpu: &GpuAlphaErosion, queries: &[AlphaErosionQuery]) {
    let got = gpu.eval(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (query, result)) in queries.iter().zip(got.iter()).enumerate() {
        pin(idx, query, result);
    }
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAlphaErosion::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn interior_dissolve_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAlphaErosion::new(&ctx);
    // threshold = 0.2 + (0.6 - 0.2) * 0.5 = 0.4, window [0.4, 0.6]; n = 0.5 is
    // the interior midpoint (margin 0.1 from each endpoint), so alpha is a
    // genuine smoothstep value and the glow band is lit.
    let query = AlphaErosionQuery::new(7, 0.5, 0.5, 0.2, [0.4, 0.6, 0.8], 2.0, 0.2, 0.6);
    check(&ctx, &gpu, &[query]);
}

#[test]
fn fully_eroded_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAlphaErosion::new(&ctx);
    // n = 0.2 sits well below the window [0.4, 0.6], so the sample is fully
    // eroded (alpha 0) and the glow is zero.
    let query = AlphaErosionQuery::new(11, 0.5, 0.2, 0.2, [0.4, 0.6, 0.8], 2.0, 0.2, 0.6);
    check(&ctx, &gpu, &[query]);
}

#[test]
fn fully_opaque_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAlphaErosion::new(&ctx);
    // n = 0.85 sits well above the window [0.4, 0.6], so the sample is fully
    // opaque (alpha 1) and the glow is zero.
    let query = AlphaErosionQuery::new(13, 0.5, 0.85, 0.2, [0.4, 0.6, 0.8], 2.0, 0.2, 0.6);
    check(&ctx, &gpu, &[query]);
}

#[test]
fn glow_band_center_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAlphaErosion::new(&ctx);
    // Constant threshold 0.3 (start == end), window [0.3, 0.7]; n = 0.5 is the
    // window centre, so the rim-glow band weight peaks at 1.0 and scales the
    // colour by the intensity (0.4*3, 0.6*3, 0.8*3). Margin 0.2 from each
    // endpoint keeps the smoothstep off its ties.
    let query = AlphaErosionQuery::new(17, 0.5, 0.5, 0.4, [0.4, 0.6, 0.8], 3.0, 0.3, 0.3);
    check(&ctx, &gpu, &[query]);
}

#[test]
fn hard_step_cutoff_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAlphaErosion::new(&ctx);
    // edge_width 0 (< MIN_EDGE) collapses the soft edge to a hard cutoff at the
    // threshold (constant 0.5) and yields no glow. Both samples are far from the
    // n == threshold tie so the two devices take the same branch: n = 0.2 erodes
    // to 0, n = 0.8 stays opaque at 1.
    let eroded = AlphaErosionQuery::new(23, 0.5, 0.2, 0.0, [0.5, 0.5, 0.5], 2.0, 0.5, 0.5);
    let opaque = AlphaErosionQuery::new(29, 0.5, 0.8, 0.0, [0.5, 0.5, 0.5], 2.0, 0.5, 0.5);
    check(&ctx, &gpu, &[eroded, opaque]);
}

#[test]
fn age_clamp_endpoints_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAlphaErosion::new(&ctx);
    // Out-of-range ages clamp: age -0.5 -> threshold = start = 0.25 (window
    // [0.25, 0.45], n = 0.05 fully eroded); age 1.5 -> threshold = end = 0.65
    // (window [0.65, 0.85], n = 0.98 fully opaque). Both n values stay a wide
    // margin from the window endpoints, and the clamp value itself is continuous.
    let birth = AlphaErosionQuery::new(31, -0.5, 0.05, 0.2, [0.3, 0.5, 0.7], 1.5, 0.25, 0.65);
    let death = AlphaErosionQuery::new(37, 1.5, 0.98, 0.2, [0.3, 0.5, 0.7], 1.5, 0.25, 0.65);
    check(&ctx, &gpu, &[birth, death]);
}

#[test]
fn hash_sweep_is_bit_exact_and_normalized() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAlphaErosion::new(&ctx);
    // Sweep many seeds through benign erosion parameters: the hashed word must be
    // bit-exact and noise01 must match the normalized reference, consistent with
    // hashed / u32::MAX.
    let queries: Vec<AlphaErosionQuery> = (0..128u32)
        .map(|seed| AlphaErosionQuery::new(seed, 0.5, 0.5, 0.2, [0.5, 0.5, 0.5], 1.0, 0.2, 0.6))
        .collect();
    let got = gpu.eval(&ctx, &queries);
    assert_eq!(got.len(), queries.len(), "result count must match");
    for (idx, (query, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = golden(query);
        assert_eq!(
            result.hashed, want.hashed,
            "seed {idx} hashed: gpu {} vs cpu {}",
            result.hashed, want.hashed
        );
        assert!(
            close(result.noise01, want.noise01),
            "seed {idx} noise01: gpu {} vs cpu {}",
            result.noise01,
            want.noise01
        );
        // Anchor the host replication to the public golden: the normalized word
        // reproduces hash_noise01.
        let normalized = want.hashed as f32 / u32::MAX as f32;
        assert!(
            close(normalized, want.noise01),
            "seed {idx} host hash is inconsistent with hash_noise01"
        );
    }
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAlphaErosion::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // One batch mixing deterministic fixtures with many random queries,
    // dispatched together so the per-thread indexing and the contiguous storage
    // layout are both exercised, then pinned element-for-element.
    let mut queries = vec![
        AlphaErosionQuery::new(7, 0.5, 0.5, 0.2, [0.4, 0.6, 0.8], 2.0, 0.2, 0.6),
        AlphaErosionQuery::new(23, 0.5, 0.2, 0.0, [0.5, 0.5, 0.5], 2.0, 0.5, 0.5),
        AlphaErosionQuery::new(17, 0.5, 0.5, 0.4, [0.4, 0.6, 0.8], 3.0, 0.3, 0.3),
    ];
    for _ in 0..48 {
        queries.push(rand_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn many_queries_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAlphaErosion::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep (several workgroups' worth) pins the hash and the dissolve
    // across many random erosion configurations.
    let queries: Vec<AlphaErosionQuery> = (0..200).map(|_| rand_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
