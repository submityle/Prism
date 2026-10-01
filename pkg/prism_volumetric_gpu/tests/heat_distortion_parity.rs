//! Real-device parity for the heat-distortion twin:
//! [`GpuHeatDistortion`](prism_volumetric_gpu::heat_distortion::GpuHeatDistortion)
//! must reproduce the `CPU` golden
//! [`heat_distortion`](prism_render_architecture::particle::heat_distortion)
//! across a zero-normal sample (no base offset), a strength that saturates the
//! magnitude clamp, a sample whose distorted `UV` stays well inside the unit
//! square, a decorrelated rolling-shimmer sample, and randomized batches
//! compared element-for-element.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The integer hash value-noise is reproduced bit for bit (`WGSL` unsigned
//! integers wrap exactly like Rust's `wrapping_mul` / `^` / `>>`), so the
//! rolling shimmer agrees to the last mantissa bit. The surrounding continuous
//! algebra (the base offset, the rational falloff, the clamps) is a fixed,
//! non-reorderable sequence of multiplies, adds and divides; a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The comparison therefore
//! allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` on every `f32` field.
//!
//! # Conditioning
//!
//! Every fixture is kept well away from the one discontinuous crack: the
//! value-noise floor split. The random batch rejects any sample whose rolling
//! coordinate `t = base_uv * freq + phase` has a fractional part near `0` or
//! `1`, so `CPU` and `GPU` select the same lattice cell regardless of a few
//! units in the last place of slack. The continuous clamps are self-tolerant at
//! their bounds (the clamped value equals the bound there), so they need no
//! special conditioning.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::heat_distortion`；
//! no third-party engine source or derived code.

use prism_volumetric_gpu::heat_distortion::{golden, GpuHeatDistortion, HeatQuery, HeatResult};
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
fn uniform(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + lcg(state) * (hi - lo)
}

/// A pseudo-random value in `[-span, span)` drawn from `state`.
fn signed(state: &mut u64, span: f32) -> f32 {
    (lcg(state) * 2.0 - 1.0) * span
}

/// How far a rolling coordinate's fractional part must stay from an integer so
/// the value-noise floor split lands on the same lattice cell on both devices.
const FRAC_GUARD: f32 = 1.0e-2;

/// Returns whether `t`'s fractional part is comfortably away from `0` and `1`,
/// so a few units in the last place cannot flip the `floor` cell index.
fn frac_is_safe(t: f32) -> bool {
    let frac = t - t.floor();
    frac > FRAC_GUARD && frac < 1.0 - FRAC_GUARD
}

/// Builds a clearly-conditioned heat-distortion sample by rejection sampling:
/// the parameters are drawn from well-spread ranges and accepted only when both
/// rolling coordinates sit away from a lattice boundary, so the one
/// discontinuous branch (the value-noise floor split) is never on a tie. The
/// `UV` and offset magnitudes keep the distorted `UV` inside the unit square.
fn rand_query(state: &mut u64) -> HeatQuery {
    loop {
        let base_uv = [signed(state, 3.0), signed(state, 3.0)];
        let phase = signed(state, 2.0);
        let freq = uniform(state, 0.5, 4.0);
        let tx = base_uv[0] * freq + phase;
        let ty = base_uv[1] * freq + phase;
        if !frac_is_safe(tx) || !frac_is_safe(ty) {
            continue;
        }
        return HeatQuery::new(
            [signed(state, 2.0), signed(state, 2.0)],
            uniform(state, 0.1, 2.0),
            uniform(state, 0.01, 0.3),
            uniform(state, 0.0, 0.5),
            uniform(state, 0.0, 0.1),
            uniform(state, 0.05, 0.3),
            uniform(state, 0.0, 10.0),
            [uniform(state, 0.35, 0.65), uniform(state, 0.35, 0.65)],
            base_uv,
            phase,
            freq,
        );
    }
}

/// Pins one `GPU` result against the `CPU` golden for `query`: the clamped
/// offset, the distorted `UV` and the rolling shimmer must each agree within
/// bound.
fn pin(idx: usize, query: &HeatQuery, got: &HeatResult) {
    let want = golden(query);
    for lane in 0..2 {
        assert!(
            close(got.offset[lane], want.offset[lane]),
            "sample {idx} offset[{lane}]: gpu {} vs cpu {}",
            got.offset[lane],
            want.offset[lane]
        );
        assert!(
            close(got.distorted_uv[lane], want.distorted_uv[lane]),
            "sample {idx} distorted_uv[{lane}]: gpu {} vs cpu {}",
            got.distorted_uv[lane],
            want.distorted_uv[lane]
        );
        assert!(
            close(got.roll[lane], want.roll[lane]),
            "sample {idx} roll[{lane}]: gpu {} vs cpu {}",
            got.roll[lane],
            want.roll[lane]
        );
    }
}

/// Dispatches `queries` on the `GPU` and pins every result against the
/// reference.
fn check(ctx: &GpuContext, gpu: &GpuHeatDistortion, queries: &[HeatQuery]) {
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
    let gpu = GpuHeatDistortion::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn zero_normal_gives_zero_offset_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHeatDistortion::new(&ctx);
    // A flat-facing particle refracts nothing: the base offset is zero, so the
    // distorted UV equals the sampled UV. The rolling shimmer is still sampled
    // at a non-integer lattice coordinate.
    let query = HeatQuery::new(
        [0.0, 0.0],
        5.0,
        4.0,
        0.1,
        0.01,
        0.2,
        1.0,
        [0.5, 0.5],
        [0.3, 0.7],
        1.25,
        4.0,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn clamp_saturates_large_offset_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHeatDistortion::new(&ctx);
    // A huge strength drives the scaled offset far past the small max_offset, so
    // both devices clamp each component to the bound (and the clamp is
    // self-tolerant at its edge). Distance 0 keeps the falloff exactly 1.
    let query = HeatQuery::new(
        [1.0, -1.0],
        100.0,
        100.0,
        0.0,
        0.0,
        0.03,
        0.0,
        [0.5, 0.5],
        [0.2, 0.9],
        0.75,
        2.0,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn interior_uv_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHeatDistortion::new(&ctx);
    // A modest offset keeps the distorted UV well inside the unit square, so the
    // domain clamp is inactive and the falloff shapes the magnitude. The rolling
    // coordinates stay off the lattice boundaries.
    let query = HeatQuery::new(
        [0.6, -0.4],
        1.5,
        0.2,
        0.3,
        0.05,
        0.25,
        3.0,
        [0.45, 0.55],
        [1.3, -0.7],
        0.4,
        1.5,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn decorrelated_rolling_channels_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHeatDistortion::new(&ctx);
    // A symmetric base_uv would collapse to equal rolling channels only if the
    // two seeds were shared; the independent seeds keep them distinct, and the
    // GPU must reproduce both. The 0.3 fractional coordinate is safely off the
    // lattice boundary.
    let query = HeatQuery::new(
        [0.2, 0.2],
        1.0,
        0.1,
        0.2,
        0.0,
        0.2,
        2.0,
        [0.5, 0.5],
        [2.3, 2.3],
        0.0,
        1.0,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHeatDistortion::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // One batch mixing deterministic fixtures with many random samples,
    // dispatched together so the per-thread indexing and the contiguous storage
    // layout are both exercised, then pinned element-for-element.
    let mut queries = vec![
        HeatQuery::new(
            [0.0, 0.0],
            5.0,
            4.0,
            0.1,
            0.01,
            0.2,
            1.0,
            [0.5, 0.5],
            [0.3, 0.7],
            1.25,
            4.0,
        ),
        HeatQuery::new(
            [1.0, -1.0],
            100.0,
            100.0,
            0.0,
            0.0,
            0.03,
            0.0,
            [0.5, 0.5],
            [0.2, 0.9],
            0.75,
            2.0,
        ),
    ];
    for _ in 0..48 {
        queries.push(rand_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn many_samples_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHeatDistortion::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep (several workgroups' worth) pins all three outputs across
    // many random parameter sets.
    let queries: Vec<HeatQuery> = (0..200).map(|_| rand_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
