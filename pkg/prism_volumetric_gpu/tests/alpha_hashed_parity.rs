//! Real-device parity for the hashed-alpha twin:
//! [`GpuAlphaHashed`](prism_volumetric_gpu::alpha_hashed::GpuAlphaHashed) must
//! reproduce the `CPU` golden
//! ([`alpha_hashed`](prism_render_architecture::particle::alpha_hashed)) across
//! a fully transparent guard (`alpha == 0`, always discarded), a fully opaque
//! guard (`alpha == 1`, always kept), a `min_threshold` floor that clamps the
//! blended hash, a strongly anisotropic footprint, a weakly anisotropic
//! footprint and a randomized batch compared element-for-element.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The integer hash path (`mix`, `hash3`, the exponent-field `exp2_pow` and the
//! `floor`-based `quantize_coord`) is reproduced bit-for-bit, so the integer
//! lattice cell, the chosen `LOD` level and the keep/discard decision agree
//! exactly: the keep flag is compared with `==`. The continuous fields
//! (`threshold`, `lod_scale`, `coverage`, `hash_probe`) are a fixed,
//! non-reorderable sequence of multiplies, adds, divides and one `sqrt`; a `GPU`
//! may fuse a multiply-add the scalar reference leaves separate, perturbing the
//! low mantissa bits by a few units in the last place, so they are compared with
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3`.
//!
//! # Conditioning
//!
//! The `floor`-driven `LOD` level and lattice selection are discontinuous at
//! octave and cell boundaries: a sub-`ULP` difference in the pixel scale or a
//! quantized coordinate would flip the integer level or cell and produce a
//! completely different (yet individually correct) hash on each device. Every
//! random fixture is therefore rejection-sampled away from those cracks: the
//! octave fraction of `approx_log2(pix_scale)` is kept mid-step, the fractional
//! part of every quantized coordinate at both bracketing levels is kept away
//! from the `floor` tie, the two bracketing lattice hashes are forced to differ
//! so the two-level blend is genuine, the fragment `alpha` is kept clear of the
//! threshold tie, and the clamp fixture is required to actually hit its floor.
//!
//! Provenance: 孪生自本仓
//! `prism_render_architecture::particle::alpha_hashed`；no third-party engine
//! source or derived code.

use prism_render_architecture::particle::alpha_hashed::hash3;
use prism_volumetric_gpu::alpha_hashed::{
    golden, AlphaHashedQuery, AlphaHashedResult, GpuAlphaHashed,
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

/// Comparison epsilon mirroring the golden `CMP_EPS`, used only for host-side
/// conditioning of the random fixtures.
const CMP_EPS: f32 = 1.0e-6;

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

/// A pseudo-random value in `[-span, span)` drawn from `state`.
fn signed(state: &mut u64, span: f32) -> f32 {
    (lcg(state) * 2.0 - 1.0) * span
}

/// Host replica of the golden `approx_log2`: the piecewise-linear `log2` read
/// straight from the `f32` exponent and mantissa bits. Used only to measure a
/// fixture's distance from an octave boundary; it calls no transcendental.
fn approx_log2(x: f32) -> f32 {
    let bits = x.to_bits();
    let exponent = ((bits >> 23) & 0xff) as i32 - 127;
    let mantissa_bits = (bits & 0x007f_ffff) | 0x3f80_0000;
    let mantissa = f32::from_bits(mantissa_bits);
    (exponent as f32) + (mantissa - 1.0)
}

/// Host replica of the golden `exp2_pow`: rebuilds `2^level` by writing the
/// biased exponent field. Used only to recompute the two bracketing lattice
/// scales for conditioning; it calls no transcendental.
fn exp2_pow(level: i32) -> f32 {
    let clamped = level.clamp(-126, 127);
    let biased = (clamped + 127) as u32;
    f32::from_bits(biased << 23)
}

/// Host replica of the golden `quantize_coord`.
fn quantize_coord(coord: f32, scale: f32) -> i32 {
    (coord * scale).floor() as i32
}

/// The alpha a fixture should carry: either a specific guard value or a random
/// draw kept clear of the threshold tie.
#[derive(Clone, Copy)]
enum AlphaMode {
    /// A fixed alpha (used by the opaque/transparent/clamp-discard guards).
    Fixed(f32),
    /// A random alpha, rejection-sampled away from the threshold tie.
    Random,
}

/// Rejection-samples a fully crack-safe query. `strong` selects a strongly vs
/// weakly anisotropic derivative pair, `min_threshold` sets the clamp floor,
/// `want_clamp` requires the blended hash to actually fall below that floor (so
/// the clamp is exercised) or, when false, to stay strictly inside the open
/// interval (so the clamp is inert), and `alpha_mode` fixes or randomizes the
/// fragment alpha.
fn conditioned(
    state: &mut u64,
    strong: bool,
    min_threshold: f32,
    want_clamp: bool,
    alpha_mode: AlphaMode,
) -> AlphaHashedQuery {
    let hash_scale = 8.0_f32;
    loop {
        let anchor = [signed(state, 5.0), signed(state, 5.0), signed(state, 5.0)];
        let (ddx, ddy) = if strong {
            // One axis much longer than the other: a strongly anisotropic
            // footprint whose larger length drives the LOD scale.
            (
                [3.0 + lcg(state) * 2.0, 0.3, 0.2],
                [0.2, 0.4 + lcg(state) * 0.2, 0.1],
            )
        } else {
            // Nearly balanced axes: a weakly anisotropic, near-round footprint.
            (
                [1.0 + lcg(state) * 0.2, 0.1, 0.0],
                [0.1, 1.0 + lcg(state) * 0.2, 0.0],
            )
        };
        let seed = (*state >> 16) as u32 ^ 0x51ed_270b;
        let probe = [
            signed(state, 60.0) as i32,
            signed(state, 60.0) as i32,
            signed(state, 60.0) as i32,
        ];
        let alpha = match alpha_mode {
            AlphaMode::Fixed(v) => v,
            AlphaMode::Random => 0.15 + lcg(state) * 0.7,
        };
        let query = AlphaHashedQuery::new(
            anchor,
            ddx,
            ddy,
            alpha,
            hash_scale,
            min_threshold,
            seed,
            probe,
        );
        let g = golden(&query);

        // Recreate the level selection to measure crack distance.
        let scale = hash_scale.max(CMP_EPS);
        let footprint = g.lod_scale.max(CMP_EPS);
        let pix_scale = (1.0 / (scale * footprint)).max(CMP_EPS);
        let level = approx_log2(pix_scale);
        let frac_level = level - level.floor();
        // Octave fraction mid-step so a sub-ULP wobble can't flip the level.
        if !(0.25..=0.75).contains(&frac_level) {
            continue;
        }
        let coarse = level.floor() as i32;
        let scale_lo = exp2_pow(coarse) * scale;
        let scale_hi = exp2_pow(coarse + 1) * scale;

        // Every quantized coordinate's fraction away from the floor tie.
        let mut cell_safe = true;
        for s in [scale_lo, scale_hi] {
            for c in anchor {
                let p = c * s;
                let qf = p - p.floor();
                if !(0.2..=0.8).contains(&qf) {
                    cell_safe = false;
                }
            }
        }
        if !cell_safe {
            continue;
        }

        // The two bracketing lattice hashes must differ so the blend is genuine.
        let h0 = hash3(
            quantize_coord(anchor[0], scale_lo),
            quantize_coord(anchor[1], scale_lo),
            quantize_coord(anchor[2], scale_lo),
            seed,
        );
        let h1 = hash3(
            quantize_coord(anchor[0], scale_hi),
            quantize_coord(anchor[1], scale_hi),
            quantize_coord(anchor[2], scale_hi),
            seed,
        );
        if (h0 - h1).abs() < 0.05 {
            continue;
        }
        let blended = h0 + (h1 - h0) * frac_level;

        if want_clamp {
            // The blend must sit below the floor so the clamp is actually hit.
            if blended > min_threshold - 0.05 {
                continue;
            }
        } else {
            // The blend must stay strictly inside the open interval so neither
            // the floor nor the opaque clamp is engaged.
            if blended < min_threshold + 0.05 || blended > 0.95 {
                continue;
            }
        }

        // Keep the fragment alpha clear of the threshold tie unless it is a
        // deliberate guard value, which is already far from any interior
        // threshold.
        if let AlphaMode::Random = alpha_mode
            && (alpha - g.threshold).abs() < 0.08
        {
            continue;
        }

        return query;
    }
}

/// A generic crack-safe random query with an inert clamp and a random alpha.
fn rand_query(state: &mut u64) -> AlphaHashedQuery {
    let strong = lcg(state) > 0.5;
    conditioned(state, strong, 1.0e-3, false, AlphaMode::Random)
}

/// Pins one `GPU` result against the `CPU` golden for `query`: every continuous
/// field must agree within bound and the keep/discard flag exactly.
fn pin(idx: usize, query: &AlphaHashedQuery, got: &AlphaHashedResult) {
    let want = golden(query);
    assert!(
        close(got.threshold, want.threshold),
        "query {idx} threshold: gpu {} vs cpu {}",
        got.threshold,
        want.threshold
    );
    assert!(
        close(got.lod_scale, want.lod_scale),
        "query {idx} lod_scale: gpu {} vs cpu {}",
        got.lod_scale,
        want.lod_scale
    );
    assert_eq!(
        got.keep, want.keep,
        "query {idx} keep: gpu {} vs cpu {}",
        got.keep, want.keep
    );
    assert!(
        close(got.coverage, want.coverage),
        "query {idx} coverage: gpu {} vs cpu {}",
        got.coverage,
        want.coverage
    );
    assert!(
        close(got.hash_probe, want.hash_probe),
        "query {idx} hash_probe: gpu {} vs cpu {}",
        got.hash_probe,
        want.hash_probe
    );
}

/// Dispatches `queries` on the `GPU` and pins every result against the
/// reference.
fn check(ctx: &GpuContext, gpu: &GpuAlphaHashed, queries: &[AlphaHashedQuery]) {
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
    let gpu = GpuAlphaHashed::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn transparent_fragment_is_always_discarded() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAlphaHashed::new(&ctx);
    let mut state = 0x0a1b_2c3d_4e5f_6071_u64;
    // alpha == 0 is below any strictly-positive threshold, so the fragment is
    // always discarded; coverage collapses to zero.
    let query = conditioned(&mut state, false, 1.0e-3, false, AlphaMode::Fixed(0.0));
    let want = golden(&query);
    assert!(!want.keep, "a fully transparent fragment must be discarded");
    check(&ctx, &gpu, &[query]);
}

#[test]
fn opaque_fragment_is_always_kept() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAlphaHashed::new(&ctx);
    let mut state = 0x1122_3344_5566_7788_u64;
    // alpha == 1 is at or above any threshold (bounded strictly below 1 by the
    // 24-bit hash), so the fragment always survives; coverage is one.
    let query = conditioned(&mut state, false, 1.0e-3, false, AlphaMode::Fixed(1.0));
    let want = golden(&query);
    assert!(want.keep, "a fully opaque fragment must be kept");
    check(&ctx, &gpu, &[query]);
}

#[test]
fn threshold_floor_clamp_is_honored() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAlphaHashed::new(&ctx);
    let mut state = 0x9988_7766_5544_3322_u64;
    // A high min_threshold with a low blended hash forces the clamp to the
    // floor; alpha below the floor is then discarded on both devices.
    let query = conditioned(&mut state, false, 0.6, true, AlphaMode::Fixed(0.2));
    let want = golden(&query);
    assert!(
        (want.threshold - 0.6).abs() < 1.0e-3,
        "the threshold must pin to the floor, got {}",
        want.threshold
    );
    assert!(
        !want.keep,
        "alpha below the clamped floor must be discarded"
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn strong_anisotropy_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAlphaHashed::new(&ctx);
    let mut state = 0xfeed_face_dead_beef_u64;
    // A strongly elongated footprint: the long axis dominates the LOD scale and
    // selects a coarser lattice level.
    let query = conditioned(&mut state, true, 1.0e-3, false, AlphaMode::Random);
    check(&ctx, &gpu, &[query]);
}

#[test]
fn weak_anisotropy_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAlphaHashed::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A near-round footprint with balanced axes: the LOD scale is close to the
    // isotropic case.
    let query = conditioned(&mut state, false, 1.0e-3, false, AlphaMode::Random);
    check(&ctx, &gpu, &[query]);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAlphaHashed::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // One batch mixing the guard, clamp and anisotropy fixtures with many random
    // queries, dispatched together so the per-thread indexing and the contiguous
    // storage layout are both exercised, then pinned element-for-element.
    let mut queries = vec![
        conditioned(&mut state, false, 1.0e-3, false, AlphaMode::Fixed(0.0)),
        conditioned(&mut state, false, 1.0e-3, false, AlphaMode::Fixed(1.0)),
        conditioned(&mut state, false, 0.6, true, AlphaMode::Fixed(0.2)),
        conditioned(&mut state, true, 1.0e-3, false, AlphaMode::Random),
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
    let gpu = GpuAlphaHashed::new(&ctx);
    let mut state = 0x00c0_ffee_1234_abcd_u64;
    // A larger sweep (several workgroups' worth) pins every field across many
    // random crack-safe geometries.
    let queries: Vec<AlphaHashedQuery> = (0..200).map(|_| rand_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
