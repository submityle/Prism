//! Real-device parity for the integer-`lcm` `u32` twin:
//! [`GpuIntegerGcdLcm`](prism_volumetric_gpu::integer_gcd_lcm::GpuIntegerGcdLcm)
//! must reproduce the `CPU` golden
//! [`integer_gcd_lcm`](prism_render_architecture::particle::integer_gcd_lcm)
//! least-common-multiple query for query, including the explicit `u32`
//! overflow flag.
//!
//! The golden standard is a `u64` domain, so the oracle widens each `u32`
//! operand to `u64`, calls
//! [`lcm_u64`](prism_render_architecture::particle::integer_gcd_lcm::lcm_u64)
//! (and
//! [`lcm_checked_u64`](prism_render_architecture::particle::integer_gcd_lcm::lcm_checked_u64)
//! for the overflow oracle) and compares against the `u32` twin: when the true
//! `lcm` fits in `u32` the twin must report the value with the flag clear, and
//! when it exceeds `u32::MAX` the twin must report the flag set. The fixtures
//! cover the degenerate `0` cases, coprime pairs, pairs sharing a common
//! factor, equal pairs, explicit values straddling the `u32::MAX` boundary and
//! a rejection-sampled batch精调 to both sides of that boundary.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every operation is pure unsigned integer arithmetic with no rounding, so
//! `CPU` (restricted to the `u32` domain) and `GPU` must agree bit for bit. The
//! comparison is an exact `==` on both the `u32` value and the overflow flag
//! with no tolerance: any mismatch is a genuine port bug. The greatest-common-
//! divisor primitive is twinned separately by the `integer_gcd` module and is
//! out of scope here.
//!
//! Provenance: twinned from this repository's
//! [`integer_gcd_lcm`](prism_render_architecture::particle::integer_gcd_lcm);
//! no third-party engine source or derived code.

use prism_render_architecture::particle::integer_gcd_lcm::{lcm_checked_u64, lcm_u64};
use prism_volumetric_gpu::integer_gcd_lcm::{
    GpuIntegerGcdLcm, IntegerGcdLcmQuery, IntegerGcdLcmResult,
};
use prism_volumetric_gpu::GpuContext;

/// A deterministic linear-congruential generator so the randomized fixtures are
/// reproducible bit for bit across runs and platforms. Pure integer math, no
/// external math library.
fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

/// The expected `u32`-domain answer for one operand pair: widen to `u64`, call
/// the golden reference and decide the overflow flag against `u32::MAX`.
fn oracle(a: u32, b: u32) -> IntegerGcdLcmResult {
    let full = lcm_u64(u64::from(a), u64::from(b));
    // For `u32` inputs the checked variant never reports `None` (the product
    // fits in `u64`); this is the same value, kept to exercise the API the host
    // uses to classify overflow.
    debug_assert_eq!(lcm_checked_u64(u64::from(a), u64::from(b)), Some(full));
    if full <= u64::from(u32::MAX) {
        IntegerGcdLcmResult {
            lcm: full as u32,
            overflow: false,
        }
    } else {
        IntegerGcdLcmResult {
            lcm: 0,
            overflow: true,
        }
    }
}

/// Explicit fixtures covering the degenerate, structured and boundary cases.
fn structured_pairs() -> Vec<(u32, u32)> {
    vec![
        // Degenerate zero cases: lcm is 0, flag clear.
        (0, 0),
        (0, 7),
        (7, 0),
        (0, u32::MAX),
        (u32::MAX, 0),
        // Coprime pairs with a small product.
        (17, 5),
        (9, 28),
        (1, 1),
        (1, u32::MAX),
        (u32::MAX, 1),
        // Pairs sharing a common factor.
        (12, 18),
        (100, 75),
        (48, 36),
        (1071, 462),
        (1000, 250),
        // Equal pairs: lcm == operand, flag clear.
        (13, 13),
        (0xDEAD_BEEF, 0xDEAD_BEEF),
        (u32::MAX, u32::MAX),
        // Just below the u32::MAX boundary: coprime consecutive values whose
        // product is 4_294_901_760 <= u32::MAX (4_294_967_295).
        (65_535, 65_536),
        (65_536, 65_535),
        // Just above the boundary: 65537 is prime, so the pair is coprime and
        // the product 4_295_032_832 > u32::MAX.
        (65_537, 65_536),
        (65_536, 65_537),
        // A multiple relationship whose lcm is the larger operand (fits).
        (3, 4_294_967_292),
        // Large coprime pair that clearly overflows.
        (4_294_967_291, 4_294_967_279),
    ]
}

/// Rejection-samples operand pairs whose true `lcm` lands just on either side of
/// the `u32::MAX` boundary, so both the fits and the overflow branch are
/// exercised with near-critical magnitudes.
fn boundary_pairs() -> Vec<(u32, u32)> {
    let mut under: Vec<(u32, u32)> = Vec::new();
    let mut over: Vec<(u32, u32)> = Vec::new();
    let mut state = 0x0BAD_F00D_1234_5678_u64;
    let limit = u64::from(u32::MAX);
    // Window around the boundary (both sides) within which a sample is kept.
    let window: u64 = 64_000_000;
    // Bounded number of draws so the sampler always terminates; the window is
    // wide enough to collect both branches well inside this budget.
    for _ in 0..4_000_000u32 {
        if under.len() >= 48 && over.len() >= 48 {
            break;
        }
        // Operands up to ~2^18 so random lcm values can reach the boundary.
        let a = ((lcg_next(&mut state) >> 40) as u32 & 0x0003_FFFF) | 1;
        let b = ((lcg_next(&mut state) >> 40) as u32 & 0x0003_FFFF) | 1;
        let full = lcm_u64(u64::from(a), u64::from(b));
        if full <= limit && limit - full <= window && under.len() < 48 {
            under.push((a, b));
        } else if full > limit && full - limit <= window && over.len() < 48 {
            over.push((a, b));
        }
    }
    under.extend(over);
    under
}

/// Runs a batch of operand pairs on-device and asserts an exact match against
/// the oracle, query for query.
fn check_pairs(gpu: &GpuIntegerGcdLcm, ctx: &GpuContext, pairs: &[(u32, u32)]) {
    let queries: Vec<IntegerGcdLcmQuery> = pairs
        .iter()
        .map(|&(a, b)| IntegerGcdLcmQuery::new(a, b))
        .collect();
    let got = gpu.run(ctx, &queries);
    assert_eq!(got.len(), pairs.len());
    for (idx, &(a, b)) in pairs.iter().enumerate() {
        let want = oracle(a, b);
        assert_eq!(
            got[idx], want,
            "lcm mismatch at {idx} for inputs a={a} b={b}"
        );
    }
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuIntegerGcdLcm::new(&ctx);
    assert!(gpu.run(&ctx, &[]).is_empty());
}

#[test]
fn structured_cases_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuIntegerGcdLcm::new(&ctx);
    check_pairs(&gpu, &ctx, &structured_pairs());
}

#[test]
fn boundary_cases_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuIntegerGcdLcm::new(&ctx);
    let pairs = boundary_pairs();
    // Both branches must be represented so the overflow guard is exercised on
    // near-critical magnitudes.
    assert!(pairs.iter().any(|&(a, b)| !oracle(a, b).overflow));
    assert!(pairs.iter().any(|&(a, b)| oracle(a, b).overflow));
    check_pairs(&gpu, &ctx, &pairs);
}

#[test]
fn lcg_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuIntegerGcdLcm::new(&ctx);
    let mut pairs: Vec<(u32, u32)> = Vec::new();
    let mut state = 0x0123_4567_89AB_CDEF_u64;
    for _ in 0..256 {
        // Full-range operands exercise both the fits and the overflow branch.
        let a = (lcg_next(&mut state) >> 32) as u32;
        let b = (lcg_next(&mut state) >> 32) as u32;
        pairs.push((a, b));
        // Masked operands make non-trivial shared factors more common and keep
        // some products well inside u32.
        pairs.push((a & 0xFFFF, b & 0xFFFF));
        pairs.push((a & 0xFF, b & 0xFF));
    }
    check_pairs(&gpu, &ctx, &pairs);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuIntegerGcdLcm::new(&ctx);
    // Interleave structured and boundary pairs in one dispatch so adjacent
    // threads within a workgroup hit different branches.
    let mut pairs = structured_pairs();
    pairs.extend(boundary_pairs());
    check_pairs(&gpu, &ctx, &pairs);
}
