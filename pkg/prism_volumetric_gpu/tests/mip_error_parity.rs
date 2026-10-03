//! Real-device parity for the texture-streaming mip-shortfall twin:
//! [`GpuMipError`](prism_volumetric_gpu::mip_error::GpuMipError) must reproduce
//! the stateless `u32` charge of the `CPU` golden
//! [`PageDemand::mip_error`](prism_render_architecture::texture_streaming::feedback::PageDemand::mip_error)
//! — the saturating resident-minus-desired level gap and the missing-page
//! sentinel — across the resident-fine-enough, equal-level, shortfall, missing,
//! and boundary cases plus a randomized sweep compared query-for-query.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The reference method is public, so it is called directly as the oracle: a
//! [`PageDemand`](prism_render_architecture::texture_streaming::feedback::PageDemand)
//! is built from each query's levels (the other fields are irrelevant to
//! `mip_error`) and
//! [`mip_error`](prism_render_architecture::texture_streaming::feedback::PageDemand::mip_error)
//! supplies the expected charge. A passing `GPU == oracle` run is direct
//! evidence the kernel computes the same charge.
//!
//! # Parity criterion
//!
//! The charge is a pure `u32` map built from an integer subtraction and ordered
//! comparisons, so `CPU` and `GPU` agree bit-for-bit and every query is asserted
//! with an exact `==`.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::texture_streaming::feedback::PageDemand::mip_error`；无第三方引擎源码或衍生代码。

use prism_render_architecture::texture_streaming::feedback::{
    PageDemand, MISSING_PAGE_MIP_PENALTY,
};
use prism_render_architecture::texture_streaming::{TexturePageKey, TextureSemantic};
use prism_volumetric_gpu::mip_error::{GpuMipError, MipErrorQuery, MipErrorResult};
use prism_volumetric_gpu::GpuContext;

/// Builds a reference [`PageDemand`] whose `mip_error`-relevant fields match a
/// query. The key, semantic, importance, byte cost and frame are irrelevant to
/// the charge under test and are fixed to representative constants.
fn demand(desired_mip: u8, resident_mip: Option<u8>) -> PageDemand {
    PageDemand {
        key: TexturePageKey {
            texture: 1,
            mip: desired_mip,
            layer: 0,
            x: 0,
            y: 0,
        },
        semantic: TextureSemantic::Color,
        desired_mip,
        resident_mip,
        screen_importance: 500,
        byte_cost: 4096,
        frame: 1,
    }
}

/// Computes the reference charge for one query by calling the golden directly.
fn oracle(q: &MipErrorQuery) -> MipErrorResult {
    let desired = q.desired_mip as u8;
    let resident = if q.has_resident != 0 {
        Some(q.resident_mip as u8)
    } else {
        None
    };
    MipErrorResult {
        mip_error: demand(desired, resident).mip_error(),
    }
}

/// Pins one `GPU` result against the oracle: the integer charge exactly.
fn check_result(idx: usize, got: &MipErrorResult, want: &MipErrorResult) {
    assert_eq!(
        got.mip_error, want.mip_error,
        "query {idx} mip_error: gpu {} vs cpu {}",
        got.mip_error, want.mip_error
    );
}

/// Dispatches every query and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuMipError, queries: &[MipErrorQuery]) {
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

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping mip_error parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuMipError::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn resident_fine_enough_charges_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMipError::new(&ctx);
    // Resident exactly as fine as desired, and resident finer than desired
    // (smaller index): both saturate to a zero charge.
    let queries = [
        MipErrorQuery::new(2, Some(2)),
        MipErrorQuery::new(2, Some(1)),
    ];
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got[0].mip_error, 0, "equal levels charge zero");
    assert_eq!(got[1].mip_error, 0, "finer resident charges zero");
    check(&ctx, &gpu, &queries);
}

#[test]
fn shortfall_counts_levels() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMipError::new(&ctx);
    // Resident three levels coarser than desired charges exactly three.
    let queries = [MipErrorQuery::new(1, Some(4))];
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got[0].mip_error, 3, "a three-level gap charges three");
    check(&ctx, &gpu, &queries);
}

#[test]
fn missing_page_uses_fixed_penalty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMipError::new(&ctx);
    // A page with nothing resident is charged the missing-page sentinel,
    // independent of the desired level.
    let queries = [MipErrorQuery::new(0, None), MipErrorQuery::new(7, None)];
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(
        got[0].mip_error, MISSING_PAGE_MIP_PENALTY,
        "a missing page is charged the sentinel"
    );
    assert_eq!(
        got[1].mip_error, MISSING_PAGE_MIP_PENALTY,
        "the sentinel is independent of the desired level"
    );
    check(&ctx, &gpu, &queries);
}

#[test]
fn boundary_levels_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMipError::new(&ctx);
    // Extremes of the u8 level range: finest desired against the coarsest
    // resident (maximal gap), and the coarsest desired against a finer resident
    // (saturating to zero).
    let queries = [
        MipErrorQuery::new(0, Some(255)),
        MipErrorQuery::new(255, Some(0)),
        MipErrorQuery::new(255, Some(255)),
        MipErrorQuery::new(0, Some(0)),
    ];
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got[0].mip_error, 255, "maximal gap charges the full span");
    assert_eq!(got[1].mip_error, 0, "far-finer resident saturates to zero");
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMipError::new(&ctx);
    let mut state = 0x2c9e_7a51_f083_b64d_u64;
    let mut queries = Vec::new();
    // Several workgroups' worth of queries spanning the u8 level range, with
    // roughly one in four pages fully missing, so both the shortfall branch and
    // the missing-page sentinel are exercised across the dispatch.
    while queries.len() < 300 {
        let desired = lcg(&mut state) % 256;
        let resident_present = !lcg(&mut state).is_multiple_of(4);
        let resident = if resident_present {
            Some(lcg(&mut state) % 256)
        } else {
            None
        };
        queries.push(MipErrorQuery::new(desired, resident));
    }
    check(&ctx, &gpu, &queries);
}
