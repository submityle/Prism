//! Real-device parity for the `CIE` color twin:
//! [`GpuCieXyz`](prism_volumetric_gpu::cie_xyz::GpuCieXyz) must reproduce the
//! `CPU` golden
//! [`cie_xyz`](prism_render_architecture::particle::cie_xyz) across the primary
//! `sRGB` channels (each mapping to a known `XYZ` column), the `D65` white round
//! trip, the two divide-by-zero guards (a black `XYZ` falling back to the `D65`
//! chromaticity and a zero-luminance `xyY` collapsing to the origin), the
//! `Bradford` identity and `D65` <-> `D50` adaptation, and a randomized batch
//! compared element-for-element.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each transform is a fixed, non-reorderable sequence of multiplies, adds and
//! divides, so `CPU` and `GPU` evaluate the same closed form in the same order.
//! They are not bit-exact: a `GPU` may fuse a multiply-add the scalar reference
//! leaves separate, perturbing the low mantissa bits by a few units in the last
//! place. The comparison therefore allows `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3` on every `f32` field.
//!
//! # Conditioning
//!
//! Every random fixture is kept well away from the two degeneracy cracks: the
//! `XYZ` fed to the `xyY` split has a tristimulus sum comfortably above zero,
//! the `xyY` fed to the reconstruction has a luminance `y` comfortably above
//! zero, and the `Bradford` reference whites are the physical `D65`/`D50`
//! whites, whose cone responses are all positive, so `CPU` and `GPU` stay on the
//! same side of every guard regardless of a few units in the last place of
//! slack. The degenerate branches are pinned by separate deterministic fixtures
//! whose guarded values are exactly zero.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::cie_xyz`；
//! no third-party engine source or derived code.
#![forbid(unsafe_code)]

use prism_render_architecture::particle::cie_xyz::{LinearSrgb, Xyy, Xyz, D50_XYZ, D65_XYZ};
use prism_volumetric_gpu::cie_xyz::{golden, CieXyzQuery, CieXyzResult, GpuCieXyz};
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
    lo + lcg(state) * (hi - lo)
}

/// Builds a clearly-conditioned color query by sampling each input well away
/// from the two guard thresholds: the `XYZ` fed to the split has a tristimulus
/// sum above `0.3`, the `xyY` has a luminance `y` in `[0.1, 0.9)`, and the
/// `Bradford` whites are the physical `D65`/`D50` whites.
fn rand_query(state: &mut u64) -> CieXyzQuery {
    let rgb = LinearSrgb::new(
        range(state, 0.0, 1.0),
        range(state, 0.0, 1.0),
        range(state, 0.0, 1.0),
    );
    // Each XYZ component is at least 0.1, so the tristimulus sum is at least 0.3
    // and the split never approaches its non-positive-sum guard.
    let xyz = Xyz::new(
        range(state, 0.1, 1.0),
        range(state, 0.1, 1.0),
        range(state, 0.1, 1.0),
    );
    // Chromaticity x in (0, 0.7) and y in [0.1, 0.9) keeps 1 - x - y and the
    // luminance ratio well-conditioned, far from the non-positive-y guard.
    let xyy = Xyy::new(
        range(state, 0.05, 0.7),
        range(state, 0.1, 0.9),
        range(state, 0.1, 1.0),
    );
    let bsrc = Xyz::new(
        range(state, 0.1, 1.0),
        range(state, 0.1, 1.0),
        range(state, 0.1, 1.0),
    );
    CieXyzQuery::new(rgb, xyz, xyy, bsrc, D65_XYZ, D50_XYZ)
}

/// Pins one `GPU` result against the `CPU` golden for `query`: every component
/// of all five transforms must agree within bound.
fn pin(idx: usize, query: &CieXyzQuery, got: &CieXyzResult) {
    let want = golden(query);
    let fields = [
        ("to_xyz.x", got.to_xyz.x, want.to_xyz.x),
        ("to_xyz.y", got.to_xyz.y, want.to_xyz.y),
        ("to_xyz.z", got.to_xyz.z, want.to_xyz.z),
        ("to_srgb.r", got.to_srgb.r, want.to_srgb.r),
        ("to_srgb.g", got.to_srgb.g, want.to_srgb.g),
        ("to_srgb.b", got.to_srgb.b, want.to_srgb.b),
        ("to_xyy.x", got.to_xyy.x, want.to_xyy.x),
        ("to_xyy.y", got.to_xyy.y, want.to_xyy.y),
        ("to_xyy.big_y", got.to_xyy.big_y, want.to_xyy.big_y),
        ("from_xyy.x", got.from_xyy.x, want.from_xyy.x),
        ("from_xyy.y", got.from_xyy.y, want.from_xyy.y),
        ("from_xyy.z", got.from_xyy.z, want.from_xyy.z),
        ("bradford.x", got.bradford.x, want.bradford.x),
        ("bradford.y", got.bradford.y, want.bradford.y),
        ("bradford.z", got.bradford.z, want.bradford.z),
    ];
    for (name, g, c) in fields {
        assert!(close(g, c), "query {idx} {name}: gpu {g} vs cpu {c}");
    }
}

/// Dispatches `queries` on the `GPU` and pins every result against the
/// reference.
fn check(ctx: &GpuContext, gpu: &GpuCieXyz, queries: &[CieXyzQuery]) {
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

/// A well-conditioned deterministic query used wherever a particular transform
/// is under test and the other inputs just need to be clear of the guards.
fn baseline() -> CieXyzQuery {
    CieXyzQuery::new(
        LinearSrgb::new(0.25, 0.5, 0.75),
        Xyz::new(0.5, 0.4, 0.3),
        Xyy::new(0.3, 0.35, 0.5),
        Xyz::new(0.4, 0.5, 0.6),
        D65_XYZ,
        D50_XYZ,
    )
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCieXyz::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn primary_channels_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCieXyz::new(&ctx);
    // Each pure linear-sRGB primary maps to one column of the primaries matrix;
    // dispatched together they exercise the sRGB -> XYZ conversion per channel.
    let queries = [
        CieXyzQuery::new(
            LinearSrgb::new(1.0, 0.0, 0.0),
            Xyz::new(0.5, 0.4, 0.3),
            Xyy::new(0.3, 0.35, 0.5),
            Xyz::new(0.4, 0.5, 0.6),
            D65_XYZ,
            D50_XYZ,
        ),
        CieXyzQuery::new(
            LinearSrgb::new(0.0, 1.0, 0.0),
            Xyz::new(0.5, 0.4, 0.3),
            Xyy::new(0.3, 0.35, 0.5),
            Xyz::new(0.4, 0.5, 0.6),
            D65_XYZ,
            D50_XYZ,
        ),
        CieXyzQuery::new(
            LinearSrgb::new(0.0, 0.0, 1.0),
            Xyz::new(0.5, 0.4, 0.3),
            Xyy::new(0.3, 0.35, 0.5),
            Xyz::new(0.4, 0.5, 0.6),
            D65_XYZ,
            D50_XYZ,
        ),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn white_maps_to_d65_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCieXyz::new(&ctx);
    // Linear-sRGB white maps to the D65 tristimulus triple; the result is pinned
    // against the reference, which produces the D65 constant.
    let query = CieXyzQuery::new(
        LinearSrgb::new(1.0, 1.0, 1.0),
        D65_XYZ,
        Xyy::new(0.3127, 0.329, 1.0),
        Xyz::new(0.4, 0.5, 0.6),
        D65_XYZ,
        D50_XYZ,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn xyz_to_xyy_black_falls_back_to_d65_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCieXyz::new(&ctx);
    // A black XYZ (tristimulus sum zero) takes the non-positive-sum guard, so the
    // split falls back to the D65 white chromaticity while preserving Y = 0. The
    // other inputs stay clear of their guards.
    let query = CieXyzQuery::new(
        LinearSrgb::new(0.25, 0.5, 0.75),
        Xyz::new(0.0, 0.0, 0.0),
        Xyy::new(0.3, 0.35, 0.5),
        Xyz::new(0.4, 0.5, 0.6),
        D65_XYZ,
        D50_XYZ,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn xyy_to_xyz_zero_luminance_collapses_to_origin_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCieXyz::new(&ctx);
    // An xyY with chromaticity y = 0 takes the non-positive-y guard, so the
    // reconstruction collapses to the XYZ origin on both devices.
    let query = CieXyzQuery::new(
        LinearSrgb::new(0.25, 0.5, 0.75),
        Xyz::new(0.5, 0.4, 0.3),
        Xyy::new(0.3, 0.0, 0.5),
        Xyz::new(0.4, 0.5, 0.6),
        D65_XYZ,
        D50_XYZ,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn xyy_roundtrip_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCieXyz::new(&ctx);
    // A single baseline color exercises every transform at once, including the
    // well-conditioned xyY split and reconstruction.
    check(&ctx, &gpu, &[baseline()]);
}

#[test]
fn bradford_identity_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCieXyz::new(&ctx);
    // Adapting between identical whites is the identity on the source color; the
    // result is pinned against the reference, which returns the input.
    let query = CieXyzQuery::new(
        LinearSrgb::new(0.25, 0.5, 0.75),
        Xyz::new(0.5, 0.4, 0.3),
        Xyy::new(0.3, 0.35, 0.5),
        Xyz::new(0.4, 0.5, 0.6),
        D65_XYZ,
        D65_XYZ,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn bradford_white_to_white_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCieXyz::new(&ctx);
    // Adapting the source white itself from D65 to D50 must land on the D50
    // white; the GPU result is pinned against the reference answer.
    let query = CieXyzQuery::new(
        LinearSrgb::new(0.25, 0.5, 0.75),
        Xyz::new(0.5, 0.4, 0.3),
        Xyy::new(0.3, 0.35, 0.5),
        D65_XYZ,
        D65_XYZ,
        D50_XYZ,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCieXyz::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // One batch mixing deterministic fixtures with many random colors,
    // dispatched together so the per-thread indexing and the contiguous storage
    // layout are both exercised, then pinned element-for-element.
    let mut queries = vec![baseline()];
    for _ in 0..48 {
        queries.push(rand_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn many_colors_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCieXyz::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep (several workgroups' worth) pins every transform across many
    // random colors.
    let queries: Vec<CieXyzQuery> = (0..200).map(|_| rand_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
