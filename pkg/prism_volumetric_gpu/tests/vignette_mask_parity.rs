//! Real-device parity for the `vignette`-mask twin:
//! [`GpuVignetteMask`](prism_volumetric_gpu::vignette_mask::GpuVignetteMask)
//! must reproduce the `CPU` golden
//! [`vignette_mask`](prism_render_architecture::particle::vignette_mask) across
//! the optical center (fully lit), a point strictly inside the inner radius
//! (also fully lit), a corner beyond the outer radius (saturated to
//! `1 - intensity`), a mid-band point, a box (`roundness = 0`) versus round
//! (`roundness = 1`) pair, an aspect-stretched frame, a color-apply check and a
//! randomized batch compared element-for-element.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds, divides
//! and one `sqrt`, so `CPU` and `GPU` evaluate the same closed form in the same
//! order. They are not bit-exact: a `GPU` may fuse a multiply-add the scalar
//! reference leaves separate, perturbing the low mantissa bits by a few units in
//! the last place. The comparison therefore allows `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3` on the mask and every color channel.
//!
//! # Conditioning
//!
//! Every random fixture is kept well away from the band edges: rejection
//! sampling requires the shape distance to sit a margin inside the inner radius,
//! a margin into the mid-band, or a margin beyond the outer radius, so a few
//! units in the last place of slack never flips a `uv` across a `smoothstep`
//! clamp boundary. The inner and outer radii are always separated by a healthy
//! span, so the degenerate near-zero-width branch is never on a tie.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::vignette_mask`；
//! no third-party engine source or derived code.

use prism_render_architecture::particle::vignette_mask::VignetteParams;
use prism_volumetric_gpu::vignette_mask::{
    golden, GpuVignetteMask, VignetteMaskQuery, VignetteMaskResult,
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

/// Margin, in the aspect-corrected `uv` metric, kept between a fixture's shape
/// distance and either band edge so a few units in the last place never flips a
/// `uv` across a `smoothstep` clamp boundary.
const BAND_MARGIN: f32 = 0.03;

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

/// Builds a clearly-conditioned query by rejection sampling: the inner and outer
/// radii always keep a healthy span, and the `uv` is accepted only when its
/// shape distance sits a [`BAND_MARGIN`] clear of each band edge, so both
/// devices stay on the same side of every `smoothstep` clamp boundary.
fn rand_query(state: &mut u64) -> VignetteMaskQuery {
    loop {
        let center = [range(state, 0.4, 0.6), range(state, 0.4, 0.6)];
        let inner = range(state, 0.1, 0.3);
        let outer = inner + range(state, 0.15, 0.4);
        let intensity = range(state, 0.2, 1.0);
        let roundness = range(state, 0.0, 1.0);
        let aspect = range(state, 0.5, 2.0);
        let params = VignetteParams::new(center, inner, outer, intensity, roundness, aspect);
        let uv = [range(state, 0.0, 1.0), range(state, 0.0, 1.0)];
        let dist = params.shape_distance(uv);
        // Reject any uv sitting within the margin of either band edge.
        let near_inner = (dist - inner).abs() < BAND_MARGIN;
        let near_outer = (dist - outer).abs() < BAND_MARGIN;
        if near_inner || near_outer {
            continue;
        }
        let color = [
            range(state, 0.0, 1.0),
            range(state, 0.0, 1.0),
            range(state, 0.0, 1.0),
        ];
        return VignetteMaskQuery::new(params, uv, color);
    }
}

/// Pins one `GPU` result against the `CPU` golden for `query`: the mask weight
/// and all three masked color channels must agree within bound.
fn pin(idx: usize, query: &VignetteMaskQuery, got: &VignetteMaskResult) {
    let want = golden(query);
    assert!(
        close(got.mask, want.mask),
        "query {idx} mask: gpu {} vs cpu {}",
        got.mask,
        want.mask
    );
    for ch in 0..3 {
        assert!(
            close(got.applied[ch], want.applied[ch]),
            "query {idx} applied[{ch}]: gpu {} vs cpu {}",
            got.applied[ch],
            want.applied[ch]
        );
    }
}

/// Dispatches `queries` on the `GPU` and pins every result against the
/// reference.
fn check(ctx: &GpuContext, gpu: &GpuVignetteMask, queries: &[VignetteMaskQuery]) {
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

/// A representative round `vignette`: centered, soft band, strong edge.
fn round_params() -> VignetteParams {
    VignetteParams::new([0.5, 0.5], 0.2, 0.5, 0.8, 1.0, 1.0)
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVignetteMask::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn center_is_fully_lit() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVignetteMask::new(&ctx);
    // At the optical center the distance is zero, well inside the inner radius,
    // so the mask is exactly 1.0 and the color passes through untouched.
    let query = VignetteMaskQuery::new(round_params(), [0.5, 0.5], [0.4, 0.6, 1.0]);
    check(&ctx, &gpu, &[query]);
}

#[test]
fn inside_inner_radius_is_fully_lit() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVignetteMask::new(&ctx);
    // A point 0.1 off-center on the x-axis is strictly inside the 0.2 inner
    // radius (a 0.1 margin), so it stays fully lit.
    let query = VignetteMaskQuery::new(round_params(), [0.6, 0.5], [0.2, 0.5, 0.9]);
    check(&ctx, &gpu, &[query]);
}

#[test]
fn corner_saturates_to_edge_value() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVignetteMask::new(&ctx);
    // The corner distance (~0.707) is well beyond the 0.5 outer radius, so the
    // mask saturates to 1 - intensity and the color is scaled by that factor.
    let query = VignetteMaskQuery::new(round_params(), [0.0, 0.0], [0.3, 0.7, 0.5]);
    check(&ctx, &gpu, &[query]);
}

#[test]
fn mid_band_point_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVignetteMask::new(&ctx);
    // On the x-axis the distance equals |uv.x - 0.5| = 0.35, squarely in the
    // mid-band (0.15 clear of each edge), so the smoothstep-then-rational
    // transition is exercised rather than a clamped endpoint.
    let query = VignetteMaskQuery::new(round_params(), [0.85, 0.5], [0.5, 0.5, 0.5]);
    check(&ctx, &gpu, &[query]);
}

#[test]
fn box_versus_round_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVignetteMask::new(&ctx);
    // A diagonal point under the box (roundness 0) and round (roundness 1)
    // metrics: the two take different shape distances, so both branches of the
    // roundness blend are pinned in one batch.
    let round = VignetteParams::new([0.5, 0.5], 0.2, 0.9, 0.8, 1.0, 1.0);
    let square = VignetteParams::new([0.5, 0.5], 0.2, 0.9, 0.8, 0.0, 1.0);
    let uv = [0.85, 0.85];
    let color = [0.6, 0.4, 0.2];
    check(
        &ctx,
        &gpu,
        &[
            VignetteMaskQuery::new(round, uv, color),
            VignetteMaskQuery::new(square, uv, color),
        ],
    );
}

#[test]
fn aspect_stretched_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVignetteMask::new(&ctx);
    // A 2:1 aspect scales the x-offset, so a point offset on x darkens faster
    // than the same offset on y; both axes are pinned.
    let params = VignetteParams::new([0.5, 0.5], 0.2, 0.5, 0.7, 1.0, 2.0);
    let color = [0.5, 0.5, 0.5];
    check(
        &ctx,
        &gpu,
        &[
            VignetteMaskQuery::new(params, [0.68, 0.5], color),
            VignetteMaskQuery::new(params, [0.5, 0.72], color),
        ],
    );
}

#[test]
fn apply_scales_color_by_mask() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVignetteMask::new(&ctx);
    // A fully-darkened corner and a fully-lit center with the same color pin the
    // per-channel multiply at both ends of the mask range.
    let params = round_params();
    let color = [0.4, 0.6, 1.0];
    check(
        &ctx,
        &gpu,
        &[
            VignetteMaskQuery::new(params, [0.0, 0.0], color),
            VignetteMaskQuery::new(params, [0.5, 0.5], color),
        ],
    );
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVignetteMask::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // One batch mixing deterministic fixtures with many random queries,
    // dispatched together so the per-thread indexing and the contiguous storage
    // layout are both exercised, then pinned element-for-element.
    let mut queries = vec![
        VignetteMaskQuery::new(round_params(), [0.5, 0.5], [0.4, 0.6, 1.0]),
        VignetteMaskQuery::new(round_params(), [0.0, 0.0], [0.3, 0.7, 0.5]),
        VignetteMaskQuery::new(round_params(), [0.85, 0.5], [0.5, 0.5, 0.5]),
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
    let gpu = GpuVignetteMask::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep (several workgroups' worth) pins the mask and masked color
    // across many random vignette parameterizations.
    let queries: Vec<VignetteMaskQuery> = (0..200).map(|_| rand_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
