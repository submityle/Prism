//! Real-device parity for the hierarchical `Z`-buffer (`HZB`) occlusion twin:
//! [`GpuOcclusion`](prism_volumetric_gpu::occlusion::GpuOcclusion) must
//! reproduce the `CPU` golden
//! [`occlusion`](prism_render_architecture::particle::occlusion) per-query
//! accessors — `mip_count`, `mip_size`, `mip_texel_count`, `select_mip`, the
//! [`ScreenRect`] extents/validity/clamp,
//! [`is_occluded`](prism_render_architecture::particle::occlusion::is_occluded)
//! and
//! [`conservative_false_negative_bound`](prism_render_architecture::particle::occlusion::conservative_false_negative_bound)
//! — across an empty batch, power-of-two and non-power-of-two pyramids, a 1x1
//! pyramid, `mip`-size saturation, the covering-`mip` selection chain, a
//! strictly occluded candidate, a candidate in front, a valid and a degenerate
//! rectangle, a clamp into bounds, a clamp fully outside the viewport, the
//! false-negative bound at several `mip` levels and a large pseudo-random batch
//! compared lane for lane.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every integer output (`mip` count, `mip` size, texel count, selected `mip`,
//! the false-negative bound) and every discrete flag (rectangle validity and
//! the occlusion verdict) is asserted bit-exact, because the kernel runs the
//! same integer shifts and the same `>` comparisons the reference does. Only
//! the continuous rectangle extents and clamped edges allow a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`), since a `GPU` may round a
//! subtraction a few units in the last place differently from the scalar
//! reference. The fixtures (and the random batch) place depths well clear of
//! the [`CMP_EPS`](prism_render_architecture::particle::occlusion::CMP_EPS)
//! band and rectangle edges well clear of degeneracy, so the exact-flag
//! assertions are unconditional.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::occlusion`；
//! standard `Hi-Z` max-depth occlusion culling; no third-party engine source or
//! derived code.

use prism_render_architecture::particle::occlusion::{HzbPyramid, ScreenRect};
use prism_volumetric_gpu::occlusion::{cpu_reference, GpuOcclusion, GpuOcclusionQuery};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on the continuous rectangle extents and clamped edges.
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

/// A query carrying neutral, boundary-clear defaults; individual tests override
/// only the fields they probe. The pyramid is a large power-of-two, the
/// rectangle is a clearly valid unit square, the depths are far apart so the
/// occlusion verdict is unambiguous and every integer selector is in range.
fn base_query() -> GpuOcclusionQuery {
    GpuOcclusionQuery {
        pyramid: HzbPyramid::new(256, 256),
        rect: ScreenRect::new(0.0, 0.0, 1.0, 1.0),
        clamp_width: 128.0,
        clamp_height: 128.0,
        nearest_depth: 0.2,
        hzb_sampled_depth: 0.5,
        mip_query_level: 0,
        select_rect_w: 1,
        select_rect_h: 1,
        fnb_mip_level: 0,
    }
}

/// Runs the `GPU` dispatch and asserts strict lane-for-lane parity against the
/// `CPU` golden: all integer and discrete outputs match exactly, and the
/// continuous rectangle extents and clamped edges match within tolerance.
/// Returns the `GPU` verdicts for extra per-test assertions.
fn check(
    ctx: &GpuContext,
    gpu: &GpuOcclusion,
    queries: &[GpuOcclusionQuery],
) -> Vec<prism_volumetric_gpu::occlusion::GpuOcclusionResult> {
    let got = gpu.eval(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (lane, (g, q)) in got.iter().zip(queries.iter()).enumerate() {
        let want = cpu_reference(q);
        assert_eq!(g.mip_count, want.mip_count, "lane {lane}: mip_count");
        assert_eq!(g.mip_size, want.mip_size, "lane {lane}: mip_size");
        assert_eq!(
            g.mip_texel_count, want.mip_texel_count,
            "lane {lane}: mip_texel_count"
        );
        assert_eq!(
            g.selected_mip, want.selected_mip,
            "lane {lane}: selected_mip"
        );
        assert_eq!(g.rect_valid, want.rect_valid, "lane {lane}: rect_valid");
        assert_eq!(
            g.clamped_valid, want.clamped_valid,
            "lane {lane}: clamped_valid"
        );
        assert_eq!(g.occluded, want.occluded, "lane {lane}: occluded");
        assert_eq!(
            g.false_negative_bound, want.false_negative_bound,
            "lane {lane}: false_negative_bound"
        );
        assert!(
            close(g.rect_width, want.rect_width),
            "lane {lane}: rect_width gpu {} vs cpu {}",
            g.rect_width,
            want.rect_width
        );
        assert!(
            close(g.rect_height, want.rect_height),
            "lane {lane}: rect_height gpu {} vs cpu {}",
            g.rect_height,
            want.rect_height
        );
        assert!(
            close(g.clamped_rect.min_x, want.clamped_rect.min_x)
                && close(g.clamped_rect.min_y, want.clamped_rect.min_y)
                && close(g.clamped_rect.max_x, want.clamped_rect.max_x)
                && close(g.clamped_rect.max_y, want.clamped_rect.max_y),
            "lane {lane}: clamped_rect gpu {:?} vs cpu {:?}",
            g.clamped_rect,
            want.clamped_rect
        );
    }
    got
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

#[test]
fn empty_batch_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOcclusion::new(&ctx);
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch yields an empty result");
}

#[test]
fn mip_chain_power_of_two() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOcclusion::new(&ctx);
    // 256x256 -> 9 levels; mip 1 is 128x128 with 16384 texels.
    let q = GpuOcclusionQuery {
        pyramid: HzbPyramid::new(256, 256),
        mip_query_level: 1,
        ..base_query()
    };
    let got = check(&ctx, &gpu, &[q]);
    assert_eq!(got[0].mip_count, 9, "256x256 has 9 levels");
    assert_eq!(got[0].mip_size, (128, 128), "mip 1 halves each dimension");
    assert_eq!(got[0].mip_texel_count, 128 * 128);
}

#[test]
fn mip_chain_non_power_of_two_uses_max_edge() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOcclusion::new(&ctx);
    // max edge 100 -> 50 -> 25 -> 12 -> 6 -> 3 -> 1 == 7 levels.
    let q = GpuOcclusionQuery {
        pyramid: HzbPyramid::new(100, 37),
        mip_query_level: 2,
        ..base_query()
    };
    let got = check(&ctx, &gpu, &[q]);
    assert_eq!(got[0].mip_count, 7, "max-edge halving gives 7 levels");
    // mip 2: 100 >> 2 == 25, 37 >> 2 == 9.
    assert_eq!(got[0].mip_size, (25, 9));
    assert_eq!(got[0].mip_texel_count, 25 * 9);
}

#[test]
fn one_by_one_pyramid_is_single_level() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOcclusion::new(&ctx);
    // Zero base dims clamp up to 1x1, a single level.
    let q = GpuOcclusionQuery {
        pyramid: HzbPyramid::new(0, 0),
        mip_query_level: 5,
        select_rect_w: 999,
        select_rect_h: 999,
        ..base_query()
    };
    let got = check(&ctx, &gpu, &[q]);
    assert_eq!(got[0].mip_count, 1, "a 1x1 pyramid has one level");
    assert_eq!(got[0].mip_size, (1, 1), "every level saturates to 1x1");
    assert_eq!(got[0].selected_mip, 0, "select clamps to the only level");
}

#[test]
fn mip_size_saturates_past_last_level() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOcclusion::new(&ctx);
    let q = GpuOcclusionQuery {
        pyramid: HzbPyramid::new(8, 8),
        mip_query_level: 99,
        ..base_query()
    };
    let got = check(&ctx, &gpu, &[q]);
    assert_eq!(got[0].mip_size, (1, 1), "a level past the tail stays 1x1");
}

#[test]
fn select_mip_chain() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOcclusion::new(&ctx);
    let big = HzbPyramid::new(1024, 1024);
    let queries = [
        // span 4 -> 2 -> 1 == level 2.
        GpuOcclusionQuery {
            pyramid: big,
            select_rect_w: 4,
            select_rect_h: 3,
            ..base_query()
        },
        // span 8 -> 4 -> 2 -> 1 == level 3.
        GpuOcclusionQuery {
            pyramid: big,
            select_rect_w: 8,
            select_rect_h: 5,
            ..base_query()
        },
        // small rect stays at the base level.
        GpuOcclusionQuery {
            pyramid: HzbPyramid::new(256, 256),
            select_rect_w: 1,
            select_rect_h: 1,
            ..base_query()
        },
        // huge rect clamps to the last level of an 8x8 pyramid (mip_count-1 == 3).
        GpuOcclusionQuery {
            pyramid: HzbPyramid::new(8, 8),
            select_rect_w: 100_000,
            select_rect_h: 100_000,
            ..base_query()
        },
    ];
    let got = check(&ctx, &gpu, &queries);
    assert_eq!(got[0].selected_mip, 2);
    assert_eq!(got[1].selected_mip, 3);
    assert_eq!(got[2].selected_mip, 0);
    assert_eq!(got[3].selected_mip, 3);
}

#[test]
fn occluded_when_strictly_behind() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOcclusion::new(&ctx);
    // nearest 0.9 is far behind the sampled 0.5, so the candidate is occluded.
    let q = GpuOcclusionQuery {
        nearest_depth: 0.9,
        hzb_sampled_depth: 0.5,
        ..base_query()
    };
    let got = check(&ctx, &gpu, &[q]);
    assert!(
        got[0].occluded,
        "a strictly-behind candidate must be occluded"
    );
}

#[test]
fn visible_when_in_front() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOcclusion::new(&ctx);
    // nearest 0.2 is well in front of the sampled 0.5, so it stays visible.
    let q = GpuOcclusionQuery {
        nearest_depth: 0.2,
        hzb_sampled_depth: 0.5,
        ..base_query()
    };
    let got = check(&ctx, &gpu, &[q]);
    assert!(!got[0].occluded, "an in-front candidate must stay visible");
}

#[test]
fn screen_rect_valid_extents() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOcclusion::new(&ctx);
    let q = GpuOcclusionQuery {
        rect: ScreenRect::new(1.0, 2.0, 5.0, 8.0),
        ..base_query()
    };
    let got = check(&ctx, &gpu, &[q]);
    assert!(got[0].rect_valid, "a positive-area rect is valid");
    assert!(close(got[0].rect_width, 4.0), "width {}", got[0].rect_width);
    assert!(
        close(got[0].rect_height, 6.0),
        "height {}",
        got[0].rect_height
    );
}

#[test]
fn screen_rect_degenerate_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOcclusion::new(&ctx);
    // Zero width (min_x == max_x) is a degenerate rect: invalid, zero width.
    let q = GpuOcclusionQuery {
        rect: ScreenRect::new(3.0, 3.0, 3.0, 9.0),
        ..base_query()
    };
    let got = check(&ctx, &gpu, &[q]);
    assert!(!got[0].rect_valid, "a zero-width rect is invalid");
    assert!(close(got[0].rect_width, 0.0), "width {}", got[0].rect_width);
}

#[test]
fn clamp_into_bounds() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOcclusion::new(&ctx);
    let q = GpuOcclusionQuery {
        rect: ScreenRect::new(-5.0, -5.0, 200.0, 400.0),
        clamp_width: 128.0,
        clamp_height: 256.0,
        ..base_query()
    };
    let got = check(&ctx, &gpu, &[q]);
    let c = got[0].clamped_rect;
    assert!(close(c.min_x, 0.0) && close(c.min_y, 0.0));
    assert!(close(c.max_x, 128.0) && close(c.max_y, 256.0));
    assert!(got[0].clamped_valid, "the clamped rect still has area");
}

#[test]
fn clamp_fully_outside_collapses() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOcclusion::new(&ctx);
    // A rect wholly past the viewport collapses onto the far edge and is invalid.
    let q = GpuOcclusionQuery {
        rect: ScreenRect::new(500.0, 500.0, 900.0, 900.0),
        clamp_width: 128.0,
        clamp_height: 128.0,
        ..base_query()
    };
    let got = check(&ctx, &gpu, &[q]);
    assert!(
        !got[0].clamped_valid,
        "a rect fully outside the viewport collapses to an invalid rect"
    );
}

#[test]
fn false_negative_bound_grows_and_saturates() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOcclusion::new(&ctx);
    let levels = [0u32, 1, 4, 31, 255];
    let queries: Vec<GpuOcclusionQuery> = levels
        .iter()
        .map(|&lvl| GpuOcclusionQuery {
            fnb_mip_level: lvl,
            ..base_query()
        })
        .collect();
    let got = check(&ctx, &gpu, &queries);
    assert_eq!(got[0].false_negative_bound, 1);
    assert_eq!(got[1].false_negative_bound, 2);
    assert_eq!(got[2].false_negative_bound, 16);
    assert_eq!(got[3].false_negative_bound, u32::MAX);
    assert_eq!(got[4].false_negative_bound, u32::MAX);
}

#[test]
fn random_batch_matches_lane_for_lane() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOcclusion::new(&ctx);

    let mut state: u64 = 0x5eed_1234_abcd_9f01;
    let mut queries: Vec<GpuOcclusionQuery> = Vec::new();
    let mut saw_occluded = false;
    let mut saw_visible = false;
    let mut saw_valid = false;
    let mut saw_invalid = false;

    for _ in 0..4096 {
        // Base extent in 1..=1024 on each axis.
        let bw = 1 + (lcg(&mut state) * 1024.0) as u32;
        let bh = 1 + (lcg(&mut state) * 1024.0) as u32;

        // A rectangle somewhere in a 256-pixel viewport; occasionally force a
        // zero-width or zero-height degenerate rect (edges exactly equal is an
        // integer-clear boundary, never an f32 tie).
        let min_x = lcg(&mut state) * 200.0 - 20.0;
        let min_y = lcg(&mut state) * 200.0 - 20.0;
        let mut max_x = min_x + lcg(&mut state) * 60.0;
        let mut max_y = min_y + lcg(&mut state) * 60.0;
        let degen = (lcg(&mut state) * 5.0) as u32;
        if degen == 0 {
            max_x = min_x;
        } else if degen == 1 {
            max_y = min_y;
        }

        // Depths kept at least 1e-2 apart so the CMP_EPS verdict cannot flip.
        let sampled = lcg(&mut state);
        let delta = 0.01 + lcg(&mut state) * 0.5;
        let nearest = if (lcg(&mut state) * 2.0) as u32 == 0 {
            sampled + delta
        } else {
            sampled - delta
        };

        let q = GpuOcclusionQuery {
            pyramid: HzbPyramid::new(bw, bh),
            rect: ScreenRect::new(min_x, min_y, max_x, max_y),
            clamp_width: 128.0 + lcg(&mut state) * 128.0,
            clamp_height: 128.0 + lcg(&mut state) * 128.0,
            nearest_depth: nearest,
            hzb_sampled_depth: sampled,
            mip_query_level: (lcg(&mut state) * 12.0) as u32,
            select_rect_w: (lcg(&mut state) * 300.0) as u32,
            select_rect_h: (lcg(&mut state) * 300.0) as u32,
            fnb_mip_level: (lcg(&mut state) * 34.0) as u32,
        };

        let want = cpu_reference(&q);
        saw_occluded |= want.occluded;
        saw_visible |= !want.occluded;
        saw_valid |= want.rect_valid;
        saw_invalid |= !want.rect_valid;
        queries.push(q);
    }

    // `check` asserts full lane-for-lane parity against the CPU golden.
    check(&ctx, &gpu, &queries);

    assert!(
        saw_occluded && saw_visible,
        "random batch should mix occluded and visible verdicts"
    );
    assert!(
        saw_valid && saw_invalid,
        "random batch should mix valid and degenerate rectangles"
    );
}
