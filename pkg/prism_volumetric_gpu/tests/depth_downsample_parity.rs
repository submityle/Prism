//! Real-device parity for the `2x2` depth-down-sample `mip`-chain twin:
//! [`GpuDepthDownsample`](prism_volumetric_gpu::depth_downsample::GpuDepthDownsample)
//! must reproduce the `CPU` golden
//! [`DepthMipChain::build`](prism_render_architecture::particle::depth_downsample::DepthMipChain::build)
//! texel for texel across degenerate bases, `1x1`, even and odd extents, every
//! [`ReduceOp`], and large random depth fields at several resolutions.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! `Min`, `Max` and `CheckerboardMinMax` are pure comparisons and exact copies,
//! so they are bit-identical on both engines; `Average` sums the same gathered
//! samples in the same row-major order and divides by the same integer count.
//! The comparison allows `abs_diff <= 1e-5` or `rel_diff <= 1e-5` — loose
//! enough to admit a legal fused multiply-add on the running sum, yet tight
//! enough to fail a wrong port (a dropped odd-boundary sample, a wrong
//! operator, a mis-sized level, a wrong checkerboard parity).
//!
//! Provenance: standard `2x2` reduction `mip` pyramid; no Unreal Engine source
//! or derived code.

use prism_render_architecture::particle::depth_downsample::{DepthMipChain, ReduceOp};
use prism_volumetric_gpu::depth_downsample::{DepthDownsampleQuery, GpuDepthDownsample};
use prism_volumetric_gpu::GpuContext;

/// Absolute/relative parity bound. A `GPU` may fuse a multiply-add the scalar
/// reference leaves separate on the `Average` running sum, perturbing the low
/// mantissa bits by a few units in the last place; `1e-5` admits that legal
/// slack while still failing a genuinely wrong port.
const EPS: f32 = 1.0e-5;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= EPS
}

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears. Returns a value in `[0, 1)`.
fn lcg(state: &mut u64) -> f32 {
    // Knuth multiplier / increment; the shift takes the high bits where the
    // generator mixes best.
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    // 24 usable mantissa bits mapped onto [0, 1).
    (bits & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// The reference packed chain: every level of
/// [`DepthMipChain::build`](prism_render_architecture::particle::depth_downsample::DepthMipChain::build)
/// concatenated back-to-back row-major, the exact layout the twin returns.
fn expected_packed(width: usize, height: usize, base: &[f32], op: ReduceOp) -> Vec<f32> {
    let chain = DepthMipChain::build(width, height, base, op).expect("valid base");
    let mut packed = Vec::with_capacity(chain.total_texel_count());
    for level in 0..chain.mip_count() {
        packed.extend_from_slice(chain.mip(level).expect("level in range"));
    }
    packed
}

/// Runs the `GPU` build and compares the whole packed chain against the `CPU`
/// golden level for level, returning the `GPU` chain for any extra assertions.
fn check(
    ctx: &GpuContext,
    gpu: &GpuDepthDownsample,
    width: usize,
    height: usize,
    base: &[f32],
    op: ReduceOp,
) -> Vec<f32> {
    let query = DepthDownsampleQuery {
        width,
        height,
        base,
        op,
    };
    let got = gpu.eval(ctx, &query).expect("valid base builds a chain");
    let want = expected_packed(width, height, base, op);
    assert_eq!(
        got.len(),
        want.len(),
        "packed chain length ({width}x{height}, {op:?})"
    );
    for (texel, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            close(g, w),
            "texel {texel}: gpu {g} vs cpu {w} ({width}x{height}, {op:?})"
        );
    }
    got
}

#[test]
fn degenerate_zero_dim_returns_none() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDepthDownsample::new(&ctx);
    // A zero dimension yields no chain and must not panic, matching the
    // reference `build` returning `None`.
    assert!(
        gpu.eval(
            &ctx,
            &DepthDownsampleQuery {
                width: 0,
                height: 4,
                base: &[],
                op: ReduceOp::Min,
            }
        )
        .is_none(),
        "zero width yields no chain"
    );
    assert!(
        gpu.eval(
            &ctx,
            &DepthDownsampleQuery {
                width: 4,
                height: 0,
                base: &[],
                op: ReduceOp::Max,
            }
        )
        .is_none(),
        "zero height yields no chain"
    );
}

#[test]
fn length_mismatch_returns_none() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDepthDownsample::new(&ctx);
    // 2x2 needs four texels; two is a mismatch, so no chain is built.
    assert!(
        gpu.eval(
            &ctx,
            &DepthDownsampleQuery {
                width: 2,
                height: 2,
                base: &[1.0, 2.0],
                op: ReduceOp::Average,
            }
        )
        .is_none(),
        "a base length mismatch yields no chain"
    );
}

#[test]
fn degenerate_1x1_single_level() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDepthDownsample::new(&ctx);
    // A 1x1 base is already its own top level: the chain is the single texel,
    // untouched, for every operator.
    for op in ReduceOp::all() {
        let got = check(&ctx, &gpu, 1, 1, &[7.0], op);
        assert_eq!(got.len(), 1, "a 1x1 chain has one texel ({op:?})");
        assert!(close(got[0], 7.0), "the lone texel passes through ({op:?})");
    }
}

#[test]
fn even_square_each_op() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDepthDownsample::new(&ctx);
    // 4x4 -> 2x2 -> 1x1: three levels, exact halving, no odd boundary.
    let base: Vec<f32> = (0..16_u16).map(f32::from).collect();
    for op in ReduceOp::all() {
        let got = check(&ctx, &gpu, 4, 4, &base, op);
        // 16 + 4 + 1 packed texels.
        assert_eq!(got.len(), 21, "4x4 chain packs 21 texels ({op:?})");
    }
    // The Max top level is the global maximum (15), the Min top level the
    // global minimum (0): pin the extremes the pyramid must preserve.
    let max_chain = check(&ctx, &gpu, 4, 4, &base, ReduceOp::Max);
    assert!(close(max_chain[20], 15.0), "max top is the global maximum");
    let min_chain = check(&ctx, &gpu, 4, 4, &base, ReduceOp::Min);
    assert!(close(min_chain[20], 0.0), "min top is the global minimum");
}

#[test]
fn odd_extent_mip_chain_each_op() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDepthDownsample::new(&ctx);
    // 5x3 -> 3x2 -> 2x1 -> 1x1: four levels, each step rounding an odd
    // dimension up and keeping its boundary row/column.
    let base: Vec<f32> = (0..15_u16).map(f32::from).collect();
    for op in ReduceOp::all() {
        let got = check(&ctx, &gpu, 5, 3, &base, op);
        // 15 + 6 + 2 + 1 packed texels.
        assert_eq!(got.len(), 24, "5x3 chain packs 24 texels ({op:?})");
    }
}

#[test]
fn odd_boundary_preserves_extreme_min() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDepthDownsample::new(&ctx);
    // Width 3 (odd): the lone right column becomes its own coarse texel and must
    // keep the tiny value there under Min — a dropped boundary sample would lose
    // it. `check` already asserts full parity; this pins the surviving minimum.
    let got = check(&ctx, &gpu, 3, 1, &[5.0, 5.0, 0.001], ReduceOp::Min);
    // 3x1 -> 2x1 -> 1x1: packed is base [5, 5, 0.001], then level 1
    // [min(5, 5), 0.001] = [5, 0.001], then level 2 [min(5, 0.001)] = [0.001].
    assert_eq!(got.len(), 6, "3x1 chain packs 3 + 2 + 1 texels");
    assert!(
        close(got[3], 5.0),
        "left coarse texel reduces the full pair"
    );
    assert!(
        close(got[4], 0.001),
        "the lone odd-boundary minimum becomes its own coarse texel"
    );
    assert!(
        close(got[5], 0.001),
        "the odd-boundary minimum survives to the 1x1 top"
    );
}

#[test]
fn checkerboard_alternates_min_max() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDepthDownsample::new(&ctx);
    // 4x2 base; the first reduction's destination is 2x1 -> texel (0,0) parity
    // even => Min, texel (1,0) parity odd => Max. `check` asserts full parity;
    // this pins the alternation the checkerboard operator encodes.
    let base = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let got = check(&ctx, &gpu, 4, 2, &base, ReduceOp::CheckerboardMinMax);
    // Packed: 8 base texels then the 2x1 level.
    assert!(close(got[8], 1.0), "even destination texel takes the min");
    assert!(close(got[9], 8.0), "odd destination texel takes the max");
}

#[test]
fn large_random_fields_match_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDepthDownsample::new(&ctx);
    let mut state = 0x5eed_dead_beef_0001_u64;

    // A spread of even and odd resolutions so every chain exercises both exact
    // halving and conservative odd-boundary rounding across many levels.
    let dims = [(1, 1), (2, 1), (3, 5), (7, 7), (16, 9), (31, 17), (64, 48)];
    for &(width, height) in &dims {
        let base: Vec<f32> = (0..width * height)
            .map(|_| lcg(&mut state) * 1000.0)
            .collect();
        for op in ReduceOp::all() {
            // `check` asserts texel-for-texel parity against the CPU golden
            // across the whole packed chain.
            let _ = check(&ctx, &gpu, width, height, &base, op);
        }
    }
}
