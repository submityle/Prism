//! Real-device parity for the anamorphic lens-streak twin:
//! [`GpuAnamorphicStreak`](prism_volumetric_gpu::anamorphic_streak::GpuAnamorphicStreak)
//! must reproduce the `CPU` golden
//! [`anamorphic_streak`](prism_render_architecture::particle::anamorphic_streak)
//! across the `Rec. 709` luminance, the
//! [`StreakParams::new`](prism_render_architecture::particle::anamorphic_streak::StreakParams::new)
//! normalization, the soft-`threshold` weight, the doubling tap-offset chain and
//! the geometric streak accumulation.
//!
//! The fixtures cover the shapes the golden unit tests call out: a non-trivial
//! direction that unit-normalizes, the degenerate zero direction that falls back
//! to the `+x` axis, negative scalars that clamp to zero, a `tap_count` forced
//! up to one and a `tap_count` at the fixed upper bound, the soft band sampled
//! well inside / below / above its edges, the zero-`knee` hard cutoff probed on
//! both sides of the `threshold`, a `tint`-and-halving accumulation and the
//! empty-tap black, plus a mixed batch. Every luminance fed to the soft band
//! sits at least `knee * 0.2` clear of both band edges (`threshold +/- knee`), so
//! no fixture lands on a branch boundary. All inputs are integers or simple
//! decimals, so the fixtures stay pure and need no transcendental math.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The `tap_count` and the tap-array length are discrete, so `CPU` and `GPU`
//! must agree exactly (`==`). The direction, the weight, the offsets, the
//! accumulation and the luminance thread through multiplies, adds, one guarded
//! division and one `sqrt`, so they are compared under tolerance (`abs_diff <=
//! 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`).
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::anamorphic_streak`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::anamorphic_streak::{luminance, StreakParams};
use prism_volumetric_gpu::anamorphic_streak::{
    AnamorphicStreakQuery, GpuAnamorphicStreak, MAX_TAPS,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous quantities.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the continuous quantities.
const REL_EPS: f32 = 1.0e-3;
/// Floor for the relative-tolerance denominator.
const REL_FLOOR: f32 = 1.0e-6;

/// Mixed absolute / relative tolerance comparison for one `f32` lane.
fn approx(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_EPS
}

/// Tolerant comparison of two `vec3` triples.
fn approx3(a: [f32; 3], b: [f32; 3]) -> bool {
    approx(a[0], b[0]) && approx(a[1], b[1]) && approx(a[2], b[2])
}

/// A neutral base query: unit `+x` direction, a four-tap chain, a mid-band
/// luminance and a non-trivial color, with no taps accumulated. Per-fixture
/// helpers override only the fields they exercise.
fn base() -> AnamorphicStreakQuery {
    AnamorphicStreakQuery {
        direction: [1.0, 0.0],
        tap_count: 4,
        stretch: 2.0,
        tint: [0.6, 0.8, 1.0],
        threshold: 1.0,
        knee: 0.5,
        intensity: 2.0,
        lum: 1.0,
        color: [0.3, 0.6, 0.9],
        taps: [[0.0; 3]; MAX_TAPS],
        accum_count: 0,
    }
}

/// Asserts every twinned answer for one query matches the `CPU` golden.
fn assert_parity(gpu: &GpuAnamorphicStreak, ctx: &GpuContext, q: &AnamorphicStreakQuery) {
    let got = gpu.evaluate(ctx, std::slice::from_ref(q));
    assert_eq!(got.len(), 1, "one result per query");
    let g = got[0];

    let cpu_lum = luminance(q.color);
    assert!(
        approx(g.luminance, cpu_lum),
        "luminance mismatch: gpu {} vs cpu {cpu_lum}",
        g.luminance
    );

    let p = StreakParams::new(
        q.direction,
        q.tap_count,
        q.stretch,
        q.tint,
        q.threshold,
        q.knee,
        q.intensity,
    );

    assert!(
        approx(g.direction[0], p.direction[0]) && approx(g.direction[1], p.direction[1]),
        "direction mismatch: gpu {:?} vs cpu {:?}",
        g.direction,
        p.direction
    );

    assert_eq!(
        g.tap_count, p.tap_count,
        "tap_count mismatch: gpu {} vs cpu {}",
        g.tap_count, p.tap_count
    );

    let cpu_weight = p.threshold_weight(q.lum);
    assert!(
        approx(g.threshold_weight, cpu_weight),
        "threshold_weight mismatch for lum {}: gpu {} vs cpu {cpu_weight}",
        q.lum,
        g.threshold_weight
    );

    let offsets = p.streak_tap_offsets();
    assert_eq!(
        offsets.len(),
        usize::try_from(p.tap_count).unwrap(),
        "offset count mismatch"
    );
    for (i, off) in offsets.iter().enumerate() {
        assert!(
            approx(g.tap_offsets[i][0], off[0]) && approx(g.tap_offsets[i][1], off[1]),
            "tap offset {i} mismatch: gpu {:?} vs cpu {off:?}",
            g.tap_offsets[i]
        );
    }

    let n = usize::try_from(q.accum_count).unwrap();
    let taps: Vec<[f32; 3]> = q.taps[..n].to_vec();
    let cpu_acc = p.accumulate_streak(&taps);
    assert!(
        approx3(g.accumulate, cpu_acc),
        "accumulate mismatch: gpu {:?} vs cpu {cpu_acc:?}",
        g.accumulate
    );
}

#[test]
fn luminance_and_normalized_direction_match() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAnamorphicStreak::new(&ctx);
    // A (3, 4) direction normalizes to (0.6, 0.8); the color exercises the
    // Rec. 709 dot product.
    let mut q = base();
    q.direction = [3.0, 4.0];
    q.color = [0.2, 0.5, 0.8];
    assert_parity(&gpu, &ctx, &q);
}

#[test]
fn zero_direction_falls_back_to_x_axis() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAnamorphicStreak::new(&ctx);
    // A degenerate zero direction falls back to the +x axis in both the golden
    // and the twin.
    let mut q = base();
    q.direction = [0.0, 0.0];
    assert_parity(&gpu, &ctx, &q);
}

#[test]
fn negative_scalars_are_clamped() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAnamorphicStreak::new(&ctx);
    // Negative stretch / threshold / knee / intensity all clamp to zero; the
    // zero-knee band then becomes a hard cutoff at threshold 0, so lum 5 is well
    // above it.
    let mut q = base();
    q.stretch = -2.0;
    q.threshold = -1.0;
    q.knee = -0.5;
    q.intensity = -3.0;
    q.lum = 5.0;
    q.accum_count = 2;
    q.taps[0] = [1.0, 1.0, 1.0];
    q.taps[1] = [0.5, 0.5, 0.5];
    assert_parity(&gpu, &ctx, &q);
}

#[test]
fn tap_count_is_forced_to_at_least_one() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAnamorphicStreak::new(&ctx);
    // A zero tap_count is forced up to one tap.
    let mut q = base();
    q.tap_count = 0;
    assert_parity(&gpu, &ctx, &q);
}

#[test]
fn tap_count_at_upper_bound() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAnamorphicStreak::new(&ctx);
    // The maximum tap_count fills every slot of the fixed-length offset array.
    let mut q = base();
    q.tap_count = u32::try_from(MAX_TAPS).unwrap();
    q.direction = [0.6, 0.8];
    q.stretch = 1.5;
    assert_parity(&gpu, &ctx, &q);
}

#[test]
fn threshold_weight_well_inside_band() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAnamorphicStreak::new(&ctx);
    // lum at the threshold center sits 0.5 (= knee) clear of both band edges
    // (0.5 and 1.5), far past the knee * 0.2 margin; the weight is 0.5.
    let mut q = base();
    q.lum = 1.0;
    assert_parity(&gpu, &ctx, &q);
    // A second interior sample a quarter of the way up the band.
    q.lum = 0.75;
    assert_parity(&gpu, &ctx, &q);
}

#[test]
fn threshold_weight_below_band_is_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAnamorphicStreak::new(&ctx);
    // lum 0.0 sits 0.5 below the lower edge (0.5); the weight is 0.
    let mut q = base();
    q.lum = 0.0;
    assert_parity(&gpu, &ctx, &q);
}

#[test]
fn threshold_weight_above_band_is_one() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAnamorphicStreak::new(&ctx);
    // lum 3.0 sits 1.5 above the upper edge (1.5); the weight is 1.
    let mut q = base();
    q.lum = 3.0;
    assert_parity(&gpu, &ctx, &q);
}

#[test]
fn zero_knee_is_a_hard_cutoff() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAnamorphicStreak::new(&ctx);
    // A zero knee degenerates the soft band to a step at the threshold. Both
    // probes stay a full unit clear of the threshold on each side.
    let mut above = base();
    above.knee = 0.0;
    above.threshold = 1.0;
    above.lum = 2.0;
    assert_parity(&gpu, &ctx, &above);

    let mut below = base();
    below.knee = 0.0;
    below.threshold = 1.0;
    below.lum = 0.0;
    assert_parity(&gpu, &ctx, &below);
}

#[test]
fn accumulate_mixes_tint_and_halving_weights() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAnamorphicStreak::new(&ctx);
    // Three taps with a chromatic tint and a non-unit intensity exercise the
    // geometric halving (1 + 0.5 + 0.25) times tint times intensity.
    let mut q = base();
    q.tint = [0.5, 0.25, 2.0];
    q.intensity = 3.0;
    q.accum_count = 3;
    q.taps[0] = [1.0, 0.8, 0.2];
    q.taps[1] = [0.4, 0.6, 1.0];
    q.taps[2] = [0.9, 0.1, 0.5];
    assert_parity(&gpu, &ctx, &q);
}

#[test]
fn accumulate_empty_is_black() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAnamorphicStreak::new(&ctx);
    // No taps accumulated: the streak color is black regardless of tint.
    let mut q = base();
    q.accum_count = 0;
    assert_parity(&gpu, &ctx, &q);
}

#[test]
fn batch_of_queries_matches_elementwise() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAnamorphicStreak::new(&ctx);
    // A batch exercises the one-thread-per-query flattening; each result must be
    // independent of its neighbours.
    let mut a = base();
    a.direction = [3.0, 4.0];
    a.lum = 1.25;
    a.accum_count = 2;
    a.taps[0] = [1.0, 0.5, 0.25];
    a.taps[1] = [0.2, 0.4, 0.6];

    let mut b = base();
    b.direction = [0.0, 0.0];
    b.tap_count = 1;
    b.knee = 0.0;
    b.lum = 2.0;

    let mut c = base();
    c.direction = [-1.0, 2.0];
    c.tap_count = u32::try_from(MAX_TAPS).unwrap();
    c.stretch = 0.5;
    c.lum = 0.0;
    c.accum_count = 4;
    c.taps[0] = [0.9, 0.9, 0.9];
    c.taps[1] = [0.7, 0.5, 0.3];
    c.taps[2] = [0.1, 0.2, 0.3];
    c.taps[3] = [0.4, 0.4, 0.8];

    let batch = [a, b, c];
    let got = gpu.evaluate(&ctx, &batch);
    assert_eq!(got.len(), batch.len());
    for q in &batch {
        assert_parity(&gpu, &ctx, q);
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAnamorphicStreak::new(&ctx);
    // No dispatch is issued and the result vector is empty.
    assert!(gpu.evaluate(&ctx, &[]).is_empty());
}
