//! Real-device parity for the `sRGB` `gamut`-clip twin:
//! [`GpuGamutClip`](prism_volumetric_gpu::gamut_clip::GpuGamutClip) must
//! reproduce the `CPU` golden
//! [`gamut_clip`](prism_render_architecture::particle::gamut_clip) routines
//! across all three strategies — the per-channel
//! [`clip_naive`](prism_render_architecture::particle::gamut_clip::clip_naive),
//! the equal-luma
//! [`clip_preserve_luma`](prism_render_architecture::particle::gamut_clip::clip_preserve_luma)
//! and the knee
//! [`soft_clip`](prism_render_architecture::particle::gamut_clip::soft_clip) —
//! over interior, overexposed, negative, mixed and neutral-gray colors, plus a
//! random sweep, each compared channel for channel.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernels are portable core-`WGSL`, so they need no optional device
//! feature.
//!
//! # Parity criterion
//!
//! Each color is a fixed, non-reorderable sequence of comparisons, clamps,
//! weighted adds and at most one divide, so `CPU` and `GPU` evaluate the same
//! closed form in the same order. They are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The comparison therefore
//! allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` — loose enough to admit a
//! legal fused multiply-add contraction, yet tight enough to fail a genuinely
//! wrong port (a dropped luma weight, a wrong knee sign, a missing wall clamp).
//! Fixtures deliberately avoid the soft-knee boundary ties and the
//! preserve-luma divide-degenerate region so neither side straddles a branch.
//!
//! Provenance: standard `sRGB` `gamut` clip (per-channel clamp, equal-luma
//! desaturation, monotone rational soft knee); no third-party engine source or
//! derived code.

use prism_render_architecture::particle::gamut_clip::{
    clip_naive, clip_preserve_luma, luma_rec709, soft_clip, Rgb,
};
use prism_volumetric_gpu::gamut_clip::{GamutClipMode, GamutClipQuery, GpuGamutClip};
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

/// Applies the `CPU` golden routine for `mode` to one color.
fn reference(c: Rgb, mode: GamutClipMode, knee: f32) -> Rgb {
    match mode {
        GamutClipMode::Naive => clip_naive(c),
        GamutClipMode::PreserveLuma => clip_preserve_luma(c),
        GamutClipMode::Soft => soft_clip(c, knee),
    }
}

/// Runs the `GPU` `gamut` clip and asserts channel-for-channel parity against
/// the `CPU` golden for `mode`, returning the `GPU` result for extra checks.
fn check(
    ctx: &GpuContext,
    gpu: &GpuGamutClip,
    colors: &[Rgb],
    mode: GamutClipMode,
    knee: f32,
) -> Vec<Rgb> {
    let query = GamutClipQuery {
        colors: colors.to_vec(),
        mode,
        knee,
    };
    let got = gpu.eval(ctx, &query);

    assert_eq!(
        got.len(),
        colors.len(),
        "result length must match the input"
    );

    for (idx, (g, &c)) in got.iter().zip(colors.iter()).enumerate() {
        let w = reference(c, mode, knee);
        assert!(
            close(g.r, w.r) && close(g.g, w.g) && close(g.b, w.b),
            "color {idx} ({mode:?}, knee {knee}): gpu {g:?} vs cpu {w:?}"
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

/// The shared fixture set: interior, neutral gray, overexposed, negative and
/// mixed over/under colors. Chosen so no channel lands on a soft-knee boundary
/// for the knees used below and no preserve-luma input hits the divide guard.
fn fixtures() -> Vec<Rgb> {
    vec![
        Rgb::new(0.3, 0.5, 0.7),
        Rgb::new(0.1, 0.4, 0.9),
        Rgb::gray(0.5),
        Rgb::gray(0.25),
        Rgb::new(1.6, 0.1, 0.1),
        Rgb::new(1.2, 0.82, 0.3),
        Rgb::new(5.0, 10.0, 100.0),
        Rgb::new(-0.3, 0.5, 0.6),
        Rgb::new(-5.0, -10.0, -100.0),
        Rgb::new(1.5, -0.2, 0.4),
    ]
}

#[test]
fn empty_input_is_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGamutClip::new(&ctx);
    // An empty color slice short-circuits: a storage buffer cannot be
    // zero-sized, so no dispatch is issued and the result is empty.
    for mode in [
        GamutClipMode::Naive,
        GamutClipMode::PreserveLuma,
        GamutClipMode::Soft,
    ] {
        let out = check(&ctx, &gpu, &[], mode, 0.2);
        assert!(out.is_empty(), "an empty input stays empty");
    }
}

#[test]
fn naive_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGamutClip::new(&ctx);
    let colors = fixtures();
    // The naive clip ignores the knee; pass an arbitrary value to prove it.
    let got = check(&ctx, &gpu, &colors, GamutClipMode::Naive, 0.2);
    for c in &got {
        assert!(c.r >= -EPS && c.r <= 1.0 + EPS, "red lands in gamut");
        assert!(c.g >= -EPS && c.g <= 1.0 + EPS, "green lands in gamut");
        assert!(c.b >= -EPS && c.b <= 1.0 + EPS, "blue lands in gamut");
    }
}

#[test]
fn preserve_luma_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGamutClip::new(&ctx);
    let colors = fixtures();
    check(&ctx, &gpu, &colors, GamutClipMode::PreserveLuma, 0.2);
}

#[test]
fn preserve_luma_keeps_luma_when_displayable() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGamutClip::new(&ctx);
    // Inputs whose own luma is inside 0..=1: the equal-luma gray is displayable,
    // so the mapped color must keep that luma (the invariant the strategy sells)
    // on the device as well as on the host.
    let colors = vec![
        Rgb::new(1.6, 0.1, 0.1),
        Rgb::new(-0.3, 0.5, 0.6),
        Rgb::new(1.5, -0.2, 0.4),
        Rgb::new(1.2, 0.82, 0.3),
    ];
    let got = check(&ctx, &gpu, &colors, GamutClipMode::PreserveLuma, 0.2);
    for (g, &c) in got.iter().zip(colors.iter()) {
        let before = luma_rec709(&c);
        if (-REL_FLOOR..=1.0 + REL_FLOOR).contains(&before) {
            assert!(
                close(luma_rec709(g), before),
                "preserve-luma must keep luma: {before} vs {}",
                luma_rec709(g)
            );
        }
    }
}

#[test]
fn soft_matches_reference_across_knees() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGamutClip::new(&ctx);
    let colors = fixtures();
    // Two knees whose boundaries (knee and 1 - knee) avoid every fixture channel
    // so neither side straddles the roll-off branch.
    for &knee in &[0.17_f32, 0.23_f32] {
        let got = check(&ctx, &gpu, &colors, GamutClipMode::Soft, knee);
        for c in &got {
            assert!(c.r > -EPS && c.r < 1.0 + EPS, "red rolls into gamut");
            assert!(c.g > -EPS && c.g < 1.0 + EPS, "green rolls into gamut");
            assert!(c.b > -EPS && c.b < 1.0 + EPS, "blue rolls into gamut");
        }
    }
}

#[test]
fn random_sweep_matches_reference_for_every_mode() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGamutClip::new(&ctx);
    let mut state = 0x5eed_4a7d_0bad_c0de_u64;
    // Random colors spanning the domain (`[-1.5, 2.5)` per channel) exercise the
    // interior, both walls and the undisplayable-luma fallback.
    let mut colors = Vec::with_capacity(96);
    for _ in 0..96 {
        let ch = |s: &mut u64| lcg(s) * 4.0 - 1.5;
        colors.push(Rgb::new(ch(&mut state), ch(&mut state), ch(&mut state)));
    }
    // The naive clamp and the soft knee are continuous everywhere, so no random
    // input can straddle a branch between host and device. Preserve-luma has one
    // discontinuous fallback at the luma boundary, so drop colors whose luma sits
    // in a thin band around 0 or 1 before comparing that mode.
    let margin = 1.0e-2_f32;
    let luma_safe: Vec<Rgb> = colors
        .iter()
        .copied()
        .filter(|c| {
            let l = luma_rec709(c);
            (l > margin && l < 1.0 - margin) || l < -margin || l > 1.0 + margin
        })
        .collect();
    for mode in [
        GamutClipMode::Naive,
        GamutClipMode::PreserveLuma,
        GamutClipMode::Soft,
    ] {
        let knee = 0.05 + lcg(&mut state) * 0.4;
        let set = if mode == GamutClipMode::PreserveLuma {
            luma_safe.as_slice()
        } else {
            colors.as_slice()
        };
        check(&ctx, &gpu, set, mode, knee);
    }
}
