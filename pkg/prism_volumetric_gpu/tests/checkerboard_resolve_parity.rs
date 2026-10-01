//! Real-device parity for the checkerboard-resolve twin:
//! [`GpuCheckerboardResolve`](prism_volumetric_gpu::checkerboard_resolve::GpuCheckerboardResolve)
//! must reproduce the `CPU` golden
//! [`resolve`](prism_render_architecture::particle::checkerboard_resolve::resolve)
//! across the whole weave. The fixtures cover an even image at half coverage,
//! odd dimensions, the frame-parity flip (`frame` `0` versus `1`), the three
//! characteristic history weights (`0` pure spatial, `1` pure clamped history,
//! `0.5` blend), random images at several resolutions and the border/corner
//! cells where the spatial fill and the neighbour box clamp onto whichever
//! neighbours exist. Every full-resolution pixel is compared on all four `RGBA`
//! channels.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each output pixel is a fixed, non-reorderable sequence of multiplies, adds,
//! `min` / `max` / `clamp` and one easing polynomial, so `CPU` and `GPU`
//! evaluate the same closed form in the same order. They are not bit-exact: a
//! `GPU` may fuse a multiply-add the scalar reference leaves separate,
//! perturbing the low mantissa bits by a few units in the last place. The
//! comparison therefore allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` — loose
//! enough to admit a legal fused multiply-add contraction, yet tight enough to
//! fail a genuinely wrong port (a swapped fill axis, a dropped neighbour, a
//! missing clamp, a wrong blend weight). The fixtures stay clear of the
//! `GRAD_EPS` edge-aware fallback threshold so no fixture straddles that
//! integer-free branch boundary.
//!
//! Provenance: standard spatial checkerboard reconstruction with a local
//! neighbour-box history clamp; no third-party engine source or derived code.

use prism_render_architecture::particle::checkerboard_resolve::{
    resolve, CheckerboardConfig, CheckerboardResolution, Rgba,
};
use prism_volumetric_gpu::checkerboard_resolve::{
    CheckerboardResolveQuery, GpuCheckerboardResolve,
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

/// Builds a `width * height` row-major image of pseudo-random `RGBA` quads in
/// `[0, scale)` drawn from `state`.
fn random_image(width: u32, height: u32, scale: f32, state: &mut u64) -> Vec<[f32; 4]> {
    let count = width as usize * height as usize;
    let mut pixels = Vec::with_capacity(count);
    for _ in 0..count {
        pixels.push([
            lcg(state) * scale,
            lcg(state) * scale,
            lcg(state) * scale,
            lcg(state) * scale,
        ]);
    }
    pixels
}

/// Converts the flat `[f32; 4]` layout into the reference `Rgba` buffer.
fn to_rgba(pixels: &[[f32; 4]]) -> Vec<Rgba> {
    pixels
        .iter()
        .map(|p| Rgba::new(p[0], p[1], p[2], p[3]))
        .collect()
}

/// Runs the `GPU` resolve and asserts pixel-for-pixel, channel-for-channel
/// parity against the `CPU` golden [`resolve`], returning the `GPU` result for
/// any extra per-test assertions.
fn check(
    ctx: &GpuContext,
    gpu: &GpuCheckerboardResolve,
    width: u32,
    height: u32,
    current: &[[f32; 4]],
    history: &[[f32; 4]],
    frame: u32,
    history_weight: f32,
) -> Vec<[f32; 4]> {
    let query = CheckerboardResolveQuery {
        full_width: width,
        full_height: height,
        current: current.to_vec(),
        history: history.to_vec(),
        frame,
        history_weight,
    };
    let got = gpu.eval(ctx, &query);

    let res = CheckerboardResolution::new(width, height);
    let want = resolve(
        res,
        &to_rgba(current),
        &to_rgba(history),
        frame,
        CheckerboardConfig::new(history_weight),
    );

    assert_eq!(
        got.len(),
        want.len(),
        "pixel count must match the reference"
    );
    for (idx, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let w = [w.r, w.g, w.b, w.a];
        for channel in 0..4 {
            assert!(
                close(g[channel], w[channel]),
                "pixel {idx} channel {channel}: gpu {} vs cpu {} \
                 ({width}x{height}, frame {frame}, weight {history_weight})",
                g[channel],
                w[channel]
            );
        }
    }
    got
}

#[test]
fn empty_image_short_circuits_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCheckerboardResolve::new(&ctx);
    // A zero-area image yields an empty vector and issues no dispatch (a storage
    // buffer cannot be zero-sized), exactly as the reference returns empty.
    let query = CheckerboardResolveQuery {
        full_width: 0,
        full_height: 0,
        current: Vec::new(),
        history: Vec::new(),
        frame: 0,
        history_weight: 0.5,
    };
    let got = gpu.eval(&ctx, &query);
    assert!(got.is_empty(), "a zero-area image stays empty");
}

#[test]
fn one_by_one_copies_shaded_and_falls_back_when_missing() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCheckerboardResolve::new(&ctx);
    let current = vec![[0.25, 0.5, 0.75, 1.0]];
    let history = vec![[0.9, 0.1, 0.4, 0.2]];
    // frame 0: the single cell is shaded and copied through.
    let shaded = check(&ctx, &gpu, 1, 1, &current, &history, 0, 1.0);
    assert!(close(shaded[0][0], 0.25), "shaded cell copies current");
    // frame 1: the single cell is missing and isolated; the fill, the box and
    // the clamped history all collapse onto the cell's own stored value.
    let missing = check(&ctx, &gpu, 1, 1, &current, &history, 1, 1.0);
    assert!(
        close(missing[0][0], 0.25),
        "an isolated missing cell falls back to its own value"
    );
}

#[test]
fn even_area_half_coverage_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCheckerboardResolve::new(&ctx);
    let mut state = 0x1357_9bdf_0246_8ace_u64;
    // An even 4x4 image shades exactly half its cells every frame; the other
    // half is rebuilt.
    let current = random_image(4, 4, 2.0, &mut state);
    let history = random_image(4, 4, 2.0, &mut state);
    for frame in 0..2u32 {
        for &weight in &[0.0f32, 0.5, 1.0] {
            check(&ctx, &gpu, 4, 4, &current, &history, frame, weight);
        }
    }
}

#[test]
fn odd_dimensions_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCheckerboardResolve::new(&ctx);
    let mut state = 0x2b7e_1516_28ae_d2a6_u64;
    // Odd extents give the two parities an unequal split that swaps with the
    // frame, and put many cells on the clamp-to-edge border.
    for &(w, h) in &[(3u32, 5u32), (5, 3), (7, 1), (1, 7)] {
        let current = random_image(w, h, 2.5, &mut state);
        let history = random_image(w, h, 2.5, &mut state);
        for frame in 0..2u32 {
            check(&ctx, &gpu, w, h, &current, &history, frame, 0.5);
        }
    }
}

#[test]
fn frame_parity_flip_rebuilds_the_other_half() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCheckerboardResolve::new(&ctx);
    let mut state = 0x3243_f6a8_885a_308d_u64;
    // The same fixture at frame 0 and frame 1 shades complementary cells; both
    // must match the reference, confirming the parity weave flips correctly.
    let current = random_image(6, 5, 3.0, &mut state);
    let history = random_image(6, 5, 3.0, &mut state);
    let a = check(&ctx, &gpu, 6, 5, &current, &history, 0, 0.5);
    let b = check(&ctx, &gpu, 6, 5, &current, &history, 1, 0.5);
    // The two frames shade opposite parities, so at least one interior pixel
    // must differ between them (the fixture is not vacuously constant).
    assert!(
        a.iter().zip(b.iter()).any(|(p, q)| !close(p[0], q[0])),
        "the two frame parities must reconstruct differently"
    );
}

#[test]
fn history_weight_zero_is_pure_spatial() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCheckerboardResolve::new(&ctx);
    let mut state = 0xa409_3822_299f_31d0_u64;
    // With weight 0 the history is ignored entirely; parity against the
    // reference pins the pure spatial rebuild. A deliberately wild history
    // confirms it never leaks in.
    let current = random_image(8, 6, 3.0, &mut state);
    let history = random_image(8, 6, 100.0, &mut state);
    check(&ctx, &gpu, 8, 6, &current, &history, 1, 0.0);
}

#[test]
fn history_weight_one_uses_box_clamped_history() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCheckerboardResolve::new(&ctx);
    // A missing interior cell with four known neighbours and an out-of-range
    // history must take the box-clamped history at weight 1. The neighbours
    // bracket [0.2, 0.6]; the history 9.0 clamps to 0.6, never leaking the raw
    // value. frame 1 makes the centre of a 3x3 missing.
    let lo = 0.2f32;
    let hi = 0.6f32;
    let mut current = vec![[0.0f32; 4]; 9];
    let res = CheckerboardResolution::new(3, 3);
    let idx = |x: u32, y: u32| (y * 3 + x) as usize;
    current[idx(0, 1)] = [lo, lo, lo, lo];
    current[idx(2, 1)] = [hi, hi, hi, hi];
    current[idx(1, 0)] = [lo, lo, lo, lo];
    current[idx(1, 2)] = [hi, hi, hi, hi];
    let mut history = vec![[0.0f32; 4]; 9];
    history[idx(1, 1)] = [9.0, 9.0, 9.0, 9.0];
    let got = check(&ctx, &gpu, 3, 3, &current, &history, 1, 1.0);
    let center = got[idx(1, 1)];
    for &value in &center {
        assert!(
            value <= hi + EPS,
            "clamped history must stay inside the neighbour box, got {value}"
        );
        assert!(
            value >= lo - EPS,
            "clamped history must stay inside the neighbour box, got {value}"
        );
    }
    // Sanity: the reference uses the same resolution, so the pixel count agrees.
    assert_eq!(got.len(), res.full_pixel_count());
}

#[test]
fn random_images_multiple_resolutions_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCheckerboardResolve::new(&ctx);
    let mut state = 0x5eed_4a7d_0bad_c0de_u64;
    // Sweep even, odd, tall, wide and prime resolutions, both frame parities and
    // several weights so the parity weave, the edge-aware fill, the neighbour
    // box and the blend are all exercised against random inputs.
    let resolutions = [
        (2u32, 2u32),
        (3, 7),
        (5, 5),
        (6, 10),
        (9, 4),
        (13, 11),
        (16, 9),
    ];
    for &(w, h) in &resolutions {
        for frame in 0..2u32 {
            let weight = 0.25 + lcg(&mut state) * 0.5;
            let current = random_image(w, h, 4.0, &mut state);
            let history = random_image(w, h, 4.0, &mut state);
            check(&ctx, &gpu, w, h, &current, &history, frame, weight);
        }
    }
}
