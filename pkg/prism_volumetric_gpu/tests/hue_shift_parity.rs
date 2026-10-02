//! Real-device parity for the hue/saturation/value colour twin:
//! [`GpuHueShift`](prism_volumetric_gpu::hue_shift::GpuHueShift) must reproduce
//! the `CPU` golden
//! [`hue_shift`](prism_render_architecture::particle::hue_shift) across the
//! `RGB` -> `HSV`/`HSL` conversions, the `HSV`/`HSL` -> `RGB` inverses, the hue
//! shift, the saturation/value/lightness scales and the `NTSC` `YIQ` chroma
//! rotation, pinned on the primary colours, a few deterministic fixtures and a
//! randomized batch compared element-for-element.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each transform is a fixed, non-reorderable sequence of multiplies, adds,
//! divides, `floor`, `abs`, `min` and `max`, so `CPU` and `GPU` evaluate the
//! same closed form in the same order. They are not bit-exact: a `GPU` may fuse
//! a multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The comparison therefore
//! allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` on every `f32` field.
//!
//! # Conditioning
//!
//! Every random fixture is kept well away from the two six-sextant degeneracy
//! cracks: the random `RGB` has its three channels pairwise separated by a
//! healthy margin, so no colour approaches the achromatic (`max == min`) grey
//! singularity nor sits on a hue-sector boundary where the max/min comparison
//! that selects the sextant branch could tip. The `HSV`/`HSL` inputs fed to the
//! inverse conversions keep their hue away from the integer sector boundaries
//! and their saturation/value/lightness comfortably inside `(0, 1)`. The
//! `(cos_d, sin_d)` pair is a unit-circle point taken from a rational
//! parameterisation, so the host supplies an exact cosine/sine with no
//! transcendental in the fixture. Because the three channels are packed
//! verbatim, both devices take the same comparison branch regardless of a few
//! units in the last place of slack.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::hue_shift`；
//! no third-party engine source or derived code.
#![forbid(unsafe_code)]

use prism_render_architecture::particle::hue_shift::{Hsl, Hsv, Rgb};
use prism_volumetric_gpu::hue_shift::{golden, GpuHueShift, HueShiftQuery, HueShiftResult};
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

/// Minimum pairwise channel separation for a random `RGB`, keeping each colour
/// clear of the grey singularity and the hue-sector boundaries.
const CHANNEL_GAP: f32 = 0.08;

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

/// Draws an `RGB` whose three channels are pairwise separated by at least
/// [`CHANNEL_GAP`], rejection-sampling so the colour never approaches the grey
/// singularity or a hue-sector boundary.
fn rand_rgb(state: &mut u64) -> Rgb {
    loop {
        let r = range(state, 0.1, 0.9);
        let g = range(state, 0.1, 0.9);
        let b = range(state, 0.1, 0.9);
        if (r - g).abs() >= CHANNEL_GAP
            && (g - b).abs() >= CHANNEL_GAP
            && (r - b).abs() >= CHANNEL_GAP
        {
            return Rgb::new(r, g, b);
        }
    }
}

/// Draws a hue in `[0, 6)` kept at least `0.2` away from the integer sector
/// boundaries, so the inverse conversion stays inside one piecewise branch.
fn rand_hue(state: &mut u64) -> f32 {
    let sector = (lcg(state) * 6.0) as u32 % 6;
    sector as f32 + range(state, 0.2, 0.8)
}

/// Builds a clearly-conditioned query: a well-separated `RGB`, an `HSV` and
/// `HSL` with hues away from the sector boundaries and mid-range
/// saturation/value/lightness, a bounded hue shift, a positive scale factor and
/// a unit-circle `(cos_d, sin_d)` pair from a rational parameterisation.
fn rand_query(state: &mut u64) -> HueShiftQuery {
    let rgb = rand_rgb(state);
    let hsv = Hsv::new(
        rand_hue(state),
        range(state, 0.2, 0.9),
        range(state, 0.2, 0.9),
    );
    let hsl = Hsl::new(
        rand_hue(state),
        range(state, 0.2, 0.9),
        range(state, 0.2, 0.9),
    );
    let delta_sextants = range(state, -3.0, 3.0);
    let factor = range(state, 0.0, 1.5);
    // A rational point on the unit circle (the t -> ((1-t^2)/(1+t^2),
    // 2t/(1+t^2)) parameterisation) gives an exact cosine/sine pair with no
    // transcendental in the fixture.
    let t = range(state, -1.0, 1.0);
    let denom = 1.0 + t * t;
    let cos_d = (1.0 - t * t) / denom;
    let sin_d = (2.0 * t) / denom;
    HueShiftQuery::new(rgb, hsv, hsl, delta_sextants, factor, cos_d, sin_d)
}

/// A well-conditioned deterministic query used wherever a particular transform
/// is under test and the other inputs just need to be clear of the guards.
fn baseline() -> HueShiftQuery {
    HueShiftQuery::new(
        Rgb::new(0.7, 0.4, 0.15),
        Hsv::new(2.4, 0.6, 0.75),
        Hsl::new(4.3, 0.55, 0.4),
        1.3,
        0.65,
        // cos(0) = 1, sin(0) = 0: the identity YIQ rotation.
        1.0,
        0.0,
    )
}

/// Pins one `GPU` result against the `CPU` golden for `query`: every component
/// of all nine transforms must agree within bound.
fn pin(idx: usize, query: &HueShiftQuery, got: &HueShiftResult) {
    let want = golden(query);
    let fields = [
        ("to_hsv.h", got.to_hsv.h, want.to_hsv.h),
        ("to_hsv.s", got.to_hsv.s, want.to_hsv.s),
        ("to_hsv.v", got.to_hsv.v, want.to_hsv.v),
        ("to_hsl.h", got.to_hsl.h, want.to_hsl.h),
        ("to_hsl.s", got.to_hsl.s, want.to_hsl.s),
        ("to_hsl.l", got.to_hsl.l, want.to_hsl.l),
        ("from_hsv.r", got.from_hsv.r, want.from_hsv.r),
        ("from_hsv.g", got.from_hsv.g, want.from_hsv.g),
        ("from_hsv.b", got.from_hsv.b, want.from_hsv.b),
        ("from_hsl.r", got.from_hsl.r, want.from_hsl.r),
        ("from_hsl.g", got.from_hsl.g, want.from_hsl.g),
        ("from_hsl.b", got.from_hsl.b, want.from_hsl.b),
        ("shifted.r", got.shifted.r, want.shifted.r),
        ("shifted.g", got.shifted.g, want.shifted.g),
        ("shifted.b", got.shifted.b, want.shifted.b),
        ("saturated.r", got.saturated.r, want.saturated.r),
        ("saturated.g", got.saturated.g, want.saturated.g),
        ("saturated.b", got.saturated.b, want.saturated.b),
        ("valued.r", got.valued.r, want.valued.r),
        ("valued.g", got.valued.g, want.valued.g),
        ("valued.b", got.valued.b, want.valued.b),
        ("lightened.r", got.lightened.r, want.lightened.r),
        ("lightened.g", got.lightened.g, want.lightened.g),
        ("lightened.b", got.lightened.b, want.lightened.b),
        ("yiq_rotated.r", got.yiq_rotated.r, want.yiq_rotated.r),
        ("yiq_rotated.g", got.yiq_rotated.g, want.yiq_rotated.g),
        ("yiq_rotated.b", got.yiq_rotated.b, want.yiq_rotated.b),
    ];
    for (name, g, c) in fields {
        assert!(close(g, c), "query {idx} {name}: gpu {g} vs cpu {c}");
    }
}

/// Dispatches `queries` on the `GPU` and pins every result against the
/// reference.
fn check(ctx: &GpuContext, gpu: &GpuHueShift, queries: &[HueShiftQuery]) {
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
    let gpu = GpuHueShift::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn primary_colours_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHueShift::new(&ctx);
    // The three primaries sit at hue 0, 2 and 4; each pins the RGB <-> HSV/HSL
    // conversions and the shift against a known sextant centre.
    let queries = [
        HueShiftQuery::new(
            Rgb::new(0.9, 0.1, 0.2),
            Hsv::new(0.5, 0.8, 0.9),
            Hsl::new(0.5, 0.7, 0.45),
            2.0,
            0.5,
            1.0,
            0.0,
        ),
        HueShiftQuery::new(
            Rgb::new(0.2, 0.9, 0.1),
            Hsv::new(2.5, 0.8, 0.9),
            Hsl::new(2.5, 0.7, 0.45),
            2.0,
            0.5,
            1.0,
            0.0,
        ),
        HueShiftQuery::new(
            Rgb::new(0.1, 0.2, 0.9),
            Hsv::new(4.5, 0.8, 0.9),
            Hsl::new(4.5, 0.7, 0.45),
            2.0,
            0.5,
            1.0,
            0.0,
        ),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn achromatic_grey_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHueShift::new(&ctx);
    // A pure grey takes the achromatic guard (chroma delta = 0) on both devices,
    // pinning the hue-0, saturation-0 fallback and the value/lightness guards.
    let query = HueShiftQuery::new(
        Rgb::new(0.4, 0.4, 0.4),
        Hsv::new(0.0, 0.0, 0.0),
        Hsl::new(0.0, 0.0, 0.0),
        1.0,
        0.5,
        1.0,
        0.0,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn baseline_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHueShift::new(&ctx);
    // A single baseline colour exercises every transform at once.
    check(&ctx, &gpu, &[baseline()]);
}

#[test]
fn yiq_quarter_turn_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHueShift::new(&ctx);
    // cos(90 deg) = 0, sin(90 deg) = 1: a quarter-turn YIQ rotation supplied as
    // an exact rational pair, pinning the chroma-plane rotation.
    let query = HueShiftQuery::new(
        Rgb::new(0.6, 0.3, 0.2),
        Hsv::new(1.4, 0.7, 0.8),
        Hsl::new(3.3, 0.6, 0.5),
        0.0,
        1.0,
        0.0,
        1.0,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHueShift::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // One batch mixing deterministic fixtures with many random colours,
    // dispatched together so the per-thread indexing and the contiguous storage
    // layout are both exercised, then pinned element-for-element.
    let mut queries = vec![baseline()];
    for _ in 0..48 {
        queries.push(rand_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn many_colours_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHueShift::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep (several workgroups' worth) pins every transform across
    // many random colours.
    let queries: Vec<HueShiftQuery> = (0..200).map(|_| rand_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
