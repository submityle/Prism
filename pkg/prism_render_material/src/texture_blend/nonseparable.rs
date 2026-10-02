//! **Non-separable blend modes** (W3C Compositing and Blending Level 1 / PDF
//! 1.7): Hue, Saturation, Color and Luminosity.
//!
//! Unlike the separable modes in [`modes`](super::modes), these cannot be
//! evaluated one channel at a time: they move whole-colour attributes (hue,
//! saturation, luminosity) between the backdrop `cb` and source `cs`, so they
//! operate on the RGB triple as a unit. They are the standard primitive behind
//! colourising, tinting and recolour workflows (e.g. re-tinting an albedo under
//! a decal while preserving the backdrop's shading) and complete the W3C blend
//! family alongside the separable modes.
//!
//! The four modes are built from the spec's helper functions:
//! * `Lum(C) = 0.3 R + 0.59 G + 0.11 B` -- the perceptual luminosity;
//! * `Sat(C) = max(R,G,B) - min(R,G,B)` -- the chroma range;
//! * `SetLum(C, l)` shifts all channels by `l - Lum(C)` then `ClipColor`
//!   projects any out-of-gamut channel back into `[0, 1]` **about the
//!   luminosity** so `Lum` is preserved;
//! * `SetSat(C, s)` linearly remaps the min/mid/max channels so the chroma
//!   range becomes `s` with min pinned to `0`.
//!
//! The modes are then:
//! * `Hue        = SetLum(SetSat(cs, Sat(cb)), Lum(cb))`;
//! * `Saturation = SetLum(SetSat(cb, Sat(cs)), Lum(cb))`;
//! * `Color      = SetLum(cs, Lum(cb))`;
//! * `Luminosity = SetLum(cb, Lum(cs))` ( = `Color` with layers swapped).
//!
//! Colours are combined in scene-linear light (so the result matches a GPU
//! layer blend); the [`blend_nonseparable_rgba8`] compositor mixes the blended
//! colour toward the backdrop by a scalar `opacity` and keeps the backdrop
//! alpha, mirroring [`blend_rgba8`](super::blend_rgba8).
//!
//! Everything is deterministic analytic `f32` arithmetic -- no AI/ML.
//!
//! # References
//! * W3C, "Compositing and Blending Level 1" (non-separable blend modes).
//! * Adobe, PDF 1.7 reference, 7.2.4.

use alloc::vec::Vec;

use crate::{linear_to_srgb, srgb_to_linear, ColorSpace, Rgba8Image};

/// A non-separable (whole-colour) blend mode `B(cb, cs) -> rgb`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NonSeparableBlendMode {
    /// Hue of the source with the saturation and luminosity of the backdrop.
    Hue,
    /// Saturation of the source with the hue and luminosity of the backdrop.
    Saturation,
    /// Hue and saturation of the source with the luminosity of the backdrop.
    Color,
    /// Luminosity of the source with the hue and saturation of the backdrop.
    Luminosity,
}

/// Perceptual luminosity `0.3 R + 0.59 G + 0.11 B` (W3C weights).
#[inline]
#[must_use]
fn lum(c: [f32; 3]) -> f32 {
    0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2]
}

/// Chroma range `max - min`.
#[inline]
#[must_use]
fn sat(c: [f32; 3]) -> f32 {
    c[0].max(c[1]).max(c[2]) - c[0].min(c[1]).min(c[2])
}

/// Project an out-of-`[0, 1]` colour back into gamut about its luminosity so
/// `Lum` is preserved (W3C `ClipColor`).
#[inline]
#[must_use]
fn clip_color(c: [f32; 3]) -> [f32; 3] {
    let l = lum(c);
    let n = c[0].min(c[1]).min(c[2]);
    let x = c[0].max(c[1]).max(c[2]);
    let mut out = c;
    if n < 0.0 {
        let d = l - n;
        if d != 0.0 {
            for v in &mut out {
                *v = l + (*v - l) * l / d;
            }
        }
    }
    if x > 1.0 {
        let d = x - l;
        if d != 0.0 {
            for v in &mut out {
                *v = l + (*v - l) * (1.0 - l) / d;
            }
        }
    }
    out
}

/// Set the luminosity of `c` to `l` (shift all channels, then `ClipColor`).
#[inline]
#[must_use]
fn set_lum(c: [f32; 3], l: f32) -> [f32; 3] {
    let d = l - lum(c);
    clip_color([c[0] + d, c[1] + d, c[2] + d])
}

/// Set the chroma range of `c` to `s`, pinning the min channel to `0` and
/// linearly remapping the mid channel (W3C `SetSat`).
#[inline]
#[must_use]
fn set_sat(c: [f32; 3], s: f32) -> [f32; 3] {
    // Identify min / mid / max channel indices (stable for ties).
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&a, &b| {
        c[a].partial_cmp(&c[b])
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    let (imin, imid, imax) = (idx[0], idx[1], idx[2]);
    let mut out = [0.0f32; 3];
    let range = c[imax] - c[imin];
    if range > 0.0 {
        out[imid] = (c[imid] - c[imin]) * s / range;
        out[imax] = s;
    } else {
        out[imid] = 0.0;
        out[imax] = 0.0;
    }
    out[imin] = 0.0;
    out
}

/// Evaluate a non-separable blend mode on scene-linear RGB triples.
#[must_use]
pub fn blend_nonseparable(mode: NonSeparableBlendMode, cb: [f32; 3], cs: [f32; 3]) -> [f32; 3] {
    match mode {
        NonSeparableBlendMode::Hue => set_lum(set_sat(cs, sat(cb)), lum(cb)),
        NonSeparableBlendMode::Saturation => set_lum(set_sat(cb, sat(cs)), lum(cb)),
        NonSeparableBlendMode::Color => set_lum(cs, lum(cb)),
        NonSeparableBlendMode::Luminosity => set_lum(cb, lum(cs)),
    }
}

/// Composite a `top` layer over a `base` backdrop with a non-separable blend
/// `mode` and scalar `opacity` in `[0, 1]`, gamma-correctly under `space`.
///
/// Colours are blended in scene-linear light; the blended colour is linearly
/// mixed toward the backdrop by `opacity`. Alpha is taken from the backdrop.
/// Returns `None` when the two images differ in size or are empty.
#[must_use]
pub fn blend_nonseparable_rgba8(
    base: &Rgba8Image,
    top: &Rgba8Image,
    mode: NonSeparableBlendMode,
    opacity: f32,
    space: ColorSpace,
) -> Option<Rgba8Image> {
    let (w, h) = (base.width(), base.height());
    if w == 0 || h == 0 || top.width() != w || top.height() != h {
        return None;
    }
    let t = opacity.clamp(0.0, 1.0);
    let bb = base.as_slice();
    let ts = top.as_slice();

    let lift = |c: u8| -> f32 {
        match space {
            ColorSpace::Linear => f32::from(c) / 255.0,
            ColorSpace::Srgb => srgb_to_linear(c),
        }
    };
    let round_u8 = |x: f32| -> u8 {
        let cc = if x.is_nan() { 0.0 } else { x.clamp(0.0, 1.0) };
        (cc * 255.0 + 0.5).floor() as u8
    };
    let encode = |lin: f32| -> u8 {
        match space {
            ColorSpace::Linear => round_u8(lin),
            ColorSpace::Srgb => linear_to_srgb(lin),
        }
    };

    let mut out = Vec::with_capacity((w * h) as usize);
    for (b, s) in bb.iter().zip(ts.iter()) {
        let cb = [lift(b[0]), lift(b[1]), lift(b[2])];
        let cs = [lift(s[0]), lift(s[1]), lift(s[2])];
        let blended = blend_nonseparable(mode, cb, cs);
        let mut px = [0u8; 4];
        for k in 0..3 {
            px[k] = encode(cb[k] + t * (blended[k] - cb[k]));
        }
        px[3] = b[3];
        out.push(px);
    }
    Rgba8Image::new(w, h, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    const EPS: f32 = 1.0e-5;

    fn lum_ref(c: [f32; 3]) -> f32 {
        0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2]
    }
    fn sat_ref(c: [f32; 3]) -> f32 {
        let mx = c[0].max(c[1]).max(c[2]);
        let mn = c[0].min(c[1]).min(c[2]);
        mx - mn
    }

    fn colours() -> Vec<[f32; 3]> {
        [
            [0.2, 0.6, 0.1],
            [0.9, 0.3, 0.4],
            [0.05, 0.5, 0.95],
            [0.7, 0.7, 0.2],
            [0.1, 0.1, 0.1],
            [0.95, 0.9, 0.85],
        ]
        .to_vec()
    }

    #[test]
    fn color_preserves_backdrop_luminosity() {
        for cb in colours() {
            for cs in colours() {
                let out = blend_nonseparable(NonSeparableBlendMode::Color, cb, cs);
                assert!(
                    (lum_ref(out) - lum_ref(cb)).abs() < EPS,
                    "cb={cb:?} cs={cs:?}"
                );
            }
        }
    }

    #[test]
    fn luminosity_takes_source_luminosity() {
        for cb in colours() {
            for cs in colours() {
                let out = blend_nonseparable(NonSeparableBlendMode::Luminosity, cb, cs);
                assert!(
                    (lum_ref(out) - lum_ref(cs)).abs() < EPS,
                    "cb={cb:?} cs={cs:?}"
                );
            }
        }
    }

    #[test]
    fn luminosity_is_color_with_layers_swapped() {
        for cb in colours() {
            for cs in colours() {
                let a = blend_nonseparable(NonSeparableBlendMode::Luminosity, cb, cs);
                let b = blend_nonseparable(NonSeparableBlendMode::Color, cs, cb);
                for k in 0..3 {
                    assert!((a[k] - b[k]).abs() < EPS, "cb={cb:?} cs={cs:?}");
                }
            }
        }
    }

    #[test]
    fn hue_keeps_backdrop_lum_and_sat() {
        for cb in colours() {
            for cs in colours() {
                let out = blend_nonseparable(NonSeparableBlendMode::Hue, cb, cs);
                assert!(
                    (lum_ref(out) - lum_ref(cb)).abs() < EPS,
                    "lum cb={cb:?} cs={cs:?}"
                );
                // SetSat gives the backdrop's chroma range; SetLum/ClipColor can
                // only shrink it when a channel clips, so sat(out) <= sat(cb)+eps.
                assert!(
                    sat_ref(out) <= sat_ref(cb) + 1.0e-4,
                    "sat cb={cb:?} cs={cs:?}"
                );
            }
        }
    }

    #[test]
    fn saturation_keeps_backdrop_lum_and_takes_source_sat() {
        for cb in colours() {
            for cs in colours() {
                let out = blend_nonseparable(NonSeparableBlendMode::Saturation, cb, cs);
                assert!(
                    (lum_ref(out) - lum_ref(cb)).abs() < EPS,
                    "lum cb={cb:?} cs={cs:?}"
                );
                assert!(
                    sat_ref(out) <= sat_ref(cs) + 1.0e-4,
                    "sat cb={cb:?} cs={cs:?}"
                );
            }
        }
    }

    #[test]
    fn achromatic_source_color_is_grey_at_backdrop_lum() {
        // Color(cb, grey) sets a flat (zero-chroma) source to the backdrop's
        // luminosity, so the result is the achromatic colour (L, L, L).
        for cb in colours() {
            for g in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
                let out = blend_nonseparable(NonSeparableBlendMode::Color, cb, [g, g, g]);
                let l = lum_ref(cb);
                for k in 0..3 {
                    assert!((out[k] - l).abs() < EPS, "cb={cb:?} g={g} out={out:?}");
                }
            }
        }
    }

    #[test]
    fn output_within_unit_range() {
        for cb in colours() {
            for cs in colours() {
                for mode in [
                    NonSeparableBlendMode::Hue,
                    NonSeparableBlendMode::Saturation,
                    NonSeparableBlendMode::Color,
                    NonSeparableBlendMode::Luminosity,
                ] {
                    let out = blend_nonseparable(mode, cb, cs);
                    for k in 0..3 {
                        assert!(
                            out[k] >= -1.0e-4 && out[k] <= 1.0 + 1.0e-4,
                            "{mode:?} {out:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn compositor_opacity_zero_is_backdrop() {
        let base = Rgba8Image::new(2, 2, vec![[10u8, 120, 240, 200]; 4]).unwrap();
        let top = Rgba8Image::new(2, 2, vec![[200u8, 50, 90, 30]; 4]).unwrap();
        let out = blend_nonseparable_rgba8(
            &base,
            &top,
            NonSeparableBlendMode::Color,
            0.0,
            ColorSpace::Srgb,
        )
        .unwrap();
        assert_eq!(out.as_slice(), base.as_slice());

        // Size mismatch -> None.
        let small = Rgba8Image::new(1, 1, vec![[0u8; 4]]).unwrap();
        assert!(blend_nonseparable_rgba8(
            &base,
            &small,
            NonSeparableBlendMode::Hue,
            1.0,
            ColorSpace::Linear
        )
        .is_none());
    }
}
