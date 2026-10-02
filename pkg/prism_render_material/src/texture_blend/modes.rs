//! Separable layer **blend modes** (W3C Compositing and Blending Level 1 / PDF
//! 1.7 blend functions) and an `RGBA8` layer compositor.
//!
//! Blend modes are the fixed-function compositing primitive behind texture
//! authoring and runtime decal/detail layering: a *detail* layer `cs` is
//! combined with a *backdrop* `cb` through a per-channel function
//! `B(cb, cs) -> [0, 1]`. "Separable" means each channel is blended
//! independently (unlike the non-separable hue/saturation/colour/luminosity
//! modes), so the whole family is pure scalar arithmetic.
//!
//! All functions assume both operands are already in `[0, 1]`; every mode here
//! maps `[0, 1]^2 -> [0, 1]`, so a composited layer never leaves the valid
//! range. The identities the oracles pin down:
//! * `multiply(a, 1) = a`, `multiply(a, 0) = 0`;
//! * `screen(a, b) = 1 - (1 - a)(1 - b)`, `screen(a, 0) = a`;
//! * `overlay(a, b) = hard_light(b, a)` (overlay is hard-light with the layers
//!   swapped) and `hard_light(a, b) = overlay(b, a)`;
//! * `darken = min`, `lighten = max`;
//! * `soft_light(a, 0.5) = a` (0.5 is the neutral grey);
//! * `color_dodge(a, 0) = a`, `color_burn(a, 1) = a`;
//! * `linear_dodge(a, b) = min(a + b, 1)`,
//!   `linear_burn(a, b) = max(a + b - 1, 0)`;
//! * `difference` and `exclusion` are symmetric and `difference(a, a) = 0`.
//!
//! The [`blend_rgba8`] compositor blends a top layer over a backdrop in
//! scene-linear light (so the result matches a GPU layer blend), with a scalar
//! `opacity` that linearly mixes the blended colour back toward the backdrop;
//! the backdrop alpha is preserved.
//!
//! Everything is deterministic analytic `f32` arithmetic -- no AI/ML.
//!
//! # References
//! * W3C, "Compositing and Blending Level 1" (separable blend modes).
//! * Adobe, PDF 1.7 reference, 7.2.4 (blend functions).

use alloc::vec::Vec;

use bevy_math::ops;

use crate::{linear_to_srgb, srgb_to_linear, ColorSpace, Rgba8Image};

/// A separable per-channel blend mode `B(backdrop, source) -> [0, 1]`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BlendMode {
    /// Source replaces backdrop.
    Normal,
    /// `a * b` -- always darkens.
    Multiply,
    /// `1 - (1 - a)(1 - b)` -- always lightens.
    Screen,
    /// Multiply/screen depending on the backdrop; boosts contrast.
    Overlay,
    /// `min(a, b)`.
    Darken,
    /// `max(a, b)`.
    Lighten,
    /// Brighten the backdrop toward white driven by the source.
    ColorDodge,
    /// Darken the backdrop toward black driven by the source.
    ColorBurn,
    /// Multiply/screen depending on the source (overlay with layers swapped).
    HardLight,
    /// Gentler contrast boost; `0.5` source is neutral.
    SoftLight,
    /// `|a - b|`.
    Difference,
    /// `a + b - 2ab`.
    Exclusion,
    /// `min(a + b, 1)` (additive / linear dodge).
    LinearDodge,
    /// `max(a + b - 1, 0)` (linear burn).
    LinearBurn,
}

/// Evaluate a separable blend mode for a single channel. Both operands are
/// assumed to lie in `[0, 1]`; the result is in `[0, 1]`.
#[must_use]
pub fn blend_channel(mode: BlendMode, cb: f32, cs: f32) -> f32 {
    match mode {
        BlendMode::Normal => cs,
        BlendMode::Multiply => cb * cs,
        BlendMode::Screen => cb + cs - cb * cs,
        BlendMode::Overlay => hard_light(cs, cb),
        BlendMode::Darken => cb.min(cs),
        BlendMode::Lighten => cb.max(cs),
        BlendMode::ColorDodge => {
            if cb <= 0.0 {
                0.0
            } else if cs >= 1.0 {
                1.0
            } else {
                (cb / (1.0 - cs)).min(1.0)
            }
        }
        BlendMode::ColorBurn => {
            if cb >= 1.0 {
                1.0
            } else if cs <= 0.0 {
                0.0
            } else {
                1.0 - ((1.0 - cb) / cs).min(1.0)
            }
        }
        BlendMode::HardLight => hard_light(cb, cs),
        BlendMode::SoftLight => soft_light(cb, cs),
        BlendMode::Difference => (cb - cs).abs(),
        BlendMode::Exclusion => cb + cs - 2.0 * cb * cs,
        BlendMode::LinearDodge => (cb + cs).min(1.0),
        BlendMode::LinearBurn => (cb + cs - 1.0).max(0.0),
    }
}

/// Hard-light: multiply/screen chosen by the source channel.
#[inline]
fn hard_light(cb: f32, cs: f32) -> f32 {
    if cs <= 0.5 {
        2.0 * cb * cs
    } else {
        // screen(cb, 2*cs - 1)
        let s = 2.0 * cs - 1.0;
        cb + s - cb * s
    }
}

/// Soft-light (W3C definition; the `d(cb)` helper uses a `sqrt` branch).
#[inline]
fn soft_light(cb: f32, cs: f32) -> f32 {
    if cs <= 0.5 {
        cb - (1.0 - 2.0 * cs) * cb * (1.0 - cb)
    } else {
        let d = if cb <= 0.25 {
            ((16.0 * cb - 12.0) * cb + 4.0) * cb
        } else {
            ops::sqrt(cb)
        };
        cb + (2.0 * cs - 1.0) * (d - cb)
    }
}

/// Composite a `top` layer over a `base` backdrop with the given blend `mode`
/// and scalar `opacity` in `[0, 1]`, gamma-correctly under `space`.
///
/// Colour channels are blended in scene-linear light; the blended colour is
/// then linearly mixed toward the backdrop by `opacity`
/// (`out = base + opacity * (blended - base)`). Alpha is taken from the
/// backdrop. Returns `None` when the two images differ in size or are empty.
#[must_use]
pub fn blend_rgba8(
    base: &Rgba8Image,
    top: &Rgba8Image,
    mode: BlendMode,
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
        let mut px = [0u8; 4];
        for k in 0..3 {
            let cb = lift(b[k]);
            let cs = lift(s[k]);
            let blended = blend_channel(mode, cb, cs);
            px[k] = encode(cb + t * (blended - cb));
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

    const GRID: [f32; 7] = [0.0, 0.15, 0.3, 0.5, 0.7, 0.85, 1.0];

    #[test]
    fn multiply_identities() {
        for &a in &GRID {
            assert!((blend_channel(BlendMode::Multiply, a, 1.0) - a).abs() < 1.0e-6);
            assert!(blend_channel(BlendMode::Multiply, a, 0.0).abs() < 1.0e-6);
        }
    }

    #[test]
    fn screen_identity_and_formula() {
        for &a in &GRID {
            assert!((blend_channel(BlendMode::Screen, a, 0.0) - a).abs() < 1.0e-6);
            for &b in &GRID {
                let want = 1.0 - (1.0 - a) * (1.0 - b);
                assert!((blend_channel(BlendMode::Screen, a, b) - want).abs() < 1.0e-6);
            }
        }
    }

    #[test]
    fn overlay_is_hard_light_swapped() {
        for &a in &GRID {
            for &b in &GRID {
                let ov = blend_channel(BlendMode::Overlay, a, b);
                let hl = blend_channel(BlendMode::HardLight, b, a);
                assert!((ov - hl).abs() < 1.0e-6, "a={a} b={b}");
            }
        }
    }

    #[test]
    fn darken_lighten_are_min_max() {
        for &a in &GRID {
            for &b in &GRID {
                assert!((blend_channel(BlendMode::Darken, a, b) - a.min(b)).abs() < 1.0e-6);
                assert!((blend_channel(BlendMode::Lighten, a, b) - a.max(b)).abs() < 1.0e-6);
            }
        }
    }

    #[test]
    fn soft_light_neutral_grey_is_identity() {
        for &a in &GRID {
            assert!(
                (blend_channel(BlendMode::SoftLight, a, 0.5) - a).abs() < 1.0e-6,
                "a={a}"
            );
        }
    }

    #[test]
    fn dodge_burn_identities() {
        for &a in &GRID {
            assert!((blend_channel(BlendMode::ColorDodge, a, 0.0) - a).abs() < 1.0e-6);
            assert!((blend_channel(BlendMode::ColorBurn, a, 1.0) - a).abs() < 1.0e-6);
        }
    }

    #[test]
    fn linear_dodge_burn_formulas() {
        for &a in &GRID {
            for &b in &GRID {
                assert!(
                    (blend_channel(BlendMode::LinearDodge, a, b) - (a + b).min(1.0)).abs() < 1.0e-6
                );
                assert!(
                    (blend_channel(BlendMode::LinearBurn, a, b) - (a + b - 1.0).max(0.0)).abs()
                        < 1.0e-6
                );
            }
        }
    }

    #[test]
    fn difference_exclusion_symmetric() {
        for &a in &GRID {
            assert!(blend_channel(BlendMode::Difference, a, a).abs() < 1.0e-6);
            for &b in &GRID {
                let d1 = blend_channel(BlendMode::Difference, a, b);
                let d2 = blend_channel(BlendMode::Difference, b, a);
                assert!((d1 - d2).abs() < 1.0e-6);
                let e1 = blend_channel(BlendMode::Exclusion, a, b);
                let e2 = blend_channel(BlendMode::Exclusion, b, a);
                assert!((e1 - e2).abs() < 1.0e-6);
            }
        }
    }

    #[test]
    fn every_mode_stays_in_unit_range() {
        let modes = [
            BlendMode::Normal,
            BlendMode::Multiply,
            BlendMode::Screen,
            BlendMode::Overlay,
            BlendMode::Darken,
            BlendMode::Lighten,
            BlendMode::ColorDodge,
            BlendMode::ColorBurn,
            BlendMode::HardLight,
            BlendMode::SoftLight,
            BlendMode::Difference,
            BlendMode::Exclusion,
            BlendMode::LinearDodge,
            BlendMode::LinearBurn,
        ];
        for &m in &modes {
            for &a in &GRID {
                for &b in &GRID {
                    let r = blend_channel(m, a, b);
                    assert!(
                        (-1.0e-6..=1.0 + 1.0e-6).contains(&r),
                        "{m:?} a={a} b={b} -> {r}"
                    );
                }
            }
        }
    }

    #[test]
    fn compositor_opacity_zero_returns_base() {
        let base = Rgba8Image::new(2, 2, vec![[40u8, 80, 120, 200]; 4]).unwrap();
        let top = Rgba8Image::new(2, 2, vec![[200u8, 10, 90, 255]; 4]).unwrap();
        let out = blend_rgba8(&base, &top, BlendMode::Multiply, 0.0, ColorSpace::Linear).unwrap();
        assert_eq!(out.as_slice(), base.as_slice());
    }

    #[test]
    fn compositor_size_mismatch_is_none() {
        let base = Rgba8Image::new(2, 2, vec![[10u8, 10, 10, 255]; 4]).unwrap();
        let top = Rgba8Image::new(3, 2, vec![[10u8, 10, 10, 255]; 6]).unwrap();
        assert!(blend_rgba8(&base, &top, BlendMode::Normal, 1.0, ColorSpace::Linear).is_none());
    }
}
