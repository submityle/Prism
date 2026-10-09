//! Real vector (TrueType outline) glyph rasterisation — the "crisp text" path.
//!
//! The embedded 5x7 [`crate`] bitmap keeps text legible everywhere (including
//! `no_std`), but it is a coarse 5x7 grid: scaled up it looks blocky. This
//! module adds the quality tier the roadmap in
//! `docs/prism_loom_text_rendering_design_zh.md` calls **B (CPU vector)**: it
//! parses a real embedded OpenType face with [`ab_glyph`] and rasterises each
//! glyph's Bézier outline into analytic per-pixel coverage at the exact target
//! size, so labels stay sharp at any scale.
//!
//! It is gated behind the `vector` feature (which pulls in `ab_glyph` and
//! `std`); when the feature is off the crate still compiles and callers fall
//! back to the bitmap table. The face is `FiraMono` (SIL Open Font License,
//! see `assets/Fira-OFL-License.txt`); it is monospace, which matches Loom's
//! current metric shaper. This module contains no Unreal Engine source or
//! derived code.

use alloc::vec;
use alloc::vec::Vec;
use std::sync::OnceLock;

use ab_glyph::{Font, FontRef, PxScale, ScaleFont};

/// The embedded monospace face used for crisp CPU text. `FiraMono` subset,
/// SIL OFL 1.1 (`assets/Fira-OFL-License.txt`).
static FONT_BYTES: &[u8] = include_bytes!("../assets/FiraMono-subset.ttf");

fn face() -> Option<&'static FontRef<'static>> {
    static FACE: OnceLock<Option<FontRef<'static>>> = OnceLock::new();
    FACE.get_or_init(|| FontRef::try_from_slice(FONT_BYTES).ok())
        .as_ref()
}

/// Scaled, size-dependent monospace metrics in device pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Metrics {
    /// Distance from baseline up to the top of the ascenders (positive).
    pub ascent: f32,
    /// Distance from baseline down to the bottom of the descenders (negative).
    pub descent: f32,
    /// Horizontal pen advance for one monospace cell.
    pub advance: f32,
}

impl Metrics {
    /// Total line box height (`ascent - descent`).
    #[must_use]
    pub fn line_height(&self) -> f32 {
        self.ascent - self.descent
    }
}

/// Returns monospace metrics at `px` em size, or `None` if the embedded face
/// failed to parse (the caller then uses the bitmap fallback).
#[must_use]
pub fn metrics(px: f32) -> Option<Metrics> {
    let font = face()?;
    let scaled = font.as_scaled(PxScale::from(px));
    // Advance of a representative glyph; the face is monospace so any covered
    // glyph has the same advance. Fall back to the space glyph.
    let gid = font.glyph_id('M');
    let advance = scaled.h_advance(gid);
    Some(Metrics {
        ascent: scaled.ascent(),
        descent: scaled.descent(),
        advance,
    })
}

/// A rasterised glyph's coverage mask plus its placement relative to the pen
/// origin (the point on the baseline where the glyph starts).
pub struct Coverage {
    /// Mask width in device pixels.
    pub width: usize,
    /// Mask height in device pixels.
    pub height: usize,
    /// X offset of the mask's left edge from the pen origin (left bearing).
    pub left: f32,
    /// Y offset of the mask's top edge from the baseline (negative = above).
    pub top: f32,
    /// Row-major coverage in `0.0..=1.0`, `width * height` entries.
    pub data: Vec<f32>,
}

/// Rasterises `ch` at `px` em size into an analytic coverage mask, or `None`
/// when the glyph has no outline (e.g. the space) or the face is unavailable.
#[must_use]
pub fn glyph_coverage(ch: char, px: f32) -> Option<Coverage> {
    let font = face()?;
    let scale = PxScale::from(px);
    let gid = font.glyph_id(ch);
    let glyph = gid.with_scale_and_position(scale, ab_glyph::point(0.0, 0.0));
    let outlined = font.outline_glyph(glyph)?;
    let bounds = outlined.px_bounds();
    let width = (bounds.max.x - bounds.min.x).ceil().max(0.0) as usize;
    let height = (bounds.max.y - bounds.min.y).ceil().max(0.0) as usize;
    if width == 0 || height == 0 {
        return None;
    }
    let mut data = vec![0.0f32; width * height];
    outlined.draw(|x, y, c| {
        let (x, y) = (x as usize, y as usize);
        if x < width && y < height {
            data[y * width + x] = c;
        }
    });
    Some(Coverage {
        width,
        height,
        left: bounds.min.x,
        top: bounds.min.y,
        data,
    })
}

/// Whether the embedded vector face parsed successfully. When `false`, callers
/// must use the bitmap fallback.
#[must_use]
pub fn available() -> bool {
    face().is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn face_parses() {
        assert!(available(), "embedded FiraMono subset must parse");
    }

    #[test]
    fn metrics_scale_linearly() {
        let a = metrics(16.0).expect("metrics");
        let b = metrics(32.0).expect("metrics");
        // Doubling the em size doubles advance and ascent (within rounding).
        assert!((b.advance - a.advance * 2.0).abs() < 0.01);
        assert!((b.ascent - a.ascent * 2.0).abs() < 0.01);
        assert!(a.advance > 0.0);
    }

    #[test]
    fn letter_has_ink_but_space_does_not() {
        let cov = glyph_coverage('A', 48.0).expect("A has an outline");
        assert!(cov.width > 0 && cov.height > 0);
        let inked: f32 = cov.data.iter().copied().sum();
        assert!(inked > 1.0, "A should cover several pixels, got {inked}");
        // The space glyph carries no outline.
        assert!(glyph_coverage(' ', 48.0).is_none());
    }
}
