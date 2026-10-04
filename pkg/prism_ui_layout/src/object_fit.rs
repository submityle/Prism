//! CSS `object-fit` / `object-position` content fitting.
//!
//! Replaced content such as an image or video has an *intrinsic* size that
//! rarely matches the box it is drawn into. CSS resolves this with two
//! properties: `object-fit` chooses how the content is scaled, and
//! `object-position` chooses where the scaled content sits inside any leftover
//! (or overflowing) space. This module reproduces that geometry on the
//! engine-agnostic [`Rect`]/[`Size`] types so renderers can place textures the
//! same way a browser would.
//!
//! The five `object-fit` keywords are modelled by [`ObjectFit`]:
//!
//! * `fill` stretches to the box, ignoring the aspect ratio;
//! * `contain` scales to the largest size that fits entirely inside the box;
//! * `cover` scales to the smallest size that fully covers the box;
//! * `none` keeps the intrinsic size;
//! * `scale-down` picks whichever of `none` and `contain` is smaller.
//!
//! `contain`, `cover`, `none` and `scale-down` all preserve the intrinsic
//! aspect ratio; only `fill` distorts it.
//!
//! # Example
//!
//! ```
//! use prism_ui_layout::geometry::{Point, Rect, Size};
//! use prism_ui_layout::object_fit::{fit, ObjectFit, ObjectPosition};
//!
//! // A 200x100 image inside a 100x100 box, scaled to contain and centered.
//! let container = Rect::new(Point::new(0.0, 0.0), Size::new(100.0, 100.0));
//! let placed = fit(
//!     ObjectFit::Contain,
//!     Size::new(200.0, 100.0),
//!     container,
//!     ObjectPosition::CENTER,
//! );
//!
//! // Width-limited: 100 wide, 50 tall, vertically centered in the 100-tall box.
//! assert!((placed.size.width - 100.0).abs() < 1e-3);
//! assert!((placed.size.height - 50.0).abs() < 1e-3);
//! assert!((placed.location.y - 25.0).abs() < 1e-3);
//! ```

use crate::geometry::{Point, Rect, Size};

/// How replaced content is scaled to fit its box (CSS `object-fit`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ObjectFit {
    /// Stretch to fill the box exactly, ignoring the intrinsic aspect ratio.
    #[default]
    Fill,
    /// Scale to the largest size that fits entirely within the box.
    Contain,
    /// Scale to the smallest size that fully covers the box.
    Cover,
    /// Keep the intrinsic size regardless of the box.
    None,
    /// Use whichever of [`ObjectFit::None`] and [`ObjectFit::Contain`] yields
    /// the smaller rendered size (never upscales).
    ScaleDown,
}

/// Placement of the scaled content within the box (CSS `object-position`).
///
/// Each component is a fraction of the free space along that axis: `0.0`
/// aligns to the start edge, `0.5` centers, and `1.0` aligns to the end edge.
/// When the content overflows the box (as with [`ObjectFit::Cover`]) the free
/// space is negative, so the same fractions shift which region stays visible.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObjectPosition {
    /// Horizontal placement fraction.
    pub x: f32,
    /// Vertical placement fraction.
    pub y: f32,
}

impl ObjectPosition {
    /// Centered placement (`50% 50%`), the CSS initial value.
    pub const CENTER: Self = Self { x: 0.5, y: 0.5 };
    /// Top-left placement (`0% 0%`).
    pub const TOP_LEFT: Self = Self { x: 0.0, y: 0.0 };
    /// Bottom-right placement (`100% 100%`).
    pub const BOTTOM_RIGHT: Self = Self { x: 1.0, y: 1.0 };

    /// Builds a placement from explicit horizontal and vertical fractions.
    #[must_use]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

impl Default for ObjectPosition {
    fn default() -> Self {
        Self::CENTER
    }
}

/// Smaller of two finite `f32` values (no_std-safe, avoids `f32::min`).
fn min_f(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}

/// Larger of two finite `f32` values (no_std-safe, avoids `f32::max`).
fn max_f(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

/// Scales an intrinsic size into a `container` while preserving aspect ratio.
///
/// With `cover == false` the result is the largest size fitting inside the
/// container (`contain`); with `cover == true` it is the smallest size that
/// covers it (`cover`). A degenerate intrinsic extent collapses to zero.
fn scaled_to(iw: f32, ih: f32, cw: f32, ch: f32, cover: bool) -> Size<f32> {
    if iw <= 0.0 || ih <= 0.0 {
        return Size::new(0.0, 0.0);
    }
    let sx = cw / iw;
    let sy = ch / ih;
    let scale = if cover { max_f(sx, sy) } else { min_f(sx, sy) };
    Size::new(iw * scale, ih * scale)
}

/// Resolves the rendered size of the content before positioning.
fn fitted_size(mode: ObjectFit, intrinsic: Size<f32>, container: Size<f32>) -> Size<f32> {
    let (iw, ih) = (intrinsic.width, intrinsic.height);
    let (cw, ch) = (container.width, container.height);
    match mode {
        ObjectFit::Fill => container,
        ObjectFit::None => intrinsic,
        ObjectFit::Contain => scaled_to(iw, ih, cw, ch, false),
        ObjectFit::Cover => scaled_to(iw, ih, cw, ch, true),
        ObjectFit::ScaleDown => {
            let contained = scaled_to(iw, ih, cw, ch, false);
            // Both candidates preserve the ratio, so compare on width alone.
            if iw <= contained.width {
                intrinsic
            } else {
                contained
            }
        }
    }
}

/// Computes the box the content occupies inside `container` for the given
/// `mode` and `position`.
///
/// The returned [`Rect`] is expressed in the same coordinate space as
/// `container`. For [`ObjectFit::Cover`] the result can extend beyond the
/// container (the caller is expected to clip).
#[must_use]
pub fn fit(
    mode: ObjectFit,
    intrinsic: Size<f32>,
    container: Rect,
    position: ObjectPosition,
) -> Rect {
    let size = fitted_size(mode, intrinsic, container.size);
    let free_x = container.size.width - size.width;
    let free_y = container.size.height - size.height;
    let x = container.location.x + position.x * free_x;
    let y = container.location.y + position.y * free_y;
    Rect::new(Point::new(x, y), size)
}

#[cfg(test)]
mod tests {
    use super::{fit, fitted_size, ObjectFit, ObjectPosition};
    use crate::geometry::{Point, Rect, Size};

    /// Small deterministic PRNG for property tests (`SplitMix64`).
    struct SplitMix64(u64);

    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        /// Uniform `f32` in `[lo, hi]`.
        fn range(&mut self, lo: f32, hi: f32) -> f32 {
            let frac = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
            lo + frac * (hi - lo)
        }
    }

    fn container() -> Rect {
        Rect::new(Point::new(7.0, 11.0), Size::new(120.0, 80.0))
    }

    fn ratio_preserved(intrinsic: Size<f32>, out: Size<f32>) -> bool {
        // iw/ih == ow/oh  <=>  iw*oh == ih*ow
        let lhs = intrinsic.width * out.height;
        let rhs = intrinsic.height * out.width;
        (lhs - rhs).abs() <= 1e-2 + lhs.abs() * 1e-4
    }

    #[test]
    fn fill_matches_container_exactly() {
        let c = container();
        let r = fit(ObjectFit::Fill, Size::new(37.0, 999.0), c, ObjectPosition::TOP_LEFT);
        assert_eq!(r.size, c.size);
        assert_eq!(r.location, c.location);
    }

    #[test]
    fn none_keeps_intrinsic_size() {
        let intrinsic = Size::new(37.0, 52.0);
        let r = fit(ObjectFit::None, intrinsic, container(), ObjectPosition::CENTER);
        assert_eq!(r.size, intrinsic);
    }

    #[test]
    fn contain_fits_inside_and_touches_an_edge() {
        let mut rng = SplitMix64(0x00C0_FFEE_1234_5678);
        let c = container();
        for _ in 0..3000 {
            let intrinsic = Size::new(rng.range(1.0, 400.0), rng.range(1.0, 400.0));
            let out = fitted_size(ObjectFit::Contain, intrinsic, c.size);
            assert!(ratio_preserved(intrinsic, out), "{intrinsic:?} -> {out:?}");
            assert!(out.width <= c.size.width + 1e-2, "{out:?} wider than box");
            assert!(out.height <= c.size.height + 1e-2, "{out:?} taller than box");
            let touches = (out.width - c.size.width).abs() < 1e-2
                || (out.height - c.size.height).abs() < 1e-2;
            assert!(touches, "contain must touch an edge: {out:?}");
        }
    }

    #[test]
    fn cover_covers_and_touches_an_edge() {
        let mut rng = SplitMix64(0x5EED_1111_2222_3333);
        let c = container();
        for _ in 0..3000 {
            let intrinsic = Size::new(rng.range(1.0, 400.0), rng.range(1.0, 400.0));
            let out = fitted_size(ObjectFit::Cover, intrinsic, c.size);
            assert!(ratio_preserved(intrinsic, out), "{intrinsic:?} -> {out:?}");
            assert!(out.width >= c.size.width - 1e-2, "{out:?} narrower than box");
            assert!(out.height >= c.size.height - 1e-2, "{out:?} shorter than box");
            let touches = (out.width - c.size.width).abs() < 1e-2
                || (out.height - c.size.height).abs() < 1e-2;
            assert!(touches, "cover must touch an edge: {out:?}");
        }
    }

    #[test]
    fn scale_down_never_upscales_and_matches_bounds() {
        let c = container();
        // Larger-than-box intrinsic: scale-down == contain.
        let big = Size::new(400.0, 400.0);
        let sd_big = fitted_size(ObjectFit::ScaleDown, big, c.size);
        let contain_big = fitted_size(ObjectFit::Contain, big, c.size);
        assert!((sd_big.width - contain_big.width).abs() < 1e-3);
        assert!((sd_big.height - contain_big.height).abs() < 1e-3);
        // Smaller-than-box intrinsic: scale-down == none (no upscaling).
        let small = Size::new(20.0, 10.0);
        let sd_small = fitted_size(ObjectFit::ScaleDown, small, c.size);
        assert_eq!(sd_small, small);
    }

    #[test]
    fn center_position_splits_free_space_evenly() {
        let c = container();
        // Contain of a wide image: width-limited, free vertical space.
        let out = fit(ObjectFit::Contain, Size::new(240.0, 80.0), c, ObjectPosition::CENTER);
        // 240x80 into 120x80 => scale 0.5 => 120x40, 40 free vertical.
        assert!((out.size.width - 120.0).abs() < 1e-3);
        assert!((out.size.height - 40.0).abs() < 1e-3);
        assert!((out.location.x - c.location.x).abs() < 1e-3);
        assert!((out.location.y - (c.location.y + 20.0)).abs() < 1e-3);
    }

    #[test]
    fn position_fractions_align_to_edges() {
        let c = container();
        let intrinsic = Size::new(240.0, 80.0); // contain => 120x40
        let start = fit(ObjectFit::Contain, intrinsic, c, ObjectPosition::new(0.0, 0.0));
        assert!((start.top() - c.top()).abs() < 1e-3);
        let end = fit(ObjectFit::Contain, intrinsic, c, ObjectPosition::new(1.0, 1.0));
        assert!((end.bottom() - c.bottom()).abs() < 1e-3);
    }

    #[test]
    fn cover_overflows_symmetrically_when_centered() {
        let c = container();
        // Tall image covering a wide box overflows vertically.
        let intrinsic = Size::new(80.0, 240.0);
        let out = fit(ObjectFit::Cover, intrinsic, c, ObjectPosition::CENTER);
        // scale = max(120/80, 80/240) = 1.5 => 120x360, overflow 280 vertical.
        assert!((out.size.width - 120.0).abs() < 1e-3);
        assert!((out.size.height - 360.0).abs() < 1e-3);
        let top_overflow = c.top() - out.top();
        let bottom_overflow = out.bottom() - c.bottom();
        assert!((top_overflow - bottom_overflow).abs() < 1e-3, "centered overflow symmetric");
    }

    #[test]
    fn degenerate_intrinsic_collapses_for_ratio_fits() {
        let c = container();
        for mode in [ObjectFit::Contain, ObjectFit::Cover] {
            let out = fitted_size(mode, Size::new(0.0, 50.0), c.size);
            assert_eq!(out, Size::new(0.0, 0.0));
        }
    }

    #[test]
    fn defaults_are_css_initial_values() {
        assert_eq!(ObjectFit::default(), ObjectFit::Fill);
        assert_eq!(ObjectPosition::default(), ObjectPosition::CENTER);
    }
}
