//! Pure-arithmetic geometry primitives shared by the motion layer.
//!
//! [`Rect`] is an axis-aligned rectangle in logical pixels and [`Transform`]
//! is the axis-aligned affine map `screen = (tx + sx * x, ty + sy * y)` used to
//! drive `FLIP` layout animation and shared-element transitions. Both types
//! implement [`Lerp`] so they can be fed straight into
//! [`prism_ui_anim::Tween`].
//!
//! All operations here use only `+ - * /`, comparisons and `.abs()`, so the
//! module is deterministic and `no_std`-friendly.

use prism_ui_anim::Lerp;

/// Smallest denominator magnitude treated as non-zero when forming ratios.
///
/// Guards the scale terms of [`Transform::from_rects`] against division by a
/// (near-)zero current dimension, which would otherwise produce a non-finite
/// scale for a collapsed box.
const MIN_DENOM: f32 = 1.0e-6;

/// An axis-aligned rectangle in logical pixels.
///
/// The origin is the top-left corner, with `x` increasing to the right and `y`
/// increasing downward, matching the conventions of the Loom layout engine.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rect {
    /// Left edge.
    pub x: f32,
    /// Top edge.
    pub y: f32,
    /// Width; expected to be non-negative.
    pub width: f32,
    /// Height; expected to be non-negative.
    pub height: f32,
}

impl Rect {
    /// A zero-sized rectangle at the origin.
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        width: 0.0,
        height: 0.0,
    };

    /// Builds a rectangle from its top-left corner and size.
    #[must_use]
    pub const fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// The x coordinate of the right edge (`x + width`).
    #[must_use]
    pub fn right(&self) -> f32 {
        self.x + self.width
    }

    /// The y coordinate of the bottom edge (`y + height`).
    #[must_use]
    pub fn bottom(&self) -> f32 {
        self.y + self.height
    }

    /// The x coordinate of the rectangle's center.
    #[must_use]
    pub fn center_x(&self) -> f32 {
        self.x + self.width * 0.5
    }

    /// The y coordinate of the rectangle's center.
    #[must_use]
    pub fn center_y(&self) -> f32 {
        self.y + self.height * 0.5
    }

    /// Whether either dimension is non-positive, i.e. the box is collapsed.
    #[must_use]
    pub fn is_degenerate(&self) -> bool {
        self.width <= 0.0 || self.height <= 0.0
    }
}

impl Lerp for Rect {
    #[inline]
    fn lerp(&self, other: &Self, t: f32) -> Self {
        Self {
            x: self.x.lerp(&other.x, t),
            y: self.y.lerp(&other.y, t),
            width: self.width.lerp(&other.width, t),
            height: self.height.lerp(&other.height, t),
        }
    }
}

/// An axis-aligned affine transform `screen = (tx + sx * x, ty + sy * y)`.
///
/// This is the minimal transform needed for layout and shared-element motion:
/// a per-axis translation plus a per-axis scale, with no rotation or shear.
/// The identity transform leaves coordinates unchanged.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform {
    /// Translation along x.
    pub tx: f32,
    /// Translation along y.
    pub ty: f32,
    /// Scale along x.
    pub sx: f32,
    /// Scale along y.
    pub sy: f32,
}

impl Transform {
    /// The identity transform: no translation and unit scale.
    pub const IDENTITY: Self = Self {
        tx: 0.0,
        ty: 0.0,
        sx: 1.0,
        sy: 1.0,
    };

    /// Builds a transform from explicit translation and scale terms.
    #[must_use]
    pub const fn new(tx: f32, ty: f32, sx: f32, sy: f32) -> Self {
        Self { tx, ty, sx, sy }
    }

    /// The identity transform (same as [`Transform::IDENTITY`]).
    #[must_use]
    pub const fn identity() -> Self {
        Self::IDENTITY
    }

    /// Whether this transform is exactly the identity.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.tx == 0.0 && self.ty == 0.0 && self.sx == 1.0 && self.sy == 1.0
    }

    /// The affine map taking `current` onto `prev`.
    ///
    /// This is the "Invert" step of `FLIP`: given an element that now lives at
    /// `current`, the returned transform makes it *appear* at `prev`. Playing
    /// this transform back toward [`Transform::IDENTITY`] animates the element
    /// from its old box to its new one.
    ///
    /// If a `current` dimension is (near) zero the corresponding scale falls
    /// back to `1.0` so the result stays finite for collapsed boxes.
    #[must_use]
    pub fn from_rects(prev: &Rect, current: &Rect) -> Self {
        let sx = ratio(prev.width, current.width);
        let sy = ratio(prev.height, current.height);
        Self {
            tx: prev.x - sx * current.x,
            ty: prev.y - sy * current.y,
            sx,
            sy,
        }
    }

    /// Applies the transform to a point, returning `(tx + sx * x, ty + sy * y)`.
    #[must_use]
    pub fn apply_point(&self, x: f32, y: f32) -> (f32, f32) {
        (self.tx + self.sx * x, self.ty + self.sy * y)
    }

    /// Applies the transform to a rectangle, mapping its corners.
    #[must_use]
    pub fn apply_rect(&self, rect: &Rect) -> Rect {
        let (x, y) = self.apply_point(rect.x, rect.y);
        Rect::new(x, y, self.sx * rect.width, self.sy * rect.height)
    }
}

impl Default for Transform {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Lerp for Transform {
    #[inline]
    fn lerp(&self, other: &Self, t: f32) -> Self {
        Self {
            tx: self.tx.lerp(&other.tx, t),
            ty: self.ty.lerp(&other.ty, t),
            sx: self.sx.lerp(&other.sx, t),
            sy: self.sy.lerp(&other.sy, t),
        }
    }
}

/// Forms `num / den`, falling back to `1.0` when `den` is (near) zero.
fn ratio(num: f32, den: f32) -> f32 {
    if den.abs() > MIN_DENOM {
        num / den
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1.0e-6;

    fn close(a: f32, b: f32) {
        assert!((a - b).abs() < EPS, "expected {a} ~= {b}");
    }

    #[test]
    fn rect_edges_and_center() {
        let r = Rect::new(10.0, 20.0, 30.0, 40.0);
        close(r.right(), 40.0);
        close(r.bottom(), 60.0);
        close(r.center_x(), 25.0);
        close(r.center_y(), 40.0);
        assert!(!r.is_degenerate());
        assert!(Rect::new(0.0, 0.0, 0.0, 5.0).is_degenerate());
    }

    #[test]
    fn rect_lerp_is_componentwise() {
        let a = Rect::new(0.0, 0.0, 10.0, 10.0);
        let b = Rect::new(10.0, 20.0, 30.0, 50.0);
        let m = a.lerp(&b, 0.5);
        close(m.x, 5.0);
        close(m.y, 10.0);
        close(m.width, 20.0);
        close(m.height, 30.0);
    }

    #[test]
    fn from_rects_maps_current_onto_prev() {
        let prev = Rect::new(0.0, 0.0, 50.0, 50.0);
        let current = Rect::new(100.0, 200.0, 100.0, 100.0);
        let t = Transform::from_rects(&prev, &current);
        // Scale halves the box.
        close(t.sx, 0.5);
        close(t.sy, 0.5);
        // The transform must send `current` back onto `prev`.
        let mapped = t.apply_rect(&current);
        close(mapped.x, prev.x);
        close(mapped.y, prev.y);
        close(mapped.width, prev.width);
        close(mapped.height, prev.height);
    }

    #[test]
    fn from_rects_degenerate_scale_falls_back_to_one() {
        let prev = Rect::new(0.0, 0.0, 10.0, 10.0);
        let current = Rect::new(5.0, 5.0, 0.0, 0.0);
        let t = Transform::from_rects(&prev, &current);
        close(t.sx, 1.0);
        close(t.sy, 1.0);
        assert!(t.tx.abs() < 100.0);
    }

    #[test]
    fn transform_identity_and_lerp() {
        assert!(Transform::IDENTITY.is_identity());
        assert!(Transform::default().is_identity());
        let start = Transform::new(-100.0, 0.0, 2.0, 2.0);
        let mid = start.lerp(&Transform::IDENTITY, 0.5);
        close(mid.tx, -50.0);
        close(mid.sx, 1.5);
        let end = start.lerp(&Transform::IDENTITY, 1.0);
        assert!(end.is_identity());
    }
}
