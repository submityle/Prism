//! Physical and logical geometry plus the scale-factor conversions between them.
//!
//! A window reports two coordinate spaces: *physical* pixels (actual device
//! pixels, integer) and *logical* units (density-independent, used by UI and
//! input). They are related by the window's `scale_factor` (physical =
//! logical × scale). Keeping both explicit avoids the classic `HiDPI` bugs where
//! one space is silently used where the other is meant.
//!
//! All conversions are `no_std`-safe: they use only arithmetic and Rust's
//! saturating float-to-int casts (no `round`/`floor` from `std`, no `libm`).

/// Rounds a non-negative logical dimension to the nearest physical pixel.
///
/// Uses Rust's saturating float-to-int cast (`NaN` → 0, overflow and +∞ →
/// `u32::MAX`, negatives → 0) after a `+0.5` round-half-up, so it never panics
/// and needs no `std` float intrinsics.
#[must_use]
fn round_to_u32(value: f32) -> u32 {
    if value.is_nan() || value <= 0.0 {
        return 0;
    }
    // `value` is positive here (finite or +inf). The saturating float-to-int
    // cast clamps anything ≥ `u32::MAX` (and +inf) to `u32::MAX`, so the
    // round-half-up `+0.5` can never overflow.
    (value + 0.5) as u32
}

/// A size in physical pixels (actual device pixels).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct PhysicalSize {
    /// Width in physical pixels.
    pub width: u32,
    /// Height in physical pixels.
    pub height: u32,
}

impl PhysicalSize {
    /// Builds a physical size.
    #[must_use]
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// Whether either dimension is zero (a window that cannot be drawn into).
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.width == 0 || self.height == 0
    }

    /// Width divided by height, or `0.0` if the height is zero.
    #[must_use]
    pub fn aspect_ratio(self) -> f32 {
        if self.height == 0 {
            0.0
        } else {
            self.width as f32 / self.height as f32
        }
    }

    /// Converts to logical units by dividing by `scale_factor`.
    ///
    /// A non-positive or non-finite `scale_factor` is treated as `1.0` so the
    /// result is always well-defined.
    #[must_use]
    pub fn to_logical(self, scale_factor: f32) -> LogicalSize {
        let scale = sane_scale(scale_factor);
        LogicalSize::new(self.width as f32 / scale, self.height as f32 / scale)
    }
}

/// A size in logical (density-independent) units.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct LogicalSize {
    /// Width in logical units.
    pub width: f32,
    /// Height in logical units.
    pub height: f32,
}

impl LogicalSize {
    /// Builds a logical size.
    #[must_use]
    pub const fn new(width: f32, height: f32) -> Self {
        Self { width, height }
    }

    /// Converts to physical pixels by multiplying by `scale_factor` and
    /// rounding half-up. A non-positive or non-finite `scale_factor` is treated
    /// as `1.0`.
    #[must_use]
    pub fn to_physical(self, scale_factor: f32) -> PhysicalSize {
        let scale = sane_scale(scale_factor);
        PhysicalSize::new(round_to_u32(self.width * scale), round_to_u32(self.height * scale))
    }
}

/// A position in physical pixels, relative to the top-left of its surface.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct PhysicalPosition {
    /// Horizontal offset in physical pixels (right is positive).
    pub x: i32,
    /// Vertical offset in physical pixels (down is positive).
    pub y: i32,
}

impl PhysicalPosition {
    /// Builds a physical position.
    #[must_use]
    pub const fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    /// Converts to logical units by dividing by `scale_factor` (treated as
    /// `1.0` when non-positive or non-finite).
    #[must_use]
    pub fn to_logical(self, scale_factor: f32) -> LogicalPosition {
        let scale = sane_scale(scale_factor);
        LogicalPosition::new(self.x as f32 / scale, self.y as f32 / scale)
    }
}

/// A position in logical (density-independent) units.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct LogicalPosition {
    /// Horizontal offset in logical units (right is positive).
    pub x: f32,
    /// Vertical offset in logical units (down is positive).
    pub y: f32,
}

impl LogicalPosition {
    /// Builds a logical position.
    #[must_use]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

/// Clamps a scale factor to a sane, strictly-positive value, folding invalid
/// inputs (`<= 0`, `NaN`, infinite) to `1.0`.
#[must_use]
pub(crate) fn sane_scale(scale_factor: f32) -> f32 {
    if scale_factor.is_finite() && scale_factor > 0.0 {
        scale_factor
    } else {
        1.0
    }
}
