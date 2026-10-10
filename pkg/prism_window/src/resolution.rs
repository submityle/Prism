//! Window resolution: physical size paired with a scale factor, plus the
//! resize constraints that bound it.

use crate::geometry::{sane_scale, LogicalSize, PhysicalSize};

/// Inclusive bounds on a window's physical size, used to clamp OS-driven
/// resizes. Defaults allow any size from `1×1` up to `u32::MAX`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct WindowResizeConstraints {
    /// Minimum width in physical pixels (at least `1`).
    pub min_width: u32,
    /// Minimum height in physical pixels (at least `1`).
    pub min_height: u32,
    /// Maximum width in physical pixels.
    pub max_width: u32,
    /// Maximum height in physical pixels.
    pub max_height: u32,
}

impl Default for WindowResizeConstraints {
    fn default() -> Self {
        Self {
            min_width: 1,
            min_height: 1,
            max_width: u32::MAX,
            max_height: u32::MAX,
        }
    }
}

impl WindowResizeConstraints {
    /// Builds constraints, normalizing them so that each minimum is at least
    /// `1` and no maximum is below its matching minimum (maxima are lifted to
    /// the minimum when inverted), keeping [`clamp`](Self::clamp) well-defined.
    #[must_use]
    pub fn new(min_width: u32, min_height: u32, max_width: u32, max_height: u32) -> Self {
        let min_width = min_width.max(1);
        let min_height = min_height.max(1);
        Self {
            min_width,
            min_height,
            max_width: max_width.max(min_width),
            max_height: max_height.max(min_height),
        }
    }

    /// Clamps a physical size into the allowed range.
    #[must_use]
    pub fn clamp(&self, size: PhysicalSize) -> PhysicalSize {
        PhysicalSize::new(
            size.width.clamp(self.min_width, self.max_width),
            size.height.clamp(self.min_height, self.max_height),
        )
    }
}

/// A window's resolution: its physical pixel size together with the display
/// scale factor relating physical to logical units.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct WindowResolution {
    physical: PhysicalSize,
    scale_factor: f32,
}

impl Default for WindowResolution {
    fn default() -> Self {
        // A conventional default windowed size at scale 1.0.
        Self {
            physical: PhysicalSize::new(1280, 720),
            scale_factor: 1.0,
        }
    }
}

impl WindowResolution {
    /// Builds a resolution from a physical size at scale factor `1.0`.
    #[must_use]
    pub const fn new(width: u32, height: u32) -> Self {
        Self {
            physical: PhysicalSize::new(width, height),
            scale_factor: 1.0,
        }
    }

    /// Returns this resolution with its scale factor replaced (invalid values
    /// fold to `1.0`).
    #[must_use]
    pub fn with_scale_factor(mut self, scale_factor: f32) -> Self {
        self.scale_factor = sane_scale(scale_factor);
        self
    }

    /// The physical pixel size.
    #[must_use]
    pub const fn physical_size(&self) -> PhysicalSize {
        self.physical
    }

    /// The physical width in pixels.
    #[must_use]
    pub const fn physical_width(&self) -> u32 {
        self.physical.width
    }

    /// The physical height in pixels.
    #[must_use]
    pub const fn physical_height(&self) -> u32 {
        self.physical.height
    }

    /// The logical size (physical divided by scale factor).
    #[must_use]
    pub fn logical_size(&self) -> LogicalSize {
        self.physical.to_logical(self.scale_factor)
    }

    /// The scale factor relating physical to logical units.
    #[must_use]
    pub const fn scale_factor(&self) -> f32 {
        self.scale_factor
    }

    /// Replaces the physical size.
    pub fn set_physical_size(&mut self, size: PhysicalSize) {
        self.physical = size;
    }

    /// Replaces the scale factor (invalid values fold to `1.0`).
    pub fn set_scale_factor(&mut self, scale_factor: f32) {
        self.scale_factor = sane_scale(scale_factor);
    }
}
