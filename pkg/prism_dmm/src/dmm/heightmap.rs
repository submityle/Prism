//! Displacement height sampling: the [`DisplacementMap`] trait and a
//! texture-backed bilinear implementation.
//!
//! The baker queries a scalar displacement height at the interpolated `UV` of
//! every micro-vertex. Any source of heights can be plugged in by implementing
//! [`DisplacementMap`]; the bundled [`TextureDisplacementMap`] stores a dense
//! `f32` height texture and reconstructs intermediate heights with **bilinear
//! filtering** under a selectable [`WrapMode`].

use alloc::vec::Vec;
use libm::floorf;

/// A source of scalar displacement heights addressed by `UV` coordinate.
pub trait DisplacementMap {
    /// Samples the displacement height at texture coordinate `(u, v)`.
    ///
    /// Coordinates follow the usual convention where `(0.0, 0.0)` is one
    /// corner of the texture and `(1.0, 1.0)` the opposite corner. The height
    /// is returned in whatever units the map was authored in; the baker later
    /// normalizes it into `[0, 1]` before quantization.
    fn sample_height(&self, u: f32, v: f32) -> f32;
}

/// How `UV` coordinates outside `[0, 1)` are resolved to texels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WrapMode {
    /// Tile the texture, wrapping coordinates modulo the texture size.
    Repeat,
    /// Clamp coordinates to the nearest edge texel.
    Clamp,
}

/// A dense `f32` height texture sampled with bilinear filtering.
///
/// Texels are stored row-major: the height at integer texel `(x, y)` lives at
/// `data[y * width + x]`. Sampling uses the half-texel-center convention, so
/// `UV` `(0.5 / width, 0.5 / height)` maps exactly onto texel `(0, 0)`.
#[derive(Debug, Clone, PartialEq)]
pub struct TextureDisplacementMap {
    width: u32,
    height: u32,
    data: Vec<f32>,
    wrap: WrapMode,
}

impl TextureDisplacementMap {
    /// Creates a height texture from a row-major `data` buffer.
    ///
    /// # Panics
    ///
    /// Panics when `width == 0`, `height == 0`, or when `data.len()` does not
    /// equal `width * height`.
    #[must_use]
    pub fn new(width: u32, height: u32, data: Vec<f32>, wrap: WrapMode) -> Self {
        assert!(width > 0, "texture width must be non-zero");
        assert!(height > 0, "texture height must be non-zero");
        let expected = width as usize * height as usize;
        assert!(
            data.len() == expected,
            "height data length does not match width * height"
        );
        Self {
            width,
            height,
            data,
            wrap,
        }
    }

    /// Returns the texture width in texels.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Returns the texture height in texels.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Returns the row-major height buffer.
    #[must_use]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Returns the configured wrap mode.
    #[must_use]
    pub const fn wrap(&self) -> WrapMode {
        self.wrap
    }

    /// Resolves a (possibly out-of-range) integer texel coordinate along one
    /// axis of `size` texels according to the wrap mode.
    fn resolve(&self, coord: i32, size: u32) -> usize {
        let size_i = size as i32;
        let wrapped = match self.wrap {
            WrapMode::Repeat => coord.rem_euclid(size_i),
            WrapMode::Clamp => coord.clamp(0, size_i - 1),
        };
        wrapped as usize
    }

    /// Fetches the raw height at an integer texel coordinate, applying the wrap
    /// mode to both axes.
    fn texel(&self, x: i32, y: i32) -> f32 {
        let xi = self.resolve(x, self.width);
        let yi = self.resolve(y, self.height);
        self.data[yi * self.width as usize + xi]
    }
}

impl DisplacementMap for TextureDisplacementMap {
    fn sample_height(&self, u: f32, v: f32) -> f32 {
        // Convert the normalized coordinate into continuous texel space using
        // the half-texel-center convention, then bilinearly blend the four
        // surrounding texels.
        let fx = u * self.width as f32 - 0.5;
        let fy = v * self.height as f32 - 0.5;
        let x0f = floorf(fx);
        let y0f = floorf(fy);
        let tx = fx - x0f;
        let ty = fy - y0f;
        let x0 = x0f as i32;
        let y0 = y0f as i32;

        let h00 = self.texel(x0, y0);
        let h10 = self.texel(x0 + 1, y0);
        let h01 = self.texel(x0, y0 + 1);
        let h11 = self.texel(x0 + 1, y0 + 1);

        let top = h00 + (h10 - h00) * tx;
        let bottom = h01 + (h11 - h01) * tx;
        top + (bottom - top) * ty
    }
}
