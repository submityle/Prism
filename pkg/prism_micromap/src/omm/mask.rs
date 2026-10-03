//! The alpha-coverage mask abstraction and a texture-backed implementation.

use alloc::vec::Vec;

/// Addressing mode applied to texture coordinates outside `0..=1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WrapMode {
    /// Wrap coordinates modulo `1` (tiling), matching `GPU` repeat sampling.
    Repeat,
    /// Clamp coordinates to the `[0, 1]` edge.
    Clamp,
}

impl WrapMode {
    /// Applies the wrap mode to a single coordinate, returning a value in
    /// `[0, 1)` for [`WrapMode::Repeat`] or `[0, 1]` for [`WrapMode::Clamp`].
    #[must_use]
    pub fn apply(self, coord: f32) -> f32 {
        match self {
            Self::Clamp => coord.clamp(0.0, 1.0),
            Self::Repeat => {
                let fract = coord - libm::floorf(coord);
                // `floorf` of a negative value already yields a non-negative
                // fract, but guard the exact-integer boundary explicitly.
                if fract < 0.0 {
                    fract + 1.0
                } else {
                    fract
                }
            }
        }
    }
}

/// A coverage mask queryable at arbitrary texture coordinates.
///
/// Implementations return a scalar alpha in `[0, 1]`; the baker compares it
/// against [`AlphaMask::threshold`] to decide opacity. Keeping this a trait
/// lets callers back the baker with a decoded texture, a procedural mask, or a
/// test fixture without the baker depending on any image format.
pub trait AlphaMask {
    /// Samples the alpha coverage at texture coordinate `(u, v)`.
    fn sample_alpha(&self, u: f32, v: f32) -> f32;

    /// The alpha cutoff: samples with `alpha >= threshold` are opaque.
    fn threshold(&self) -> f32;

    /// Convenience predicate: `true` when the sample at `(u, v)` is opaque.
    fn is_opaque(&self, u: f32, v: f32) -> bool {
        self.sample_alpha(u, v) >= self.threshold()
    }

    /// Grid resolution `(width, height)` in texels when the mask is backed by a
    /// dense texture, enabling the exact texel-conservative classification
    /// path. Returns [`None`] for procedural or resolution-free masks.
    fn grid_resolution(&self) -> Option<(u32, u32)> {
        None
    }

    /// Fetches the alpha of texel `(x, y)` when [`AlphaMask::grid_resolution`]
    /// is [`Some`]. The default samples the texel center and is only meaningful
    /// once a grid resolution is reported.
    fn fetch_texel(&self, x: u32, y: u32) -> f32 {
        match self.grid_resolution() {
            Some((w, h)) => {
                let u = (x as f32 + 0.5) / w as f32;
                let v = (y as f32 + 0.5) / h as f32;
                self.sample_alpha(u, v)
            }
            None => f32::NAN,
        }
    }
}

/// A dense, row-major alpha texture sampled with nearest-texel filtering.
///
/// Nearest filtering is deliberate: an `OMM` baker must classify against the
/// exact texels the hardware alpha test will read, and nearest sampling keeps
/// the `CPU` golden and the `GPU` twin bit-identical.
#[derive(Debug, Clone, PartialEq)]
pub struct TextureAlphaMask {
    width: u32,
    height: u32,
    alpha: Vec<f32>,
    threshold: f32,
    wrap: WrapMode,
}

impl TextureAlphaMask {
    /// Creates a mask from a row-major `width * height` alpha buffer using
    /// [`WrapMode::Clamp`].
    ///
    /// # Panics
    ///
    /// Panics when `alpha.len() != width * height`, when either dimension is
    /// zero, or when `width * height` overflows [`usize`].
    #[must_use]
    pub fn new(width: u32, height: u32, alpha: Vec<f32>, threshold: f32) -> Self {
        Self::with_wrap(width, height, alpha, threshold, WrapMode::Clamp)
    }

    /// Creates a mask with an explicit [`WrapMode`].
    ///
    /// # Panics
    ///
    /// Panics when `alpha.len() != width * height`, when either dimension is
    /// zero, or when `width * height` overflows [`usize`].
    #[must_use]
    pub fn with_wrap(
        width: u32,
        height: u32,
        alpha: Vec<f32>,
        threshold: f32,
        wrap: WrapMode,
    ) -> Self {
        assert!(width > 0 && height > 0, "mask dimensions must be non-zero");
        let expected = (width as usize)
            .checked_mul(height as usize)
            .expect("mask dimensions overflow usize");
        assert_eq!(alpha.len(), expected, "alpha buffer length mismatch");
        Self {
            width,
            height,
            alpha,
            threshold,
            wrap,
        }
    }

    /// The mask width in texels.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// The mask height in texels.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// The configured wrap mode.
    #[must_use]
    pub const fn wrap(&self) -> WrapMode {
        self.wrap
    }

    /// Fetches the alpha of texel `(x, y)` without filtering or wrapping.
    ///
    /// Returns [`None`] when the coordinate is out of range.
    #[must_use]
    pub fn texel(&self, x: u32, y: u32) -> Option<f32> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let idx = y as usize * self.width as usize + x as usize;
        Some(self.alpha[idx])
    }

    /// Converts a wrapped coordinate in `[0, 1]` to a texel index on `size`
    /// texels, clamped to the last texel.
    #[inline]
    fn coord_to_texel(coord: f32, size: u32) -> u32 {
        let scaled = libm::floorf(coord * size as f32);
        if scaled <= 0.0 {
            0
        } else {
            let idx = scaled as u32;
            idx.min(size - 1)
        }
    }
}

impl AlphaMask for TextureAlphaMask {
    fn sample_alpha(&self, u: f32, v: f32) -> f32 {
        let wu = self.wrap.apply(u);
        let wv = self.wrap.apply(v);
        let x = Self::coord_to_texel(wu, self.width);
        let y = Self::coord_to_texel(wv, self.height);
        let idx = y as usize * self.width as usize + x as usize;
        self.alpha[idx]
    }

    fn threshold(&self) -> f32 {
        self.threshold
    }

    fn grid_resolution(&self) -> Option<(u32, u32)> {
        Some((self.width, self.height))
    }

    fn fetch_texel(&self, x: u32, y: u32) -> f32 {
        self.texel(x, y).unwrap_or(0.0)
    }
}
