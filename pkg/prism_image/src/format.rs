//! The pixel formats an [`Image`](crate::Image) can carry, and their mapping
//! onto the RHI.

use prism_render_driver::TextureFormat;

/// The pixel format of an [`Image`](crate::Image).
///
/// The set is the `RGBA`/`BGRA` 8-bit family: every variant stores four bytes
/// per texel, which is what makes an image's buffer tightly packed at a fixed
/// stride of `width * 4` bytes per row. Each variant maps one-to-one onto an
/// RHI [`TextureFormat`], so an uploaded image keeps the exact numeric
/// interpretation (channel order and transfer function) it was authored with.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
#[non_exhaustive]
pub enum ImageFormat {
    /// Four 8-bit unsigned-normalized channels in `RGBA` order, linear.
    #[default]
    Rgba8Unorm,
    /// Four 8-bit unsigned-normalized channels in `RGBA` order, `sRGB`-encoded.
    Rgba8UnormSrgb,
    /// Four 8-bit unsigned-normalized channels in `BGRA` order, linear.
    ///
    /// This is the native layout of some windowing and compositor pipelines; it
    /// is kept distinct from [`Self::Rgba8Unorm`] so the GPU samples the
    /// intended channel order rather than a swapped one.
    Bgra8Unorm,
    /// Four 8-bit unsigned-normalized channels in `BGRA` order,
    /// `sRGB`-encoded.
    Bgra8UnormSrgb,
}

impl ImageFormat {
    /// The RHI texture format an image in this format uploads as.
    #[must_use]
    pub const fn texture_format(self) -> TextureFormat {
        match self {
            Self::Rgba8Unorm => TextureFormat::Rgba8Unorm,
            Self::Rgba8UnormSrgb => TextureFormat::Rgba8UnormSrgb,
            Self::Bgra8Unorm => TextureFormat::Bgra8Unorm,
            Self::Bgra8UnormSrgb => TextureFormat::Bgra8UnormSrgb,
        }
    }

    /// The number of bytes one texel occupies in an image's buffer.
    ///
    /// Always four for this family, so an image's rows are `width * 4` bytes
    /// apart with no padding.
    #[must_use]
    pub const fn bytes_per_texel(self) -> u32 {
        4
    }

    /// Whether the format carries an `sRGB` transfer function.
    #[must_use]
    pub const fn is_srgb(self) -> bool {
        matches!(self, Self::Rgba8UnormSrgb | Self::Bgra8UnormSrgb)
    }
}
