//! What an image can be wrong about.

use core::fmt;

/// An image that cannot be uploaded as authored.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ImageError {
    /// One of the extents is zero: there is no texel to upload.
    EmptyExtent,
    /// The texel data does not match the extents and format.
    ///
    /// Every byte would be interpreted as texels, so a short buffer would read
    /// past its own end and a long one would upload junk.
    SizeMismatch {
        /// The byte length the extents and format require.
        expected: usize,
        /// The byte length the data has.
        found: usize,
    },
    /// The format describes depth (or depth + stencil), not colour.
    ///
    /// A depth attachment is not sampleable content and has no texel colours; an
    /// image holding "depth data" is a mistake in the material, not a texture.
    NotColor {
        /// `true` when the format also carries a stencil plane.
        depth_stencil: bool,
    },
}

impl fmt::Display for ImageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyExtent => write!(f, "the image has a zero-sized axis"),
            Self::SizeMismatch { expected, found } => write!(
                f,
                "the image data is {found} bytes, the extents and format require {expected}"
            ),
            Self::NotColor { depth_stencil } => write!(
                f,
                "the format is a depth{} format, not a colour format",
                if *depth_stencil { "/stencil" } else { "" }
            ),
        }
    }
}
