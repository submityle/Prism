//! The CPU-side [`Image`] container: tightly packed pixel bytes and their
//! extent, with no GPU objects attached.

use prism_render_driver::{Extent3d, ImageDataLayout, RenderDevice, RenderQueue};

use crate::format::ImageFormat;
use crate::gpu::{self, GpuImage};

/// A CPU-side image: a byte buffer of tightly packed texels plus the extent
/// that gives those bytes their shape.
///
/// Rows are contiguous with a stride of `width * bytes_per_texel` and no
/// padding; for the [`ImageFormat`] family this crate supports that is always
/// four bytes per texel. The buffer is layer-major for multi-layer extents.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Image {
    extent: Extent3d,
    format: ImageFormat,
    data: Vec<u8>,
}

/// Why an [`Image`] could not be constructed from raw bytes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ImageError {
    /// The buffer length was not exactly
    /// `width * height * depth_or_array_layers * bytes_per_texel`.
    LengthMismatch {
        /// The number of bytes the extent requires.
        expected: usize,
        /// The number of bytes supplied.
        actual: usize,
    },
    /// The extent had a zero axis, which cannot become a GPU texture.
    ZeroExtent,
}

impl core::fmt::Display for ImageError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::LengthMismatch { expected, actual } => write!(
                f,
                "image byte length mismatch: expected {expected} bytes, got {actual}"
            ),
            Self::ZeroExtent => f.write_str("image extent has a zero axis"),
        }
    }
}

impl std::error::Error for ImageError {}

impl Image {
    /// Builds an image in `format` from a tightly packed `data` buffer.
    ///
    /// # Errors
    ///
    /// Returns [`ImageError::ZeroExtent`] when any extent axis is zero, and
    /// [`ImageError::LengthMismatch`] when `data` is not exactly the number of
    /// bytes the extent requires.
    pub fn new(extent: Extent3d, format: ImageFormat, data: Vec<u8>) -> Result<Self, ImageError> {
        if extent.width == 0 || extent.height == 0 || extent.depth_or_array_layers == 0 {
            return Err(ImageError::ZeroExtent);
        }
        let expected = expected_byte_len(extent, format);
        if data.len() != expected {
            return Err(ImageError::LengthMismatch {
                expected,
                actual: data.len(),
            });
        }
        Ok(Self::from_parts(extent, format, data))
    }

    /// Builds a single-layer [`ImageFormat::Rgba8Unorm`] image from decoded
    /// `RGBA` bytes.
    ///
    /// This is the entry point for externally decoded data (a decoder, or a
    /// texture the application generated itself): the buffer is expected to be
    /// `width * height * 4` bytes of tightly packed `RGBA8`.
    ///
    /// # Errors
    ///
    /// Returns [`ImageError::LengthMismatch`] unless `data` is exactly
    /// `width * height * 4` bytes, and [`ImageError::ZeroExtent`] if either
    /// dimension is zero.
    pub fn from_rgba8(width: u32, height: u32, data: Vec<u8>) -> Result<Self, ImageError> {
        Self::new(
            Extent3d::new_2d(width, height),
            ImageFormat::Rgba8Unorm,
            data,
        )
    }

    /// Builds an image from parts the caller has already validated.
    ///
    /// Crate-internal: used by the generators, which compute a buffer whose
    /// length matches the extent by construction.
    pub(crate) fn from_parts(extent: Extent3d, format: ImageFormat, data: Vec<u8>) -> Self {
        Self {
            extent,
            format,
            data,
        }
    }

    /// The image's extent in texels.
    #[must_use]
    pub const fn extent(&self) -> Extent3d {
        self.extent
    }

    /// The image's width in texels.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.extent.width
    }

    /// The image's height in texels.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.extent.height
    }

    /// The image's pixel format.
    #[must_use]
    pub const fn format(&self) -> ImageFormat {
        self.format
    }

    /// The tightly packed pixel bytes.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// The byte distance between the starts of consecutive rows.
    #[must_use]
    pub fn row_stride(&self) -> usize {
        self.extent.width as usize * self.format.bytes_per_texel() as usize
    }

    /// The total size of the pixel buffer in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the pixel buffer is empty (a degenerate zero-extent image).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// The `(x, y)` texel as an `[r, g, b, a]` byte quad.
    ///
    /// # Panics
    ///
    /// Panics if `x >= width` or `y >= height`.
    #[must_use]
    pub fn texel(&self, x: u32, y: u32) -> [u8; 4] {
        assert!(
            x < self.extent.width && y < self.extent.height,
            "texel ({x}, {y}) is outside the {}x{} image",
            self.extent.width,
            self.extent.height,
        );
        let offset = y as usize * self.row_stride() + x as usize * 4;
        [
            self.data[offset],
            self.data[offset + 1],
            self.data[offset + 2],
            self.data[offset + 3],
        ]
    }

    /// The RHI [upload layout](ImageDataLayout) describing this image's buffer.
    ///
    /// Offsets and strides are explicit rather than `None` so an upload can
    /// never be confused by a backend that infers a different row pitch.
    #[must_use]
    pub fn data_layout(&self) -> ImageDataLayout {
        ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(self.row_stride() as u32),
            rows_per_image: Some(self.extent.height),
        }
    }

    /// Uploads the image into a fresh GPU texture and returns its handles.
    ///
    /// See [`gpu::upload`] for the destination texture's exact shape and
    /// usages.
    #[must_use]
    pub fn upload<D: RenderDevice, Q: RenderQueue>(&self, device: &D, queue: &Q) -> GpuImage {
        gpu::upload(self, device, queue)
    }
}

/// The number of bytes a tightly packed buffer of `extent` in `format` needs.
fn expected_byte_len(extent: Extent3d, format: ImageFormat) -> usize {
    let texels = u64::from(extent.width)
        * u64::from(extent.height)
        * u64::from(extent.depth_or_array_layers);
    let bytes = texels * u64::from(format.bytes_per_texel());
    usize::try_from(bytes).unwrap_or(usize::MAX)
}
