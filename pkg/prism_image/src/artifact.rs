//! Versioned canonical image artifact consumed by build and runtime layers.

use alloc::vec::Vec;

use crate::{Image, ImageError};

/// Schema version for canonical RGBA8 image artifacts.
pub const IMAGE_ARTIFACT_SCHEMA: u32 = 1;
const MAGIC: &[u8; 8] = b"PRIMG001";
const HEADER_LEN: usize = 8 + 4 + 4 + 4 + 4;

/// A deterministic, tightly-packed RGBA8 image artifact.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImageArtifact {
    /// Image width in texels.
    pub width: u32,
    /// Image height in texels.
    pub height: u32,
    /// Row-major RGBA8 texels.
    pub rgba8: Vec<u8>,
}

/// Failure while decoding a canonical image artifact.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageArtifactError {
    /// Header magic or schema is invalid.
    InvalidHeader,
    /// Payload length does not match dimensions.
    InvalidDimensions,
    /// Payload ended before its declared length.
    Truncated,
    /// Bytes remained after the complete artifact.
    TrailingBytes,
}

impl core::fmt::Display for ImageArtifactError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidHeader => f.write_str("invalid canonical image artifact header"),
            Self::InvalidDimensions => f.write_str("canonical image dimensions are invalid"),
            Self::Truncated => f.write_str("canonical image artifact is truncated"),
            Self::TrailingBytes => f.write_str("canonical image artifact has trailing bytes"),
        }
    }
}

impl std::error::Error for ImageArtifactError {}

impl ImageArtifact {
    /// Encodes the canonical artifact wire format.
    pub fn encode(&self) -> Result<Vec<u8>, ImageArtifactError> {
        let expected = pixel_len(self.width, self.height)?;
        if self.rgba8.len() != expected {
            return Err(ImageArtifactError::InvalidDimensions);
        }
        let payload_len =
            u32::try_from(self.rgba8.len()).map_err(|_| ImageArtifactError::InvalidDimensions)?;
        let mut bytes = Vec::with_capacity(HEADER_LEN + self.rgba8.len());
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&IMAGE_ARTIFACT_SCHEMA.to_le_bytes());
        bytes.extend_from_slice(&self.width.to_le_bytes());
        bytes.extend_from_slice(&self.height.to_le_bytes());
        bytes.extend_from_slice(&payload_len.to_le_bytes());
        bytes.extend_from_slice(&self.rgba8);
        Ok(bytes)
    }

    /// Decodes and validates one canonical artifact.
    pub fn decode(bytes: &[u8]) -> Result<Self, ImageArtifactError> {
        if bytes.len() < HEADER_LEN || &bytes[..8] != MAGIC {
            return Err(ImageArtifactError::InvalidHeader);
        }
        if u32::from_le_bytes(bytes[8..12].try_into().unwrap()) != IMAGE_ARTIFACT_SCHEMA {
            return Err(ImageArtifactError::InvalidHeader);
        }
        let width = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let height = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
        let payload_len = usize::try_from(u32::from_le_bytes(bytes[20..24].try_into().unwrap()))
            .map_err(|_| ImageArtifactError::InvalidDimensions)?;
        if payload_len != pixel_len(width, height)? {
            return Err(ImageArtifactError::InvalidDimensions);
        }
        let end = HEADER_LEN
            .checked_add(payload_len)
            .ok_or(ImageArtifactError::InvalidDimensions)?;
        if end > bytes.len() {
            return Err(ImageArtifactError::Truncated);
        }
        if end != bytes.len() {
            return Err(ImageArtifactError::TrailingBytes);
        }
        Ok(Self {
            width,
            height,
            rgba8: bytes[HEADER_LEN..end].to_vec(),
        })
    }

    /// Converts the artifact into the runtime image asset with its default sampler.
    pub fn to_image(&self) -> Result<Image, ImageError> {
        Image::new(
            prism_render_driver::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
            prism_render_driver::TextureFormat::Rgba8Unorm,
            self.rgba8.clone(),
        )
    }
}

fn pixel_len(width: u32, height: u32) -> Result<usize, ImageArtifactError> {
    usize::try_from(u64::from(width) * u64::from(height) * 4)
        .map_err(|_| ImageArtifactError::InvalidDimensions)
}
