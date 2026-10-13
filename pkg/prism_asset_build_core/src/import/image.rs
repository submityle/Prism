//! PNG import into a strict RGBA8 runtime artifact.

use crate::job::{Processor, ProcessorError};
use alloc::vec::Vec;
use std::io::Cursor;

/// Runtime artifact kind for canonical RGBA8 images.
pub const IMAGE_ARTIFACT_KIND: u128 = 0x5052_4953_4d5f_494d_4147_455f_5238;
/// Version of the canonical RGBA8 image artifact wire format.
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

/// PNG import or canonical artifact validation failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ImageImportError {
    /// The source is not a decodable PNG or the decoder rejected it.
    InvalidPng(String),
    /// The PNG uses a color/depth combination this artifact contract does not support.
    UnsupportedFormat,
    /// Image dimensions or payload length overflow the artifact contract.
    InvalidDimensions,
    /// The canonical artifact ended before all declared bytes were available.
    Truncated,
    /// The canonical artifact header or version is invalid.
    InvalidArtifact,
}

impl std::fmt::Display for ImageImportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidPng(error) => write!(f, "invalid PNG: {error}"),
            Self::UnsupportedFormat => f.write_str("PNG format cannot be normalized to RGBA8"),
            Self::InvalidDimensions => f.write_str("image dimensions overflow the RGBA8 artifact"),
            Self::Truncated => f.write_str("RGBA8 image artifact is truncated"),
            Self::InvalidArtifact => f.write_str("invalid RGBA8 image artifact header"),
        }
    }
}

impl std::error::Error for ImageImportError {}

impl ImageArtifact {
    /// Encodes the canonical artifact wire format.
    pub fn encode(&self) -> Result<Vec<u8>, ImageImportError> {
        let expected = pixel_len(self.width, self.height)?;
        if self.rgba8.len() != expected {
            return Err(ImageImportError::InvalidDimensions);
        }
        let payload_len =
            u32::try_from(self.rgba8.len()).map_err(|_| ImageImportError::InvalidDimensions)?;
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
    pub fn decode(bytes: &[u8]) -> Result<Self, ImageImportError> {
        if bytes.len() < HEADER_LEN || &bytes[..8] != MAGIC {
            return Err(ImageImportError::InvalidArtifact);
        }
        let schema = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if schema != IMAGE_ARTIFACT_SCHEMA {
            return Err(ImageImportError::InvalidArtifact);
        }
        let width = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let height = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
        let payload_len = usize::try_from(u32::from_le_bytes(bytes[20..24].try_into().unwrap()))
            .map_err(|_| ImageImportError::InvalidDimensions)?;
        let expected = pixel_len(width, height)?;
        if payload_len != expected {
            return Err(ImageImportError::InvalidDimensions);
        }
        let end = HEADER_LEN
            .checked_add(payload_len)
            .ok_or(ImageImportError::InvalidDimensions)?;
        if end > bytes.len() {
            return Err(ImageImportError::Truncated);
        }
        if end != bytes.len() {
            return Err(ImageImportError::InvalidArtifact);
        }
        Ok(Self {
            width,
            height,
            rgba8: bytes[HEADER_LEN..end].to_vec(),
        })
    }
}

/// Decodes PNG with the mature `png` crate and emits the canonical RGBA8 artifact.
#[derive(Clone, Copy, Debug, Default)]
pub struct CanonicalPngProcessor;

impl CanonicalPngProcessor {
    /// Imports one PNG source into a deterministic RGBA8 artifact.
    pub fn import(bytes: &[u8]) -> Result<ImageArtifact, ImageImportError> {
        let mut decoder = png::Decoder::new(Cursor::new(bytes));
        decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
        let mut reader = decoder
            .read_info()
            .map_err(|error| ImageImportError::InvalidPng(error.to_string()))?;
        let output_len = reader
            .output_buffer_size()
            .ok_or(ImageImportError::InvalidDimensions)?;
        let mut buffer = vec![0; output_len];
        let info = reader
            .next_frame(&mut buffer)
            .map_err(|error| ImageImportError::InvalidPng(error.to_string()))?;
        let source = &buffer[..info.buffer_size()];
        let rgba8 = match (info.color_type, info.bit_depth) {
            (png::ColorType::Rgba, png::BitDepth::Eight) => source.to_vec(),
            (png::ColorType::Rgb, png::BitDepth::Eight) => source
                .chunks_exact(3)
                .flat_map(|rgb| [rgb[0], rgb[1], rgb[2], 255])
                .collect(),
            (png::ColorType::Grayscale, png::BitDepth::Eight) => source
                .iter()
                .flat_map(|&gray| [gray, gray, gray, 255])
                .collect(),
            (png::ColorType::GrayscaleAlpha, png::BitDepth::Eight) => source
                .chunks_exact(2)
                .flat_map(|gray_alpha| [gray_alpha[0], gray_alpha[0], gray_alpha[0], gray_alpha[1]])
                .collect(),
            _ => return Err(ImageImportError::UnsupportedFormat),
        };
        let artifact = ImageArtifact {
            width: info.width,
            height: info.height,
            rgba8,
        };
        artifact.encode()?;
        Ok(artifact)
    }
}

impl Processor for CanonicalPngProcessor {
    fn process(&self, inputs: &[&[u8]]) -> Result<Vec<u8>, ProcessorError> {
        let [source] = inputs else {
            return Err(ProcessorError::new(
                "PNG importer expects exactly one source",
            ));
        };
        Self::import(source)
            .and_then(|artifact| artifact.encode())
            .map_err(|error| ProcessorError::new(error.to_string()))
    }
}

fn pixel_len(width: u32, height: u32) -> Result<usize, ImageImportError> {
    usize::try_from(u64::from(width) * u64::from(height) * 4)
        .map_err(|_| ImageImportError::InvalidDimensions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use png::Encoder;

    fn png_bytes(color: png::ColorType, data: &[u8], width: u32, height: u32) -> Vec<u8> {
        let mut bytes = Vec::new();
        let mut encoder = Encoder::new(&mut bytes, width, height);
        encoder.set_color(color);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();
        writer.write_image_data(data).unwrap();
        drop(writer);
        bytes
    }

    #[test]
    fn imports_rgb_png_to_canonical_rgba8() {
        let source = png_bytes(png::ColorType::Rgb, &[1, 2, 3, 4, 5, 6], 2, 1);
        let artifact = CanonicalPngProcessor::import(&source).unwrap();
        assert_eq!(artifact.width, 2);
        assert_eq!(artifact.height, 1);
        assert_eq!(artifact.rgba8, vec![1, 2, 3, 255, 4, 5, 6, 255]);
        assert_eq!(
            ImageArtifact::decode(&artifact.encode().unwrap()).unwrap(),
            artifact
        );
    }

    #[test]
    fn rejects_invalid_png_and_tampered_artifact() {
        assert!(matches!(
            CanonicalPngProcessor::import(b"not png"),
            Err(ImageImportError::InvalidPng(_))
        ));
        let source = png_bytes(png::ColorType::Rgba, &[1, 2, 3, 4], 1, 1);
        let bytes = CanonicalPngProcessor::import(&source)
            .unwrap()
            .encode()
            .unwrap();
        assert_eq!(
            ImageArtifact::decode(&bytes[..bytes.len() - 1]),
            Err(ImageImportError::Truncated)
        );
    }
}
