//! PNG import into a strict RGBA8 runtime artifact.

use super::ImporterPlugin;
use crate::job::{Processor, ProcessorError};
use crate::recipe::ImportProcessor;
use alloc::vec::Vec;
use prism_image::{ImageArtifact, ImageArtifactError, ImageMipArtifact, MipGenerationSettings};
use std::io::Cursor;

/// Runtime artifact kind for canonical RGBA8 images.
pub const IMAGE_ARTIFACT_KIND: u128 = 0x5052_4953_4d5f_494d_4147_455f_5238;
/// PNG import or canonical artifact validation failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ImageImportError {
    /// The source is not a decodable PNG or the decoder rejected it.
    InvalidPng(String),
    /// The PNG uses a color/depth combination this artifact contract does not support.
    UnsupportedFormat,
    /// Canonical runtime artifact validation failed.
    Artifact(ImageArtifactError),
}

impl std::fmt::Display for ImageImportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidPng(error) => write!(f, "invalid PNG: {error}"),
            Self::UnsupportedFormat => f.write_str("PNG format cannot be normalized to RGBA8"),
            Self::Artifact(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for ImageImportError {}

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
            .ok_or(ImageImportError::Artifact(
                ImageArtifactError::InvalidDimensions,
            ))?;
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
        artifact.encode().map_err(ImageImportError::Artifact)?;
        Ok(artifact)
    }

    /// Imports one PNG into the complete deterministic RGBA8 mip artifact.
    pub fn import_mip_artifact(bytes: &[u8]) -> Result<ImageMipArtifact, ImageImportError> {
        let base = Self::import(bytes)?;
        let chain = prism_image::ImageMipChain::generate(
            base.width,
            base.height,
            &base.rgba8,
            MipGenerationSettings::default(),
        )
        .map_err(|_| ImageImportError::Artifact(ImageArtifactError::InvalidMipChain))?;
        ImageMipArtifact::from_chain(&chain).map_err(ImageImportError::Artifact)
    }
}

impl Processor for CanonicalPngProcessor {
    fn process(&self, inputs: &[&[u8]]) -> Result<Vec<u8>, ProcessorError> {
        let [source] = inputs else {
            return Err(ProcessorError::new(
                "PNG importer expects exactly one source",
            ));
        };
        Self::import_mip_artifact(source)
            .and_then(|artifact| artifact.encode().map_err(ImageImportError::Artifact))
            .map_err(|error| ProcessorError::new(error.to_string()))
    }
}

impl ImportProcessor for CanonicalPngProcessor {
    const IMPORTER_ID: u128 = 0x504e47;
    const IMPORTER_VERSION: u32 = 2;
}

impl ImporterPlugin for CanonicalPngProcessor {
    const EXTENSIONS: &'static [&'static str] = &["png"];
    const MIME_TYPES: &'static [&'static str] = &["image/png"];
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
    fn process_emits_complete_odd_dimension_mip_chain() {
        let source = png_bytes(
            png::ColorType::Rgba,
            &(0..(3 * 5 * 4))
                .map(|value| value as u8)
                .collect::<Vec<_>>(),
            3,
            5,
        );
        let bytes = CanonicalPngProcessor.process(&[&source]).unwrap();
        let artifact = ImageMipArtifact::decode(&bytes).unwrap();
        assert_eq!((artifact.width, artifact.height), (3, 5));
        assert_eq!(
            artifact
                .levels
                .iter()
                .map(|level| (level.width, level.height))
                .collect::<Vec<_>>(),
            vec![(3, 5), (2, 3), (1, 2), (1, 1)]
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
            Err(ImageArtifactError::Truncated)
        );
    }
}
