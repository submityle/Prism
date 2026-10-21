//! PNG import into a strict RGBA8 runtime artifact.

use super::ImporterPlugin;
use crate::job::{Processor, ProcessorError};
use crate::processor::{
    AnalyzeContext, ArtifactType, ArtifactWriter, ExecuteContext, ProcessPlan, ProcessReport,
    ProcessRequest, ProcessorCategory, ProcessorDescriptor,
};
use crate::recipe::ImportProcessor;
use alloc::vec::Vec;
use prism_image::{ImageArtifact, ImageArtifactError, ImageMipArtifact, MipGenerationSettings};
use std::io::{BufRead, BufReader, Cursor, Read, Seek, SeekFrom, Write};
use std::sync::OnceLock;

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
    /// The SDK accepts exactly this single primary input name.
    pub const SOURCE_INPUT: &'static str = "source";
    /// Finite source bound: default SDK analysis admits verification plus a 33-byte IHDR.
    pub const MAX_SOURCE_BYTES: u64 = 32 * 1024 * 1024;
    /// Maximum width or height, checked before decoder pixel allocations.
    pub const MAX_DIMENSION: u32 = 8192;
    /// Upper bound for the transformed decoder buffer and base RGBA8 pixels.
    pub const MAX_DECODED_BYTES: u64 = 64 * 1024 * 1024;
    /// Maximum aggregate RGBA8 bytes in the complete, ceil-halved mip chain.
    pub const MAX_MIP_BYTES: u64 = 64 * 1024 * 1024;
    /// Maximum complete encoded artifact, including the mip table.
    pub const MAX_ENCODED_BYTES: u64 = 64 * 1024 * 1024;
    /// Best-effort decoder allocation limit, separate from its caller-owned pixel buffer.
    pub const DECODER_MEMORY_BYTES: u64 = 8 * 1024 * 1024;
    /// The sole accepted SDK settings encoding. Empty, extended, or unknown settings fail.
    ///
    /// Contract: expand palette/transparency, strip 16-bit to 8-bit, ColorSrgb,
    /// Preserve alpha, complete mip chain; ignore text/ICC metadata, reject animation.
    /// All caps are implementation policy and versioned with this exact encoding.
    pub const CANONICAL_SETTINGS: &'static [u8] = b"prism.png.v3;rgba8;srgb;alpha=preserve;mips=full;metadata=ignore-text-icc;animation=reject;source=33554432;dimension=8192;decoded=67108864;mips-bytes=67108864;encoded=67108864;decoder=8388608";

    /// Canonical request settings for the SDK; no ambient defaults are inferred.
    pub fn canonical_settings() -> Vec<u8> {
        Self::CANONICAL_SETTINGS.to_vec()
    }

    /// Imports one PNG source into a deterministic RGBA8 artifact.
    pub fn import(bytes: &[u8]) -> Result<ImageArtifact, ImageImportError> {
        check_source_len(bytes.len() as u64).map_err(import_error)?;
        decode_png(Cursor::new(bytes)).map_err(import_error)
    }

    /// Imports one PNG into the complete deterministic RGBA8 mip artifact.
    pub fn import_mip_artifact(bytes: &[u8]) -> Result<ImageMipArtifact, ImageImportError> {
        mip_artifact(Self::import(bytes)?)
    }
}

fn import_error(error: impl std::fmt::Display) -> ImageImportError {
    ImageImportError::InvalidPng(error.to_string())
}

fn processor_error(error: impl std::fmt::Display) -> ProcessorError {
    ProcessorError::new(error.to_string())
}

fn check_source_len(len: u64) -> Result<(), ProcessorError> {
    if !(33..=CanonicalPngProcessor::MAX_SOURCE_BYTES).contains(&len) {
        return Err(processor_error(
            "PNG source byte limit exceeded or truncated IHDR",
        ));
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct PngLayout {
    base_bytes: u64,
    mip_bytes: u64,
    encoded_bytes: u64,
}

impl PngLayout {
    /// No allocations: size every ceil-halved level and its wire header first.
    fn new(mut width: u32, mut height: u32) -> Result<Self, ProcessorError> {
        if width == 0
            || height == 0
            || width > CanonicalPngProcessor::MAX_DIMENSION
            || height > CanonicalPngProcessor::MAX_DIMENSION
        {
            return Err(processor_error("PNG dimension limit exceeded"));
        }
        let base_bytes = u64::from(width) * u64::from(height) * 4;
        if base_bytes > CanonicalPngProcessor::MAX_DECODED_BYTES {
            return Err(processor_error("PNG decoded byte limit exceeded"));
        }
        let mut mip_bytes = 0;
        let mut levels = 0;
        loop {
            mip_bytes += u64::from(width) * u64::from(height) * 4;
            levels += 1;
            if mip_bytes > CanonicalPngProcessor::MAX_MIP_BYTES || levels > 32 {
                return Err(processor_error("PNG mip byte or level limit exceeded"));
            }
            if width == 1 && height == 1 {
                break;
            }
            width = width.div_ceil(2);
            height = height.div_ceil(2);
        }
        // PRIMG002: 28-byte header, then a 16-byte header per mip.
        let encoded_bytes = 28 + 16 * levels + mip_bytes;
        if encoded_bytes > CanonicalPngProcessor::MAX_ENCODED_BYTES {
            return Err(processor_error("PNG encoded byte limit exceeded"));
        }
        Ok(Self {
            base_bytes,
            mip_bytes,
            encoded_bytes,
        })
    }

    fn resources(self, source_len: u64) -> crate::BuildResources {
        crate::BuildResources {
            // Conservative simultaneous heap reservation: decoder scratch and
            // ancillary metadata copies (the decoder's limit is best-effort),
            // transformed/base pixels, mip generation + artifact/codec copies,
            // encoded output, buffered input and CAS verification scratch.
            memory_bytes: 3 * CanonicalPngProcessor::DECODER_MEMORY_BYTES
                + 2 * self.base_bytes
                + 4 * self.mip_bytes
                + self.encoded_bytes
                + 3 * 64 * 1024,
            // A verified open scans the source, then decoding reads it again.
            disk_read_bytes: 2 * source_len,
            disk_write_bytes: self.encoded_bytes,
            ..crate::BuildResources::default()
        }
    }
}

fn png_decoder<R: BufRead + Seek>(input: R) -> png::Decoder<R> {
    let mut decoder = png::Decoder::new_with_limits(
        input,
        png::Limits {
            bytes: CanonicalPngProcessor::DECODER_MEMORY_BYTES as usize,
        },
    );
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    decoder.set_ignore_text_chunk(true);
    decoder.set_ignore_iccp_chunk(true);
    decoder
}

fn decode_png<R: BufRead + Seek>(input: R) -> Result<ImageArtifact, ProcessorError> {
    let mut decoder = png_decoder(input);
    let header = decoder.read_header_info().map_err(processor_error)?;
    let layout = PngLayout::new(header.width, header.height)?;
    let mut reader = decoder.read_info().map_err(processor_error)?;
    if reader.info().animation_control.is_some() {
        return Err(processor_error("animated PNG is unsupported"));
    }
    let output_len = reader
        .output_buffer_size()
        .ok_or_else(|| processor_error("PNG decoded size overflow"))?;
    if output_len as u64 > layout.base_bytes {
        return Err(processor_error(
            "PNG decoded buffer exceeds analyzed RGBA8 bound",
        ));
    }
    let mut buffer = vec![0; output_len];
    let info = reader.next_frame(&mut buffer).map_err(processor_error)?;
    reader.finish().map_err(processor_error)?;
    // Release decoder scratch before allocating a converted pixel buffer.
    drop(reader);
    buffer.truncate(info.buffer_size());
    let rgba8 = match (info.color_type, info.bit_depth) {
        // RGBA8 (including expanded palette/tRNS) already has the target layout.
        (png::ColorType::Rgba, png::BitDepth::Eight) => buffer,
        (png::ColorType::Rgb, png::BitDepth::Eight) => {
            let mut rgba8 = Vec::with_capacity(layout.base_bytes as usize);
            for rgb in buffer.chunks_exact(3) {
                rgba8.extend_from_slice(&[rgb[0], rgb[1], rgb[2], 255]);
            }
            rgba8
        }
        (png::ColorType::Grayscale, png::BitDepth::Eight) => {
            let mut rgba8 = Vec::with_capacity(layout.base_bytes as usize);
            for &gray in &buffer {
                rgba8.extend_from_slice(&[gray, gray, gray, 255]);
            }
            rgba8
        }
        (png::ColorType::GrayscaleAlpha, png::BitDepth::Eight) => {
            let mut rgba8 = Vec::with_capacity(layout.base_bytes as usize);
            for gray_alpha in buffer.chunks_exact(2) {
                rgba8.extend_from_slice(&[
                    gray_alpha[0],
                    gray_alpha[0],
                    gray_alpha[0],
                    gray_alpha[1],
                ]);
            }
            rgba8
        }
        _ => return Err(processor_error(ImageImportError::UnsupportedFormat)),
    };
    let artifact = ImageArtifact {
        width: info.width,
        height: info.height,
        rgba8,
    };
    if artifact.rgba8.len() as u64 != layout.base_bytes {
        return Err(processor_error("PNG decoded dimensions differ from IHDR"));
    }
    Ok(artifact)
}

fn mip_artifact(base: ImageArtifact) -> Result<ImageMipArtifact, ImageImportError> {
    let chain = prism_image::ImageMipChain::generate(
        base.width,
        base.height,
        &base.rgba8,
        MipGenerationSettings {
            semantic: prism_image::TextureSemantic::ColorSrgb,
            alpha: prism_image::AlphaCoveragePolicy::Preserve,
            max_levels: 32,
            max_total_bytes: CanonicalPngProcessor::MAX_MIP_BYTES as usize,
        },
    )
    .map_err(|_| ImageImportError::Artifact(ImageArtifactError::InvalidMipChain))?;
    // The generated chain owns level zero; release the original before the
    // artifact conversion copies the chain into its public representation.
    drop(base);
    ImageMipArtifact::from_chain(&chain).map_err(ImageImportError::Artifact)
}

impl crate::processor::Processor for CanonicalPngProcessor {
    fn descriptor(&self) -> &ProcessorDescriptor {
        static DESCRIPTOR: OnceLock<ProcessorDescriptor> = OnceLock::new();
        DESCRIPTOR.get_or_init(|| ProcessorDescriptor {
            identity: Self::identity(),
            name: "Canonical PNG RGBA8 mip importer".into(),
            category: ProcessorCategory::Importer,
            input_types: vec![ArtifactType {
                kind: crate::source::SOURCE_ARTIFACT_KIND,
                schema: crate::source::SOURCE_ARTIFACT_SCHEMA,
            }],
            output_types: vec![ArtifactType {
                kind: IMAGE_ARTIFACT_KIND,
                schema: prism_image::IMAGE_MIP_ARTIFACT_SCHEMA,
            }],
            settings_schema: 0x5052_4953_4d5f_504e_475f_5345_545f_3033,
            deterministic: true,
            // PNG inflation and mip generation have no intra-loop polling.
            supports_cancellation: false,
        })
    }

    fn analyze(
        &self,
        ctx: &mut AnalyzeContext<'_>,
        request: &ProcessRequest,
    ) -> Result<ProcessPlan, ProcessorError> {
        if request.primary_inputs.len() != 1 || request.primary_inputs[0] != Self::SOURCE_INPUT {
            return Err(processor_error(
                "PNG importer expects one primary input named source",
            ));
        }
        if request.canonical_settings != Self::CANONICAL_SETTINGS {
            return Err(processor_error(
                "PNG importer requires exact canonical settings",
            ));
        }
        let mut input = ctx.open_input(Self::SOURCE_INPUT)?;
        let source_len = input.seek(SeekFrom::End(0)).map_err(processor_error)?;
        check_source_len(source_len)?;
        input.rewind().map_err(processor_error)?;
        let mut header = [0; 33];
        input.read_exact(&mut header).map_err(processor_error)?;
        // Let the mature decoder validate the fixed IHDR; no pixel allocation.
        if &header[..8] != b"\x89PNG\r\n\x1a\n"
            || header[8..12] != 13u32.to_be_bytes()
            || &header[12..16] != b"IHDR"
        {
            return Err(processor_error("invalid PNG IHDR"));
        }
        let mut decoder = png_decoder(Cursor::new(header));
        let info = decoder.read_header_info().map_err(processor_error)?;
        let layout = PngLayout::new(info.width, info.height)?;
        Ok(ProcessPlan {
            output: ArtifactType {
                kind: IMAGE_ARTIFACT_KIND,
                schema: prism_image::IMAGE_MIP_ARTIFACT_SCHEMA,
            },
            alignment: 4,
            canonical_settings: Self::canonical_settings(),
            resources: layout.resources(source_len),
            cacheable: true,
        })
    }

    fn execute(
        &self,
        ctx: &mut ExecuteContext<'_, '_>,
        plan: &ProcessPlan,
        output: &mut ArtifactWriter<'_>,
    ) -> Result<ProcessReport, ProcessorError> {
        ctx.check_cancelled()?;
        if plan.canonical_settings != Self::CANONICAL_SETTINGS {
            return Err(processor_error("invalid frozen PNG settings"));
        }
        let base = {
            let mut input = ctx.input(Self::SOURCE_INPUT)?;
            let len = input.seek(SeekFrom::End(0)).map_err(processor_error)?;
            check_source_len(len)?;
            input.rewind().map_err(processor_error)?;
            decode_png(BufReader::with_capacity(64 * 1024, input))?
        };
        ctx.check_cancelled()?;
        let layout = PngLayout::new(base.width, base.height)?;
        let artifact = mip_artifact(base).map_err(processor_error)?;
        ctx.check_cancelled()?;
        let encoded = artifact.encode().map_err(processor_error)?;
        if encoded.len() as u64 != layout.encoded_bytes {
            return Err(processor_error(
                "PNG encoded size differs from analyzed mip layout",
            ));
        }
        ctx.check_cancelled()?;
        for chunk in encoded.chunks(64 * 1024) {
            output.write_all(chunk).map_err(processor_error)?;
        }
        ctx.check_cancelled()?;
        Ok(ProcessReport {
            output_bytes: output.bytes_written(),
        })
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
    const IMPORTER_VERSION: u32 = 3;
}

impl ImporterPlugin for CanonicalPngProcessor {
    const EXTENSIONS: &'static [&'static str] = &["png"];
    const MIME_TYPES: &'static [&'static str] = &["image/png"];
}

#[cfg(test)]
mod tests {
    use super::*;
    use png::Encoder;

    #[test]
    fn source_length_policy_has_closed_finite_boundaries() {
        assert!(check_source_len(32).is_err());
        assert!(check_source_len(33).is_ok());
        assert!(check_source_len(CanonicalPngProcessor::MAX_SOURCE_BYTES).is_ok());
        assert!(check_source_len(CanonicalPngProcessor::MAX_SOURCE_BYTES + 1).is_err());
        assert!(check_source_len(u64::MAX).is_err());
    }

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
