//! Versioned canonical image artifact consumed by build and runtime layers.

use alloc::vec::Vec;

use crate::{
    AlphaCoveragePolicy, Image, ImageError, ImageMipChain, ImageMipLevel, MipGenerationSettings,
    TextureSemantic,
};

/// Schema version for canonical RGBA8 image artifacts.
pub const IMAGE_ARTIFACT_SCHEMA: u32 = 1;
/// Schema version for the complete semantic RGBA8 mip-chain artifact.
pub const IMAGE_MIP_ARTIFACT_SCHEMA: u32 = 2;
const MAGIC: &[u8; 8] = b"PRIMG001";
const MIP_MAGIC: &[u8; 8] = b"PRIMG002";
const HEADER_LEN: usize = 8 + 4 + 4 + 4 + 4;
const MIP_HEADER_LEN: usize = 8 + 4 + 4 + 4 + 1 + 1 + 2 + 4;
const MIP_LEVEL_HEADER_LEN: usize = 4 + 4 + 8;
const MAX_ARTIFACT_LEVELS: usize = 32;
const MAX_ARTIFACT_BYTES: usize = 256 * 1024 * 1024;

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
    /// A mip-chain metadata tag or level table is invalid.
    InvalidMipChain,
    /// The encoded artifact exceeds the bounded decoder budget.
    BudgetExceeded,
}

impl core::fmt::Display for ImageArtifactError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidHeader => f.write_str("invalid canonical image artifact header"),
            Self::InvalidDimensions => f.write_str("canonical image dimensions are invalid"),
            Self::Truncated => f.write_str("canonical image artifact is truncated"),
            Self::TrailingBytes => f.write_str("canonical image artifact has trailing bytes"),
            Self::InvalidMipChain => f.write_str("canonical image mip chain is invalid"),
            Self::BudgetExceeded => f.write_str("canonical image artifact exceeds its byte budget"),
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
        if bytes.len() >= 8 && &bytes[..8] == MIP_MAGIC {
            let full = ImageMipArtifact::decode(bytes)?;
            let base = full
                .levels
                .first()
                .ok_or(ImageArtifactError::InvalidMipChain)?;
            return Ok(Self {
                width: full.width,
                height: full.height,
                rgba8: base.rgba8.clone(),
            });
        }
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

    /// Encodes this level-zero image together with a generated complete mip chain.
    pub fn encode_with_mips(
        &self,
        settings: MipGenerationSettings,
    ) -> Result<Vec<u8>, ImageArtifactError> {
        let chain = ImageMipChain::generate(self.width, self.height, &self.rgba8, settings)
            .map_err(|_| ImageArtifactError::InvalidMipChain)?;
        ImageMipArtifact::from_chain(&chain)?.encode()
    }
}

/// One encoded level in a complete canonical image artifact.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImageArtifactMipLevel {
    /// Level width in texels.
    pub width: u32,
    /// Level height in texels.
    pub height: u32,
    /// Tightly packed RGBA8 bytes.
    pub rgba8: Vec<u8>,
}

/// Complete semantic mip-chain artifact used by the build and runtime paths.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImageMipArtifact {
    /// Level-zero width.
    pub width: u32,
    /// Level-zero height.
    pub height: u32,
    /// Texture reduction semantic preserved in the artifact metadata.
    pub semantic: TextureSemantic,
    /// Alpha reduction rule preserved in the artifact metadata.
    pub alpha: AlphaCoveragePolicy,
    /// Strictly ordered level-zero-to-one-texel levels.
    pub levels: Vec<ImageArtifactMipLevel>,
}

impl ImageMipArtifact {
    /// Converts a validated in-memory chain to its wire representation.
    pub fn from_chain(chain: &ImageMipChain) -> Result<Self, ImageArtifactError> {
        if chain.levels().is_empty() || chain.levels().len() > MAX_ARTIFACT_LEVELS {
            return Err(ImageArtifactError::InvalidMipChain);
        }
        Ok(Self {
            width: chain.level(0).unwrap().width(),
            height: chain.level(0).unwrap().height(),
            semantic: chain.semantic(),
            alpha: chain.alpha_policy(),
            levels: chain
                .levels()
                .iter()
                .map(|level| ImageArtifactMipLevel {
                    width: level.width(),
                    height: level.height(),
                    rgba8: level.rgba8().to_vec(),
                })
                .collect(),
        })
    }

    /// Encodes the strict versioned mip-chain wire format.
    pub fn encode(&self) -> Result<Vec<u8>, ImageArtifactError> {
        let chain = self.to_chain()?;
        let total = chain.total_bytes();
        if total > MAX_ARTIFACT_BYTES || chain.levels().len() > MAX_ARTIFACT_LEVELS {
            return Err(ImageArtifactError::BudgetExceeded);
        }
        let mut bytes = Vec::with_capacity(
            MIP_HEADER_LEN + chain.levels().len() * MIP_LEVEL_HEADER_LEN + total,
        );
        bytes.extend_from_slice(MIP_MAGIC);
        bytes.extend_from_slice(&IMAGE_MIP_ARTIFACT_SCHEMA.to_le_bytes());
        bytes.extend_from_slice(&self.width.to_le_bytes());
        bytes.extend_from_slice(&self.height.to_le_bytes());
        bytes.push(semantic_tag(self.semantic));
        let (alpha_tag, threshold) = alpha_tag(self.alpha);
        bytes.push(alpha_tag);
        bytes.extend_from_slice(&threshold.to_le_bytes());
        bytes.extend_from_slice(&(chain.levels().len() as u32).to_le_bytes());
        for level in chain.levels() {
            bytes.extend_from_slice(&level.width().to_le_bytes());
            bytes.extend_from_slice(&level.height().to_le_bytes());
            bytes.extend_from_slice(&(level.rgba8().len() as u64).to_le_bytes());
            bytes.extend_from_slice(level.rgba8());
        }
        Ok(bytes)
    }

    /// Decodes and fully validates a mip-chain artifact, including every byte.
    pub fn decode(bytes: &[u8]) -> Result<Self, ImageArtifactError> {
        if bytes.len() < MIP_HEADER_LEN || &bytes[..8] != MIP_MAGIC {
            return Err(ImageArtifactError::InvalidHeader);
        }
        if u32::from_le_bytes(bytes[8..12].try_into().unwrap()) != IMAGE_MIP_ARTIFACT_SCHEMA {
            return Err(ImageArtifactError::InvalidHeader);
        }
        let width = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let height = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
        let semantic = decode_semantic(bytes[20])?;
        let alpha = decode_alpha(
            bytes[21],
            u16::from_le_bytes(bytes[22..24].try_into().unwrap()),
        )?;
        let count = usize::try_from(u32::from_le_bytes(bytes[24..28].try_into().unwrap()))
            .map_err(|_| ImageArtifactError::InvalidMipChain)?;
        if count == 0 || count > MAX_ARTIFACT_LEVELS {
            return Err(ImageArtifactError::InvalidMipChain);
        }
        let mut offset = MIP_HEADER_LEN;
        let mut levels = Vec::with_capacity(count);
        let mut total = 0usize;
        for _ in 0..count {
            let end_header = offset
                .checked_add(MIP_LEVEL_HEADER_LEN)
                .ok_or(ImageArtifactError::InvalidMipChain)?;
            if end_header > bytes.len() {
                return Err(ImageArtifactError::Truncated);
            }
            let level_width = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
            let level_height =
                u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap());
            let len = usize::try_from(u64::from_le_bytes(
                bytes[offset + 8..end_header].try_into().unwrap(),
            ))
            .map_err(|_| ImageArtifactError::BudgetExceeded)?;
            offset = end_header;
            let end = offset
                .checked_add(len)
                .ok_or(ImageArtifactError::BudgetExceeded)?;
            if end > bytes.len() {
                return Err(ImageArtifactError::Truncated);
            }
            total = total
                .checked_add(len)
                .ok_or(ImageArtifactError::BudgetExceeded)?;
            if total > MAX_ARTIFACT_BYTES {
                return Err(ImageArtifactError::BudgetExceeded);
            }
            levels.push(ImageArtifactMipLevel {
                width: level_width,
                height: level_height,
                rgba8: bytes[offset..end].to_vec(),
            });
            offset = end;
        }
        if offset != bytes.len() {
            return Err(ImageArtifactError::TrailingBytes);
        }
        let artifact = Self {
            width,
            height,
            semantic,
            alpha,
            levels,
        };
        artifact.to_chain()?;
        Ok(artifact)
    }

    /// Converts the decoded artifact into the runtime image retaining every level.
    pub fn to_image(&self) -> Result<Image, ImageError> {
        self.to_chain()
            .map_err(|_| ImageError::SizeMismatch {
                expected: 0,
                found: 0,
            })?
            .to_image()
    }

    fn to_chain(&self) -> Result<ImageMipChain, ImageArtifactError> {
        let levels = self
            .levels
            .iter()
            .cloned()
            .map(|level| {
                ImageMipLevel::new(level.width, level.height, level.rgba8)
                    .map_err(|_| ImageArtifactError::InvalidMipChain)
            })
            .collect::<Result<Vec<_>, _>>()?;
        ImageMipChain::from_levels(
            levels,
            self.semantic,
            self.alpha,
            MipGenerationSettings {
                semantic: self.semantic,
                alpha: self.alpha,
                max_levels: MAX_ARTIFACT_LEVELS as u32,
                max_total_bytes: MAX_ARTIFACT_BYTES,
            },
        )
        .map_err(|_| ImageArtifactError::InvalidMipChain)
    }
}

fn semantic_tag(value: TextureSemantic) -> u8 {
    match value {
        TextureSemantic::ColorSrgb => 0,
        TextureSemantic::ColorLinear => 1,
        TextureSemantic::NormalMap => 2,
        TextureSemantic::Mask => 3,
    }
}

fn decode_semantic(value: u8) -> Result<TextureSemantic, ImageArtifactError> {
    match value {
        0 => Ok(TextureSemantic::ColorSrgb),
        1 => Ok(TextureSemantic::ColorLinear),
        2 => Ok(TextureSemantic::NormalMap),
        3 => Ok(TextureSemantic::Mask),
        _ => Err(ImageArtifactError::InvalidMipChain),
    }
}

fn alpha_tag(value: AlphaCoveragePolicy) -> (u8, u16) {
    match value {
        AlphaCoveragePolicy::Preserve => (0, 0),
        AlphaCoveragePolicy::Coverage { threshold } => (1, u16::from(threshold)),
        AlphaCoveragePolicy::Mask { threshold } => (2, u16::from(threshold)),
    }
}

fn decode_alpha(tag: u8, threshold: u16) -> Result<AlphaCoveragePolicy, ImageArtifactError> {
    let threshold = u8::try_from(threshold).map_err(|_| ImageArtifactError::InvalidMipChain)?;
    match tag {
        0 if threshold == 0 => Ok(AlphaCoveragePolicy::Preserve),
        1 => Ok(AlphaCoveragePolicy::Coverage { threshold }),
        2 => Ok(AlphaCoveragePolicy::Mask { threshold }),
        _ => Err(ImageArtifactError::InvalidMipChain),
    }
}

fn pixel_len(width: u32, height: u32) -> Result<usize, ImageArtifactError> {
    usize::try_from(u64::from(width) * u64::from(height) * 4)
        .map_err(|_| ImageArtifactError::InvalidDimensions)
}
