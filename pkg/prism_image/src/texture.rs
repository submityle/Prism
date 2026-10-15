//! Deterministic semantic texture mip generation.

use alloc::vec;
use alloc::vec::Vec;

use crate::{Image, ImageError};
use prism_render_driver::{SamplerDescriptor, TextureFormat};

/// The interpretation used while reducing texels.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TextureSemantic {
    /// RGB is encoded sRGB and is filtered in linear light.
    ColorSrgb,
    /// RGB is already linear and is filtered directly.
    ColorLinear,
    /// RGB is a tangent-space normal and is renormalized after filtering.
    NormalMap,
    /// RGB is filtered linearly; alpha is treated as a binary mask.
    Mask,
}

/// The alpha rule applied at every generated level.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AlphaCoveragePolicy {
    /// Average alpha as a regular scalar channel.
    Preserve,
    /// Preserve the parent level's covered-pixel fraction at this level.
    Coverage { threshold: u8 },
    /// Keep alpha as a deterministic binary mask at the supplied threshold.
    Mask { threshold: u8 },
}

/// Bounded, explicit inputs to mip generation and artifact decoding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MipGenerationSettings {
    /// How RGB channels are interpreted.
    pub semantic: TextureSemantic,
    /// How alpha is reduced.
    pub alpha: AlphaCoveragePolicy,
    /// Maximum number of levels, including level zero.
    pub max_levels: u32,
    /// Maximum bytes in all generated RGBA8 levels.
    pub max_total_bytes: usize,
}

impl Default for MipGenerationSettings {
    fn default() -> Self {
        Self {
            semantic: TextureSemantic::ColorSrgb,
            alpha: AlphaCoveragePolicy::Preserve,
            max_levels: 32,
            max_total_bytes: 256 * 1024 * 1024,
        }
    }
}

/// Why a semantic mip chain could not be built or validated.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MipChainError {
    /// Width or height is zero.
    InvalidDimensions,
    /// The source byte length does not match width and height.
    InvalidSourceLength,
    /// The requested level budget is zero or too small.
    LevelBudgetExceeded,
    /// The aggregate byte budget was exceeded.
    ByteBudgetExceeded,
    /// A chain has a missing, oversized, or inconsistent level.
    InvalidChain,
}

impl core::fmt::Display for MipChainError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::InvalidDimensions => "mip chain dimensions are invalid",
            Self::InvalidSourceLength => "mip chain source length is invalid",
            Self::LevelBudgetExceeded => "mip chain level budget was exceeded",
            Self::ByteBudgetExceeded => "mip chain byte budget was exceeded",
            Self::InvalidChain => "mip chain levels are inconsistent",
        })
    }
}

impl std::error::Error for MipChainError {}

/// One tightly packed RGBA8 mip level.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImageMipLevel {
    width: u32,
    height: u32,
    rgba8: Vec<u8>,
}

impl ImageMipLevel {
    /// Creates a validated mip level.
    pub fn new(width: u32, height: u32, rgba8: Vec<u8>) -> Result<Self, MipChainError> {
        if width == 0 || height == 0 {
            return Err(MipChainError::InvalidDimensions);
        }
        let expected = pixel_bytes(width, height).ok_or(MipChainError::InvalidDimensions)?;
        if rgba8.len() != expected {
            return Err(MipChainError::InvalidSourceLength);
        }
        Ok(Self {
            width,
            height,
            rgba8,
        })
    }

    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }
    #[must_use]
    pub fn rgba8(&self) -> &[u8] {
        &self.rgba8
    }
}

/// A complete, strictly ordered level-zero-to-one-texel mip chain.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImageMipChain {
    levels: Vec<ImageMipLevel>,
    semantic: TextureSemantic,
    alpha: AlphaCoveragePolicy,
}

impl ImageMipChain {
    /// Generates all levels, using ceil-halving for odd dimensions.
    pub fn generate(
        width: u32,
        height: u32,
        rgba8: &[u8],
        settings: MipGenerationSettings,
    ) -> Result<Self, MipChainError> {
        if width == 0 || height == 0 {
            return Err(MipChainError::InvalidDimensions);
        }
        if settings.max_levels == 0 {
            return Err(MipChainError::LevelBudgetExceeded);
        }
        let expected = pixel_bytes(width, height).ok_or(MipChainError::InvalidDimensions)?;
        if rgba8.len() != expected {
            return Err(MipChainError::InvalidSourceLength);
        }
        let mut levels = Vec::new();
        let mut current = ImageMipLevel::new(width, height, rgba8.to_vec())?;
        let mut total = current.rgba8.len();
        if total > settings.max_total_bytes {
            return Err(MipChainError::ByteBudgetExceeded);
        }
        levels.push(current.clone());
        while current.width > 1 || current.height > 1 {
            if levels.len() as u32 >= settings.max_levels {
                return Err(MipChainError::LevelBudgetExceeded);
            }
            let next_width = current.width.div_ceil(2);
            let next_height = current.height.div_ceil(2);
            let next = reduce_level(&current, next_width, next_height, settings);
            total = total
                .checked_add(next.rgba8.len())
                .ok_or(MipChainError::ByteBudgetExceeded)?;
            if total > settings.max_total_bytes {
                return Err(MipChainError::ByteBudgetExceeded);
            }
            current = next;
            levels.push(current.clone());
        }
        Ok(Self {
            levels,
            semantic: settings.semantic,
            alpha: settings.alpha,
        })
    }

    /// Validates an already encoded chain without generating new data.
    pub fn from_levels(
        levels: Vec<ImageMipLevel>,
        semantic: TextureSemantic,
        alpha: AlphaCoveragePolicy,
        settings: MipGenerationSettings,
    ) -> Result<Self, MipChainError> {
        if levels.is_empty() || levels.len() as u32 > settings.max_levels {
            return Err(MipChainError::LevelBudgetExceeded);
        }
        let mut total = 0usize;
        for (index, level) in levels.iter().enumerate() {
            let expected_width = levels[0].width.div_ceil(1u32 << index.min(31));
            let expected_height = levels[0].height.div_ceil(1u32 << index.min(31));
            if level.width != expected_width || level.height != expected_height {
                return Err(MipChainError::InvalidChain);
            }
            total = total
                .checked_add(level.rgba8.len())
                .ok_or(MipChainError::ByteBudgetExceeded)?;
        }
        if levels.last().map(|level| (level.width, level.height)) != Some((1, 1)) {
            return Err(MipChainError::InvalidChain);
        }
        if total > settings.max_total_bytes {
            return Err(MipChainError::ByteBudgetExceeded);
        }
        Ok(Self {
            levels,
            semantic,
            alpha,
        })
    }

    #[must_use]
    pub fn levels(&self) -> &[ImageMipLevel] {
        &self.levels
    }
    #[must_use]
    pub fn level(&self, index: usize) -> Option<&ImageMipLevel> {
        self.levels.get(index)
    }
    #[must_use]
    pub const fn semantic(&self) -> TextureSemantic {
        self.semantic
    }
    #[must_use]
    pub const fn alpha_policy(&self) -> AlphaCoveragePolicy {
        self.alpha
    }
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        self.levels.iter().map(|level| level.rgba8.len()).sum()
    }

    /// Converts the chain into an image retaining all levels.
    pub fn to_image(&self) -> Result<Image, ImageError> {
        Image::from_mip_chain(
            self,
            TextureFormat::Rgba8Unorm,
            SamplerDescriptor::default(),
        )
    }
}

fn reduce_level(
    source: &ImageMipLevel,
    width: u32,
    height: u32,
    settings: MipGenerationSettings,
) -> ImageMipLevel {
    let mut output = vec![0u8; (width as usize) * (height as usize) * 4];
    for y in 0..height {
        for x in 0..width {
            let x0 = x * 2;
            let y0 = y * 2;
            let x1 = (x0 + 1).min(source.width - 1);
            let y1 = (y0 + 1).min(source.height - 1);
            let mut samples = [(0u8, 0u8, 0u8, 0u8); 4];
            let mut count = 0usize;
            for sy in [y0, y1] {
                for sx in [x0, x1] {
                    if sx < source.width
                        && sy < source.height
                        && (count == 0 || sx != x0 || sy != y0)
                    {
                        let offset = ((sy * source.width + sx) as usize) * 4;
                        samples[count] = (
                            source.rgba8[offset],
                            source.rgba8[offset + 1],
                            source.rgba8[offset + 2],
                            source.rgba8[offset + 3],
                        );
                        count += 1;
                    }
                }
            }
            let out = ((y * width + x) as usize) * 4;
            let mut rgb = [0.0f32; 3];
            let mut alpha = 0.0f32;
            for &(r, g, b, a) in samples.iter().take(count) {
                let values = [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0];
                let values = match settings.semantic {
                    TextureSemantic::ColorSrgb => values.map(srgb_to_linear),
                    TextureSemantic::NormalMap => [
                        values[0] * 2.0 - 1.0,
                        values[1] * 2.0 - 1.0,
                        values[2] * 2.0 - 1.0,
                    ],
                    _ => values,
                };
                rgb[0] += values[0];
                rgb[1] += values[1];
                rgb[2] += values[2];
                alpha += a as f32 / 255.0;
            }
            let divisor = count as f32;
            rgb[0] /= divisor;
            rgb[1] /= divisor;
            rgb[2] /= divisor;
            alpha /= divisor;
            if settings.semantic == TextureSemantic::NormalMap {
                let length = (rgb[0] * rgb[0] + rgb[1] * rgb[1] + rgb[2] * rgb[2]).sqrt();
                if length > f32::EPSILON {
                    rgb[0] /= length;
                    rgb[1] /= length;
                    rgb[2] /= length;
                }
                rgb = rgb.map(|value| value * 0.5 + 0.5);
            } else if settings.semantic == TextureSemantic::ColorSrgb {
                rgb = rgb.map(linear_to_srgb);
            }
            output[out] = to_byte(rgb[0]);
            output[out + 1] = to_byte(rgb[1]);
            output[out + 2] = to_byte(rgb[2]);
            output[out + 3] = to_byte(alpha);
        }
    }
    let mut level = ImageMipLevel {
        width,
        height,
        rgba8: output,
    };
    apply_alpha_policy(&mut level, source, settings.alpha);
    level
}

fn apply_alpha_policy(
    level: &mut ImageMipLevel,
    parent: &ImageMipLevel,
    policy: AlphaCoveragePolicy,
) {
    match policy {
        AlphaCoveragePolicy::Preserve => {}
        AlphaCoveragePolicy::Mask { threshold } => {
            for pixel in level.rgba8.chunks_exact_mut(4) {
                pixel[3] = if pixel[3] >= threshold { 255 } else { 0 };
            }
        }
        AlphaCoveragePolicy::Coverage { threshold } => {
            let parent_count = parent
                .rgba8
                .chunks_exact(4)
                .filter(|p| p[3] >= threshold)
                .count();
            let target = parent_count as f32 / (parent.rgba8.len() / 4) as f32;
            let mut values: Vec<u8> = level.rgba8.chunks_exact(4).map(|p| p[3]).collect();
            values.sort_unstable_by(|a, b| b.cmp(a));
            let keep = (target * values.len() as f32).round() as usize;
            let cutoff = keep
                .checked_sub(1)
                .and_then(|index| values.get(index))
                .copied()
                .unwrap_or(255);
            for pixel in level.rgba8.chunks_exact_mut(4) {
                pixel[3] = if pixel[3] >= cutoff {
                    pixel[3].max(threshold)
                } else {
                    0
                };
            }
        }
    }
}

fn pixel_bytes(width: u32, height: u32) -> Option<usize> {
    usize::try_from(
        u64::from(width)
            .checked_mul(u64::from(height))?
            .checked_mul(4)?,
    )
    .ok()
}

fn srgb_to_linear(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}
fn linear_to_srgb(value: f32) -> f32 {
    let value = value.clamp(0.0, 1.0);
    if value <= 0.0031308 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}
fn to_byte(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0 + 0.5).floor() as u8
}
