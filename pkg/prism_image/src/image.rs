//! The image asset.

use alloc::vec::Vec;

use prism_asset::Asset;
use prism_render_driver::{Extent3d, SamplerDescriptor, TextureFormat};

use crate::error::ImageError;

/// One image: texel data and the description it is uploaded with.
///
/// The data is the whole level-0 image, tightly packed at the format's texel
/// size — the layout a renderer uploads with one row of `bytes_per_row` per
/// texel row. Mip levels, layers, and block-compressed payloads arrive with the
/// streaming slice; an image today is one 2D level whose size the extents and
/// format fully determine (which is exactly why it can be validated here).
#[derive(Clone, Debug, PartialEq)]
pub struct Image {
    extent: Extent3d,
    format: TextureFormat,
    data: Vec<u8>,
    mip_levels: Vec<Vec<u8>>,
    sampler: SamplerDescriptor,
}

impl Image {
    /// An image of `extent` in `format` over `data`, with the default sampler.
    ///
    /// Fails when the description and the data disagree (see [`ImageError`]) —
    /// the renderer never sees an image it cannot upload faithfully.
    pub fn new(extent: Extent3d, format: TextureFormat, data: Vec<u8>) -> Result<Self, ImageError> {
        Self::with_sampler(extent, format, data, SamplerDescriptor::default())
    }

    /// An image with an explicit sampler.
    ///
    /// The sampler lives with the image because "how this content is filtered and
    /// addressed" is authored alongside the content (a UI atlas clamps, a terrain
    /// layer repeats with mipmapping); a material may still sample a texture
    /// through its own sampler, but the image's default has to be right.
    pub fn with_sampler(
        extent: Extent3d,
        format: TextureFormat,
        data: Vec<u8>,
        sampler: SamplerDescriptor,
    ) -> Result<Self, ImageError> {
        if extent.width == 0 || extent.height == 0 || extent.depth_or_array_layers == 0 {
            return Err(ImageError::EmptyExtent);
        }
        if format.has_depth() || format.has_stencil() {
            return Err(ImageError::NotColor {
                depth_stencil: format.has_stencil(),
            });
        }
        let expected = bytes_for(extent, format);
        if data.len() != expected {
            return Err(ImageError::SizeMismatch {
                expected,
                found: data.len(),
            });
        }
        Ok(Self {
            extent,
            format,
            data: data.clone(),
            mip_levels: vec![data],
            sampler,
        })
    }

    /// Creates an RGBA8 image from a validated semantic mip chain.
    pub fn from_mip_chain(
        chain: &crate::ImageMipChain,
        format: TextureFormat,
        sampler: SamplerDescriptor,
    ) -> Result<Self, ImageError> {
        if format != TextureFormat::Rgba8Unorm {
            return Err(ImageError::SizeMismatch {
                expected: 0,
                found: format.bytes_per_texel() as usize,
            });
        }
        let Some(base) = chain.level(0) else {
            return Err(ImageError::EmptyExtent);
        };
        let levels = chain
            .levels()
            .iter()
            .map(|level| level.rgba8().to_vec())
            .collect::<Vec<_>>();
        let image = Self {
            extent: Extent3d::new_2d(base.width(), base.height()),
            format,
            data: base.rgba8().to_vec(),
            mip_levels: levels,
            sampler,
        };
        image.validate_mip_levels()?;
        Ok(image)
    }

    /// The image's texel extents.
    #[must_use]
    pub const fn extent(&self) -> Extent3d {
        self.extent
    }

    /// The image's texel format.
    #[must_use]
    pub const fn format(&self) -> TextureFormat {
        self.format
    }

    /// The tightly packed level-0 texel bytes.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Returns the tightly packed bytes for `level`, including level zero.
    #[must_use]
    pub fn mip_data(&self, level: u32) -> Option<&[u8]> {
        self.mip_levels
            .get(usize::try_from(level).ok()?)
            .map(Vec::as_slice)
    }

    /// The number of stored mip levels.
    #[must_use]
    pub fn mip_level_count(&self) -> u32 {
        self.mip_levels.len() as u32
    }

    /// The texel extent of a stored mip level.
    #[must_use]
    pub fn mip_extent(&self, level: u32) -> Option<Extent3d> {
        if level >= self.mip_level_count() {
            return None;
        }
        Some(
            self.extent
                .mip_level_size(level, prism_render_driver::TextureDimension::D2),
        )
    }

    /// The sampler the image wants to be sampled through.
    #[must_use]
    pub const fn sampler(&self) -> &SamplerDescriptor {
        &self.sampler
    }

    /// The byte length of one texel row of level 0.
    ///
    /// This is the row pitch an upload must use, and the row pitch a readback
    /// comes back in — the two have to agree for a round trip to be byte-exact.
    #[must_use]
    pub const fn bytes_per_row(&self) -> u32 {
        self.extent.width * self.format.bytes_per_texel()
    }

    fn validate_mip_levels(&self) -> Result<(), ImageError> {
        if self.mip_levels.is_empty() || self.mip_levels[0] != self.data {
            return Err(ImageError::SizeMismatch {
                expected: self.data.len(),
                found: self.mip_levels.first().map_or(0, Vec::len),
            });
        }
        for (index, level) in self.mip_levels.iter().enumerate() {
            let mip = self
                .extent
                .mip_level_size(index as u32, prism_render_driver::TextureDimension::D2);
            let expected = bytes_for(mip, self.format);
            if level.len() != expected {
                return Err(ImageError::SizeMismatch {
                    expected,
                    found: level.len(),
                });
            }
        }
        Ok(())
    }
}

/// The level-0 byte length `extent` and `format` require.
#[must_use]
pub fn bytes_for(extent: Extent3d, format: TextureFormat) -> usize {
    let texels = u64::from(extent.width)
        .saturating_mul(u64::from(extent.height))
        .saturating_mul(u64::from(extent.depth_or_array_layers));
    let bytes = texels.saturating_mul(u64::from(format.bytes_per_texel()));
    usize::try_from(bytes).unwrap_or(usize::MAX)
}

impl Asset for Image {
    const TYPE_NAME: &'static str = "prism_image::Image";
}
