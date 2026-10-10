//! Texture extents, dimensions, and creation/view descriptors.

use crate::flags::TextureUsages;
use crate::format::TextureFormat;
use alloc::string::String;

/// The size of a texture in texels, including array layers / depth.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Extent3d {
    /// Width in texels.
    pub width: u32,
    /// Height in texels (1 for 1D textures).
    pub height: u32,
    /// Depth for 3D textures, or array layer count otherwise.
    pub depth_or_array_layers: u32,
}

impl Extent3d {
    /// A 2D extent with a single layer.
    #[must_use]
    pub const fn new_2d(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            depth_or_array_layers: 1,
        }
    }

    /// The maximum number of mip levels a texture of this size supports, i.e.
    /// `floor(log2(max_dim)) + 1`. Depth/array layers do not reduce.
    #[must_use]
    pub const fn max_mip_levels(self, dimension: TextureDimension) -> u32 {
        let mut max = self.width;
        if self.height > max {
            max = self.height;
        }
        if matches!(dimension, TextureDimension::D3) && self.depth_or_array_layers > max {
            max = self.depth_or_array_layers;
        }
        if max == 0 {
            return 1;
        }
        32 - (max.leading_zeros())
    }

    /// The extent of mip level `level`, clamped so each axis is at least 1.
    #[must_use]
    pub const fn mip_level_size(self, level: u32, dimension: TextureDimension) -> Self {
        let shift = level;
        let width = max_u32(shr_or_zero(self.width, shift), 1);
        let height = max_u32(shr_or_zero(self.height, shift), 1);
        let depth = match dimension {
            TextureDimension::D3 => max_u32(shr_or_zero(self.depth_or_array_layers, shift), 1),
            _ => self.depth_or_array_layers,
        };
        Self {
            width,
            height,
            depth_or_array_layers: depth,
        }
    }
}

/// `const`-friendly `max` for `u32` (`core::cmp::max` is not `const`).
const fn max_u32(a: u32, b: u32) -> u32 {
    if a > b {
        a
    } else {
        b
    }
}

/// `const`-friendly right shift that saturates to `0` once `shift` reaches the
/// bit width, avoiding the shift-overflow panic for mip levels beyond the
/// smallest extent.
const fn shr_or_zero(value: u32, shift: u32) -> u32 {
    if shift >= u32::BITS {
        0
    } else {
        value >> shift
    }
}

/// The dimensionality of a texture's storage.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum TextureDimension {
    /// A 1D texture (a line of texels).
    D1,
    /// A 2D texture (the common case).
    #[default]
    D2,
    /// A 3D volume texture.
    D3,
}

/// How a texture is interpreted when bound as a view.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum TextureViewDimension {
    /// A single 1D image.
    D1,
    /// A single 2D image.
    #[default]
    D2,
    /// An array of 2D images.
    D2Array,
    /// A cube of six 2D faces.
    Cube,
    /// An array of cubes.
    CubeArray,
    /// A 3D volume.
    D3,
}

/// Which plane(s) of a texture a view or copy targets.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum TextureAspect {
    /// All planes the format exposes.
    #[default]
    All,
    /// The depth plane only.
    DepthOnly,
    /// The stencil plane only.
    StencilOnly,
}

/// A request to create a texture.
#[derive(Clone, PartialEq, Debug)]
pub struct TextureDescriptor {
    /// A debug label surfaced in GPU tooling.
    pub label: Option<String>,
    /// The texel dimensions and layer/depth count.
    pub size: Extent3d,
    /// The number of mip levels.
    pub mip_level_count: u32,
    /// The MSAA sample count (1 means no multisampling).
    pub sample_count: u32,
    /// The storage dimensionality.
    pub dimension: TextureDimension,
    /// The texel format.
    pub format: TextureFormat,
    /// The permitted GPU usages.
    pub usage: TextureUsages,
}

impl TextureDescriptor {
    /// Creates a single-mip, non-multisampled 2D texture descriptor.
    #[must_use]
    pub fn new_2d(size: Extent3d, format: TextureFormat, usage: TextureUsages) -> Self {
        Self {
            label: None,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage,
        }
    }

    /// Whether this descriptor requests multisampling.
    #[must_use]
    pub const fn is_multisampled(&self) -> bool {
        self.sample_count > 1
    }
}

/// A request to create a view over an existing texture.
#[derive(Clone, PartialEq, Debug)]
pub struct TextureViewDescriptor {
    /// A debug label surfaced in GPU tooling.
    pub label: Option<String>,
    /// The format the view reinterprets the texture as (defaults to the
    /// texture's own format when `None`).
    pub format: Option<TextureFormat>,
    /// How the view is dimensioned.
    pub dimension: TextureViewDimension,
    /// Which plane(s) the view exposes.
    pub aspect: TextureAspect,
    /// The first visible mip level.
    pub base_mip_level: u32,
    /// The number of visible mip levels, or `None` for all remaining.
    pub mip_level_count: Option<u32>,
    /// The first visible array layer.
    pub base_array_layer: u32,
    /// The number of visible array layers, or `None` for all remaining.
    pub array_layer_count: Option<u32>,
}

impl Default for TextureViewDescriptor {
    fn default() -> Self {
        Self {
            label: None,
            format: None,
            dimension: TextureViewDimension::D2,
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        }
    }
}
