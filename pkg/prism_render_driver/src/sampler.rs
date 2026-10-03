//! Sampler state descriptors.

use crate::compare::CompareFunction;
use alloc::string::String;

/// How texture coordinates outside `[0, 1]` are resolved.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum AddressMode {
    /// Coordinates are clamped to the edge texel.
    #[default]
    ClampToEdge,
    /// Coordinates wrap around (tiling).
    Repeat,
    /// Coordinates mirror on each repeat.
    MirrorRepeat,
    /// Coordinates outside the range read the configured border color.
    ClampToBorder,
}

/// How texels are selected or blended when sampling.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum FilterMode {
    /// Pick the nearest texel.
    #[default]
    Nearest,
    /// Linearly blend neighbouring texels.
    Linear,
}

/// The border color used with [`AddressMode::ClampToBorder`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum SamplerBorderColor {
    /// Transparent black (`0, 0, 0, 0`).
    #[default]
    TransparentBlack,
    /// Opaque black (`0, 0, 0, 1`).
    OpaqueBlack,
    /// Opaque white (`1, 1, 1, 1`).
    OpaqueWhite,
}

/// A request to create a texture sampler.
#[derive(Clone, PartialEq, Debug)]
pub struct SamplerDescriptor {
    /// A debug label surfaced in GPU tooling.
    pub label: Option<String>,
    /// Addressing along U (x).
    pub address_mode_u: AddressMode,
    /// Addressing along V (y).
    pub address_mode_v: AddressMode,
    /// Addressing along W (z).
    pub address_mode_w: AddressMode,
    /// Filtering between magnified texels.
    pub mag_filter: FilterMode,
    /// Filtering between minified texels.
    pub min_filter: FilterMode,
    /// Filtering between mip levels.
    pub mipmap_filter: FilterMode,
    /// The lowest mip LOD the sampler clamps to.
    pub lod_min_clamp: f32,
    /// The highest mip LOD the sampler clamps to.
    pub lod_max_clamp: f32,
    /// If set, makes this a comparison sampler (for shadow mapping).
    pub compare: Option<CompareFunction>,
    /// The anisotropy ratio (1 disables anisotropic filtering).
    pub anisotropy_clamp: u16,
    /// The border color used with [`AddressMode::ClampToBorder`].
    pub border_color: Option<SamplerBorderColor>,
}

impl Default for SamplerDescriptor {
    fn default() -> Self {
        Self {
            label: None,
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mipmap_filter: FilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: 32.0,
            compare: None,
            anisotropy_clamp: 1,
            border_color: None,
        }
    }
}

impl SamplerDescriptor {
    /// A trilinear, repeating sampler — a sensible default for color textures.
    #[must_use]
    pub fn linear_repeat() -> Self {
        Self {
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            address_mode_w: AddressMode::Repeat,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            ..Self::default()
        }
    }

    /// Whether this sampler performs depth comparison (shadow sampling).
    #[must_use]
    pub const fn is_comparison(&self) -> bool {
        self.compare.is_some()
    }
}
