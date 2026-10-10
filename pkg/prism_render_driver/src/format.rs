//! Texture and vertex formats with their layout math.

/// How a texture format's samples are interpreted by a shader when sampled.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum TextureSampleType {
    /// Normalized or sRGB data sampled as floating point and filterable.
    Float,
    /// Floating-point data that cannot be linearly filtered (for example
    /// 32-bit float color on some tiers).
    UnfilterableFloat,
    /// Depth data sampled as floating point.
    Depth,
    /// Signed integer data, sampled without filtering.
    Sint,
    /// Unsigned integer data, sampled without filtering.
    Uint,
}

/// Which logical planes a texture format exposes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum FormatAspects {
    /// A single color plane.
    Color,
    /// A depth plane only.
    Depth,
    /// A stencil plane only.
    Stencil,
    /// Combined depth and stencil planes.
    DepthStencil,
}

/// A GPU texture format.
///
/// This is a curated set covering the color, HDR, depth, and stencil formats
/// Prism's renderer uses; it is intentionally smaller than the full hardware
/// matrix and grows as backends require. Each variant knows its texel byte
/// size, sample type, sRGB-ness, and aspects via the query methods below.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[non_exhaustive]
pub enum TextureFormat {
    /// 8-bit single-channel unsigned normalized.
    R8Unorm,
    /// 8-bit single-channel unsigned integer.
    R8Uint,
    /// 16-bit single-channel float.
    R16Float,
    /// 32-bit single-channel float.
    R32Float,
    /// 32-bit single-channel unsigned integer.
    R32Uint,
    /// Two 8-bit unsigned normalized channels.
    Rg8Unorm,
    /// Two 16-bit float channels.
    Rg16Float,
    /// Two 32-bit float channels.
    Rg32Float,
    /// Four 8-bit unsigned normalized channels (linear).
    Rgba8Unorm,
    /// Four 8-bit unsigned normalized channels (sRGB-encoded).
    Rgba8UnormSrgb,
    /// Four 8-bit unsigned normalized channels in BGRA order (linear).
    Bgra8Unorm,
    /// Four 8-bit unsigned normalized channels in BGRA order (sRGB-encoded).
    Bgra8UnormSrgb,
    /// Packed 10/10/10/2 unsigned normalized, common for HDR-ish color.
    Rgb10a2Unorm,
    /// Packed 11/11/10 float, common for HDR color without alpha.
    Rg11b10Float,
    /// Four 16-bit float channels.
    Rgba16Float,
    /// Four 32-bit float channels.
    Rgba32Float,
    /// 32-bit depth float.
    Depth32Float,
    /// 24-bit depth plus 8-bit stencil.
    Depth24PlusStencil8,
}

impl TextureFormat {
    /// The byte size of a single texel (uncompressed formats only).
    #[must_use]
    pub const fn bytes_per_texel(self) -> u32 {
        match self {
            Self::R8Unorm | Self::R8Uint => 1,
            Self::R16Float | Self::Rg8Unorm => 2,
            Self::R32Float
            | Self::R32Uint
            | Self::Rg16Float
            | Self::Rgba8Unorm
            | Self::Rgba8UnormSrgb
            | Self::Bgra8Unorm
            | Self::Bgra8UnormSrgb
            | Self::Rgb10a2Unorm
            | Self::Rg11b10Float
            | Self::Depth32Float
            | Self::Depth24PlusStencil8 => 4,
            Self::Rg32Float | Self::Rgba16Float => 8,
            Self::Rgba32Float => 16,
        }
    }

    /// The shader sample type for this format.
    #[must_use]
    pub const fn sample_type(self) -> TextureSampleType {
        match self {
            Self::R8Uint | Self::R32Uint => TextureSampleType::Uint,
            Self::R32Float | Self::Rg32Float | Self::Rgba32Float => {
                TextureSampleType::UnfilterableFloat
            }
            Self::Depth32Float | Self::Depth24PlusStencil8 => TextureSampleType::Depth,
            _ => TextureSampleType::Float,
        }
    }

    /// Which planes this format exposes.
    #[must_use]
    pub const fn aspects(self) -> FormatAspects {
        match self {
            Self::Depth32Float => FormatAspects::Depth,
            Self::Depth24PlusStencil8 => FormatAspects::DepthStencil,
            _ => FormatAspects::Color,
        }
    }

    /// Whether the format carries an sRGB transfer function.
    #[must_use]
    pub const fn is_srgb(self) -> bool {
        matches!(self, Self::Rgba8UnormSrgb | Self::Bgra8UnormSrgb)
    }

    /// Whether the format has a depth plane.
    #[must_use]
    pub const fn has_depth(self) -> bool {
        matches!(
            self.aspects(),
            FormatAspects::Depth | FormatAspects::DepthStencil
        )
    }

    /// Whether the format has a stencil plane.
    #[must_use]
    pub const fn has_stencil(self) -> bool {
        matches!(
            self.aspects(),
            FormatAspects::Stencil | FormatAspects::DepthStencil
        )
    }

    /// Whether the format is usable as a depth/stencil attachment.
    #[must_use]
    pub const fn is_depth_stencil(self) -> bool {
        self.has_depth() || self.has_stencil()
    }

    /// Whether the format can be linearly filtered when sampled.
    #[must_use]
    pub const fn is_filterable(self) -> bool {
        matches!(self.sample_type(), TextureSampleType::Float)
    }
}

/// The scalar/vector layout of a single vertex attribute as fed to the vertex
/// stage.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[non_exhaustive]
pub enum VertexFormat {
    /// One 32-bit float.
    Float32,
    /// Two 32-bit floats.
    Float32x2,
    /// Three 32-bit floats.
    Float32x3,
    /// Four 32-bit floats.
    Float32x4,
    /// Two 16-bit floats.
    Float16x2,
    /// Four 16-bit floats.
    Float16x4,
    /// One 32-bit unsigned integer.
    Uint32,
    /// Two 32-bit unsigned integers.
    Uint32x2,
    /// Four 32-bit unsigned integers.
    Uint32x4,
    /// One 32-bit signed integer.
    Sint32,
    /// Four 8-bit unsigned normalized values.
    Unorm8x4,
    /// Four 8-bit signed normalized values.
    Snorm8x4,
}

impl VertexFormat {
    /// The byte size this attribute occupies in a vertex buffer.
    #[must_use]
    pub const fn size(self) -> u64 {
        match self {
            Self::Unorm8x4 | Self::Snorm8x4 | Self::Float32 | Self::Uint32 | Self::Sint32 => 4,
            Self::Float16x2 | Self::Float32x2 | Self::Uint32x2 => 8,
            Self::Float32x3 => 12,
            Self::Float16x4 | Self::Float32x4 | Self::Uint32x4 => 16,
        }
    }

    /// The number of scalar components.
    #[must_use]
    pub const fn components(self) -> u32 {
        match self {
            Self::Float32 | Self::Uint32 | Self::Sint32 => 1,
            Self::Float32x2 | Self::Float16x2 | Self::Uint32x2 => 2,
            Self::Float32x3 => 3,
            Self::Float32x4
            | Self::Float16x4
            | Self::Uint32x4
            | Self::Unorm8x4
            | Self::Snorm8x4 => 4,
        }
    }
}
