//! Virtual-resource descriptions and screen-relative sizing.
//!
//! A pass declares *what* a resource looks like, not where it lives. These
//! descriptions are backend-agnostic requests that the compiler later lowers to
//! concrete [`prism_render_driver`] descriptors once the frame's swapchain size
//! is known. Keeping sizing symbolic (e.g. "half the swapchain") is what lets a
//! graph be authored once and run at any resolution, and lets the compiler reason
//! about which transient resources can share memory.

use prism_render_driver::{
    BufferDescriptor, BufferUsages, Extent3d, TextureDescriptor, TextureDimension, TextureFormat,
    TextureUsages,
};

/// How a texture's pixel extent is derived.
///
/// Absolute sizes are used verbatim; swapchain-relative sizes are resolved
/// against the frame's render target so one graph scales to any resolution. The
/// divisor supports classic half/quarter-res passes (bloom, SSAO) without
/// hard-coding pixels.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum SizeClass {
    /// A fixed pixel extent independent of the swapchain.
    Absolute {
        /// Width in texels.
        width: u32,
        /// Height in texels.
        height: u32,
        /// Depth (3D) or array-layer count.
        depth_or_array_layers: u32,
    },
    /// A fraction of the swapchain extent: `ceil(swapchain / divisor)`.
    ///
    /// A divisor of `1` tracks the swapchain exactly; `2` is half-res, and so
    /// on. The divisor is clamped to at least `1` at resolution time.
    SwapchainRelative {
        /// The denominator applied to both swapchain axes.
        divisor: u32,
        /// Depth (3D) or array-layer count (not scaled by the divisor).
        depth_or_array_layers: u32,
    },
}

impl SizeClass {
    /// An absolute 2D extent with a single layer.
    #[must_use]
    pub const fn absolute_2d(width: u32, height: u32) -> Self {
        Self::Absolute {
            width,
            height,
            depth_or_array_layers: 1,
        }
    }

    /// Tracks the full swapchain extent.
    #[must_use]
    pub const fn full_screen() -> Self {
        Self::SwapchainRelative {
            divisor: 1,
            depth_or_array_layers: 1,
        }
    }

    /// A fraction of the swapchain extent.
    #[must_use]
    pub const fn screen_fraction(divisor: u32) -> Self {
        Self::SwapchainRelative {
            divisor,
            depth_or_array_layers: 1,
        }
    }

    /// Resolves this size class against a concrete `swapchain` extent.
    #[must_use]
    pub const fn resolve(self, swapchain: Extent3d) -> Extent3d {
        match self {
            Self::Absolute {
                width,
                height,
                depth_or_array_layers,
            } => Extent3d {
                width: max1(width),
                height: max1(height),
                depth_or_array_layers: max1(depth_or_array_layers),
            },
            Self::SwapchainRelative {
                divisor,
                depth_or_array_layers,
            } => {
                let d = max1(divisor);
                Extent3d {
                    width: max1(swapchain.width.div_ceil(d)),
                    height: max1(swapchain.height.div_ceil(d)),
                    depth_or_array_layers: max1(depth_or_array_layers),
                }
            }
        }
    }
}

/// `const` floor of `1` for extents (zero-sized textures are illegal).
const fn max1(v: u32) -> u32 {
    if v == 0 {
        1
    } else {
        v
    }
}

/// A virtual texture description.
///
/// Mirrors [`TextureDescriptor`] but with symbolic sizing and *inferred* usage:
/// the compiler widens [`Self::usage`] with the usages implied by how passes
/// actually access the resource, so authors rarely set usage by hand.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct TextureDesc {
    /// How the extent is derived.
    pub size: SizeClass,
    /// The number of mip levels.
    pub mip_level_count: u32,
    /// The MSAA sample count (`1` means no multisampling).
    pub sample_count: u32,
    /// The storage dimensionality.
    pub dimension: TextureDimension,
    /// The texel format.
    pub format: TextureFormat,
    /// Usage bits explicitly requested; the compiler unions in inferred bits.
    pub usage: TextureUsages,
}

impl TextureDesc {
    /// A full-screen 2D color target in `format`.
    #[must_use]
    pub fn color(format: TextureFormat) -> Self {
        Self {
            size: SizeClass::full_screen(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::RENDER_ATTACHMENT,
        }
    }

    /// A full-screen 2D depth target in `format`.
    #[must_use]
    pub fn depth(format: TextureFormat) -> Self {
        Self {
            size: SizeClass::full_screen(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::RENDER_ATTACHMENT,
        }
    }

    /// Overrides the size class, returning `self` for chaining.
    #[must_use]
    pub fn with_size(mut self, size: SizeClass) -> Self {
        self.size = size;
        self
    }

    /// Sets the sample count, returning `self` for chaining.
    #[must_use]
    pub fn with_samples(mut self, sample_count: u32) -> Self {
        self.sample_count = sample_count;
        self
    }

    /// Adds usage bits, returning `self` for chaining.
    #[must_use]
    pub fn with_usage(mut self, usage: TextureUsages) -> Self {
        self.usage = self.usage.union(usage);
        self
    }

    /// Lowers to a concrete driver [`TextureDescriptor`] at `swapchain`
    /// resolution, folding in any `extra` usage inferred by the compiler.
    #[must_use]
    pub fn lower(&self, swapchain: Extent3d, extra: TextureUsages) -> TextureDescriptor {
        TextureDescriptor {
            label: None,
            size: self.size.resolve(swapchain),
            mip_level_count: self.mip_level_count.max(1),
            sample_count: self.sample_count.max(1),
            dimension: self.dimension,
            format: self.format,
            usage: self.usage.union(extra),
        }
    }
}

/// A virtual buffer description.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct BufferDesc {
    /// The size in bytes.
    pub size: u64,
    /// Usage bits explicitly requested; the compiler unions in inferred bits.
    pub usage: BufferUsages,
}

impl BufferDesc {
    /// A buffer of `size` bytes with the given `usage`.
    #[must_use]
    pub const fn new(size: u64, usage: BufferUsages) -> Self {
        Self { size, usage }
    }

    /// Lowers to a concrete driver [`BufferDescriptor`], folding in `extra`
    /// usage inferred by the compiler.
    #[must_use]
    pub fn lower(&self, extra: BufferUsages) -> BufferDescriptor {
        BufferDescriptor {
            label: None,
            size: self.size.max(1),
            usage: self.usage.union(extra),
            mapped_at_creation: false,
        }
    }
}
