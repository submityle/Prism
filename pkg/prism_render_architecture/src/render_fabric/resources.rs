//! Typed resource metadata. Allocation and aliasing remain graph-owned.
use super::*;
use crate::frame_graph::ResourceLifetime;
pub use prism_render_driver::{
    BufferId, BufferUsages, TextureDimension, TextureFormat, TextureId, TextureUsages,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResourceLayout {
    Buffer {
        bytes: u64,
        usage: BufferUsages,
    },
    Texture {
        extent: [u32; 3],
        dimension: TextureDimension,
        format: TextureFormat,
        mip_levels: u32,
        samples: u32,
        usage: TextureUsages,
    },
}

impl ResourceLayout {
    /// Logical uncompressed footprint; hardware placement requirements must be
    /// supplied by the backend, not guessed by this declaration adapter.
    pub fn size(self) -> Option<u64> {
        match self {
            Self::Buffer { bytes, .. } => (bytes > 0).then_some(bytes),
            Self::Texture {
                extent,
                dimension,
                format,
                mip_levels,
                samples,
                ..
            } => {
                if extent.contains(&0) || mip_levels == 0 || !matches!(samples, 1 | 2 | 4 | 8 | 16)
                {
                    return None;
                }
                if dimension == TextureDimension::D1 && (extent[1] != 1 || extent[2] != 1)
                    || samples > 1 && (dimension != TextureDimension::D2 || mip_levels != 1)
                {
                    return None;
                }
                let largest = if dimension == TextureDimension::D3 {
                    *extent.iter().max()?
                } else {
                    extent[0].max(extent[1])
                };
                if mip_levels > u32::BITS - largest.leading_zeros() {
                    return None;
                }
                let mut size = 0u64;
                for mip in 0..mip_levels {
                    let width = u64::from((extent[0] >> mip).max(1));
                    let height = u64::from((extent[1] >> mip).max(1));
                    let depth = u64::from(if dimension == TextureDimension::D3 {
                        (extent[2] >> mip).max(1)
                    } else {
                        extent[2]
                    });
                    let bytes = width
                        .checked_mul(height)?
                        .checked_mul(depth)?
                        .checked_mul(u64::from(format.bytes_per_texel()))?
                        .checked_mul(u64::from(samples))?;
                    size = size.checked_add(bytes)?;
                }
                Some(size)
            }
        }
    }

    pub fn supports(self, access: AccessKind) -> bool {
        match self {
            Self::Buffer { usage, .. } => match access {
                AccessKind::UniformRead => usage.contains(BufferUsages::UNIFORM),
                AccessKind::IndexRead => usage.contains(BufferUsages::INDEX),
                AccessKind::StorageRead | AccessKind::StorageWrite => {
                    usage.contains(BufferUsages::STORAGE)
                }
                AccessKind::IndirectRead => usage.contains(BufferUsages::INDIRECT),
                AccessKind::TransferRead => usage.contains(BufferUsages::COPY_SRC),
                AccessKind::TransferWrite => usage.contains(BufferUsages::COPY_DST),
                _ => false,
            },
            Self::Texture {
                usage,
                format,
                samples,
                ..
            } => match access {
                AccessKind::SampledRead => usage.contains(TextureUsages::TEXTURE_BINDING),
                AccessKind::StorageRead | AccessKind::StorageWrite => {
                    !format.is_depth_stencil()
                        && !format.is_srgb()
                        && samples == 1
                        && usage.contains(TextureUsages::STORAGE_BINDING)
                }
                AccessKind::ColorAttachment => {
                    !format.is_depth_stencil() && usage.contains(TextureUsages::RENDER_ATTACHMENT)
                }
                AccessKind::DepthAttachment | AccessKind::DepthAttachmentRead => {
                    format.has_depth() && usage.contains(TextureUsages::RENDER_ATTACHMENT)
                }
                AccessKind::TransferRead => usage.contains(TextureUsages::COPY_SRC),
                AccessKind::TransferWrite => usage.contains(TextureUsages::COPY_DST),
                AccessKind::UniformRead | AccessKind::IndexRead | AccessKind::IndirectRead => false,
            },
        }
    }
}

/// Semantic resource template: no frame-local ID or physical handle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResourceTemplate {
    pub key: ResourceKey,
    pub lifetime: ResourceLifetime,
    pub layout: ResourceLayout,
    pub alignment: u64,
}

/// Concrete externally owned storage, not a replacement resource allocator.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PhysicalResource {
    Buffer { id: BufferId, offset: u64 },
    Texture(TextureId),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrameResourceBinding {
    pub key: ResourceKey,
    pub frame: u64,
    /// Committed content version of imported/history storage before this graph.
    pub version: u32,
    pub physical: PhysicalResource,
    pub layout: ResourceLayout,
    pub alignment: u64,
}

impl FrameResourceBinding {
    pub(super) fn matches(self, template: ResourceTemplate) -> bool {
        if self.layout != template.layout || self.alignment != template.alignment {
            return false;
        }
        match (self.physical, self.layout) {
            (PhysicalResource::Buffer { offset, .. }, ResourceLayout::Buffer { bytes, .. }) => {
                offset.is_multiple_of(template.alignment) && offset.checked_add(bytes).is_some()
            }
            (PhysicalResource::Texture(_), ResourceLayout::Texture { .. }) => true,
            _ => false,
        }
    }

    pub(super) fn overlaps(self, other: Self) -> bool {
        match (self.physical, other.physical) {
            (PhysicalResource::Texture(a), PhysicalResource::Texture(b)) => a == b,
            (
                PhysicalResource::Buffer { id: a, offset: x },
                PhysicalResource::Buffer { id: b, offset: y },
            ) => {
                a == b
                    && x < y.saturating_add(other.layout.size().unwrap_or(u64::MAX))
                    && y < x.saturating_add(self.layout.size().unwrap_or(u64::MAX))
            }
            _ => false,
        }
    }
}
