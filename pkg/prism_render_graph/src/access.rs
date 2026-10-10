//! The access vocabulary passes use to declare how they touch a resource.
//!
//! A render graph earns its keep by turning a handful of high-level intents
//! ("I sample this texture in the fragment stage", "I write this as a color
//! attachment") into the exact [`prism_render_driver`] state triples the
//! barrier solver needs. This module is that translation layer: each access
//! records *what* a pass does to a resource, and knows how to lower itself to a
//! driver [`BufferState`] / [`TextureState`] and to the GPU usage bits the
//! resource must therefore be created with.

use prism_render_driver::{
    Accesses, BufferState, BufferUsages, PipelineStages, SubresourceRange, TextureLayout,
    TextureState, TextureUsages,
};

/// How a pass accesses a texture subresource range.
///
/// The variant fixes the layout and access class; the pass kind supplies the
/// pipeline stage (fragment for raster, compute for compute, transfer for
/// copies), so authors do not hand-write stage masks.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum TextureUse {
    /// Sampled or read-only shader access (`ShaderReadOnly` layout).
    Sampled,
    /// Read-only storage-image binding.
    StorageRead,
    /// Read/write storage-image binding.
    StorageWrite,
    /// Written as a color render attachment.
    ColorAttachment,
    /// Read as a color attachment (blend/load without clobbering).
    ColorAttachmentRead,
    /// Read/write depth-stencil attachment.
    DepthAttachment,
    /// Read-only depth-stencil attachment (depth test with no writes; allows
    /// concurrent sampling).
    DepthAttachmentRead,
    /// Same-pixel input-attachment read (tile-local, drives subpass merging).
    InputAttachment,
    /// Transfer source (copy/blit read).
    CopySrc,
    /// Transfer destination (copy/clear/blit write).
    CopyDst,
    /// Presented to a surface.
    Present,
}

impl TextureUse {
    /// Whether this use writes the texture (participates in WAW/RAW hazards and
    /// bumps the SSA version).
    #[must_use]
    pub const fn is_write(self) -> bool {
        matches!(
            self,
            Self::StorageWrite | Self::ColorAttachment | Self::DepthAttachment | Self::CopyDst
        )
    }

    /// Whether this use is an attachment (color/depth/input), which must be
    /// declared explicitly because it shapes the render pass.
    #[must_use]
    pub const fn is_attachment(self) -> bool {
        matches!(
            self,
            Self::ColorAttachment
                | Self::ColorAttachmentRead
                | Self::DepthAttachment
                | Self::DepthAttachmentRead
                | Self::InputAttachment
        )
    }

    /// The GPU usage bit a texture must carry to support this use.
    #[must_use]
    pub const fn required_usage(self) -> TextureUsages {
        match self {
            Self::Sampled | Self::InputAttachment => TextureUsages::TEXTURE_BINDING,
            Self::StorageRead | Self::StorageWrite => TextureUsages::STORAGE_BINDING,
            Self::ColorAttachment
            | Self::ColorAttachmentRead
            | Self::DepthAttachment
            | Self::DepthAttachmentRead => TextureUsages::RENDER_ATTACHMENT,
            Self::CopySrc => TextureUsages::COPY_SRC,
            Self::CopyDst => TextureUsages::COPY_DST,
            // Present needs no extra creation usage beyond RENDER_ATTACHMENT,
            // which the swapchain image already carries.
            Self::Present => TextureUsages::NONE,
        }
    }

    /// The target layout for this use.
    #[must_use]
    pub const fn layout(self) -> TextureLayout {
        match self {
            Self::Sampled | Self::InputAttachment => TextureLayout::ShaderReadOnly,
            Self::StorageRead | Self::StorageWrite => TextureLayout::General,
            Self::ColorAttachment | Self::ColorAttachmentRead => TextureLayout::ColorAttachment,
            Self::DepthAttachment => TextureLayout::DepthStencilAttachment,
            Self::DepthAttachmentRead => TextureLayout::DepthStencilReadOnly,
            Self::CopySrc => TextureLayout::TransferSrc,
            Self::CopyDst => TextureLayout::TransferDst,
            Self::Present => TextureLayout::Present,
        }
    }

    /// Lowers to a driver [`TextureState`] at the given pipeline `stages`.
    #[must_use]
    pub fn state(self, stages: PipelineStages) -> TextureState {
        let (accesses, layout) = match self {
            Self::Sampled | Self::InputAttachment => {
                (Accesses::SHADER_READ, TextureLayout::ShaderReadOnly)
            }
            Self::StorageRead => (Accesses::SHADER_READ, TextureLayout::General),
            Self::StorageWrite => (
                Accesses::SHADER_READ.union(Accesses::SHADER_WRITE),
                TextureLayout::General,
            ),
            Self::ColorAttachment => (
                Accesses::COLOR_ATTACHMENT_WRITE,
                TextureLayout::ColorAttachment,
            ),
            Self::ColorAttachmentRead => (
                Accesses::COLOR_ATTACHMENT_READ.union(Accesses::COLOR_ATTACHMENT_WRITE),
                TextureLayout::ColorAttachment,
            ),
            Self::DepthAttachment => (
                Accesses::DEPTH_STENCIL_READ.union(Accesses::DEPTH_STENCIL_WRITE),
                TextureLayout::DepthStencilAttachment,
            ),
            Self::DepthAttachmentRead => (
                Accesses::DEPTH_STENCIL_READ,
                TextureLayout::DepthStencilReadOnly,
            ),
            Self::CopySrc => (Accesses::TRANSFER_READ, TextureLayout::TransferSrc),
            Self::CopyDst => (Accesses::TRANSFER_WRITE, TextureLayout::TransferDst),
            Self::Present => (Accesses::NONE, TextureLayout::Present),
        };
        TextureState::new(self.stages_for(stages), accesses, layout)
    }

    /// Resolves the effective stage mask: attachment/transfer uses pin their
    /// own stage; shader uses adopt the pass's shader stage.
    #[must_use]
    fn stages_for(self, pass_stages: PipelineStages) -> PipelineStages {
        match self {
            Self::ColorAttachment | Self::ColorAttachmentRead => {
                PipelineStages::COLOR_ATTACHMENT_OUTPUT
            }
            Self::DepthAttachment | Self::DepthAttachmentRead => {
                PipelineStages::EARLY_FRAGMENT_TESTS.union(PipelineStages::LATE_FRAGMENT_TESTS)
            }
            Self::CopySrc | Self::CopyDst => PipelineStages::TRANSFER,
            Self::Present => PipelineStages::PRESENT,
            // Sampled / storage / input: run in whatever shader stage the pass
            // declares (fragment for raster, compute for compute).
            _ => pass_stages,
        }
    }
}

/// How a pass accesses a buffer.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum BufferUse {
    /// Uniform buffer read in a shader.
    Uniform,
    /// Read-only storage buffer.
    StorageRead,
    /// Read/write storage buffer.
    StorageWrite,
    /// Index buffer for indexed draws.
    Index,
    /// Vertex attribute buffer.
    Vertex,
    /// Indirect draw/dispatch argument source.
    Indirect,
    /// Transfer source.
    CopySrc,
    /// Transfer destination.
    CopyDst,
}

impl BufferUse {
    /// Whether this use writes the buffer.
    #[must_use]
    pub const fn is_write(self) -> bool {
        matches!(self, Self::StorageWrite | Self::CopyDst)
    }

    /// The GPU usage bit a buffer must carry to support this use.
    #[must_use]
    pub const fn required_usage(self) -> BufferUsages {
        match self {
            Self::Uniform => BufferUsages::UNIFORM,
            Self::StorageRead | Self::StorageWrite => BufferUsages::STORAGE,
            Self::Index => BufferUsages::INDEX,
            Self::Vertex => BufferUsages::VERTEX,
            Self::Indirect => BufferUsages::INDIRECT,
            Self::CopySrc => BufferUsages::COPY_SRC,
            Self::CopyDst => BufferUsages::COPY_DST,
        }
    }

    /// Lowers to a driver [`BufferState`] at the given pipeline `stages`.
    #[must_use]
    pub fn state(self, stages: PipelineStages) -> BufferState {
        match self {
            Self::Uniform => BufferState::uniform(stages),
            Self::StorageRead => BufferState::storage_read(stages),
            Self::StorageWrite => BufferState::storage_write(stages),
            Self::Index => BufferState::index(),
            Self::Vertex => BufferState::vertex(),
            Self::Indirect => BufferState::indirect(),
            Self::CopySrc => BufferState::copy_src(),
            Self::CopyDst => BufferState::copy_dst(),
        }
    }
}

/// A full subresource range covering every mip and layer, the default for
/// whole-resource accesses.
#[must_use]
pub fn full_range() -> SubresourceRange {
    SubresourceRange::all()
}
