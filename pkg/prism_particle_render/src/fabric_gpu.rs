//! Framework-independent physical publication for the Particle Render Fabric.
//!
//! The native RenderApp owns the original `BufferId` table entries.  This
//! packet carries those exact IDs and the semantic frame items to the host
//! adapter; it never allocates a replacement buffer or treats a raw ID as a
//! Bevy resource handle.

use prism_render_architecture::{
    particle::indirect_draw::RendererGeometry,
    render_fabric::{FrameRenderItem, RenderObjectId, ResourcePayload},
};
use prism_render_driver::{BufferId, BufferState, BufferUsages};

/// One renderer-owned Particle buffer published for a SceneFabric import.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ParticleFabricGpuBuffer {
    /// Canonical semantic resource declaration.
    pub semantic: ResourcePayload,
    /// ID in the producing RenderApp's renderer-owned table.
    pub buffer: BufferId,
    /// Buffer-table generation used as the physical publication version.
    pub version: u32,
    /// Complete physical allocation size.
    pub size: u64,
    /// Physical usage flags authenticated by the producer.
    pub usage: BufferUsages,
    /// State expected when the first SceneFabric pass imports the buffer.
    pub entry_state: BufferState,
}

/// One Particle emitter publication with real renderer-owned resources.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParticleFabricGpuDraw {
    /// Stable emitter identity used by the shared frame item.
    pub object_id: RenderObjectId,
    /// Stable authored consumer ordinal for renderer binding validation.
    pub renderer_index: u32,
    /// Temporal identity carried into the shared frame.
    pub history_key: u32,
    /// Minimum live-particle position in world space.
    pub bounds_min: [f32; 3],
    /// Maximum live-particle position in world space.
    pub bounds_max: [f32; 3],
    /// Exact geometry contract consumed by the particle renderer.
    pub geometry: RendererGeometry,
    /// Bridge-written renderer-input stream.
    pub stream: ParticleFabricGpuBuffer,
    /// Renderer-written compact visible-index stream.
    pub alive: ParticleFabricGpuBuffer,
}

/// Same-frame semantic and physical Particle publication exported by RenderApp.
#[derive(Clone, Debug, PartialEq)]
pub struct ParticleFabricScenePublication {
    frame: u64,
    items: Vec<FrameRenderItem>,
    draws: Vec<ParticleFabricGpuDraw>,
}

impl ParticleFabricScenePublication {
    /// Creates a publication after the producer has authenticated its frame.
    #[must_use]
    pub fn new(frame: u64, items: Vec<FrameRenderItem>, draws: Vec<ParticleFabricGpuDraw>) -> Self {
        Self {
            frame,
            items,
            draws,
        }
    }

    #[must_use]
    /// Returns the authenticated render frame number.
    pub const fn frame(&self) -> u64 {
        self.frame
    }

    #[must_use]
    /// Returns the semantic frame items to be imported by the host.
    pub fn items(&self) -> &[FrameRenderItem] {
        &self.items
    }

    #[must_use]
    /// Returns the physical per-emitter bindings.
    pub fn draws(&self) -> &[ParticleFabricGpuDraw] {
        &self.draws
    }

    #[must_use]
    /// Reports whether the publication contains no emitter work.
    pub fn is_empty(&self) -> bool {
        self.draws.is_empty()
    }
}
