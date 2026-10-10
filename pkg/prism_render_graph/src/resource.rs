//! The graph's virtual-resource table: transient, imported, and persistent.
//!
//! Every texture or buffer a frame touches is registered here first as a pure
//! *description*. Only after compilation does the executor bind real driver
//! objects to them. Separating the three lifetime classes is what lets the
//! compiler alias transient memory safely while leaving imported swapchain
//! images and cross-frame history buffers untouched.

use crate::desc::{BufferDesc, TextureDesc};
use alloc::string::String;
use prism_render_driver::{
    BufferId, BufferState, BufferUsages, TextureId, TextureState, TextureUsages, TextureViewId,
};

/// The lifetime class of a virtual resource.
///
/// This is the single most important property for the compiler: it decides
/// whether a resource participates in transient memory aliasing, whether the
/// graph allocates it, and whether its contents survive the frame.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Lifetime {
    /// Lives only this frame; the graph allocates it and may alias its memory
    /// with other transients whose lifetimes do not overlap.
    Transient,
    /// Supplied by the outside world (swapchain image, resident asset); the
    /// graph only tracks state transitions, never allocation.
    Imported,
    /// Allocated by the graph but preserved across frames (history buffers);
    /// transitions like any resource but is never aliased.
    Persistent,
}

impl Lifetime {
    /// Whether resources of this class may share physical memory via aliasing.
    #[must_use]
    pub const fn is_aliasable(self) -> bool {
        matches!(self, Self::Transient)
    }
}

/// An externally-owned texture injected into the graph.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ImportedTexture {
    /// The real driver texture.
    pub texture: TextureId,
    /// A default whole-resource view used when the texture is an attachment.
    pub default_view: TextureViewId,
    /// The texture's state on entry to the frame (what the previous owner left
    /// it in), used to seed the barrier solver.
    pub entry_state: TextureState,
    /// The state the graph must leave the texture in on frame exit (e.g.
    /// [`TextureState::present`] for a swapchain image).
    pub exit_state: TextureState,
    /// Mip level count, for subresource-granular tracking.
    pub mip_levels: u32,
    /// Array layer count.
    pub array_layers: u32,
}

/// An externally-owned buffer injected into the graph.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ImportedBuffer {
    /// The real driver buffer.
    pub buffer: BufferId,
    /// The buffer's state on entry to the frame.
    pub entry_state: BufferState,
}

/// A virtual texture: its description, lifetime, and (after realization) the
/// driver objects bound to it.
#[derive(Clone, Debug)]
pub struct TextureResource {
    /// A debug name surfaced in diagnostics and barrier logs.
    pub name: String,
    /// The symbolic description.
    pub desc: TextureDesc,
    /// The lifetime class.
    pub lifetime: Lifetime,
    /// Import payload, present iff `lifetime == Imported`.
    pub imported: Option<ImportedTexture>,
    /// Usage bits inferred from how passes access this resource, unioned into
    /// the creation descriptor by the compiler.
    pub inferred_usage: TextureUsages,
    /// The current SSA write version (number of writes recorded so far).
    pub version: u32,
    /// The driver texture bound during realization (transient/persistent get a
    /// freshly created one; imported reuses its own).
    pub realized: Option<RealizedTexture>,
}

/// The concrete driver objects bound to a texture resource for one frame.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RealizedTexture {
    /// The driver texture.
    pub texture: TextureId,
    /// A whole-resource default view (for attachment use).
    pub default_view: TextureViewId,
}

/// A virtual buffer: its description, lifetime, and realized driver object.
#[derive(Clone, Debug)]
pub struct BufferResource {
    /// A debug name surfaced in diagnostics and barrier logs.
    pub name: String,
    /// The symbolic description.
    pub desc: BufferDesc,
    /// The lifetime class.
    pub lifetime: Lifetime,
    /// Import payload, present iff `lifetime == Imported`.
    pub imported: Option<ImportedBuffer>,
    /// Usage bits inferred from how passes access this resource.
    pub inferred_usage: BufferUsages,
    /// The current SSA write version.
    pub version: u32,
    /// The driver buffer bound during realization.
    pub realized: Option<BufferId>,
}

impl TextureResource {
    /// Whether this resource is a transient the compiler may alias.
    #[must_use]
    pub const fn is_transient(&self) -> bool {
        matches!(self.lifetime, Lifetime::Transient)
    }
}

impl BufferResource {
    /// Whether this resource is a transient the compiler may alias.
    #[must_use]
    pub const fn is_transient(&self) -> bool {
        matches!(self.lifetime, Lifetime::Transient)
    }
}
