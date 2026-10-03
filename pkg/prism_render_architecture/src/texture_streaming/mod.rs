//! Texture mip and virtual-texture streaming.
//!
//! This subsystem owns the `CPU`-side contract for streaming fixed-size tiles
//! (pages) of individual mip levels into a bounded physical pool. It is split
//! into three cooperating layers:
//!
//! * [`residency`] — the [`residency::TextureResidencyTable`] lifecycle state
//!   machine tracking each [`TexturePageKey`] as `NotResident`, `Requested`, or
//!   `Resident`, plus its priority, byte cost, and recency.
//! * [`feedback`] — turns per-page streaming feedback (desired vs resident mip,
//!   screen importance, [`TextureSemantic`]) into a fixed-point priority using
//!   only integer arithmetic, so the ordering is exact and deterministic.
//! * [`scheduler`] — a byte-budget-constrained greedy pass that picks the
//!   resident set and reports the frame's uploads and evictions.
//!
//! Two further layers turn that `CPU`-side decision into a device-ready backend
//! core, still holding no `GPU` handle so the crate stays device-free:
//!
//! * [`pool`] — the [`pool::PhysicalPagePool`] slot allocator that binds each
//!   resident [`TexturePageKey`] to a physical tile slot with a deterministic,
//!   lowest-free-first policy, and turns a [`scheduler::StreamingPlan`] into the
//!   frame's [`pool::PageUpload`] copy list.
//! * [`indirection`] — the [`indirection::GpuPageTable`] serialiser that packs
//!   the resident `(key, slot)` set into a flat, sorted, binary-searchable `u32`
//!   buffer whose compare order matches [`TexturePageKey`], so a shader resolves
//!   a page coordinate with the same comparison sequence as the golden
//!   [`indirection::GpuPageTable::lookup`].
//!
//! The only device-side work left to a scene-layer twin is recording the
//! staging-to-atlas copies described by [`pool::PageUpload`] and uploading
//! [`indirection::GpuPageTable::words`]; this crate computes both deterministically.

pub mod feedback;
pub mod indirection;
pub mod pool;
pub mod residency;
pub mod scheduler;

pub use feedback::{PageDemand, SemanticWeights, MAX_SCREEN_IMPORTANCE, MIP_URGENCY};
pub use indirection::{GpuPageTable, PAGE_TABLE_ENTRY_WORDS};
pub use pool::{PageUpload, PhysicalPagePool};
pub use residency::{PageRecord, PageResidency, TextureResidencyTable};
pub use scheduler::{schedule, schedule_and_apply, StreamingPlan};

/// Address of one streamed texture tile: a mip level of a layer, at a page grid
/// coordinate, within a texture. Ordered so residency containers iterate
/// deterministically.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TexturePageKey {
    /// Source texture identifier.
    pub texture: u32,
    /// Mip level of this page (0 = finest).
    pub mip: u8,
    /// Array layer or cube face.
    pub layer: u16,
    /// Page-grid X coordinate within the mip.
    pub x: u16,
    /// Page-grid Y coordinate within the mip.
    pub y: u16,
}

/// Perceptual channel a texture carries, used to bias streaming priority.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum TextureSemantic {
    /// Base color / albedo.
    Color,
    /// Tangent-space normal map.
    Normal,
    /// Packed roughness, metalness, and ambient-occlusion (`AO`) channels.
    RoughnessMetalAo,
    /// Height / displacement.
    Height,
    /// Coverage or selection mask.
    Mask,
    /// High-dynamic-range (`HDR`) data.
    Hdr,
}
