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
//! No layer holds a `GPU` handle: the backend keys its physical store on
//! [`TexturePageKey`] and consults these tables to decide what to upload and
//! drop, pending the `GPU` backend.

pub mod feedback;
pub mod residency;
pub mod scheduler;

pub use feedback::{PageDemand, SemanticWeights, MAX_SCREEN_IMPORTANCE, MIP_URGENCY};
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
