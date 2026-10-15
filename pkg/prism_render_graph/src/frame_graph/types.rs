//! Backend-neutral declaration records; compilation is owned by the shared SSA kernel.
use alloc::borrow::Cow;
use alloc::vec::Vec;
use core::ops::Range;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
/// A declaration resource-table index, stable until the builder is cleared.
pub struct ResourceId(
    /// The checked table index.
    pub u32,
);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
/// An original declaration pass-table index (not a scheduled position).
pub struct PassId(
    /// The checked table index.
    pub u32,
);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Logical queue class used to group consecutive scheduled passes.
pub enum QueueClass {
    /// Raster and present work.
    Graphics,
    /// Compute work eligible for a separate queue.
    Compute,
    /// Copy and transfer work.
    Transfer,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Ownership policy; only transient resources may share heap regions.
pub enum ResourceLifetime {
    /// Externally owned resource; never allocated or aliased by the graph.
    Imported,
    /// Graph-owned resource whose contents survive the frame.
    Persistent,
    /// Frame-local resource eligible for lifetime-disjoint aliasing.
    Transient,
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// Backend-neutral resource footprint and lifetime declaration.
pub struct ResourceDescriptor {
    /// Human-readable diagnostic name.
    pub name: Cow<'static, str>,
    /// Ownership and aliasing policy.
    pub lifetime: ResourceLifetime,
    /// Unpadded byte footprint; zero is valid for topology-only declarations.
    pub size: u64,
    /// Required byte-offset alignment; zero is normalized to one.
    pub alignment: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Logical resource use, independent of backend layouts and subresources.
pub enum AccessKind {
    /// Shader sampled read.
    SampledRead,
    /// Read-only storage access.
    StorageRead,
    /// Storage write producing a new SSA version.
    StorageWrite,
    /// Color attachment write.
    ColorAttachment,
    /// Depth attachment write.
    DepthAttachment,
    /// Indirect dispatch or draw argument read.
    IndirectRead,
    /// Copy-source read.
    TransferRead,
    /// Copy-destination write.
    TransferWrite,
}

impl AccessKind {
    /// Whether the access mints a new resource version.
    pub const fn writes(self) -> bool {
        matches!(
            self,
            Self::StorageWrite
                | Self::ColorAttachment
                | Self::DepthAttachment
                | Self::TransferWrite
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// A resource use in declaration order; writes are versioned during lowering.
pub struct ResourceAccess {
    /// Resource-table index.
    pub resource: ResourceId,
    /// Logical access mode.
    pub kind: AccessKind,
}

#[derive(Clone, Debug)]
/// A startup pass declaration; prerequisites supplement resource hazards.
/// Implicit resource versions follow a stable topological order of explicit
/// prerequisites, with original pass IDs breaking ties. Table order alone
/// therefore never overrides an explicit prerequisite.
pub struct PassDescriptor {
    /// Human-readable diagnostic name.
    pub name: Cow<'static, str>,
    /// Logical queue class (not a physical submission promise).
    pub queue: QueueClass,
    /// Resource uses in declaration order.
    pub accesses: Vec<ResourceAccess>,
    /// Explicit prerequisites, indexed by original pass ID.
    pub depends_on: Vec<PassId>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// A resource write in scheduled order.
pub struct ResourceVersion {
    /// Resource-table index.
    pub resource: ResourceId,
    /// Original pass ID that produced the version.
    pub writer: PassId,
    /// SSA output version, beginning at one.
    pub version: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// A logical RAW/WAR/WAW hazard projected from the authoritative SSA solution.
pub struct Barrier {
    /// Resource-table index.
    pub resource: ResourceId,
    /// Original prerequisite pass ID.
    pub source: PassId,
    /// Original dependent pass ID.
    pub destination: PassId,
    /// Access mode in the prerequisite pass.
    pub before: AccessKind,
    /// Access mode in the dependent pass.
    pub after: AccessKind,
    /// Whether the hazard crosses logical queue classes.
    pub queue_transfer: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// Consecutive same-queue work with dependency waits on earlier batches.
pub struct QueueBatch {
    /// Logical queue class (not a physical submission promise).
    pub queue: QueueClass,
    /// Half-open scheduled-position range, not original pass IDs.
    pub pass_range: Range<u32>,
    /// Sorted, deduplicated indices of prerequisite batches.
    pub waits_for: Vec<u32>,
}
