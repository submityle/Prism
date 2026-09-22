use alloc::borrow::Cow;
use core::ops::Range;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ResourceId(pub u32);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PassId(pub u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QueueClass {
    Graphics,
    Compute,
    Transfer,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResourceLifetime {
    Imported,
    Persistent,
    Transient,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResourceDescriptor {
    pub name: Cow<'static, str>,
    pub lifetime: ResourceLifetime,
    pub size: u64,
    pub alignment: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AccessKind {
    SampledRead,
    StorageRead,
    StorageWrite,
    ColorAttachment,
    DepthAttachment,
    IndirectRead,
    TransferRead,
    TransferWrite,
}

impl AccessKind {
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
pub struct ResourceAccess {
    pub resource: ResourceId,
    pub kind: AccessKind,
}

#[derive(Clone, Debug)]
pub struct PassDescriptor {
    pub name: Cow<'static, str>,
    pub queue: QueueClass,
    pub accesses: Vec<ResourceAccess>,
    pub depends_on: Vec<PassId>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResourceVersion {
    pub resource: ResourceId,
    pub writer: PassId,
    pub version: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Barrier {
    pub resource: ResourceId,
    pub source: PassId,
    pub destination: PassId,
    pub before: AccessKind,
    pub after: AccessKind,
    pub queue_transfer: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QueueBatch {
    pub queue: QueueClass,
    pub pass_range: Range<u32>,
    pub waits_for: Vec<u32>,
}
