//! ECS-driven GPU frame graph contracts.

use std::borrow::Cow;

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
}

#[derive(Default)]
pub struct GpuFrameGraphBuilder {
    passes: Vec<PassDescriptor>,
}

impl GpuFrameGraphBuilder {
    pub fn add_pass(&mut self, descriptor: PassDescriptor) -> PassId {
        let id = PassId(self.passes.len() as u32);
        self.passes.push(descriptor);
        id
    }

    pub fn passes(&self) -> &[PassDescriptor] {
        &self.passes
    }

    pub fn clear(&mut self) {
        self.passes.clear();
    }
}

/// Immutable output of frame-graph compilation.
#[derive(Default)]
pub struct CompiledGpuFrameGraph {
    pub execution_order: Vec<PassId>,
    pub submission_count: u32,
    pub transient_bytes: u64,
}
