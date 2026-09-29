//! ECS-driven GPU frame graph with explicit resource lifetimes and hazards.

mod builder;
mod compiler;
mod transient;
mod types;

#[cfg(test)]
mod tests;

pub use builder::GpuFrameGraphBuilder;
pub use compiler::{CompileError, CompiledGpuFrameGraph};
pub use transient::{AliasOverlap, TransientAllocation, TransientRegion};
pub use types::{
    AccessKind, Barrier, PassDescriptor, PassId, QueueBatch, QueueClass, ResourceAccess,
    ResourceDescriptor, ResourceId, ResourceLifetime, ResourceVersion,
};
