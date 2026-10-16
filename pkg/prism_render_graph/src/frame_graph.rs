//! Declaration-only facade for startup topology graphs.
//!
//! No scheduler or hazard solver lives here: all declarations lower into the
//! same SSA kernel as executable `RenderGraph` passes.
mod builder;
mod codegen;
mod transient;
mod types;

pub use crate::plan::CompileError;
pub use builder::GpuFrameGraphBuilder;
pub use codegen::{
    declare_passes, AccessDeclaration, BindGroupContract, CodegenError, GeneratedPass,
    PassSkeletonDeclaration, PipelineContract, ResourceDeclaration, SkeletonAbi,
};
pub use transient::{AliasOverlap, TransientAllocation, TransientRegion};
pub use types::{
    AccessInput, AccessKind, Barrier, PassDescriptor, PassId, QueueBatch, QueueClass,
    ResourceAccess, ResourceDescriptor, ResourceId, ResourceLifetime, ResourceVersion,
};
