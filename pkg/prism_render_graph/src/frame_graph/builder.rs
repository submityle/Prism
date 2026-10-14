//! Admission-checked declaration frontend; no graph algorithms live here.
use super::{
    CompileError, CompiledGpuFrameGraph, PassDescriptor, PassId, ResourceDescriptor, ResourceId,
};
use crate::capacity::GraphLimits;
use alloc::vec::Vec;

/// Startup topology declaration builder backed by the authoritative SSA kernel.
#[derive(Default)]
pub struct GpuFrameGraphBuilder {
    resources: Vec<ResourceDescriptor>,
    passes: Vec<PassDescriptor>,
    limits: GraphLimits,
}

impl GpuFrameGraphBuilder {
    /// Creates an empty graph with admission limits.
    pub fn with_limits(limits: GraphLimits) -> Self {
        Self {
            limits,
            ..Self::default()
        }
    }

    /// Adds a resource. Use [`Self::try_add_resource`] for dynamic declarations.
    /// # Panics
    /// The startup declaration must fit the configured resource limit.
    pub fn add_resource(&mut self, descriptor: ResourceDescriptor) -> ResourceId {
        self.try_add_resource(descriptor)
            .expect("startup graph resource capacity precondition")
    }
    /// Checks admission before narrowing an ID or growing the resource table.
    pub fn try_add_resource(
        &mut self,
        descriptor: ResourceDescriptor,
    ) -> Result<ResourceId, CompileError> {
        let id = ResourceId(GraphLimits::next_id(
            "resources",
            self.resources.len(),
            self.limits.resources,
        )?);
        self.resources.push(descriptor);
        Ok(id)
    }
    /// Adds a pass. Use [`Self::try_add_pass`] for dynamic declarations.
    /// # Panics
    /// The startup declaration must fit pass/access limits.
    pub fn add_pass(&mut self, descriptor: PassDescriptor) -> PassId {
        self.try_add_pass(descriptor)
            .expect("startup graph pass capacity precondition")
    }
    /// Rejects an oversized pass before inserting it in the table.
    pub fn try_add_pass(&mut self, descriptor: PassDescriptor) -> Result<PassId, CompileError> {
        let id = PassId(GraphLimits::next_id(
            "passes",
            self.passes.len(),
            self.limits.passes,
        )?);
        GraphLimits::check_count(
            "accesses",
            descriptor.accesses.len(),
            self.limits.accesses_per_pass,
        )?;
        GraphLimits::check_count(
            "dependencies",
            descriptor.depends_on.len(),
            self.limits.passes,
        )?;
        self.passes.push(descriptor);
        Ok(id)
    }
    /// Resource declaration table.
    pub fn resources(&self) -> &[ResourceDescriptor] {
        &self.resources
    }
    /// Pass declaration table, including empty passes.
    pub fn passes(&self) -> &[PassDescriptor] {
        &self.passes
    }
    /// Compiles through the same SSA kernel used by executable `RenderGraph`.
    pub fn compile(&self) -> Result<CompiledGpuFrameGraph, CompileError> {
        CompiledGpuFrameGraph::compile(&self.resources, &self.passes, self.limits)
    }
    /// Clears declarations, retaining the configured limits.
    pub fn clear(&mut self) {
        self.resources.clear();
        self.passes.clear();
    }
}
