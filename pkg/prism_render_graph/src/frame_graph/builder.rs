//! Admission-checked declaration frontend; no graph algorithms live here.
use super::{AccessInput, PassDescriptor, PassId, ResourceDescriptor, ResourceId};
use crate::capacity::GraphLimits;
use crate::plan::{CompileError, ExecutionPlan};
use alloc::vec::Vec;

/// Startup topology declaration builder backed by the authoritative SSA kernel.
#[derive(Clone, Default)]
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
    pub fn compile(&self) -> Result<ExecutionPlan, CompileError> {
        crate::graph::RenderGraph::compile_declarations::<u32>(
            &self.resources,
            &self.passes,
            self.limits,
            None,
        )
    }
    /// Compiles with one explicit SSA input version per declared pass/access.
    ///
    /// Rows use original [`PassId`] order; columns use each pass's declared
    /// access order (not the compiler's texture-first normalized ordinals).
    /// Version zero denotes initial contents. Other inputs must already exist
    /// when lowering visits the access in the stable explicit-dependency order;
    /// add control dependencies when a producer must be lowered first.
    ///
    /// Reads may retain older versions. Writes may build on older inputs too,
    /// but still mint the next version of the resource's latest physical write;
    /// callers never choose output versions. RAW/WAR/WAW edges and the final
    /// schedule remain owned by the same kernel as [`Self::compile`].
    ///
    /// # Errors
    /// Returns [`CompileError::InvalidInputShape`] for a row/column mismatch,
    /// [`CompileError::InvalidHandle`] for a not-yet-existing input version,
    /// or the same checked admission, overflow and cycle errors as `compile`.
    pub fn compile_with_inputs(&self, inputs: &[Vec<u32>]) -> Result<ExecutionPlan, CompileError> {
        crate::graph::RenderGraph::compile_declarations(
            &self.resources,
            &self.passes,
            self.limits,
            Some(inputs),
        )
    }
    /// Orders only explicit control prerequisites using the shared kernel.
    ///
    /// Returns original pass IDs, including empty passes, with the same stable
    /// tie-break as declaration compilation. Resource accesses are not lowered
    /// or validated; this does not derive SSA, hazards, aliasing or barriers.
    ///
    /// # Errors
    /// Returns checked pass/dependency admission errors or a dependency cycle.
    pub fn dependency_order(&self) -> Result<Vec<PassId>, CompileError> {
        crate::graph::RenderGraph::declaration_dependency_order(&self.passes, self.limits)
    }
    /// Compiles explicit versions and symbolic producer outputs in one lowering.
    ///
    /// Rows and columns use original pass/declaration-access order, just like
    /// [`Self::compile_with_inputs`]. A producer must be a different pass's
    /// write to the same resource. References supplement control prerequisites;
    /// the existing kernel orders lowering, and the checked builder alone
    /// allocates each output version. Reads may retain older producer outputs.
    ///
    /// # Errors
    /// Returns [`CompileError::InvalidInputSource`] for an invalid producer,
    /// or the same shape, admission, version and cycle errors as the old APIs.
    pub fn compile_with_sources(
        &self,
        inputs: &[Vec<AccessInput>],
    ) -> Result<ExecutionPlan, CompileError> {
        crate::graph::RenderGraph::compile_declarations(
            &self.resources,
            &self.passes,
            self.limits,
            Some(inputs),
        )
    }
    /// Clears declarations, retaining the configured limits.
    pub fn clear(&mut self) {
        self.resources.clear();
        self.passes.clear();
    }
}
