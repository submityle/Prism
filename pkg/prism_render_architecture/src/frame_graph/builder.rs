use super::{
    CompileError, CompiledGpuFrameGraph, PassDescriptor, PassId, ResourceDescriptor, ResourceId,
};

#[derive(Default)]
pub struct GpuFrameGraphBuilder {
    resources: Vec<ResourceDescriptor>,
    passes: Vec<PassDescriptor>,
}

impl GpuFrameGraphBuilder {
    pub fn add_resource(&mut self, descriptor: ResourceDescriptor) -> ResourceId {
        let id = ResourceId(self.resources.len() as u32);
        self.resources.push(descriptor);
        id
    }
    pub fn add_pass(&mut self, descriptor: PassDescriptor) -> PassId {
        let id = PassId(self.passes.len() as u32);
        self.passes.push(descriptor);
        id
    }
    pub fn resources(&self) -> &[ResourceDescriptor] {
        &self.resources
    }
    pub fn passes(&self) -> &[PassDescriptor] {
        &self.passes
    }
    pub fn compile(&self) -> Result<CompiledGpuFrameGraph, CompileError> {
        CompiledGpuFrameGraph::compile(&self.resources, &self.passes)
    }
    pub fn clear(&mut self) {
        self.resources.clear();
        self.passes.clear();
    }
}
