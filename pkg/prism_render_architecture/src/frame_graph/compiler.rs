use core::fmt;

use super::transient::TransientAllocation;
use super::{
    AccessKind, Barrier, PassDescriptor, PassId, QueueBatch, ResourceDescriptor, ResourceId,
    ResourceVersion,
};

mod schedule;
use schedule::{dependencies, queue_batches, topological_order};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CompileError {
    InvalidResource { pass: PassId, resource: ResourceId },
    InvalidDependency { pass: PassId, dependency: PassId },
    CyclicDependency,
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GPU frame graph compile error: {self:?}")
    }
}
impl std::error::Error for CompileError {}

#[derive(Debug, Default)]
pub struct CompiledGpuFrameGraph {
    pub execution_order: Vec<PassId>,
    pub resource_versions: Vec<ResourceVersion>,
    pub barriers: Vec<Barrier>,
    pub queue_batches: Vec<QueueBatch>,
    pub transient_offsets: Vec<Option<u64>>,
    pub submission_count: u32,
    pub transient_bytes: u64,
    /// Validated, aliased transient-heap allocation plan (see
    /// [`TransientAllocation`]). `transient_offsets` / `transient_bytes` are
    /// derived views of this plan, kept for existing consumers.
    pub transient: TransientAllocation,
}

impl CompiledGpuFrameGraph {
    pub(crate) fn compile(
        resources: &[ResourceDescriptor],
        passes: &[PassDescriptor],
    ) -> Result<Self, CompileError> {
        validate(resources, passes)?;
        let dependencies = dependencies(resources, passes);
        let execution_order = topological_order(&dependencies)?;
        let (resource_versions, barriers) = hazards(passes, &execution_order);
        let queue_batches = queue_batches(passes, &execution_order, &dependencies);
        let transient = TransientAllocation::plan(resources, passes, &execution_order);
        Ok(Self {
            execution_order,
            resource_versions,
            barriers,
            submission_count: queue_batches.len() as u32,
            queue_batches,
            transient_offsets: transient.offsets(),
            transient_bytes: transient.heap_bytes(),
            transient,
        })
    }
}

fn validate(
    resources: &[ResourceDescriptor],
    passes: &[PassDescriptor],
) -> Result<(), CompileError> {
    for (index, pass) in passes.iter().enumerate() {
        let pass_id = PassId(index as u32);
        for access in &pass.accesses {
            if access.resource.0 as usize >= resources.len() {
                return Err(CompileError::InvalidResource {
                    pass: pass_id,
                    resource: access.resource,
                });
            }
        }
        for &dependency in &pass.depends_on {
            if dependency.0 as usize >= passes.len() || dependency == pass_id {
                return Err(CompileError::InvalidDependency {
                    pass: pass_id,
                    dependency,
                });
            }
        }
    }
    Ok(())
}

fn hazards(passes: &[PassDescriptor], order: &[PassId]) -> (Vec<ResourceVersion>, Vec<Barrier>) {
    let mut versions = Vec::new();
    let mut barriers = Vec::new();
    let mut writers: Vec<Option<(PassId, AccessKind, u32)>> = Vec::new();
    let mut readers: Vec<Vec<(PassId, AccessKind)>> = Vec::new();
    for &pass_id in order {
        let pass = &passes[pass_id.0 as usize];
        for access in &pass.accesses {
            let index = access.resource.0 as usize;
            writers.resize(writers.len().max(index + 1), None);
            readers.resize_with(readers.len().max(index + 1), Vec::new);
            let previous_version = writers[index].map_or(0, |value| value.2);
            if let Some((source, before, _)) = writers[index] {
                barriers.push(Barrier {
                    resource: access.resource,
                    source,
                    destination: pass_id,
                    before,
                    after: access.kind,
                    queue_transfer: passes[source.0 as usize].queue != pass.queue,
                });
            }
            if access.kind.writes() {
                for &(source, before) in &readers[index] {
                    barriers.push(Barrier {
                        resource: access.resource,
                        source,
                        destination: pass_id,
                        before,
                        after: access.kind,
                        queue_transfer: passes[source.0 as usize].queue != pass.queue,
                    });
                }
                readers[index].clear();
                let version = previous_version + 1;
                versions.push(ResourceVersion {
                    resource: access.resource,
                    writer: pass_id,
                    version,
                });
                writers[index] = Some((pass_id, access.kind, version));
            } else {
                readers[index].push((pass_id, access.kind));
            }
        }
    }
    (versions, barriers)
}
