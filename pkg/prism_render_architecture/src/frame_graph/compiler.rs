use core::fmt;

use super::{
    AccessKind, Barrier, PassDescriptor, PassId, QueueBatch, ResourceDescriptor, ResourceId,
    ResourceLifetime, ResourceVersion,
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
        let (transient_offsets, transient_bytes) =
            transient_layout(resources, passes, &execution_order);
        Ok(Self {
            execution_order,
            resource_versions,
            barriers,
            submission_count: queue_batches.len() as u32,
            queue_batches,
            transient_offsets,
            transient_bytes,
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

fn transient_layout(
    resources: &[ResourceDescriptor],
    passes: &[PassDescriptor],
    order: &[PassId],
) -> (Vec<Option<u64>>, u64) {
    let mut lifetimes = vec![None::<(u32, u32)>; resources.len()];
    for (position, &pass_id) in order.iter().enumerate() {
        for access in &passes[pass_id.0 as usize].accesses {
            let life = &mut lifetimes[access.resource.0 as usize];
            *life = Some(
                life.map_or((position as u32, position as u32), |(first, _)| {
                    (first, position as u32)
                }),
            );
        }
    }
    let mut cursor = 0_u64;
    let mut offsets = vec![None; resources.len()];
    let mut blocks: Vec<(u64, u64, u32)> = Vec::new();
    let mut pending: Vec<_> = resources
        .iter()
        .enumerate()
        .filter_map(|(index, resource)| {
            (resource.lifetime == ResourceLifetime::Transient)
                .then(|| lifetimes[index].map(|life| (index, life)))
                .flatten()
        })
        .collect();
    pending.sort_by_key(|(index, (first, _))| (*first, *index));
    for (index, (first, last)) in pending {
        let resource = &resources[index];
        let alignment = resource.alignment.max(1);
        if let Some(block) = blocks.iter_mut().find(|(offset, size, available_after)| {
            *available_after < first && *size >= resource.size && *offset % alignment == 0
        }) {
            offsets[index] = Some(block.0);
            block.2 = last;
        } else {
            cursor = cursor.div_ceil(alignment) * alignment;
            offsets[index] = Some(cursor);
            blocks.push((cursor, resource.size, last));
            cursor += resource.size;
        }
    }
    (offsets, cursor)
}
