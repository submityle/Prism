//! Declaration-to-SSA adapter and projections of the shared compiler product.
use super::{Barrier, PassDescriptor, PassId, QueueBatch, ResourceDescriptor, ResourceVersion};
use crate::capacity::GraphLimits;
use crate::compile::kernel::{self, Access};
pub use crate::plan::CompileError;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

/// Startup topology plan; all derived fields are views of the shared kernel.
#[derive(Debug, Default)]
pub struct CompiledGpuFrameGraph {
    /// Single source of truth for the runtime pass-count assertion.
    pub declared_pass_count: usize,
    /// Deterministically ordered original pass IDs.
    pub execution_order: Vec<PassId>,
    /// SSA writes in scheduled order.
    pub resource_versions: Vec<ResourceVersion>,
    /// Shared logical hazards projected to declaration access kinds.
    pub barriers: Vec<Barrier>,
    /// Consecutive same-queue batches with dependency waits.
    pub queue_batches: Vec<QueueBatch>,
    /// Per-resource offsets derived from transient metadata.
    pub transient_offsets: Vec<Option<u64>>,
    /// Number of queue batches, not physical submissions on an auto-sync backend.
    pub submission_count: u32,
    /// Transient heap address extent.
    pub transient_bytes: u64,
    /// Validated alias metadata.
    pub transient: super::TransientAllocation,
}

impl CompiledGpuFrameGraph {
    pub(crate) fn compile(
        resources: &[ResourceDescriptor],
        passes: &[PassDescriptor],
        limits: GraphLimits,
    ) -> Result<Self, CompileError> {
        GraphLimits::check_count("resources", resources.len(), limits.resources)?;
        GraphLimits::check_count("passes", passes.len(), limits.passes)?;
        for (p, pass) in passes.iter().enumerate() {
            GraphLimits::check_count("accesses", pass.accesses.len(), limits.accesses_per_pass)?;
            for access in &pass.accesses {
                if access.resource.0 as usize >= resources.len() {
                    return Err(CompileError::InvalidResource {
                        pass: PassId(p as u32),
                        resource: access.resource,
                    });
                }
            }
            for &dependency in &pass.depends_on {
                if dependency.0 as usize >= passes.len() || dependency.0 as usize == p {
                    return Err(CompileError::InvalidDependency {
                        pass: PassId(p as u32),
                        dependency,
                    });
                }
            }
        }
        let accesses = normalize_with_limits(resources, passes, limits)?;
        let explicit: Vec<_> = passes
            .iter()
            .map(|p| p.depends_on.iter().map(|p| p.0 as usize).collect())
            .collect();
        let names: Vec<String> = passes.iter().map(|p| p.name.clone().into_owned()).collect();
        let solved = kernel::solve(&accesses, &vec![true; passes.len()], &explicit, &names)?;
        let execution_order: Vec<_> = solved.order.iter().map(|&p| PassId(p as u32)).collect();
        let mut resource_versions = Vec::new();
        for &p in &solved.order {
            for (a, access) in accesses[p].iter().enumerate() {
                if let Some(version) = access.output {
                    resource_versions.push(ResourceVersion {
                        resource: passes[p].accesses[a].resource,
                        writer: PassId(p as u32),
                        version,
                    });
                }
            }
        }
        let barriers = solved
            .hazards
            .iter()
            .map(|h| Barrier {
                resource: passes[h.destination].accesses[h.after].resource,
                source: PassId(h.source as u32),
                destination: PassId(h.destination as u32),
                before: passes[h.source].accesses[h.before].kind,
                after: passes[h.destination].accesses[h.after].kind,
                queue_transfer: passes[h.source].queue != passes[h.destination].queue,
            })
            .collect();
        let queues = passes.iter().map(|p| p.queue).collect::<Vec<_>>();
        let queue_batches =
            crate::compile::schedule::queue_batches(&queues, &solved.order, &solved.dependencies);
        let intervals = kernel::lifetimes(&accesses, &solved.order, resources.len());
        let transient = super::TransientAllocation::from_intervals(resources, &intervals)?;
        Ok(Self {
            declared_pass_count: passes.len(),
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

fn normalize_with_limits(
    resources: &[ResourceDescriptor],
    passes: &[PassDescriptor],
    limits: GraphLimits,
) -> Result<Vec<Vec<Access>>, CompileError> {
    let mut versions = vec![0; resources.len()];
    passes
        .iter()
        .map(|p| {
            p.accesses
                .iter()
                .map(|a| {
                    let input = versions[a.resource.0 as usize];
                    let output = if a.kind.writes() {
                        let version = limits.next_version(input)?;
                        versions[a.resource.0 as usize] = version;
                        Some(version)
                    } else {
                        None
                    };
                    Ok(Access {
                        resource: a.resource.0 as usize,
                        input,
                        output,
                    })
                })
                .collect()
        })
        .collect()
}
