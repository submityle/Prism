//! Deterministic topological scheduling of the surviving passes.
//!
//! The graph derives execution order purely from SSA data flow — no author
//! ever wires an edge by hand. Three hazard classes become edges:
//!
//! - **Read-after-write / build-on**: a pass that observes version `v` of a
//!   resource must run after the pass that produced `v`.
//! - **Write-after-read**: a pass that produces `v+1` must run after every pass
//!   that still reads `v`, so a writer never clobbers a live reader.
//!
//! - **Write-after-write**: output versions serialize physical storage even
//!   when a writer builds on a stale input handle.
//!
//! A Kahn topological sort then linearizes the DAG. Ties — passes that become
//! ready simultaneously — are broken by original insertion index, so a given
//! graph always yields byte-for-byte the same schedule, which matters for
//! reproducible captures and golden tests.

use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;
use core::ops::Range;

use super::kernel;
use crate::frame_graph::{QueueBatch, QueueClass};
use crate::pass::{PassFlags, PassKind, PassNode};
use crate::plan::CompileError;

/// Flattens typed resource namespaces without deriving any dependencies.
pub(crate) fn normalize(passes: &[PassNode], texture_count: usize) -> Vec<Vec<kernel::Access>> {
    passes
        .iter()
        .map(|pass| {
            pass.texture_accesses
                .iter()
                .map(|acc| kernel::Access {
                    resource: acc.resource.get() as usize,
                    input: acc.input_version,
                    output: acc.produces.then_some(acc.output_version),
                })
                .chain(pass.buffer_accesses.iter().map(|acc| kernel::Access {
                    resource: texture_count + acc.resource.get() as usize,
                    input: acc.input_version,
                    output: acc.produces.then_some(acc.output_version),
                }))
                .collect()
        })
        .collect()
}

/// Shared SSA solver is the only source of dependencies, hazards and order.
pub(crate) fn schedule(
    passes: &[PassNode],
    alive: &[bool],
    accesses: &[Vec<kernel::Access>],
) -> Result<kernel::Solved, CompileError> {
    let names = passes.iter().map(|p| p.name.clone()).collect::<Vec<_>>();
    let explicit: Vec<Vec<usize>> = passes
        .iter()
        .map(|pass| pass.dependencies.clone())
        .collect();
    kernel::solve(accesses, alive, &explicit, &names)
}

/// Compute without async eligibility stays on the graphics queue.
pub(crate) fn queue_class(pass: &PassNode) -> QueueClass {
    match pass.kind {
        PassKind::Compute if pass.flags.contains(PassFlags::ASYNC_COMPUTE) => QueueClass::Compute,
        PassKind::Transfer => QueueClass::Transfer,
        _ => QueueClass::Graphics,
    }
}

/// Groups the scheduled passes into consecutive queue submissions.
///
/// `order` contains original pass indices and `dependencies` is indexed by
/// those same indices. A batch waits only on earlier batches that contain one
/// of its dependency passes; the result is deterministic because both the
/// ranges and wait sets are built in schedule order.
pub(crate) fn queue_batches(
    queues: &[QueueClass],
    order: &[usize],
    dependencies: &[BTreeSet<usize>],
) -> Vec<QueueBatch> {
    if order.is_empty() {
        return Vec::new();
    }

    let mut batches: Vec<(QueueClass, Range<u32>)> = Vec::new();
    let mut pass_to_batch = vec![u32::MAX; queues.len()];
    for (position, &pass) in order.iter().enumerate() {
        let queue = queues[pass];
        if batches
            .last()
            .is_none_or(|(last_queue, _)| *last_queue != queue)
        {
            batches.push((queue, position as u32..position as u32));
        }
        let batch = batches.len() - 1;
        batches[batch].1.end = position as u32 + 1;
        pass_to_batch[pass] = batch as u32;
    }

    batches
        .into_iter()
        .enumerate()
        .map(|(batch_index, (queue, pass_range))| {
            let mut waits_for = BTreeSet::new();
            for position in pass_range.clone() {
                let pass = order[position as usize];
                for &dependency in &dependencies[pass] {
                    let dependency_batch = pass_to_batch[dependency];
                    if dependency_batch != u32::MAX && dependency_batch != batch_index as u32 {
                        waits_for.insert(dependency_batch);
                    }
                }
            }
            QueueBatch {
                queue,
                pass_range,
                waits_for: waits_for.into_iter().collect(),
            }
        })
        .collect()
}
