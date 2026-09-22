use alloc::collections::BTreeSet;

use super::{CompileError, PassDescriptor, PassId, QueueBatch, ResourceDescriptor};

pub(super) fn dependencies(
    resources: &[ResourceDescriptor],
    passes: &[PassDescriptor],
) -> Vec<BTreeSet<usize>> {
    let mut deps: Vec<_> = passes
        .iter()
        .map(|pass| {
            pass.depends_on
                .iter()
                .map(|id| id.0 as usize)
                .collect::<BTreeSet<_>>()
        })
        .collect();
    let mut last_writer = vec![None; resources.len()];
    let mut readers = vec![BTreeSet::new(); resources.len()];
    for (pass_index, pass) in passes.iter().enumerate() {
        for access in &pass.accesses {
            let resource = access.resource.0 as usize;
            if access.kind.writes() {
                if let Some(writer) = last_writer[resource] {
                    deps[pass_index].insert(writer);
                }
                deps[pass_index].extend(readers[resource].iter().copied());
                readers[resource].clear();
                last_writer[resource] = Some(pass_index);
            } else {
                if let Some(writer) = last_writer[resource] {
                    deps[pass_index].insert(writer);
                }
                readers[resource].insert(pass_index);
            }
        }
    }
    deps
}

pub(super) fn topological_order(
    dependencies: &[BTreeSet<usize>],
) -> Result<Vec<PassId>, CompileError> {
    let mut emitted = vec![false; dependencies.len()];
    let mut order = Vec::with_capacity(dependencies.len());
    while order.len() != dependencies.len() {
        let Some(next) = dependencies.iter().enumerate().position(|(index, deps)| {
            !emitted[index] && deps.iter().all(|dependency| emitted[*dependency])
        }) else {
            return Err(CompileError::CyclicDependency);
        };
        emitted[next] = true;
        order.push(PassId(next as u32));
    }
    Ok(order)
}

pub(super) fn queue_batches(
    passes: &[PassDescriptor],
    order: &[PassId],
    dependencies: &[BTreeSet<usize>],
) -> Vec<QueueBatch> {
    let mut batches: Vec<QueueBatch> = Vec::new();
    let mut pass_to_batch = vec![0_u32; passes.len()];
    for (position, &pass_id) in order.iter().enumerate() {
        let queue = passes[pass_id.0 as usize].queue;
        if batches.last().is_none_or(|batch| batch.queue != queue) {
            batches.push(QueueBatch {
                queue,
                pass_range: position as u32..position as u32 + 1,
                waits_for: Vec::new(),
            });
        } else {
            batches.last_mut().unwrap().pass_range.end += 1;
        }
        pass_to_batch[pass_id.0 as usize] = (batches.len() - 1) as u32;
    }
    for &pass_id in order {
        let batch_id = pass_to_batch[pass_id.0 as usize];
        let waits = &mut batches[batch_id as usize].waits_for;
        for dependency in &dependencies[pass_id.0 as usize] {
            let dependency_batch = pass_to_batch[*dependency];
            if dependency_batch != batch_id {
                waits.push(dependency_batch);
            }
        }
        waits.sort_unstable();
        waits.dedup();
    }
    batches
}
