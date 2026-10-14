//! Sole SSA hazard/schedule/lifetime/first-fit derivation for all frontends.
//!
//! Adapters only supply versioned access records. Texture/buffer namespaces are
//! flattened by the executable adapter; no algorithm depends on a backend.

use crate::plan::{AliasSlot, CompileError};
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Access {
    pub resource: usize,
    pub input: u32,
    pub output: Option<u32>,
}

/// Access indices preserve provenance so adapters project barriers, never solve hazards.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct Hazard {
    pub source: usize,
    pub before: usize,
    pub destination: usize,
    pub after: usize,
}

pub(crate) struct Solved {
    pub order: Vec<usize>,
    pub dependencies: Vec<BTreeSet<usize>>,
    pub hazards: Vec<Hazard>,
}

pub(crate) fn solve(
    accesses: &[Vec<Access>],
    alive: &[bool],
    explicit: &[Vec<usize>],
    names: &[String],
) -> Result<Solved, CompileError> {
    let mut producers = BTreeMap::new();
    let mut readers: BTreeMap<(usize, u32), Vec<(usize, usize)>> = BTreeMap::new();
    for (p, list) in accesses.iter().enumerate().filter(|(p, _)| alive[*p]) {
        for (a, access) in list.iter().enumerate() {
            if let Some(v) = access.output {
                producers.insert((access.resource, v), (p, a));
            } else {
                readers
                    .entry((access.resource, access.input))
                    .or_default()
                    .push((p, a));
            }
        }
    }
    let mut dependencies = vec![BTreeSet::new(); accesses.len()];
    let mut hazards = BTreeSet::new();
    for (p, list) in accesses.iter().enumerate().filter(|(p, _)| alive[*p]) {
        dependencies[p].extend(explicit[p].iter().copied().filter(|d| alive[*d]));
        for (a, access) in list.iter().enumerate() {
            let mut add = |source: usize, before: usize| {
                if source != p {
                    dependencies[p].insert(source);
                    hazards.insert(Hazard {
                        source,
                        before,
                        destination: p,
                        after: a,
                    });
                }
            };
            if let Some(&(source, before)) = producers.get(&(access.resource, access.input)) {
                add(source, before);
            }
            if let Some(output) = access.output {
                // Physical storage has a single owner: stale write handles may
                // build on an older input, but must not overwrite live versions.
                let replaced = output - 1;
                if let Some(&(source, before)) = producers.get(&(access.resource, replaced)) {
                    add(source, before);
                }
                if let Some(prior_readers) = readers.get(&(access.resource, replaced)) {
                    for &(source, before) in prior_readers {
                        add(source, before);
                    }
                }
            }
        }
    }
    let n = accesses.len();
    let mut successors = vec![Vec::new(); n];
    let mut indegrees = vec![0; n];
    for (p, deps) in dependencies.iter().enumerate() {
        indegrees[p] = deps.len();
        for &d in deps {
            successors[d].push(p);
        }
    }
    let mut ready: BTreeSet<_> = (0..n).filter(|&p| alive[p] && indegrees[p] == 0).collect();
    let mut order = Vec::new();
    while let Some(p) = ready.pop_first() {
        order.push(p);
        for &s in &successors[p] {
            indegrees[s] -= 1;
            if indegrees[s] == 0 {
                ready.insert(s);
            }
        }
    }
    let blocked: Vec<_> = (0..n).filter(|&p| alive[p] && indegrees[p] > 0).collect();
    if !blocked.is_empty() {
        // Every blocked vertex has a blocked predecessor. Following one yields
        // an actual cycle, not a list mislabelling downstream victims as cyclic.
        let blocked_set: BTreeSet<_> = blocked.iter().copied().collect();
        let mut path = Vec::new();
        let mut visited = BTreeMap::new();
        let mut next = blocked[0];
        let cycle = loop {
            if let Some(&start) = visited.get(&next) {
                let mut cycle = path[start..].to_vec();
                cycle.push(next);
                cycle.reverse(); // producer -> consumer edges
                break cycle;
            }
            visited.insert(next, path.len());
            path.push(next);
            next = *dependencies[next]
                .iter()
                .find(|d| blocked_set.contains(d))
                .expect("Kahn residual vertex has a residual predecessor");
        };
        let cycle_names = cycle.iter().map(|&p| names[p].clone()).collect();
        return Err(CompileError::Cycle {
            unscheduled: blocked.len(),
            blocked,
            cycle,
            names: cycle_names,
        });
    }
    Ok(Solved {
        order,
        dependencies,
        hazards: hazards.into_iter().collect(),
    })
}

pub(crate) fn lifetimes(
    accesses: &[Vec<Access>],
    order: &[usize],
    count: usize,
) -> Vec<Option<(usize, usize)>> {
    let mut lives = vec![None; count];
    for (pos, &p) in order.iter().enumerate() {
        for access in &accesses[p] {
            let slot = &mut lives[access.resource];
            *slot = Some(slot.map_or((pos, pos), |(first, _)| (first, pos)));
        }
    }
    lives
}

#[derive(Clone, Copy)]
pub(crate) struct AllocationRequest {
    pub size: u64,
    pub alignment: u64,
    pub first: usize,
    pub last: usize,
}

pub(crate) fn align_up(value: u64, alignment: u64) -> Result<u64, CompileError> {
    let alignment = alignment.max(1);
    let remainder = value % alignment;
    if remainder == 0 {
        Ok(value)
    } else {
        value
            .checked_add(alignment - remainder)
            .ok_or(CompileError::MemoryOverflow)
    }
}

/// Deterministic first fit; heap size is the address extent, not live-byte peak.
pub(crate) fn allocate(
    requests: &[Option<AllocationRequest>],
) -> Result<(Vec<Option<AliasSlot>>, u64), CompileError> {
    let mut pending: Vec<_> = requests
        .iter()
        .enumerate()
        .filter_map(|(i, r)| r.map(|r| (i, r)))
        .collect();
    pending.sort_by_key(|(i, r)| (r.first, *i));
    let mut slots = vec![None; requests.len()];
    let mut blocks: Vec<(u64, u64, usize)> = Vec::new();
    let mut cursor = 0;
    for (i, r) in pending {
        let alignment = r.alignment.max(1);
        let offset = if let Some(block) = blocks.iter_mut().find(|(offset, size, last)| {
            *last < r.first && *size >= r.size && *offset % alignment == 0
        }) {
            block.2 = r.last;
            block.0
        } else {
            let offset = align_up(cursor, alignment)?;
            cursor = offset
                .checked_add(r.size)
                .ok_or(CompileError::MemoryOverflow)?;
            blocks.push((offset, r.size, r.last));
            offset
        };
        slots[i] = Some(AliasSlot {
            offset,
            size: r.size,
        });
    }
    Ok((slots, cursor))
}
