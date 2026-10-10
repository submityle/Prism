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
//! (Write-after-write falls out for free: SSA gives each write a distinct
//! output version, and the later writer builds on the earlier one, which is
//! already a build-on edge.)
//!
//! A Kahn topological sort then linearizes the DAG. Ties — passes that become
//! ready simultaneously — are broken by original insertion index, so a given
//! graph always yields byte-for-byte the same schedule, which matters for
//! reproducible captures and golden tests.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec;
use alloc::vec::Vec;

use crate::pass::PassNode;
use crate::plan::CompileError;

/// Orders the alive passes, or reports a dependency cycle.
///
/// Only passes with `alive[i] == true` participate; dead passes contribute no
/// edges and never appear in the result.
pub(crate) fn schedule(passes: &[PassNode], alive: &[bool]) -> Result<Vec<usize>, CompileError> {
    let n = passes.len();

    let mut tex_producer: BTreeMap<(u32, u32), usize> = BTreeMap::new();
    let mut buf_producer: BTreeMap<(u32, u32), usize> = BTreeMap::new();
    let mut tex_readers: BTreeMap<(u32, u32), Vec<usize>> = BTreeMap::new();
    let mut buf_readers: BTreeMap<(u32, u32), Vec<usize>> = BTreeMap::new();

    for (idx, pass) in passes.iter().enumerate() {
        if !alive[idx] {
            continue;
        }
        for acc in &pass.texture_accesses {
            if acc.produces {
                tex_producer.insert((acc.resource.get(), acc.output_version), idx);
            } else {
                tex_readers
                    .entry((acc.resource.get(), acc.input_version))
                    .or_default()
                    .push(idx);
            }
        }
        for acc in &pass.buffer_accesses {
            if acc.produces {
                buf_producer.insert((acc.resource.get(), acc.output_version), idx);
            } else {
                buf_readers
                    .entry((acc.resource.get(), acc.input_version))
                    .or_default()
                    .push(idx);
            }
        }
    }

    // Deduplicated edges so Kahn in-degrees stay exact.
    let mut edges: BTreeSet<(usize, usize)> = BTreeSet::new();
    for (idx, pass) in passes.iter().enumerate() {
        if !alive[idx] {
            continue;
        }
        for acc in &pass.texture_accesses {
            // RAW / build-on: depend on whoever produced the observed version.
            if let Some(&prod) = tex_producer.get(&(acc.resource.get(), acc.input_version))
                && prod != idx
            {
                edges.insert((prod, idx));
            }
            // WAR: a writer runs after every reader of the version it replaces.
            if acc.produces
                && let Some(readers) = tex_readers.get(&(acc.resource.get(), acc.input_version))
            {
                for &r in readers {
                    if r != idx {
                        edges.insert((r, idx));
                    }
                }
            }
        }
        for acc in &pass.buffer_accesses {
            if let Some(&prod) = buf_producer.get(&(acc.resource.get(), acc.input_version))
                && prod != idx
            {
                edges.insert((prod, idx));
            }
            if acc.produces
                && let Some(readers) = buf_readers.get(&(acc.resource.get(), acc.input_version))
            {
                for &r in readers {
                    if r != idx {
                        edges.insert((r, idx));
                    }
                }
            }
        }
    }

    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut indeg: Vec<usize> = vec![0; n];
    for &(a, b) in &edges {
        succ[a].push(b);
        indeg[b] += 1;
    }

    // Ready set ordered by pass index for a deterministic tie-break.
    let mut ready: BTreeSet<usize> = BTreeSet::new();
    for (idx, &is_alive) in alive.iter().enumerate() {
        if is_alive && indeg[idx] == 0 {
            ready.insert(idx);
        }
    }

    let mut order: Vec<usize> = Vec::new();
    while let Some(&p) = ready.iter().next() {
        ready.remove(&p);
        order.push(p);
        for &s in &succ[p] {
            indeg[s] -= 1;
            if indeg[s] == 0 {
                ready.insert(s);
            }
        }
    }

    let alive_count = alive.iter().filter(|&&a| a).count();
    if order.len() != alive_count {
        return Err(CompileError::Cycle {
            unscheduled: alive_count - order.len(),
        });
    }
    Ok(order)
}
