//! Schedule ambiguity detection (design §23.4).
//!
//! Two systems are *ambiguous* when their [`Access`] sets conflict — one writes
//! a component or resource the other reads or writes, or one is exclusive — yet
//! the schedule contains **no ordering edge** (directly or transitively) that
//! fixes which runs first. Their relative order is then an implementation
//! detail of the topological tie-break, so a change to insertion order can
//! silently change behaviour. That is exactly the class of non-determinism
//! §23.4 wants surfaced.
//!
//! [`Schedule::ambiguities`](crate::schedule::Schedule::ambiguities) returns the
//! full set; [`Schedule::assert_no_ambiguities`](crate::schedule::Schedule::assert_no_ambiguities)
//! turns it into a hard gate (the deterministic analogue of the cycle panic),
//! suitable for a CI check. Phases are totally chained, so only systems sharing
//! a phase (and lacking an explicit `before`/`after`/`chain`) can ever be
//! reported.

use alloc::string::String;
use alloc::vec::Vec;

use crate::component::ComponentId;
use crate::query::Access;
use crate::resource::ResourceId;

/// One pair of systems whose access conflicts without an ordering edge.
#[derive(Clone, Debug)]
pub struct Ambiguity {
    /// Node index of the first system in the schedule.
    pub first: usize,
    /// Node index of the second system in the schedule.
    pub second: usize,
    /// Name of the first system (from [`System::name`](crate::system::System::name)).
    pub first_name: String,
    /// Name of the second system.
    pub second_name: String,
    /// Components both systems touch where at least one writes.
    pub components: Vec<ComponentId>,
    /// Resources both systems touch where at least one writes.
    pub resources: Vec<ResourceId>,
    /// `true` when the conflict is because one system is exclusive (borrows the
    /// whole world); in that case `components`/`resources` may be empty.
    pub whole_world: bool,
}

/// The result of an ambiguity analysis over a [`Schedule`](crate::schedule::Schedule).
///
/// Deterministic: pairs are reported in ascending `(first, second)` node-index
/// order.
#[derive(Clone, Debug, Default)]
pub struct Ambiguities {
    pairs: Vec<Ambiguity>,
}

impl Ambiguities {
    /// The detected ambiguous pairs, in deterministic order.
    #[inline]
    #[must_use]
    pub fn pairs(&self) -> &[Ambiguity] {
        &self.pairs
    }

    /// Number of ambiguous pairs.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    /// Whether no ambiguities were found.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    /// Iterate over the ambiguous pairs.
    #[inline]
    pub fn iter(&self) -> core::slice::Iter<'_, Ambiguity> {
        self.pairs.iter()
    }

    /// A human-readable, multi-line report (one line per pair). Empty string
    /// when there are no ambiguities.
    #[must_use]
    pub fn report(&self) -> String {
        use core::fmt::Write as _;
        let mut out = String::new();
        if self.pairs.is_empty() {
            return out;
        }
        let _ = write!(
            out,
            "prism_ecs schedule: {} system ambiguit{} detected (conflicting access, no order):",
            self.pairs.len(),
            if self.pairs.len() == 1 { "y" } else { "ies" }
        );
        for pair in &self.pairs {
            let _ = write!(out, "\n  - `{}` vs `{}`", pair.first_name, pair.second_name);
            if pair.whole_world {
                let _ = write!(out, " (exclusive / whole-world access)");
            }
            if !pair.components.is_empty() {
                let _ = write!(out, " components: ");
                for (i, c) in pair.components.iter().enumerate() {
                    let _ = write!(out, "{}#{}", if i == 0 { "" } else { ", " }, c.index());
                }
            }
            if !pair.resources.is_empty() {
                let _ = write!(out, " resources: ");
                for (i, r) in pair.resources.iter().enumerate() {
                    let _ = write!(out, "{}#{}", if i == 0 { "" } else { ", " }, r.index());
                }
            }
        }
        out
    }
}

/// Core detection: given each node's [`Access`], its display name, and the
/// resolved ordering edges `(from, to)`, report every conflicting unordered
/// pair. The graph is assumed acyclic (the caller computes the order first,
/// which panics on a cycle).
pub(crate) fn detect(accesses: &[&Access], names: &[String], edges: &[(usize, usize)]) -> Ambiguities {
    let n = accesses.len();
    debug_assert_eq!(n, names.len());

    // Adjacency list from the edge set.
    let mut adj: Vec<Vec<usize>> = alloc::vec![Vec::new(); n];
    for &(from, to) in edges {
        adj[from].push(to);
    }

    // Transitive reachability per source node via iterative DFS. `reach[s][t]`
    // is true when `t` is ordered after `s` (directly or transitively).
    let mut reach: Vec<Vec<bool>> = alloc::vec![alloc::vec![false; n]; n];
    let mut stack: Vec<usize> = Vec::new();
    for (s, row) in reach.iter_mut().enumerate() {
        stack.clear();
        stack.push(s);
        while let Some(u) = stack.pop() {
            for &v in &adj[u] {
                if !row[v] {
                    row[v] = true;
                    stack.push(v);
                }
            }
        }
    }

    let mut pairs: Vec<Ambiguity> = Vec::new();
    for first in 0..n {
        for second in (first + 1)..n {
            let ordered = reach[first][second] || reach[second][first];
            if ordered {
                continue;
            }
            let a = accesses[first];
            let b = accesses[second];
            if a.is_compatible(b) {
                continue;
            }
            let (components, resources) = conflicting_ids(a, b);
            pairs.push(Ambiguity {
                first,
                second,
                first_name: names[first].clone(),
                second_name: names[second].clone(),
                components,
                resources,
                whole_world: a.writes_everything() || b.writes_everything(),
            });
        }
    }

    Ambiguities { pairs }
}

/// The specific components and resources that make `a` and `b` conflict: an
/// item written by one and read or written by the other. Deduplicated and in a
/// deterministic (first-seen) order.
fn conflicting_ids(a: &Access, b: &Access) -> (Vec<ComponentId>, Vec<ResourceId>) {
    let mut components: Vec<ComponentId> = Vec::new();
    for w in a.writes() {
        if (b.writes().contains(w) || b.reads().contains(w)) && !components.contains(w) {
            components.push(*w);
        }
    }
    for w in b.writes() {
        if a.reads().contains(w) && !components.contains(w) {
            components.push(*w);
        }
    }

    let mut resources: Vec<ResourceId> = Vec::new();
    for w in a.resource_writes() {
        if (b.resource_writes().contains(w) || b.resource_reads().contains(w))
            && !resources.contains(w)
        {
            resources.push(*w);
        }
    }
    for w in b.resource_writes() {
        if a.resource_reads().contains(w) && !resources.contains(w) {
            resources.push(*w);
        }
    }

    (components, resources)
}
