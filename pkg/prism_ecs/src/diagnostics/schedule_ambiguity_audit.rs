//! Schedule-ambiguity blast-radius and contention census (design §23.4 /
//! §16.6).
//!
//! Two systems are *ambiguous* when their access sets conflict — one writes a
//! component or resource the other reads or writes, or one is exclusive — yet
//! the schedule contains no ordering edge that fixes which runs first (design
//! §23.4). [`Schedule::ambiguities`](crate::schedule::Schedule::ambiguities)
//! returns the full set of such pairs as an
//! [`Ambiguities`](crate::schedule::Ambiguities); each
//! [`Ambiguity`](crate::schedule::Ambiguity) names the two systems and the
//! components / resources they contend over (or flags a whole-world exclusive
//! conflict).
//!
//! The raw pair list answers "which two systems clash", but not the two
//! questions that actually drive remediation on a large schedule:
//!
//! * **which system is the hotspot** — a system that appears in many ambiguous
//!   pairs is one well-placed `before`/`after`/`chain` edge away from
//!   collapsing a whole cluster of ambiguities, so ranking systems by how many
//!   pairs they sit in points straight at the highest-leverage fix;
//! * **which data is the magnet** — a component or resource that recurs across
//!   many ambiguous pairs is a shared mutable choke point whose access pattern
//!   (or a dedicated ordering set) is the structural fix, not the individual
//!   pairs.
//!
//! This report folds a slice of [`Ambiguity`] pairs (as returned by
//! [`Ambiguities::pairs`](crate::schedule::Ambiguities::pairs)) into those two
//! rankings plus a rolled-up summary — total pairs, how many are whole-world
//! versus component versus resource conflicts, the number of distinct systems
//! involved, and the worst single-system involvement — without touching the
//! schedule or world. It is purely read-only post-analysis over the detector's
//! own output.
//!
//! Everything is deterministic (design §14): the input pairs are already
//! reported in ascending `(first, second)` node order, and the derived
//! rankings are sorted by node index, then component index, then resource
//! index — independent of input order and internal hashing.

use alloc::string::String;
use alloc::vec::Vec;

use crate::component::ComponentId;
use crate::resource::ResourceId;
use crate::schedule::Ambiguity;
use crate::collections::HashMap;

/// Integer permille (`parts per thousand`) of `num / den`, returning `0` when
/// `den` is zero.
#[inline]
fn permille(num: u64, den: u64) -> u64 {
    (num * 1000).checked_div(den).unwrap_or(0)
}

/// One system's involvement across the ambiguous pairs: how many pairs it
/// appears in (as either side). A high count marks a hotspot where one
/// ordering edge can resolve many ambiguities at once (design §23.4).
#[derive(Clone, Debug)]
pub struct AmbiguousSystemEntry {
    /// Schedule node index of the system.
    pub node: usize,
    /// Name of the system (from [`System::name`](crate::system::System::name)).
    pub name: String,
    /// Number of ambiguous pairs this system participates in.
    pub pair_count: usize,
}

/// One component's recurrence across the ambiguous pairs: how many pairs list
/// it as contended. A high count marks a shared mutable choke point (design
/// §23.4).
#[derive(Clone, Copy, Debug)]
pub struct ContendedComponentEntry {
    /// The contended component.
    pub component: ComponentId,
    /// Number of ambiguous pairs that list this component as contended.
    pub pair_count: usize,
}

/// One resource's recurrence across the ambiguous pairs: how many pairs list it
/// as contended (design §23.4).
#[derive(Clone, Copy, Debug)]
pub struct ContendedResourceEntry {
    /// The contended resource.
    pub resource: ResourceId,
    /// Number of ambiguous pairs that list this resource as contended.
    pub pair_count: usize,
}

/// Read-only census over a schedule's ambiguous pairs: per-system involvement
/// and per-datum contention rankings plus a rolled-up summary (design §23.4 /
/// §16.6).
#[derive(Clone, Debug)]
pub struct ScheduleAmbiguityAudit {
    pair_count: usize,
    whole_world_pair_count: usize,
    component_conflict_pair_count: usize,
    resource_conflict_pair_count: usize,
    max_system_involvement: usize,
    systems: Vec<AmbiguousSystemEntry>,
    components: Vec<ContendedComponentEntry>,
    resources: Vec<ContendedResourceEntry>,
}

impl ScheduleAmbiguityAudit {
    /// Fold a slice of ambiguous `pairs` (as returned by
    /// [`Ambiguities::pairs`](crate::schedule::Ambiguities::pairs)) into the
    /// involvement / contention rankings and summary.
    ///
    /// A system is counted once per pair it appears in; a component or resource
    /// is counted once per pair that lists it (duplicates within a single
    /// pair's list are collapsed). Derived rankings are returned sorted by node
    /// index, then component index, then resource index (design §14).
    pub fn from_pairs(pairs: &[Ambiguity]) -> Self {
        let mut whole_world_pair_count = 0usize;
        let mut component_conflict_pair_count = 0usize;
        let mut resource_conflict_pair_count = 0usize;

        let mut system_hits: HashMap<usize, (String, usize)> = HashMap::default();
        let mut component_hits: HashMap<ComponentId, usize> = HashMap::default();
        let mut resource_hits: HashMap<ResourceId, usize> = HashMap::default();

        for pair in pairs {
            if pair.whole_world {
                whole_world_pair_count += 1;
            }
            if !pair.components.is_empty() {
                component_conflict_pair_count += 1;
            }
            if !pair.resources.is_empty() {
                resource_conflict_pair_count += 1;
            }

            // Each system is counted once per pair (the two sides always differ).
            system_hits
                .entry(pair.first)
                .or_insert_with(|| (pair.first_name.clone(), 0))
                .1 += 1;
            system_hits
                .entry(pair.second)
                .or_insert_with(|| (pair.second_name.clone(), 0))
                .1 += 1;

            // Collapse duplicates within a single pair's lists so each datum is
            // counted at most once per pair.
            let mut seen_components: Vec<ComponentId> = Vec::new();
            for &component in &pair.components {
                if !seen_components.contains(&component) {
                    seen_components.push(component);
                    *component_hits.entry(component).or_insert(0) += 1;
                }
            }
            let mut seen_resources: Vec<ResourceId> = Vec::new();
            for &resource in &pair.resources {
                if !seen_resources.contains(&resource) {
                    seen_resources.push(resource);
                    *resource_hits.entry(resource).or_insert(0) += 1;
                }
            }
        }

        let mut systems: Vec<AmbiguousSystemEntry> = system_hits
            .into_iter()
            .map(|(node, (name, pair_count))| AmbiguousSystemEntry {
                node,
                name,
                pair_count,
            })
            .collect();
        systems.sort_unstable_by_key(|entry| entry.node);

        let mut components: Vec<ContendedComponentEntry> = component_hits
            .into_iter()
            .map(|(component, pair_count)| ContendedComponentEntry {
                component,
                pair_count,
            })
            .collect();
        components.sort_unstable_by_key(|entry| entry.component.index());

        let mut resources: Vec<ContendedResourceEntry> = resource_hits
            .into_iter()
            .map(|(resource, pair_count)| ContendedResourceEntry {
                resource,
                pair_count,
            })
            .collect();
        resources.sort_unstable_by_key(|entry| entry.resource.index());

        let max_system_involvement = systems.iter().map(|entry| entry.pair_count).max().unwrap_or(0);

        Self {
            pair_count: pairs.len(),
            whole_world_pair_count,
            component_conflict_pair_count,
            resource_conflict_pair_count,
            max_system_involvement,
            systems,
            components,
            resources,
        }
    }

    /// Total number of ambiguous pairs analysed.
    #[inline]
    pub fn pair_count(&self) -> usize {
        self.pair_count
    }

    /// Number of pairs whose conflict is a whole-world (exclusive) borrow.
    #[inline]
    pub fn whole_world_pair_count(&self) -> usize {
        self.whole_world_pair_count
    }

    /// Number of pairs that contend over at least one component.
    #[inline]
    pub fn component_conflict_pair_count(&self) -> usize {
        self.component_conflict_pair_count
    }

    /// Number of pairs that contend over at least one resource.
    #[inline]
    pub fn resource_conflict_pair_count(&self) -> usize {
        self.resource_conflict_pair_count
    }

    /// Number of distinct systems that appear in at least one ambiguous pair.
    #[inline]
    pub fn involved_system_count(&self) -> usize {
        self.systems.len()
    }

    /// Number of distinct components contended across the ambiguous pairs.
    #[inline]
    pub fn contended_component_count(&self) -> usize {
        self.components.len()
    }

    /// Number of distinct resources contended across the ambiguous pairs.
    #[inline]
    pub fn contended_resource_count(&self) -> usize {
        self.resources.len()
    }

    /// Highest number of ambiguous pairs any single system participates in
    /// (`0` when there are no ambiguities) — the top hotspot's involvement.
    #[inline]
    pub fn max_system_involvement(&self) -> usize {
        self.max_system_involvement
    }

    /// Whether the schedule has no ambiguities at all.
    #[inline]
    pub fn is_clean(&self) -> bool {
        self.pair_count == 0
    }

    /// Whether any ambiguity is a whole-world (exclusive) conflict.
    #[inline]
    pub fn has_whole_world(&self) -> bool {
        self.whole_world_pair_count > 0
    }

    /// Permille (parts per thousand) of ambiguous pairs that are whole-world
    /// (exclusive) conflicts, `0` when there are no pairs.
    #[inline]
    pub fn whole_world_permille(&self) -> u64 {
        permille(self.whole_world_pair_count as u64, self.pair_count as u64)
    }

    /// Per-system involvement entries, sorted by ascending node index.
    #[inline]
    pub fn systems(&self) -> &[AmbiguousSystemEntry] {
        &self.systems
    }

    /// Per-component contention entries, sorted by ascending component index.
    #[inline]
    pub fn components(&self) -> &[ContendedComponentEntry] {
        &self.components
    }

    /// Per-resource contention entries, sorted by ascending resource index.
    #[inline]
    pub fn resources(&self) -> &[ContendedResourceEntry] {
        &self.resources
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn pair(
        first: usize,
        second: usize,
        components: Vec<ComponentId>,
        resources: Vec<ResourceId>,
        whole_world: bool,
    ) -> Ambiguity {
        Ambiguity {
            first,
            second,
            first_name: String::from("sys"),
            second_name: String::from("sys"),
            components,
            resources,
            whole_world,
        }
    }

    #[test]
    fn empty_is_clean() {
        let audit = ScheduleAmbiguityAudit::from_pairs(&[]);
        assert!(audit.is_clean());
        assert_eq!(audit.pair_count(), 0);
        assert_eq!(audit.involved_system_count(), 0);
        assert_eq!(audit.contended_component_count(), 0);
        assert_eq!(audit.contended_resource_count(), 0);
        assert_eq!(audit.max_system_involvement(), 0);
        assert_eq!(audit.whole_world_permille(), 0);
        assert!(!audit.has_whole_world());
        assert!(audit.systems().is_empty());
    }

    #[test]
    fn single_pair_component_conflict() {
        let pairs = vec![pair(0, 1, vec![ComponentId::new(7)], vec![], false)];
        let audit = ScheduleAmbiguityAudit::from_pairs(&pairs);
        assert!(!audit.is_clean());
        assert_eq!(audit.pair_count(), 1);
        assert_eq!(audit.component_conflict_pair_count(), 1);
        assert_eq!(audit.resource_conflict_pair_count(), 0);
        assert_eq!(audit.whole_world_pair_count(), 0);
        assert_eq!(audit.involved_system_count(), 2);
        assert_eq!(audit.contended_component_count(), 1);
        assert_eq!(audit.max_system_involvement(), 1);
        assert_eq!(audit.components()[0].component, ComponentId::new(7));
        assert_eq!(audit.components()[0].pair_count, 1);
    }

    #[test]
    fn whole_world_pair_counted() {
        let pairs = vec![pair(0, 1, vec![], vec![], true)];
        let audit = ScheduleAmbiguityAudit::from_pairs(&pairs);
        assert_eq!(audit.whole_world_pair_count(), 1);
        assert!(audit.has_whole_world());
        assert_eq!(audit.whole_world_permille(), 1000);
        assert_eq!(audit.component_conflict_pair_count(), 0);
        assert_eq!(audit.resource_conflict_pair_count(), 0);
    }

    #[test]
    fn resource_conflict_pair_counted() {
        let pairs = vec![pair(2, 5, vec![], vec![ResourceId::new(3)], false)];
        let audit = ScheduleAmbiguityAudit::from_pairs(&pairs);
        assert_eq!(audit.resource_conflict_pair_count(), 1);
        assert_eq!(audit.contended_resource_count(), 1);
        assert_eq!(audit.resources()[0].resource, ResourceId::new(3));
        assert_eq!(audit.resources()[0].pair_count, 1);
    }

    #[test]
    fn system_involvement_hotspot() {
        // Node 0 appears in three pairs; nodes 1, 2, 3 appear once each.
        let pairs = vec![
            pair(0, 1, vec![ComponentId::new(1)], vec![], false),
            pair(0, 2, vec![ComponentId::new(1)], vec![], false),
            pair(0, 3, vec![ComponentId::new(1)], vec![], false),
        ];
        let audit = ScheduleAmbiguityAudit::from_pairs(&pairs);
        assert_eq!(audit.involved_system_count(), 4);
        assert_eq!(audit.max_system_involvement(), 3);
        let hotspot = audit.systems().iter().find(|e| e.node == 0).unwrap();
        assert_eq!(hotspot.pair_count, 3);
        for other in audit.systems().iter().filter(|e| e.node != 0) {
            assert_eq!(other.pair_count, 1);
        }
    }

    #[test]
    fn contended_component_ranked() {
        // Component 9 recurs in two pairs; component 4 in one.
        let pairs = vec![
            pair(0, 1, vec![ComponentId::new(9)], vec![], false),
            pair(2, 3, vec![ComponentId::new(9), ComponentId::new(4)], vec![], false),
        ];
        let audit = ScheduleAmbiguityAudit::from_pairs(&pairs);
        assert_eq!(audit.contended_component_count(), 2);
        let nine = audit.components().iter().find(|e| e.component == ComponentId::new(9)).unwrap();
        let four = audit.components().iter().find(|e| e.component == ComponentId::new(4)).unwrap();
        assert_eq!(nine.pair_count, 2);
        assert_eq!(four.pair_count, 1);
    }

    #[test]
    fn entries_sorted() {
        let pairs = vec![
            pair(5, 2, vec![ComponentId::new(8), ComponentId::new(1)], vec![ResourceId::new(6), ResourceId::new(2)], false),
        ];
        let audit = ScheduleAmbiguityAudit::from_pairs(&pairs);
        let sys_nodes: Vec<usize> = audit.systems().iter().map(|e| e.node).collect();
        assert_eq!(sys_nodes, vec![2, 5]);
        let comp_idx: Vec<u32> = audit.components().iter().map(|e| e.component.index()).collect();
        assert_eq!(comp_idx, vec![1, 8]);
        let res_idx: Vec<u32> = audit.resources().iter().map(|e| e.resource.index()).collect();
        assert_eq!(res_idx, vec![2, 6]);
    }

    #[test]
    fn duplicate_component_within_pair_collapsed() {
        // A single pair that lists the same component twice counts it once.
        let pairs = vec![pair(0, 1, vec![ComponentId::new(3), ComponentId::new(3)], vec![], false)];
        let audit = ScheduleAmbiguityAudit::from_pairs(&pairs);
        assert_eq!(audit.contended_component_count(), 1);
        assert_eq!(audit.components()[0].pair_count, 1);
    }

    #[test]
    fn mixed_rollup() {
        let pairs = vec![
            pair(0, 1, vec![ComponentId::new(2)], vec![], false),
            pair(0, 2, vec![], vec![ResourceId::new(5)], false),
            pair(3, 4, vec![], vec![], true),
        ];
        let audit = ScheduleAmbiguityAudit::from_pairs(&pairs);
        assert_eq!(audit.pair_count(), 3);
        assert_eq!(audit.component_conflict_pair_count(), 1);
        assert_eq!(audit.resource_conflict_pair_count(), 1);
        assert_eq!(audit.whole_world_pair_count(), 1);
        assert_eq!(audit.involved_system_count(), 5);
        assert_eq!(audit.max_system_involvement(), 2); // node 0 in two pairs
        assert_eq!(audit.whole_world_permille(), 333);
    }
}
