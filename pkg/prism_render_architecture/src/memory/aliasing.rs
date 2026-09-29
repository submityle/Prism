//! Memory aliasing: sharing one physical range between resources whose
//! lifetimes never overlap.
//!
//! A frame graph produces many transient resources, but only a fraction are
//! alive at any instant. Two resources whose live intervals are disjoint can be
//! backed by the same bytes ("aliased"), which is how renderers fit a large
//! working set into a small pool. Deciding the assignment is exactly interval
//! graph colouring: each colour is one physical region, and two intervals sharing
//! a colour must not overlap.
//!
//! This module computes that assignment deterministically. Lifetimes are given
//! as inclusive `[first_use, last_use]` timeline points (frame graph pass
//! indices, timeline values, ...). The greedy sweep below processes intervals in
//! start order and is optimal for interval graphs: it uses exactly the maximum
//! number of simultaneously-live resources, which is the provable lower bound.
//!
//! The plan is pure `CPU` data; issuing the aliasing memory barriers the `GPU`
//! needs when a region changes owner is *pending the GPU backend*.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::fmt;

/// One resource's memory requirement and inclusive live interval.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResourceLifetime {
    /// Caller-chosen identifier, unique within a plan.
    pub id: u64,
    /// Bytes the resource needs while alive.
    pub size: u64,
    /// First timeline point at which the resource is live (inclusive).
    pub first_use: u64,
    /// Last timeline point at which the resource is live (inclusive).
    pub last_use: u64,
}

impl ResourceLifetime {
    /// Whether two lifetimes are simultaneously live and therefore cannot alias.
    ///
    /// Intervals are inclusive, so touching endpoints (`a.last == b.first`) count
    /// as overlapping: both resources are live on that timeline point.
    #[must_use]
    pub const fn overlaps(&self, other: &Self) -> bool {
        self.first_use <= other.last_use && other.first_use <= self.last_use
    }
}

/// Reason [`plan_aliasing`] rejected its input.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AliasError {
    /// A resource requested zero bytes.
    ZeroSize(u64),
    /// A lifetime had `first_use > last_use`.
    InvalidLifetime {
        /// Offending resource id.
        id: u64,
        /// The bad start point.
        first_use: u64,
        /// The bad end point.
        last_use: u64,
    },
    /// The requested placement alignment was not a non-zero power of two.
    InvalidAlignment(u64),
    /// Two resources shared an id.
    DuplicateId(u64),
}

impl fmt::Display for AliasError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroSize(id) => write!(formatter, "resource {id} requested zero bytes"),
            Self::InvalidLifetime {
                id,
                first_use,
                last_use,
            } => write!(
                formatter,
                "resource {id} has an inverted lifetime [{first_use}, {last_use}]"
            ),
            Self::InvalidAlignment(alignment) => {
                write!(formatter, "alignment {alignment} is not a power of two")
            }
            Self::DuplicateId(id) => write!(formatter, "resource id {id} appears more than once"),
        }
    }
}

impl std::error::Error for AliasError {}

/// One physical region shared by a set of non-overlapping resources.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AliasGroup {
    /// Offset of the region within the aliasing pool.
    pub offset: u64,
    /// Size of the region: the largest member's size.
    pub size: u64,
    /// Resource ids assigned to this region, in assignment order.
    pub members: Vec<u64>,
}

/// The computed aliasing assignment for a set of resources.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AliasPlan {
    groups: Vec<AliasGroup>,
    assignment: BTreeMap<u64, usize>,
    total_bytes: u64,
}

impl AliasPlan {
    /// The shared regions, ordered by offset.
    #[must_use]
    pub fn groups(&self) -> &[AliasGroup] {
        &self.groups
    }

    /// Number of physical regions the plan uses (the "colour" count).
    #[must_use]
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    /// Total bytes the aliasing pool must reserve for this plan.
    #[must_use]
    pub const fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// Index into [`groups`](Self::groups) that a resource was assigned to.
    #[must_use]
    pub fn group_index(&self, id: u64) -> Option<usize> {
        self.assignment.get(&id).copied()
    }

    /// Byte offset a resource was placed at, or `None` if it is not in the plan.
    #[must_use]
    pub fn offset_of(&self, id: u64) -> Option<u64> {
        self.group_index(id).map(|index| self.groups[index].offset)
    }
}

/// Working record for a colour while the greedy sweep runs.
struct OpenGroup {
    /// Largest `last_use` currently assigned; the group is reusable strictly
    /// after this point.
    free_after: u64,
    /// Largest member size so far.
    size: u64,
    /// Members assigned in order.
    members: Vec<u64>,
}

/// Computes an aliasing assignment for `resources`, packing regions end to end
/// with each region start rounded up to `alignment`.
///
/// The result uses the minimum possible number of regions for the given
/// lifetimes. Returns an [`AliasError`] if any input is malformed; an empty
/// input yields an empty plan.
pub fn plan_aliasing(
    resources: &[ResourceLifetime],
    alignment: u64,
) -> Result<AliasPlan, AliasError> {
    if !alignment.is_power_of_two() {
        return Err(AliasError::InvalidAlignment(alignment));
    }

    let mut seen: BTreeMap<u64, ()> = BTreeMap::new();
    for resource in resources {
        if resource.size == 0 {
            return Err(AliasError::ZeroSize(resource.id));
        }
        if resource.first_use > resource.last_use {
            return Err(AliasError::InvalidLifetime {
                id: resource.id,
                first_use: resource.first_use,
                last_use: resource.last_use,
            });
        }
        if seen.insert(resource.id, ()).is_some() {
            return Err(AliasError::DuplicateId(resource.id));
        }
    }

    // Deterministic sweep order: by start, then end, then id.
    let mut order: Vec<usize> = (0..resources.len()).collect();
    order.sort_by(|&lhs, &rhs| {
        let a = &resources[lhs];
        let b = &resources[rhs];
        a.first_use
            .cmp(&b.first_use)
            .then(a.last_use.cmp(&b.last_use))
            .then(a.id.cmp(&b.id))
    });

    let mut open: Vec<OpenGroup> = Vec::new();
    let mut group_of: Vec<usize> = alloc::vec![0; resources.len()];

    for &index in &order {
        let resource = &resources[index];
        // First colour whose most recent member ends strictly before this start.
        let slot = open
            .iter()
            .position(|group| group.free_after < resource.first_use);
        match slot {
            Some(group_index) => {
                let group = &mut open[group_index];
                group.free_after = group.free_after.max(resource.last_use);
                group.size = group.size.max(resource.size);
                group.members.push(resource.id);
                group_of[index] = group_index;
            }
            None => {
                group_of[index] = open.len();
                open.push(OpenGroup {
                    free_after: resource.last_use,
                    size: resource.size,
                    members: alloc::vec![resource.id],
                });
            }
        }
    }

    // Lay the colours out end to end, aligning each region start.
    let mut offset = 0u64;
    let mut groups = Vec::with_capacity(open.len());
    for group in open {
        offset = align_up(offset, alignment).expect("aligned pool offset fits in u64");
        groups.push(AliasGroup {
            offset,
            size: group.size,
            members: group.members,
        });
        offset += group.size;
    }

    let mut assignment = BTreeMap::new();
    for (index, resource) in resources.iter().enumerate() {
        assignment.insert(resource.id, group_of[index]);
    }

    Ok(AliasPlan {
        groups,
        assignment,
        total_bytes: offset,
    })
}

/// Rounds `value` up to a power-of-two `align`, returning `None` on overflow.
fn align_up(value: u64, align: u64) -> Option<u64> {
    debug_assert!(align.is_power_of_two(), "alignment must be a power of two");
    let mask = align - 1;
    value.checked_add(mask).map(|rounded| rounded & !mask)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lt(id: u64, size: u64, first: u64, last: u64) -> ResourceLifetime {
        ResourceLifetime {
            id,
            size,
            first_use: first,
            last_use: last,
        }
    }

    #[test]
    fn disjoint_lifetimes_share_one_region() {
        // Three resources live in sequence: [0,1], [2,3], [4,5].
        let resources = [lt(1, 100, 0, 1), lt(2, 200, 2, 3), lt(3, 50, 4, 5)];
        let plan = plan_aliasing(&resources, 1).unwrap();
        assert_eq!(plan.group_count(), 1);
        // Region sized to the largest member.
        assert_eq!(plan.groups()[0].size, 200);
        assert_eq!(plan.total_bytes(), 200);
        // All three land at the same offset.
        assert_eq!(plan.offset_of(1), Some(0));
        assert_eq!(plan.offset_of(2), Some(0));
        assert_eq!(plan.offset_of(3), Some(0));
    }

    #[test]
    fn fully_overlapping_lifetimes_never_alias() {
        // All three live across the whole window.
        let resources = [lt(1, 100, 0, 9), lt(2, 100, 0, 9), lt(3, 100, 0, 9)];
        let plan = plan_aliasing(&resources, 1).unwrap();
        assert_eq!(plan.group_count(), 3);
        assert_eq!(plan.total_bytes(), 300);
        // Distinct offsets.
        let offsets = [
            plan.offset_of(1).unwrap(),
            plan.offset_of(2).unwrap(),
            plan.offset_of(3).unwrap(),
        ];
        assert_eq!(offsets, [0, 100, 200]);
    }

    #[test]
    fn touching_endpoints_count_as_overlap() {
        // [0,2] and [2,4] share point 2, so they must not alias.
        let resources = [lt(1, 64, 0, 2), lt(2, 64, 2, 4)];
        let plan = plan_aliasing(&resources, 1).unwrap();
        assert_eq!(plan.group_count(), 2);
    }

    #[test]
    fn group_count_equals_peak_concurrency() {
        // Peak of two simultaneously-live resources at t=2..3.
        let resources = [
            lt(1, 10, 0, 3),
            lt(2, 10, 2, 5),
            lt(3, 10, 4, 7),
            lt(4, 10, 6, 9),
        ];
        let plan = plan_aliasing(&resources, 1).unwrap();
        assert_eq!(plan.group_count(), 2);
    }

    #[test]
    fn placement_offsets_respect_alignment() {
        // Two overlapping resources force two aligned regions.
        let resources = [lt(1, 100, 0, 9), lt(2, 100, 0, 9)];
        let plan = plan_aliasing(&resources, 256).unwrap();
        assert_eq!(plan.groups()[0].offset, 0);
        // Second region rounded up from 100 to the next 256 boundary.
        assert_eq!(plan.groups()[1].offset, 256);
        assert_eq!(plan.total_bytes(), 356);
    }

    #[test]
    fn members_within_a_group_are_pairwise_disjoint() {
        let resources = [
            lt(1, 10, 0, 1),
            lt(2, 10, 0, 1),
            lt(3, 10, 2, 3),
            lt(4, 10, 2, 3),
        ];
        let plan = plan_aliasing(&resources, 1).unwrap();
        // Verify the core aliasing invariant directly.
        for group in plan.groups() {
            for (i, &lhs) in group.members.iter().enumerate() {
                for &rhs in &group.members[i + 1..] {
                    let a = resources.iter().find(|r| r.id == lhs).unwrap();
                    let b = resources.iter().find(|r| r.id == rhs).unwrap();
                    assert!(!a.overlaps(b), "aliased members {lhs} and {rhs} overlap");
                }
            }
        }
    }

    #[test]
    fn rejects_malformed_input() {
        assert_eq!(
            plan_aliasing(&[lt(1, 0, 0, 1)], 1),
            Err(AliasError::ZeroSize(1))
        );
        assert_eq!(
            plan_aliasing(&[lt(7, 10, 5, 2)], 1),
            Err(AliasError::InvalidLifetime {
                id: 7,
                first_use: 5,
                last_use: 2
            })
        );
        assert_eq!(
            plan_aliasing(&[lt(1, 10, 0, 1), lt(1, 10, 2, 3)], 1),
            Err(AliasError::DuplicateId(1))
        );
        assert_eq!(
            plan_aliasing(&[lt(1, 10, 0, 1)], 3),
            Err(AliasError::InvalidAlignment(3))
        );
    }

    #[test]
    fn empty_input_yields_empty_plan() {
        let plan = plan_aliasing(&[], 16).unwrap();
        assert_eq!(plan.group_count(), 0);
        assert_eq!(plan.total_bytes(), 0);
    }

    #[test]
    fn plan_is_deterministic_regardless_of_input_order() {
        let ordered = [lt(1, 10, 0, 1), lt(2, 20, 2, 3), lt(3, 30, 4, 5)];
        let shuffled = [lt(3, 30, 4, 5), lt(1, 10, 0, 1), lt(2, 20, 2, 3)];
        let a = plan_aliasing(&ordered, 8).unwrap();
        let b = plan_aliasing(&shuffled, 8).unwrap();
        // Same grouping and placement independent of listing order.
        assert_eq!(a.group_count(), b.group_count());
        assert_eq!(a.total_bytes(), b.total_bytes());
        assert_eq!(a.offset_of(1), b.offset_of(1));
        assert_eq!(a.offset_of(2), b.offset_of(2));
        assert_eq!(a.offset_of(3), b.offset_of(3));
    }
}
