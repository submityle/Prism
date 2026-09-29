//! Aliased transient-heap allocation planning for the GPU frame graph.
//!
//! The scheduler assigns every `Transient` resource a byte offset inside a
//! single transient heap, reusing (aliasing) the same region for resources
//! whose execution-order lifetimes do not overlap. This module turns those raw
//! offsets into a validated, backend-consumable plan:
//!
//! * a [`TransientRegion`] per resource (offset / size / alignment),
//! * the total heap size the backend must allocate,
//! * the *alias groups* (sets of resources that share one heap region), and
//! * the VRAM saved versus giving every pass its own standalone allocation.
//!
//! The plan is backend-agnostic: `wgpu` 30 does not expose placed/aliased heaps,
//! so this stays pure logic. A backend that gains heap aliasing consumes
//! [`TransientAllocation::alias_groups`] to bind several resources to the same
//! memory, relying on [`TransientAllocation::validate_alias_safety`] to prove
//! the reuse is lifetime-safe.

use alloc::collections::BTreeMap;
use core::ops::Range;

use super::{PassDescriptor, PassId, ResourceDescriptor, ResourceId, ResourceLifetime};

/// A byte region within the transient heap assigned to a single resource.
///
/// `offset` is guaranteed to be a multiple of `alignment`, and the region spans
/// `offset..offset + size`. Two resources in the same alias group share an
/// identical `offset` (and therefore overlap in memory), which is only sound
/// because their lifetimes are disjoint.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TransientRegion {
    /// Byte offset of the region from the start of the transient heap.
    pub offset: u64,
    /// Byte size the resource occupies.
    pub size: u64,
    /// Alignment the offset satisfies (always `>= 1`).
    pub alignment: u64,
}

/// A validated, aliased allocation plan for the frame graph's transient heap.
///
/// Built by [`TransientAllocation::plan`] from the resource table, the pass
/// table and the compiled execution order. Every accessor is derived from the
/// same first-fit assignment the compiler already relied on, so the plan can
/// never disagree with `transient_offsets` / `transient_bytes`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TransientAllocation {
    /// Total heap size in bytes (the peak the first-fit assignment reached).
    heap_bytes: u64,
    /// Per-resource region, indexed by `ResourceId`. `None` for non-transient
    /// resources and for transient resources that no pass ever touches.
    regions: Vec<Option<TransientRegion>>,
    /// Per-resource inclusive execution-order lifetime as a half-open range
    /// `first..last + 1`, indexed by `ResourceId`. `None` mirrors `regions`.
    lifetimes: Vec<Option<Range<u32>>>,
    /// Resources sharing each heap offset, ordered by ascending offset and, for
    /// members, by ascending `ResourceId`. A group with more than one member is
    /// a true aliased region; a singleton group owns its offset outright.
    alias_groups: Vec<Vec<ResourceId>>,
    /// Sum of the standalone footprints (`size`) of every planned transient
    /// resource, i.e. the heap that would be needed with no aliasing at all.
    individual_bytes: u64,
}

/// A lifetime overlap discovered inside a single alias group.
///
/// Returned by [`TransientAllocation::validate_alias_safety`]; its presence is a
/// planner bug because aliasing two resources whose lifetimes overlap would let
/// the backend corrupt live data.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AliasOverlap {
    /// Shared heap offset of the offending alias group.
    pub offset: u64,
    /// First resource of the overlapping pair.
    pub first: ResourceId,
    /// Second resource of the overlapping pair.
    pub second: ResourceId,
}

impl TransientAllocation {
    /// Builds the aliased plan from the compiled graph.
    ///
    /// `order` is the topologically sorted execution order; each resource's
    /// lifetime is the inclusive span of execution positions that access it.
    /// Transient resources are assigned by first-fit: a resource reuses an
    /// existing region whenever that region is free (its last user precedes the
    /// new resource's first user), is large enough, and keeps the offset
    /// aligned. Everything else is derived from that assignment.
    ///
    /// # Panics
    ///
    /// Debug builds assert [`Self::validate_alias_safety`] holds, so a planner
    /// regression that aliases overlapping lifetimes trips immediately in tests.
    pub(crate) fn plan(
        resources: &[ResourceDescriptor],
        passes: &[PassDescriptor],
        order: &[PassId],
    ) -> Self {
        let lifetimes = compute_lifetimes(resources, passes, order);

        // First-fit assignment: reuse a free, large-enough, aligned block or
        // grow the heap. Mirrors the golden reuse rule so offsets never drift.
        let mut cursor = 0_u64;
        let mut regions = vec![None; resources.len()];
        let mut individual_bytes = 0_u64;
        // (offset, size, last_user_position) for each live block.
        let mut blocks: Vec<(u64, u64, u32)> = Vec::new();

        let mut pending: Vec<(usize, u32, u32)> = resources
            .iter()
            .enumerate()
            .filter_map(|(index, resource)| {
                if resource.lifetime != ResourceLifetime::Transient {
                    return None;
                }
                lifetimes[index]
                    .as_ref()
                    .map(|life| (index, life.start, life.end - 1))
            })
            .collect();
        pending.sort_by_key(|(index, first, _)| (*first, *index));

        for (index, first, last) in pending {
            let resource = &resources[index];
            let alignment = resource.alignment.max(1);
            individual_bytes += resource.size;
            if let Some(block) = blocks.iter_mut().find(|(offset, size, available_after)| {
                *available_after < first && *size >= resource.size && *offset % alignment == 0
            }) {
                regions[index] = Some(TransientRegion {
                    offset: block.0,
                    size: resource.size,
                    alignment,
                });
                block.2 = last;
            } else {
                cursor = cursor.div_ceil(alignment) * alignment;
                regions[index] = Some(TransientRegion {
                    offset: cursor,
                    size: resource.size,
                    alignment,
                });
                blocks.push((cursor, resource.size, last));
                cursor += resource.size;
            }
        }

        let alias_groups = group_by_offset(&regions);

        let plan = Self {
            heap_bytes: cursor,
            regions,
            lifetimes,
            alias_groups,
            individual_bytes,
        };
        debug_assert!(
            plan.validate_alias_safety().is_ok(),
            "transient alias plan aliased overlapping lifetimes: {:?}",
            plan.validate_alias_safety()
        );
        plan
    }

    /// Total heap size the backend must allocate for all transient resources.
    pub fn heap_bytes(&self) -> u64 {
        self.heap_bytes
    }

    /// The assigned region for `resource`, or `None` when it is not a planned
    /// transient (persistent, imported, or never accessed).
    pub fn region(&self, resource: ResourceId) -> Option<TransientRegion> {
        self.regions
            .get(resource.0 as usize)
            .copied()
            .flatten()
    }

    /// Per-resource regions indexed by `ResourceId`, for bulk consumers.
    pub fn regions(&self) -> &[Option<TransientRegion>] {
        &self.regions
    }

    /// Groups of resources that share a heap offset, ascending by offset.
    ///
    /// A group with more than one member is a true aliased region whose members
    /// are guaranteed disjoint in time by [`Self::validate_alias_safety`].
    pub fn alias_groups(&self) -> &[Vec<ResourceId>] {
        &self.alias_groups
    }

    /// Raw offsets indexed by `ResourceId`, matching the legacy
    /// `transient_offsets` product the topology check consumed.
    pub fn offsets(&self) -> Vec<Option<u64>> {
        self.regions
            .iter()
            .map(|region| region.map(|region| region.offset))
            .collect()
    }

    /// Standalone heap the graph would need with no aliasing (sum of sizes).
    pub fn individual_bytes(&self) -> u64 {
        self.individual_bytes
    }

    /// Bytes saved by aliasing versus one standalone allocation per resource:
    /// `individual_bytes - heap_bytes`. Zero when nothing could be reused.
    pub fn savings_bytes(&self) -> u64 {
        self.individual_bytes.saturating_sub(self.heap_bytes)
    }

    /// Theoretical peak of simultaneously live transient bytes: the maximum,
    /// over every execution position, of the summed sizes of the resources
    /// alive there. This is the lower bound any allocator could reach; the gap
    /// to [`Self::heap_bytes`] is alignment / fragmentation overhead.
    pub fn peak_live_bytes(&self) -> u64 {
        let horizon = self
            .lifetimes
            .iter()
            .flatten()
            .map(|life| life.end)
            .max()
            .unwrap_or(0);
        let mut peak = 0_u64;
        for position in 0..horizon {
            let mut live = 0_u64;
            for (index, life) in self.lifetimes.iter().enumerate() {
                let Some(life) = life else { continue };
                if life.contains(&position)
                    && let Some(region) = self.regions[index]
                {
                    live += region.size;
                }
            }
            peak = peak.max(live);
        }
        peak
    }

    /// Proves the plan is alias-safe: within every alias group, all resource
    /// lifetimes are pairwise disjoint, so the backend can bind them to the same
    /// heap region without one clobbering another's live contents.
    ///
    /// Returns the first [`AliasOverlap`] found, or `Ok(())` when the plan is
    /// sound. A correctly built plan is always `Ok`; a violation signals a
    /// planner bug (see [`Self::plan`]'s debug assertion).
    pub fn validate_alias_safety(&self) -> Result<(), AliasOverlap> {
        for group in &self.alias_groups {
            let offset = self
                .regions
                .get(group[0].0 as usize)
                .copied()
                .flatten()
                .map_or(0, |region| region.offset);
            for (position, &first) in group.iter().enumerate() {
                // A zero-byte region owns no memory and cannot corrupt a peer,
                // so it is never an unsafe alias regardless of lifetime overlap.
                if self.region_bytes(first) == 0 {
                    continue;
                }
                let Some(first_life) = self.lifetime_of(first) else {
                    continue;
                };
                for &second in &group[position + 1..] {
                    if self.region_bytes(second) == 0 {
                        continue;
                    }
                    let Some(second_life) = self.lifetime_of(second) else {
                        continue;
                    };
                    if ranges_overlap(&first_life, &second_life) {
                        return Err(AliasOverlap {
                            offset,
                            first,
                            second,
                        });
                    }
                }
            }
        }
        Ok(())
    }

    fn region_bytes(&self, resource: ResourceId) -> u64 {
        self.regions
            .get(resource.0 as usize)
            .copied()
            .flatten()
            .map_or(0, |region| region.size)
    }

    fn lifetime_of(&self, resource: ResourceId) -> Option<Range<u32>> {
        self.lifetimes
            .get(resource.0 as usize)
            .cloned()
            .flatten()
    }
}

/// Inclusive execution-order lifetimes as half-open ranges (`first..last + 1`),
/// indexed by `ResourceId`. `None` when no pass accesses the resource.
fn compute_lifetimes(
    resources: &[ResourceDescriptor],
    passes: &[PassDescriptor],
    order: &[PassId],
) -> Vec<Option<Range<u32>>> {
    let mut lifetimes = vec![None::<Range<u32>>; resources.len()];
    for (position, &pass_id) in order.iter().enumerate() {
        let position = position as u32;
        for access in &passes[pass_id.0 as usize].accesses {
            let life = &mut lifetimes[access.resource.0 as usize];
            *life = Some(match life {
                Some(existing) => existing.start..position + 1,
                None => position..position + 1,
            });
        }
    }
    lifetimes
}

/// Partitions the assigned resources into alias groups keyed by heap offset,
/// ascending by offset and, within a group, by `ResourceId`.
///
/// Zero-byte regions are excluded: a region that owns no bytes occupies the
/// range `offset..offset` (empty), so it can never overlap another region in
/// memory and aliasing it is a no-op. Topology-only graphs that model resources
/// with `size == 0` therefore pile every such resource onto offset 0 without it
/// being a real alias, and grouping them there would raise a spurious
/// [`AliasOverlap`] between resources whose lifetimes legitimately coincide.
fn group_by_offset(regions: &[Option<TransientRegion>]) -> Vec<Vec<ResourceId>> {
    let mut by_offset: BTreeMap<u64, Vec<ResourceId>> = BTreeMap::new();
    for (index, region) in regions.iter().enumerate() {
        if let Some(region) = region {
            if region.size == 0 {
                continue;
            }
            by_offset
                .entry(region.offset)
                .or_default()
                .push(ResourceId(index as u32));
        }
    }
    by_offset.into_values().collect()
}

/// Half-open range overlap: `a` and `b` share a position iff each starts before
/// the other ends.
fn ranges_overlap(a: &Range<u32>, b: &Range<u32>) -> bool {
    a.start < b.end && b.start < a.end
}

#[cfg(test)]
mod tests {
    use super::*;

    fn region(offset: u64, size: u64, alignment: u64) -> Option<TransientRegion> {
        Some(TransientRegion {
            offset,
            size,
            alignment,
        })
    }

    #[test]
    fn validate_alias_safety_accepts_disjoint_lifetimes_in_a_group() {
        // Two resources aliased to offset 0, lifetimes [0,1] and [2,3]: disjoint.
        let plan = TransientAllocation {
            heap_bytes: 1024,
            regions: vec![region(0, 1024, 256), region(0, 1024, 256)],
            lifetimes: vec![Some(0..2), Some(2..4)],
            alias_groups: vec![vec![ResourceId(0), ResourceId(1)]],
            individual_bytes: 2048,
        };
        assert_eq!(plan.validate_alias_safety(), Ok(()));
    }

    #[test]
    fn validate_alias_safety_flags_overlapping_lifetimes_in_a_group() {
        // Two resources sharing offset 0 whose lifetimes [0,2] and [1,3] overlap
        // at positions 1..2 — an unsound alias the validator must reject.
        let plan = TransientAllocation {
            heap_bytes: 1024,
            regions: vec![region(0, 1024, 256), region(0, 1024, 256)],
            lifetimes: vec![Some(0..3), Some(1..4)],
            alias_groups: vec![vec![ResourceId(0), ResourceId(1)]],
            individual_bytes: 2048,
        };
        assert_eq!(
            plan.validate_alias_safety(),
            Err(AliasOverlap {
                offset: 0,
                first: ResourceId(0),
                second: ResourceId(1),
            })
        );
    }

    #[test]
    fn touching_lifetimes_are_disjoint() {
        // [0,0] then [1,1]: last_a (0) < first_b (1), so half-open 0..1 and 1..2
        // do not overlap — the boundary case the first-fit reuse rule allows.
        assert!(!ranges_overlap(&(0..1), &(1..2)));
        assert!(ranges_overlap(&(0..2), &(1..3)));
    }

    #[test]
    fn zero_sized_transients_never_alias_even_when_lifetimes_overlap() {
        // Topology-only graphs model transients with size 0 (byte budgets are
        // resolved later at runtime). Every such resource lands on offset 0, but
        // an empty region owns no memory and cannot corrupt a peer, so two
        // overlapping zero-byte resources must not be grouped or flagged.
        let regions = vec![region(0, 0, 16), region(0, 0, 16)];
        assert!(group_by_offset(&regions).is_empty());

        let plan = TransientAllocation {
            heap_bytes: 0,
            regions,
            lifetimes: vec![Some(0..5), Some(0..5)],
            alias_groups: group_by_offset(&[region(0, 0, 16), region(0, 0, 16)]),
            individual_bytes: 0,
        };
        assert_eq!(plan.validate_alias_safety(), Ok(()));
    }

    #[test]
    fn peak_live_bytes_tracks_simultaneous_demand() {
        // r0 live [0,2] (1024), r1 live [1,3] (512): overlap at 1..3 => 1536.
        let plan = TransientAllocation {
            heap_bytes: 1536,
            regions: vec![region(0, 1024, 256), region(1024, 512, 256)],
            lifetimes: vec![Some(0..3), Some(1..4)],
            alias_groups: vec![vec![ResourceId(0)], vec![ResourceId(1)]],
            individual_bytes: 1536,
        };
        assert_eq!(plan.peak_live_bytes(), 1536);
        assert_eq!(plan.savings_bytes(), 0);
    }
}
