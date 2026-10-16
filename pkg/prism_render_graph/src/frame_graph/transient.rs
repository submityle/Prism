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
//! the reuse is lifetime-safe. Executable `RenderGraph` plans use
//! [`crate::AliasPlan`] as their authoritative alias product.

#[cfg(test)]
use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::ops::Range;

use super::ResourceId;

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
/// Built by the declaration adapter from the resource table and shared
/// lifetime intervals. Executable `RenderGraph` plans expose the authoritative
/// [`crate::AliasPlan`] instead of duplicating this projection.
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
    /// Total heap size the backend must allocate for all transient resources.
    pub fn heap_bytes(&self) -> u64 {
        self.heap_bytes
    }

    /// The assigned region for `resource`, or `None` when it is not a planned
    /// transient (persistent, imported, or never accessed).
    pub fn region(&self, resource: ResourceId) -> Option<TransientRegion> {
        self.regions.get(resource.0 as usize).copied().flatten()
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

    /// Raw offsets indexed by `ResourceId` for declaration-front-end
    /// diagnostics.
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
    /// planner bug, rather than a recoverable declaration error.
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
        self.lifetimes.get(resource.0 as usize).cloned().flatten()
    }
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
#[cfg(test)]
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
#[path = "../../text/unit/transient.rs"]
mod tests;
