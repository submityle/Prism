//! Frame-level GPU memory budget: turns the deduplicated logical allocation set
//! ([`frame_resource_map`](super::frame_resource_map)) into a concrete per-frame
//! byte plan the render graph sizes its hair storage pool from.
//!
//! [`frame_resource_map`](super::frame_resource_map) established *which* physical
//! buffer each per-frame `(pass, binding)` view aliases; this module answers
//! *how many bytes* each of those distinct allocations costs and totals them by
//! [`HairAllocationResidency`]. Because it walks the deduplicated allocation set
//! (guide positions counted once, not once per aliasing pass) it computes the
//! true working-set footprint, not the inflated sum of every pass's binding
//! table.
//!
//! Per-allocation sizes are never re-derived here: each allocation's byte size
//! is read from the same authoritative binding layout the single-pass ABI
//! publishes ([`pass_bindings`](super::pass_layout::pass_bindings) /
//! [`optional_pass_bindings`](super::optional_pass_layout::optional_pass_bindings)),
//! so strides and element counts stay in lock-step with the buffer contracts.
//! Every aliasing view of one allocation reports the identical size (all guide
//! position views are `guide_particles * 16` bytes), so first-seen deduplication
//! is size-stable.
//!
//! This is the fuller companion to
//! [`async_pipeline`](super::async_pipeline)'s handoff residency: that models
//! only the frame-replicated handoff state (persistent guide + resolve output)
//! versus the shared input, whereas this covers the complete working set,
//! including per-frame scratch, split three ways so the graph can decide what to
//! ring-buffer, what to reuse, and what to upload once.
//!
//! Everything is deterministic, pure-integer, and never panics.

use alloc::vec::Vec;

use crate::hair::frame_resource_map::{
    allocation_of, HairAllocationResidency, HairFrameAllocation,
};
use crate::hair::frame_schedule::{per_frame_schedule, HairFrameConfig, HairScheduledPass};
use crate::hair::gpu_dispatch::HairGpuCounts;
use crate::hair::optional_pass_layout::{optional_pass_bindings, HairOptionalExtents};
use crate::hair::pass_layout::{pass_bindings, HairBindingLayout, HairGpuExtents};

/// The sized budget of one distinct per-frame allocation: its logical identity,
/// its residency class, and the byte size to reserve for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HairAllocationBudget {
    /// The deduplicated logical allocation this row sizes.
    pub allocation: HairFrameAllocation,
    /// How it must be backed across the frame timeline.
    pub residency: HairAllocationResidency,
    /// Bytes to reserve, from the authoritative single-pass binding layout
    /// (clamped up to one element for an empty groom).
    pub byte_size: usize,
}

/// Resolves the sized binding layout for one scheduled pass into `scratch`,
/// dispatching to the fixed-spine or optional binding contract.
fn pass_layouts(
    pass: HairScheduledPass,
    counts: &HairGpuCounts,
    extents: HairGpuExtents,
    optional_extents: HairOptionalExtents,
    scratch: &mut Vec<HairBindingLayout>,
) {
    match pass {
        HairScheduledPass::Main(main) => pass_bindings(main, counts, extents, scratch),
        HairScheduledPass::Optional(optional) => {
            optional_pass_bindings(optional, counts, optional_extents, scratch);
        }
    }
}

/// Walks the per-frame schedule for `config` and collects the deduplicated,
/// sized allocation budget, appending to `out` (cleared first).
///
/// For every `(pass, binding)` in schedule order the binding's authoritative
/// byte size is joined with its logical allocation
/// ([`allocation_of`]); the first appearance of each distinct allocation is
/// kept, so aliased views (guide positions across many passes) collapse to a
/// single sized row. Bindings that map to no per-frame allocation are skipped.
/// Deterministic; never panics.
pub fn plan_frame_budget(
    config: HairFrameConfig,
    counts: &HairGpuCounts,
    extents: HairGpuExtents,
    optional_extents: HairOptionalExtents,
    out: &mut Vec<HairAllocationBudget>,
) {
    out.clear();
    let mut schedule = Vec::new();
    per_frame_schedule(config, &mut schedule);
    let mut layouts = Vec::new();
    for pass in schedule {
        pass_layouts(pass, counts, extents, optional_extents, &mut layouts);
        for layout in &layouts {
            let Some(allocation) = allocation_of(pass, layout.binding) else {
                continue;
            };
            if out.iter().any(|row| row.allocation == allocation) {
                continue;
            }
            out.push(HairAllocationBudget {
                allocation,
                residency: allocation.residency(),
                byte_size: layout.byte_size,
            });
        }
    }
}

/// Sums the byte sizes of every allocation in `budget` whose residency matches
/// `residency`. Deterministic; never panics.
#[must_use]
pub fn residency_bytes(
    budget: &[HairAllocationBudget],
    residency: HairAllocationResidency,
) -> usize {
    budget
        .iter()
        .filter(|row| row.residency == residency)
        .map(|row| row.byte_size)
        .sum()
}

/// The per-frame hair storage footprint split by residency class, from which the
/// render graph derives the resident pool size at any pipeline depth.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct HairFrameMemoryPlan {
    /// Bytes of persistent state for one frame (guide positions / previous
    /// positions / render output) — ring-buffered per in-flight frame.
    pub persistent_bytes: usize,
    /// Bytes of transient per-frame scratch (root frames, `VBD` targets,
    /// self-collision corrections, `LOD` keep mask) — also distinct per in-flight
    /// frame since overlapping frames must not share it.
    pub scratch_bytes: usize,
    /// Bytes of read-only uploaded input — one shared copy, independent of the
    /// pipeline depth.
    pub upload_bytes: usize,
}

impl HairFrameMemoryPlan {
    /// The working set one in-flight frame owns exclusively: persistent state
    /// plus scratch. Uploaded input is excluded because it is shared.
    #[must_use]
    pub fn per_frame_working_bytes(&self) -> usize {
        self.persistent_bytes + self.scratch_bytes
    }

    /// The full resident hair-storage cost at pipeline depth `frames_in_flight`:
    /// the per-frame working set replicated across the ring plus the single
    /// shared upload allocation. A depth of zero is treated as one.
    #[must_use]
    pub fn resident_bytes(&self, frames_in_flight: u32) -> usize {
        let depth = frames_in_flight.max(1) as usize;
        self.per_frame_working_bytes() * depth + self.upload_bytes
    }
}

/// Plans the per-frame hair memory footprint for `config`, summing the
/// deduplicated allocation budget into its three residency classes.
/// Deterministic; never panics.
#[must_use]
pub fn plan_frame_memory(
    config: HairFrameConfig,
    counts: &HairGpuCounts,
    extents: HairGpuExtents,
    optional_extents: HairOptionalExtents,
) -> HairFrameMemoryPlan {
    let mut budget = Vec::new();
    plan_frame_budget(config, counts, extents, optional_extents, &mut budget);
    HairFrameMemoryPlan {
        persistent_bytes: residency_bytes(&budget, HairAllocationResidency::Persistent),
        scratch_bytes: residency_bytes(&budget, HairAllocationResidency::Scratch),
        upload_bytes: residency_bytes(&budget, HairAllocationResidency::Upload),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hair::gpu_dispatch::HairComputePass;
    use crate::hair::optional_pass_dispatch::HairOptionalPass;
    use crate::hair::sim_pass_buffers::HairSimPassExtent;
    use crate::hair::solver::HairSolverKind;

    fn sample_counts() -> HairGpuCounts {
        HairGpuCounts {
            roots: 100,
            guide_strands: 100,
            guide_particles: 3200,
            render_strands: 50_000,
            light_texels: 4096,
        }
    }

    fn sample_extents() -> HairGpuExtents {
        HairGpuExtents {
            collider_count: 8,
            render_points: 200_000,
            sim_pass: HairSimPassExtent {
                scalp_vertex_count: 500,
                scalp_index_count: 1500,
                sdf_primitive_count: 12,
            },
            ..HairGpuExtents::default()
        }
    }

    fn sample_optional_extents() -> HairOptionalExtents {
        HairOptionalExtents {
            collider_count: 8,
            cell_count: 4096,
        }
    }

    #[test]
    fn guide_positions_is_sized_once_and_aliased_views_agree() {
        let counts = sample_counts();
        let extents = sample_extents();
        let optional = sample_optional_extents();
        let mut budget = Vec::new();
        plan_frame_budget(
            HairFrameConfig::default(),
            &counts,
            extents,
            optional,
            &mut budget,
        );
        let guide_rows: Vec<&HairAllocationBudget> = budget
            .iter()
            .filter(|row| row.allocation == HairFrameAllocation::GuidePositions)
            .collect();
        assert_eq!(guide_rows.len(), 1);
        // Guide positions is one vec4 per guide particle.
        assert_eq!(guide_rows[0].byte_size, 3200 * 16);

        // Independently sizing every aliasing view yields the identical bytes,
        // so first-seen dedup is size-stable.
        let mut wind = Vec::new();
        pass_bindings(HairComputePass::Wind, &counts, extents, &mut wind);
        let mut interp = Vec::new();
        pass_bindings(HairComputePass::Interpolate, &counts, extents, &mut interp);
        assert_eq!(wind[0].byte_size, interp[0].byte_size);
        assert_eq!(wind[0].byte_size, guide_rows[0].byte_size);
    }

    #[test]
    fn default_plan_matches_manual_residency_sums() {
        let counts = sample_counts();
        let extents = sample_extents();
        let optional = sample_optional_extents();
        let mut budget = Vec::new();
        plan_frame_budget(
            HairFrameConfig::default(),
            &counts,
            extents,
            optional,
            &mut budget,
        );

        // Persistent = guide positions + prev positions + render out points.
        let guide = 3200 * 16;
        let prev = 3200 * 16;
        let out_points = 200_000 * 16;
        let plan = plan_frame_memory(HairFrameConfig::default(), &counts, extents, optional);
        assert_eq!(plan.persistent_bytes, guide + prev + out_points);
        assert_eq!(
            plan.persistent_bytes,
            residency_bytes(&budget, HairAllocationResidency::Persistent)
        );
        // Every budgeted byte falls into exactly one residency class.
        let total: usize = budget.iter().map(|row| row.byte_size).sum();
        assert_eq!(
            total,
            plan.persistent_bytes + plan.scratch_bytes + plan.upload_bytes
        );
    }

    #[test]
    fn resident_bytes_rings_the_working_set_and_shares_upload() {
        let counts = sample_counts();
        let extents = sample_extents();
        let optional = sample_optional_extents();
        let plan = plan_frame_memory(HairFrameConfig::default(), &counts, extents, optional);

        let working = plan.per_frame_working_bytes();
        assert_eq!(working, plan.persistent_bytes + plan.scratch_bytes);
        // Depth 0 clamps to 1.
        assert_eq!(plan.resident_bytes(0), working + plan.upload_bytes);
        assert_eq!(plan.resident_bytes(1), working + plan.upload_bytes);
        assert_eq!(plan.resident_bytes(2), 2 * working + plan.upload_bytes);
        // The ring grows the working set but never the shared upload copy.
        let delta = plan.resident_bytes(3) - plan.resident_bytes(2);
        assert_eq!(delta, working);
    }

    #[test]
    fn vbd_config_budgets_scratch_targets() {
        let counts = sample_counts();
        let extents = sample_extents();
        let optional = sample_optional_extents();
        let config = HairFrameConfig {
            solver: HairSolverKind::Vbd,
            self_collision: false,
        };
        let mut budget = Vec::new();
        plan_frame_budget(config, &counts, extents, optional, &mut budget);
        let targets = budget
            .iter()
            .find(|row| row.allocation == HairFrameAllocation::VbdTargets)
            .expect("VBD config must budget the inertial-target scratch");
        assert_eq!(targets.residency, HairAllocationResidency::Scratch);
        // Targets is one vec4 per guide particle.
        assert_eq!(targets.byte_size, 3200 * 16);
    }

    #[test]
    fn self_collision_config_budgets_the_csr_buffers() {
        let counts = sample_counts();
        let extents = sample_extents();
        let optional = sample_optional_extents();
        let config = HairFrameConfig {
            solver: HairSolverKind::Xpbd,
            self_collision: true,
        };
        let mut budget = Vec::new();
        plan_frame_budget(config, &counts, extents, optional, &mut budget);
        for allocation in [
            HairFrameAllocation::SelfCollisionCorrections,
            HairFrameAllocation::SelfCollisionCellKeys,
            HairFrameAllocation::SelfCollisionCellStarts,
            HairFrameAllocation::SelfCollisionCellIndices,
        ] {
            assert!(
                budget.iter().any(|row| row.allocation == allocation),
                "{allocation:?} must be budgeted when self-collision is enabled",
            );
        }
        // Enabling self-collision only grows the budget (no rows lost).
        let mut base = Vec::new();
        plan_frame_budget(
            HairFrameConfig::default(),
            &counts,
            extents,
            optional,
            &mut base,
        );
        assert!(budget.len() > base.len());
    }

    #[test]
    fn empty_groom_never_panics_and_reserves_one_element() {
        let counts = HairGpuCounts {
            roots: 0,
            guide_strands: 0,
            guide_particles: 0,
            render_strands: 0,
            light_texels: 0,
        };
        let extents = HairGpuExtents::default();
        let optional = HairOptionalExtents::default();
        let config = HairFrameConfig {
            solver: HairSolverKind::Vbd,
            self_collision: true,
        };
        let mut budget = Vec::new();
        plan_frame_budget(config, &counts, extents, optional, &mut budget);
        // Every allocation clamps to at least one element, so no zero-byte rows.
        assert!(budget.iter().all(|row| row.byte_size > 0));
        let plan = plan_frame_memory(config, &counts, extents, optional);
        assert!(plan.resident_bytes(2) > 0);
    }

    #[test]
    fn optional_binding_maps_are_covered() {
        // Guard: the optional passes still resolve their aliasing binding-0 to
        // guide positions, so their sizes fold into the shared allocation.
        let counts = sample_counts();
        let optional = sample_optional_extents();
        let mut vbd = Vec::new();
        optional_pass_bindings(HairOptionalPass::VbdSolve, &counts, optional, &mut vbd);
        assert_eq!(
            allocation_of(
                HairScheduledPass::Optional(HairOptionalPass::VbdSolve),
                vbd[0].binding
            ),
            Some(HairFrameAllocation::GuidePositions)
        );
    }
}
