//! Frame-level bind resolution: annotates every scheduled `@group(0)` binding
//! slot with the physical allocation that backs it.
//!
//! [`frame_pass_layout`](super::frame_pass_layout) already produces the ordered
//! per-frame bind table — for each scheduled pass, its dense `@group(0)`
//! bindings with sizes and access. [`frame_resource_map`](super::frame_resource_map)
//! separately answers *which physical buffer* a given `(pass, binding)` view
//! aliases. What the render graph actually needs while walking a frame — slot by
//! slot — is those two joined: at every binding slot, the resolved physical
//! [`HairFrameAllocation`] to bind there, so that the many aliasing views of a
//! shared buffer all point at the same allocation. `Wind` binding-0 and
//! `Interpolate` binding-0 both resolve to the single guide-positions
//! allocation; if the graph bound a distinct buffer per `(pass, binding)` the
//! guide solve would silently lose its integration across passes.
//!
//! This module performs that join. For each pass in the per-frame schedule it
//! walks the authoritative single-pass binding table
//! ([`plan_pass`](super::pass_layout::plan_pass) /
//! [`plan_optional_pass`](super::optional_pass_layout::plan_optional_pass)) and
//! resolves every slot to its logical allocation
//! ([`allocation_of`]) plus that allocation's residency class. Empty-domain
//! passes are skipped exactly like
//! [`plan_frame_passes`](super::frame_pass_layout::plan_frame_passes) so the
//! resolved list matches the dispatch list one-for-one. Every in-scope
//! per-frame binding resolves to an allocation, so no slot is dropped; a slot
//! that resolves to nothing (never reached for the per-frame schedule) is
//! skipped rather than panicking.
//!
//! Byte size and access are copied straight from the single-pass ABI, so this
//! module never re-derives sizing; it only adds the physical identity that lets
//! the render graph share buffers correctly. Everything is deterministic,
//! pure-integer, and never panics.

use alloc::vec::Vec;

use crate::hair::frame_resource_map::{
    allocation_of, HairAllocationResidency, HairFrameAllocation,
};
use crate::hair::frame_schedule::{per_frame_schedule, HairFrameConfig, HairScheduledPass};
use crate::hair::gpu_buffers::HairBufferAccess;
use crate::hair::gpu_dispatch::HairGpuCounts;
use crate::hair::optional_pass_layout::{plan_optional_pass, HairOptionalExtents};
use crate::hair::pass_layout::{plan_pass, HairBindingLayout, HairGpuExtents};

/// One resolved `@group(0)` binding slot: the slot index and its access/size
/// copied from the single-pass ABI, joined with the physical allocation that
/// backs it and that allocation's residency class.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HairResolvedBinding {
    /// `@group(0) @binding(binding)` index (dense within the pass).
    pub binding: u32,
    /// Read-only input vs. mutated-in-place storage, from the single-pass ABI.
    pub access: HairBufferAccess,
    /// Bytes the slot's buffer occupies, from the single-pass ABI.
    pub byte_size: usize,
    /// `true` when the kernel writes this slot.
    pub is_output: bool,
    /// The deduplicated physical allocation this slot binds. Aliasing views of
    /// one buffer (guide positions across many passes) all resolve here to the
    /// same variant.
    pub allocation: HairFrameAllocation,
    /// How that allocation is backed across the frame timeline.
    pub residency: HairAllocationResidency,
}

/// One scheduled pass with every `@group(0)` slot resolved to its physical
/// allocation, in dense binding order. Contains a [`Vec`], so it is not `Copy`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct HairScheduledBindPlan {
    /// The scheduled pass (main-spine or optional) this bind plan is for.
    pub pass: HairScheduledPass,
    /// The pass's resolved binding slots, in dense binding order `0..len`.
    pub slots: Vec<HairResolvedBinding>,
}

/// Resolves one scheduled pass's authoritative binding table into `out`
/// (cleared first): every `@group(0)` slot joined with the physical allocation
/// it binds and that allocation's residency. Slots that resolve to no per-frame
/// allocation are skipped (never reached for the per-frame schedule). Never
/// panics.
pub fn resolve_pass_bindings(
    pass: HairScheduledPass,
    counts: &HairGpuCounts,
    extents: HairGpuExtents,
    optional_extents: HairOptionalExtents,
    out: &mut Vec<HairResolvedBinding>,
) {
    out.clear();
    let bindings: Vec<HairBindingLayout> = match pass {
        HairScheduledPass::Main(main) => plan_pass(main, counts, extents).bindings,
        HairScheduledPass::Optional(optional) => {
            plan_optional_pass(optional, counts, optional_extents).bindings
        }
    };
    for layout in &bindings {
        let Some(allocation) = allocation_of(pass, layout.binding) else {
            continue;
        };
        out.push(HairResolvedBinding {
            binding: layout.binding,
            access: layout.access,
            byte_size: layout.byte_size,
            is_output: layout.is_output,
            allocation,
            residency: allocation.residency(),
        });
    }
}

/// Resolves the whole per-frame schedule for `config` into ordered
/// [`HairScheduledBindPlan`] rows, appending to `out` (cleared first).
///
/// Walks [`per_frame_schedule`], resolving each pass's `@group(0)` slots to
/// their physical allocations. Empty-domain passes (`workgroup_count == 0`) are
/// skipped so the list matches
/// [`plan_frame_passes`](super::frame_pass_layout::plan_frame_passes) and
/// [`plan_frame_dispatches`](super::frame_schedule::plan_frame_dispatches)
/// one-for-one. Deterministic; never panics.
pub fn plan_frame_bindings(
    config: HairFrameConfig,
    counts: &HairGpuCounts,
    extents: HairGpuExtents,
    optional_extents: HairOptionalExtents,
    out: &mut Vec<HairScheduledBindPlan>,
) {
    out.clear();
    let mut schedule = Vec::new();
    per_frame_schedule(config, &mut schedule);
    let mut slots = Vec::new();
    for pass in schedule {
        let workgroup_count = pass.workgroup_count(counts);
        if workgroup_count == 0 {
            continue;
        }
        resolve_pass_bindings(pass, counts, extents, optional_extents, &mut slots);
        out.push(HairScheduledBindPlan {
            pass,
            slots: slots.clone(),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hair::gpu_dispatch::HairComputePass;
    use crate::hair::optional_pass_dispatch::HairOptionalPass;
    use crate::hair::sim_pass_buffers::HairSimPassExtent;
    use crate::hair::solver::HairSolverKind;

    fn counts() -> HairGpuCounts {
        HairGpuCounts {
            roots: 100,
            guide_strands: 100,
            guide_particles: 3200,
            render_strands: 50_000,
            light_texels: 4096,
        }
    }

    fn extents() -> HairGpuExtents {
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

    fn optional_extents() -> HairOptionalExtents {
        HairOptionalExtents {
            collider_count: 8,
            cell_count: 4096,
        }
    }

    #[test]
    fn every_scheduled_slot_resolves_to_an_allocation() {
        let counts = counts();
        let extents = extents();
        let optional = optional_extents();
        let mut plan = Vec::new();
        plan_frame_bindings(
            HairFrameConfig::default(),
            &counts,
            extents,
            optional,
            &mut plan,
        );
        assert!(!plan.is_empty());
        // No pass appears with an empty slot list, and each pass's resolved slot
        // count equals its authoritative binding count (no slot dropped).
        for row in &plan {
            let mut slots = Vec::new();
            resolve_pass_bindings(row.pass, &counts, extents, optional, &mut slots);
            assert_eq!(row.slots.len(), slots.len());
            assert!(!row.slots.is_empty());
        }
    }

    #[test]
    fn aliasing_binding_zero_resolves_to_shared_guide_positions() {
        let counts = counts();
        let extents = extents();
        let optional = optional_extents();
        let mut plan = Vec::new();
        plan_frame_bindings(
            HairFrameConfig::default(),
            &counts,
            extents,
            optional,
            &mut plan,
        );
        // Wind binding-0, SdfCollision binding-0 and Interpolate binding-0 all
        // resolve to the single guide-positions allocation.
        for pass in [
            HairComputePass::Wind,
            HairComputePass::SdfCollision,
            HairComputePass::Interpolate,
        ] {
            let row = plan
                .iter()
                .find(|row| row.pass == HairScheduledPass::Main(pass))
                .expect("pass must be scheduled");
            let slot0 = row
                .slots
                .iter()
                .find(|slot| slot.binding == 0)
                .expect("binding 0 must resolve");
            assert_eq!(slot0.allocation, HairFrameAllocation::GuidePositions);
            assert_eq!(slot0.residency, HairAllocationResidency::Persistent);
        }
    }

    #[test]
    fn interpolate_reads_guide_positions_while_solver_writes_it() {
        let counts = counts();
        let extents = extents();
        let optional = optional_extents();
        let mut plan = Vec::new();
        plan_frame_bindings(
            HairFrameConfig::default(),
            &counts,
            extents,
            optional,
            &mut plan,
        );
        let interp = plan
            .iter()
            .find(|row| row.pass == HairScheduledPass::Main(HairComputePass::Interpolate))
            .expect("interpolate scheduled");
        let interp0 = interp.slots.iter().find(|s| s.binding == 0).unwrap();
        assert_eq!(interp0.allocation, HairFrameAllocation::GuidePositions);
        assert_eq!(interp0.access, HairBufferAccess::Read);
        assert!(!interp0.is_output);

        let sim = plan
            .iter()
            .find(|row| row.pass == HairScheduledPass::Main(HairComputePass::GuideSim))
            .expect("guide sim scheduled");
        let sim0 = sim.slots.iter().find(|s| s.binding == 0).unwrap();
        assert_eq!(sim0.allocation, HairFrameAllocation::GuidePositions);
        assert_eq!(sim0.access, HairBufferAccess::ReadWrite);
    }

    #[test]
    fn resolved_output_slot_is_flagged() {
        let counts = counts();
        let extents = extents();
        let optional = optional_extents();
        let mut plan = Vec::new();
        plan_frame_bindings(
            HairFrameConfig::default(),
            &counts,
            extents,
            optional,
            &mut plan,
        );
        let interp = plan
            .iter()
            .find(|row| row.pass == HairScheduledPass::Main(HairComputePass::Interpolate))
            .expect("interpolate scheduled");
        let out_slot = interp
            .slots
            .iter()
            .find(|s| s.allocation == HairFrameAllocation::RenderOutPoints)
            .expect("interpolate writes render out points");
        assert!(out_slot.is_output);
        assert_eq!(out_slot.access, HairBufferAccess::ReadWrite);
        assert_eq!(out_slot.residency, HairAllocationResidency::Persistent);
    }

    #[test]
    fn byte_size_matches_single_pass_abi() {
        let counts = counts();
        let extents = extents();
        let optional = optional_extents();
        let mut plan = Vec::new();
        plan_frame_bindings(
            HairFrameConfig::default(),
            &counts,
            extents,
            optional,
            &mut plan,
        );
        // Every resolved slot's byte size equals the authoritative single-pass
        // binding layout (never re-derived here).
        for row in &plan {
            if let HairScheduledPass::Main(main) = row.pass {
                let authoritative = plan_pass(main, &counts, extents);
                for slot in &row.slots {
                    let layout = authoritative
                        .bindings
                        .iter()
                        .find(|b| b.binding == slot.binding)
                        .expect("binding present in authoritative table");
                    assert_eq!(slot.byte_size, layout.byte_size);
                }
            }
        }
    }

    #[test]
    fn vbd_and_self_collision_slots_resolve_to_scratch() {
        let counts = counts();
        let extents = extents();
        let optional = optional_extents();
        let config = HairFrameConfig {
            solver: HairSolverKind::Vbd,
            self_collision: true,
        };
        let mut plan = Vec::new();
        plan_frame_bindings(config, &counts, extents, optional, &mut plan);
        let vbd = plan
            .iter()
            .find(|row| row.pass == HairScheduledPass::Optional(HairOptionalPass::VbdSolve))
            .expect("VBD solve scheduled");
        let targets = vbd
            .slots
            .iter()
            .find(|s| s.allocation == HairFrameAllocation::VbdTargets)
            .expect("VBD targets slot present");
        assert_eq!(targets.residency, HairAllocationResidency::Scratch);

        let corrections = plan
            .iter()
            .flat_map(|row| row.slots.iter())
            .find(|s| s.allocation == HairFrameAllocation::SelfCollisionCorrections)
            .expect("self-collision corrections slot present");
        assert_eq!(corrections.residency, HairAllocationResidency::Scratch);
    }

    #[test]
    fn bind_plan_matches_dispatch_list_one_for_one() {
        let counts = counts();
        let extents = extents();
        let optional = optional_extents();
        let config = HairFrameConfig {
            solver: HairSolverKind::Vbd,
            self_collision: true,
        };
        let mut bind_plan = Vec::new();
        plan_frame_bindings(config, &counts, extents, optional, &mut bind_plan);
        let mut schedule = Vec::new();
        per_frame_schedule(config, &mut schedule);
        let expected: Vec<HairScheduledPass> = schedule
            .into_iter()
            .filter(|p| p.workgroup_count(&counts) != 0)
            .collect();
        let actual: Vec<HairScheduledPass> = bind_plan.iter().map(|row| row.pass).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn empty_groom_yields_empty_plan_without_panicking() {
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
        let mut plan = Vec::new();
        plan_frame_bindings(config, &counts, extents, optional, &mut plan);
        assert!(plan.is_empty());
    }
}
