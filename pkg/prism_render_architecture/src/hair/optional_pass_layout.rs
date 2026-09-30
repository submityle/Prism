//! Per-pass bind-group layout confluence for the hair *optional* compute passes.
//!
//! [`pass_layout`](super::pass_layout) joins the dispatch dimension and the
//! sized `@group(0)` binding table for every pass on the fixed main spine
//! (`HairComputePass`). The optional / alternative-solver passes
//! ([`HairOptionalPass`]) live off that spine — the guide-`VBD` solver that can
//! replace `GuideSim`, and the self-collision `accumulate`→`apply` pair that can
//! be inserted after `SdfCollision` — and own their byte-layout contracts in
//! [`vbd_pass_buffers`](super::vbd_pass_buffers) and
//! [`self_collision_pass_buffers`](super::self_collision_pass_buffers) rather
//! than in the six main buffer modules. [`optional_pass_dispatch`] answers *how
//! many workgroups* each optional pass dispatches; this module is the matching
//! confluence, producing — in one call — the dispatch dimension *and* the dense
//! `0..binding_count` [`HairBindingLayout`] table plus the immediate
//! (push-constant) block size the render graph needs to bind and issue one
//! optional pass.
//!
//! It re-exports no state and allocates no device resources; it only *joins* the
//! existing contracts, so the two optional buffer modules stay the single source
//! of truth for their strides and the join here can never drift (the tests
//! assert `bindings.len() == pass.binding_count()` for every optional pass). The
//! row type is [`HairBindingLayout`] from [`pass_layout`](super::pass_layout), so
//! the render graph consumes one binding-layout row for main and optional passes
//! alike; the two optional buffer enums each declare their own local access
//! enum, which this module normalises onto the canonical
//! [`HairBufferAccess`](super::gpu_buffers::HairBufferAccess).
//!
//! Everything is pure, integer and deterministic: an empty groom
//! ([`HairGpuCounts::default`] + [`HairOptionalExtents::default`]) yields zero
//! workgroups and clamped one-element byte sizes without panicking, exactly like
//! the buffer contracts it joins.

use alloc::vec::Vec;

use crate::hair::gpu_buffers::HairBufferAccess;
use crate::hair::gpu_dispatch::HairGpuCounts;
use crate::hair::optional_pass_dispatch::HairOptionalPass;
use crate::hair::pass_layout::HairBindingLayout;
use crate::hair::self_collision_pass_buffers::{
    HairBufferAccess as SelfCollisionAccess, HairSelfCollisionBuffer,
    PARAMS_IMMEDIATE_BYTES as SELF_COLLISION_PARAMS_BYTES,
};
use crate::hair::vbd_pass_buffers::{
    HairBufferAccess as VbdAccess, HairVbdBuffer, PARAMS_IMMEDIATE_BYTES as VBD_PARAMS_BYTES,
};

/// The non-dispatch element counts the optional passes size their non-domain
/// buffers against, gathered into one struct so a caller can plan any optional
/// pass from a single value alongside [`HairGpuCounts`]. Mirrors
/// [`HairGpuExtents`](super::pass_layout::HairGpuExtents) for the main spine.
///
/// The domain-sized buffers (`positions`, `prev_positions`, rest lengths, …)
/// come straight from [`HairGpuCounts`]; only the two counts here are external
/// to the dispatch domains.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct HairOptionalExtents {
    /// Analytic body collider count bound by the guide-`VBD` solver
    /// ([`HairVbdBuffer::Colliders`]). Matches `HairVbdParams.collider_count`.
    pub collider_count: u32,
    /// Occupied self-collision grid cell count
    /// ([`GridCsr::cell_count`](super::self_collision_grid::GridCsr)), sizing the
    /// CSR `cell_keys` / `cell_starts` bindings of the self-collision passes.
    pub cell_count: u32,
}

/// A fully-planned optional hair compute pass: the pass identity, its 1-D
/// dispatch dimension (domain element count and derived workgroup count), the
/// dense list of `@group(0)` bindings with their sizes, the summed binding-table
/// byte footprint and the immediate (push-constant) block size. This is
/// everything the render graph needs to build a bind group and issue the
/// dispatch for one optional pass. Contains a [`Vec`], so it is not `Copy`.
///
/// Parallels [`HairPassPlan`](super::pass_layout::HairPassPlan) for the main
/// spine; both plan types now carry `params_immediate_bytes` (the main spine's
/// push-constant sizing is centralised in
/// [`pass_params`](super::pass_params), the optional passes' in their own
/// buffer modules), so the two are symmetric.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct HairOptionalPassPlan {
    /// The optional compute pass this plan binds and dispatches.
    pub pass: HairOptionalPass,
    /// Number of domain elements this dispatch covers.
    pub domain_count: u32,
    /// Workgroups to dispatch along X (`ceil(domain_count / 64)`); `0` skips.
    pub workgroup_count: u32,
    /// The pass's `@group(0)` bindings, in dense binding order `0..len`.
    pub bindings: Vec<HairBindingLayout>,
    /// Sum of every binding's `byte_size` — the pass's total storage footprint.
    pub total_binding_bytes: usize,
    /// Byte size of the pass's `var<immediate>` params block (push constants).
    pub params_immediate_bytes: usize,
}

/// Normalises the guide-`VBD` module's local access enum onto the canonical
/// [`HairBufferAccess`].
#[must_use]
fn normalise_vbd_access(access: VbdAccess) -> HairBufferAccess {
    match access {
        VbdAccess::Read => HairBufferAccess::Read,
        VbdAccess::ReadWrite => HairBufferAccess::ReadWrite,
    }
}

/// Normalises the self-collision module's local access enum onto the canonical
/// [`HairBufferAccess`].
#[must_use]
fn normalise_self_collision_access(access: SelfCollisionAccess) -> HairBufferAccess {
    match access {
        SelfCollisionAccess::Read => HairBufferAccess::Read,
        SelfCollisionAccess::ReadWrite => HairBufferAccess::ReadWrite,
    }
}

/// Byte size of the `var<immediate>` params block declared by the `WESL` twin of
/// `pass`. The self-collision `accumulate` and `apply` kernels share one params
/// block (they live in one file), so both report the same size.
#[must_use]
pub fn optional_params_immediate_bytes(pass: HairOptionalPass) -> usize {
    match pass {
        HairOptionalPass::VbdSolve => VBD_PARAMS_BYTES,
        HairOptionalPass::SelfCollisionAccumulate | HairOptionalPass::SelfCollisionApply => {
            SELF_COLLISION_PARAMS_BYTES
        }
    }
}

/// Appends the `@group(0)` binding layouts of the optional `pass` (in dense
/// binding order) to `out`, which is cleared first. Dispatches to the pass's
/// owning buffer-contract enum for every per-binding value, so the strides and
/// access modes here are exactly those the source modules publish. Never panics;
/// an empty groom yields clamped one-element byte sizes.
pub fn optional_pass_bindings(
    pass: HairOptionalPass,
    counts: &HairGpuCounts,
    extents: HairOptionalExtents,
    out: &mut Vec<HairBindingLayout>,
) {
    out.clear();
    match pass {
        HairOptionalPass::VbdSolve => {
            for buffer in HairVbdBuffer::ALL {
                let access = normalise_vbd_access(buffer.access());
                out.push(HairBindingLayout {
                    binding: buffer.binding(),
                    access,
                    stride: buffer.stride(),
                    element_count: buffer.element_count(counts, extents.collider_count),
                    byte_size: buffer.byte_size(counts, extents.collider_count),
                    is_output: access == HairBufferAccess::ReadWrite,
                });
            }
        }
        HairOptionalPass::SelfCollisionAccumulate | HairOptionalPass::SelfCollisionApply => {
            // Both self-collision kernels bind the identical five-buffer group.
            for buffer in HairSelfCollisionBuffer::ALL {
                let access = normalise_self_collision_access(buffer.access());
                out.push(HairBindingLayout {
                    binding: buffer.binding(),
                    access,
                    stride: buffer.stride(),
                    element_count: buffer.element_count(counts, extents.cell_count),
                    byte_size: buffer.byte_size(counts, extents.cell_count),
                    is_output: access == HairBufferAccess::ReadWrite,
                });
            }
        }
    }
}

/// Fully plans one optional pass: joins its dispatch dimension
/// ([`optional_pass_dispatch`]) with its sized binding table
/// ([`optional_pass_bindings`]) and its immediate block size into one
/// [`HairOptionalPassPlan`]. Never panics; an empty groom yields a plan with
/// zero workgroups and clamped one-element byte sizes.
#[must_use]
pub fn plan_optional_pass(
    pass: HairOptionalPass,
    counts: &HairGpuCounts,
    extents: HairOptionalExtents,
) -> HairOptionalPassPlan {
    let mut bindings = Vec::new();
    optional_pass_bindings(pass, counts, extents, &mut bindings);
    let total_binding_bytes = bindings.iter().map(|binding| binding.byte_size).sum();
    HairOptionalPassPlan {
        pass,
        domain_count: counts.domain_count(pass.domain()),
        workgroup_count: pass.workgroup_count(counts),
        bindings,
        total_binding_bytes,
        params_immediate_bytes: optional_params_immediate_bytes(pass),
    }
}

/// Resolves `passes` into ordered [`HairOptionalPassPlan`] entries for the given
/// groom, appending to `out` (cleared first). Passes whose domain is empty
/// (`workgroup_count == 0`) are skipped, mirroring
/// [`plan_optional_dispatches`](super::optional_pass_dispatch::plan_optional_dispatches)
/// so the plan list matches the dispatch list one-for-one. Input order is
/// preserved. Never panics.
pub fn plan_optional_passes(
    passes: &[HairOptionalPass],
    counts: &HairGpuCounts,
    extents: HairOptionalExtents,
    out: &mut Vec<HairOptionalPassPlan>,
) {
    out.clear();
    for &pass in passes {
        let plan = plan_optional_pass(pass, counts, extents);
        if plan.workgroup_count == 0 {
            continue;
        }
        out.push(plan);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hair::gpu_buffers::HairSimBuffer;
    use crate::hair::optional_pass_dispatch::self_collision_passes;

    fn counts() -> HairGpuCounts {
        HairGpuCounts {
            guide_strands: 100,
            guide_particles: 3200,
            ..HairGpuCounts::default()
        }
    }

    #[test]
    fn vbd_plan_has_six_dense_bindings() {
        let plan = plan_optional_pass(
            HairOptionalPass::VbdSolve,
            &counts(),
            HairOptionalExtents {
                collider_count: 3,
                cell_count: 0,
            },
        );
        assert_eq!(
            plan.bindings.len() as u32,
            HairOptionalPass::VbdSolve.binding_count()
        );
        assert_eq!(plan.bindings.len(), 6);
        for (index, binding) in plan.bindings.iter().enumerate() {
            assert_eq!(binding.binding, index as u32);
        }
        // Positions / PrevPositions / Targets are the read-write outputs.
        let outputs: Vec<u32> = plan
            .bindings
            .iter()
            .filter(|binding| binding.is_output)
            .map(|binding| binding.binding)
            .collect();
        assert_eq!(outputs, alloc::vec![0, 1, 2]);
    }

    #[test]
    fn self_collision_accumulate_and_apply_share_the_same_five_bindings() {
        let extents = HairOptionalExtents {
            collider_count: 0,
            cell_count: 64,
        };
        let accumulate = plan_optional_pass(
            HairOptionalPass::SelfCollisionAccumulate,
            &counts(),
            extents,
        );
        let apply = plan_optional_pass(HairOptionalPass::SelfCollisionApply, &counts(), extents);
        assert_eq!(accumulate.bindings.len(), 5);
        assert_eq!(accumulate.bindings, apply.bindings);
        for (index, binding) in accumulate.bindings.iter().enumerate() {
            assert_eq!(binding.binding, index as u32);
        }
        // Positions + Corrections are the read-write buffers.
        let outputs: Vec<u32> = accumulate
            .bindings
            .iter()
            .filter(|binding| binding.is_output)
            .map(|binding| binding.binding)
            .collect();
        assert_eq!(outputs, alloc::vec![0, 1]);
    }

    #[test]
    fn binding_count_matches_dispatch_metadata_for_every_optional_pass() {
        let extents = HairOptionalExtents {
            collider_count: 2,
            cell_count: 32,
        };
        for pass in [
            HairOptionalPass::VbdSolve,
            HairOptionalPass::SelfCollisionAccumulate,
            HairOptionalPass::SelfCollisionApply,
        ] {
            let plan = plan_optional_pass(pass, &counts(), extents);
            assert_eq!(plan.bindings.len() as u32, pass.binding_count());
        }
    }

    #[test]
    fn params_immediate_bytes_match_shader_blocks() {
        assert_eq!(
            optional_params_immediate_bytes(HairOptionalPass::VbdSolve),
            48
        );
        assert_eq!(
            optional_params_immediate_bytes(HairOptionalPass::SelfCollisionAccumulate),
            20
        );
        assert_eq!(
            optional_params_immediate_bytes(HairOptionalPass::SelfCollisionApply),
            20
        );
    }

    #[test]
    fn vbd_positions_stride_is_byte_identical_to_the_sim() {
        let plan = plan_optional_pass(
            HairOptionalPass::VbdSolve,
            &counts(),
            HairOptionalExtents::default(),
        );
        let vbd_positions = plan.bindings[0];
        assert_eq!(vbd_positions.binding, 0);
        // The persistent guide state must be byte-shareable between solvers.
        assert_eq!(vbd_positions.stride, HairSimBuffer::Positions.stride());
    }

    #[test]
    fn total_binding_bytes_sums_the_binding_table() {
        let plan = plan_optional_pass(
            HairOptionalPass::VbdSolve,
            &counts(),
            HairOptionalExtents {
                collider_count: 4,
                cell_count: 0,
            },
        );
        let summed: usize = plan.bindings.iter().map(|binding| binding.byte_size).sum();
        assert_eq!(plan.total_binding_bytes, summed);
    }

    #[test]
    fn workgroup_count_matches_the_dispatch_planner() {
        let counts = counts();
        for pass in [
            HairOptionalPass::VbdSolve,
            HairOptionalPass::SelfCollisionAccumulate,
        ] {
            let plan = plan_optional_pass(pass, &counts, HairOptionalExtents::default());
            assert_eq!(plan.workgroup_count, pass.workgroup_count(&counts));
        }
    }

    #[test]
    fn empty_groom_yields_zero_workgroups_but_clamped_bytes() {
        let plan = plan_optional_pass(
            HairOptionalPass::VbdSolve,
            &HairGpuCounts::default(),
            HairOptionalExtents::default(),
        );
        assert_eq!(plan.workgroup_count, 0);
        // Every binding still reserves at least one element so the WebGPU
        // storage binding is valid even for an empty groom.
        for binding in &plan.bindings {
            assert!(binding.byte_size >= binding.stride);
        }
    }

    #[test]
    fn plan_optional_passes_skips_empty_domains_and_preserves_order() {
        // Non-empty groom: the accumulate→apply pair both survive, in order.
        let extents = HairOptionalExtents {
            collider_count: 0,
            cell_count: 16,
        };
        let mut out = Vec::new();
        plan_optional_passes(&self_collision_passes(), &counts(), extents, &mut out);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].pass, HairOptionalPass::SelfCollisionAccumulate);
        assert_eq!(out[1].pass, HairOptionalPass::SelfCollisionApply);

        // Empty groom: both are skipped.
        let mut empty = Vec::new();
        plan_optional_passes(
            &self_collision_passes(),
            &HairGpuCounts::default(),
            extents,
            &mut empty,
        );
        assert!(empty.is_empty());
    }
}
