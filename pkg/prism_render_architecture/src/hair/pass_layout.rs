//! Per-pass bind-group layout confluence for the hair compute pipeline.
//!
//! [`gpu_dispatch`] answers *how many workgroups* each [`HairComputePass`]
//! dispatches; the six device-free buffer-contract modules
//! ([`gpu_buffers`], [`import_buffers`], [`interp_buffers`],
//! [`lod_dither_buffers`], [`shadow_buffers`], [`sim_pass_buffers`]) answer
//! *which `@group(0)` storage buffers* each pass binds and their per-binding
//! `std430` stride, access and element count. Each source module owns exactly
//! one concern, but the render graph, when it comes to bind a pass, needs both
//! halves at once: the workgroup count *and* the fully-sized binding table.
//!
//! This module is the confluence: given a [`HairComputePass`], the shared
//! [`HairGpuCounts`] and the non-dispatch [`HairGpuExtents`], it produces —
//! in a single call — a [`HairPassPlan`] carrying the dispatch dimension and a
//! dense `0..binding_count` list of [`HairBindingLayout`] entries. It re-exports
//! no state and allocates no device resources; it only *joins* the existing
//! contracts, so the six buffer modules stay the single source of truth for
//! their own strides and the join here can never drift from them (the tests
//! assert `bindings.len() == pass.binding_count()` for every pass).
//!
//! Everything is pure, integer and deterministic: an empty groom
//! ([`HairGpuCounts::default`] + [`HairGpuExtents::default`]) yields zero
//! workgroups and clamped one-element byte sizes without panicking, matching
//! the empty-domain behaviour the buffer contracts already guarantee.

use alloc::vec::Vec;

use crate::hair::gpu_buffers::{HairBufferAccess, HairSimBuffer};
use crate::hair::gpu_dispatch::{
    dispatch_groups, HairComputePass, HairGpuCounts, HAIR_WORKGROUP_SIZE,
};
use crate::hair::import_buffers::{HairImportExtent, HairResampleBuffer, HairRootBindBuffer};
use crate::hair::interp_buffers::HairInterpBuffer;
use crate::hair::lod_dither_buffers::HairLodDitherBuffer;
use crate::hair::pass_params::params_immediate_bytes;
use crate::hair::shadow_buffers::{
    HairDeepOpacityBuffer, HairShadowExtent, HairTransmittanceBuffer,
};
use crate::hair::sim_pass_buffers::{
    HairRootSkinningBuffer, HairSdfCollisionBuffer, HairSimPassExtent, HairWindBuffer,
};

/// The non-dispatch extents every hair pass sizes its buffers against, gathered
/// into one struct so a caller can plan any pass from a single value alongside
/// [`HairGpuCounts`]. Each field feeds the passes that consume it; passes whose
/// buffers are sized purely from [`HairGpuCounts`] (`Wind`, `LodDither`) ignore
/// all of them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct HairGpuExtents {
    /// Analytic body collider count bound by `GuideSim`
    /// ([`HairSimBuffer::Colliders`]).
    pub collider_count: u32,
    /// Total interpolated render control points across every render strand,
    /// sizing `Interpolate`'s output point buffer.
    pub render_points: u32,
    /// Scalp mesh and raw-pool extents for the import passes (`RootBind`,
    /// `Resample`).
    pub import: HairImportExtent,
    /// Deformed scalp and `SDF` primitive extents for the per-frame `Simulate`
    /// passes (`RootSkinning`, `SdfCollision`).
    pub sim_pass: HairSimPassExtent,
    /// Sample-pool and depth-slice extents for the self-shadow build passes
    /// (`Transmittance`, `DeepOpacity`).
    pub shadow: HairShadowExtent,
}

/// One resolved `@group(0)` storage binding: its binding index, how the kernel
/// accesses it, the `std430` element stride, the element count for the current
/// groom, the total byte size (`stride * count`, clamped so an empty groom
/// still reserves one element), and whether the pass writes it. Mirrors the
/// per-binding metadata each buffer-contract enum exposes, collapsed into one
/// row so callers need not re-query the source enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HairBindingLayout {
    /// `@group(0) @binding(binding)` index (dense within a pass).
    pub binding: u32,
    /// Read-only input vs. mutated-in-place storage.
    pub access: HairBufferAccess,
    /// `std430` array stride of one element, in bytes.
    pub stride: usize,
    /// Number of elements bound for the current groom.
    pub element_count: u32,
    /// Allocation size in bytes (`stride * element_count`, clamped `>= stride`).
    pub byte_size: usize,
    /// `true` when the kernel writes this buffer (`var<storage, read_write>`
    /// output), so the render graph can flag it for read-back / persistence.
    pub is_output: bool,
}

/// A fully-planned hair compute pass: the pass identity, its 1-D dispatch
/// dimension (domain element count and derived workgroup count) and the dense
/// list of `@group(0)` bindings with their sizes, plus the summed binding-table
/// byte footprint. This is everything the render graph needs to build a bind
/// group and issue the dispatch for one pass. Contains a [`Vec`], so it is not
/// `Copy`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct HairPassPlan {
    /// The compute pass this plan binds and dispatches.
    pub pass: HairComputePass,
    /// Number of domain elements this dispatch covers.
    pub domain_count: u32,
    /// Workgroups to dispatch along X (`ceil(domain_count / 64)`); `0` skips.
    pub workgroup_count: u32,
    /// The pass's `@group(0)` bindings, in dense binding order `0..len`.
    pub bindings: Vec<HairBindingLayout>,
    /// Sum of every binding's `byte_size` — the pass's total storage footprint.
    pub total_binding_bytes: usize,
    /// Byte size of the pass's `var<immediate>` params block (push constants),
    /// from [`params_immediate_bytes`]. The render graph reserves this
    /// push-constant range when it builds the pass's compute pipeline. Mirrors
    /// [`HairOptionalPassPlan::params_immediate_bytes`](super::optional_pass_layout::HairOptionalPassPlan)
    /// so both plan types carry the full pipeline-layout footprint.
    pub params_immediate_bytes: usize,
}

/// Appends the `@group(0)` binding layouts of `pass` (in dense binding order) to
/// `out`, which is cleared first. Dispatches to the pass's owning buffer-contract
/// enum for every per-binding value, so the strides and access modes here are
/// exactly those the source modules publish. Never panics; an empty groom yields
/// clamped one-element byte sizes.
pub fn pass_bindings(
    pass: HairComputePass,
    counts: &HairGpuCounts,
    extents: HairGpuExtents,
    out: &mut Vec<HairBindingLayout>,
) {
    out.clear();
    match pass {
        HairComputePass::RootBind => {
            for buffer in HairRootBindBuffer::ALL {
                out.push(HairBindingLayout {
                    binding: buffer.binding(),
                    access: buffer.access(),
                    stride: buffer.stride(),
                    element_count: buffer.element_count(counts, extents.import),
                    byte_size: buffer.byte_size(counts, extents.import),
                    is_output: buffer.is_output(),
                });
            }
        }
        HairComputePass::Resample => {
            for buffer in HairResampleBuffer::ALL {
                out.push(HairBindingLayout {
                    binding: buffer.binding(),
                    access: buffer.access(),
                    stride: buffer.stride(),
                    element_count: buffer.element_count(counts, extents.import),
                    byte_size: buffer.byte_size(counts, extents.import),
                    is_output: buffer.is_output(),
                });
            }
        }
        HairComputePass::RootSkinning => {
            for buffer in HairRootSkinningBuffer::ALL {
                out.push(HairBindingLayout {
                    binding: buffer.binding(),
                    access: buffer.access(),
                    stride: buffer.stride(),
                    element_count: buffer.element_count(counts, extents.sim_pass),
                    byte_size: buffer.byte_size(counts, extents.sim_pass),
                    is_output: buffer.is_output(),
                });
            }
        }
        HairComputePass::Wind => {
            for buffer in HairWindBuffer::ALL {
                out.push(HairBindingLayout {
                    binding: buffer.binding(),
                    access: buffer.access(),
                    stride: buffer.stride(),
                    element_count: buffer.element_count(counts),
                    byte_size: buffer.byte_size(counts),
                    is_output: buffer.is_output(),
                });
            }
        }
        HairComputePass::GuideSim => {
            for buffer in HairSimBuffer::ALL {
                // `HairSimBuffer` has no dedicated `is_output`; a written buffer
                // is exactly a `read_write` one (`Positions` / `PrevPositions`,
                // the persistent solver state), matching the other passes.
                let access = buffer.access();
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
        HairComputePass::SdfCollision => {
            for buffer in HairSdfCollisionBuffer::ALL {
                out.push(HairBindingLayout {
                    binding: buffer.binding(),
                    access: buffer.access(),
                    stride: buffer.stride(),
                    element_count: buffer.element_count(counts, extents.sim_pass),
                    byte_size: buffer.byte_size(counts, extents.sim_pass),
                    is_output: buffer.is_output(),
                });
            }
        }
        HairComputePass::Interpolate => {
            for buffer in HairInterpBuffer::ALL {
                out.push(HairBindingLayout {
                    binding: buffer.binding(),
                    access: buffer.access(),
                    stride: buffer.stride(),
                    element_count: buffer.element_count(counts, extents.render_points),
                    byte_size: buffer.byte_size(counts, extents.render_points),
                    is_output: buffer.is_output(),
                });
            }
        }
        HairComputePass::LodDither => {
            for buffer in HairLodDitherBuffer::ALL {
                out.push(HairBindingLayout {
                    binding: buffer.binding(),
                    access: buffer.access(),
                    stride: buffer.stride(),
                    element_count: buffer.element_count(counts),
                    byte_size: buffer.byte_size(counts),
                    is_output: buffer.is_output(),
                });
            }
        }
        HairComputePass::Transmittance => {
            for buffer in HairTransmittanceBuffer::ALL {
                out.push(HairBindingLayout {
                    binding: buffer.binding(),
                    access: buffer.access(),
                    stride: buffer.stride(),
                    element_count: buffer.element_count(counts, extents.shadow),
                    byte_size: buffer.byte_size(counts, extents.shadow),
                    is_output: buffer.is_output(),
                });
            }
        }
        HairComputePass::DeepOpacity => {
            for buffer in HairDeepOpacityBuffer::ALL {
                out.push(HairBindingLayout {
                    binding: buffer.binding(),
                    access: buffer.access(),
                    stride: buffer.stride(),
                    element_count: buffer.element_count(counts, extents.shadow),
                    byte_size: buffer.byte_size(counts, extents.shadow),
                    is_output: buffer.is_output(),
                });
            }
        }
    }
}

/// Plans a single `pass` into a [`HairPassPlan`]: joins its dispatch dimension
/// (domain element count → workgroup count) with its fully-sized `@group(0)`
/// binding table in one call. Never panics.
#[must_use]
pub fn plan_pass(
    pass: HairComputePass,
    counts: &HairGpuCounts,
    extents: HairGpuExtents,
) -> HairPassPlan {
    let mut bindings = Vec::new();
    pass_bindings(pass, counts, extents, &mut bindings);
    let total_binding_bytes = bindings.iter().map(|b| b.byte_size).sum();
    let domain_count = counts.domain_count(pass.domain());
    let workgroup_count = dispatch_groups(domain_count, HAIR_WORKGROUP_SIZE);
    HairPassPlan {
        pass,
        domain_count,
        workgroup_count,
        bindings,
        total_binding_bytes,
        params_immediate_bytes: params_immediate_bytes(pass),
    }
}

/// Plans each pass in `passes` into `out` (cleared first), preserving input
/// order. Unlike the dispatch planner, empty-domain passes are *not* dropped —
/// a bind-group layout is still meaningful for a pass that happens to have zero
/// work this frame — so callers gate on `workgroup_count == 0` themselves.
/// Never panics.
pub fn plan_passes(
    passes: &[HairComputePass],
    counts: &HairGpuCounts,
    extents: HairGpuExtents,
    out: &mut Vec<HairPassPlan>,
) {
    out.clear();
    for &pass in passes {
        out.push(plan_pass(pass, counts, extents));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A non-trivial groom exercising every extent field.
    fn sample_counts() -> HairGpuCounts {
        HairGpuCounts {
            roots: 100,
            guide_strands: 100,
            guide_particles: 3200,
            render_strands: 5000,
            light_texels: 4096,
        }
    }

    fn sample_extents() -> HairGpuExtents {
        HairGpuExtents {
            collider_count: 8,
            render_points: 160_000,
            import: HairImportExtent {
                scalp_vertex_count: 2048,
                scalp_index_count: 6144,
                raw_point_count: 40_000,
            },
            sim_pass: HairSimPassExtent {
                scalp_vertex_count: 2048,
                scalp_index_count: 6144,
                sdf_primitive_count: 12,
            },
            shadow: HairShadowExtent {
                sample_count: 65_536,
                layer_count: 8,
            },
        }
    }

    #[test]
    fn binding_count_matches_dispatch_contract_for_every_pass() {
        // The confluence invariant: the join here binds exactly as many buffers
        // as `gpu_dispatch` promises the kernel declares, for all ten passes.
        let counts = sample_counts();
        let extents = sample_extents();
        let mut out = Vec::new();
        for pass in HairComputePass::ALL {
            pass_bindings(pass, &counts, extents, &mut out);
            assert_eq!(
                out.len() as u32,
                pass.binding_count(),
                "binding count mismatch for {pass:?}"
            );
        }
    }

    #[test]
    fn bindings_are_dense_from_zero() {
        let counts = sample_counts();
        let extents = sample_extents();
        let mut out = Vec::new();
        for pass in HairComputePass::ALL {
            pass_bindings(pass, &counts, extents, &mut out);
            for (i, binding) in out.iter().enumerate() {
                assert_eq!(binding.binding, i as u32, "sparse binding in {pass:?}");
            }
        }
    }

    #[test]
    fn byte_size_is_stride_times_count_clamped() {
        let counts = sample_counts();
        let extents = sample_extents();
        let mut out = Vec::new();
        for pass in HairComputePass::ALL {
            pass_bindings(pass, &counts, extents, &mut out);
            for binding in &out {
                let raw = binding
                    .stride
                    .saturating_mul(binding.element_count as usize);
                let expected = raw.max(binding.stride);
                assert_eq!(binding.byte_size, expected, "byte size in {pass:?}");
            }
        }
    }

    #[test]
    fn empty_groom_clamps_to_one_element_and_never_panics() {
        let counts = HairGpuCounts::default();
        let extents = HairGpuExtents::default();
        let mut out = Vec::new();
        for pass in HairComputePass::ALL {
            pass_bindings(pass, &counts, extents, &mut out);
            assert_eq!(out.len() as u32, pass.binding_count());
            for binding in &out {
                // Empty groom → the contracts clamp element allocation to one.
                assert_eq!(binding.byte_size, binding.stride);
            }
        }
    }

    #[test]
    fn plan_pass_matches_dispatch_and_binding_join() {
        let counts = sample_counts();
        let extents = sample_extents();
        for pass in HairComputePass::ALL {
            let plan = plan_pass(pass, &counts, extents);
            assert_eq!(plan.pass, pass);
            // Dispatch half matches gpu_dispatch exactly.
            let domain = counts.domain_count(pass.domain());
            assert_eq!(plan.domain_count, domain);
            assert_eq!(
                plan.workgroup_count,
                dispatch_groups(domain, HAIR_WORKGROUP_SIZE)
            );
            // Binding half matches pass_bindings exactly.
            let mut expected = Vec::new();
            pass_bindings(pass, &counts, extents, &mut expected);
            assert_eq!(plan.bindings, expected);
            let sum: usize = expected.iter().map(|b| b.byte_size).sum();
            assert_eq!(plan.total_binding_bytes, sum);
            // Immediate-params half matches pass_params exactly.
            assert_eq!(plan.params_immediate_bytes, params_immediate_bytes(pass));
        }
    }

    #[test]
    fn access_and_output_agree_with_source_enums() {
        let counts = sample_counts();
        let extents = sample_extents();
        // GuideSim: Positions is read_write output, Goals is read-only input.
        let sim = plan_pass(HairComputePass::GuideSim, &counts, extents);
        let positions = sim
            .bindings
            .iter()
            .find(|b| b.binding == HairSimBuffer::Positions.binding())
            .unwrap();
        assert_eq!(positions.access, HairBufferAccess::ReadWrite);
        assert!(positions.is_output);
        let goals = sim
            .bindings
            .iter()
            .find(|b| b.binding == HairSimBuffer::Goals.binding())
            .unwrap();
        assert_eq!(goals.access, HairBufferAccess::Read);
        assert!(!goals.is_output);
    }

    #[test]
    fn plan_passes_preserves_order_and_keeps_empty_domains() {
        let counts = HairGpuCounts {
            roots: 0,
            guide_strands: 0,
            guide_particles: 0,
            render_strands: 128,
            light_texels: 0,
        };
        let extents = HairGpuExtents::default();
        let mut plans = Vec::new();
        plan_passes(&HairComputePass::ALL, &counts, extents, &mut plans);
        // Every pass is retained (unlike plan_dispatches), in ALL order.
        assert_eq!(plans.len(), HairComputePass::ALL.len());
        for (plan, &pass) in plans.iter().zip(HairComputePass::ALL.iter()) {
            assert_eq!(plan.pass, pass);
        }
        // Interpolate (render_strands = 128) has work; RootBind (roots = 0) does not.
        let interp = plans
            .iter()
            .find(|p| p.pass == HairComputePass::Interpolate)
            .unwrap();
        assert_eq!(interp.workgroup_count, 2); // ceil(128 / 64)
        let root_bind = plans
            .iter()
            .find(|p| p.pass == HairComputePass::RootBind)
            .unwrap();
        assert_eq!(root_bind.workgroup_count, 0);
        // Even a zero-work pass still carries its full binding table.
        assert_eq!(
            root_bind.bindings.len() as u32,
            HairComputePass::RootBind.binding_count()
        );
    }
}
