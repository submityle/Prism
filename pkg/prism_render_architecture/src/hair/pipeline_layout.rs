//! Device-independent compute *pipeline layout* descriptor for the hair spine.
//!
//! The per-pass modules already answer every question needed to *build a bind
//! group and issue a dispatch* for a running frame: [`pass_layout`] and
//! [`optional_pass_layout`] answer *which `@group(0)` storage buffers a pass
//! binds and how large each is*, while [`pass_params`] and
//! [`optional_pass_layout`] answer *how many push-constant bytes* the pass
//! declares. Those are the *runtime* halves — they depend on the current
//! groom's element counts.
//!
//! Creating the `wgpu` pipeline, however, needs the *count-independent* shape:
//! a bind-group-layout (binding indices, storage read/read-write types and
//! per-binding minimum sizes) plus the push-constant range. That shape never
//! changes with the groom size — only the bound buffers' element counts do —
//! so the render graph builds one pipeline layout per pass at load time and
//! reuses it every frame. This module is that count-independent join: given a
//! [`HairComputePass`], [`HairOptionalPass`] or [`HairScheduledPass`] it
//! produces a [`HairComputePipelineLayout`] carrying the dense
//! [`HairBindGroupLayoutEntry`] list and the optional [`HairPushConstantRange`].
//!
//! It derives the entries from the very same buffer-contract confluence the
//! runtime path uses ([`pass_bindings`] / [`optional_pass_bindings`]) but keeps
//! only the count-independent fields: the binding index, the storage binding
//! type (from [`HairBufferAccess`]) and the `std430` element stride as the
//! `min_binding_size`. Because a bound `array<T>`'s minimum size is one element
//! stride regardless of element count, the resulting layout is identical for
//! any [`HairGpuCounts`] — the tests assert this by re-deriving entries from a
//! non-default count and comparing. Every hair pass binds exactly one
//! `@group(0)` group of `var<storage>` buffers and one `var<immediate>`
//! push-constant block, so [`HairComputePipelineLayout::bind_group_count`] is
//! always `1` and visibility is always the compute stage.
//!
//! Everything is pure, integer and deterministic; no device resources are
//! allocated and nothing panics.

use alloc::vec::Vec;

use crate::hair::frame_schedule::HairScheduledPass;
use crate::hair::gpu_buffers::HairBufferAccess;
use crate::hair::gpu_dispatch::{HairComputePass, HairGpuCounts};
use crate::hair::optional_pass_dispatch::HairOptionalPass;
use crate::hair::optional_pass_layout::{
    optional_params_immediate_bytes, optional_pass_bindings, HairOptionalExtents,
};
use crate::hair::pass_layout::{pass_bindings, HairBindingLayout, HairGpuExtents};
use crate::hair::pass_params::params_immediate_bytes;

/// How a `@group(0)` storage binding is declared in the compute pipeline
/// layout. Derived one-to-one from [`HairBufferAccess`]; the render graph maps
/// each variant to the matching `wgpu` `BufferBindingType::Storage` read-only
/// flag.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HairBufferBindingType {
    /// `var<storage, read>` — read-only storage (`read_only: true`).
    StorageReadOnly,
    /// `var<storage, read_write>` — mutated-in-place storage (`read_only:
    /// false`).
    StorageReadWrite,
}

impl HairBufferBindingType {
    /// Maps a buffer access mode to its bind-group-layout storage type.
    #[must_use]
    pub fn from_access(access: HairBufferAccess) -> Self {
        match access {
            HairBufferAccess::Read => Self::StorageReadOnly,
            HairBufferAccess::ReadWrite => Self::StorageReadWrite,
        }
    }

    /// `true` for a read-only storage binding.
    #[must_use]
    pub fn is_read_only(self) -> bool {
        matches!(self, Self::StorageReadOnly)
    }
}

/// One `@group(0)` bind-group-layout entry for a hair compute pass: the binding
/// index, the storage buffer binding type, the count-independent minimum bound
/// size in bytes (one `std430` element stride) and whether the entry carries a
/// dynamic offset. Hair passes never use dynamic offsets, so
/// `has_dynamic_offset` is always `false`; every entry is visible only to the
/// compute stage (see [`HairBindGroupLayoutEntry::VISIBILITY_IS_COMPUTE`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HairBindGroupLayoutEntry {
    /// `@group(0) @binding(binding)` index (dense within a pass).
    pub binding: u32,
    /// Read-only vs. read-write storage binding type.
    pub ty: HairBufferBindingType,
    /// Minimum bound size in bytes: one `std430` array element stride. This is
    /// count-independent — a bound `array<T>` requires at least one element.
    pub min_binding_size: u64,
    /// Always `false`: hair bindings are never bound with a dynamic offset.
    pub has_dynamic_offset: bool,
}

impl HairBindGroupLayoutEntry {
    /// Every hair storage binding is visible only to the compute stage; the
    /// render graph maps this to `wgpu` `ShaderStages::COMPUTE`. Encoded as an
    /// associated constant because the value is invariant across all hair
    /// passes.
    pub const VISIBILITY_IS_COMPUTE: bool = true;
}

/// The push-constant (`var<immediate>`) range a hair compute pass declares.
/// Every hair pass places its immediate block at offset `0` and exposes it to
/// the compute stage; `size` is the block's byte size (from
/// [`params_immediate_bytes`] / [`optional_params_immediate_bytes`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HairPushConstantRange {
    /// Byte offset of the range's start; always `0` for hair passes.
    pub start: u32,
    /// Byte size of the push-constant / immediate block (always a multiple of
    /// four and non-zero for every hair pass).
    pub size: u32,
}

impl HairPushConstantRange {
    /// Hair push-constant ranges are always visible to the compute stage; the
    /// render graph maps this to `wgpu` `ShaderStages::COMPUTE`.
    pub const STAGE_IS_COMPUTE: bool = true;
}

/// The device-independent layout descriptor for one hair compute pipeline: the
/// dense list of `@group(0)` bind-group-layout entries and, when the pass
/// declares one, its push-constant range. This is everything the render graph
/// needs to create a `wgpu` pipeline layout (one bind-group layout plus the
/// push-constant range) that is reused every frame regardless of groom size.
/// Contains a [`Vec`], so it is not `Copy`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct HairComputePipelineLayout {
    /// The single `@group(0)` group's entries, in dense binding order `0..len`.
    pub entries: Vec<HairBindGroupLayoutEntry>,
    /// The push-constant range, or `None` when the pass declares no immediate
    /// block (never the case for the current hair passes, all of which declare
    /// a non-zero immediate block).
    pub push_constant: Option<HairPushConstantRange>,
}

impl HairComputePipelineLayout {
    /// Every hair pass binds exactly one `@group(0)` group, so this is always
    /// `1`.
    #[must_use]
    pub fn bind_group_count(&self) -> u32 {
        1
    }

    /// `true` when the pass declares a push-constant / immediate block.
    #[must_use]
    pub fn has_push_constants(&self) -> bool {
        self.push_constant.is_some()
    }

    /// Number of `@group(0)` bindings in the layout.
    #[must_use]
    pub fn binding_count(&self) -> usize {
        self.entries.len()
    }
}

/// Derives the count-independent bind-group-layout entries from a runtime
/// binding table, keeping only the binding index, storage type and one-element
/// stride (as `min_binding_size`) and discarding the groom-dependent element
/// counts / byte sizes.
fn layout_entries(bindings: &[HairBindingLayout]) -> Vec<HairBindGroupLayoutEntry> {
    let mut entries = Vec::with_capacity(bindings.len());
    for binding in bindings {
        entries.push(HairBindGroupLayoutEntry {
            binding: binding.binding,
            ty: HairBufferBindingType::from_access(binding.access),
            min_binding_size: binding.stride as u64,
            has_dynamic_offset: false,
        });
    }
    entries
}

/// Wraps an immediate-block byte size into a push-constant range, mapping a
/// zero-size block to `None`.
fn push_constant_of(size_bytes: usize) -> Option<HairPushConstantRange> {
    if size_bytes == 0 {
        None
    } else {
        Some(HairPushConstantRange {
            start: 0,
            size: size_bytes as u32,
        })
    }
}

/// Builds the count-independent pipeline layout for a fixed per-frame spine
/// [`HairComputePass`]. The bind-group entries come from [`pass_bindings`]
/// evaluated at [`HairGpuCounts::default`] / [`HairGpuExtents::default`] (the
/// binding index, storage type and stride are count-independent), and the
/// push-constant range from [`params_immediate_bytes`].
#[must_use]
pub fn main_pipeline_layout(pass: HairComputePass) -> HairComputePipelineLayout {
    let mut bindings = Vec::new();
    pass_bindings(
        pass,
        &HairGpuCounts::default(),
        HairGpuExtents::default(),
        &mut bindings,
    );
    HairComputePipelineLayout {
        entries: layout_entries(&bindings),
        push_constant: push_constant_of(params_immediate_bytes(pass)),
    }
}

/// Builds the count-independent pipeline layout for an optional / alternative
/// [`HairOptionalPass`] (`VBD` solve, self-collision accumulate / apply). Mirrors
/// [`main_pipeline_layout`] using [`optional_pass_bindings`] and
/// [`optional_params_immediate_bytes`].
#[must_use]
pub fn optional_pipeline_layout(pass: HairOptionalPass) -> HairComputePipelineLayout {
    let mut bindings = Vec::new();
    optional_pass_bindings(
        pass,
        &HairGpuCounts::default(),
        HairOptionalExtents::default(),
        &mut bindings,
    );
    HairComputePipelineLayout {
        entries: layout_entries(&bindings),
        push_constant: push_constant_of(optional_params_immediate_bytes(pass)),
    }
}

/// Builds the pipeline layout for any scheduled pass, delegating to
/// [`main_pipeline_layout`] or [`optional_pipeline_layout`] so the frame
/// scheduler and the render graph share one layout source.
#[must_use]
pub fn scheduled_pipeline_layout(pass: HairScheduledPass) -> HairComputePipelineLayout {
    match pass {
        HairScheduledPass::Main(main) => main_pipeline_layout(main),
        HairScheduledPass::Optional(optional) => optional_pipeline_layout(optional),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn main_entry_count_matches_binding_count() {
        for pass in HairComputePass::ALL {
            let layout = main_pipeline_layout(pass);
            assert_eq!(layout.entries.len(), pass.binding_count() as usize);
            assert_eq!(layout.binding_count(), pass.binding_count() as usize);
        }
    }

    #[test]
    fn main_bindings_are_dense_and_never_dynamic() {
        for pass in HairComputePass::ALL {
            let layout = main_pipeline_layout(pass);
            for (index, entry) in layout.entries.iter().enumerate() {
                assert_eq!(entry.binding, index as u32);
                assert!(!entry.has_dynamic_offset);
            }
        }
    }

    #[test]
    fn guide_sim_binding_types_and_min_sizes() {
        let layout = main_pipeline_layout(HairComputePass::GuideSim);
        // b0 Positions -> read_write, 16-byte vec4 stride.
        assert_eq!(
            layout.entries[0].ty,
            HairBufferBindingType::StorageReadWrite
        );
        assert_eq!(layout.entries[0].min_binding_size, 16);
        // b2 Goals -> read-only.
        assert_eq!(layout.entries[2].ty, HairBufferBindingType::StorageReadOnly);
        assert_eq!(layout.entries[2].min_binding_size, 16);
        // b3 RestLengths -> read-only f32 stride 4.
        assert_eq!(layout.entries[3].ty, HairBufferBindingType::StorageReadOnly);
        assert_eq!(layout.entries[3].min_binding_size, 4);
        // b5 Colliders -> read-only, 32-byte (two vec4) stride.
        assert_eq!(layout.entries[5].ty, HairBufferBindingType::StorageReadOnly);
        assert_eq!(layout.entries[5].min_binding_size, 32);
    }

    #[test]
    fn main_push_constant_matches_params_and_is_present() {
        for pass in HairComputePass::ALL {
            let layout = main_pipeline_layout(pass);
            let range = layout
                .push_constant
                .expect("every main hair pass declares an immediate block");
            assert_eq!(range.start, 0);
            assert_eq!(range.size, params_immediate_bytes(pass) as u32);
            assert!(range.size > 0);
            assert!(layout.has_push_constants());
        }
    }

    #[test]
    fn entries_are_count_independent() {
        // Re-derive GuideSim entries from a non-default groom and assert the
        // count-independent fields match the default-derived layout.
        let layout = main_pipeline_layout(HairComputePass::GuideSim);
        let counts = HairGpuCounts {
            roots: 7,
            guide_strands: 11,
            guide_particles: 53,
            render_strands: 128,
            light_texels: 4096,
        };
        let extents = HairGpuExtents {
            collider_count: 9,
            ..HairGpuExtents::default()
        };
        let mut bindings = Vec::new();
        pass_bindings(HairComputePass::GuideSim, &counts, extents, &mut bindings);
        let rederived = layout_entries(&bindings);
        assert_eq!(rederived, layout.entries);
    }

    #[test]
    fn vbd_optional_layout() {
        let layout = optional_pipeline_layout(HairOptionalPass::VbdSolve);
        assert_eq!(layout.entries.len(), 6);
        assert_eq!(layout.binding_count(), 6);
        let range = layout
            .push_constant
            .expect("VBD declares an immediate block");
        assert_eq!(range.size, 48);
        assert_eq!(
            range.size,
            optional_params_immediate_bytes(HairOptionalPass::VbdSolve) as u32
        );
    }

    #[test]
    fn self_collision_optional_layout() {
        for pass in [
            HairOptionalPass::SelfCollisionAccumulate,
            HairOptionalPass::SelfCollisionApply,
        ] {
            let layout = optional_pipeline_layout(pass);
            assert_eq!(layout.entries.len(), 5);
            let range = layout
                .push_constant
                .expect("self-collision declares an immediate block");
            assert_eq!(range.size, 20);
            assert_eq!(range.size, optional_params_immediate_bytes(pass) as u32);
            for (index, entry) in layout.entries.iter().enumerate() {
                assert_eq!(entry.binding, index as u32);
                assert!(!entry.has_dynamic_offset);
            }
        }
    }

    #[test]
    fn scheduled_delegates_to_main_and_optional() {
        assert_eq!(
            scheduled_pipeline_layout(HairScheduledPass::Main(HairComputePass::GuideSim)),
            main_pipeline_layout(HairComputePass::GuideSim)
        );
        assert_eq!(
            scheduled_pipeline_layout(HairScheduledPass::Optional(HairOptionalPass::VbdSolve)),
            optional_pipeline_layout(HairOptionalPass::VbdSolve)
        );
    }

    #[test]
    fn bind_group_count_is_always_one() {
        for pass in HairComputePass::ALL {
            assert_eq!(main_pipeline_layout(pass).bind_group_count(), 1);
        }
        for pass in HairOptionalPass::ALL {
            assert_eq!(optional_pipeline_layout(pass).bind_group_count(), 1);
        }
    }

    #[test]
    fn binding_type_maps_from_access() {
        assert_eq!(
            HairBufferBindingType::from_access(HairBufferAccess::Read),
            HairBufferBindingType::StorageReadOnly
        );
        assert_eq!(
            HairBufferBindingType::from_access(HairBufferAccess::ReadWrite),
            HairBufferBindingType::StorageReadWrite
        );
        assert!(HairBufferBindingType::StorageReadOnly.is_read_only());
        assert!(!HairBufferBindingType::StorageReadWrite.is_read_only());
    }
}
