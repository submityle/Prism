//! Bin-driven queueing: packed draw bins → queued indirect draws.
//!
//! `prism_render_visibility::binning` groups the visible set into deterministic
//! draw bins and packs them into one command stream: one indirect-argument slot
//! per instance, one header per (pipeline class, geometry, LOD) group. This
//! module is the app-side half of that contract. It turns each bin's slot range
//! into the [`IndirectArgs`] a multi-draw reads, resolves the bin's pipeline
//! through the same curve the direct path uses ([`SceneRegistry::bin_variant`]
//! rebuilds the full permutation from the bin's compact class, so a material
//! specializes to *one* pipeline, not two), and appends one [`PhaseItem`] to the
//! phase its packed class names.
//!
//! The two halves must agree on the slot arithmetic: the offset is
//! `(command_buffer_start + bin.command_start) * DRAW_INDEXED_INDIRECT_ARGS_SIZE`,
//! exactly what `GpuDrawBinHeader::from_range_at` publishes to the GPU, so the
//! header a shader reads and the CPU's queued range cannot drift.
//!
//! The bin path binds geometry and pipeline per bin, but the *material* it draws
//! with is per instance: a bin batches every material that shares the
//! permutation class, so the per-instance material index (GPU scene) is what
//! selects the bindings at draw time. The item's `material` is the bin's
//! representative — the instance the header was built from — which is what a
//! non-bindless fallback and diagnostics use.

use prism_render_architecture::gpu_scene::CpuRenderScene;
use prism_render_driver::{RenderDevice, TextureFormat, DRAW_INDEXED_INDIRECT_ARGS_SIZE};
use prism_render_graph::BufferHandle;
use prism_render_visibility::{DrawBinRange, ViewDrawBins};

use crate::draw::DrawFunctionId;
use crate::mesh_pipeline::{MeshPipelineProvider, MeshPipelineSpecializer};
use crate::queue::{phase_for_bin_class, IndirectArgs, PhaseItem, RenderQueues};
use crate::scene::SceneRegistry;

/// The indirect argument range one packed bin owns.
///
/// `command_base` is the stream's [`ViewDrawBins::command_buffer_start`] after
/// [`pack_draw_bin_streams`](prism_render_visibility::pack_draw_bin_streams):
/// bins store stream-local slots, and the shared argument buffer is what the
/// absolute slot is relative to. The addition saturates, mirroring
/// `GpuDrawBinHeader::from_range_at`, so a pathological total cannot wrap into
/// another stream's arguments.
#[must_use]
pub fn bin_indirect_args(
    args: BufferHandle,
    range: &DrawBinRange,
    command_base: u32,
) -> IndirectArgs {
    IndirectArgs {
        args,
        offset: u64::from(command_base.saturating_add(range.command_start))
            * u64::from(DRAW_INDEXED_INDIRECT_ARGS_SIZE),
        count: range.command_capacity,
        stride: DRAW_INDEXED_INDIRECT_ARGS_SIZE,
    }
}

/// What queueing one bin stream did.
///
/// A caller can tell "nothing was visible" (`queued == 0` with no skips) from
/// "the stream and the registry disagree" (a non-zero skip count), which is what
/// an asset-streaming bug looks like from here.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct BinQueueOutcome {
    /// Bins queued as indirect draws.
    pub queued: usize,
    /// Bins skipped because their packed class did not name a phase.
    pub unknown_class: usize,
    /// Bins skipped because the pipeline or the representative instance was
    /// missing from the registry.
    pub unresolved: usize,
}

impl BinQueueOutcome {
    /// The number of bins the stream contained.
    #[must_use]
    pub const fn total(self) -> usize {
        self.queued + self.unknown_class + self.unresolved
    }
}

impl RenderQueues {
    /// Queues a packed bin stream as indirect draws, one item per bin.
    ///
    /// Each bin becomes one [`PhaseItem`] whose [`IndirectArgs`] cover the bin's
    /// slot range, so the bin draw function issues a single multi-draw for the
    /// whole bin and the CPU never learns the instance count (the cull pass
    /// writes it). The item carries no element range: `first`, `count`, and
    /// `base_vertex` live in the argument slot, and `lod_level` is the stream's
    /// cluster/LOD selector for diagnostics.
    ///
    /// Items are ordered by their absolute command slot, so the queue records
    /// bins in the same order the argument buffer stores them. A bin whose class
    /// does not decode, or whose pipeline/instance is not resident, is skipped
    /// and counted in the returned [`BinQueueOutcome`].
    pub fn fill_bins<D: RenderDevice, P: MeshPipelineProvider>(
        &mut self,
        device: &D,
        specializer: &mut MeshPipelineSpecializer,
        registry: &SceneRegistry,
        meshes: &crate::RenderAssets<crate::RenderMesh>,
        instances: &CpuRenderScene,
        args: BufferHandle,
        bins: &ViewDrawBins,
        depth_format: Option<TextureFormat>,
        sample_count: u32,
        draw: DrawFunctionId,
        provider: &P,
    ) -> BinQueueOutcome {
        let mut outcome = BinQueueOutcome::default();
        for range in &bins.bins {
            let Some(phase) = phase_for_bin_class(range.key.pipeline_class) else {
                outcome.unknown_class += 1;
                continue;
            };
            let (Some(variant), Some(mesh_class)) = (
                registry.bin_variant(&range.key, depth_format, sample_count),
                registry.mesh_class(range.key.vertex_buffer_class),
            ) else {
                outcome.unresolved += 1;
                continue;
            };
            // A class is a vertex stage; the buffers are the geometry's mesh's.
            let Some(buffers) = registry
                .geometry_mesh(range.key.geometry)
                .and_then(|mesh| meshes.get(mesh))
                .map(|prepared| prepared.buffers.clone())
            else {
                outcome.unresolved += 1;
                continue;
            };
            let Some(instance) = instances.get(range.representative_scene) else {
                outcome.unresolved += 1;
                continue;
            };
            let pipeline = specializer
                .specialize_with_provider(device, variant, &mesh_class.layout, provider)
                .id();
            self.push(
                phase,
                PhaseItem {
                    pipeline,
                    draw,
                    geometry: range.key.geometry,
                    vertices: buffers.vertices,
                    indices: buffers.indices,
                    first: 0,
                    count: 0,
                    base_vertex: 0,
                    indirect: Some(bin_indirect_args(args, range, bins.command_buffer_start)),
                    lod_level: range.key.lod_or_cluster,
                    material: instance.material,
                    scene: range.representative_scene,
                    order: u64::from(
                        bins.command_buffer_start
                            .saturating_add(range.command_start),
                    ),
                },
            );
            outcome.queued += 1;
        }
        outcome
    }
}
