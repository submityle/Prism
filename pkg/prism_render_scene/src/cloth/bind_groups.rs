//! Per-piece `GPU` buffer allocation and bind-group assembly for the cloth
//! subsystem.
//!
//! [`super::pipeline`] built the eleven compute pipelines and the five group-0
//! layouts once at startup; this slice turns one cloth piece's CPU-golden solver
//! state into the resident device buffers those pipelines read and write, and
//! wires them into the five bind groups the dispatch slice records against.
//!
//! The buffer set mirrors the golden sizing contract in
//! [`prism_render_architecture::cloth::gpu::buffers`]: three particle pools
//! (`positions` / `velocities` / `prev_positions`, each `array<vec4<f32>>` with
//! the inverse mass packed into `.w`), the two read-only constraint arrays, the
//! analytic collider list, the self-collision spatial-hash cell table plus its
//! per-particle linked-list `next` array, the painted-backstop planes, the
//! render-vertex embed bindings with their output position pool, and one uniform
//! block per pass. Each record is uploaded with a plain `bytemuck` cast of the
//! `#[repr(C)]` mirrors in [`super::abi`], so the host bytes and the `WESL`
//! `struct`s stay layout-identical (the `size_of` contract tests in that module
//! pin the strides).
//!
//! An empty optional array (no colliders, no bending hinges, no backstops on a
//! piece) is padded up to a single zeroed element so the binding is always
//! valid; the matching per-pass uniform count is zero, so the dispatch bounds
//! check discards the placeholder before it is ever read.

#![allow(
    dead_code,
    reason = "the per-piece cloth buffer set and its five bind groups are consumed by the Core3d dispatch slice and the plugin that lands next; the device-free buffer-plan sizing is exercised by the contract tests below"
)]

use bevy_render::{
    render_resource::{
        BindGroup, BindGroupEntries, Buffer, BufferDescriptor, BufferInitDescriptor, BufferUsages,
    },
    renderer::RenderDevice,
};
use bytemuck::Pod;
use prism_render_architecture::cloth::gpu::buffers::{
    BufferCounts, PersistentBufferSet, BACKSTOP_STRIDE, CONSTRAINT_STRIDE, EMBED_STRIDE,
    HASH_CELL_STRIDE, PARTICLE_VEC_STRIDE,
};

use super::abi::{
    GpuClothBackstop, GpuClothBackstopParams, GpuClothBendingConstraint, GpuClothBodyParams,
    GpuClothCollider, GpuClothConstraint, GpuClothEmbedBinding, GpuClothEmbedParams,
    GpuClothSelfParams, GpuClothSimParams,
};
use super::pipeline::ClothComputePipelines;

/// The packed byte size of one analytic collider proxy ([`GpuClothCollider`]).
/// Colliders are scene-supplied per solve rather than sized by
/// [`BufferCounts`], so the stride lives here next to the plan that consumes it.
pub(crate) const COLLIDER_STRIDE: u32 = size_of::<GpuClothCollider>() as u32;

/// The packed byte size of one per-particle spatial-hash `next` link (a single
/// `u32` particle index).
pub(crate) const HASH_NEXT_STRIDE: u32 = 4;

/// Minimum storage-buffer size in bytes.
///
/// `wgpu` rejects a zero-sized storage buffer, so every resident buffer is
/// clamped to at least one 16-byte `std430` row even when its element count is
/// zero. The placeholder is never read: the owning pass's uniform count is zero
/// and its per-element bounds check discards the sole padded element.
const MIN_STORAGE_BYTES: u64 = 16;

/// The CPU-golden host state uploaded to build one piece's resident buffers.
///
/// Every slice is a `#[repr(C)]` mirror ready for a `bytemuck` cast; the
/// packing from the architecture-layer solver types into these mirrors is the
/// extract slice's job, so this factory stays a pure "bytes → device buffers"
/// step that can be reasoned about without the golden solver in scope.
pub(crate) struct ClothPieceUpload<'a> {
    /// Particle positions with inverse mass packed into `.w` (`<= 0` = pinned).
    pub(crate) positions: &'a [[f32; 4]],
    /// Particle velocities; `.w` is unused padding kept for the 16-byte stride.
    pub(crate) velocities: &'a [[f32; 4]],
    /// Graph-colored distance constraints (stretch / shear), all colors packed.
    pub(crate) constraints: &'a [GpuClothConstraint],
    /// Dihedral bending hinges; empty when the piece has no bending energy.
    pub(crate) bending: &'a [GpuClothBendingConstraint],
    /// Analytic body-collision proxies; empty when the piece hits no colliders.
    pub(crate) colliders: &'a [GpuClothCollider],
    /// Painted backstop planes, one per constrained particle; empty when unused.
    pub(crate) backstops: &'a [GpuClothBackstop],
    /// Render-vertex embed bindings driving the skinning pass; empty when the
    /// piece renders its sim mesh directly.
    pub(crate) embed_bindings: &'a [GpuClothEmbedBinding],
    /// Number of render-mesh vertices (sizes the embed output position pool).
    pub(crate) render_vertex_count: u32,
    /// Number of self-collision hash cells (sizes the cell-header table).
    pub(crate) hash_cell_count: u32,
    /// Initial per-substep solver uniform for the `cloth_sim.wesl` passes.
    pub(crate) sim_params: GpuClothSimParams,
    /// Initial body-collision dispatch uniform.
    pub(crate) body_params: GpuClothBodyParams,
    /// Initial self-collision dispatch uniform.
    pub(crate) self_params: GpuClothSelfParams,
    /// Initial backstop dispatch uniform.
    pub(crate) backstop_params: GpuClothBackstopParams,
    /// Initial skin-embed dispatch uniform.
    pub(crate) embed_params: GpuClothEmbedParams,
}

/// Every resident device buffer for one cloth piece.
///
/// Built once from a [`ClothPieceUpload`] and kept across frames; the solver
/// dispatches read and write them in place. The uniform buffers are re-written
/// each frame with fresh per-substep scalars via [`RenderQueue`] writes owned by
/// the dispatch slice.
pub(crate) struct ClothPieceGpuBuffers {
    /// Particle positions (read-write; `.w` = inverse mass).
    pub(crate) positions: Buffer,
    /// Particle velocities (read-write).
    pub(crate) velocities: Buffer,
    /// Pre-substep position snapshot (read-write) driving velocity recovery.
    pub(crate) prev_positions: Buffer,
    /// Packed distance constraints (read-only).
    pub(crate) constraints: Buffer,
    /// Packed dihedral bending hinges (read-only).
    pub(crate) bending: Buffer,
    /// Analytic collision proxies (read-only).
    pub(crate) colliders: Buffer,
    /// Self-collision hash cell-header table (read-write).
    pub(crate) hash_cells: Buffer,
    /// Per-particle spatial-hash linked-list `next` array (read-write).
    pub(crate) particle_next: Buffer,
    /// Painted backstop planes (read-only).
    pub(crate) backstops: Buffer,
    /// Skinned render-vertex positions (read-write output of the embed pass).
    pub(crate) render_positions: Buffer,
    /// Render-vertex embed bindings (read-only).
    pub(crate) embed_bindings: Buffer,
    /// `cloth_sim.wesl` per-substep uniform.
    pub(crate) sim_params: Buffer,
    /// Body-collision pass uniform.
    pub(crate) body_params: Buffer,
    /// Self-collision pass uniform.
    pub(crate) self_params: Buffer,
    /// Backstop pass uniform.
    pub(crate) backstop_params: Buffer,
    /// Skin-embed pass uniform.
    pub(crate) embed_params: Buffer,
}

impl ClothPieceGpuBuffers {
    /// Allocates and uploads every resident buffer for one piece.
    ///
    /// The three particle pools carry `COPY_SRC` so the async readback slice can
    /// copy simulated positions back for a `CPU` fallback path or debugging; the
    /// remaining storage buffers are device-local (`STORAGE | COPY_DST`). Empty
    /// optional arrays pad to one zeroed element (see [`MIN_STORAGE_BYTES`]).
    pub(crate) fn create(device: &RenderDevice, upload: &ClothPieceUpload<'_>) -> Self {
        let positions = readable_storage(device, "prism cloth positions", upload.positions);
        let velocities = readable_storage(device, "prism cloth velocities", upload.velocities);
        // The pre-step snapshot starts equal to the initial positions so the
        // first frame's velocity-recovery pass reads a coherent delta of zero.
        let prev_positions =
            readable_storage(device, "prism cloth prev positions", upload.positions);

        let constraints = read_only_storage(device, "prism cloth constraints", upload.constraints);
        let bending = read_only_storage(device, "prism cloth bending", upload.bending);
        let colliders = read_only_storage(device, "prism cloth colliders", upload.colliders);
        let backstops = read_only_storage(device, "prism cloth backstops", upload.backstops);
        let embed_bindings =
            read_only_storage(device, "prism cloth embed bindings", upload.embed_bindings);

        // The hash table and the per-particle `next` links are produced by the
        // build pass every frame, so they start zeroed rather than uploaded.
        let hash_cells = zeroed_storage(
            device,
            "prism cloth hash cells",
            u64::from(upload.hash_cell_count) * u64::from(HASH_CELL_STRIDE),
        );
        let particle_next = zeroed_storage(
            device,
            "prism cloth particle next",
            u64::from(upload.positions.len() as u32) * u64::from(HASH_NEXT_STRIDE),
        );
        // The embed pass writes the skinned render vertices; the pool starts
        // zeroed and is fully overwritten on the first skinning dispatch.
        let render_positions = zeroed_storage(
            device,
            "prism cloth render positions",
            u64::from(upload.render_vertex_count) * u64::from(PARTICLE_VEC_STRIDE),
        );

        let sim_params = uniform(device, "prism cloth sim params", &upload.sim_params);
        let body_params = uniform(device, "prism cloth body params", &upload.body_params);
        let self_params = uniform(device, "prism cloth self params", &upload.self_params);
        let backstop_params = uniform(
            device,
            "prism cloth backstop params",
            &upload.backstop_params,
        );
        let embed_params = uniform(device, "prism cloth embed params", &upload.embed_params);

        Self {
            positions,
            velocities,
            prev_positions,
            constraints,
            bending,
            colliders,
            hash_cells,
            particle_next,
            backstops,
            render_positions,
            embed_bindings,
            sim_params,
            body_params,
            self_params,
            backstop_params,
            embed_params,
        }
    }
}

/// The five group-0 bind groups one piece dispatches against.
///
/// Present only once its backing [`ClothPieceGpuBuffers`] exists; the dispatch
/// node treats a present set as "safe to record". Each group's entry order
/// matches its layout exactly (see [`super::pipeline`]).
pub(crate) struct ClothPieceBindGroups {
    /// group 0 for all six `cloth_sim.wesl` kernels.
    pub(crate) sim: BindGroup,
    /// group 0 for the `cloth_body_collision` pass.
    pub(crate) body: BindGroup,
    /// group 0 for both self-collision passes (hash build + resolve).
    pub(crate) self_collision: BindGroup,
    /// group 0 for the `cloth_backstop` pass.
    pub(crate) backstop: BindGroup,
    /// group 0 for the `cloth_skin_embed` pass.
    pub(crate) embed: BindGroup,
}

impl ClothPieceBindGroups {
    /// Builds the five bind groups binding `buffers` against the shared
    /// pipeline layouts.
    ///
    /// The `positions` buffer is bound read-write by the sim / body / self /
    /// backstop groups and read-only (as the skinning source) by the embed
    /// group; a single storage buffer satisfies both because it is allocated
    /// with `STORAGE` usage and each group declares its own access mode.
    pub(crate) fn create(
        device: &RenderDevice,
        pipelines: &ClothComputePipelines,
        buffers: &ClothPieceGpuBuffers,
    ) -> Self {
        let sim = device.create_bind_group(
            "prism cloth sim",
            &pipelines.sim_layout,
            &BindGroupEntries::sequential((
                buffers.positions.as_entire_binding(),
                buffers.velocities.as_entire_binding(),
                buffers.prev_positions.as_entire_binding(),
                buffers.constraints.as_entire_binding(),
                buffers.bending.as_entire_binding(),
                buffers.sim_params.as_entire_binding(),
            )),
        );
        let body = device.create_bind_group(
            "prism cloth body",
            &pipelines.body_layout,
            &BindGroupEntries::sequential((
                buffers.positions.as_entire_binding(),
                buffers.colliders.as_entire_binding(),
                buffers.body_params.as_entire_binding(),
            )),
        );
        let self_collision = device.create_bind_group(
            "prism cloth self",
            &pipelines.self_layout,
            &BindGroupEntries::sequential((
                buffers.positions.as_entire_binding(),
                buffers.hash_cells.as_entire_binding(),
                buffers.particle_next.as_entire_binding(),
                buffers.self_params.as_entire_binding(),
            )),
        );
        let backstop = device.create_bind_group(
            "prism cloth backstop",
            &pipelines.backstop_layout,
            &BindGroupEntries::sequential((
                buffers.positions.as_entire_binding(),
                buffers.backstops.as_entire_binding(),
                buffers.backstop_params.as_entire_binding(),
            )),
        );
        let embed = device.create_bind_group(
            "prism cloth embed",
            &pipelines.embed_layout,
            &BindGroupEntries::sequential((
                buffers.positions.as_entire_binding(),
                buffers.render_positions.as_entire_binding(),
                buffers.embed_bindings.as_entire_binding(),
                buffers.embed_params.as_entire_binding(),
            )),
        );
        Self {
            sim,
            body,
            self_collision,
            backstop,
            embed,
        }
    }
}

/// Uploads a read-write storage buffer that is also copyable back to the host
/// (`STORAGE | COPY_DST | COPY_SRC`), padding an empty slice to one zeroed
/// element so the binding is valid.
fn readable_storage<T: Pod>(device: &RenderDevice, label: &str, data: &[T]) -> Buffer {
    storage_with_data(
        device,
        label,
        data,
        BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
    )
}

/// Uploads a device-local read-only storage buffer (`STORAGE | COPY_DST`),
/// padding an empty slice to one zeroed element so the binding is valid.
fn read_only_storage<T: Pod>(device: &RenderDevice, label: &str, data: &[T]) -> Buffer {
    storage_with_data(
        device,
        label,
        data,
        BufferUsages::STORAGE | BufferUsages::COPY_DST,
    )
}

/// Uploads `data` (or one zeroed element when empty) into a storage buffer with
/// the given usage.
fn storage_with_data<T: Pod>(
    device: &RenderDevice,
    label: &str,
    data: &[T],
    usage: BufferUsages,
) -> Buffer {
    let placeholder = [T::zeroed()];
    let contents: &[T] = if data.is_empty() { &placeholder } else { data };
    device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(contents),
        usage,
    })
}

/// Allocates a zeroed device-local read-write storage buffer of `byte_len`
/// bytes (clamped to at least one `std430` row), for the pass-produced hash and
/// render-position pools.
fn zeroed_storage(device: &RenderDevice, label: &str, byte_len: u64) -> Buffer {
    device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size: byte_len.max(MIN_STORAGE_BYTES),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

/// Uploads a `#[repr(C)]` uniform record into a `UNIFORM | COPY_DST` buffer.
fn uniform<T: Pod>(device: &RenderDevice, label: &str, value: &T) -> Buffer {
    device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::bytes_of(value),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    })
}

/// Device-free byte-size plan for one piece's resident buffers.
///
/// Wraps the golden [`PersistentBufferSet`] (which sizes the particle pools,
/// constraint, hash and embed buffers from a [`BufferCounts`]) and adds the two
/// counts the golden contract leaves to the render layer: the analytic collider
/// list (scene-supplied per solve) and the storage-buffer floor every binding is
/// clamped to. Kept pure so the sizing can be unit-tested without a
/// [`RenderDevice`].
pub(crate) struct ClothBufferPlan {
    set: PersistentBufferSet,
    collider_count: u32,
}

impl ClothBufferPlan {
    /// Builds a plan from the golden element counts plus the collider count.
    #[must_use]
    pub(crate) fn new(counts: BufferCounts, collider_count: u32) -> Self {
        Self {
            set: PersistentBufferSet::new(counts),
            collider_count,
        }
    }

    /// Bytes for one particle position pool (`positions` and `prev_positions`
    /// each allocate this size).
    #[must_use]
    pub(crate) fn position_bytes(&self) -> u64 {
        clamp_storage(self.set.position_bytes())
    }

    /// Bytes for the packed distance-constraint buffer.
    #[must_use]
    pub(crate) fn constraint_bytes(&self) -> u64 {
        clamp_storage(self.set.constraint_bytes())
    }

    /// Bytes for the self-collision hash cell-header table.
    #[must_use]
    pub(crate) fn hash_cell_bytes(&self) -> u64 {
        clamp_storage(self.set.hash_cell_bytes())
    }

    /// Bytes for the per-particle spatial-hash `next` array.
    #[must_use]
    pub(crate) fn hash_next_bytes(&self) -> u64 {
        clamp_storage(self.set.hash_entry_bytes())
    }

    /// Bytes for the render-vertex embed-binding buffer.
    #[must_use]
    pub(crate) fn embed_bytes(&self) -> u64 {
        clamp_storage(self.set.embed_bytes())
    }

    /// Bytes for the painted-backstop plane buffer.
    #[must_use]
    pub(crate) fn backstop_bytes(&self) -> u64 {
        clamp_storage(self.set.backstop_bytes())
    }

    /// Bytes for the analytic collider list.
    #[must_use]
    pub(crate) fn collider_bytes(&self) -> u64 {
        clamp_storage(self.collider_count.saturating_mul(COLLIDER_STRIDE))
    }
}

/// Clamps a golden byte size up to the storage-buffer floor, matching the
/// runtime padding [`zeroed_storage`] and [`storage_with_data`] apply.
#[must_use]
fn clamp_storage(bytes: u32) -> u64 {
    u64::from(bytes).max(MIN_STORAGE_BYTES)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counts() -> BufferCounts {
        BufferCounts {
            particles: 100,
            constraints: 400,
            hash_cells: 64,
            render_vertices: 500,
            backstops: 100,
        }
    }

    /// The collider stride the plan sizes against must equal the packed
    /// `#[repr(C)]` collider record the `bytemuck` upload casts.
    #[test]
    fn collider_stride_matches_abi_record() {
        assert_eq!(COLLIDER_STRIDE, size_of::<GpuClothCollider>() as u32);
    }

    /// A populated plan reproduces the golden per-buffer byte sizes so the
    /// render allocation can never drift from the architecture-layer contract.
    #[test]
    fn plan_reproduces_golden_byte_sizes() {
        let plan = ClothBufferPlan::new(counts(), 8);
        let golden = PersistentBufferSet::new(counts());
        assert_eq!(plan.position_bytes(), u64::from(golden.position_bytes()));
        assert_eq!(
            plan.constraint_bytes(),
            u64::from(golden.constraint_bytes())
        );
        assert_eq!(plan.hash_cell_bytes(), u64::from(golden.hash_cell_bytes()));
        assert_eq!(plan.hash_next_bytes(), u64::from(golden.hash_entry_bytes()));
        assert_eq!(plan.embed_bytes(), u64::from(golden.embed_bytes()));
        assert_eq!(plan.backstop_bytes(), u64::from(golden.backstop_bytes()));
        assert_eq!(
            plan.collider_bytes(),
            u64::from(8u32.saturating_mul(COLLIDER_STRIDE))
        );
    }

    /// Positions and previous positions share the same per-copy size, matching
    /// the golden double-buffered position pool.
    #[test]
    fn position_and_prev_share_one_copy_size() {
        let plan = ClothBufferPlan::new(counts(), 0);
        assert_eq!(plan.position_bytes(), u64::from(100 * PARTICLE_VEC_STRIDE));
    }

    /// An empty piece clamps every buffer to the storage floor so no binding is
    /// ever zero-sized, mirroring the runtime placeholder padding.
    #[test]
    fn empty_plan_clamps_to_storage_floor() {
        let plan = ClothBufferPlan::new(BufferCounts::default(), 0);
        assert_eq!(plan.position_bytes(), MIN_STORAGE_BYTES);
        assert_eq!(plan.constraint_bytes(), MIN_STORAGE_BYTES);
        assert_eq!(plan.hash_cell_bytes(), MIN_STORAGE_BYTES);
        assert_eq!(plan.hash_next_bytes(), MIN_STORAGE_BYTES);
        assert_eq!(plan.embed_bytes(), MIN_STORAGE_BYTES);
        assert_eq!(plan.backstop_bytes(), MIN_STORAGE_BYTES);
        assert_eq!(plan.collider_bytes(), MIN_STORAGE_BYTES);
    }

    /// The `CONSTRAINT_STRIDE` / `BACKSTOP_STRIDE` / `EMBED_STRIDE` imports are
    /// the golden strides the plan's byte sizes derive from; pin them so a
    /// golden drift is caught here rather than at dispatch time.
    #[test]
    fn golden_strides_are_stable() {
        assert_eq!(CONSTRAINT_STRIDE, 16);
        assert_eq!(BACKSTOP_STRIDE, 32);
        assert_eq!(EMBED_STRIDE, 32);
        assert_eq!(HASH_CELL_STRIDE, 8);
        assert_eq!(PARTICLE_VEC_STRIDE, 16);
    }
}
