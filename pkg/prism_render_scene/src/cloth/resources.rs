//! Render-world resources that hold the resident `GPU` cloth pieces.
//!
//! The [`ClothComputePipelines`](super::pipeline::ClothComputePipelines) own the
//! pipelines and layouts once for the whole app, and
//! [`bind_groups`](super::bind_groups) knows how to build one piece's resident
//! buffers and five bind groups. This module is the render-world container that
//! ties a piece's bind groups to the ordered golden dispatch schedule the
//! [`dispatch`](super::dispatch) node records.
//!
//! The resource is deliberately a plain `Vec` of pieces: the extract stage
//! (which snapshots the main-world cloth garments into these render resources)
//! rebuilds it each frame, and an empty vector makes the dispatch node a
//! genuine no-op rather than a fake solve.

use bevy_ecs::entity::Entity;
use bevy_ecs::resource::Resource;
use bevy_platform::collections::{HashMap, HashSet};

use prism_render_architecture::cloth::gpu::pipeline::PlannedDispatch;

use super::bind_groups::{ClothPieceBindGroups, ClothPieceGpuBuffers, ClothPieceUpload};

/// One resident cloth piece the dispatch node can record a frame's solve for.
///
/// Owns the resident buffers (so their `wgpu` handles outlive the bind groups
/// that reference them), the five bind groups the eleven kernels dispatch
/// against, and the flat, ordered [`PlannedDispatch`] schedule the golden
/// `prepare` stage produced for this piece's colored constraint graph and
/// substep/iteration counts.
pub(crate) struct ClothGpuPiece {
    /// The resident buffer set backing every bind group of this piece.
    ///
    /// Held to keep the `wgpu` buffer handles alive for as long as the bind
    /// groups that reference them, and — for a piece that persists across frames
    /// — to receive the per-frame dynamic-input rewrites through
    /// [`ClothPieceGpuBuffers::write_dynamic`]. The dispatch node still binds
    /// through the derived bind groups rather than this field directly.
    pub(crate) buffers: ClothPieceGpuBuffers,
    /// The seven group-0 bind groups, one per shader-interface layout.
    pub(crate) bind_groups: ClothPieceBindGroups,
    /// The ordered dispatch schedule in exact golden record order.
    pub(crate) dispatches: Vec<PlannedDispatch>,
    /// The buffer-topology fingerprint this piece was allocated for. The prepare
    /// stage reuses this resident piece only while a re-derived signature still
    /// matches; any change (a re-authored mesh or an LOD tier swap that resizes a
    /// pool) forces a fresh allocation instead of a rewrite.
    pub(crate) signature: ClothPieceSignature,
}

impl ClothGpuPiece {
    /// Builds a piece from its resident buffers, bind groups and golden
    /// schedule. Kept explicit (rather than a struct literal at the call site)
    /// so the extract stage constructs pieces through one documented entry.
    #[must_use]
    pub(crate) fn new(
        buffers: ClothPieceGpuBuffers,
        bind_groups: ClothPieceBindGroups,
        dispatches: Vec<PlannedDispatch>,
        signature: ClothPieceSignature,
    ) -> Self {
        Self {
            buffers,
            bind_groups,
            dispatches,
            signature,
        }
    }
}

/// The buffer-topology fingerprint that decides whether a resident cloth piece
/// can be reused across frames.
///
/// A persistent piece keeps its simulation state resident on the device and only
/// restreams the small per-frame dynamic inputs (see
/// [`ClothPieceGpuBuffers::write_dynamic`]). That in-place reuse is only sound
/// while every resident buffer is still exactly the right size for this frame's
/// upload, so the prepare stage compares this fingerprint — one element count per
/// resident pool — and falls back to a fresh allocation the moment any pool would
/// need to grow or shrink (a re-authored mesh, or an LOD tier swap that changes
/// the simulated resolution). Reuse when equal is therefore always memory-safe:
/// each streamed write fits the existing allocation exactly.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ClothPieceSignature {
    /// Particle count (sizes the position, velocity and `prev` pools; the
    /// velocity buffer always mirrors the position buffer's length).
    positions: usize,
    /// Distance/attachment constraint record count.
    constraints: usize,
    /// Dihedral bending hinge count.
    bending: usize,
    /// Analytic body-collider proxy count.
    colliders: usize,
    /// Painted-backstop plane count.
    backstops: usize,
    /// Render-vertex embed binding count.
    embed_bindings: usize,
    /// Sim-mesh triangle count.
    triangles: usize,
    /// `CSR` adjacency offset-array length.
    csr_offsets: usize,
    /// `CSR` adjacency entry-array length.
    csr_entries: usize,
    /// Self-collision hash-cell count (sizes the hash-cell table).
    hash_cell_count: u32,
    /// Render-mesh vertex count (sizes the embed output pool).
    render_vertex_count: u32,
}

impl ClothPieceSignature {
    /// Derives the fingerprint from the exact upload that would build the resident
    /// buffers, so the reuse gate is defined by the one type that also drives the
    /// allocation. Keeping this next to the reuse logic means the set of pools
    /// that must match to reuse cannot silently drift from the set that
    /// [`ClothPieceGpuBuffers::create`] actually allocates.
    #[must_use]
    pub(crate) fn from_upload(upload: &ClothPieceUpload<'_>) -> Self {
        Self {
            positions: upload.positions.len(),
            constraints: upload.constraints.len(),
            bending: upload.bending.len(),
            colliders: upload.colliders.len(),
            backstops: upload.backstops.len(),
            embed_bindings: upload.embed_bindings.len(),
            triangles: upload.triangles.len(),
            csr_offsets: upload.csr_offsets.len(),
            csr_entries: upload.csr_entries.len(),
            hash_cell_count: upload.hash_cell_count,
            render_vertex_count: upload.render_vertex_count,
        }
    }
}

/// The render-world set of resident `GPU` cloth pieces, persisted across frames.
///
/// Pieces are keyed by their main-world [`Entity`] so a garment's device state
/// survives from one frame to the next: the [`prepare_cloth_pieces`] stage reuses
/// a resident piece in place (evolving its simulation state on the `GPU`) whenever
/// the buffer topology is unchanged, allocates a fresh piece when a garment first
/// appears or its topology changes, and evicts pieces whose entity despawned. The
/// separate `order` list records which entities produced a schedulable piece
/// *this* frame, in extract iteration order, so the
/// [`dispatch_cloth`](super::dispatch::dispatch_cloth) node walks a deterministic
/// sequence while the resident map may also hold gated-out garments (LOD-collapsed
/// or budget-deferred) whose state is being preserved for a later frame.
///
/// [`prepare_cloth_pieces`]: super::prepare::prepare_cloth_pieces
#[derive(Resource, Default)]
pub(crate) struct ClothGpuPieces {
    /// Every resident piece, keyed by the garment's stable main-world entity.
    /// Owns the persistent `GPU` buffers across frames.
    resident: HashMap<Entity, ClothGpuPiece>,
    /// The entities that produced a schedulable piece this frame, in extract
    /// order. A subset of `resident`'s keys (gated-out garments stay resident but
    /// are absent here), and the exact set the dispatch node records.
    order: Vec<Entity>,
}

impl ClothGpuPieces {
    /// Returns `true` when no piece is scheduled this frame, so the dispatch node
    /// can skip recording a compute pass entirely. Resident-but-gated garments do
    /// not count: they hold state without scheduling work.
    #[must_use]
    pub(crate) fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    /// Opens a new frame: clears the per-frame dispatch order while keeping every
    /// resident piece, so a reused piece can continue evolving its simulation
    /// state in place.
    pub(crate) fn begin_frame(&mut self) {
        self.order.clear();
    }

    /// Returns the resident piece for `entity`, if one is cached, for the prepare
    /// stage to test its signature and, on a match, restream its dynamic inputs.
    pub(crate) fn resident_mut(&mut self, entity: Entity) -> Option<&mut ClothGpuPiece> {
        self.resident.get_mut(&entity)
    }

    /// Records that a resident piece is scheduled this frame (the reuse path,
    /// after its dynamic inputs were restreamed in place).
    pub(crate) fn mark_active(&mut self, entity: Entity) {
        self.order.push(entity);
    }

    /// Installs a freshly built piece and schedules it this frame (the allocate
    /// path, for a new garment or a changed topology).
    pub(crate) fn install(&mut self, entity: Entity, piece: ClothGpuPiece) {
        self.resident.insert(entity, piece);
        self.order.push(entity);
    }

    /// Evicts resident pieces whose entity is absent from `live` (the garments
    /// still present in this frame's extracted set). A garment that is merely
    /// gated out this frame stays in `live`, so its resident state is preserved
    /// and re-offered next frame; only a genuinely despawned garment is dropped,
    /// freeing its `GPU` buffers.
    pub(crate) fn retain_live(&mut self, live: &HashSet<Entity>) {
        self.resident.retain(|entity, _| live.contains(entity));
    }

    /// Iterates this frame's scheduled pieces in deterministic dispatch order.
    /// Every entity in `order` was installed or marked active this frame, so the
    /// lookup never misses; the `filter_map` simply keeps the borrow ergonomic.
    pub(crate) fn active_pieces(&self) -> impl Iterator<Item = &ClothGpuPiece> + '_ {
        self.order
            .iter()
            .filter_map(|entity| self.resident.get(entity))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::cloth::abi::{
        GpuClothAeroParams, GpuClothBackstop, GpuClothBackstopParams, GpuClothBendingConstraint,
        GpuClothBodyParams, GpuClothCollider, GpuClothConstraint, GpuClothEmbedBinding,
        GpuClothEmbedParams, GpuClothSelfParams, GpuClothSimParams,
    };

    /// Owned backing store for a synthetic [`ClothPieceUpload`], so a test can
    /// vary one pool's element count and re-derive the buffer signature without
    /// touching the device. Every pool is given a distinct count so a bug that
    /// read the wrong pool's length would collapse two fields onto one another.
    struct UploadFixture {
        positions: Vec<[f32; 4]>,
        velocities: Vec<[f32; 4]>,
        constraints: Vec<GpuClothConstraint>,
        bending: Vec<GpuClothBendingConstraint>,
        colliders: Vec<GpuClothCollider>,
        backstops: Vec<GpuClothBackstop>,
        embed_bindings: Vec<GpuClothEmbedBinding>,
        triangles: Vec<[u32; 3]>,
        csr_offsets: Vec<u32>,
        csr_entries: Vec<u32>,
        render_vertex_count: u32,
        hash_cell_count: u32,
    }

    impl UploadFixture {
        /// A fixture whose pools all carry distinct, non-trivial counts.
        fn distinct() -> Self {
            Self {
                positions: vec![[0.0; 4]; 4],
                velocities: vec![[0.0; 4]; 4],
                constraints: vec![GpuClothConstraint::default(); 6],
                bending: vec![GpuClothBendingConstraint::default(); 3],
                colliders: vec![GpuClothCollider::default(); 2],
                backstops: vec![GpuClothBackstop::default(); 5],
                embed_bindings: vec![GpuClothEmbedBinding::default(); 7],
                triangles: vec![[0u32; 3]; 8],
                csr_offsets: vec![0u32; 5],
                csr_entries: vec![0u32; 9],
                render_vertex_count: 11,
                hash_cell_count: 13,
            }
        }

        /// Borrows the owned pools into a [`ClothPieceUpload`] for signing.
        fn upload(&self) -> ClothPieceUpload<'_> {
            ClothPieceUpload {
                positions: &self.positions,
                velocities: &self.velocities,
                constraints: &self.constraints,
                bending: &self.bending,
                colliders: &self.colliders,
                backstops: &self.backstops,
                embed_bindings: &self.embed_bindings,
                triangles: &self.triangles,
                csr_offsets: &self.csr_offsets,
                csr_entries: &self.csr_entries,
                aero_params: GpuClothAeroParams::default(),
                render_vertex_count: self.render_vertex_count,
                hash_cell_count: self.hash_cell_count,
                sim_params: GpuClothSimParams::default(),
                body_params: GpuClothBodyParams::default(),
                self_params: GpuClothSelfParams::default(),
                backstop_params: GpuClothBackstopParams::default(),
                embed_params: GpuClothEmbedParams::default(),
            }
        }
    }

    #[test]
    fn identical_topology_yields_equal_signature() {
        let a = UploadFixture::distinct();
        let b = UploadFixture::distinct();
        // Identical pool sizes must fingerprint identically, which is what lets the
        // prepare stage reuse the resident piece and restream only dynamic inputs.
        assert_eq!(
            ClothPieceSignature::from_upload(&a.upload()),
            ClothPieceSignature::from_upload(&b.upload()),
        );
    }

    #[test]
    fn growing_the_particle_pool_breaks_reuse() {
        let base = UploadFixture::distinct();
        let base_sig = ClothPieceSignature::from_upload(&base.upload());

        // A larger particle pool means the resident position/velocity/prev buffers
        // are too small; reuse would overrun them, so the signature must differ.
        let mut grown = UploadFixture::distinct();
        grown.positions.push([0.0; 4]);
        grown.velocities.push([0.0; 4]);
        assert_ne!(base_sig, ClothPieceSignature::from_upload(&grown.upload()));
    }

    #[test]
    fn growing_the_collider_pool_breaks_reuse() {
        let base = UploadFixture::distinct();
        let base_sig = ClothPieceSignature::from_upload(&base.upload());

        // The collider pool is restreamed by `write_dynamic`; growing it without a
        // realloc would overrun the resident buffer, so the signature must differ.
        let mut more = UploadFixture::distinct();
        more.colliders.push(GpuClothCollider::default());
        assert_ne!(base_sig, ClothPieceSignature::from_upload(&more.upload()));
    }

    #[test]
    fn growing_the_backstop_pool_breaks_reuse() {
        let base = UploadFixture::distinct();
        let base_sig = ClothPieceSignature::from_upload(&base.upload());

        // The backstop pool is restreamed by `write_dynamic` (its planes anchor on
        // the animated skinned surface); growing it without a realloc would overrun
        // the resident buffer, so the signature must differ to force a fresh piece.
        let mut more = UploadFixture::distinct();
        more.backstops.push(GpuClothBackstop::default());
        assert_ne!(base_sig, ClothPieceSignature::from_upload(&more.upload()));
    }

    #[test]
    fn resizing_the_hash_grid_breaks_reuse() {
        let base = UploadFixture::distinct();
        let base_sig = ClothPieceSignature::from_upload(&base.upload());

        // An LOD tier swap can resize the self-collision hash grid; even though the
        // count is a scalar rather than a pool length, it sizes the cell table, so
        // a change must still force a fresh allocation.
        let mut more_cells = UploadFixture::distinct();
        more_cells.hash_cell_count += 1;
        assert_ne!(
            base_sig,
            ClothPieceSignature::from_upload(&more_cells.upload()),
        );
    }
}

