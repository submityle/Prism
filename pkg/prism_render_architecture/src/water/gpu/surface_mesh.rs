//! The water-surface *meshing* compute pass `CPU` contract: the bridge that
//! turns the assembled ocean displacement/normal textures into the four
//! per-vertex storage arrays the raster draw consumes.
//!
//! The spectrum passes ([`super::spectral_plan`], [`super::fft_plan`]) end by
//! writing *textures*: an assembled displacement field and a normal/foam field
//! (`water_spectrum_assemble`). The surface raster draw
//! ([`super::surface_bindings`]) instead reads *per-vertex storage arrays*
//! (`base_positions`, `surface_uvs`, `displacement`, `normal_foam`) indexed by
//! `@builtin(vertex_index)`. Something has to sample the textures at each
//! vertex's `uv` and scatter the result into those arrays — that is the
//! `water_surface_mesh` kernel (wired to a `WaterKernel` variant in a later
//! slice), and this module owns its `CPU`-testable sizing and dispatch
//! contract.
//!
//! Like the rest of [`super`], this is pure integer bookkeeping: no `GPU`
//! handles, no floats, no wall clock. The output strides mirror
//! [`SURFACE_VERTEX_RECORD_STRIDE`](super::surface_bindings::SURFACE_VERTEX_RECORD_STRIDE) (the stride the draw reads back), so the
//! producer and consumer of the per-vertex arrays can never silently disagree.
//! Clamp-free pathological grids never overflow the byte counts the backend
//! allocates: every multiply saturates.
//!
//! Shipping oceans run the identical producer→consumer split. `WaveWorks`,
//! `Crest` and `UE5` Water all evaluate the displacement/normal fields into
//! GPU textures once per cascade and then sample them while building the
//! clipmap/projected-grid vertices; `water_surface_mesh` is that sampling pass
//! factored into its own dispatch so the draw node stays a dumb index-buffer
//! executor.

use crate::water::kernels::linear_group_count;

use super::surface_bindings::SurfaceGrid;

/// Linear lane width of the per-vertex meshing sweep. Matches the 64-lane
/// `workgroup.x` the future `water_surface_mesh` compute kernel will declare;
/// the kernel descriptor and this constant are unified when the pass is wired
/// into `WaterKernel` in a later slice.
pub const SURFACE_MESH_LANES: u32 = 64;

/// The assembled ocean textures the meshing pass samples, in binding order.
///
/// Both are `rgba32float` fields produced by the spectrum assemble pass: the
/// displacement field (`xyz` world offset, `w` foldover/Jacobian) and the
/// normal/foam field (`xyz` surface normal, `w` whitecap/foam coverage). The
/// meshing kernel reads them with a sampler at each vertex `uv`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SurfaceMeshSource {
    /// The assembled displacement texture (`xyz` offset, `w` foldover).
    Displacement,
    /// The assembled normal/foam texture (`xyz` normal, `w` foam coverage).
    NormalFoam,
}

impl SurfaceMeshSource {
    /// Every sampled source texture the meshing pass binds, in binding order.
    pub const ALL: [SurfaceMeshSource; 2] = [
        SurfaceMeshSource::Displacement,
        SurfaceMeshSource::NormalFoam,
    ];
}

/// The four per-vertex storage arrays the meshing pass writes, in the
/// [`super::surface_bindings::SurfaceBinding`] slot order the raster draw reads
/// them back (`@binding(1..=4)`).
///
/// Each is an `array<vec4<f32>>` at [`SURFACE_VERTEX_RECORD_STRIDE`](super::surface_bindings::SURFACE_VERTEX_RECORD_STRIDE), one record
/// per surface vertex. The meshing kernel is the sole writer; the vertex stage
/// of `water_surface_raster.wesl` is the sole reader.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SurfaceMeshOutput {
    /// Undisplaced base lattice position (`xyz`; `w = 1`).
    BasePositions,
    /// Per-vertex texture coordinate (`xy`; `zw` pad/cascade blend).
    SurfaceUvs,
    /// Sampled displacement at the vertex `uv` (`xyz` world offset).
    Displacement,
    /// Sampled surface normal (`xyz`) and foam coverage (`w`).
    NormalFoam,
}

impl SurfaceMeshOutput {
    /// Every output array in the raster draw's binding order.
    pub const ALL: [SurfaceMeshOutput; 4] = [
        SurfaceMeshOutput::BasePositions,
        SurfaceMeshOutput::SurfaceUvs,
        SurfaceMeshOutput::Displacement,
        SurfaceMeshOutput::NormalFoam,
    ];

    /// The `@group(0)` binding slot this output occupies when the raster draw
    /// later reads it back. Mirrors
    /// [`super::surface_bindings::SurfaceBinding::index`] for the four
    /// `StorageRead` per-vertex arrays, keeping the producer's write targets and
    /// the consumer's read slots on one numbering.
    #[must_use]
    pub fn draw_binding_index(self) -> u32 {
        match self {
            SurfaceMeshOutput::BasePositions => 1,
            SurfaceMeshOutput::SurfaceUvs => 2,
            SurfaceMeshOutput::Displacement => 3,
            SurfaceMeshOutput::NormalFoam => 4,
        }
    }

    /// Bytes one output array needs for `grid`: `vertex_count` records at
    /// [`SURFACE_VERTEX_RECORD_STRIDE`](super::surface_bindings::SURFACE_VERTEX_RECORD_STRIDE). Saturating. Shares
    /// [`SurfaceGrid::vertex_array_bytes`] so the producer buffer and the draw's
    /// consumer binding size identically.
    #[must_use]
    pub fn array_bytes(self, grid: SurfaceGrid) -> u32 {
        grid.vertex_array_bytes()
    }
}

/// Total bytes the meshing pass writes for `grid`: the four per-vertex arrays,
/// each `vertex_count` records at [`SURFACE_VERTEX_RECORD_STRIDE`](super::surface_bindings::SURFACE_VERTEX_RECORD_STRIDE). Saturating,
/// so a pathological grid can never overflow the allocation the backend reads.
#[must_use]
pub fn surface_mesh_output_bytes(grid: SurfaceGrid) -> u32 {
    let one = grid.vertex_array_bytes();
    one.saturating_mul(SurfaceMeshOutput::ALL.len() as u32)
}

/// A fully described surface-meshing compute dispatch for one surface patch.
///
/// Pure integer bookkeeping the backend turns into a `dispatch_workgroups`
/// call: the linear vertex extent and the number of 64-lane workgroups that
/// cover it. Carries no `GPU` handles.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SurfaceMeshDispatch {
    /// Surface vertices the pass assembles (the linear dispatch extent).
    pub vertex_count: u32,
    /// Workgroups that cover `vertex_count` at the kernel's lane width.
    pub workgroup_count: u32,
}

impl SurfaceMeshDispatch {
    /// Whether this dispatch launches no workgroups (a degenerate grid with no
    /// vertices to assemble).
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.workgroup_count == 0
    }
}

/// Plan the `water_surface_mesh` dispatch for `grid`.
///
/// The pass is a flat per-vertex sweep: one invocation assembles one surface
/// vertex, so the launch is sized against [`SurfaceGrid::vertex_count`] at the
/// meshing lane width ([`SURFACE_MESH_LANES`], 64). The group count is a
/// saturating ceiling
/// division, so a zero-vertex grid yields zero workgroups rather than a divide
/// by zero.
#[must_use]
pub fn plan_surface_mesh_dispatch(grid: SurfaceGrid) -> SurfaceMeshDispatch {
    let vertex_count = grid.vertex_count();
    SurfaceMeshDispatch {
        vertex_count,
        workgroup_count: linear_group_count(vertex_count, SURFACE_MESH_LANES),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        plan_surface_mesh_dispatch, surface_mesh_output_bytes, SurfaceMeshOutput,
        SurfaceMeshSource, SURFACE_MESH_LANES,
    };
    use crate::water::gpu::surface_bindings::{
        SurfaceBinding, SurfaceBindingKind, SurfaceGrid, SURFACE_VERTEX_RECORD_STRIDE,
    };
    use alloc::vec::Vec;

    fn grid(verts_x: u32, verts_z: u32) -> SurfaceGrid {
        SurfaceGrid { verts_x, verts_z }
    }

    #[test]
    fn meshing_contract_shape_is_stable() {
        // The meshing pass contract: four per-vertex output arrays, two sampled
        // source textures, and a 64-lane linear vertex sweep. These are the
        // shapes the future `water_surface_mesh` kernel descriptor must match
        // when the pass is wired into `WaterKernel`.
        assert_eq!(SURFACE_MESH_LANES, 64);
        assert_eq!(SurfaceMeshOutput::ALL.len(), 4);
        assert_eq!(SurfaceMeshSource::ALL.len(), 2);
    }

    #[test]
    fn meshing_contract_matches_the_kernel_descriptor() {
        // The meshing lane width and the four-storage / two-sampled resource
        // shape here are the single source of truth the `WaterKernel::SurfaceMesh`
        // descriptor must mirror, so the host dispatch sizing and the device
        // bind-group layout can never silently disagree.
        use crate::water::kernels::{DispatchDomain, WaterKernel};
        let descriptor = WaterKernel::SurfaceMesh.descriptor();
        assert_eq!(descriptor.workgroup.x, SURFACE_MESH_LANES);
        assert_eq!(descriptor.workgroup.y, 1);
        assert_eq!(descriptor.workgroup.z, 1);
        assert_eq!(descriptor.domain, DispatchDomain::Vertices);
        assert_eq!(
            descriptor.layout.storage_buffers,
            SurfaceMeshOutput::ALL.len() as u32
        );
        assert_eq!(
            descriptor.layout.sampled_textures,
            SurfaceMeshSource::ALL.len() as u32
        );
        assert_eq!(descriptor.layout.uniform_buffers, 1);
        assert_eq!(descriptor.layout.storage_textures, 0);
    }

    #[test]
    fn output_slots_match_the_draw_storage_bindings() {
        // Every per-vertex array this pass writes is read back by the raster
        // draw at the same `@group(0)` slot and the same stride; the producer
        // and consumer share one numbering.
        for out in SurfaceMeshOutput::ALL {
            let slot = out.draw_binding_index();
            let binding = SurfaceBinding::ALL
                .into_iter()
                .find(|b| b.index() == slot)
                .expect("output slot has a matching draw binding");
            assert!(
                binding.visible_in_vertex(),
                "{out:?} feeds the vertex stage"
            );
        }
        // The four outputs cover exactly the four storage-read draw bindings.
        let storage_slots: Vec<u32> = SurfaceBinding::ALL
            .into_iter()
            .filter(|b| matches!(b.kind(), SurfaceBindingKind::StorageRead))
            .map(|b| b.index())
            .collect();
        let mut produced: Vec<u32> = SurfaceMeshOutput::ALL
            .into_iter()
            .map(|o| o.draw_binding_index())
            .collect();
        produced.sort_unstable();
        assert_eq!(produced, storage_slots);
    }

    #[test]
    fn total_output_bytes_is_four_vertex_arrays() {
        let g = grid(4, 3);
        assert_eq!(g.vertex_count(), 12);
        let one = g.vertex_array_bytes();
        assert_eq!(one, 12 * SURFACE_VERTEX_RECORD_STRIDE);
        assert_eq!(surface_mesh_output_bytes(g), one.saturating_mul(4));
        for out in SurfaceMeshOutput::ALL {
            assert_eq!(out.array_bytes(g), one);
        }
    }

    #[test]
    fn output_bytes_saturate_on_a_pathological_grid() {
        let g = grid(u32::MAX, u32::MAX);
        assert_eq!(g.vertex_array_bytes(), u32::MAX);
        assert_eq!(surface_mesh_output_bytes(g), u32::MAX);
    }

    #[test]
    fn dispatch_covers_every_vertex_at_the_lane_width() {
        let g = grid(10, 7);
        let vc = g.vertex_count();
        assert_eq!(vc, 70);
        let lanes = SURFACE_MESH_LANES;
        let plan = plan_surface_mesh_dispatch(g);
        assert_eq!(plan.vertex_count, vc);
        assert_eq!(plan.workgroup_count, 2); // ceil(70 / 64)
                                             // Every vertex is covered and no empty tail group is launched.
        assert!(plan.workgroup_count.saturating_mul(lanes) >= vc);
        assert!((plan.workgroup_count - 1).saturating_mul(lanes) < vc);
        assert!(!plan.is_empty());
    }

    #[test]
    fn exact_multiple_launches_no_tail_group() {
        // 64 vertices (verts_x * verts_z) is exactly one lane group.
        let g = grid(8, 8);
        assert_eq!(g.vertex_count(), 64);
        let plan = plan_surface_mesh_dispatch(g);
        assert_eq!(plan.workgroup_count, 1);
    }

    #[test]
    fn degenerate_grid_dispatches_nothing() {
        for g in [grid(0, 4), grid(4, 0), grid(1, 1), grid(0, 0)] {
            let plan = plan_surface_mesh_dispatch(g);
            if g.vertex_count() == 0 {
                assert!(plan.is_empty(), "{g:?}");
                assert_eq!(plan.workgroup_count, 0);
            }
        }
    }

    #[test]
    fn dispatch_plan_is_deterministic() {
        let g = grid(33, 17);
        assert_eq!(plan_surface_mesh_dispatch(g), plan_surface_mesh_dispatch(g));
    }

    #[test]
    fn sources_are_the_two_assembled_textures() {
        assert_eq!(SurfaceMeshSource::ALL.len(), 2);
        assert_ne!(
            SurfaceMeshSource::Displacement,
            SurfaceMeshSource::NormalFoam
        );
    }
}
