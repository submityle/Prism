//! `CPU`-side geometry and per-view uniform construction for the water-surface
//! raster draw.
//!
//! The sibling [`super::surface_pipeline`] slice builds the raster *pipelines*
//! (the shared `@group(0)` layout and the four per-frontend pipelines); this
//! slice builds the two host-side buffers the draw node uploads for every
//! surface patch:
//!
//! * the **triangle-list index buffer** that drives the vertexless draw. The
//!   `water_surface_raster.wesl` vertex stage has no vertex buffer: it reads the
//!   displaced grid out of the `@group(0)` storage arrays indexed by
//!   `@builtin(vertex_index)`, so the index buffer alone selects which lattice
//!   vertices form each triangle. [`surface_index_data`] emits exactly the
//!   `two-triangles-per-quad` list the
//!   [`SurfaceGrid`](prism_render_architecture::water::gpu::SurfaceGrid) draw
//!   call ([`plan_surface_draw_call`](prism_render_architecture::water::gpu::plan_surface_draw_call))
//!   sizes.
//!
//! * the **`WaterSurfaceView` per-view uniform**: a `#[repr(C)]` host mirror of
//!   the `WaterSurfaceView` `struct` in `water_surface_raster.wesl`, laid out
//!   byte-for-byte at the `std140` offsets the `CPU` contract
//!   ([`prism_render_architecture::water::gpu::surface_bindings::view`]) pins,
//!   so a plain `bytemuck` cast uploads it and a layout drift fails the build
//!   rather than silently misreading on the device.
//!
//! Both are pure data: no `wgpu` handles, no wall clock. The draw node (next
//! slice) allocates the device index buffer from [`surface_index_data`], fills
//! a [`GpuWaterSurfaceView`] from the camera / light / water parameters, and
//! records `draw_indexed`.
//!
//! ## Winding
//!
//! The lattice vertex at column `ix` (`0..verts_x`) and row `iz` (`0..verts_z`)
//! lives at linear index `iz * verts_x + ix`, matching the row-major order the
//! solver writes its per-vertex storage arrays in. Each interior quad is split
//! into two triangles across the `v00`–`v11` diagonal, wound so the surface's
//! up-face (`+y` world normal) is the front face — consistent with the
//! per-vertex normals (which point out of the water) and the
//! [`FrontFace::Ccw`](bevy_render::render_resource::FrontFace) the pipeline
//! declares. (Culling is disabled, so the camera may still cross the waterline
//! and see the back face; the winding only fixes which side `@builtin(front_facing)`
//! reports.)

use bytemuck::{Pod, Zeroable};

use prism_render_architecture::water::gpu::surface_bindings::view;
use prism_render_architecture::water::gpu::SurfaceGrid;

/// Build the triangle-list index buffer for one surface patch.
///
/// Returns `quad_count * 6` indices (two triangles per interior quad), each a
/// valid lattice-vertex index in `0..vertex_count`, in a deterministic order
/// (row-major quads, lower-right triangle then upper-left). A grid thinner than
/// two vertices on either axis has no interior quad and yields an empty buffer,
/// matching
/// [`SurfaceDrawCall::is_empty`](prism_render_architecture::water::gpu::SurfaceDrawCall::is_empty);
/// the draw node must skip such a patch rather than issue a zero-index draw.
///
/// The length is exactly
/// [`SurfaceGrid::index_count`](prism_render_architecture::water::gpu::SurfaceGrid::index_count),
/// so the device buffer the node allocates from this slice and the draw call
/// the arch contract plans agree by construction.
#[must_use]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "consumed by the water-surface raster draw node (the following slice); exercised now by the unit tests in this module"
    )
)]
pub(crate) fn surface_index_data(grid: SurfaceGrid) -> Vec<u32> {
    let verts_x = grid.verts_x;
    let verts_z = grid.verts_z;

    // A degenerate patch (fewer than two vertices on an axis) has no quad.
    if verts_x < 2 || verts_z < 2 {
        return Vec::new();
    }

    // `index_count` is saturating `u32`; a realistic surface grid is at most a
    // few thousand vertices per side, so the cast to `usize` is exact on every
    // target the engine builds for.
    let mut indices = Vec::with_capacity(grid.index_count() as usize);

    let quads_x = verts_x - 1;
    let quads_z = verts_z - 1;
    for iz in 0..quads_z {
        let row0 = iz * verts_x;
        let row1 = (iz + 1) * verts_x;
        for ix in 0..quads_x {
            let v00 = row0 + ix;
            let v10 = row0 + ix + 1;
            let v01 = row1 + ix;
            let v11 = row1 + ix + 1;

            // Lower-right triangle, then upper-left; both wound so the `+y`
            // up-face is the front face (see the module winding note).
            indices.extend_from_slice(&[v00, v11, v10, v00, v01, v11]);
        }
    }

    indices
}

/// Host mirror of the `WaterSurfaceView` uniform `struct` in
/// `water_surface_raster.wesl`, laid out byte-for-byte at the `std140` offsets
/// [`prism_render_architecture::water::gpu::surface_bindings::view`] pins.
///
/// A `mat4x4<f32>` (64 bytes) followed by seven 16-byte-aligned `vec4<f32>`
/// rows — already a multiple of 16, so no tail padding. The `offset_of` /
/// `size_of` contract tests below pin every field to the arch-side offset
/// constant, so a drift between this host record, the shader `struct` and the
/// contract fails the build rather than corrupting the draw at run time.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "uploaded by the water-surface raster draw node (the following slice); exercised now by the layout contract tests in this module"
    )
)]
pub(crate) struct GpuWaterSurfaceView {
    /// Clip-from-world transform for the surface vertices (column-major, the
    /// `wgpu`/`WGSL` convention).
    pub clip_from_world: [[f32; 4]; 4],
    /// World-space camera position (`xyz`); `w` is the refraction screen offset.
    pub world_camera_position: [f32; 4],
    /// Direction *towards* the key light (`xyz`, normalized); `w` unused.
    pub sun_direction: [f32; 4],
    /// Key-light illuminance (`rgb`); `w` unused.
    pub sun_illuminance: [f32; 4],
    /// Base perceptual roughness (`x`), reflectance (`y`), optical thickness
    /// (`z`), foam whiten strength (`w`).
    pub surface_params: [f32; 4],
    /// Base water albedo (`rgb`) and the minimum surface alpha (`w`).
    pub water_color: [f32; 4],
    /// `NPR` ramp step count (`x`), toon foam threshold (`y`), author tint
    /// strength (`z`), hybrid shore blend bias (`w`).
    pub style_params: [f32; 4],
    /// Framebuffer size in pixels (`xy`) for the refraction screen lookup; `zw`
    /// unused.
    pub viewport: [f32; 4],
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_architecture::water::gpu::plan_surface_draw_call;

    // ------------------------------------------------------------------ mesh

    #[test]
    fn index_count_matches_the_draw_call() {
        for grid in [
            SurfaceGrid {
                verts_x: 2,
                verts_z: 2,
            },
            SurfaceGrid {
                verts_x: 4,
                verts_z: 3,
            },
            SurfaceGrid {
                verts_x: 129,
                verts_z: 129,
            },
        ] {
            let indices = surface_index_data(grid);
            assert_eq!(
                indices.len() as u32,
                grid.index_count(),
                "{grid:?}: index data length must equal the planned index count"
            );
            assert_eq!(
                indices.len() as u32,
                plan_surface_draw_call(grid).index_count
            );
        }
    }

    #[test]
    fn every_index_is_a_valid_lattice_vertex() {
        let grid = SurfaceGrid {
            verts_x: 7,
            verts_z: 5,
        };
        let count = grid.vertex_count();
        for &i in &surface_index_data(grid) {
            assert!(i < count, "index {i} out of bounds for {count} vertices");
        }
    }

    #[test]
    fn degenerate_grids_emit_no_indices() {
        for grid in [
            SurfaceGrid {
                verts_x: 1,
                verts_z: 8,
            },
            SurfaceGrid {
                verts_x: 8,
                verts_z: 1,
            },
            SurfaceGrid {
                verts_x: 0,
                verts_z: 0,
            },
            SurfaceGrid {
                verts_x: 1,
                verts_z: 1,
            },
        ] {
            assert!(surface_index_data(grid).is_empty(), "{grid:?}");
        }
    }

    #[test]
    fn index_data_is_deterministic() {
        let grid = SurfaceGrid {
            verts_x: 33,
            verts_z: 17,
        };
        assert_eq!(surface_index_data(grid), surface_index_data(grid));
    }

    #[test]
    fn one_quad_is_two_triangles_sharing_the_diagonal() {
        // 2x2 lattice => one quad => two triangles => six indices.
        // Linear indices: v00=0, v10=1, v01=2, v11=3.
        let indices = surface_index_data(SurfaceGrid {
            verts_x: 2,
            verts_z: 2,
        });
        assert_eq!(indices, vec![0, 3, 1, 0, 2, 3]);
    }

    #[test]
    fn every_triangle_faces_up() {
        // The up-face (+y world normal) must be the front face. Reconstruct each
        // vertex position as (ix, 0, iz) from its linear index and check the
        // triangle normal points +y (cross of its two edges has positive y).
        let grid = SurfaceGrid {
            verts_x: 6,
            verts_z: 4,
        };
        let verts_x = grid.verts_x;
        let pos = |i: u32| -> (f32, f32) {
            let ix = i % verts_x;
            let iz = i / verts_x;
            (ix as f32, iz as f32)
        };
        let indices = surface_index_data(grid);
        assert_eq!(indices.len() % 3, 0);
        for tri in indices.chunks_exact(3) {
            let (ax, az) = pos(tri[0]);
            let (bx, bz) = pos(tri[1]);
            let (cx, cz) = pos(tri[2]);
            // Edges in the x-z plane; the y of (e1 x e2) is -(e1x*e2z - e1z*e2x)
            // because the world up axis is +y with x,z spanning the plane.
            let e1x = bx - ax;
            let e1z = bz - az;
            let e2x = cx - ax;
            let e2z = cz - az;
            let normal_y = e1z * e2x - e1x * e2z;
            assert!(
                normal_y > 0.0,
                "triangle {tri:?} is wound the wrong way (normal_y = {normal_y})"
            );
        }
    }

    #[test]
    fn triangles_tile_the_patch_without_gaps() {
        // Across the whole patch every interior quad contributes exactly two
        // triangles, so the triangle count equals quad_count and the total
        // signed area equals the patch area ((verts_x-1)*(verts_z-1)).
        let grid = SurfaceGrid {
            verts_x: 9,
            verts_z: 6,
        };
        let verts_x = grid.verts_x;
        let pos = |i: u32| -> (f32, f32) { ((i % verts_x) as f32, (i / verts_x) as f32) };
        let indices = surface_index_data(grid);
        let triangles = indices.len() / 3;
        assert_eq!(triangles as u32, grid.quad_count() * 2);

        let mut area2 = 0.0_f32;
        for tri in indices.chunks_exact(3) {
            let (ax, az) = pos(tri[0]);
            let (bx, bz) = pos(tri[1]);
            let (cx, cz) = pos(tri[2]);
            // Twice the unsigned triangle area in the x-z plane.
            area2 += ((bx - ax) * (cz - az) - (bz - az) * (cx - ax)).abs();
        }
        let patch_area = (grid.quad_count() as f32) * 2.0;
        assert!(
            (area2 - patch_area).abs() < 1e-3,
            "tiled area {area2} must equal twice the patch area {patch_area}"
        );
    }

    // --------------------------------------------------------------- uniform

    #[test]
    fn uniform_size_matches_the_contract() {
        assert_eq!(size_of::<GpuWaterSurfaceView>(), view::SIZE as usize);
        // No tail padding beyond the std140-aligned fields.
        assert_eq!(size_of::<GpuWaterSurfaceView>() % 16, 0);
    }

    #[test]
    fn uniform_field_offsets_match_the_contract() {
        assert_eq!(
            core::mem::offset_of!(GpuWaterSurfaceView, clip_from_world),
            view::CLIP_FROM_WORLD_OFFSET as usize
        );
        assert_eq!(
            core::mem::offset_of!(GpuWaterSurfaceView, world_camera_position),
            view::WORLD_CAMERA_POSITION_OFFSET as usize
        );
        assert_eq!(
            core::mem::offset_of!(GpuWaterSurfaceView, sun_direction),
            view::SUN_DIRECTION_OFFSET as usize
        );
        assert_eq!(
            core::mem::offset_of!(GpuWaterSurfaceView, sun_illuminance),
            view::SUN_ILLUMINANCE_OFFSET as usize
        );
        assert_eq!(
            core::mem::offset_of!(GpuWaterSurfaceView, surface_params),
            view::SURFACE_PARAMS_OFFSET as usize
        );
        assert_eq!(
            core::mem::offset_of!(GpuWaterSurfaceView, water_color),
            view::WATER_COLOR_OFFSET as usize
        );
        assert_eq!(
            core::mem::offset_of!(GpuWaterSurfaceView, style_params),
            view::STYLE_PARAMS_OFFSET as usize
        );
        assert_eq!(
            core::mem::offset_of!(GpuWaterSurfaceView, viewport),
            view::VIEWPORT_OFFSET as usize
        );
    }

    #[test]
    fn uniform_is_a_plain_byte_castable_record() {
        // A zeroed view is all-zero bytes (Pod/Zeroable), and the byte view
        // covers the full uniform binding the contract sizes.
        let view_uniform = GpuWaterSurfaceView::default();
        let bytes = bytemuck::bytes_of(&view_uniform);
        assert_eq!(bytes.len(), view::SIZE as usize);
        assert!(bytes.iter().all(|&b| b == 0));
    }
}
