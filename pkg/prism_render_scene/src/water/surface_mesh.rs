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

use prism_render_architecture::water::gpu::SurfaceGrid;

use crate::water::abi::GpuWaterSurfaceMeshParams;

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

/// Reconstruct the [`SurfaceGrid`] from the compute sweep's authored mesh
/// params so the raster draw node sizes and fills the index buffer from the
/// same lattice resolution the sweep placed, with no second source of truth.
///
/// Reads only the vertex counts: `grid_dims[0]` vertices along x and
/// `grid_dims[1]` along z (see [`GpuWaterSurfaceMeshParams::grid_dims`]). The
/// remaining lanes (`[2]` total vertex count, `[3]` lane width) are sweep-side
/// bookkeeping the raster path recomputes from the grid, so they are
/// intentionally not read here.
#[must_use]
pub(crate) fn surface_grid_from_params(params: &GpuWaterSurfaceMeshParams) -> SurfaceGrid {
    SurfaceGrid {
        verts_x: params.grid_dims[0],
        verts_z: params.grid_dims[1],
    }
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

/// The authored, semantic inputs the raster draw packs into one
/// [`GpuWaterSurfaceView`] uniform.
///
/// This is the per-frame, per-view camera / key-light / shading-style state in
/// plain engine units, *before* it is laid out at the `std140` offsets the
/// `water_surface_raster.wesl` `WaterSurfaceView` `struct` reads. Keeping the
/// authored values here (rather than hand-packing `vec4` rows at the call site)
/// keeps the byte layout a single-sourced concern of [`build_surface_view`] and
/// lets the draw node pass readable, named fields.
pub(crate) struct SurfaceViewParams {
    /// Clip-from-world transform for the surface vertices (column-major).
    pub clip_from_world: [[f32; 4]; 4],
    /// World-space camera position.
    pub camera_world_position: [f32; 3],
    /// Screen-space refraction offset scale (packed into
    /// `world_camera_position.w`); `0.0` disables the refraction displacement.
    pub refraction_screen_offset: f32,
    /// Direction *towards* the key light (world space). The shader renormalizes
    /// and falls back to straight-up for a degenerate direction, so a non-unit
    /// or zero vector is safe.
    pub sun_direction: [f32; 3],
    /// Key-light illuminance (`rgb`, linear).
    pub sun_illuminance: [f32; 3],
    /// Base perceptual roughness.
    pub roughness: f32,
    /// Base reflectance (`f0` at normal incidence).
    pub reflectance: f32,
    /// Optical thickness driving the `Beer-Lambert` body attenuation.
    pub optical_thickness: f32,
    /// Foam whiten strength: how strongly foam coverage washes the body white.
    pub foam_whiten: f32,
    /// Base water albedo (`rgb`, linear).
    pub water_albedo: [f32; 3],
    /// Minimum surface alpha (the `Fresnel`-weighted alpha never falls below
    /// this), so still water keeps a floor of opacity.
    pub min_alpha: f32,
    /// `NPR` ramp step count (number of discrete toon bands).
    pub npr_ramp_steps: f32,
    /// `NPR` toon foam threshold: foam coverage above this draws a hard white
    /// edge.
    pub toon_foam_threshold: f32,
    /// Custom-frontend emissive tint strength.
    pub tint_strength: f32,
    /// Hybrid-frontend shore blend bias added to foam coverage when crossfading
    /// the `PBR` body into the stylized shallows.
    pub hybrid_shore_bias: f32,
    /// Framebuffer size in pixels (`width`, `height`) for the refraction screen
    /// lookup.
    pub viewport_size: [f32; 2],
}

/// Pack the authored per-view state into the `std140` [`GpuWaterSurfaceView`]
/// uniform the raster draw uploads.
///
/// This is the single place the semantic inputs map onto the shader's `vec4`
/// rows, in the exact order `water_surface_raster.wesl` reads them:
///
/// * `world_camera_position = (camera_xyz, refraction_screen_offset)`,
/// * `sun_direction = (dir_xyz, 0)` and `sun_illuminance = (rgb, 0)`,
/// * `surface_params = (roughness, reflectance, optical_thickness, foam_whiten)`,
/// * `water_color = (albedo_rgb, min_alpha)`,
/// * `style_params = (npr_ramp_steps, toon_foam_threshold, tint_strength, hybrid_shore_bias)`,
/// * `viewport = (width, height, 0, 0)`.
///
/// Pure: the same inputs always produce the same bytes, so the uniform upload
/// is deterministic frame to frame. The unused `vec4` lanes (`sun_*.w`,
/// `viewport.zw`) are zeroed rather than left undefined.
#[must_use]
pub(crate) fn build_surface_view(params: &SurfaceViewParams) -> GpuWaterSurfaceView {
    let [cx, cy, cz] = params.camera_world_position;
    let [sx, sy, sz] = params.sun_direction;
    let [ir, ig, ib] = params.sun_illuminance;
    let [ar, ag, ab] = params.water_albedo;
    let [vw, vh] = params.viewport_size;

    GpuWaterSurfaceView {
        clip_from_world: params.clip_from_world,
        world_camera_position: [cx, cy, cz, params.refraction_screen_offset],
        sun_direction: [sx, sy, sz, 0.0],
        sun_illuminance: [ir, ig, ib, 0.0],
        surface_params: [
            params.roughness,
            params.reflectance,
            params.optical_thickness,
            params.foam_whiten,
        ],
        water_color: [ar, ag, ab, params.min_alpha],
        style_params: [
            params.npr_ramp_steps,
            params.toon_foam_threshold,
            params.tint_strength,
            params.hybrid_shore_bias,
        ],
        viewport: [vw, vh, 0.0, 0.0],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // The std140 offset/size contract lives in the architecture crate; the
    // offset and byte-stability tests below assert this packing matches it.
    use prism_render_architecture::water::gpu::plan_surface_draw_call;
    use prism_render_architecture::water::gpu::surface_bindings::view;

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

    // ------------------------------------------------- surface_grid_from_params

    /// Authored mesh params whose bookkeeping lanes deliberately disagree with
    /// the lattice vertex counts, so a test proves the reconstruction reads the
    /// resolution lanes and nothing else.
    fn mesh_params(verts_x: u32, verts_z: u32) -> GpuWaterSurfaceMeshParams {
        GpuWaterSurfaceMeshParams {
            // [2]/[3] are intentionally bogus: the grid must come from [0]/[1].
            grid_dims: [verts_x, verts_z, 0xDEAD_BEEF, 0x7FFF_FFFF],
            patch_origin: [1.0, 2.0, 3.0, 0.0],
            patch_extent: [10.0, 20.0, 1.0, 1.0],
        }
    }

    #[test]
    fn grid_from_params_reads_the_lattice_vertex_counts() {
        let grid = surface_grid_from_params(&mesh_params(7, 5));
        assert_eq!(grid.verts_x, 7);
        assert_eq!(grid.verts_z, 5);
    }

    #[test]
    fn grid_from_params_ignores_the_bookkeeping_lanes() {
        // Flipping only [2]/[3] must not change the reconstructed grid.
        let a = surface_grid_from_params(&mesh_params(4, 6));
        let mut params = mesh_params(4, 6);
        params.grid_dims[2] = 0;
        params.grid_dims[3] = 0;
        let b = surface_grid_from_params(&params);
        assert_eq!(a, b);
    }

    #[test]
    fn grid_from_params_feeds_an_index_buffer_of_the_contract_length() {
        // The reconstructed grid and the index data agree by construction: the
        // slice the draw node allocates is exactly `index_count` long.
        let params = mesh_params(5, 4);
        let grid = surface_grid_from_params(&params);
        assert_eq!(surface_index_data(grid).len() as u32, grid.index_count());
    }

    #[test]
    fn grid_from_params_degenerate_lattice_yields_an_empty_index_buffer() {
        // A single-row patch has no interior quad; the draw node must skip it.
        let grid = surface_grid_from_params(&mesh_params(1, 9));
        assert!(surface_index_data(grid).is_empty());
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

    // ---------------------------------------------------------- build_surface_view

    /// A representative, all-distinct set of authored inputs so each packed lane
    /// is traceable back to exactly one source field.
    fn sample_params() -> SurfaceViewParams {
        SurfaceViewParams {
            clip_from_world: [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ],
            camera_world_position: [100.0, 200.0, 300.0],
            refraction_screen_offset: 0.25,
            sun_direction: [0.0, 1.0, 0.0],
            sun_illuminance: [10.0, 11.0, 12.0],
            roughness: 0.3,
            reflectance: 0.02,
            optical_thickness: 0.7,
            foam_whiten: 0.8,
            water_albedo: [0.01, 0.1, 0.2],
            min_alpha: 0.15,
            npr_ramp_steps: 4.0,
            toon_foam_threshold: 0.6,
            tint_strength: 0.5,
            hybrid_shore_bias: 0.1,
            viewport_size: [1920.0, 1080.0],
        }
    }

    #[test]
    fn build_passes_the_clip_matrix_through_unchanged() {
        let params = sample_params();
        let view_uniform = build_surface_view(&params);
        assert_eq!(view_uniform.clip_from_world, params.clip_from_world);
    }

    #[test]
    fn build_packs_the_camera_and_refraction_offset() {
        let params = sample_params();
        let view_uniform = build_surface_view(&params);
        // xyz is the camera position; w carries the refraction screen offset.
        assert_eq!(
            view_uniform.world_camera_position,
            [100.0, 200.0, 300.0, 0.25]
        );
    }

    #[test]
    fn build_packs_the_key_light_with_zero_tail_lanes() {
        let params = sample_params();
        let view_uniform = build_surface_view(&params);
        assert_eq!(view_uniform.sun_direction, [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(view_uniform.sun_illuminance, [10.0, 11.0, 12.0, 0.0]);
    }

    #[test]
    fn build_packs_the_shading_params_in_contract_order() {
        let params = sample_params();
        let view_uniform = build_surface_view(&params);
        // surface_params = (roughness, reflectance, optical_thickness, foam_whiten).
        assert_eq!(view_uniform.surface_params, [0.3, 0.02, 0.7, 0.8]);
        // water_color = (albedo_rgb, min_alpha).
        assert_eq!(view_uniform.water_color, [0.01, 0.1, 0.2, 0.15]);
        // style_params = (ramp_steps, foam_threshold, tint_strength, shore_bias).
        assert_eq!(view_uniform.style_params, [4.0, 0.6, 0.5, 0.1]);
    }

    #[test]
    fn build_packs_the_viewport_with_zero_tail_lanes() {
        let params = sample_params();
        let view_uniform = build_surface_view(&params);
        // Only xy (framebuffer size) are read; zw must be zeroed, not undefined.
        assert_eq!(view_uniform.viewport, [1920.0, 1080.0, 0.0, 0.0]);
    }

    #[test]
    fn build_is_deterministic_and_byte_stable() {
        let params = sample_params();
        let a = build_surface_view(&params);
        let b = build_surface_view(&params);
        assert_eq!(a, b);
        // The packed uniform fills exactly the contract-sized binding.
        assert_eq!(bytemuck::bytes_of(&a).len(), view::SIZE as usize);
    }
}
