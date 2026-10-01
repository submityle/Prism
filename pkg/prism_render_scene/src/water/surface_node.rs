//! The water-surface raster draw's deterministic, `GPU`-free plan.
//!
//! Mirroring the compute half ([`dispatch`](super::dispatch), whose ordering
//! and sizing all live in the float-free arch plan), the raster draw is split
//! into a pure planner here and a dumb device executor (the draw system that
//! lands in the following slice). This module owns the planner: it resolves one
//! resident [`WaterBody`](super::body::WaterBody) into the complete set of facts
//! the draw system needs — which shading frontend to key the pipeline on, the
//! lattice resolution to size the index buffer from, and how many displaced
//! vertices the vertex stage indexes — or decides the body draws nothing at all.
//!
//! Keeping the decision here (rather than inside the draw system, which needs a
//! device the sandbox lacks) is what makes every branch — simulation-only
//! bodies, an un-swept surface, a degenerate grid — unit-testable on `CPU`
//! without a `GPU`, exactly like [`dispatch`](super::dispatch)'s kernel→group
//! mapping is.

use prism_render_architecture::water::gpu::SurfaceGrid;

use super::body::WaterBody;
use super::surface_mesh::surface_grid_from_params;
use super::surface_shading::WaterSurfaceShading;

/// Everything the water-surface raster draw system needs to render one body's
/// displaced surface, resolved once from the body on the `CPU`.
///
/// Produced by [`plan_surface_draw`] and consumed by the raster draw system
/// (following slice): the system keys its pipeline on
/// [`shading.frontend`](WaterSurfaceShading::frontend), packs the per-view
/// uniform from `shading`, sizes and fills the index buffer from `grid`
/// ([`surface_index_data`](super::surface_mesh::surface_index_data) +
/// [`SurfaceGrid::index_count`]), and reads the four per-vertex storage arrays
/// of `vertex_count` displaced vertices the compute sweep already filled.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "consumed by the water-surface raster draw system (the following slice); its fields are exercised now by this module's unit tests"
    )
)]
pub(crate) struct SurfaceDraw {
    /// The authored shading state (frontend + body-constant style) the draw
    /// keys its pipeline on and packs into the per-view uniform.
    pub(crate) shading: WaterSurfaceShading,
    /// The lattice the index buffer is sized and wound from, reconstructed from
    /// the body's authored mesh params so there is no second source of truth.
    pub(crate) grid: SurfaceGrid,
    /// The displaced render-mesh vertex count the vertex stage indexes through
    /// the four `@group(0)` storage arrays the compute sweep filled.
    pub(crate) vertex_count: u32,
}

/// Resolve the raster draw for one water `body`, or `None` when the body draws
/// no surface this frame.
///
/// Returns `None` — an honest no-op, never a fabricated draw — in exactly three
/// cases, each a real "nothing to draw" condition rather than an error:
///
/// * the body carries no [`WaterSurfaceShading`], i.e. it is simulation-only
///   and never opted into an on-screen surface;
/// * the surface-mesh compute sweep placed no vertices
///   (`surface_vertex_count == 0`), so the storage arrays the vertex stage
///   indexes are empty;
/// * the reconstructed [`SurfaceGrid`] is degenerate (fewer than two vertices
///   on an axis), so it has no quad and
///   [`index_count`](SurfaceGrid::index_count) is zero — a draw of zero indices.
///
/// Otherwise it returns the fully-resolved [`SurfaceDraw`]. Pure: the same body
/// always yields the same plan, so every branch is unit-tested without a `GPU`.
#[must_use]
pub(crate) fn plan_surface_draw(body: &WaterBody) -> Option<SurfaceDraw> {
    let shading = body.surface_shading?;
    if body.surface_vertex_count == 0 {
        return None;
    }
    let grid = surface_grid_from_params(&body.surface_mesh_params);
    if grid.index_count() == 0 {
        return None;
    }
    Some(SurfaceDraw {
        shading,
        grid,
        vertex_count: body.surface_vertex_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::water::abi::GpuWaterSurfaceMeshParams;
    use prism_render_architecture::water::ShadingFrontend;

    /// A body with a swept 4x3 surface and `PBR` shading: the normal "draws a
    /// surface" case every assertion starts from, then mutates one field.
    fn drawable_body() -> WaterBody {
        WaterBody {
            surface_shading: Some(WaterSurfaceShading::new(ShadingFrontend::Pbr)),
            surface_vertex_count: 12,
            surface_mesh_params: GpuWaterSurfaceMeshParams {
                grid_dims: [4, 3, 12, 1],
                ..Default::default()
            },
            ..WaterBody::default()
        }
    }

    #[test]
    fn plans_a_draw_for_a_swept_shaded_body() {
        let body = drawable_body();

        let draw = plan_surface_draw(&body).expect("a swept, shaded body draws a surface");

        assert_eq!(draw.shading.frontend, ShadingFrontend::Pbr);
        assert_eq!(draw.vertex_count, 12);
        assert_eq!(draw.grid.verts_x, 4);
        assert_eq!(draw.grid.verts_z, 3);
        // 2 quads x 3 quads x 6 indices = 36 triangle-list indices.
        assert_eq!(draw.grid.index_count(), 36);
    }

    #[test]
    fn carries_the_requested_frontend_through() {
        for frontend in [
            ShadingFrontend::Pbr,
            ShadingFrontend::Npr,
            ShadingFrontend::Custom,
            ShadingFrontend::Hybrid,
        ] {
            let mut body = drawable_body();
            body.surface_shading = Some(WaterSurfaceShading::new(frontend));

            let draw = plan_surface_draw(&body).expect("a swept, shaded body draws a surface");

            assert_eq!(draw.shading.frontend, frontend);
        }
    }

    #[test]
    fn simulation_only_body_draws_nothing() {
        let mut body = drawable_body();
        body.surface_shading = None;

        assert!(plan_surface_draw(&body).is_none());
    }

    #[test]
    fn unswept_surface_draws_nothing() {
        let mut body = drawable_body();
        body.surface_vertex_count = 0;

        assert!(plan_surface_draw(&body).is_none());
    }

    #[test]
    fn degenerate_grid_draws_nothing() {
        // A single column (fewer than two vertices on x) spans no quad, so the
        // index buffer would be empty: there is nothing to draw.
        let mut body = drawable_body();
        body.surface_mesh_params.grid_dims = [1, 3, 3, 1];

        assert!(plan_surface_draw(&body).is_none());
    }
}
