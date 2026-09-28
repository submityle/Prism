//! GPU Ground-Truth Ambient Occlusion (GTAO) subsystem.
//!
//! CPU golden and shader twin live in [`prism_render_shading::ao`] and
//! `shaders/gtao.wesl`; this module is the render-world plumbing that runs the
//! twin.  GTAO slots between the visibility raster and the shading resolve as
//! two compute steps:
//!
//! 1. a geometry prepass decodes the visibility buffer into a linear
//!    view-depth texture and a view-space normal texture, and
//! 2. the GTAO kernel reads those to write per-pixel ambient visibility, which
//!    the resolve stage multiplies into its indirect/ambient term.
//!
//! Following the rest of the shading pipeline, it is split into cohesive files.
//! This slice lands the per-view texture layer; the geometry prepass, the GTAO
//! compute pipeline and the resolve consumption follow in subsequent slices.
//!
//! * [`resources`] — the per-view [`resources::ViewGtaoTextures`] (linear
//!   depth, view normal, ambient occlusion) and the prepare system that keeps
//!   them sized to the viewport.

mod resources;

pub(crate) use resources::prepare_gtao_textures;
