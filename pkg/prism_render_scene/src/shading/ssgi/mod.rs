//! GPU screen-space global illumination (SSGI) subsystem.
//!
//! The CPU golden lives in [`prism_render_shading::screen_space::gi`] and the
//! shader twin in `shaders/ssgi.wesl`; this module is the render-world plumbing
//! that runs them. SSGI integrates one indirect *diffuse* bounce by casting
//! several cosine-weighted hemisphere rays per pixel and marching them across
//! the *same* reverse-Z Hi-Z pyramid and current-frame colour pyramid the
//! reflection subsystem already builds. Hits pick up on-screen radiance (colour
//! bleeding); misses take the IBL/SH ambient the resolve evaluates, so the
//! gather augments the ambient term without introducing energy discontinuities.
//!
//! Because SSGI reuses SSR's rebuilt inputs, the additional plumbing is a
//! trace plus a two-pass composite; it lands across cohesive files matching the
//! rest of the shading pipeline:
//!
//! * [`abi`] — the [`abi::GpuSsgiConfig`] gather immediate and the
//!   [`abi::GpuSsgiCompositeParams`] composite immediate shared with the SSGI
//!   shaders.
//! * [`resources`] — the per-view [`resources::ViewSsgiTextures`] (the gather
//!   output plus the scratch base copy) and their viewport-sized allocator.
//! * [`denoise`] — the edge-aware spatial denoise pipeline, its per-view bind
//!   group and the `Core3d` node that runs the joint bilateral blur between the
//!   trace and the composite.
//! * [`trace`] — the diffuse-hemisphere gather pipeline, its per-view bind
//!   group and the `Core3d` node that marches the reverse-Z Hi-Z pyramid and
//!   samples the current-frame colour at each hit, writing the pre-albedo mean
//!   indirect radiance plus a blend confidence.
//! * [`composite`] — the copy + fold pipelines, per-view bind groups and the
//!   `Core3d` node that lifts the SSR-composited `scene_color` into the scratch
//!   base and folds the gather in under an energy-conserving substitution of
//!   the resolve's flat ambient.

mod abi;
mod composite;
mod denoise;
mod resources;
mod trace;

#[cfg(test)]
mod shader_tests;

pub(crate) use composite::{
    init_ssgi_composite_pipeline, prepare_ssgi_composite_bind_groups, ssgi_composite_pass,
};
pub(crate) use denoise::{
    init_ssgi_denoise_pipeline, prepare_ssgi_denoise_bind_groups, ssgi_denoise_pass,
};
pub(crate) use resources::prepare_ssgi_textures;
pub(crate) use trace::{init_ssgi_trace_pipeline, prepare_ssgi_trace_bind_groups, ssgi_trace_pass};
