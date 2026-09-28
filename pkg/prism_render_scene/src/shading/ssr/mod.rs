//! GPU screen-space reflection (SSR) subsystem.
//!
//! The CPU golden and shader twin live in
//! [`prism_render_shading::screen_space`] and `shaders/ssr.wesl`; this module
//! is the render-world plumbing that runs the twin.  SSR reflects the view ray
//! off each surface, marches the reflected ray across the reverse-Z HZB
//! "nearest depth" pyramid, reprojects a hit into the previous frame's colour,
//! and produces a reflected-radiance-plus-confidence buffer the shading resolve
//! blends over the prefiltered IBL specular wherever the trace is reliable.
//!
//! Following the rest of the shading pipeline, the plumbing lands as cohesive
//! files across successive slices:
//!
//! * `abi` — the immediate block shared with `ssr.wesl`.
//! * `resources` — the per-view SSR trace target and its allocator.
//! * `pipeline` — the SSR trace compute pipeline plus bind-group layout.
//! * `bind_groups` — the per-view HZB/depth/normal/history bind group.
//! * `dispatch` — the `Core3d` node recording the SSR trace.
//!
//! The shading resolve binds the trace target into its view bind group and
//! mixes the reflected radiance over the prefiltered environment specular by
//! the per-pixel confidence, falling back to IBL on misses and faded traces.

#[cfg(test)]
mod shader_tests;
