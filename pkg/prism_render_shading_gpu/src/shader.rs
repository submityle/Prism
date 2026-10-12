//! The single-sourced WGSL module and its entry-point metadata.
//!
//! The shader text lives in `src/shaders/surface.wgsl` and is pulled in once
//! here, so the parity test, a demo binary, and any future surface pass all
//! compile the identical source rather than three drifting copies.

/// The PBR surface shader source, as the WGSL text the RHI backend compiles.
///
/// It declares the reusable `evaluate_principled_direct` function — the twin of
/// `prism_render_shading::lighting::evaluate_principled_direct` restricted to
/// the base lobe — plus the `evaluate_surfaces` batch driver the parity test
/// dispatches. See the file header for the exact mirror scope.
pub const SURFACE_SHADER_WGSL: &str = include_str!("shaders/surface.wgsl");

/// The `@compute` entry point that shades a whole batch of surfaces.
pub const EVALUATE_ENTRY_POINT: &str = "evaluate_surfaces";

/// The workgroup size declared on [`EVALUATE_ENTRY_POINT`]'s `@workgroup_size`.
///
/// Kept next to the shader so the dispatch math cannot drift from the WGSL.
pub const WORKGROUP_SIZE: u32 = 64;

/// A debug label attached to the compiled shader module.
pub const SURFACE_SHADER_LABEL: &str = "prism_render_shading_gpu::surface";
