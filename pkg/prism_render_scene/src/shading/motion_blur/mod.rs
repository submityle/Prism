//! Motion-blur reconstruction subsystem.
//!
//! The CPU golden lives in [`prism_render_shading::motion_blur`] and the shader
//! twin in `shaders/motion_blur.wesl`; this module is the render-world plumbing
//! that runs them. Motion blur is a shared post-processing base, not a peer of
//! the PBR/NPR shading fronts: every illumination model resolves the same
//! velocity buffer, and a stylized front can pin a crisp shutter while a
//! photographic front opens it for cinematic streaking.
//!
//! The subsystem is a three-pass compute chain (`McGuire` et al. 2012):
//!
//! * [`pipeline`] builds the `tile_max` / `neighbor_max` / `reconstruct`
//!   compute pipelines and their group-0 layouts at `RenderStartup`.
//! * [`resources`] allocates the per-view tile textures
//!   (`ceil(dim / tile)`-sized) and the full-resolution blurred output during
//!   `PrepareResources`, gated on the enable and the resident visibility + SSR
//!   buffers it reads.
//! * [`bind_groups`] wires those textures into the three group-0 bind groups
//!   during `PrepareBindGroups`.
//! * [`dispatch`] records the `TileMax` -> `NeighborMax` -> reconstruction chain as
//!   a `Core3d` node.
//!
//! [`settings`] is the single render-world resource gating the subsystem and
//! feeding the golden tunables into the [`abi`] immediate block, mirroring the
//! golden [`prism_render_shading::MotionBlurParams`] defaults.
//!
//! Motion vectors are *not* reconstructed here: the resolve pass already writes
//! a per-pixel `cur_uv - prev_uv` G-buffer (camera + object motion), consumed
//! directly. Depth for the soft-depth term is the SSR geometry prepass's
//! reverse-Z device depth, unprojected to a linear view distance — reusing that
//! pass rather than duplicating a depth prepass, the same coupling the VSM
//! receiver pass relies on.

mod abi;
mod bind_groups;
mod dispatch;
mod pipeline;
mod resources;
mod settings;

pub(crate) use bind_groups::prepare_motion_blur_bind_groups;
pub(crate) use dispatch::motion_blur_pass;
pub(crate) use pipeline::init_motion_blur_pipeline;
pub(crate) use resources::prepare_motion_blur_textures;
pub(crate) use settings::PrismMotionBlurSettings;

#[cfg(test)]
mod shader_tests;
