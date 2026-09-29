//! Froxel volumetric fog subsystem.
//!
//! The CPU golden lives in [`prism_render_shading::volumetrics`] and the shader
//! twin in `shaders/volumetrics.wesl`; this module is the render-world plumbing
//! that runs them. Volumetric fog integrates *participating media* — fog,
//! dust, god rays, self-lit magical haze — into a view-frustum-aligned 3D grid
//! of "froxels". Each froxel stores its medium's scattering/absorption
//! coefficients and the light already in-scattered toward the eye; a final
//! front-to-back column march folds each slice's energy-conserving
//! in-scattering under the transmittance accumulated by nearer slices, so
//! nearer media correctly occlude farther media. The per-view result is applied
//! to the lit scene as `final = background * transmittance + in_scattering`.
//!
//! Fog is a subsystem on the shared GPU-driven base rather than a peer of the
//! PBR/NPR shading fronts: it consumes the depth the opaque pass already
//! resolves and the shadow maps the lighting pass already produces, and its
//! single-scattering integration is illumination-model agnostic (a stylized
//! front can drive the same froxel grid with an authored phase/colour).
//!
//! Following the rest of the shading pipeline, the plumbing lands as cohesive
//! files, mirroring the TAA / VSM compute-dispatch paradigm:
//!
//! * [`settings`] — the [`settings::PrismVolumetricsSettings`] render-world
//!   resource: the enable flag and the medium/grid/light tunables the passes
//!   read (self-owned, since there is no architecture contract for fog).
//! * [`abi`] — the two immediate blocks shared with `volumetrics.wesl`.
//! * [`pipeline`] — the scatter + integrate + apply compute pipelines and their
//!   owned group-0 layouts (plus the apply pass's linear-clamp sampler).
//! * [`resources`] — the persistent per-view froxel storage volumes (cached by
//!   [`bevy_render::view::RetainedViewEntity`]) and the per-frame immediate
//!   blocks, resolved into [`resources::ViewVolumetrics`].
//! * [`bind_groups`] — the per-view scatter + integrate bind groups.
//! * [`dispatch`] — the `Core3d` scheduling-system pass recording the scatter ->
//!   integrate -> apply dispatches and the copy-back over `scene_color`.

mod abi;
mod bind_groups;
mod dispatch;
mod pipeline;
mod resources;
mod settings;

#[cfg(test)]
mod shader_tests;

pub(crate) use bind_groups::prepare_volumetrics_bind_groups;
pub(crate) use dispatch::volumetrics_pass;
pub(crate) use pipeline::init_volumetrics_pipeline;
pub(crate) use resources::{prepare_volumetrics_resources, VolumetricsTextureCache};
pub(crate) use settings::PrismVolumetricsSettings;
