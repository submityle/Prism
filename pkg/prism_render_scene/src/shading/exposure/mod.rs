//! Exposure and eye-adaptation subsystem.
//!
//! The CPU golden lives in [`prism_render_shading::exposure`] and the shader
//! twin in `shaders/exposure.wesl`; this module is the render-world plumbing
//! that will run them. Exposure turns a physical camera (aperture / shutter /
//! ISO) or a measured scene luminance into the single scalar the shading pass
//! multiplies radiance by before it is written to the HDR buffer. That
//! *pre-exposure* keeps HDR values in a well-conditioned float range and drives
//! temporal *eye adaptation*; it is distinct from the display tone-map curve
//! applied later in the post chain.
//!
//! Exposure is a shared post-processing base, not a peer of the PBR/NPR shading
//! fronts: every illumination model writes pre-exposed radiance through the
//! same multiplier, and a stylized front can drive the same auto-exposure
//! metering (or pin a fixed EV for a flat, illustrative look). The remaining
//! slices land the luminance-histogram build over the resolved HDR target, the
//! percentile-trimmed average reduction, the eye-adaptation state carried
//! across frames, and the pre-exposure multiply consumed by the resolve pass —
//! each with its first live consumer so no committed ABI is dead, matching the
//! SSR / SSGI / volumetrics precedent.

mod abi;
mod average;
mod histogram;
mod resources;

#[cfg(test)]
mod shader_tests;

pub(crate) use average::{
    exposure_average_pass, init_exposure_average_pipeline, prepare_exposure_average_bind_groups,
};
pub(crate) use histogram::{
    exposure_histogram_pass, init_exposure_histogram_pipeline,
    prepare_exposure_histogram_bind_groups,
};
pub(crate) use resources::{prepare_exposure_buffers, ViewExposureBuffers};
