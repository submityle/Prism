//! Main-world configuration resource for the NPR stylized front end.
//!
//! The lighting *math* for the stylized (NPR) illumination axis lives in the
//! backend-neutral golden reference [`prism_render_shading::StylizedParams`] and
//! its GPU mirror ([`super::abi::GpuStylizedParams`] / the `lighting.wesl`
//! twin).  Because `prism_render_shading` does not depend on `bevy_ecs`, it
//! cannot itself be an ECS [`Resource`]; this thin wrapper lives in the scene
//! crate so applications can drive the frame-global stylized look from the main
//! world.
//!
//! Extraction ([`super::extract::extract_lights`]) copies these parameters into
//! the per-frame [`super::abi::GpuLightEnvironment`] the resolve pass consumes.
//! The [`Default`] value is [`StylizedParams::default`], i.e. the historical
//! four-band toon lobe, so simply installing this resource is a behavioural
//! no-op until the parameters are mutated.

use bevy_ecs::prelude::Resource;
use prism_render_shading::StylizedParams;

/// Frame-global controls for the stylized (NPR) illumination axis.
///
/// This is a per-frame *look* control, not a per-material one: it sets the
/// default cel ramp, half-Lambert wrap, stepped shadow, stylized specular, and
/// rim response the resolve pass applies to stylized-shaded surfaces.  It is
/// deliberately orthogonal to the surface `illumination` axis (Lit / Stylized /
/// Unlit / Custom) and does not preclude later per-material overrides or light
/// layers; it is the frame-constant baseline those refinements build on.
///
/// The wrapped [`StylizedParams`] defaults to [`StylizedParams::default`] (the
/// legacy four-band toon lobe), so inserting this resource without mutating it
/// leaves the shaded result bit-for-bit identical to the historical toon path.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Default)]
pub struct StylizedLighting {
    /// The stylized front-end parameters uploaded to the GPU each frame.
    pub params: StylizedParams,
}

impl StylizedLighting {
    /// Builds a configuration from explicit stylized parameters.
    #[must_use]
    pub const fn new(params: StylizedParams) -> Self {
        Self { params }
    }
}

impl From<StylizedParams> for StylizedLighting {
    fn from(params: StylizedParams) -> Self {
        Self { params }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_legacy_toon() {
        assert_eq!(StylizedLighting::default().params, StylizedParams::default());
    }

    #[test]
    fn new_and_from_wrap_params() {
        let params = StylizedParams::with_bands(5);
        assert_eq!(StylizedLighting::new(params).params, params);
        assert_eq!(StylizedLighting::from(params).params, params);
    }
}
