//! Render-world extraction of the active environment probe's radiance source.
//!
//! The prefilter precompute needs to know *which* GPU cube-map to convolve.
//! Bevy's [`EnvironmentMapLight`] lives only in the main world, so this
//! `ExtractSchedule` system mirrors the first visible probe's `specular_map`
//! asset id and intensity into [`ExtractedIblSource`].  The prefilter pass then
//! looks the id up in `RenderAssets<GpuImage>` to bind the source cube.
//!
//! Selection matches [`crate::lighting::extract`]'s SH probe: the base mip of
//! the `specular_map` is the raw radiance environment, and the first visible
//! probe wins so the prefiltered specular and the diffuse SH agree on a source.

use bevy_asset::AssetId;
use bevy_camera::visibility::ViewVisibility;
use bevy_ecs::prelude::*;
use bevy_image::Image;
use bevy_light::EnvironmentMapLight;
use bevy_render::Extract;

/// The radiance cube-map the prefilter pass should convolve this frame.
///
/// `specular_map` is [`None`] when no visible environment probe exists, in
/// which case the prefilter pass stays idle and the resolve keeps its ambient
/// fallback.  `intensity` scales the prefiltered radiance to match the probe's
/// exposure, mirroring the SH probe's per-light scaling.
#[derive(Resource, Default, Debug, Clone, PartialEq)]
pub(crate) struct ExtractedIblSource {
    /// Asset id of the first visible probe's radiance `specular_map`, if any.
    pub(crate) specular_map: Option<AssetId<Image>>,
    /// The probe's intensity multiplier applied to the prefiltered radiance.
    pub(crate) intensity: f32,
}

/// `ExtractSchedule` system recording the active radiance source into
/// [`ExtractedIblSource`].
///
/// Hidden probes are skipped so a later visible probe can still win; if none
/// qualify the source is cleared to [`None`].
pub(crate) fn extract_ibl_source(
    mut extracted: ResMut<ExtractedIblSource>,
    environment_maps: Extract<Query<(&EnvironmentMapLight, Option<&ViewVisibility>)>>,
) {
    extracted.specular_map = None;
    extracted.intensity = 0.0;

    for (map, visibility) in &environment_maps {
        if visibility.is_some_and(|visibility| !visibility.get()) {
            continue;
        }
        extracted.specular_map = Some(map.specular_map.id());
        extracted.intensity = map.intensity;
        return;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_source_is_empty() {
        let source = ExtractedIblSource::default();
        assert!(source.specular_map.is_none());
        assert_eq!(source.intensity, 0.0);
    }
}
