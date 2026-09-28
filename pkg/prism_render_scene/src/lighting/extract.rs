//! Render-world extraction of Bevy lights into the flat GPU light ABI.
//!
//! This system runs in [`ExtractSchedule`] and mirrors every enabled
//! [`DirectionalLight`], [`PointLight`], and [`SpotLight`] plus the
//! [`GlobalAmbientLight`] resource into the byte-for-byte records the resolve
//! compute shader consumes.  The unit conversions are intentionally faithful to
//! Bevy's own render-world extraction (`bevy_pbr/src/render/light.rs`) so the
//! GPU path and the CPU golden reference agree numerically:
//!
//! * A directional light's illuminance is `color_linear * illuminance_lux`, and
//!   its direction-to-light is `GlobalTransform::back()` (the local `+Z` axis,
//!   i.e. the opposite of the light's forward ray into the scene).
//! * A point/spot light's luminous power (lumens) is converted to radiant
//!   intensity (candela) with `intensity / (4 * PI)`, matching
//!   `light.rs:{574,713}`.
//! * A spot light's cone axis is `GlobalTransform::forward()` (local `-Z`); its
//!   outer half-angle is clamped strictly below `PI / 2` and the inner angle is
//!   clamped to the outer, exactly as Bevy clamps `SpotLight` before upload.
//! * Ambient light is `color_linear * brightness`; this is a deliberate
//!   approximation of Bevy's `cd/m^2` ambient term onto the reference's flat
//!   ambient input, which the resolve pass folds into the indirect term.

use bevy_asset::Assets;
use bevy_color::{Color, ColorToComponents};
use bevy_ecs::prelude::*;
use bevy_image::Image;
use bevy_light::{
    DirectionalLight, EnvironmentMapLight, GlobalAmbientLight, PointLight, SpotLight,
};
use bevy_math::ops;
use bevy_camera::visibility::ViewVisibility;
use bevy_render::Extract;
use bevy_transform::components::GlobalTransform;
use core::f32::consts::{FRAC_PI_2, PI};
use prism_render_shading::{PunctualLight, SphericalHarmonicsL2};

use super::abi::{GpuDirectionalLight, GpuLightEnvironment, GpuPunctualLight};
use super::probe::EnvironmentProbeCache;

/// Reciprocal of the full sphere solid angle used to turn a punctual light's
/// luminous power (lumens) into radiant intensity (candela).
const INV_FOUR_PI: f32 = 1.0 / (4.0 * PI);

/// Largest outer half-angle a spot cone may use; Bevy requires the outer angle
/// to stay strictly below `PI / 2` so the cone projection remains finite.
const MAX_SPOT_OUTER_ANGLE: f32 = FRAC_PI_2 - 1.0e-4;

/// The lighting inputs extracted from the main world for a single frame.
///
/// The three collections mirror the three GPU storage buffers one-to-one; the
/// environment carries the frame-constant ambient/probe state plus the light
/// counts the shader loops over.
#[derive(Resource, Default, Debug, Clone, PartialEq)]
pub struct ExtractedLights {
    /// Every enabled directional light, already in world space.
    pub directionals: Vec<GpuDirectionalLight>,
    /// Every enabled point and spot light, already in world space.
    pub punctuals: Vec<GpuPunctualLight>,
    /// Frame-constant ambient/probe environment and light counts.
    pub environment: GpuLightEnvironment,
}

impl ExtractedLights {
    /// Clears the per-frame collections while keeping the allocations.
    fn clear(&mut self) {
        self.directionals.clear();
        self.punctuals.clear();
        self.environment = GpuLightEnvironment::default();
    }

    /// Refreshes the environment counts from the current buffer lengths.
    fn sync_counts(&mut self) {
        self.environment.directional_count = self.directionals.len() as u32;
        self.environment.punctual_count = self.punctuals.len() as u32;
    }
}

/// Returns the linear RGB triple of a Bevy [`Color`], dropping alpha.
fn linear_rgb(color: Color) -> [f32; 3] {
    let [red, green, blue, _] = color.to_linear().to_f32_array();
    [red, green, blue]
}

/// Scales a linear RGB triple by a scalar.
fn scale_rgb(rgb: [f32; 3], scalar: f32) -> [f32; 3] {
    [rgb[0] * scalar, rgb[1] * scalar, rgb[2] * scalar]
}

/// Extracts every enabled light and the ambient environment into
/// [`ExtractedLights`].
///
/// Lights whose [`ViewVisibility`] has been computed to hidden are skipped so
/// the GPU buffers only carry contributing emitters.  Entities without a
/// `ViewVisibility` component (the common case for lights added without the
/// full visibility bundle) are treated as visible.
pub(crate) fn extract_lights(
    mut extracted: ResMut<ExtractedLights>,
    ambient: Extract<Option<Res<GlobalAmbientLight>>>,
    directionals: Extract<
        Query<(&DirectionalLight, &GlobalTransform, Option<&ViewVisibility>)>,
    >,
    points: Extract<Query<(&PointLight, &GlobalTransform, Option<&ViewVisibility>)>>,
    spots: Extract<Query<(&SpotLight, &GlobalTransform, Option<&ViewVisibility>)>>,
    environment_maps: Extract<Query<(&EnvironmentMapLight, Option<&ViewVisibility>)>>,
    images: Extract<Res<Assets<Image>>>,
    mut probe_cache: ResMut<EnvironmentProbeCache>,
) {
    extracted.clear();

    if let Some(ambient) = ambient.as_deref() {
        extracted.environment.ambient =
            scale_rgb(linear_rgb(ambient.color), ambient.brightness);
    }

    // Image-based lighting: project the first readable environment probe into
    // the SH radiance vector the resolve pass prefers over the flat ambient
    // term.  Compressed or unreadable maps leave `has_image_based` false so the
    // ambient fallback above still applies.
    extract_environment_probe(
        &mut extracted.environment,
        &environment_maps,
        &images,
        &mut probe_cache,
    );

    for (light, transform, visibility) in &directionals {
        if is_hidden(visibility) {
            continue;
        }
        let direction_to_light = transform.back();
        extracted.directionals.push(GpuDirectionalLight {
            direction_to_light: [
                direction_to_light.x,
                direction_to_light.y,
                direction_to_light.z,
            ],
            visibility: 1.0,
            illuminance: scale_rgb(linear_rgb(light.color), light.illuminance),
            _padding: 0.0,
        });
    }

    for (light, transform, visibility) in &points {
        if is_hidden(visibility) {
            continue;
        }
        let intensity = scale_rgb(linear_rgb(light.color), light.intensity * INV_FOUR_PI);
        let position = transform.translation();
        let point = PunctualLight::point(
            [position.x, position.y, position.z],
            intensity,
            light.range,
        );
        extracted.punctuals.push(GpuPunctualLight::from(point));
    }

    for (light, transform, visibility) in &spots {
        if is_hidden(visibility) {
            continue;
        }
        let intensity = scale_rgb(linear_rgb(light.color), light.intensity * INV_FOUR_PI);
        let position = transform.translation();
        let axis = transform.forward();
        // Match Bevy's spot clamps: keep the outer angle strictly inside a
        // hemisphere and the inner angle at or inside the outer angle.
        let outer_angle = light.outer_angle.clamp(0.0, MAX_SPOT_OUTER_ANGLE);
        let inner_angle = light.inner_angle.clamp(0.0, outer_angle);
        let spot = PunctualLight::spot(
            [position.x, position.y, position.z],
            intensity,
            light.range,
            [axis.x, axis.y, axis.z],
            ops::cos(inner_angle),
            ops::cos(outer_angle),
        );
        extracted.punctuals.push(GpuPunctualLight::from(spot));
    }

    extracted.sync_counts();
}

/// Scales every SH coefficient of a probe by `intensity`.
fn scale_probe(probe: &mut SphericalHarmonicsL2, intensity: f32) {
    for coefficient in probe.coefficients.iter_mut() {
        coefficient[0] *= intensity;
        coefficient[1] *= intensity;
        coefficient[2] *= intensity;
    }
}

/// Projects the first visible, CPU-readable [`EnvironmentMapLight`] into
/// `environment` as an SH radiance probe scaled by the light's intensity.
///
/// The `specular_map` base mip is the raw radiance environment, so it is the
/// source the golden probe expects; the diffuse map is already pre-convolved.
/// Maps that cannot be projected (missing, compressed, non-cube) are skipped so
/// a later readable probe can still win, and the ambient fallback stands if
/// none qualify.
fn extract_environment_probe(
    environment: &mut GpuLightEnvironment,
    environment_maps: &Query<(&EnvironmentMapLight, Option<&ViewVisibility>)>,
    images: &Assets<Image>,
    probe_cache: &mut EnvironmentProbeCache,
) {
    for (map, visibility) in environment_maps {
        if is_hidden(visibility) {
            continue;
        }
        let handle = &map.specular_map;
        let Some(image) = images.get(handle) else {
            continue;
        };
        let Some(mut probe) = probe_cache.get_or_project(handle.id(), image) else {
            continue;
        };
        scale_probe(&mut probe, map.intensity);
        environment.set_spherical_harmonics(&probe);
        return;
    }
}

/// A light is contributing unless it carries an explicitly-hidden
/// [`ViewVisibility`].
fn is_hidden(visibility: Option<&ViewVisibility>) -> bool {
    visibility.is_some_and(|visibility| !visibility.get())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_rgb_drops_alpha() {
        let rgb = linear_rgb(Color::WHITE);
        assert!((rgb[0] - 1.0).abs() < 1.0e-6);
        assert!((rgb[1] - 1.0).abs() < 1.0e-6);
        assert!((rgb[2] - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn scale_rgb_multiplies_each_channel() {
        assert_eq!(scale_rgb([1.0, 2.0, 3.0], 2.0), [2.0, 4.0, 6.0]);
    }

    #[test]
    fn hidden_view_visibility_is_skipped() {
        assert!(!is_hidden(None));
    }

    #[test]
    fn counts_track_collection_lengths() {
        let mut lights = ExtractedLights::default();
        lights.directionals.push(GpuDirectionalLight::default());
        lights.punctuals.push(GpuPunctualLight::default());
        lights.punctuals.push(GpuPunctualLight::default());
        lights.sync_counts();
        assert_eq!(lights.environment.directional_count, 1);
        assert_eq!(lights.environment.punctual_count, 2);
    }

    #[test]
    fn clear_resets_environment_and_collections() {
        let mut lights = ExtractedLights::default();
        lights.directionals.push(GpuDirectionalLight::default());
        lights.environment.ambient = [1.0, 1.0, 1.0];
        lights.clear();
        assert!(lights.directionals.is_empty());
        assert_eq!(lights.environment.ambient, [0.0, 0.0, 0.0]);
    }
}
