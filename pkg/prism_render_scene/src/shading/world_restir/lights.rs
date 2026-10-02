//! Per-frame candidate light list feeding the world-space `ReSTIR` seed pass's
//! `@binding(2)`.
//!
//! The seed kernel's streaming `RIS` draws point-emitter candidates towards
//! each occupied cell's visible point (see `shaders/world_restir_seed.wesl`),
//! so this module repacks the render world's resident extracted punctual lights
//! (owned by the prism [`crate::lighting`] subsystem) into the frozen
//! [`GpuWorldRestirLight`] layout and keeps them in a resident storage buffer
//! the seed bind group binds. It mirrors the structure of
//! [`crate::lighting::LightGpuBuffers`] — a `RawBufferVec` rebuilt and uploaded
//! every frame, padded with one neutral element so the storage binding is never
//! zero-sized in an unlit scene — scaled to the single array the seed pass
//! needs.
//!
//! Only *punctual* (point/spot) lights seed the world-space table: the
//! [`GpuWorldRestirLight`] record is a bare point emitter (a world position plus
//! linear-RGB radiance), so directional (infinitely distant) lights, which
//! carry a direction rather than a world position, have no faithful
//! point-emitter form here and are left to the resolve pass's analytic term.
//!
//! The seed pass is opt-in exactly like the fill pass: when
//! [`PrismWorldRestirSettings::enabled`] is `false` the resident list is
//! cleared and nothing is uploaded or dispatched.

use bevy_ecs::{prelude::*, world::FromWorld};
use bevy_render::{
    render_resource::{Buffer, BufferUsages, RawBufferVec},
    renderer::{RenderDevice, RenderQueue},
};

use crate::lighting::{ExtractedLights, GpuPunctualLight};

use super::abi::GpuWorldRestirLight;
use super::settings::PrismWorldRestirSettings;

/// Packs one extracted punctual light into the seed pass's point-emitter
/// candidate record.
///
/// [`GpuWorldRestirLight`] is a bare point emitter: the seed kernel's
/// `build_sample` forms the candidate radiance as
/// `color * (intensity * artistic_gain)`, so this folds the light's analytic
/// shadow `visibility` term (clamped non-negative) into the linear-RGB
/// `color` and leaves the scalar `intensity` lane at `1`. Folding visibility
/// means the `RIS` stream resamples by the *shadowed* contribution, so occluded
/// lights are proportionally less likely to survive the reservoir.
///
/// Two approximations fall out of that bare point-emitter layout, and both
/// match the CPU golden `prism_render_shading` world-`ReSTIR` estimator exactly
/// (it is likewise a point-emitter estimator with no cone or range term):
/// * the spot cone (`spot_scale` / `spot_offset` / `direction`) is dropped — a
///   spot seeds as an omnidirectional point emitter;
/// * the influence `range` window is dropped — distance falloff is the
///   estimator's pure geometric `1 / dist^2`.
pub(crate) fn pack_punctual_light(light: &GpuPunctualLight) -> GpuWorldRestirLight {
    let visibility = light.visibility.max(0.0);
    GpuWorldRestirLight {
        position: light.position,
        intensity: 1.0,
        color: [
            light.intensity[0] * visibility,
            light.intensity[1] * visibility,
            light.intensity[2] * visibility,
        ],
        _pad0: 0.0,
    }
}

/// Resident per-frame candidate light list the world-space `ReSTIR` seed pass
/// resamples from, bound at the seed group's `@binding(2)`.
#[derive(Resource)]
pub(crate) struct WorldRestirLights {
    /// Packed point-emitter candidates, one [`GpuWorldRestirLight`] per active
    /// punctual light. Padded with a single neutral element on upload when the
    /// scene is unlit so the storage binding is never zero-sized.
    lights: RawBufferVec<GpuWorldRestirLight>,
    /// Count of *active* candidates the seed pass may draw from (`0` in an unlit
    /// scene). This is the authoritative `light_count` fed to the seed immediate
    /// block — it stays `0` even when the buffer is padded to one element, so
    /// the kernel's `pick_light` is never asked to index an all-pad list.
    count: u32,
}

impl FromWorld for WorldRestirLights {
    fn from_world(_: &mut World) -> Self {
        let mut lights = RawBufferVec::new(BufferUsages::STORAGE);
        lights.set_label(Some("prism world-space ReSTIR lights"));
        Self { lights, count: 0 }
    }
}

impl WorldRestirLights {
    /// Repacks the extracted punctual lights into the candidate array and
    /// records the active count. Directional lights are intentionally skipped
    /// (see the module docs). The allocation is retained across frames.
    fn rebuild(&mut self, extracted: &ExtractedLights) {
        self.lights.clear();
        self.lights
            .extend(extracted.punctuals.iter().map(pack_punctual_light));
        self.count = extracted.punctuals.len() as u32;
    }

    /// Drops every candidate and zeroes the active count (the opt-out path when
    /// the subsystem is disabled). The retained GPU allocation is not freed, so
    /// re-enabling the subsystem reuses it without a reallocation.
    fn clear(&mut self) {
        self.lights.clear();
        self.count = 0;
    }

    /// Streams the packed candidates to the GPU. An empty list is padded with a
    /// single neutral element so the storage binding is never zero-sized; the
    /// active [`count`](Self::count) stays `0`, so this padding element is never
    /// resampled.
    fn upload(&mut self, device: &RenderDevice, queue: &RenderQueue) {
        if self.lights.is_empty() {
            self.lights.push(GpuWorldRestirLight::default());
        }
        self.lights.write_buffer(device, queue);
    }

    /// The resident candidate storage buffer, once it has been uploaded at least
    /// once. Bound at the seed group's `@binding(2)` by a follow-up slice.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "seed bind group binds this candidate buffer at @binding(2) in a follow-up \
                      slice, so no host path reads it yet"
        )
    )]
    pub(crate) fn buffer(&self) -> Option<&Buffer> {
        self.lights.buffer()
    }

    /// Count of active candidates, the authoritative `light_count` for the seed
    /// immediate block (`0` in an unlit scene, independent of the padded buffer
    /// length).
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "seed dispatch forwards this into GpuWorldRestirSeedParams::from_settings in \
                      a follow-up slice, so no host path reads it yet"
        )
    )]
    pub(crate) fn count(&self) -> u32 {
        self.count
    }
}

/// `PrepareResources` system repacking and uploading the per-frame world-space
/// `ReSTIR` candidate light list from the extracted lights.
///
/// Gated solely on [`PrismWorldRestirSettings::enabled`]: when the subsystem is
/// disabled the resident list is cleared (so a later enable starts from a clean
/// slate) and nothing is uploaded. Otherwise the extracted punctual lights are
/// repacked and streamed, ready for the seed bind group to bind at
/// `@binding(2)`.
pub(crate) fn prepare_world_restir_lights(
    settings: Res<PrismWorldRestirSettings>,
    extracted: Res<ExtractedLights>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    mut lights: ResMut<WorldRestirLights>,
) {
    if !settings.enabled {
        lights.clear();
        return;
    }
    lights.rebuild(&extracted);
    lights.upload(&device, &queue);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A punctual light with distinct, easy-to-trace fields.
    fn sample_punctual() -> GpuPunctualLight {
        GpuPunctualLight {
            position: [1.0, 2.0, 3.0],
            range: 25.0,
            intensity: [4.0, 8.0, 12.0],
            spot_scale: 2.0,
            direction: [0.0, -1.0, 0.0],
            spot_offset: -1.5,
            visibility: 0.5,
            _padding: [0.0; 3],
        }
    }

    #[test]
    fn pack_forwards_position_and_folds_visibility_into_colour() {
        let packed = pack_punctual_light(&sample_punctual());
        // Position passes straight through as the candidate's secondary point.
        assert_eq!(packed.position, [1.0, 2.0, 3.0]);
        // The scalar intensity lane is the neutral `1`; the radiance magnitude
        // lives entirely in the colour so the seed kernel's
        // `color * (intensity * gain)` reproduces it.
        assert_eq!(packed.intensity, 1.0);
        // Visibility (0.5) is folded into the linear-RGB colour.
        assert_eq!(packed.color, [2.0, 4.0, 6.0]);
        assert_eq!(packed._pad0, 0.0);
    }

    #[test]
    fn pack_clamps_negative_visibility_to_zero() {
        let mut light = sample_punctual();
        light.visibility = -3.0;
        let packed = pack_punctual_light(&light);
        // A non-physical negative shadow term never yields negative radiance.
        assert_eq!(packed.color, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn pack_with_full_visibility_preserves_the_raw_intensity() {
        let mut light = sample_punctual();
        light.visibility = 1.0;
        let packed = pack_punctual_light(&light);
        // A fully-lit emitter keeps its raw radiant intensity as the colour.
        assert_eq!(packed.color, [4.0, 8.0, 12.0]);
    }

    #[test]
    fn rebuild_packs_every_punctual_and_skips_directionals() {
        let mut world = World::new();
        let mut lights = WorldRestirLights::from_world(&mut world);
        let mut extracted = ExtractedLights::default();
        extracted.punctuals.push(sample_punctual());
        let mut second = sample_punctual();
        second.position = [9.0, 9.0, 9.0];
        extracted.punctuals.push(second);
        // A directional light must not inflate the point-emitter candidate list.
        extracted
            .directionals
            .push(crate::lighting::GpuDirectionalLight::default());

        lights.rebuild(&extracted);

        assert_eq!(lights.count(), 2);
        assert_eq!(lights.lights.values().len(), 2);
        assert_eq!(lights.lights.values()[0].position, [1.0, 2.0, 3.0]);
        assert_eq!(lights.lights.values()[1].position, [9.0, 9.0, 9.0]);
    }

    #[test]
    fn rebuild_of_an_unlit_scene_is_empty_with_zero_count() {
        let mut world = World::new();
        let mut lights = WorldRestirLights::from_world(&mut world);
        lights.rebuild(&ExtractedLights::default());
        assert_eq!(lights.count(), 0);
        assert_eq!(lights.lights.values().len(), 0);
    }

    #[test]
    fn buffer_is_absent_until_the_first_upload() {
        // `upload` needs a `RenderDevice`, which the sandbox has no GPU for, so
        // the resident buffer stays absent until a real frame uploads it. This
        // exercises the `buffer()` accessor the seed bind group will consume.
        let mut world = World::new();
        let lights = WorldRestirLights::from_world(&mut world);
        assert!(lights.buffer().is_none());
    }

    #[test]
    fn clear_drops_candidates_and_zeroes_the_count() {
        let mut world = World::new();
        let mut lights = WorldRestirLights::from_world(&mut world);
        let mut extracted = ExtractedLights::default();
        extracted.punctuals.push(sample_punctual());
        lights.rebuild(&extracted);
        assert_eq!(lights.count(), 1);

        lights.clear();
        assert_eq!(lights.count(), 0);
        assert_eq!(lights.lights.values().len(), 0);
    }
}
