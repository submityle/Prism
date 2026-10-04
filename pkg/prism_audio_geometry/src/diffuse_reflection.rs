//! Diffuse (scattered) first-order reflection field.
//!
//! A real wall never reflects like a mirror. The classic image-source bounce
//! in [`crate::reflection_path`] models only the *specular* share of a
//! reflection — the fraction `sqrt(1 - s)` of the surface's reflected energy
//! that leaves at the mirror angle, where `s` is the ISO 17497 scattering
//! coefficient reported by
//! [`BandedAcousticMaterial::scattering`](prism_audio_spatial::material_spectrum::BandedAcousticMaterial::scattering).
//! The complementary `sqrt(s)` share — the energy a rough, diffusing surface
//! scatters away from the mirror angle, exposed as
//! [`BandedAcousticMaterial::diffuse_reflection`](prism_audio_spatial::material_spectrum::BandedAcousticMaterial::diffuse_reflection)
//! — had no consumer: every surface in the scene silently discarded it, so a
//! stone cavern with plaster-rough walls radiated the same early field as a
//! hall of mirrors. That is audibly wrong. The scattered energy is exactly
//! what fills the gaps between the discrete specular images, builds the dense
//! early diffuse response of a reverberant room, and softens the comb-filtered
//! ring of a purely specular model.
//!
//! This module is that missing consumer. For a given source and listener it
//! walks every triangle the scene exposes and, for each surface visible to
//! both, gathers the Lambert-weighted diffuse energy the surface scatters from
//! the source toward the listener. Because the many scattering surfaces are
//! mutually incoherent — their path lengths differ by far more than a
//! wavelength and their phases are unrelated — their contributions add as
//! *energy*, not as pressure: the field sums each surface's per-band squared
//! amplitude and takes the root of the total. The result is one aggregate
//! reverberant send, [`DiffuseReflectionField`], carrying a per-band diffuse
//! gain, its broadband magnitude, and the energy-weighted mean arrival delay
//! that places the diffuse cloud in time relative to the direct sound.
//!
//! # Gain convention
//!
//! Every gain this module produces is a *relative* linear amplitude referenced
//! to the free-field direct arrival at the query's `base_distance`, matching
//! the convention the specular ([`crate::reflection_path`]) and diffraction
//! ([`crate::diffraction_path`]) builders already use: a surface's diffuse
//! contribution is its diffuse reflection coefficient scaled by the geometric
//! spreading ratio `base_distance / path_length` and the Lambertian radiation
//! weight, and the aggregate per-band send is clamped to `0.0..=1.0` so the
//! diffuse field can never exceed the direct reference it is measured against.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML. The
//! model is textbook geometrical room acoustics: Lambert's cosine law for
//! diffuse radiation (scattered amplitude proportional to
//! `sqrt(cos_in * cos_out)`), incoherent energy superposition of statistically
//! independent reflections (Kuttruff, *Room Acoustics*), and the same
//! `base_distance / path_length` spreading law and `BandGains` energy algebra
//! the rest of this crate already applies.
//!
//! # Relationship
//!
//! [`crate::reflection_path`] takes a surface's
//! [`specular_reflection`](prism_audio_spatial::material_spectrum::BandedAcousticMaterial::specular_reflection)
//! and routes each bounce as a discrete, delayed, *coherent*
//! [`PropagationPath`](prism_audio_spatial::propagation::PropagationPath) the
//! voice renders through its own delay line. This module takes the orthogonal
//! [`diffuse_reflection`](prism_audio_spatial::material_spectrum::BandedAcousticMaterial::diffuse_reflection)
//! split those paths discard and collapses the whole scattering surface set
//! into a single *incoherent* [`DiffuseReflectionField`] — a reverberant send a
//! downstream late/diffuse stage (design sections 14 and 17) feeds into its
//! decorrelated reverberator rather than a per-sample delay line. The two are
//! complementary halves of the same reflected energy and never double-count:
//! together, specular `(sqrt(1 - s))^2` plus diffuse `(sqrt(s))^2` conserve the
//! surface's total reflected energy `s + (1 - s) = 1`.

use bevy_math::ops;
use prism_audio_core::math::Sample;
use prism_audio_spatial::band_spectrum::BandGains;
use prism_audio_spatial::doppler::SPEED_OF_SOUND_MPS;
use prism_audio_spatial::geometry::{Emitter, Listener};
use prism_audio_spatial::PROPAGATION_BAND_COUNT;

use crate::config::GeometricConfig;
use crate::diffraction_edges::distance;
use crate::scene::AcousticScene;

/// The aggregate diffuse (scattered) first-order reflection arriving at the
/// listener, summed incoherently across every scattering surface in the scene.
///
/// This is a single reverberant *send*, not a per-path arrival: a downstream
/// diffuse/late stage drives its decorrelated reverberator with [`send`] and
/// positions the diffuse cloud in time with [`mean_delay_seconds`]. It is the
/// counterpart to the discrete specular bounces
/// [`crate::reflection_path`] produces.
///
/// [`send`]: DiffuseReflectionField::send
/// [`mean_delay_seconds`]: DiffuseReflectionField::mean_delay_seconds
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiffuseReflectionField {
    /// Per-band diffuse send: the incoherent sum of every contributing
    /// surface's scattered amplitude, expressed as a relative linear gain in
    /// `0.0..=1.0` referenced to the free-field direct arrival (see the
    /// module's gain convention).
    pub send: BandGains,
    /// Broadband magnitude of [`send`](Self::send), the
    /// [`broadband_rms`](BandGains::broadband_rms) of the per-band diffuse
    /// gains; a convenient scalar level for gating and metering.
    pub broadband_gain: Sample,
    /// Energy-weighted mean arrival delay of the diffuse cloud, in seconds:
    /// `sum(surface_energy * path_delay) / sum(surface_energy)`, where each
    /// surface's delay is its total source-to-surface-to-listener path length
    /// divided by the speed of sound. Positions the diffuse send in time
    /// relative to the direct arrival.
    pub mean_delay_seconds: Sample,
    /// Number of scattering surfaces that contributed audible diffuse energy to
    /// this field (visible to both source and listener and on the source side).
    pub contributing_surfaces: usize,
}

impl DiffuseReflectionField {
    /// A field with no diffuse energy: silent send, zero gain, zero delay, no
    /// contributing surfaces. Returned when scattering is disabled, the scene
    /// is empty, or no surface scatters audible energy toward the listener.
    pub const SILENT: Self = Self {
        send: BandGains::SILENT,
        broadband_gain: 0.0,
        mean_delay_seconds: 0.0,
        contributing_surfaces: 0,
    };

    /// Returns `true` when the field's broadband gain exceeds the supplied
    /// audibility `floor`, i.e. the diffuse send is worth rendering.
    pub fn is_audible(&self, floor: Sample) -> bool {
        self.broadband_gain > floor
    }
}

impl Default for DiffuseReflectionField {
    fn default() -> Self {
        Self::SILENT
    }
}

/// Lambertian radiation weight for a diffuse bounce, as a linear *amplitude*
/// factor.
///
/// A perfectly diffusing (Lambertian) surface radiates scattered power
/// proportional to `cos_in * cos_out`, where `cos_in` is the cosine of the
/// incidence angle (source direction to surface normal) and `cos_out` the
/// cosine of the exitance angle (surface normal to listener direction). The
/// corresponding amplitude weight is the square root of that power,
/// `sqrt(max(cos_in, 0) * max(cos_out, 0))`; negative cosines (a grazing or
/// back-facing geometry) clamp to zero so a surface the source or listener
/// cannot see from the front contributes nothing.
pub fn lambert_weight(cos_in: Sample, cos_out: Sample) -> Sample {
    let a = cos_in.max(0.0);
    let b = cos_out.max(0.0);
    ops::sqrt(a * b)
}

/// Resolves the aggregate diffuse (scattered) first-order reflection field for
/// a source and listener in a scene.
///
/// Walks every triangle the scene exposes and, for each surface the source and
/// listener can both see on the same side, accumulates the Lambert-weighted
/// diffuse energy it scatters from the source toward the listener, then sums
/// those mutually incoherent contributions in the energy domain into a single
/// [`DiffuseReflectionField`]. `base_distance` is the free-field direct
/// distance each surface's gain is referenced against (the same reference the
/// specular and diffraction builders use).
///
/// Returns [`DiffuseReflectionField::SILENT`] when reflections are disabled
/// ([`GeometricConfig::reflections_enabled`] is `false` or
/// [`GeometricConfig::max_reflections`] is `0`), the scene is empty, no surface
/// scatters toward the listener, or the aggregate broadband send does not clear
/// [`GeometricConfig::min_gain`].
///
/// The computation runs at control rate, allocates nothing on the heap, never
/// panics, and is deterministic: surfaces are visited in ascending triangle
/// index order and all arithmetic flows through
/// [`bevy_math::ops`](bevy_math::ops) for cross-platform-stable `f32` results.
pub fn resolve_diffuse_reflections(
    scene: &AcousticScene,
    listener: &Listener,
    emitter: &Emitter,
    config: &GeometricConfig,
    base_distance: Sample,
) -> DiffuseReflectionField {
    if !config.reflections_enabled || config.max_reflections == 0 || scene.is_empty() {
        return DiffuseReflectionField::SILENT;
    }

    let eps = config.surface_epsilon_m.max(0.0);
    let mut energy = [0.0 as Sample; PROPAGATION_BAND_COUNT];
    let mut delay_energy_sum: Sample = 0.0;
    let mut total_energy_weight: Sample = 0.0;
    let mut surfaces: usize = 0;

    for triangle in 0..scene.triangle_count() {
        let Some([a, b, c]) = scene.triangle(triangle) else {
            continue;
        };
        let Some(normal) = scene.triangle_normal(triangle) else {
            continue;
        };

        // Both legs of the scattering path meet at the surface centroid, the
        // representative point for a first-order diffuse bounce off the face.
        let centroid = (a + b + c) / 3.0;
        let to_source = emitter.position - centroid;
        let to_listener = listener.position - centroid;
        let d_source = distance(emitter.position, centroid);
        let d_listener = distance(listener.position, centroid);
        if d_source <= 0.0 || d_listener <= 0.0 {
            continue;
        }

        // Incidence and exitance cosines against the surface normal. The mesh
        // normal has an arbitrary winding-defined sign, so orient it toward the
        // source: once the source sits on the positive side, the listener must
        // too for the surface to scatter source energy toward it.
        let unit_source = to_source / d_source;
        let unit_listener = to_listener / d_listener;
        let mut cos_in = unit_source.dot(normal);
        let mut cos_out = unit_listener.dot(normal);
        if cos_in < 0.0 {
            cos_in = -cos_in;
            cos_out = -cos_out;
        }
        if cos_in <= 0.0 || cos_out <= 0.0 {
            continue;
        }

        // Both legs must be unobstructed for the surface to be mutually
        // visible to source and listener.
        if scene.segment_blocked(listener.position, centroid, eps)
            || scene.segment_blocked(centroid, emitter.position, eps)
        {
            continue;
        }

        let path_length = d_source + d_listener;
        let spreading = (base_distance / path_length).clamp(0.0, 1.0);
        let lambert = lambert_weight(cos_in, cos_out);
        let diffuse = scene
            .material(triangle)
            .diffuse_reflection()
            .scaled(spreading * lambert);
        let bands = diffuse.bands();

        let mut surface_energy: Sample = 0.0;
        for (slot, &gain) in energy.iter_mut().zip(bands.iter()) {
            let e = gain * gain;
            *slot += e;
            surface_energy += e;
        }
        if surface_energy <= 0.0 {
            continue;
        }

        let delay = path_length / SPEED_OF_SOUND_MPS;
        delay_energy_sum += surface_energy * delay;
        total_energy_weight += surface_energy;
        surfaces += 1;
    }

    if surfaces == 0 || total_energy_weight <= 0.0 {
        return DiffuseReflectionField::SILENT;
    }

    let mut send_bands = [0.0 as Sample; PROPAGATION_BAND_COUNT];
    for (slot, &e) in send_bands.iter_mut().zip(energy.iter()) {
        *slot = ops::sqrt(e).clamp(0.0, 1.0);
    }
    let send = BandGains::new(send_bands);
    let broadband_gain = send.broadband_rms();
    if broadband_gain <= config.min_gain {
        return DiffuseReflectionField::SILENT;
    }
    let mean_delay_seconds = delay_energy_sum / total_energy_weight;

    DiffuseReflectionField {
        send,
        broadband_gain,
        mean_delay_seconds,
        contributing_surfaces: surfaces,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use bevy_math::{Quat, Vec3};
    use prism_audio_spatial::material_spectrum::BandedAcousticMaterial;

    use crate::material_map::MaterialTable;

    const EPS: Sample = 1.0e-6;

    /// Builds a two-triangle floor on the `y = 0` plane with the given uniform
    /// reflection and scattering, spanning `x,z` in `[-10, 10]`.
    fn rough_floor(reflection: Sample, scattering: Sample) -> AcousticScene {
        let vertices = vec![
            Vec3::new(-10.0, 0.0, -10.0),
            Vec3::new(-10.0, 0.0, 10.0),
            Vec3::new(10.0, 0.0, 10.0),
            Vec3::new(10.0, 0.0, -10.0),
        ];
        let indices = vec![[0u32, 1, 2], [0, 2, 3]];
        let material = BandedAcousticMaterial::new(
            BandGains::uniform(reflection),
            BandGains::SILENT,
            scattering,
        );
        AcousticScene::new(vertices, indices, MaterialTable::uniform(material))
            .expect("floor scene builds")
    }

    /// Single-triangle half-floor variant for the incoherent-accumulation test.
    fn half_floor(reflection: Sample, scattering: Sample) -> AcousticScene {
        let vertices = vec![
            Vec3::new(-10.0, 0.0, -10.0),
            Vec3::new(-10.0, 0.0, 10.0),
            Vec3::new(10.0, 0.0, 10.0),
        ];
        let indices = vec![[0u32, 1, 2]];
        let material = BandedAcousticMaterial::new(
            BandGains::uniform(reflection),
            BandGains::SILENT,
            scattering,
        );
        AcousticScene::new(vertices, indices, MaterialTable::uniform(material))
            .expect("half floor scene builds")
    }

    fn empty_scene() -> AcousticScene {
        AcousticScene::new(vec![], vec![], MaterialTable::uniform(BandedAcousticMaterial::new(
            BandGains::SILENT,
            BandGains::SILENT,
            0.0,
        )))
        .expect("empty scene builds")
    }

    fn listener_at(position: Vec3) -> Listener {
        Listener::new(position, Quat::IDENTITY, Vec3::ZERO)
    }

    fn emitter_at(position: Vec3) -> Emitter {
        Emitter::point(position, Vec3::ZERO)
    }

    #[test]
    fn empty_scene_is_silent() {
        let field = resolve_diffuse_reflections(
            &empty_scene(),
            &listener_at(Vec3::new(-4.0, 2.0, 0.0)),
            &emitter_at(Vec3::new(4.0, 2.0, 0.0)),
            &GeometricConfig::new(48_000),
            8.0,
        );
        assert_eq!(field, DiffuseReflectionField::SILENT);
        assert_eq!(field.contributing_surfaces, 0);
    }

    #[test]
    fn reflections_disabled_is_silent() {
        let field = resolve_diffuse_reflections(
            &rough_floor(0.9, 0.5),
            &listener_at(Vec3::new(-4.0, 2.0, 0.0)),
            &emitter_at(Vec3::new(4.0, 2.0, 0.0)),
            &GeometricConfig::new(48_000).without_reflections(),
            8.0,
        );
        assert_eq!(field, DiffuseReflectionField::SILENT);
    }

    #[test]
    fn zero_max_reflections_is_silent() {
        let field = resolve_diffuse_reflections(
            &rough_floor(0.9, 0.5),
            &listener_at(Vec3::new(-4.0, 2.0, 0.0)),
            &emitter_at(Vec3::new(4.0, 2.0, 0.0)),
            &GeometricConfig::new(48_000).with_max_reflections(0),
            8.0,
        );
        assert_eq!(field, DiffuseReflectionField::SILENT);
    }

    #[test]
    fn zero_scattering_is_silent() {
        let field = resolve_diffuse_reflections(
            &rough_floor(0.9, 0.0),
            &listener_at(Vec3::new(-4.0, 2.0, 0.0)),
            &emitter_at(Vec3::new(4.0, 2.0, 0.0)),
            &GeometricConfig::new(48_000),
            8.0,
        );
        assert_eq!(field, DiffuseReflectionField::SILENT);
    }

    #[test]
    fn rough_floor_scatters_audible_diffuse_field() {
        let config = GeometricConfig::new(48_000);
        let field = resolve_diffuse_reflections(
            &rough_floor(0.9, 0.5),
            &listener_at(Vec3::new(-4.0, 2.0, 0.0)),
            &emitter_at(Vec3::new(4.0, 2.0, 0.0)),
            &config,
            8.0,
        );
        assert_eq!(field.contributing_surfaces, 2);
        assert!(field.broadband_gain > config.min_gain);
        assert!(field.mean_delay_seconds > 0.0);
        for band in field.send.bands() {
            assert!(band > 0.0 && band <= 1.0);
        }
    }

    #[test]
    fn opposite_sides_do_not_scatter() {
        // Source below the floor, listener above: the surface cannot scatter
        // the source's energy to the listener.
        let field = resolve_diffuse_reflections(
            &rough_floor(0.9, 0.5),
            &listener_at(Vec3::new(-4.0, 2.0, 0.0)),
            &emitter_at(Vec3::new(4.0, -2.0, 0.0)),
            &GeometricConfig::new(48_000),
            8.0,
        );
        assert_eq!(field, DiffuseReflectionField::SILENT);
    }

    #[test]
    fn lambert_weight_matches_cosine_law() {
        assert!((lambert_weight(1.0, 1.0) - 1.0).abs() < EPS);
        assert!(lambert_weight(-0.5, 0.5).abs() < EPS);
        assert!((lambert_weight(0.25, 0.25) - 0.25).abs() < EPS);
        assert!(lambert_weight(0.0, 0.7).abs() < EPS);
    }

    #[test]
    fn two_triangles_accumulate_over_one() {
        let config = GeometricConfig::new(48_000);
        let listener = listener_at(Vec3::new(-4.0, 2.0, 0.0));
        let emitter = emitter_at(Vec3::new(4.0, 2.0, 0.0));
        let full = resolve_diffuse_reflections(
            &rough_floor(0.9, 0.5),
            &listener,
            &emitter,
            &config,
            8.0,
        );
        let half = resolve_diffuse_reflections(
            &half_floor(0.9, 0.5),
            &listener,
            &emitter,
            &config,
            8.0,
        );
        assert!(full.contributing_surfaces >= half.contributing_surfaces);
        assert!(full.broadband_gain >= half.broadband_gain - EPS);
    }

    #[test]
    fn higher_scattering_sends_more() {
        let config = GeometricConfig::new(48_000);
        let listener = listener_at(Vec3::new(-4.0, 2.0, 0.0));
        let emitter = emitter_at(Vec3::new(4.0, 2.0, 0.0));
        let rough = resolve_diffuse_reflections(
            &rough_floor(0.9, 0.9),
            &listener,
            &emitter,
            &config,
            8.0,
        );
        let smooth = resolve_diffuse_reflections(
            &rough_floor(0.9, 0.1),
            &listener,
            &emitter,
            &config,
            8.0,
        );
        assert!(rough.broadband_gain > smooth.broadband_gain);
    }

    #[test]
    fn default_is_silent() {
        let field = DiffuseReflectionField::default();
        assert_eq!(field, DiffuseReflectionField::SILENT);
        assert!(!field.is_audible(0.0));
    }

    #[test]
    fn single_triangle_mean_delay_matches_geometry() {
        let listener = listener_at(Vec3::new(-4.0, 2.0, 0.0));
        let emitter = emitter_at(Vec3::new(4.0, 2.0, 0.0));
        let scene = half_floor(0.9, 0.5);
        let field = resolve_diffuse_reflections(
            &scene,
            &listener,
            &emitter,
            &GeometricConfig::new(48_000),
            8.0,
        );
        assert_eq!(field.contributing_surfaces, 1);
        // Centroid of the single triangle (-10,0,-10),(-10,0,10),(10,0,10).
        let centroid = (Vec3::new(-10.0, 0.0, -10.0)
            + Vec3::new(-10.0, 0.0, 10.0)
            + Vec3::new(10.0, 0.0, 10.0))
            / 3.0;
        let expected = (listener.position.distance(centroid)
            + emitter.position.distance(centroid))
            / SPEED_OF_SOUND_MPS;
        assert!((field.mean_delay_seconds - expected).abs() < 1.0e-4);
    }
}
