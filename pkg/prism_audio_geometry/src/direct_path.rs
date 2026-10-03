//! The direct line-of-sight arrival and its transmission through partitions.
//!
//! The direct path is the straight line from emitter to listener. In an open
//! field it is unobstructed; in a built scene it may cross one or more
//! partitions, each attenuating the sound by its transmission loss. This module
//! marches that segment through [`AcousticScene`], multiplies the per-surface
//! transmission gains, and reports both a
//! [`PropagationPath`] for the (possibly transmitted) direct arrival and the
//! [`OcclusionFactors`] the occlusion model consumes.
//!
//! # Gain convention
//!
//! A path's `gain` is expressed *relative to the direct free-field arrival at
//! the emitter's base distance*: a clear direct path is `1.0`, and a path that
//! crosses partitions carries the product of their linear transmission gains.
//! Distance attenuation itself is applied downstream from the base distance, so
//! this module never folds spherical spreading into the direct gain. Secondary
//! arrivals (reflections, diffraction) fold the extra spreading of their longer
//! routes relative to this same base distance.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//!
//! Consumes [`crate::scene::AcousticScene`] and the spatial crate's
//! [`transmission`](prism_audio_spatial::propagation) acoustics; its result is
//! assembled by [`crate::backend::GeometricBackend`].

use prism_audio_core::math::Sample;
use prism_audio_spatial::doppler::SPEED_OF_SOUND_MPS;
use prism_audio_spatial::geometry::{Emitter, Listener};
use prism_audio_spatial::occlusion::OcclusionFactors;
use prism_audio_spatial::propagation::{
    AcousticMaterial, PathKind, PropagationPath, FULL_BAND_CUTOFF_HZ,
};

use crate::config::GeometricConfig;
use crate::scene::AcousticScene;

/// The resolved direct arrival plus the blocking it implies.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirectResult {
    /// The direct/transmission arrival. Only meaningful when `audible` is true.
    pub path: PropagationPath,
    /// Blocking of the direct line of sight, feeding the occlusion model and
    /// the reverberant send regardless of whether `path` is audible.
    pub occlusion: OcclusionFactors,
    /// Whether `path` should be rendered. A fully blocked path with
    /// transmission disabled, or one quieter than the audibility floor, is not
    /// audible even though `occlusion` still describes the geometry.
    pub audible: bool,
    /// Straight-line emitter-to-listener distance (metres), the base distance
    /// secondary arrivals attenuate against.
    pub base_distance: Sample,
}

/// Resolves the direct arrival of `emitter` at `listener` through `scene`.
#[must_use]
pub fn resolve_direct(
    scene: &AcousticScene,
    listener: &Listener,
    emitter: &Emitter,
    config: &GeometricConfig,
) -> DirectResult {
    let local = listener.localize(emitter);
    let distance = local.distance;
    let delay_seconds = distance / SPEED_OF_SOUND_MPS;

    // Walk every partition between listener and emitter, folding in each
    // transmission gain. An empty scene (or a clear line) leaves the product at
    // unity.
    let mut transmitted = 1.0_f32;
    let mut crossings = 0_u32;
    scene.march_segment(
        listener.position,
        emitter.position,
        config.surface_epsilon_m,
        |hit| {
            transmitted *= hit.material.transmission_gain();
            crossings += 1;
            // Stop once the running product is already below the floor: nothing
            // beyond can make it audible again.
            transmitted > config.min_gain
        },
    );

    if crossings == 0 {
        // Clear line of sight: a unity, full-band direct arrival.
        return DirectResult {
            path: PropagationPath {
                kind: PathKind::Direct,
                delay_seconds,
                gain: 1.0,
                cutoff_hz: FULL_BAND_CUTOFF_HZ,
                direction: local.direction,
            },
            occlusion: OcclusionFactors::OPEN,
            audible: true,
            base_distance: distance,
        };
    }

    // Blocked: the surviving fraction is the transmitted product. Both the
    // direct and reverberant blocking scale with how much was stopped.
    let blocked = (1.0 - transmitted).clamp(0.0, 1.0);
    let occlusion = OcclusionFactors::new(blocked, blocked);
    let audible = config.transmission_enabled && transmitted > config.min_gain;

    DirectResult {
        path: PropagationPath {
            kind: PathKind::Transmission,
            delay_seconds,
            gain: if audible { transmitted } else { 0.0 },
            cutoff_hz: FULL_BAND_CUTOFF_HZ,
            direction: local.direction,
        },
        occlusion,
        audible,
        base_distance: distance,
    }
}

/// The linear transmission gain accumulated by crossing `materials` in order.
///
/// A small helper mirroring the per-surface fold used by [`resolve_direct`],
/// exposed for callers that already hold the partition list (for example a
/// reflection sub-segment that passes through a window).
#[must_use]
pub fn transmission_product(materials: impl IntoIterator<Item = AcousticMaterial>) -> Sample {
    let mut product = 1.0_f32;
    for material in materials {
        product *= material.transmission_gain();
    }
    product.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::{resolve_direct, transmission_product};
    use alloc::vec;
    use bevy_math::Vec3;
    use prism_audio_spatial::geometry::{Emitter, Listener};
    use prism_audio_spatial::propagation::{AcousticMaterial, PathKind};

    use crate::config::GeometricConfig;
    use crate::material_map::MaterialTable;
    use crate::scene::AcousticScene;

    fn wall(material: AcousticMaterial) -> AcousticScene {
        let vertices = vec![
            Vec3::new(0.0, -2.0, -2.0),
            Vec3::new(0.0, 2.0, -2.0),
            Vec3::new(0.0, 2.0, 2.0),
            Vec3::new(0.0, -2.0, 2.0),
        ];
        let indices = vec![[0, 1, 2], [0, 2, 3]];
        AcousticScene::new(vertices, indices, MaterialTable::uniform(material)).unwrap()
    }

    fn empty_scene() -> AcousticScene {
        AcousticScene::new(vec![], vec![], MaterialTable::default()).unwrap()
    }

    #[test]
    fn clear_line_is_unity_direct() {
        let scene = empty_scene();
        let listener = Listener::default();
        let emitter = Emitter::point(Vec3::new(0.0, 0.0, -10.0), Vec3::ZERO);
        let cfg = GeometricConfig::new(48_000);
        let r = resolve_direct(&scene, &listener, &emitter, &cfg);
        assert!(r.audible);
        assert_eq!(r.path.kind, PathKind::Direct);
        assert!((r.path.gain - 1.0).abs() < 1e-6);
        assert_eq!(r.occlusion, prism_audio_spatial::occlusion::OcclusionFactors::OPEN);
        assert!((r.base_distance - 10.0).abs() < 1e-4);
    }

    #[test]
    fn wall_transmits_and_blocks() {
        // 6 dB transmission loss -> ~0.5 linear.
        let scene = wall(AcousticMaterial::new(6.0206, 0.0));
        let listener = Listener::new(Vec3::new(-3.0, 0.0, 0.0), bevy_math::Quat::IDENTITY, Vec3::ZERO);
        let emitter = Emitter::point(Vec3::new(3.0, 0.0, 0.0), Vec3::ZERO);
        let cfg = GeometricConfig::new(48_000);
        let r = resolve_direct(&scene, &listener, &emitter, &cfg);
        assert!(r.audible);
        assert_eq!(r.path.kind, PathKind::Transmission);
        assert!((r.path.gain - 0.5).abs() < 2e-2);
        assert!(r.occlusion.obstruction > 0.4 && r.occlusion.obstruction < 0.6);
    }

    #[test]
    fn transmission_disabled_silences_but_reports_blocking() {
        let scene = wall(AcousticMaterial::new(40.0, 0.0));
        let listener = Listener::new(Vec3::new(-3.0, 0.0, 0.0), bevy_math::Quat::IDENTITY, Vec3::ZERO);
        let emitter = Emitter::point(Vec3::new(3.0, 0.0, 0.0), Vec3::ZERO);
        let cfg = GeometricConfig::new(48_000);
        let disabled = GeometricConfig {
            transmission_enabled: false,
            ..cfg
        };
        let r = resolve_direct(&scene, &listener, &emitter, &disabled);
        assert!(!r.audible);
        assert!(r.occlusion.obstruction > 0.9);
    }

    #[test]
    fn transmission_product_folds_materials() {
        let half = AcousticMaterial::new(6.0206, 0.0);
        let p = transmission_product([half, half]);
        assert!((p - 0.25).abs() < 2e-2);
        assert!((transmission_product([]) - 1.0).abs() < 1e-6);
    }
}
