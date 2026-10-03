//! The assembled geometric propagation backend.
//!
//! [`GeometricBackend`] implements
//! [`PropagationBackend`](prism_audio_spatial::propagation::PropagationBackend)
//! by tracing an [`AcousticScene`]: it resolves the direct arrival (and the
//! blocking it implies), then, within the configured budget, the first-order
//! specular reflections and the shadow-edge diffractions. The arrivals are
//! merged loudest-first into the caller's bounded path buffer, with the direct
//! arrival always first when it is audible.
//!
//! This is the real geometric sibling of the spatial crate's
//! [`FreeFieldBackend`](prism_audio_spatial::propagation::FreeFieldBackend): the
//! same trait, the same bounded output, but driven by actual geometry instead
//! of an open field.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//!
//! Implements [`prism_audio_spatial::propagation::PropagationBackend`] by
//! composing [`crate::direct_path`], [`crate::reflection_path`], and
//! [`crate::diffraction_path`] over a [`crate::scene::AcousticScene`] and a
//! [`crate::config::GeometricConfig`].

use alloc::vec::Vec;
use core::cmp::Ordering;

use prism_audio_spatial::geometry::{Emitter, Listener};
use prism_audio_spatial::occlusion::OcclusionFactors;
use prism_audio_spatial::propagation::{PropagationBackend, PropagationPath, PropagationSummary};

use crate::config::GeometricConfig;
use crate::diffraction_path::resolve_diffraction;
use crate::direct_path::resolve_direct;
use crate::reflection_path::resolve_reflections;
use crate::scene::AcousticScene;

/// A [`PropagationBackend`] that ray-traces a triangle-mesh scene to resolve the
/// direct, reflected, and diffracted arrivals of a source at a listener.
#[derive(Debug, Clone)]
pub struct GeometricBackend {
    scene: AcousticScene,
    config: GeometricConfig,
}

impl GeometricBackend {
    /// Builds a backend over `scene` with the budget and feature switches in
    /// `config`.
    #[inline]
    #[must_use]
    pub fn new(scene: AcousticScene, config: GeometricConfig) -> Self {
        Self { scene, config }
    }

    /// The scene this backend traces.
    #[inline]
    #[must_use]
    pub fn scene(&self) -> &AcousticScene {
        &self.scene
    }

    /// The control-rate configuration this backend runs with.
    #[inline]
    #[must_use]
    pub fn config(&self) -> &GeometricConfig {
        &self.config
    }

    /// Replaces the scene, keeping the configuration (for streaming geometry
    /// updates without rebuilding the backend).
    #[inline]
    pub fn set_scene(&mut self, scene: AcousticScene) {
        self.scene = scene;
    }
}

impl PropagationBackend for GeometricBackend {
    fn query(
        &self,
        listener: &Listener,
        emitter: &Emitter,
        paths: &mut [PropagationPath],
    ) -> PropagationSummary {
        if paths.is_empty() {
            return PropagationSummary {
                direct: OcclusionFactors::OPEN,
                path_count: 0,
            };
        }

        let direct = resolve_direct(&self.scene, listener, emitter, &self.config);
        let base_distance = direct.base_distance;

        let mut written = 0usize;
        if direct.audible {
            paths[0] = direct.path;
            written = 1;
        }

        // Specular reflections can arrive whether or not the direct line is
        // clear; shadow-edge diffraction only matters once the direct line is
        // at least partly blocked, so gate it on measured occlusion.
        let mut secondary: Vec<PropagationPath> =
            resolve_reflections(&self.scene, listener, emitter, &self.config, base_distance);
        if direct.occlusion.direct_factor() > 0.0 {
            secondary.extend(resolve_diffraction(
                &self.scene,
                listener,
                emitter,
                &self.config,
                base_distance,
            ));
        }
        secondary.sort_by(|lhs, rhs| rhs.gain.partial_cmp(&lhs.gain).unwrap_or(Ordering::Equal));

        for path in secondary {
            if written >= paths.len() {
                break;
            }
            paths[written] = path;
            written += 1;
        }

        PropagationSummary {
            direct: direct.occlusion,
            path_count: written,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::GeometricBackend;
    use alloc::vec;
    use bevy_math::{Quat, Vec3};
    use prism_audio_spatial::geometry::{Emitter, Listener};
    use prism_audio_spatial::propagation::{
        PathKind, PropagationBackend, PropagationPath, MAX_PROPAGATION_PATHS,
    };

    use crate::config::GeometricConfig;
    use crate::material_map::MaterialTable;
    use crate::scene::AcousticScene;

    fn buffer() -> [PropagationPath; MAX_PROPAGATION_PATHS] {
        [PropagationPath::SILENT; MAX_PROPAGATION_PATHS]
    }

    fn empty_scene() -> AcousticScene {
        AcousticScene::new(vec![], vec![], MaterialTable::default()).unwrap()
    }

    #[test]
    fn empty_buffer_reports_nothing() {
        let backend = GeometricBackend::new(empty_scene(), GeometricConfig::new(48_000));
        let listener = Listener::default();
        let emitter = Emitter::point(Vec3::new(0.0, 0.0, -5.0), Vec3::ZERO);
        let summary = backend.query(&listener, &emitter, &mut []);
        assert_eq!(summary.path_count, 0);
    }

    #[test]
    fn open_scene_is_a_single_direct_path() {
        let backend = GeometricBackend::new(empty_scene(), GeometricConfig::new(48_000));
        let listener = Listener::default();
        let emitter = Emitter::point(Vec3::new(0.0, 0.0, -5.0), Vec3::ZERO);
        let mut paths = buffer();
        let summary = backend.query(&listener, &emitter, &mut paths);
        assert_eq!(summary.path_count, 1);
        assert_eq!(paths[0].kind, PathKind::Direct);
        assert!((paths[0].gain - 1.0).abs() < 1e-6);
        assert_eq!(summary.direct, prism_audio_spatial::occlusion::OcclusionFactors::OPEN);
    }

    #[test]
    fn reflector_adds_a_second_arrival() {
        // Reflective floor at y = 0 with listener/source above it.
        let vertices = vec![
            Vec3::new(-10.0, 0.0, -10.0),
            Vec3::new(10.0, 0.0, -10.0),
            Vec3::new(10.0, 0.0, 10.0),
            Vec3::new(-10.0, 0.0, 10.0),
        ];
        let indices = vec![[0, 1, 2], [0, 2, 3]];
        let material = prism_audio_spatial::propagation::AcousticMaterial::new(0.0, 0.9);
        let scene = AcousticScene::new(vertices, indices, MaterialTable::uniform(material)).unwrap();
        let backend = GeometricBackend::new(scene, GeometricConfig::new(48_000));
        let listener = Listener::new(Vec3::new(-4.0, 2.0, 0.0), Quat::IDENTITY, Vec3::ZERO);
        let emitter = Emitter::point(Vec3::new(4.0, 2.0, 0.0), Vec3::ZERO);
        let mut paths = buffer();
        let summary = backend.query(&listener, &emitter, &mut paths);
        assert_eq!(summary.path_count, 2);
        assert_eq!(paths[0].kind, PathKind::Direct);
        assert_eq!(paths[1].kind, PathKind::Reflection);
        // The reflection is quieter and later than the direct arrival.
        assert!(paths[1].gain < paths[0].gain);
        assert!(paths[1].delay_seconds > paths[0].delay_seconds);
    }

    #[test]
    fn barrier_blocks_direct_and_bends_over_the_edge() {
        // A tall wall between the two, forcing diffraction over its top edge.
        let vertices = vec![
            Vec3::new(0.0, -5.0, -5.0),
            Vec3::new(0.0, 1.0, -5.0),
            Vec3::new(0.0, 1.0, 5.0),
            Vec3::new(0.0, -5.0, 5.0),
        ];
        let indices = vec![[0, 1, 2], [0, 2, 3]];
        let material = prism_audio_spatial::propagation::AcousticMaterial::new(80.0, 0.0);
        let scene = AcousticScene::new(vertices, indices, MaterialTable::uniform(material)).unwrap();
        let cfg = GeometricConfig::new(48_000).without_reflections();
        let backend = GeometricBackend::new(scene, cfg);
        // Both below the top edge (y = 1), opposite sides: direct line crosses
        // the wall and is heavily attenuated; the edge route is clear.
        let listener = Listener::new(Vec3::new(-3.0, 0.0, 0.0), Quat::IDENTITY, Vec3::ZERO);
        let emitter = Emitter::point(Vec3::new(3.0, 0.0, 0.0), Vec3::ZERO);
        let mut paths = buffer();
        let summary = backend.query(&listener, &emitter, &mut paths);
        // The direct line is blocked, so occlusion is reported.
        assert!(summary.direct.direct_factor() > 0.5);
        // At least one diffracted arrival bends over the edge.
        assert!(paths[..summary.path_count]
            .iter()
            .any(|p| p.kind == PathKind::Diffraction));
    }
}
