//! A [`PropagationBackend`] driven by the device kernels.
//!
//! [`GpuPropagationBackend`] is the drop-in, single-query sibling of the `CPU`
//! [`GeometricBackend`](prism_audio_geometry::GeometricBackend): it satisfies
//! the spatial crate's
//! [`PropagationBackend`](prism_audio_spatial::propagation::PropagationBackend)
//! trait so a spatial voice can source its arrivals from the `GPU`. Where
//! [`GpuGeometryBackend`] exposes the batched device API, this type adapts it to
//! the per-source control-rate trait the spatial layer consumes.
//!
//! # Division of labour (identical to the `CPU` backend)
//!
//! The direct line-of-sight / transmission arrival and the first-order specular
//! reflections are resolved on the device through [`GpuGeometryBackend::resolve`]
//! for the single query. Shadow-edge diffraction stays on the `CPU`: it is a
//! least-detour search that does not parallelise per query, so, exactly like
//! [`GeometricBackend`](prism_audio_geometry::GeometricBackend), it is resolved
//! with [`resolve_diffraction`] and only when the direct line is measurably
//! blocked. The device reflections (already de-duplicated, sorted loudest-first
//! and capped at `config.max_reflections`) are merged with the diffraction fills
//! and re-sorted loudest-first, with the audible direct arrival kept first --
//! the same merge policy [`GeometricBackend`](prism_audio_geometry::GeometricBackend)
//! applies, so the two track arrival-for-arrival within the device's
//! floating-point tolerance.
//!
//! # Provenance
//!
//! Original work; standard `wgpu` compute mirroring the classic image-source /
//! ray-march acoustics already implemented on the `CPU` in
//! [`prism_audio_geometry`]; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//!
//! Implements [`prism_audio_spatial::propagation::PropagationBackend`] by
//! composing [`GpuGeometryBackend`] (device direct + reflection) with
//! [`prism_audio_geometry::diffraction_path::resolve_diffraction`] (`CPU`
//! diffraction) over a device-resident [`GpuScene`] and a
//! [`GeometricConfig`](prism_audio_geometry::GeometricConfig).

use alloc::vec::Vec;
use core::cmp::Ordering;

use prism_audio_geometry::diffraction_path::resolve_diffraction;
use prism_audio_geometry::scene::AcousticScene;
use prism_audio_geometry::GeometricConfig;
use prism_audio_spatial::geometry::{Emitter, Listener};
use prism_audio_spatial::occlusion::OcclusionFactors;
use prism_audio_spatial::propagation::{
    PathKind, PropagationBackend, PropagationPath, PropagationSummary,
};

use crate::backend::GpuGeometryBackend;
use crate::context::GpuContext;
use crate::scene_upload::GpuScene;

/// A device-driven [`PropagationBackend`] over one uploaded scene.
///
/// It owns the compute context, the device-resident scene, the compiled
/// kernels, and a host copy of the scene (needed for the `CPU` diffraction
/// stage). Build it once per scene/configuration and query it per source at
/// control rate; it is not real-time safe (it dispatches the device and
/// allocates scratch) and is meant to run off the audio thread.
///
/// Callers that have many sources to resolve against the same scene at once
/// should prefer the batched [`GpuGeometryBackend::resolve`] directly; this
/// adapter exists for the single-source trait the spatial layer consumes.
pub struct GpuPropagationBackend {
    /// The compute context the kernels dispatch on.
    context: GpuContext,
    /// The compiled direct and reflection kernels.
    backend: GpuGeometryBackend,
    /// The device-resident triangle mesh the kernels trace.
    device_scene: GpuScene,
    /// The host copy of the scene, traced by the `CPU` diffraction stage.
    scene: AcousticScene,
    /// The control-rate budget and feature switches.
    config: GeometricConfig,
}

impl GpuPropagationBackend {
    /// Builds a backend that traces `scene` on `context` under `config`.
    ///
    /// The scene is uploaded to the device once and the kernels are compiled
    /// once; the host copy of `scene` is retained so the `CPU` diffraction
    /// stage can trace the same geometry.
    #[must_use]
    pub fn new(context: GpuContext, scene: AcousticScene, config: GeometricConfig) -> Self {
        let device_scene = GpuScene::upload(&context, &scene);
        let backend = GpuGeometryBackend::new(&context);
        Self {
            context,
            backend,
            device_scene,
            scene,
            config,
        }
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
}

impl PropagationBackend for GpuPropagationBackend {
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

        // Resolve direct + specular reflections on the device for this query.
        let mut resolved = self.backend.resolve(
            &self.context,
            &self.device_scene,
            &self.config,
            &[(*listener, *emitter)],
        );
        let resolved = resolved
            .pop()
            .expect("one query resolves to exactly one result");
        let factors = resolved.direct;

        // The device backend returns the audible direct/transmission arrival
        // first (when any), then the capped specular reflections. Split them so
        // the reflections can be merged with the diffraction fills, exactly as
        // the CPU backend does.
        let mut direct_path: Option<PropagationPath> = None;
        let mut secondary: Vec<PropagationPath> = Vec::with_capacity(resolved.paths.len());
        for path in resolved.paths {
            if direct_path.is_none()
                && matches!(path.kind, PathKind::Direct | PathKind::Transmission)
            {
                direct_path = Some(path);
            } else {
                secondary.push(path);
            }
        }

        // Shadow-edge diffraction only matters once the direct line is at least
        // partly blocked. The base distance is the straight listener-to-emitter
        // length, matching the CPU direct stage so the diffraction spreading is
        // identical.
        if factors.direct_factor() > 0.0 {
            let base_distance = listener.localize(emitter).distance;
            secondary.extend(resolve_diffraction(
                &self.scene,
                listener,
                emitter,
                &self.config,
                base_distance,
            ));
        }
        secondary.sort_by(|lhs, rhs| rhs.gain.partial_cmp(&lhs.gain).unwrap_or(Ordering::Equal));

        let mut written = 0usize;
        if let Some(direct) = direct_path {
            paths[0] = direct;
            written = 1;
        }
        for path in secondary {
            if written >= paths.len() {
                break;
            }
            paths[written] = path;
            written += 1;
        }

        PropagationSummary {
            direct: factors,
            path_count: written,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::GpuPropagationBackend;

    use alloc::vec;
    use bevy_math::{Quat, Vec3};
    use prism_audio_geometry::material_map::MaterialTable;
    use prism_audio_geometry::scene::AcousticScene;
    use prism_audio_geometry::{GeometricBackend, GeometricConfig};
    use prism_audio_spatial::geometry::{Emitter, Listener};
    use prism_audio_spatial::propagation::{
        AcousticMaterial, PathKind, PropagationBackend, PropagationPath, MAX_PROPAGATION_PATHS,
    };

    use crate::context::GpuContext;

    const TOL: f32 = 1.0e-4;

    fn buffer() -> [PropagationPath; MAX_PROPAGATION_PATHS] {
        [PropagationPath::SILENT; MAX_PROPAGATION_PATHS]
    }

    fn floor_scene() -> AcousticScene {
        let vertices = vec![
            Vec3::new(-10.0, 0.0, -10.0),
            Vec3::new(10.0, 0.0, -10.0),
            Vec3::new(10.0, 0.0, 10.0),
            Vec3::new(-10.0, 0.0, 10.0),
        ];
        let indices = vec![[0, 1, 2], [0, 2, 3]];
        let material = AcousticMaterial::new(0.0, 0.9);
        AcousticScene::new(vertices, indices, MaterialTable::uniform(material))
            .expect("floor scene builds")
    }

    fn barrier_scene() -> AcousticScene {
        let vertices = vec![
            Vec3::new(0.0, -5.0, -5.0),
            Vec3::new(0.0, 1.0, -5.0),
            Vec3::new(0.0, 1.0, 5.0),
            Vec3::new(0.0, -5.0, 5.0),
        ];
        let indices = vec![[0, 1, 2], [0, 2, 3]];
        let material = AcousticMaterial::new(80.0, 0.0);
        AcousticScene::new(vertices, indices, MaterialTable::uniform(material))
            .expect("barrier scene builds")
    }

    fn assert_tracks_cpu(
        scene: AcousticScene,
        config: GeometricConfig,
        listener: &Listener,
        emitter: &Emitter,
    ) {
        let Some(context) = GpuContext::try_headless() else {
            return;
        };
        let cpu = GeometricBackend::new(scene.clone(), config);
        let gpu = GpuPropagationBackend::new(context, scene, config);

        let mut cpu_paths = buffer();
        let mut gpu_paths = buffer();
        let cpu_summary = cpu.query(listener, emitter, &mut cpu_paths);
        let gpu_summary = gpu.query(listener, emitter, &mut gpu_paths);

        assert_eq!(gpu_summary.path_count, cpu_summary.path_count);
        assert!((gpu_summary.direct.obstruction - cpu_summary.direct.obstruction).abs() < TOL);
        assert!((gpu_summary.direct.occlusion - cpu_summary.direct.occlusion).abs() < TOL);
        for (got, want) in gpu_paths[..gpu_summary.path_count]
            .iter()
            .zip(cpu_paths[..cpu_summary.path_count].iter())
        {
            assert_eq!(got.kind, want.kind);
            assert!((got.gain - want.gain).abs() < TOL);
            assert!((got.delay_seconds - want.delay_seconds).abs() < TOL);
            assert!(got.direction.dot(want.direction) > 1.0 - TOL);
        }
    }

    #[test]
    fn empty_buffer_reports_nothing() {
        let Some(context) = GpuContext::try_headless() else {
            return;
        };
        let backend =
            GpuPropagationBackend::new(context, floor_scene(), GeometricConfig::new(48_000));
        let listener = Listener::new(Vec3::new(-4.0, 2.0, 0.0), Quat::IDENTITY, Vec3::ZERO);
        let emitter = Emitter::point(Vec3::new(4.0, 2.0, 0.0), Vec3::ZERO);
        let summary = backend.query(&listener, &emitter, &mut []);
        assert_eq!(summary.path_count, 0);
    }

    #[test]
    fn direct_and_reflection_track_cpu() {
        let listener = Listener::new(Vec3::new(-4.0, 2.0, 0.0), Quat::IDENTITY, Vec3::ZERO);
        let emitter = Emitter::point(Vec3::new(4.0, 2.0, 0.0), Vec3::ZERO);
        assert_tracks_cpu(floor_scene(), GeometricConfig::new(48_000), &listener, &emitter);
    }

    #[test]
    fn diffraction_over_barrier_tracks_cpu() {
        let listener = Listener::new(Vec3::new(-3.0, 0.0, 0.0), Quat::IDENTITY, Vec3::ZERO);
        let emitter = Emitter::point(Vec3::new(3.0, 0.0, 0.0), Vec3::ZERO);
        let config = GeometricConfig::new(48_000).without_reflections();
        assert_tracks_cpu(barrier_scene(), config, &listener, &emitter);
    }

    #[test]
    fn barrier_backend_bends_over_the_edge() {
        let Some(context) = GpuContext::try_headless() else {
            return;
        };
        let config = GeometricConfig::new(48_000).without_reflections();
        let backend = GpuPropagationBackend::new(context, barrier_scene(), config);
        let listener = Listener::new(Vec3::new(-3.0, 0.0, 0.0), Quat::IDENTITY, Vec3::ZERO);
        let emitter = Emitter::point(Vec3::new(3.0, 0.0, 0.0), Vec3::ZERO);
        let mut paths = buffer();
        let summary = backend.query(&listener, &emitter, &mut paths);
        assert!(summary.direct.direct_factor() > 0.5);
        assert!(paths[..summary.path_count]
            .iter()
            .any(|path| path.kind == PathKind::Diffraction));
    }
}
