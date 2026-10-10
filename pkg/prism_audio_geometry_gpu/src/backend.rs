//! Device-side geometric acoustics backend: the public entry point of the
//! crate.
//!
//! [`GpuGeometryBackend`] compiles the two compute kernels once, then
//! [`resolve`](GpuGeometryBackend::resolve)s a whole batch of listener/emitter
//! queries against one uploaded [`GpuScene`]. For each query it decodes the
//! device results into the same [`PropagationPath`] set and
//! [`OcclusionFactors`] the `CPU` [`GeometricBackend`](prism_audio_geometry::GeometricBackend)
//! produces, assembling them with the identical merge policy: the direct (or
//! transmitted) arrival first, then the surviving specular reflections sorted
//! loudest-first and capped, with the whole list truncated to
//! [`MAX_PROPAGATION_PATHS`].
//!
//! # Honest division of labour
//!
//! Edge diffraction stays on the `CPU` crate (it is a least-detour search that
//! does not parallelise per query), so this backend resolves `direct +
//! reflection` on the device. The golden comparison runs on clear-line scenes
//! whose `CPU` diffraction stage is empty, so the two agree arrival-for-arrival;
//! callers that need diffraction fills on shadowed lines keep using
//! [`prism_audio_geometry`].
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
//! Consumes the same [`AcousticScene`](prism_audio_geometry::AcousticScene) /
//! [`GeometricConfig`] and spatial
//! [`Listener`](prism_audio_spatial::geometry::Listener) /
//! [`Emitter`](prism_audio_spatial::geometry::Emitter) as
//! [`prism_audio_geometry`], and produces the same
//! [`PropagationPath`](prism_audio_spatial::propagation::PropagationPath) set,
//! so it is a drop-in device sibling of the `CPU` backend on clear-line scenes.

use alloc::vec::Vec;
use core::cmp::Ordering;

use bevy_math::Vec3;
use prism_audio_geometry::GeometricConfig;
use prism_audio_spatial::doppler::SPEED_OF_SOUND_MPS;
use prism_audio_spatial::geometry::{Emitter, Listener};
use prism_audio_spatial::occlusion::OcclusionFactors;
use prism_audio_spatial::propagation::{
    PathKind, PropagationPath, FULL_BAND_CUTOFF_HZ, MAX_PROPAGATION_PATHS,
};
use prism_audio_spatial::BandGains;

use crate::context::GpuContext;
use crate::direct::{cpu_direct, DirectKernel};
use crate::params::GeometryParams;
use crate::query::{GpuDirectResult, GpuQuery, GpuReflectionCandidate, DIRECT_KIND_DIRECT};
use crate::reflection::{cpu_reflection, ReflectionKernel};
use crate::scene_upload::GpuScene;

/// Largest absolute delay difference (seconds) under which two reflections are
/// treated as the same arrival, mirroring the `CPU` reflection stage's
/// de-duplication tolerance.
const DUPLICATE_DELAY_TOLERANCE: f32 = 1.0e-6;

/// Smallest direction cosine above which two reflections of equal delay are
/// treated as the same arrival, mirroring the `CPU` reflection stage.
const DUPLICATE_DIRECTION_COSINE: f32 = 0.9999;

/// The direct and reflection arrivals the backend resolved for one query.
///
/// Mirrors the output of a single
/// [`GeometricBackend::query`](prism_audio_geometry::GeometricBackend) call: the
/// occlusion factors the direct line implies, plus the ordered propagation
/// paths (direct/transmission first, then capped reflections).
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedQuery {
    /// Obstruction/occlusion factors the direct line implies.
    pub direct: OcclusionFactors,
    /// Ordered arrivals: the audible direct or transmitted path first (if any),
    /// then the surviving specular reflections, truncated to
    /// [`MAX_PROPAGATION_PATHS`].
    pub paths: Vec<PropagationPath>,
}

/// Compiled device backend that resolves direct and specular-reflection
/// arrivals for a batch of queries against one uploaded scene.
pub struct GpuGeometryBackend {
    /// The direct line-of-sight / transmission kernel.
    direct: DirectKernel,
    /// The first-order specular reflection kernel.
    reflection: ReflectionKernel,
}

impl GpuGeometryBackend {
    /// Compiles both compute pipelines on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuGeometryBackend {
        GpuGeometryBackend {
            direct: DirectKernel::new(ctx),
            reflection: ReflectionKernel::new(ctx),
        }
    }

    /// Resolves every `(listener, emitter)` query against `scene`.
    ///
    /// The returned vector is parallel to `queries`. Each entry carries the
    /// direct occlusion factors and the ordered propagation paths, assembled
    /// with the same policy as the `CPU`
    /// [`GeometricBackend`](prism_audio_geometry::GeometricBackend): the audible
    /// direct or transmitted arrival first, then de-duplicated specular
    /// reflections sorted loudest-first and capped at `config.max_reflections`,
    /// with the whole list truncated to [`MAX_PROPAGATION_PATHS`].
    #[must_use]
    pub fn resolve(
        &self,
        ctx: &GpuContext,
        scene: &GpuScene,
        config: &GeometricConfig,
        queries: &[(Listener, Emitter)],
    ) -> Vec<ResolvedQuery> {
        if queries.is_empty() {
            return Vec::new();
        }

        let triangle_count = scene.triangle_count();
        let params = GeometryParams {
            triangle_count: triangle_count as u32,
            query_count: queries.len() as u32,
            min_gain: config.min_gain,
            surface_epsilon: config.surface_epsilon_m,
            speed_of_sound: SPEED_OF_SOUND_MPS,
            transmission_enabled: u32::from(config.transmission_enabled),
            _pad0: 0,
            _pad1: 0,
        };

        let gpu_queries: Vec<GpuQuery> = queries
            .iter()
            .map(|(listener, emitter)| GpuQuery::new(listener, emitter))
            .collect();

        let directs = self.direct.dispatch(ctx, scene, &params, &gpu_queries);
        let reflections_disabled = reflections_disabled(config, scene);
        let candidates = if reflections_disabled {
            Vec::new()
        } else {
            self.reflection.dispatch(ctx, scene, &params, &gpu_queries)
        };

        assemble(
            &directs,
            &candidates,
            triangle_count,
            config,
            reflections_disabled,
        )
    }

    /// Resolves every query entirely on the host, using the kernels' `CPU`
    /// twins ([`cpu_direct`]/[`cpu_reflection`]).
    ///
    /// This produces results bit-identical to the device twins without a `GPU`,
    /// so callers on a headless adapter (or a platform where
    /// [`GpuContext::try_headless`] returns `None`) still get the same arrivals.
    /// It shares the exact assembly policy of [`resolve`](Self::resolve).
    #[must_use]
    pub fn resolve_cpu(
        &self,
        scene: &GpuScene,
        config: &GeometricConfig,
        queries: &[(Listener, Emitter)],
    ) -> Vec<ResolvedQuery> {
        if queries.is_empty() {
            return Vec::new();
        }

        let triangle_count = scene.triangle_count();
        let params = GeometryParams {
            triangle_count: triangle_count as u32,
            query_count: queries.len() as u32,
            min_gain: config.min_gain,
            surface_epsilon: config.surface_epsilon_m,
            speed_of_sound: SPEED_OF_SOUND_MPS,
            transmission_enabled: u32::from(config.transmission_enabled),
            _pad0: 0,
            _pad1: 0,
        };

        let triangles = scene.triangles();
        let reflections_disabled = reflections_disabled(config, scene);

        let mut directs = Vec::with_capacity(queries.len());
        let mut candidates = Vec::new();
        if !reflections_disabled {
            candidates.reserve(queries.len() * triangle_count);
        }
        for (listener, emitter) in queries {
            let query = GpuQuery::new(listener, emitter);
            directs.push(cpu_direct(triangles, &query, &params));
            if !reflections_disabled {
                for index in 0..triangle_count {
                    candidates.push(cpu_reflection(triangles, &query, index, &params));
                }
            }
        }

        assemble(
            &directs,
            &candidates,
            triangle_count,
            config,
            reflections_disabled,
        )
    }
}

/// Whether the reflection stage should be skipped entirely for `scene` under
/// `config`, matching the `CPU` backend's host gate.
#[must_use]
fn reflections_disabled(config: &GeometricConfig, scene: &GpuScene) -> bool {
    !config.reflections_enabled || config.max_reflections == 0 || scene.is_empty()
}

/// Decodes the direct results and (optionally) the flat reflection candidates
/// into one [`ResolvedQuery`] per query, applying the shared merge policy.
///
/// `candidates` is laid out as `query_count * triangle_count` records where slot
/// `q * triangle_count + t` is query `q`'s candidate off triangle `t`; it is
/// empty when reflections are disabled.
#[must_use]
fn assemble(
    directs: &[GpuDirectResult],
    candidates: &[GpuReflectionCandidate],
    triangle_count: usize,
    config: &GeometricConfig,
    reflections_disabled: bool,
) -> Vec<ResolvedQuery> {
    let mut resolved = Vec::with_capacity(directs.len());
    for (index, direct) in directs.iter().enumerate() {
        let factors = OcclusionFactors::new(direct.obstruction, direct.occlusion);

        let mut reflections: Vec<PropagationPath> = Vec::new();
        if !reflections_disabled {
            let start = index * triangle_count;
            let slice = &candidates[start..start + triangle_count];
            for candidate in slice {
                if candidate.valid == 0 {
                    continue;
                }
                let path = PropagationPath {
                    kind: PathKind::Reflection,
                    delay_seconds: candidate.delay_seconds,
                    gain: candidate.gain,
                    cutoff_hz: FULL_BAND_CUTOFF_HZ,
                    bands: BandGains::new([
                        candidate.bands[0],
                        candidate.bands[1],
                        candidate.bands[2],
                    ]),
                    direction: Vec3::new(
                        candidate.direction[0],
                        candidate.direction[1],
                        candidate.direction[2],
                    ),
                };
                if is_duplicate(&reflections, &path) {
                    continue;
                }
                reflections.push(path);
            }
            reflections.sort_by(|l, r| r.gain.partial_cmp(&l.gain).unwrap_or(Ordering::Equal));
            reflections.truncate(config.max_reflections);
        }

        let mut paths: Vec<PropagationPath> = Vec::new();
        if direct.audible != 0 {
            let kind = if direct.kind == DIRECT_KIND_DIRECT {
                PathKind::Direct
            } else {
                PathKind::Transmission
            };
            paths.push(PropagationPath {
                kind,
                delay_seconds: direct.delay_seconds,
                gain: direct.gain,
                cutoff_hz: direct.cutoff_hz,
                bands: BandGains::new([direct.bands[0], direct.bands[1], direct.bands[2]]),
                direction: Vec3::new(
                    direct.direction[0],
                    direct.direction[1],
                    direct.direction[2],
                ),
            });
        }
        paths.extend(reflections);
        paths.truncate(MAX_PROPAGATION_PATHS);

        resolved.push(ResolvedQuery {
            direct: factors,
            paths,
        });
    }

    resolved
}

/// Returns whether `candidate` duplicates an already-retained reflection,
/// mirroring the `CPU` reflection stage: equal within
/// [`DUPLICATE_DELAY_TOLERANCE`] seconds of delay and sharing a direction whose
/// cosine exceeds [`DUPLICATE_DIRECTION_COSINE`].
#[must_use]
fn is_duplicate(existing: &[PropagationPath], candidate: &PropagationPath) -> bool {
    existing.iter().any(|path| {
        (path.delay_seconds - candidate.delay_seconds).abs() < DUPLICATE_DELAY_TOLERANCE
            && path.direction.dot(candidate.direction) > DUPLICATE_DIRECTION_COSINE
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use bevy_math::{Quat, Vec3};
    use prism_audio_geometry::direct_path::resolve_direct;
    use prism_audio_geometry::reflection_path::resolve_reflections;
    use prism_audio_geometry::{AcousticScene, GeometricBackend, MaterialTable};
    use prism_audio_spatial::propagation::{AcousticMaterial, PropagationBackend};
    use prism_audio_spatial::BandedAcousticMaterial;

    use crate::direct::cpu_direct;
    use crate::query::DIRECT_KIND_TRANSMISSION;
    use crate::reflection::cpu_reflection;
    use crate::scene_upload::GpuTriangle;

    /// Absolute tolerance for comparing two independent `f32` arithmetic paths.
    const TOL: f32 = 1.0e-4;
    const TWIN_TOL: f32 = 1.0e-5;

    /// Packs an [`AcousticScene`] into the device triangle layout exactly as
    /// [`GpuScene::upload`] does, but without a device, for the host twins.
    fn pack(scene: &AcousticScene) -> Vec<GpuTriangle> {
        let mut triangles = Vec::with_capacity(scene.triangle_count());
        for index in 0..scene.triangle_count() {
            let [a, b, c] = scene
                .triangle(index)
                .unwrap_or([Vec3::ZERO, Vec3::ZERO, Vec3::ZERO]);
            let normal = scene.triangle_normal(index).unwrap_or(Vec3::ZERO);
            let material = scene.material(index);
            let transmission = material.transmission().bands();
            let reflection = material.reflection().bands();
            triangles.push(GpuTriangle {
                a: [a.x, a.y, a.z, 0.0],
                b: [b.x, b.y, b.z, 0.0],
                c: [c.x, c.y, c.z, 0.0],
                normal: [normal.x, normal.y, normal.z, 0.0],
                transmission: [transmission[0], transmission[1], transmission[2], 0.0],
                reflection: [reflection[0], reflection[1], reflection[2], 0.0],
                scattering: material.scattering(),
                _pad: [0.0, 0.0, 0.0],
            });
        }
        triangles
    }

    /// Builds the [`GeometryParams`] a dispatch would upload for `scene` and a
    /// one-query batch under `config`.
    fn params_for(scene: &AcousticScene, config: &GeometricConfig) -> GeometryParams {
        GeometryParams {
            triangle_count: scene.triangle_count() as u32,
            query_count: 1,
            min_gain: config.min_gain,
            surface_epsilon: config.surface_epsilon_m,
            speed_of_sound: SPEED_OF_SOUND_MPS,
            transmission_enabled: u32::from(config.transmission_enabled),
            _pad0: 0,
            _pad1: 0,
        }
    }

    /// A floor spanning `[-10, 10]` on the `x`/`z` plane at `y = 0`, highly
    /// reflective and acoustically opaque to transmission.
    fn floor_scene() -> AcousticScene {
        let vertices = Vec::from([
            Vec3::new(-10.0, 0.0, -10.0),
            Vec3::new(10.0, 0.0, -10.0),
            Vec3::new(10.0, 0.0, 10.0),
            Vec3::new(-10.0, 0.0, 10.0),
        ]);
        let indices = Vec::from([[0, 1, 2], [0, 2, 3]]);
        let table = MaterialTable::uniform_scalar(AcousticMaterial::new(0.0, 0.9));
        AcousticScene::new(vertices, indices, table).expect("floor scene builds")
    }

    /// A single opaque-ish wall on the `x = 0` plane with a 6.02 dB
    /// transmission loss (half-amplitude), used for the transmission twin.
    fn wall_scene() -> AcousticScene {
        let vertices = Vec::from([
            Vec3::new(0.0, -5.0, -5.0),
            Vec3::new(0.0, 5.0, -5.0),
            Vec3::new(0.0, 5.0, 5.0),
            Vec3::new(0.0, -5.0, 5.0),
        ]);
        let indices = Vec::from([[0, 1, 2], [0, 2, 3]]);
        let table = MaterialTable::uniform_scalar(AcousticMaterial::new(6.020_6, 0.0));
        AcousticScene::new(vertices, indices, table).expect("wall scene builds")
    }

    fn listener_at(position: Vec3) -> Listener {
        Listener::new(position, Quat::IDENTITY, Vec3::ZERO)
    }

    #[test]
    fn direct_twin_matches_cpu_on_clear_line() {
        let scene = floor_scene();
        let config = GeometricConfig::new(48_000);
        let listener = listener_at(Vec3::new(-4.0, 2.0, 0.0));
        let emitter = Emitter::point(Vec3::new(4.0, 2.0, 0.0), Vec3::ZERO);

        let triangles = pack(&scene);
        let params = params_for(&scene, &config);
        let query = GpuQuery::new(&listener, &emitter);
        let twin = cpu_direct(&triangles, &query, &params);

        let reference = resolve_direct(&scene, &listener, &emitter, &config);

        assert_eq!(twin.audible != 0, reference.audible);
        assert_eq!(twin.kind, DIRECT_KIND_DIRECT);
        assert_eq!(reference.path.kind, PathKind::Direct);
        assert!((twin.base_distance - reference.base_distance).abs() < TOL);
        assert!((twin.gain - reference.path.gain).abs() < TOL);
        assert!((twin.delay_seconds - reference.path.delay_seconds).abs() < TOL);
        assert!((twin.obstruction - reference.occlusion.obstruction).abs() < TOL);
        assert!((twin.occlusion - reference.occlusion.occlusion).abs() < TOL);
        let twin_dir = Vec3::new(twin.direction[0], twin.direction[1], twin.direction[2]);
        assert!(twin_dir.dot(reference.path.direction) > 1.0 - TOL);
    }

    #[test]
    fn direct_twin_matches_cpu_through_wall() {
        let scene = wall_scene();
        let config = GeometricConfig::new(48_000);
        let listener = listener_at(Vec3::new(-3.0, 0.0, 0.0));
        let emitter = Emitter::point(Vec3::new(3.0, 0.0, 0.0), Vec3::ZERO);

        let triangles = pack(&scene);
        let params = params_for(&scene, &config);
        let query = GpuQuery::new(&listener, &emitter);
        let twin = cpu_direct(&triangles, &query, &params);

        let reference = resolve_direct(&scene, &listener, &emitter, &config);

        assert_eq!(twin.audible != 0, reference.audible);
        assert_eq!(twin.kind, DIRECT_KIND_TRANSMISSION);
        assert_eq!(reference.path.kind, PathKind::Transmission);
        assert!((twin.gain - reference.path.gain).abs() < TOL);
        assert!(twin.gain > 0.4 && twin.gain < 0.6);
        assert!((twin.obstruction - reference.occlusion.obstruction).abs() < TOL);
        assert!((twin.occlusion - reference.occlusion.occlusion).abs() < TOL);
    }

    #[test]
    fn reflection_twin_matches_cpu() {
        let scene = floor_scene();
        let config = GeometricConfig::new(48_000);
        let listener = listener_at(Vec3::new(-4.0, 2.0, 0.0));
        let emitter = Emitter::point(Vec3::new(4.0, 2.0, 0.0), Vec3::ZERO);

        let triangles = pack(&scene);
        let params = params_for(&scene, &config);
        let query = GpuQuery::new(&listener, &emitter);

        let base_distance = resolve_direct(&scene, &listener, &emitter, &config).base_distance;
        let reference = resolve_reflections(&scene, &listener, &emitter, &config, base_distance);

        let mut twins: Vec<PropagationPath> = Vec::new();
        for index in 0..triangles.len() {
            let candidate = cpu_reflection(&triangles, &query, index, &params);
            if candidate.valid == 0 {
                continue;
            }
            let path = PropagationPath {
                kind: PathKind::Reflection,
                delay_seconds: candidate.delay_seconds,
                gain: candidate.gain,
                cutoff_hz: FULL_BAND_CUTOFF_HZ,
                bands: BandGains::new([candidate.bands[0], candidate.bands[1], candidate.bands[2]]),
                direction: Vec3::new(
                    candidate.direction[0],
                    candidate.direction[1],
                    candidate.direction[2],
                ),
            };
            if is_duplicate(&twins, &path) {
                continue;
            }
            twins.push(path);
        }
        twins.sort_by(|l, r| r.gain.partial_cmp(&l.gain).unwrap_or(Ordering::Equal));
        twins.truncate(config.max_reflections);

        assert_eq!(twins.len(), reference.len());
        assert_eq!(reference.len(), 1);
        let twin = twins[0];
        let want = reference[0];
        assert_eq!(twin.kind, PathKind::Reflection);
        assert!((twin.gain - want.gain).abs() < TOL);
        assert!((twin.delay_seconds - want.delay_seconds).abs() < TOL);
        assert!(twin.direction.dot(want.direction) > 1.0 - TOL);
    }

    /// A wall whose per-band transmission colours the arrival: the opaque wall
    /// of [`wall_scene`] carries a scalar loss, but a real window darkens high
    /// frequencies more than low. This builds the same geometry with a banded
    /// material so the twins can prove the colour survives the march.
    fn wall_scene_banded(material: BandedAcousticMaterial) -> AcousticScene {
        let vertices = Vec::from([
            Vec3::new(0.0, -5.0, -5.0),
            Vec3::new(0.0, 5.0, -5.0),
            Vec3::new(0.0, 5.0, 5.0),
            Vec3::new(0.0, -5.0, 5.0),
        ]);
        let indices = Vec::from([[0, 1, 2], [0, 2, 3]]);
        AcousticScene::new(vertices, indices, MaterialTable::uniform(material))
            .expect("banded wall scene builds")
    }

    /// The floor of [`floor_scene`] rebuilt with a banded reflection spectrum and
    /// surface roughness, so the specular bounce keeps a frequency colour.
    fn floor_scene_banded(material: BandedAcousticMaterial) -> AcousticScene {
        let vertices = Vec::from([
            Vec3::new(-10.0, 0.0, -10.0),
            Vec3::new(10.0, 0.0, -10.0),
            Vec3::new(10.0, 0.0, 10.0),
            Vec3::new(-10.0, 0.0, 10.0),
        ]);
        let indices = Vec::from([[0, 1, 2], [0, 2, 3]]);
        AcousticScene::new(vertices, indices, MaterialTable::uniform(material))
            .expect("banded floor scene builds")
    }

    #[test]
    fn direct_twin_carries_banded_transmission_colour() {
        let material =
            BandedAcousticMaterial::new(BandGains::SILENT, BandGains::new([0.8, 0.4, 0.1]), 0.0);
        let scene = wall_scene_banded(material);
        let config = GeometricConfig::new(48_000);
        let listener = listener_at(Vec3::new(-3.0, 0.0, 0.0));
        let emitter = Emitter::point(Vec3::new(3.0, 0.0, 0.0), Vec3::ZERO);

        let triangles = pack(&scene);
        let params = params_for(&scene, &config);
        let query = GpuQuery::new(&listener, &emitter);
        let twin = cpu_direct(&triangles, &query, &params);

        let reference = resolve_direct(&scene, &listener, &emitter, &config);
        let want = reference.path.bands.bands();

        assert_eq!(twin.kind, DIRECT_KIND_TRANSMISSION);
        assert_eq!(reference.path.kind, PathKind::Transmission);
        assert!(twin.audible != 0 && reference.audible);
        // The peak-normalised gain tracks the brightest band (the low band at
        // 0.8), matching the CPU golden rather than a broadband average.
        assert!((twin.gain - reference.path.gain).abs() < TOL);
        assert!((twin.gain - 0.8).abs() < TOL);
        // The arrival darkens towards high frequency, and the twin reproduces
        // the CPU colour band-by-band instead of collapsing to one number.
        assert!(twin.bands[2] < twin.bands[0]);
        for (&want_band, &twin_band) in want.iter().zip(twin.bands.iter()) {
            assert!((twin_band - want_band).abs() < TOL);
        }
    }

    #[test]
    fn reflection_twin_carries_banded_colour() {
        let material = BandedAcousticMaterial::new(
            BandGains::new([0.9, 0.6, 0.3]),
            BandGains::new([0.8, 0.5, 0.2]),
            0.25,
        );
        let scene = floor_scene_banded(material);
        let config = GeometricConfig::new(48_000);
        let listener = listener_at(Vec3::new(-4.0, 2.0, 0.0));
        let emitter = Emitter::point(Vec3::new(4.0, 2.0, 0.0), Vec3::ZERO);

        let triangles = pack(&scene);
        let params = params_for(&scene, &config);
        let query = GpuQuery::new(&listener, &emitter);

        let base_distance = resolve_direct(&scene, &listener, &emitter, &config).base_distance;
        let reference = resolve_reflections(&scene, &listener, &emitter, &config, base_distance);

        let mut twins: Vec<PropagationPath> = Vec::new();
        for index in 0..triangles.len() {
            let candidate = cpu_reflection(&triangles, &query, index, &params);
            if candidate.valid == 0 {
                continue;
            }
            let path = PropagationPath {
                kind: PathKind::Reflection,
                delay_seconds: candidate.delay_seconds,
                gain: candidate.gain,
                cutoff_hz: FULL_BAND_CUTOFF_HZ,
                bands: BandGains::new([candidate.bands[0], candidate.bands[1], candidate.bands[2]]),
                direction: Vec3::new(
                    candidate.direction[0],
                    candidate.direction[1],
                    candidate.direction[2],
                ),
            };
            if is_duplicate(&twins, &path) {
                continue;
            }
            twins.push(path);
        }
        twins.sort_by(|l, r| r.gain.partial_cmp(&l.gain).unwrap_or(Ordering::Equal));
        twins.truncate(config.max_reflections);

        assert_eq!(reference.len(), 1);
        assert_eq!(twins.len(), 1);
        let twin = twins[0];
        let want = reference[0];
        let twin_bands = twin.bands.bands();
        let want_bands = want.bands.bands();
        assert!((twin.gain - want.gain).abs() < TOL);
        for band in 0..3 {
            assert!((twin_bands[band] - want_bands[band]).abs() < TOL);
        }
        // The specular reflection keeps a clear low-to-high colour gradient
        // (bright low band, dim high band), not a flat spectrum.
        assert!((twin_bands[0] - twin_bands[2]).abs() > 0.1);
    }

    #[test]
    fn gpu_direct_matches_twin() {
        let Some(ctx) = GpuContext::try_headless() else {
            return;
        };
        let scene = floor_scene();
        let config = GeometricConfig::new(48_000);
        let listener = listener_at(Vec3::new(-4.0, 2.0, 0.0));
        let emitter = Emitter::point(Vec3::new(4.0, 2.0, 0.0), Vec3::ZERO);

        let gpu_scene = GpuScene::upload(&ctx, &scene);
        let params = params_for(&scene, &config);
        let query = GpuQuery::new(&listener, &emitter);
        let kernel = DirectKernel::new(&ctx);
        let device = kernel.dispatch(&ctx, &gpu_scene, &params, &[query]);
        let twin = cpu_direct(gpu_scene.triangles(), &query, &params);

        assert_eq!(device.len(), 1);
        assert_eq!(device[0], twin);
    }

    #[test]
    fn gpu_reflection_matches_twin() {
        let Some(ctx) = GpuContext::try_headless() else {
            return;
        };
        let scene = floor_scene();
        let config = GeometricConfig::new(48_000);
        let listener = listener_at(Vec3::new(-4.0, 2.0, 0.0));
        let emitter = Emitter::point(Vec3::new(4.0, 2.0, 0.0), Vec3::ZERO);

        let gpu_scene = GpuScene::upload(&ctx, &scene);
        let params = params_for(&scene, &config);
        let query = GpuQuery::new(&listener, &emitter);
        let kernel = ReflectionKernel::new(&ctx);
        let device = kernel.dispatch(&ctx, &gpu_scene, &params, &[query]);

        assert_eq!(device.len(), gpu_scene.triangle_count());
        for (index, candidate) in device.iter().enumerate() {
            let twin = cpu_reflection(gpu_scene.triangles(), &query, index, &params);
            assert_reflection_matches(candidate, &twin);
        }
    }

    /// Compares a device reflection candidate against its CPU twin.
    ///
    /// Validity and padding are integer or canonical-zero fields and must match
    /// exactly. The floating-point legs are compared within [`TWIN_TOL`] because
    /// the shader fuses `dot` into a hardware multiply-add while the scalar twin
    /// evaluates the same expression without fusion; that difference is bounded
    /// by one unit in the last place and cannot be removed across devices.
    fn assert_reflection_matches(device: &GpuReflectionCandidate, twin: &GpuReflectionCandidate) {
        assert_eq!(device.valid, twin.valid);
        if device.valid == 0 {
            return;
        }
        for axis in 0..3 {
            assert!((device.direction[axis] - twin.direction[axis]).abs() < TWIN_TOL);
        }
        assert!((device.delay_seconds - twin.delay_seconds).abs() < TWIN_TOL);
        assert!((device.gain - twin.gain).abs() < TWIN_TOL);
    }

    #[test]
    fn backend_empty_scene_is_direct_only() {
        let Some(ctx) = GpuContext::try_headless() else {
            return;
        };
        let scene = AcousticScene::new(Vec::new(), Vec::new(), MaterialTable::default())
            .expect("empty scene builds");
        let config = GeometricConfig::new(48_000);
        let gpu_scene = GpuScene::upload(&ctx, &scene);
        let backend = GpuGeometryBackend::new(&ctx);

        let listener = listener_at(Vec3::new(-4.0, 2.0, 0.0));
        let emitter = Emitter::point(Vec3::new(4.0, 2.0, 0.0), Vec3::ZERO);
        let resolved = backend.resolve(&ctx, &gpu_scene, &config, &[(listener, emitter)]);

        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].paths.len(), 1);
        assert_eq!(resolved[0].paths[0].kind, PathKind::Direct);
        assert!((resolved[0].paths[0].gain - 1.0).abs() < TOL);
        assert!(resolved[0].direct.direct_factor() < TOL);
    }

    #[test]
    fn backend_floor_reflection_matches_cpu() {
        let Some(ctx) = GpuContext::try_headless() else {
            return;
        };
        let scene = floor_scene();
        let config = GeometricConfig::new(48_000);
        let gpu_scene = GpuScene::upload(&ctx, &scene);
        let backend = GpuGeometryBackend::new(&ctx);

        let listener = listener_at(Vec3::new(-4.0, 2.0, 0.0));
        let emitter = Emitter::point(Vec3::new(4.0, 2.0, 0.0), Vec3::ZERO);
        let resolved = backend.resolve(&ctx, &gpu_scene, &config, &[(listener, emitter)]);

        assert_eq!(resolved.len(), 1);
        let query = &resolved[0];
        assert_eq!(query.paths.len(), 2);
        assert_eq!(query.paths[0].kind, PathKind::Direct);
        assert_eq!(query.paths[1].kind, PathKind::Reflection);
        assert!(query.paths[1].gain < query.paths[0].gain);
        assert!(query.paths[1].delay_seconds > query.paths[0].delay_seconds);

        let cpu = GeometricBackend::new(scene, config);
        let mut reference = [PropagationPath::SILENT; MAX_PROPAGATION_PATHS];
        let summary = cpu.query(&listener, &emitter, &mut reference);
        assert_eq!(summary.path_count, 2);
        for (got, want) in query.paths.iter().zip(reference.iter()) {
            assert_eq!(got.kind, want.kind);
            assert!((got.gain - want.gain).abs() < TOL);
            assert!((got.delay_seconds - want.delay_seconds).abs() < TOL);
            assert!(got.direction.dot(want.direction) > 1.0 - TOL);
        }
    }
}
