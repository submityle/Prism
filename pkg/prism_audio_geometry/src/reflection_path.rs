//! First-order specular reflections resolved with the image-source method.
//!
//! A specular reflection is the mirror image of the emitter in a reflecting
//! face: a wave leaves the source, bounces once off a surface, and reaches the
//! listener along a longer, delayed, scaled route. The classic image-source
//! construction mirrors the emitter across each candidate face, draws the line
//! from the listener to that image, and keeps the crossing point as the
//! reflection point when it lands on the face and both sub-segments are clear.
//!
//! # Gain convention
//!
//! Each reflection carries the surface's linear reflection coefficient scaled
//! by the extra spherical spreading of its longer route relative to the direct
//! arrival's base distance (`base_distance / path_length`), matching the
//! convention documented in [`crate::direct_path`]. The result is clamped to
//! `[0, 1]`; reflections quieter than the configured floor are discarded.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//!
//! Consumes [`crate::scene::AcousticScene`] and emits
//! [`PropagationPath`](prism_audio_spatial::propagation::PropagationPath)s of
//! kind [`Reflection`](prism_audio_spatial::propagation::PathKind::Reflection);
//! merged with the direct and diffracted arrivals by
//! [`crate::backend::GeometricBackend`].

use alloc::vec::Vec;
use core::cmp::Ordering;

use bevy_math::ops;
use bevy_math::Vec3;
use prism_audio_core::math::Sample;
use prism_audio_spatial::doppler::SPEED_OF_SOUND_MPS;
use prism_audio_spatial::geometry::{Emitter, Listener};
use prism_audio_spatial::propagation::{PathKind, PropagationPath, FULL_BAND_CUTOFF_HZ};

use crate::config::GeometricConfig;
use crate::scene::AcousticScene;

/// Barycentric tolerance when testing whether the reflection point lands on the
/// candidate face. A small positive slack keeps points on a shared edge valid
/// for both adjoining triangles instead of falling through the seam.
const BARYCENTRIC_TOLERANCE: Sample = 1.0e-4;

/// Resolves the first-order specular reflections of `emitter` at `listener`.
///
/// Returns the audible reflections sorted loudest-first, at most
/// [`GeometricConfig::max_reflections`]. An empty scene, a disabled reflection
/// stage, or a zero reflection budget yields an empty list.
#[must_use]
pub fn resolve_reflections(
    scene: &AcousticScene,
    listener: &Listener,
    emitter: &Emitter,
    config: &GeometricConfig,
    base_distance: Sample,
) -> Vec<PropagationPath> {
    let mut paths = Vec::new();
    if !config.reflections_enabled || config.max_reflections == 0 || scene.is_empty() {
        return paths;
    }

    let eps = config.surface_epsilon_m.max(0.0);
    for triangle in 0..scene.triangle_count() {
        let Some([a, b, c]) = scene.triangle(triangle) else {
            continue;
        };
        let Some(normal) = scene.triangle_normal(triangle) else {
            continue;
        };

        // Signed distances of both endpoints to the reflector's plane. A valid
        // specular bounce needs the source and listener on the same side.
        let d_listener = (listener.position - a).dot(normal);
        let d_source = (emitter.position - a).dot(normal);
        if d_listener * d_source <= 0.0 {
            continue;
        }

        // Mirror the source across the plane and intersect the listener->image
        // ray with the plane: that crossing is the specular reflection point.
        let image = emitter.position - 2.0 * d_source * normal;
        let direction = image - listener.position;
        let denom = direction.dot(normal);
        if denom.abs() <= f32::EPSILON {
            continue;
        }
        let t = -d_listener / denom;
        if !(t > 0.0 && t < 1.0) {
            continue;
        }
        let point = listener.position + t * direction;
        if !point_in_triangle(point, a, b, c) {
            continue;
        }

        // Both legs of the bounce must have a clear line to the reflection
        // point; a wall between would prevent this specular arrival.
        if scene.segment_blocked(listener.position, point, eps)
            || scene.segment_blocked(point, emitter.position, eps)
        {
            continue;
        }

        let path_length = distance(listener.position, point) + distance(point, emitter.position);
        if path_length <= 0.0 {
            continue;
        }

        let spreading = (base_distance / path_length).clamp(0.0, 1.0);
        let gain = (scene.material(triangle).reflection_gain() * spreading).clamp(0.0, 1.0);
        if gain <= config.min_gain {
            continue;
        }

        let local = listener.localize(&Emitter::point(point, Vec3::ZERO));
        let candidate = PropagationPath {
            kind: PathKind::Reflection,
            delay_seconds: path_length / SPEED_OF_SOUND_MPS,
            gain,
            cutoff_hz: FULL_BAND_CUTOFF_HZ,
            direction: local.direction,
        };
        if !is_duplicate(&paths, &candidate) {
            paths.push(candidate);
        }
    }

    paths.sort_by(|lhs, rhs| rhs.gain.partial_cmp(&lhs.gain).unwrap_or(Ordering::Equal));
    paths.truncate(config.max_reflections);
    paths
}

/// Whether `p` lies within triangle `a`, `b`, `c` (coplanar barycentric test
/// with a small positive slack so shared edges belong to both faces).
#[must_use]
fn point_in_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> bool {
    let v0 = c - a;
    let v1 = b - a;
    let v2 = p - a;
    let dot00 = v0.dot(v0);
    let dot01 = v0.dot(v1);
    let dot02 = v0.dot(v2);
    let dot11 = v1.dot(v1);
    let dot12 = v1.dot(v2);
    let denom = dot00 * dot11 - dot01 * dot01;
    if denom.abs() <= f32::EPSILON {
        return false;
    }
    let inv = 1.0 / denom;
    let u = (dot11 * dot02 - dot01 * dot12) * inv;
    let v = (dot00 * dot12 - dot01 * dot02) * inv;
    u >= -BARYCENTRIC_TOLERANCE
        && v >= -BARYCENTRIC_TOLERANCE
        && (u + v) <= 1.0 + BARYCENTRIC_TOLERANCE
}

/// Whether `candidate` duplicates an arrival already kept (same delay and
/// direction within a tight tolerance). Two coplanar triangles sharing an edge
/// can both host a reflection point on that seam; this keeps only one.
#[must_use]
fn is_duplicate(paths: &[PropagationPath], candidate: &PropagationPath) -> bool {
    paths.iter().any(|existing| {
        (existing.delay_seconds - candidate.delay_seconds).abs() < 1.0e-6
            && existing.direction.dot(candidate.direction) > 0.9999
    })
}

/// Deterministic Euclidean distance (routes through [`bevy_math::ops::sqrt`]).
#[inline]
#[must_use]
fn distance(a: Vec3, b: Vec3) -> Sample {
    let d = a - b;
    ops::sqrt(d.dot(d))
}

#[cfg(test)]
mod tests {
    use super::{point_in_triangle, resolve_reflections};
    use alloc::vec;
    use bevy_math::{Quat, Vec3};
    use prism_audio_spatial::geometry::{Emitter, Listener};
    use prism_audio_spatial::propagation::{AcousticMaterial, PathKind};

    use crate::config::GeometricConfig;
    use crate::material_map::MaterialTable;
    use crate::scene::AcousticScene;

    // A large floor in the plane y = 0 spanning x,z in [-10, 10].
    fn floor(material: AcousticMaterial) -> AcousticScene {
        let vertices = vec![
            Vec3::new(-10.0, 0.0, -10.0),
            Vec3::new(10.0, 0.0, -10.0),
            Vec3::new(10.0, 0.0, 10.0),
            Vec3::new(-10.0, 0.0, 10.0),
        ];
        let indices = vec![[0, 1, 2], [0, 2, 3]];
        AcousticScene::new(vertices, indices, MaterialTable::uniform(material)).unwrap()
    }

    #[test]
    fn floor_bounce_is_found() {
        // Listener and source both 2 m above a reflective floor, 8 m apart.
        let scene = floor(AcousticMaterial::new(0.0, 0.9));
        let listener = Listener::new(Vec3::new(-4.0, 2.0, 0.0), Quat::IDENTITY, Vec3::ZERO);
        let emitter = Emitter::point(Vec3::new(4.0, 2.0, 0.0), Vec3::ZERO);
        let cfg = GeometricConfig::new(48_000);
        let base = (emitter.position - listener.position).length();
        let paths = resolve_reflections(&scene, &listener, &emitter, &cfg, base);
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].kind, PathKind::Reflection);
        // The bounced route is longer than the direct one, so it is delayed.
        assert!(paths[0].delay_seconds > base / 343.0);
        assert!(paths[0].gain > 0.0 && paths[0].gain <= 1.0);
    }

    #[test]
    fn disabled_stage_yields_nothing() {
        let scene = floor(AcousticMaterial::new(0.0, 0.9));
        let listener = Listener::new(Vec3::new(-4.0, 2.0, 0.0), Quat::IDENTITY, Vec3::ZERO);
        let emitter = Emitter::point(Vec3::new(4.0, 2.0, 0.0), Vec3::ZERO);
        let cfg = GeometricConfig::new(48_000).without_reflections();
        let base = (emitter.position - listener.position).length();
        assert!(resolve_reflections(&scene, &listener, &emitter, &cfg, base).is_empty());
    }

    #[test]
    fn same_side_requirement_rejects_through_bounces() {
        // Source below the floor, listener above: no shared-side bounce exists.
        let scene = floor(AcousticMaterial::new(0.0, 0.9));
        let listener = Listener::new(Vec3::new(-4.0, 2.0, 0.0), Quat::IDENTITY, Vec3::ZERO);
        let emitter = Emitter::point(Vec3::new(4.0, -2.0, 0.0), Vec3::ZERO);
        let cfg = GeometricConfig::new(48_000);
        let base = (emitter.position - listener.position).length();
        assert!(resolve_reflections(&scene, &listener, &emitter, &cfg, base).is_empty());
    }

    #[test]
    fn budget_caps_reflection_count() {
        let scene = floor(AcousticMaterial::new(0.0, 0.9));
        let listener = Listener::new(Vec3::new(-4.0, 2.0, 0.0), Quat::IDENTITY, Vec3::ZERO);
        let emitter = Emitter::point(Vec3::new(4.0, 2.0, 0.0), Vec3::ZERO);
        let cfg = GeometricConfig::new(48_000).with_max_reflections(0);
        let base = (emitter.position - listener.position).length();
        assert!(resolve_reflections(&scene, &listener, &emitter, &cfg, base).is_empty());
    }

    #[test]
    fn barycentric_contains_centroid_only() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(1.0, 0.0, 0.0);
        let c = Vec3::new(0.0, 1.0, 0.0);
        let centroid = (a + b + c) / 3.0;
        assert!(point_in_triangle(centroid, a, b, c));
        assert!(!point_in_triangle(Vec3::new(2.0, 2.0, 0.0), a, b, c));
    }
}
