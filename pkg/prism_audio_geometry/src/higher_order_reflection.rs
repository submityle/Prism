//! Higher-order specular reflections resolved with the recursive image-source
//! method.
//!
//! [`crate::reflection_path`] resolves the single-bounce arrivals; this module
//! resolves the second- and higher-order bounces a wave takes when it reflects
//! off several surfaces in sequence before reaching the listener (a corner
//! reflector returning energy, a corridor flutter, the dense early-reflection
//! cluster of a shoebox room). It is the geometric sibling of the shoebox
//! image-source expansions in the spatial crate, but driven by the actual
//! triangle mesh instead of six axis-aligned walls.
//!
//! # Method
//!
//! For an ordered face sequence `f_1, f_2, …, f_n` (the order in which the wave
//! strikes them, leaving the source and arriving at the listener), the image of
//! the source is mirrored successively: `I_0 = S`, `I_k = mirror(I_{k-1}, f_k)`.
//! The listener sees the final image `I_n`. The reflection points are then
//! back-traced from the listener: `P_n` is where the segment `listener -> I_n`
//! crosses `f_n`, `P_{n-1}` is where `P_n -> I_{n-1}` crosses `f_{n-1}`, and so
//! on down to `P_1 -> S`. A sequence is a physical arrival only when every
//! crossing lands inside its triangle and every one of the `n + 1` real legs
//! (`listener -> P_n -> … -> P_1 -> source`) has a clear line of sight.
//!
//! # Gain convention
//!
//! A kept arrival carries the per-band product of the specular reflection of
//! every surface it bounced off (modelling the colouration accumulating in
//! series), scaled by the extra spherical spreading of its longer route
//! relative to the direct arrival's base distance (`base_distance /
//! path_length`), matching [`crate::reflection_path`]. The coloured spectrum is
//! factored into a broadband gain plus a relative colour; arrivals quieter than
//! the configured floor are discarded.
//!
//! # Determinism and budget
//!
//! Faces are visited in index order and the depth-first expansion is bounded by
//! [`crate::config::MAX_SUPPORTED_REFLECTION_ORDER`] and a fixed per-query
//! evaluation budget ([`MAX_IMAGE_SOURCE_EVALUATIONS`]), so the resolved set is
//! bit-reproducible and the control-rate cost is bounded regardless of mesh
//! size. Every transcendental routes through [`bevy_math::ops`]; the resolver
//! allocates only control-rate scratch and cannot panic.
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
//! kind [`Reflection`](prism_audio_spatial::propagation::PathKind::Reflection),
//! merged with the direct, first-order reflected, and diffracted arrivals by
//! [`crate::backend::GeometricBackend`]. It complements, and never duplicates,
//! the single-bounce arrivals from [`crate::reflection_path`].

use alloc::vec::Vec;
use core::cmp::Ordering;

use bevy_math::ops;
use bevy_math::Vec3;
use prism_audio_core::math::Sample;
use prism_audio_spatial::doppler::SPEED_OF_SOUND_MPS;
use prism_audio_spatial::geometry::{Emitter, Listener};
use prism_audio_spatial::propagation::{PathKind, PropagationPath, FULL_BAND_CUTOFF_HZ};
use prism_audio_spatial::BandGains;

use crate::config::{GeometricConfig, MAX_SUPPORTED_REFLECTION_ORDER};
use crate::scene::AcousticScene;

/// Barycentric tolerance when testing whether a back-traced reflection point
/// lands on its candidate face. Matches [`crate::reflection_path`] so a point on
/// a shared edge stays valid for both adjoining triangles.
const BARYCENTRIC_TOLERANCE: Sample = 1.0e-4;

/// Hard cap on the number of image-source mirror evaluations performed per
/// query. The depth-first expansion of face sequences is `O(F^order)` in the
/// worst case; this bound keeps a large mesh from exploding the control-rate
/// cost while leaving every realistic small acoustic mesh fully explored. Faces
/// are visited in index order, so which sequences the budget covers is
/// deterministic.
const MAX_IMAGE_SOURCE_EVALUATIONS: usize = 4096;

/// The fixed invariants of one higher-order reflection query, bundled so the
/// recursive expansion and validation stay within the argument budget.
struct ImageSourceTrace<'a> {
    scene: &'a AcousticScene,
    listener: &'a Listener,
    emitter: &'a Emitter,
    config: &'a GeometricConfig,
    base_distance: Sample,
    eps: Sample,
    max_order: usize,
}

/// Mutable depth-first state threaded through the recursive expansion: the face
/// index chain, the successive mirrored source images aligned with it, the
/// remaining evaluation budget, and the collected arrivals.
struct TraceState {
    /// Face indices of the current chain `f_1, …, f_k` in strike order.
    faces: Vec<usize>,
    /// The successive mirrored source images aligned with [`Self::faces`], where
    /// `images[k]` is the source mirrored through `faces[0..=k]`.
    images: Vec<Vec3>,
    /// Remaining mirror evaluations before the query stops expanding.
    budget: usize,
    /// Arrivals accepted so far.
    out: Vec<PropagationPath>,
}

/// Resolves the second- and higher-order specular reflections of `emitter` at
/// `listener`, up to [`GeometricConfig::max_reflection_order`].
///
/// Returns the audible multi-bounce arrivals sorted loudest-first, at most
/// [`GeometricConfig::max_reflections`]. An empty scene, a disabled reflection
/// stage, or a configured order below `2` yields an empty list (the single
/// bounce is [`crate::reflection_path`]'s job and is never duplicated here).
#[must_use]
pub fn resolve_higher_order_reflections(
    scene: &AcousticScene,
    listener: &Listener,
    emitter: &Emitter,
    config: &GeometricConfig,
    base_distance: Sample,
) -> Vec<PropagationPath> {
    if !config.reflections_enabled || scene.is_empty() {
        return Vec::new();
    }
    let max_order = config.max_reflection_order.min(MAX_SUPPORTED_REFLECTION_ORDER);
    if max_order < 2 {
        return Vec::new();
    }

    let trace = ImageSourceTrace {
        scene,
        listener,
        emitter,
        config,
        base_distance,
        eps: config.surface_epsilon_m.max(0.0),
        max_order,
    };
    let mut state = TraceState {
        faces: Vec::with_capacity(max_order),
        images: Vec::with_capacity(max_order),
        budget: MAX_IMAGE_SOURCE_EVALUATIONS,
        out: Vec::new(),
    };

    expand(&trace, emitter.position, &mut state);

    let mut paths = state.out;
    paths.sort_by(|lhs, rhs| rhs.gain.partial_cmp(&lhs.gain).unwrap_or(Ordering::Equal));
    paths.truncate(config.max_reflections);
    paths
}

/// Depth-first extension of the current image-source chain by one more face.
///
/// `current_image` is the source mirrored through every face already on the
/// chain (`I_k` for a chain of length `k`); each new face mirrors it again. Once
/// the chain is at least two faces long, the completed chain is validated and,
/// if physical and audible, pushed to the output before recursing deeper.
fn expand(trace: &ImageSourceTrace<'_>, current_image: Vec3, state: &mut TraceState) {
    if state.faces.len() >= trace.max_order {
        return;
    }
    for face in 0..trace.scene.triangle_count() {
        if state.budget == 0 {
            return;
        }
        // A specular bounce cannot use the same face twice in a row.
        if state.faces.last() == Some(&face) {
            continue;
        }
        let Some(normal) = trace.scene.triangle_normal(face) else {
            continue;
        };
        let Some([anchor, _, _]) = trace.scene.triangle(face) else {
            continue;
        };
        let signed = (current_image - anchor).dot(normal);
        // An image already on the plane cannot be mirrored into a new one.
        if ops::abs(signed) <= f32::EPSILON {
            continue;
        }
        let next_image = current_image - 2.0 * signed * normal;
        state.budget -= 1;

        state.faces.push(face);
        state.images.push(next_image);
        if state.faces.len() >= 2
            && let Some(path) = validate_chain(trace, &state.faces, &state.images)
            && !is_duplicate(&state.out, &path)
        {
            state.out.push(path);
        }
        expand(trace, next_image, state);
        state.faces.pop();
        state.images.pop();
    }
}

/// Validates one completed image-source chain and, when it is a physical and
/// audible arrival, builds its [`PropagationPath`].
///
/// Back-traces the reflection points from the listener, requiring every
/// crossing to land inside its triangle and every real leg to be unobstructed,
/// then accumulates the per-band specular product and the total route length.
#[must_use]
fn validate_chain(
    trace: &ImageSourceTrace<'_>,
    faces: &[usize],
    images: &[Vec3],
) -> Option<PropagationPath> {
    let n = faces.len();
    debug_assert!((2..=MAX_SUPPORTED_REFLECTION_ORDER).contains(&n));
    debug_assert_eq!(images.len(), n);

    // Back-trace the reflection points from the listener toward the source.
    // `points[k]` is the bounce point on `faces[k]`.
    let mut points = [Vec3::ZERO; MAX_SUPPORTED_REFLECTION_ORDER];
    let mut from = trace.listener.position;
    for k in (0..n).rev() {
        let face = faces[k];
        let image = images[k];
        let normal = trace.scene.triangle_normal(face)?;
        let [a, b, c] = trace.scene.triangle(face)?;
        let d_from = (from - a).dot(normal);
        let direction = image - from;
        let denom = direction.dot(normal);
        if ops::abs(denom) <= f32::EPSILON {
            return None;
        }
        let t = -d_from / denom;
        if !(t > 0.0 && t < 1.0) {
            return None;
        }
        let point = from + t * direction;
        if !point_in_triangle(point, a, b, c) {
            return None;
        }
        points[k] = point;
        from = point;
    }

    // Walk the real route source -> P_1 -> … -> P_n -> listener, requiring each
    // leg clear and accumulating length and the series specular colouration.
    let mut previous = trace.emitter.position;
    let mut path_length = 0.0;
    let mut reflection = BandGains::UNITY;
    for k in 0..n {
        let point = points[k];
        if trace.scene.segment_blocked(previous, point, trace.eps) {
            return None;
        }
        path_length += distance(previous, point);
        reflection = reflection.combine(trace.scene.material(faces[k]).specular_reflection());
        previous = point;
    }
    if trace
        .scene
        .segment_blocked(previous, trace.listener.position, trace.eps)
    {
        return None;
    }
    path_length += distance(previous, trace.listener.position);
    if path_length <= 0.0 {
        return None;
    }

    let spreading = (trace.base_distance / path_length).clamp(0.0, 1.0);
    let (gain, bands) = reflection.scaled(spreading).split_peak();
    if gain <= trace.config.min_gain {
        return None;
    }

    // The arrival comes from the direction of the last bounce point.
    let local = trace
        .listener
        .localize(&Emitter::point(points[n - 1], Vec3::ZERO));
    Some(PropagationPath {
        kind: PathKind::Reflection,
        delay_seconds: path_length / SPEED_OF_SOUND_MPS,
        gain,
        cutoff_hz: FULL_BAND_CUTOFF_HZ,
        bands,
        direction: local.direction,
    })
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
    if ops::abs(denom) <= f32::EPSILON {
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
/// direction within a tight tolerance). Different face sequences can converge on
/// an almost identical route; this keeps only one.
#[must_use]
fn is_duplicate(paths: &[PropagationPath], candidate: &PropagationPath) -> bool {
    paths.iter().any(|existing| {
        ops::abs(existing.delay_seconds - candidate.delay_seconds) < 1.0e-6
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
    use super::{point_in_triangle, resolve_higher_order_reflections};
    use alloc::vec;
    use bevy_math::{Quat, Vec3};
    use prism_audio_spatial::geometry::{Emitter, Listener};
    use prism_audio_spatial::propagation::{AcousticMaterial, PathKind};

    use crate::config::GeometricConfig;
    use crate::material_map::MaterialTable;
    use crate::scene::AcousticScene;

    // A right-angle corner reflector: wall A in the plane x = 0 and wall B in
    // the plane z = 0, both reflective and large enough to host the bounces.
    fn corner(material: AcousticMaterial) -> AcousticScene {
        let vertices = vec![
            // Wall A (x = 0), spanning y in [-5, 5], z in [-10, 10].
            Vec3::new(0.0, -5.0, -10.0),
            Vec3::new(0.0, 5.0, -10.0),
            Vec3::new(0.0, 5.0, 10.0),
            Vec3::new(0.0, -5.0, 10.0),
            // Wall B (z = 0), spanning x in [-10, 10], y in [-5, 5].
            Vec3::new(-10.0, -5.0, 0.0),
            Vec3::new(10.0, -5.0, 0.0),
            Vec3::new(10.0, 5.0, 0.0),
            Vec3::new(-10.0, 5.0, 0.0),
        ];
        let indices = vec![
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
        ];
        AcousticScene::new(vertices, indices, MaterialTable::uniform_scalar(material)).unwrap()
    }

    #[test]
    fn second_order_corner_bounce_is_found() {
        let scene = corner(AcousticMaterial::new(80.0, 0.9));
        // Both in the +x,+z quadrant so the direct line never touches a wall.
        let listener = Listener::new(Vec3::new(4.0, 0.0, 2.0), Quat::IDENTITY, Vec3::ZERO);
        let emitter = Emitter::point(Vec3::new(2.0, 0.0, 4.0), Vec3::ZERO);
        let base = (emitter.position - listener.position).length();
        let cfg = GeometricConfig::new(48_000).with_max_reflection_order(2);
        let paths = resolve_higher_order_reflections(&scene, &listener, &emitter, &cfg, base);

        assert!(!paths.is_empty(), "expected a double-bounce corner arrival");
        let arrival = &paths[0];
        assert_eq!(arrival.kind, PathKind::Reflection);
        // The two-bounce route is far longer than the direct line.
        assert!(arrival.delay_seconds > base / 343.0);
        assert!(arrival.gain > 0.0 && arrival.gain <= 1.0);
    }

    #[test]
    fn first_order_only_yields_no_higher_order() {
        let scene = corner(AcousticMaterial::new(80.0, 0.9));
        let listener = Listener::new(Vec3::new(4.0, 0.0, 2.0), Quat::IDENTITY, Vec3::ZERO);
        let emitter = Emitter::point(Vec3::new(2.0, 0.0, 4.0), Vec3::ZERO);
        let base = (emitter.position - listener.position).length();
        // Default order is 1: this stage contributes nothing.
        let cfg = GeometricConfig::new(48_000);
        assert!(resolve_higher_order_reflections(&scene, &listener, &emitter, &cfg, base).is_empty());
    }

    #[test]
    fn disabled_stage_yields_nothing() {
        let scene = corner(AcousticMaterial::new(80.0, 0.9));
        let listener = Listener::new(Vec3::new(4.0, 0.0, 2.0), Quat::IDENTITY, Vec3::ZERO);
        let emitter = Emitter::point(Vec3::new(2.0, 0.0, 4.0), Vec3::ZERO);
        let base = (emitter.position - listener.position).length();
        let cfg = GeometricConfig::new(48_000)
            .with_max_reflection_order(3)
            .without_reflections();
        assert!(resolve_higher_order_reflections(&scene, &listener, &emitter, &cfg, base).is_empty());
    }

    #[test]
    fn second_order_bounce_is_quieter_than_first_order() {
        // A single reflective floor supports only a first-order bounce, so the
        // second-order stage finds nothing even when enabled.
        let vertices = vec![
            Vec3::new(-10.0, 0.0, -10.0),
            Vec3::new(10.0, 0.0, -10.0),
            Vec3::new(10.0, 0.0, 10.0),
            Vec3::new(-10.0, 0.0, 10.0),
        ];
        let indices = vec![[0, 1, 2], [0, 2, 3]];
        let material = AcousticMaterial::new(80.0, 0.9);
        let scene =
            AcousticScene::new(vertices, indices, MaterialTable::uniform_scalar(material)).unwrap();
        let listener = Listener::new(Vec3::new(-4.0, 2.0, 0.0), Quat::IDENTITY, Vec3::ZERO);
        let emitter = Emitter::point(Vec3::new(4.0, 2.0, 0.0), Vec3::ZERO);
        let base = (emitter.position - listener.position).length();
        let cfg = GeometricConfig::new(48_000).with_max_reflection_order(3);
        // Two coplanar triangles cannot form a non-degenerate double bounce.
        assert!(resolve_higher_order_reflections(&scene, &listener, &emitter, &cfg, base).is_empty());
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
