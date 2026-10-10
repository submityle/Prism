//! Coupled reflection-and-diffraction arrivals.
//!
//! A lone specular bounce ([`crate::reflection_path`]) and a lone shadow-edge
//! bend ([`crate::diffraction_path`]) are the first-order secondary arrivals a
//! wave takes to a shadowed listener. The next route up the energy ladder
//! *couples* the two mechanisms: the wave reflects off one face **and** bends
//! over one diffracting edge, in either order. This is the arrival that lets a
//! source heard around a corner still pick up the colour of the wall it
//! glanced off on the way, and it is exactly the second-order term Steam
//! Audio's path tracer spends its budget on once the pure first-order set is
//! exhausted.
//!
//! This module resolves both orderings with the same image-source and
//! least-detour primitives the pure resolvers already use, so a coupled arrival
//! is geometrically consistent with the lone bounce and the lone bend it
//! extends rather than a parallel approximation:
//!
//! - **Reflect-then-diffract**: mirror the emitter in the reflecting face to an
//!   image `I`, find the least-detour corner `C` on the edge as seen from `I`
//!   and the listener, then intersect the segment `I`..`C` with the face to
//!   recover the reflection point `P`. The physical route is
//!   `emitter -> P -> C -> listener`.
//! - **Diffract-then-reflect**: mirror the *listener* in the reflecting face to
//!   an image `L'`, find the least-detour corner `C` as seen from the emitter
//!   and `L'`, then intersect `C`..`L'` with the face to recover `P`. The
//!   physical route is `emitter -> C -> P -> listener`.
//!
//! # Gain convention
//!
//! A coupled arrival multiplies the reflecting surface's specular per-band
//! reflection by the edge's diffraction colour (a Maekawa low-pass or a UTD
//! wedge spectrum, matching [`crate::diffraction_path`]), then folds in the
//! extra spherical spreading of its longer three-leg route relative to the
//! direct arrival's base distance (`base_distance / path_length`). The result
//! is factored into a flat broadband gain and a relative colour exactly as the
//! pure resolvers do; arrivals quieter than the configured floor are discarded.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//!
//! Consumes [`crate::scene::AcousticScene`], reuses
//! [`crate::reflection_path::point_in_triangle`] and the
//! [`crate::diffraction_edges`] detour/wedge primitives, and emits
//! [`PropagationPath`](prism_audio_spatial::propagation::PropagationPath)s of
//! kind [`Diffraction`](prism_audio_spatial::propagation::PathKind::Diffraction)
//! (a coupled arrival carries diffraction colour and a shadow low-pass corner).
//! Merged, loudest-first, with the direct, reflected, and diffracted arrivals
//! by [`crate::backend::GeometricBackend`]; enabled by
//! [`crate::config::GeometricConfig::with_coupled_paths`].

use alloc::vec::Vec;
use core::cmp::Ordering;

use bevy_math::Vec3;
use prism_audio_core::math::Sample;
use prism_audio_spatial::doppler::SPEED_OF_SOUND_MPS;
use prism_audio_spatial::geometry::{Emitter, Listener};
use prism_audio_spatial::propagation::{
    diffraction_cutoff_hz, diffraction_gain, edge_path_difference, PathKind, PropagationPath,
};
use prism_audio_spatial::BandGains;

use crate::config::{DiffractionModel, GeometricConfig};
use crate::diffraction_edges::{
    diffracting_edges, distance, least_detour_point, utd_wedge, EdgeData,
};
use crate::reflection_path::point_in_triangle;
use crate::scene::AcousticScene;

/// Minimum over-the-edge detour (metres) a coupled bend must have to count as a
/// genuine diffraction rather than a near-grazing coincidence. Shares the floor
/// used by [`crate::higher_order_diffraction`] so the two resolvers agree on
/// when an edge is actually in shadow.
const MIN_EDGE_DETOUR_M: Sample = 1.0e-3;

/// Hard ceiling on the number of (face, edge) pairs the coupled resolver
/// evaluates per query. Each pair is checked in both orderings, so the real
/// work is bounded at twice this. The product of faces and edges is quadratic
/// in scene size; this cap keeps a dense mesh from blowing the control-rate
/// budget while still covering every pair in the small, local scenes the
/// coupled term matters for.
const MAX_COUPLED_EVALUATIONS: u32 = 4096;

/// Resolves the coupled reflection-and-diffraction arrivals of `emitter` at
/// `listener`.
///
/// Returns the audible coupled arrivals sorted loudest-first, at most
/// [`GeometricConfig::max_coupled_paths`]. A disabled coupled stage, a disabled
/// reflection or diffraction stage, a zero coupled budget, or an empty scene
/// yields an empty list: a coupled arrival needs both mechanisms, so it is only
/// traced when both are enabled.
#[must_use]
pub fn resolve_coupled_paths(
    scene: &AcousticScene,
    listener: &Listener,
    emitter: &Emitter,
    config: &GeometricConfig,
    base_distance: Sample,
) -> Vec<PropagationPath> {
    if !config.coupled_enabled
        || !config.reflections_enabled
        || !config.diffraction_enabled
        || config.max_coupled_paths == 0
        || scene.is_empty()
    {
        return Vec::new();
    }

    let solver = CoupledSolver {
        scene,
        listener,
        emitter,
        config,
        base_distance,
        eps: config.surface_epsilon_m.max(0.0),
    };
    solver.resolve()
}

/// Packs the per-query inputs shared by both coupled orderings so the geometry
/// helpers stay single-argument-list simple instead of threading six values
/// through every call.
struct CoupledSolver<'a> {
    scene: &'a AcousticScene,
    listener: &'a Listener,
    emitter: &'a Emitter,
    config: &'a GeometricConfig,
    base_distance: Sample,
    eps: Sample,
}

impl CoupledSolver<'_> {
    /// Evaluates every (face, edge) pair in both orderings within the budget,
    /// keeping the loudest audible coupled arrivals.
    #[must_use]
    fn resolve(&self) -> Vec<PropagationPath> {
        let edges = diffracting_edges(self.scene);
        let mut paths: Vec<PropagationPath> = Vec::new();
        let mut remaining = MAX_COUPLED_EVALUATIONS;

        for triangle in 0..self.scene.triangle_count() {
            let Some([a, b, c]) = self.scene.triangle(triangle) else {
                continue;
            };
            let Some(normal) = self.scene.triangle_normal(triangle) else {
                continue;
            };
            let face = Face {
                a,
                b,
                c,
                normal,
                bands: self.scene.material(triangle).specular_reflection(),
            };

            for edge in &edges {
                if remaining == 0 {
                    return finalise(paths, self.config.max_coupled_paths);
                }
                remaining -= 1;

                // Reflecting on a face off one of the very endpoints of the edge
                // we also bend over is a degenerate coincidence, not a distinct
                // second-order route; skip it so it cannot alias a lone bend.
                if edge_on_face(edge, &[a, b, c]) {
                    continue;
                }

                if let Some(path) = self.reflect_then_diffract(&face, edge)
                    && !is_duplicate(&paths, &path)
                {
                    paths.push(path);
                }
                if let Some(path) = self.diffract_then_reflect(&face, edge)
                    && !is_duplicate(&paths, &path)
                {
                    paths.push(path);
                }
            }
        }

        finalise(paths, self.config.max_coupled_paths)
    }

    /// `emitter -> reflect at P on face -> bend at corner C on edge -> listener`.
    #[must_use]
    fn reflect_then_diffract(&self, face: &Face, edge: &EdgeData) -> Option<PropagationPath> {
        // Mirror the emitter across the face: the reflected leg behaves as a
        // straight ray from this image.
        let d_source = (self.emitter.position - face.a).dot(face.normal);
        let image = self.emitter.position - 2.0 * d_source * face.normal;

        // The bend corner is the least-detour point as seen from the image and
        // the listener; the reflection point is where image..corner pierces the
        // face.
        let corner = least_detour_point(edge.start, edge.end, self.listener.position, image);
        let point = Self::plane_crossing(face, image, corner)?;

        // All three physical legs must be clear.
        if self
            .scene
            .segment_blocked(self.emitter.position, point, self.eps)
            || self.scene.segment_blocked(point, corner, self.eps)
            || self
                .scene
                .segment_blocked(corner, self.listener.position, self.eps)
        {
            return None;
        }

        let delta = edge_path_difference(self.listener.position, corner, image);
        if delta < MIN_EDGE_DETOUR_M {
            return None;
        }

        let length = distance(self.emitter.position, point)
            + distance(point, corner)
            + distance(corner, self.listener.position);
        let wedge_source = image;
        let wedge_receiver = self.listener.position;
        self.shade(ShadeInputs {
            face,
            edge,
            corner,
            delta,
            length,
            wedge_source,
            wedge_receiver,
            // The last leg lands on the listener from the corner, so the corner
            // is the arrival point the listener localises.
            arrival_point: corner,
        })
    }

    /// `emitter -> bend at corner C on edge -> reflect at P on face -> listener`.
    #[must_use]
    fn diffract_then_reflect(&self, face: &Face, edge: &EdgeData) -> Option<PropagationPath> {
        // Mirror the listener across the face: the reflected leg into the
        // listener behaves as a straight ray to this image.
        let d_listener = (self.listener.position - face.a).dot(face.normal);
        let image = self.listener.position - 2.0 * d_listener * face.normal;

        let corner = least_detour_point(edge.start, edge.end, image, self.emitter.position);
        let point = Self::plane_crossing(face, corner, image)?;

        if self
            .scene
            .segment_blocked(self.emitter.position, corner, self.eps)
            || self.scene.segment_blocked(corner, point, self.eps)
            || self
                .scene
                .segment_blocked(point, self.listener.position, self.eps)
        {
            return None;
        }

        let delta = edge_path_difference(image, corner, self.emitter.position);
        if delta < MIN_EDGE_DETOUR_M {
            return None;
        }

        let length = distance(self.emitter.position, corner)
            + distance(corner, point)
            + distance(point, self.listener.position);
        let wedge_source = self.emitter.position;
        let wedge_receiver = image;
        self.shade(ShadeInputs {
            face,
            edge,
            corner,
            delta,
            length,
            wedge_source,
            wedge_receiver,
            // The last leg lands on the listener from the reflection point, so
            // that point is the arrival direction the listener localises.
            arrival_point: point,
        })
    }

    /// Intersects segment `from`..`to` with `face`'s plane, returning the
    /// crossing point only when it lies strictly between the endpoints and
    /// inside the triangle.
    #[must_use]
    fn plane_crossing(face: &Face, from: Vec3, to: Vec3) -> Option<Vec3> {
        let dir = to - from;
        let denom = dir.dot(face.normal);
        if denom.abs() <= f32::EPSILON {
            return None;
        }
        let d_from = (from - face.a).dot(face.normal);
        let t = -d_from / denom;
        if !(t > 0.0 && t < 1.0) {
            return None;
        }
        let point = from + t * dir;
        if !point_in_triangle(point, face.a, face.b, face.c) {
            return None;
        }
        Some(point)
    }

    /// Shades a resolved coupled route into a [`PropagationPath`], combining the
    /// surface reflection colour with the edge diffraction colour and folding in
    /// the longer route's spreading. Returns `None` when the arrival falls below
    /// the audibility floor.
    #[must_use]
    fn shade(&self, inputs: ShadeInputs<'_>) -> Option<PropagationPath> {
        if inputs.length <= 0.0 {
            return None;
        }
        let spreading = (self.base_distance / inputs.length).clamp(0.0, 1.0);
        let cutoff_hz = diffraction_cutoff_hz(inputs.delta, self.config.sample_rate);

        let full = match self.config.diffraction_model {
            DiffractionModel::Maekawa => {
                let barrier = diffraction_gain(inputs.delta, self.config.diffraction_freq_hz);
                inputs
                    .face
                    .bands
                    .combine(BandGains::from_lowpass_cutoff(cutoff_hz))
                    .scaled(barrier * spreading)
            }
            DiffractionModel::Utd => {
                let wedge = utd_wedge(
                    inputs.edge,
                    inputs.corner,
                    inputs.wedge_source,
                    inputs.wedge_receiver,
                );
                inputs
                    .face
                    .bands
                    .combine(wedge.band_gains())
                    .scaled(spreading)
            }
        };

        let (gain, bands) = full.split_peak();
        if gain <= self.config.min_gain {
            return None;
        }

        let local = self
            .listener
            .localize(&Emitter::point(inputs.arrival_point, Vec3::ZERO));
        Some(PropagationPath {
            kind: PathKind::Diffraction,
            delay_seconds: inputs.length / SPEED_OF_SOUND_MPS,
            gain,
            cutoff_hz,
            bands,
            direction: local.direction,
        })
    }
}

/// A reflecting face resolved once per triangle: its three vertices, outward
/// normal, and specular per-band reflection colour.
struct Face {
    a: Vec3,
    b: Vec3,
    c: Vec3,
    normal: Vec3,
    bands: BandGains,
}

/// The resolved geometry and spectral inputs a coupled route hands to
/// [`CoupledSolver::shade`], packed so the shader stays one argument list.
struct ShadeInputs<'a> {
    face: &'a Face,
    edge: &'a EdgeData,
    corner: Vec3,
    delta: Sample,
    length: Sample,
    wedge_source: Vec3,
    wedge_receiver: Vec3,
    arrival_point: Vec3,
}

/// Sorts coupled arrivals loudest-first and keeps at most `budget` of them.
#[must_use]
fn finalise(mut paths: Vec<PropagationPath>, budget: usize) -> Vec<PropagationPath> {
    paths.sort_by(|lhs, rhs| rhs.gain.partial_cmp(&lhs.gain).unwrap_or(Ordering::Equal));
    paths.truncate(budget);
    paths
}

/// Whether `edge`'s two endpoints are both vertices of the triangle `face`.
/// Such an edge borders the very face we would reflect off, so coupling the two
/// is a degenerate self-interaction rather than a distinct second-order route.
#[must_use]
fn edge_on_face(edge: &EdgeData, face: &[Vec3; 3]) -> bool {
    let on_face = |p: Vec3| face.contains(&p);
    on_face(edge.start) && on_face(edge.end)
}

/// Whether `candidate` duplicates an arrival already kept (same delay and
/// arrival direction within a tight tolerance). The two orderings and two
/// triangles sharing a seam can resolve to the same physical route; this keeps
/// only one.
#[must_use]
fn is_duplicate(paths: &[PropagationPath], candidate: &PropagationPath) -> bool {
    paths.iter().any(|existing| {
        (existing.delay_seconds - candidate.delay_seconds).abs() < 1.0e-6
            && existing.direction.dot(candidate.direction) > 0.9999
    })
}

#[cfg(test)]
mod tests {
    use super::resolve_coupled_paths;
    use alloc::vec;
    use bevy_math::{Quat, Vec3};
    use prism_audio_spatial::doppler::SPEED_OF_SOUND_MPS;
    use prism_audio_spatial::geometry::{Emitter, Listener};
    use prism_audio_spatial::propagation::{AcousticMaterial, PathKind};

    use crate::config::{DiffractionModel, GeometricConfig};
    use crate::material_map::MaterialTable;
    use crate::scene::AcousticScene;

    // A reflecting floor plus a standing barrier. The floor is a y = 0 quad
    // spanning x, z in [-10, 10]; the barrier is an x = 0 quad spanning
    // y in [0, 2], z in [-6, 6] with its free top edge at y = 2. A listener and
    // emitter placed below the top edge on opposite sides are shadowed by the
    // barrier (no clear direct line) yet can be heard by a wave that glances off
    // the floor and bends over the barrier's top edge: the coupled arrival.
    fn floor_and_barrier() -> AcousticScene {
        let vertices = vec![
            // Reflecting floor at y = 0.
            Vec3::new(-10.0, 0.0, -10.0),
            Vec3::new(10.0, 0.0, -10.0),
            Vec3::new(10.0, 0.0, 10.0),
            Vec3::new(-10.0, 0.0, 10.0),
            // Barrier at x = 0, top edge at y = 2.
            Vec3::new(0.0, 0.0, -6.0),
            Vec3::new(0.0, 2.0, -6.0),
            Vec3::new(0.0, 2.0, 6.0),
            Vec3::new(0.0, 0.0, 6.0),
        ];
        let indices = vec![[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]];
        AcousticScene::new(
            vertices,
            indices,
            MaterialTable::uniform_scalar(AcousticMaterial::new(60.0, 0.9)),
        )
        .unwrap()
    }

    fn endpoints() -> (Listener, Emitter) {
        let listener = Listener::new(Vec3::new(-4.0, 0.5, 0.0), Quat::IDENTITY, Vec3::ZERO);
        let emitter = Emitter::point(Vec3::new(4.0, 0.5, 0.0), Vec3::ZERO);
        (listener, emitter)
    }

    fn base_distance(listener: &Listener, emitter: &Emitter) -> f32 {
        (emitter.position - listener.position).length()
    }

    #[test]
    fn reflect_and_bend_arrival_is_found() {
        let scene = floor_and_barrier();
        let (listener, emitter) = endpoints();
        let base = base_distance(&listener, &emitter);
        let cfg = GeometricConfig::new(48_000).with_coupled_paths();
        let paths = resolve_coupled_paths(&scene, &listener, &emitter, &cfg, base);

        assert!(
            !paths.is_empty(),
            "a floor glance that bends over the barrier should produce a coupled arrival"
        );
        for path in &paths {
            assert_eq!(path.kind, PathKind::Diffraction);
            // The three-leg reflect-and-bend route is strictly longer than the
            // straight line, so it arrives later than the direct wave would.
            assert!(path.delay_seconds > base / SPEED_OF_SOUND_MPS);
            assert!(path.gain > 0.0 && path.gain <= 1.0);
            for band in path.bands.bands() {
                assert!(band.is_finite() && (0.0..=1.0).contains(&band));
            }
        }
    }

    #[test]
    fn coupled_arrivals_are_sorted_loudest_first() {
        let scene = floor_and_barrier();
        let (listener, emitter) = endpoints();
        let base = base_distance(&listener, &emitter);
        let cfg = GeometricConfig::new(48_000).with_coupled_paths();
        let paths = resolve_coupled_paths(&scene, &listener, &emitter, &cfg, base);
        for pair in paths.windows(2) {
            assert!(pair[0].gain >= pair[1].gain);
        }
    }

    #[test]
    fn utd_model_also_resolves_the_coupled_route() {
        let scene = floor_and_barrier();
        let (listener, emitter) = endpoints();
        let base = base_distance(&listener, &emitter);
        let cfg = GeometricConfig::new(48_000)
            .with_coupled_paths()
            .with_diffraction_model(DiffractionModel::Utd);
        let paths = resolve_coupled_paths(&scene, &listener, &emitter, &cfg, base);
        assert!(!paths.is_empty());
        for path in &paths {
            assert_eq!(path.kind, PathKind::Diffraction);
            assert!(path.gain > 0.0 && path.gain <= 1.0);
        }
    }

    #[test]
    fn default_config_emits_no_coupled() {
        let scene = floor_and_barrier();
        let (listener, emitter) = endpoints();
        let base = base_distance(&listener, &emitter);
        // Coupling is opt-in: the default config leaves it off even though both
        // the reflection and diffraction stages are on.
        let cfg = GeometricConfig::new(48_000);
        assert!(resolve_coupled_paths(&scene, &listener, &emitter, &cfg, base).is_empty());
    }

    #[test]
    fn disabled_reflections_yield_nothing() {
        let scene = floor_and_barrier();
        let (listener, emitter) = endpoints();
        let base = base_distance(&listener, &emitter);
        // A coupled arrival needs the reflection mechanism; without it there is
        // no face to glance off, so nothing couples.
        let cfg = GeometricConfig::new(48_000)
            .with_coupled_paths()
            .without_reflections();
        assert!(resolve_coupled_paths(&scene, &listener, &emitter, &cfg, base).is_empty());
    }

    #[test]
    fn disabled_diffraction_yields_nothing() {
        let scene = floor_and_barrier();
        let (listener, emitter) = endpoints();
        let base = base_distance(&listener, &emitter);
        // Likewise it needs the diffraction mechanism; without it there is no
        // edge to bend over.
        let cfg = GeometricConfig::new(48_000)
            .with_coupled_paths()
            .without_diffraction();
        assert!(resolve_coupled_paths(&scene, &listener, &emitter, &cfg, base).is_empty());
    }

    #[test]
    fn zero_budget_yields_nothing() {
        let scene = floor_and_barrier();
        let (listener, emitter) = endpoints();
        let base = base_distance(&listener, &emitter);
        let cfg = GeometricConfig::new(48_000)
            .with_coupled_paths()
            .with_max_coupled_paths(0);
        assert!(resolve_coupled_paths(&scene, &listener, &emitter, &cfg, base).is_empty());
    }

    #[test]
    fn budget_caps_coupled_count() {
        let scene = floor_and_barrier();
        let (listener, emitter) = endpoints();
        let base = base_distance(&listener, &emitter);
        let cfg = GeometricConfig::new(48_000)
            .with_coupled_paths()
            .with_max_coupled_paths(1);
        let paths = resolve_coupled_paths(&scene, &listener, &emitter, &cfg, base);
        assert!(paths.len() <= 1);
    }

    #[test]
    fn empty_scene_is_silent() {
        let scene = AcousticScene::new(
            vec![],
            vec![],
            MaterialTable::uniform_scalar(AcousticMaterial::new(60.0, 0.9)),
        )
        .unwrap();
        let (listener, emitter) = endpoints();
        let base = base_distance(&listener, &emitter);
        let cfg = GeometricConfig::new(48_000).with_coupled_paths();
        assert!(resolve_coupled_paths(&scene, &listener, &emitter, &cfg, base).is_empty());
    }
}
