//! Arbitrary-order interleaved reflection-and-diffraction sequences.
//!
//! [`crate::coupled_path`] resolves the order-2 coupling: exactly one specular
//! bounce and one shadow-edge bend, in either order. Real rooms hide sources
//! behind richer chains - a wave glances off a wall and then bends around two
//! successive corners, or bends over an edge, reflects off a baffle, and bends
//! again - where the only audible route interleaves reflections and
//! diffractions three or more interactions deep. This module traces those
//! mixed sequences, matching the arbitrarily interleaved reflection and
//! diffraction paths Steam Audio's path tracer spends its remaining budget on
//! once the pure-reflection, pure-diffraction, and order-2 coupled sets are
//! exhausted.
//!
//! # Scope
//!
//! To avoid double-counting the lower resolvers, this module owns **only**
//! sequences with three or more total interactions that contain **both** at
//! least one reflection and at least one diffraction:
//!
//! - pure higher-order reflections stay with [`crate::higher_order_reflection`],
//! - pure higher-order diffractions stay with [`crate::higher_order_diffraction`],
//! - the lone-bounce-plus-lone-bend order-2 coupling stays with
//!   [`crate::coupled_path`].
//!
//! # Method
//!
//! Each interaction is a [`Interaction::Reflect`] off a face or a
//! [`Interaction::Diffract`] over a diffracting edge. For every ordered mixed
//! sequence (enumerated in deterministic face-then-edge index order, with no
//! two identical interactions adjacent and no edge reused), the taut stationary
//! route emitter -> p0 -> p1 -> ... -> listener is found by Gauss-Seidel
//! relaxation: a reflection point is re-placed at the specular crossing implied
//! by mirroring its outgoing neighbour across the face (Fermat's equal-angle
//! law), and a diffraction corner is re-placed at the least-detour point on its
//! edge, each given its current neighbours. The sweep converges to the
//! string-pulled-taut path shared with the pure resolvers. A sequence survives
//! only when every reflection point lands inside its triangle on a genuine
//! specular crossing, every diffraction corner bends by a real detour, every
//! leg is unobstructed, and the whole route is longer than the straight line.
//!
//! # Gain convention
//!
//! Each interaction attenuates independently and the factors multiply: a
//! reflection folds in the face's specular per-band reflection colour; a
//! diffraction folds in the edge's Maekawa low-pass colour (and broadband
//! barrier gain) or its UTD wedge spectrum, exactly as the single-mechanism
//! resolvers document. The product is scaled by the extra spherical spreading
//! of the longer route (`base_distance / path_length`), then factored into a
//! flat broadband gain and a relative colour with `cutoff_hz` set to the
//! tightest corner along the route. Arrivals below the configured floor drop.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//!
//! Reads a [`crate::scene::AcousticScene`], reuses
//! [`crate::reflection_path::point_in_triangle`] and the
//! [`crate::diffraction_edges`] detour/wedge primitives shared with the pure
//! resolvers, and emits
//! [`PropagationPath`](prism_audio_spatial::propagation::PropagationPath)s of
//! kind [`Diffraction`](prism_audio_spatial::propagation::PathKind::Diffraction)
//! (a coupled arrival carries diffraction colour and a shadow low-pass corner).
//! Merged, loudest-first, with the direct, reflected, diffracted, and order-2
//! coupled arrivals by [`crate::backend::GeometricBackend`]; enabled by
//! [`crate::config::GeometricConfig::with_max_coupled_order`].

use alloc::vec::Vec;
use core::cmp::Ordering;

use bevy_math::Vec3;
use prism_audio_core::math::Sample;
use prism_audio_spatial::doppler::SPEED_OF_SOUND_MPS;
use prism_audio_spatial::geometry::{Emitter, Listener};
use prism_audio_spatial::propagation::{
    diffraction_cutoff_hz, diffraction_gain, edge_path_difference, PathKind, PropagationPath,
    FULL_BAND_CUTOFF_HZ,
};
use prism_audio_spatial::BandGains;

use crate::config::{DiffractionModel, GeometricConfig, MAX_SUPPORTED_COUPLED_ORDER};
use crate::diffraction_edges::{
    diffracting_edges, distance, least_detour_point, utd_wedge, EdgeData,
};
use crate::reflection_path::point_in_triangle;
use crate::scene::AcousticScene;

/// Fixed number of Gauss-Seidel sweeps used to pull a mixed route taut. The
/// per-interaction update is contractive for the sparse chains that survive, so
/// a fixed count converges deterministically without a tolerance loop that
/// could vary across targets. Matches the relaxation budget of
/// [`crate::higher_order_diffraction`] with headroom for the extra reflection
/// projection.
const RELAXATION_SWEEPS: u32 = 32;

/// Upper bound on the number of mixed sequences evaluated per query. The number
/// of ordered interaction sequences grows factorially with the face and edge
/// counts, so this bound keeps a dense mesh from stalling the control-rate
/// query; sequences are explored in deterministic index order, so the retained
/// set is stable.
const MAX_SEQUENCE_EVALUATIONS: usize = 4096;

/// Minimum per-corner detour (metres) for a diffraction to count as a genuine
/// bend. A corner adding less than this merely re-derives a route one order
/// lower and is rejected so the orders never double-count the same arrival.
/// Shares the floor used by [`crate::coupled_path`] and
/// [`crate::higher_order_diffraction`].
const MIN_EDGE_DETOUR_M: Sample = 1.0e-3;

/// One interaction along a mixed route: a specular bounce off a scene triangle
/// or a shadow bend over a diffracting edge.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Interaction {
    /// Specular reflection off the triangle with this index.
    Reflect(usize),
    /// Diffraction over the edge with this index into the resolved edge list.
    Diffract(usize),
}

/// Resolves the arbitrary-order interleaved reflection-and-diffraction arrivals
/// of `emitter` at `listener`.
///
/// Returns the audible mixed sequences (three or more interactions, at least
/// one reflection and at least one diffraction) sorted loudest-first, at most
/// [`GeometricConfig::max_coupled_paths`]. A disabled coupled stage, a disabled
/// reflection or diffraction stage, a zero coupled budget, an empty scene, or a
/// coupled order below `3` yields an empty list: order-2 coupling is owned by
/// [`crate::coupled_path`], and a mixed sequence needs both mechanisms enabled.
#[must_use]
pub fn resolve_coupled_sequences(
    scene: &AcousticScene,
    listener: &Listener,
    emitter: &Emitter,
    config: &GeometricConfig,
    base_distance: Sample,
) -> Vec<PropagationPath> {
    let max_order = config.max_coupled_order.min(MAX_SUPPORTED_COUPLED_ORDER);
    if !config.coupled_enabled
        || !config.reflections_enabled
        || !config.diffraction_enabled
        || config.max_coupled_paths == 0
        || scene.is_empty()
        || max_order < 3
    {
        return Vec::new();
    }

    let edges = diffracting_edges(scene);
    if edges.is_empty() || scene.triangle_count() == 0 {
        return Vec::new();
    }

    let faces: Vec<Face> = (0..scene.triangle_count())
        .filter_map(|triangle| {
            let [a, b, c] = scene.triangle(triangle)?;
            let normal = scene.triangle_normal(triangle)?;
            Some(Face {
                a,
                b,
                c,
                normal,
                bands: scene.material(triangle).specular_reflection(),
            })
        })
        .collect();
    if faces.is_empty() {
        return Vec::new();
    }

    let solver = SequenceSolver {
        scene,
        faces: &faces,
        edges: &edges,
        listener,
        emitter,
        config,
        base_distance,
        eps: config.surface_epsilon_m.max(0.0),
        max_order,
    };
    solver.run()
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

/// Immutable context shared across the recursive sequence enumeration, bundled
/// into one struct so the per-sequence routines stay well under the argument
/// ceiling.
struct SequenceSolver<'a> {
    scene: &'a AcousticScene,
    faces: &'a [Face],
    edges: &'a [EdgeData],
    listener: &'a Listener,
    emitter: &'a Emitter,
    config: &'a GeometricConfig,
    base_distance: Sample,
    eps: Sample,
    max_order: usize,
}

impl SequenceSolver<'_> {
    /// Enumerates every ordered mixed interaction sequence and returns the
    /// surviving arrivals, sorted loudest-first and capped to the coupled
    /// budget.
    #[must_use]
    fn run(&self) -> Vec<PropagationPath> {
        let mut paths = Vec::new();
        let mut sequence = Vec::new();
        let mut budget = MAX_SEQUENCE_EVALUATIONS;
        self.extend(&mut sequence, &mut paths, &mut budget);
        paths.sort_by(|lhs, rhs| rhs.gain.partial_cmp(&lhs.gain).unwrap_or(Ordering::Equal));
        paths.truncate(self.config.max_coupled_paths);
        paths
    }

    /// Depth-first growth of the current interaction `sequence`: evaluates it
    /// once it is a mixed order-3-or-more chain, then recurses with each legal
    /// next interaction until the configured order or the evaluation budget is
    /// exhausted. Faces and edges are offered in deterministic index order; no
    /// two adjacent interactions may be identical and no edge may be reused.
    fn extend(
        &self,
        sequence: &mut Vec<Interaction>,
        paths: &mut Vec<PropagationPath>,
        budget: &mut usize,
    ) {
        if sequence.len() >= 3 && is_mixed(sequence) {
            if *budget == 0 {
                return;
            }
            *budget -= 1;
            if let Some(path) = self.evaluate(sequence)
                && !is_duplicate(paths, &path)
            {
                paths.push(path);
            }
        }
        if sequence.len() >= self.max_order || *budget == 0 {
            return;
        }
        for face in 0..self.faces.len() {
            let cand = Interaction::Reflect(face);
            if self.may_push(sequence, cand) {
                sequence.push(cand);
                self.extend(sequence, paths, budget);
                sequence.pop();
                if *budget == 0 {
                    return;
                }
            }
        }
        for edge in 0..self.edges.len() {
            let cand = Interaction::Diffract(edge);
            if self.may_push(sequence, cand) {
                sequence.push(cand);
                self.extend(sequence, paths, budget);
                sequence.pop();
                if *budget == 0 {
                    return;
                }
            }
        }
    }

    /// Whether `cand` may legally follow `sequence`: never immediately after an
    /// identical interaction (a degenerate self-interaction), and an edge never
    /// reused anywhere along the route (a repeated edge re-folds an existing
    /// corner). Faces may recur non-adjacently, as a wave genuinely can glance
    /// off the same wall twice between other interactions.
    #[must_use]
    fn may_push(&self, sequence: &[Interaction], cand: Interaction) -> bool {
        if sequence.last() == Some(&cand) {
            return false;
        }
        if let Interaction::Diffract(edge) = cand
            && sequence.contains(&Interaction::Diffract(edge))
        {
            return false;
        }
        true
    }

    /// Relaxes the taut route for one `sequence`, validates it, and shades the
    /// surviving arrival. Returns [`None`] when the route is obstructed, not
    /// longer than the straight line, degenerate at any interaction, or
    /// inaudible.
    #[must_use]
    fn evaluate(&self, sequence: &[Interaction]) -> Option<PropagationPath> {
        let mut points = self.relax(sequence);

        // Snap every reflection point to its exact specular crossing from the
        // final neighbours and reject the sequence if any reflection is not a
        // genuine in-triangle bounce.
        for slot in 0..points.len() {
            if let Interaction::Reflect(face) = sequence[slot] {
                let (prev, next) = neighbours(self.emitter, self.listener, &points, slot);
                let point = specular_point(&self.faces[face], prev, next)?;
                points[slot] = point;
            }
        }

        // Every diffraction corner must bend by a genuine detour, else the route
        // re-derives one an order lower.
        for (slot, interaction) in sequence.iter().enumerate() {
            if let Interaction::Diffract(_) = interaction {
                let (prev, next) = neighbours(self.emitter, self.listener, &points, slot);
                if edge_path_difference(prev, points[slot], next) < MIN_EDGE_DETOUR_M {
                    return None;
                }
            }
        }

        if !self.route_is_clear(&points) {
            return None;
        }

        let total_length = self.route_length(&points);
        if total_length <= self.base_distance {
            return None;
        }

        let spreading = (self.base_distance / total_length).clamp(0.0, 1.0);
        let (gain, bands, cutoff_hz) = self.shade(sequence, &points, spreading);
        if gain <= self.config.min_gain {
            return None;
        }

        let last = *points.last()?;
        let local = self.listener.localize(&Emitter::point(last, Vec3::ZERO));
        Some(PropagationPath {
            kind: PathKind::Diffraction,
            delay_seconds: total_length / SPEED_OF_SOUND_MPS,
            gain,
            cutoff_hz,
            bands,
            direction: local.direction,
        })
    }

    /// Gauss-Seidel relaxation of each interaction point toward the taut route,
    /// starting reflections from the triangle centroid and diffractions from the
    /// edge midpoint.
    #[must_use]
    fn relax(&self, sequence: &[Interaction]) -> Vec<Vec3> {
        let mut points: Vec<Vec3> = sequence
            .iter()
            .map(|interaction| match *interaction {
                Interaction::Reflect(face) => {
                    let f = &self.faces[face];
                    (f.a + f.b + f.c) / 3.0
                }
                Interaction::Diffract(edge) => {
                    let e = &self.edges[edge];
                    (e.start + e.end) * 0.5
                }
            })
            .collect();
        for _ in 0..RELAXATION_SWEEPS {
            for slot in 0..points.len() {
                let (prev, next) = neighbours(self.emitter, self.listener, &points, slot);
                points[slot] = match sequence[slot] {
                    Interaction::Reflect(face) => {
                        specular_point(&self.faces[face], prev, next).unwrap_or(points[slot])
                    }
                    Interaction::Diffract(edge) => {
                        let e = &self.edges[edge];
                        least_detour_point(e.start, e.end, prev, next)
                    }
                };
            }
        }
        points
    }

    /// Whether every leg emitter -> p0 -> ... -> listener reaches its endpoint
    /// without being swallowed by another surface.
    #[must_use]
    fn route_is_clear(&self, points: &[Vec3]) -> bool {
        let Some(&first) = points.first() else {
            return false;
        };
        if self
            .scene
            .segment_blocked(self.emitter.position, first, self.eps)
        {
            return false;
        }
        for leg in points.windows(2) {
            if self.scene.segment_blocked(leg[0], leg[1], self.eps) {
                return false;
            }
        }
        let Some(&last) = points.last() else {
            return false;
        };
        !self
            .scene
            .segment_blocked(last, self.listener.position, self.eps)
    }

    /// Total length of the route emitter -> points -> listener.
    #[must_use]
    fn route_length(&self, points: &[Vec3]) -> Sample {
        let Some(&first) = points.first() else {
            return 0.0;
        };
        let mut total = distance(self.emitter.position, first);
        for leg in points.windows(2) {
            total += distance(leg[0], leg[1]);
        }
        if let Some(&last) = points.last() {
            total += distance(last, self.listener.position);
        }
        total
    }

    /// Multiplies the per-interaction reflection and diffraction colours into the
    /// route's scalar gain (including `spreading`), spectral `bands`, and
    /// tightest `cutoff_hz`.
    #[must_use]
    fn shade(
        &self,
        sequence: &[Interaction],
        points: &[Vec3],
        spreading: Sample,
    ) -> (Sample, BandGains, Sample) {
        let mut spectrum = BandGains::UNITY;
        let mut scalar = spreading;
        let mut cutoff_hz = FULL_BAND_CUTOFF_HZ;
        for (slot, interaction) in sequence.iter().enumerate() {
            match *interaction {
                Interaction::Reflect(face) => {
                    spectrum = spectrum.combine(self.faces[face].bands);
                }
                Interaction::Diffract(edge) => {
                    let (prev, next) = neighbours(self.emitter, self.listener, points, slot);
                    let corner = points[slot];
                    let delta = edge_path_difference(prev, corner, next);
                    let edge_cutoff = diffraction_cutoff_hz(delta, self.config.sample_rate);
                    cutoff_hz = cutoff_hz.min(edge_cutoff);
                    match self.config.diffraction_model {
                        DiffractionModel::Maekawa => {
                            scalar *= diffraction_gain(delta, self.config.diffraction_freq_hz);
                            spectrum =
                                spectrum.combine(BandGains::from_lowpass_cutoff(edge_cutoff));
                        }
                        DiffractionModel::Utd => {
                            let wedge = utd_wedge(&self.edges[edge], corner, prev, next);
                            spectrum = spectrum.combine(wedge.band_gains());
                        }
                    }
                }
            }
        }
        let (gain, bands) = spectrum.scaled(scalar).split_peak();
        (gain.clamp(0.0, 1.0), bands, cutoff_hz)
    }
}

/// The points flanking the interaction at `slot` along the route: the emitter
/// before the first interaction and the listener after the last.
#[must_use]
fn neighbours(
    emitter: &Emitter,
    listener: &Listener,
    points: &[Vec3],
    slot: usize,
) -> (Vec3, Vec3) {
    let prev = if slot == 0 {
        emitter.position
    } else {
        points[slot - 1]
    };
    let next = if slot + 1 == points.len() {
        listener.position
    } else {
        points[slot + 1]
    };
    (prev, next)
}

/// The specular reflection point of the ray `prev` -> (face) -> `next`: mirror
/// `next` across the face plane and intersect the straight line from `prev` to
/// that image with the plane (Fermat's equal-angle law). Returns [`None`] when
/// the crossing is parallel, lands outside the segment, or lands outside the
/// triangle - i.e. when the bounce is not physical.
#[must_use]
fn specular_point(face: &Face, prev: Vec3, next: Vec3) -> Option<Vec3> {
    let d_next = (next - face.a).dot(face.normal);
    let next_image = next - 2.0 * d_next * face.normal;
    let dir = next_image - prev;
    let denom = dir.dot(face.normal);
    if denom.abs() <= f32::EPSILON {
        return None;
    }
    let d_prev = (prev - face.a).dot(face.normal);
    let t = -d_prev / denom;
    if !(t > 0.0 && t < 1.0) {
        return None;
    }
    let point = prev + t * dir;
    if !point_in_triangle(point, face.a, face.b, face.c) {
        return None;
    }
    Some(point)
}

/// Whether `sequence` contains at least one reflection and at least one
/// diffraction - the defining property of a route this module owns.
#[must_use]
fn is_mixed(sequence: &[Interaction]) -> bool {
    let has_reflect = sequence
        .iter()
        .any(|i| matches!(i, Interaction::Reflect(_)));
    let has_diffract = sequence
        .iter()
        .any(|i| matches!(i, Interaction::Diffract(_)));
    has_reflect && has_diffract
}

/// Whether `candidate` duplicates an arrival already kept (same delay and
/// arrival direction within a tight tolerance). Different interleavings can
/// relax to the same physical route; this keeps only one.
#[must_use]
fn is_duplicate(paths: &[PropagationPath], candidate: &PropagationPath) -> bool {
    paths.iter().any(|existing| {
        (existing.delay_seconds - candidate.delay_seconds).abs() < 1.0e-6
            && existing.direction.dot(candidate.direction) > 0.9999
    })
}

#[cfg(test)]
mod tests {
    use super::resolve_coupled_sequences;
    use alloc::vec;
    use bevy_math::{Quat, Vec3};
    use prism_audio_spatial::doppler::SPEED_OF_SOUND_MPS;
    use prism_audio_spatial::geometry::{Emitter, Listener};
    use prism_audio_spatial::propagation::{AcousticMaterial, PathKind};

    use crate::config::GeometricConfig;
    use crate::material_map::MaterialTable;
    use crate::scene::AcousticScene;

    // An L-shaped pair of non-coplanar barriers standing on a reflecting floor.
    // The two barriers meet at the vertical line x = 0, z = 0 and rise to a top
    // edge at y = 2; the floor is a y = 0 quad. A source tucked low on one side
    // of the corner reaches a raised listener on the other side only by routes
    // that bend over the two top edges in turn - and one such route first
    // glances off the floor, giving a mixed reflect-then-bend-twice sequence
    // this module owns.
    fn l_corner_on_floor() -> AcousticScene {
        let vertices = vec![
            // Reflecting floor at y = 0.
            Vec3::new(-10.0, 0.0, -10.0),
            Vec3::new(10.0, 0.0, -10.0),
            Vec3::new(10.0, 0.0, 10.0),
            Vec3::new(-10.0, 0.0, 10.0),
            // Barrier A in x = 0, spanning z in [-6, 0], top edge at y = 2.
            Vec3::new(0.0, 0.0, -6.0),
            Vec3::new(0.0, 2.0, -6.0),
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            // Barrier B in z = 0, spanning x in [0, 6], top edge at y = 2.
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(6.0, 2.0, 0.0),
            Vec3::new(6.0, 0.0, 0.0),
        ];
        let indices = vec![
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [8, 9, 10],
            [8, 10, 11],
        ];
        AcousticScene::new(
            vertices,
            indices,
            MaterialTable::uniform_scalar(AcousticMaterial::new(60.0, 0.9)),
        )
        .unwrap()
    }

    fn endpoints() -> (Listener, Emitter) {
        let listener = Listener::new(Vec3::new(-3.0, 1.5, -3.0), Quat::IDENTITY, Vec3::ZERO);
        let emitter = Emitter::point(Vec3::new(3.0, 0.5, 3.0), Vec3::ZERO);
        (listener, emitter)
    }

    fn base_distance(listener: &Listener, emitter: &Emitter) -> f32 {
        (emitter.position - listener.position).length()
    }

    #[test]
    fn mixed_sequence_arrival_is_found() {
        let scene = l_corner_on_floor();
        let (listener, emitter) = endpoints();
        let base = base_distance(&listener, &emitter);
        let cfg = GeometricConfig::new(48_000)
            .with_coupled_paths()
            .with_max_coupled_order(3);
        let paths = resolve_coupled_sequences(&scene, &listener, &emitter, &cfg, base);

        assert!(
            !paths.is_empty(),
            "a floor glance that bends over both corner edges should produce a mixed arrival"
        );
        for path in &paths {
            assert_eq!(path.kind, PathKind::Diffraction);
            // The reflect-and-bend-twice route is strictly longer than the
            // straight line, so it arrives later than a direct wave would.
            assert!(path.delay_seconds > base / SPEED_OF_SOUND_MPS);
            assert!(path.gain > 0.0 && path.gain <= 1.0);
            for band in path.bands.bands() {
                assert!(band.is_finite() && (0.0..=1.0).contains(&band));
            }
        }
    }

    #[test]
    fn mixed_arrivals_are_sorted_loudest_first() {
        let scene = l_corner_on_floor();
        let (listener, emitter) = endpoints();
        let base = base_distance(&listener, &emitter);
        let cfg = GeometricConfig::new(48_000)
            .with_coupled_paths()
            .with_max_coupled_order(4);
        let paths = resolve_coupled_sequences(&scene, &listener, &emitter, &cfg, base);
        for pair in paths.windows(2) {
            assert!(pair[0].gain >= pair[1].gain);
        }
    }

    #[test]
    fn default_order_emits_no_sequence() {
        let scene = l_corner_on_floor();
        let (listener, emitter) = endpoints();
        let base = base_distance(&listener, &emitter);
        // Order-2 coupling is owned by `coupled_path`; the default coupled order
        // is 2, so this module stays silent even with coupling on.
        let cfg = GeometricConfig::new(48_000).with_coupled_paths();
        assert!(resolve_coupled_sequences(&scene, &listener, &emitter, &cfg, base).is_empty());
    }

    #[test]
    fn disabled_coupling_yields_nothing() {
        let scene = l_corner_on_floor();
        let (listener, emitter) = endpoints();
        let base = base_distance(&listener, &emitter);
        // Raising the order enables coupling; turning it back off must silence
        // the module even at order 3.
        let cfg = GeometricConfig::new(48_000)
            .with_max_coupled_order(3)
            .without_coupled_paths();
        assert!(resolve_coupled_sequences(&scene, &listener, &emitter, &cfg, base).is_empty());
    }

    #[test]
    fn disabled_reflections_yield_nothing() {
        let scene = l_corner_on_floor();
        let (listener, emitter) = endpoints();
        let base = base_distance(&listener, &emitter);
        // A mixed sequence needs the reflection mechanism.
        let cfg = GeometricConfig::new(48_000)
            .with_max_coupled_order(3)
            .without_reflections();
        assert!(resolve_coupled_sequences(&scene, &listener, &emitter, &cfg, base).is_empty());
    }

    #[test]
    fn disabled_diffraction_yields_nothing() {
        let scene = l_corner_on_floor();
        let (listener, emitter) = endpoints();
        let base = base_distance(&listener, &emitter);
        // Likewise it needs the diffraction mechanism.
        let cfg = GeometricConfig::new(48_000)
            .with_max_coupled_order(3)
            .without_diffraction();
        assert!(resolve_coupled_sequences(&scene, &listener, &emitter, &cfg, base).is_empty());
    }

    #[test]
    fn zero_budget_yields_nothing() {
        let scene = l_corner_on_floor();
        let (listener, emitter) = endpoints();
        let base = base_distance(&listener, &emitter);
        let cfg = GeometricConfig::new(48_000)
            .with_max_coupled_order(3)
            .with_max_coupled_paths(0);
        assert!(resolve_coupled_sequences(&scene, &listener, &emitter, &cfg, base).is_empty());
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
        let cfg = GeometricConfig::new(48_000)
            .with_coupled_paths()
            .with_max_coupled_order(3);
        assert!(resolve_coupled_sequences(&scene, &listener, &emitter, &cfg, base).is_empty());
    }
}
