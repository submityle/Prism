//! Multi-edge (higher-order) diffraction resolved by taut-path relaxation.
//!
//! The single-edge resolver in [`crate::diffraction_path`] bends a wave over
//! exactly one edge. Real rooms routinely hide a source behind two or more
//! successive corners (an L-shaped corridor, a doorway seen from across a bent
//! hallway, the far lip of a baffle maze), where the only audible route bends
//! over a *sequence* of edges in turn. This module traces those routes, aligning
//! with the sequential edge-diffraction paths that Steam Audio's path tracer
//! emits for occluded sources.
//!
//! # Method
//!
//! For every ordered sequence of distinct diffracting edges (length `2` up to
//! the configured order), the taut bent route emitter -> c0 -> c1 -> ... ->
//! listener is found by Gauss-Seidel relaxation: each corner is repeatedly
//! re-placed at the least-detour point on its edge given its current
//! neighbours, which converges to the string-pulled-taut path. A sequence is
//! kept only when every leg is unobstructed, the whole route is longer than the
//! straight line, and every edge contributes a genuine bend (so a route that
//! merely re-derives a lower-order path is rejected rather than double-counted).
//!
//! # Gain convention
//!
//! Each edge attenuates independently, so the per-edge shadow gains multiply
//! (Maekawa barrier gains, or UTD wedge coefficients, one factor per corner),
//! scaled by the extra spherical spreading of the longer route
//! (`base_distance / path_length`) exactly as the single-edge resolver and
//! [`crate::direct_path`] document. The spectral `bands` are the product of each
//! edge's low-pass colour, and `cutoff_hz` is the tightest (lowest) corner along
//! the route. The result is clamped to `[0, 1]`; arrivals below the floor drop.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//!
//! Shares the diffracting-edge primitives in [`crate::diffraction_edges`] with
//! the single-edge [`crate::diffraction_path`], reads a
//! [`crate::scene::AcousticScene`], and emits
//! [`PropagationPath`](prism_audio_spatial::propagation::PropagationPath)s of
//! kind [`Diffraction`](prism_audio_spatial::propagation::PathKind::Diffraction)
//! that [`crate::backend::GeometricBackend`] merges with the direct, reflected,
//! and single-edge diffracted arrivals.

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

use crate::config::{DiffractionModel, GeometricConfig, MAX_SUPPORTED_DIFFRACTION_ORDER};
use crate::diffraction_edges::{diffracting_edges, distance, least_detour_point, utd_wedge, EdgeData};
use crate::scene::AcousticScene;

/// Fixed number of Gauss-Seidel sweeps used to pull a multi-edge route taut.
/// The per-leg detour is convex and the sweep is contractive for the sparse
/// corner chains that survive, so a fixed count converges deterministically
/// without a tolerance loop that could vary across targets.
const RELAXATION_SWEEPS: u32 = 24;

/// Upper bound on the number of edge sequences evaluated per query. The number
/// of ordered distinct-edge sequences grows factorially with the edge count, so
/// this bound keeps a dense mesh from stalling the control-rate query; sequences
/// are explored in deterministic edge-index order, so the retained set is stable.
const MAX_DIFFRACTION_SEQUENCE_EVALUATIONS: usize = 4096;

/// Minimum per-edge detour (metres) for an edge to count as a genuine bend. A
/// route whose edge adds less than this merely re-derives a lower-order path and
/// is rejected so the orders never double-count the same arrival.
const MIN_EDGE_DETOUR_M: Sample = 1.0e-3;

/// Resolves the multi-edge diffracted arrivals of `emitter` at `listener`.
///
/// Returns the audible higher-order (two-or-more-edge) diffractions sorted
/// loudest-first, at most [`GeometricConfig::max_diffractions`]. Single-edge
/// bends are owned by [`crate::diffraction_path`] and never produced here. An
/// empty scene, a disabled diffraction stage, a zero diffraction budget, or an
/// effective order below `2` yields an empty list.
#[must_use]
pub fn resolve_higher_order_diffraction(
    scene: &AcousticScene,
    listener: &Listener,
    emitter: &Emitter,
    config: &GeometricConfig,
    base_distance: Sample,
) -> Vec<PropagationPath> {
    let max_order = config
        .max_diffraction_order
        .min(MAX_SUPPORTED_DIFFRACTION_ORDER);
    if !config.diffraction_enabled || config.max_diffractions == 0 || scene.is_empty() || max_order < 2
    {
        return Vec::new();
    }

    let edges = diffracting_edges(scene);
    if edges.len() < 2 {
        return Vec::new();
    }

    let solver = SequenceSolver {
        scene,
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

/// Immutable context shared across the recursive sequence enumeration, bundled
/// into one struct so the per-sequence routines stay well under the argument
/// ceiling.
struct SequenceSolver<'a> {
    scene: &'a AcousticScene,
    edges: &'a [EdgeData],
    listener: &'a Listener,
    emitter: &'a Emitter,
    config: &'a GeometricConfig,
    base_distance: Sample,
    eps: Sample,
    max_order: usize,
}

impl SequenceSolver<'_> {
    /// Enumerates every ordered distinct-edge sequence and returns the surviving
    /// arrivals, sorted loudest-first and capped to the diffraction budget.
    #[must_use]
    fn run(&self) -> Vec<PropagationPath> {
        let mut paths = Vec::new();
        let mut sequence = Vec::new();
        let mut budget = MAX_DIFFRACTION_SEQUENCE_EVALUATIONS;
        self.extend(&mut sequence, &mut paths, &mut budget);
        paths.sort_by(|lhs, rhs| rhs.gain.partial_cmp(&lhs.gain).unwrap_or(Ordering::Equal));
        paths.truncate(self.config.max_diffractions);
        paths
    }

    /// Depth-first growth of the current edge `sequence`: evaluates it once it
    /// reaches length two, then recurses with each not-yet-used edge until the
    /// configured order or the evaluation budget is exhausted.
    fn extend(&self, sequence: &mut Vec<usize>, paths: &mut Vec<PropagationPath>, budget: &mut usize) {
        if sequence.len() >= 2 {
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
        for next in 0..self.edges.len() {
            if sequence.contains(&next) {
                continue;
            }
            sequence.push(next);
            self.extend(sequence, paths, budget);
            sequence.pop();
            if *budget == 0 {
                return;
            }
        }
    }

    /// Relaxes the taut route for one edge `sequence`, validates it, and shades
    /// the surviving arrival. Returns [`None`] when the route is obstructed,
    /// not longer than the straight line, degenerate at any edge, or inaudible.
    #[must_use]
    fn evaluate(&self, sequence: &[usize]) -> Option<PropagationPath> {
        let corners = self.relax(sequence);
        if !self.route_is_clear(&corners) {
            return None;
        }

        let total_length = self.route_length(&corners);
        if total_length <= self.base_distance {
            return None;
        }

        // Every edge must bend genuinely, otherwise this route re-derives a
        // lower-order path and must not be counted again here.
        for (slot, &corner) in corners.iter().enumerate() {
            let (prev, next) = self.neighbours(&corners, slot);
            if edge_path_difference(prev, corner, next) < MIN_EDGE_DETOUR_M {
                return None;
            }
        }

        let spreading = (self.base_distance / total_length).clamp(0.0, 1.0);
        let (gain, bands, cutoff_hz) = self.shade(sequence, &corners, spreading);
        if gain <= self.config.min_gain {
            return None;
        }

        let last = *corners.last()?;
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

    /// Gauss-Seidel relaxation of the corner on each edge toward the taut route,
    /// starting from the edge midpoints.
    #[must_use]
    fn relax(&self, sequence: &[usize]) -> Vec<Vec3> {
        let mut corners: Vec<Vec3> = sequence
            .iter()
            .map(|&edge| {
                let data = &self.edges[edge];
                (data.start + data.end) * 0.5
            })
            .collect();
        for _ in 0..RELAXATION_SWEEPS {
            for slot in 0..corners.len() {
                let (prev, next) = self.neighbours(&corners, slot);
                let data = &self.edges[sequence[slot]];
                corners[slot] = least_detour_point(data.start, data.end, prev, next);
            }
        }
        corners
    }

    /// The points flanking corner `slot` along the route: the emitter before the
    /// first corner and the listener after the last.
    #[must_use]
    fn neighbours(&self, corners: &[Vec3], slot: usize) -> (Vec3, Vec3) {
        let prev = if slot == 0 {
            self.emitter.position
        } else {
            corners[slot - 1]
        };
        let next = if slot + 1 == corners.len() {
            self.listener.position
        } else {
            corners[slot + 1]
        };
        (prev, next)
    }

    /// Whether every leg emitter -> c0 -> ... -> listener reaches its endpoint
    /// without being swallowed by another surface.
    #[must_use]
    fn route_is_clear(&self, corners: &[Vec3]) -> bool {
        let Some(&first) = corners.first() else {
            return false;
        };
        if self.scene.segment_blocked(self.emitter.position, first, self.eps) {
            return false;
        }
        for leg in corners.windows(2) {
            if self.scene.segment_blocked(leg[0], leg[1], self.eps) {
                return false;
            }
        }
        let Some(&last) = corners.last() else {
            return false;
        };
        !self
            .scene
            .segment_blocked(last, self.listener.position, self.eps)
    }

    /// Total length of the bent route emitter -> corners -> listener.
    #[must_use]
    fn route_length(&self, corners: &[Vec3]) -> Sample {
        let Some(&first) = corners.first() else {
            return 0.0;
        };
        let mut total = distance(self.emitter.position, first);
        for leg in corners.windows(2) {
            total += distance(leg[0], leg[1]);
        }
        if let Some(&last) = corners.last() {
            total += distance(last, self.listener.position);
        }
        total
    }

    /// Multiplies the per-edge shadow gains and colours into the route's scalar
    /// gain (including `spreading`), spectral `bands`, and tightest `cutoff_hz`.
    #[must_use]
    fn shade(&self, sequence: &[usize], corners: &[Vec3], spreading: Sample) -> (Sample, BandGains, Sample) {
        let mut gain = spreading;
        let mut bands = BandGains::UNITY;
        let mut cutoff_hz = FULL_BAND_CUTOFF_HZ;
        for (slot, &corner) in corners.iter().enumerate() {
            let (prev, next) = self.neighbours(corners, slot);
            let delta = edge_path_difference(prev, corner, next);
            let edge_cutoff = diffraction_cutoff_hz(delta, self.config.sample_rate);
            cutoff_hz = cutoff_hz.min(edge_cutoff);
            match self.config.diffraction_model {
                DiffractionModel::Maekawa => {
                    gain *= diffraction_gain(delta, self.config.diffraction_freq_hz);
                    bands = bands.combine(BandGains::from_lowpass_cutoff(edge_cutoff));
                }
                DiffractionModel::Utd => {
                    let wedge = utd_wedge(&self.edges[sequence[slot]], corner, prev, next);
                    gain *= wedge.relative_gain(self.config.diffraction_freq_hz);
                    bands = bands.combine(wedge.band_gains());
                }
            }
        }
        (gain.clamp(0.0, 1.0), bands, cutoff_hz)
    }
}

/// Whether `candidate` duplicates an arrival already kept (same delay and
/// direction within a tight tolerance).
#[must_use]
fn is_duplicate(paths: &[PropagationPath], candidate: &PropagationPath) -> bool {
    paths.iter().any(|existing| {
        (existing.delay_seconds - candidate.delay_seconds).abs() < 1.0e-6
            && existing.direction.dot(candidate.direction) > 0.9999
    })
}

#[cfg(test)]
mod tests {
    use super::resolve_higher_order_diffraction;
    use alloc::vec;
    use bevy_math::{Quat, Vec3};
    use prism_audio_spatial::geometry::{Emitter, Listener};
    use prism_audio_spatial::propagation::{AcousticMaterial, PathKind};

    use crate::config::GeometricConfig;
    use crate::material_map::MaterialTable;
    use crate::scene::AcousticScene;

    // An L-shaped pair of baffles that force a two-edge bend. One barrier lies
    // in the plane x = 0 (spanning y in [-5, 1], z in [-5, 0]) with its free
    // top edge at y = 1; a second barrier lies in the plane z = 0 (spanning
    // y in [-5, 1], x in [0, 5]) with its free top edge also at y = 1. Together
    // they wrap a right-angle corner so a source tucked behind both can only be
    // heard by bending over both top edges in turn.
    fn l_corner() -> AcousticScene {
        let vertices = vec![
            // Barrier A in x = 0.
            Vec3::new(0.0, -5.0, -5.0),
            Vec3::new(0.0, 1.0, -5.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, -5.0, 0.0),
            // Barrier B in z = 0.
            Vec3::new(0.0, -5.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(5.0, 1.0, 0.0),
            Vec3::new(5.0, -5.0, 0.0),
        ];
        let indices = vec![[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]];
        AcousticScene::new(
            vertices,
            indices,
            MaterialTable::uniform_scalar(AcousticMaterial::new(60.0, 0.0)),
        )
        .unwrap()
    }

    // A single finite barrier offers only one diffracting silhouette, so no
    // multi-edge route can form.
    fn single_barrier() -> AcousticScene {
        let vertices = vec![
            Vec3::new(0.0, -1.0, -5.0),
            Vec3::new(0.0, 1.0, -5.0),
            Vec3::new(0.0, 1.0, 5.0),
            Vec3::new(0.0, -1.0, 5.0),
        ];
        let indices = vec![[0, 1, 2], [0, 2, 3]];
        AcousticScene::new(
            vertices,
            indices,
            MaterialTable::uniform_scalar(AcousticMaterial::new(60.0, 0.0)),
        )
        .unwrap()
    }

    #[test]
    fn second_order_bend_wraps_the_corner() {
        let scene = l_corner();
        // Listener high above barrier A's side; emitter low behind barrier B's
        // side, so the only clear route bends over both top edges.
        let listener = Listener::new(Vec3::new(-3.0, 3.0, -3.0), Quat::IDENTITY, Vec3::ZERO);
        let emitter = Emitter::point(Vec3::new(3.0, -3.0, 3.0), Vec3::ZERO);
        let cfg = GeometricConfig::new(48_000).with_max_diffraction_order(2);
        let base = (emitter.position - listener.position).length();
        let paths = resolve_higher_order_diffraction(&scene, &listener, &emitter, &cfg, base);
        assert!(!paths.is_empty());
        assert_eq!(paths[0].kind, PathKind::Diffraction);
        // The two-edge detour is strictly longer than the straight line.
        assert!(paths[0].delay_seconds > base / 343.0);
        assert!(paths[0].gain > 0.0 && paths[0].gain <= 1.0);
        for band in paths[0].bands.bands() {
            assert!(band.is_finite() && (0.0..=1.0).contains(&band));
        }
    }

    #[test]
    fn default_order_emits_no_higher_order() {
        let scene = l_corner();
        let listener = Listener::new(Vec3::new(-3.0, 3.0, -3.0), Quat::IDENTITY, Vec3::ZERO);
        let emitter = Emitter::point(Vec3::new(3.0, -3.0, 3.0), Vec3::ZERO);
        // Default order is 1: single-edge bends belong to `diffraction_path`.
        let cfg = GeometricConfig::new(48_000);
        let base = (emitter.position - listener.position).length();
        assert!(resolve_higher_order_diffraction(&scene, &listener, &emitter, &cfg, base).is_empty());
    }

    #[test]
    fn disabled_stage_yields_nothing() {
        let scene = l_corner();
        let listener = Listener::new(Vec3::new(-3.0, 3.0, -3.0), Quat::IDENTITY, Vec3::ZERO);
        let emitter = Emitter::point(Vec3::new(3.0, -3.0, 3.0), Vec3::ZERO);
        let cfg = GeometricConfig::new(48_000)
            .with_max_diffraction_order(2)
            .without_diffraction();
        let base = (emitter.position - listener.position).length();
        assert!(resolve_higher_order_diffraction(&scene, &listener, &emitter, &cfg, base).is_empty());
    }

    #[test]
    fn single_edge_scene_has_no_higher_order() {
        let scene = single_barrier();
        let listener = Listener::new(Vec3::new(-3.0, 0.0, 0.0), Quat::IDENTITY, Vec3::ZERO);
        let emitter = Emitter::point(Vec3::new(3.0, 0.0, 0.0), Vec3::ZERO);
        let cfg = GeometricConfig::new(48_000).with_max_diffraction_order(3);
        let base = (emitter.position - listener.position).length();
        // The lone quad's four boundary edges are coplanar silhouettes of one
        // screen; no two-edge taut route survives the clear-leg test.
        let paths = resolve_higher_order_diffraction(&scene, &listener, &emitter, &cfg, base);
        assert!(paths.iter().all(|p| p.delay_seconds > base / 343.0));
    }

    #[test]
    fn empty_scene_is_silent() {
        let scene = AcousticScene::new(
            vec![],
            vec![],
            MaterialTable::uniform_scalar(AcousticMaterial::new(60.0, 0.0)),
        )
        .unwrap();
        let listener = Listener::new(Vec3::new(-3.0, 0.0, 0.0), Quat::IDENTITY, Vec3::ZERO);
        let emitter = Emitter::point(Vec3::new(3.0, 0.0, 0.0), Vec3::ZERO);
        let cfg = GeometricConfig::new(48_000).with_max_diffraction_order(3);
        let base = (emitter.position - listener.position).length();
        assert!(resolve_higher_order_diffraction(&scene, &listener, &emitter, &cfg, base).is_empty());
    }
}
