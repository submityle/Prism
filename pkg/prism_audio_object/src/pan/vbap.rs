//! Vector Base Amplitude Panning (VBAP) over an arbitrary 3D speaker set.
//!
//! Given a source unit direction and a set of loudspeaker unit vectors, VBAP
//! (Pulkki 1997) distributes the source onto the two or three speakers that
//! bracket its direction so that the gain-weighted sum of the speaker vectors
//! points back along the source direction. This module implements:
//!
//! * **Triplet VBAP** for three or more speakers: for every speaker triplet it
//!   solves `dir = g0*s0 + g1*s1 + g2*s2` with the scalar-triple-product
//!   (Cramer) rule. A triplet is *valid* when all three gains are
//!   non-negative, meaning the direction lies inside the spherical triangle of
//!   those speakers. Among valid triplets the one whose speakers hug the
//!   direction most tightly (largest summed cosine) is chosen, so panning
//!   stays local rather than smearing across a large triangle.
//! * **Pairwise VBAP** for exactly two speakers: it solves the 2x2 Gram system
//!   in the plane the pair spans.
//! * **Single speaker**: unit gain.
//! * A **nearest-speaker fallback** for directions outside every triangle
//!   (for example straight down when no speaker is below the horizon).
//!
//! All returned gain vectors are renormalised to **constant power** (the sum of
//! squared gains is one when any speaker is driven), so panning a source around
//! the array keeps its loudness steady.
//!
//! # Determinism
//!
//! Every length/normalisation step routes through [`bevy_math::ops`]; triplet
//! selection breaks ties by index. The result is bit-reproducible across
//! targets.
//!
//! # Provenance
//! Original work implementing the publicly documented VBAP method: Ville
//! Pulkki, "Virtual Sound Source Positioning Using Vector Base Amplitude
//! Panning", J. Audio Eng. Soc. 45(6), 1997. No Unreal Engine, Unity, Godot,
//! Wwise, FMOD, Steam Audio, Dolby, or Google Resonance Audio source or
//! derived code; no AI/ML.
//!
//! # Relationship
//! The object-to-bed panning core of design section 44.2. Consumed by
//! [`crate::pan`] via [`crate::pan::channel_layout::SpeakerArray`].

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use bevy_math::{ops, Vec3};

use prism_audio_core::math::Sample;

/// Minimum magnitude of a triplet determinant (scalar triple product) for the
/// triplet to be treated as linearly independent.
const DET_EPSILON: Sample = 1e-6;

/// Small negative tolerance when deciding whether a solved gain counts as
/// non-negative (absorbs rounding at a triangle edge).
const GAIN_EPSILON: Sample = -1e-5;

/// Renormalises `gains` so the sum of squares is one (constant power). A
/// nearly all-zero vector is left untouched.
fn normalize_constant_power(gains: &mut [Sample]) {
    let mut energy: Sample = 0.0;
    for &g in gains.iter() {
        energy += g * g;
    }
    if energy <= 1e-20 {
        return;
    }
    let inv = 1.0 / ops::sqrt(energy);
    for g in gains.iter_mut() {
        *g *= inv;
    }
}

/// Returns the index of the speaker whose direction is closest (largest dot
/// product) to `direction`, or `None` when `speakers` is empty.
#[must_use]
pub fn nearest_speaker(direction: Vec3, speakers: &[Vec3]) -> Option<usize> {
    let mut best = None;
    let mut best_dot = Sample::NEG_INFINITY;
    for (i, &s) in speakers.iter().enumerate() {
        let d = s.dot(direction);
        if d > best_dot {
            best_dot = d;
            best = Some(i);
        }
    }
    best
}

/// Solves the two-speaker Gram system for gains, clamped to non-negative.
fn pairwise_gains(direction: Vec3, s0: Vec3, s1: Vec3) -> (Sample, Sample) {
    let a = s0.dot(s0);
    let b = s0.dot(s1);
    let c = s1.dot(s1);
    let d0 = s0.dot(direction);
    let d1 = s1.dot(direction);
    let det = a * c - b * b;
    if ops::abs(det) < DET_EPSILON {
        // Collinear speakers: fall back to the nearer one.
        return if d0 >= d1 { (1.0, 0.0) } else { (0.0, 1.0) };
    }
    let g0 = (d0 * c - d1 * b) / det;
    let g1 = (a * d1 - b * d0) / det;
    (g0.max(0.0), g1.max(0.0))
}

/// Computes VBAP gains for `direction` over `speakers`, returning one gain per
/// speaker (constant-power normalised).
///
/// The returned vector always has length `speakers.len()`. An empty speaker
/// set yields an empty vector; a zero-length `direction` collapses to the
/// nearest-speaker (first) fallback.
#[must_use]
pub fn vbap_gains(direction: Vec3, speakers: &[Vec3]) -> Vec<Sample> {
    let mut gains = Vec::new();
    gains.resize(speakers.len(), 0.0);
    if speakers.is_empty() {
        return gains;
    }

    let dir = direction.normalize_or_zero();
    if dir == Vec3::ZERO {
        // No bearing: put everything on the first speaker.
        gains[0] = 1.0;
        return gains;
    }

    match speakers.len() {
        1 => {
            gains[0] = 1.0;
            return gains;
        }
        2 => {
            let (g0, g1) = pairwise_gains(dir, speakers[0], speakers[1]);
            gains[0] = g0;
            gains[1] = g1;
            normalize_constant_power(&mut gains);
            return gains;
        }
        _ => {}
    }

    // Triplet search: pick the valid triplet whose speakers hug `dir` most
    // tightly (largest summed cosine).
    let mut best: Option<(usize, usize, usize, Sample, Sample, Sample)> = None;
    let mut best_score = Sample::NEG_INFINITY;
    for i in 0..speakers.len() {
        for j in (i + 1)..speakers.len() {
            for k in (j + 1)..speakers.len() {
                let s0 = speakers[i];
                let s1 = speakers[j];
                let s2 = speakers[k];
                let det = s0.dot(s1.cross(s2));
                if ops::abs(det) < DET_EPSILON {
                    continue;
                }
                let inv_det = 1.0 / det;
                let g0 = dir.dot(s1.cross(s2)) * inv_det;
                let g1 = s0.dot(dir.cross(s2)) * inv_det;
                let g2 = s0.dot(s1.cross(dir)) * inv_det;
                if g0 < GAIN_EPSILON || g1 < GAIN_EPSILON || g2 < GAIN_EPSILON {
                    continue;
                }
                let score = s0.dot(dir) + s1.dot(dir) + s2.dot(dir);
                if score > best_score {
                    best_score = score;
                    best = Some((i, j, k, g0.max(0.0), g1.max(0.0), g2.max(0.0)));
                }
            }
        }
    }

    if let Some((i, j, k, g0, g1, g2)) = best {
        gains[i] = g0;
        gains[j] = g1;
        gains[k] = g2;
        normalize_constant_power(&mut gains);
        return gains;
    }

    // Outside every triangle: nearest-speaker fallback.
    if let Some(idx) = nearest_speaker(dir, speakers) {
        gains[idx] = 1.0;
    }
    gains
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bed::direction_from_angles;

    const EPS: Sample = 1e-4;

    fn close(a: Sample, b: Sample) -> bool {
        ops::abs(a - b) <= EPS
    }

    fn sum_sq(g: &[Sample]) -> Sample {
        g.iter().map(|&x| x * x).sum()
    }

    fn cube_speakers() -> Vec<Vec3> {
        // Eight speakers at +/-45 azimuth corners, horizon and overhead.
        vec![
            direction_from_angles(-45.0, 0.0),
            direction_from_angles(45.0, 0.0),
            direction_from_angles(-135.0, 0.0),
            direction_from_angles(135.0, 0.0),
            direction_from_angles(-45.0, 45.0),
            direction_from_angles(45.0, 45.0),
            direction_from_angles(-135.0, 45.0),
            direction_from_angles(135.0, 45.0),
        ]
    }

    #[test]
    fn gains_are_constant_power() {
        let speakers = cube_speakers();
        let dir = direction_from_angles(10.0, 10.0);
        let g = vbap_gains(dir, &speakers);
        assert!(close(sum_sq(&g), 1.0));
        assert!(g.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn direction_at_speaker_drives_mostly_that_speaker() {
        let speakers = cube_speakers();
        let g = vbap_gains(speakers[1], &speakers);
        assert!(close(sum_sq(&g), 1.0));
        // The coincident speaker should dominate (near unit, others near zero).
        assert!(g[1] > 0.99);
        for (i, &gi) in g.iter().enumerate() {
            if i != 1 {
                assert!(gi < 1e-2);
            }
        }
    }

    #[test]
    fn pairwise_two_speakers_constant_power() {
        let speakers = vec![
            direction_from_angles(-30.0, 0.0),
            direction_from_angles(30.0, 0.0),
        ];
        let center = direction_from_angles(0.0, 0.0);
        let g = vbap_gains(center, &speakers);
        assert!(close(sum_sq(&g), 1.0));
        // Symmetric placement => equal split.
        assert!(close(g[0], g[1]));
    }

    #[test]
    fn pairwise_at_speaker_is_unit() {
        let speakers = vec![
            direction_from_angles(-30.0, 0.0),
            direction_from_angles(30.0, 0.0),
        ];
        let g = vbap_gains(speakers[0], &speakers);
        assert!(close(g[0], 1.0));
        assert!(close(g[1], 0.0));
    }

    #[test]
    fn single_speaker_gets_unit_gain() {
        let speakers = vec![direction_from_angles(0.0, 0.0)];
        let g = vbap_gains(direction_from_angles(90.0, 0.0), &speakers);
        assert!(close(g[0], 1.0));
    }

    #[test]
    fn below_array_falls_back_to_nearest() {
        let speakers = cube_speakers();
        // Straight down: no speaker below the horizon, so no triangle contains
        // it; nearest-speaker fallback should drive exactly one speaker.
        let down = Vec3::new(0.0, -1.0, 0.0);
        let g = vbap_gains(down, &speakers);
        let driven = g.iter().filter(|&&x| x > 0.0).count();
        assert_eq!(driven, 1);
        assert!(close(sum_sq(&g), 1.0));
    }

    #[test]
    fn empty_speakers_yields_empty() {
        let g = vbap_gains(Vec3::NEG_Z, &[]);
        assert!(g.is_empty());
    }

    #[test]
    fn zero_direction_uses_first_speaker() {
        let speakers = cube_speakers();
        let g = vbap_gains(Vec3::ZERO, &speakers);
        assert!(close(g[0], 1.0));
    }
}
