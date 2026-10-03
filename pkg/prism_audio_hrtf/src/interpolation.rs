//! Direction interpolation of HRIRs with inter-aural time-difference (ITD)
//! alignment.
//!
//! A measured [`HrtfDataset`](crate::dataset::HrtfDataset) only samples a
//! finite set of directions. To render a source at an arbitrary
//! azimuth/elevation we interpolate between the nearest measured HRIRs. Doing
//! this naively - summing the raw impulse responses - is wrong: each HRIR
//! carries its own onset delay (the ITD), so a direct weighted sum super-poses
//! two pulses at different times and produces destructive **comb filtering**.
//!
//! This module therefore performs **time-aligned interpolation**:
//!
//! 1. Select the nearest measurements by great-circle angle on the unit
//!    sphere.
//! 2. Weight them with Shepard inverse-distance weighting (an exact-match
//!    collapses to a single weight).
//! 3. For each ear, estimate every selected HRIR's onset delay, shift each one
//!    so its onset sits at time zero, and accumulate the *aligned* weighted
//!    sum. This blends spectral shape without comb filtering.
//! 4. Re-apply a single interpolated onset delay (the weighted mean of the
//!    per-measurement onsets), reconstructing a physically plausible ITD for
//!    the target direction.
//!
//! # Real-time contract
//!
//! [`interpolate`] is **allocation free, lock free, and panic free**: it uses
//! fixed-size stack arrays for the neighbour set and writes into
//! caller-provided output buffers. It may run on the audio thread when a
//! source's direction changes (typically feeding
//! [`crate::binaural::BinauralRenderer::set_hrir`]).
//!
//! # Determinism
//!
//! All angle math routes through [`bevy_math::ops`] (libm-backed) and onset
//! detection is a deterministic threshold scan, so interpolation is
//! bit-reproducible and golden-comparable across targets.
//!
//! # Provenance
//!
//! This module is engine-agnostic and contains **no Unreal Engine, Unity,
//! Godot, Wwise, FMOD, or Steam Audio source or derived code**. Inverse-
//! distance (Shepard) interpolation, onset/group-delay alignment, and the
//! spherical direction geometry are implemented from standard, publicly
//! documented DSP and scattered-data-interpolation knowledge.

use bevy_math::{ops, Vec3};
use prism_audio_core::math::{Sample, MIN_AUDIBLE_GAIN};

use crate::dataset::HrtfDataset;

/// Maximum number of measured HRIRs blended for a single target direction.
///
/// Four neighbours suffice to cover a target inside a quad of a typical
/// azimuth/elevation grid while keeping the blend cheap and the onset spread
/// small.
pub const MAX_NEIGHBORS: usize = 4;

/// Fraction of an HRIR's peak magnitude used as the onset-detection threshold.
const ONSET_FRACTION: Sample = 0.15;

/// Angular distance (radians) below which a measurement is treated as an exact
/// match, collapsing the blend to that single point.
const EXACT_MATCH_ANGLE: Sample = 1.0e-4;

/// Smoothing term added to the angular distance in the Shepard weight to keep
/// weights finite and well conditioned near a measurement.
const SHEPARD_SMOOTHING: Sample = 1.0e-3;

/// Which ear an HRIR belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ear {
    Left,
    Right,
}

/// Summary of an [`interpolate`] call, useful for diagnostics and testing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InterpolationInfo {
    /// Number of measured HRIRs actually blended (`1..=MAX_NEIGHBORS`).
    pub neighbors_used: usize,
    /// The reconstructed left-ear onset delay, in samples.
    pub left_delay: usize,
    /// The reconstructed right-ear onset delay, in samples.
    pub right_delay: usize,
}

/// The listener-local unit direction for an `azimuth`/`elevation` pair
/// (radians), matching the [`prism_audio_spatial`] convention.
///
/// Inverts `azimuth = atan2(x, -z)` and `elevation = atan2(y, hypot(x, z))`:
/// `x = cos(el) sin(az)`, `y = sin(el)`, `z = -cos(el) cos(az)`.
#[must_use]
#[inline]
pub fn direction_from_angles(azimuth: Sample, elevation: Sample) -> Vec3 {
    let (sin_el, cos_el) = ops::sin_cos(elevation);
    let (sin_az, cos_az) = ops::sin_cos(azimuth);
    Vec3::new(cos_el * sin_az, sin_el, -cos_el * cos_az)
}

/// Great-circle angle (radians, `[0, pi]`) between two unit directions.
#[must_use]
#[inline]
pub fn angular_distance(a: Vec3, b: Vec3) -> Sample {
    let dot = a.dot(b).clamp(-1.0, 1.0);
    ops::acos(dot)
}

/// Estimates an HRIR's onset delay (leading group delay) in samples.
///
/// Returns the index of the first sample whose magnitude reaches
/// [`ONSET_FRACTION`] of the response's peak magnitude. A silent response
/// yields `0`. This is the classic threshold-based time-of-arrival estimate
/// used for ITD extraction.
#[must_use]
pub fn estimate_onset_delay(ir: &[Sample]) -> usize {
    let mut peak = 0.0;
    for &s in ir {
        let a = ops::abs(s);
        if a > peak {
            peak = a;
        }
    }
    if peak <= MIN_AUDIBLE_GAIN {
        return 0;
    }
    let threshold = peak * ONSET_FRACTION;
    for (i, &s) in ir.iter().enumerate() {
        if ops::abs(s) >= threshold {
            return i;
        }
    }
    0
}

/// Interpolates the dataset's HRIRs for the target `azimuth`/`elevation`
/// (radians) into `out_left` and `out_right`, with ITD alignment.
///
/// Only the azimuth/elevation direction is interpolated here (single-sphere
/// assumption); source distance is handled separately by
/// [`crate::nearfield`]. The output buffers are written up to
/// `min(out_left.len(), out_right.len(), dataset.hrir_len())` samples; any
/// trailing elements are left untouched.
///
/// Returns `None` if the dataset is empty or if either output buffer is
/// shorter than one sample; otherwise returns an [`InterpolationInfo`].
///
/// Real-time safe: no allocation, no panic.
pub fn interpolate(
    dataset: &HrtfDataset,
    azimuth: Sample,
    elevation: Sample,
    out_left: &mut [Sample],
    out_right: &mut [Sample],
) -> Option<InterpolationInfo> {
    let len = dataset.hrir_len().min(out_left.len()).min(out_right.len());
    if len == 0 || dataset.is_empty() {
        return None;
    }

    let target = direction_from_angles(azimuth, elevation);

    // Fixed-capacity nearest-neighbour set, sorted ascending by angle.
    let mut idx = [0usize; MAX_NEIGHBORS];
    let mut ang = [Sample::INFINITY; MAX_NEIGHBORS];
    let mut count = 0usize;

    for (m, meas) in dataset.measurements().iter().enumerate() {
        let dir = direction_from_angles(meas.azimuth, meas.elevation);
        let d = angular_distance(target, dir);
        insert_neighbor(&mut idx, &mut ang, &mut count, m, d);
    }

    // Convert angular distances to normalised Shepard weights.
    let mut weight = [0.0 as Sample; MAX_NEIGHBORS];
    if ang[0] <= EXACT_MATCH_ANGLE {
        // Snap to the single closest measurement to avoid diluting an exact hit.
        weight[0] = 1.0;
        // Collapse the blend to the single exact-match measurement.
        count = 1;
    } else {
        let mut sum = 0.0;
        for i in 0..count {
            let w = 1.0 / (ang[i] + SHEPARD_SMOOTHING);
            weight[i] = w;
            sum += w;
        }
        let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
        for w in weight.iter_mut().take(count) {
            *w *= inv;
        }
    }

    let left_delay = blend_ear(dataset, Ear::Left, &idx, &weight, count, len, out_left);
    let right_delay = blend_ear(dataset, Ear::Right, &idx, &weight, count, len, out_right);

    Some(InterpolationInfo {
        neighbors_used: count,
        left_delay,
        right_delay,
    })
}

/// Inserts `(candidate, distance)` into the ascending-sorted neighbour set,
/// keeping at most [`MAX_NEIGHBORS`] closest entries.
#[inline]
fn insert_neighbor(
    idx: &mut [usize; MAX_NEIGHBORS],
    ang: &mut [Sample; MAX_NEIGHBORS],
    count: &mut usize,
    candidate: usize,
    distance: Sample,
) {
    // Reject if the set is full and this candidate is farther than the worst.
    if *count == MAX_NEIGHBORS && distance >= ang[MAX_NEIGHBORS - 1] {
        return;
    }
    // Find the insertion point.
    let mut pos = *count;
    while pos > 0 && ang[pos - 1] > distance {
        pos -= 1;
    }
    // Shift the tail down by one (dropping the last if full).
    let last = (*count).min(MAX_NEIGHBORS - 1);
    let mut j = last;
    while j > pos {
        ang[j] = ang[j - 1];
        idx[j] = idx[j - 1];
        j -= 1;
    }
    ang[pos] = distance;
    idx[pos] = candidate;
    if *count < MAX_NEIGHBORS {
        *count += 1;
    }
}

/// Blends one ear's HRIRs with ITD alignment, writing `len` samples into `out`
/// and returning the reconstructed onset delay.
fn blend_ear(
    dataset: &HrtfDataset,
    ear: Ear,
    idx: &[usize; MAX_NEIGHBORS],
    weight: &[Sample; MAX_NEIGHBORS],
    count: usize,
    len: usize,
    out: &mut [Sample],
) -> usize {
    for s in out.iter_mut().take(len) {
        *s = 0.0;
    }

    let mut delay_acc = 0.0;
    for i in 0..count {
        let hrir = match ear {
            Ear::Left => dataset.left_hrir(idx[i]),
            Ear::Right => dataset.right_hrir(idx[i]),
        };
        if hrir.is_empty() {
            continue;
        }
        let onset = estimate_onset_delay(hrir);
        let w = weight[i];
        delay_acc += w * onset as Sample;

        // Accumulate the onset-aligned (delay-removed) weighted response.
        for (j, o) in out.iter_mut().enumerate().take(len) {
            let src = j + onset;
            if src < hrir.len() {
                *o += w * hrir[src];
            }
        }
    }

    // Re-apply the interpolated onset delay by shifting the aligned response
    // right in place (read from lower indices, so a reverse scan is safe).
    let delay = round_to_usize(delay_acc).min(len.saturating_sub(1));
    if delay > 0 {
        let mut i = len;
        while i > 0 {
            i -= 1;
            out[i] = if i >= delay { out[i - delay] } else { 0.0 };
        }
    }
    delay
}

/// Rounds a non-negative sample to the nearest `usize` (deterministic).
#[inline]
fn round_to_usize(x: Sample) -> usize {
    if x <= 0.0 {
        0
    } else {
        ops::round(x) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::Measurement;
    use alloc::vec;
    use alloc::vec::Vec;
    use core::f32::consts::{FRAC_PI_2, PI};

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    /// Dataset with an impulse at a per-direction delay so tests can check
    /// alignment: front (delay 2), right (delay 5), left (delay 5).
    fn delayed_dataset() -> HrtfDataset {
        let len = 16;
        let dirs = [
            (0.0, 0.0),        // front
            (FRAC_PI_2, 0.0),  // right
            (-FRAC_PI_2, 0.0), // left
        ];
        let delays = [2usize, 5, 5];
        let mut measurements = Vec::new();
        let mut left = vec![0.0f32; dirs.len() * len];
        let mut right = vec![0.0f32; dirs.len() * len];
        for (m, ((az, el), d)) in dirs.iter().zip(delays.iter()).enumerate() {
            measurements.push(Measurement::new(*az, *el, 1.0));
            left[m * len + d] = 1.0;
            right[m * len + d] = 1.0;
        }
        HrtfDataset::from_samples(48_000, len, measurements, left, right).unwrap()
    }

    #[test]
    fn direction_round_trips_through_angles() {
        let d = direction_from_angles(0.0, 0.0);
        assert!(approx(d.x, 0.0, 1e-6));
        assert!(approx(d.y, 0.0, 1e-6));
        assert!(approx(d.z, -1.0, 1e-6));
        let r = direction_from_angles(FRAC_PI_2, 0.0);
        assert!(approx(r.x, 1.0, 1e-6));
        let up = direction_from_angles(0.0, FRAC_PI_2);
        assert!(approx(up.y, 1.0, 1e-6));
    }

    #[test]
    fn angular_distance_is_symmetric_and_bounded() {
        let a = direction_from_angles(0.0, 0.0);
        let b = direction_from_angles(PI, 0.0);
        assert!(approx(angular_distance(a, b), PI, 1e-4));
        assert!(approx(angular_distance(a, a), 0.0, 1e-6));
        assert!(approx(angular_distance(a, b), angular_distance(b, a), 1e-6));
    }

    #[test]
    fn onset_detection_finds_pulse() {
        let ir = [0.0, 0.0, 0.0, 1.0, 0.2, 0.0];
        assert_eq!(estimate_onset_delay(&ir), 3);
        let silent = [0.0, 0.0, 0.0];
        assert_eq!(estimate_onset_delay(&silent), 0);
    }

    #[test]
    fn exact_match_reproduces_measurement() {
        let ds = delayed_dataset();
        let mut l = vec![0.0f32; 16];
        let mut r = vec![0.0f32; 16];
        let info = interpolate(&ds, FRAC_PI_2, 0.0, &mut l, &mut r).unwrap();
        assert_eq!(info.neighbors_used, 1);
        // Right measurement had its pulse at delay 5; must be reconstructed there.
        assert_eq!(info.right_delay, 5);
        assert!(approx(r[5], 1.0, 1e-6));
        assert!(approx(l[5], 1.0, 1e-6));
    }

    #[test]
    fn interpolated_direction_preserves_unit_energy() {
        let ds = delayed_dataset();
        let mut l = vec![0.0f32; 16];
        let mut r = vec![0.0f32; 16];
        // Halfway between front (delay 2) and right (delay 5).
        let info = interpolate(&ds, FRAC_PI_2 * 0.5, 0.0, &mut l, &mut r).unwrap();
        assert!(info.neighbors_used >= 2);
        // The aligned blend keeps total amplitude ~1 (no comb cancellation),
        // and the reconstructed delay lies between the two source delays.
        let sum_r: f32 = r.iter().sum();
        assert!(approx(sum_r, 1.0, 1e-4), "sum was {sum_r}");
        assert!(info.right_delay >= 2 && info.right_delay <= 5);
    }

    #[test]
    fn alignment_avoids_comb_cancellation() {
        // Two equally weighted pulses at different delays. A naive average
        // would place two half-height pulses (peak 0.5); the aligned blend
        // keeps a single unit-height pulse.
        let ds = delayed_dataset();
        let mut l = vec![0.0f32; 16];
        let mut r = vec![0.0f32; 16];
        interpolate(&ds, FRAC_PI_2 * 0.5, 0.0, &mut l, &mut r).unwrap();
        let peak = r.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
        assert!(peak > 0.9, "aligned peak collapsed to {peak}");
    }

    #[test]
    fn empty_output_returns_none() {
        let ds = delayed_dataset();
        let mut l: [f32; 0] = [];
        let mut r = vec![0.0f32; 16];
        assert!(interpolate(&ds, 0.0, 0.0, &mut l, &mut r).is_none());
    }

    #[test]
    fn shorter_output_is_truncated_safely() {
        let ds = delayed_dataset();
        let mut l = vec![0.0f32; 4];
        let mut r = vec![0.0f32; 4];
        // Front pulse is at delay 2, which fits in a 4-sample window.
        let info = interpolate(&ds, 0.0, 0.0, &mut l, &mut r).unwrap();
        assert_eq!(info.left_delay, 2);
        assert!(approx(l[2], 1.0, 1e-6));
    }

    #[test]
    fn neighbor_insertion_keeps_closest() {
        let mut idx = [0usize; MAX_NEIGHBORS];
        let mut ang = [f32::INFINITY; MAX_NEIGHBORS];
        let mut count = 0;
        for (i, d) in [(0, 0.9), (1, 0.1), (2, 0.5), (3, 0.3), (4, 0.7)] {
            insert_neighbor(&mut idx, &mut ang, &mut count, i, d);
        }
        assert_eq!(count, MAX_NEIGHBORS);
        // Sorted ascending, farthest (0.9, idx 0) evicted.
        assert_eq!(idx[0], 1);
        assert!(ang[0] <= ang[1] && ang[1] <= ang[2] && ang[2] <= ang[3]);
        assert!(ang[MAX_NEIGHBORS - 1] < 0.9);
    }
}
