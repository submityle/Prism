//! Image-source early reflections for rectangular ("shoebox") rooms.
//!
//! Early reflections are the first few wall bounces that reach a listener after
//! the direct sound. Their timing and direction are the dominant cue for room
//! size and source distance, so a spatial engine renders them explicitly rather
//! than folding them into a late-reverb tail. For an axis-aligned rectangular
//! room the reflection paths have a closed form: every reflected path is
//! equivalent to a straight line from a mirrored copy of the source (an *image
//! source*) to the listener. Enumerating the images over a small integer index
//! grid therefore yields, per bounce, an exact delay, gain, and arrival
//! direction with no ray tracing and no floating-point error accumulation.
//!
//! This module computes that reflection tap list at control rate
//! ([`compute_early_reflections`]) and, optionally, renders it in real time
//! through a single shared delay line with per-tap equal-power panning
//! ([`EarlyReflectionRenderer`]).
//!
//! # Geometry
//!
//! The room is the axis-aligned box `[min, max]` (metres, world space). For one
//! axis with span `w = max - min` and a source coordinate `sr = source - min`
//! relative to the low wall, the image at integer index `n` is
//!
//! ```text
//! image_r = n * w + sr           (n even)
//! image_r = (n + 1) * w - sr     (n odd)
//! image_world = min + image_r
//! ```
//!
//! so `n = 0` reproduces the source (the direct path), `n = 1` is the mirror in
//! the high wall, `n = -1` the mirror in the low wall, and larger `|n|` are the
//! repeated bounces. The reflection order along that axis is `|n|`, and the
//! total order of an image `(nx, ny, nz)` is `|nx| + |ny| + |nz|`.
//!
//! # Gain and delay
//!
//! The arrival direction is the unit vector from the image to the listener,
//! expressed in the listener's local frame (matching
//! [`crate::geometry::Listener::localize`], so `-Z` is forward). The path length
//! is the Euclidean distance from the image to the listener; the delay is
//! `distance / sound_speed` and the geometric-spreading gain is `1 / distance`
//! (clamped at a small minimum distance to bound near-coincident gains).
//!
//! Each wall absorbs energy by its coefficient `alpha in [0, 1]`; the amplitude
//! reflection coefficient is `beta = sqrt(1 - alpha)` (energy to amplitude). A
//! path that crosses a wall `k` times multiplies its gain by `beta^k`. The wall
//! crossing counts per axis follow directly from the image index `n`:
//!
//! ```text
//! n >= 0:  high_count = (n + 1) / 2,  low_count = n / 2       (integer division)
//! n <  0:  low_count  = (|n| + 1) / 2, high_count = |n| / 2
//! ```
//!
//! With `alpha = 0` every `beta = 1` (perfect reflector); with `alpha = 1` on a
//! face, `beta = 0`, so any path touching that face is silenced.
//!
//! # Real-time contract
//!
//! [`compute_early_reflections`] runs at control rate, is stack-only, allocates
//! nothing, and never panics: degenerate rooms, out-of-box points, zero sample
//! rates, and over-large orders all produce a finite, bounded result. The
//! renderer allocates its delay line once in [`EarlyReflectionRenderer::new`];
//! [`EarlyReflectionRenderer::process_block`] is allocation free, lock free, and
//! panic free.
//!
//! # Determinism
//!
//! All length and transcendental math routes through [`bevy_math::ops`], and
//! sample-accurate delays are accumulated with integer stepping (never a
//! floating-point-to-integer cast), so the tap list is bit-reproducible and can
//! be golden-compared sample-for-sample.
//!
//! # Provenance
//!
//! This module is engine-agnostic and contains **no Unreal Engine, Unity,
//! Godot, Wwise, FMOD, Steam Audio, or Google Resonance Audio source or derived
//! code**. The rectangular-room image method is publicly documented acoustics:
//! see J. B. Allen and D. A. Berkley, "Image method for efficiently simulating
//! small-room acoustics," Journal of the Acoustical Society of America 65(4),
//! 1979. Everything here is implemented from that public description.

use alloc::vec;
use alloc::vec::Vec;

use bevy_math::{Vec3, ops};
use prism_audio_core::math::{Sample, equal_power_pan};

use crate::geometry::Listener;

/// Maximum number of reflection taps a single query returns. When more images
/// than this fall within the requested order, the strongest by gain are kept
/// and the weakest are evicted. Bounded at 32 to stay within serde's derived
/// array support and to keep all storage on the stack.
pub const MAX_EARLY_REFLECTIONS: usize = 32;

/// Maximum reflection order (total wall bounces `|nx| + |ny| + |nz|`) the
/// enumerator will consider. Requested orders above this are clamped.
pub const MAX_REFLECTION_ORDER: usize = 4;

/// Default speed of sound in dry air at room temperature, in metres per second.
pub const DEFAULT_SOUND_SPEED: Sample = 343.0;

/// Upper bound on the integer delay a tap may report, in samples. Caps the
/// integer-stepping delay conversion so it always terminates in bounded time
/// (2^18 samples is roughly 5.5 seconds at 48 kHz, far beyond any early
/// reflection).
const MAX_DELAY_SAMPLES: usize = 1 << 18;

/// Distances below this (metres) are clamped when forming the `1 / distance`
/// spreading gain, bounding the gain of near-coincident paths.
const MIN_DISTANCE_METRES: Sample = 0.1;

/// Axis spans at or below this (metres) are treated as degenerate (zero width);
/// that axis contributes only the `n = 0` image.
const DEGENERATE_SPAN: Sample = 1.0e-6;

/// Distances below this (metres) collapse the arrival direction to local
/// forward, avoiding a division by a near-zero length.
const COINCIDENT_EPSILON: Sample = 1.0e-6;

/// An axis-aligned rectangular ("shoebox") room.
///
/// `min` and `max` are opposite corners in world space (metres); `new`
/// normalises them so `max >= min` component-wise. `wall_absorption` holds the
/// six energy absorption coefficients (each clamped to `[0, 1]`) ordered
/// `[x_low, x_high, y_low, y_high, z_low, z_high]`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ShoeboxRoom {
    /// Low corner (component-wise minimum) in world-space metres.
    pub min: Vec3,
    /// High corner (component-wise maximum) in world-space metres.
    pub max: Vec3,
    /// Per-face energy absorption in `[0, 1]`, ordered
    /// `[x_low, x_high, y_low, y_high, z_low, z_high]`.
    pub wall_absorption: [Sample; 6],
}

impl ShoeboxRoom {
    /// Builds a room from two corners and the six face absorption coefficients.
    ///
    /// The corners are sorted component-wise so `max >= min`, and every
    /// absorption is clamped to `[0, 1]`. Construction never panics.
    #[must_use]
    pub fn new(corner_a: Vec3, corner_b: Vec3, wall_absorption: [Sample; 6]) -> Self {
        let min = corner_a.min(corner_b);
        let max = corner_a.max(corner_b);
        let mut clamped = [0.0; 6];
        for i in 0..6 {
            clamped[i] = wall_absorption[i].clamp(0.0, 1.0);
        }
        Self { min, max, wall_absorption: clamped }
    }

    /// A room with the given corners and perfectly reflecting walls
    /// (`absorption = 0` on every face).
    #[must_use]
    #[inline]
    pub fn rigid(corner_a: Vec3, corner_b: Vec3) -> Self {
        Self::new(corner_a, corner_b, [0.0; 6])
    }

    /// The room's interior span (`max - min`) in metres.
    #[must_use]
    #[inline]
    pub fn size(&self) -> Vec3 {
        self.max - self.min
    }
}

/// One early-reflection tap: a single image-source arrival.
///
/// `direction` is a unit vector in the listener's local frame (`-Z` forward,
/// `+X` right, `+Y` up). `delay_samples` is the integer path delay,
/// `gain` the linear amplitude, `order` the total wall-bounce count, and
/// `is_direct` marks the zero-order direct path.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ReflectionTap {
    /// Path delay in whole samples.
    pub delay_samples: usize,
    /// Linear amplitude gain (spreading times wall reflection product).
    pub gain: Sample,
    /// Arrival direction, listener-local unit vector.
    pub direction: Vec3,
    /// Total reflection order `|nx| + |ny| + |nz|` (0 for the direct path).
    pub order: u32,
    /// True only for the zero-order direct path.
    pub is_direct: bool,
}

impl ReflectionTap {
    /// A silent placeholder tap (used to initialise fixed-size storage).
    const SILENT: Self = Self {
        delay_samples: 0,
        gain: 0.0,
        direction: Vec3::NEG_Z,
        order: 0,
        is_direct: false,
    };
}

/// Converts a delay in seconds to whole samples by integer stepping, avoiding a
/// floating-point-to-integer cast. Coarse 1024-sample strides keep the loop
/// bounded, a fine pass adds the remainder, and a final half-sample compare
/// rounds to nearest. The result is clamped to [`MAX_DELAY_SAMPLES`].
#[must_use]
fn seconds_to_samples(seconds: Sample, sample_rate: Sample) -> usize {
    let cap = MAX_DELAY_SAMPLES as Sample;
    let exact = (seconds * sample_rate).max(0.0).min(cap);
    let mut n: usize = 0;
    let mut acc: Sample = 0.0;
    while acc + 1024.0 <= exact {
        acc += 1024.0;
        n += 1024;
    }
    while acc + 1.0 <= exact {
        acc += 1.0;
        n += 1;
    }
    if exact - acc >= 0.5 {
        n += 1;
    }
    n
}

/// Raises `base` to a small integer power by repeated multiplication (the
/// exponent is at most the reflection order, so no `powi`/`powf` is needed).
#[must_use]
#[inline]
fn pow_small(base: Sample, exponent: u32) -> Sample {
    let mut acc: Sample = 1.0;
    let mut k = 0;
    while k < exponent {
        acc *= base;
        k += 1;
    }
    acc
}

/// Folds a source coordinate to its image-source coordinate along one axis.
///
/// Returns the image position relative to the low wall (`image_r`); the world
/// coordinate is `low + image_r`. See the module docs for the derivation.
#[must_use]
#[inline]
fn fold_axis(n: i32, span: Sample, source_rel: Sample) -> Sample {
    if n % 2 == 0 {
        n as Sample * span + source_rel
    } else {
        (n + 1) as Sample * span - source_rel
    }
}

/// Low- and high-wall crossing counts for an image index `n` along one axis.
#[must_use]
#[inline]
fn wall_counts(n: i32) -> (u32, u32) {
    if n >= 0 {
        let m = n.unsigned_abs();
        let low = m / 2;
        let high = m.div_ceil(2);
        (low, high)
    } else {
        let m = n.unsigned_abs();
        let low = m.div_ceil(2);
        let high = m / 2;
        (low, high)
    }
}

/// Enumerates the early-reflection taps for a shoebox room.
///
/// Images are enumerated over the integer grid `(nx, ny, nz)` with
/// `|nx| + |ny| + |nz| <= order` (order clamped to [`MAX_REFLECTION_ORDER`]);
/// the zero-order image is the direct path and is flagged
/// [`ReflectionTap::is_direct`]. Degenerate axes (span below
/// [`DEGENERATE_SPAN`]) contribute only their `n = 0` image. `sample_rate` and
/// `sound_speed` are floored to a small positive value so no division by zero
/// occurs. Taps are written into `out`; if more images qualify than
/// `min(out.len(), MAX_EARLY_REFLECTIONS)`, the strongest by gain are kept.
///
/// Returns the number of taps written. Runs at control rate, allocates nothing,
/// and never panics.
///
/// # Examples
///
/// ```
/// use bevy_math::Vec3;
/// use prism_audio_spatial::early_reflections::{
///     compute_early_reflections, ReflectionTap, ShoeboxRoom, DEFAULT_SOUND_SPEED,
/// };
/// use prism_audio_spatial::geometry::Listener;
///
/// let room = ShoeboxRoom::rigid(Vec3::splat(-5.0), Vec3::splat(5.0));
/// let listener = Listener::default(); // at the origin, facing -Z
/// let source = Vec3::new(0.0, 0.0, -3.0); // 3 m straight ahead
/// let mut taps = [ReflectionTap {
///     delay_samples: 0,
///     gain: 0.0,
///     direction: Vec3::NEG_Z,
///     order: 0,
///     is_direct: false,
/// }; 32];
///
/// let count = compute_early_reflections(
///     &room, &listener, source, 1, 48_000.0, DEFAULT_SOUND_SPEED, &mut taps,
/// );
/// assert!(count >= 1);
/// // The direct path is present and points forward (local -Z).
/// let direct = taps[..count].iter().find(|t| t.is_direct).unwrap();
/// assert!(direct.direction.z < 0.0);
/// ```
#[must_use]
#[expect(
    clippy::needless_range_loop,
    reason = "index-parallel access to out plus arithmetic on the loop index is clearer than iterator adaptors"
)]
pub fn compute_early_reflections(
    room: &ShoeboxRoom,
    listener: &Listener,
    source: Vec3,
    order: usize,
    sample_rate: Sample,
    sound_speed: Sample,
    out: &mut [ReflectionTap],
) -> usize {
    let cap = out.len().min(MAX_EARLY_REFLECTIONS);
    if cap == 0 {
        return 0;
    }

    let order_i = order.min(MAX_REFLECTION_ORDER) as i32;
    let fs = sample_rate.max(1.0);
    let c = sound_speed.max(1.0);

    let span = room.max - room.min;
    let source_rel = source - room.min;

    // Amplitude reflection coefficients per face: beta = sqrt(1 - alpha).
    let beta = [
        ops::sqrt(1.0 - room.wall_absorption[0]),
        ops::sqrt(1.0 - room.wall_absorption[1]),
        ops::sqrt(1.0 - room.wall_absorption[2]),
        ops::sqrt(1.0 - room.wall_absorption[3]),
        ops::sqrt(1.0 - room.wall_absorption[4]),
        ops::sqrt(1.0 - room.wall_absorption[5]),
    ];

    // Per-axis index ranges: a degenerate axis is pinned to n = 0.
    let nx_max = if span.x > DEGENERATE_SPAN { order_i } else { 0 };
    let ny_max = if span.y > DEGENERATE_SPAN { order_i } else { 0 };
    let nz_max = if span.z > DEGENERATE_SPAN { order_i } else { 0 };

    let inv = listener.orientation.inverse();
    let mut count: usize = 0;

    let mut nx = -nx_max;
    while nx <= nx_max {
        let mut ny = -ny_max;
        while ny <= ny_max {
            let mut nz = -nz_max;
            while nz <= nz_max {
                let total = nx.unsigned_abs() + ny.unsigned_abs() + nz.unsigned_abs();
                if total as i32 > order_i {
                    nz += 1;
                    continue;
                }

                // Image world position from the per-axis folding formula.
                let image = Vec3::new(
                    room.min.x + fold_axis(nx, span.x, source_rel.x),
                    room.min.y + fold_axis(ny, span.y, source_rel.y),
                    room.min.z + fold_axis(nz, span.z, source_rel.z),
                );

                let delta = image - listener.position;
                let distance = ops::sqrt(delta.dot(delta));

                // Wall reflection product across the six faces.
                let (xl, xh) = wall_counts(nx);
                let (yl, yh) = wall_counts(ny);
                let (zl, zh) = wall_counts(nz);
                let wall_gain = pow_small(beta[0], xl)
                    * pow_small(beta[1], xh)
                    * pow_small(beta[2], yl)
                    * pow_small(beta[3], yh)
                    * pow_small(beta[4], zl)
                    * pow_small(beta[5], zh);

                let spread = 1.0 / distance.max(MIN_DISTANCE_METRES);
                let gain = spread * wall_gain;

                let direction = if distance <= COINCIDENT_EPSILON {
                    Vec3::NEG_Z
                } else {
                    inv * (delta / distance)
                };

                let delay_samples = seconds_to_samples(distance / c, fs);
                let is_direct = total == 0;

                let tap = ReflectionTap {
                    delay_samples,
                    gain,
                    direction,
                    order: total,
                    is_direct,
                };

                if count < cap {
                    out[count] = tap;
                    count += 1;
                } else {
                    // Replace the weakest stored tap if this one is stronger.
                    let mut min_index = 0;
                    let mut min_gain = out[0].gain;
                    for i in 1..cap {
                        if out[i].gain < min_gain {
                            min_gain = out[i].gain;
                            min_index = i;
                        }
                    }
                    if tap.gain > min_gain {
                        out[min_index] = tap;
                    }
                }

                nz += 1;
            }
            ny += 1;
        }
        nx += 1;
    }

    count
}

/// Real-time renderer for a fixed early-reflection tap list.
///
/// A single delay line is fed the dry (mono) input; each tap reads it at its own
/// delay, scales by its gain, and pans to stereo by equal-power panning on the
/// tap direction's lateral (`x`) component. The delay line is allocated once in
/// [`EarlyReflectionRenderer::new`]; [`EarlyReflectionRenderer::process_block`]
/// allocates nothing and never panics.
pub struct EarlyReflectionRenderer {
    delay_line: Vec<Sample>,
    write_pos: usize,
    taps: [ReflectionTap; MAX_EARLY_REFLECTIONS],
    tap_count: usize,
    max_delay: usize,
}

impl EarlyReflectionRenderer {
    /// Creates a renderer with a delay line long enough for `max_delay` samples.
    /// The delay line (`max_delay + 1` samples) is the only allocation.
    #[must_use]
    pub fn new(max_delay: usize) -> Self {
        let size = max_delay.saturating_add(1);
        Self {
            delay_line: vec![0.0; size],
            write_pos: 0,
            taps: [ReflectionTap::SILENT; MAX_EARLY_REFLECTIONS],
            tap_count: 0,
            max_delay,
        }
    }

    /// Loads a tap list (control rate). Up to [`MAX_EARLY_REFLECTIONS`] taps are
    /// copied; each tap delay is clamped to the renderer's `max_delay`.
    pub fn set_taps(&mut self, taps: &[ReflectionTap]) {
        let n = taps.len().min(MAX_EARLY_REFLECTIONS);
        for (slot, tap) in self.taps.iter_mut().zip(taps.iter().take(n)) {
            let mut copy = *tap;
            copy.delay_samples = copy.delay_samples.min(self.max_delay);
            *slot = copy;
        }
        self.tap_count = n;
    }

    /// Number of active taps.
    #[must_use]
    #[inline]
    pub fn tap_count(&self) -> usize {
        self.tap_count
    }

    /// Clears the delay-line history (silences the tail) without dropping taps.
    pub fn reset(&mut self) {
        for s in &mut self.delay_line {
            *s = 0.0;
        }
        self.write_pos = 0;
    }

    /// Renders one block: pushes `dry` through the shared delay line and sums
    /// the panned taps into `out_l` / `out_r`. Processes
    /// `min(dry.len(), out_l.len(), out_r.len())` samples. Allocation, lock, and
    /// panic free.
    pub fn process_block(&mut self, dry: &[Sample], out_l: &mut [Sample], out_r: &mut [Sample]) {
        let len = dry.len().min(out_l.len()).min(out_r.len());
        let size = self.delay_line.len();
        if size == 0 {
            return;
        }

        for i in 0..len {
            self.delay_line[self.write_pos] = dry[i];

            let mut acc_l: Sample = 0.0;
            let mut acc_r: Sample = 0.0;
            for t in 0..self.tap_count {
                let tap = self.taps[t];
                let d = tap.delay_samples.min(size - 1);
                let read = (self.write_pos + size - d) % size;
                let s = self.delay_line[read] * tap.gain;
                let pan = tap.direction.x.clamp(-1.0, 1.0);
                let (l, r) = equal_power_pan(pan);
                acc_l += s * l;
                acc_r += s * r;
            }

            out_l[i] = acc_l;
            out_r[i] = acc_r;
            self.write_pos = (self.write_pos + 1) % size;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::Quat;
    use core::f32::consts::FRAC_PI_2;

    fn approx(a: Sample, b: Sample, eps: Sample) -> bool {
        (a - b).abs() <= eps
    }

    fn empty_taps() -> [ReflectionTap; MAX_EARLY_REFLECTIONS] {
        [ReflectionTap::SILENT; MAX_EARLY_REFLECTIONS]
    }

    #[test]
    fn cube_center_first_order_is_symmetric() {
        // Cube [0, 4]^3, listener and source both at the centre.
        let room = ShoeboxRoom::rigid(Vec3::ZERO, Vec3::splat(4.0));
        let listener = Listener {
            position: Vec3::splat(2.0),
            ..Listener::default()
        };
        let source = Vec3::splat(2.0);
        let mut taps = empty_taps();
        let n = compute_early_reflections(
            &room, &listener, source, 1, 48_000.0, DEFAULT_SOUND_SPEED, &mut taps,
        );
        // 1 direct + 6 first-order images.
        assert_eq!(n, 7);
        let firsts: Vec<&ReflectionTap> =
            taps[..n].iter().filter(|t| t.order == 1).collect();
        assert_eq!(firsts.len(), 6);
        // All six share one delay and one gain (distance == box side == 4 m).
        let d0 = firsts[0].delay_samples;
        let g0 = firsts[0].gain;
        for f in &firsts {
            assert_eq!(f.delay_samples, d0);
            assert!(approx(f.gain, g0, 1e-6));
        }
    }

    #[test]
    fn direct_delay_matches_distance_over_speed() {
        let room = ShoeboxRoom::rigid(Vec3::splat(-10.0), Vec3::splat(10.0));
        let listener = Listener::default();
        let source = Vec3::new(0.0, 0.0, -3.43); // 3.43 m ahead
        let mut taps = empty_taps();
        let n = compute_early_reflections(
            &room, &listener, source, 0, 100.0, 343.0, &mut taps,
        );
        assert_eq!(n, 1);
        assert!(taps[0].is_direct);
        // 3.43 m / 343 m/s = 0.01 s * 100 Hz = 1 sample.
        assert_eq!(taps[0].delay_samples, 1);
    }

    #[test]
    fn gain_falls_as_inverse_distance() {
        let room = ShoeboxRoom::rigid(Vec3::splat(-50.0), Vec3::splat(50.0));
        let listener = Listener::default();
        let mut taps = empty_taps();

        let near = Vec3::new(0.0, 0.0, -2.0);
        let far = Vec3::new(0.0, 0.0, -4.0);
        let _ = compute_early_reflections(
            &room, &listener, near, 0, 48_000.0, 343.0, &mut taps,
        );
        let g_near = taps[0].gain;
        let _ = compute_early_reflections(
            &room, &listener, far, 0, 48_000.0, 343.0, &mut taps,
        );
        let g_far = taps[0].gain;
        // Doubling distance halves the 1/r gain.
        assert!(approx(g_near / g_far, 2.0, 1e-4));
    }

    #[test]
    fn delay_grows_with_room_size() {
        let listener = Listener::default();
        let source = Vec3::ZERO;
        let mut taps = empty_taps();

        let small = ShoeboxRoom::rigid(Vec3::splat(-2.0), Vec3::splat(2.0));
        let big = ShoeboxRoom::rigid(Vec3::splat(-8.0), Vec3::splat(8.0));

        // Place listener at the centre of each; compare a first-order delay.
        let n_small = compute_early_reflections(
            &small, &listener, source, 1, 48_000.0, 343.0, &mut taps,
        );
        let small_first = taps[..n_small]
            .iter()
            .filter(|t| t.order == 1)
            .map(|t| t.delay_samples)
            .max()
            .unwrap();
        let n_big = compute_early_reflections(
            &big, &listener, source, 1, 48_000.0, 343.0, &mut taps,
        );
        let big_first = taps[..n_big]
            .iter()
            .filter(|t| t.order == 1)
            .map(|t| t.delay_samples)
            .max()
            .unwrap();
        assert!(big_first > small_first);
    }

    #[test]
    fn zero_absorption_reflects_fully() {
        let room = ShoeboxRoom::rigid(Vec3::ZERO, Vec3::splat(4.0));
        let listener = Listener {
            position: Vec3::splat(2.0),
            ..Listener::default()
        };
        let source = Vec3::splat(2.0);
        let mut taps = empty_taps();
        let n = compute_early_reflections(
            &room, &listener, source, 1, 48_000.0, 343.0, &mut taps,
        );
        // First-order gain must equal pure 1/distance (beta == 1 everywhere).
        let first = taps[..n].iter().find(|t| t.order == 1).unwrap();
        assert!(approx(first.gain, 1.0 / 4.0, 1e-5));
    }

    #[test]
    fn full_absorption_silences_that_face() {
        // Absorb the x-high wall (index 1) completely; keep others rigid.
        let room = ShoeboxRoom::new(
            Vec3::ZERO,
            Vec3::splat(4.0),
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        );
        let listener = Listener {
            position: Vec3::splat(2.0),
            ..Listener::default()
        };
        let source = Vec3::splat(2.0);
        let mut taps = empty_taps();
        let n = compute_early_reflections(
            &room, &listener, source, 1, 48_000.0, 343.0, &mut taps,
        );
        // The x-high image (nx = 1) must be silenced; the x-low image is intact.
        // Identify by direction: x-high image is to the listener's +X.
        let x_high = taps[..n]
            .iter()
            .find(|t| t.order == 1 && t.direction.x > 0.5);
        let x_low = taps[..n]
            .iter()
            .find(|t| t.order == 1 && t.direction.x < -0.5)
            .unwrap();
        assert!(x_high.is_none() || approx(x_high.unwrap().gain, 0.0, 1e-6));
        assert!(x_low.gain > 0.0);
    }

    #[test]
    fn reflection_order_counts_are_correct() {
        let room = ShoeboxRoom::rigid(Vec3::splat(-5.0), Vec3::splat(5.0));
        let listener = Listener::default();
        let source = Vec3::new(0.0, 0.0, -1.0);
        let mut taps = empty_taps();
        let n = compute_early_reflections(
            &room, &listener, source, 2, 48_000.0, 343.0, &mut taps,
        );
        // Every reported order is within [0, 2].
        for t in &taps[..n] {
            assert!(t.order <= 2);
        }
        // Exactly one direct path.
        assert_eq!(taps[..n].iter().filter(|t| t.is_direct).count(), 1);
    }

    #[test]
    fn head_rotation_moves_arrival_into_local_frame() {
        // Source straight ahead in the world; rotate the head 90 degrees about
        // +Y so world -Z maps to the listener's +X (to the right).
        let room = ShoeboxRoom::rigid(Vec3::splat(-10.0), Vec3::splat(10.0));
        let listener = Listener {
            orientation: Quat::from_rotation_y(FRAC_PI_2),
            ..Listener::default()
        };
        let source = Vec3::new(0.0, 0.0, -3.0);
        let mut taps = empty_taps();
        let n = compute_early_reflections(
            &room, &listener, source, 0, 48_000.0, 343.0, &mut taps,
        );
        assert_eq!(n, 1);
        // With a +90 deg yaw the forward world source lands on local +X.
        assert!(taps[0].direction.x > 0.9);
        assert!(taps[0].direction.z.abs() < 0.2);
    }

    #[test]
    fn degenerate_inputs_do_not_panic() {
        let mut taps = empty_taps();

        // Zero-size box.
        let flat = ShoeboxRoom::rigid(Vec3::ZERO, Vec3::ZERO);
        let listener = Listener::default();
        let _ = compute_early_reflections(
            &flat, &listener, Vec3::ZERO, 3, 48_000.0, 343.0, &mut taps,
        );

        // Point outside the box, zero sample rate, and over-large order.
        let room = ShoeboxRoom::rigid(Vec3::splat(-1.0), Vec3::splat(1.0));
        let outside = Vec3::new(100.0, -50.0, 25.0);
        let _ = compute_early_reflections(
            &room, &listener, outside, 999, 0.0, 0.0, &mut taps,
        );

        // Source coincident with the listener.
        let _ = compute_early_reflections(
            &room, &listener, listener.position, 1, 48_000.0, 343.0, &mut taps,
        );

        // Empty output slice.
        let mut none: [ReflectionTap; 0] = [];
        assert_eq!(
            compute_early_reflections(&room, &listener, Vec3::ZERO, 2, 48_000.0, 343.0, &mut none),
            0,
        );
    }

    #[test]
    fn overflow_keeps_strongest_taps() {
        // A tiny output slice forces eviction; the kept taps must be the
        // strongest present (their minimum gain must not fall below any
        // rejected candidate, verified by re-running with full capacity).
        let room = ShoeboxRoom::rigid(Vec3::splat(-3.0), Vec3::splat(3.0));
        let listener = Listener::default();
        let source = Vec3::new(0.5, 0.0, -0.5);

        let mut full = empty_taps();
        let n_full = compute_early_reflections(
            &room, &listener, source, 3, 48_000.0, 343.0, &mut full,
        );
        assert!(n_full > 4);

        let mut small = [ReflectionTap::SILENT; 4];
        let n_small = compute_early_reflections(
            &room, &listener, source, 3, 48_000.0, 343.0, &mut small,
        );
        assert_eq!(n_small, 4);

        // The weakest kept tap must be >= the 4th strongest overall.
        let mut gains: Vec<Sample> =
            full[..n_full].iter().map(|t| t.gain).collect();
        gains.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let fourth_strongest = gains[3];
        let kept_min = small.iter().map(|t| t.gain).fold(Sample::INFINITY, Sample::min);
        assert!(kept_min >= fourth_strongest - 1e-6);
    }

    #[test]
    fn seconds_to_samples_rounds_to_nearest() {
        assert_eq!(seconds_to_samples(0.0, 48_000.0), 0);
        // 0.01 s * 48000 = 480 samples exactly.
        assert_eq!(seconds_to_samples(0.01, 48_000.0), 480);
        // 100.4 samples -> 100; 100.6 -> 101.
        assert_eq!(seconds_to_samples(100.4 / 48_000.0, 48_000.0), 100);
        assert_eq!(seconds_to_samples(100.6 / 48_000.0, 48_000.0), 101);
        // Clamped at the cap.
        assert_eq!(seconds_to_samples(1.0e9, 48_000.0), MAX_DELAY_SAMPLES);
    }

    #[test]
    fn renderer_delays_and_pans() {
        let mut renderer = EarlyReflectionRenderer::new(16);
        let taps = [
            ReflectionTap {
                delay_samples: 0,
                gain: 1.0,
                direction: Vec3::NEG_Z, // centre -> equal L/R
                order: 0,
                is_direct: true,
            },
            ReflectionTap {
                delay_samples: 3,
                gain: 0.5,
                direction: Vec3::X, // hard right
                order: 1,
                is_direct: false,
            },
        ];
        renderer.set_taps(&taps);
        assert_eq!(renderer.tap_count(), 2);

        let mut dry = [0.0; 8];
        dry[0] = 1.0; // an impulse
        let mut l = [0.0; 8];
        let mut r = [0.0; 8];
        renderer.process_block(&dry, &mut l, &mut r);

        // Direct tap at sample 0: centre pan -> equal, non-zero energy.
        assert!(l[0] > 0.0);
        assert!(approx(l[0], r[0], 1e-6));
        // Right-panned reflection at sample 3: R louder than L.
        assert!(r[3] > l[3]);
        assert!(r[3] > 0.0);
    }

    #[test]
    fn renderer_reset_clears_tail() {
        let mut renderer = EarlyReflectionRenderer::new(8);
        let taps = [ReflectionTap {
            delay_samples: 2,
            gain: 1.0,
            direction: Vec3::NEG_Z,
            order: 1,
            is_direct: false,
        }];
        renderer.set_taps(&taps);
        let mut dry = [1.0; 4];
        let mut l = [0.0; 4];
        let mut r = [0.0; 4];
        renderer.process_block(&dry, &mut l, &mut r);
        renderer.reset();
        // After reset, a silent input yields silence (no leftover tail).
        dry = [0.0; 4];
        renderer.process_block(&dry, &mut l, &mut r);
        for i in 0..4 {
            assert!(approx(l[i], 0.0, 1e-9));
            assert!(approx(r[i], 0.0, 1e-9));
        }
    }
}
