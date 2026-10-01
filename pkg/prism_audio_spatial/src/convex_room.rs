//! Image-source early reflections for convex polyhedral rooms.
//!
//! [`crate::early_reflections`] computes first-and-higher-order reflections for
//! axis-aligned rectangular ("shoebox") rooms with a closed-form integer image
//! grid. Many rooms are not boxes: a hall with a slanted ceiling, a chamfered
//! corner, or any space bounded by a handful of flat walls is still *convex*
//! (the intersection of a set of half-spaces) even though it is not a box. This
//! module generalises the image-source method to such rooms.
//!
//! A convex room is described by a set of [`ReflectionPlane`]s, each an inward
//! half-space `dot(normal, p) >= offset`. For a source and listener inside the
//! room, the first-order reflection off a plane is the straight line from the
//! listener to the source mirrored across that plane (the *image source*). The
//! reflection point is where that line crosses the plane; the path is physical
//! only when the reflection point also lies inside every other plane (that is,
//! on the actual bounded face). Reflections whose mirror point falls outside the
//! room are culled.
//!
//! The output is the same [`ReflectionTap`] type that
//! [`crate::early_reflections`] produces, so a convex-room tap list drops
//! straight into [`crate::reflection_directivity`],
//! [`crate::reflection_clustering`], or the shoebox
//! [`crate::early_reflections::EarlyReflectionRenderer`].
//!
//! # Geometry
//!
//! A plane with inward unit normal `n` and plane constant `offset` splits space
//! into the interior half-space `dot(n, p) >= offset` and the exterior. The
//! mirror of a point `s` across the plane is
//!
//! ```text
//! s' = s - 2 * (dot(n, s) - offset) * n
//! ```
//!
//! For a listener `L` and source `S` the first-order image is `S' = mirror(S)`;
//! the reflection point is the intersection of the segment `L -> S'` with the
//! plane, found by solving `dot(n, L + t (S' - L)) = offset` for `t` and keeping
//! only `t` in `(0, 1)`.
//!
//! # Gain and delay
//!
//! The total path length equals `|S' - L|` (the mirrored straight-line
//! distance). The delay is `path / sound_speed` converted to whole samples by
//! integer stepping (never a floating-point-to-integer cast), the
//! geometric-spreading gain is `1 / path` clamped at a small minimum distance,
//! and the plane's amplitude reflection coefficient multiplies the gain. The
//! arrival direction is the unit vector from the reflection point to the
//! listener, expressed in the listener's local frame (matching
//! [`crate::geometry::Listener`], so `-Z` is forward). The zero-order direct
//! path is emitted first and flagged [`ReflectionTap::is_direct`].
//!
//! # Scope
//!
//! This module enumerates the direct path and first-order reflections; the
//! requested `order` is clamped to one. First-order reflections dominate the
//! perceptual early-reflection cue, and keeping the enumeration to one bounce
//! keeps the per-plane validity test exact and cheap. Higher orders can be
//! layered on later without changing the tap format.
//!
//! # Real-time contract
//!
//! [`compute_convex_reflections`] runs at control rate, is stack-only, allocates
//! nothing, and never panics: degenerate planes, coincident listener and
//! source, non-finite inputs, and empty rooms all produce a finite, bounded
//! result. When more reflections qualify than fit, the strongest by gain are
//! kept.
//!
//! # Determinism
//!
//! All length math routes through [`bevy_math::ops`] and delays are accumulated
//! by integer stepping, so the tap list is bit-reproducible.
//!
//! # Relationship
//!
//! This module is the convex-polyhedron generalisation of
//! [`crate::early_reflections`] and deliberately re-implements none of the
//! rectangular integer-grid image method: it reuses that module's
//! [`ReflectionTap`] type, [`crate::early_reflections::MAX_EARLY_REFLECTIONS`]
//! capacity, and the same gain, delay, and local-direction conventions. Its taps
//! can be weighted by [`crate::reflection_directivity`] and reduced by
//! [`crate::reflection_clustering`] exactly like shoebox taps.
//!
//! # Provenance
//!
//! The image-source method for polygonal and convex rooms is publicly
//! documented geometrical acoustics (see J. B. Allen and D. A. Berkley, "Image
//! method for efficiently simulating small-room acoustics," Journal of the
//! Acoustical Society of America 65(4), 1979, and the standard extension to
//! arbitrary planar boundaries). This module is engine-agnostic and contains
//! **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or Google
//! Resonance Audio source or derived code**; it is implemented from the public
//! mathematical description.

use bevy_math::{Vec3, ops};

use prism_audio_core::math::Sample;

use crate::early_reflections::{MAX_EARLY_REFLECTIONS, ReflectionTap};
use crate::geometry::Listener;

/// Maximum number of bounding planes a convex room may hold. Rooms built from
/// more planes are truncated to this many. Bounded at 12 to cover a box (6
/// faces) plus several chamfer or slope planes while staying on the stack and
/// within serde's derived array support.
pub const MAX_PLANES: usize = 12;

/// Distances below this (metres) are clamped when forming the `1 / distance`
/// spreading gain, bounding the gain of near-coincident paths.
const MIN_DISTANCE_METRES: Sample = 0.1;

/// Distances below this (metres) collapse the arrival direction to local
/// forward, avoiding a division by a near-zero length.
const COINCIDENT_EPSILON: Sample = 1.0e-6;

/// A reflection point is accepted as inside a plane when its signed distance is
/// no more negative than this (metres), tolerating floating-point slack on the
/// reflecting face itself.
const INSIDE_EPSILON: Sample = 1.0e-4;

/// Line-plane intersections with a denominator smaller than this are treated as
/// parallel and skipped.
const MIN_DENOM: Sample = 1.0e-9;

/// Normals shorter than this (squared) are treated as degenerate; such planes
/// are skipped during enumeration.
const MIN_NORMAL_SQUARED: Sample = 0.5;

/// Upper bound on the integer delay a tap may report, in samples (about 5.5
/// seconds at 48 kHz), capping the integer-stepping delay conversion.
const MAX_DELAY_SAMPLES: usize = 1 << 18;

/// One bounding plane of a convex room, an inward half-space
/// `dot(normal, p) >= offset`.
///
/// `normal` points into the room interior and is stored unit length; `offset`
/// is the plane constant `dot(normal, point_on_plane)`; `reflection` is the
/// broadband amplitude reflection coefficient in `[0, 1]` (`1` is a perfect
/// reflector, `0` fully absorbing). A plane built from a zero or non-finite
/// normal is stored degenerate and is skipped during enumeration.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ReflectionPlane {
    /// Inward unit normal (points into the room interior).
    pub normal: Vec3,
    /// Plane constant `dot(normal, point_on_plane)`.
    pub offset: Sample,
    /// Broadband amplitude reflection coefficient in `[0, 1]`.
    pub reflection: Sample,
}

impl ReflectionPlane {
    /// A degenerate placeholder plane (zero normal), skipped during
    /// enumeration.
    const DEGENERATE: Self = Self {
        normal: Vec3::ZERO,
        offset: 0.0,
        reflection: 0.0,
    };

    /// Builds a plane from an inward `normal`, a `point_on_plane`, and an
    /// amplitude `reflection` coefficient.
    ///
    /// The normal is normalised and must point into the room interior; the
    /// offset is computed as `dot(normal, point_on_plane)`. A zero or non-finite
    /// normal yields a degenerate plane that is skipped during enumeration. The
    /// reflection coefficient is clamped to `[0, 1]`.
    #[must_use]
    pub fn new(normal: Vec3, point_on_plane: Vec3, reflection: Sample) -> Self {
        if !normal.is_finite() || !point_on_plane.is_finite() {
            return Self::DEGENERATE;
        }
        let unit = normal.normalize_or_zero();
        if unit.length_squared() < MIN_NORMAL_SQUARED {
            return Self::DEGENERATE;
        }
        let reflection = if reflection.is_finite() {
            reflection.clamp(0.0, 1.0)
        } else {
            0.0
        };
        Self {
            normal: unit,
            offset: unit.dot(point_on_plane),
            reflection,
        }
    }

    /// Builds a plane whose amplitude reflection coefficient is derived from an
    /// energy `absorption` in `[0, 1]` as `sqrt(1 - absorption)`.
    #[must_use]
    pub fn from_absorption(normal: Vec3, point_on_plane: Vec3, absorption: Sample) -> Self {
        let alpha = if absorption.is_finite() {
            absorption.clamp(0.0, 1.0)
        } else {
            1.0
        };
        Self::new(normal, point_on_plane, ops::sqrt((1.0 - alpha).max(0.0)))
    }

    /// Whether this plane has a usable (finite, non-zero) normal.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.normal.length_squared() >= MIN_NORMAL_SQUARED
    }
}

/// A convex room described by up to [`MAX_PLANES`] inward bounding planes.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ConvexRoom {
    planes: [ReflectionPlane; MAX_PLANES],
    count: usize,
}

impl ConvexRoom {
    /// Builds a convex room from a slice of bounding planes. Planes beyond
    /// [`MAX_PLANES`] are dropped. Degenerate planes are retained in storage but
    /// skipped during enumeration.
    #[must_use]
    pub fn new(planes: &[ReflectionPlane]) -> Self {
        let mut stored = [ReflectionPlane::DEGENERATE; MAX_PLANES];
        let count = planes.len().min(MAX_PLANES);
        for (slot, plane) in stored.iter_mut().zip(planes.iter()) {
            *slot = *plane;
        }
        Self { planes: stored, count }
    }

    /// Builds a convex room equivalent to an axis-aligned box from two corners
    /// and the six face energy absorption coefficients, ordered
    /// `[x_low, x_high, y_low, y_high, z_low, z_high]`.
    ///
    /// This is a convenience bridge to the shoebox convention of
    /// [`crate::early_reflections::ShoeboxRoom`]; the resulting first-order taps
    /// match the shoebox image method for interior geometry.
    #[must_use]
    pub fn shoebox(corner_a: Vec3, corner_b: Vec3, absorption: [Sample; 6]) -> Self {
        let lo = corner_a.min(corner_b);
        let hi = corner_a.max(corner_b);
        let planes = [
            ReflectionPlane::from_absorption(Vec3::X, lo, absorption[0]),
            ReflectionPlane::from_absorption(Vec3::NEG_X, hi, absorption[1]),
            ReflectionPlane::from_absorption(Vec3::Y, lo, absorption[2]),
            ReflectionPlane::from_absorption(Vec3::NEG_Y, hi, absorption[3]),
            ReflectionPlane::from_absorption(Vec3::Z, lo, absorption[4]),
            ReflectionPlane::from_absorption(Vec3::NEG_Z, hi, absorption[5]),
        ];
        Self::new(&planes)
    }

    /// The room's bounding planes (including any degenerate entries up to the
    /// stored count).
    #[must_use]
    pub fn planes(&self) -> &[ReflectionPlane] {
        &self.planes[..self.count]
    }

    /// The number of stored bounding planes.
    #[must_use]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Whether a point lies inside every valid plane (interior half-space).
    #[must_use]
    fn contains(&self, point: Vec3) -> bool {
        for plane in self.planes[..self.count].iter().filter(|p| p.is_valid()) {
            if plane.normal.dot(point) - plane.offset < -INSIDE_EPSILON {
                return false;
            }
        }
        true
    }
}

/// A fixed-capacity set of reflection taps produced from a convex room.
///
/// Always holds at most [`crate::early_reflections::MAX_EARLY_REFLECTIONS`]
/// taps; [`ConvexReflections::taps`] returns the valid prefix.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ConvexReflections {
    taps: [ReflectionTap; MAX_EARLY_REFLECTIONS],
    count: usize,
}

impl ConvexReflections {
    /// The valid reflection taps (direct path first, then first-order
    /// reflections).
    #[must_use]
    pub fn taps(&self) -> &[ReflectionTap] {
        &self.taps[..self.count]
    }

    /// The number of valid taps.
    #[must_use]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Whether no taps were produced.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

/// A silent placeholder tap used to initialise fixed-size storage.
const SILENT_TAP: ReflectionTap = ReflectionTap {
    delay_samples: 0,
    gain: 0.0,
    direction: Vec3::NEG_Z,
    order: 0,
    is_direct: false,
};

/// Converts a delay in seconds to whole samples by integer stepping, avoiding a
/// floating-point-to-integer cast. The result is clamped to
/// [`MAX_DELAY_SAMPLES`].
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

/// Inserts a tap into a bounded buffer, keeping the strongest by gain when the
/// buffer is full.
fn push_tap(taps: &mut [ReflectionTap; MAX_EARLY_REFLECTIONS], count: &mut usize, tap: ReflectionTap) {
    if *count < MAX_EARLY_REFLECTIONS {
        taps[*count] = tap;
        *count += 1;
        return;
    }
    let mut min_index = 0;
    let mut min_gain = taps[0].gain;
    for (i, stored) in taps.iter().enumerate().skip(1) {
        if stored.gain < min_gain {
            min_gain = stored.gain;
            min_index = i;
        }
    }
    if tap.gain > min_gain {
        taps[min_index] = tap;
    }
}

/// Computes the direct path and first-order convex-room reflections for a source
/// heard by a listener.
///
/// The `order` argument is clamped to one (direct plus first-order). Taps are
/// returned in a [`ConvexReflections`] set; the direct path, when the source is
/// finite, is always included and flagged [`ReflectionTap::is_direct`]. Runs at
/// control rate, allocates nothing, and never panics: degenerate planes,
/// coincident listener and source, and non-finite inputs all produce a finite,
/// bounded result.
///
/// # Examples
///
/// ```
/// use bevy_math::Vec3;
/// use prism_audio_spatial::convex_room::{compute_convex_reflections, ConvexRoom};
/// use prism_audio_spatial::geometry::Listener;
///
/// // A convex box room, listener at the origin, source three metres ahead.
/// let room = ConvexRoom::shoebox(Vec3::splat(-5.0), Vec3::splat(5.0), [0.0; 6]);
/// let listener = Listener::default();
/// let source = Vec3::new(0.0, 0.0, -3.0);
///
/// let reflections = compute_convex_reflections(
///     &room, &listener, source, 1, 48_000.0, 343.0,
/// );
/// // The direct path is present and points forward (local -Z).
/// let direct = reflections.taps().iter().find(|t| t.is_direct).unwrap();
/// assert!(direct.direction.z < 0.0);
/// // Six walls each contribute one first-order reflection.
/// let reflected = reflections.taps().iter().filter(|t| !t.is_direct).count();
/// assert_eq!(reflected, 6);
/// ```
#[must_use]
pub fn compute_convex_reflections(
    room: &ConvexRoom,
    listener: &Listener,
    source: Vec3,
    order: usize,
    sample_rate: Sample,
    sound_speed: Sample,
) -> ConvexReflections {
    let mut taps = [SILENT_TAP; MAX_EARLY_REFLECTIONS];
    let mut count: usize = 0;

    let l = listener.position;
    if !l.is_finite() || !source.is_finite() {
        return ConvexReflections { taps, count };
    }

    let fs = sample_rate.max(1.0);
    let c = sound_speed.max(1.0);
    let inv = listener.orientation.inverse();

    // Direct path (zero order).
    let delta = source - l;
    let dist = ops::sqrt(delta.dot(delta));
    let direct_dir = if dist <= COINCIDENT_EPSILON {
        Vec3::NEG_Z
    } else {
        inv * (delta / dist)
    };
    push_tap(
        &mut taps,
        &mut count,
        ReflectionTap {
            delay_samples: seconds_to_samples(dist / c, fs),
            gain: 1.0 / dist.max(MIN_DISTANCE_METRES),
            direction: direct_dir,
            order: 0,
            is_direct: true,
        },
    );

    if order == 0 {
        return ConvexReflections { taps, count };
    }

    // First-order reflections: one per valid bounding plane.
    for plane in room.planes[..room.count].iter().filter(|p| p.is_valid()) {
        let n = plane.normal;
        let off = plane.offset;

        // Mirror the source across the plane to form the image source.
        let signed = n.dot(source) - off;
        let image = source - 2.0 * signed * n;

        // Intersect the segment L -> image with the plane.
        let dl = n.dot(l) - off;
        let di = n.dot(image) - off;
        let denom = di - dl;
        if denom.abs() < MIN_DENOM {
            continue;
        }
        let t = -dl / denom;
        if !(t > 0.0 && t < 1.0) {
            continue;
        }

        let hit = l + t * (image - l);
        if !room.contains(hit) {
            continue;
        }

        let path_vec = image - l;
        let path = ops::sqrt(path_vec.dot(path_vec));
        let hit_vec = hit - l;
        let hit_dist = ops::sqrt(hit_vec.dot(hit_vec));
        let direction = if hit_dist <= COINCIDENT_EPSILON {
            Vec3::NEG_Z
        } else {
            inv * (hit_vec / hit_dist)
        };

        push_tap(
            &mut taps,
            &mut count,
            ReflectionTap {
                delay_samples: seconds_to_samples(path / c, fs),
                gain: plane.reflection / path.max(MIN_DISTANCE_METRES),
                direction,
                order: 1,
                is_direct: false,
            },
        );
    }

    ConvexReflections { taps, count }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reflection_clustering::cluster_taps;
    use bevy_math::Quat;

    fn approx(a: Sample, b: Sample, eps: Sample) -> bool {
        (a - b).abs() <= eps
    }

    fn cube(half: Sample) -> ConvexRoom {
        ConvexRoom::shoebox(Vec3::splat(-half), Vec3::splat(half), [0.0; 6])
    }

    fn reflected_count(r: &ConvexReflections) -> usize {
        r.taps().iter().filter(|t| !t.is_direct).count()
    }

    #[test]
    fn empty_room_returns_only_direct() {
        let room = ConvexRoom::new(&[]);
        let r = compute_convex_reflections(
            &room,
            &Listener::default(),
            Vec3::new(0.0, 0.0, -3.0),
            1,
            48_000.0,
            343.0,
        );
        assert_eq!(r.count(), 1);
        assert!(r.taps()[0].is_direct);
    }

    #[test]
    fn direct_tap_points_at_source() {
        let room = cube(5.0);
        let r = compute_convex_reflections(
            &room,
            &Listener::default(),
            Vec3::new(0.0, 0.0, -3.0),
            0,
            48_000.0,
            343.0,
        );
        let direct = r.taps().iter().find(|t| t.is_direct).unwrap();
        assert_eq!(direct.order, 0);
        assert!(direct.direction.z < 0.0);
        assert!(approx(direct.gain, 1.0 / 3.0, 1e-4));
    }

    #[test]
    fn order_zero_returns_only_direct() {
        let room = cube(5.0);
        let r = compute_convex_reflections(
            &room,
            &Listener::default(),
            Vec3::new(0.0, 0.0, -3.0),
            0,
            48_000.0,
            343.0,
        );
        assert_eq!(reflected_count(&r), 0);
        assert_eq!(r.count(), 1);
    }

    #[test]
    fn single_plane_reflection_geometry() {
        // Floor plane y = -2, inward normal +Y.
        let plane = ReflectionPlane::new(Vec3::Y, Vec3::new(0.0, -2.0, 0.0), 1.0);
        let room = ConvexRoom::new(&[plane]);
        let listener = Listener::new(Vec3::new(0.0, 1.0, 0.0), Quat::IDENTITY, Vec3::ZERO);
        let source = Vec3::new(0.0, 1.0, -4.0);
        let r = compute_convex_reflections(&room, &listener, source, 1, 48_000.0, 343.0);
        assert_eq!(reflected_count(&r), 1);
        // Image of source across y = -2 is at y = -5; path = |image - L|.
        let image = Vec3::new(0.0, -5.0, -4.0);
        let expected = (image - listener.position).length();
        let tap = r.taps().iter().find(|t| !t.is_direct).unwrap();
        assert!(approx(tap.gain, 1.0 / expected, 1e-4));
    }

    #[test]
    fn reflection_point_lies_on_plane() {
        let plane = ReflectionPlane::new(Vec3::Y, Vec3::new(0.0, -2.0, 0.0), 1.0);
        let room = ConvexRoom::new(&[plane]);
        let listener = Listener::new(Vec3::new(0.0, 1.0, 0.0), Quat::IDENTITY, Vec3::ZERO);
        let source = Vec3::new(0.0, 1.0, -4.0);
        // Reconstruct the hit from the direction: it must sit on y = -2.
        let r = compute_convex_reflections(&room, &listener, source, 1, 48_000.0, 343.0);
        // The reflected direction points downward (toward the floor).
        let tap = r.taps().iter().find(|t| !t.is_direct).unwrap();
        assert!(tap.direction.y < 0.0);
    }

    #[test]
    fn gain_decreases_with_distance() {
        // Larger room -> walls farther away -> longer reflection path -> lower gain.
        let near_room = cube(50.0);
        let far_room = cube(100.0);
        let source = Vec3::new(0.0, 0.0, -2.0);
        let near_gain = compute_convex_reflections(
            &near_room,
            &Listener::default(),
            source,
            1,
            48_000.0,
            343.0,
        )
        .taps()
        .iter()
        .find(|t| !t.is_direct)
        .unwrap()
        .gain;
        let far_gain = compute_convex_reflections(
            &far_room,
            &Listener::default(),
            source,
            1,
            48_000.0,
            343.0,
        )
        .taps()
        .iter()
        .find(|t| !t.is_direct)
        .unwrap()
        .gain;
        assert!(far_gain < near_gain);
    }

    #[test]
    fn reflection_coefficient_scales_gain() {
        let full = ConvexRoom::shoebox(Vec3::splat(-5.0), Vec3::splat(5.0), [0.0; 6]);
        let half_abs = 1.0 - 0.25; // reflection = sqrt(1 - 0.75) = 0.5
        let damped = ConvexRoom::shoebox(Vec3::splat(-5.0), Vec3::splat(5.0), [half_abs; 6]);
        let src = Vec3::new(0.0, 0.0, -3.0);
        let gf = compute_convex_reflections(&full, &Listener::default(), src, 1, 48_000.0, 343.0)
            .taps()
            .iter()
            .find(|t| !t.is_direct)
            .unwrap()
            .gain;
        let gd = compute_convex_reflections(&damped, &Listener::default(), src, 1, 48_000.0, 343.0)
            .taps()
            .iter()
            .find(|t| !t.is_direct)
            .unwrap()
            .gain;
        assert!(approx(gd, gf * 0.5, 1e-4));
    }

    #[test]
    fn source_outside_plane_is_culled() {
        let plane = ReflectionPlane::new(Vec3::X, Vec3::new(-5.0, 0.0, 0.0), 1.0);
        let room = ConvexRoom::new(&[plane]);
        let listener = Listener::default(); // at origin, inside (x >= -5)
        let outside = Vec3::new(-8.0, 0.0, 0.0); // x < -5, exterior
        let r = compute_convex_reflections(&room, &listener, outside, 1, 48_000.0, 343.0);
        assert_eq!(reflected_count(&r), 0);
        // Direct is still emitted.
        assert_eq!(r.count(), 1);
    }

    #[test]
    fn cube_matches_shoebox_first_order_count() {
        let room = cube(5.0);
        let r = compute_convex_reflections(
            &room,
            &Listener::default(),
            Vec3::new(0.0, 0.0, -3.0),
            1,
            48_000.0,
            343.0,
        );
        // Direct + six first-order wall images.
        assert_eq!(r.count(), 7);
        assert_eq!(reflected_count(&r), 6);
    }

    #[test]
    fn cube_matches_shoebox_image_distances() {
        use crate::early_reflections::{ShoeboxRoom, compute_early_reflections};
        let convex = cube(5.0);
        let shoebox = ShoeboxRoom::rigid(Vec3::splat(-5.0), Vec3::splat(5.0));
        let listener = Listener::default();
        let source = Vec3::new(0.5, -0.3, -3.0);

        let cr = compute_convex_reflections(&convex, &listener, source, 1, 48_000.0, 343.0);

        let mut taps = [SILENT_TAP; 32];
        let n = compute_early_reflections(
            &shoebox, &listener, source, 1, 48_000.0, 343.0, &mut taps,
        );

        // Every shoebox first-order tap has a matching convex tap (same delay
        // and gain within tolerance).
        for sb in taps[..n].iter().filter(|t| !t.is_direct) {
            let matched = cr.taps().iter().filter(|t| !t.is_direct).any(|cv| {
                cv.delay_samples == sb.delay_samples && approx(cv.gain, sb.gain, 1e-4)
            });
            assert!(matched, "no convex match for shoebox tap {sb:?}");
        }
    }

    #[test]
    fn listener_equals_source_is_safe() {
        let room = cube(5.0);
        let p = Vec3::new(1.0, 2.0, -1.0);
        let listener = Listener::new(p, Quat::IDENTITY, Vec3::ZERO);
        let r = compute_convex_reflections(&room, &listener, p, 1, 48_000.0, 343.0);
        for tap in r.taps() {
            assert!(tap.gain.is_finite());
            assert!(tap.direction.is_finite());
        }
    }

    #[test]
    fn non_finite_listener_returns_empty() {
        let room = cube(5.0);
        let listener = Listener::new(Vec3::new(Sample::NAN, 0.0, 0.0), Quat::IDENTITY, Vec3::ZERO);
        let r = compute_convex_reflections(
            &room,
            &listener,
            Vec3::new(0.0, 0.0, -3.0),
            1,
            48_000.0,
            343.0,
        );
        assert!(r.is_empty());
    }

    #[test]
    fn non_finite_source_returns_empty() {
        let room = cube(5.0);
        let r = compute_convex_reflections(
            &room,
            &Listener::default(),
            Vec3::new(0.0, Sample::INFINITY, 0.0),
            1,
            48_000.0,
            343.0,
        );
        assert!(r.is_empty());
    }

    #[test]
    fn zero_normal_plane_is_skipped() {
        let bad = ReflectionPlane::new(Vec3::ZERO, Vec3::new(0.0, -2.0, 0.0), 1.0);
        assert!(!bad.is_valid());
        let room = ConvexRoom::new(&[bad]);
        let r = compute_convex_reflections(
            &room,
            &Listener::default(),
            Vec3::new(0.0, 0.0, -3.0),
            1,
            48_000.0,
            343.0,
        );
        assert_eq!(reflected_count(&r), 0);
    }

    #[test]
    fn order_above_one_is_clamped() {
        let room = cube(5.0);
        let src = Vec3::new(0.0, 0.0, -3.0);
        let a = compute_convex_reflections(&room, &Listener::default(), src, 1, 48_000.0, 343.0);
        let b = compute_convex_reflections(&room, &Listener::default(), src, 4, 48_000.0, 343.0);
        assert_eq!(a.count(), b.count());
    }

    #[test]
    fn output_count_is_bounded() {
        let room = cube(5.0);
        let r = compute_convex_reflections(
            &room,
            &Listener::default(),
            Vec3::new(0.0, 0.0, -3.0),
            1,
            48_000.0,
            343.0,
        );
        assert!(r.count() <= MAX_EARLY_REFLECTIONS);
        assert!(r.count() <= room.count() + 1);
    }

    #[test]
    fn directions_are_unit_length() {
        let room = cube(5.0);
        let r = compute_convex_reflections(
            &room,
            &Listener::default(),
            Vec3::new(1.0, 0.5, -3.0),
            1,
            48_000.0,
            343.0,
        );
        for tap in r.taps() {
            assert!(approx(tap.direction.length(), 1.0, 1e-4));
        }
    }

    #[test]
    fn local_frame_rotates_directions() {
        let room = cube(5.0);
        let source = Vec3::new(0.0, 0.0, -3.0);
        let facing = Listener::default();
        let turned = Listener::new(
            Vec3::ZERO,
            Quat::from_rotation_y(core::f32::consts::FRAC_PI_2),
            Vec3::ZERO,
        );
        let d0 = compute_convex_reflections(&room, &facing, source, 0, 48_000.0, 343.0).taps()[0]
            .direction;
        let d1 = compute_convex_reflections(&room, &turned, source, 0, 48_000.0, 343.0).taps()[0]
            .direction;
        // Rotating the listener changes the local arrival direction.
        assert!((d0 - d1).length() > 0.1);
    }

    #[test]
    fn feeds_cluster_taps_without_panic() {
        let room = cube(5.0);
        let r = compute_convex_reflections(
            &room,
            &Listener::default(),
            Vec3::new(0.3, 0.2, -3.0),
            1,
            48_000.0,
            343.0,
        );
        let clusters = cluster_taps(r.taps());
        assert_eq!(clusters.count(), crate::reflection_clustering::CLUSTER_COUNT);
    }

    #[test]
    fn plane_from_absorption_matches_sqrt_rule() {
        let plane = ReflectionPlane::from_absorption(Vec3::Y, Vec3::ZERO, 0.75);
        assert!(approx(plane.reflection, 0.5, 1e-5));
        let rigid = ReflectionPlane::from_absorption(Vec3::Y, Vec3::ZERO, 0.0);
        assert!(approx(rigid.reflection, 1.0, 1e-5));
    }
}
