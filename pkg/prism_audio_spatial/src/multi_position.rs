//! Multi-position sources: mapping one *logical* sound to several world-space
//! points and folding them into a single control-rate spatialisation.
//!
//! Some sounds are not points. A river, a motorway, a crackling fire wall, or a
//! crowd occupies an *extent*: it should be heard from several directions at
//! once and it should not "snap" to a single spot as the listener moves along
//! it. The classic engine answer (aligned with Wwise Multi-Position) is to give
//! one voice several spatial positions and combine them, rather than spawning a
//! voice per point.
//!
//! This module composes the per-source orchestration in
//! [`spatializer`](crate::spatializer) over a set of positions and reduces the
//! results to one [`SpatialParams`] the caller can drive a single voice with.
//! Three [combination modes](MultiPositionMode) trade cost for realism:
//!
//! * [`Nearest`](MultiPositionMode::Nearest) — snap to the closest point.
//! * [`Blend`](MultiPositionMode::Blend) — one representative image at the
//!   gain-weighted mean direction.
//! * [`Envelop`](MultiPositionMode::Envelop) — every point contributes at once,
//!   summing energy and widening the [spread](crate::spread) to span the
//!   angular extent of the points, so the source wraps around the listener.
//!
//! # Control rate, not audio rate
//!
//! [`resolve_multi`] is a pure function evaluated once per control block, just
//! like [`resolve`]. It allocates nothing (all per-position scratch lives in
//! fixed stack buffers sized for [`MAX_POSITIONS`]), locks nothing, and cannot
//! panic, so it is safe to call from a device callback. Positions beyond
//! [`MAX_POSITIONS`] are ignored rather than triggering growth.
//!
//! # Combination policy
//!
//! Direction is always combined on the **unit direction vectors** in the
//! listener frame (never on raw azimuth/elevation), so a source split across
//! front and back does not average to a bogus sideways angle. Each position is
//! weighted by its own resolved direct gain, so nearer/louder points pull the
//! combined image toward themselves. When exactly opposed directions cancel (a
//! source split evenly to the left and right of the listener), the combined
//! vector is null and the image falls back to the nearest point's direction so
//! it stays well defined. Pitch and the direct low-pass corner are
//! gain-weighted means; the wet send takes the strongest contribution. Loudness
//! combines per mode: [`Blend`](MultiPositionMode::Blend) keeps the single
//! loudest contribution (one image, not louder for being averaged), while
//! [`Envelop`](MultiPositionMode::Envelop) sums power across the (assumed
//! uncorrelated) points and clamps to unity.
//!
//! # Determinism
//!
//! Every angle and length routes through [`bevy_math::ops`] rather than `f32`
//! intrinsics, so a given configuration reduces to bit-identical parameters
//! across targets and can be golden-compared.
//!
//! # Provenance
//!
//! This module is engine-agnostic and contains **no Unreal Engine, Unity,
//! Godot, Wwise, or FMOD source or derived code**. Representing a distributed
//! source by several positions and combining them by nearest/weighted/all is a
//! standard, publicly documented spatial-audio technique.

use bevy_math::{Vec3, ops};
use prism_audio_core::math::{MIN_AUDIBLE_GAIN, Sample};

use crate::geometry::{Emitter, Listener, LocalSource};
use crate::occlusion::OcclusionFactors;
use crate::spatializer::{SourceDescriptor, SpatialParams, resolve};
use crate::spread::SpreadParams;

use core::f32::consts::PI;

/// Largest number of positions a single logical source may map to. Positions
/// beyond this are ignored so the combination stays allocation free.
pub const MAX_POSITIONS: usize = 16;

/// One world-space position of a multi-position source, together with the
/// occlusion state on the path from that point to the listener.
///
/// Plain data: cheap to copy and, with the `serialize` feature,
/// (de)serializable. Bundling the [`OcclusionFactors`] with the [`Emitter`]
/// keeps [`resolve_multi`] free of parallel slices that could disagree in
/// length.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct PositionInput {
    /// World-space pose and velocity of this point of the source.
    pub emitter: Emitter,
    /// Occlusion/obstruction blocking factors on the path from this point to
    /// the listener. Use [`OcclusionFactors::OPEN`] for a clear line of sight.
    pub factors: OcclusionFactors,
}

impl PositionInput {
    /// Creates a position input with an explicit occlusion state.
    #[must_use]
    #[inline]
    pub fn new(emitter: Emitter, factors: OcclusionFactors) -> Self {
        Self { emitter, factors }
    }

    /// Creates a fully audible (open line of sight) position input.
    #[must_use]
    #[inline]
    pub fn open(emitter: Emitter) -> Self {
        Self { emitter, factors: OcclusionFactors::OPEN }
    }
}

/// How the positions of a multi-position source combine into one image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum MultiPositionMode {
    /// Use only the closest position. Cheapest and fully localised: the source
    /// snaps to whichever point is nearest, ignoring the rest. Good for sparse
    /// alternatives where only one point is ever relevant at a time.
    Nearest,
    /// One representative image at the gain-weighted mean direction, taking the
    /// loudness of the single strongest point. Models a source you localise to
    /// one place but whose exact spot is an average of a few emitters (a
    /// machine with several vents, a chord of nearby pipes).
    #[default]
    Blend,
    /// Every position contributes at once: energy sums across the points
    /// (clamped to unity), the image sits at the gain-weighted centroid, and
    /// the [spread](crate::spread) widens to span the angular extent of the
    /// points so the source wraps around the listener. Models large,
    /// distributed sources (a river, a crowd, a wall of fire).
    Envelop,
}

/// Reduces several world-space positions of one logical source to a single
/// control-rate [`SpatialParams`].
///
/// Each position in `positions` (up to [`MAX_POSITIONS`]; extras are ignored)
/// is spatialised with the shared `descriptor` against the live `listener`,
/// then combined per `mode`. `sample_rate` is the device rate, used only to
/// clamp the air-absorption corner to Nyquist.
///
/// The result is a pure function of the inputs (no audio state, no allocation,
/// no locks, no panics), so it is safe to call from a real-time thread. When
/// `positions` is empty the source is silent and centred.
///
/// # Examples
///
/// ```
/// # use bevy_math::Vec3;
/// # use prism_audio_spatial::geometry::{Emitter, Listener};
/// # use prism_audio_spatial::multi_position::{
/// #     resolve_multi, MultiPositionMode, PositionInput,
/// # };
/// # use prism_audio_spatial::spatializer::SourceDescriptor;
/// let listener = Listener::default();
/// // A river heard from the left and the right at once.
/// let positions = [
///     PositionInput::open(Emitter::point(Vec3::new(-8.0, 0.0, -8.0), Vec3::ZERO)),
///     PositionInput::open(Emitter::point(Vec3::new(8.0, 0.0, -8.0), Vec3::ZERO)),
/// ];
/// let params = resolve_multi(
///     &listener,
///     &positions,
///     &SourceDescriptor::default(),
///     48_000,
///     MultiPositionMode::Envelop,
/// );
/// // Symmetric points straddle the listener, so the image sits ahead and wraps.
/// assert!(params.azimuth.abs() < 1e-3);
/// assert!(params.spread.spread > 0.0);
/// ```
#[must_use]
pub fn resolve_multi(
    listener: &Listener,
    positions: &[PositionInput],
    descriptor: &SourceDescriptor,
    sample_rate: u32,
    mode: MultiPositionMode,
) -> SpatialParams {
    let count = positions.len().min(MAX_POSITIONS);
    if count == 0 {
        return silent(sample_rate);
    }

    // Fixed stack scratch: per-position resolved params, local directions, and
    // weights. No allocation regardless of position count.
    let mut params: [SpatialParams; MAX_POSITIONS] = [silent(sample_rate); MAX_POSITIONS];
    let mut dirs: [Vec3; MAX_POSITIONS] = [Vec3::NEG_Z; MAX_POSITIONS];
    let mut weights: [Sample; MAX_POSITIONS] = [0.0; MAX_POSITIONS];

    let mut nearest_idx = 0usize;
    let mut nearest_distance = Sample::INFINITY;

    for (i, input) in positions.iter().take(count).enumerate() {
        let local: LocalSource = listener.localize(&input.emitter);
        params[i] = resolve(listener, &input.emitter, descriptor, input.factors, sample_rate);
        dirs[i] = local.direction;
        weights[i] = params[i].direct_gain;
        if local.distance < nearest_distance {
            nearest_distance = local.distance;
            nearest_idx = i;
        }
    }

    match mode {
        MultiPositionMode::Nearest => params[nearest_idx],
        MultiPositionMode::Blend => {
            combine(&params[..count], &dirs[..count], &weights[..count], nearest_idx, false)
        }
        MultiPositionMode::Envelop => {
            combine(&params[..count], &dirs[..count], &weights[..count], nearest_idx, true)
        }
    }
}

/// Silent, centred parameters used for an empty source and as scratch fill.
#[must_use]
#[inline]
fn silent(sample_rate: u32) -> SpatialParams {
    SpatialParams {
        direct_gain: 0.0,
        pitch_ratio: 1.0,
        azimuth: 0.0,
        elevation: 0.0,
        direct_cutoff_hz: (sample_rate as Sample) * 0.5,
        wet_gain: 0.0,
        spread: SpreadParams::POINT,
    }
}

/// Combines resolved per-position params, directions, and weights into one
/// image. `envelop` selects power-summed loudness plus angular-extent spread;
/// otherwise the strongest single contribution and a gain-weighted spread are
/// used. `fallback` is the position whose image is used when every weight is
/// below audibility (so direction stays well defined).
#[must_use]
fn combine(
    params: &[SpatialParams],
    dirs: &[Vec3],
    weights: &[Sample],
    fallback: usize,
    envelop: bool,
) -> SpatialParams {
    let mut total_weight = 0.0;
    let mut dir_sum = Vec3::ZERO;
    let mut pitch_sum = 0.0;
    let mut cutoff_sum = 0.0;
    let mut spread_sum = 0.0;
    let mut focus_sum = 0.0;
    let mut max_gain = 0.0;
    let mut max_wet = 0.0;
    let mut power_sum = 0.0;

    for i in 0..params.len() {
        let w = weights[i];
        let p = &params[i];
        total_weight += w;
        dir_sum += dirs[i] * w;
        pitch_sum += p.pitch_ratio * w;
        cutoff_sum += p.direct_cutoff_hz * w;
        spread_sum += p.spread.spread * w;
        focus_sum += p.spread.focus * w;
        power_sum += p.direct_gain * p.direct_gain;
        if p.direct_gain > max_gain {
            max_gain = p.direct_gain;
        }
        if p.wet_gain > max_wet {
            max_wet = p.wet_gain;
        }
    }

    // Direction and gain-weighted means fall back to the reference position when
    // the source is effectively silent, so the image never jumps to a
    // degenerate direction.
    let (direction, pitch_ratio, cutoff, base_spread, focus) = if total_weight > MIN_AUDIBLE_GAIN {
        (
            normalize_or(dir_sum, dirs[fallback]),
            pitch_sum / total_weight,
            cutoff_sum / total_weight,
            spread_sum / total_weight,
            focus_sum / total_weight,
        )
    } else {
        let p = &params[fallback];
        (
            dirs[fallback],
            p.pitch_ratio,
            p.direct_cutoff_hz,
            p.spread.spread,
            p.spread.focus,
        )
    };

    let direct_gain = if envelop {
        // Uncorrelated points add in power; clamp to unity.
        ops::sqrt(power_sum).min(1.0)
    } else {
        max_gain
    };

    // Envelopment widens the arc to span how far the widest point sits from the
    // combined direction; a single point (or a tight cluster) leaves it alone.
    let (spread, half_width) = if envelop {
        let extent = angular_extent(dirs, weights, direction);
        let extent_spread = (extent / PI).clamp(0.0, 1.0);
        let s = base_spread.max(extent_spread);
        (s, s * PI)
    } else {
        (base_spread, base_spread * PI)
    };

    let local = LocalSource { direction, distance: 0.0, radial_velocity: 0.0 };

    SpatialParams {
        direct_gain,
        pitch_ratio,
        azimuth: local.azimuth(),
        elevation: local.elevation(),
        direct_cutoff_hz: cutoff,
        wet_gain: max_wet,
        spread: SpreadParams { spread, focus, half_width },
    }
}

/// Largest gain-weighted angle (radians) between any contributing direction and
/// the combined `centre` direction. Silent points do not widen the arc.
#[must_use]
fn angular_extent(dirs: &[Vec3], weights: &[Sample], centre: Vec3) -> Sample {
    let mut extent = 0.0;
    for i in 0..dirs.len() {
        if weights[i] <= MIN_AUDIBLE_GAIN {
            continue;
        }
        let cos_angle = dirs[i].dot(centre).clamp(-1.0, 1.0);
        let angle = ops::acos(cos_angle);
        if angle > extent {
            extent = angle;
        }
    }
    extent
}

/// Normalises `v`, falling back to `fallback` (assumed unit) when `v` is too
/// short to have a well-defined direction. Deterministic.
#[must_use]
#[inline]
fn normalize_or(v: Vec3, fallback: Vec3) -> Vec3 {
    let len = ops::sqrt(v.dot(v));
    if len <= MIN_AUDIBLE_GAIN { fallback } else { v / len }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 48_000;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    fn front(z: f32) -> PositionInput {
        PositionInput::open(Emitter::point(Vec3::new(0.0, 0.0, z), Vec3::ZERO))
    }

    #[test]
    fn empty_source_is_silent_and_centred() {
        let params = resolve_multi(
            &Listener::default(),
            &[],
            &SourceDescriptor::default(),
            SR,
            MultiPositionMode::Blend,
        );
        assert_eq!(params.direct_gain, 0.0);
        assert!(approx(params.azimuth, 0.0, 1e-6));
        assert!(approx(params.pitch_ratio, 1.0, 1e-6));
    }

    #[test]
    fn single_position_matches_resolve() {
        let listener = Listener::default();
        let descriptor = SourceDescriptor::default();
        let emitter = Emitter::point(Vec3::new(2.0, 0.0, -5.0), Vec3::ZERO);
        let single = resolve(&listener, &emitter, &descriptor, OcclusionFactors::OPEN, SR);
        for mode in [
            MultiPositionMode::Nearest,
            MultiPositionMode::Blend,
            MultiPositionMode::Envelop,
        ] {
            let multi = resolve_multi(
                &listener,
                &[PositionInput::open(emitter)],
                &descriptor,
                SR,
                mode,
            );
            assert!(approx(multi.azimuth, single.azimuth, 1e-4));
            assert!(approx(multi.elevation, single.elevation, 1e-4));
            assert!(approx(multi.direct_gain, single.direct_gain, 1e-4));
            assert!(approx(multi.pitch_ratio, single.pitch_ratio, 1e-4));
        }
    }

    #[test]
    fn nearest_picks_the_closest_point() {
        let listener = Listener::default();
        let descriptor = SourceDescriptor::default();
        let near = Emitter::point(Vec3::new(3.0, 0.0, 0.0), Vec3::ZERO); // to the right
        let far = Emitter::point(Vec3::new(0.0, 0.0, -50.0), Vec3::ZERO); // far ahead
        let near_only = resolve(&listener, &near, &descriptor, OcclusionFactors::OPEN, SR);
        let params = resolve_multi(
            &listener,
            &[PositionInput::open(far), PositionInput::open(near)],
            &descriptor,
            SR,
            MultiPositionMode::Nearest,
        );
        // Should equal spatialising the near (right) point alone.
        assert!(approx(params.azimuth, near_only.azimuth, 1e-4));
        assert!(approx(params.direct_gain, near_only.direct_gain, 1e-4));
    }

    #[test]
    fn symmetric_points_centre_the_blend() {
        // Equal points to the left and right must average to straight ahead.
        let listener = Listener::default();
        let left = PositionInput::open(Emitter::point(Vec3::new(-6.0, 0.0, -6.0), Vec3::ZERO));
        let right = PositionInput::open(Emitter::point(Vec3::new(6.0, 0.0, -6.0), Vec3::ZERO));
        let params = resolve_multi(
            &listener,
            &[left, right],
            &SourceDescriptor::default(),
            SR,
            MultiPositionMode::Blend,
        );
        assert!(approx(params.azimuth, 0.0, 1e-3));
    }

    #[test]
    fn blend_pulls_toward_the_louder_point() {
        // A near (loud) right point and a far (quiet) left point: the weighted
        // image should sit to the right of centre.
        let listener = Listener::default();
        let loud_right = PositionInput::open(Emitter::point(Vec3::new(2.0, 0.0, 0.0), Vec3::ZERO));
        let quiet_left = PositionInput::open(Emitter::point(Vec3::new(-40.0, 0.0, 0.0), Vec3::ZERO));
        let params = resolve_multi(
            &listener,
            &[loud_right, quiet_left],
            &SourceDescriptor::default(),
            SR,
            MultiPositionMode::Blend,
        );
        assert!(params.azimuth > 0.0);
    }

    #[test]
    fn envelop_sums_power_and_widens_spread() {
        // Two audible points straddling the listener: envelop is at least as
        // loud as either alone and produces a non-zero spread, while blend of
        // the same points keeps the loudest single gain and a point image.
        let listener = Listener::default();
        let left = PositionInput::open(Emitter::point(Vec3::new(-4.0, 0.0, -4.0), Vec3::ZERO));
        let right = PositionInput::open(Emitter::point(Vec3::new(4.0, 0.0, -4.0), Vec3::ZERO));
        let one = resolve(
            &listener,
            &Emitter::point(Vec3::new(-4.0, 0.0, -4.0), Vec3::ZERO),
            &SourceDescriptor::default(),
            OcclusionFactors::OPEN,
            SR,
        );
        let envelop = resolve_multi(
            &listener,
            &[left, right],
            &SourceDescriptor::default(),
            SR,
            MultiPositionMode::Envelop,
        );
        let blend = resolve_multi(
            &listener,
            &[left, right],
            &SourceDescriptor::default(),
            SR,
            MultiPositionMode::Blend,
        );
        assert!(envelop.direct_gain >= one.direct_gain);
        assert!(envelop.direct_gain <= 1.0);
        // Points 180 deg apart around the listener widen the arc substantially.
        assert!(envelop.spread.spread > 0.0);
        assert!(envelop.spread.half_width > 0.0);
        // Blend keeps a representative (loudest) gain, not the power sum.
        assert!(approx(blend.direct_gain, one.direct_gain, 1e-4));
    }

    #[test]
    fn positions_beyond_cap_are_ignored() {
        // More than MAX_POSITIONS entries must not panic or allocate; the extra
        // ones are simply dropped.
        let listener = Listener::default();
        let mut many = [front(-5.0); MAX_POSITIONS + 4];
        // Make one of the *ignored* tail entries wildly loud/near; it must not
        // affect the result because it is past the cap.
        many[MAX_POSITIONS] = front(-0.1);
        let params = resolve_multi(
            &listener,
            &many,
            &SourceDescriptor::default(),
            SR,
            MultiPositionMode::Nearest,
        );
        let front_only = resolve(
            &listener,
            &Emitter::point(Vec3::new(0.0, 0.0, -5.0), Vec3::ZERO),
            &SourceDescriptor::default(),
            OcclusionFactors::OPEN,
            SR,
        );
        assert!(approx(params.direct_gain, front_only.direct_gain, 1e-4));
    }

    #[test]
    fn silent_points_keep_direction_well_defined() {
        // Two points at (and beyond) a linear model's max distance: every gain
        // collapses to exactly zero, so the combined weight is sub-audible. The
        // image must still resolve to a finite direction (from the reference
        // position) rather than NaN.
        use crate::attenuation::{Attenuation, DistanceModel};
        let listener = Listener::default();
        let descriptor = SourceDescriptor {
            attenuation: Attenuation::new(DistanceModel::Linear, 1.0, 10.0, 1.0),
            ..SourceDescriptor::default()
        };
        // Distance == max_distance (10 m) => linear gain hits zero.
        let a = PositionInput::open(Emitter::point(Vec3::new(-10.0, 0.0, 0.0), Vec3::ZERO));
        let b = PositionInput::open(Emitter::point(Vec3::new(10.0, 0.0, 0.0), Vec3::ZERO));
        for mode in [MultiPositionMode::Blend, MultiPositionMode::Envelop] {
            let params = resolve_multi(&listener, &[a, b], &descriptor, SR, mode);
            assert!(params.azimuth.is_finite());
            assert!(params.elevation.is_finite());
            assert!(approx(params.direct_gain, 0.0, 1e-6));
        }
    }
}
