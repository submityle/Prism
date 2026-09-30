//! Angular *spread* and *focus* shaping: how wide a source's image appears
//! around the listener, and how tightly its energy concentrates within that
//! width.
//!
//! A point source panned to a single direction sounds pinned to one spot; that
//! is correct at a distance but wrong up close, where a real sound (a river, a
//! crowd, a nearby machine) wraps around the head. **Spread** models this: a
//! spread of `0` keeps the source a point, while a spread of `1` distributes
//! its energy all the way around the listener. **Focus** then controls how the
//! energy is weighted *within* the spread arc: low focus spreads energy evenly
//! (diffuse), high focus concentrates it toward the source's true direction.
//!
//! # Model
//!
//! Spread is realised by *multi-point spreading*: the source is replaced by a
//! symmetric fan of virtual sub-sources across an arc of half-width
//! `spread * PI` centred on its true azimuth. Each virtual source is panned
//! independently by any [`Panner`] and the results are summed with normalised
//! weights. A focus-controlled raised-cosine window sets those weights, so the
//! same fan smoothly ranges from "evenly enveloping" (focus `0`) to "sharp at
//! the centre" (focus `1`). At spread `0` the arc collapses and every tap lands
//! on the true direction, recovering an exact point source.
//!
//! Because spread typically shrinks with distance (far sources become points,
//! near sources wrap around), [`Spread`] carries a distance curve and
//! [`Spread::resolve`] evaluates it per control block into [`SpreadParams`].
//!
//! # Real-time contract
//!
//! Everything here is **allocation free, lock free, and panic free**. Tap
//! generation writes into a caller-provided slice and gain accumulation uses a
//! fixed stack buffer sized for the largest supported layout, so the whole path
//! is safe on the audio thread.
//!
//! # Determinism
//!
//! The focus window and tap offsets route their transcendental math through
//! [`bevy_math::ops`] (libm) rather than `f32` intrinsics, so spreading is
//! bit-reproducible across targets.
//!
//! # Provenance
//!
//! This module is engine-agnostic and contains **no Unreal Engine, Unity,
//! Godot, Wwise, or FMOD source or derived code**. Multi-point source
//! spreading with a focus-weighted arc is a standard, publicly documented
//! spatial-audio technique.

use core::f32::consts::{FRAC_PI_2, PI};

use bevy_math::ops;
use prism_audio_core::math::Sample;

use crate::panner::Panner;

/// Largest number of virtual sub-sources a spread fan will ever use. Odd so the
/// centre (the true direction) is always sampled exactly.
pub const MAX_SPREAD_TAPS: usize = 9;

/// Largest channel count any supported layout produces (7.1). Sizes the
/// stack scratch buffer used while accumulating spread gains.
const MAX_PAN_CHANNELS: usize = 8;

/// How aggressively full focus concentrates energy toward the arc centre. The
/// focus value in `[0, 1]` scales into a cosine-window exponent in
/// `[0, MAX_FOCUS_POWER]`.
const MAX_FOCUS_POWER: Sample = 6.0;

/// Authoring-time spread/focus descriptor with a distance curve.
///
/// Spread interpolates linearly from [`near_spread`](Spread::near_spread) at
/// [`near_distance`](Spread::near_distance) to [`far_spread`](Spread::far_spread)
/// at [`far_distance`](Spread::far_distance), holding flat outside that range.
/// Plain data: cheap to copy and, with the `serialize` feature,
/// (de)serializable.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Spread {
    /// Distance (m) at or below which spread equals [`near_spread`](Spread::near_spread).
    pub near_distance: Sample,
    /// Distance (m) at or beyond which spread equals [`far_spread`](Spread::far_spread).
    pub far_distance: Sample,
    /// Spread amount in `[0, 1]` at the near distance (`1` = fully enveloping).
    pub near_spread: Sample,
    /// Spread amount in `[0, 1]` at the far distance (`0` = point source).
    pub far_spread: Sample,
    /// Focus in `[0, 1]`: energy concentration within the arc (`0` = even,
    /// `1` = concentrated at the centre).
    pub focus: Sample,
}

impl Default for Spread {
    /// Enveloping up close, collapsing to a point by 20 m, with even
    /// (unfocused) energy across the arc.
    #[inline]
    fn default() -> Self {
        Self {
            near_distance: 1.0,
            far_distance: 20.0,
            near_spread: 1.0,
            far_spread: 0.0,
            focus: 0.0,
        }
    }
}

impl Spread {
    /// Creates a spread descriptor, sanitising its parameters: spreads and
    /// focus are clamped to `[0, 1]`, distances to non-negative, and
    /// `far_distance` is raised to at least `near_distance` so the curve is
    /// never inverted.
    #[must_use]
    #[inline]
    pub fn new(
        near_distance: Sample,
        far_distance: Sample,
        near_spread: Sample,
        far_spread: Sample,
        focus: Sample,
    ) -> Self {
        let near_distance = near_distance.max(0.0);
        let far_distance = far_distance.max(near_distance);
        Self {
            near_distance,
            far_distance,
            near_spread: near_spread.clamp(0.0, 1.0),
            far_spread: far_spread.clamp(0.0, 1.0),
            focus: focus.clamp(0.0, 1.0),
        }
    }

    /// Evaluates the spread curve at `distance` (metres) into concrete DSP
    /// targets. The returned spread is clamped to `[0, 1]`; the arc half-width
    /// is `spread * PI` radians.
    #[must_use]
    pub fn resolve(&self, distance: Sample) -> SpreadParams {
        let span = self.far_distance - self.near_distance;
        let spread = if span <= 0.0 {
            // Degenerate curve: near and far coincide, so pick the near value.
            self.near_spread
        } else {
            let t = ((distance - self.near_distance) / span).clamp(0.0, 1.0);
            self.near_spread + (self.far_spread - self.near_spread) * t
        };
        let spread = spread.clamp(0.0, 1.0);
        SpreadParams {
            spread,
            focus: self.focus,
            half_width: spread * PI,
        }
    }
}

/// Resolved, control-rate spread targets.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpreadParams {
    /// Effective spread in `[0, 1]`.
    pub spread: Sample,
    /// Effective focus in `[0, 1]`.
    pub focus: Sample,
    /// Half-width of the angular arc in radians, in `[0, PI]`.
    pub half_width: Sample,
}

impl SpreadParams {
    /// A point source: zero spread, zero arc.
    pub const POINT: Self = Self {
        spread: 0.0,
        focus: 0.0,
        half_width: 0.0,
    };
}

/// One virtual sub-source in a spread fan.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpreadTap {
    /// Azimuth offset from the true direction in radians (added to the centre
    /// azimuth before panning).
    pub azimuth_offset: Sample,
    /// Normalised weight in `[0, 1]`; the weights across a fan sum to `1`.
    pub azimuth_weight: Sample,
}

/// Fills `out` with a symmetric fan of [`SpreadTap`]s for the given arc
/// `half_width` (radians) and `focus`, returning how many taps were written.
///
/// The number of taps is `out.len()` capped at [`MAX_SPREAD_TAPS`] and forced
/// odd (so the centre tap always samples the true direction). Offsets are
/// spaced evenly across `[-half_width, half_width]`; weights follow a
/// focus-controlled cosine window `cos(u * PI/2)^(focus * MAX_FOCUS_POWER)`
/// (where `u` is the normalised offset in `[-1, 1]`) and are normalised to sum
/// to `1`. With `focus == 0` every tap is equally weighted (maximally
/// diffuse); higher focus concentrates weight at the centre.
///
/// Returns `0` (writing nothing) only when `out` is empty.
#[must_use]
pub fn spread_taps(half_width: Sample, focus: Sample, out: &mut [SpreadTap]) -> usize {
    let cap = out.len().min(MAX_SPREAD_TAPS);
    if cap == 0 {
        return 0;
    }
    // Force an odd count so the centre is sampled exactly.
    let n = if cap.is_multiple_of(2) { cap - 1 } else { cap };
    if n == 1 {
        out[0] = SpreadTap { azimuth_offset: 0.0, azimuth_weight: 1.0 };
        return 1;
    }

    let half_width = half_width.max(0.0);
    let focus = focus.clamp(0.0, 1.0);
    let power = focus * MAX_FOCUS_POWER;
    let mid = (n / 2) as Sample; // e.g. n=5 -> mid=2

    // First pass: offsets + unnormalised weights, accumulating their sum.
    let mut sum = 0.0;
    for (i, tap) in out.iter_mut().take(n).enumerate() {
        let centred = i as Sample - mid; // in [-mid, mid]
        let u = centred / mid; // normalised to [-1, 1]
        let offset = half_width * u;
        // cos(u * PI/2) in [0, 1]; ^0 == 1 gives a flat (diffuse) window.
        let window = ops::cos(u * FRAC_PI_2).max(0.0);
        let weight = ops::powf(window, power);
        tap.azimuth_offset = offset;
        tap.azimuth_weight = weight;
        sum += weight;
    }

    // Second pass: normalise (sum >= 1 because the centre tap contributes 1).
    let inv = if sum > 0.0 { 1.0 / sum } else { 1.0 };
    for tap in out.iter_mut().take(n) {
        tap.azimuth_weight *= inv;
    }
    n
}

/// Pans a source with spread: accumulates the weighted per-channel gains of a
/// [`SpreadTap`] fan into `out`.
///
/// `panner` supplies the per-tap point-panning; `center_azimuth` is the
/// source's true azimuth (radians); `params` selects the arc and focus. `out`
/// is fully overwritten (not added to): every channel it can hold for the
/// panner's layout is set to the spread result. At [`SpreadParams::POINT`] this
/// is identical to a single [`Panner::compute_gains`] call.
///
/// Allocation/lock/panic free: uses a fixed stack scratch buffer sized for the
/// largest supported layout.
pub fn compute_spread_gains(
    panner: &dyn Panner,
    center_azimuth: Sample,
    params: SpreadParams,
    out: &mut [Sample],
) {
    let count = panner.layout().channel_count().min(out.len());
    for slot in out.iter_mut().take(count) {
        *slot = 0.0;
    }

    let mut taps = [SpreadTap { azimuth_offset: 0.0, azimuth_weight: 0.0 }; MAX_SPREAD_TAPS];
    let n = spread_taps(params.half_width, params.focus, &mut taps);

    let mut scratch = [0.0 as Sample; MAX_PAN_CHANNELS];
    for tap in taps.iter().take(n) {
        panner.compute_gains(center_azimuth + tap.azimuth_offset, &mut scratch);
        for c in 0..count {
            out[c] += tap.azimuth_weight * scratch[c];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::panner::VbapPanner;
    use core::f32::consts::FRAC_PI_2;
    use prism_audio_core::buffer::ChannelLayout;

    fn approx(a: Sample, b: Sample, eps: Sample) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn resolve_hits_curve_endpoints() {
        let spread = Spread::default();
        let near = spread.resolve(0.5); // below near_distance
        let far = spread.resolve(100.0); // beyond far_distance
        assert!(approx(near.spread, 1.0, 1e-6));
        assert!(approx(far.spread, 0.0, 1e-6));
        assert!(approx(near.half_width, PI, 1e-5));
        assert!(approx(far.half_width, 0.0, 1e-6));
    }

    #[test]
    fn resolve_is_monotonic_across_the_curve() {
        let spread = Spread::default();
        let a = spread.resolve(2.0).spread;
        let b = spread.resolve(10.0).spread;
        let c = spread.resolve(18.0).spread;
        // Spread decreases with distance for the default (near>far) curve.
        assert!(a > b && b > c);
    }

    #[test]
    fn new_sanitises_parameters() {
        let s = Spread::new(-5.0, -1.0, 2.0, -0.5, 3.0);
        assert!(s.near_distance >= 0.0);
        assert!(s.far_distance >= s.near_distance);
        assert!((0.0..=1.0).contains(&s.near_spread));
        assert!((0.0..=1.0).contains(&s.far_spread));
        assert!((0.0..=1.0).contains(&s.focus));
    }

    #[test]
    fn degenerate_curve_uses_near_value() {
        let s = Spread::new(5.0, 5.0, 0.7, 0.2, 0.0);
        assert!(approx(s.resolve(5.0).spread, 0.7, 1e-6));
        assert!(approx(s.resolve(50.0).spread, 0.7, 1e-6));
    }

    #[test]
    fn single_tap_is_a_point() {
        let mut taps = [SpreadTap { azimuth_offset: 0.0, azimuth_weight: 0.0 }; 1];
        let n = spread_taps(PI, 0.0, &mut taps);
        assert_eq!(n, 1);
        assert!(approx(taps[0].azimuth_offset, 0.0, 1e-6));
        assert!(approx(taps[0].azimuth_weight, 1.0, 1e-6));
    }

    #[test]
    fn taps_are_symmetric_and_sum_to_one() {
        let mut taps = [SpreadTap { azimuth_offset: 0.0, azimuth_weight: 0.0 }; MAX_SPREAD_TAPS];
        let n = spread_taps(1.0, 0.3, &mut taps);
        assert_eq!(n, MAX_SPREAD_TAPS);
        let sum: Sample = taps.iter().take(n).map(|t| t.azimuth_weight).sum();
        assert!(approx(sum, 1.0, 1e-5));
        // Endpoints sit at +/- half_width and the fan is symmetric.
        assert!(approx(taps[0].azimuth_offset, -1.0, 1e-5));
        assert!(approx(taps[n - 1].azimuth_offset, 1.0, 1e-5));
        for k in 0..n / 2 {
            assert!(approx(
                taps[k].azimuth_offset,
                -taps[n - 1 - k].azimuth_offset,
                1e-5
            ));
            assert!(approx(
                taps[k].azimuth_weight,
                taps[n - 1 - k].azimuth_weight,
                1e-5
            ));
        }
    }

    #[test]
    fn even_buffer_is_forced_odd() {
        let mut taps = [SpreadTap { azimuth_offset: 0.0, azimuth_weight: 0.0 }; 4];
        let n = spread_taps(1.0, 0.0, &mut taps);
        assert_eq!(n, 3);
    }

    #[test]
    fn zero_focus_is_uniform_high_focus_is_centred() {
        let mut diffuse = [SpreadTap { azimuth_offset: 0.0, azimuth_weight: 0.0 }; 5];
        let mut focused = [SpreadTap { azimuth_offset: 0.0, azimuth_weight: 0.0 }; 5];
        let nd = spread_taps(1.0, 0.0, &mut diffuse);
        let nf = spread_taps(1.0, 1.0, &mut focused);
        assert_eq!(nd, 5);
        assert_eq!(nf, 5);
        // Diffuse: every tap equally weighted.
        for t in diffuse.iter().take(nd) {
            assert!(approx(t.azimuth_weight, 0.2, 1e-5));
        }
        // Focused: centre outweighs the edges.
        let mid = nf / 2;
        assert!(focused[mid].azimuth_weight > focused[0].azimuth_weight);
        assert!(focused[mid].azimuth_weight > focused[nf - 1].azimuth_weight);
    }

    #[test]
    fn point_spread_equals_plain_pan() {
        let panner = VbapPanner::new(ChannelLayout::Stereo);
        let az = FRAC_PI_2; // hard right
        let mut plain = [0.0; 2];
        panner.compute_gains(az, &mut plain);
        let mut spread = [0.0; 2];
        compute_spread_gains(&panner, az, SpreadParams::POINT, &mut spread);
        assert!(approx(spread[0], plain[0], 1e-5));
        assert!(approx(spread[1], plain[1], 1e-5));
    }

    #[test]
    fn spreading_widens_a_hard_panned_source() {
        let panner = VbapPanner::new(ChannelLayout::Stereo);
        let az = FRAC_PI_2; // hard right: point pan puts ~nothing on the left
        let mut point = [0.0; 2];
        compute_spread_gains(&panner, az, SpreadParams::POINT, &mut point);
        let wide = SpreadParams { spread: 0.8, focus: 0.0, half_width: 0.8 * PI };
        let mut spread = [0.0; 2];
        compute_spread_gains(&panner, az, wide, &mut spread);
        // Left channel (index 0) gains energy as the source widens.
        assert!(spread[0] > point[0]);
    }

    #[test]
    fn spread_gains_are_finite_for_full_envelope() {
        let panner = VbapPanner::new(ChannelLayout::Surround5_1);
        let params = SpreadParams { spread: 1.0, focus: 0.5, half_width: PI };
        let mut out = [0.0; 6];
        compute_spread_gains(&panner, 0.3, params, &mut out);
        for g in out {
            assert!(g.is_finite() && g >= 0.0);
        }
    }
}
