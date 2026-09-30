//! Steerable virtual-microphone beamforming for higher-order Ambisonics.
//!
//! Given a scene-based Ambisonic field (`ACN`/`SN3D`, exactly as produced by
//! [`crate::hoa::encode_hoa`]), this module extracts a single mono "virtual
//! microphone" signal aimed at an arbitrary look direction. The virtual mic is
//! a *modal beamformer*: it weights each Ambisonic degree `n` by a scalar `g_n`
//! and forms an axisymmetric beam whose response, as a function of the angle
//! `gamma` between the source and the look direction, is
//!
//! ```text
//! B(gamma) = ( sum_n g_n * P_n(cos gamma) ) / ( sum_n g_n )
//! ```
//!
//! where `P_n` is the Legendre polynomial. The per-degree weights `g_n` select
//! the classic beam families offered by [`BeamPattern`]. The normalisation
//! `sum_n g_n` fixes the on-axis response to exactly `1.0`, so a unit-amplitude
//! source lying on the look axis is reproduced with unit gain regardless of
//! order or pattern.
//!
//! This complements [`crate::hoa_decode`]: that module renders a field to many
//! loudspeakers, whereas this one collapses a field to a single steerable pickup
//! (spot-mic isolation, up-mixing analysis, direction-of-arrival probes, or
//! feeding a mono effect send from a chosen bearing).
//!
//! # Provenance
//!
//! This module is engine-agnostic and contains **no Unreal Engine, Unity,
//! Godot, Wwise, FMOD, or Steam Audio source or derived code**. Modal
//! beamforming of a spherical-harmonic field is publicly documented acoustics:
//! the beam families and their per-degree weights are drawn from F. Zotter and
//! M. Frank, "Ambisonics" (Springer, 2019, beamforming chapter) and J. Daniel,
//! "Representation de champs acoustiques" (2001) for the in-phase and max-rE
//! weightings. Everything here is implemented from that public literature.
//!
//! # Beam families
//!
//! All weights are normalised so that `g_0 = 1`.
//!
//! - [`BeamPattern::Basic`]: `g_n = 1`. The plain projection beam; identical in
//!   shape to [`crate::hoa::decode_hoa`]. Narrowest first-null main lobe but
//!   with unweighted side lobes.
//! - [`BeamPattern::MaxDi`]: `g_n = 2n + 1` (hypercardioid). Maximises the
//!   directivity index, which reaches the theoretical `(order + 1)^2`. Has
//!   negative side lobes.
//! - [`BeamPattern::MaxRe`]: `g_n = P_n(r_E)` (reuses
//!   [`crate::hoa_decode::max_re_gains`]). Maximises the energy-vector length,
//!   concentrating reproduced energy toward the look direction.
//! - [`BeamPattern::InPhase`]: `g_n = (L!)^2 / ((L + n)! (L - n)!)` with
//!   `L = order`. The strictly non-negative beam (no phase reversal in any side
//!   lobe), the smoothest / most artefact-free pickup.
//!
//! # Coordinate / axis convention
//!
//! Identical to [`crate::hoa`]: directions are listener-local and match Bevy
//! (`-Z` forward, `+X` right, `+Y` up); acoustic axes are `front = -z`,
//! `left = -x`, `up = +y`.
//!
//! # Real-time contract
//!
//! [`Beamformer::beam`] is **allocation free, lock free, and panic free**: the
//! per-channel weights and the normalisation are baked once in
//! [`Beamformer::new`] (control rate), and the hot path is a bounded dot product
//! over fixed-size stack arrays with a divide-by-zero guard. Degenerate inputs
//! (short buffers, a zero-length look direction, out-of-range order) are clamped,
//! never panicked.
//!
//! # Determinism
//!
//! All direction encoding routes through [`crate::hoa::encode_hoa`], whose math
//! flows through [`bevy_math::ops`] (libm-backed) rather than `f32` intrinsics,
//! so beamforming is bit-reproducible across targets and can be golden-compared
//! sample-for-sample. This is enforced by the workspace lints.

use bevy_math::Vec3;

use prism_audio_core::math::Sample;

use crate::hoa::{MAX_HOA_CHANNELS, MAX_HOA_ORDER, acn_index, encode_hoa, hoa_channel_count};
use crate::hoa_decode::{MAX_ORDER_WEIGHTS, max_re_gains};

/// Guard threshold below which the beam normalisation is treated as zero and
/// replaced by `1.0`, keeping [`Beamformer::beam`] panic and NaN free.
const NORM_EPSILON: Sample = 1.0e-9;

/// The classic modal beam families selectable for a [`Beamformer`].
///
/// Each variant fixes the per-degree weight rule `g_n`; see the module-level
/// documentation for the closed forms and their acoustic trade-offs. All are
/// normalised so that `g_0 = 1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum BeamPattern {
    /// Plain projection beam, `g_n = 1`. Matches [`crate::hoa::decode_hoa`].
    Basic,
    /// Maximum-directivity (hypercardioid) beam, `g_n = 2n + 1`.
    MaxDi,
    /// Maximum energy-vector beam, `g_n = P_n(r_E)`.
    MaxRe,
    /// Strictly non-negative (in-phase) beam, no side-lobe phase reversal.
    InPhase,
}

/// Factorial `k!` as a [`Sample`], via an integer-stepped multiply recurrence.
///
/// Only ever evaluated at control rate for small `k` (at most `2 * order`, and
/// `order <= MAX_HOA_ORDER`), so precision loss is irrelevant here.
fn factorial(k: usize) -> Sample {
    let mut acc: Sample = 1.0;
    let mut i = 2usize;
    while i <= k {
        acc *= i as Sample;
        i += 1;
    }
    acc
}

/// Returns the per-degree beam weights `g_n` for `pattern` at `order`,
/// normalised so that `g_0 = 1`. Degrees beyond `order` are `0`.
///
/// `order` is clamped to [`MAX_HOA_ORDER`]. This is a pure control-rate helper;
/// [`Beamformer::new`] bakes these into a per-channel table for the hot path.
///
/// # Examples
///
/// ```
/// use prism_audio_spatial::hoa_beamform::{BeamPattern, beam_gains};
///
/// // The maximum-directivity beam uses g_n = 2n + 1.
/// let g = beam_gains(BeamPattern::MaxDi, 2);
/// assert_eq!(g[0], 1.0);
/// assert_eq!(g[1], 3.0);
/// assert_eq!(g[2], 5.0);
/// ```
#[must_use]
pub fn beam_gains(pattern: BeamPattern, order: usize) -> [Sample; MAX_ORDER_WEIGHTS] {
    let order = order.min(MAX_HOA_ORDER);
    let mut g = [0.0 as Sample; MAX_ORDER_WEIGHTS];
    match pattern {
        BeamPattern::Basic => {
            for w in g.iter_mut().take(order + 1) {
                *w = 1.0;
            }
        }
        BeamPattern::MaxDi => {
            for (n, w) in g.iter_mut().take(order + 1).enumerate() {
                *w = 2.0 * n as Sample + 1.0;
            }
        }
        BeamPattern::MaxRe => {
            // Reuse the shared max-rE weights (already normalised to g_0 = 1).
            g = max_re_gains(order);
        }
        BeamPattern::InPhase => {
            // g_n = (L!)^2 / ((L + n)! (L - n)!), L = order. g_0 = 1 by construction.
            let l = order;
            let lfact = factorial(l);
            let numer = lfact * lfact;
            for (n, w) in g.iter_mut().take(order + 1).enumerate() {
                let denom = factorial(l + n) * factorial(l - n);
                *w = if denom.abs() > NORM_EPSILON { numer / denom } else { 0.0 };
            }
        }
    }
    g
}

/// A pre-baked steerable modal beamformer over an Ambisonic field.
///
/// [`Beamformer::new`] fixes the beam family and order and precomputes a
/// per-`ACN`-channel weight table plus the on-axis normalisation, so the hot
/// path [`Beamformer::beam`] only encodes the look direction and takes a bounded
/// dot product. Reuse one instance across many frames; rebuild only when the
/// pattern or order changes.
#[derive(Debug, Clone)]
pub struct Beamformer {
    order: usize,
    /// `per_channel_gains[acn_index(n, m)] = g_n`; channels beyond `order` are 0.
    per_channel_gains: [Sample; MAX_HOA_CHANNELS],
    /// On-axis normalisation `sum_n g_n`, fixing `B(0) = 1`.
    norm: Sample,
}

impl Beamformer {
    /// Builds a beamformer of the given `pattern` and `order` (clamped to
    /// [`MAX_HOA_ORDER`]).
    ///
    /// The per-degree weights from [`beam_gains`] are expanded across each
    /// degree's `2n + 1` `ACN` channels, and the normalisation `sum_n g_n` is
    /// accumulated once here so that a unit source on the look axis reads back
    /// as `1.0`.
    #[must_use]
    pub fn new(pattern: BeamPattern, order: usize) -> Self {
        let order = order.min(MAX_HOA_ORDER);
        let weights = beam_gains(pattern, order);
        let mut per_channel_gains = [0.0 as Sample; MAX_HOA_CHANNELS];
        let mut norm = 0.0 as Sample;
        for n in 0..=order {
            let g_n = weights[n];
            norm += g_n;
            let n_isize = n as isize;
            let mut m = -n_isize;
            while m <= n_isize {
                per_channel_gains[acn_index(n, m)] = g_n;
                m += 1;
            }
        }
        Self { order, per_channel_gains, norm }
    }

    /// The beamformer's Ambisonic order.
    #[must_use]
    pub const fn order(&self) -> usize {
        self.order
    }

    /// The active per-`ACN`-channel weight table (`(order + 1)^2` entries).
    #[must_use]
    pub fn per_channel_gains(&self) -> &[Sample] {
        &self.per_channel_gains[..hoa_channel_count(self.order)]
    }

    /// Forms the mono virtual-microphone output aimed at `look_dir` from the
    /// Ambisonic field `coeffs`.
    ///
    /// The look direction is encoded to `ACN`/`SN3D` harmonics and dotted with
    /// the field through the per-channel weight table, then divided by the
    /// baked normalisation. Only the leading `min((order + 1)^2, coeffs.len())`
    /// channels participate; a shorter `coeffs` simply truncates the sum.
    ///
    /// Real-time safe: allocation, lock, and panic free. A degenerate
    /// (near-zero) normalisation is replaced by `1.0`.
    #[must_use]
    pub fn beam(&self, coeffs: &[Sample], look_dir: Vec3) -> Sample {
        let count = hoa_channel_count(self.order);
        let mut enc = [0.0 as Sample; MAX_HOA_CHANNELS];
        encode_hoa(look_dir, self.order, &mut enc);
        let n = count.min(coeffs.len());
        let mut acc = 0.0 as Sample;
        for i in 0..n {
            acc += coeffs[i] * enc[i] * self.per_channel_gains[i];
        }
        let denom = if self.norm.abs() > NORM_EPSILON { self.norm } else { 1.0 };
        acc / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hoa::decode_hoa;
    use crate::hoa_decode::max_re_radius;
    use bevy_math::ops;

    const EPS: Sample = 1.0e-4;

    fn approx(a: Sample, b: Sample) -> bool {
        (a - b).abs() <= EPS
    }

    /// Directivity index from per-degree weights: DI = (sum g_n)^2 / sum(g_n^2 / (2n+1)).
    fn directivity_index(g: &[Sample; MAX_ORDER_WEIGHTS], order: usize) -> Sample {
        let mut num = 0.0 as Sample;
        let mut den = 0.0 as Sample;
        for n in 0..=order {
            num += g[n];
            den += g[n] * g[n] / (2.0 * n as Sample + 1.0);
        }
        (num * num) / den
    }

    /// Standard axisymmetric energy-vector radius:
    /// r_E = 2 sum_{n<order} (n+1) a_n a_{n+1} / sum_n (2n+1) a_n^2.
    fn energy_vector_radius(g: &[Sample; MAX_ORDER_WEIGHTS], order: usize) -> Sample {
        let mut num = 0.0 as Sample;
        let mut den = 0.0 as Sample;
        for n in 0..order {
            num += 2.0 * (n as Sample + 1.0) * g[n] * g[n + 1];
        }
        for n in 0..=order {
            den += (2.0 * n as Sample + 1.0) * g[n] * g[n];
        }
        num / den
    }

    /// A great-circle direction at angle `gamma` (radians) from `-Z` forward,
    /// swept in the front/right (x, -z) plane. Built via libm-backed ops.
    fn dir_at_angle(gamma: Sample) -> Vec3 {
        let (s, c) = ops::sin_cos(gamma);
        Vec3::new(s, 0.0, -c)
    }

    #[test]
    fn all_patterns_normalise_g0_to_one() {
        for order in 0..=MAX_HOA_ORDER {
            for pat in [
                BeamPattern::Basic,
                BeamPattern::MaxDi,
                BeamPattern::MaxRe,
                BeamPattern::InPhase,
            ] {
                let g = beam_gains(pat, order);
                assert!(approx(g[0], 1.0), "g0 != 1 for {pat:?} order {order}");
            }
        }
    }

    #[test]
    fn basic_gains_are_all_unity() {
        let g = beam_gains(BeamPattern::Basic, 3);
        for n in 0..=3 {
            assert!(approx(g[n], 1.0));
        }
    }

    #[test]
    fn maxdi_gains_are_two_n_plus_one() {
        let g = beam_gains(BeamPattern::MaxDi, 3);
        assert!(approx(g[0], 1.0));
        assert!(approx(g[1], 3.0));
        assert!(approx(g[2], 5.0));
        assert!(approx(g[3], 7.0));
    }

    #[test]
    fn inphase_gains_match_closed_form_and_decrease() {
        // order 2: [1, 2/3, 1/6].
        let g = beam_gains(BeamPattern::InPhase, 2);
        assert!(approx(g[0], 1.0));
        assert!(approx(g[1], 2.0 / 3.0));
        assert!(approx(g[2], 1.0 / 6.0));
        // Monotonically decreasing and strictly positive up to order.
        for n in 0..2 {
            assert!(g[n] > g[n + 1], "not decreasing at n={n}");
            assert!(g[n + 1] > 0.0);
        }
    }

    #[test]
    fn maxre_gains_reuse_shared_weights() {
        for order in 0..=MAX_HOA_ORDER {
            let g = beam_gains(BeamPattern::MaxRe, order);
            let shared = max_re_gains(order);
            for n in 0..MAX_ORDER_WEIGHTS {
                assert!(approx(g[n], shared[n]));
            }
        }
    }

    #[test]
    fn unit_output_on_look_axis_for_all_patterns() {
        let order = 3;
        let dir = Vec3::new(0.4, -0.3, -0.7).normalize();
        let mut field = [0.0 as Sample; MAX_HOA_CHANNELS];
        encode_hoa(dir, order, &mut field);
        for pat in [
            BeamPattern::Basic,
            BeamPattern::MaxDi,
            BeamPattern::MaxRe,
            BeamPattern::InPhase,
        ] {
            let bf = Beamformer::new(pat, order);
            let out = bf.beam(&field, dir);
            assert!(approx(out, 1.0), "{pat:?}: on-axis output {out} != 1");
        }
    }

    #[test]
    fn basic_beam_matches_decode_hoa() {
        let order = 2;
        let src = Vec3::new(0.2, 0.5, -0.8).normalize();
        let mut field = [0.0 as Sample; MAX_HOA_CHANNELS];
        encode_hoa(src, order, &mut field);
        let bf = Beamformer::new(BeamPattern::Basic, order);
        for look in [
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(-0.3, 0.4, 0.5).normalize(),
        ] {
            let beam = bf.beam(&field, look);
            let decode = decode_hoa(&field, look, order);
            assert!(approx(beam, decode), "beam {beam} != decode {decode}");
        }
    }

    #[test]
    fn inphase_has_no_negative_side_lobes() {
        for order in 1..=MAX_HOA_ORDER {
            let src = Vec3::new(0.0, 0.0, -1.0);
            let mut field = [0.0 as Sample; MAX_HOA_CHANNELS];
            encode_hoa(src, order, &mut field);
            let bf = Beamformer::new(BeamPattern::InPhase, order);
            let mut i = 0usize;
            while i <= 180 {
                let gamma = core::f32::consts::PI * (i as Sample) / 180.0;
                let out = bf.beam(&field, dir_at_angle(gamma));
                assert!(out >= -1.0e-4, "InPhase order {order}: negative lobe {out} at i={i}");
                i += 1;
            }
        }
    }

    #[test]
    fn maxdi_has_a_negative_side_lobe() {
        // Contrast to in-phase: the hypercardioid must dip below zero somewhere.
        for order in 1..=MAX_HOA_ORDER {
            let src = Vec3::new(0.0, 0.0, -1.0);
            let mut field = [0.0 as Sample; MAX_HOA_CHANNELS];
            encode_hoa(src, order, &mut field);
            let bf = Beamformer::new(BeamPattern::MaxDi, order);
            let mut min = Sample::INFINITY;
            let mut i = 0usize;
            while i <= 180 {
                let gamma = core::f32::consts::PI * (i as Sample) / 180.0;
                let out = bf.beam(&field, dir_at_angle(gamma));
                if out < min {
                    min = out;
                }
                i += 1;
            }
            assert!(min < -1.0e-3, "MaxDi order {order}: expected negative lobe, min={min}");
        }
    }

    #[test]
    fn maxdi_directivity_index_is_order_plus_one_squared() {
        for order in 0..=MAX_HOA_ORDER {
            let g = beam_gains(BeamPattern::MaxDi, order);
            let di = directivity_index(&g, order);
            let expected = (order as Sample + 1.0) * (order as Sample + 1.0);
            assert!(approx(di, expected), "order {order}: DI {di} != {expected}");
        }
    }

    #[test]
    fn maxdi_directivity_beats_basic() {
        for order in 1..=MAX_HOA_ORDER {
            let di_maxdi = directivity_index(&beam_gains(BeamPattern::MaxDi, order), order);
            let di_basic = directivity_index(&beam_gains(BeamPattern::Basic, order), order);
            assert!(di_maxdi > di_basic, "order {order}: maxdi {di_maxdi} !> basic {di_basic}");
        }
    }

    #[test]
    fn maxre_energy_vector_beats_basic_and_matches_radius() {
        for order in 1..=MAX_HOA_ORDER {
            let re_maxre = energy_vector_radius(&beam_gains(BeamPattern::MaxRe, order), order);
            let re_basic = energy_vector_radius(&beam_gains(BeamPattern::Basic, order), order);
            assert!(re_maxre > re_basic, "order {order}: maxre {re_maxre} !> basic {re_basic}");
            // The optimised radius should track the design radius closely.
            let target = max_re_radius(order);
            assert!(
                (re_maxre - target).abs() < 5.0e-2,
                "order {order}: r_E {re_maxre} far from target {target}"
            );
        }
    }

    #[test]
    fn short_buffer_and_zero_direction_do_not_panic() {
        let order = 3;
        let bf = Beamformer::new(BeamPattern::MaxRe, order);
        // Empty field.
        let _ = bf.beam(&[], Vec3::new(0.0, 0.0, -1.0));
        // Field shorter than the channel count.
        let short = [0.5 as Sample; 4];
        let _ = bf.beam(&short, Vec3::new(0.0, 0.0, -1.0));
        // Zero-length look direction (collapses to W only).
        let mut field = [0.0 as Sample; MAX_HOA_CHANNELS];
        encode_hoa(Vec3::new(0.0, 0.0, -1.0), order, &mut field);
        let out = bf.beam(&field, Vec3::ZERO);
        assert!(out.is_finite());
    }

    #[test]
    fn order_is_clamped_to_max() {
        let bf = Beamformer::new(BeamPattern::Basic, 99);
        assert_eq!(bf.order(), MAX_HOA_ORDER);
        assert_eq!(bf.per_channel_gains().len(), MAX_HOA_CHANNELS);
        let g = beam_gains(BeamPattern::MaxDi, 99);
        assert!(approx(g[MAX_HOA_ORDER], 2.0 * MAX_HOA_ORDER as Sample + 1.0));
    }
}
