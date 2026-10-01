//! Higher-order Ambisonics (HOA): scene-based spatial encoding and decoding up
//! to third order.
//!
//! This module generalises the first-order encoder in [`crate::ambisonics`] to
//! arbitrary order `N` (capped at [`MAX_HOA_ORDER`]). Where first-order
//! Ambisonics captures a soundfield with four components (`W, X, Y, Z`),
//! higher-order representations add the finer angular detail carried by the
//! degree-`n` spherical harmonics, sharpening a source's apparent size and
//! improving localisation and decoder robustness. A source is *encoded* into a
//! set of `(N + 1)^2` channels once, then *decoded* toward any speaker (or
//! virtual sampling) direction, keeping *where a sound is* independent of *how
//! it is finally reproduced*.
//!
//! # Provenance
//!
//! Higher-order Ambisonics is a publicly documented, decades-old body of
//! acoustics knowledge that extends Michael Gerzon's first-order theory. The
//! spherical-harmonic soundfield decomposition, the associated Legendre
//! recurrences used to evaluate it, and the channel ordering / normalisation
//! conventions applied here (**`ACN`** component ordering with **`SN3D`**
//! Schmidt semi-normalised gains, i.e. the openly specified `AmbiX` set) are
//! all standard textbook material. Everything here is implemented from that
//! public literature. It contains **no Unreal Engine, Unity, Godot, Wwise, or
//! FMOD source or derived code**.
//!
//! # Channel order and normalisation
//!
//! Unlike [`crate::ambisonics`], which stores its four first-order channels in
//! the engine's fixed `W, X, Y, Z` buffer slots, this module lays channels out
//! in pure **`ACN`** order: the component of degree `n` and order `m`
//! (`-n <= m <= n`) lives at index [`acn_index`]`(n, m) = n^2 + n + m`. The
//! first four `ACN` channels are therefore `W, Y, Z, X` (not `W, X, Y, Z`), so
//! a first-order HOA buffer is *not* interchangeable with an
//! [`crate::ambisonics`] FOA buffer without a channel remap. All gains are
//! `SN3D`, matching the FOA encoder's zeroth-order unit `W`.
//!
//! # Coordinate / axis convention
//!
//! Directions are listener-local and match Bevy (`-Z` forward, `+X` right,
//! `+Y` up). As in [`crate::ambisonics`] the acoustic axes are
//! `front = -dir.z`, `left = -dir.x`, `up = dir.y`; the elevation sine is
//! `sin(beta) = up` and the horizontal radius is
//! `cos(beta) = sqrt(front^2 + left^2)`.
//!
//! # Real-time contract
//!
//! The free functions and [`HoaEncoderNode::process`] are **allocation free,
//! lock free, and panic free**. Spherical-harmonic (re)evaluation from a
//! direction happens only in the non-real-time setters
//! ([`HoaEncoderNode::set_direction`] /
//! [`HoaEncoderNode::set_direction_immediate`]); the hot path merely advances
//! the per-channel [`Smoothed`] gains and multiplies.
//!
//! # Determinism
//!
//! All length/normalisation math flows through [`bevy_math::ops`] (libm-backed)
//! rather than `f32` intrinsics; the harmonics themselves are evaluated with
//! polynomial (Chebyshev / associated-Legendre) recurrences using only
//! multiplies and adds, so encoding and decoding are bit-reproducible across
//! targets and can be golden-compared sample-for-sample.
//!
//! # Field rotation
//!
//! Rotating a *higher*-order soundfield in the spherical-harmonic domain
//! requires the Wigner-D (real SH) rotation matrices, which are deferred to a
//! later milestone. Until then, a moving/rotating source is handled by
//! re-encoding from its current listener-local direction each control tick
//! (exactly what [`HoaEncoderNode::set_direction`] does); first-order fields
//! can still be rotated directly with [`crate::ambisonics::rotate_foa`].

use bevy_math::{Vec3, ops};

use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
use prism_audio_core::math::Sample;
use prism_audio_core::param::{Ramp, Smoothed};

use core::f32::consts::SQRT_2;

/// Highest Ambisonic order supported by this module (third order).
pub const MAX_HOA_ORDER: usize = 3;

/// Number of channels in a full third-order buffer, `(MAX_HOA_ORDER + 1)^2`.
pub const MAX_HOA_CHANNELS: usize = 16;

/// The `SN3D`-normalised zeroth-order (`W`) gain. Under `SN3D` the omni
/// component is unit-weighted.
const W_GAIN: Sample = 1.0;

/// Below this horizontal radius the azimuth is degenerate (the direction points
/// straight up or down), so the azimuth is pinned to `front` (`cos = 1`,
/// `sin = 0`) to avoid a divide-by-zero.
const AZIMUTH_EPSILON: Sample = 1.0e-6;

/// Returns the number of channels in an order-`order` Ambisonic buffer,
/// `(order + 1)^2`.
///
/// ```
/// use prism_audio_spatial::hoa::hoa_channel_count;
/// assert_eq!(hoa_channel_count(0), 1); // W only
/// assert_eq!(hoa_channel_count(1), 4); // first order
/// assert_eq!(hoa_channel_count(3), 16); // third order
/// ```
#[inline]
#[must_use]
pub const fn hoa_channel_count(order: usize) -> usize {
    (order + 1) * (order + 1)
}

/// Returns the flat `ACN` buffer index of the spherical-harmonic component of
/// degree `n` and order `m` (`-n <= m <= n`): `n^2 + n + m`.
#[inline]
#[must_use]
pub const fn acn_index(n: usize, m: isize) -> usize {
    #[expect(
        clippy::cast_possible_wrap,
        reason = "n is a small Ambisonic degree; n*n + n never approaches isize::MAX"
    )]
    let base = (n * n + n) as isize;
    #[expect(
        clippy::cast_sign_loss,
        reason = "base + m is non-negative for valid -n <= m <= n"
    )]
    let idx = (base + m) as usize;
    idx
}

/// Odd double factorial `(2m - 1)!! = 1 * 3 * ... * (2m - 1)` for `m >= 1`.
#[inline]
#[must_use]
fn dblfact_odd(m: usize) -> Sample {
    let mut acc: Sample = 1.0;
    let mut k = 1usize;
    while k < 2 * m {
        acc *= k as Sample;
        k += 2;
    }
    acc
}

/// Ordinary factorial `k!` for small `k` (used only for `SN3D` normalisation).
#[inline]
#[must_use]
fn factorial(k: usize) -> Sample {
    let mut acc: Sample = 1.0;
    let mut i = 2usize;
    while i <= k {
        acc *= i as Sample;
        i += 1;
    }
    acc
}

/// The `SN3D` normalisation weight for degree `n`, absolute order `am`:
/// `sqrt((n - am)! / (n + am)!)`. The extra `sqrt(2)` for `m != 0` is folded
/// into the trigonometric term instead (see [`fill_hoa_coeffs`]).
#[inline]
#[must_use]
fn sn3d_norm(n: usize, am: usize) -> Sample {
    ops::sqrt(factorial(n - am) / factorial(n + am))
}

/// Evaluates the full set of `SN3D` real spherical-harmonic gains for `order`
/// into `coeffs` (in `ACN` order), leaving indices beyond `(order + 1)^2`
/// untouched.
///
/// The implementation maps the Bevy direction to the acoustic axes, evaluates
/// `cos(m*alpha)` / `sin(m*alpha)` with the Chebyshev recurrence and the
/// associated Legendre functions `P_n^m(sin beta)` with the standard upward
/// recurrences (no Condon-Shortley phase), then combines them with the `SN3D`
/// normalisation.
#[expect(
    clippy::needless_range_loop,
    reason = "the associated-Legendre recurrences walk p[n][m] triangularly by degree/order; index arithmetic between neighbouring terms is intrinsic and clearer than iterator adaptors"
)]
fn fill_hoa_coeffs(direction: Vec3, order: usize, coeffs: &mut [Sample; MAX_HOA_CHANNELS]) {
    // Zeroth order: the omni pressure is always unit under SN3D.
    coeffs[0] = W_GAIN;
    if order == 0 {
        return;
    }

    let dir = direction.normalize_or_zero();
    if dir == Vec3::ZERO {
        // A zero-length direction carries no bearing: leave it omnidirectional
        // (W only) instead of collapsing to an arbitrary axis.
        return;
    }
    let front = -dir.z;
    let left = -dir.x;
    let up = dir.y;

    let sin_beta = up;
    let horiz = ops::sqrt(front * front + left * left);
    let (cos_a1, sin_a1) = if horiz > AZIMUTH_EPSILON {
        (front / horiz, left / horiz)
    } else {
        (1.0, 0.0)
    };

    // cos(m*alpha) and sin(m*alpha) via the Chebyshev recurrence.
    let mut cos_m = [0.0 as Sample; MAX_HOA_ORDER + 1];
    let mut sin_m = [0.0 as Sample; MAX_HOA_ORDER + 1];
    cos_m[0] = 1.0;
    sin_m[0] = 0.0;
    cos_m[1] = cos_a1;
    sin_m[1] = sin_a1;
    for m in 2..=order {
        cos_m[m] = 2.0 * cos_a1 * cos_m[m - 1] - cos_m[m - 2];
        sin_m[m] = 2.0 * cos_a1 * sin_m[m - 1] - sin_m[m - 2];
    }

    // Associated Legendre P[n][m] for 0 <= m <= n <= order, argument sin_beta.
    let mut p = [[0.0 as Sample; MAX_HOA_ORDER + 1]; MAX_HOA_ORDER + 1];
    p[0][0] = 1.0;
    // Diagonal: P[m][m] = (2m-1)!! * horiz^m (horiz = cos beta).
    let mut horiz_pow: Sample = 1.0;
    for m in 1..=order {
        horiz_pow *= horiz;
        p[m][m] = dblfact_odd(m) * horiz_pow;
    }
    // First sub-diagonal: P[m+1][m] = sin_beta * (2m + 1) * P[m][m].
    for m in 0..order {
        p[m + 1][m] = sin_beta * ((2 * m + 1) as Sample) * p[m][m];
    }
    // Upward recurrence in n for the remaining terms.
    for m in 0..=order {
        for n in (m + 2)..=order {
            let nn = n as Sample;
            let mm = m as Sample;
            p[n][m] =
                ((2.0 * nn - 1.0) * sin_beta * p[n - 1][m] - (nn + mm - 1.0) * p[n - 2][m])
                    / (nn - mm);
        }
    }

    // Assemble the SN3D real harmonics into ACN slots (degree 0 already done).
    for n in 1..=order {
        let n_isize = n as isize;
        for m in -n_isize..=n_isize {
            let am = m.unsigned_abs();
            let norm = sn3d_norm(n, am);
            let trig = if m == 0 {
                1.0
            } else if m > 0 {
                SQRT_2 * cos_m[am]
            } else {
                SQRT_2 * sin_m[am]
            };
            coeffs[acn_index(n, m)] = norm * p[n][am] * trig;
        }
    }
}

/// Encodes a source `direction` into `SN3D`/`ACN` Ambisonic gains of the given
/// `order`, writing them into `out` and returning the number of channels
/// written.
///
/// `order` is clamped to [`MAX_HOA_ORDER`]. At most `(order + 1)^2` gains are
/// produced; if `out` is shorter, only the leading channels that fit are
/// written and that shorter count is returned. Multiply a mono sample by these
/// gains to distribute it across the Ambisonic channels. A zero-length
/// `direction` collapses to `W` only (omnidirectional).
///
/// This is the one-shot / offline path; the real-time path uses
/// [`HoaEncoderNode`] so that direction changes are click-free.
#[inline]
pub fn encode_hoa(direction: Vec3, order: usize, out: &mut [Sample]) -> usize {
    let order = order.min(MAX_HOA_ORDER);
    let count = hoa_channel_count(order);
    let mut coeffs = [0.0 as Sample; MAX_HOA_CHANNELS];
    fill_hoa_coeffs(direction, order, &mut coeffs);
    let written = count.min(out.len());
    out[..written].copy_from_slice(&coeffs[..written]);
    written
}

/// Decodes an Ambisonic field toward a single `speaker_direction`, returning
/// the mono signal that speaker should emit.
///
/// This is the order-`N` "basic"/projection decode: the speaker direction is
/// re-encoded to `SN3D`/`ACN` harmonics and dotted with `coeffs`, then scaled
/// by `1 / (order + 1)` (the `SN3D` self-energy at the source direction). By the spherical-harmonic addition theorem this
/// makes a unit source encoded at direction `d` decode back to exactly `1.0` at
/// a coincident speaker (`d`), while off-axis speakers receive progressively
/// less as the order rises (sharper directivity).
///
/// `order` is clamped to [`MAX_HOA_ORDER`]; only the leading
/// `min((order + 1)^2, coeffs.len())` channels participate.
#[must_use]
pub fn decode_hoa(coeffs: &[Sample], speaker_direction: Vec3, order: usize) -> Sample {
    let order = order.min(MAX_HOA_ORDER);
    let count = hoa_channel_count(order);
    let mut enc = [0.0 as Sample; MAX_HOA_CHANNELS];
    fill_hoa_coeffs(speaker_direction, order, &mut enc);
    let n = count.min(coeffs.len());
    // Under SN3D the per-degree self-energy is unity, so the addition theorem
    // sums to (order + 1) at the source direction; dividing by that yields a
    // unit response there. (This reduces to FOA's 1 / (N + 1) = 0.5.)
    let mut acc: Sample = 0.0;
    for (c, e) in coeffs[..n].iter().zip(enc[..n].iter()) {
        acc += c * e;
    }
    acc / ((order + 1) as Sample)
}

/// A real-time node that encodes a mono source into higher-order Ambisonics
/// with click-free direction changes.
///
/// * Input port `0`: mono ([`ChannelLayout::Mono`](prism_audio_core::buffer::ChannelLayout::Mono)).
/// * Output port `0`: an Ambisonic bus whose channels are laid out in `ACN`
///   order (see [`acn_index`]); the node writes its
///   [`active_channels`](Self::active_channels) leading channels.
///
/// The `SN3D` encoding gains are held as [`Smoothed`] parameters so that moving
/// the source is a per-sample glide rather than a step. Direction changes are
/// applied off the audio thread through [`set_direction`](Self::set_direction)
/// / [`set_direction_immediate`](Self::set_direction_immediate); the hot path
/// only advances and applies the smoothers and never allocates or panics.
#[derive(Debug, Clone)]
pub struct HoaEncoderNode {
    gains: [Smoothed; MAX_HOA_CHANNELS],
    order: usize,
    active_channels: usize,
}

impl HoaEncoderNode {
    /// Creates an encoder of the given `order` (clamped to [`MAX_HOA_ORDER`])
    /// settled at "pressure only": the `W` gain starts at `1.0` (`SN3D`) and
    /// every higher component at `0.0`, i.e. an omnidirectional source until a
    /// direction is set.
    #[inline]
    #[must_use]
    pub fn new(order: usize) -> Self {
        let order = order.min(MAX_HOA_ORDER);
        let active_channels = hoa_channel_count(order);
        let mut gains = [Smoothed::new(0.0); MAX_HOA_CHANNELS];
        gains[0] = Smoothed::new(W_GAIN);
        Self { gains, order, active_channels }
    }

    /// The Ambisonic order this encoder produces.
    #[inline]
    #[must_use]
    pub const fn order(&self) -> usize {
        self.order
    }

    /// The number of active `ACN` channels this encoder drives,
    /// `(order + 1)^2`.
    #[inline]
    #[must_use]
    pub const fn active_channels(&self) -> usize {
        self.active_channels
    }

    /// Retargets the encoding gains toward a new source `direction`, gliding
    /// over `ramp`. **Non-real-time**: this re-evaluates the spherical
    /// harmonics and must be called off the audio thread (e.g. from the
    /// control/update tick).
    pub fn set_direction(&mut self, direction: Vec3, ramp: Ramp) {
        let mut coeffs = [0.0 as Sample; MAX_HOA_CHANNELS];
        fill_hoa_coeffs(direction, self.order, &mut coeffs);
        for (k, (slot, &coeff)) in self.gains.iter_mut().zip(coeffs.iter()).enumerate() {
            let target = if k < self.active_channels { coeff } else { 0.0 };
            slot.set_target(target, ramp);
        }
    }

    /// Snaps the encoding gains to a new source `direction` immediately (no
    /// glide). Intended for initial placement or teleports; using it on a
    /// moving source may click. **Non-real-time.**
    #[inline]
    pub fn set_direction_immediate(&mut self, direction: Vec3) {
        self.set_direction(direction, Ramp::Immediate);
    }

    /// Returns the current (instantaneous) encoding gains in `ACN` order
    /// without advancing the smoothers. Useful for tests and metering.
    #[inline]
    #[must_use]
    pub fn current_gains(&self) -> [Sample; MAX_HOA_CHANNELS] {
        let mut out = [0.0 as Sample; MAX_HOA_CHANNELS];
        for (slot, gain) in out.iter_mut().zip(self.gains.iter()) {
            *slot = gain.current();
        }
        out
    }
}

impl AudioNode for HoaEncoderNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);

        // Degenerate wiring guard: never index a missing channel, never panic.
        if input.channels() == 0 || output.channels() == 0 {
            return;
        }

        let mono = input.channel(0);
        let frames = mono.len().min(output.active_frames());
        let out_channels = output.channels().min(self.active_channels);

        for k in 0..self.active_channels {
            if k < out_channels {
                let dst = output.channel_mut(k);
                let gain = &mut self.gains[k];
                for i in 0..frames {
                    dst[i] = mono[i] * gain.next_sample();
                }
            } else {
                // Keep every active smoother phase-aligned even when the output
                // bus has fewer channels than the encoder's order.
                let gain = &mut self.gains[k];
                for _ in 0..frames {
                    gain.next_sample();
                }
            }
        }
    }

    fn reset(&mut self) {
        for k in 0..MAX_HOA_CHANNELS {
            let settled = self.gains[k].target();
            self.gains[k] = Smoothed::new(settled);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};

    const EPS: Sample = 1.0e-5;

    fn approx(a: Sample, b: Sample) -> bool {
        (a - b).abs() <= EPS
    }

    const FRONT: Vec3 = Vec3::new(0.0, 0.0, -1.0);
    const LEFT: Vec3 = Vec3::new(-1.0, 0.0, 0.0);
    const UP: Vec3 = Vec3::new(0.0, 1.0, 0.0);

    #[test]
    fn channel_count_matches_square_law() {
        assert_eq!(hoa_channel_count(0), 1);
        assert_eq!(hoa_channel_count(1), 4);
        assert_eq!(hoa_channel_count(2), 9);
        assert_eq!(hoa_channel_count(3), 16);
    }

    #[test]
    fn acn_index_orders_first_order_as_w_y_z_x() {
        assert_eq!(acn_index(0, 0), 0);
        assert_eq!(acn_index(1, -1), 1);
        assert_eq!(acn_index(1, 0), 2);
        assert_eq!(acn_index(1, 1), 3);
        assert_eq!(acn_index(3, 3), 15);
    }

    #[test]
    fn first_order_reduces_to_sn3d_acn_gains() {
        // Front (Bevy -Z): W=1, Y(left)=0, Z(up)=0, X(front)=1.
        let mut out = [0.0 as Sample; 4];
        let written = encode_hoa(FRONT, 1, &mut out);
        assert_eq!(written, 4);
        assert!(approx(out[0], 1.0)); // W
        assert!(approx(out[1], 0.0)); // Y
        assert!(approx(out[2], 0.0)); // Z
        assert!(approx(out[3], 1.0)); // X

        // Left excites Y only.
        let mut ly = [0.0 as Sample; 4];
        encode_hoa(LEFT, 1, &mut ly);
        assert!(approx(ly[1], 1.0));
        assert!(approx(ly[3], 0.0));

        // Up excites Z only.
        let mut uz = [0.0 as Sample; 4];
        encode_hoa(UP, 1, &mut uz);
        assert!(approx(uz[2], 1.0));
        assert!(approx(uz[1], 0.0));
    }

    #[test]
    fn order_is_clamped_and_returns_written_count() {
        let mut out = [0.0 as Sample; MAX_HOA_CHANNELS];
        // Absurd order clamps to third order (16 channels).
        assert_eq!(encode_hoa(FRONT, 99, &mut out), 16);
        // A short buffer only receives what fits.
        let mut short = [0.0 as Sample; 5];
        assert_eq!(encode_hoa(FRONT, 3, &mut short), 5);
    }

    #[test]
    fn degenerate_direction_is_omnidirectional() {
        let mut out = [0.0 as Sample; MAX_HOA_CHANNELS];
        let written = encode_hoa(Vec3::ZERO, 3, &mut out);
        assert_eq!(written, 16);
        assert!(approx(out[0], 1.0)); // W stays unity
        for &c in &out[1..16] {
            assert!(approx(c, 0.0), "higher components should vanish, got {c}");
        }
    }

    #[test]
    fn straight_up_direction_does_not_panic() {
        // Purely vertical directions drive the horizontal radius to zero.
        let mut up = [0.0 as Sample; MAX_HOA_CHANNELS];
        let mut down = [0.0 as Sample; MAX_HOA_CHANNELS];
        encode_hoa(UP, 3, &mut up);
        encode_hoa(Vec3::new(0.0, -1.0, 0.0), 3, &mut down);
        // Z (ACN 2) is +1 up, -1 down under SN3D.
        assert!(up[2] > 0.9);
        assert!(down[2] < -0.9);
    }

    #[test]
    fn decode_returns_unity_at_the_encoded_direction() {
        for &order in &[1usize, 2, 3] {
            let dir = Vec3::new(0.3, -0.5, -0.8);
            let mut coeffs = [0.0 as Sample; MAX_HOA_CHANNELS];
            let n = encode_hoa(dir, order, &mut coeffs);
            let here = decode_hoa(&coeffs[..n], dir, order);
            assert!(approx(here, 1.0), "order {order}: decode at source = {here}");
        }
    }

    #[test]
    fn decode_is_weakest_at_the_antipode() {
        let dir = FRONT;
        let anti = Vec3::new(0.0, 0.0, 1.0);
        let mut coeffs = [0.0 as Sample; MAX_HOA_CHANNELS];
        let n = encode_hoa(dir, 3, &mut coeffs);
        let here = decode_hoa(&coeffs[..n], dir, 3);
        let there = decode_hoa(&coeffs[..n], anti, 3);
        assert!(here > there, "front {here} should beat antipode {there}");
    }

    #[test]
    fn higher_order_is_more_directional_off_axis() {
        // Encode a front source, sample a speaker 90 degrees to the side. The
        // higher the order, the smaller the leaked off-axis response.
        let side = LEFT;
        let mut o1 = [0.0 as Sample; MAX_HOA_CHANNELS];
        let mut o3 = [0.0 as Sample; MAX_HOA_CHANNELS];
        let n1 = encode_hoa(FRONT, 1, &mut o1);
        let n3 = encode_hoa(FRONT, 3, &mut o3);
        let leak1 = decode_hoa(&o1[..n1], side, 1).abs();
        let leak3 = decode_hoa(&o3[..n3], side, 3).abs();
        assert!(leak3 < leak1, "order3 leak {leak3} should be < order1 {leak1}");
    }

    #[test]
    fn w_only_field_decodes_to_a_direction_independent_constant() {
        // A pure omni (W = 1) field decodes to 1 / (order + 1) everywhere.
        let order = 2;
        let denom = (order + 1) as Sample;
        let mut coeffs = [0.0 as Sample; MAX_HOA_CHANNELS];
        coeffs[0] = 1.0;
        for dir in [FRONT, LEFT, UP, Vec3::new(1.0, 2.0, -3.0)] {
            let v = decode_hoa(&coeffs[..hoa_channel_count(order)], dir, order);
            assert!(approx(v, 1.0 / denom), "expected constant, got {v}");
        }
    }

    #[test]
    fn encoding_is_linear_in_the_source_amplitude() {
        // encode gains are amplitude-independent; scaling the mono sample scales
        // every channel identically, so decode scales linearly too.
        let dir = Vec3::new(-0.2, 0.6, -0.7);
        let mut coeffs = [0.0 as Sample; MAX_HOA_CHANNELS];
        let n = encode_hoa(dir, 3, &mut coeffs);
        let mut scaled = coeffs;
        for s in &mut scaled[..n] {
            *s *= 2.5;
        }
        let base = decode_hoa(&coeffs[..n], dir, 3);
        let scaled_out = decode_hoa(&scaled[..n], dir, 3);
        assert!(approx(scaled_out, 2.5 * base));
    }

    #[test]
    fn node_metadata_tracks_the_requested_order() {
        let node = HoaEncoderNode::new(2);
        assert_eq!(node.order(), 2);
        assert_eq!(node.active_channels(), 9);
        // Clamped high, and a settled encoder starts omnidirectional (W only).
        let clamped = HoaEncoderNode::new(99);
        assert_eq!(clamped.order(), MAX_HOA_ORDER);
        assert_eq!(clamped.active_channels(), MAX_HOA_CHANNELS);
        let g = clamped.current_gains();
        assert!(approx(g[0], 1.0));
        assert!(g[1..].iter().all(|&s| approx(s, 0.0)));
    }

    #[test]
    fn node_encodes_front_like_the_free_function() {
        // Use a first-order encoder so the four ACN channels fit a Quad-sized
        // discrete container (there is no 9-channel layout in the core buffer).
        let mut node = HoaEncoderNode::new(1);
        assert_eq!(node.order(), 1);
        assert_eq!(node.active_channels(), 4);
        node.set_direction_immediate(FRONT);

        let mut input = AudioBuffer::new(ChannelLayout::Mono, 8);
        input.channel_mut(0).copy_from_slice(&[1.0; 8]);
        let mut outputs = [AudioBuffer::new(ChannelLayout::Quad, 8)];
        let ctx = RenderContext { sample_rate: 48_000, frames: 8, playhead: 0 };
        let inputs = [input];
        {
            let mut io = ProcessIo::new(&inputs, &mut outputs);
            node.process(&ctx, &mut io);
        }
        let out = &outputs[0];

        let mut expect = [0.0 as Sample; MAX_HOA_CHANNELS];
        encode_hoa(FRONT, 1, &mut expect);
        for (k, &e) in expect.iter().enumerate().take(4) {
            assert!(
                out.channel(k).iter().all(|&s| approx(s, e)),
                "channel {k} mismatch",
            );
        }
    }

    #[test]
    fn node_glides_between_directions_without_zipper_jumps() {
        let mut node = HoaEncoderNode::new(1);
        node.set_direction_immediate(FRONT); // X = 1
        node.set_direction(LEFT, Ramp::Linear { samples: 8 });

        let mut input = AudioBuffer::new(ChannelLayout::Mono, 8);
        input.channel_mut(0).copy_from_slice(&[1.0; 8]);
        let mut outputs = [AudioBuffer::new(ChannelLayout::Quad, 8)];
        let ctx = RenderContext { sample_rate: 48_000, frames: 8, playhead: 0 };
        let inputs = [input];
        {
            let mut io = ProcessIo::new(&inputs, &mut outputs);
            node.process(&ctx, &mut io);
        }
        let out = &outputs[0];

        // ACN 3 = X (front) should fall, ACN 1 = Y (left) should rise.
        let x = out.channel(3);
        let y = out.channel(1);
        assert!(x[0] > x[7], "X should fall: {} -> {}", x[0], x[7]);
        assert!(y[7] > y[0], "Y should rise: {} -> {}", y[0], y[7]);
        for w in x.windows(2) {
            assert!((w[1] - w[0]).abs() < 0.3);
        }
    }

    #[test]
    fn reset_settles_gains_at_their_targets() {
        let mut node = HoaEncoderNode::new(1);
        node.set_direction(LEFT, Ramp::Linear { samples: 64 });

        let mut input = AudioBuffer::new(ChannelLayout::Mono, 4);
        input.channel_mut(0).copy_from_slice(&[1.0; 4]);
        let mut outputs = [AudioBuffer::new(ChannelLayout::Quad, 4)];
        let ctx = RenderContext { sample_rate: 48_000, frames: 4, playhead: 0 };
        let inputs = [input];
        {
            let mut io = ProcessIo::new(&inputs, &mut outputs);
            node.process(&ctx, &mut io);
        }

        node.reset();
        let g = node.current_gains();
        let mut target = [0.0 as Sample; MAX_HOA_CHANNELS];
        encode_hoa(LEFT, 1, &mut target);
        for k in 0..4 {
            assert!(approx(g[k], target[k]), "channel {k}: {} vs {}", g[k], target[k]);
        }
    }

    #[test]
    fn process_does_not_panic_on_short_output_buffer() {
        // A mono output must not panic; extra components are dropped while the
        // smoothers still advance.
        let mut node = HoaEncoderNode::new(3);
        node.set_direction_immediate(FRONT);
        let mut input = AudioBuffer::new(ChannelLayout::Mono, 4);
        input.channel_mut(0).copy_from_slice(&[1.0; 4]);
        let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, 4)];
        let ctx = RenderContext { sample_rate: 48_000, frames: 4, playhead: 0 };
        let inputs = [input];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx, &mut io);
        assert!(outputs[0].channel(0).iter().all(|&s| approx(s, 1.0)));
    }
}
