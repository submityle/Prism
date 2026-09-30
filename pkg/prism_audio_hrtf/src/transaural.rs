//! Transaural rendering: crosstalk cancellation for loudspeaker playback of
//! binaural signals.
//!
//! [`binaural`](crate::binaural) produces a two-channel signal meant for
//! headphones, where the left ear hears only the left channel and the right
//! ear only the right. Over a pair of stereo loudspeakers that assumption
//! breaks: each ear also hears the *opposite* speaker after a short delay and
//! head-shadow attenuation (the acoustic "crosstalk"). Left unaddressed,
//! crosstalk collapses the binaural cues and the spatial image folds back
//! toward the speakers.
//!
//! This module implements the classic *recursive crosstalk canceller*: given a
//! binaural target `b = [b_l, b_r]` we compute speaker feeds `s = [s_l, s_r]`
//! such that, after the acoustic path to the ears, the ears receive `b` again.
//!
//! # Model
//!
//! Model the symmetric speaker-to-ear system, normalised by the ipsilateral
//! (same-side) path, as
//!
//! ```text
//! e = C s,   C = [[1, beta], [beta, 1]]
//! ```
//!
//! where `beta` is the contralateral/ipsilateral ratio: a pure delay of `d`
//! samples (the inter-aural path difference), a broadband head-shadow gain
//! `g < 1`, and a one-pole low-pass that captures the head shadowing the far
//! ear's high frequencies. Because `|beta| < 1`, the inverse `C^-1` is realised
//! *exactly* by the single-path recursive structure
//!
//! ```text
//! s_l[n] = b_l[n] - beta * s_r[n]
//! s_r[n] = b_r[n] - beta * s_l[n]
//! ```
//!
//! which back-substitutes to `e = C s = b`. Since `beta` carries a delay of
//! `d >= 1` samples there is no algebraic (delay-free) loop, so the recursion
//! is real-time realisable with two `d`-sample delay lines. The round-trip loop
//! gain is `g^2 < 1`, so the canceller is unconditionally stable for any
//! physical head-shadow gain.
//!
//! This canceller inverts the *crosstalk* only; it does not equalise the
//! ipsilateral magnitude/phase response, which is left to an upstream headphone
//! or speaker EQ stage. That separation keeps the hot path a pair of one-pole
//! filters plus delay reads.
//!
//! # Real-time contract
//!
//! [`CrosstalkCanceller::process_block`] is **allocation free, lock free, and
//! panic free**. The delay lines and filter state are sized once in
//! [`CrosstalkCanceller::new`] (control rate); the hot path only reads/writes
//! fixed buffers. Degenerate parameters (zero delay, gain out of range) are
//! clamped, never panicked.
//!
//! # Determinism
//!
//! The geometry helper routes its transcendental math through
//! [`bevy_math::ops`] (libm-backed) rather than `f32` intrinsics, so derived
//! coefficients are bit-reproducible across targets. The runtime recursion is
//! plain multiply/add and is golden-comparable sample-for-sample.
//!
//! # Provenance
//!
//! This module is engine-agnostic and contains **no Unreal Engine, Unity,
//! Godot, Wwise, FMOD, or Steam Audio source or derived code**. Recursive
//! crosstalk cancellation for transaural stereo is long-established, publicly
//! documented acoustics: B. B. Bauer, "Stereophonic Earphones and Binaural
//! Loudspeakers" (JAES, 1961); M. R. Schroeder and B. S. Atal, "Computer
//! Simulation of Sound Transmission in Rooms" (1963); D. H. Cooper and J. L.
//! Bauck, "Prospects for Transaural Recording" (JAES, 1989); and W. G.
//! Gardner, "3-D Audio Using Loudspeakers" (1998). The Woodworth inter-aural
//! delay used by the geometry helper is standard psychoacoustics. Everything
//! here is implemented from that public literature.

use alloc::vec;
use alloc::vec::Vec;

use bevy_math::ops;

use prism_audio_core::math::Sample;

/// Default speed of sound in dry air at 20 C, in metres per second.
pub const DEFAULT_SOUND_SPEED: Sample = 343.0;

/// Default broadband contralateral head-shadow gain (`< 1`), a typical value
/// for a source near +/- 30 degrees off-axis.
pub const DEFAULT_CONTRALATERAL_GAIN: Sample = 0.7;

/// Minimum guaranteed crosstalk delay in samples. A delay of at least one
/// sample is what breaks the recursion's algebraic loop, so the design clamps
/// to this floor even for degenerate (head-on) speaker geometry.
pub const MIN_CROSSTALK_DELAY: usize = 1;

/// Description of the speaker-to-ear crosstalk path used by a
/// [`CrosstalkCanceller`].
///
/// These are control-rate design parameters. Build them directly for a known
/// path, or from a symmetric speaker layout with
/// [`CrosstalkParams::from_geometry`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct CrosstalkParams {
    /// Broadband contralateral gain `g` relative to the ipsilateral path.
    /// Clamped to `[0, 0.999]` on use so the loop gain stays below unity.
    pub contralateral_gain: Sample,
    /// Inter-aural crosstalk delay `d` in samples (path difference between the
    /// near and far ear). Clamped to at least [`MIN_CROSSTALK_DELAY`].
    pub delay_samples: usize,
    /// Head-shadow low-pass cutoff for the contralateral path, in hertz. A
    /// non-positive or non-finite value disables the shadow filter (the
    /// contralateral branch is then a plain delayed gain).
    pub shadow_cutoff_hz: Sample,
    /// Sample rate the delay and cutoff were designed for, in hertz.
    pub sample_rate: Sample,
}

impl CrosstalkParams {
    /// Builds parameters directly from a contralateral gain, an integer delay,
    /// and a head-shadow cutoff.
    #[must_use]
    pub fn new(
        contralateral_gain: Sample,
        delay_samples: usize,
        shadow_cutoff_hz: Sample,
        sample_rate: Sample,
    ) -> Self {
        Self { contralateral_gain, delay_samples, shadow_cutoff_hz, sample_rate }
    }

    /// Derives the crosstalk path from a symmetric two-speaker layout.
    ///
    /// `speaker_azimuth_rad` is the half-angle of the stereo pair measured from
    /// the forward axis to one speaker (for the classic 60-degree pair, pass
    /// `30 degrees` in radians). The inter-aural delay uses the Woodworth
    /// approximation `itd = (a / c) * (phi + sin phi)` with head radius `a` and
    /// sound speed `c`. The head-shadow cutoff is placed near the frequency
    /// whose wavelength equals the head circumference, a standard rule of thumb.
    ///
    /// Non-finite or out-of-range inputs collapse to safe defaults (minimum
    /// delay, no shadow) rather than panicking.
    #[must_use]
    pub fn from_geometry(
        speaker_azimuth_rad: Sample,
        head_radius: Sample,
        sample_rate: Sample,
        sound_speed: Sample,
    ) -> Self {
        let a = if head_radius.is_finite() && head_radius > 0.0 {
            head_radius
        } else {
            crate::nearfield::DEFAULT_HEAD_RADIUS
        };
        let c = if sound_speed.is_finite() && sound_speed > 0.0 {
            sound_speed
        } else {
            DEFAULT_SOUND_SPEED
        };
        // Clamp azimuth to the Woodworth-valid front quadrant.
        let phi = speaker_azimuth_rad.clamp(0.0, core::f32::consts::FRAC_PI_2);
        let itd_seconds = (a / c) * (phi + ops::sin(phi));
        let delay_samples = seconds_to_samples(itd_seconds, sample_rate);
        // Place the head-shadow corner near c / (2 * pi * a): the frequency
        // where the head becomes acoustically large. This darkens the far-ear
        // branch above roughly this corner.
        let shadow_cutoff_hz = c / (core::f32::consts::TAU * a);
        Self {
            contralateral_gain: DEFAULT_CONTRALATERAL_GAIN,
            delay_samples,
            shadow_cutoff_hz,
            sample_rate,
        }
    }
}

/// Converts a non-negative duration in seconds to a rounded, clamped sample
/// count of at least [`MIN_CROSSTALK_DELAY`].
fn seconds_to_samples(seconds: Sample, sample_rate: Sample) -> usize {
    if !seconds.is_finite() || !sample_rate.is_finite() || sample_rate <= 0.0 {
        return MIN_CROSSTALK_DELAY;
    }
    let raw = seconds * sample_rate;
    if !(raw.is_finite() && raw >= 0.0) {
        return MIN_CROSSTALK_DELAY;
    }
    // Round to nearest by integer stepping, avoiding a lint-flagged float cast.
    let mut n: usize = 0;
    let mut acc: Sample = 0.5;
    // Bound the loop so a pathological `raw` cannot spin unboundedly.
    while acc < raw && n < MAX_DELAY_STEPS {
        acc += 1.0;
        n += 1;
    }
    n.max(MIN_CROSSTALK_DELAY)
}

/// Upper bound on the integer delay the geometry helper will resolve, guarding
/// [`seconds_to_samples`] against pathological inputs (about 21 ms at 48 kHz).
const MAX_DELAY_STEPS: usize = 1 << 10;

/// One-pole low-pass smoothing coefficient from a cutoff frequency.
///
/// Returns `1.0` (pass-through) when the cutoff is disabled or exceeds Nyquist,
/// so a canceller with the shadow filter off is a plain delayed gain.
fn shadow_alpha(cutoff_hz: Sample, sample_rate: Sample) -> Sample {
    if !cutoff_hz.is_finite() || cutoff_hz <= 0.0 || !sample_rate.is_finite() || sample_rate <= 0.0
    {
        return 1.0;
    }
    let nyquist = 0.5 * sample_rate;
    if cutoff_hz >= nyquist {
        return 1.0;
    }
    let x = -core::f32::consts::TAU * cutoff_hz / sample_rate;
    (1.0 - ops::exp(x)).clamp(0.0, 1.0)
}

/// A real-time recursive crosstalk canceller for transaural stereo playback.
///
/// Construct once from [`CrosstalkParams`], then stream binaural blocks through
/// [`process_block`](CrosstalkCanceller::process_block). Reuse a single
/// instance across the stream so the delay lines and shadow filters carry
/// state; call [`reset`](CrosstalkCanceller::reset) to clear it on a
/// discontinuity.
///
/// # Examples
///
/// With the head-shadow filter disabled the canceller reduces to a plain
/// delayed gain, so the speaker feeds reconstruct the binaural target exactly
/// at the ears. Here the contralateral gain is `g = 0.5` and the delay is
/// `d = 1` sample:
///
/// ```
/// use prism_audio_hrtf::{CrosstalkCanceller, CrosstalkParams};
///
/// // gain 0.5, 1-sample delay, shadow filter disabled (cutoff <= 0).
/// let params = CrosstalkParams::new(0.5, 1, 0.0, 48_000.0);
/// let mut canceller = CrosstalkCanceller::new(&params);
///
/// let in_l = [1.0, 0.0];
/// let in_r = [0.0, 1.0];
/// let mut s_l = [0.0; 2];
/// let mut s_r = [0.0; 2];
/// let frames = canceller.process_block(&in_l, &in_r, &mut s_l, &mut s_r);
/// assert_eq!(frames, 2);
///
/// // Pre-compensated (crosstalk-cancelled) speaker feeds.
/// assert_eq!(s_l, [1.0, 0.0]);
/// assert_eq!(s_r, [0.0, 0.5]);
///
/// // Propagate the feeds through the symmetric crosstalk path:
/// //   e_l[n] = s_l[n] + g * s_r[n - d],  e_r[n] = s_r[n] + g * s_l[n - d].
/// let g = 0.5;
/// let e_l = [s_l[0], s_l[1] + g * s_r[0]];
/// let e_r = [s_r[0], s_r[1] + g * s_l[0]];
/// assert_eq!(e_l, in_l); // left ear hears only the left target
/// assert_eq!(e_r, in_r); // right ear hears only the right target
/// ```
#[derive(Debug, Clone)]
pub struct CrosstalkCanceller {
    gain: Sample,
    alpha: Sample,
    delay: usize,
    /// Ring buffer of past left speaker outputs, length `delay`.
    line_l: Vec<Sample>,
    /// Ring buffer of past right speaker outputs, length `delay`.
    line_r: Vec<Sample>,
    idx: usize,
    /// Head-shadow one-pole state for the left and right contralateral branches.
    shadow_l: Sample,
    shadow_r: Sample,
}

impl CrosstalkCanceller {
    /// Builds a canceller from crosstalk parameters.
    ///
    /// The contralateral gain is clamped to `[0, 0.999]` (keeping loop gain
    /// below unity), and the delay to at least [`MIN_CROSSTALK_DELAY`].
    #[must_use]
    pub fn new(params: &CrosstalkParams) -> Self {
        let gain = params.contralateral_gain.clamp(0.0, 0.999);
        let delay = params.delay_samples.max(MIN_CROSSTALK_DELAY);
        let alpha = shadow_alpha(params.shadow_cutoff_hz, params.sample_rate);
        Self {
            gain,
            alpha,
            delay,
            line_l: vec![0.0; delay],
            line_r: vec![0.0; delay],
            idx: 0,
            shadow_l: 0.0,
            shadow_r: 0.0,
        }
    }

    /// The clamped contralateral gain in use.
    #[must_use]
    pub const fn contralateral_gain(&self) -> Sample {
        self.gain
    }

    /// The crosstalk delay in samples.
    #[must_use]
    pub const fn delay_samples(&self) -> usize {
        self.delay
    }

    /// Clears all delay-line and shadow-filter state.
    pub fn reset(&mut self) {
        for s in &mut self.line_l {
            *s = 0.0;
        }
        for s in &mut self.line_r {
            *s = 0.0;
        }
        self.idx = 0;
        self.shadow_l = 0.0;
        self.shadow_r = 0.0;
    }

    /// Applies crosstalk cancellation to one binaural block, writing the
    /// speaker feeds into `out_l` / `out_r`.
    ///
    /// Returns the number of frames processed, the minimum of all four slice
    /// lengths, so mismatched buffers truncate rather than panic. Real-time
    /// safe: allocation, lock, and panic free.
    pub fn process_block(
        &mut self,
        in_l: &[Sample],
        in_r: &[Sample],
        out_l: &mut [Sample],
        out_r: &mut [Sample],
    ) -> usize {
        let frames = in_l
            .len()
            .min(in_r.len())
            .min(out_l.len())
            .min(out_r.len());
        for n in 0..frames {
            // Speaker outputs from `delay` samples ago (the slot about to be
            // overwritten holds exactly s[n - delay]).
            let past_l = self.line_l[self.idx];
            let past_r = self.line_r[self.idx];
            // Head-shadow low-pass on each contralateral branch.
            self.shadow_l += self.alpha * (past_l - self.shadow_l);
            self.shadow_r += self.alpha * (past_r - self.shadow_r);
            let cross_from_r = self.gain * self.shadow_r;
            let cross_from_l = self.gain * self.shadow_l;
            // Recursive canceller: subtract the reconstructed crosstalk.
            let s_l = in_l[n] - cross_from_r;
            let s_r = in_r[n] - cross_from_l;
            out_l[n] = s_l;
            out_r[n] = s_r;
            self.line_l[self.idx] = s_l;
            self.line_r[self.idx] = s_r;
            self.idx += 1;
            if self.idx >= self.delay {
                self.idx = 0;
            }
        }
        frames
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FS: Sample = 48_000.0;

    fn canceller(gain: Sample, delay: usize) -> CrosstalkCanceller {
        CrosstalkCanceller::new(&CrosstalkParams::new(gain, delay, 0.0, FS))
    }

    /// Simulate the symmetric acoustic path `e = C s` for a pure delayed gain
    /// (shadow off) so we can check the ears receive the intended binaural
    /// signal after cancellation.
    fn acoustic_path(
        s_l: &[Sample],
        s_r: &[Sample],
        gain: Sample,
        delay: usize,
    ) -> (Vec<Sample>, Vec<Sample>) {
        let n = s_l.len();
        let mut e_l = vec![0.0; n];
        let mut e_r = vec![0.0; n];
        for i in 0..n {
            let cross_l = if i >= delay { gain * s_r[i - delay] } else { 0.0 };
            let cross_r = if i >= delay { gain * s_l[i - delay] } else { 0.0 };
            e_l[i] = s_l[i] + cross_l;
            e_r[i] = s_r[i] + cross_r;
        }
        (e_l, e_r)
    }

    #[test]
    fn zero_gain_is_pass_through() {
        let mut c = canceller(0.0, 4);
        let in_l = [1.0, 0.5, -0.25, 0.0, 0.1];
        let in_r = [-0.3, 0.2, 0.4, 0.0, -0.1];
        let mut out_l = [0.0; 5];
        let mut out_r = [0.0; 5];
        let frames = c.process_block(&in_l, &in_r, &mut out_l, &mut out_r);
        assert_eq!(frames, 5);
        assert_eq!(out_l, in_l);
        assert_eq!(out_r, in_r);
    }

    #[test]
    fn cancellation_reconstructs_binaural_at_the_ears() {
        // Feed a binaural target through the canceller, then through the
        // simulated acoustic path; the ears must receive the target back.
        let gain = 0.7;
        let delay = 8;
        let mut c = canceller(gain, delay);
        let n = 256;
        let mut b_l = vec![0.0 as Sample; n];
        let mut b_r = vec![0.0 as Sample; n];
        // Two distinct impulses so left/right are clearly separated.
        b_l[0] = 1.0;
        b_r[16] = 1.0;
        let mut s_l = vec![0.0; n];
        let mut s_r = vec![0.0; n];
        c.process_block(&b_l, &b_r, &mut s_l, &mut s_r);
        let (e_l, e_r) = acoustic_path(&s_l, &s_r, gain, delay);
        for i in 0..n {
            assert!(
                (e_l[i] - b_l[i]).abs() < 1.0e-4,
                "left ear {} != target {} at {i}",
                e_l[i],
                b_l[i]
            );
            assert!(
                (e_r[i] - b_r[i]).abs() < 1.0e-4,
                "right ear {} != target {} at {i}",
                e_r[i],
                b_r[i]
            );
        }
    }

    #[test]
    fn is_stable_for_physical_gain() {
        // A sustained input must not blow up: bounded input, bounded output.
        let mut c = canceller(0.9, 5);
        let n = 4096;
        let in_l = vec![0.5 as Sample; n];
        let in_r = vec![-0.5 as Sample; n];
        let mut out_l = vec![0.0; n];
        let mut out_r = vec![0.0; n];
        c.process_block(&in_l, &in_r, &mut out_l, &mut out_r);
        for &v in out_l.iter().chain(out_r.iter()) {
            assert!(v.is_finite());
            // Bound: 1 / (1 - g) is the DC gain of the recursion.
            assert!(v.abs() < 1.0 / (1.0 - 0.9) + 1.0, "unstable sample {v}");
        }
    }

    #[test]
    fn gain_is_clamped_below_unity() {
        let c = canceller(5.0, 3);
        assert!(c.contralateral_gain() < 1.0);
        assert!(c.contralateral_gain() > 0.99 - 1.0e-6);
    }

    #[test]
    fn delay_is_clamped_to_minimum() {
        let c = canceller(0.5, 0);
        assert_eq!(c.delay_samples(), MIN_CROSSTALK_DELAY);
    }

    #[test]
    fn reset_makes_output_reproducible() {
        let mut c = canceller(0.6, 4);
        let in_l = [0.3, -0.2, 0.5, 0.1, 0.0, -0.4];
        let in_r = [0.1, 0.2, -0.3, 0.4, -0.1, 0.2];
        let mut a_l = [0.0; 6];
        let mut a_r = [0.0; 6];
        c.process_block(&in_l, &in_r, &mut a_l, &mut a_r);
        c.reset();
        let mut b_l = [0.0; 6];
        let mut b_r = [0.0; 6];
        c.process_block(&in_l, &in_r, &mut b_l, &mut b_r);
        assert_eq!(a_l, b_l);
        assert_eq!(a_r, b_r);
    }

    #[test]
    fn mismatched_buffers_truncate() {
        let mut c = canceller(0.5, 2);
        let in_l = [1.0, 2.0, 3.0, 4.0];
        let in_r = [1.0, 2.0];
        let mut out_l = [0.0; 4];
        let mut out_r = [0.0; 3];
        let frames = c.process_block(&in_l, &in_r, &mut out_l, &mut out_r);
        assert_eq!(frames, 2);
    }

    #[test]
    fn empty_block_is_noop() {
        let mut c = canceller(0.5, 2);
        let frames = c.process_block(&[], &[], &mut [], &mut []);
        assert_eq!(frames, 0);
    }

    #[test]
    fn geometry_produces_sane_parameters() {
        let p = CrosstalkParams::from_geometry(
            core::f32::consts::FRAC_PI_6, // 30 degrees
            crate::nearfield::DEFAULT_HEAD_RADIUS,
            FS,
            DEFAULT_SOUND_SPEED,
        );
        // 30-degree pair -> a few tens of microseconds of ITD -> a handful of
        // samples at 48 kHz, always at least the minimum.
        assert!(p.delay_samples >= MIN_CROSSTALK_DELAY);
        assert!(p.delay_samples < 64);
        assert!(p.shadow_cutoff_hz > 0.0);
        assert!(p.contralateral_gain > 0.0 && p.contralateral_gain < 1.0);
    }

    #[test]
    fn geometry_delay_grows_with_angle() {
        let a = crate::nearfield::DEFAULT_HEAD_RADIUS;
        let narrow = CrosstalkParams::from_geometry(0.1, a, FS, DEFAULT_SOUND_SPEED);
        let wide = CrosstalkParams::from_geometry(1.4, a, FS, DEFAULT_SOUND_SPEED);
        assert!(wide.delay_samples >= narrow.delay_samples);
    }

    #[test]
    fn geometry_falls_back_on_bad_inputs() {
        let p = CrosstalkParams::from_geometry(Sample::NAN, -1.0, FS, -5.0);
        assert!(p.delay_samples >= MIN_CROSSTALK_DELAY);
        assert!(p.sample_rate == FS);
    }

    #[test]
    fn shadow_alpha_disabled_above_nyquist() {
        assert_eq!(shadow_alpha(0.0, FS), 1.0);
        assert_eq!(shadow_alpha(30_000.0, FS), 1.0);
        let a = shadow_alpha(1_000.0, FS);
        assert!(a > 0.0 && a < 1.0);
    }

    #[test]
    fn shadow_darkens_but_stays_stable() {
        // With the shadow filter engaged the canceller must still be stable and
        // reconstruct low frequencies (DC) at the ears.
        let params = CrosstalkParams::new(0.7, 8, 4_000.0, FS);
        let mut c = CrosstalkCanceller::new(&params);
        let n = 2048;
        let in_l = vec![0.4 as Sample; n];
        let in_r = vec![0.4 as Sample; n];
        let mut out_l = vec![0.0; n];
        let mut out_r = vec![0.0; n];
        c.process_block(&in_l, &in_r, &mut out_l, &mut out_r);
        for &v in out_l.iter().chain(out_r.iter()) {
            assert!(v.is_finite());
        }
    }
}
