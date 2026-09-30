//! Occlusion and obstruction: attenuate and low-pass a source when geometry
//! blocks the path between it and the listener.
//!
//! This module follows the semantic split popularised by mainstream audio
//! middleware, expressed here as two independent, normalised factors:
//!
//! * **Obstruction** - only the *direct* path is blocked (the source and the
//!   listener share the same acoustic space, but an object sits between them).
//!   The direct sound is attenuated and low-passed; any reverb/aux (wet) send
//!   is left intact, because reflected energy still reaches the listener.
//! * **Occlusion** - *both* the direct and the reverberant paths are blocked
//!   (the source is in a different space entirely). The direct sound is
//!   attenuated and low-passed, *and* the wet send is scaled down.
//!
//! # Layering
//!
//! Geometry is supplied by a pluggable [`OcclusionQuery`] backend (typically
//! backed by physics ray casts in a higher layer) rather than computed here, so
//! this crate stays free of any physics dependency. [`NullOcclusionQuery`]
//! provides an always-open default for tests and headless use.
//!
//! The mapping from factors to DSP parameters is a pure, control-rate function
//! ([`Occlusion::resolve`]). [`OcclusionNode`] applies the *direct-path*
//! processing (smoothed gain + low-pass) to a signal; the resolved
//! [`OcclusionParams::wet_gain`] is exposed for the graph to apply to the aux
//! send, because routing lives outside a single node.
//!
//! # Real-time contract
//!
//! [`OcclusionNode::process`] is allocation, lock, and panic free. Following
//! the same discipline as [`crate::air`], the low-pass corner is only ever
//! re-designed off the audio thread (in [`OcclusionNode::set_factors`]); the
//! per-sample hot path advances a [`Smoothed`] gain and runs a fixed biquad, so
//! no coefficient math happens per sample.
//!
//! # Determinism
//!
//! The cut-off is interpolated in the log-frequency domain through
//! [`bevy_math::ops`] (libm-backed `ln`/`exp`) rather than `f32` intrinsics, so
//! results are bit-reproducible across targets.
//!
//! # Provenance
//!
//! This is a from-scratch implementation of the well-known occlusion vs
//! obstruction model (direct-path gain + low-pass, plus wet-send scaling for
//! occlusion). It contains **no Unreal Engine, Unity, Godot, Wwise, or FMOD
//! source or derived code**; only publicly documented acoustics knowledge is
//! used.

use bevy_math::ops;
use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
use prism_audio_core::math::{Sample, db_to_linear};
use prism_audio_core::nodes::biquad::{Biquad, BiquadCoeffs, BiquadKind};
use prism_audio_core::param::{Ramp, Smoothed};

use crate::geometry::{Emitter, Listener};

/// Butterworth (maximally flat) quality factor for the direct-path low-pass,
/// matching the convention used by [`crate::air`].
const LOWPASS_Q: Sample = core::f32::consts::FRAC_1_SQRT_2;

/// Two normalised blocking factors describing how much geometry sits between an
/// emitter and the listener.
///
/// Both fields are linear factors in `[0, 1]`, where `0` is a fully open path
/// and `1` is fully blocked. They are clamped into range by [`Self::new`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct OcclusionFactors {
    /// How much the *direct* path alone is blocked, in `[0, 1]`.
    pub obstruction: Sample,
    /// How much *both* the direct and reverberant paths are blocked, in
    /// `[0, 1]`.
    pub occlusion: Sample,
}

impl OcclusionFactors {
    /// A fully open path: nothing is blocked.
    pub const OPEN: Self = Self {
        obstruction: 0.0,
        occlusion: 0.0,
    };

    /// Builds factors, clamping both inputs into `[0, 1]`.
    #[inline]
    #[must_use]
    pub fn new(obstruction: Sample, occlusion: Sample) -> Self {
        Self {
            obstruction: obstruction.clamp(0.0, 1.0),
            occlusion: occlusion.clamp(0.0, 1.0),
        }
    }

    /// The effective blocking of the *direct* path.
    ///
    /// Both obstruction and occlusion attenuate the direct sound, so the direct
    /// path uses the stronger of the two.
    #[inline]
    #[must_use]
    pub fn direct_factor(&self) -> Sample {
        self.obstruction.max(self.occlusion)
    }
}

impl Default for OcclusionFactors {
    #[inline]
    fn default() -> Self {
        Self::OPEN
    }
}

/// A pluggable geometry backend that reports how blocked a source is.
///
/// Implementors (typically in a physics-aware layer) trace the listener-to-
/// emitter path and return normalised [`OcclusionFactors`]. Keeping this as a
/// trait lets the spatial crate stay physics-agnostic.
pub trait OcclusionQuery: Send {
    /// Reports the current blocking factors for `emitter` heard from
    /// `listener`.
    fn query(&self, listener: &Listener, emitter: &Emitter) -> OcclusionFactors;
}

/// An [`OcclusionQuery`] that always reports a fully open path.
///
/// Useful as a default backend for tests, headless rendering, or scenes with no
/// occluding geometry.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NullOcclusionQuery;

impl OcclusionQuery for NullOcclusionQuery {
    #[inline]
    fn query(&self, _listener: &Listener, _emitter: &Emitter) -> OcclusionFactors {
        OcclusionFactors::OPEN
    }
}

/// Configuration mapping [`OcclusionFactors`] to concrete DSP parameters.
///
/// The mapping is linear in decibels for gain (fully blocked reaches
/// `max_attenuation_db` of attenuation) and logarithmic in frequency for the
/// low-pass corner (gliding from `open_cutoff_hz` at factor `0` down to
/// `blocked_cutoff_hz` at factor `1`).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Occlusion {
    /// Attenuation applied to a fully blocked path, in decibels (>= 0). At
    /// factor `0` no attenuation is applied.
    pub max_attenuation_db: Sample,
    /// Low-pass corner for a fully open path, in Hz. Effectively "no filtering"
    /// when set at or above the Nyquist frequency.
    pub open_cutoff_hz: Sample,
    /// Low-pass corner for a fully blocked path, in Hz.
    pub blocked_cutoff_hz: Sample,
}

impl Default for Occlusion {
    /// A moderate, generic profile: up to 24 dB of attenuation, gliding the
    /// corner from 20 kHz (open) down to 700 Hz (fully blocked).
    #[inline]
    fn default() -> Self {
        Self {
            max_attenuation_db: 24.0,
            open_cutoff_hz: 20_000.0,
            blocked_cutoff_hz: 700.0,
        }
    }
}

/// Resolved, control-rate DSP targets for a given set of factors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OcclusionParams {
    /// Linear gain applied to the direct signal, in `[0, 1]`.
    pub direct_gain: Sample,
    /// Low-pass corner applied to the direct signal, in Hz.
    pub direct_cutoff_hz: Sample,
    /// Multiplicative scale for the source's reverb/aux (wet) send, in `[0, 1]`.
    /// Only driven by the occlusion factor; obstruction leaves it at unity.
    pub wet_gain: Sample,
}

impl Occlusion {
    /// Sanitises the configuration: attenuation is forced non-negative and the
    /// two corner frequencies are ordered so `blocked <= open`.
    #[inline]
    #[must_use]
    pub fn new(max_attenuation_db: Sample, open_cutoff_hz: Sample, blocked_cutoff_hz: Sample) -> Self {
        let att = max_attenuation_db.max(0.0);
        let open = open_cutoff_hz.max(0.0);
        let blocked = blocked_cutoff_hz.max(0.0).min(open);
        Self {
            max_attenuation_db: att,
            open_cutoff_hz: open,
            blocked_cutoff_hz: blocked,
        }
    }

    /// Maps `factors` to concrete DSP targets.
    ///
    /// * The direct gain uses the stronger of the two factors
    ///   ([`OcclusionFactors::direct_factor`]).
    /// * The direct cut-off glides in log-frequency from `open_cutoff_hz` to
    ///   `blocked_cutoff_hz` over the same direct factor.
    /// * The wet gain is driven by the occlusion factor alone (obstruction does
    ///   not touch the reverberant path).
    #[must_use]
    pub fn resolve(&self, factors: OcclusionFactors) -> OcclusionParams {
        let direct = factors.direct_factor();
        let direct_gain = db_to_linear(-self.max_attenuation_db * direct);
        let wet_gain = db_to_linear(-self.max_attenuation_db * factors.occlusion);
        OcclusionParams {
            direct_gain,
            direct_cutoff_hz: self.interp_cutoff(direct),
            wet_gain,
        }
    }

    /// Interpolates the low-pass corner in the log-frequency domain for a
    /// blocking factor in `[0, 1]`.
    ///
    /// `cutoff = open * (blocked / open) ^ factor`, evaluated as
    /// `exp(ln(open) + factor * (ln(blocked) - ln(open)))`. Falls back to
    /// linear endpoints if either corner is non-positive.
    fn interp_cutoff(&self, factor: Sample) -> Sample {
        let f = factor.clamp(0.0, 1.0);
        // Snap the endpoints so the fully-open/fully-blocked corners are exact
        // (avoids ln/exp round-trip error at the boundaries).
        if f <= 0.0 {
            return self.open_cutoff_hz;
        }
        if f >= 1.0 {
            return self.blocked_cutoff_hz;
        }
        if self.open_cutoff_hz <= 0.0 || self.blocked_cutoff_hz <= 0.0 {
            // Degenerate corner(s): fall back to a plain linear blend.
            return self.open_cutoff_hz + (self.blocked_cutoff_hz - self.open_cutoff_hz) * f;
        }
        let ln_open = ops::ln(self.open_cutoff_hz);
        let ln_blocked = ops::ln(self.blocked_cutoff_hz);
        ops::exp(ln_open + f * (ln_blocked - ln_open))
    }
}

/// A real-time node that applies the *direct-path* occlusion processing:
/// a per-sample smoothed gain followed by a low-pass filter.
///
/// The wet-send scale is *not* applied here (routing is external); read it from
/// the last [`OcclusionParams`] returned by [`Self::set_factors`] and apply it
/// to the aux send in the graph.
pub struct OcclusionNode {
    config: Occlusion,
    gain: Smoothed,
    filter: Biquad,
    cutoff_hz: Sample,
    wet_gain: Sample,
}

impl OcclusionNode {
    /// Builds a node for `channels` channels at `sample_rate`, starting fully
    /// open (unit gain, `open_cutoff_hz`).
    #[must_use]
    pub fn new(config: Occlusion, channels: usize, sample_rate: u32) -> Self {
        let params = config.resolve(OcclusionFactors::OPEN);
        let coeffs = BiquadCoeffs::design(
            BiquadKind::LowPass,
            sample_rate,
            params.direct_cutoff_hz,
            LOWPASS_Q,
            0.0,
        );
        Self {
            config,
            gain: Smoothed::new(params.direct_gain),
            filter: Biquad::new(coeffs, channels),
            cutoff_hz: params.direct_cutoff_hz,
            wet_gain: params.wet_gain,
        }
    }

    /// Updates the blocking factors, gliding the direct gain over `ramp` and
    /// re-tuning the low-pass corner. **Non-real-time**: this re-designs the
    /// biquad, so call it from the control thread, not from
    /// [`AudioNode::process`]. Filter state is preserved (no click).
    ///
    /// Returns the resolved [`OcclusionParams`] so the caller can apply
    /// [`OcclusionParams::wet_gain`] to the aux send.
    pub fn set_factors(
        &mut self,
        factors: OcclusionFactors,
        ramp: Ramp,
        sample_rate: u32,
    ) -> OcclusionParams {
        let params = self.config.resolve(factors);
        self.gain.set_target(params.direct_gain, ramp);
        if params.direct_cutoff_hz != self.cutoff_hz {
            self.cutoff_hz = params.direct_cutoff_hz;
            self.filter.set_coeffs(BiquadCoeffs::design(
                BiquadKind::LowPass,
                sample_rate,
                self.cutoff_hz,
                LOWPASS_Q,
                0.0,
            ));
        }
        self.wet_gain = params.wet_gain;
        params
    }

    /// The most recently resolved wet-send scale, in `[0, 1]`.
    #[inline]
    #[must_use]
    pub fn wet_gain(&self) -> Sample {
        self.wet_gain
    }

    /// The current direct-path low-pass corner, in Hz.
    #[inline]
    #[must_use]
    pub fn cutoff_hz(&self) -> Sample {
        self.cutoff_hz
    }
}

impl AudioNode for OcclusionNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        if input.channels() == 0 || output.channels() == 0 {
            return;
        }

        // Match the output length to the input, then copy channel by channel.
        // `set_active_frames` saturates at capacity and the per-channel `min`
        // keeps the copy in bounds, so nothing here can panic.
        let frames = input.active_frames();
        output.set_active_frames(frames);
        let channels = input.channels().min(output.channels());
        for c in 0..channels {
            let src = input.channel(c);
            let dst = output.channel_mut(c);
            let n = src.len().min(dst.len());
            dst[..n].copy_from_slice(&src[..n]);
        }

        // Apply the per-sample smoothed gain uniformly across channels first so
        // every channel shares the same gain trajectory, then low-pass in place.
        let active = output.active_frames();
        for i in 0..active {
            let g = self.gain.next_sample();
            for c in 0..channels {
                output.channel_mut(c)[i] *= g;
            }
        }
        self.filter.process_inplace(output);
    }

    fn reset(&mut self) {
        self.gain = Smoothed::new(self.gain.target());
        self.filter.reset();
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

    fn ctx(sample_rate: u32, frames: usize) -> RenderContext {
        RenderContext {
            sample_rate,
            frames,
            playhead: 0,
        }
    }

    #[test]
    fn factors_are_clamped() {
        let f = OcclusionFactors::new(-0.5, 2.0);
        assert!(approx(f.obstruction, 0.0));
        assert!(approx(f.occlusion, 1.0));
    }

    #[test]
    fn direct_factor_is_the_stronger_block() {
        let f = OcclusionFactors::new(0.3, 0.8);
        assert!(approx(f.direct_factor(), 0.8));
        let f = OcclusionFactors::new(0.9, 0.2);
        assert!(approx(f.direct_factor(), 0.9));
    }

    #[test]
    fn open_path_is_transparent() {
        let occ = Occlusion::default();
        let p = occ.resolve(OcclusionFactors::OPEN);
        assert!(approx(p.direct_gain, 1.0));
        assert!(approx(p.wet_gain, 1.0));
        assert!(approx(p.direct_cutoff_hz, occ.open_cutoff_hz));
    }

    #[test]
    fn fully_blocked_reaches_max_attenuation() {
        let occ = Occlusion::new(24.0, 20_000.0, 700.0);
        let p = occ.resolve(OcclusionFactors::new(1.0, 1.0));
        // -24 dB is a factor of ~0.0631.
        assert!(approx(p.direct_gain, db_to_linear(-24.0)));
        assert!(approx(p.wet_gain, db_to_linear(-24.0)));
        assert!(approx(p.direct_cutoff_hz, 700.0));
    }

    #[test]
    fn obstruction_does_not_scale_wet_send() {
        let occ = Occlusion::default();
        // Pure obstruction: direct is attenuated, wet stays open.
        let p = occ.resolve(OcclusionFactors::new(1.0, 0.0));
        assert!(p.direct_gain < 0.2, "direct gain={}", p.direct_gain);
        assert!(approx(p.wet_gain, 1.0));
    }

    #[test]
    fn cutoff_is_monotonic_in_factor() {
        let occ = Occlusion::default();
        let a = occ.resolve(OcclusionFactors::new(0.25, 0.0)).direct_cutoff_hz;
        let b = occ.resolve(OcclusionFactors::new(0.5, 0.0)).direct_cutoff_hz;
        let c = occ.resolve(OcclusionFactors::new(0.75, 0.0)).direct_cutoff_hz;
        assert!(a > b && b > c, "cutoffs a={a} b={b} c={c}");
        assert!(c > occ.blocked_cutoff_hz - 1.0 && a < occ.open_cutoff_hz);
    }

    #[test]
    fn cutoff_midpoint_is_geometric_mean() {
        let occ = Occlusion::new(24.0, 16_000.0, 250.0);
        let mid = occ.resolve(OcclusionFactors::new(0.5, 0.0)).direct_cutoff_hz;
        // Log-domain midpoint = sqrt(open * blocked).
        let expected = ops::sqrt(16_000.0 * 250.0);
        assert!((mid - expected).abs() / expected < 1.0e-4, "mid={mid} exp={expected}");
    }

    #[test]
    fn degenerate_cutoffs_fall_back_to_linear() {
        let occ = Occlusion {
            max_attenuation_db: 12.0,
            open_cutoff_hz: 0.0,
            blocked_cutoff_hz: 0.0,
        };
        // Must not produce NaN/inf via ln(0).
        let p = occ.resolve(OcclusionFactors::new(0.5, 0.5));
        assert!(p.direct_cutoff_hz.is_finite());
    }

    #[test]
    fn null_query_is_always_open() {
        let q = NullOcclusionQuery;
        let listener = Listener::default();
        let emitter = Emitter::point(bevy_math::Vec3::new(1.0, 0.0, 0.0), bevy_math::Vec3::ZERO);
        assert_eq!(q.query(&listener, &emitter), OcclusionFactors::OPEN);
    }

    #[test]
    fn node_open_path_preserves_signal_energy() {
        let sample_rate = 48_000;
        let frames = 64;
        let mut node = OcclusionNode::new(Occlusion::default(), 1, sample_rate);
        // Fully open: unit gain, corner at Nyquist-ish (no meaningful roll-off
        // for a low-frequency test tone). Settle the gain immediately.
        node.set_factors(OcclusionFactors::OPEN, Ramp::Immediate, sample_rate);

        let mut input = AudioBuffer::new(ChannelLayout::Mono, frames);
        for (n, s) in input.channel_mut(0).iter_mut().enumerate() {
            // Low-frequency ramp that the open low-pass leaves essentially intact.
            *s = 0.5 * (n as Sample / frames as Sample);
        }
        input.set_active_frames(frames);
        let inputs = [input];
        let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, frames)];

        let c = ctx(sample_rate, frames);
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&c, &mut io);

        assert_eq!(outputs[0].active_frames(), frames);
        // The tail sample (largest input) should remain close to its input.
        let last = outputs[0].channel(0)[frames - 1];
        assert!(last > 0.4, "open path attenuated too much: {last}");
    }

    #[test]
    fn node_blocked_path_attenuates() {
        let sample_rate = 48_000;
        let frames = 128;
        let mut node = OcclusionNode::new(Occlusion::default(), 1, sample_rate);
        node.set_factors(OcclusionFactors::new(1.0, 1.0), Ramp::Immediate, sample_rate);

        let mut input = AudioBuffer::new(ChannelLayout::Mono, frames);
        for s in input.channel_mut(0).iter_mut() {
            *s = 0.8;
        }
        input.set_active_frames(frames);
        let inputs = [input];
        let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, frames)];

        let c = ctx(sample_rate, frames);
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&c, &mut io);

        // Fully blocked: the DC-ish input is attenuated well below its level.
        let tail = outputs[0].channel(0)[frames - 1];
        assert!(tail.abs() < 0.3, "blocked path not attenuated: {tail}");
    }
}
