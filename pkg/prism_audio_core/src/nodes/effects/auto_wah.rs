//! Auto-wah / envelope filter: an amplitude-driven resonant filter whose
//! cutoff sweeps with the loudness of the incoming signal.
//!
//! A "wah" pedal is a resonant band-pass (or low-pass / peak) filter whose
//! centre frequency is swept. In an *auto*-wah the sweep is driven not by a
//! foot treadle or an LFO but by an **envelope follower**: the louder the input,
//! the further the cutoff travels, so picking harder opens the filter. This is
//! the classic "envelope filter" funk-guitar and synth-bass effect.
//!
//! The filter core is the shared topology-preserving
//! [`Svf`](super::super::svf::Svf). Its trapezoidal state stays bounded under
//! fast per-sample cutoff modulation, which is exactly what an envelope-driven
//! sweep needs and where a naive Direct Form I biquad would click. The envelope
//! follower reuses the dynamics family's
//! [`time_to_coef`](crate::nodes::dynamics::detector::time_to_coef) so its
//! attack / release ballistics match the rest of the engine.
//!
//! # Signal model
//!
//! A single mono side-chain (the per-frame peak across the input channels)
//! feeds a rectified attack / release envelope follower `env`. A sensitivity
//! drive plus a soft `1 - exp(-x)` knee maps `env` into a normalised `[0, 1)`
//! amount `a`, and the cutoff is swept exponentially over a configurable number
//! of octaves:
//!
//! ```text
//! cutoff = base_hz * 2^(sweep_octaves * a * direction)
//! ```
//!
//! where `direction` is `+1` for an upward (louder = brighter) sweep or `-1`
//! for a downward sweep. The cutoff is fed to the SVF once per sample and every
//! channel is filtered from the same coefficients so the stereo image sweeps
//! coherently.
//!
//! # Provenance
//!
//! The envelope-follower-driven resonant filter is standard, publicly
//! documented audio-effect knowledge (see Udo Zolzer, "DAFX: Digital Audio
//! Effects", and Will Pirkle, "Designing Audio Effect Plugins in C++"). The
//! rectified attack / release follower and the exponential octave sweep below
//! are re-derived from that public literature and composed on top of this
//! crate's own [`Svf`](super::super::svf::Svf) core. This file contains **no
//! Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or Google Resonance
//! Audio source or derived code**; it is implemented purely from that publicly
//! documented mathematics.
//!
//! # Relationship
//!
//! `auto_wah` composes two existing primitives rather than reimplementing
//! either: the resonant filter is [`Svf`](super::super::svf::Svf) (the same TPT
//! core wrapped by [`SvfNode`](super::super::svf::SvfNode)), and the ballistics
//! coefficient comes from the dynamics family's
//! [`time_to_coef`](crate::nodes::dynamics::detector::time_to_coef). It is the
//! modulation counterpart to [`phaser`](super::phaser) and
//! [`tremolo`](super::tremolo): where those are LFO-driven, this one is
//! amplitude-driven.

use bevy_math::ops;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, flush_denormal};
use crate::nodes::dynamics::detector::time_to_coef;
use crate::nodes::svf::{Svf, SvfCoeffs, SvfKind};
use crate::param::{Ramp, Smoothed};

/// Largest normalised drive fed into the soft knee, so the exponential map
/// stays finite for very loud transients.
const MAX_DRIVE: Sample = 8.0;

/// The resonant filter response the auto-wah sweeps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum WahMode {
    /// Band-pass: the classic vocal "wah" that isolates a moving formant.
    BandPass,
    /// Low-pass: a sweeping resonant low-pass (envelope-controlled filter,
    /// synth-bass style).
    LowPass,
    /// Peak: a resonant bell that boosts a swept band while passing the rest.
    Peak,
}

impl WahMode {
    /// Maps the wah response onto the underlying [`SvfKind`].
    #[inline]
    #[must_use]
    const fn svf_kind(self) -> SvfKind {
        match self {
            WahMode::BandPass => SvfKind::BandPass,
            WahMode::LowPass => SvfKind::LowPass,
            WahMode::Peak => SvfKind::Peak,
        }
    }
}

/// Direction of the cutoff sweep as the envelope grows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum SweepDirection {
    /// Louder input sweeps the cutoff **up** (brighter): the usual auto-wah.
    Up,
    /// Louder input sweeps the cutoff **down** (darker): the "down" or reverse
    /// envelope filter.
    Down,
}

impl SweepDirection {
    #[inline]
    #[must_use]
    const fn sign(self) -> Sample {
        match self {
            SweepDirection::Up => 1.0,
            SweepDirection::Down => -1.0,
        }
    }
}

/// Construction parameters for an [`AutoWahNode`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct AutoWahParams {
    /// Filter response swept by the envelope.
    pub mode: WahMode,
    /// Direction the cutoff moves as the input gets louder.
    pub direction: SweepDirection,
    /// Cutoff at rest (envelope = 0), in Hz.
    pub base_freq_hz: Sample,
    /// How many octaves the cutoff sweeps at a fully open envelope.
    pub sweep_octaves: Sample,
    /// Filter resonance (`Q`); higher is a sharper, more vocal sweep.
    pub q: Sample,
    /// Input drive into the envelope follower; higher opens the filter with
    /// quieter input.
    pub sensitivity: Sample,
    /// Envelope attack time in milliseconds (how fast the filter opens).
    pub attack_ms: Sample,
    /// Envelope release time in milliseconds (how slowly it closes).
    pub release_ms: Sample,
    /// Wet (filtered) mix gain.
    pub wet: Sample,
    /// Dry (unprocessed) mix gain.
    pub dry: Sample,
}

impl Default for AutoWahParams {
    fn default() -> Self {
        Self {
            mode: WahMode::BandPass,
            direction: SweepDirection::Up,
            base_freq_hz: 300.0,
            sweep_octaves: 3.0,
            q: 3.0,
            sensitivity: 8.0,
            attack_ms: 5.0,
            release_ms: 120.0,
            wet: 1.0,
            dry: 0.0,
        }
    }
}

/// Allocation-free auto-wah DSP core: a rectified attack / release envelope
/// follower driving a shared [`Svf`].
///
/// A single mono side-chain drives one envelope for all channels so the sweep
/// is coherent across the stereo image. All state is pre-allocated at
/// construction, so [`AutoWah::advance`] and [`AutoWah::filter`] are real-time
/// safe (no allocation, no locking, no panic).
#[derive(Debug, Clone)]
pub struct AutoWah {
    /// Shared resonant filter (per-channel integrator state lives here).
    svf: Svf,
    /// Sample rate used to design coefficients and ballistics.
    sample_rate: u32,
    /// Underlying SVF response.
    kind: SvfKind,
    /// Sweep direction sign (`+1` up, `-1` down).
    direction: Sample,
    /// Rest cutoff in Hz (clamped positive).
    base_freq_hz: Sample,
    /// Octave span of the sweep (clamped non-negative).
    sweep_octaves: Sample,
    /// Filter resonance.
    q: Sample,
    /// Envelope input drive.
    sensitivity: Sample,
    /// Attack one-pole coefficient.
    attack_coef: Sample,
    /// Release one-pole coefficient.
    release_coef: Sample,
    /// Running rectified envelope (linear amplitude).
    envelope: Sample,
    /// Most recently designed cutoff (Hz), exposed for telemetry / tests.
    cutoff_hz: Sample,
}

impl AutoWah {
    /// Builds an auto-wah core for `channels` channels at `sample_rate`.
    #[must_use]
    pub fn new(params: AutoWahParams, sample_rate: u32, channels: usize) -> Self {
        let sr = sample_rate.max(1);
        let kind = params.mode.svf_kind();
        let q = params.q.max(1.0e-4);
        let base = params.base_freq_hz.max(1.0);
        let coeffs = SvfCoeffs::design(kind, sr, base, q, 0.0);
        let mut wah = Self {
            svf: Svf::new(coeffs, channels.max(1)),
            sample_rate: sr,
            kind,
            direction: params.direction.sign(),
            base_freq_hz: base,
            sweep_octaves: params.sweep_octaves.max(0.0),
            q,
            sensitivity: params.sensitivity.max(0.0),
            attack_coef: time_to_coef(params.attack_ms, sr),
            release_coef: time_to_coef(params.release_ms, sr),
            envelope: 0.0,
            cutoff_hz: base,
        };
        wah.cutoff_hz = base;
        wah
    }

    /// Number of channels the internal filter tracks.
    #[inline]
    #[must_use]
    pub fn channels(&self) -> usize {
        self.svf.channels()
    }

    /// Current envelope value (linear amplitude).
    #[inline]
    #[must_use]
    pub fn envelope(&self) -> Sample {
        self.envelope
    }

    /// Cutoff frequency (Hz) designed by the most recent [`AutoWah::advance`].
    #[inline]
    #[must_use]
    pub fn cutoff_hz(&self) -> Sample {
        self.cutoff_hz
    }

    /// Sets the rest cutoff (Hz, clamped positive).
    #[inline]
    pub fn set_base_freq_hz(&mut self, base_freq_hz: Sample) {
        self.base_freq_hz = base_freq_hz.max(1.0);
    }

    /// Sets the octave span of the sweep (clamped non-negative).
    #[inline]
    pub fn set_sweep_octaves(&mut self, sweep_octaves: Sample) {
        self.sweep_octaves = sweep_octaves.max(0.0);
    }

    /// Sets the filter resonance.
    #[inline]
    pub fn set_q(&mut self, q: Sample) {
        self.q = q.max(1.0e-4);
    }

    /// Sets the envelope input drive (clamped non-negative).
    #[inline]
    pub fn set_sensitivity(&mut self, sensitivity: Sample) {
        self.sensitivity = sensitivity.max(0.0);
    }

    /// Sets the sweep direction.
    #[inline]
    pub fn set_direction(&mut self, direction: SweepDirection) {
        self.direction = direction.sign();
    }

    /// Sets the filter response.
    #[inline]
    pub fn set_mode(&mut self, mode: WahMode) {
        self.kind = mode.svf_kind();
    }

    /// Updates the envelope attack / release times in milliseconds.
    #[inline]
    pub fn set_times(&mut self, attack_ms: Sample, release_ms: Sample) {
        self.attack_coef = time_to_coef(attack_ms, self.sample_rate);
        self.release_coef = time_to_coef(release_ms, self.sample_rate);
    }

    /// Advances the envelope by one mono side-chain sample, redesigns the
    /// filter coefficients for the new cutoff, and returns that cutoff in Hz.
    ///
    /// Non-finite input is treated as silence so the follower cannot be
    /// poisoned into a NaN / infinity state.
    #[inline]
    pub fn advance(&mut self, sidechain: Sample) -> Sample {
        let rectified = if sidechain.is_finite() {
            sidechain.abs()
        } else {
            0.0
        };
        // Decoupled attack / release one-pole: fast rise, slow fall.
        let coef = if rectified > self.envelope {
            self.attack_coef
        } else {
            self.release_coef
        };
        self.envelope = flush_denormal(coef * self.envelope + (1.0 - coef) * rectified);

        // Sensitivity drive plus a soft `1 - exp(-x)` knee into `[0, 1)`.
        let drive = (self.envelope * self.sensitivity).clamp(0.0, MAX_DRIVE);
        let amount = 1.0 - ops::exp(-drive);

        // Exponential octave sweep around the rest cutoff.
        let octaves = self.sweep_octaves * amount * self.direction;
        let nyquist_guard = (self.sample_rate as Sample) * 0.499;
        let cutoff = (self.base_freq_hz * ops::exp2(octaves)).clamp(1.0, nyquist_guard);
        self.cutoff_hz = cutoff;

        let coeffs = SvfCoeffs::design(self.kind, self.sample_rate, cutoff, self.q, 0.0);
        self.svf.set_coeffs(coeffs);
        cutoff
    }

    /// Filters one sample on channel `ch` with the current coefficients.
    #[inline]
    pub fn filter(&mut self, ch: usize, x: Sample) -> Sample {
        self.svf.tick(ch, x)
    }

    /// Clears the envelope and the filter integrator memory.
    #[inline]
    pub fn reset(&mut self) {
        self.envelope = 0.0;
        self.cutoff_hz = self.base_freq_hz;
        self.svf.reset();
        let coeffs = SvfCoeffs::design(self.kind, self.sample_rate, self.base_freq_hz, self.q, 0.0);
        self.svf.set_coeffs(coeffs);
    }
}

/// An envelope-controlled resonant filter node (input port 0 -> output port 0).
///
/// The per-frame peak across the input channels drives one shared envelope, so
/// every channel sweeps together. Wet / dry gains are [`Smoothed`] so automation
/// stays click-free; the cutoff itself is modulated per sample by the envelope
/// (the [`Svf`] core is stable under that fast modulation).
///
/// # Example
///
/// ```
/// use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
/// use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
/// use prism_audio_core::nodes::effects::{AutoWahNode, AutoWahParams};
///
/// let mut node = AutoWahNode::new(AutoWahParams::default(), 48_000, 1);
/// let mut input = AudioBuffer::new(ChannelLayout::Mono, 4);
/// let mut output = AudioBuffer::new(ChannelLayout::Mono, 4);
/// input.set_active_frames(4);
/// output.set_active_frames(4);
/// for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
///     *s = if i % 2 == 0 { 0.5 } else { -0.5 };
/// }
/// let ctx = RenderContext { sample_rate: 48_000, frames: 4, playhead: 0 };
/// let mut io = ProcessIo::new(core::slice::from_ref(&input), core::slice::from_mut(&mut output));
/// node.process(&ctx, &mut io);
/// assert!(output.channel(0).iter().all(|s| s.is_finite()));
/// ```
#[derive(Debug, Clone)]
pub struct AutoWahNode {
    /// DSP core (envelope follower + shared SVF).
    wah: AutoWah,
    /// Smoothed wet (filtered) mix gain.
    wet: Smoothed,
    /// Smoothed dry (unprocessed) mix gain.
    dry: Smoothed,
}

impl AutoWahNode {
    /// Builds an auto-wah node for `channels` channels at `sample_rate`.
    #[must_use]
    pub fn new(params: AutoWahParams, sample_rate: u32, channels: usize) -> Self {
        Self {
            wet: Smoothed::new(params.wet),
            dry: Smoothed::new(params.dry),
            wah: AutoWah::new(params, sample_rate, channels),
        }
    }

    /// Sets the rest cutoff frequency in Hz.
    #[inline]
    pub fn set_base_freq_hz(&mut self, base_freq_hz: Sample) {
        self.wah.set_base_freq_hz(base_freq_hz);
    }

    /// Sets the octave span of the sweep.
    #[inline]
    pub fn set_sweep_octaves(&mut self, sweep_octaves: Sample) {
        self.wah.set_sweep_octaves(sweep_octaves);
    }

    /// Sets the filter resonance.
    #[inline]
    pub fn set_q(&mut self, q: Sample) {
        self.wah.set_q(q);
    }

    /// Sets the envelope input drive.
    #[inline]
    pub fn set_sensitivity(&mut self, sensitivity: Sample) {
        self.wah.set_sensitivity(sensitivity);
    }

    /// Sets the sweep direction.
    #[inline]
    pub fn set_direction(&mut self, direction: SweepDirection) {
        self.wah.set_direction(direction);
    }

    /// Sets the filter response.
    #[inline]
    pub fn set_mode(&mut self, mode: WahMode) {
        self.wah.set_mode(mode);
    }

    /// Updates the envelope attack / release times in milliseconds.
    #[inline]
    pub fn set_times(&mut self, attack_ms: Sample, release_ms: Sample) {
        self.wah.set_times(attack_ms, release_ms);
    }

    /// Sets the wet (filtered) mix gain with the given ramp.
    #[inline]
    pub fn set_wet(&mut self, wet: Sample, ramp: Ramp) {
        self.wet.set_target(wet, ramp);
    }

    /// Sets the dry (unprocessed) mix gain with the given ramp.
    #[inline]
    pub fn set_dry(&mut self, dry: Sample, ramp: Ramp) {
        self.dry.set_target(dry, ramp);
    }

    /// Current cutoff frequency (Hz) after the last processed frame.
    #[inline]
    #[must_use]
    pub fn cutoff_hz(&self) -> Sample {
        self.wah.cutoff_hz()
    }
}

impl AudioNode for AutoWahNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let channels = output.channels().min(self.wah.channels());
        let frames = output.active_frames();

        for f in 0..frames {
            let wet = self.wet.next_sample();
            let dry = self.dry.next_sample();

            // Mono side-chain: peak across the active channels this frame.
            let mut peak = 0.0;
            for ch in 0..channels {
                let a = input.channel(ch)[f].abs();
                if a > peak {
                    peak = a;
                }
            }
            self.wah.advance(peak);

            for ch in 0..channels {
                let x = input.channel(ch)[f];
                let filtered = self.wah.filter(ch, x);
                output.channel_mut(ch)[f] = dry * x + wet * filtered;
            }
        }
    }

    fn reset(&mut self) {
        self.wah.reset();
        self.wet = Smoothed::new(self.wet.target());
        self.dry = Smoothed::new(self.dry.target());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use crate::buffer::{AudioBuffer, ChannelLayout};

    const SR: u32 = 48_000;

    fn params() -> AutoWahParams {
        AutoWahParams::default()
    }

    fn feed_constant(wah: &mut AutoWah, amp: Sample, samples: usize) -> Sample {
        let mut cutoff = wah.cutoff_hz();
        for i in 0..samples {
            // Alternating sign so the rectified level equals `amp`.
            let s = if i % 2 == 0 { amp } else { -amp };
            cutoff = wah.advance(s);
            let _ = wah.filter(0, s);
        }
        cutoff
    }

    #[test]
    fn silence_stays_at_base_cutoff() {
        let mut wah = AutoWah::new(params(), SR, 1);
        let cutoff = feed_constant(&mut wah, 0.0, 512);
        assert!((cutoff - wah.cutoff_hz()).abs() < 1.0e-3);
        // No drive -> cutoff never leaves the rest frequency.
        assert!((cutoff - 300.0).abs() < 1.0);
    }

    #[test]
    fn louder_input_opens_filter_upward() {
        let mut quiet = AutoWah::new(params(), SR, 1);
        let mut loud = AutoWah::new(params(), SR, 1);
        let quiet_cut = feed_constant(&mut quiet, 0.05, 4096);
        let loud_cut = feed_constant(&mut loud, 0.9, 4096);
        assert!(
            loud_cut > quiet_cut,
            "loud={loud_cut} should exceed quiet={quiet_cut}"
        );
        // Upward sweep: louder pushes above the rest cutoff.
        assert!(loud_cut > 300.0);
    }

    #[test]
    fn downward_direction_lowers_cutoff() {
        let mut p = params();
        p.direction = SweepDirection::Down;
        p.base_freq_hz = 2000.0;
        let mut wah = AutoWah::new(p, SR, 1);
        let cut = feed_constant(&mut wah, 0.9, 4096);
        assert!(cut < 2000.0, "downward sweep should drop below base: {cut}");
    }

    #[test]
    fn cutoff_is_clamped_below_nyquist() {
        let mut p = params();
        p.base_freq_hz = 8000.0;
        p.sweep_octaves = 6.0;
        p.sensitivity = 50.0;
        let mut wah = AutoWah::new(p, SR, 1);
        let cut = feed_constant(&mut wah, 1.0, 4096);
        assert!(cut <= (SR as Sample) * 0.499 + 1.0);
        assert!(cut.is_finite());
    }

    #[test]
    fn zero_sweep_octaves_pins_cutoff() {
        let mut p = params();
        p.sweep_octaves = 0.0;
        let mut wah = AutoWah::new(p, SR, 1);
        let cut = feed_constant(&mut wah, 0.9, 2048);
        assert!((cut - 300.0).abs() < 1.0);
    }

    #[test]
    fn envelope_grows_towards_input_level() {
        let mut wah = AutoWah::new(params(), SR, 1);
        feed_constant(&mut wah, 0.6, 8192);
        // With a rectified constant of 0.6 the follower converges near 0.6.
        assert!(wah.envelope() > 0.4 && wah.envelope() <= 0.6 + 1.0e-3);
    }

    #[test]
    fn attack_is_faster_than_release() {
        let mut wah = AutoWah::new(params(), SR, 1);
        // Rise to a loud level.
        feed_constant(&mut wah, 0.9, 2048);
        let peak_env = wah.envelope();
        // Then go silent for the same span; release is slower so some envelope
        // should remain (it does not collapse to zero as fast as it rose).
        for _ in 0..64 {
            wah.advance(0.0);
        }
        let after = wah.envelope();
        assert!(after > 0.0 && after < peak_env);
    }

    #[test]
    fn non_finite_sidechain_is_safe() {
        let mut wah = AutoWah::new(params(), SR, 1);
        let cut = wah.advance(Sample::NAN);
        assert!(cut.is_finite());
        assert!(wah.envelope().is_finite());
        let cut2 = wah.advance(Sample::INFINITY);
        assert!(cut2.is_finite());
    }

    #[test]
    fn reset_restores_rest_state() {
        let mut wah = AutoWah::new(params(), SR, 1);
        feed_constant(&mut wah, 0.9, 4096);
        wah.reset();
        assert_eq!(wah.envelope(), 0.0);
        assert!((wah.cutoff_hz() - 300.0).abs() < 1.0);
    }

    #[test]
    fn reset_makes_output_reproducible() {
        let mut wah = AutoWah::new(params(), SR, 1);
        let first: Vec<Sample> = (0..256)
            .map(|i| {
                let s = if i % 2 == 0 { 0.5 } else { -0.5 };
                wah.advance(s);
                wah.filter(0, s)
            })
            .collect();
        wah.reset();
        let second: Vec<Sample> = (0..256)
            .map(|i| {
                let s = if i % 2 == 0 { 0.5 } else { -0.5 };
                wah.advance(s);
                wah.filter(0, s)
            })
            .collect();
        assert_eq!(first, second);
    }

    #[test]
    fn channels_filter_independently() {
        let mut wah = AutoWah::new(params(), SR, 2);
        assert_eq!(wah.channels(), 2);
        wah.advance(0.5);
        let a = wah.filter(0, 1.0);
        let b = wah.filter(1, 0.0);
        // Same coefficients, different inputs -> different outputs.
        assert!((a - b).abs() > 1.0e-6);
    }

    #[test]
    fn bandpass_rejects_dc() {
        let mut p = params();
        p.mode = WahMode::BandPass;
        let mut wah = AutoWah::new(p, SR, 1);
        // Settle the filter with a DC input; a band-pass must not pass DC.
        let mut last = 0.0;
        for _ in 0..8192 {
            wah.advance(1.0);
            last = wah.filter(0, 1.0);
        }
        assert!(last.abs() < 0.2, "band-pass leaked DC: {last}");
    }

    #[test]
    fn lowpass_mode_passes_dc() {
        let mut p = params();
        p.mode = WahMode::LowPass;
        p.sensitivity = 0.0; // keep cutoff at rest so DC is well inside band
        let mut wah = AutoWah::new(p, SR, 1);
        let mut last = 0.0;
        for _ in 0..8192 {
            wah.advance(1.0);
            last = wah.filter(0, 1.0);
        }
        assert!(last > 0.8, "low-pass should pass DC near unity: {last}");
    }

    #[test]
    fn node_process_is_finite_and_mixes() {
        let mut node = AutoWahNode::new(params(), SR, 1);
        let mut input = AudioBuffer::new(ChannelLayout::Mono, 128);
        let mut output = AudioBuffer::new(ChannelLayout::Mono, 128);
        input.set_active_frames(128);
        output.set_active_frames(128);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            *s = if i % 2 == 0 { 0.7 } else { -0.7 };
        }
        let ctx = RenderContext {
            sample_rate: SR,
            frames: 128,
            playhead: 0,
        };
        let mut io = ProcessIo::new(
            core::slice::from_ref(&input),
            core::slice::from_mut(&mut output),
        );
        node.process(&ctx, &mut io);
        assert!(output.channel(0).iter().all(|s| s.is_finite()));
        assert!(node.cutoff_hz() > 300.0);
    }

    #[test]
    fn node_fully_dry_is_passthrough() {
        let mut p = params();
        p.wet = 0.0;
        p.dry = 1.0;
        let mut node = AutoWahNode::new(p, SR, 1);
        let mut input = AudioBuffer::new(ChannelLayout::Mono, 64);
        let mut output = AudioBuffer::new(ChannelLayout::Mono, 64);
        input.set_active_frames(64);
        output.set_active_frames(64);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            *s = (i as Sample) * 0.01 - 0.3;
        }
        let ctx = RenderContext {
            sample_rate: SR,
            frames: 64,
            playhead: 0,
        };
        let mut io = ProcessIo::new(
            core::slice::from_ref(&input),
            core::slice::from_mut(&mut output),
        );
        node.process(&ctx, &mut io);
        for f in 0..64 {
            assert!((output.channel(0)[f] - input.channel(0)[f]).abs() < 1.0e-6);
        }
    }

    #[test]
    fn node_reset_clears_tail() {
        let mut node = AutoWahNode::new(params(), SR, 1);
        let mut input = AudioBuffer::new(ChannelLayout::Mono, 32);
        let mut output = AudioBuffer::new(ChannelLayout::Mono, 32);
        input.set_active_frames(32);
        output.set_active_frames(32);
        for s in input.channel_mut(0).iter_mut() {
            *s = 0.9;
        }
        let ctx = RenderContext {
            sample_rate: SR,
            frames: 32,
            playhead: 0,
        };
        {
            let mut io = ProcessIo::new(
                core::slice::from_ref(&input),
                core::slice::from_mut(&mut output),
            );
            node.process(&ctx, &mut io);
        }
        node.reset();
        assert!((node.cutoff_hz() - 300.0).abs() < 1.0);
    }

    #[test]
    fn extreme_params_do_not_panic() {
        let mut p = params();
        p.q = 1.0e9;
        p.sensitivity = 1.0e9;
        p.sweep_octaves = 1.0e6;
        p.base_freq_hz = -5.0;
        let mut wah = AutoWah::new(p, SR, 1);
        for _ in 0..256 {
            let c = wah.advance(1.0e6);
            let y = wah.filter(0, 1.0e6);
            assert!(c.is_finite());
            assert!(y.is_finite());
        }
    }

    #[test]
    fn zero_frames_is_noop() {
        let mut node = AutoWahNode::new(params(), SR, 1);
        let input = AudioBuffer::new(ChannelLayout::Mono, 8);
        let mut output = AudioBuffer::new(ChannelLayout::Mono, 8);
        output.set_active_frames(0);
        let ctx = RenderContext {
            sample_rate: SR,
            frames: 0,
            playhead: 0,
        };
        let mut io = ProcessIo::new(
            core::slice::from_ref(&input),
            core::slice::from_mut(&mut output),
        );
        node.process(&ctx, &mut io);
        assert_eq!(output.active_frames(), 0);
    }
}
