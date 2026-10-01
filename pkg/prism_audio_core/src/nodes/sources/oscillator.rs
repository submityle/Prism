//! Band-limited virtual-analog oscillator source node.
//!
//! [`OscillatorNode`] is a *source* (zero inputs, one output) that synthesizes a
//! classic analog waveform — sine, sawtooth, square, or triangle — from a
//! phase accumulator. The naive sawtooth and square shapes contain hard
//! discontinuities that alias badly when sampled; this node suppresses that
//! aliasing with **`PolyBLEP`** (polynomial band-limited step) correction, which
//! rounds the sample or two straddling each discontinuity so the spectrum stays
//! (nearly) band-limited without the cost of full BLIT/wavetable synthesis.
//!
//! The band-limited triangle is derived by running the band-limited square
//! through a leaky integrator, which is the standard virtual-analog trick: the
//! integral of a square is a triangle, and the leak keeps the integrator from
//! accumulating DC drift.
//!
//! # Real-time contract
//!
//! Construction pre-computes all state, so [`OscillatorNode::process`] performs
//! no allocation, no locking, and no panicking: it is a pure per-sample state
//! machine. Amplitude is driven through a [`Smoothed`] value so that gain
//! automation never produces zipper-noise clicks.

use bevy_math::ops;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::Sample;
use crate::param::{Ramp, Smoothed};

/// Full turn in radians, used to map the normalized phase to the sine argument.
const TAU: Sample = core::f32::consts::TAU;

/// The set of classic analog waveforms [`OscillatorNode`] can synthesize.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum Waveform {
    /// Pure sine wave. Naturally band-limited, so no `PolyBLEP` is applied.
    Sine,
    /// Sawtooth (ramp) wave, band-limited with a single `PolyBLEP` correction at
    /// the wrap discontinuity.
    Saw,
    /// Square (50% duty pulse) wave, band-limited with two `PolyBLEP`
    /// corrections: one at the rising edge and one at the falling edge.
    Square,
    /// Triangle wave, produced by leaky-integrating the band-limited square.
    Triangle,
}

/// A band-limited oscillator source node.
///
/// Zero inputs, one output. Every output channel receives the same mono
/// waveform so that stereo/surround downstream nodes see a coherent signal.
#[derive(Debug, Clone)]
pub struct OscillatorNode {
    /// Currently selected output waveform.
    waveform: Waveform,
    /// Oscillator frequency in Hertz. Always non-negative.
    ///
    /// Frequency is stored as a plain scalar rather than a [`Smoothed`] value:
    /// because the phase accumulator is continuous, an instantaneous frequency
    /// change only alters the slope of the waveform, not its instantaneous
    /// amplitude, so it introduces no discontinuity (click). Amplitude, by
    /// contrast, is a multiplicative gain applied per sample and *is* smoothed.
    frequency: Sample,
    /// Smoothed linear output amplitude (a gain multiplier, not decibels).
    amplitude: Smoothed,
    /// Normalized phase accumulator in `[0.0, 1.0)`.
    phase: Sample,
    /// Leaky-integrator state carrying the band-limited triangle between
    /// samples.
    triangle_state: Sample,
}

impl OscillatorNode {
    /// Creates an oscillator emitting `waveform` at `frequency_hz`, with its
    /// amplitude settled at `amplitude` (a linear multiplier).
    ///
    /// Negative frequencies are clamped to `0.0`.
    #[must_use]
    pub fn new(waveform: Waveform, frequency_hz: Sample, amplitude: Sample) -> Self {
        Self {
            waveform,
            frequency: frequency_hz.max(0.0),
            amplitude: Smoothed::new(amplitude),
            phase: 0.0,
            triangle_state: 0.0,
        }
    }

    /// Sets the oscillator frequency in Hertz (clamped to be non-negative).
    ///
    /// The change takes effect immediately; because phase is continuous this is
    /// click-free without smoothing.
    #[inline]
    pub fn set_frequency(&mut self, hz: Sample) {
        self.frequency = hz.max(0.0);
    }

    /// Selects a new output waveform, taking effect on the next processed
    /// sample.
    #[inline]
    pub fn set_waveform(&mut self, waveform: Waveform) {
        self.waveform = waveform;
    }

    /// Sets a new target amplitude (linear), gliding toward it with `ramp` to
    /// avoid zipper noise.
    #[inline]
    pub fn set_amplitude(&mut self, linear: Sample, ramp: Ramp) {
        self.amplitude.set_target(linear, ramp);
    }

    /// Returns the current frequency in Hertz.
    #[inline]
    #[must_use]
    pub fn frequency(&self) -> Sample {
        self.frequency
    }

    /// Returns the currently selected waveform.
    #[inline]
    #[must_use]
    pub fn waveform(&self) -> Waveform {
        self.waveform
    }

    /// Returns the target amplitude the oscillator is gliding toward.
    #[inline]
    #[must_use]
    pub fn amplitude(&self) -> Sample {
        self.amplitude.target()
    }

    /// Computes the next raw (pre-amplitude) waveform sample for the current
    /// phase, then advances the phase accumulator by `dt` (the per-sample phase
    /// increment `frequency / sample_rate`).
    #[inline]
    fn next_value(&mut self, dt: Sample) -> Sample {
        let t = self.phase;
        let value = match self.waveform {
            Waveform::Sine => ops::sin(TAU * t),
            Waveform::Saw => {
                // Naive bipolar ramp minus the PolyBLEP correction at the wrap.
                let naive = 2.0 * t - 1.0;
                naive - poly_blep(t, dt)
            }
            Waveform::Square => band_limited_square(t, dt),
            Waveform::Triangle => {
                // Leaky one-pole integrator of the band-limited square. The
                // integral of a square is a triangle; the `(1 - dt)` leak keeps
                // the result DC-free and bounded (this is the standard
                // virtual-analog PolyBLEP triangle). Amplitude is naturally
                // below unity and tapers with frequency.
                let square = band_limited_square(t, dt);
                self.triangle_state = dt * square + (1.0 - dt) * self.triangle_state;
                self.triangle_state
            }
        };

        // Advance and wrap the phase into [0, 1). `dt` is always non-negative,
        // so truncating with `as u32` yields the integer part (avoids the
        // no_std-unavailable `floor`) and handles `dt >= 1` (freq > sr) too.
        self.phase += dt;
        if self.phase >= 1.0 {
            self.phase -= (self.phase as u32) as Sample;
        }

        value
    }
}

impl AudioNode for OscillatorNode {
    fn process(&mut self, ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let out = io.output(0);
        let channels = out.channels();
        if channels == 0 {
            return;
        }

        // Per-sample phase increment. `sample_rate` is validated non-zero by
        // the graph; guard defensively so `process` can never divide by zero.
        let sample_rate = ctx.sample_rate.max(1) as Sample;
        let dt = self.frequency / sample_rate;

        // Generate the mono signal into channel 0, advancing the amplitude
        // smoother once per frame.
        {
            let buf = out.channel_mut(0);
            for s in buf.iter_mut() {
                let amp = self.amplitude.next_sample();
                *s = self.next_value(dt) * amp;
            }
        }

        // Replicate the mono signal into every remaining channel.
        for ch in 1..channels {
            let (src, dst) = out.channel_pair_mut(0, ch);
            dst.copy_from_slice(src);
        }
    }

    fn reset(&mut self) {
        self.phase = 0.0;
        self.triangle_state = 0.0;
        self.amplitude = Smoothed::new(self.amplitude.target());
    }

    fn latency_frames(&self) -> u32 {
        0
    }
}

/// Two-sample `PolyBLEP` correction for a rising step discontinuity of unit
/// height located at phase `0.0` (equivalently `1.0`).
///
/// `t` is the normalized phase in `[0, 1)` and `dt` is the per-sample phase
/// increment. The function returns a small residual that, when subtracted from
/// (or added to) a naive waveform near its discontinuity, rounds the corner
/// over the two samples straddling the jump and thereby suppresses aliasing.
/// It returns `0.0` away from the discontinuity.
#[inline]
pub(crate) fn poly_blep(t: Sample, dt: Sample) -> Sample {
    if dt <= 0.0 {
        return 0.0;
    }
    if t < dt {
        // Just after the discontinuity: normalize distance into [0, 1).
        let x = t / dt;
        x + x - x * x - 1.0
    } else if t > 1.0 - dt {
        // Just before the next discontinuity: normalize into (-1, 0].
        let x = (t - 1.0) / dt;
        x * x + x + x + 1.0
    } else {
        0.0
    }
}

/// Band-limited square wave for phase `t` with per-sample increment `dt`.
///
/// This is equivalent to the difference of two half-phase-shifted sawtooths:
/// the naive `+1 / -1` square is corrected with a `PolyBLEP` at the rising edge
/// (phase `0`) and a second `PolyBLEP` at the falling edge (phase `0.5`).
#[inline]
fn band_limited_square(t: Sample, dt: Sample) -> Sample {
    // Naive bipolar square (50% duty cycle).
    let mut value = if t < 0.5 { 1.0 } else { -1.0 };
    // Correct the rising edge at phase 0.
    value += poly_blep(t, dt);
    // Correct the falling edge, which sits half a period away at phase 0.5.
    let mut t_half = t + 0.5;
    if t_half >= 1.0 {
        t_half -= 1.0;
    }
    value -= poly_blep(t_half, dt);
    value
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{AudioBuffer, ChannelLayout};
    use bevy_math::ops;

    /// Builds a render context for the given sample rate and frame count.
    fn ctx(sample_rate: u32, frames: usize) -> RenderContext {
        RenderContext {
            sample_rate,
            frames,
            playhead: 0,
        }
    }

    /// Renders one mono block from `node` and returns the output buffer.
    fn render(node: &mut OscillatorNode, sample_rate: u32, frames: usize) -> AudioBuffer {
        let out = AudioBuffer::new(ChannelLayout::Mono, frames);
        let inputs: [AudioBuffer; 0] = [];
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(sample_rate, frames), &mut io);
        let [buf] = outputs;
        buf
    }

    #[test]
    fn sine_zero_crossings_match_frequency() {
        let sample_rate = 48_000;
        let frames = 4_800; // 0.1 s
        let freq = 480.0; // 48 whole periods in 0.1 s
        let mut node = OscillatorNode::new(Waveform::Sine, freq, 1.0);
        let out = render(&mut node, sample_rate, frames);
        let ch = out.channel(0);

        // Count rising zero crossings; one per period.
        let mut crossings = 0u32;
        for w in ch.windows(2) {
            if w[0] <= 0.0 && w[1] > 0.0 {
                crossings += 1;
            }
        }
        let expected = (freq * frames as Sample / sample_rate as Sample) as u32; // 48
        let diff = crossings.abs_diff(expected);
        assert!(diff <= 1, "crossings={crossings} expected={expected}");
    }

    #[test]
    fn all_waveforms_stay_bounded() {
        let sample_rate = 48_000;
        let frames = 2_048;
        let amp = 0.5;
        // PolyBLEP overshoot and the triangle integrator can slightly exceed
        // the nominal amplitude, so allow a modest tolerance.
        let tolerance = amp * 1.3;
        for &wave in &[
            Waveform::Sine,
            Waveform::Saw,
            Waveform::Square,
            Waveform::Triangle,
        ] {
            let mut node = OscillatorNode::new(wave, 440.0, amp);
            let out = render(&mut node, sample_rate, frames);
            for &s in out.channel(0) {
                assert!(
                    s.abs() <= tolerance,
                    "waveform={wave:?} sample={s} exceeded {tolerance}"
                );
            }
        }
    }

    #[test]
    fn sine_matches_reference_series() {
        let sample_rate = 48_000;
        let frames = 256;
        let amp = 0.75;
        let freq = 1_000.0;
        let mut node = OscillatorNode::new(Waveform::Sine, freq, amp);
        let out = render(&mut node, sample_rate, frames);
        let ch = out.channel(0);

        // Reproduce the phase accumulation exactly to compare per sample.
        let dt = freq / sample_rate as Sample;
        let mut phase = 0.0f32;
        for &produced in ch {
            let expected = amp * ops::sin(TAU * phase);
            assert!(
                (produced - expected).abs() < 1e-5,
                "produced={produced} expected={expected}"
            );
            phase += dt;
            if phase >= 1.0 {
                phase -= (phase as u32) as Sample;
            }
        }
    }

    #[test]
    fn reset_makes_output_reproducible() {
        let sample_rate = 48_000;
        let frames = 128;
        let mut node = OscillatorNode::new(Waveform::Saw, 220.0, 0.9);

        let first = render(&mut node, sample_rate, frames);
        node.reset();
        let second = render(&mut node, sample_rate, frames);

        assert_eq!(first.channel(0), second.channel(0));
    }

    #[test]
    fn reset_clears_phase_and_integrator() {
        let sample_rate = 48_000;
        let frames = 64;
        let mut node = OscillatorNode::new(Waveform::Triangle, 330.0, 1.0);
        let _ = render(&mut node, sample_rate, frames);
        assert!(node.phase != 0.0 || node.triangle_state != 0.0);

        node.reset();
        assert_eq!(node.phase, 0.0);
        assert_eq!(node.triangle_state, 0.0);
    }
}
