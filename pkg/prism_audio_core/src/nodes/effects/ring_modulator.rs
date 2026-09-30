//! Ring modulator: multiplies the input by a bipolar carrier oscillator.
//!
//! Ring modulation is the archetypal analog "clangorous"/metallic effect. The
//! input signal is multiplied sample-by-sample by a carrier waveform whose mean
//! is zero (a true *ring* modulator suppresses the carrier itself, unlike a
//! plain amplitude modulator that adds a DC offset to the carrier). For an
//! input partial at `f_in` and a sine carrier at `f_c` the product creates a
//! pair of sidebands at `f_in +/- f_c` while the original partial disappears,
//! which is why ring modulation produces inharmonic, bell-like, or robotic
//! timbres rather than a simple tremolo.
//!
//! Unlike [`TremoloNode`](crate::nodes::effects::TremoloNode) - which is a
//! *unipolar* amplitude modulation that keeps the carrier's DC term so the
//! signal only dips in level - this node uses the raw bipolar carrier in
//! `[-1, 1]`, so it inverts the signal's polarity twice per carrier cycle and
//! generates new sum-and-difference partials.
//!
//! A single carrier is shared across every channel so the modulation stays
//! phase-coherent across the stereo (or surround) image. A wet/dry `mix`
//! blends the modulated signal against the untouched input.
//!
//! # Real-time contract
//!
//! The carrier oscillator is allocated once in [`RingModulatorNode::new`].
//! [`process`](crate::graph::AudioNode::process) performs no allocation, takes
//! no locks, and cannot panic: mismatched channel counts and zero-length blocks
//! degrade gracefully.
//!
//! # Provenance
//!
//! Ring modulation (multiplication by a suppressed-carrier bipolar oscillator,
//! yielding sum-and-difference sidebands) is a classic analog technique
//! described in the standard literature (e.g. Zoelzer, "DAFX: Digital Audio
//! Effects"). This module reuses only this crate's own [`Lfo`] carrier,
//! [`Sample`] type, and denormal-flushing primitive. It contains **no Unreal
//! Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or Google Resonance Audio
//! source or derived code**; it is implemented purely from that publicly
//! documented theory.

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, flush_denormal};
use crate::modulation::{Lfo, LfoWaveform};

/// Configuration for a [`RingModulatorNode`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct RingModulatorParams {
    /// Carrier frequency in hertz (clamped to be non-negative). Sub-audio-rate
    /// values approach a tremolo; audio-rate values create inharmonic
    /// sidebands.
    pub carrier_hz: Sample,
    /// Carrier waveform. A sine gives the cleanest two-sideband spectrum; other
    /// shapes add extra carrier harmonics and denser sidebands.
    pub waveform: LfoWaveform,
    /// Wet/dry blend in `[0, 1]`: `0` is the untouched input, `1` is fully
    /// modulated.
    pub mix: Sample,
}

impl Default for RingModulatorParams {
    fn default() -> Self {
        Self {
            carrier_hz: 440.0,
            waveform: LfoWaveform::Sine,
            mix: 1.0,
        }
    }
}

/// A ring-modulator node: `output = input * ((1 - mix) + mix * carrier)`.
#[derive(Debug)]
pub struct RingModulatorNode {
    /// The shared bipolar carrier oscillator, advanced once per frame.
    carrier: Lfo,
    /// Wet/dry blend (clamped to `[0, 1]`).
    mix: Sample,
}

impl RingModulatorNode {
    /// Builds a ring modulator for a stream sampled at `sample_rate`.
    #[must_use]
    pub fn new(sample_rate: u32, params: RingModulatorParams) -> Self {
        Self {
            carrier: Lfo::new(sample_rate, params.carrier_hz.max(0.0), params.waveform),
            mix: params.mix.clamp(0.0, 1.0),
        }
    }

    /// Sets the carrier frequency in hertz (clamped to be non-negative).
    pub fn set_carrier_hz(&mut self, sample_rate: u32, carrier_hz: Sample) {
        self.carrier.set_frequency(sample_rate, carrier_hz.max(0.0));
    }

    /// Sets the carrier waveform.
    pub fn set_waveform(&mut self, waveform: LfoWaveform) {
        self.carrier.set_waveform(waveform);
    }

    /// Sets the wet/dry blend (clamped to `[0, 1]`).
    pub fn set_mix(&mut self, mix: Sample) {
        self.mix = mix.clamp(0.0, 1.0);
    }
}

impl AudioNode for RingModulatorNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let channels = output.channels().min(input.channels());
        let frames = output.active_frames();
        if frames == 0 || channels == 0 {
            return;
        }

        let mix = self.mix;
        let dry = 1.0 - mix;

        for f in 0..frames {
            // One carrier value per frame, shared across channels for a
            // phase-coherent stereo/surround image.
            let carrier = self.carrier.next_sample();
            let gain = dry + mix * carrier;
            for ch in 0..channels {
                output.channel_mut(ch)[f] = flush_denormal(input.channel(ch)[f] * gain);
            }
        }
    }

    fn reset(&mut self) {
        self.carrier.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{AudioBuffer, ChannelLayout};

    const SR: u32 = 48_000;

    fn ctx(frames: usize) -> RenderContext {
        RenderContext {
            sample_rate: SR,
            frames,
            playhead: 0,
        }
    }

    fn dc(layout: ChannelLayout, frames: usize, values: &[Sample]) -> AudioBuffer {
        let mut buf = AudioBuffer::new(layout, frames);
        buf.set_active_frames(frames);
        for ch in 0..layout.channel_count() {
            let v = values[ch % values.len()];
            for s in buf.channel_mut(ch) {
                *s = v;
            }
        }
        buf
    }

    fn run(node: &mut RingModulatorNode, input: &AudioBuffer) -> AudioBuffer {
        let frames = input.active_frames();
        let mut out = AudioBuffer::new(input.layout(), input.capacity_frames());
        out.set_active_frames(frames);
        let inputs = [input.clone()];
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(frames), &mut io);
        let [o] = outputs;
        o
    }

    #[test]
    fn full_wet_output_tracks_the_carrier() {
        // With a unit DC input and full wet, the output equals the carrier.
        let input = dc(ChannelLayout::Mono, 256, &[1.0]);
        let params = RingModulatorParams {
            carrier_hz: 1000.0,
            waveform: LfoWaveform::Sine,
            mix: 1.0,
        };
        let mut node = RingModulatorNode::new(SR, params);
        let out = run(&mut node, &input);
        // A sine carrier starts at phase 0 (value 0) and stays bounded.
        assert!(out.channel(0)[0].abs() < 1e-6, "first {}", out.channel(0)[0]);
        for &s in out.channel(0) {
            assert!((-1.0..=1.0).contains(&s), "out of range {s}");
        }
        // Somewhere in the block the carrier must have swung meaningfully.
        let peak = out.channel(0).iter().fold(0.0_f32, |m, &s| m.max(s.abs()));
        assert!(peak > 0.9, "carrier never swung: peak {peak}");
    }

    #[test]
    fn mix_zero_is_bypass() {
        let input = dc(ChannelLayout::Stereo, 128, &[0.7, -0.3]);
        let params = RingModulatorParams {
            carrier_hz: 500.0,
            waveform: LfoWaveform::Sine,
            mix: 0.0,
        };
        let mut node = RingModulatorNode::new(SR, params);
        let out = run(&mut node, &input);
        for ch in 0..2 {
            for (o, i) in out.channel(ch).iter().zip(input.channel(ch)) {
                assert!((o - i).abs() < 1e-6, "not bypassed: {o} vs {i}");
            }
        }
    }

    #[test]
    fn square_carrier_inverts_polarity() {
        // A square carrier is +1 for the first half cycle, -1 for the second.
        // At 8 Hz carrier and 32 Hz rate there are 4 samples per cycle.
        let input = dc(ChannelLayout::Mono, 4, &[1.0]);
        let params = RingModulatorParams {
            carrier_hz: 8.0,
            waveform: LfoWaveform::Square,
            mix: 1.0,
        };
        let mut node = RingModulatorNode::new(32, params);
        let out = run(&mut node, &input);
        // phases 0.0, 0.25 -> +1 ; 0.5, 0.75 -> -1.
        assert!((out.channel(0)[0] - 1.0).abs() < 1e-6);
        assert!((out.channel(0)[1] - 1.0).abs() < 1e-6);
        assert!((out.channel(0)[2] + 1.0).abs() < 1e-6);
        assert!((out.channel(0)[3] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn magnitude_never_exceeds_input() {
        // |carrier| <= 1 so full-wet ring modulation cannot boost level.
        let input = dc(ChannelLayout::Mono, 512, &[0.8]);
        let params = RingModulatorParams {
            carrier_hz: 733.0,
            waveform: LfoWaveform::Sine,
            mix: 1.0,
        };
        let mut node = RingModulatorNode::new(SR, params);
        let out = run(&mut node, &input);
        for &s in out.channel(0) {
            assert!(s.abs() <= 0.8 + 1e-6, "boosted: {s}");
        }
    }

    #[test]
    fn carrier_is_shared_across_channels() {
        // Identical per-channel input must yield identical per-channel output.
        let input = dc(ChannelLayout::Stereo, 200, &[0.5, 0.5]);
        let params = RingModulatorParams::default();
        let mut node = RingModulatorNode::new(SR, params);
        let out = run(&mut node, &input);
        for (l, r) in out.channel(0).iter().zip(out.channel(1)) {
            assert!((l - r).abs() < 1e-6, "channels diverged: {l} vs {r}");
        }
    }

    #[test]
    fn dry_wet_blend_is_a_linear_mix() {
        let input = dc(ChannelLayout::Mono, 64, &[1.0]);
        let mut wet = RingModulatorNode::new(
            SR,
            RingModulatorParams {
                carrier_hz: 300.0,
                waveform: LfoWaveform::Sine,
                mix: 1.0,
            },
        );
        let mut half = RingModulatorNode::new(
            SR,
            RingModulatorParams {
                carrier_hz: 300.0,
                waveform: LfoWaveform::Sine,
                mix: 0.5,
            },
        );
        let out_wet = run(&mut wet, &input);
        let out_half = run(&mut half, &input);
        // half = input*0.5 + wet*0.5 (input is unit DC).
        for (h, w) in out_half.channel(0).iter().zip(out_wet.channel(0)) {
            let expected = 0.5 * 1.0 + 0.5 * w;
            assert!((h - expected).abs() < 1e-5, "{h} vs {expected}");
        }
    }

    #[test]
    fn parameters_are_clamped() {
        let input = dc(ChannelLayout::Mono, 32, &[1.0]);
        let mut node = RingModulatorNode::new(
            SR,
            RingModulatorParams {
                carrier_hz: -100.0,
                waveform: LfoWaveform::Sine,
                mix: 5.0,
            },
        );
        // Negative carrier clamps to 0 Hz (a stationary sine at value 0), mix
        // clamps to 1 -> full-wet output of a zero carrier is silence.
        let out = run(&mut node, &input);
        for &s in out.channel(0) {
            assert!(s.is_finite(), "non-finite {s}");
            assert!(s.abs() < 1e-6, "expected silence, got {s}");
        }
        node.set_mix(-3.0);
        node.set_carrier_hz(SR, -5.0);
        let out2 = run(&mut node, &input);
        // mix clamps to 0 -> bypass.
        for &s in out2.channel(0) {
            assert!((s - 1.0).abs() < 1e-6, "expected bypass, got {s}");
        }
    }

    #[test]
    fn reset_restarts_the_carrier() {
        let input = dc(ChannelLayout::Mono, 100, &[1.0]);
        let params = RingModulatorParams {
            carrier_hz: 640.0,
            waveform: LfoWaveform::Sine,
            mix: 1.0,
        };
        let mut node = RingModulatorNode::new(SR, params);
        let first = run(&mut node, &input);
        node.reset();
        let second = run(&mut node, &input);
        for (a, b) in first.channel(0).iter().zip(second.channel(0)) {
            assert!((a - b).abs() < 1e-6, "reset not reproducible: {a} vs {b}");
        }
    }

    #[test]
    fn zero_frames_do_not_panic() {
        let mut node = RingModulatorNode::new(SR, RingModulatorParams::default());
        let input = dc(ChannelLayout::Mono, 16, &[1.0]);
        let mut out = AudioBuffer::new(ChannelLayout::Mono, 16);
        out.set_active_frames(0);
        let inputs = [input];
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(0), &mut io);
        let [o] = outputs;
        assert_eq!(o.active_frames(), 0);
    }

    #[test]
    fn output_is_finite_for_all_waveforms() {
        for wf in [
            LfoWaveform::Sine,
            LfoWaveform::Triangle,
            LfoWaveform::Sawtooth,
            LfoWaveform::Square,
        ] {
            let input = dc(ChannelLayout::Mono, 128, &[0.9]);
            let mut node = RingModulatorNode::new(
                SR,
                RingModulatorParams {
                    carrier_hz: 1234.0,
                    waveform: wf,
                    mix: 0.75,
                },
            );
            let out = run(&mut node, &input);
            for &s in out.channel(0) {
                assert!(s.is_finite(), "non-finite for {wf:?}: {s}");
            }
        }
    }
}
