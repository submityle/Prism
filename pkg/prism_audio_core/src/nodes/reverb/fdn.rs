//! Feedback Delay Network (`FDN`) reverberator.
//!
//! An `FDN` recirculates a bank of delay lines through an orthogonal mixing
//! matrix. Each line feeds every other line (via the matrix) on every pass, so
//! after a handful of round-trips the echo pattern becomes statistically dense
//! and indistinguishable from the diffuse tail of a real room. Because the
//! mixing matrix is *lossless* (orthonormal), the only energy loss is the
//! per-line feedback gain, which is chosen to hit a target `RT60` decay time,
//! and an optional per-line one-pole low-pass that makes the high frequencies
//! die away faster than the lows — exactly what happens as sound is absorbed by
//! air and soft surfaces.
//!
//! This design follows the classic Jot / Stautner-Puckette formulation:
//!
//! - Delay-line lengths are mutually prime so their comb resonances never line
//!   up, avoiding a metallic, ringing colour.
//! - The feedback matrix is a normalised Sylvester-Hadamard matrix, which is
//!   orthogonal (energy preserving) yet costs only additions and sign flips.
//! - A single mono recirculating network is excited by the channel average and
//!   read out through two decorrelated tap weightings to synthesise a wide
//!   stereo image.
//!
//! All storage is allocated at construction, so
//! [`FdnReverb::process`](crate::graph::AudioNode::process) performs no
//! allocation, locking, or panicking and is safe on the audio callback thread.

use alloc::vec::Vec;

use bevy_math::ops;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, flush_denormal};
use crate::param::{Ramp, Smoothed};

/// Maximum number of delay lines the network can hold. Fixed so the hot-path
/// scratch buffers can live on the stack instead of the heap.
const MAX_LINES: usize = 8;

/// Reference sample rate the prime delay-line table is expressed at. Lengths
/// are scaled from this rate to the runtime rate at construction.
const REFERENCE_RATE: Sample = 48_000.0;

/// Prime delay-line lengths (in frames at [`REFERENCE_RATE`]).
///
/// These are all primes spanning roughly 27-55 ms; being mutually prime keeps
/// their comb resonances from aligning into an audible ringing pitch.
const PRIME_LENGTHS: [usize; MAX_LINES] = [1327, 1523, 1721, 1873, 2069, 2213, 2411, 2647];

/// Largest feedback gain permitted per line. Kept just below unity so a
/// sustained tail always decays rather than building to infinity.
const MAX_FEEDBACK: Sample = 0.9999;

/// Number of delay lines in the feedback network.
///
/// Both variants are powers of two so the Sylvester-Hadamard mixing matrix is
/// well defined. More lines give a denser, smoother tail at a higher CPU cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum FdnOrder {
    /// Four delay lines: lighter and slightly grainier.
    Four,
    /// Eight delay lines: denser and smoother (the default).
    Eight,
}

impl FdnOrder {
    /// Returns the number of delay lines this order uses.
    #[inline]
    #[must_use]
    pub const fn count(self) -> usize {
        match self {
            FdnOrder::Four => 4,
            FdnOrder::Eight => 8,
        }
    }
}

/// Construction parameters for an [`FdnReverb`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct FdnReverbParams {
    /// Number of delay lines in the network.
    pub order: FdnOrder,
    /// Scales every delay-line length; larger values model a bigger space.
    /// Clamped to `[0.1, 4.0]`.
    pub room_size: Sample,
    /// Target reverberation time (the `RT60`, i.e. the seconds for the tail to
    /// fall 60 dB) in seconds. Clamped to at least `0.01`.
    pub decay_rt60_seconds: Sample,
    /// High-frequency damping in `[0.0, 0.999]`; higher values roll the tail's
    /// treble off faster, modelling air and soft-surface absorption.
    pub damping: Sample,
    /// Wet (reverberated) mix gain.
    pub wet: Sample,
    /// Dry (unprocessed input) mix gain.
    pub dry: Sample,
}

impl Default for FdnReverbParams {
    fn default() -> Self {
        Self {
            order: FdnOrder::Eight,
            room_size: 1.0,
            decay_rt60_seconds: 2.2,
            damping: 0.3,
            wet: 0.4,
            dry: 1.0,
        }
    }
}

/// A Feedback Delay Network reverb (input port 0 -> output port 0).
///
/// The mono recirculating core is excited by the average of the input channels
/// and read back through two decorrelated tap weightings, producing a wide,
/// diffuse stereo tail. The output is `dry * input + wet * network`. Damping and
/// the wet/dry gains are [`Smoothed`] so automation stays click-free; the decay
/// time is recomputed in place (no allocation) when changed.
#[derive(Debug, Clone)]
pub struct FdnReverb {
    /// Runtime sample rate in Hz, used to convert the `RT60` into per-line
    /// feedback gains.
    sample_rate: u32,
    /// Number of active delay lines (`order.count()`).
    lines: usize,
    /// One recirculating delay ring per active line; each has a prime length.
    rings: Vec<Vec<Sample>>,
    /// Length in frames of each ring (mirrors `rings[i].len()`).
    lengths: [usize; MAX_LINES],
    /// Per-line read/write cursor into the corresponding ring.
    pos: [usize; MAX_LINES],
    /// Per-line one-pole damping filter state (the previous low-passed sample).
    damp_state: [Sample; MAX_LINES],
    /// Per-line feedback gain derived from the target `RT60`.
    feedback_gain: [Sample; MAX_LINES],
    /// Normalised Sylvester-Hadamard mixing matrix (orthonormal).
    hadamard: [[Sample; MAX_LINES]; MAX_LINES],
    /// Per-line left-output tap weight.
    out_l: [Sample; MAX_LINES],
    /// Per-line right-output tap weight.
    out_r: [Sample; MAX_LINES],
    /// Cached target `RT60` (seconds) so `set_decay` can recompute gains.
    rt60_seconds: Sample,
    /// Cached room-size multiplier applied to the prime lengths.
    room_size: Sample,
    /// Smoothed damping coefficient in `[0.0, 0.999]`.
    damping: Smoothed,
    /// Smoothed wet (reverberated) mix gain.
    wet: Smoothed,
    /// Smoothed dry (unprocessed input) mix gain.
    dry: Smoothed,
}

/// Computes the per-line feedback gain that yields the requested `RT60`.
///
/// A signal circulating a line of `len` frames loops every `len / sr` seconds;
/// to lose 60 dB (a factor of `1000`) over `rt60` seconds the per-loop gain is
/// `10^(-3 * loop_time / rt60)`.
#[inline]
fn feedback_for_rt60(len: usize, sample_rate: u32, rt60: Sample) -> Sample {
    let loop_time = len as Sample / sample_rate as Sample;
    let rt = rt60.max(0.01);
    let g = ops::exp(-3.0 * loop_time / rt * core::f32::consts::LN_10);
    g.clamp(0.0, MAX_FEEDBACK)
}

impl FdnReverb {
    /// Builds a reverb for `channels` channels running at `sample_rate` Hz.
    ///
    /// All delay rings, the mixing matrix, and the tap weights are allocated
    /// and computed here; the initial parameter values start settled (no
    /// glide).
    #[must_use]
    pub fn new(sample_rate: u32, channels: usize, params: FdnReverbParams) -> Self {
        let _ = channels;
        let sr = sample_rate.max(1);
        let lines = params.order.count();
        let room_size = params.room_size.clamp(0.1, 4.0);
        let rt60 = params.decay_rt60_seconds.max(0.01);
        let scale = room_size * (sr as Sample) / REFERENCE_RATE;

        let mut lengths = [0usize; MAX_LINES];
        let mut feedback_gain = [0.0; MAX_LINES];
        for (i, len) in lengths.iter_mut().enumerate().take(lines) {
            let scaled = ops::round(PRIME_LENGTHS[i] as Sample * scale) as usize;
            *len = scaled.max(2);
        }
        for (i, g) in feedback_gain.iter_mut().enumerate().take(lines) {
            *g = feedback_for_rt60(lengths[i], sr, rt60);
        }

        let mut rings = Vec::with_capacity(lines);
        for len in lengths.iter().take(lines) {
            let mut ring = Vec::with_capacity(*len);
            ring.resize(*len, 0.0);
            rings.push(ring);
        }

        // Normalised Sylvester-Hadamard matrix: h[i][j] = (-1)^popcount(i & j)
        // divided by sqrt(N) so the transform is orthonormal (energy-preserving).
        let norm = 1.0 / ops::sqrt(lines as Sample);
        let mut hadamard = [[0.0; MAX_LINES]; MAX_LINES];
        for (i, row) in hadamard.iter_mut().enumerate().take(lines) {
            for (j, cell) in row.iter_mut().enumerate().take(lines) {
                let even = (i & j).count_ones() % 2 == 0;
                *cell = if even { norm } else { -norm };
            }
        }

        // Decorrelated output taps: the left bus sums every line in phase while
        // the right bus alternates sign, so the two channels are diffuse and
        // wide rather than a mono duplicate.
        let mut out_l = [0.0; MAX_LINES];
        let mut out_r = [0.0; MAX_LINES];
        for (i, (l, r)) in out_l.iter_mut().zip(out_r.iter_mut()).enumerate().take(lines) {
            *l = 1.0;
            *r = if i % 2 == 0 { 1.0 } else { -1.0 };
        }

        Self {
            sample_rate: sr,
            lines,
            rings,
            lengths,
            pos: [0; MAX_LINES],
            damp_state: [0.0; MAX_LINES],
            feedback_gain,
            hadamard,
            out_l,
            out_r,
            rt60_seconds: rt60,
            room_size,
            damping: Smoothed::new(params.damping.clamp(0.0, 0.999)),
            wet: Smoothed::new(params.wet),
            dry: Smoothed::new(params.dry),
        }
    }

    /// Returns the number of active delay lines.
    #[inline]
    #[must_use]
    pub fn lines(&self) -> usize {
        self.lines
    }

    /// Returns the cached room-size multiplier.
    #[inline]
    #[must_use]
    pub fn room_size(&self) -> Sample {
        self.room_size
    }

    /// Returns the target `RT60` decay time in seconds.
    #[inline]
    #[must_use]
    pub fn decay_rt60_seconds(&self) -> Sample {
        self.rt60_seconds
    }

    /// Sets a new target `RT60` (seconds), recomputing the per-line feedback
    /// gains in place. Allocation-free, so it is safe to call between blocks.
    #[inline]
    pub fn set_decay(&mut self, rt60_seconds: Sample) {
        let rt = rt60_seconds.max(0.01);
        self.rt60_seconds = rt;
        let sr = self.sample_rate;
        for (i, g) in self.feedback_gain.iter_mut().enumerate().take(self.lines) {
            *g = feedback_for_rt60(self.lengths[i], sr, rt);
        }
    }

    /// Sets the high-frequency damping coefficient, clamped to `[0.0, 0.999]`,
    /// gliding with `ramp`.
    #[inline]
    pub fn set_damping(&mut self, damping: Sample, ramp: Ramp) {
        self.damping.set_target(damping.clamp(0.0, 0.999), ramp);
    }

    /// Sets the wet (reverberated) mix gain, gliding with `ramp`.
    #[inline]
    pub fn set_wet(&mut self, wet: Sample, ramp: Ramp) {
        self.wet.set_target(wet, ramp);
    }

    /// Sets the dry (unprocessed input) mix gain, gliding with `ramp`.
    #[inline]
    pub fn set_dry(&mut self, dry: Sample, ramp: Ramp) {
        self.dry.set_target(dry, ramp);
    }
}

impl AudioNode for FdnReverb {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let channels = output.channels().min(input.channels());
        let frames = output.active_frames().min(input.active_frames());
        let lines = self.lines;
        let out_norm = 1.0 / lines as Sample;

        for f in 0..frames {
            // The recirculating network runs once per frame (it is mono), so
            // the smoothers advance once per frame here.
            let damp = self.damping.next_sample().clamp(0.0, 0.999);
            let wet = self.wet.next_sample();
            let dry = self.dry.next_sample();

            // Gather each line's delayed output and low-pass it, forming the
            // damped feedback the mixing matrix will redistribute.
            let mut delayed = [0.0; MAX_LINES];
            let mut feedback = [0.0; MAX_LINES];
            for i in 0..lines {
                let s = self.rings[i][self.pos[i]];
                delayed[i] = s;
                let lp = (1.0 - damp) * s + damp * self.damp_state[i];
                self.damp_state[i] = flush_denormal(lp);
                feedback[i] = lp * self.feedback_gain[i];
            }

            // Excite the network with the average of the input channels so a
            // mono or stereo source drives a single coherent tail.
            let mut excite = 0.0;
            if channels > 0 {
                for ch in 0..channels {
                    excite += input.channel(ch)[f];
                }
                excite *= 1.0 / channels as Sample;
            }

            // Mix through the orthonormal matrix, inject the excitation, and
            // accumulate the two decorrelated output taps from the pre-mix
            // delayed samples.
            let mut wl = 0.0;
            let mut wr = 0.0;
            for (i, &d) in delayed.iter().enumerate().take(lines) {
                let mut mixed = 0.0;
                for (j, &fb) in feedback.iter().enumerate().take(lines) {
                    mixed += self.hadamard[i][j] * fb;
                }
                let v = flush_denormal(excite + mixed);
                self.rings[i][self.pos[i]] = v;
                self.pos[i] = if self.pos[i] + 1 == self.lengths[i] {
                    0
                } else {
                    self.pos[i] + 1
                };
                wl += d * self.out_l[i];
                wr += d * self.out_r[i];
            }
            wl *= out_norm;
            wr *= out_norm;

            // Blend the dry input with the appropriate wet tap per channel.
            for ch in 0..channels {
                let x = input.channel(ch)[f];
                let wv = if ch % 2 == 1 { wr } else { wl };
                output.channel_mut(ch)[f] = dry * x + wet * wv;
            }
        }
    }

    fn reset(&mut self) {
        for ring in &mut self.rings {
            for s in ring.iter_mut() {
                *s = 0.0;
            }
        }
        self.pos = [0; MAX_LINES];
        self.damp_state = [0.0; MAX_LINES];
        self.damping = Smoothed::new(self.damping.target());
        self.wet = Smoothed::new(self.wet.target());
        self.dry = Smoothed::new(self.dry.target());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{AudioBuffer, ChannelLayout};

    fn ctx(frames: usize) -> RenderContext {
        RenderContext {
            sample_rate: 48_000,
            frames,
            playhead: 0,
        }
    }

    fn stereo(frames: usize) -> AudioBuffer {
        AudioBuffer::new(ChannelLayout::Stereo, frames)
    }

    #[test]
    fn dry_passthrough_when_wet_zero() {
        let params = FdnReverbParams {
            wet: 0.0,
            dry: 1.0,
            ..FdnReverbParams::default()
        };
        let mut node = FdnReverb::new(48_000, 2, params);
        let mut input = stereo(32);
        for ch in 0..2 {
            for (i, s) in input.channel_mut(ch).iter_mut().enumerate() {
                *s = (i as Sample) * 0.01 - 0.15 + ch as Sample * 0.02;
            }
        }
        let inputs = [input.clone()];
        let mut outputs = [stereo(32)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(32), &mut io);
        assert_eq!(outputs[0].channel(0), inputs[0].channel(0));
        assert_eq!(outputs[0].channel(1), inputs[0].channel(1));
    }

    #[test]
    fn tail_has_energy_and_is_bounded() {
        let params = FdnReverbParams {
            wet: 1.0,
            dry: 0.0,
            ..FdnReverbParams::default()
        };
        let mut node = FdnReverb::new(48_000, 2, params);
        let n = 8_192;
        let mut input = stereo(n);
        input.channel_mut(0)[0] = 1.0;
        input.channel_mut(1)[0] = 1.0;
        let inputs = [input];
        let mut outputs = [stereo(n)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(n), &mut io);

        let mut energy = 0.0f32;
        for ch in 0..2 {
            for &s in outputs[0].channel(ch) {
                assert!(s.is_finite(), "non-finite tail sample: {s}");
                assert!(s.abs() < 8.0, "tail sample exploded: {s}");
                energy += s * s;
            }
        }
        assert!(energy > 1e-3, "reverb tail carried no energy: {energy}");
    }

    #[test]
    fn decaying_tail_shrinks_over_time() {
        // A shorter RT60 must leave less late energy than a longer one.
        fn late_energy(rt60: Sample) -> Sample {
            let params = FdnReverbParams {
                wet: 1.0,
                dry: 0.0,
                decay_rt60_seconds: rt60,
                ..FdnReverbParams::default()
            };
            let mut node = FdnReverb::new(48_000, 1, params);
            let n = 24_000;
            let mut input = AudioBuffer::new(ChannelLayout::Mono, n);
            input.channel_mut(0)[0] = 1.0;
            let inputs = [input];
            let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, n)];
            let mut io = ProcessIo::new(&inputs, &mut outputs);
            node.process(&ctx(n), &mut io);
            outputs[0].channel(0)[n / 2..].iter().map(|s| s * s).sum()
        }
        let short = late_energy(0.4);
        let long = late_energy(3.5);
        assert!(long > short, "longer RT60 should retain more late energy: short={short} long={long}");
    }

    #[test]
    fn reset_clears_tail() {
        let params = FdnReverbParams {
            wet: 1.0,
            dry: 0.0,
            ..FdnReverbParams::default()
        };
        let mut node = FdnReverb::new(48_000, 1, params);
        let n = 512;
        let mut input = AudioBuffer::new(ChannelLayout::Mono, n);
        input.channel_mut(0)[0] = 1.0;
        let inputs = [input];
        let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, n)];
        {
            let mut io = ProcessIo::new(&inputs, &mut outputs);
            node.process(&ctx(n), &mut io);
        }
        node.reset();
        let silence = [AudioBuffer::new(ChannelLayout::Mono, n)];
        let mut out2 = [AudioBuffer::new(ChannelLayout::Mono, n)];
        {
            let mut io = ProcessIo::new(&silence, &mut out2);
            node.process(&ctx(n), &mut io);
        }
        for &s in out2[0].channel(0) {
            assert!(s.abs() < 1e-12, "tail not cleared after reset: {s}");
        }
    }

    #[test]
    fn latency_is_zero() {
        let node = FdnReverb::new(48_000, 2, FdnReverbParams::default());
        assert_eq!(node.latency_frames(), 0);
    }
}
