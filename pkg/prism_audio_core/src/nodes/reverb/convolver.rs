//! Time-domain convolution reverb.
//!
//! A convolution reverb reproduces the *exact* acoustic signature of a real (or
//! synthetic) space by convolving the dry input with a measured impulse
//! response: the recording of how that space answers a single click. Whereas an
//! algorithmic reverb *models* a room, a convolver *replays* one, capturing the
//! precise early-reflection pattern and modal tail of a cathedral, plate, or
//! speaker cabinet.
//!
//! # Why time-domain?
//!
//! Production convolvers usually run in the frequency domain (partitioned FFT)
//! for long impulse responses. This kernel is `no_std` and pulls in no external
//! FFT, so it convolves directly in the time domain: each output sample is the
//! dot product of the impulse response with the most recent inputs held in a
//! per-channel ring buffer. That is `O(M)` per sample for an `M`-tap response,
//! which is ideal for the short-to-moderate impulse responses used for early
//! reflections, cabinet emulation, and small rooms, and keeps the hot path
//! allocation- and branch-light.
//!
//! Because the ring buffer and coefficient table are sized at construction,
//! [`Convolver::process`](crate::graph::AudioNode::process) never allocates,
//! locks, or panics.

use alloc::vec::Vec;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, flush_denormal};
use crate::param::{Ramp, Smoothed};

/// Per-channel convolution state: the impulse response and a matching ring of
/// recent inputs.
#[derive(Debug, Clone)]
struct ChannelConv {
    /// Impulse-response coefficients (`h[0]` is the immediate, zero-delay tap).
    ir: Vec<Sample>,
    /// Ring buffer of the most recent inputs; length equals `ir.len()`.
    hist: Vec<Sample>,
    /// Write cursor into `hist` (the slot holding the current input sample).
    pos: usize,
}

impl ChannelConv {
    /// Builds channel state from an impulse response, cloning it into an owned
    /// buffer. An empty response is replaced by a single silent tap so the
    /// modular ring arithmetic never divides by zero.
    fn new(ir: &[Sample]) -> Self {
        let len = ir.len().max(1);
        let mut coeffs = Vec::with_capacity(len);
        if ir.is_empty() {
            coeffs.push(0.0);
        } else {
            coeffs.extend_from_slice(ir);
        }
        let mut hist = Vec::with_capacity(len);
        hist.resize(len, 0.0);
        Self {
            ir: coeffs,
            hist,
            pos: 0,
        }
    }

    /// Convolves one input sample, returning the corresponding wet output.
    ///
    /// Computes `y[n] = sum_k h[k] * x[n - k]` by walking the impulse response
    /// against the ring of past inputs, then advances the write cursor.
    #[inline]
    fn process_sample(&mut self, x: Sample) -> Sample {
        let len = self.hist.len();
        self.hist[self.pos] = x;
        let mut acc = 0.0;
        for (k, &coeff) in self.ir.iter().enumerate() {
            let idx = (self.pos + len - k) % len;
            acc += coeff * self.hist[idx];
        }
        self.pos = if self.pos + 1 == len { 0 } else { self.pos + 1 };
        flush_denormal(acc)
    }

    /// Clears the input history back to silence.
    fn reset(&mut self) {
        for s in &mut self.hist {
            *s = 0.0;
        }
        self.pos = 0;
    }
}

/// A per-channel time-domain convolution reverb (input port 0 -> output
/// port 0).
///
/// The output is `dry * input + wet * (input * impulse_response)`, computed
/// independently per channel from that channel's own impulse response. The
/// wet/dry gains are [`Smoothed`] so automation stays click-free. The node has
/// zero reported latency: the first impulse-response tap is the immediate,
/// in-phase sample.
#[derive(Debug, Clone)]
pub struct Convolver {
    /// One convolution engine per channel.
    channels: Vec<ChannelConv>,
    /// Smoothed wet (convolved) mix gain.
    wet: Smoothed,
    /// Smoothed dry (unprocessed input) mix gain.
    dry: Smoothed,
}

impl Convolver {
    /// Builds a convolver from one impulse response per channel.
    ///
    /// `irs[ch]` is the response applied to channel `ch`; each is copied into
    /// pre-allocated storage. At least one channel is always created (an empty
    /// `irs` yields a single silent-response channel). The initial `wet`/`dry`
    /// gains start settled.
    #[must_use]
    pub fn new(irs: &[&[Sample]], wet: Sample, dry: Sample) -> Self {
        let count = irs.len().max(1);
        let mut channels = Vec::with_capacity(count);
        if irs.is_empty() {
            channels.push(ChannelConv::new(&[]));
        } else {
            for ir in irs {
                channels.push(ChannelConv::new(ir));
            }
        }
        Self {
            channels,
            wet: Smoothed::new(wet),
            dry: Smoothed::new(dry),
        }
    }

    /// Returns the number of channels (impulse responses) this convolver holds.
    #[inline]
    #[must_use]
    pub fn channels(&self) -> usize {
        self.channels.len()
    }

    /// Returns the impulse-response length (in taps) for `channel`.
    ///
    /// # Panics
    ///
    /// Panics if `channel >= channels()`.
    #[inline]
    #[must_use]
    pub fn ir_len(&self, channel: usize) -> usize {
        self.channels[channel].ir.len()
    }

    /// Sets the wet (convolved) mix gain, gliding with `ramp`.
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

impl AudioNode for Convolver {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let channels = output
            .channels()
            .min(input.channels())
            .min(self.channels.len());
        // Snapshot the smoother state so every channel replays the identical
        // per-frame gain sequence, then commit the advanced state once.
        let wet_start = self.wet;
        let dry_start = self.dry;
        let mut committed_wet = wet_start;
        let mut committed_dry = dry_start;

        for (ch, conv) in self.channels.iter_mut().enumerate().take(channels) {
            let mut wet = wet_start;
            let mut dry = dry_start;
            let src = input.channel(ch);
            let dst = output.channel_mut(ch);
            for (d, &x) in dst.iter_mut().zip(src.iter()) {
                let w = wet.next_sample();
                let dgain = dry.next_sample();
                let wet_sample = conv.process_sample(x);
                *d = dgain * x + w * wet_sample;
            }
            committed_wet = wet;
            committed_dry = dry;
        }

        self.wet = committed_wet;
        self.dry = committed_dry;
    }

    fn reset(&mut self) {
        for conv in &mut self.channels {
            conv.reset();
        }
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

    fn mono(frames: usize) -> AudioBuffer {
        AudioBuffer::new(ChannelLayout::Mono, frames)
    }

    #[test]
    fn impulse_response_reproduces_ir() {
        // An impulse in, wet-only, must reproduce the impulse response exactly.
        let ir = [0.5, -0.25, 0.125, 0.0625, -0.03125];
        let irs: [&[Sample]; 1] = [&ir];
        let mut node = Convolver::new(&irs, 1.0, 0.0);
        let n = 8;
        let mut input = mono(n);
        input.channel_mut(0)[0] = 1.0;
        let inputs = [input];
        let mut outputs = [mono(n)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(n), &mut io);
        let out = outputs[0].channel(0);
        for (i, &c) in ir.iter().enumerate() {
            assert!((out[i] - c).abs() < 1e-6, "tap {i}: got {} want {c}", out[i]);
        }
        for &s in &out[ir.len()..] {
            assert!(s.abs() < 1e-6, "energy past the impulse response: {s}");
        }
    }

    #[test]
    fn convolution_matches_manual_reference() {
        // Verify a full convolution against a hand-rolled reference sum.
        let ir = [0.2, 0.5, -0.1];
        let irs: [&[Sample]; 1] = [&ir];
        let mut node = Convolver::new(&irs, 1.0, 0.0);
        let x = [1.0, 0.5, -0.3, 0.7, 0.0, -0.2];
        let n = x.len();
        let mut input = mono(n);
        input.channel_mut(0).copy_from_slice(&x);
        let inputs = [input];
        let mut outputs = [mono(n)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(n), &mut io);
        let out = outputs[0].channel(0);
        for i in 0..n {
            let mut want = 0.0;
            for (k, &c) in ir.iter().enumerate() {
                if i >= k {
                    want += c * x[i - k];
                }
            }
            assert!((out[i] - want).abs() < 1e-6, "sample {i}: got {} want {want}", out[i]);
        }
    }

    #[test]
    fn dry_passthrough_when_wet_zero() {
        let ir = [0.9, 0.4, 0.2];
        let irs: [&[Sample]; 1] = [&ir];
        let mut node = Convolver::new(&irs, 0.0, 1.0);
        let n = 6;
        let mut input = mono(n);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            *s = i as Sample + 1.0;
        }
        let inputs = [input.clone()];
        let mut outputs = [mono(n)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(n), &mut io);
        assert_eq!(outputs[0].channel(0), inputs[0].channel(0));
    }

    #[test]
    fn tail_is_finite_and_bounded() {
        let ir: Vec<Sample> = (0..64).map(|k| 0.5 - (k as Sample) * 0.005).collect();
        let irs: [&[Sample]; 1] = [&ir];
        let mut node = Convolver::new(&irs, 1.0, 0.0);
        let n = 128;
        let mut input = mono(n);
        input.channel_mut(0)[0] = 1.0;
        let inputs = [input];
        let mut outputs = [mono(n)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(n), &mut io);
        let mut energy = 0.0f32;
        for &s in outputs[0].channel(0) {
            assert!(s.is_finite(), "non-finite output: {s}");
            assert!(s.abs() < 4.0, "output exploded: {s}");
            energy += s * s;
        }
        assert!(energy > 1e-3, "convolution produced no energy: {energy}");
    }

    #[test]
    fn reset_clears_tail() {
        let ir = [0.8, 0.6, 0.4, 0.2];
        let irs: [&[Sample]; 1] = [&ir];
        let mut node = Convolver::new(&irs, 1.0, 0.0);
        let n = 16;
        let mut input = mono(n);
        input.channel_mut(0)[0] = 1.0;
        let inputs = [input];
        let mut outputs = [mono(n)];
        {
            let mut io = ProcessIo::new(&inputs, &mut outputs);
            node.process(&ctx(n), &mut io);
        }
        node.reset();
        let silence = [mono(n)];
        let mut out2 = [mono(n)];
        {
            let mut io = ProcessIo::new(&silence, &mut out2);
            node.process(&ctx(n), &mut io);
        }
        for &s in out2[0].channel(0) {
            assert!(s.abs() < 1e-12, "tail not cleared after reset: {s}");
        }
    }

    #[test]
    fn stereo_channels_use_independent_responses() {
        let ir_l = [1.0, 0.0, 0.0];
        let ir_r = [0.0, 0.0, 1.0];
        let irs: [&[Sample]; 2] = [&ir_l, &ir_r];
        let mut node = Convolver::new(&irs, 1.0, 0.0);
        let n = 6;
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, n);
        input.channel_mut(0)[0] = 1.0;
        input.channel_mut(1)[0] = 1.0;
        let inputs = [input];
        let mut outputs = [AudioBuffer::new(ChannelLayout::Stereo, n)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(n), &mut io);
        assert!((outputs[0].channel(0)[0] - 1.0).abs() < 1e-6);
        assert!((outputs[0].channel(1)[2] - 1.0).abs() < 1e-6);
        assert!(outputs[0].channel(1)[0].abs() < 1e-6);
    }

    #[test]
    fn latency_is_zero() {
        let ir = [1.0, 0.5];
        let irs: [&[Sample]; 1] = [&ir];
        let node = Convolver::new(&irs, 1.0, 0.0);
        assert_eq!(node.latency_frames(), 0);
    }
}
