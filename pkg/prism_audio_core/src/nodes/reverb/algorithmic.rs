//! Schroeder/Freeverb-style algorithmic room reverberator.
//!
//! Where the [`FdnReverb`](super::fdn::FdnReverb) recirculates a matrixed bank
//! of delay lines and the [`Convolver`](super::convolver::Convolver) replays a
//! measured impulse response, the [`AlgorithmicRoom`] *models* a room from three
//! classic building blocks chained in series:
//!
//! 1. **Pre-delay** — a short delay before any reflections, modelling the time
//!    the direct sound takes to reach the first wall. It cleanly separates the
//!    dry source from its ambience and is the single most important cue for
//!    perceived room size.
//! 2. **Early reflections** — a handful of discrete taps off a delay line, the
//!    sparse first-order echoes bouncing off nearby surfaces. Their pattern and
//!    spacing tell the ear the geometry and scale of the space.
//! 3. **Late reverberation** — a Schroeder/Freeverb tail: eight parallel
//!    low-pass feedback comb filters (which build a dense, coloured decay) feed
//!    four series all-pass diffusers (which smear the combs' regular echoes into
//!    a smooth, uncorrelated wash).
//!
//! The comb/all-pass tunings are the well-known Freeverb constants (expressed at
//! 44.1 kHz and rescaled to the runtime rate at construction). Odd channels add
//! a small `STEREO_SPREAD` offset to every delay length so the left and right
//! tails decorrelate into a wide stereo image, and a `width` control cross-mixes
//! stereo pairs from fully mono to fully wide.
//!
//! Every delay line and filter is allocated at construction, so
//! [`AlgorithmicRoom::process`](crate::graph::AudioNode::process) never
//! allocates, locks, or panics on the audio thread.

use alloc::vec::Vec;

use bevy_math::ops;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, flush_denormal};
use crate::param::{Ramp, Smoothed};

/// Sample rate the tuning tables below are expressed at; lengths are rescaled
/// to the runtime rate at construction.
const REFERENCE_RATE: Sample = 44_100.0;

/// Freeverb comb-filter delay lengths (frames at [`REFERENCE_RATE`]). Their
/// mutually irregular spacing keeps the combined comb resonances from fusing
/// into an audible pitch.
const COMB_TUNING: [usize; 8] = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617];

/// Freeverb all-pass diffuser delay lengths (frames at [`REFERENCE_RATE`]).
const ALLPASS_TUNING: [usize; 4] = [556, 441, 341, 225];

/// Extra frames added to every delay length on odd channels so the left and
/// right tails decorrelate into a wide stereo image.
const STEREO_SPREAD: usize = 23;

/// Input attenuation feeding the comb bank, matching Freeverb's fixed gain so
/// the summed comb outputs stay well below clipping.
const FIXED_GAIN: Sample = 0.015;

/// Maps the `[0, 1]` room-size control onto the comb feedback range
/// `[ROOM_OFFSET, ROOM_OFFSET + ROOM_SCALE]`.
const ROOM_SCALE: Sample = 0.28;
/// Base comb feedback for a room size of zero (see [`ROOM_SCALE`]).
const ROOM_OFFSET: Sample = 0.7;
/// Scales the `[0, 1]` damping control onto the comb low-pass coefficient.
const DAMP_SCALE: Sample = 0.4;
/// Fixed feedback coefficient of every all-pass diffuser (Freeverb's value).
const ALLPASS_FEEDBACK: Sample = 0.5;

/// Largest channel count the width cross-mix scratch buffer supports; covers
/// every [`ChannelLayout`](crate::buffer::ChannelLayout) the engine ships.
const MAX_CHANNELS: usize = 8;

/// Early-reflection tap times in milliseconds, a fixed diffuse first-echo
/// pattern independent of room size.
const ER_TAPS_MS: [Sample; 6] = [4.3, 8.9, 13.7, 19.1, 24.3, 30.5];
/// Gain of each early-reflection tap in [`ER_TAPS_MS`], decaying with delay.
const ER_GAINS: [Sample; 6] = [0.85, 0.72, 0.60, 0.48, 0.36, 0.24];

/// A low-pass feedback comb filter (one of the eight parallel combs that build
/// the dense late tail).
#[derive(Debug, Clone)]
struct Comb {
    /// Delay-line storage; its length sets the comb's echo period.
    buffer: Vec<Sample>,
    /// Read/write cursor into `buffer`.
    pos: usize,
    /// One-pole low-pass state held in the feedback path (the damping filter).
    filter_store: Sample,
    /// Feedback gain, derived from the room-size control.
    feedback: Sample,
    /// Low-pass damping coefficient in `[0, 1)`; higher rolls treble off faster.
    damp: Sample,
}

impl Comb {
    /// Allocates a comb with a `len`-frame delay line (clamped to at least one).
    fn new(len: usize, feedback: Sample, damp: Sample) -> Self {
        let len = len.max(1);
        let mut buffer = Vec::with_capacity(len);
        buffer.resize(len, 0.0);
        Self {
            buffer,
            pos: 0,
            filter_store: 0.0,
            feedback,
            damp,
        }
    }

    /// Processes one sample: reads the delayed output, low-passes it, and writes
    /// `input + damped_feedback` back into the line.
    #[inline]
    fn process(&mut self, input: Sample) -> Sample {
        let output = self.buffer[self.pos];
        self.filter_store =
            flush_denormal(output * (1.0 - self.damp) + self.filter_store * self.damp);
        self.buffer[self.pos] = flush_denormal(input + self.filter_store * self.feedback);
        self.pos += 1;
        if self.pos >= self.buffer.len() {
            self.pos = 0;
        }
        output
    }

    /// Zeroes the delay line and low-pass state.
    fn clear(&mut self) {
        for s in &mut self.buffer {
            *s = 0.0;
        }
        self.pos = 0;
        self.filter_store = 0.0;
    }
}

/// A Schroeder all-pass diffuser (one of the four series stages that smear the
/// comb tail into a smooth wash).
#[derive(Debug, Clone)]
struct Allpass {
    /// Delay-line storage; its length sets the diffusion density.
    buffer: Vec<Sample>,
    /// Read/write cursor into `buffer`.
    pos: usize,
}

impl Allpass {
    /// Allocates an all-pass with a `len`-frame delay line (clamped to at least
    /// one).
    fn new(len: usize) -> Self {
        let len = len.max(1);
        let mut buffer = Vec::with_capacity(len);
        buffer.resize(len, 0.0);
        Self { buffer, pos: 0 }
    }

    /// Processes one sample through the classic Schroeder all-pass difference
    /// equation with fixed feedback [`ALLPASS_FEEDBACK`].
    #[inline]
    fn process(&mut self, input: Sample) -> Sample {
        let buffered = self.buffer[self.pos];
        let output = -input + buffered;
        self.buffer[self.pos] = flush_denormal(input + buffered * ALLPASS_FEEDBACK);
        self.pos += 1;
        if self.pos >= self.buffer.len() {
            self.pos = 0;
        }
        output
    }

    /// Zeroes the delay line.
    fn clear(&mut self) {
        for s in &mut self.buffer {
            *s = 0.0;
        }
        self.pos = 0;
    }
}

/// Per-channel reverb chain: pre-delay -> early reflections + late tail.
#[derive(Debug, Clone)]
struct ChannelReverb {
    /// Pre-delay ring buffer; the direct sound is delayed by its full length.
    predelay: Vec<Sample>,
    /// Cursor into `predelay`.
    pd_pos: usize,
    /// Early-reflection tapped delay line, fed by the pre-delayed signal.
    er_line: Vec<Sample>,
    /// Cursor into `er_line`.
    er_pos: usize,
    /// Early-reflection taps as `(delay_frames, gain)` pairs.
    er_taps: Vec<(usize, Sample)>,
    /// The eight parallel low-pass feedback combs forming the dense tail.
    combs: Vec<Comb>,
    /// The four series all-pass diffusers smoothing the comb tail.
    allpasses: Vec<Allpass>,
}

impl ChannelReverb {
    /// Builds one channel's chain. `spread` adds a per-channel offset (used to
    /// decorrelate odd channels), `feedback`/`damp` seed the combs, and the
    /// pre-delay/early-reflection lines are sized from the sample rate.
    fn new(
        sample_rate: u32,
        pre_delay_ms: Sample,
        spread: usize,
        feedback: Sample,
        damp: Sample,
    ) -> Self {
        let sr = sample_rate as Sample;
        let rate_scale = sr / REFERENCE_RATE;

        let pd_frames = ms_to_frames(pre_delay_ms, sr).max(1);
        let mut predelay = Vec::with_capacity(pd_frames);
        predelay.resize(pd_frames, 0.0);

        // Size the early-reflection line to the longest tap and pre-compute the
        // integer tap delays / gains once, so the hot path only indexes.
        let mut er_taps = Vec::with_capacity(ER_TAPS_MS.len());
        let mut max_tap = 1usize;
        for (ms, gain) in ER_TAPS_MS.iter().zip(ER_GAINS.iter()) {
            let d = ms_to_frames(*ms, sr).max(1);
            max_tap = max_tap.max(d);
            er_taps.push((d, *gain));
        }
        let er_len = max_tap + 1;
        let mut er_line = Vec::with_capacity(er_len);
        er_line.resize(er_len, 0.0);

        let mut combs = Vec::with_capacity(COMB_TUNING.len());
        for tuning in COMB_TUNING {
            let len = scale_len(tuning, rate_scale) + spread;
            combs.push(Comb::new(len, feedback, damp));
        }
        let mut allpasses = Vec::with_capacity(ALLPASS_TUNING.len());
        for tuning in ALLPASS_TUNING {
            let len = scale_len(tuning, rate_scale) + spread;
            allpasses.push(Allpass::new(len));
        }

        Self {
            predelay,
            pd_pos: 0,
            er_line,
            er_pos: 0,
            er_taps,
            combs,
            allpasses,
        }
    }

    /// Processes one sample. Returns `(early, late)` so the caller can weight
    /// the early reflections independently of the late tail.
    #[inline]
    fn process(&mut self, x: Sample) -> (Sample, Sample) {
        // Pre-delay: emit the oldest sample, then store the new one.
        let pd = self.predelay[self.pd_pos];
        self.predelay[self.pd_pos] = x;
        self.pd_pos += 1;
        if self.pd_pos >= self.predelay.len() {
            self.pd_pos = 0;
        }

        // Early reflections: tap the pre-delayed signal at several offsets.
        let er_len = self.er_line.len();
        self.er_line[self.er_pos] = pd;
        let mut early = 0.0;
        for &(delay, gain) in &self.er_taps {
            let idx = (self.er_pos + er_len - delay) % er_len;
            early += self.er_line[idx] * gain;
        }
        self.er_pos += 1;
        if self.er_pos >= er_len {
            self.er_pos = 0;
        }

        // Late tail: parallel combs summed, then series all-pass diffusers.
        let late_in = pd * FIXED_GAIN;
        let mut late = 0.0;
        for comb in &mut self.combs {
            late += comb.process(late_in);
        }
        for allpass in &mut self.allpasses {
            late = allpass.process(late);
        }

        (early, late)
    }

    /// Updates every comb's feedback and damping in place (allocation-free).
    #[inline]
    fn set_tail(&mut self, feedback: Sample, damp: Sample) {
        for comb in &mut self.combs {
            comb.feedback = feedback;
            comb.damp = damp;
        }
    }

    /// Zeroes all delay lines and filter state.
    fn clear(&mut self) {
        for s in &mut self.predelay {
            *s = 0.0;
        }
        self.pd_pos = 0;
        for s in &mut self.er_line {
            *s = 0.0;
        }
        self.er_pos = 0;
        for comb in &mut self.combs {
            comb.clear();
        }
        for allpass in &mut self.allpasses {
            allpass.clear();
        }
    }
}

/// Converts a duration in milliseconds to a whole number of frames.
#[inline]
fn ms_to_frames(ms: Sample, sample_rate: Sample) -> usize {
    ops::round(ms.max(0.0) * 0.001 * sample_rate) as usize
}

/// Rescales a reference-rate delay length to the runtime rate.
#[inline]
fn scale_len(reference_frames: usize, rate_scale: Sample) -> usize {
    (ops::round(reference_frames as Sample * rate_scale) as usize).max(1)
}

/// Maps the `[0, 1]` room-size control onto the comb feedback coefficient.
#[inline]
fn feedback_for_room(room_size: Sample) -> Sample {
    room_size.clamp(0.0, 1.0) * ROOM_SCALE + ROOM_OFFSET
}

/// Maps the `[0, 1]` damping control onto the comb low-pass coefficient.
#[inline]
fn damp_coefficient(damping: Sample) -> Sample {
    (damping.clamp(0.0, 1.0) * DAMP_SCALE).clamp(0.0, 0.999)
}

/// Construction parameters for an [`AlgorithmicRoom`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct AlgorithmicRoomParams {
    /// Room size in `[0, 1]`; larger values raise the comb feedback for a
    /// longer, more sustained tail.
    pub room_size: Sample,
    /// High-frequency damping in `[0, 1]`; higher values roll the tail's treble
    /// off faster, modelling absorptive surfaces.
    pub damping: Sample,
    /// Stereo width in `[0, 1]`; `0` collapses the tail to mono, `1` keeps the
    /// fully decorrelated left/right image.
    pub width: Sample,
    /// Pre-delay before any reflections, in milliseconds (clamped to `>= 0`).
    pub pre_delay_ms: Sample,
    /// Mix level of the early reflections relative to the late tail.
    pub early_level: Sample,
    /// Wet (reverberated) mix gain.
    pub wet: Sample,
    /// Dry (unprocessed input) mix gain.
    pub dry: Sample,
}

impl Default for AlgorithmicRoomParams {
    fn default() -> Self {
        Self {
            room_size: 0.7,
            damping: 0.35,
            width: 1.0,
            pre_delay_ms: 12.0,
            early_level: 0.5,
            wet: 0.4,
            dry: 1.0,
        }
    }
}

/// A Schroeder/Freeverb-style algorithmic room reverb (input port 0 -> output
/// port 0).
///
/// Each channel owns an independent pre-delay, early-reflection tap line, and
/// Freeverb comb/all-pass tail; odd channels are offset by [`STEREO_SPREAD`] so
/// the tail is wide, and `width` cross-mixes stereo pairs. The output is
/// `dry * input + wet * (early_level * early + late)`.
#[derive(Debug, Clone)]
pub struct AlgorithmicRoom {
    /// Number of channels processed (mirrors the I/O buffer channel count).
    channels: usize,
    /// Per-channel reverb chains.
    chans: Vec<ChannelReverb>,
    /// Cached room-size control, so `set_room_size` can recompute feedback.
    room_size: Sample,
    /// Cached damping control, so `set_damping` can recompute the coefficient.
    damping: Sample,
    /// Smoothed early-reflection mix level.
    early_level: Smoothed,
    /// Smoothed stereo width in `[0, 1]`.
    width: Smoothed,
    /// Smoothed wet (reverberated) mix gain.
    wet: Smoothed,
    /// Smoothed dry (unprocessed input) mix gain.
    dry: Smoothed,
}

impl AlgorithmicRoom {
    /// Builds a room reverb for `channels` channels at `sample_rate` Hz. All
    /// delay lines and filters are allocated here; parameters start settled.
    #[must_use]
    pub fn new(sample_rate: u32, channels: usize, params: AlgorithmicRoomParams) -> Self {
        let sr = sample_rate.max(1);
        let channels = channels.max(1);
        let feedback = feedback_for_room(params.room_size);
        let damp = damp_coefficient(params.damping);

        let mut chans = Vec::with_capacity(channels);
        for ch in 0..channels {
            let spread = if ch % 2 == 1 { STEREO_SPREAD } else { 0 };
            chans.push(ChannelReverb::new(
                sr,
                params.pre_delay_ms,
                spread,
                feedback,
                damp,
            ));
        }

        Self {
            channels,
            chans,
            room_size: params.room_size.clamp(0.0, 1.0),
            damping: params.damping.clamp(0.0, 1.0),
            early_level: Smoothed::new(params.early_level),
            width: Smoothed::new(params.width.clamp(0.0, 1.0)),
            wet: Smoothed::new(params.wet),
            dry: Smoothed::new(params.dry),
        }
    }

    /// Returns the number of channels this reverb processes.
    #[inline]
    #[must_use]
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Returns the cached room-size control in `[0, 1]`.
    #[inline]
    #[must_use]
    pub fn room_size(&self) -> Sample {
        self.room_size
    }

    /// Sets the room size in `[0, 1]`, recomputing every comb's feedback in
    /// place (allocation-free).
    #[inline]
    pub fn set_room_size(&mut self, room_size: Sample) {
        self.room_size = room_size.clamp(0.0, 1.0);
        let feedback = feedback_for_room(self.room_size);
        let damp = damp_coefficient(self.damping);
        for chan in &mut self.chans {
            chan.set_tail(feedback, damp);
        }
    }

    /// Sets the damping control in `[0, 1]`, recomputing every comb's low-pass
    /// coefficient in place (allocation-free).
    #[inline]
    pub fn set_damping(&mut self, damping: Sample) {
        self.damping = damping.clamp(0.0, 1.0);
        let feedback = feedback_for_room(self.room_size);
        let damp = damp_coefficient(self.damping);
        for chan in &mut self.chans {
            chan.set_tail(feedback, damp);
        }
    }

    /// Sets the stereo width in `[0, 1]`, gliding with `ramp`.
    #[inline]
    pub fn set_width(&mut self, width: Sample, ramp: Ramp) {
        self.width.set_target(width.clamp(0.0, 1.0), ramp);
    }

    /// Sets the early-reflection mix level, gliding with `ramp`.
    #[inline]
    pub fn set_early_level(&mut self, early_level: Sample, ramp: Ramp) {
        self.early_level.set_target(early_level, ramp);
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

impl AudioNode for AlgorithmicRoom {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let channels = output
            .channels()
            .min(input.channels())
            .min(self.channels);
        let frames = output.active_frames().min(input.active_frames());

        for f in 0..frames {
            let early_level = self.early_level.next_sample();
            let width = self.width.next_sample().clamp(0.0, 1.0);
            let wet = self.wet.next_sample();
            let dry = self.dry.next_sample();

            // Freeverb stereo-width weights: wet1 keeps the channel's own tail,
            // wet2 bleeds in its pair partner.
            let wet1 = width * 0.5 + 0.5;
            let wet2 = (1.0 - width) * 0.5;

            // Compute each channel's wet signal into a stack scratch buffer so
            // the width cross-mix can read neighbours without extra allocation.
            let mut wet_buf = [0.0; MAX_CHANNELS];
            let active = channels.min(MAX_CHANNELS);
            for (ch, slot) in wet_buf.iter_mut().enumerate().take(active) {
                let x = input.channel(ch)[f];
                let (early, late) = self.chans[ch].process(x);
                *slot = early_level * early + late;
            }

            for ch in 0..channels {
                let x = input.channel(ch)[f];
                let own = if ch < active { wet_buf[ch] } else { 0.0 };
                // Cross-mix with the paired channel (ch ^ 1) for stereo width.
                let partner = ch ^ 1;
                let mixed = if partner < active {
                    own * wet1 + wet_buf[partner] * wet2
                } else {
                    own
                };
                output.channel_mut(ch)[f] = dry * x + wet * mixed;
            }
        }
    }

    fn reset(&mut self) {
        for chan in &mut self.chans {
            chan.clear();
        }
        self.early_level = Smoothed::new(self.early_level.target());
        self.width = Smoothed::new(self.width.target());
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

    fn stereo(frames: usize) -> AudioBuffer {
        AudioBuffer::new(ChannelLayout::Stereo, frames)
    }

    #[test]
    fn dry_passthrough_when_wet_zero() {
        let params = AlgorithmicRoomParams {
            wet: 0.0,
            dry: 1.0,
            ..AlgorithmicRoomParams::default()
        };
        let mut node = AlgorithmicRoom::new(48_000, 2, params);
        let mut input = stereo(64);
        for ch in 0..2 {
            for (i, s) in input.channel_mut(ch).iter_mut().enumerate() {
                *s = (i as Sample) * 0.005 - 0.1 + ch as Sample * 0.01;
            }
        }
        let inputs = [input.clone()];
        let mut outputs = [stereo(64)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(64), &mut io);
        for ch in 0..2 {
            for (o, i) in outputs[0].channel(ch).iter().zip(inputs[0].channel(ch)) {
                assert!((o - i).abs() < 1e-6, "dry path altered: {o} vs {i}");
            }
        }
    }

    #[test]
    fn tail_has_energy_and_is_bounded() {
        let params = AlgorithmicRoomParams {
            wet: 1.0,
            dry: 0.0,
            ..AlgorithmicRoomParams::default()
        };
        let mut node = AlgorithmicRoom::new(48_000, 2, params);
        let n = 16_384;
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
    fn larger_room_sustains_longer() {
        fn late_energy(room_size: Sample) -> Sample {
            let params = AlgorithmicRoomParams {
                wet: 1.0,
                dry: 0.0,
                room_size,
                ..AlgorithmicRoomParams::default()
            };
            let mut node = AlgorithmicRoom::new(48_000, 1, params);
            let n = 48_000;
            let mut input = mono(n);
            input.channel_mut(0)[0] = 1.0;
            let inputs = [input];
            let mut outputs = [mono(n)];
            let mut io = ProcessIo::new(&inputs, &mut outputs);
            node.process(&ctx(n), &mut io);
            outputs[0].channel(0)[n / 2..].iter().map(|s| s * s).sum()
        }
        let small = late_energy(0.1);
        let large = late_energy(0.95);
        assert!(
            large > small,
            "larger room should retain more late energy: small={small} large={large}"
        );
    }

    #[test]
    fn pre_delay_defers_first_output() {
        // With a long pre-delay and no early/late contribution near t=0, the
        // very first wet samples must be silent.
        let params = AlgorithmicRoomParams {
            wet: 1.0,
            dry: 0.0,
            pre_delay_ms: 20.0,
            early_level: 1.0,
            ..AlgorithmicRoomParams::default()
        };
        let mut node = AlgorithmicRoom::new(48_000, 1, params);
        let n = 4_096;
        let mut input = mono(n);
        input.channel_mut(0)[0] = 1.0;
        let inputs = [input];
        let mut outputs = [mono(n)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(n), &mut io);
        // 20 ms at 48 kHz is 960 frames; the first few hundred must be silent.
        for &s in &outputs[0].channel(0)[..400] {
            assert!(s.abs() < 1e-9, "output before pre-delay elapsed: {s}");
        }
    }

    #[test]
    fn reset_clears_tail() {
        let params = AlgorithmicRoomParams {
            wet: 1.0,
            dry: 0.0,
            ..AlgorithmicRoomParams::default()
        };
        let mut node = AlgorithmicRoom::new(48_000, 1, params);
        let n = 1_024;
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
    fn latency_is_zero() {
        let node = AlgorithmicRoom::new(48_000, 2, AlgorithmicRoomParams::default());
        assert_eq!(node.latency_frames(), 0);
    }
}
