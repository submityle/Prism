//! Real-time granular texture processor (grain cloud).
//!
//! A [`GranularNode`] continuously records its input into a mono circular
//! capture buffer and sprays short, enveloped "grains" that read back from the
//! recent past at a controllable rate. Each grain gets independently
//! randomized position, pitch, and stereo pan, so a steady input dissolves
//! into an evolving cloud of overlapping micro-events: shimmering pads,
//! time-smeared ambiences, pitched clouds, or glitchy stutters depending on the
//! controls.
//!
//! The core controls mirror the classic granular vocabulary found in GRM
//! Tools, Ableton's Granulator, Reaktor, and the granular synths exposed by
//! modern game-audio middleware:
//!
//! - **grain size** -- the duration of each windowed read.
//! - **density** -- how many grains spawn per second.
//! - **position** -- how far behind the write head grains start reading, with a
//!   jitter spread for diffusion.
//! - **pitch** -- a base playback ratio plus a per-grain semitone jitter.
//! - **spread** -- how widely grains scatter across the stereo field.
//!
//! Each grain is weighted by a Hann window so overlapping grains cross-fade
//! smoothly with no clicks, and the summed cloud is scaled by an overlap-based
//! normalization so loudness stays roughly constant as density and grain size
//! change. All storage (capture ring and a fixed pool of
//! [`MAX_GRAINS`] grains) is allocated at construction, so
//! [`GranularNode::process`](crate::graph::AudioNode::process) performs no heap
//! allocation, takes no locks, and never panics.
//!
//! # Provenance
//!
//! Implemented from first principles as a windowed overlap-add grain
//! scheduler reading a circular capture buffer. No source code or derivative
//! code from UE, Unity, Godot, Wwise, FMOD, Steam Audio, Google Resonance
//! Audio, or Web Audio was consulted or copied; only the well-known public
//! concept of granular synthesis informs the design. There is no AI or machine
//! learning of any kind: grain randomization is a deterministic classic
//! `xorshift64` pseudo-random generator seeded from the parameters.
//!
//! # Relationship
//!
//! The grain pitch jitter reuses [`semitones_to_ratio`] from the sibling
//! [`pitch_shifter`](crate::nodes::effects::pitch_shifter) module and clamps to
//! its [`MIN_PITCH_RATIO`] / [`MAX_PITCH_RATIO`] range, rather than duplicating
//! the semitone-to-ratio conversion. Grains perform their own lightweight
//! linear interpolation into the capture ring; this is deliberately simpler
//! than, and independent of, the resampling in
//! [`sample_player`](crate::nodes::sources::sample_player), because grains read
//! a shared live-captured ring rather than an owned static asset.
//!
//! # Real-time contract
//!
//! The capture ring and the [`MAX_GRAINS`] grain pool are sized and allocated
//! in [`GranularNode::new`]. `process` only reads inputs, advances integer and
//! float cursors, and writes outputs; it flushes denormals on every capture
//! write and on every output sample, performs no allocation or locking, and is
//! panic-free (every ring access is reduced modulo the capture length and every
//! read position is clamped to the valid recorded window).

use alloc::{vec, vec::Vec};

use bevy_math::ops;

use core::f32::consts::TAU;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, equal_power_pan, flush_denormal, lerp};
use crate::nodes::effects::pitch_shifter::{MAX_PITCH_RATIO, MIN_PITCH_RATIO, semitones_to_ratio};
use crate::param::{Ramp, Smoothed};

/// Maximum number of grains that can sound simultaneously.
///
/// Spawn requests beyond this cap are dropped, bounding both CPU cost and
/// output amplitude.
pub const MAX_GRAINS: usize = 64;

/// Length of the mono capture ring, in seconds of recent input history.
pub const DEFAULT_CAPTURE_SECONDS: Sample = 4.0;

/// Smallest allowed grain duration, in milliseconds.
pub const MIN_GRAIN_MS: Sample = 2.0;

/// Largest allowed grain duration, in milliseconds.
pub const MAX_GRAIN_MS: Sample = 2000.0;

/// Smallest allowed spawn density, in grains per second.
pub const MIN_DENSITY_HZ: Sample = 0.1;

/// Largest allowed spawn density, in grains per second.
pub const MAX_DENSITY_HZ: Sample = 200.0;

/// Largest allowed stereo scatter (full left-to-right spread).
pub const MAX_SPREAD: Sample = 1.0;

/// Default pseudo-random seed (a golden-ratio-derived odd constant).
pub const DEFAULT_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// Reciprocal of 2^24, used to map a 24-bit random word onto `[0, 1)`.
const INV_RNG_SCALE: Sample = 1.0 / 16_777_216.0;

/// Construction and automation parameters for a [`GranularNode`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct GranularParams {
    /// Grain duration in milliseconds (clamped to [`MIN_GRAIN_MS`] ..
    /// [`MAX_GRAIN_MS`]).
    pub grain_size_ms: Sample,
    /// Spawn rate in grains per second (clamped to [`MIN_DENSITY_HZ`] ..
    /// [`MAX_DENSITY_HZ`]).
    pub density_hz: Sample,
    /// How far behind the write head each grain starts reading, in seconds.
    pub position_seconds: Sample,
    /// Random spread added to the read position, in milliseconds.
    pub position_jitter_ms: Sample,
    /// Base playback ratio applied to every grain (1 = original pitch).
    pub pitch_ratio: Sample,
    /// Peak random per-grain detune, in semitones (bipolar).
    pub pitch_jitter_semitones: Sample,
    /// Stereo scatter in `[0, MAX_SPREAD]` (0 = centered, 1 = full width).
    pub spread: Sample,
    /// Wet (granular cloud) mix gain.
    pub wet: Sample,
    /// Dry (unprocessed input) mix gain.
    pub dry: Sample,
    /// Seed for the deterministic grain randomizer.
    pub seed: u64,
}

impl Default for GranularParams {
    fn default() -> Self {
        Self {
            grain_size_ms: 80.0,
            density_hz: 20.0,
            position_seconds: 0.25,
            position_jitter_ms: 50.0,
            pitch_ratio: 1.0,
            pitch_jitter_semitones: 0.0,
            spread: 0.6,
            wet: 1.0,
            dry: 0.0,
            seed: DEFAULT_SEED,
        }
    }
}

impl GranularParams {
    /// Returns a copy with every field clamped to its valid range and any
    /// non-finite value replaced by the default.
    #[must_use]
    pub fn sanitised(self) -> Self {
        let d = Self::default();
        let fix = |x: Sample, lo: Sample, hi: Sample, fallback: Sample| {
            if x.is_finite() { x.clamp(lo, hi) } else { fallback }
        };
        Self {
            grain_size_ms: fix(self.grain_size_ms, MIN_GRAIN_MS, MAX_GRAIN_MS, d.grain_size_ms),
            density_hz: fix(self.density_hz, MIN_DENSITY_HZ, MAX_DENSITY_HZ, d.density_hz),
            position_seconds: fix(
                self.position_seconds,
                0.0,
                DEFAULT_CAPTURE_SECONDS,
                d.position_seconds,
            ),
            position_jitter_ms: fix(self.position_jitter_ms, 0.0, 10_000.0, d.position_jitter_ms),
            pitch_ratio: fix(self.pitch_ratio, MIN_PITCH_RATIO, MAX_PITCH_RATIO, d.pitch_ratio),
            pitch_jitter_semitones: fix(self.pitch_jitter_semitones, 0.0, 48.0, 0.0),
            spread: fix(self.spread, 0.0, MAX_SPREAD, d.spread),
            wet: fix(self.wet, 0.0, 8.0, d.wet),
            dry: fix(self.dry, 0.0, 8.0, d.dry),
            seed: self.seed,
        }
    }
}

/// A single scheduled grain reading from the capture ring.
#[derive(Debug, Clone, Copy, Default)]
struct Grain {
    /// Whether this slot is currently sounding.
    active: bool,
    /// Absolute read position (frame index into the recorded timeline).
    pos: f64,
    /// Samples elapsed since the grain was spawned.
    age: u32,
    /// Total grain lifetime in samples.
    length: u32,
    /// Reciprocal of `length`, precomputed for the window phase.
    inv_length: Sample,
    /// Read advance per output sample (effective pitch ratio).
    pitch: Sample,
    /// Overlap-normalized amplitude.
    gain: Sample,
    /// Left-channel equal-power pan gain.
    gain_l: Sample,
    /// Right-channel equal-power pan gain.
    gain_r: Sample,
}

/// Deterministic `xorshift64`-star pseudo-random generator for grain jitter.
#[derive(Debug, Clone)]
struct GrainRng {
    state: u64,
}

impl GrainRng {
    /// Creates a generator from `seed`, forcing a non-zero odd state.
    fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    /// Advances the generator and returns the next 64-bit word.
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Returns a uniform sample in `[0, 1)`.
    fn next_unit(&mut self) -> Sample {
        ((self.next_u64() >> 40) as u32 as Sample) * INV_RNG_SCALE
    }

    /// Returns a uniform sample in `[-1, 1)`.
    fn next_bipolar(&mut self) -> Sample {
        self.next_unit() * 2.0 - 1.0
    }
}

/// A real-time granular texture processor (input port 0 -> output port 0).
///
/// The output is `dry * input + wet * cloud`, where `cloud` is the sum of all
/// active windowed grains panned across the output channels. Wet and dry gains
/// are [`Smoothed`] so mix automation stays click-free; grain-shaping controls
/// take effect on the next spawned grain.
#[derive(Debug, Clone)]
pub struct GranularNode {
    /// Sample rate in Hz.
    sample_rate: u32,
    /// Output (and dry passthrough) channel count.
    channels: usize,
    /// Mono circular capture buffer of recent input.
    capture: Vec<Sample>,
    /// Length of [`Self::capture`] in frames.
    capture_len: usize,
    /// Total frames written so far (monotonic; newest index is this minus one).
    write_total: u64,
    /// Fixed pool of grains.
    grains: Vec<Grain>,
    /// Deterministic jitter source.
    rng: GrainRng,
    /// Seed retained for [`AudioNode::reset`].
    seed: u64,
    /// Frames accumulated toward the next spawn.
    spawn_timer: Sample,
    /// Current grain length in frames (from `grain_size_ms`).
    grain_len_frames: Sample,
    /// Current spawn density in grains per second.
    density_hz: Sample,
    /// Current read offset behind the write head, in frames.
    position_frames: Sample,
    /// Current random read-position spread, in frames.
    position_jitter_frames: Sample,
    /// Current base playback ratio.
    pitch_ratio: Sample,
    /// Current peak per-grain detune in semitones.
    pitch_jitter_semitones: Sample,
    /// Current stereo scatter in `[0, MAX_SPREAD]`.
    spread: Sample,
    /// Overlap-based amplitude normalization applied to each grain.
    gain_norm: Sample,
    /// Smoothed wet mix gain.
    wet: Smoothed,
    /// Smoothed dry mix gain.
    dry: Smoothed,
}

/// Default smoothing ramp for wet/dry automation, in seconds.
const MIX_RAMP_SECONDS: Sample = 0.01;

impl GranularNode {
    /// Builds a granular node for `channels` output channels at `sample_rate`.
    ///
    /// The capture ring is sized to hold [`DEFAULT_CAPTURE_SECONDS`] plus one
    /// maximum grain of history, and the grain pool is pre-allocated to
    /// [`MAX_GRAINS`]. Parameters are sanitized before use.
    ///
    /// # Example
    ///
    /// ```
    /// use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
    /// use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
    /// use prism_audio_core::nodes::effects::{GranularNode, GranularParams};
    ///
    /// let mut node = GranularNode::new(48_000, 1, GranularParams::default());
    /// assert_eq!(node.latency_frames(), 0);
    /// let mut input = AudioBuffer::new(ChannelLayout::Mono, 8);
    /// let mut output = AudioBuffer::new(ChannelLayout::Mono, 8);
    /// input.set_active_frames(8);
    /// output.set_active_frames(8);
    /// for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
    ///     *s = if i % 2 == 0 { 0.5 } else { -0.5 };
    /// }
    /// let ctx = RenderContext { sample_rate: 48_000, frames: 8, playhead: 0 };
    /// let mut io = ProcessIo::new(core::slice::from_ref(&input), core::slice::from_mut(&mut output));
    /// node.process(&ctx, &mut io);
    /// assert!(output.channel(0).iter().all(|s| s.is_finite()));
    /// ```
    #[must_use]
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "capture length is a positive, bounded frame count"
    )]
    pub fn new(sample_rate: u32, channels: usize, params: GranularParams) -> Self {
        let p = params.sanitised();
        let channels = channels.max(1);
        let secs = f64::from(DEFAULT_CAPTURE_SECONDS) + f64::from(MAX_GRAIN_MS) / 1000.0;
        let capture_len = ((f64::from(sample_rate) * secs) as usize + 2).max(2);
        let mut node = Self {
            sample_rate,
            channels,
            capture: vec![0.0; capture_len],
            capture_len,
            write_total: 0,
            grains: vec![Grain::default(); MAX_GRAINS],
            rng: GrainRng::new(p.seed),
            seed: p.seed,
            spawn_timer: 0.0,
            grain_len_frames: 0.0,
            density_hz: p.density_hz,
            position_frames: 0.0,
            position_jitter_frames: 0.0,
            pitch_ratio: p.pitch_ratio,
            pitch_jitter_semitones: p.pitch_jitter_semitones,
            spread: p.spread,
            gain_norm: 1.0,
            wet: Smoothed::new(p.wet),
            dry: Smoothed::new(p.dry),
        };
        node.grain_len_frames = node.ms_to_frames(p.grain_size_ms);
        node.position_frames = p.position_seconds * node.sample_rate as Sample;
        node.position_jitter_frames = node.ms_to_frames(p.position_jitter_ms);
        node.recompute_gain();
        node
    }

    /// Converts milliseconds to frames at the node's sample rate.
    fn ms_to_frames(&self, ms: Sample) -> Sample {
        ms * self.sample_rate as Sample / 1000.0
    }

    /// Recomputes the overlap-based amplitude normalization.
    ///
    /// The expected number of simultaneous grains is `density * grain_seconds`;
    /// scaling by its inverse square root keeps summed power roughly constant
    /// as density and grain size change.
    fn recompute_gain(&mut self) {
        let grain_seconds = self.grain_len_frames / self.sample_rate as Sample;
        let overlap = (self.density_hz * grain_seconds).max(1.0);
        self.gain_norm = 1.0 / ops::sqrt(overlap);
    }

    /// Returns the configured output channel count.
    #[must_use]
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Returns the capture ring length in frames.
    #[must_use]
    pub fn capture_frames(&self) -> usize {
        self.capture_len
    }

    /// Returns the current base playback ratio.
    #[must_use]
    pub fn pitch_ratio(&self) -> Sample {
        self.pitch_ratio
    }

    /// Returns the current spawn density in grains per second.
    #[must_use]
    pub fn density(&self) -> Sample {
        self.density_hz
    }

    /// Returns the number of grains currently sounding.
    #[must_use]
    pub fn active_grains(&self) -> usize {
        self.grains.iter().filter(|g| g.active).count()
    }

    /// Sets the grain duration in milliseconds (clamped and gain-compensated).
    pub fn set_grain_size_ms(&mut self, ms: Sample) {
        let ms = if ms.is_finite() {
            ms.clamp(MIN_GRAIN_MS, MAX_GRAIN_MS)
        } else {
            MIN_GRAIN_MS
        };
        self.grain_len_frames = self.ms_to_frames(ms);
        self.recompute_gain();
    }

    /// Sets the spawn density in grains per second (clamped and compensated).
    pub fn set_density(&mut self, hz: Sample) {
        self.density_hz = if hz.is_finite() {
            hz.clamp(MIN_DENSITY_HZ, MAX_DENSITY_HZ)
        } else {
            MIN_DENSITY_HZ
        };
        self.recompute_gain();
    }

    /// Sets how far behind the write head grains start reading, in seconds.
    pub fn set_position_seconds(&mut self, seconds: Sample) {
        let s = if seconds.is_finite() {
            seconds.clamp(0.0, DEFAULT_CAPTURE_SECONDS)
        } else {
            0.0
        };
        self.position_frames = s * self.sample_rate as Sample;
    }

    /// Sets the random read-position spread, in milliseconds.
    pub fn set_position_jitter_ms(&mut self, ms: Sample) {
        let ms = if ms.is_finite() {
            ms.clamp(0.0, 10_000.0)
        } else {
            0.0
        };
        self.position_jitter_frames = self.ms_to_frames(ms);
    }

    /// Sets the base playback ratio (clamped to the pitch-shifter range).
    pub fn set_pitch_ratio(&mut self, ratio: Sample) {
        self.pitch_ratio = if ratio.is_finite() {
            ratio.clamp(MIN_PITCH_RATIO, MAX_PITCH_RATIO)
        } else {
            1.0
        };
    }

    /// Sets the base pitch from a semitone offset.
    pub fn set_pitch_semitones(&mut self, semitones: Sample) {
        let st = if semitones.is_finite() { semitones } else { 0.0 };
        self.set_pitch_ratio(semitones_to_ratio(st));
    }

    /// Sets the peak per-grain random detune, in semitones.
    pub fn set_pitch_jitter_semitones(&mut self, semitones: Sample) {
        self.pitch_jitter_semitones = if semitones.is_finite() {
            semitones.clamp(0.0, 48.0)
        } else {
            0.0
        };
    }

    /// Sets the stereo scatter in `[0, MAX_SPREAD]`.
    pub fn set_spread(&mut self, spread: Sample) {
        self.spread = if spread.is_finite() {
            spread.clamp(0.0, MAX_SPREAD)
        } else {
            0.0
        };
    }

    /// Sets the smoothed wet (cloud) mix gain.
    pub fn set_wet(&mut self, gain: Sample) {
        let g = if gain.is_finite() { gain.max(0.0) } else { 0.0 };
        self.wet
            .set_target(g, Ramp::linear_seconds(MIX_RAMP_SECONDS, self.sample_rate));
    }

    /// Sets the smoothed dry (passthrough) mix gain.
    pub fn set_dry(&mut self, gain: Sample) {
        let g = if gain.is_finite() { gain.max(0.0) } else { 0.0 };
        self.dry
            .set_target(g, Ramp::linear_seconds(MIX_RAMP_SECONDS, self.sample_rate));
    }

    /// Spawns one grain into a free pool slot, if any is available.
    fn spawn_grain(&mut self) {
        let Some(slot) = self.grains.iter().position(|g| !g.active) else {
            return;
        };
        let length = grain_length(self.grain_len_frames);
        let inv_length = 1.0 / length as Sample;
        let detune = self.rng.next_bipolar() * self.pitch_jitter_semitones;
        let pitch =
            (self.pitch_ratio * semitones_to_ratio(detune)).clamp(MIN_PITCH_RATIO, MAX_PITCH_RATIO);
        let back = self.position_frames + self.rng.next_unit() * self.position_jitter_frames;
        let newest = newest_frame(self.write_total);
        let pos = newest - f64::from(back);
        let pan = (self.rng.next_bipolar() * self.spread).clamp(-1.0, 1.0);
        let (gain_l, gain_r) = equal_power_pan(pan);
        let g = &mut self.grains[slot];
        g.active = true;
        g.age = 0;
        g.length = length;
        g.inv_length = inv_length;
        g.pitch = pitch;
        g.pos = pos;
        g.gain = self.gain_norm;
        g.gain_l = gain_l;
        g.gain_r = gain_r;
    }
}

/// Rounds a grain length in frames to at least two samples.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "grain length is a small positive frame count after rounding"
)]
fn grain_length(frames: Sample) -> u32 {
    (ops::round(frames) as u32).max(2)
}

/// Returns the newest recorded frame index as a float timeline position.
#[expect(
    clippy::cast_precision_loss,
    reason = "frame counts stay far below 2^52 so the f64 conversion is exact"
)]
fn newest_frame(write_total: u64) -> f64 {
    write_total.saturating_sub(1) as f64
}

/// Reduces an absolute frame index to a capture-ring slot.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the modular index is strictly less than len, which fits usize"
)]
fn ring_index(abs: u64, len: usize) -> usize {
    (abs % len as u64) as usize
}

/// Reads the capture ring at absolute position `pos` with linear
/// interpolation, clamped to the valid recorded window.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "pos is clamped non-negative before flooring; frame counts stay below 2^52"
)]
fn read_capture(capture: &[Sample], write_total: u64, len: usize, pos: f64) -> Sample {
    if write_total == 0 {
        return 0.0;
    }
    let newest = (write_total - 1) as f64;
    let oldest = write_total.saturating_sub(len as u64) as f64;
    let p = pos.clamp(oldest, newest);
    let i0 = p as u64;
    let frac = (p - i0 as f64) as Sample;
    let i1 = (i0 + 1).min(write_total - 1);
    let s0 = capture[ring_index(i0, len)];
    let s1 = capture[ring_index(i1, len)];
    lerp(s0, s1, frac)
}

impl AudioNode for GranularNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let channels = output.channels().min(self.channels);
        let frames = output.active_frames();
        let in_channels = input.channels();
        let cap_len = self.capture_len;
        let inv_in = if in_channels > 0 {
            1.0 / in_channels as Sample
        } else {
            0.0
        };

        for f in 0..frames {
            // Downmix the input frame to mono and record it.
            let mut mono = 0.0;
            for ch in 0..in_channels {
                mono += input.channel(ch)[f];
            }
            mono *= inv_in;
            let widx = ring_index(self.write_total, cap_len);
            self.capture[widx] = flush_denormal(mono);
            self.write_total += 1;

            // Schedule any grains that fall due this frame.
            if self.density_hz > 0.0 {
                self.spawn_timer += 1.0;
                let interval = (self.sample_rate as Sample / self.density_hz).max(1.0);
                let mut guard = 0;
                while self.spawn_timer >= interval && guard < MAX_GRAINS {
                    self.spawn_grain();
                    self.spawn_timer -= interval;
                    guard += 1;
                }
            }

            // Render all active grains into a stereo accumulator.
            let wt = self.write_total;
            let mut wet_l = 0.0;
            let mut wet_r = 0.0;
            for g in self.grains.iter_mut() {
                if !g.active {
                    continue;
                }
                let phase = g.age as Sample * g.inv_length;
                let env = 0.5 - 0.5 * ops::cos(TAU * phase);
                let s = read_capture(&self.capture, wt, cap_len, g.pos) * env * g.gain;
                wet_l += s * g.gain_l;
                wet_r += s * g.gain_r;
                g.pos += f64::from(g.pitch);
                g.age += 1;
                if g.age >= g.length {
                    g.active = false;
                }
            }

            let wet = self.wet.next_sample();
            let dry = self.dry.next_sample();
            for ch in 0..channels {
                let dry_in = if ch < in_channels {
                    input.channel(ch)[f]
                } else {
                    0.0
                };
                let wet_c = if channels == 1 {
                    wet_l + wet_r
                } else if ch == 0 {
                    wet_l
                } else if ch == 1 {
                    wet_r
                } else if ch % 2 == 0 {
                    wet_l
                } else {
                    wet_r
                };
                output.channel_mut(ch)[f] = flush_denormal(dry * dry_in + wet * wet_c);
            }
        }
    }

    fn reset(&mut self) {
        for s in &mut self.capture {
            *s = 0.0;
        }
        self.write_total = 0;
        self.spawn_timer = 0.0;
        for g in &mut self.grains {
            *g = Grain::default();
        }
        self.rng = GrainRng::new(self.seed);
        self.wet = Smoothed::new(self.wet.target());
        self.dry = Smoothed::new(self.dry.target());
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

    fn mono(frames: usize) -> AudioBuffer {
        AudioBuffer::new(ChannelLayout::Mono, frames)
    }

    fn stereo(frames: usize) -> AudioBuffer {
        AudioBuffer::new(ChannelLayout::Stereo, frames)
    }

    /// Single-bin Goertzel power estimate at `freq` over `signal`.
    fn goertzel(signal: &[Sample], freq: Sample, sample_rate: Sample) -> Sample {
        let w = TAU * freq / sample_rate;
        let coeff = 2.0 * ops::cos(w);
        let mut s_prev = 0.0;
        let mut s_prev2 = 0.0;
        for &x in signal {
            let s = x + coeff * s_prev - s_prev2;
            s_prev2 = s_prev;
            s_prev = s;
        }
        s_prev * s_prev + s_prev2 * s_prev2 - coeff * s_prev * s_prev2
    }

    fn render(node: &mut GranularNode, input: &AudioBuffer, out_channels: usize) -> AudioBuffer {
        let frames = input.active_frames();
        let layout = if out_channels == 1 {
            ChannelLayout::Mono
        } else {
            ChannelLayout::Stereo
        };
        let inputs = [input.clone()];
        let mut outputs = [AudioBuffer::new(layout, frames)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(frames), &mut io);
        outputs.into_iter().next().unwrap()
    }

    fn sine_mono(frames: usize, freq: Sample, amp: Sample) -> AudioBuffer {
        let mut buf = mono(frames);
        for (i, s) in buf.channel_mut(0).iter_mut().enumerate() {
            *s = amp * ops::sin(TAU * freq * i as Sample / SR as Sample);
        }
        buf
    }

    #[test]
    fn capture_ring_covers_default_window() {
        let node = GranularNode::new(SR, 2, GranularParams::default());
        // At least DEFAULT_CAPTURE_SECONDS of history.
        assert!(node.capture_frames() as Sample >= DEFAULT_CAPTURE_SECONDS * SR as Sample);
        assert_eq!(node.channels(), 2);
        assert!((node.pitch_ratio() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn latency_is_zero() {
        let node = GranularNode::new(SR, 2, GranularParams::default());
        assert_eq!(node.latency_frames(), 0);
    }

    #[test]
    fn silence_in_silence_out() {
        let mut node = GranularNode::new(SR, 2, GranularParams::default());
        let out = render(&mut node, &stereo(2048), 2);
        for ch in 0..2 {
            for &y in out.channel(ch) {
                assert!(y.abs() < 1e-9, "{y}");
            }
        }
    }

    #[test]
    fn zero_frames_is_safe() {
        let mut node = GranularNode::new(SR, 2, GranularParams::default());
        let mut input = stereo(4);
        input.set_active_frames(0);
        let mut output = stereo(4);
        output.set_active_frames(0);
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(0), &mut io);
        assert_eq!(outputs[0].active_frames(), 0);
    }

    #[test]
    fn wet_zero_dry_one_is_passthrough() {
        let params = GranularParams {
            wet: 0.0,
            dry: 1.0,
            ..GranularParams::default()
        };
        let mut node = GranularNode::new(SR, 1, params);
        let input = sine_mono(1024, 440.0, 0.5);
        let out = render(&mut node, &input, 1);
        assert_eq!(out.channel(0), input.channel(0));
    }

    #[test]
    fn wet_only_builds_a_non_silent_cloud() {
        let params = GranularParams {
            wet: 1.0,
            dry: 0.0,
            spread: 0.0,
            ..GranularParams::default()
        };
        let mut node = GranularNode::new(SR, 2, params);
        let input = sine_mono(SR as usize, 440.0, 0.5);
        let out = render(&mut node, &input, 2);
        // Analyze the back half, after the ring has filled and grains sound.
        let half = out.active_frames() / 2;
        let energy: Sample = out.channel(0)[half..].iter().map(|v| v * v).sum();
        assert!(energy > 1.0, "expected an audible cloud, energy = {energy}");
    }

    #[test]
    fn output_is_finite_and_bounded_under_max_density() {
        let params = GranularParams {
            density_hz: MAX_DENSITY_HZ,
            grain_size_ms: 200.0,
            wet: 1.0,
            dry: 1.0,
            spread: 1.0,
            pitch_jitter_semitones: 12.0,
            ..GranularParams::default()
        };
        let mut node = GranularNode::new(SR, 2, params);
        let input = sine_mono(SR as usize, 220.0, 0.9);
        let out = render(&mut node, &input, 2);
        for ch in 0..2 {
            for &y in out.channel(ch) {
                assert!(y.is_finite() && y.abs() < 16.0, "{y}");
            }
        }
    }

    #[test]
    fn pitch_ratio_two_moves_energy_up_an_octave() {
        let base = 500.0;
        let params = GranularParams {
            pitch_ratio: 2.0,
            pitch_jitter_semitones: 0.0,
            density_hz: 80.0,
            grain_size_ms: 120.0,
            position_seconds: 0.3,
            position_jitter_ms: 40.0,
            spread: 0.0,
            wet: 1.0,
            dry: 0.0,
            ..GranularParams::default()
        };
        let mut node = GranularNode::new(SR, 1, params);
        let input = sine_mono(SR as usize, base, 0.6);
        let out = render(&mut node, &input, 1);
        let tail = &out.channel(0)[(SR as usize / 2)..];
        let e_base = goertzel(tail, base, SR as Sample);
        let e_octave = goertzel(tail, base * 2.0, SR as Sample);
        assert!(
            e_octave > e_base * 3.0,
            "octave energy {e_octave} should dominate base energy {e_base}"
        );
    }

    #[test]
    fn same_seed_is_bit_identical() {
        let params = GranularParams {
            spread: 0.8,
            pitch_jitter_semitones: 7.0,
            ..GranularParams::default()
        };
        let input = sine_mono(8192, 330.0, 0.5);
        let mut a = GranularNode::new(SR, 2, params);
        let mut b = GranularNode::new(SR, 2, params);
        let oa = render(&mut a, &input, 2);
        let ob = render(&mut b, &input, 2);
        for ch in 0..2 {
            assert_eq!(oa.channel(ch), ob.channel(ch));
        }
    }

    #[test]
    fn different_seed_decorrelates() {
        let mut params = GranularParams {
            spread: 0.9,
            pitch_jitter_semitones: 9.0,
            ..GranularParams::default()
        };
        let input = sine_mono(SR as usize, 330.0, 0.5);
        params.seed = 1;
        let mut a = GranularNode::new(SR, 2, params);
        params.seed = 777_777;
        let mut b = GranularNode::new(SR, 2, params);
        let oa = render(&mut a, &input, 2);
        let ob = render(&mut b, &input, 2);
        // Compare the back half, after the capture ring has filled with enough
        // recent history for grains to read distinct randomized positions.
        let half = oa.active_frames() / 2;
        let diff: Sample = oa.channel(0)[half..]
            .iter()
            .zip(ob.channel(0)[half..].iter())
            .map(|(x, y)| (x - y).abs())
            .sum();
        assert!(diff > 1e-3, "distinct seeds should produce distinct clouds");
    }

    #[test]
    fn reset_restores_deterministic_output() {
        let mut node = GranularNode::new(SR, 2, GranularParams::default());
        let input = sine_mono(8192, 400.0, 0.5);
        let first = render(&mut node, &input, 2);
        node.reset();
        let second = render(&mut node, &input, 2);
        for ch in 0..2 {
            assert_eq!(first.channel(ch), second.channel(ch));
        }
    }

    #[test]
    fn active_grains_stay_bounded() {
        let params = GranularParams {
            density_hz: MAX_DENSITY_HZ,
            grain_size_ms: MAX_GRAIN_MS,
            ..GranularParams::default()
        };
        let mut node = GranularNode::new(SR, 2, params);
        let input = sine_mono(SR as usize, 220.0, 0.5);
        let _ = render(&mut node, &input, 2);
        let active = node.active_grains();
        assert!(active > 0 && active <= MAX_GRAINS, "active = {active}");
    }

    #[test]
    fn stereo_spread_decorrelates_channels() {
        let params = GranularParams {
            spread: 1.0,
            wet: 1.0,
            dry: 0.0,
            ..GranularParams::default()
        };
        let mut node = GranularNode::new(SR, 2, params);
        let input = sine_mono(SR as usize, 440.0, 0.6);
        let out = render(&mut node, &input, 2);
        let half = out.active_frames() / 2;
        let diff: Sample = out.channel(0)[half..]
            .iter()
            .zip(out.channel(1)[half..].iter())
            .map(|(l, r)| (l - r).abs())
            .sum();
        assert!(diff > 1e-3, "full spread should decorrelate L/R");
    }

    #[test]
    fn mono_output_sums_both_pan_legs() {
        let params = GranularParams {
            spread: 1.0,
            wet: 1.0,
            dry: 0.0,
            ..GranularParams::default()
        };
        let mut node = GranularNode::new(SR, 1, params);
        let input = sine_mono(SR as usize, 440.0, 0.5);
        let out = render(&mut node, &input, 1);
        let half = out.active_frames() / 2;
        let energy: Sample = out.channel(0)[half..].iter().map(|v| v * v).sum();
        assert!(energy > 1.0, "mono cloud should be audible, energy = {energy}");
    }

    #[test]
    fn sanitised_clamps_and_replaces_non_finite() {
        let bad = GranularParams {
            grain_size_ms: Sample::INFINITY,
            density_hz: -5.0,
            position_seconds: 1_000.0,
            position_jitter_ms: Sample::NAN,
            pitch_ratio: 100.0,
            pitch_jitter_semitones: -3.0,
            spread: 9.0,
            wet: Sample::NEG_INFINITY,
            dry: -1.0,
            seed: 42,
        };
        let s = bad.sanitised();
        assert!(s.grain_size_ms.is_finite());
        assert!(s.density_hz >= MIN_DENSITY_HZ && s.density_hz <= MAX_DENSITY_HZ);
        assert!(s.position_seconds <= DEFAULT_CAPTURE_SECONDS);
        assert!(s.position_jitter_ms.is_finite());
        assert!(s.pitch_ratio <= MAX_PITCH_RATIO);
        assert!(s.pitch_jitter_semitones >= 0.0);
        assert!(s.spread <= MAX_SPREAD);
        assert!(s.wet >= 0.0);
        assert!(s.dry >= 0.0);
        assert_eq!(s.seed, 42);
    }

    #[test]
    fn setters_clamp_and_reject_non_finite() {
        let mut node = GranularNode::new(SR, 2, GranularParams::default());
        node.set_density(1_000.0);
        assert!((node.density() - MAX_DENSITY_HZ).abs() < 1e-3);
        node.set_density(Sample::NAN);
        assert!((node.density() - MIN_DENSITY_HZ).abs() < 1e-3);
        node.set_pitch_ratio(100.0);
        assert!((node.pitch_ratio() - MAX_PITCH_RATIO).abs() < 1e-3);
        node.set_pitch_semitones(12.0);
        assert!((node.pitch_ratio() - 2.0).abs() < 1e-2);
        node.set_pitch_ratio(Sample::INFINITY);
        assert!((node.pitch_ratio() - 1.0).abs() < 1e-6);
    }
}
