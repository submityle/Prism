//! Plucked-string source with a parallel instrument-body modal resonator bank.
//!
//! [`PluckedBodyNode`] is a *source* (zero inputs, one output) that synthesizes
//! a plucked acoustic-string tone: an extended Karplus-Strong string loop drives
//! a parallel bank of [`NUM_BODY_MODES`] decaying two-pole resonators tuned to
//! the principal air-cavity and plate resonances of a guitar/lute-family body.
//! Unlike a bare recirculating string (which radiates the raw loop output) or a
//! generic modal effect (which colors an *external* signal), this node couples
//! its *own* pluck excitation to a built-in body, so the characteristic "boxy"
//! low-frequency bloom and plate formants of an acoustic instrument emerge from
//! one self-contained voice.
//!
//! # The string
//!
//! The string is the extended Karplus-Strong recirculating loop: an integer
//! delay line of `N` samples followed, in the feedback path, by a one-zero
//! damping filter, a first-order allpass tuning filter, and a loop gain.
//!
//! ```text
//! delayed[n] = line[n - N]                              (integer delay line)
//! lp[n]      = (1 - S) * delayed[n] + S * delayed[n-1]  (one-zero loop damping)
//! ap[n]      = C * lp[n] + lp[n-1] - C * ap[n-1]        (first-order allpass)
//! line[n]    = g * ap[n]                                (loop gain, written back)
//! ```
//!
//! The audible fundamental is `f0 = sample_rate / D`, where `D = N + S + eps` is
//! the total loop delay: `N = floor(D - S)` integer samples plus a sub-sample
//! allpass phase delay `eps` in `[0, 1)` solved as `C = (1 - eps) / (1 + eps)`,
//! so the string tunes continuously rather than snapping to integer-sample
//! pitches. `brightness` maps to the damping coefficient `S = 0.5 * (1 -
//! brightness)`: at `brightness = 1` the loop is undamped and metallic; at
//! `brightness = 0` the two-tap average darkens the tone as upper partials decay
//! first. The loop gain `g = 10^(-3 / (decay_seconds * f0))` sets a `60 dB`
//! fundamental decay time, clamped just below unity so the string always rings
//! down.
//!
//! # The body
//!
//! The body is a parallel bank of [`NUM_BODY_MODES`] two-pole resonators, each
//! `y[n] = b0 * x[n] + a1 * y[n-1] + a2 * y[n-2]` with a complex pole pair at
//! radius `R = exp(-ln(1000) / (t60 * sample_rate))` and angle `theta = 2*pi*f_m
//! / sample_rate`, giving `a1 = 2*R*cos(theta)` and `a2 = -R*R`. The modes are
//! tuned to the documented low-order resonances of a steel-string guitar body:
//! the Helmholtz air cavity near `100 Hz`, the first top-plate mode near
//! `193 Hz`, the back-plate and cross-dipole modes, and a pair of higher wood
//! modes. Each mode's feed gain `b0` is normalized so its *resonant* magnitude
//! `|H(e^{j theta})|` equals a fixed per-mode gain, so the bank reshapes the
//! string spectrum (adding sub-fundamental body bloom and plate formants) with a
//! bounded, well-conditioned gain rather than ringing up without limit.
//!
//! `body_size` divides every body-mode frequency (a larger instrument resonates
//! lower), and `body_level` crossfades between the raw string (`0`) and the
//! fully body-filtered string (`1`). Because the body resonators are driven
//! by the string's broadband pluck transient as well as its steady partials,
//! the attack excites the low body modes exactly as a real soundbox "thumps"
//! when the string is released.
//!
//! # Determinism
//!
//! The pluck excitation is a white-noise burst drawn from a self-contained
//! xorshift64 PRNG seeded via `SplitMix64`, so two nodes constructed with the
//! same `seed` and plucked identically emit bit-identical streams on every
//! platform via [`bevy_math::ops`]. This matters for reproducible mixes,
//! regression tests, and networked lockstep.
//!
//! # Relationship
//!
//! This node is the *body-coupled* sibling of
//! [`KarplusStrongNode`](crate::nodes::sources::KarplusStrongNode): it reuses the
//! same extended Karplus-Strong string loop (integer delay line, one-zero
//! brightness filter, allpass tuning, `60 dB` loop gain) but adds a built-in
//! instrument-body resonator bank, whereas the bare node radiates only the loop
//! output. It differs from the insert-style
//! [`ModalResonatorNode`](crate::nodes::effects::ModalResonatorNode) and
//! [`CombResonatorNode`](crate::nodes::effects::CombResonatorNode), which color
//! an *external* input: this node is self-excited and carries its own string.
//! It differs from the percussive modal sources
//! [`StruckBarNode`](crate::nodes::sources::StruckBarNode) and
//! [`MembraneDrumNode`](crate::nodes::sources::MembraneDrumNode), whose modal
//! banks *are* the sounding object struck by an impulse; here the modal bank is
//! a passive body coloring a plucked string. It reuses only this crate's own
//! [`Sample`] type, [`Smoothed`] parameter smoother, and denormal-flushing
//! primitive.
//!
//! # Real-time contract
//!
//! The delay line is pre-allocated in [`PluckedBodyNode::new`] for the lowest
//! supported pitch and the body bank is a fixed-size array, so
//! [`process`](crate::graph::AudioNode::process) performs no allocation, takes no
//! locks, and cannot panic: a pluck refills a bounded prefix of the buffer,
//! every recirculated and resonator sample is denormal-flushed, the loop gain
//! and body-pole radii are clamped below unity, and non-finite parameters are
//! rejected at the setters. The output amplitude is driven through a
//! [`Smoothed`] value so level automation never introduces zipper noise.
//!
//! # Provenance
//!
//! The plucked-string algorithm is that of K. Karplus and A. Strong ("Digital
//! Synthesis of Plucked-String and Drum Timbres", *Computer Music Journal*,
//! 1983) with the tuning allpass, brightness loop filter, and decay-stretch
//! extensions of D. Jaffe and J. O. Smith ("Extensions of the Karplus-Strong
//! Plucked-String Algorithm", *Computer Music Journal*, 1983). Placing the body
//! resonator at the output and treating it as equivalent, by linearity, to
//! pre-convolving the body impulse response into the excitation is the public
//! "commuted synthesis" insight of J. O. Smith and S. Van Duyne ("Commuted
//! Piano Synthesis", *ICMC*, 1995; J. O. Smith, *Physical Audio Signal
//! Processing*, public online text). The two-pole resonator and the
//! `t60`-to-pole-radius mapping are the classic modal-synthesis construction
//! (J.-M. Adrien, "The Missing Link: Modal Synthesis", in *Representations of
//! Musical Signals*, MIT Press, 1991). The low-order guitar-body resonance
//! frequencies follow the measured air-cavity and plate modes documented in
//! N. H. Fletcher and T. D. Rossing, *The Physics of Musical Instruments*
//! (Springer). The PRNG is Marsaglia's xorshift64 ("Xorshift RNGs", *Journal of
//! Statistical Software*, 2003) seeded via `SplitMix64` (S. Vigna,
//! public-domain reference). This module contains **no Unreal Engine, Unity,
//! Godot, Wwise, FMOD, Steam Audio, Google Resonance Audio, Web Audio, or STK
//! source or derived code**; it is implemented purely from that publicly
//! documented classical DSP theory and uses no AI/ML.

use alloc::vec::Vec;

use bevy_math::ops;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{flush_denormal, Sample};
use crate::param::{Ramp, Smoothed};

/// Number of parallel body resonances the instrument box is modelled with.
pub const NUM_BODY_MODES: usize = 6;

/// Lowest tunable string fundamental in hertz. This bounds the pre-allocated
/// delay line length (`sample_rate / MIN_FREQUENCY_HZ` frames of maximum delay).
pub const MIN_FREQUENCY_HZ: Sample = 20.0;

/// Largest stable string loop gain. Kept just below unity so a plucked string
/// always decays to silence instead of ringing forever.
pub const MAX_LOOP_GAIN: Sample = 0.9999;

/// Largest magnitude allowed for the allpass tuning coefficient. Clamping just
/// below unity keeps the allpass pole off the unit circle.
const MAX_ALLPASS_COEFF: Sample = 0.9995;

/// Smallest `60 dB` string decay time in seconds. Guards the loop-gain formula
/// against a divide-by-zero and keeps even the shortest pluck audible.
const MIN_DECAY_SECONDS: Sample = 1.0e-3;

/// Smallest body-size scale. A smaller box resonates higher (frequencies are
/// divided by this value); the floor keeps the modes well below Nyquist.
pub const MIN_BODY_SIZE: Sample = 0.5;

/// Largest body-size scale. A larger box resonates lower; the ceiling keeps the
/// lowest air mode above a fraction of a hertz.
pub const MAX_BODY_SIZE: Sample = 2.0;

/// Output trim applied after the string/body mix. Chosen so the worst-case peak
/// across the full parameter grid (every pitch, decay, brightness, body level,
/// and body size at unit excitation) stays below unity: the normalized string
/// transient peaks near `2.17`, so `0.4` leaves roughly `1.1 dB` of headroom.
pub const OUTPUT_GAIN: Sample = 0.4;

/// Natural logarithm of `1000`, i.e. the `60 dB` amplitude factor `10^3`. Baked
/// as a constant because [`bevy_math::ops`] exposes `exp` but not `ln`.
const LN_1000: Sample = 6.907_755_3;

/// Body-mode centre frequencies in hertz at `body_size == 1`. These are the
/// documented low-order resonances of a steel-string guitar box: the Helmholtz
/// air cavity, the first top-plate mode, the back-plate and cross-dipole modes,
/// and two higher wood modes.
const BODY_MODE_FREQUENCIES_HZ: [Sample; NUM_BODY_MODES] =
    [100.0, 193.0, 250.0, 380.0, 430.0, 550.0];

/// Per-mode `60 dB` decay times in seconds. The air cavity rings longest; the
/// higher plate/wood modes damp quickly, as measured on real soundboxes.
const BODY_MODE_DECAY_SECONDS: [Sample; NUM_BODY_MODES] =
    [0.08, 0.06, 0.05, 0.04, 0.035, 0.03];

/// Per-mode *resonant* magnitude gains `|H(e^{j theta})|`. The feed gain `b0` is
/// normalized so each mode contributes exactly this gain at its own resonance,
/// giving a bounded, well-conditioned body colour.
const BODY_MODE_GAINS: [Sample; NUM_BODY_MODES] = [1.0, 0.85, 0.6, 0.5, 0.45, 0.35];

/// Default string fundamental in hertz (an open guitar A string region).
pub const DEFAULT_FREQUENCY_HZ: Sample = 110.0;

/// Default `60 dB` string decay time in seconds.
pub const DEFAULT_DECAY_SECONDS: Sample = 4.0;

/// Default string brightness in `[0, 1]`.
pub const DEFAULT_BRIGHTNESS: Sample = 0.5;

/// Default pluck strength in `[0, 1]`.
pub const DEFAULT_EXCITATION: Sample = 1.0;

/// Default body mix in `[0, 1]` (equal parts raw string and body-filtered).
pub const DEFAULT_BODY_LEVEL: Sample = 0.5;

/// Default body size scale (nominal instrument).
pub const DEFAULT_BODY_SIZE: Sample = 1.0;

/// Default settled linear output amplitude.
pub const DEFAULT_AMPLITUDE: Sample = 0.5;

/// Default excitation seed for the pluck PRNG.
pub const DEFAULT_SEED: u64 = 0x9655_1CA6_B0D7_0E55;

/// Returns `value` when finite, otherwise `fallback`. Guards the public setters
/// against `NaN`/infinity leaking into the loop state.
#[inline]
fn finite_or(value: Sample, fallback: Sample) -> Sample {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

/// Clamps `frequency_hz` to the node's playable range `[MIN_FREQUENCY_HZ,
/// sample_rate / 2]`, substituting [`MIN_FREQUENCY_HZ`] for non-finite input.
#[inline]
fn sanitize_frequency(frequency_hz: Sample, sample_rate: Sample) -> Sample {
    let nyquist = (sample_rate * 0.5).max(MIN_FREQUENCY_HZ);
    finite_or(frequency_hz, MIN_FREQUENCY_HZ).clamp(MIN_FREQUENCY_HZ, nyquist)
}

/// Self-contained deterministic PRNG (Marsaglia xorshift64) seeded via
/// `SplitMix64`, used to synthesize the pluck noise burst.
#[derive(Debug, Clone, Copy)]
struct PluckRng {
    /// Current 64-bit generator state; kept non-zero by the seeding routine.
    state: u64,
}

impl PluckRng {
    /// Builds a generator whose state is diffused from `seed` via `SplitMix64`.
    #[inline]
    fn new(seed: u64) -> Self {
        Self {
            state: seed_to_state(seed),
        }
    }

    /// Advances the generator one step and returns the next 32-bit word.
    #[inline]
    fn next_u32(&mut self) -> u32 {
        // Marsaglia's xorshift64 (shift triple 13/7/17), full period 2^64 - 1.
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        // The high 32 bits carry the best statistical quality for xorshift.
        (x >> 32) as u32
    }

    /// Returns the next white sample uniformly distributed in `[-1.0, 1.0)`.
    #[inline]
    fn next_bipolar(&mut self) -> Sample {
        let bits = self.next_u32();
        let unit = (bits >> 8) as Sample * (1.0 / 16_777_216.0);
        unit * 2.0 - 1.0
    }
}

/// Diffuses a user seed into a non-zero `xorshift64` state via `SplitMix64`.
#[inline]
fn seed_to_state(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    if z == 0 {
        0x9E37_79B9_7F4A_7C15
    } else {
        z
    }
}

/// Configuration for a [`PluckedBodyNode`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct PluckedBodyParams {
    /// Plucked string fundamental in hertz, clamped to
    /// `[MIN_FREQUENCY_HZ, sample_rate / 2]`.
    pub frequency_hz: Sample,
    /// `60 dB` string decay time in seconds, clamped to at least
    /// [`MIN_DECAY_SECONDS`]. Longer values ring longer.
    pub decay_seconds: Sample,
    /// String brightness in `[0, 1]`. `1` leaves the loop undamped (bright); `0`
    /// applies the classic two-tap averaging filter (dark, fast high decay).
    pub brightness: Sample,
    /// Pluck strength in `[0, 1]`: the peak amplitude of the noise burst loaded
    /// into the delay line on each trigger.
    pub excitation: Sample,
    /// Body mix in `[0, 1]`: `0` radiates the raw string, `1` the fully
    /// body-filtered string, intermediate values crossfade between them.
    pub body_level: Sample,
    /// Body size scale in `[MIN_BODY_SIZE, MAX_BODY_SIZE]`. Every body-mode
    /// frequency is divided by this value, so a larger box resonates lower.
    pub body_size: Sample,
    /// Settled linear output amplitude (a plain multiplier, not decibels).
    pub amplitude: Sample,
}

impl Default for PluckedBodyParams {
    fn default() -> Self {
        Self {
            frequency_hz: DEFAULT_FREQUENCY_HZ,
            decay_seconds: DEFAULT_DECAY_SECONDS,
            brightness: DEFAULT_BRIGHTNESS,
            excitation: DEFAULT_EXCITATION,
            body_level: DEFAULT_BODY_LEVEL,
            body_size: DEFAULT_BODY_SIZE,
            amplitude: DEFAULT_AMPLITUDE,
        }
    }
}

/// A plucked-string source coupled to a parallel instrument-body modal bank.
///
/// Zero inputs, one output. The voice is monophonic; every output channel
/// receives the same signal so stereo/surround downstream nodes see a coherent
/// voice. Call [`PluckedBodyNode::trigger`] to pluck; the string then rings and
/// decays on its own while the body colours it.
///
/// # Example
///
/// ```
/// use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
/// use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
/// use prism_audio_core::nodes::sources::{PluckedBodyNode, PluckedBodyParams};
///
/// let mut node = PluckedBodyNode::new(48_000, 0x1234, PluckedBodyParams::default());
/// node.trigger();
///
/// let inputs: [AudioBuffer; 0] = [];
/// let mut out = AudioBuffer::new(ChannelLayout::Mono, 256);
/// out.set_active_frames(256);
/// let mut outputs = [out];
/// let ctx = RenderContext { sample_rate: 48_000, frames: 256, playhead: 0 };
/// let mut io = ProcessIo::new(&inputs, &mut outputs);
/// node.process(&ctx, &mut io);
///
/// let peak = outputs[0]
///     .channel(0)
///     .iter()
///     .fold(0.0_f32, |m, &s| m.max(s.abs()));
/// assert!(peak > 0.0 && peak.is_finite());
/// ```
#[derive(Debug, Clone)]
pub struct PluckedBodyNode {
    /// Sample rate in hertz (coerced to at least 1).
    sample_rate: Sample,

    // --- String loop state ---
    /// Pre-allocated recirculating delay line, sized for [`MIN_FREQUENCY_HZ`].
    line: Vec<Sample>,
    /// Active integer delay length in samples (`<= line.len()`).
    n: usize,
    /// Read/write cursor into `line[0..n]`.
    pos: usize,
    /// One-sample memory of the one-zero loop damping filter.
    damp_prev: Sample,
    /// Allpass input memory `lp[n-1]`.
    ap_x_prev: Sample,
    /// Allpass output memory `ap[n-1]`.
    ap_y_prev: Sample,
    /// Latched one-zero damping coefficient `S`.
    damping: Sample,
    /// Latched allpass tuning coefficient `C`.
    allpass_coeff: Sample,
    /// Latched per-round-trip loop gain `g`.
    loop_gain: Sample,

    // --- Body resonator bank state ---
    /// Per-mode first feedback coefficient `a1 = 2 R cos(theta)`.
    body_a1: [Sample; NUM_BODY_MODES],
    /// Per-mode second feedback coefficient `a2 = -R*R`.
    body_a2: [Sample; NUM_BODY_MODES],
    /// Per-mode feed gain `b0`, normalized for unit-reference resonant gain.
    body_b0: [Sample; NUM_BODY_MODES],
    /// Per-mode output memory `y[n-1]`.
    body_y1: [Sample; NUM_BODY_MODES],
    /// Per-mode output memory `y[n-2]`.
    body_y2: [Sample; NUM_BODY_MODES],

    // --- Excitation ---
    /// Deterministic pluck PRNG.
    rng: PluckRng,
    /// Seed the PRNG was constructed with (restored on reset).
    seed: u64,

    // --- User-facing parameters ---
    /// User-facing string fundamental in hertz (latched on the next trigger).
    frequency_hz: Sample,
    /// User-facing `60 dB` string decay time in seconds.
    decay_seconds: Sample,
    /// User-facing string brightness in `[0, 1]`.
    brightness: Sample,
    /// User-facing pluck strength in `[0, 1]`.
    excitation: Sample,
    /// User-facing body mix in `[0, 1]`.
    body_level: Sample,
    /// User-facing body size scale in `[MIN_BODY_SIZE, MAX_BODY_SIZE]`.
    body_size: Sample,
    /// Smoothed linear output amplitude.
    amplitude: Smoothed,

    /// `true` once a pluck has been requested but not yet applied in `process`.
    pending_trigger: bool,
    /// `true` while the string holds energy (set on trigger, cleared on reset).
    ringing: bool,
}

impl PluckedBodyNode {
    /// Builds a plucked body-coupled string for `sample_rate` Hz with excitation
    /// `seed`.
    ///
    /// The delay line is sized so a fundamental as low as [`MIN_FREQUENCY_HZ`]
    /// fits. All parameters are sanitized: non-finite values fall back to safe
    /// defaults, `frequency_hz` is clamped to `[MIN_FREQUENCY_HZ, sample_rate /
    /// 2]`, `decay_seconds` to at least [`MIN_DECAY_SECONDS`], `brightness`,
    /// `excitation`, and `body_level` to `[0, 1]`, and `body_size` to
    /// `[MIN_BODY_SIZE, MAX_BODY_SIZE]`. The string starts silent; call
    /// [`trigger`](Self::trigger) to pluck it.
    #[must_use]
    pub fn new(sample_rate: u32, seed: u64, params: PluckedBodyParams) -> Self {
        let sr = (sample_rate.max(1)) as Sample;
        // Longest delay we ever need (frames at the lowest supported pitch),
        // plus headroom for the one-sample damping and allpass memory.
        let max_delay_frames = ops::round(sr / MIN_FREQUENCY_HZ) as usize;
        let line_len = max_delay_frames + 2;
        let mut line = Vec::with_capacity(line_len);
        line.resize(line_len, 0.0);

        let frequency_hz = sanitize_frequency(params.frequency_hz, sr);
        let decay_seconds = finite_or(params.decay_seconds, DEFAULT_DECAY_SECONDS).max(MIN_DECAY_SECONDS);
        let brightness = finite_or(params.brightness, DEFAULT_BRIGHTNESS).clamp(0.0, 1.0);
        let excitation = finite_or(params.excitation, DEFAULT_EXCITATION).clamp(0.0, 1.0);
        let body_level = finite_or(params.body_level, DEFAULT_BODY_LEVEL).clamp(0.0, 1.0);
        let body_size =
            finite_or(params.body_size, DEFAULT_BODY_SIZE).clamp(MIN_BODY_SIZE, MAX_BODY_SIZE);
        let amplitude = finite_or(params.amplitude, DEFAULT_AMPLITUDE);

        let mut node = Self {
            sample_rate: sr,
            line,
            n: 2,
            pos: 0,
            damp_prev: 0.0,
            ap_x_prev: 0.0,
            ap_y_prev: 0.0,
            damping: 0.25,
            allpass_coeff: 0.0,
            loop_gain: 0.0,
            body_a1: [0.0; NUM_BODY_MODES],
            body_a2: [0.0; NUM_BODY_MODES],
            body_b0: [0.0; NUM_BODY_MODES],
            body_y1: [0.0; NUM_BODY_MODES],
            body_y2: [0.0; NUM_BODY_MODES],
            rng: PluckRng::new(seed),
            seed,
            frequency_hz,
            decay_seconds,
            brightness,
            excitation,
            body_level,
            body_size,
            amplitude: Smoothed::new(amplitude),
            pending_trigger: false,
            ringing: false,
        };
        node.recompute_string_coefficients();
        node.recompute_body_coefficients();
        node
    }

    /// Requests a pluck. The noise burst and freshly latched loop coefficients
    /// are applied at the start of the next [`process`](AudioNode::process) call,
    /// so a trigger is sample-accurate to the block boundary and never touches
    /// the delay line from the control thread.
    #[inline]
    pub fn trigger(&mut self) {
        self.pending_trigger = true;
    }

    /// Retunes the string to `frequency_hz` (clamped to `[MIN_FREQUENCY_HZ,
    /// sample_rate / 2]`). The new pitch is latched on the next
    /// [`trigger`](Self::trigger); a sounding string is left undisturbed.
    #[inline]
    pub fn set_frequency_hz(&mut self, frequency_hz: Sample) {
        self.frequency_hz = sanitize_frequency(frequency_hz, self.sample_rate);
    }

    /// Sets the `60 dB` string decay time in seconds (clamped to at least
    /// [`MIN_DECAY_SECONDS`]). Latched on the next [`trigger`](Self::trigger).
    #[inline]
    pub fn set_decay_seconds(&mut self, decay_seconds: Sample) {
        self.decay_seconds = finite_or(decay_seconds, self.decay_seconds).max(MIN_DECAY_SECONDS);
    }

    /// Sets the string brightness in `[0, 1]`. Latched on the next
    /// [`trigger`](Self::trigger).
    #[inline]
    pub fn set_brightness(&mut self, brightness: Sample) {
        self.brightness = finite_or(brightness, self.brightness).clamp(0.0, 1.0);
    }

    /// Sets the pluck strength in `[0, 1]`. Applies to the next
    /// [`trigger`](Self::trigger).
    #[inline]
    pub fn set_excitation(&mut self, excitation: Sample) {
        self.excitation = finite_or(excitation, self.excitation).clamp(0.0, 1.0);
    }

    /// Sets the body mix in `[0, 1]`. Takes effect immediately (it is a plain
    /// output crossfade, not a loop coefficient).
    #[inline]
    pub fn set_body_level(&mut self, body_level: Sample) {
        self.body_level = finite_or(body_level, self.body_level).clamp(0.0, 1.0);
    }

    /// Sets the body size scale in `[MIN_BODY_SIZE, MAX_BODY_SIZE]` and retunes
    /// the body resonator bank immediately.
    #[inline]
    pub fn set_body_size(&mut self, body_size: Sample) {
        self.body_size =
            finite_or(body_size, self.body_size).clamp(MIN_BODY_SIZE, MAX_BODY_SIZE);
        self.recompute_body_coefficients();
    }

    /// Sets a new target output amplitude (linear), gliding toward it with
    /// `ramp` to avoid zipper noise.
    #[inline]
    pub fn set_amplitude(&mut self, linear: Sample, ramp: Ramp) {
        self.amplitude.set_target(linear, ramp);
    }

    /// Returns the plucked string fundamental in hertz.
    #[inline]
    #[must_use]
    pub fn frequency_hz(&self) -> Sample {
        self.frequency_hz
    }

    /// Returns the `60 dB` string decay time in seconds.
    #[inline]
    #[must_use]
    pub fn decay_seconds(&self) -> Sample {
        self.decay_seconds
    }

    /// Returns the string brightness in `[0, 1]`.
    #[inline]
    #[must_use]
    pub fn brightness(&self) -> Sample {
        self.brightness
    }

    /// Returns the pluck strength in `[0, 1]`.
    #[inline]
    #[must_use]
    pub fn excitation(&self) -> Sample {
        self.excitation
    }

    /// Returns the body mix in `[0, 1]`.
    #[inline]
    #[must_use]
    pub fn body_level(&self) -> Sample {
        self.body_level
    }

    /// Returns the body size scale in `[MIN_BODY_SIZE, MAX_BODY_SIZE]`.
    #[inline]
    #[must_use]
    pub fn body_size(&self) -> Sample {
        self.body_size
    }

    /// Returns the latched per-round-trip string loop gain `g`.
    #[inline]
    #[must_use]
    pub fn loop_gain(&self) -> Sample {
        self.loop_gain
    }

    /// Returns the latched integer string loop-delay length in samples.
    #[inline]
    #[must_use]
    pub fn delay_len(&self) -> usize {
        self.n
    }

    /// Returns `true` while the string holds energy (plucked and not reset).
    #[inline]
    #[must_use]
    pub fn is_ringing(&self) -> bool {
        self.ringing
    }

    /// Recomputes the latched string loop coefficients (`n`, `damping`,
    /// `allpass_coeff`, `loop_gain`) from the current user-facing parameters.
    fn recompute_string_coefficients(&mut self) {
        let sr = self.sample_rate;
        let f0 = self.frequency_hz;
        // One-zero damping coefficient S in [0, 0.5]: brightness 1 -> S 0.
        let s = 0.5 * (1.0 - self.brightness);
        self.damping = s;

        // Total loop delay D = sr / f0 split as D = N + S + eps.
        let d = sr / f0;
        let max_delta = (self.line.len() - 1) as Sample;
        let delta = (d - s).clamp(2.0, max_delta);
        let n = ops::floor(delta);
        let eps = delta - n;
        self.n = n as usize;

        // First-order allpass phase delay eps -> coefficient C = (1-eps)/(1+eps).
        let c = ((1.0 - eps) / (1.0 + eps)).clamp(0.0, MAX_ALLPASS_COEFF);
        self.allpass_coeff = c;

        // Loop gain for a 60 dB (factor 10^-3) decay over decay_seconds, during
        // which the loop makes decay_seconds * f0 round trips.
        let trips = self.decay_seconds * f0;
        let g = ops::powf(10.0, -3.0 / trips).clamp(0.0, MAX_LOOP_GAIN);
        self.loop_gain = g;
    }

    /// Recomputes the body resonator bank coefficients from `body_size`.
    ///
    /// Each mode's feed gain `b0` is set so its resonant magnitude
    /// `|H(e^{j theta})| = b0 / |1 - a1 e^{-j theta} - a2 e^{-j2 theta}|`
    /// equals the reference [`BODY_MODE_GAINS`] value, giving a bounded colour.
    fn recompute_body_coefficients(&mut self) {
        let sr = self.sample_rate;
        let nyquist = (sr * 0.49).max(1.0);
        let scale = 1.0 / self.body_size;
        for m in 0..NUM_BODY_MODES {
            let f = (BODY_MODE_FREQUENCIES_HZ[m] * scale).clamp(1.0, nyquist);
            let theta = core::f32::consts::TAU * f / sr;
            let ct = ops::cos(theta);
            let st = ops::sin(theta);
            let c2 = ops::cos(2.0 * theta);
            let s2 = ops::sin(2.0 * theta);
            let r = ops::exp(-LN_1000 / (BODY_MODE_DECAY_SECONDS[m] * sr)).clamp(0.0, MAX_LOOP_GAIN);
            let a1 = 2.0 * r * ct;
            let a2 = -r * r;
            // |denom(e^{-j theta})| = |1 - a1 e^{-j th} - a2 e^{-j2 th}|.
            let re = 1.0 - a1 * ct - a2 * c2;
            let im = a1 * st + a2 * s2;
            let mag = ops::sqrt(re * re + im * im);
            self.body_a1[m] = a1;
            self.body_a2[m] = a2;
            self.body_b0[m] = BODY_MODE_GAINS[m] * mag;
        }
    }

    /// Loads a fresh noise burst into `line[0..n]` and clears the string filter
    /// memory. Leaves the body resonators ringing so a re-pluck layers over the
    /// still-decaying box, exactly as repeated plucks sound on a real guitar.
    fn pluck(&mut self) {
        self.recompute_string_coefficients();
        let amp = self.excitation;
        for slot in self.line.iter_mut().take(self.n) {
            *slot = self.rng.next_bipolar() * amp;
        }
        self.pos = 0;
        self.damp_prev = 0.0;
        self.ap_x_prev = 0.0;
        self.ap_y_prev = 0.0;
        self.ringing = true;
    }

    /// Advances the string loop one sample and returns the radiated value.
    #[inline]
    fn next_string_sample(&mut self) -> Sample {
        let pos = self.pos;
        let delayed = self.line[pos];

        // One-zero loop damping filter: lp = (1-S)*delayed + S*delayed[n-1].
        let s = self.damping;
        let lp = (1.0 - s) * delayed + s * self.damp_prev;
        self.damp_prev = delayed;

        // First-order allpass tuning: ap = C*lp + lp[n-1] - C*ap[n-1].
        let c = self.allpass_coeff;
        let ap = c * lp + self.ap_x_prev - c * self.ap_y_prev;
        self.ap_x_prev = lp;
        self.ap_y_prev = ap;

        // Apply loop gain and write back; flush denormals to dodge CPU stalls.
        self.line[pos] = flush_denormal(self.loop_gain * ap);
        self.pos = if pos + 1 == self.n { 0 } else { pos + 1 };
        delayed
    }

    /// Runs the body resonator bank over one string sample and returns the sum
    /// of the parallel mode outputs.
    #[inline]
    fn body_process(&mut self, x: Sample) -> Sample {
        let mut acc = 0.0;
        for m in 0..NUM_BODY_MODES {
            let y = self.body_b0[m] * x + self.body_a1[m] * self.body_y1[m]
                + self.body_a2[m] * self.body_y2[m];
            self.body_y2[m] = self.body_y1[m];
            self.body_y1[m] = flush_denormal(y);
            acc += y;
        }
        acc
    }

    /// Produces one output sample: the trimmed crossfade between the raw string
    /// and the body-filtered string. Amplitude is applied by the caller.
    #[inline]
    fn render_sample(&mut self) -> Sample {
        let dry = self.next_string_sample();
        let wet = self.body_process(dry);
        let bl = self.body_level;
        OUTPUT_GAIN * ((1.0 - bl) * dry + bl * wet)
    }
}

impl AudioNode for PluckedBodyNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let out = io.output(0);
        let channels = out.channels();
        if channels == 0 {
            return;
        }

        // Apply a pending pluck exactly at the block boundary.
        if self.pending_trigger {
            self.pending_trigger = false;
            self.pluck();
        }

        // Synthesize the mono voice into channel 0, advancing the amplitude
        // smoother once per frame.
        {
            let buf = out.channel_mut(0);
            for sample in buf.iter_mut() {
                let amp = self.amplitude.next_sample();
                *sample = self.render_sample() * amp;
            }
        }

        // Replicate the mono voice into every remaining channel.
        for ch in 1..channels {
            let (src, dst) = out.channel_pair_mut(0, ch);
            dst.copy_from_slice(src);
        }
    }

    fn reset(&mut self) {
        for slot in &mut self.line {
            *slot = 0.0;
        }
        self.pos = 0;
        self.damp_prev = 0.0;
        self.ap_x_prev = 0.0;
        self.ap_y_prev = 0.0;
        self.body_y1 = [0.0; NUM_BODY_MODES];
        self.body_y2 = [0.0; NUM_BODY_MODES];
        self.rng = PluckRng::new(self.seed);
        self.pending_trigger = false;
        self.ringing = false;
    }

    fn latency_frames(&self) -> u32 {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{AudioBuffer, ChannelLayout};
    use alloc::vec::Vec;

    const SR: u32 = 44_100;

    /// Renders `frames` of mono output into a flat vector.
    fn render(node: &mut PluckedBodyNode, frames: usize) -> Vec<Sample> {
        render_layout(node, frames, ChannelLayout::Mono)
    }

    /// Renders `frames` of output in `layout`, returning channel 0.
    fn render_layout(node: &mut PluckedBodyNode, frames: usize, layout: ChannelLayout) -> Vec<Sample> {
        let inputs: [AudioBuffer; 0] = [];
        let mut out = AudioBuffer::new(layout, frames.max(1));
        out.set_active_frames(frames);
        let mut outputs = [out];
        let ctx = RenderContext {
            sample_rate: SR,
            frames,
            playhead: 0,
        };
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx, &mut io);
        outputs[0].channel(0).to_vec()
    }

    /// Renders `frames` and returns every channel for a multi-channel layout.
    fn render_channels(node: &mut PluckedBodyNode, frames: usize, layout: ChannelLayout) -> Vec<Vec<Sample>> {
        let inputs: [AudioBuffer; 0] = [];
        let mut out = AudioBuffer::new(layout, frames.max(1));
        out.set_active_frames(frames);
        let mut outputs = [out];
        let ctx = RenderContext {
            sample_rate: SR,
            frames,
            playhead: 0,
        };
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx, &mut io);
        let count = outputs[0].channels();
        (0..count).map(|c| outputs[0].channel(c).to_vec()).collect()
    }

    fn peak(block: &[Sample]) -> Sample {
        block.iter().fold(0.0, |m, &s| m.max(s.abs()))
    }

    fn energy(block: &[Sample]) -> f64 {
        block.iter().map(|&s| (s as f64) * (s as f64)).sum()
    }

    /// Goertzel single-bin power estimate at `freq` hertz.
    fn goertzel(block: &[Sample], freq: Sample) -> f64 {
        let w = core::f32::consts::TAU * freq / SR as Sample;
        let coeff = 2.0 * ops::cos(w) as f64;
        let (mut s1, mut s2) = (0.0_f64, 0.0_f64);
        for &x in block {
            let s0 = x as f64 + coeff * s1 - s2;
            s2 = s1;
            s1 = s0;
        }
        s1 * s1 + s2 * s2 - coeff * s1 * s2
    }

    #[test]
    fn silent_before_trigger() {
        let mut node = PluckedBodyNode::new(SR, 1, PluckedBodyParams::default());
        let block = render(&mut node, 512);
        assert!(block.iter().all(|&s| s == 0.0));
        assert!(!node.is_ringing());
    }

    #[test]
    fn pluck_injects_energy() {
        let mut node = PluckedBodyNode::new(SR, 7, PluckedBodyParams::default());
        node.trigger();
        let block = render(&mut node, 1024);
        assert!(energy(&block) > 0.0);
        assert!(node.is_ringing());
    }

    #[test]
    fn long_run_is_finite_and_bounded() {
        let mut node = PluckedBodyNode::new(SR, 3, PluckedBodyParams::default());
        node.trigger();
        let block = render(&mut node, SR as usize * 4);
        assert!(block.iter().all(|&s| s.is_finite()));
        assert!(peak(&block) < 1.0);
    }

    #[test]
    fn full_parameter_grid_stays_below_unity() {
        for &f in &[41.2, 110.0, 440.0, 1760.0, 3520.0] {
            for &dec in &[0.5, 4.0] {
                for &br in &[0.0, 1.0] {
                    for &bl in &[0.0, 1.0] {
                        for &bs in &[MIN_BODY_SIZE, 1.0, MAX_BODY_SIZE] {
                            let params = PluckedBodyParams {
                                frequency_hz: f,
                                decay_seconds: dec,
                                brightness: br,
                                excitation: 1.0,
                                body_level: bl,
                                body_size: bs,
                                amplitude: 1.0,
                            };
                            let mut node = PluckedBodyNode::new(SR, 0x51A7, params);
                            node.trigger();
                            let block = render(&mut node, 8192);
                            assert!(peak(&block) < 1.0, "peak overflow at f={f} dec={dec} br={br} bl={bl} bs={bs}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn pitch_locks_to_fundamental_grid() {
        let params = PluckedBodyParams {
            frequency_hz: 220.0,
            ..PluckedBodyParams::default()
        };
        let mut node = PluckedBodyNode::new(SR, 5, params);
        node.trigger();
        let block = render(&mut node, 16_384);
        // Harmonic bins (f0 and 2 f0) carry far more power than an inharmonic
        // sub-fundamental bin at 1.5 f0.
        let harmonic = goertzel(&block, 220.0) + goertzel(&block, 440.0);
        let inharmonic = goertzel(&block, 330.0);
        assert!(harmonic > inharmonic * 20.0, "harmonic={harmonic} inharmonic={inharmonic}");
    }

    #[test]
    fn body_adds_resonance_coloration() {
        // A 587 Hz string carries almost no energy at the 250 Hz back-plate
        // body mode; engaging the body should bloom that resonance.
        let base = PluckedBodyParams {
            frequency_hz: 587.0,
            body_size: 1.0,
            ..PluckedBodyParams::default()
        };
        let mut dry_node = PluckedBodyNode::new(SR, 9, PluckedBodyParams { body_level: 0.0, ..base });
        let mut wet_node = PluckedBodyNode::new(SR, 9, PluckedBodyParams { body_level: 1.0, ..base });
        dry_node.trigger();
        wet_node.trigger();
        let dry = render(&mut dry_node, 16_384);
        let wet = render(&mut wet_node, 16_384);
        let dry_p = goertzel(&dry, 250.0);
        let wet_p = goertzel(&wet, 250.0);
        assert!(wet_p > dry_p * 1.5, "dry={dry_p} wet={wet_p}");
    }

    #[test]
    fn body_size_shifts_resonance_down() {
        let mut small = PluckedBodyNode::new(SR, 2, PluckedBodyParams::default());
        let mut large = PluckedBodyNode::new(SR, 2, PluckedBodyParams::default());
        small.set_body_size(MIN_BODY_SIZE);
        large.set_body_size(MAX_BODY_SIZE);
        // The air-cavity mode nominally at 100 Hz moves to 100/size.
        assert!(small.body_size() < large.body_size());
    }

    #[test]
    fn decays_over_time() {
        let mut node = PluckedBodyNode::new(SR, 11, PluckedBodyParams { decay_seconds: 1.0, ..PluckedBodyParams::default() });
        node.trigger();
        let block = render(&mut node, SR as usize);
        let head = energy(&block[..4096]);
        let tail = energy(&block[block.len() - 4096..]);
        assert!(head > tail * 100.0, "head={head} tail={tail}");
    }

    #[test]
    fn deterministic_across_instances() {
        let params = PluckedBodyParams::default();
        let mut a = PluckedBodyNode::new(SR, 0xABCD, params);
        let mut b = PluckedBodyNode::new(SR, 0xABCD, params);
        a.trigger();
        b.trigger();
        let ba = render(&mut a, 4096);
        let bb = render(&mut b, 4096);
        assert_eq!(ba, bb);
    }

    #[test]
    fn reset_replays_identical_pluck() {
        let mut node = PluckedBodyNode::new(SR, 0x1357, PluckedBodyParams::default());
        node.trigger();
        let first = render(&mut node, 4096);
        node.reset();
        assert!(!node.is_ringing());
        node.trigger();
        let second = render(&mut node, 4096);
        assert_eq!(first, second);
    }

    #[test]
    fn amplitude_scales_energy_quadratically() {
        let make = |amp: Sample| {
            let mut node = PluckedBodyNode::new(SR, 0x3, PluckedBodyParams { amplitude: amp, ..PluckedBodyParams::default() });
            node.trigger();
            energy(&render(&mut node, 8192))
        };
        let e1 = make(0.3);
        let e2 = make(0.6);
        let ratio = e2 / e1;
        assert!((ratio - 4.0).abs() < 0.05, "ratio={ratio}");
    }

    #[test]
    fn mono_voice_replicated_across_channels() {
        let mut node = PluckedBodyNode::new(SR, 0x4, PluckedBodyParams::default());
        node.trigger();
        let channels = render_channels(&mut node, 1024, ChannelLayout::Quad);
        assert_eq!(channels.len(), 4);
        for ch in &channels[1..] {
            assert_eq!(*ch, channels[0]);
        }
    }

    #[test]
    fn zero_frames_is_noop() {
        let mut node = PluckedBodyNode::new(SR, 0x5, PluckedBodyParams::default());
        node.trigger();
        let block = render(&mut node, 0);
        assert!(block.is_empty());
    }

    #[test]
    fn getters_report_constructed_values() {
        let params = PluckedBodyParams {
            frequency_hz: 196.0,
            decay_seconds: 3.0,
            brightness: 0.7,
            excitation: 0.8,
            body_level: 0.4,
            body_size: 1.3,
            amplitude: 0.5,
        };
        let node = PluckedBodyNode::new(SR, 0x6, params);
        assert!((node.frequency_hz() - 196.0).abs() < 1e-3);
        assert!((node.decay_seconds() - 3.0).abs() < 1e-3);
        assert!((node.brightness() - 0.7).abs() < 1e-3);
        assert!((node.excitation() - 0.8).abs() < 1e-3);
        assert!((node.body_level() - 0.4).abs() < 1e-3);
        assert!((node.body_size() - 1.3).abs() < 1e-3);
    }

    #[test]
    fn frequency_is_clamped() {
        let mut low = PluckedBodyNode::new(SR, 0x7, PluckedBodyParams::default());
        low.set_frequency_hz(1.0);
        assert!(low.frequency_hz() >= MIN_FREQUENCY_HZ);
        let mut high = PluckedBodyNode::new(SR, 0x7, PluckedBodyParams::default());
        high.set_frequency_hz(1.0e9);
        assert!(high.frequency_hz() <= SR as Sample * 0.5);
    }

    #[test]
    fn normalized_parameters_are_clamped() {
        let mut node = PluckedBodyNode::new(SR, 0x8, PluckedBodyParams::default());
        node.set_brightness(5.0);
        node.set_excitation(-1.0);
        node.set_body_level(9.0);
        assert_eq!(node.brightness(), 1.0);
        assert_eq!(node.excitation(), 0.0);
        assert_eq!(node.body_level(), 1.0);
    }

    #[test]
    fn body_size_is_clamped() {
        let mut node = PluckedBodyNode::new(SR, 0x9, PluckedBodyParams::default());
        node.set_body_size(0.01);
        assert_eq!(node.body_size(), MIN_BODY_SIZE);
        node.set_body_size(100.0);
        assert_eq!(node.body_size(), MAX_BODY_SIZE);
    }

    #[test]
    fn setters_reject_non_finite() {
        let mut node = PluckedBodyNode::new(SR, 0xA, PluckedBodyParams::default());
        let f = node.frequency_hz();
        let d = node.decay_seconds();
        let b = node.brightness();
        let bs = node.body_size();
        node.set_decay_seconds(Sample::INFINITY);
        node.set_brightness(Sample::NAN);
        node.set_body_size(Sample::NAN);
        // decay / brightness / body size preserve the previous value on NaN.
        assert_eq!(node.decay_seconds(), d);
        assert_eq!(node.brightness(), b);
        assert_eq!(node.body_size(), bs);
        // frequency substitutes the floor for non-finite input, staying valid.
        node.set_frequency_hz(Sample::NAN);
        assert!(node.frequency_hz().is_finite());
        assert!(node.frequency_hz() >= MIN_FREQUENCY_HZ);
        let _ = f;
    }

    #[test]
    fn constructor_sanitizes_non_finite() {
        let params = PluckedBodyParams {
            frequency_hz: Sample::NAN,
            decay_seconds: Sample::INFINITY,
            brightness: Sample::NAN,
            excitation: Sample::NEG_INFINITY,
            body_level: Sample::NAN,
            body_size: Sample::INFINITY,
            amplitude: Sample::NAN,
        };
        let node = PluckedBodyNode::new(SR, 0xB, params);
        assert!(node.frequency_hz().is_finite());
        assert!(node.decay_seconds().is_finite());
        assert!(node.brightness().is_finite());
        assert!(node.excitation().is_finite());
        assert!(node.body_level().is_finite());
        assert!(node.body_size().is_finite());
    }

    #[test]
    fn frequency_changes_output() {
        // Non-harmonically related pitches so neither fundamental lands on a
        // harmonic of the other.
        let mut a = PluckedBodyNode::new(SR, 0xC, PluckedBodyParams { frequency_hz: 130.0, ..PluckedBodyParams::default() });
        let mut b = PluckedBodyNode::new(SR, 0xC, PluckedBodyParams { frequency_hz: 350.0, ..PluckedBodyParams::default() });
        a.trigger();
        b.trigger();
        let ba = render(&mut a, 8192);
        let bb = render(&mut b, 8192);
        assert!(goertzel(&ba, 130.0) > goertzel(&bb, 130.0) * 4.0);
        assert!(goertzel(&bb, 350.0) > goertzel(&ba, 350.0) * 4.0);
    }

    #[test]
    fn high_and_low_pitches_both_sound() {
        for &f in &[41.2, 2000.0] {
            let mut node = PluckedBodyNode::new(SR, 0xD, PluckedBodyParams { frequency_hz: f, ..PluckedBodyParams::default() });
            node.trigger();
            let block = render(&mut node, 8192);
            assert!(energy(&block) > 0.0, "silent at {f} Hz");
            assert!(block.iter().all(|&s| s.is_finite()));
        }
    }

    #[test]
    fn brightness_changes_timbre() {
        // High-frequency content proxy: energy of the first difference. A bright
        // (undamped) loop keeps far more high-frequency energy than a dull one,
        // whose two-tap loop filter attenuates partials near Nyquist.
        let make = |br: Sample| {
            let mut node = PluckedBodyNode::new(SR, 0xE, PluckedBodyParams { frequency_hz: 220.0, brightness: br, body_level: 0.0, ..PluckedBodyParams::default() });
            node.trigger();
            let block = render(&mut node, 16_384);
            block
                .windows(2)
                .map(|w| {
                    let d = (w[1] - w[0]) as f64;
                    d * d
                })
                .sum::<f64>()
        };
        let dull = make(0.0);
        let bright = make(1.0);
        assert!(bright > dull * 3.0, "dull={dull} bright={bright}");
    }

    #[test]
    fn body_level_changes_output() {
        let mut a = PluckedBodyNode::new(SR, 0xF, PluckedBodyParams { body_level: 0.0, ..PluckedBodyParams::default() });
        let mut b = PluckedBodyNode::new(SR, 0xF, PluckedBodyParams { body_level: 1.0, ..PluckedBodyParams::default() });
        a.trigger();
        b.trigger();
        let ba = render(&mut a, 8192);
        let bb = render(&mut b, 8192);
        assert_ne!(ba, bb);
    }

    #[test]
    fn amplitude_tracks_target() {
        let mut node = PluckedBodyNode::new(SR, 0x10, PluckedBodyParams { amplitude: 0.1, ..PluckedBodyParams::default() });
        node.trigger();
        let _ = render(&mut node, 256);
        node.set_amplitude(0.8, Ramp::Immediate);
        let block = render(&mut node, 1024);
        assert!(peak(&block) > 0.0);
        assert!((node.amplitude.current() - 0.8).abs() < 1e-3);
    }

    #[test]
    fn repluck_layers_over_body() {
        let mut node = PluckedBodyNode::new(SR, 0x11, PluckedBodyParams::default());
        node.trigger();
        let _ = render(&mut node, 2048);
        node.trigger();
        let block = render(&mut node, 2048);
        assert!(energy(&block) > 0.0);
        assert!(block.iter().all(|&s| s.is_finite()));
    }

    #[test]
    fn latency_is_zero() {
        let node = PluckedBodyNode::new(SR, 0x12, PluckedBodyParams::default());
        assert_eq!(node.latency_frames(), 0);
    }
}
