//! Frequency-dependent atmospheric absorption, modelled as a distance-driven
//! low-pass filter.
//!
//! As sound travels through air, high frequencies are absorbed more strongly
//! than low frequencies, so a distant source sounds progressively duller. This
//! module computes the physically correct absorption coefficient from the
//! **ISO 9613-1:1993** standard (including the oxygen and nitrogen molecular
//! relaxation frequencies) and maps it onto a distance-dependent low-pass
//! cut-off, which is realised with the shared RBJ biquad from
//! [`prism_audio_core`].
//!
//! # Model
//!
//! [`absorption_db_per_metre`] evaluates the ISO 9613-1 pure-tone attenuation
//! coefficient `alpha` (in dB per metre) for a given atmosphere and frequency.
//! [`AirAbsorption::cutoff_hz`] then scans a fixed logarithmic frequency grid
//! and picks the lowest frequency whose accumulated absorption over the
//! propagation distance reaches a perceptual threshold; that frequency becomes
//! the low-pass corner. Greater distances push the corner lower, matching the
//! ear's experience of far-away sound.
//!
//! # Real-time contract
//!
//! [`AirAbsorptionNode::process`] is **allocation free, lock free, and panic
//! free**: it copies the input to the output and runs one biquad pass. All
//! coefficient (re)design happens in the non-real-time setters
//! ([`AirAbsorptionNode::set_distance`], [`AirAbsorptionNode::set_conditions`]).
//!
//! # Determinism
//!
//! Every transcendental (`exp`, `powf`, `sqrt`) routes through
//! [`bevy_math::ops`] (libm-backed) rather than an `f32` intrinsic, so the
//! absorption coefficient and the derived cut-off are bit-reproducible across
//! targets. The coefficient is evaluated in `f32`: the crate stays `no_std`
//! compatible and has no `f64`/libm dependency available, and the `f32` result
//! is comfortably accurate for the perceptual low-pass mapping.
//!
//! # Provenance
//!
//! The absorption formula is the publicly documented **ISO 9613-1:1993**
//! ("Acoustics -- Attenuation of sound during propagation outdoors -- Part 1:
//! Calculation of the absorption of sound by the atmosphere") pure-tone
//! coefficient, as reproduced in standard acoustics references (e.g. Bass et
//! al., "Atmospheric absorption of sound: Further developments", J. Acoust.
//! Soc. Am. 97(1), 1995). The low-pass itself uses the Robert Bristow-Johnson
//! audio-EQ cookbook biquad already implemented in [`prism_audio_core`]. This
//! file contains **no Unreal Engine, Unity, Godot, Wwise, or FMOD source or
//! derived code**; it is implemented purely from that publicly documented
//! acoustics and signal-processing knowledge.

use bevy_math::ops;

use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
use prism_audio_core::math::Sample;
use prism_audio_core::nodes::biquad::{Biquad, BiquadCoeffs, BiquadKind};

/// Reference static pressure of one standard atmosphere, in kilopascals.
const REFERENCE_PRESSURE_KPA: Sample = 101.325;
/// Reference air temperature (20 degrees Celsius), in kelvin.
const REFERENCE_TEMPERATURE_K: Sample = 293.15;
/// Celsius-to-kelvin additive offset.
const CELSIUS_TO_KELVIN: Sample = 273.15;
/// Triple-point temperature of water, in kelvin (used by the saturation
/// vapour-pressure fit).
const WATER_TRIPLE_POINT_K: Sample = 273.16;
/// Smallest strictly-positive value used to keep divisors and `powf` bases
/// away from zero without ever panicking.
const EPSILON: Sample = 1.0e-6;
/// Quality factor for the low-pass corner (Butterworth, maximally flat).
const LOWPASS_Q: Sample = core::f32::consts::FRAC_1_SQRT_2;
/// Absolute lower bound (Hz) for any derived cut-off frequency.
const MIN_CUTOFF_HZ: Sample = 20.0;

/// Fixed logarithmic frequency grid (Hz) scanned by
/// [`AirAbsorption::cutoff_hz`].
///
/// 32 points spanning roughly 40 Hz to 24 kHz. Storing the grid as a `const`
/// keeps the cut-off search entirely on the stack (no heap allocation).
const FREQUENCY_GRID: [Sample; 32] = [
    40.0, 49.1675, 60.436, 74.2871, 91.3127, 112.2403, 137.9643, 169.5838, 208.4502, 256.2241,
    314.9472, 387.1289, 475.8536, 584.9128, 718.967, 883.7445, 1086.2867, 1335.2491, 1641.2701,
    2017.4271, 2479.7942, 3048.1294, 3746.7197, 4605.4175, 5660.9165, 6958.3223, 8553.076,
    10513.325, 12922.838, 15884.578, 19525.11, 24000.0,
];

/// The atmosphere a sound propagates through.
///
/// This is plain, authoring-time description data (not a real-time node): cheap
/// to copy and, with the `serialize` feature, (de)serialisable. It feeds
/// [`absorption_db_per_metre`] and [`AirAbsorption`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct AtmosphericConditions {
    /// Air temperature in degrees Celsius.
    pub temperature_c: Sample,
    /// Relative humidity as a percentage in `[0, 100]`.
    pub relative_humidity_percent: Sample,
    /// Static air pressure in kilopascals.
    pub pressure_kpa: Sample,
}

impl Default for AtmosphericConditions {
    /// Standard reference atmosphere: 20 degrees Celsius, 50 % relative
    /// humidity, one standard atmosphere (101.325 kPa).
    #[inline]
    fn default() -> Self {
        Self {
            temperature_c: 20.0,
            relative_humidity_percent: 50.0,
            pressure_kpa: REFERENCE_PRESSURE_KPA,
        }
    }
}

impl AtmosphericConditions {
    /// Creates a set of atmospheric conditions from raw measurements.
    #[inline]
    #[must_use]
    pub fn new(
        temperature_c: Sample,
        relative_humidity_percent: Sample,
        pressure_kpa: Sample,
    ) -> Self {
        Self {
            temperature_c,
            relative_humidity_percent,
            pressure_kpa,
        }
    }
}

/// Computes the ISO 9613-1:1993 pure-tone atmospheric absorption coefficient
/// `alpha`, in **decibels per metre**, for `freq_hz` in the given atmosphere.
///
/// The coefficient increases with frequency (high frequencies are absorbed
/// faster) and depends on temperature, humidity, and pressure through the
/// oxygen and nitrogen molecular relaxation frequencies.
///
/// Inputs are sanitised so the evaluation can never divide by zero, take the
/// `powf` of a non-positive base, or panic: the frequency, pressure ratio, and
/// absolute temperature are each floored to a small positive value, and
/// relative humidity is clamped to be non-negative.
///
/// # Examples
///
/// ```
/// use prism_audio_spatial::air::{absorption_db_per_metre, AtmosphericConditions};
///
/// let air = AtmosphericConditions::default();
/// let low = absorption_db_per_metre(&air, 1_000.0);
/// let high = absorption_db_per_metre(&air, 8_000.0);
/// assert!(high > low);
/// ```
#[must_use]
pub fn absorption_db_per_metre(conditions: &AtmosphericConditions, freq_hz: Sample) -> Sample {
    // Sanitised, strictly positive working values.
    let f = freq_hz.max(EPSILON);
    let f2 = f * f;
    let pr = (conditions.pressure_kpa / REFERENCE_PRESSURE_KPA).max(EPSILON);
    let temperature_k = (conditions.temperature_c + CELSIUS_TO_KELVIN).max(EPSILON);
    // Relative temperature (ratio to the 293.15 K reference).
    let tr = (temperature_k / REFERENCE_TEMPERATURE_K).max(EPSILON);
    let humidity = conditions.relative_humidity_percent.max(0.0);

    // Molar concentration of water vapour, `h` (percent), from the ISO 9613-1
    // saturation-vapour-pressure fit:
    //   C = -6.8346 * (273.16 / T)^1.261 + 4.6151
    //   psat / pr = 10^C
    let saturation_exponent =
        -6.8346 * ops::powf(WATER_TRIPLE_POINT_K / temperature_k, 1.261) + 4.6151;
    let saturation_ratio = ops::exp(saturation_exponent * core::f32::consts::LN_10);
    let humidity_molar = humidity * saturation_ratio / pr;

    // Oxygen relaxation frequency (Hz).
    let fr_o =
        pr * (24.0 + 4.04e4 * humidity_molar * (0.02 + humidity_molar) / (0.391 + humidity_molar));
    // Nitrogen relaxation frequency (Hz).
    let tr_inv_cbrt = ops::powf(tr, -1.0 / 3.0);
    let fr_n = pr
        * ops::powf(tr, -0.5)
        * (9.0 + 280.0 * humidity_molar * ops::exp(-4.170 * (tr_inv_cbrt - 1.0)));

    // Both relaxation frequencies are strictly positive (pr > 0, and the
    // parenthesised terms are >= 24 and >= 9 respectively), so the divisors
    // `fr + f^2 / fr` below can never be zero.
    let oxygen_term = 0.01275 * ops::exp(-2239.1 / temperature_k) / (fr_o + f2 / fr_o);
    let nitrogen_term = 0.1068 * ops::exp(-3352.0 / temperature_k) / (fr_n + f2 / fr_n);

    let classical = 1.84e-11 * (1.0 / pr) * ops::sqrt(tr);
    let relaxation = ops::powf(tr, -2.5) * (oxygen_term + nitrogen_term);

    8.686 * f2 * (classical + relaxation)
}

/// Turns [`AtmosphericConditions`] into a distance-dependent low-pass corner.
///
/// The corner is the lowest frequency on `FREQUENCY_GRID` whose accumulated
/// absorption over the propagation distance reaches
/// [`threshold_db`](AirAbsorption::threshold_db). This is plain description
/// data; the actual filtering lives in [`AirAbsorptionNode`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct AirAbsorption {
    /// The atmosphere the sound propagates through.
    pub conditions: AtmosphericConditions,
    /// Accumulated absorption (dB) that marks a frequency as "rolled off". The
    /// lowest grid frequency reaching this over the distance becomes the corner.
    pub threshold_db: Sample,
}

impl Default for AirAbsorption {
    /// The standard reference atmosphere with a 3 dB roll-off threshold.
    #[inline]
    fn default() -> Self {
        Self {
            conditions: AtmosphericConditions::default(),
            threshold_db: 3.0,
        }
    }
}

impl AirAbsorption {
    /// Creates an air-absorption model for `conditions` with the default 3 dB
    /// roll-off threshold.
    #[inline]
    #[must_use]
    pub fn new(conditions: AtmosphericConditions) -> Self {
        Self {
            conditions,
            ..Self::default()
        }
    }

    /// Creates an air-absorption model with an explicit roll-off threshold.
    #[inline]
    #[must_use]
    pub fn with_threshold(conditions: AtmosphericConditions, threshold_db: Sample) -> Self {
        Self {
            conditions,
            threshold_db,
        }
    }

    /// Returns the low-pass corner frequency (Hz) for a source `distance_m`
    /// metres away, rendered at `sample_rate`.
    ///
    /// The search walks `FREQUENCY_GRID` from low to high and returns the first
    /// frequency whose absorption over the distance reaches
    /// [`threshold_db`](Self::threshold_db). If no grid point reaches the
    /// threshold (a very short distance) the grid's upper bound is used. The
    /// result is clamped to `[MIN_CUTOFF_HZ, sample_rate * 0.499]` so it is
    /// always a valid low-pass corner. The corner is **monotonically
    /// non-increasing** in distance: farther sources are duller.
    #[must_use]
    pub fn cutoff_hz(&self, distance_m: Sample, sample_rate: u32) -> Sample {
        let max_cutoff = ((sample_rate as Sample) * 0.499).max(MIN_CUTOFF_HZ);
        let distance = distance_m.max(0.0);

        // Default to the grid's upper bound: nothing has rolled off yet.
        let mut cutoff = FREQUENCY_GRID[FREQUENCY_GRID.len() - 1];
        for &f in &FREQUENCY_GRID {
            let accumulated = absorption_db_per_metre(&self.conditions, f) * distance;
            if accumulated >= self.threshold_db {
                cutoff = f;
                break;
            }
        }

        cutoff.clamp(MIN_CUTOFF_HZ, max_cutoff)
    }
}

/// A real-time audio node that applies distance-based atmospheric low-pass
/// filtering (input port 0 -> output port 0).
///
/// Construct it with [`AirAbsorptionNode::new`], update the propagation
/// distance with [`AirAbsorptionNode::set_distance`] (or the atmosphere with
/// [`AirAbsorptionNode::set_conditions`]) off the audio thread, and call
/// [`AudioNode::process`] on the audio thread.
#[derive(Debug, Clone)]
pub struct AirAbsorptionNode {
    /// The absorption model (atmosphere + threshold) used to derive the corner.
    absorption: AirAbsorption,
    /// The shared RBJ biquad realising the low-pass. State is preserved across
    /// coefficient updates for click-free retuning.
    filter: Biquad,
    /// The most recently set propagation distance, in metres. Kept so the
    /// corner can be recomputed when only the atmosphere changes.
    distance_m: Sample,
    /// The current low-pass corner frequency, in Hz.
    cutoff_hz: Sample,
}

impl AirAbsorptionNode {
    /// Builds a node for a `channels`-wide signal at `sample_rate`, with the
    /// given `conditions` and initial source `distance_m`.
    #[must_use]
    pub fn new(
        channels: usize,
        sample_rate: u32,
        conditions: AtmosphericConditions,
        distance_m: Sample,
    ) -> Self {
        let absorption = AirAbsorption::new(conditions);
        let cutoff_hz = absorption.cutoff_hz(distance_m, sample_rate);
        let coeffs =
            BiquadCoeffs::design(BiquadKind::LowPass, sample_rate, cutoff_hz, LOWPASS_Q, 0.0);
        Self {
            absorption,
            filter: Biquad::new(coeffs, channels),
            distance_m,
            cutoff_hz,
        }
    }

    /// Retunes the low-pass for a new propagation distance (non-real-time).
    ///
    /// Recomputes the corner and re-designs the biquad, preserving filter state
    /// so no click is introduced. Do **not** call this from
    /// [`AudioNode::process`].
    pub fn set_distance(&mut self, distance_m: Sample, sample_rate: u32) {
        self.distance_m = distance_m;
        self.retune(sample_rate);
    }

    /// Retunes the low-pass for a new atmosphere (non-real-time).
    ///
    /// Recomputes the corner at the current distance and re-designs the biquad,
    /// preserving filter state.
    pub fn set_conditions(&mut self, conditions: AtmosphericConditions, sample_rate: u32) {
        self.absorption.conditions = conditions;
        self.retune(sample_rate);
    }

    /// Returns the current low-pass corner frequency, in Hz.
    #[inline]
    #[must_use]
    pub fn cutoff_hz(&self) -> Sample {
        self.cutoff_hz
    }

    /// Returns the current propagation distance, in metres.
    #[inline]
    #[must_use]
    pub fn distance_m(&self) -> Sample {
        self.distance_m
    }

    /// Recomputes the corner from the current distance/atmosphere and updates
    /// the biquad coefficients, keeping filter state.
    fn retune(&mut self, sample_rate: u32) {
        self.cutoff_hz = self.absorption.cutoff_hz(self.distance_m, sample_rate);
        self.filter.set_coeffs(BiquadCoeffs::design(
            BiquadKind::LowPass,
            sample_rate,
            self.cutoff_hz,
            LOWPASS_Q,
            0.0,
        ));
    }
}

impl AudioNode for AirAbsorptionNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);

        // Match the output's active length to the input's, then copy the signal
        // through channel by channel. `set_active_frames` saturates at the
        // output capacity, and the per-channel `min` keeps the copy in bounds,
        // so no branch here can panic.
        let frames = input.active_frames();
        output.set_active_frames(frames);
        let channels = input.channels().min(output.channels());
        for c in 0..channels {
            let src = input.channel(c);
            let dst = output.channel_mut(c);
            let n = src.len().min(dst.len());
            dst[..n].copy_from_slice(&src[..n]);
        }

        // Apply the low-pass in place on the output buffer.
        self.filter.process_inplace(output);
    }

    fn reset(&mut self) {
        self.filter.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};

    fn ctx(sample_rate: u32, frames: usize) -> RenderContext {
        RenderContext {
            sample_rate,
            frames,
            playhead: 0,
        }
    }

    fn energy(buf: &AudioBuffer) -> Sample {
        buf.channel(0).iter().map(|s| s * s).sum()
    }

    #[test]
    fn absorption_is_positive_and_monotonic_in_frequency() {
        let air = AtmosphericConditions::default();
        let mut previous = 0.0;
        for &f in &FREQUENCY_GRID {
            let alpha = absorption_db_per_metre(&air, f);
            assert!(alpha > 0.0, "alpha must be positive at {f} Hz, got {alpha}");
            assert!(
                alpha > previous,
                "alpha must increase with frequency: {alpha} !> {previous} at {f} Hz",
            );
            previous = alpha;
        }
    }

    #[test]
    fn absorption_at_1khz_matches_iso_reference_magnitude() {
        // ISO 9613-1 at 20 C / 50 % RH / 101.325 kPa gives ~0.0047 dB/m at 1 kHz.
        let air = AtmosphericConditions::default();
        let alpha = absorption_db_per_metre(&air, 1_000.0);
        assert!(
            (0.001..0.02).contains(&alpha),
            "1 kHz absorption {alpha} dB/m outside the expected 0.001..0.02 band",
        );
    }

    #[test]
    fn absorption_handles_degenerate_inputs_without_panicking() {
        // Non-positive frequency, zero humidity, and extreme pressure/temperature
        // must all stay finite and non-negative (no divide-by-zero / NaN).
        let air = AtmosphericConditions::new(-300.0, 0.0, 0.0);
        let a0 = absorption_db_per_metre(&air, 0.0);
        let a1 = absorption_db_per_metre(&air, -100.0);
        assert!(a0.is_finite() && a0 >= 0.0);
        assert!(a1.is_finite() && a1 >= 0.0);
    }

    #[test]
    fn cutoff_is_non_increasing_with_distance() {
        let model = AirAbsorption::default();
        let sr = 48_000;
        let distances = [0.0, 1.0, 5.0, 20.0, 100.0, 1_000.0];
        let mut previous = Sample::INFINITY;
        for d in distances {
            let cutoff = model.cutoff_hz(d, sr);
            assert!(
                cutoff <= previous + 1.0e-3,
                "cutoff must not rise with distance: {cutoff} > {previous} at {d} m",
            );
            assert!((MIN_CUTOFF_HZ..=(sr as Sample) * 0.499).contains(&cutoff));
            previous = cutoff;
        }
    }

    #[test]
    fn short_distance_keeps_the_full_band() {
        // At zero distance nothing is absorbed, so the corner sits at the top.
        let model = AirAbsorption::default();
        let sr = 48_000;
        let cutoff = model.cutoff_hz(0.0, sr);
        assert!(cutoff >= (sr as Sample) * 0.499 - 1.0);
    }

    #[test]
    fn process_preserves_frame_count() {
        let sr = 48_000;
        let frames = 256;
        let mut node = AirAbsorptionNode::new(1, sr, AtmosphericConditions::default(), 10.0);
        let mut input = AudioBuffer::new(ChannelLayout::Mono, frames);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            *s = if i % 2 == 0 { 1.0 } else { -1.0 };
        }
        let inputs = [input];
        let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, frames)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(sr, frames), &mut io);
        assert_eq!(outputs[0].active_frames(), frames);
    }

    #[test]
    fn low_frequencies_pass_but_high_frequencies_are_attenuated() {
        let sr = 48_000;
        let frames = 512;
        // A long distance drives the corner well below Nyquist.
        let mut node = AirAbsorptionNode::new(1, sr, AtmosphericConditions::default(), 100.0);
        assert!(
            node.cutoff_hz() < (sr as Sample) * 0.4,
            "expected a low corner at 100 m, got {} Hz",
            node.cutoff_hz(),
        );

        // Direct current (constant 1.0) is a pure low frequency: it should pass
        // through a low-pass essentially untouched.
        {
            let mut input = AudioBuffer::new(ChannelLayout::Mono, frames);
            input.channel_mut(0).fill(1.0);
            let inputs = [input];
            let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, frames)];
            let mut io = ProcessIo::new(&inputs, &mut outputs);
            node.process(&ctx(sr, frames), &mut io);
            // Skip the initial transient; the tail should settle near unity.
            let tail = &outputs[0].channel(0)[frames - 16..];
            for &s in tail {
                assert!((s - 1.0).abs() < 0.05, "DC should pass, got {s}");
            }
        }

        // The Nyquist tone (alternating +/-1) is the highest representable
        // frequency and must lose most of its energy.
        node.reset();
        let mut input = AudioBuffer::new(ChannelLayout::Mono, frames);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            *s = if i % 2 == 0 { 1.0 } else { -1.0 };
        }
        let input_energy = energy(&input);
        let inputs = [input];
        let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, frames)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(sr, frames), &mut io);
        let output_energy = energy(&outputs[0]);
        assert!(
            output_energy < input_energy * 0.25,
            "high-frequency energy should drop sharply: {output_energy} vs {input_energy}",
        );
    }

    #[test]
    fn reset_clears_filter_state() {
        let sr = 48_000;
        let frames = 128;
        let mut node = AirAbsorptionNode::new(1, sr, AtmosphericConditions::default(), 100.0);

        // Excite the filter with a burst so its internal state is non-zero.
        let mut input = AudioBuffer::new(ChannelLayout::Mono, frames);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            *s = if i % 2 == 0 { 1.0 } else { -1.0 };
        }
        let inputs = [input];
        let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, frames)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(sr, frames), &mut io);

        // After reset, feeding silence must produce exact silence: no residual
        // ringing from the previous block.
        node.reset();
        let silence = [AudioBuffer::new(ChannelLayout::Mono, frames)];
        let mut quiet_out = [AudioBuffer::new(ChannelLayout::Mono, frames)];
        let mut io = ProcessIo::new(&silence, &mut quiet_out);
        node.process(&ctx(sr, frames), &mut io);
        for &s in quiet_out[0].channel(0) {
            assert_eq!(s, 0.0, "state must be cleared after reset");
        }
    }
}
