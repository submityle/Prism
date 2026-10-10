//! Acoustic time format: the sample clock and speed of sound a propagation
//! query needs to turn a path length into a delay.
//!
//! Every geometric-acoustics query ultimately answers two questions about an
//! arrival: *how loud* and *how late*. The loudness is pure geometry, but the
//! lateness depends on two numbers that are properties of the *rendering
//! context*, not of the scene: the audio sample rate (how many samples make a
//! second) and the speed of sound in the medium (how many metres make a
//! second). Threading those two scalars through every routing entry point as
//! loose positional arguments is error-prone -- it is easy to transpose them --
//! and it bloats signatures. This type bundles them into one cohesive value and
//! owns the one operation that consumes them: converting a path length in metres
//! to a whole-sample delay.
//!
//! # Model
//!
//! A distance `d` metres travels in `d / c` seconds at speed of sound `c`, which
//! is `d / c * sample_rate` samples. The conversion rounds to the nearest whole
//! sample by integer stepping (never a float-to-int cast, so it is
//! deterministic and golden-comparable) and clamps to [`MAX_DELAY_SAMPLES`] so a
//! pathological distance can never overflow a delay-line index. A non-positive
//! speed of sound is floored to `1.0` m/s so the division is always finite.
//!
//! # Relationship
//!
//! This is the shared delay-conversion vocabulary for the crate's routing
//! backends. [`crate::portal_graph::route_portals`] takes an [`AcousticFormat`]
//! in place of a loose `(sample_rate, sound_speed)` pair, and its recursive
//! search carries the format rather than the two scalars. It reuses the
//! [`Sample`] scalar from [`prism_audio_core`] and does not redefine the speed
//! of sound: [`AcousticFormat::DEFAULT_SOUND_SPEED`] is the single canonical
//! dry-air value shared with [`crate::portal_graph::DEFAULT_SOUND_SPEED`].
//!
//! # Real-time contract
//!
//! [`AcousticFormat`] is a `Copy` value with no heap allocation and no interior
//! mutability; [`AcousticFormat::delay_samples`] is allocation free, lock free,
//! and cannot panic for any finite or non-finite input. It is cheap enough to
//! pass by value on the audio thread, though in practice routing runs at control
//! rate.
//!
//! # Provenance
//!
//! This is an elementary unit conversion (distance over speed times sample rate)
//! with a deterministic integer rounding step. It is pure classic DSP with no AI
//! or ML. It is engine-agnostic and contains **no Unreal Engine, Unity, Godot,
//! Wwise, FMOD, Steam Audio, or Google Resonance Audio source or derived
//! code**; it is implemented purely from the physical definition of propagation
//! delay.

use prism_audio_core::math::Sample;

/// Default speed of sound in dry air at room temperature (metres per second).
///
/// This is the single canonical value re-exported as
/// [`crate::portal_graph::DEFAULT_SOUND_SPEED`].
pub const DEFAULT_SOUND_SPEED: Sample = 343.0;

/// Largest representable delay in whole samples (matches the crate's other
/// delay-line modules); accumulated distances that exceed it are clamped.
pub const MAX_DELAY_SAMPLES: usize = 1 << 18;

/// The sample clock and speed of sound a propagation query needs to convert a
/// path length into a delay.
///
/// Construct one with [`AcousticFormat::new`] or
/// [`AcousticFormat::with_default_speed`], then call
/// [`AcousticFormat::delay_samples`] to turn a metre distance into whole
/// samples.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct AcousticFormat {
    /// Audio sample rate in hertz (samples per second).
    pub sample_rate: Sample,
    /// Speed of sound in the propagation medium, in metres per second.
    pub sound_speed: Sample,
}

impl AcousticFormat {
    /// Dry-air default re-exported as [`DEFAULT_SOUND_SPEED`].
    pub const DEFAULT_SOUND_SPEED: Sample = DEFAULT_SOUND_SPEED;

    /// Builds a format from an explicit sample rate and speed of sound.
    #[inline]
    #[must_use]
    pub const fn new(sample_rate: Sample, sound_speed: Sample) -> Self {
        Self {
            sample_rate,
            sound_speed,
        }
    }

    /// Builds a format at `sample_rate` using [`DEFAULT_SOUND_SPEED`] for the
    /// medium (dry air at room temperature).
    #[inline]
    #[must_use]
    pub const fn with_default_speed(sample_rate: Sample) -> Self {
        Self::new(sample_rate, DEFAULT_SOUND_SPEED)
    }

    /// The speed of sound used for conversions, floored to `1.0` m/s so the
    /// delay division is always finite even when a caller supplies a
    /// non-positive speed.
    #[inline]
    #[must_use]
    pub fn effective_sound_speed(&self) -> Sample {
        self.sound_speed.max(1.0)
    }

    /// Converts a path length in metres to a whole-sample delay.
    ///
    /// The result rounds to the nearest sample and is clamped to
    /// [`MAX_DELAY_SAMPLES`]. Negative, zero, or non-finite distances map to the
    /// safe delay `0`.
    #[inline]
    #[must_use]
    pub fn delay_samples(&self, distance_metres: Sample) -> usize {
        seconds_to_samples(
            distance_metres / self.effective_sound_speed(),
            self.sample_rate,
        )
    }
}

/// Converts a delay in seconds to whole samples by integer stepping (no
/// float-to-int cast), rounding to nearest and clamping to [`MAX_DELAY_SAMPLES`].
#[must_use]
fn seconds_to_samples(seconds: Sample, sample_rate: Sample) -> usize {
    let cap = MAX_DELAY_SAMPLES as Sample;
    let exact = (seconds * sample_rate).max(0.0).min(cap);
    let mut n: usize = 0;
    let mut acc: Sample = 0.0;
    while acc + 1024.0 <= exact {
        acc += 1024.0;
        n += 1024;
    }
    while acc + 1.0 <= exact {
        acc += 1.0;
        n += 1;
    }
    if exact - acc >= 0.5 {
        n += 1;
    }
    n
}

#[cfg(test)]
mod tests {
    use super::{AcousticFormat, DEFAULT_SOUND_SPEED, MAX_DELAY_SAMPLES};

    #[test]
    fn delay_samples_rounds_to_nearest() {
        let fmt = AcousticFormat::new(48_000.0, 1.0);
        // At 1 m/s the distance in metres equals the delay in seconds.
        assert_eq!(fmt.delay_samples(0.0), 0);
        assert_eq!(fmt.delay_samples(0.01), 480);
        assert_eq!(fmt.delay_samples(100.4 / 48_000.0), 100);
        assert_eq!(fmt.delay_samples(100.6 / 48_000.0), 101);
    }

    #[test]
    fn delay_samples_clamps_and_floors_speed() {
        let huge = AcousticFormat::new(48_000.0, 1.0);
        assert_eq!(huge.delay_samples(1.0e9), MAX_DELAY_SAMPLES);

        // A non-positive speed is floored to 1 m/s rather than dividing by zero.
        let zero_speed = AcousticFormat::new(48_000.0, 0.0);
        assert_eq!(zero_speed.effective_sound_speed(), 1.0);
        assert_eq!(zero_speed.delay_samples(0.01), 480);
    }

    #[test]
    fn with_default_speed_uses_dry_air() {
        let fmt = AcousticFormat::with_default_speed(44_100.0);
        assert_eq!(fmt.sound_speed, DEFAULT_SOUND_SPEED);
        assert_eq!(fmt.sample_rate, 44_100.0);
    }

    #[test]
    fn negative_distance_is_safe() {
        let fmt = AcousticFormat::with_default_speed(48_000.0);
        assert_eq!(fmt.delay_samples(-5.0), 0);
    }
}
