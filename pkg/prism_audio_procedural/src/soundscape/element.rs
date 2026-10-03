//! A single soundscape element: one ambience voice and its scatter rules.
//!
//! An element is one kind of environmental sound (a bird call, a gust of wind,
//! a dripping pipe, distant traffic) together with the rules that govern how it
//! is scattered around the listener: how often it fires, how likely each slot
//! is to actually sound, how its pitch and gain are randomised, how far from
//! the listener it may be placed, how many copies may sound at once, whether it
//! belongs indoors or outdoors, and how its activity is weighted between day
//! and night. The element holds only data and the few pure helpers that turn
//! the daylight factor into an effective trigger probability; the actual
//! scattering is performed by [`crate::soundscape::scheduler`].
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the palette color-point of design section 37; grouped into a
//! [`crate::soundscape::palette::SoundscapePalette`] and consumed by
//! [`crate::soundscape::scheduler::SoundscapeScheduler`]. The `id` is echoed
//! into every emitted [`crate::soundscape::scheduler::ScatteredOneShot`] so the
//! host knows which voice to spawn.

use crate::dsp::lerp;
use prism_audio_core::math::Sample;

/// A single ambience element and the rules for scattering it.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SoundscapeElement {
    id: u32,
    mean_interval_s: Sample,
    interval_jitter: Sample,
    trigger_prob: Sample,
    pitch_low_semitones: Sample,
    pitch_high_semitones: Sample,
    gain_low: Sample,
    gain_high: Sample,
    scatter_radius_m: Sample,
    nominal_duration_s: Sample,
    concurrency_cap: u16,
    day_weight: Sample,
    night_weight: Sample,
    allow_indoor: bool,
    allow_outdoor: bool,
}

impl SoundscapeElement {
    /// Builds an element with sane defaults for `id`.
    ///
    /// Defaults describe a mid-activity outdoor voice: one scheduling slot
    /// roughly every two seconds with heavy jitter, a certain trigger, no pitch
    /// shift, unit gain, a `20 m` scatter radius, a one-second nominal duration,
    /// a cap of four concurrent copies, and equal day/night weighting. Use the
    /// `with_*` builders to shape it. All numeric inputs are sanitised.
    #[inline]
    #[must_use]
    pub fn new(id: u32) -> Self {
        Self {
            id,
            mean_interval_s: 2.0,
            interval_jitter: 0.6,
            trigger_prob: 1.0,
            pitch_low_semitones: 0.0,
            pitch_high_semitones: 0.0,
            gain_low: 1.0,
            gain_high: 1.0,
            scatter_radius_m: 20.0,
            nominal_duration_s: 1.0,
            concurrency_cap: 4,
            day_weight: 1.0,
            night_weight: 1.0,
            allow_indoor: true,
            allow_outdoor: true,
        }
    }

    /// Sets the mean scheduling interval (seconds) and its jitter fraction.
    ///
    /// Each fired slot draws its next interval as `mean * (1 +/- jitter)`.
    /// `mean` is floored at `1 ms` so the scheduler always advances; `jitter`
    /// is clamped to `[0, 0.99]`.
    #[inline]
    #[must_use]
    pub fn with_interval(mut self, mean_s: Sample, jitter: Sample) -> Self {
        self.mean_interval_s = sanitize_positive(mean_s, 0.001);
        self.interval_jitter = clamp01(jitter).min(0.99);
        self
    }

    /// Sets the per-slot trigger probability in `[0, 1]`.
    #[inline]
    #[must_use]
    pub fn with_trigger_prob(mut self, prob: Sample) -> Self {
        self.trigger_prob = clamp01(prob);
        self
    }

    /// Sets the random pitch range in semitones (`low..=high`).
    #[inline]
    #[must_use]
    pub fn with_pitch_range(mut self, low_semitones: Sample, high_semitones: Sample) -> Self {
        let (lo, hi) = ordered(low_semitones, high_semitones);
        self.pitch_low_semitones = lo;
        self.pitch_high_semitones = hi;
        self
    }

    /// Sets the random linear-gain range (`low..=high`, non-negative).
    #[inline]
    #[must_use]
    pub fn with_gain_range(mut self, low: Sample, high: Sample) -> Self {
        let (lo, hi) = ordered(low.max(0.0), high.max(0.0));
        self.gain_low = lo;
        self.gain_high = hi;
        self
    }

    /// Sets the scatter radius (metres) around the listener. Floored at `0`.
    #[inline]
    #[must_use]
    pub fn with_scatter_radius(mut self, radius_m: Sample) -> Self {
        self.scatter_radius_m = sanitize_positive(radius_m, 0.0);
        self
    }

    /// Sets the nominal one-shot duration (seconds) used to estimate how many
    /// copies are concurrently sounding. Floored at `1 ms`.
    #[inline]
    #[must_use]
    pub fn with_nominal_duration(mut self, duration_s: Sample) -> Self {
        self.nominal_duration_s = sanitize_positive(duration_s, 0.001);
        self
    }

    /// Sets the maximum number of concurrently sounding copies (floored at `1`).
    #[inline]
    #[must_use]
    pub fn with_concurrency_cap(mut self, cap: u16) -> Self {
        self.concurrency_cap = cap.max(1);
        self
    }

    /// Sets the day and night activity weights (each clamped to `[0, 1]`).
    ///
    /// The weights cross-fade with the daylight factor, so a nocturnal element
    /// (`day = 0`, `night = 1`) is silent at noon and active at midnight.
    #[inline]
    #[must_use]
    pub fn with_day_night(mut self, day_weight: Sample, night_weight: Sample) -> Self {
        self.day_weight = clamp01(day_weight);
        self.night_weight = clamp01(night_weight);
        self
    }

    /// Sets whether the element may sound indoors and/or outdoors.
    #[inline]
    #[must_use]
    pub fn with_environments(mut self, allow_indoor: bool, allow_outdoor: bool) -> Self {
        self.allow_indoor = allow_indoor;
        self.allow_outdoor = allow_outdoor;
        self
    }

    /// Returns the element id echoed into every emitted one-shot.
    #[inline]
    #[must_use]
    pub fn id(self) -> u32 {
        self.id
    }

    /// Returns the mean scheduling interval in seconds.
    #[inline]
    #[must_use]
    pub fn mean_interval_s(self) -> Sample {
        self.mean_interval_s
    }

    /// Returns the interval jitter fraction in `[0, 0.99]`.
    #[inline]
    #[must_use]
    pub fn interval_jitter(self) -> Sample {
        self.interval_jitter
    }

    /// Returns the base per-slot trigger probability.
    #[inline]
    #[must_use]
    pub fn trigger_prob(self) -> Sample {
        self.trigger_prob
    }

    /// Returns the random pitch range in semitones as `(low, high)`.
    #[inline]
    #[must_use]
    pub fn pitch_range(self) -> (Sample, Sample) {
        (self.pitch_low_semitones, self.pitch_high_semitones)
    }

    /// Returns the random gain range as `(low, high)`.
    #[inline]
    #[must_use]
    pub fn gain_range(self) -> (Sample, Sample) {
        (self.gain_low, self.gain_high)
    }

    /// Returns the scatter radius in metres.
    #[inline]
    #[must_use]
    pub fn scatter_radius_m(self) -> Sample {
        self.scatter_radius_m
    }

    /// Returns the nominal one-shot duration in seconds.
    #[inline]
    #[must_use]
    pub fn nominal_duration_s(self) -> Sample {
        self.nominal_duration_s
    }

    /// Returns the per-element concurrency cap.
    #[inline]
    #[must_use]
    pub fn concurrency_cap(self) -> u16 {
        self.concurrency_cap
    }

    /// Returns `true` when the element is allowed in the given environment.
    #[inline]
    #[must_use]
    pub fn allowed_indoor(self) -> bool {
        self.allow_indoor
    }

    /// Returns `true` when the element is allowed outdoors.
    #[inline]
    #[must_use]
    pub fn allowed_outdoor(self) -> bool {
        self.allow_outdoor
    }

    /// Returns the day/night activity weight for a `daylight` factor in
    /// `[0, 1]` (`lerp(night, day, daylight)`).
    #[inline]
    #[must_use]
    pub fn daylight_weight(self, daylight: Sample) -> Sample {
        lerp(self.night_weight, self.day_weight, daylight)
    }

    /// Returns the effective per-slot trigger probability after applying the
    /// day/night weight and an external `density` multiplier, clamped to
    /// `[0, 1]`.
    #[inline]
    #[must_use]
    pub fn effective_trigger_prob(self, daylight: Sample, density: Sample) -> Sample {
        clamp01(self.trigger_prob * self.daylight_weight(daylight) * clamp01(density))
    }

    /// Returns `true` when the element may sound in the given indoor state.
    #[inline]
    #[must_use]
    pub fn permitted_indoors(self, indoor: bool) -> bool {
        if indoor {
            self.allow_indoor
        } else {
            self.allow_outdoor
        }
    }
}

#[inline]
fn clamp01(x: Sample) -> Sample {
    if x.is_finite() { x.clamp(0.0, 1.0) } else { 0.0 }
}

#[inline]
fn sanitize_positive(x: Sample, floor: Sample) -> Sample {
    if x.is_finite() { x.max(floor) } else { floor }
}

#[inline]
fn ordered(a: Sample, b: Sample) -> (Sample, Sample) {
    let a = if a.is_finite() { a } else { 0.0 };
    let b = if b.is_finite() { b } else { 0.0 };
    if a <= b { (a, b) } else { (b, a) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builders_sanitise_inputs() {
        let e = SoundscapeElement::new(3)
            .with_interval(-5.0, 2.0)
            .with_trigger_prob(2.0)
            .with_pitch_range(7.0, -7.0)
            .with_gain_range(-1.0, 0.5)
            .with_concurrency_cap(0);
        assert!(e.mean_interval_s() >= 0.001);
        assert!(e.interval_jitter() <= 0.99);
        assert_eq!(e.trigger_prob(), 1.0);
        assert_eq!(e.pitch_range(), (-7.0, 7.0));
        assert_eq!(e.gain_range(), (0.0, 0.5));
        assert_eq!(e.concurrency_cap(), 1);
    }

    #[test]
    fn nocturnal_element_is_quiet_in_daylight() {
        let owl = SoundscapeElement::new(1)
            .with_trigger_prob(1.0)
            .with_day_night(0.0, 1.0);
        assert!(owl.effective_trigger_prob(1.0, 1.0) < 1e-6);
        assert!((owl.effective_trigger_prob(0.0, 1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn density_scales_trigger() {
        let e = SoundscapeElement::new(2).with_trigger_prob(1.0);
        assert!((e.effective_trigger_prob(1.0, 0.5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn environment_gate() {
        let bird = SoundscapeElement::new(4).with_environments(false, true);
        assert!(bird.permitted_indoors(false));
        assert!(!bird.permitted_indoors(true));
    }
}
