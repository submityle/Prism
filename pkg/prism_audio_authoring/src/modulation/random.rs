//! Deterministic random modulation source.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the random modulation source of design section 12. Driven by the
//! shared seeded generator in `crate::rng`, it offers both a stepped
//! (sample-and-hold of noise) and a smoothly interpolated mode, and implements
//! `super::source::Modulator` for use in the modulation matrix.

use prism_audio_core::Sample;

use super::source::{ModContext, Modulator};
use crate::rng::Rng;

/// Interpolation behaviour between successive random targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum RandomMode {
    /// Hold each drawn value until the next clock edge (stepped noise).
    Stepped,
    /// Linearly interpolate from the previous value to the next across the
    /// clock period (smooth random).
    Smooth,
}

/// A rate-clocked random source producing reproducible stepped or smooth noise.
///
/// The output polarity is selectable: bipolar values span `[-1, 1)` while
/// unipolar values span `[0, 1)`. Because every draw comes from the seeded
/// [`Rng`], a given seed reproduces an identical modulation stream.
#[derive(Debug, Clone)]
pub struct RandomModulator {
    rng: Rng,
    mode: RandomMode,
    bipolar: bool,
    prev: Sample,
    next: Sample,
    phase: Sample,
    increment: Sample,
    value: Sample,
}

impl RandomModulator {
    /// Creates a random source clocked at `rate_hz`, seeded with `seed`.
    ///
    /// # Panics
    ///
    /// Panics if `sample_rate` is zero.
    #[must_use]
    pub fn new(
        sample_rate: u32,
        rate_hz: Sample,
        mode: RandomMode,
        bipolar: bool,
        seed: u64,
    ) -> Self {
        assert!(sample_rate > 0, "sample_rate must be non-zero");
        let mut rng = Rng::new(seed);
        let prev = draw(&mut rng, bipolar);
        let next = draw(&mut rng, bipolar);
        let mut source = Self {
            rng,
            mode,
            bipolar,
            prev,
            next,
            phase: 0.0,
            increment: 0.0,
            value: prev,
        };
        source.set_rate(sample_rate, rate_hz);
        source
    }

    /// Sets the clock rate in Hz (clamped to be non-negative).
    #[inline]
    pub fn set_rate(&mut self, sample_rate: u32, rate_hz: Sample) {
        self.increment = rate_hz.max(0.0) / sample_rate.max(1) as Sample;
    }

    /// Advances the source one sample and returns its new value.
    pub fn next_sample(&mut self) -> Sample {
        self.phase += self.increment;
        while self.phase >= 1.0 {
            self.phase -= 1.0;
            self.prev = self.next;
            self.next = draw(&mut self.rng, self.bipolar);
        }
        self.value = match self.mode {
            RandomMode::Stepped => self.prev,
            RandomMode::Smooth => self.prev + (self.next - self.prev) * self.phase,
        };
        self.value
    }
}

impl Modulator for RandomModulator {
    fn tick(&mut self, ctx: &ModContext) -> Sample {
        for _ in 0..ctx.frames {
            self.next_sample();
        }
        self.value
    }

    fn value(&self) -> Sample {
        self.value
    }

    fn reset(&mut self) {
        self.phase = 0.0;
        self.value = self.prev;
    }
}

/// Draws one value in the requested polarity.
#[inline]
fn draw(rng: &mut Rng, bipolar: bool) -> Sample {
    if bipolar {
        rng.next_bipolar()
    } else {
        rng.next_unit()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 48_000;

    #[test]
    fn stepped_holds_between_edges() {
        let mut r = RandomModulator::new(8, 1.0, RandomMode::Stepped, true, 1);
        let a = r.next_sample();
        let b = r.next_sample();
        // Within a clock period the stepped value does not change.
        assert_eq!(a, b);
    }

    #[test]
    fn same_seed_reproduces_stream() {
        let mut a = RandomModulator::new(SR, 20.0, RandomMode::Smooth, true, 42);
        let mut b = RandomModulator::new(SR, 20.0, RandomMode::Smooth, true, 42);
        for _ in 0..5000 {
            assert_eq!(a.next_sample(), b.next_sample());
        }
    }

    #[test]
    fn bipolar_values_are_in_range() {
        let mut r = RandomModulator::new(SR, 100.0, RandomMode::Smooth, true, 3);
        for _ in 0..10_000 {
            let v = r.next_sample();
            assert!((-1.0..=1.0).contains(&v), "{v}");
        }
    }

    #[test]
    fn unipolar_values_are_in_range() {
        let mut r = RandomModulator::new(SR, 100.0, RandomMode::Stepped, false, 3);
        for _ in 0..10_000 {
            let v = r.next_sample();
            assert!((0.0..=1.0).contains(&v), "{v}");
        }
    }

    #[test]
    fn smooth_differs_from_stepped() {
        let mut smooth = RandomModulator::new(SR, 10.0, RandomMode::Smooth, true, 7);
        let mut stepped = RandomModulator::new(SR, 10.0, RandomMode::Stepped, true, 7);
        let mut diff = 0.0f32;
        for _ in 0..2000 {
            diff += (smooth.next_sample() - stepped.next_sample()).abs();
        }
        assert!(diff > 1.0, "expected smooth and stepped to differ, diff={diff}");
    }

    #[test]
    fn tick_matches_sample_loop() {
        let mut a = RandomModulator::new(SR, 15.0, RandomMode::Smooth, true, 9);
        let mut b = RandomModulator::new(SR, 15.0, RandomMode::Smooth, true, 9);
        let ctx = ModContext::new(SR, 32);
        let via_tick = a.tick(&ctx);
        let mut via_loop = 0.0;
        for _ in 0..32 {
            via_loop = b.next_sample();
        }
        assert!((via_tick - via_loop).abs() < 1.0e-6);
    }
}
