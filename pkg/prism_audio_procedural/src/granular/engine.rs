//! The granular cloud engine: a seeded scheduler driving a bounded grain pool.
//!
//! A granular texture (splashing liquid, scattering debris, a hiss of sand) is
//! produced by continuously spawning short grains with randomised pitch, pan,
//! length, and amplitude, drawn around a set of control parameters. This engine
//! owns that loop. A phase accumulator fires grains at the requested grain rate;
//! each grain's parameters are jittered from the deterministic RNG around the
//! current [`GrainParams`], so the texture is lively yet an identical seed and
//! parameter history reproduce it bit-for-bit. The grain pool is fixed-capacity
//! with voice stealing, so the whole render path is allocation-free and bounded
//! regardless of how high the grain rate is driven.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements design section 47.4 (granular synthesis); schedules
//! [`crate::granular::grain::Grain`]s into a
//! [`crate::granular::pool::GrainPool`] using [`crate::rng`].

use bevy_math::ops;

use crate::granular::pool::GrainPool;
use crate::rng::ProceduralRng;
use prism_audio_core::math::Sample;

/// Control parameters for a granular cloud.
///
/// These are the "centres" each spawned grain is jittered around; drive them
/// from physics quantities (impact energy, flow speed) or from the soundscape.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct GrainParams {
    /// Grains spawned per second.
    pub grain_rate_hz: Sample,
    /// Centre tone frequency of a grain in hertz.
    pub base_freq_hz: Sample,
    /// Mean grain length in seconds.
    pub length_s: Sample,
    /// Grain length jitter as a fraction of the mean in `[0, 1]`.
    pub length_jitter: Sample,
    /// Pitch jitter in semitones (each grain is detuned by up to this much).
    pub pitch_jitter_semitones: Sample,
    /// Stereo scatter half-width in `[0, 1]`.
    pub pan_spread: Sample,
    /// Peak grain amplitude.
    pub amplitude: Sample,
}

impl Default for GrainParams {
    #[inline]
    fn default() -> Self {
        Self {
            grain_rate_hz: 120.0,
            base_freq_hz: 1_200.0,
            length_s: 0.03,
            length_jitter: 0.4,
            pitch_jitter_semitones: 7.0,
            pan_spread: 0.6,
            amplitude: 0.4,
        }
    }
}

/// A deterministic granular synthesis engine.
#[derive(Clone, Debug)]
pub struct GranularEngine {
    pool: GrainPool,
    rng: ProceduralRng,
    params: GrainParams,
    phase: Sample,
    sample_rate: u32,
}

impl GranularEngine {
    /// Creates an engine with a grain pool of `voices` grains, seeded with
    /// `seed`, at `sample_rate`.
    #[must_use]
    pub fn new(sample_rate: u32, voices: usize, seed: u64) -> Self {
        Self {
            pool: GrainPool::new(voices),
            rng: ProceduralRng::new(seed),
            params: GrainParams::default(),
            phase: 0.0,
            sample_rate: sample_rate.max(1),
        }
    }

    /// Replaces the cloud control parameters.
    #[inline]
    pub fn set_params(&mut self, params: GrainParams) {
        self.params = params;
    }

    /// Returns the current cloud control parameters.
    #[inline]
    #[must_use]
    pub fn params(&self) -> GrainParams {
        self.params
    }

    /// Returns the number of currently sounding grains.
    #[inline]
    #[must_use]
    pub fn active_grains(&self) -> usize {
        self.pool.active()
    }

    /// Spawns one grain with parameters jittered around the current controls.
    fn spawn_grain(&mut self) {
        let p = self.params;
        let fs = self.sample_rate;

        let semis = self.rng.next_bipolar() * p.pitch_jitter_semitones;
        let ratio = ops::exp2(semis / 12.0);
        let freq = (p.base_freq_hz * ratio).max(1.0);

        let len_scale = 1.0 + self.rng.next_bipolar() * p.length_jitter.clamp(0.0, 1.0);
        let length_s = (p.length_s * len_scale).max(0.001);
        let length = (length_s * fs as Sample) as u32;

        let pan = self.rng.next_bipolar() * p.pan_spread.clamp(0.0, 1.0);
        // Slight amplitude variation so grains do not phase-lock into a tone.
        let amp = p.amplitude * (0.7 + 0.3 * self.rng.next_unit());

        self.pool
            .allocate()
            .trigger(freq, length.max(1), amp, pan, fs);
    }

    /// Renders `frames` samples of the cloud into `left` and `right`,
    /// overwriting them.
    ///
    /// Grains are spawned at the control grain rate and summed from the pool.
    /// No allocation occurs regardless of the grain rate.
    pub fn render_stereo(&mut self, frames: usize, left: &mut [Sample], right: &mut [Sample]) {
        let frames = frames.min(left.len()).min(right.len());
        if frames == 0 {
            return;
        }
        let fs = self.sample_rate as Sample;
        let rate = self.params.grain_rate_hz.max(0.0);
        for (l, r) in left.iter_mut().zip(right.iter_mut()).take(frames) {
            self.phase += rate / fs;
            // Allow multiple spawns per sample for very dense clouds, bounded by
            // the pool capacity via voice stealing.
            let mut guard = 0;
            while self.phase >= 1.0 && guard < 8 {
                self.phase -= 1.0;
                self.spawn_grain();
                guard += 1;
            }
            if self.phase >= 1.0 {
                self.phase = 0.0;
            }
            let (lo, ro) = self.pool.mix();
            *l = lo;
            *r = ro;
        }
    }

    /// Renders `frames` samples of the cloud summed to mono into `out`.
    pub fn render_mono(&mut self, frames: usize, out: &mut [Sample]) {
        let frames = frames.min(out.len());
        if frames == 0 {
            return;
        }
        let fs = self.sample_rate as Sample;
        let rate = self.params.grain_rate_hz.max(0.0);
        for slot in out.iter_mut().take(frames) {
            self.phase += rate / fs;
            let mut guard = 0;
            while self.phase >= 1.0 && guard < 8 {
                self.phase -= 1.0;
                self.spawn_grain();
                guard += 1;
            }
            if self.phase >= 1.0 {
                self.phase = 0.0;
            }
            let (lo, ro) = self.pool.mix();
            *slot = 0.5 * (lo + ro);
        }
    }

    /// Silences the cloud.
    pub fn reset(&mut self) {
        self.pool.reset();
        self.phase = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(feature = "std"))]
    use alloc::vec;

    #[test]
    fn cloud_produces_sound() {
        let mut eng = GranularEngine::new(48_000, 32, 1);
        eng.set_params(GrainParams::default());
        let mut l = vec![0.0; 4_800];
        let mut r = vec![0.0; 4_800];
        eng.render_stereo(4_800, &mut l, &mut r);
        let peak = l
            .iter()
            .chain(r.iter())
            .fold(0.0f32, |m, &s| m.max(s.abs()));
        assert!(peak > 0.01, "peak={peak}");
    }

    #[test]
    fn higher_rate_spawns_more() {
        let count = |rate: Sample| {
            let mut eng = GranularEngine::new(48_000, 256, 7);
            eng.set_params(GrainParams {
                grain_rate_hz: rate,
                length_s: 0.001,
                ..GrainParams::default()
            });
            let mut out = vec![0.0; 48_000];
            eng.render_mono(48_000, &mut out);
            out.iter().filter(|&&s| s.abs() > 1e-6).count()
        };
        let slow = count(50.0);
        let fast = count(400.0);
        assert!(fast > slow, "slow={slow} fast={fast}");
    }

    #[test]
    fn zero_rate_is_silent() {
        let mut eng = GranularEngine::new(48_000, 32, 3);
        eng.set_params(GrainParams {
            grain_rate_hz: 0.0,
            ..GrainParams::default()
        });
        let mut out = vec![0.0; 4_800];
        eng.render_mono(4_800, &mut out);
        let peak = out.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
        assert_eq!(peak, 0.0);
    }

    #[test]
    fn deterministic_for_same_seed() {
        let render = || {
            let mut eng = GranularEngine::new(48_000, 64, 42);
            eng.set_params(GrainParams::default());
            let mut out = vec![0.0; 2_048];
            eng.render_mono(2_048, &mut out);
            out
        };
        assert_eq!(render(), render());
    }
}
