//! A fixed-capacity pool of grain voices with deterministic voice stealing.
//!
//! A granular cloud must be bounded: the engine can request a new grain far
//! faster than old grains finish, so there has to be a hard ceiling on
//! simultaneous voices and a deterministic rule for what happens when that
//! ceiling is hit. This pool preallocates its grains once and never allocates
//! again. [`GrainPool::allocate`] returns a free grain if one exists, otherwise
//! it steals the grain closest to finishing (the least audible one to cut), so
//! overflow degrades gracefully instead of panicking or dropping new grains.
//! Mixing ticks every active grain and sums their stereo contributions.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Backs the fixed-capacity grain cloud of design section 47.4; owns
//! [`crate::granular::grain::Grain`] voices for
//! [`crate::granular::engine::GranularEngine`].

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::granular::grain::Grain;
use prism_audio_core::math::Sample;

/// A bounded pool of grain voices.
#[derive(Clone, Debug)]
pub struct GrainPool {
    grains: Vec<Grain>,
}

impl GrainPool {
    /// Creates a pool with `capacity` grains (clamped to at least one),
    /// preallocated and idle.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        let mut grains = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            grains.push(Grain::new());
        }
        Self { grains }
    }

    /// Returns the pool capacity.
    #[inline]
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.grains.len()
    }

    /// Returns the number of currently sounding grains.
    #[must_use]
    pub fn active(&self) -> usize {
        self.grains.iter().filter(|g| g.is_active()).count()
    }

    /// Returns a grain to (re)trigger: a free one if available, otherwise the
    /// one closest to finishing (deterministic voice stealing).
    pub fn allocate(&mut self) -> &mut Grain {
        // First pass: find a free slot by index.
        let mut free = None;
        let mut steal = 0usize;
        let mut steal_progress = -1.0f32;
        for (i, g) in self.grains.iter().enumerate() {
            if !g.is_active() {
                free = Some(i);
                break;
            }
            let p = g.progress();
            if p > steal_progress {
                steal_progress = p;
                steal = i;
            }
        }
        let index = free.unwrap_or(steal);
        &mut self.grains[index]
    }

    /// Ticks every active grain and returns the summed stereo output.
    #[inline]
    pub fn mix(&mut self) -> (Sample, Sample) {
        let mut left = 0.0;
        let mut right = 0.0;
        for g in &mut self.grains {
            if g.is_active() {
                let (l, r) = g.tick();
                left += l;
                right += r;
            }
        }
        (left, right)
    }

    /// Silences every grain.
    pub fn reset(&mut self) {
        for g in &mut self.grains {
            g.silence();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_uses_free_slots_first() {
        let mut pool = GrainPool::new(4);
        pool.allocate().trigger(440.0, 100, 1.0, 0.0, 48_000);
        assert_eq!(pool.active(), 1);
        pool.allocate().trigger(440.0, 100, 1.0, 0.0, 48_000);
        assert_eq!(pool.active(), 2);
    }

    #[test]
    fn overflow_steals_without_growing() {
        let mut pool = GrainPool::new(2);
        for _ in 0..10 {
            pool.allocate().trigger(440.0, 1_000, 1.0, 0.0, 48_000);
        }
        assert_eq!(pool.capacity(), 2);
        assert!(pool.active() <= 2);
    }

    #[test]
    fn mix_sums_active_grains() {
        let mut pool = GrainPool::new(4);
        pool.allocate().trigger(440.0, 256, 1.0, 0.0, 48_000);
        pool.allocate().trigger(660.0, 256, 1.0, 0.0, 48_000);
        let mut peak = 0.0f32;
        for _ in 0..256 {
            let (l, _r) = pool.mix();
            peak = peak.max(l.abs());
        }
        assert!(peak > 0.1, "peak={peak}");
    }

    #[test]
    fn reset_silences() {
        let mut pool = GrainPool::new(4);
        pool.allocate().trigger(440.0, 256, 1.0, 0.0, 48_000);
        pool.reset();
        assert_eq!(pool.active(), 0);
    }
}
