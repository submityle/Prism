//! [`SmoothedDelta`]: an exponential moving average of frame deltas.
//!
//! Raw per-frame deltas jitter under scheduling noise; feeding them directly
//! into UI readouts (FPS, frame-time graphs) or delta-driven smoothing looks
//! noisy. [`SmoothedDelta`] keeps an exponential moving average so displays and
//! camera smoothing see a stable value while still tracking real changes.

use crate::Duration;

/// An exponential-moving-average smoother for frame deltas.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SmoothedDelta {
    smoothed_secs: f64,
    smoothing: f64,
    initialized: bool,
}

impl SmoothedDelta {
    /// Default smoothing weight for a new sample (`0.1`): heavy smoothing that
    /// still converges within a handful of frames.
    pub const DEFAULT_SMOOTHING: f64 = 0.1;

    /// Create a smoother with an explicit new-sample weight. The weight is the
    /// fraction of each new sample folded into the average and is clamped to
    /// `0.0..=1.0` (`1.0` = no smoothing, pass-through).
    #[inline]
    pub fn new(smoothing: f64) -> Self {
        Self {
            smoothed_secs: 0.0,
            smoothing: smoothing.clamp(0.0, 1.0),
            initialized: false,
        }
    }

    /// Fold `delta` into the moving average. The first sample initializes the
    /// average directly so startup is not biased toward zero. Returns `self`.
    #[inline]
    pub fn add(&mut self, delta: Duration) -> &Self {
        let sample = delta.as_secs_f64();
        if self.initialized {
            self.smoothed_secs += self.smoothing * (sample - self.smoothed_secs);
        } else {
            self.smoothed_secs = sample;
            self.initialized = true;
        }
        self
    }

    /// The current smoothed delta.
    #[inline]
    pub fn smoothed(&self) -> Duration {
        Duration::from_secs_f64(self.smoothed_secs.max(0.0))
    }

    /// The current smoothed delta in seconds (`f32`).
    #[inline]
    pub fn smoothed_secs(&self) -> f32 {
        self.smoothed_secs as f32
    }

    /// The current smoothed delta in seconds (`f64`).
    #[inline]
    pub fn smoothed_secs_f64(&self) -> f64 {
        self.smoothed_secs
    }

    /// Smoothed frames-per-second (`0.0` before the first sample or if the
    /// smoothed delta is zero).
    #[inline]
    pub fn smoothed_fps(&self) -> f64 {
        if self.initialized && self.smoothed_secs > 0.0 {
            1.0 / self.smoothed_secs
        } else {
            0.0
        }
    }

    /// Whether at least one sample has been folded in.
    #[inline]
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Reset to the uninitialized state; the next sample reinitializes.
    #[inline]
    pub fn reset(&mut self) {
        self.smoothed_secs = 0.0;
        self.initialized = false;
    }
}

impl Default for SmoothedDelta {
    #[inline]
    fn default() -> Self {
        Self::new(Self::DEFAULT_SMOOTHING)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_sample_initializes_directly() {
        let mut s = SmoothedDelta::new(0.1);
        assert!(!s.is_initialized());
        s.add(Duration::from_millis(16));
        assert!(s.is_initialized());
        assert!((s.smoothed_secs() - 0.016).abs() < 1e-6);
    }

    #[test]
    fn ema_converges_toward_steady_input() {
        let mut s = SmoothedDelta::new(0.25);
        s.add(Duration::from_millis(10));
        for _ in 0..200 {
            s.add(Duration::from_millis(20));
        }
        // Converges close to the steady 20 ms input.
        assert!((s.smoothed_secs_f64() - 0.020).abs() < 1e-4);
        assert!((s.smoothed_fps() - 50.0).abs() < 0.5);
    }

    #[test]
    fn pass_through_at_full_weight() {
        let mut s = SmoothedDelta::new(1.0);
        s.add(Duration::from_millis(10));
        s.add(Duration::from_millis(33));
        assert!((s.smoothed_secs_f64() - 0.033).abs() < 1e-9);
    }

    #[test]
    fn reset_requires_reinit() {
        let mut s = SmoothedDelta::default();
        s.add(Duration::from_millis(16));
        s.reset();
        assert!(!s.is_initialized());
        assert_eq!(s.smoothed_fps(), 0.0);
    }
}
