//! Sample-accurate parameter smoothing.
//!
//! Applying a control change (volume, filter cutoff, pan) instantaneously to a
//! per-sample gain produces an audible click known as *zipper noise*. Every
//! automatable parameter in the engine is therefore driven through a
//! [`Smoothed`] value that ramps toward its target over a configurable number
//! of samples.

use bevy_math::ops;

use crate::math::Sample;

/// The interpolation shape used when a parameter moves toward a new target.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum Ramp {
    /// Jump immediately to the target (still click-safe only for values that
    /// are not applied per-sample, e.g. an integer mode switch).
    Immediate,
    /// Move linearly to the target over the given number of samples.
    Linear {
        /// Number of samples the linear ramp spans.
        samples: u32,
    },
    /// Move exponentially (one-pole) toward the target with the given
    /// time-constant in samples. Never mathematically reaches the target, so a
    /// snap threshold is applied internally.
    Exponential {
        /// One-pole time constant in samples.
        tau_samples: f32,
    },
}

impl Ramp {
    /// Builds a linear ramp from a duration in seconds and a sample rate.
    #[inline]
    #[must_use]
    pub fn linear_seconds(seconds: f32, sample_rate: u32) -> Self {
        let samples = ops::round(seconds.max(0.0) * sample_rate as f32) as u32;
        Ramp::Linear {
            samples: samples.max(1),
        }
    }
}

/// A scalar parameter that glides toward its target on every sample tick.
///
/// The whole struct is `Copy` and holds no heap data, so it is safe to embed
/// directly inside real-time [`AudioNode`](crate::graph::AudioNode)s.
#[derive(Debug, Clone, Copy)]
pub struct Smoothed {
    current: Sample,
    target: Sample,
    /// Per-sample increment for linear ramps.
    step: Sample,
    /// Remaining samples for linear ramps.
    remaining: u32,
    /// One-pole coefficient for exponential ramps (0 disables).
    pole: Sample,
}

impl Smoothed {
    /// Creates a smoothed value that starts settled at `initial`.
    #[inline]
    #[must_use]
    pub fn new(initial: Sample) -> Self {
        Self {
            current: initial,
            target: initial,
            step: 0.0,
            remaining: 0,
            pole: 0.0,
        }
    }

    /// Returns the instantaneous value without advancing.
    #[inline]
    #[must_use]
    pub fn current(&self) -> Sample {
        self.current
    }

    /// Returns the value the parameter is gliding toward.
    #[inline]
    #[must_use]
    pub fn target(&self) -> Sample {
        self.target
    }

    /// Returns `true` when the value has reached its target.
    #[inline]
    #[must_use]
    pub fn is_settled(&self) -> bool {
        self.remaining == 0 && self.pole == 0.0
    }

    /// Sets a new target using the given ramp shape.
    pub fn set_target(&mut self, target: Sample, ramp: Ramp) {
        self.target = target;
        match ramp {
            Ramp::Immediate => {
                self.current = target;
                self.remaining = 0;
                self.step = 0.0;
                self.pole = 0.0;
            }
            Ramp::Linear { samples } => {
                let n = samples.max(1);
                self.remaining = n;
                self.step = (target - self.current) / n as Sample;
                self.pole = 0.0;
            }
            Ramp::Exponential { tau_samples } => {
                // One-pole coefficient: exp(-1 / tau).
                let tau = tau_samples.max(1.0);
                self.pole = exp_neg_inv(tau);
                self.remaining = 0;
                self.step = 0.0;
            }
        }
    }

    /// Advances the value by one sample and returns the new instantaneous
    /// value. Safe to call in the innermost DSP loop.
    #[inline]
    pub fn next_sample(&mut self) -> Sample {
        if self.remaining > 0 {
            self.current += self.step;
            self.remaining -= 1;
            if self.remaining == 0 {
                self.current = self.target;
            }
        } else if self.pole != 0.0 {
            self.current = self.target + (self.current - self.target) * self.pole;
            if (self.current - self.target).abs() < 1.0e-6 {
                self.current = self.target;
                self.pole = 0.0;
            }
        }
        self.current
    }
}

#[inline]
fn exp_neg_inv(tau: f32) -> f32 {
    // exp(-1/tau) via the deterministic, cross-platform `bevy_math::ops`.
    ops::exp(-1.0 / tau)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[test]
    fn immediate_snaps() {
        let mut s = Smoothed::new(0.0);
        s.set_target(1.0, Ramp::Immediate);
        assert_eq!(s.current(), 1.0);
        assert!(s.is_settled());
    }

    #[test]
    fn linear_reaches_target_exactly() {
        let mut s = Smoothed::new(0.0);
        s.set_target(1.0, Ramp::Linear { samples: 4 });
        let vals: Vec<f32> = (0..4).map(|_| s.next_sample()).collect();
        assert!((vals[0] - 0.25).abs() < 1e-6);
        assert_eq!(vals[3], 1.0);
        assert!(s.is_settled());
    }

    #[test]
    fn exponential_converges() {
        let mut s = Smoothed::new(0.0);
        s.set_target(1.0, Ramp::Exponential { tau_samples: 8.0 });
        for _ in 0..512 {
            s.next_sample();
        }
        assert!((s.current() - 1.0).abs() < 1e-3);
    }
}
