//! Arrival-direction encoding: a pseudo-intensity estimate of where the early
//! sound energy comes from, built from a probe cell and its six axis
//! neighbours.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Microsoft Project Acoustics, or Google Resonance Audio source or derived
//! code; no AI/ML.
//!
//! # Relationship
//! Implements the "initial arrival direction and energy -> azimuth / elevation"
//! measure of design section 43. Particle velocity is approximated from the
//! finite-difference pressure gradient (an Euler integration of the six
//! neighbour impulse responses); the time-domain acoustic intensity then
//! points along the energy flow, and the arrival direction is its negation.
//! Angles use the same listener-local convention as
//! [`prism_audio_spatial::SpatialParams`].

use bevy_math::{ops, Vec3};

use crate::solver::ImpulseResponse;

/// A probe cell together with its six axis-neighbour impulse responses, used
/// to estimate the arrival direction of the early energy.
///
/// All seven responses are expected to share a sample rate and length; shorter
/// neighbours are simply read as silent past their end.
pub struct DirectionalProbe<'a> {
    /// Impulse response at the probe cell itself.
    pub center: &'a ImpulseResponse,
    /// Neighbour one cell in the negative `x` direction.
    pub x_neg: &'a ImpulseResponse,
    /// Neighbour one cell in the positive `x` direction.
    pub x_pos: &'a ImpulseResponse,
    /// Neighbour one cell in the negative `y` direction.
    pub y_neg: &'a ImpulseResponse,
    /// Neighbour one cell in the positive `y` direction.
    pub y_pos: &'a ImpulseResponse,
    /// Neighbour one cell in the negative `z` direction.
    pub z_neg: &'a ImpulseResponse,
    /// Neighbour one cell in the positive `z` direction.
    pub z_pos: &'a ImpulseResponse,
}

impl DirectionalProbe<'_> {
    /// Estimates the unit arrival direction (pointing from the listener toward
    /// the source) over the early window `[onset, onset + window)`.
    ///
    /// Returns [`Vec3::ZERO`] when the field carries no directional energy, for
    /// instance when every neighbour is identical to the centre.
    #[must_use]
    pub fn arrival_direction(&self, onset: usize, window: usize) -> Vec3 {
        let center = self.center.samples();
        let end = onset.saturating_add(window.max(1)).min(center.len());
        if onset >= end {
            return Vec3::ZERO;
        }

        let xn = self.x_neg.samples();
        let xp = self.x_pos.samples();
        let yn = self.y_neg.samples();
        let yp = self.y_pos.samples();
        let zn = self.z_neg.samples();
        let zp = self.z_pos.samples();
        let at = |s: &[f32], i: usize| -> f32 { s.get(i).copied().unwrap_or(0.0) };

        // Euler integration of the pressure gradient gives a particle-velocity
        // proxy; the running intensity is pressure times that velocity.
        let mut vx = 0.0_f32;
        let mut vy = 0.0_f32;
        let mut vz = 0.0_f32;
        let mut intensity = Vec3::ZERO;
        for (i, &p) in center.iter().enumerate().take(end).skip(onset) {
            vx += -(at(xp, i) - at(xn, i));
            vy += -(at(yp, i) - at(yn, i));
            vz += -(at(zp, i) - at(zn, i));
            intensity += Vec3::new(p * vx, p * vy, p * vz);
        }

        // Intensity points along the energy flow (source -> listener); the
        // arrival direction is the opposite.
        let flow = -intensity;
        let len = ops::sqrt(flow.dot(flow));
        if len <= 1.0e-20 {
            Vec3::ZERO
        } else {
            flow / len
        }
    }
}

/// Converts a (not necessarily normalised) direction into listener-local
/// `(azimuth, elevation)` radians, matching the
/// [`prism_audio_spatial`] convention (`+x` right, `+y` up, `-z` forward).
///
/// A zero direction maps to straight ahead, `(0, 0)`.
#[must_use]
pub fn direction_to_angles(dir: Vec3) -> (f32, f32) {
    if dir.dot(dir) <= 1.0e-20 {
        return (0.0, 0.0);
    }
    let azimuth = ops::atan2(dir.x, -dir.z);
    let horizontal = ops::sqrt(dir.x * dir.x + dir.z * dir.z);
    let elevation = ops::atan2(dir.y, horizontal);
    (azimuth, elevation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use core::f32::consts::PI;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    fn pulse(sample_rate: f32, len: usize, at: usize) -> ImpulseResponse {
        let mut p = vec![0.0_f32; len];
        if at < len {
            p[at] = 1.0;
        }
        ImpulseResponse::new(sample_rate, p)
    }

    #[test]
    fn source_on_the_right_reads_positive_azimuth() {
        // Source toward +x: the +x neighbour (nearer the source) leads the -x
        // neighbour in time. y and z neighbours match the centre (no gradient).
        let sr = 1000.0;
        let center = pulse(sr, 32, 10);
        let x_pos = pulse(sr, 32, 9);
        let x_neg = pulse(sr, 32, 11);
        let flat = pulse(sr, 32, 10);
        let probe = DirectionalProbe {
            center: &center,
            x_neg: &x_neg,
            x_pos: &x_pos,
            y_neg: &flat,
            y_pos: &flat,
            z_neg: &flat,
            z_pos: &flat,
        };
        let dir = probe.arrival_direction(0, 32);
        assert!(dir.x > 0.0, "expected +x arrival, got {dir:?}");
        let (az, el) = direction_to_angles(dir);
        assert!(approx(az, PI / 2.0, 0.2), "azimuth {az}");
        assert!(approx(el, 0.0, 0.2), "elevation {el}");
    }

    #[test]
    fn no_gradient_gives_zero_direction() {
        let sr = 1000.0;
        let flat = pulse(sr, 16, 5);
        let probe = DirectionalProbe {
            center: &flat,
            x_neg: &flat,
            x_pos: &flat,
            y_neg: &flat,
            y_pos: &flat,
            z_neg: &flat,
            z_pos: &flat,
        };
        let dir = probe.arrival_direction(0, 16);
        assert_eq!(dir, Vec3::ZERO);
        assert_eq!(direction_to_angles(dir), (0.0, 0.0));
    }

    #[test]
    fn forward_source_is_near_zero_azimuth() {
        let (az, el) = direction_to_angles(Vec3::new(0.0, 0.0, -1.0));
        assert!(approx(az, 0.0, 1e-5));
        assert!(approx(el, 0.0, 1e-5));
    }
}
