//! `CPU` golden twin for the `GPU` `CFL` reduction.
//!
//! [`cpu_max_speed`] performs the same reduction the `WGSL` kernel does — the
//! maximum squared speed over the velocity field, followed by a single square
//! root — so a passing real-device parity test is direct evidence that the
//! ported kernel reduces to the same value, not merely that its shader
//! compiled. [`cpu_cfl_dt`] then feeds that speed through
//! [`CflConfig::suggest_dt`] to obtain the adaptive time step.
//!
//! # Why max of squared speed
//!
//! Taking the maximum of the squared speeds and rooting once at the end is
//! identical to rooting each speed and taking the maximum, because the square
//! root is monotone. Reducing the squared speed keeps the whole reduction free
//! of per-element roots, so the device kernel can fold non-negative values with
//! a single integer `atomicMax` over their bit patterns.
//!
//! # Provenance
//!
//! The `CFL` condition is a classical, openly published stability criterion.
//! This module contains no Unreal Engine source or derived code.

use glam::Vec3;

use super::config::CflConfig;

/// Returns the largest speed in `velocities`, or `0.0` when the slice is empty.
///
/// The reduction folds the squared speed `v · v` of every entry and takes one
/// square root at the end, matching the device kernel's fold-then-root order.
#[must_use]
pub fn cpu_max_speed(velocities: &[Vec3]) -> f32 {
    let mut max_sq = 0.0_f32;
    for v in velocities {
        max_sq = max_sq.max(v.dot(*v));
    }
    max_sq.sqrt()
}

/// Suggests a `CFL`-stable time step for the scene described by `velocities`.
///
/// A thin composition of [`cpu_max_speed`] and [`CflConfig::suggest_dt`]; it is
/// the reference the `GPU` [`suggest_dt`](super::gpu::GpuCflReduce::suggest_dt)
/// is checked against.
#[must_use]
pub fn cpu_cfl_dt(velocities: &[Vec3], config: &CflConfig) -> f32 {
    config.suggest_dt(cpu_max_speed(velocities))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Relative tolerance for the analytic speed checks; the reduction is a
    /// plain maximum plus one square root, so only the root contributes error.
    const TOL: f32 = 1.0e-5;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= TOL * a.abs().max(1.0)
    }

    #[test]
    fn empty_field_has_zero_speed() {
        assert_eq!(cpu_max_speed(&[]), 0.0);
    }

    #[test]
    fn single_velocity_is_its_own_speed() {
        // 3-4-0 is a 3-4-5 triangle, so the speed is exactly 5.
        let speed = cpu_max_speed(&[Vec3::new(3.0, 4.0, 0.0)]);
        assert!(close(speed, 5.0), "speed {speed}");
    }

    #[test]
    fn reports_the_fastest_of_many() {
        let field = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 3.0, 4.0),
            Vec3::new(2.0, 0.0, 0.0),
        ];
        // The middle entry has speed 5, the others 1 and 2.
        assert!(close(cpu_max_speed(&field), 5.0));
    }

    #[test]
    fn a_still_scene_takes_the_largest_step() {
        let cfg = CflConfig::default();
        assert_eq!(cpu_cfl_dt(&[], &cfg), cfg.dt_max);
        assert_eq!(cpu_cfl_dt(&[Vec3::ZERO, Vec3::ZERO], &cfg), cfg.dt_max);
    }

    #[test]
    fn a_fast_scene_is_clamped_to_the_floor() {
        let cfg = CflConfig::default();
        // A very fast particle drives the raw step below dt_min.
        let dt = cpu_cfl_dt(&[Vec3::new(1.0e6, 0.0, 0.0)], &cfg);
        assert_eq!(dt, cfg.dt_min);
    }

    #[test]
    fn a_moderate_scene_follows_the_formula() {
        let cfg = CflConfig {
            cfl_number: 0.5,
            cell_size: 1.0,
            dt_min: 1.0e-6,
            dt_max: 1.0,
        };
        // Speed 10 gives 0.5 * 1 / 10 = 0.05, inside [1e-6, 1].
        let dt = cpu_cfl_dt(&[Vec3::new(6.0, 8.0, 0.0)], &cfg);
        assert!(close(dt, 0.05), "dt {dt}");
    }
}
