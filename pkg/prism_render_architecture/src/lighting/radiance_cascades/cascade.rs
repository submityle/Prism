//! A single cascade: `probes × directions` radiance intervals plus the gather
//! that fills them from a scene.
//!
//! The scene is abstracted behind [`SceneSampler`] so this module stays
//! device-free and deterministic: a sampler answers "what radiance interval
//! does a ray from `origin` in `dir` accumulate over `[t0, t1)`?". Production
//! runtimes implement it with a ray march over an SDF / voxel / mesh scene; the
//! golden tests implement it analytically so the gather can be checked exactly.
//!
//! # Provenance
//! The per-probe directional interval store and the radial gather are from
//! Alexander Sannikov's *Radiance Cascades* (2023). Clean-room classical code.
//!
//! No Unreal Engine source is used anywhere in this module.

use alloc::vec::Vec;

use glam::Vec2;

use super::hierarchy::CascadeHierarchy;
use super::interval::RadianceInterval;

/// Scene query backing the cascade gather.
///
/// Implementations must be deterministic (same inputs ⇒ same output) so cascade
/// solves reproduce in golden tests and a future GPU twin. `t0 <= t1` always,
/// both measured along the unit vector `dir` from `origin`.
pub trait SceneSampler {
    /// Radiance and transmittance accumulated along the ray `origin + t*dir`
    /// for `t` in `[t0, t1)`.
    fn sample_interval(&self, origin: Vec2, dir: Vec2, t0: f32, t1: f32) -> RadianceInterval;
}

/// One cascade's directional radiance field.
///
/// `data` is laid out probe-major: the interval for probe `(col, row)` in
/// direction `dir` lives at `(row * cols + col) * angular + dir`.
#[derive(Clone, Debug)]
pub struct Cascade {
    level: u32,
    cols: u32,
    rows: u32,
    angular: u32,
    data: Vec<RadianceInterval>,
}

impl Cascade {
    /// Cascade level index (0 is the finest-spatial, coarsest-angular).
    #[must_use]
    pub fn level(&self) -> u32 {
        self.level
    }

    /// Probe grid dimensions `(cols, rows)`.
    #[must_use]
    pub fn dims(&self) -> (u32, u32) {
        (self.cols, self.rows)
    }

    /// Angular bin count per probe.
    #[must_use]
    pub fn angular(&self) -> u32 {
        self.angular
    }

    /// Flat index of probe `(col, row)` direction `dir`.
    #[inline]
    fn index(&self, col: u32, row: u32, dir: u32) -> usize {
        (((row * self.cols) + col) * self.angular + dir) as usize
    }

    /// Interval stored for probe `(col, row)` in direction `dir`.
    #[must_use]
    pub fn get(&self, col: u32, row: u32, dir: u32) -> RadianceInterval {
        self.data[self.index(col, row, dir)]
    }

    /// Overwrite the interval for probe `(col, row)` in direction `dir`.
    pub fn set(&mut self, col: u32, row: u32, dir: u32, value: RadianceInterval) {
        let i = self.index(col, row, dir);
        self.data[i] = value;
    }

    /// Allocate a cleared cascade sized for `level` of `hierarchy`.
    #[must_use]
    pub fn cleared(hierarchy: &CascadeHierarchy, level: u32) -> Self {
        let (cols, rows) = hierarchy.probe_dims(level);
        let angular = hierarchy.angular_count(level);
        let len = (cols * rows * angular) as usize;
        Self {
            level,
            cols,
            rows,
            angular,
            data: alloc::vec![RadianceInterval::CLEAR; len],
        }
    }

    /// Gather cascade `level`: for every probe and direction, ask `sampler` for
    /// the radiance interval over this cascade's radial shell.
    #[must_use]
    pub fn gather<S: SceneSampler>(
        hierarchy: &CascadeHierarchy,
        level: u32,
        sampler: &S,
    ) -> Self {
        let mut cascade = Self::cleared(hierarchy, level);
        let t0 = hierarchy.interval_start(level);
        let t1 = hierarchy.interval_end(level);
        for row in 0..cascade.rows {
            for col in 0..cascade.cols {
                let origin = hierarchy.probe_position(level, col, row);
                for dir in 0..cascade.angular {
                    let angle = hierarchy.bin_angle(level, dir);
                    // glam evaluates the trig internally (external crate), so this
                    // stays clear of the workspace `f32::cos`/`f32::sin` lint.
                    let d = Vec2::from_angle(angle);
                    let interval = sampler.sample_interval(origin, d, t0, t1);
                    cascade.set(col, row, dir, interval);
                }
            }
        }
        cascade
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    /// Homogeneous emissive–absorbing medium: emission `e` and absorption `a`
    /// per unit length, uniform everywhere. Over `[t0, t1)` of length `L` the
    /// analytic transmittance is `exp(-a L)` and the premultiplied radiance is
    /// `(e / a)(1 - exp(-a L))`. These composite exactly under `over`, so the
    /// medium is an ideal oracle for both gather and merge.
    struct Medium {
        emission: Vec3,
        absorption: f32,
    }
    impl SceneSampler for Medium {
        fn sample_interval(&self, _o: Vec2, _d: Vec2, t0: f32, t1: f32) -> RadianceInterval {
            // Rational emissive-absorbing medium with survival S(t)=1/(1+a t)^2.
            // Transmittance over [t0,t1) is S(t1)/S(t0) and premultiplied radiance
            // integrates e*S exactly to a closed rational form, so sub-intervals
            // composite exactly under `over` with no exp/trig (keeping the golden
            // within the workspace libm-determinism lint).
            let a = self.absorption;
            let s0 = 1.0 + a * t0.max(0.0);
            let s1 = 1.0 + a * t1.max(0.0);
            let ratio = s0 / s1;
            let tr = ratio * ratio;
            let rad = self.emission * ((s0 / a) * (1.0 - ratio));
            RadianceInterval::new(rad, tr)
        }
    }

    fn hierarchy() -> CascadeHierarchy {
        CascadeHierarchy {
            origin: Vec2::ZERO,
            base_spacing: 1.0,
            base_cols: 8,
            base_rows: 8,
            base_angular: 4,
            base_interval: 1.0,
            levels: 3,
        }
    }

    #[test]
    fn cleared_cascade_has_expected_shape() {
        let h = hierarchy();
        let c = Cascade::cleared(&h, 1);
        assert_eq!(c.dims(), (4, 4));
        assert_eq!(c.angular(), 16);
        for row in 0..4 {
            for col in 0..4 {
                for dir in 0..16 {
                    assert_eq!(c.get(col, row, dir), RadianceInterval::CLEAR);
                }
            }
        }
    }

    #[test]
    fn gather_matches_the_analytic_medium() {
        let h = hierarchy();
        let m = Medium {
            emission: Vec3::new(2.0, 1.0, 0.5),
            absorption: 0.3,
        };
        let level = 1;
        let c = Cascade::gather(&h, level, &m);
        let a = m.absorption;
        let s0 = 1.0 + a * h.interval_start(level);
        let s1 = 1.0 + a * h.interval_end(level);
        let ratio = s0 / s1;
        let tr = ratio * ratio;
        let rad = m.emission * ((s0 / a) * (1.0 - ratio));
        for row in 0..c.dims().1 {
            for col in 0..c.dims().0 {
                for dir in 0..c.angular() {
                    let got = c.get(col, row, dir);
                    assert!((got.transmittance - tr).abs() <= 1.0e-5);
                    assert!((got.radiance - rad).abs().max_element() <= 1.0e-4);
                }
            }
        }
    }

    #[test]
    fn set_then_get_round_trips() {
        let h = hierarchy();
        let mut c = Cascade::cleared(&h, 0);
        let v = RadianceInterval::new(Vec3::new(0.1, 0.2, 0.3), 0.4);
        c.set(3, 5, 2, v);
        assert_eq!(c.get(3, 5, 2), v);
        assert_eq!(c.get(3, 5, 1), RadianceInterval::CLEAR);
    }
}
