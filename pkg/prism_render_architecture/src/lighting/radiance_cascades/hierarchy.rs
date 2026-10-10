//! Cascade hierarchy dimensioning and its invariants.
//!
//! A radiance cascade hierarchy trades spatial resolution for angular
//! resolution as the radial interval grows. Starting from cascade 0:
//!
//! - **Angular resolution grows** by [`ANGULAR_SCALE`] (×4) per level, because a
//!   longer interval reaches farther and needs finer directions to resolve the
//!   same angular feature size.
//! - **Spatial resolution shrinks** by [`SPATIAL_SCALE`] (×2 probe spacing per
//!   axis) per level, because far radiance varies slowly across space.
//! - **The radial interval grows** by [`INTERVAL_SCALE`] (×4) per level, so each
//!   cascade covers a geometrically larger shell of distance.
//!
//! The elegant consequence (the "constant-memory" property of radiance
//! cascades) is that in 2D the probe *count* falls by 4× exactly as the
//! direction count rises by 4×, so **every cascade stores the same number of
//! ray records**. Intervals are also laid out contiguously: the end of
//! cascade `n` is the start of cascade `n+1`, so compositing adjacent cascades
//! reconstructs an unbroken gather from the camera plane outward. Both facts
//! are exact for power-of-two base grids and are asserted in the golden tests.
//!
//! # Provenance
//! The ×4 angular / ×4 interval / ÷4 probe-count scaling is the dimensioning
//! rule from Alexander Sannikov's *Radiance Cascades* (2023). Clean-room
//! classical formulation; no neural or data-driven components.
//!
//! # References
//! - A. Sannikov, *Radiance Cascades: A Novel Approach to Calculating Global
//!   Illumination* (2023).
//!
//! No Unreal Engine source is used anywhere in this module.

use core::f32::consts::TAU;

/// Deterministic `base^exp` for a small non-negative integer exponent.
///
/// Used instead of `f32::powi` so the hierarchy dimensioning stays within the
/// workspace's libm-determinism lint; repeated multiplication is exact for the
/// small exponents a cascade stack uses.
fn pow_u32(base: f32, exp: u32) -> f32 {
    let mut acc = 1.0_f32;
    for _ in 0..exp {
        acc *= base;
    }
    acc
}

/// Probe spacing doubles per axis each cascade (×2 ⇒ ÷4 probe count in 2D).
pub const SPATIAL_SCALE: u32 = 2;

/// Angular bin count multiplies by this each cascade.
pub const ANGULAR_SCALE: u32 = 4;

/// Radial interval length multiplies by this each cascade.
pub const INTERVAL_SCALE: f32 = 4.0;

/// Immutable description of a cascade hierarchy over a rectangular probe field.
///
/// The base (cascade 0) grid is `base_cols × base_rows` probes spaced
/// `base_spacing` world units apart, with lower-left probe centre at `origin`.
/// Each probe fans out `base_angular` directions covering the radial interval
/// `[0, base_interval)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CascadeHierarchy {
    /// World-space position of the cascade-0 probe at grid index `(0, 0)`.
    pub origin: glam::Vec2,
    /// Cascade-0 probe spacing in world units (both axes).
    pub base_spacing: f32,
    /// Cascade-0 probe columns (x). Should be a multiple of
    /// `SPATIAL_SCALE^(levels-1)` for exact down-scaling.
    pub base_cols: u32,
    /// Cascade-0 probe rows (y).
    pub base_rows: u32,
    /// Cascade-0 angular bin count (directions per probe).
    pub base_angular: u32,
    /// Cascade-0 radial interval length in world units.
    pub base_interval: f32,
    /// Number of cascades, `>= 1`. Cascade indices are `0..levels`.
    pub levels: u32,
}

impl CascadeHierarchy {
    /// Angular bin count at cascade `level`.
    #[must_use]
    pub fn angular_count(&self, level: u32) -> u32 {
        self.base_angular * ANGULAR_SCALE.pow(level)
    }

    /// Probe spacing (world units) at cascade `level`.
    #[must_use]
    pub fn spacing(&self, level: u32) -> f32 {
        self.base_spacing * (SPATIAL_SCALE.pow(level) as f32)
    }

    /// Probe grid dimensions `(cols, rows)` at cascade `level`.
    ///
    /// Each axis is divided by `SPATIAL_SCALE^level`, floored but never below 1
    /// so a coarse cascade always keeps at least one probe.
    #[must_use]
    pub fn probe_dims(&self, level: u32) -> (u32, u32) {
        let d = SPATIAL_SCALE.pow(level);
        ((self.base_cols / d).max(1), (self.base_rows / d).max(1))
    }

    /// Total probe count at cascade `level`.
    #[must_use]
    pub fn probe_count(&self, level: u32) -> u32 {
        let (c, r) = self.probe_dims(level);
        c * r
    }

    /// Number of stored ray records (`probes × directions`) at `level`.
    ///
    /// For power-of-two base grids this is identical across all cascades — the
    /// constant-memory property.
    #[must_use]
    pub fn rays(&self, level: u32) -> u32 {
        self.probe_count(level) * self.angular_count(level)
    }

    /// Start distance of the radial interval covered by cascade `level`.
    ///
    /// Intervals are contiguous: `interval_start(n+1) == interval_end(n)`. The
    /// closed form is the geometric partial sum
    /// `base * (INTERVAL_SCALE^level - 1) / (INTERVAL_SCALE - 1)`.
    #[must_use]
    pub fn interval_start(&self, level: u32) -> f32 {
        if level == 0 {
            return 0.0;
        }
        self.base_interval * (pow_u32(INTERVAL_SCALE, level) - 1.0) / (INTERVAL_SCALE - 1.0)
    }

    /// Length of the radial interval covered by cascade `level`.
    #[must_use]
    pub fn interval_length(&self, level: u32) -> f32 {
        self.base_interval * pow_u32(INTERVAL_SCALE, level)
    }

    /// End distance of the radial interval covered by cascade `level`.
    #[must_use]
    pub fn interval_end(&self, level: u32) -> f32 {
        self.interval_start(level) + self.interval_length(level)
    }

    /// Centre angle (radians) of angular bin `dir` at cascade `level`.
    ///
    /// Bins are the half-open sectors `[dir, dir+1) * TAU / count`; the centre
    /// uses the `+0.5` convention so that a child bin's centre is the exact
    /// average of its [`ANGULAR_SCALE`] parent sub-bin centres, which keeps the
    /// angular merge unbiased.
    #[must_use]
    pub fn bin_angle(&self, level: u32, dir: u32) -> f32 {
        let count = self.angular_count(level);
        TAU * (dir as f32 + 0.5) / (count as f32)
    }

    /// World-space position of probe `(col, row)` at cascade `level`.
    #[must_use]
    pub fn probe_position(&self, level: u32, col: u32, row: u32) -> glam::Vec2 {
        let s = self.spacing(level);
        self.origin + glam::Vec2::new((col as f32 + 0.5) * s, (row as f32 + 0.5) * s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hierarchy() -> CascadeHierarchy {
        CascadeHierarchy {
            origin: glam::Vec2::ZERO,
            base_spacing: 1.0,
            base_cols: 64,
            base_rows: 64,
            base_angular: 4,
            base_interval: 1.0,
            levels: 4,
        }
    }

    #[test]
    fn ray_count_is_constant_across_cascades() {
        let h = hierarchy();
        let base = h.rays(0);
        for level in 0..h.levels {
            assert_eq!(h.rays(level), base, "level {level}");
        }
    }

    #[test]
    fn angular_count_quadruples_per_level() {
        let h = hierarchy();
        for level in 1..h.levels {
            assert_eq!(h.angular_count(level), h.angular_count(level - 1) * 4);
        }
    }

    #[test]
    fn probe_spacing_doubles_per_level() {
        let h = hierarchy();
        for level in 1..h.levels {
            assert!((h.spacing(level) - h.spacing(level - 1) * 2.0).abs() <= 1.0e-6);
        }
    }

    #[test]
    fn intervals_are_contiguous() {
        let h = hierarchy();
        for level in 0..h.levels - 1 {
            let end = h.interval_end(level);
            let next = h.interval_start(level + 1);
            assert!(
                (end - next).abs() <= 1.0e-4,
                "level {level}: {end} vs {next}"
            );
        }
    }

    #[test]
    fn interval_start_matches_geometric_partial_sum() {
        let h = hierarchy();
        // base=1, scale=4 ⇒ starts are 0, 1, 5, 21.
        assert!((h.interval_start(0) - 0.0).abs() <= 1.0e-6);
        assert!((h.interval_start(1) - 1.0).abs() <= 1.0e-6);
        assert!((h.interval_start(2) - 5.0).abs() <= 1.0e-6);
        assert!((h.interval_start(3) - 21.0).abs() <= 1.0e-6);
    }

    #[test]
    fn child_bin_centre_is_the_mean_of_its_parent_subbins() {
        let h = hierarchy();
        for level in 0..h.levels - 1 {
            let child_count = h.angular_count(level);
            for d in 0..child_count {
                let child = h.bin_angle(level, d);
                let mut sum = 0.0;
                for k in 0..ANGULAR_SCALE {
                    sum += h.bin_angle(level + 1, d * ANGULAR_SCALE + k);
                }
                let mean = sum / (ANGULAR_SCALE as f32);
                assert!((child - mean).abs() <= 1.0e-5, "level {level} dir {d}");
            }
        }
    }

    #[test]
    fn coarse_cascade_keeps_at_least_one_probe() {
        let h = CascadeHierarchy {
            base_cols: 2,
            base_rows: 2,
            levels: 6,
            ..hierarchy()
        };
        assert_eq!(h.probe_dims(5), (1, 1));
    }
}
