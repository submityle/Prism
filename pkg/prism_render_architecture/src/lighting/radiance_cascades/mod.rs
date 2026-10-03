//! Radiance Cascades — a classical, hierarchy-based global-illumination solver.
//!
//! Radiance cascades compute diffuse (and low-gloss) global illumination by
//! sampling radiance on a hierarchy of probe grids. Each successive cascade
//! trades spatial resolution for angular resolution while covering a radially
//! larger shell, so near field is resolved with many probes/few directions and
//! far field with few probes/many directions. Merging the shells
//! front-to-back reconstructs a full-range directional radiance field at
//! cascade 0, which [`resolve`] integrates into per-probe irradiance.
//!
//! The design intentionally mirrors Prism's other GI modules
//! ([`super::restir_gi`], [`super::regir`]): a device-free CPU golden with a
//! scene abstraction ([`SceneSampler`]) and exhaustive invariants, ready for a
//! separate `*_gpu` wgpu twin to be validated against it.
//!
//! # Pipeline
//! 1. [`CascadeHierarchy`] fixes probe grids, angular counts, and the
//!    contiguous radial intervals, with the ×4 angular / ÷4 probe-count /
//!    ×4 interval scaling that keeps per-cascade memory constant.
//! 2. [`Cascade::gather`] fills each cascade's `probes × directions` radiance
//!    intervals from a [`SceneSampler`].
//! 3. [`merge::merge_into`] folds the hierarchy top-down, compositing each
//!    coarse cascade into its finer child (bilinear in space, averaged over the
//!    ×4 angular sub-bins).
//! 4. [`resolve::mean_radiance`] / [`resolve::fluence`] integrate cascade 0 over
//!    direction for shading.
//!
//! [`solve`] runs steps 2–3 and returns the merged cascade 0.
//!
//! # Provenance
//! Algorithm after Alexander Sannikov, *Radiance Cascades: A Novel Approach to
//! Calculating Global Illumination* (2023). Compositing is Porter–Duff "over"
//! (1984). This is a clean-room classical implementation: pure deterministic
//! quadrature with no neural, learned, or data-driven components.
//!
//! No Unreal Engine source is used anywhere in this module.

pub mod cascade;
pub mod hierarchy;
pub mod interval;
pub mod merge;
pub mod resolve;

pub use cascade::{Cascade, SceneSampler};
pub use hierarchy::{CascadeHierarchy, ANGULAR_SCALE, INTERVAL_SCALE, SPATIAL_SCALE};
pub use interval::RadianceInterval;

/// Gather every cascade and fold the hierarchy top-down, returning the merged
/// cascade 0 whose directional radiance covers the full radial range.
///
/// Equivalent to gathering each level with [`Cascade::gather`] and compositing
/// coarse-into-fine with [`merge::merge_into`] from the top cascade down to
/// cascade 0.
#[must_use]
pub fn solve<S: SceneSampler>(hierarchy: &CascadeHierarchy, sampler: &S) -> Cascade {
    let top = hierarchy.levels.saturating_sub(1);
    let mut acc = Cascade::gather(hierarchy, top, sampler);
    let mut level = top;
    while level > 0 {
        level -= 1;
        let mut child = Cascade::gather(hierarchy, level, sampler);
        merge::merge_into(hierarchy, &mut child, &acc);
        acc = child;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Vec2, Vec3};

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

    #[test]
    fn solve_returns_cascade_zero_covering_the_full_range() {
        let h = CascadeHierarchy {
            origin: Vec2::ZERO,
            base_spacing: 1.0,
            base_cols: 8,
            base_rows: 8,
            base_angular: 4,
            base_interval: 1.0,
            levels: 4,
        };
        let m = Medium {
            emission: Vec3::new(1.0, 2.0, 3.0),
            absorption: 0.25,
        };
        let merged = solve(&h, &m);
        assert_eq!(merged.level(), 0);
        assert_eq!(merged.dims(), (8, 8));
        assert_eq!(merged.angular(), 4);

        let a = m.absorption;
        let s0 = 1.0 + a * h.interval_start(0);
        let s1 = 1.0 + a * h.interval_end(h.levels - 1);
        let ratio = s0 / s1;
        let tr = ratio * ratio;
        let rad = m.emission * ((s0 / a) * (1.0 - ratio));
        for row in 0..8 {
            for col in 0..8 {
                for dir in 0..4 {
                    let got = merged.get(col, row, dir);
                    assert!((got.transmittance - tr).abs() <= 1.0e-3);
                    assert!((got.radiance - rad).abs().max_element() <= 2.0e-3);
                }
            }
        }
    }

    #[test]
    fn single_level_hierarchy_is_just_a_gather() {
        let h = CascadeHierarchy {
            origin: Vec2::ZERO,
            base_spacing: 1.0,
            base_cols: 4,
            base_rows: 4,
            base_angular: 8,
            base_interval: 2.0,
            levels: 1,
        };
        let m = Medium {
            emission: Vec3::splat(1.0),
            absorption: 0.5,
        };
        let merged = solve(&h, &m);
        let direct = Cascade::gather(&h, 0, &m);
        for row in 0..4 {
            for col in 0..4 {
                for dir in 0..8 {
                    assert_eq!(merged.get(col, row, dir), direct.get(col, row, dir));
                }
            }
        }
    }
}
