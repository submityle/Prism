//! Upward merge: composite a coarse parent cascade into its finer child.
//!
//! Merging is what turns a stack of independently-gathered shells into a single
//! full-range radiance field. For each child probe and child direction the
//! merge:
//!
//! 1. **Bilinearly interpolates** the parent cascade at the child probe's world
//!    position — the parent grid is 2× coarser, so four parent probes surround
//!    each child probe.
//! 2. **Angularly averages** the [`ANGULAR_SCALE`] parent sub-directions that
//!    subdivide this child direction's sector (the parent has 4× the bins).
//! 3. **Composites** the child's near interval *over* that averaged far
//!    interval ([`RadianceInterval::over`]).
//!
//! Because child and parent cover contiguous radial shells
//! ([`CascadeHierarchy::interval_end`] of the child equals
//! [`CascadeHierarchy::interval_start`] of the parent), the composite covers the
//! union shell. Folding from the top cascade down to cascade 0 therefore yields
//! a cascade-0 field whose every direction integrates the full radial range.
//!
//! # Provenance
//! Bilinear spatial interpolation plus angular averaging of the ×4 parent
//! sub-bins is the merge rule from Alexander Sannikov's *Radiance Cascades*
//! (2023). Clean-room classical implementation.
//!
//! [`ANGULAR_SCALE`]: super::hierarchy::ANGULAR_SCALE
//! [`CascadeHierarchy::interval_end`]: super::hierarchy::CascadeHierarchy::interval_end
//! [`CascadeHierarchy::interval_start`]: super::hierarchy::CascadeHierarchy::interval_start
//!
//! No Unreal Engine source is used anywhere in this module.

use glam::Vec3;

use super::cascade::Cascade;
use super::hierarchy::{CascadeHierarchy, ANGULAR_SCALE};
use super::interval::RadianceInterval;

/// Bilinearly sample `parent` direction `dir` at child-probe world position
/// `(col, row)` of child `level`.
fn bilinear(
    hierarchy: &CascadeHierarchy,
    parent: &Cascade,
    child_level: u32,
    col: u32,
    row: u32,
    dir: u32,
) -> RadianceInterval {
    let world = hierarchy.probe_position(child_level, col, row);
    let parent_level = child_level + 1;
    let spacing = hierarchy.spacing(parent_level);
    let origin = hierarchy.origin;
    // Continuous parent grid coordinate (probe centres sit at integer+0.5).
    let gx = (world.x - origin.x) / spacing - 0.5;
    let gy = (world.y - origin.y) / spacing - 0.5;

    let (pcols, prows) = parent.dims();
    let clamp = |v: f32, hi: u32| -> (u32, u32, f32) {
        let lo_f = v.floor();
        let frac = v - lo_f;
        let lo = lo_f.max(0.0) as u32;
        let lo = lo.min(hi - 1);
        let hi_i = (lo + 1).min(hi - 1);
        // If we clamped at the low edge, pull frac to 0; at the high edge, to the
        // sampled neighbour so interpolation degrades to nearest at borders.
        let frac = if (v < 0.0) || (lo == hi_i) { 0.0 } else { frac };
        (lo, hi_i, frac)
    };
    let (x0, x1, fx) = clamp(gx, pcols);
    let (y0, y1, fy) = clamp(gy, prows);

    let c00 = parent.get(x0, y0, dir);
    let c10 = parent.get(x1, y0, dir);
    let c01 = parent.get(x0, y1, dir);
    let c11 = parent.get(x1, y1, dir);
    let top = c00.lerp(c10, fx);
    let bot = c01.lerp(c11, fx);
    top.lerp(bot, fy)
}

/// Angularly average the [`ANGULAR_SCALE`] parent sub-bins of child `dir`,
/// each bilinearly interpolated at the child probe position.
fn parent_far(
    hierarchy: &CascadeHierarchy,
    parent: &Cascade,
    child_level: u32,
    col: u32,
    row: u32,
    child_dir: u32,
) -> RadianceInterval {
    let mut radiance = Vec3::ZERO;
    let mut transmittance = 0.0_f32;
    for k in 0..ANGULAR_SCALE {
        let pdir = child_dir * ANGULAR_SCALE + k;
        let s = bilinear(hierarchy, parent, child_level, col, row, pdir);
        radiance += s.radiance;
        transmittance += s.transmittance;
    }
    let inv = 1.0 / (ANGULAR_SCALE as f32);
    RadianceInterval::new(radiance * inv, transmittance * inv)
}

/// Composite `parent` (level `child.level() + 1`) into `child` in place.
///
/// After this call every `child` interval covers the union of its own shell and
/// the parent's shell.
pub fn merge_into(hierarchy: &CascadeHierarchy, child: &mut Cascade, parent: &Cascade) {
    debug_assert_eq!(parent.level(), child.level() + 1);
    let child_level = child.level();
    let (cols, rows) = child.dims();
    let angular = child.angular();
    for row in 0..rows {
        for col in 0..cols {
            for dir in 0..angular {
                let near = child.get(col, row, dir);
                let far = parent_far(hierarchy, parent, child_level, col, row, dir);
                child.set(col, row, dir, near.over(far));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::cascade::SceneSampler;
    use super::*;
    use glam::Vec2;

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
    fn child_merged_with_parent_equals_full_range_gather() {
        // Spatially-uniform medium ⇒ bilinear + angular average are exact, so
        // the merged child interval must equal a direct gather over the union
        // shell [start(0), end(1)).
        let h = hierarchy();
        let m = Medium {
            emission: Vec3::new(1.5, 0.75, 0.25),
            absorption: 0.4,
        };
        let mut child = Cascade::gather(&h, 0, &m);
        let parent = Cascade::gather(&h, 1, &m);
        merge_into(&h, &mut child, &parent);

        let t0 = h.interval_start(0);
        let t2 = h.interval_end(1);
        let a = m.absorption;
        let s0 = 1.0 + a * t0;
        let s1 = 1.0 + a * t2;
        let ratio = s0 / s1;
        let tr = ratio * ratio;
        let rad = m.emission * ((s0 / a) * (1.0 - ratio));
        for row in 0..child.dims().1 {
            for col in 0..child.dims().0 {
                for dir in 0..child.angular() {
                    let got = child.get(col, row, dir);
                    assert!(
                        (got.transmittance - tr).abs() <= 1.0e-4,
                        "T {} vs {}",
                        got.transmittance,
                        tr
                    );
                    assert!((got.radiance - rad).abs().max_element() <= 1.0e-4);
                }
            }
        }
    }

    #[test]
    fn merging_a_clear_parent_leaves_the_child_unchanged() {
        let h = hierarchy();
        let m = Medium {
            emission: Vec3::splat(1.0),
            absorption: 0.2,
        };
        let child_ref = Cascade::gather(&h, 0, &m);
        let mut child = child_ref.clone();
        let parent = Cascade::cleared(&h, 1); // all CLEAR
        merge_into(&h, &mut child, &parent);
        for row in 0..child.dims().1 {
            for col in 0..child.dims().0 {
                for dir in 0..child.angular() {
                    assert_eq!(child.get(col, row, dir), child_ref.get(col, row, dir));
                }
            }
        }
    }
}
