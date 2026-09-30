//! `CPU` golden twin for the `GPU` per-fragment bounds builder.
//!
//! [`cpu_bounds_fragments`] runs the identical fixed-point reduction the `WGSL`
//! kernels do — one point at a time — then de-quantises each fragment's
//! axis-aligned box and derives its bounding sphere. Because the integer
//! extrema are exact and order independent, a passing real-device parity test
//! is direct evidence that the ported kernels produced the same box and sphere
//! as this reference, not merely that the shaders compiled.
//!
//! # What each fragment reports
//!
//! For a fragment cell `c` gathering the points assigned to it:
//!
//! - `aabb_min` / `aabb_max` are the component-wise minimum and maximum of the
//!   quantised point positions.
//! - `sphere_center` is the box centre `(aabb_min + aabb_max) / 2`.
//! - `sphere_radius` is the largest distance from that centre to any point in
//!   the fragment, so the sphere tightly encloses the same point set as the box.
//!
//! A fragment that gathered no points reports a zero box and a zero sphere so
//! callers can skip it without a separate flag.
//!
//! # Provenance
//!
//! Axis-aligned extrema and a box-centred bounding sphere are elementary
//! geometry; fixed-point atomic reduction is a standard `GPU` technique. No
//! Unreal Engine source or derived code.

use glam::Vec3;

use super::super::config::NO_CELL;
use super::config::{dequantise, quantise, BoundsConfig};

/// The broad-phase proxy aggregated for one fragment cell.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct FragmentBounds {
    /// Minimum corner of the axis-aligned bounding box, or the origin when the
    /// fragment gathered no points.
    pub aabb_min: Vec3,
    /// Maximum corner of the axis-aligned bounding box, or the origin when the
    /// fragment gathered no points.
    pub aabb_max: Vec3,
    /// Centre of the bounding sphere (the box centre), or the origin when the
    /// fragment gathered no points.
    pub sphere_center: Vec3,
    /// Radius of the bounding sphere, or zero when the fragment gathered no
    /// points.
    pub sphere_radius: f32,
}

impl FragmentBounds {
    /// The proxy reported for a fragment that gathered no points.
    #[must_use]
    pub fn empty() -> FragmentBounds {
        FragmentBounds {
            aabb_min: Vec3::ZERO,
            aabb_max: Vec3::ZERO,
            sphere_center: Vec3::ZERO,
            sphere_radius: 0.0,
        }
    }
}

/// Integer extrema for one fragment, laid out exactly like the device atomics:
/// three minimum slots, three maximum slots, and one squared-radius slot.
#[derive(Clone, Copy)]
struct Extents {
    min: [i32; 3],
    max: [i32; 3],
    radius_sq: i32,
}

impl Extents {
    /// The identity element: an empty box (min above max) and a zero radius,
    /// matching the `clear` kernel's initialisation.
    const EMPTY: Extents = Extents {
        min: [i32::MAX; 3],
        max: [i32::MIN; 3],
        radius_sq: 0,
    };

    /// Whether no point has been folded in, so the box is still inverted.
    fn is_empty(&self) -> bool {
        self.min[0] > self.max[0]
    }
}

/// Builds the per-fragment bounding box and sphere for `n_cells` fragments from
/// `points` and their `cells` assignment.
///
/// The two slices must share the same length (one entry per point). A point
/// whose cell is [`NO_CELL`] or is out of range for `n_cells` is skipped, so a
/// classifier's unassigned sentinel passes through harmlessly. The returned
/// vector has exactly `n_cells` entries in cell-index order.
///
/// # Panics
///
/// Panics if `points` and `cells` do not have the same length.
#[must_use]
pub fn cpu_bounds_fragments(
    n_cells: usize,
    points: &[Vec3],
    cells: &[u32],
    config: &BoundsConfig,
) -> Vec<FragmentBounds> {
    assert!(
        points.len() == cells.len(),
        "points and cells must have equal length"
    );

    // Pass 1: reduce the axis-aligned box extents with integer extrema.
    let mut ext = vec![Extents::EMPTY; n_cells];
    for (&p, &cell) in points.iter().zip(cells) {
        let Some(c) = fragment_index(cell, n_cells) else {
            continue;
        };
        let e = &mut ext[c];
        let q = [
            quantise(p.x, config.position_scale),
            quantise(p.y, config.position_scale),
            quantise(p.z, config.position_scale),
        ];
        for (axis, &qa) in q.iter().enumerate() {
            e.min[axis] = e.min[axis].min(qa);
            e.max[axis] = e.max[axis].max(qa);
        }
    }

    // Pass 2: the box centre becomes the sphere centre once extents are known.
    let centers: Vec<Vec3> = ext
        .iter()
        .map(|e| {
            if e.is_empty() {
                Vec3::ZERO
            } else {
                (dequant_corner(e.min, config) + dequant_corner(e.max, config)) * 0.5
            }
        })
        .collect();

    // Pass 3: fold every point's squared distance to its centre into the radius.
    for (&p, &cell) in points.iter().zip(cells) {
        let Some(c) = fragment_index(cell, n_cells) else {
            continue;
        };
        if ext[c].is_empty() {
            continue;
        }
        let d = p - centers[c];
        let q = quantise(d.dot(d), config.radius_sq_scale);
        ext[c].radius_sq = ext[c].radius_sq.max(q);
    }

    // Pass 4: de-quantise into the final box and sphere.
    ext.iter()
        .zip(&centers)
        .map(|(e, &center)| finish(e, center, config))
        .collect()
}

/// Maps a raw cell tag to an in-range fragment index, dropping the [`NO_CELL`]
/// sentinel and any out-of-range index exactly as the kernels do.
fn fragment_index(cell: u32, n_cells: usize) -> Option<usize> {
    if cell == NO_CELL {
        return None;
    }
    let c = cell as usize;
    if c >= n_cells {
        None
    } else {
        Some(c)
    }
}

/// De-quantises the three integer components of a box corner.
fn dequant_corner(corner: [i32; 3], config: &BoundsConfig) -> Vec3 {
    Vec3::new(
        dequantise(corner[0], config.position_scale),
        dequantise(corner[1], config.position_scale),
        dequantise(corner[2], config.position_scale),
    )
}

/// Turns one fragment's integer extrema into its box and sphere, matching the
/// de-quantisation the device kernels perform on read-back.
fn finish(e: &Extents, center: Vec3, config: &BoundsConfig) -> FragmentBounds {
    if e.is_empty() {
        return FragmentBounds::empty();
    }
    let aabb_min = dequant_corner(e.min, config);
    let aabb_max = dequant_corner(e.max, config);
    let radius = dequantise(e.radius_sq, config.radius_sq_scale)
        .max(0.0)
        .sqrt();
    FragmentBounds {
        aabb_min,
        aabb_max,
        sphere_center: center,
        sphere_radius: radius,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for the analytic checks. Positions are chosen as small
    /// dyadic values so the fixed-point round-trip is effectively exact and only
    /// the final radius square root contributes error.
    const TOL: f32 = 1.0e-3;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= TOL
    }

    #[test]
    fn empty_cell_reports_zero_bounds() {
        let out = cpu_bounds_fragments(2, &[], &[], &BoundsConfig::default());
        assert_eq!(out.len(), 2);
        for b in out {
            assert_eq!(b, FragmentBounds::empty());
        }
    }

    #[test]
    fn zero_cells_returns_empty() {
        let out = cpu_bounds_fragments(
            0,
            &[Vec3::new(1.0, 2.0, 3.0)],
            &[0],
            &BoundsConfig::default(),
        );
        assert!(out.is_empty());
    }

    #[test]
    fn single_point_is_a_degenerate_box_and_zero_radius() {
        let out = cpu_bounds_fragments(
            1,
            &[Vec3::new(2.0, -1.0, 0.5)],
            &[0],
            &BoundsConfig::default(),
        );
        let b = out[0];
        assert!(close(b.aabb_min.x, 2.0) && close(b.aabb_max.x, 2.0));
        assert!(close(b.aabb_min.y, -1.0) && close(b.aabb_max.y, -1.0));
        assert!(close(b.aabb_min.z, 0.5) && close(b.aabb_max.z, 0.5));
        assert!(close(b.sphere_center.x, 2.0));
        assert!(close(b.sphere_center.y, -1.0));
        assert!(close(b.sphere_center.z, 0.5));
        assert!(close(b.sphere_radius, 0.0));
    }

    #[test]
    fn symmetric_pair_has_centred_sphere() {
        // Two points at (+2,0,0) and (-2,0,0): box spans [-2,2] on x, the centre
        // is the origin, and the sphere radius is the distance to either point.
        let out = cpu_bounds_fragments(
            1,
            &[Vec3::new(2.0, 0.0, 0.0), Vec3::new(-2.0, 0.0, 0.0)],
            &[0, 0],
            &BoundsConfig::default(),
        );
        let b = out[0];
        assert!(close(b.aabb_min.x, -2.0) && close(b.aabb_max.x, 2.0));
        assert!(close(b.sphere_center.x, 0.0));
        assert!(close(b.sphere_radius, 2.0));
    }

    #[test]
    fn box_encloses_a_scattered_cluster() {
        let out = cpu_bounds_fragments(
            1,
            &[
                Vec3::new(1.0, 2.0, 3.0),
                Vec3::new(-1.0, 0.0, 5.0),
                Vec3::new(0.5, -2.0, 4.0),
            ],
            &[0, 0, 0],
            &BoundsConfig::default(),
        );
        let b = out[0];
        assert!(close(b.aabb_min.x, -1.0) && close(b.aabb_max.x, 1.0));
        assert!(close(b.aabb_min.y, -2.0) && close(b.aabb_max.y, 2.0));
        assert!(close(b.aabb_min.z, 3.0) && close(b.aabb_max.z, 5.0));
        assert!(close(b.sphere_center.x, 0.0));
        assert!(close(b.sphere_center.y, 0.0));
        assert!(close(b.sphere_center.z, 4.0));
        // Every point must sit inside the reported sphere.
        for p in [
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(-1.0, 0.0, 5.0),
            Vec3::new(0.5, -2.0, 4.0),
        ] {
            let d = (p - b.sphere_center).length();
            assert!(d <= b.sphere_radius + TOL, "point escapes the sphere");
        }
    }

    #[test]
    fn unassigned_and_out_of_range_points_are_skipped() {
        let out = cpu_bounds_fragments(
            1,
            &[
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(9.0, 9.0, 9.0),
                Vec3::new(9.0, 9.0, 9.0),
            ],
            &[0, NO_CELL, 7],
            &BoundsConfig::default(),
        );
        // Only the first point counts; the sentinel and out-of-range index drop,
        // so the fragment is a degenerate box at (1,0,0).
        let b = out[0];
        assert!(close(b.aabb_min.x, 1.0) && close(b.aabb_max.x, 1.0));
        assert!(close(b.sphere_radius, 0.0));
    }
}
