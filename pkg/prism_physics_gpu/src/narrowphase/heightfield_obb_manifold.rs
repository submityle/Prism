//! `OBB`-versus-heightfield multi-point contact manifold, shared bit-for-bit
//! with the `WGSL` kernel.
//!
//! This closes the "heightfield only supports spheres and capsules" gap: where
//! [`cpu_capsule_heightfield_manifold`](super::heightfield_capsule_manifold::cpu_capsule_heightfield_manifold)
//! clips a capsule across the cells its swept footprint touches, an oriented box
//! resting on terrain needs the same treatment — up to four coplanar points
//! sharing one normal — so a solver can hold the box flush against a flat or
//! gently sloped span of ground instead of letting it pivot about one corner.
//!
//! # Algorithm
//!
//! 1. The box's `XZ` footprint is the axis-aligned box its eight corners
//!    project to on the world `XZ` plane, computed as the centre plus the
//!    per-axis absolute projection of the half extents.
//!    [`Heightfield::xz_cell_range`](super::heightfield::Heightfield::xz_cell_range)
//!    maps that box to the inclusive rectangle of candidate cells (or reports no
//!    overlap when the box misses the grid entirely).
//! 2. The candidate cells are visited in a fixed order — row outer, column
//!    inner, first cell triangle before the second — and each cell triangle runs
//!    through the shared
//!    [`obb_triangle_manifold`](super::obb_triangle_manifold) geometry,
//!    producing a one- to four-point sub-manifold per penetrated triangle. The
//!    fixed order makes the merge below deterministic.
//! 3. The sub-manifold holding the single globally deepest point becomes the
//!    *reference*; its normal is the manifold normal. The deepest point is
//!    chosen with a strictly-greater comparison in iteration order, so the first
//!    deepest wins and ties resolve by the fixed visit order.
//! 4. Every sub-manifold whose normal is within [`COPLANAR_COS`] of the
//!    reference normal contributes all of its points; sub-manifolds facing a
//!    materially different direction (the far side of a ridge, say) are dropped,
//!    because the single-normal [`ContactManifold`] contract cannot represent
//!    them and folding them in would corrupt the plane.
//! 5. The pooled coplanar points are reduced to the widest, deepest four with
//!    the shared [`reduce_to_four`](super::manifold) routine and packed into one
//!    [`ContactManifold`].
//!
//! This is the correct upgrade of the single-point sphere path and the capsule
//! path beside it, not a stub: on a flat span it yields a stable four-corner
//! manifold, and on a ridge where the triangle normals diverge it collapses
//! honestly to the deepest reference plane. The device kernel
//! (`shaders/narrowphase_obb_heightfield_manifold.wgsl`) runs the identical
//! arithmetic in the identical order, so the real-device parity test matches it
//! slot for slot.
//!
//! # Normal convention
//!
//! The reported [`ContactManifold`] stores the box index in `a` and the
//! heightfield index in `b`; its unit normal points **from the heightfield
//! toward the box**, the direction that pushes a dynamic box off static terrain,
//! inherited unchanged from the box-versus-triangle slice it is built on.
//!
//! Provenance: textbook box-versus-triangle separating-axis test and
//! Sutherland-Hodgman face clipping plus the standard four-point manifold
//! reduction; the heightfield cell triangulation is textbook. No Unreal Engine
//! source or derived code.

use super::heightfield::{Heightfield, XzAabb};
use super::manifold::{reduce_to_four, ContactManifold, ManifoldPoint};
use super::obb::Obb;
use super::obb_triangle_manifold::obb_triangle_manifold;

/// Minimum cosine between two sub-manifold normals for them to count as
/// coplanar and be merged into one manifold.
///
/// `cos(8 degrees) ~ 0.990`: triangles tilted less than roughly eight degrees
/// apart (adjacent cells of gently sloped terrain, or the two triangles of one
/// flat cell) merge their points; a sharper fold past this threshold keeps only
/// the reference side, because a single-normal manifold cannot straddle a crease.
const COPLANAR_COS: f32 = 0.990;

/// A candidate box-versus-heightfield pair: an index into the box slice and an
/// index into the heightfield slice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeightfieldObbPair {
    /// Index of the oriented bounding box in the box slice.
    pub obb: u32,
    /// Index of the heightfield in the heightfield slice.
    pub heightfield: u32,
}

impl HeightfieldObbPair {
    /// Creates a box-versus-heightfield candidate pair.
    #[must_use]
    pub fn new(obb: u32, heightfield: u32) -> HeightfieldObbPair {
        HeightfieldObbPair { obb, heightfield }
    }
}

/// The greatest penetration depth among a sub-manifold's live points.
#[must_use]
fn manifold_peak_depth(manifold: &ContactManifold) -> f32 {
    let mut peak = manifold.points[0].depth;
    for point in manifold.points.iter().take(manifold.count as usize).skip(1) {
        if point.depth > peak {
            peak = point.depth;
        }
    }
    peak
}

/// The box's axis-aligned `XZ` footprint: the centre plus the per-axis
/// absolute projection of the half extents onto world x and z. This is exactly
/// the bounding box of the eight corners' `XZ` projection, computed with the
/// same op order the kernel uses so the candidate cell set never diverges.
#[must_use]
fn obb_xz_footprint(obb: &Obb) -> XzAabb {
    let he = obb.half_extents;
    let ax = obb.axes;
    let span_x =
        he.x * ax[0].x.abs() + he.y * ax[1].x.abs() + he.z * ax[2].x.abs();
    let span_z =
        he.x * ax[0].z.abs() + he.y * ax[1].z.abs() + he.z * ax[2].z.abs();
    XzAabb {
        min_x: obb.center.x - span_x,
        max_x: obb.center.x + span_x,
        min_z: obb.center.z - span_z,
        max_z: obb.center.z + span_z,
    }
}

/// Builds the box-versus-heightfield contact manifold for one pair, or
/// [`None`] when the box touches no cell triangle.
///
/// The box's `XZ` footprint maps to candidate cells; every cell triangle runs
/// through the shared box-versus-triangle manifold geometry, the deepest point
/// selects the reference normal, and all coplanar sub-manifold points merge and
/// reduce to the widest, deepest four. The arithmetic mirrors
/// `narrowphase_obb_heightfield_manifold.wgsl` operation for operation.
#[must_use]
pub(crate) fn obb_heightfield_manifold(
    obb_id: u32,
    field_id: u32,
    obb: &Obb,
    field: &Heightfield,
) -> Option<ContactManifold> {
    let range = field.xz_cell_range(obb_xz_footprint(obb))?;

    // Gather one sub-manifold per penetrated cell triangle in the fixed visit
    // order (row outer, column inner, triangle zero before triangle one) so the
    // reference pick and the merge below are deterministic.
    let mut subs: Vec<ContactManifold> = Vec::new();
    for row in range.min_row..=range.max_row {
        for col in range.min_col..=range.max_col {
            for tri in &field.cell_triangles(row, col) {
                if let Some(sub) = obb_triangle_manifold(obb_id, field_id, obb, tri) {
                    subs.push(sub);
                }
            }
        }
    }
    if subs.is_empty() {
        return None;
    }

    // The sub-manifold carrying the globally deepest point sets the reference
    // normal; strictly-greater keeps the first deepest, matching the kernel.
    let mut ref_index = 0usize;
    let mut best_depth = manifold_peak_depth(&subs[0]);
    for (index, sub) in subs.iter().enumerate().skip(1) {
        let depth = manifold_peak_depth(sub);
        if depth > best_depth {
            best_depth = depth;
            ref_index = index;
        }
    }
    let ref_normal = subs[ref_index].normal;

    // Pool every coplanar sub-manifold's points; the reference itself always
    // qualifies (dot = 1), so the pool is never empty.
    let mut points: Vec<ManifoldPoint> = Vec::new();
    for sub in &subs {
        if sub.normal.dot(ref_normal) >= COPLANAR_COS {
            for point in sub.points.iter().take(sub.count as usize) {
                points.push(*point);
            }
        }
    }

    let reduced = reduce_to_four(&points, ref_normal);
    let count = reduced.len();
    Some(ContactManifold::new(
        obb_id, field_id, ref_normal, count, &reduced,
    ))
}

/// `CPU` golden twin of the box-versus-heightfield manifold narrow phase.
///
/// Turns a set of candidate `pairs` into one contact slot each, in input
/// order: [`Some`] carrying the merged multi-point manifold when the box
/// penetrates the terrain, or [`None`] when it is clear of every candidate
/// cell. Emitting a slot per pair (rather than compacting) keeps the manifold
/// index aligned with the pair index, which the real-device parity test relies
/// on.
///
/// # Panics
///
/// Panics if a pair references a box or heightfield index outside the
/// corresponding slice, which is never valid output from a broad phase over the
/// same sets.
#[must_use]
pub fn cpu_obb_heightfield_manifold(
    boxes: &[Obb],
    fields: &[Heightfield],
    pairs: &[HeightfieldObbPair],
) -> Vec<Option<ContactManifold>> {
    pairs
        .iter()
        .map(|pair| {
            let obb = &boxes[pair.obb as usize];
            let field = &fields[pair.heightfield as usize];
            obb_heightfield_manifold(pair.obb, pair.heightfield, obb, field)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    /// A flat `rows x cols` field at `y = 0` with unit spacing, origin at the
    /// world origin.
    fn flat_field(rows: u32, cols: u32) -> Heightfield {
        Heightfield::new(rows, cols, 1.0, Vec3::ZERO, vec![0.0; (rows * cols) as usize])
    }

    /// An axis-aligned box at `center` with half extents `he`.
    fn axis_box(center: Vec3, he: Vec3) -> Obb {
        Obb::new(center, [Vec3::X, Vec3::Y, Vec3::Z], he)
    }

    #[test]
    fn flat_box_rests_on_four_corners() {
        // A unit box centred at (1, 0.4, 1) over a flat 3x3 field: its base sits
        // 0.4 - 0.5 = -0.1 below the ground, so it penetrates by 0.1 with a +Y
        // normal and a stable four-corner manifold.
        let field = flat_field(3, 3);
        let boxes = [axis_box(Vec3::new(1.0, 0.4, 1.0), Vec3::splat(0.5))];
        let pairs = [HeightfieldObbPair::new(0, 0)];
        let m = cpu_obb_heightfield_manifold(&boxes, &[field], &pairs)[0]
            .expect("box resting on terrain must contact");
        assert_eq!(m.a, 0);
        assert_eq!(m.b, 0);
        assert!((m.normal - Vec3::Y).length() < 1.0e-5, "normal {:?}", m.normal);
        assert_eq!(m.count, 4, "a flat box should rest on four corners");
        for point in m.points.iter().take(m.count as usize) {
            assert!((point.depth - 0.1).abs() < 1.0e-5, "depth {}", point.depth);
        }
    }

    #[test]
    fn box_off_the_grid_misses() {
        // Footprint lies wholly beyond the grid: no candidate cells, no contact.
        let field = flat_field(3, 3);
        let boxes = [axis_box(Vec3::new(50.0, 0.4, 50.0), Vec3::splat(0.5))];
        let pairs = [HeightfieldObbPair::new(0, 0)];
        assert!(cpu_obb_heightfield_manifold(&boxes, &[field], &pairs)[0].is_none());
    }

    #[test]
    fn box_clear_above_terrain_misses() {
        // Box hovers well above the ground inside the footprint: cells are
        // visited but no triangle penetrates, so the result is None.
        let field = flat_field(3, 3);
        let boxes = [axis_box(Vec3::new(1.0, 5.0, 1.0), Vec3::splat(0.5))];
        let pairs = [HeightfieldObbPair::new(0, 0)];
        assert!(cpu_obb_heightfield_manifold(&boxes, &[field], &pairs)[0].is_none());
    }

    #[test]
    fn incline_box_keeps_single_reference_plane() {
        // A uniform incline rising +0.2 per unit z: every cell triangle is
        // coplanar, so a box resting on the slope reports one tilted normal.
        let heights = vec![
            0.0, 0.0, 0.0, // row 0 (z = 0)
            0.2, 0.2, 0.2, // row 1 (z = 1)
            0.4, 0.4, 0.4, // row 2 (z = 2)
        ];
        let field = Heightfield::new(3, 3, 1.0, Vec3::ZERO, heights);
        let boxes = [axis_box(Vec3::new(1.0, 0.6, 1.0), Vec3::splat(0.5))];
        let pairs = [HeightfieldObbPair::new(0, 0)];
        let m = cpu_obb_heightfield_manifold(&boxes, &[field], &pairs)[0]
            .expect("box on the incline must contact");
        assert!((m.normal.length() - 1.0).abs() < 1.0e-5, "normal must be unit");
        assert!(m.normal.dot(Vec3::Y) > 0.0, "terrain normal points up: {:?}", m.normal);
        assert!((1..=4).contains(&m.count), "count in range, got {}", m.count);
    }

    #[test]
    fn batch_preserves_slot_alignment() {
        // One hitting pair and one missing pair keep their input slots.
        let field = flat_field(3, 3);
        let boxes = [
            axis_box(Vec3::new(1.0, 0.4, 1.0), Vec3::splat(0.5)),
            axis_box(Vec3::new(50.0, 0.4, 50.0), Vec3::splat(0.5)),
        ];
        let pairs = [
            HeightfieldObbPair::new(0, 0),
            HeightfieldObbPair::new(1, 0),
        ];
        let out = cpu_obb_heightfield_manifold(&boxes, &[field], &pairs);
        assert!(out[0].is_some(), "first box contacts");
        assert!(out[1].is_none(), "second box misses");
    }
}
