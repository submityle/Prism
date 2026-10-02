//! Capsule-versus-heightfield multi-point contact manifold, shared bit-for-bit
//! with the `WGSL` kernel.
//!
//! This closes the "heightfield only supports spheres" gap: where
//! [`cpu_sphere_heightfield_narrowphase`](super::heightfield::cpu_sphere_heightfield_narrowphase)
//! reduces a sphere to a single deepest contact over the cells its footprint
//! touches, a capsule lying across terrain needs a *manifold* — up to four
//! coplanar points sharing one normal — so a solver can resist rotation and
//! hold the capsule flush against a flat span of ground instead of letting it
//! rock about one point.
//!
//! # Algorithm
//!
//! 1. The capsule's `XZ` footprint is the axis-aligned box of its two
//!    endpoints `p0`/`p1` on the world `XZ` plane, grown by the radius on
//!    every side. [`Heightfield::xz_cell_range`](super::heightfield::Heightfield::xz_cell_range)
//!    maps that box to the inclusive rectangle of candidate cells (or reports no
//!    overlap when the capsule misses the grid entirely).
//! 2. The candidate cells are visited in a fixed order — row outer, column
//!    inner, first cell triangle before the second — and each cell triangle runs
//!    through the shared
//!    [`capsule_triangle_manifold`](super::capsule_triangle_manifold) geometry,
//!    producing a one- or two-point sub-manifold per penetrated triangle. The
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
//! This is the correct upgrade of the single-point sphere path, not a stub: on a
//! flat span it yields a stable multi-point manifold, and on a ridge where the
//! triangle normals diverge it collapses honestly to the deepest reference
//! plane. The device kernel
//! (`shaders/narrowphase_capsule_heightfield_manifold.wgsl`) runs the identical
//! arithmetic in the identical order, so the real-device parity test matches it
//! slot for slot.
//!
//! # Normal convention
//!
//! The reported [`ContactManifold`] stores the capsule index in `a` and the
//! heightfield index in `b`; its unit normal points **from the heightfield
//! toward the capsule**, the direction that pushes a dynamic capsule off static
//! terrain, inherited unchanged from the capsule-versus-triangle slice it is
//! built on.
//!
//! Provenance: textbook capsule-versus-triangle clipping (closest-point and
//! segment-triangle clip from Christer Ericson, *Real-Time Collision Detection*,
//! 2004) plus the standard four-point manifold reduction; the heightfield cell
//! triangulation is textbook. No Unreal Engine source or derived code.

use super::capsule::Capsule;
use super::capsule_triangle_manifold::capsule_triangle_manifold;
use super::heightfield::{Heightfield, XzAabb};
use super::manifold::{reduce_to_four, ContactManifold, ManifoldPoint};

/// Minimum cosine between two sub-manifold normals for them to count as
/// coplanar and be merged into one manifold.
///
/// `cos(8 degrees) ~ 0.990`: triangles tilted less than roughly eight degrees
/// apart (adjacent cells of gently sloped terrain, or the two triangles of one
/// flat cell) merge their points; a sharper fold past this threshold keeps only
/// the reference side, because a single-normal manifold cannot straddle a crease.
const COPLANAR_COS: f32 = 0.990;

/// A candidate capsule-versus-heightfield pair: an index into the capsule slice
/// and an index into the heightfield slice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeightfieldCapsulePair {
    /// Index of the capsule in the capsule slice.
    pub capsule: u32,
    /// Index of the heightfield in the heightfield slice.
    pub heightfield: u32,
}

impl HeightfieldCapsulePair {
    /// Creates a capsule-versus-heightfield candidate pair.
    #[must_use]
    pub fn new(capsule: u32, heightfield: u32) -> HeightfieldCapsulePair {
        HeightfieldCapsulePair {
            capsule,
            heightfield,
        }
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

/// Builds the capsule-versus-heightfield contact manifold for one pair, or
/// [`None`] when the capsule touches no cell triangle.
///
/// The capsule's `XZ` footprint maps to candidate cells; every cell triangle
/// runs through the shared capsule-versus-triangle manifold geometry, the
/// deepest point selects the reference normal, and all coplanar sub-manifold
/// points merge and reduce to the widest, deepest four. The arithmetic mirrors
/// `narrowphase_capsule_heightfield_manifold.wgsl` operation for operation.
#[must_use]
pub(crate) fn capsule_heightfield_manifold(
    capsule_id: u32,
    field_id: u32,
    cap: &Capsule,
    field: &Heightfield,
) -> Option<ContactManifold> {
    let aabb = XzAabb {
        min_x: cap.p0.x.min(cap.p1.x) - cap.radius,
        max_x: cap.p0.x.max(cap.p1.x) + cap.radius,
        min_z: cap.p0.z.min(cap.p1.z) - cap.radius,
        max_z: cap.p0.z.max(cap.p1.z) + cap.radius,
    };
    let range = field.xz_cell_range(aabb)?;

    // Gather one sub-manifold per penetrated cell triangle in the fixed visit
    // order (row outer, column inner, triangle zero before triangle one) so the
    // reference pick and the merge below are deterministic.
    let mut subs: Vec<ContactManifold> = Vec::new();
    for row in range.min_row..=range.max_row {
        for col in range.min_col..=range.max_col {
            for tri in &field.cell_triangles(row, col) {
                if let Some(sub) = capsule_triangle_manifold(capsule_id, field_id, cap, tri) {
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
        capsule_id, field_id, ref_normal, count, &reduced,
    ))
}

/// `CPU` golden twin of the capsule-versus-heightfield manifold narrow phase.
///
/// Turns a set of candidate `pairs` into one contact slot each, in input
/// order: [`Some`] carrying the merged multi-point manifold when the capsule
/// penetrates the terrain, or [`None`] when it is clear of every candidate
/// cell. Emitting a slot per pair (rather than compacting) keeps the manifold
/// index aligned with the pair index, which the real-device parity test relies
/// on.
///
/// # Panics
///
/// Panics if a pair references a capsule or heightfield index outside the
/// corresponding slice, which is never valid output from a broad phase over the
/// same sets.
#[must_use]
pub fn cpu_capsule_heightfield_manifold(
    capsules: &[Capsule],
    fields: &[Heightfield],
    pairs: &[HeightfieldCapsulePair],
) -> Vec<Option<ContactManifold>> {
    pairs
        .iter()
        .map(|pair| {
            let cap = &capsules[pair.capsule as usize];
            let field = &fields[pair.heightfield as usize];
            capsule_heightfield_manifold(pair.capsule, pair.heightfield, cap, field)
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

    fn cap(p0: Vec3, p1: Vec3, radius: f32) -> Capsule {
        Capsule::new(p0, p1, radius)
    }

    #[test]
    fn flat_capsule_reports_stable_multipoint() {
        // Capsule lying flat along +x at height y = 0.4, radius 0.5, over a flat
        // 3x3 field (2x2 cells). Both ends sit 0.4 above the ground, so every
        // clipped corner has depth 0.5 - 0.4 = 0.1 and the normal is +Y. The
        // span crosses two columns, so the merge yields a multi-point manifold.
        let field = flat_field(3, 3);
        let capsules = [cap(Vec3::new(0.2, 0.4, 0.5), Vec3::new(1.8, 0.4, 0.5), 0.5)];
        let pairs = [HeightfieldCapsulePair::new(0, 0)];
        let m = cpu_capsule_heightfield_manifold(&capsules, &[field], &pairs)[0]
            .expect("flat capsule over terrain must contact");
        assert_eq!(m.a, 0);
        assert_eq!(m.b, 0);
        assert!((m.normal - Vec3::Y).length() < 1.0e-6, "normal {:?}", m.normal);
        assert!(m.count >= 2, "flat span should give a multi-point manifold, got {}", m.count);
        for point in m.points.iter().take(m.count as usize) {
            assert!((point.depth - 0.1).abs() < 1.0e-5, "depth {}", point.depth);
            assert!(point.position.y.abs() < 1.0e-5, "contact should sit on the face");
        }
    }

    #[test]
    fn capsule_off_the_grid_misses() {
        // Footprint lies wholly beyond the grid: no candidate cells, no contact.
        let field = flat_field(3, 3);
        let capsules = [cap(Vec3::new(50.0, 0.4, 50.0), Vec3::new(52.0, 0.4, 50.0), 0.5)];
        let pairs = [HeightfieldCapsulePair::new(0, 0)];
        assert!(cpu_capsule_heightfield_manifold(&capsules, &[field], &pairs)[0].is_none());
    }

    #[test]
    fn capsule_clear_above_terrain_misses() {
        // Capsule hovers well above the ground inside the footprint: cells are
        // visited but no triangle penetrates, so the result is None.
        let field = flat_field(3, 3);
        let capsules = [cap(Vec3::new(0.5, 5.0, 0.5), Vec3::new(1.5, 5.0, 0.5), 0.5)];
        let pairs = [HeightfieldCapsulePair::new(0, 0)];
        assert!(cpu_capsule_heightfield_manifold(&capsules, &[field], &pairs)[0].is_none());
    }

    #[test]
    fn ridge_collapses_to_deepest_coplanar_side() {
        // A gentle tent ridged along +x at z = 1 (row 1 peaked to y = 0.2): a
        // capsule laid along +z above both slopes hits triangles whose normals
        // tilt ~11 degrees apart on either side of the crease (so the two sides
        // are not coplanar). The manifold keeps the single reference plane, so
        // its normal is unit, points up off the terrain, and never straddles the
        // fold.
        let heights = vec![
            0.0, 0.0, 0.0, // row 0 (z = 0)
            0.2, 0.2, 0.2, // row 1 (z = 1), the gentle ridge
            0.0, 0.0, 0.0, // row 2 (z = 2)
        ];
        let field = Heightfield::new(3, 3, 1.0, Vec3::ZERO, heights);
        // The capsule axis sits above the peak so every contact normal points
        // upward; the near slope is strictly deeper, so the reference is
        // unambiguous.
        let capsules = [cap(Vec3::new(0.5, 0.3, 0.3), Vec3::new(0.5, 0.3, 1.7), 0.5)];
        let pairs = [HeightfieldCapsulePair::new(0, 0)];
        let m = cpu_capsule_heightfield_manifold(&capsules, &[field], &pairs)[0]
            .expect("capsule across the ridge must contact");
        assert!((m.normal.length() - 1.0).abs() < 1.0e-5, "normal must be unit");
        assert!(m.normal.dot(Vec3::Y) > 0.0, "terrain normal points up: {:?}", m.normal);
        assert!((1..=4).contains(&m.count), "count in range, got {}", m.count);
    }

    #[test]
    fn batch_preserves_slot_alignment() {
        // One hitting pair and one missing pair keep their input slots.
        let field = flat_field(3, 3);
        let capsules = [
            cap(Vec3::new(0.2, 0.4, 0.5), Vec3::new(1.8, 0.4, 0.5), 0.5),
            cap(Vec3::new(50.0, 0.4, 50.0), Vec3::new(52.0, 0.4, 50.0), 0.5),
        ];
        let pairs = [
            HeightfieldCapsulePair::new(0, 0),
            HeightfieldCapsulePair::new(1, 0),
        ];
        let out = cpu_capsule_heightfield_manifold(&capsules, &[field], &pairs);
        assert!(out[0].is_some(), "first capsule contacts");
        assert!(out[1].is_none(), "second capsule misses");
    }
}
