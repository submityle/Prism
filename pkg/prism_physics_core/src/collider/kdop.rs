//! k-DOP (discrete oriented polytope) bounding volumes.
//!
//! A *k-DOP* bounds a point set with `k` half-spaces drawn from a small, fixed
//! set of `k/2` directions. For each direction the volume stores the `[min,
//! max]` slab the point set projects onto; the intersection of those slabs is a
//! convex polytope that hugs the geometry far more tightly than an axis-aligned
//! box while staying cheap to build and test. k-DOPs are the classic mid-phase
//! bounding volume for broad-phase/BVH culling (Klosowski et al., *Efficient
//! Collision Detection Using Bounding Volume Hierarchies of k-DOPs*, 1998) and
//! ship in production engines (`OPCODE`, Chaos `FAABBVectorized` neighbours,
//! Havok mid-phase).
//!
//! # Why not just an AABB?
//!
//! An AABB is exactly a 6-DOP: three cardinal-axis slabs. Adding the cube-corner
//! and edge directions lets the volume reject pairs that are separated along a
//! diagonal -- a very common false positive for AABB culling of rotated or
//! diamond-shaped geometry. The overlap test stays a per-axis slab comparison,
//! so a 14-/18-/26-DOP costs only a handful more comparisons yet prunes many
//! more non-colliding pairs before the narrow phase runs.
//!
//! # Conservativeness
//!
//! [`KDop::overlaps`] is the standard *necessary* separating-slab test: if any
//! shared axis has disjoint slabs the volumes provably cannot intersect
//! (reject); otherwise they *may* intersect and the pair is forwarded to the
//! narrow phase. It never reports a false "disjoint", which is the only safety
//! property a culling volume must guarantee.
//!
//! # Determinism
//!
//! Every quantity is a dot product plus `min`/`max` over fixed, index-ordered
//! direction tables -- no transcendental functions and no data-dependent
//! iteration order -- so a volume built from the same points is bit-for-bit
//! reproducible, as required for cross-run state hashing.
//!
//! # Provenance
//!
//! The k-DOP construction and slab-overlap test are textbook computational
//! geometry. This module contains **no Unreal Engine source or derived code**.

use glam::Vec3;

/// The largest direction count any supported [`DopKind`] uses (26-DOP = 13).
const MAX_DOP_AXES: usize = 13;

/// Cardinal axes shared by every direction set (the AABB slabs).
const AXES_6: [Vec3; 3] = [
    Vec3::new(1.0, 0.0, 0.0),
    Vec3::new(0.0, 1.0, 0.0),
    Vec3::new(0.0, 0.0, 1.0),
];

/// Cardinal axes plus the four cube-corner diagonals.
const AXES_14: [Vec3; 7] = [
    Vec3::new(1.0, 0.0, 0.0),
    Vec3::new(0.0, 1.0, 0.0),
    Vec3::new(0.0, 0.0, 1.0),
    Vec3::new(1.0, 1.0, 1.0),
    Vec3::new(1.0, 1.0, -1.0),
    Vec3::new(1.0, -1.0, 1.0),
    Vec3::new(1.0, -1.0, -1.0),
];

/// Cardinal axes plus the six cube-edge diagonals.
const AXES_18: [Vec3; 9] = [
    Vec3::new(1.0, 0.0, 0.0),
    Vec3::new(0.0, 1.0, 0.0),
    Vec3::new(0.0, 0.0, 1.0),
    Vec3::new(1.0, 1.0, 0.0),
    Vec3::new(1.0, -1.0, 0.0),
    Vec3::new(1.0, 0.0, 1.0),
    Vec3::new(1.0, 0.0, -1.0),
    Vec3::new(0.0, 1.0, 1.0),
    Vec3::new(0.0, 1.0, -1.0),
];

/// Cardinal axes plus every cube-corner and cube-edge diagonal (13 axes).
const AXES_26: [Vec3; 13] = [
    Vec3::new(1.0, 0.0, 0.0),
    Vec3::new(0.0, 1.0, 0.0),
    Vec3::new(0.0, 0.0, 1.0),
    Vec3::new(1.0, 1.0, 1.0),
    Vec3::new(1.0, 1.0, -1.0),
    Vec3::new(1.0, -1.0, 1.0),
    Vec3::new(1.0, -1.0, -1.0),
    Vec3::new(1.0, 1.0, 0.0),
    Vec3::new(1.0, -1.0, 0.0),
    Vec3::new(1.0, 0.0, 1.0),
    Vec3::new(1.0, 0.0, -1.0),
    Vec3::new(0.0, 1.0, 1.0),
    Vec3::new(0.0, 1.0, -1.0),
];

/// The canonical direction sets the k-DOP builder supports.
///
/// Every set begins with the three cardinal axes, so
/// [`KDop::to_aabb`] is valid for all kinds and higher-order kinds strictly
/// refine the AABB.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum DopKind {
    /// 6-DOP: the three cardinal-axis slabs. Equivalent to an AABB.
    Dop6,
    /// 14-DOP: cardinal axes plus the four cube-corner diagonals.
    Dop14,
    /// 18-DOP: cardinal axes plus the six cube-edge diagonals.
    Dop18,
    /// 26-DOP: cardinal axes plus every corner and edge diagonal (13 axes).
    Dop26,
}

impl DopKind {
    /// The direction table backing this kind.
    ///
    /// Directions are intentionally left un-normalised: the slab min/max of
    /// every volume built with the same kind is scaled by the same factor, so
    /// the overlap and containment tests stay exact without a square root.
    #[must_use]
    pub fn axes(self) -> &'static [Vec3] {
        match self {
            DopKind::Dop6 => &AXES_6,
            DopKind::Dop14 => &AXES_14,
            DopKind::Dop18 => &AXES_18,
            DopKind::Dop26 => &AXES_26,
        }
    }

    /// Number of direction axes (half of `k`).
    #[must_use]
    pub fn axis_count(self) -> usize {
        self.axes().len()
    }

    /// The bounding `k` (number of half-space planes = `2 * axis_count`).
    #[must_use]
    pub fn k(self) -> usize {
        2 * self.axis_count()
    }
}

/// A k-DOP bounding volume: one `[min, max]` projection slab per direction of a
/// [`DopKind`].
///
/// Slabs are stored in a fixed-size array so the volume is `Copy` and
/// allocation-free, which matters when one is cached on every node of a
/// bounding-volume hierarchy. Only the first [`DopKind::axis_count`] entries are
/// meaningful; the tail is inert padding.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct KDop {
    kind: DopKind,
    mins: [f32; MAX_DOP_AXES],
    maxs: [f32; MAX_DOP_AXES],
}

impl KDop {
    /// An empty volume: every active slab is inverted (`min = +inf`,
    /// `max = -inf`) so the first [`grow`](KDop::grow) sets it exactly.
    #[must_use]
    pub fn empty(kind: DopKind) -> KDop {
        KDop {
            kind,
            mins: [f32::INFINITY; MAX_DOP_AXES],
            maxs: [f32::NEG_INFINITY; MAX_DOP_AXES],
        }
    }

    /// Builds the tight k-DOP of a point cloud.
    #[must_use]
    pub fn from_points(kind: DopKind, points: &[Vec3]) -> KDop {
        let mut dop = KDop::empty(kind);
        for &p in points {
            dop.grow(p);
        }
        dop
    }

    /// Builds the tight k-DOP of an axis-aligned box given by its corners.
    ///
    /// For [`DopKind::Dop6`] this reproduces the box exactly; higher kinds also
    /// bound the box's eight corners along their diagonal axes.
    #[must_use]
    pub fn from_aabb(kind: DopKind, min: Vec3, max: Vec3) -> KDop {
        let corners = [
            Vec3::new(min.x, min.y, min.z),
            Vec3::new(max.x, min.y, min.z),
            Vec3::new(min.x, max.y, min.z),
            Vec3::new(max.x, max.y, min.z),
            Vec3::new(min.x, min.y, max.z),
            Vec3::new(max.x, min.y, max.z),
            Vec3::new(min.x, max.y, max.z),
            Vec3::new(max.x, max.y, max.z),
        ];
        KDop::from_points(kind, &corners)
    }

    /// The direction set this volume is defined against.
    #[must_use]
    pub fn kind(self) -> DopKind {
        self.kind
    }

    /// Number of active slabs (`= self.kind().axis_count()`).
    #[must_use]
    pub fn axis_count(self) -> usize {
        self.kind.axis_count()
    }

    /// Expands the volume to include `point`.
    pub fn grow(&mut self, point: Vec3) {
        for (i, axis) in self.kind.axes().iter().enumerate() {
            let d = axis.dot(point);
            self.mins[i] = self.mins[i].min(d);
            self.maxs[i] = self.maxs[i].max(d);
        }
    }

    /// Expands the volume to the union of `self` and `other`.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if the two volumes use different [`DopKind`]s;
    /// unioning across kinds is a programming error because the slabs are
    /// indexed against different direction tables.
    pub fn merge(&mut self, other: &KDop) {
        debug_assert_eq!(
            self.kind, other.kind,
            "cannot merge k-DOPs of different kinds"
        );
        for i in 0..self.kind.axis_count() {
            self.mins[i] = self.mins[i].min(other.mins[i]);
            self.maxs[i] = self.maxs[i].max(other.maxs[i]);
        }
    }

    /// Returns the union of `self` and `other` without mutating either.
    #[must_use]
    pub fn merged(mut self, other: &KDop) -> KDop {
        self.merge(other);
        self
    }

    /// The `[min, max]` projection slab for direction `index`, or `None` if the
    /// index is outside the active range.
    #[must_use]
    pub fn slab(&self, index: usize) -> Option<(f32, f32)> {
        if index < self.kind.axis_count() {
            Some((self.mins[index], self.maxs[index]))
        } else {
            None
        }
    }

    /// Whether the volume is non-empty: every active slab has `min <= max`.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        (0..self.kind.axis_count()).all(|i| self.mins[i] <= self.maxs[i])
    }

    /// The axis-aligned bounding box recovered from the three cardinal slabs.
    ///
    /// Returns `(min, max)` corners. Valid for every [`DopKind`] because the
    /// cardinal axes are always the first three directions.
    #[must_use]
    pub fn to_aabb(&self) -> (Vec3, Vec3) {
        (
            Vec3::new(self.mins[0], self.mins[1], self.mins[2]),
            Vec3::new(self.maxs[0], self.maxs[1], self.maxs[2]),
        )
    }

    /// Whether `point` lies inside every active slab.
    #[must_use]
    pub fn contains_point(&self, point: Vec3) -> bool {
        if !self.is_valid() {
            return false;
        }
        for (i, axis) in self.kind.axes().iter().enumerate() {
            let d = axis.dot(point);
            if d < self.mins[i] || d > self.maxs[i] {
                return false;
            }
        }
        true
    }

    /// Conservative overlap test against another volume of the same kind.
    ///
    /// Returns `false` only when a shared axis separates the two slabs (a proof
    /// of disjointness); a `true` result means the volumes *may* intersect and
    /// the pair should be forwarded to the narrow phase. Volumes of different
    /// kinds, or any empty volume, never overlap.
    #[must_use]
    pub fn overlaps(&self, other: &KDop) -> bool {
        if self.kind != other.kind || !self.is_valid() || !other.is_valid() {
            return false;
        }
        for i in 0..self.kind.axis_count() {
            if self.maxs[i] < other.mins[i] || other.maxs[i] < self.mins[i] {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_cube_points() -> Vec<Vec3> {
        vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 1.0),
            Vec3::new(0.0, 1.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
        ]
    }

    /// An octahedron (diamond) centred at `c` with the given radius.
    fn octahedron(c: Vec3, r: f32) -> Vec<Vec3> {
        vec![
            c + Vec3::new(r, 0.0, 0.0),
            c - Vec3::new(r, 0.0, 0.0),
            c + Vec3::new(0.0, r, 0.0),
            c - Vec3::new(0.0, r, 0.0),
            c + Vec3::new(0.0, 0.0, r),
            c - Vec3::new(0.0, 0.0, r),
        ]
    }

    #[test]
    fn axis_counts_match_dop_order() {
        assert_eq!(DopKind::Dop6.axis_count(), 3);
        assert_eq!(DopKind::Dop14.axis_count(), 7);
        assert_eq!(DopKind::Dop18.axis_count(), 9);
        assert_eq!(DopKind::Dop26.axis_count(), 13);
        assert_eq!(DopKind::Dop6.k(), 6);
        assert_eq!(DopKind::Dop14.k(), 14);
        assert_eq!(DopKind::Dop18.k(), 18);
        assert_eq!(DopKind::Dop26.k(), 26);
    }

    #[test]
    fn every_kind_starts_with_cardinal_axes() {
        for kind in [
            DopKind::Dop6,
            DopKind::Dop14,
            DopKind::Dop18,
            DopKind::Dop26,
        ] {
            let axes = kind.axes();
            assert_eq!(axes[0], Vec3::X);
            assert_eq!(axes[1], Vec3::Y);
            assert_eq!(axes[2], Vec3::Z);
        }
    }

    #[test]
    fn dop6_cardinal_slabs_match_aabb() {
        let dop = KDop::from_points(DopKind::Dop6, &unit_cube_points());
        let (min, max) = dop.to_aabb();
        assert_eq!(min, Vec3::ZERO);
        assert_eq!(max, Vec3::ONE);
    }

    #[test]
    fn corner_slab_bounds_cube_diagonal() {
        let dop = KDop::from_points(DopKind::Dop14, &unit_cube_points());
        // Axis 3 is (1,1,1): the cube projects onto [0, 3].
        let (lo, hi) = dop.slab(3).expect("corner slab present");
        assert!((lo - 0.0).abs() < 1e-6);
        assert!((hi - 3.0).abs() < 1e-6);
        assert!(dop.slab(13).is_none());
    }

    #[test]
    fn contains_point_respects_diagonal_slabs() {
        let dop = KDop::from_points(DopKind::Dop14, &octahedron(Vec3::ZERO, 1.0));
        // Centre is inside; the far cube corner is inside the AABB but outside
        // the diamond's (1,1,1) slab.
        assert!(dop.contains_point(Vec3::ZERO));
        assert!(!dop.contains_point(Vec3::new(1.0, 1.0, 1.0)));
        // A pure-axis extreme is on the surface and still contained.
        assert!(dop.contains_point(Vec3::new(1.0, 0.0, 0.0)));
    }

    #[test]
    fn fourteen_dop_rejects_diagonal_gap_that_aabb_passes() {
        let a = octahedron(Vec3::ZERO, 1.0);
        let b = octahedron(Vec3::splat(0.9), 1.0);

        // Cardinal ranges overlap, so the AABB (6-DOP) cannot separate them.
        let a6 = KDop::from_points(DopKind::Dop6, &a);
        let b6 = KDop::from_points(DopKind::Dop6, &b);
        assert!(a6.overlaps(&b6));

        // The (1,1,1) corner slabs are disjoint, so the 14-DOP prunes the pair.
        let a14 = KDop::from_points(DopKind::Dop14, &a);
        let b14 = KDop::from_points(DopKind::Dop14, &b);
        assert!(!a14.overlaps(&b14));
    }

    #[test]
    fn overlap_is_symmetric_and_detects_true_contact() {
        let a = KDop::from_points(DopKind::Dop26, &octahedron(Vec3::ZERO, 1.0));
        let b = KDop::from_points(DopKind::Dop26, &octahedron(Vec3::new(0.3, 0.0, 0.0), 1.0));
        assert!(a.overlaps(&b));
        assert!(b.overlaps(&a));
    }

    #[test]
    fn merge_grows_to_the_union() {
        let mut a = KDop::from_points(DopKind::Dop18, &octahedron(Vec3::ZERO, 1.0));
        let b = KDop::from_points(DopKind::Dop18, &octahedron(Vec3::new(2.0, 0.0, 0.0), 1.0));
        let far = Vec3::new(3.0, 0.0, 0.0);
        assert!(!a.contains_point(far));
        a.merge(&b);
        assert!(a.contains_point(far));
        assert!(a.contains_point(Vec3::new(-1.0, 0.0, 0.0)));
    }

    #[test]
    fn merged_matches_building_from_all_points() {
        let a_pts = octahedron(Vec3::ZERO, 1.0);
        let b_pts = octahedron(Vec3::new(1.5, 0.5, -0.5), 0.75);
        let a = KDop::from_points(DopKind::Dop26, &a_pts);
        let b = KDop::from_points(DopKind::Dop26, &b_pts);

        let mut all = a_pts.clone();
        all.extend_from_slice(&b_pts);
        let direct = KDop::from_points(DopKind::Dop26, &all);

        assert_eq!(a.merged(&b), direct);
    }

    #[test]
    fn from_aabb_matches_from_corner_points() {
        let min = Vec3::new(-1.0, -2.0, -3.0);
        let max = Vec3::new(4.0, 5.0, 6.0);
        let from_box = KDop::from_aabb(DopKind::Dop26, min, max);
        let (lo, hi) = from_box.to_aabb();
        assert_eq!(lo, min);
        assert_eq!(hi, max);
    }

    #[test]
    fn empty_volume_is_invalid_and_never_overlaps() {
        let empty = KDop::empty(DopKind::Dop14);
        let real = KDop::from_points(DopKind::Dop14, &unit_cube_points());
        assert!(!empty.is_valid());
        assert!(!empty.overlaps(&real));
        assert!(!real.overlaps(&empty));
        assert!(!empty.contains_point(Vec3::ZERO));
    }

    #[test]
    fn different_kinds_never_overlap() {
        let a = KDop::from_points(DopKind::Dop6, &unit_cube_points());
        let b = KDop::from_points(DopKind::Dop14, &unit_cube_points());
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn construction_is_deterministic() {
        let pts = octahedron(Vec3::new(0.1, -0.2, 0.3), 1.3);
        let first = KDop::from_points(DopKind::Dop26, &pts);
        let second = KDop::from_points(DopKind::Dop26, &pts);
        assert_eq!(first, second);
    }
}
