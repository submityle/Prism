//! Axis-aligned bounding boxes and scene bounds for the `LBVH` builder.
//!
//! An [`Aabb`] is the leaf primitive the hierarchy is built over; [`SceneBounds`]
//! is the union of all leaf boxes and defines the cube that Morton coordinates
//! are quantised into. Keeping both here, separate from the Morton and tree
//! code, lets the `CPU` twin and the `GPU` kernels share one definition of the
//! quantisation domain so their codes agree bit-for-bit.
//!
//! # Provenance
//!
//! Plain bounding-box algebra; the linear `BVH` it feeds is the technique of
//! Karras, "Maximizing Parallelism in the Construction of BVHs, Octrees, and
//! k-d Trees" (High Performance Graphics 2012). No Unreal Engine source or
//! derived code.

use glam::Vec3;

/// An axis-aligned bounding box.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    /// Minimum corner.
    pub min: Vec3,
    /// Maximum corner.
    pub max: Vec3,
}

impl Aabb {
    /// Creates a box from its `min` and `max` corners.
    #[must_use]
    pub fn new(min: Vec3, max: Vec3) -> Aabb {
        Aabb { min, max }
    }

    /// The geometric centre of the box.
    #[must_use]
    pub fn centroid(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// The component-wise union (smallest box containing both `self` and
    /// `other`).
    #[must_use]
    pub fn union(&self, other: &Aabb) -> Aabb {
        Aabb {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }
}

/// The union of every leaf box: the cube Morton coordinates are quantised into.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SceneBounds {
    /// Minimum corner of the union box.
    pub min: Vec3,
    /// Maximum corner of the union box.
    pub max: Vec3,
}

impl SceneBounds {
    /// Computes the bounds as the union of every box in `boxes`.
    ///
    /// Returns [`None`] for an empty slice, since there is no domain to quantise
    /// into.
    #[must_use]
    pub fn of(boxes: &[Aabb]) -> Option<SceneBounds> {
        let first = boxes.first()?;
        let mut min = first.min;
        let mut max = first.max;
        for b in &boxes[1..] {
            min = min.min(b.min);
            max = max.max(b.max);
        }
        Some(SceneBounds { min, max })
    }

    /// The per-axis extent (`max - min`); an axis may be zero for a flat scene.
    #[must_use]
    pub fn extent(&self) -> Vec3 {
        self.max - self.min
    }
}

#[cfg(test)]
mod tests {
    use super::{Aabb, SceneBounds};
    use glam::Vec3;

    #[test]
    fn centroid_is_the_midpoint() {
        let b = Aabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 4.0, 6.0));
        assert_eq!(b.centroid(), Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn union_grows_to_contain_both() {
        let a = Aabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
        let b = Aabb::new(Vec3::new(-1.0, 2.0, 0.5), Vec3::new(0.5, 3.0, 4.0));
        let u = a.union(&b);
        assert_eq!(u.min, Vec3::new(-1.0, 0.0, 0.0));
        assert_eq!(u.max, Vec3::new(1.0, 3.0, 4.0));
    }

    #[test]
    fn scene_bounds_union_every_box() {
        let boxes = [
            Aabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0)),
            Aabb::new(Vec3::new(2.0, -1.0, 0.0), Vec3::new(3.0, 0.0, 5.0)),
        ];
        let bounds = SceneBounds::of(&boxes).expect("non-empty");
        assert_eq!(bounds.min, Vec3::new(0.0, -1.0, 0.0));
        assert_eq!(bounds.max, Vec3::new(3.0, 1.0, 5.0));
        assert_eq!(bounds.extent(), Vec3::new(3.0, 2.0, 5.0));
    }

    #[test]
    fn empty_scene_has_no_bounds() {
        assert_eq!(SceneBounds::of(&[]), None);
    }
}
