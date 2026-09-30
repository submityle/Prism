//! Sequential golden twin for `GPU` ray-vs-`BVH` scene queries.
//!
//! Once the [`Lbvh`] is built, a *ray query* finds where a directed ray first
//! enters the leaf primitives: [`cpu_bvh_raycast_closest`] returns the nearest hit
//! (`PhysX` closest-hit / Unreal `LineTrace`) and [`cpu_bvh_raycast_any`]
//! reports whether any primitive is hit at all (`PhysX` any-hit / a shadow or
//! occlusion probe), which can early-out on the first crossing.
//!
//! # Why this twin scans every leaf
//!
//! The device kernel walks the hierarchy and prunes whole subtrees whose box the
//! ray misses, exactly like the overlap-pair query in [`crate::bvh::query`].
//! This twin deliberately does **not** walk the tree: it scans every leaf box by
//! brute force. A brute-force scan is a stronger, fully independent oracle, so a
//! passing real-device parity test is evidence the `GPU` traversal and its
//! pruning agree with the ground truth over the whole primitive set, not merely
//! that two copies of the same traversal agree.
//!
//! # Slab intersection, bit-for-bit
//!
//! Both sides intersect the ray with a box using the branch-free slab method of
//! Williams et al., "An Efficient and Robust Ray-Box Intersection Algorithm"
//! (2005): reciprocate the direction per component (a zero component yields a
//! signed infinity, matching `WGSL` `1.0 / 0.0`), map each slab to an entry and
//! exit distance, then take the largest entry and smallest exit across axes.
//! Component-wise `min`/`max` are `NaN`-robust on both glam and `WGSL`, so the
//! degenerate infinities collapse consistently. The entry distance is clamped to
//! zero so a ray whose origin is inside the box reports a hit at distance zero.
//!
//! Because the only inexact step is the reciprocal (`WGSL` division carries up to
//! 2.5 `ULP` of tolerance, unlike the bit-exact add/subtract/multiply), the
//! parity test matches the primitive index exactly and the hit distance within a
//! small tolerance, the same contract the `XPBD` and fluid twins use.
//!
//! # Provenance
//!
//! Williams et al. slab intersection (2005) over the linear `BVH` of Karras,
//! "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees"
//! (High Performance Graphics 2012). No Unreal Engine source or derived code.

use glam::Vec3;

use super::config::Aabb;
use super::cpu::Lbvh;

/// A directed ray with a maximum travel distance.
///
/// The ray is the point set `origin + t * dir` for `t` in `[0, t_max]`. The
/// direction need not be normalised, but the reported hit distance `t` is then
/// measured in units of `dir`'s length rather than world units, so callers that
/// want a metric distance should pass a unit `dir`.
#[derive(Clone, Copy, Debug)]
pub struct Ray {
    /// The ray's starting point.
    pub origin: Vec3,
    /// The ray's direction; a zero component is handled by the slab test.
    pub dir: Vec3,
    /// The largest `t` a hit may lie at; hits beyond it are rejected.
    pub t_max: f32,
}

impl Ray {
    /// Creates a ray from its `origin`, `dir`ection, and maximum distance.
    #[must_use]
    pub fn new(origin: Vec3, dir: Vec3, t_max: f32) -> Ray {
        Ray { origin, dir, t_max }
    }
}

/// A single ray hit: the primitive that was struck and the entry distance.
///
/// `prim` is the original primitive index (the position in the slice passed to
/// [`cpu_build_lbvh`](super::cpu::cpu_build_lbvh)), not the Morton-sorted leaf slot;
/// `t` is the ray parameter where the ray enters that primitive's box, clamped
/// to zero when the origin is already inside.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayHit {
    /// The original primitive index that was hit.
    pub prim: u32,
    /// The ray parameter `t` at the entry point.
    pub t: f32,
}

/// The entry distance where `ray` enters `aabb`, or [`None`] if it misses.
///
/// This is the branch-free slab test shared bit-for-bit with the device kernel.
/// The direction is reciprocated per component with glam's component-wise
/// division, so a zero component becomes a signed infinity exactly as `WGSL`
/// `1.0 / 0.0` does. The per-component `min`/`max` picking the near and far slab
/// planes are `NaN`-robust on both sides, so an axis with a zero direction and an
/// origin outside the slab collapses to an empty interval and the box is missed.
///
/// The returned entry distance is clamped to zero, so an origin inside the box
/// reports a hit at distance zero rather than a negative entry. A hit is
/// accepted only when the entry does not exceed the exit and lies within
/// `ray.t_max`.
#[must_use]
fn slab_enter(ray: &Ray, aabb: &Aabb) -> Option<f32> {
    let inv = Vec3::ONE / ray.dir;
    let t0 = (aabb.min - ray.origin) * inv;
    let t1 = (aabb.max - ray.origin) * inv;
    let t_small = t0.min(t1);
    let t_big = t0.max(t1);
    let t_near = t_small.max_element().max(0.0);
    let t_far = t_big.min_element();
    if t_near <= t_far && t_near <= ray.t_max {
        Some(t_near)
    } else {
        None
    }
}

/// Finds the nearest primitive `ray` hits in `lbvh`, the golden twin of the
/// `GPU` closest-hit query.
///
/// Every leaf box is tested by brute force; the smallest entry distance wins.
/// Ties keep the lowest leaf slot (the first primitive visited), so the result
/// is deterministic even when two boxes are entered at the same distance.
/// Returns [`None`] when the ray misses every primitive or the tree is empty.
///
/// The reported [`RayHit::prim`] is the original primitive index recovered
/// through [`Lbvh::sorted_indices`], matching what the device kernel gathers.
#[must_use]
pub fn cpu_bvh_raycast_closest(lbvh: &Lbvh, ray: Ray) -> Option<RayHit> {
    let mut best: Option<RayHit> = None;
    for slot in 0..lbvh.num_leaves {
        let Some(t) = slab_enter(&ray, &lbvh.leaf_aabb[slot]) else {
            continue;
        };
        let closer = match best {
            Some(hit) => t < hit.t,
            None => true,
        };
        if closer {
            best = Some(RayHit {
                prim: lbvh.sorted_indices[slot],
                t,
            });
        }
    }
    best
}

/// Reports whether `ray` hits any primitive in `lbvh`, the golden twin of the
/// `GPU` any-hit query.
///
/// Scans leaf boxes by brute force and returns `true` on the first hit, since an
/// any-hit query only needs existence, not the nearest primitive. Returns
/// `false` for a ray that misses every primitive or an empty tree.
#[must_use]
pub fn cpu_bvh_raycast_any(lbvh: &Lbvh, ray: Ray) -> bool {
    (0..lbvh.num_leaves).any(|slot| slab_enter(&ray, &lbvh.leaf_aabb[slot]).is_some())
}

#[cfg(test)]
mod tests {
    use super::{cpu_bvh_raycast_any, cpu_bvh_raycast_closest, Ray};
    use crate::bvh::config::Aabb;
    use crate::bvh::cpu::cpu_build_lbvh;
    use glam::Vec3;

    /// A half-unit box centred at `(x, y, z)`, matching the parity-test scenes.
    fn box_at(x: f32, y: f32, z: f32) -> Aabb {
        Aabb::new(
            Vec3::new(x - 0.5, y - 0.5, z - 0.5),
            Vec3::new(x + 0.5, y + 0.5, z + 0.5),
        )
    }

    #[test]
    fn hits_a_box_in_front() {
        let lbvh = cpu_build_lbvh(&[box_at(5.0, 0.0, 0.0)]);
        let ray = Ray::new(Vec3::ZERO, Vec3::X, 100.0);
        let hit = cpu_bvh_raycast_closest(&lbvh, ray).expect("hit");
        assert_eq!(hit.prim, 0);
        assert!((hit.t - 4.5).abs() < 1e-6, "entry distance {}", hit.t);
        assert!(cpu_bvh_raycast_any(&lbvh, ray));
    }

    #[test]
    fn misses_a_box_behind_the_origin() {
        let lbvh = cpu_build_lbvh(&[box_at(5.0, 0.0, 0.0)]);
        let ray = Ray::new(Vec3::ZERO, Vec3::NEG_X, 100.0);
        assert_eq!(cpu_bvh_raycast_closest(&lbvh, ray), None);
        assert!(!cpu_bvh_raycast_any(&lbvh, ray));
    }

    #[test]
    fn misses_a_box_offset_from_the_ray() {
        let lbvh = cpu_build_lbvh(&[box_at(5.0, 0.0, 0.0)]);
        let ray = Ray::new(Vec3::new(0.0, 2.0, 0.0), Vec3::X, 100.0);
        assert_eq!(cpu_bvh_raycast_closest(&lbvh, ray), None);
        assert!(!cpu_bvh_raycast_any(&lbvh, ray));
    }

    #[test]
    fn origin_inside_the_box_hits_at_zero() {
        let lbvh = cpu_build_lbvh(&[box_at(0.0, 0.0, 0.0)]);
        let ray = Ray::new(Vec3::ZERO, Vec3::X, 100.0);
        let hit = cpu_bvh_raycast_closest(&lbvh, ray).expect("hit");
        assert_eq!(hit.prim, 0);
        assert_eq!(hit.t, 0.0);
    }

    #[test]
    fn t_max_rejects_a_far_box() {
        let lbvh = cpu_build_lbvh(&[box_at(5.0, 0.0, 0.0)]);
        let ray = Ray::new(Vec3::ZERO, Vec3::X, 3.0);
        assert_eq!(cpu_bvh_raycast_closest(&lbvh, ray), None);
        assert!(!cpu_bvh_raycast_any(&lbvh, ray));
    }

    #[test]
    fn empty_tree_never_hits() {
        let lbvh = cpu_build_lbvh(&[]);
        let ray = Ray::new(Vec3::ZERO, Vec3::X, 100.0);
        assert_eq!(cpu_bvh_raycast_closest(&lbvh, ray), None);
        assert!(!cpu_bvh_raycast_any(&lbvh, ray));
    }

    #[test]
    fn closest_of_several_boxes_wins() {
        let lbvh = cpu_build_lbvh(&[
            box_at(10.0, 0.0, 0.0),
            box_at(5.0, 0.0, 0.0),
            box_at(20.0, 0.0, 0.0),
        ]);
        let ray = Ray::new(Vec3::ZERO, Vec3::X, 100.0);
        let hit = cpu_bvh_raycast_closest(&lbvh, ray).expect("hit");
        assert_eq!(hit.prim, 1);
        assert!((hit.t - 4.5).abs() < 1e-6, "entry distance {}", hit.t);
    }
}
