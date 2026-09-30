//! Analytic axis-aligned box primitive and its single-level `BVH`.
//!
//! Alongside spheres (see [`super::sphere`]), the axis-aligned box is the other
//! workhorse *procedural* primitive on the hardware ray-tracing path: voxels,
//! bricks, collision proxies, light/portal volumes, and debug boxes are all
//! defined by a closed-form surface rather than a triangle mesh. On DXR/Vulkan
//! these ride the *procedural-primitive* path where the `BLAS` stores one
//! axis-aligned bounding box per primitive and an *intersection shader* refines
//! the hit; for a box that bounding box *is* the surface, so the intersection is
//! the classic slab test. This module is the `CPU` golden reference for that
//! path: an [`AabbPrimitive`] with a slab intersection the shader mirrors, plus
//! an [`AabbBvh`] reusing the shared binned-`SAH` [`build_linear_bvh`] and the
//! same ordered slab walk the triangle [`super::bvh::Bvh`] and [`SphereBvh`]
//! use.
//!
//! The intersection tracks which slab produced the near/far bound so it can
//! report the exact face normal (axis-aligned, unit, oriented against the
//! incident ray) and a front/back flag, using only mul/div/min/max/compare and
//! `copysign` — no transcendental call — so it is bit-reproducible on the `GPU`.
//!
//! [`SphereBvh`]: super::sphere::SphereBvh

use super::bvh::{build_linear_bvh, Aabb, BvhBuildConfig, LinearBvhNode};
use super::traversal::Ray;

/// An analytic axis-aligned box primitive in world space.
///
/// `primitive` is the caller's stable id (mirroring [`super::bvh::Triangle`] and
/// [`super::sphere::Sphere`]): the [`AabbBvh`] builder reorders boxes internally
/// but always reports hits by this id. The corners are stored normalized so that
/// `min[k] <= max[k]` on every axis; a caller that passes swapped corners has
/// them folded to a well-formed box.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AabbPrimitive {
    /// Lower corner (`min` on each axis).
    min: [f32; 3],
    /// Upper corner (`max` on each axis).
    max: [f32; 3],
    /// Caller's stable primitive id, reported unchanged on every hit.
    primitive: u32,
}

impl AabbPrimitive {
    /// Builds a box spanning corners `a`..`b` (normalized per axis so
    /// `min <= max`) with stable id `primitive`.
    #[must_use]
    pub fn new(a: [f32; 3], b: [f32; 3], primitive: u32) -> Self {
        Self {
            min: [a[0].min(b[0]), a[1].min(b[1]), a[2].min(b[2])],
            max: [a[0].max(b[0]), a[1].max(b[1]), a[2].max(b[2])],
            primitive,
        }
    }

    /// Lower corner.
    #[must_use]
    pub fn min(&self) -> [f32; 3] {
        self.min
    }

    /// Upper corner.
    #[must_use]
    pub fn max(&self) -> [f32; 3] {
        self.max
    }

    /// Caller's stable primitive id.
    #[must_use]
    pub fn primitive(&self) -> u32 {
        self.primitive
    }

    /// The box as an [`Aabb`]; this is exactly the procedural-primitive bounding
    /// box the hardware `BLAS` stores (the box *is* its own bounds).
    #[must_use]
    pub fn aabb(&self) -> Aabb {
        Aabb::new(self.min, self.max)
    }

    /// Nearest ray/box intersection inside `ray`'s `[t_min, t_max]` interval, or
    /// `None` when the ray misses the box within that interval.
    ///
    /// The reported [`AabbHit::normal`] is the axis-aligned unit face normal
    /// oriented *against* the incident ray, and [`AabbHit::front_face`] is `true`
    /// when the outward-facing side was struck (a ray originating inside the box
    /// exits through a face and reports `front_face == false` with the normal
    /// flipped inward). A zero-length ray direction never reports a hit.
    #[must_use]
    pub fn intersect(&self, ray: &Ray) -> Option<AabbHit> {
        let origin = ray.origin();
        let direction = ray.direction();
        if direction[0] == 0.0 && direction[1] == 0.0 && direction[2] == 0.0 {
            return None;
        }

        // Slab test tracking which axis produced the near (entry) and far (exit)
        // bounds so the face normal can be reconstructed.
        let mut t_near = f32::NEG_INFINITY;
        let mut t_far = f32::INFINITY;
        let mut near_axis = 0usize;
        let mut far_axis = 0usize;
        for axis in 0..3 {
            let inv = 1.0 / direction[axis];
            let mut t0 = (self.min[axis] - origin[axis]) * inv;
            let mut t1 = (self.max[axis] - origin[axis]) * inv;
            if inv < 0.0 {
                core::mem::swap(&mut t0, &mut t1);
            }
            if t0 > t_near {
                t_near = t0;
                near_axis = axis;
            }
            if t1 < t_far {
                t_far = t1;
                far_axis = axis;
            }
            if t_far < t_near {
                return None;
            }
        }

        // Select the entry hit if it lies in the interval, else the exit hit
        // (ray started inside the box), mirroring the sphere near/far choice.
        let t_min = ray.t_min();
        let t_max = ray.t_max();
        let range = t_min..=t_max;
        let (t, axis, front_face) = if range.contains(&t_near) {
            (t_near, near_axis, true)
        } else if range.contains(&t_far) {
            (t_far, far_axis, false)
        } else {
            return None;
        };

        // Geometric outward normal on the struck face: entry through the min
        // face when the ray advances along `+axis` (outward `-axis`) and through
        // the max face otherwise; the exit face is the opposite side.
        let advancing = direction[axis] > 0.0;
        let outward_sign = if front_face == advancing { -1.0 } else { 1.0 };
        let mut outward = [0.0f32; 3];
        outward[axis] = outward_sign;
        // Orient against the incident ray so shading always sees a normal facing
        // the viewer, matching [`super::sphere::SphereHit`].
        let normal = if front_face {
            outward
        } else {
            [-outward[0], -outward[1], -outward[2]]
        };

        Some(AabbHit {
            t,
            primitive: self.primitive,
            normal,
            front_face,
        })
    }
}

/// A ray/box intersection result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AabbHit {
    /// Ray parameter at the intersection (distance in `direction` lengths).
    pub t: f32,
    /// Stable id of the box that was hit.
    pub primitive: u32,
    /// Axis-aligned unit face normal oriented against the incident ray.
    pub normal: [f32; 3],
    /// `true` when the outward-facing side was struck; `false` for a back face
    /// (ray originating inside the box), whose `normal` is flipped inward.
    pub front_face: bool,
}

/// A single-level `BVH` over analytic [`AabbPrimitive`] boxes.
///
/// Empty input yields an empty hierarchy ([`AabbBvh::is_empty`]); traversal of an
/// empty hierarchy never reports a hit. The layout and ordered slab walk mirror
/// the triangle [`super::bvh::Bvh`] and [`super::sphere::SphereBvh`] so all
/// primitive kinds share one acceleration-structure contract.
#[derive(Clone, Debug, PartialEq)]
pub struct AabbBvh {
    /// Flattened `BVH` nodes; the root (when present) is index `0`.
    nodes: Vec<LinearBvhNode>,
    /// Boxes reordered so each leaf owns a contiguous slice.
    boxes: Vec<AabbPrimitive>,
}

impl AabbBvh {
    /// Builds a `BVH` over `boxes` with [`BvhBuildConfig::default`].
    #[must_use]
    pub fn build(boxes: &[AabbPrimitive]) -> Self {
        Self::build_with(boxes, BvhBuildConfig::default())
    }

    /// Builds a `BVH` over `boxes` with the given binned-`SAH` `config`.
    ///
    /// The builder runs over each box's [`AabbPrimitive::aabb`] and then reorders
    /// the boxes by the returned primitive order so every leaf's
    /// `[first_primitive, first_primitive + primitive_count)` slice indexes
    /// directly into [`AabbBvh::boxes`].
    #[must_use]
    pub fn build_with(boxes: &[AabbPrimitive], config: BvhBuildConfig) -> Self {
        let bounds: Vec<Aabb> = boxes.iter().map(AabbPrimitive::aabb).collect();
        let (nodes, order) = build_linear_bvh(&bounds, config);
        let boxes = order.iter().map(|&i| boxes[i as usize]).collect();
        Self { nodes, boxes }
    }

    /// Number of flattened `BVH` nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of boxes in the hierarchy.
    #[must_use]
    pub fn primitive_count(&self) -> usize {
        self.boxes.len()
    }

    /// True when the hierarchy holds no primitives.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Root bounds, or the empty box when the hierarchy is empty.
    #[must_use]
    pub fn bounds(&self) -> Aabb {
        self.nodes.first().map_or(Aabb::empty(), |node| node.bounds)
    }

    /// The flattened node array.
    #[must_use]
    pub fn nodes(&self) -> &[LinearBvhNode] {
        &self.nodes
    }

    /// The reordered box array (leaf slices index into this).
    #[must_use]
    pub fn boxes(&self) -> &[AabbPrimitive] {
        &self.boxes
    }

    /// Nearest intersection along `ray`, or `None` if the ray hits nothing.
    ///
    /// Walks the flattened nodes with an explicit stack, visiting the child on
    /// the near side of the split axis first so the running `t_max` shrinks as
    /// fast as possible and far subtrees are culled by the slab test.
    #[must_use]
    pub fn closest_hit(&self, ray: &Ray) -> Option<AabbHit> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut ray = *ray;
        let mut best: Option<AabbHit> = None;

        let mut stack = [0u32; 64];
        let mut sp = 0usize;
        let mut node_index = 0u32;
        loop {
            let node = &self.nodes[node_index as usize];
            if ray
                .aabb_interval(&node.bounds, ray.t_min(), ray.t_max())
                .is_some()
            {
                if node.is_leaf() {
                    let start = node.first_primitive as usize;
                    let end = start + node.primitive_count as usize;
                    for shape in &self.boxes[start..end] {
                        if let Some(hit) = shape.intersect(&ray) {
                            ray = Ray::new(ray.origin(), ray.direction(), ray.t_min(), hit.t);
                            best = Some(hit);
                        }
                    }
                    match stack_pop(&mut stack, &mut sp) {
                        Some(n) => node_index = n,
                        None => break,
                    }
                } else {
                    let first_child = node_index + 1;
                    let second_child = node.second_child;
                    let neg = ray.direction()[node.axis as usize] < 0.0;
                    let (near, far) = if neg {
                        (second_child, first_child)
                    } else {
                        (first_child, second_child)
                    };
                    if sp < stack.len() {
                        stack[sp] = far;
                        sp += 1;
                    }
                    node_index = near;
                }
            } else {
                match stack_pop(&mut stack, &mut sp) {
                    Some(n) => node_index = n,
                    None => break,
                }
            }
        }
        best
    }

    /// True when *any* box intersects `ray` inside its interval.
    ///
    /// Returns on the first hit without tracking the nearest, so it is the cheap
    /// query for shadow and ambient-occlusion rays.
    #[must_use]
    pub fn any_hit(&self, ray: &Ray) -> bool {
        if self.nodes.is_empty() {
            return false;
        }
        let mut stack = [0u32; 64];
        let mut sp = 0usize;
        let mut node_index = 0u32;
        loop {
            let node = &self.nodes[node_index as usize];
            if ray
                .aabb_interval(&node.bounds, ray.t_min(), ray.t_max())
                .is_some()
            {
                if node.is_leaf() {
                    let start = node.first_primitive as usize;
                    let end = start + node.primitive_count as usize;
                    for shape in &self.boxes[start..end] {
                        if shape.intersect(ray).is_some() {
                            return true;
                        }
                    }
                    match stack_pop(&mut stack, &mut sp) {
                        Some(n) => node_index = n,
                        None => break,
                    }
                } else {
                    let first_child = node_index + 1;
                    if sp < stack.len() {
                        stack[sp] = node.second_child;
                        sp += 1;
                    }
                    node_index = first_child;
                }
            } else {
                match stack_pop(&mut stack, &mut sp) {
                    Some(n) => node_index = n,
                    None => break,
                }
            }
        }
        false
    }
}

/// Pops the top node index off the traversal stack, or `None` when empty.
fn stack_pop(stack: &mut [u32; 64], sp: &mut usize) -> Option<u32> {
    if *sp == 0 {
        None
    } else {
        *sp -= 1;
        Some(stack[*sp])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small deterministic xorshift `RNG`, matching the other `ray_scene`
    /// suites so tests never depend on an external crate.
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }
        fn next_u32(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            (x >> 32) as u32
        }
        fn unit(&mut self) -> f32 {
            self.next_u32() as f32 / u32::MAX as f32
        }
        fn range(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * self.unit()
        }
    }

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps * (1.0 + a.abs().max(b.abs()))
    }

    /// Brute-force nearest hit over the *original* (unordered) box list, used as
    /// the ground truth the `BVH` must reproduce.
    fn brute_closest(boxes: &[AabbPrimitive], ray: &Ray) -> Option<AabbHit> {
        let mut best: Option<AabbHit> = None;
        let mut ray = *ray;
        for shape in boxes {
            if let Some(hit) = shape.intersect(&ray) {
                ray = Ray::new(ray.origin(), ray.direction(), ray.t_min(), hit.t);
                best = Some(hit);
            }
        }
        best
    }

    fn random_boxes(rng: &mut Rng, count: u32) -> Vec<AabbPrimitive> {
        (0..count)
            .map(|i| {
                let c = [
                    rng.range(-6.0, 6.0),
                    rng.range(-6.0, 6.0),
                    rng.range(-6.0, 6.0),
                ];
                let h = [
                    rng.range(0.1, 1.2),
                    rng.range(0.1, 1.2),
                    rng.range(0.1, 1.2),
                ];
                AabbPrimitive::new(
                    [c[0] - h[0], c[1] - h[1], c[2] - h[2]],
                    [c[0] + h[0], c[1] + h[1], c[2] + h[2]],
                    i,
                )
            })
            .collect()
    }

    #[test]
    fn new_normalizes_swapped_corners() {
        let b = AabbPrimitive::new([2.0, 5.0, -1.0], [-3.0, 1.0, 4.0], 7);
        assert_eq!(b.min(), [-3.0, 1.0, -1.0]);
        assert_eq!(b.max(), [2.0, 5.0, 4.0]);
        assert_eq!(b.primitive(), 7);
        assert_eq!(b.aabb(), Aabb::new([-3.0, 1.0, -1.0], [2.0, 5.0, 4.0]));
    }

    #[test]
    fn axis_hit_reports_the_face_normal() {
        let b = AabbPrimitive::new([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], 0);
        // Ray down the +x axis strikes the -x face; normal faces the ray (-x).
        let ray = Ray::infinite([-5.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        let hit = b.intersect(&ray).expect("should hit the box");
        assert!(hit.front_face);
        assert_eq!(hit.primitive, 0);
        assert!(approx(hit.t, 4.0, 1e-6));
        assert_eq!(hit.normal, [-1.0, 0.0, 0.0]);
    }

    #[test]
    fn ray_from_inside_reports_back_face() {
        let b = AabbPrimitive::new([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], 3);
        // Origin inside the box, travelling +y: exits through the +y (max) face.
        let ray = Ray::infinite([0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let hit = b.intersect(&ray).expect("inside ray still hits an exit face");
        assert!(!hit.front_face);
        assert!(approx(hit.t, 1.0, 1e-6));
        // Exit face is +y; reported normal is flipped against the ray => -y.
        assert_eq!(hit.normal, [0.0, -1.0, 0.0]);
    }

    #[test]
    fn parallel_ray_outside_slab_misses() {
        let b = AabbPrimitive::new([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], 0);
        // Travels along +x but offset in y beyond the box: never enters.
        let ray = Ray::infinite([-5.0, 3.0, 0.0], [1.0, 0.0, 0.0]);
        assert!(b.intersect(&ray).is_none());
    }

    #[test]
    fn zero_direction_never_hits() {
        let b = AabbPrimitive::new([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], 0);
        let ray = Ray::infinite([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        assert!(b.intersect(&ray).is_none());
    }

    #[test]
    fn empty_bvh_never_hits() {
        let bvh = AabbBvh::build(&[]);
        assert!(bvh.is_empty());
        assert_eq!(bvh.node_count(), 0);
        assert_eq!(bvh.primitive_count(), 0);
        assert_eq!(bvh.bounds(), Aabb::empty());
        let ray = Ray::infinite([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
        assert!(bvh.closest_hit(&ray).is_none());
        assert!(!bvh.any_hit(&ray));
    }

    #[test]
    fn bvh_preserves_primitive_ids_and_bounds() {
        let mut rng = Rng::new(0x00B0_7E55);
        let boxes = random_boxes(&mut rng, 40);
        let bvh = AabbBvh::build(&boxes);
        assert_eq!(bvh.primitive_count(), boxes.len());
        // Every reordered box keeps a valid original id and the root bounds
        // enclose all of them.
        let mut ids: Vec<u32> = bvh.boxes().iter().map(AabbPrimitive::primitive).collect();
        ids.sort_unstable();
        let expected: Vec<u32> = (0..boxes.len() as u32).collect();
        assert_eq!(ids, expected);
        let root = bvh.bounds();
        for shape in bvh.boxes() {
            let ab = shape.aabb();
            for k in 0..3 {
                assert!(root.min[k] <= ab.min[k] + 1e-4);
                assert!(root.max[k] >= ab.max[k] - 1e-4);
            }
        }
    }

    #[test]
    fn bvh_closest_matches_brute_force_bit_for_bit() {
        let mut rng = Rng::new(0xB0_1234_5678);
        let boxes = random_boxes(&mut rng, 96);
        let bvh = AabbBvh::build(&boxes);

        for _ in 0..4_000 {
            let origin = [
                rng.range(-10.0, 10.0),
                rng.range(-10.0, 10.0),
                rng.range(-10.0, 10.0),
            ];
            let dir = [
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
            ];
            if dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2] < 1e-6 {
                continue;
            }
            let ray = Ray::infinite(origin, dir);

            let expected = brute_closest(&boxes, &ray);
            let actual = bvh.closest_hit(&ray);
            match (expected, actual) {
                (None, None) => {}
                (Some(e), Some(a)) => {
                    assert_eq!(e.t.to_bits(), a.t.to_bits(), "t mismatch");
                    assert_eq!(e.primitive, a.primitive, "id mismatch");
                    assert_eq!(e.normal, a.normal, "normal mismatch");
                    assert_eq!(e.front_face, a.front_face, "face mismatch");
                }
                _ => panic!("closest_hit existence disagrees with brute force"),
            }
            assert_eq!(bvh.any_hit(&ray), expected.is_some());
        }
    }
}
