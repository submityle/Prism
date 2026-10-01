//! Analytic oriented box (`OBB`) primitive and its single-level `BVH`.
//!
//! The axis-aligned box ([`super::aabb_primitive`]) covers voxels and bricks,
//! but most real props — crates, bricks laid at an angle, collision proxies,
//! light/portal volumes attached to rotated nodes — are *oriented*. On the
//! `DXR`/Vulkan procedural-primitive path the `BLAS` stores one axis-aligned
//! bounding box per primitive and an *intersection shader* refines the hit; for
//! an oriented box the stored `AABB` is the world-space bound of the rotated box
//! and the shader runs the slab test in the box's own frame. This module is the
//! `CPU` golden reference for that path: an [`Obb`] carrying a center,
//! half-extents, and an orthonormal frame, plus an [`ObbBvh`] reusing the shared
//! binned-`SAH` [`build_linear_bvh`] over the per-box world bounds and the same
//! ordered slab walk the triangle [`super::bvh::Bvh`] uses.
//!
//! The intersection projects the ray into the box frame (one dot product per
//! axis for the origin offset and the direction), then runs the identical slab
//! test the [`super::aabb_primitive::AabbPrimitive`] uses, tracking which slab
//! produced the near/far bound so it can reconstruct the exact face normal. The
//! local face normal `±e_axis` is rotated back to world space by the stored
//! frame axis, so for an orthonormal frame it is already unit. Every step is
//! mul/div/min/max/compare — no transcendental call — so it is bit-reproducible
//! on the `GPU`. The frame axes are stored verbatim (not re-normalized) so the
//! `GPU` decode reproduces them bit-for-bit; callers pass an orthonormal frame.

use super::bvh::{build_linear_bvh, Aabb, BvhBuildConfig, LinearBvhNode};
use super::traversal::Ray;

/// An analytic oriented box primitive in world space.
///
/// `primitive` is the caller's stable id (mirroring [`super::bvh::Triangle`]):
/// the [`ObbBvh`] builder reorders boxes internally but always reports hits by
/// this id. The half-extents are stored non-negative; the frame `axes` are the
/// three local basis vectors (`u`, `v`, `w`) expressed in world space and are
/// stored verbatim — callers pass an orthonormal frame so the reported normals
/// are unit and the `GPU` decode is bit-exact.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Obb {
    /// Box center in world space.
    center: [f32; 3],
    /// Non-negative half-extents along the local `u`, `v`, `w` axes.
    half: [f32; 3],
    /// Orthonormal frame: world-space `u`, `v`, `w` basis vectors (rows).
    axes: [[f32; 3]; 3],
    /// Caller's stable primitive id, reported unchanged on every hit.
    primitive: u32,
}

impl Obb {
    /// Builds an oriented box at `center` with the given world-space `axes`
    /// (`[u, v, w]`, assumed orthonormal), per-axis `half` extents (folded to
    /// their magnitude), and stable id `primitive`.
    #[must_use]
    pub fn new(
        center: [f32; 3],
        axes: [[f32; 3]; 3],
        half: [f32; 3],
        primitive: u32,
    ) -> Self {
        Self {
            center,
            half: [half[0].abs(), half[1].abs(), half[2].abs()],
            axes,
            primitive,
        }
    }

    /// Box center in world space.
    #[must_use]
    pub fn center(&self) -> [f32; 3] {
        self.center
    }

    /// Non-negative half-extents along the local `u`, `v`, `w` axes.
    #[must_use]
    pub fn half(&self) -> [f32; 3] {
        self.half
    }

    /// Orthonormal frame axes (`[u, v, w]`) in world space.
    #[must_use]
    pub fn axes(&self) -> [[f32; 3]; 3] {
        self.axes
    }

    /// Caller's stable primitive id.
    #[must_use]
    pub fn primitive(&self) -> u32 {
        self.primitive
    }

    /// World-space axis-aligned bounds of the rotated box.
    ///
    /// This is the procedural-primitive `AABB` the hardware `BLAS` stores: the
    /// world extent along each world axis `k` is `Σ_i half[i] · |axes[i][k]|`,
    /// the standard oriented-box projection onto the world axes.
    #[must_use]
    pub fn aabb(&self) -> Aabb {
        let mut world_half = [0.0f32; 3];
        for (k, wh) in world_half.iter_mut().enumerate() {
            *wh = self.half[0] * self.axes[0][k].abs()
                + self.half[1] * self.axes[1][k].abs()
                + self.half[2] * self.axes[2][k].abs();
        }
        Aabb::new(
            [
                self.center[0] - world_half[0],
                self.center[1] - world_half[1],
                self.center[2] - world_half[2],
            ],
            [
                self.center[0] + world_half[0],
                self.center[1] + world_half[1],
                self.center[2] + world_half[2],
            ],
        )
    }

    /// Nearest ray/box intersection inside `ray`'s `[t_min, t_max]` interval, or
    /// `None` when the ray misses the box within that interval.
    ///
    /// The reported [`ObbHit::normal`] is the unit face normal oriented *against*
    /// the incident ray, and [`ObbHit::front_face`] is `true` when the
    /// outward-facing side was struck (a ray originating inside the box exits
    /// through a face and reports `front_face == false` with the normal flipped
    /// inward). A zero-length ray direction never reports a hit.
    #[must_use]
    pub fn intersect(&self, ray: &Ray) -> Option<ObbHit> {
        let world_dir = ray.direction();
        if world_dir[0] == 0.0 && world_dir[1] == 0.0 && world_dir[2] == 0.0 {
            return None;
        }
        let origin = ray.origin();
        let rel = [
            origin[0] - self.center[0],
            origin[1] - self.center[1],
            origin[2] - self.center[2],
        ];
        // Project ray origin offset and direction into the box frame.
        let lo = [
            dot(rel, self.axes[0]),
            dot(rel, self.axes[1]),
            dot(rel, self.axes[2]),
        ];
        let ld = [
            dot(world_dir, self.axes[0]),
            dot(world_dir, self.axes[1]),
            dot(world_dir, self.axes[2]),
        ];

        // Slab test in the local frame against `[-half, +half]`, tracking which
        // axis produced the near (entry) and far (exit) bounds.
        let mut t_near = f32::NEG_INFINITY;
        let mut t_far = f32::INFINITY;
        let mut near_axis = 0usize;
        let mut far_axis = 0usize;
        for axis in 0..3 {
            let inv = 1.0 / ld[axis];
            let mut t0 = (-self.half[axis] - lo[axis]) * inv;
            let mut t1 = (self.half[axis] - lo[axis]) * inv;
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

        // Local outward normal on the struck face: entry through the `-half`
        // face when the ray advances along `+axis` (outward `-axis`), through
        // the `+half` face otherwise; the exit face is the opposite side.
        let advancing = ld[axis] > 0.0;
        let outward_sign = if front_face == advancing { -1.0 } else { 1.0 };
        // Rotate the local axis normal back to world space (orthonormal frame
        // ⇒ unit length). Orient against the incident ray.
        let a = self.axes[axis];
        let outward = [a[0] * outward_sign, a[1] * outward_sign, a[2] * outward_sign];
        let normal = if front_face {
            outward
        } else {
            [-outward[0], -outward[1], -outward[2]]
        };

        Some(ObbHit {
            t,
            primitive: self.primitive,
            normal,
            front_face,
        })
    }
}

/// A ray/oriented-box intersection result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObbHit {
    /// Ray parameter at the intersection (distance in `direction` lengths).
    pub t: f32,
    /// Stable id of the box that was hit.
    pub primitive: u32,
    /// Unit face normal oriented against the incident ray.
    pub normal: [f32; 3],
    /// `true` when the outward-facing side was struck; `false` for a back face
    /// (ray originating inside the box), whose `normal` is flipped inward.
    pub front_face: bool,
}

/// A single-level `BVH` over analytic [`Obb`] primitives.
///
/// Empty input yields an empty hierarchy ([`ObbBvh::is_empty`]); traversal of an
/// empty hierarchy never reports a hit. The layout and ordered slab walk mirror
/// the triangle [`super::bvh::Bvh`] so all primitive kinds share one
/// acceleration-structure contract.
#[derive(Clone, Debug, PartialEq)]
pub struct ObbBvh {
    /// Flattened `BVH` nodes; the root (when present) is index `0`.
    nodes: Vec<LinearBvhNode>,
    /// Boxes reordered so each leaf owns a contiguous slice.
    boxes: Vec<Obb>,
}

impl ObbBvh {
    /// Builds a `BVH` over `boxes` with [`BvhBuildConfig::default`].
    #[must_use]
    pub fn build(boxes: &[Obb]) -> Self {
        Self::build_with(boxes, BvhBuildConfig::default())
    }

    /// Builds a `BVH` over `boxes` with the given binned-`SAH` `config`.
    ///
    /// The builder runs over each box's world-space [`Obb::aabb`] and then
    /// reorders the boxes by the returned primitive order so every leaf's
    /// `[first_primitive, first_primitive + primitive_count)` slice indexes
    /// directly into [`ObbBvh::boxes`].
    #[must_use]
    pub fn build_with(boxes: &[Obb], config: BvhBuildConfig) -> Self {
        let bounds: Vec<Aabb> = boxes.iter().map(Obb::aabb).collect();
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
    pub fn boxes(&self) -> &[Obb] {
        &self.boxes
    }

    /// Nearest intersection along `ray`, or `None` if the ray hits nothing.
    ///
    /// Walks the flattened nodes with an explicit stack, visiting the child on
    /// the near side of the split axis first so the running `t_max` shrinks as
    /// fast as possible and far subtrees are culled by the slab test.
    #[must_use]
    pub fn closest_hit(&self, ray: &Ray) -> Option<ObbHit> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut ray = *ray;
        let mut best: Option<ObbHit> = None;

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
                    for obb in &self.boxes[start..end] {
                        if let Some(hit) = obb.intersect(&ray) {
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
                    for obb in &self.boxes[start..end] {
                        if obb.intersect(ray).is_some() {
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

/// Dot product of two 3-vectors.
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
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

    /// Builds an orthonormal frame rotated by `angle` about the world `+z`
    /// axis using only algebraic (`sqrt`-based) trig so clippy's transcendental
    /// ban is respected: given `s = sin`, `c = cos` supplied as a unit pair.
    fn frame_z(c: f32, s: f32) -> [[f32; 3]; 3] {
        [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]
    }

    /// A unit (cos, sin) pair from a rational parameter, no transcendentals:
    /// the standard `((1 - m²)/(1 + m²), 2m/(1 + m²))` tangent half-angle map.
    fn unit_pair(m: f32) -> (f32, f32) {
        let d = 1.0 + m * m;
        ((1.0 - m * m) / d, 2.0 * m / d)
    }

    fn brute_closest(boxes: &[Obb], ray: &Ray) -> Option<ObbHit> {
        let mut best: Option<ObbHit> = None;
        let mut ray = *ray;
        for obb in boxes {
            if let Some(hit) = obb.intersect(&ray) {
                ray = Ray::new(ray.origin(), ray.direction(), ray.t_min(), hit.t);
                best = Some(hit);
            }
        }
        best
    }

    #[test]
    fn axis_aligned_frame_matches_a_plain_box() {
        // Identity frame ⇒ an OBB is a plain AABB; a +x ray hits the −x face.
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let obb = Obb::new([0.0, 0.0, 0.0], identity, [1.0, 2.0, 3.0], 5);
        let ray = Ray::infinite([-10.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        let hit = obb.intersect(&ray).expect("axis ray must hit");
        assert_eq!(hit.primitive, 5);
        assert!(approx(hit.t, 9.0, 1e-5), "t = {}", hit.t);
        assert!(hit.front_face);
        assert!(approx(hit.normal[0], -1.0, 1e-5));
    }

    #[test]
    fn rotated_box_reports_rotated_face_normal() {
        // Rotate 45° about +z (m = tan(22.5°) ≈ 0.414213...).
        let (c, s) = unit_pair(0.41421356);
        let axes = frame_z(c, s);
        let obb = Obb::new([0.0, 0.0, 0.0], axes, [1.0, 1.0, 1.0], 0);
        // A ray from +x toward the origin hits the rotated +u face; its world
        // normal is +u = (c, s, 0), oriented against the −x ray so it stays +u.
        let ray = Ray::infinite([10.0, 0.0, 0.0], [-1.0, 0.0, 0.0]);
        let hit = obb.intersect(&ray).expect("rotated box must be hit");
        assert!(hit.front_face);
        // Normal is unit and perpendicular to the local v/w plane: its x
        // component equals c (the +u axis x), facing the −x ray (positive x).
        let nlen =
            (hit.normal[0] * hit.normal[0] + hit.normal[1] * hit.normal[1] + hit.normal[2] * hit.normal[2])
                .sqrt();
        assert!(approx(nlen, 1.0, 1e-5), "normal not unit: {nlen}");
        assert!(hit.normal[0] > 0.0, "normal must face the −x ray");
    }

    #[test]
    fn ray_from_inside_reports_back_face() {
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let obb = Obb::new([0.0, 0.0, 0.0], identity, [1.0, 1.0, 1.0], 0);
        let ray = Ray::infinite([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        let hit = obb.intersect(&ray).expect("interior ray exits a face");
        assert!(approx(hit.t, 1.0, 1e-5), "t = {}", hit.t);
        assert!(!hit.front_face);
        // Exit face is +x; its ray-facing normal is flipped to −x.
        assert!(approx(hit.normal[0], -1.0, 1e-5));
    }

    #[test]
    fn zero_direction_never_hits() {
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let obb = Obb::new([0.0, 0.0, 0.0], identity, [1.0, 1.0, 1.0], 0);
        let ray = Ray::infinite([5.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        assert!(obb.intersect(&ray).is_none());
    }

    #[test]
    fn reported_hit_lies_on_a_face_with_a_unit_normal() {
        // Independent check: transform the reported hit point into the box frame
        // and verify it sits on a face (one local coordinate equals ±half, the
        // others within the slab) and the normal is unit. The oracle is this
        // local-frame slab geometry, never `intersect`.
        let mut rng = Rng::new(0x0B_B123);
        let mut hits = 0u32;
        for _ in 0..6_000 {
            let center = [rng.range(-4.0, 4.0), rng.range(-4.0, 4.0), rng.range(-4.0, 4.0)];
            let half = [rng.range(0.3, 2.0), rng.range(0.3, 2.0), rng.range(0.3, 2.0)];
            // Random orthonormal frame: rotate the identity about +z then about
            // the resulting +x by rational half-angle maps (no transcendentals).
            let (cz, sz) = unit_pair(rng.range(-2.0, 2.0));
            let (cx, sx) = unit_pair(rng.range(-2.0, 2.0));
            let rz = frame_z(cz, sz);
            // Rotate rows of rz about world +x by (cx, sx).
            let rot_x = |v: [f32; 3]| [v[0], cx * v[1] - sx * v[2], sx * v[1] + cx * v[2]];
            let axes = [rot_x(rz[0]), rot_x(rz[1]), rot_x(rz[2])];
            let obb = Obb::new(center, axes, half, 0);

            let origin = [
                center[0] + rng.range(-12.0, 12.0),
                center[1] + rng.range(-12.0, 12.0),
                center[2] + rng.range(-12.0, 12.0),
            ];
            // Aim at a jittered point inside the box, expressed in world space.
            let lx = rng.range(-1.0, 1.0) * half[0];
            let ly = rng.range(-1.0, 1.0) * half[1];
            let lz = rng.range(-1.0, 1.0) * half[2];
            let target = [
                center[0] + lx * axes[0][0] + ly * axes[1][0] + lz * axes[2][0],
                center[1] + lx * axes[0][1] + ly * axes[1][1] + lz * axes[2][1],
                center[2] + lx * axes[0][2] + ly * axes[1][2] + lz * axes[2][2],
            ];
            let dir = [target[0] - origin[0], target[1] - origin[1], target[2] - origin[2]];
            if dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2] < 1e-6 {
                continue;
            }
            let ray = Ray::infinite(origin, dir);
            let Some(hit) = obb.intersect(&ray) else {
                continue;
            };
            hits += 1;

            // Local coordinates of the hit point.
            let p = ray.at(hit.t);
            let rel = [p[0] - center[0], p[1] - center[1], p[2] - center[2]];
            let local = [dot(rel, axes[0]), dot(rel, axes[1]), dot(rel, axes[2])];
            // Exactly one local coordinate should be on a face (|local| ≈ half);
            // the other two must lie within their slab (plus a small margin).
            let mut on_face = 0;
            for k in 0..3 {
                let m = half[k];
                if (local[k].abs() - m).abs() < 2e-3 * (1.0 + m) {
                    on_face += 1;
                } else {
                    assert!(local[k].abs() <= m + 2e-3 * (1.0 + m), "point outside slab {k}");
                }
            }
            assert!(on_face >= 1, "hit point not on any face: {local:?}");

            let nlen = (hit.normal[0] * hit.normal[0]
                + hit.normal[1] * hit.normal[1]
                + hit.normal[2] * hit.normal[2])
                .sqrt();
            assert!(approx(nlen, 1.0, 2e-3), "normal not unit: {nlen}");

            let facing = hit.normal[0] * dir[0] + hit.normal[1] * dir[1] + hit.normal[2] * dir[2];
            assert!(facing <= 1e-3, "normal not oriented against the ray: {facing}");
        }
        assert!(hits > 2_000, "too few surface hits accumulated: {hits}");
    }

    #[test]
    fn empty_bvh_never_hits() {
        let bvh = ObbBvh::build(&[]);
        assert!(bvh.is_empty());
        assert_eq!(bvh.node_count(), 0);
        assert_eq!(bvh.primitive_count(), 0);
        assert_eq!(bvh.bounds(), Aabb::empty());
        let ray = Ray::infinite([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
        assert!(bvh.closest_hit(&ray).is_none());
        assert!(!bvh.any_hit(&ray));
    }

    #[test]
    fn bvh_closest_hit_matches_brute_force() {
        let mut rng = Rng::new(0xB0_FACE);
        let boxes: Vec<Obb> = (0..48)
            .map(|i| {
                let center = [rng.range(-6.0, 6.0), rng.range(-6.0, 6.0), rng.range(-6.0, 6.0)];
                let half = [rng.range(0.3, 1.2), rng.range(0.3, 1.2), rng.range(0.3, 1.2)];
                let (cz, sz) = unit_pair(rng.range(-2.0, 2.0));
                let axes = frame_z(cz, sz);
                Obb::new(center, axes, half, i)
            })
            .collect();
        let bvh = ObbBvh::build(&boxes);
        assert_eq!(bvh.primitive_count(), boxes.len());

        for _ in 0..3_000 {
            let origin = [rng.range(-10.0, 10.0), rng.range(-10.0, 10.0), rng.range(-10.0, 10.0)];
            let dir = [rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(-1.0, 1.0)];
            if dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2] < 1e-6 {
                continue;
            }
            let ray = Ray::infinite(origin, dir);
            let expected = brute_closest(&boxes, &ray);
            let actual = bvh.closest_hit(&ray);
            match (expected, actual) {
                (None, None) => {}
                (Some(e), Some(a)) => {
                    assert_eq!(e.primitive, a.primitive);
                    assert_eq!(e.t.to_bits(), a.t.to_bits());
                    assert_eq!(e.front_face, a.front_face);
                }
                (e, a) => panic!("hit disagreement: {e:?} vs {a:?}"),
            }
            assert_eq!(bvh.any_hit(&ray), brute_closest(&boxes, &ray).is_some());
        }
    }
}
