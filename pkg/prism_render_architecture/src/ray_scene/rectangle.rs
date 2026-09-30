//! Analytic oriented rectangle (parallelogram) primitive and its single-level
//! `BVH`.
//!
//! Like [`super::sphere`], [`super::cylinder`], [`super::disk`], and
//! [`super::aabb_primitive`], this is a procedural primitive for the
//! `DXR`/Vulkan `AABB` path: the `BLAS` stores one axis-aligned box per
//! rectangle and an intersection shader refines the hit. A flat rectangle is
//! the natural proxy for a rectangular area light (the `UE`-style rect light),
//! a quad emitter, a window, or a billboard, so a path tracer wants a
//! closed-form test rather than a tessellated pair of triangles.
//!
//! A [`Rectangle`] is a center-based parallelogram: the set of points
//! `center + a·axis_u + b·axis_v` with `a, b ∈ [−1, 1]`, where `axis_u` and
//! `axis_v` are *half-edge* vectors (each spans half the corresponding side).
//! The plane normal is derived as `axis_u × axis_v`, so a caller never stores a
//! redundant normal. The intersection solves the single ray/plane equation
//! `t = (center − origin) · n / (d · n)` and then solves the in-plane `2×2`
//! parallelogram-coordinate system to accept or reject the hit. Every step is
//! add/sub/mul/div/`sqrt` and comparisons, so it is bit-reproducible on the
//! `GPU` and free of any transcendental call.

use super::bvh::{build_linear_bvh, Aabb, BvhBuildConfig, LinearBvhNode};
use super::traversal::Ray;

/// An analytic oriented rectangle (parallelogram) in world space.
///
/// The surface is the set of points `center + a·axis_u + b·axis_v` for
/// `a, b ∈ [−1, 1]`, where `axis_u` and `axis_v` are *half-edge* vectors, so
/// the full side lengths are `2·|axis_u|` and `2·|axis_v|`. The edges need not
/// be orthogonal — a general parallelogram is supported — but they must be
/// linearly independent; two parallel (or zero-length) edges span no area and
/// mark the rectangle degenerate (it never reports a hit). `primitive` is the
/// caller's stable id (mirroring [`super::bvh::Triangle`] and
/// [`super::disk::Disk`]): the [`RectangleBvh`] builder reorders rectangles
/// internally but always reports hits by this id. All fields are stored
/// verbatim, so a packed rectangle round-trips through the `GPU` layout
/// bit-for-bit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rectangle {
    /// Center of the rectangle (a point on its plane).
    center: [f32; 3],
    /// Half-edge vector along the first side (length is half that side).
    axis_u: [f32; 3],
    /// Half-edge vector along the second side (length is half that side).
    axis_v: [f32; 3],
    /// Caller's stable primitive id, reported unchanged on every hit.
    primitive: u32,
}

impl Rectangle {
    /// Builds a rectangle centered at `center` with the two *half-edge* vectors
    /// `axis_u` and `axis_v`, tagged with stable id `primitive`.
    ///
    /// The edge vectors are stored verbatim (no normalization or scaling), so
    /// `2·|axis_u|` and `2·|axis_v|` are the full side lengths. Two parallel or
    /// zero-length edges span no area and yield a degenerate rectangle that
    /// never reports a hit; the derived [`Rectangle::aabb`] stays well formed in
    /// that case too.
    #[must_use]
    pub fn new(center: [f32; 3], axis_u: [f32; 3], axis_v: [f32; 3], primitive: u32) -> Self {
        Self {
            center,
            axis_u,
            axis_v,
            primitive,
        }
    }

    /// Center of the rectangle (a point on its plane).
    #[must_use]
    pub fn center(&self) -> [f32; 3] {
        self.center
    }

    /// Half-edge vector along the first side (length is half that side).
    #[must_use]
    pub fn axis_u(&self) -> [f32; 3] {
        self.axis_u
    }

    /// Half-edge vector along the second side (length is half that side).
    #[must_use]
    pub fn axis_v(&self) -> [f32; 3] {
        self.axis_v
    }

    /// Caller's stable primitive id.
    #[must_use]
    pub fn primitive(&self) -> u32 {
        self.primitive
    }

    /// Unit plane normal `normalize(axis_u × axis_v)`, or the zero vector for a
    /// degenerate rectangle (parallel or zero-length edges).
    ///
    /// This is computed on demand rather than stored, so it always agrees with
    /// the current edge vectors and never needs a re-normalization round-trip.
    #[must_use]
    pub fn normal(&self) -> [f32; 3] {
        let n = cross(self.axis_u, self.axis_v);
        let nn = dot(n, n);
        if nn > 0.0 {
            let inv = 1.0 / nn.sqrt();
            [n[0] * inv, n[1] * inv, n[2] * inv]
        } else {
            [0.0, 0.0, 0.0]
        }
    }

    /// Tight axis-aligned bounds of the rectangle.
    ///
    /// This is the procedural-primitive `AABB` the hardware `BLAS` stores. The
    /// four corners are `center ± axis_u ± axis_v`, so per world axis `i` the
    /// box reaches `center[i] ± (|axis_u[i]| + |axis_v[i]|)` — the exact
    /// projected half extent — hugging the rectangle rather than using a loose
    /// cube. A degenerate rectangle simply yields a (possibly flat) box around
    /// its edges.
    #[must_use]
    pub fn aabb(&self) -> Aabb {
        let mut min = [0.0f32; 3];
        let mut max = [0.0f32; 3];
        for (axis, slot) in min.iter_mut().zip(max.iter_mut()).enumerate() {
            let e = self.axis_u[axis].abs() + self.axis_v[axis].abs();
            *slot.0 = self.center[axis] - e;
            *slot.1 = self.center[axis] + e;
        }
        Aabb::new(min, max)
    }

    /// Nearest ray/rectangle intersection inside `ray`'s `[t_min, t_max]`
    /// interval, or `None` when the ray misses.
    ///
    /// [`RectangleHit::normal`] is the unit plane normal oriented *against* the
    /// incident ray, and [`RectangleHit::front_face`] is `true` when the ray
    /// struck the side the derived normal faces (a ray arriving from behind
    /// reports `front_face == false` with a flipped normal). A degenerate
    /// rectangle (parallel or zero-length edges), a ray parallel to the plane,
    /// or a zero-length ray direction never reports a hit.
    ///
    /// The in-plane containment test solves the `2×2` system `p = a·u + b·v`
    /// (with `p` the hit point relative to the center) using the reciprocal
    /// basis, so it is exact even when the edges are not orthogonal. The system
    /// determinant is `|u|²|v|² − (u·v)²`, which by Lagrange's identity equals
    /// `|u × v|²` — the very quantity that certifies the rectangle is
    /// non-degenerate — so one squared cross product guards both the plane test
    /// and the containment solve.
    #[must_use]
    pub fn intersect(&self, ray: &Ray) -> Option<RectangleHit> {
        let n = cross(self.axis_u, self.axis_v);
        let nn = dot(n, n);
        // A zero-area (parallel or zero-length edges) rectangle has no plane.
        if nn <= 0.0 {
            return None;
        }
        let direction = ray.direction();
        // `denom == 0` means the ray runs parallel to the plane (or has zero
        // length) and can never cross it.
        let denom = dot(direction, n);
        if denom == 0.0 {
            return None;
        }
        let oc = sub(self.center, ray.origin());
        let t = dot(oc, n) / denom;
        if t < ray.t_min() || t > ray.t_max() {
            return None;
        }
        // Solve `p = a·u + b·v` in the plane via the reciprocal basis. The
        // determinant `uu·vv − uv²` equals `nn` (Lagrange identity) and is
        // strictly positive here, so the divide is well defined.
        let p = sub(ray.at(t), self.center);
        let uu = dot(self.axis_u, self.axis_u);
        let vv = dot(self.axis_v, self.axis_v);
        let uv = dot(self.axis_u, self.axis_v);
        let pu = dot(p, self.axis_u);
        let pv = dot(p, self.axis_v);
        let a = (vv * pu - uv * pv) / nn;
        let b = (uu * pv - uv * pu) / nn;
        if !(-1.0..=1.0).contains(&a) || !(-1.0..=1.0).contains(&b) {
            return None;
        }
        let inv = 1.0 / nn.sqrt();
        let unit = [n[0] * inv, n[1] * inv, n[2] * inv];
        let front_face = denom < 0.0;
        let normal = if front_face {
            unit
        } else {
            [-unit[0], -unit[1], -unit[2]]
        };
        Some(RectangleHit {
            t,
            primitive: self.primitive,
            normal,
            front_face,
        })
    }
}

/// A ray/rectangle intersection result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RectangleHit {
    /// Ray parameter at the intersection (distance in `direction` lengths).
    pub t: f32,
    /// Stable id of the rectangle that was hit.
    pub primitive: u32,
    /// Unit surface normal oriented against the incident ray.
    pub normal: [f32; 3],
    /// `true` when the ray struck the side the derived normal faces; `false`
    /// when it arrived from behind, in which case `normal` is flipped to oppose
    /// it.
    pub front_face: bool,
}

/// Subtracts `b` from `a` componentwise.
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Euclidean dot product of two vectors.
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Right-handed cross product `a × b`.
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// A single-level `BVH` over analytic [`Rectangle`] primitives.
///
/// Empty input yields an empty hierarchy ([`RectangleBvh::is_empty`]);
/// traversal of an empty hierarchy simply never reports a hit. The layout and
/// ordered slab walk mirror the triangle [`super::bvh::Bvh`],
/// [`super::sphere::SphereBvh`], and [`super::disk::DiskBvh`] so every
/// primitive kind shares one acceleration-structure contract.
#[derive(Clone, Debug, PartialEq)]
pub struct RectangleBvh {
    /// Flattened `BVH` nodes; the root (when present) is index `0`.
    nodes: Vec<LinearBvhNode>,
    /// Rectangles reordered so each leaf owns a contiguous slice.
    rectangles: Vec<Rectangle>,
}

impl RectangleBvh {
    /// Builds a `BVH` over `rectangles` with [`BvhBuildConfig::default`].
    #[must_use]
    pub fn build(rectangles: &[Rectangle]) -> Self {
        Self::build_with(rectangles, BvhBuildConfig::default())
    }

    /// Builds a `BVH` over `rectangles` with the given binned-`SAH` `config`.
    ///
    /// The builder runs over each rectangle's [`Rectangle::aabb`] and reorders
    /// the rectangles by the returned primitive order so every leaf's
    /// `[first_primitive, first_primitive + primitive_count)` slice indexes
    /// directly into [`RectangleBvh::rectangles`].
    #[must_use]
    pub fn build_with(rectangles: &[Rectangle], config: BvhBuildConfig) -> Self {
        let bounds: Vec<Aabb> = rectangles.iter().map(Rectangle::aabb).collect();
        let (nodes, order) = build_linear_bvh(&bounds, config);
        let rectangles = order.iter().map(|&i| rectangles[i as usize]).collect();
        Self { nodes, rectangles }
    }

    /// Number of flattened `BVH` nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of rectangles in the hierarchy.
    #[must_use]
    pub fn primitive_count(&self) -> usize {
        self.rectangles.len()
    }

    /// True when the hierarchy holds no nodes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Root bounds, or an empty box when the hierarchy is empty.
    #[must_use]
    pub fn bounds(&self) -> Aabb {
        self.nodes.first().map_or_else(Aabb::empty, |n| n.bounds)
    }

    /// Flattened `BVH` nodes (root at index `0` when present).
    #[must_use]
    pub fn nodes(&self) -> &[LinearBvhNode] {
        &self.nodes
    }

    /// Rectangles in leaf-contiguous order.
    #[must_use]
    pub fn rectangles(&self) -> &[Rectangle] {
        &self.rectangles
    }

    /// Nearest intersection along `ray`, or `None` if the ray hits nothing.
    ///
    /// Walks the flattened nodes with an explicit stack, visiting the child on
    /// the near side of the split axis first so the running `t_max` shrinks as
    /// fast as possible and far subtrees are culled by the slab test.
    #[must_use]
    pub fn closest_hit(&self, ray: &Ray) -> Option<RectangleHit> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut ray = *ray;
        let mut best: Option<RectangleHit> = None;

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
                    for rectangle in &self.rectangles[start..end] {
                        if let Some(hit) = rectangle.intersect(&ray) {
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

    /// True when *any* rectangle intersects `ray` inside its interval.
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
                    for rectangle in &self.rectangles[start..end] {
                        if rectangle.intersect(ray).is_some() {
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
        (a - b).abs() <= eps
    }

    /// Axis-aligned unit square at the origin whose normal faces `+z`
    /// (half-edges of length 1 along `+x` and `+y`, so a 2×2 square).
    fn unit_square(primitive: u32) -> Rectangle {
        Rectangle::new(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            primitive,
        )
    }

    #[test]
    fn hit_from_the_front() {
        let rect = unit_square(3);
        let ray = Ray::infinite([0.0, 0.0, 5.0], [0.0, 0.0, -1.0]);
        let hit = rect.intersect(&ray).expect("front hit");
        assert_eq!(hit.primitive, 3);
        assert!(approx(hit.t, 5.0, 1e-4), "t = {}", hit.t);
        assert!(hit.front_face);
        assert!(approx(hit.normal[2], 1.0, 1e-4), "normal = {:?}", hit.normal);
    }

    #[test]
    fn hit_from_behind_reports_back_face() {
        let rect = unit_square(4);
        let ray = Ray::infinite([0.0, 0.0, -5.0], [0.0, 0.0, 1.0]);
        let hit = rect.intersect(&ray).expect("back hit");
        assert!(approx(hit.t, 5.0, 1e-4), "t = {}", hit.t);
        assert!(!hit.front_face);
        assert!(approx(hit.normal[2], -1.0, 1e-4), "normal = {:?}", hit.normal);
    }

    #[test]
    fn hit_near_corner_inside() {
        let rect = unit_square(0);
        // Crosses the plane at (0.9, 0.9): inside the [-1, 1]² square.
        let ray = Ray::infinite([0.9, 0.9, 5.0], [0.0, 0.0, -1.0]);
        assert!(rect.intersect(&ray).is_some());
    }

    #[test]
    fn ray_outside_extent_misses() {
        let rect = unit_square(0);
        // Crosses the plane at (1.5, 0, 0): outside the half-width of 1.
        let ray = Ray::infinite([1.5, 0.0, 5.0], [0.0, 0.0, -1.0]);
        assert!(rect.intersect(&ray).is_none());
    }

    #[test]
    fn ray_parallel_to_plane_misses() {
        let rect = unit_square(0);
        let ray = Ray::infinite([0.0, 0.0, 0.5], [1.0, 0.0, 0.0]);
        assert!(rect.intersect(&ray).is_none());
    }

    #[test]
    fn behind_origin_is_missed() {
        let rect = unit_square(0);
        let ray = Ray::infinite([0.0, 0.0, 5.0], [0.0, 0.0, 1.0]);
        assert!(rect.intersect(&ray).is_none());
    }

    #[test]
    fn t_max_excludes_far_hit() {
        let rect = unit_square(0);
        let ray = Ray::new([0.0, 0.0, 5.0], [0.0, 0.0, -1.0], 0.0, 4.0);
        assert!(rect.intersect(&ray).is_none());
        let ray = Ray::new([0.0, 0.0, 5.0], [0.0, 0.0, -1.0], 0.0, 6.0);
        assert!(rect.intersect(&ray).is_some());
    }

    #[test]
    fn degenerate_edges_never_hit() {
        // Parallel edges span zero area.
        let rect = Rectangle::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], 0);
        let ray = Ray::infinite([0.0, 0.0, 5.0], [0.0, 0.0, -1.0]);
        assert!(rect.intersect(&ray).is_none());
        assert_eq!(rect.normal(), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn non_orthogonal_parallelogram_containment_is_exact() {
        // Sheared parallelogram: axis_v leans into +x, so a simple per-axis
        // projection would misclassify points; the reciprocal-basis solve must
        // still accept only the true parallelogram interior.
        let rect = Rectangle::new(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            7,
        );
        // Corner point center + axis_u + axis_v = (2, 1, 0): a == b == 1 exactly.
        let ray = Ray::infinite([2.0, 1.0, 5.0], [0.0, 0.0, -1.0]);
        assert!(rect.intersect(&ray).is_some());
        // Point (2, 0, 0) lies outside the sheared parallelogram (b would be 0
        // but a would be 2), even though it is within the loose x extent.
        let ray = Ray::infinite([2.0, 0.0, 5.0], [0.0, 0.0, -1.0]);
        assert!(rect.intersect(&ray).is_none());
    }

    #[test]
    fn aabb_is_flat_for_axis_aligned_rectangle() {
        let rect = Rectangle::new([1.0, 2.0, 3.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0);
        let aabb = rect.aabb();
        assert!(approx(aabb.min[0], -1.0, 1e-4));
        assert!(approx(aabb.max[0], 3.0, 1e-4));
        assert!(approx(aabb.min[1], 1.0, 1e-4));
        assert!(approx(aabb.max[1], 3.0, 1e-4));
        assert!(approx(aabb.min[2], 3.0, 1e-4));
        assert!(approx(aabb.max[2], 3.0, 1e-4));
    }

    fn random_rectangle(rng: &mut Rng, primitive: u32) -> Rectangle {
        let center = [
            rng.range(-5.0, 5.0),
            rng.range(-5.0, 5.0),
            rng.range(-5.0, 5.0),
        ];
        // Build two half-edges that are unlikely to be parallel.
        let axis_u = [
            rng.range(0.3, 1.5),
            rng.range(-1.5, 1.5),
            rng.range(-1.5, 1.5),
        ];
        let axis_v = [
            rng.range(-1.5, 1.5),
            rng.range(0.3, 1.5),
            rng.range(-1.5, 1.5),
        ];
        Rectangle::new(center, axis_u, axis_v, primitive)
    }

    fn random_scene(rng: &mut Rng, count: u32) -> Vec<Rectangle> {
        (0..count).map(|i| random_rectangle(rng, i)).collect()
    }

    fn brute_closest(rectangles: &[Rectangle], ray: &Ray) -> Option<RectangleHit> {
        let mut best: Option<RectangleHit> = None;
        let mut ray = *ray;
        for rectangle in rectangles {
            if let Some(hit) = rectangle.intersect(&ray) {
                ray = Ray::new(ray.origin(), ray.direction(), ray.t_min(), hit.t);
                best = Some(hit);
            }
        }
        best
    }

    #[test]
    fn empty_bvh_never_hits() {
        let bvh = RectangleBvh::build(&[]);
        assert!(bvh.is_empty());
        let ray = Ray::infinite([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
        assert!(bvh.closest_hit(&ray).is_none());
        assert!(!bvh.any_hit(&ray));
    }

    #[test]
    fn bvh_closest_hit_matches_brute_force_bit_for_bit() {
        let mut rng = Rng::new(0x0d15_c000_9abc_1234u64);
        let rectangles = random_scene(&mut rng, 64);
        let bvh = RectangleBvh::build(&rectangles);

        for _ in 0..3_000 {
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

            let expected = brute_closest(&rectangles, &ray);
            let actual = bvh.closest_hit(&ray);
            match (expected, actual) {
                (None, None) => {}
                (Some(e), Some(a)) => {
                    assert_eq!(e.primitive, a.primitive);
                    assert_eq!(e.t.to_bits(), a.t.to_bits(), "t bits differ");
                    assert_eq!(e.front_face, a.front_face);
                    for k in 0..3 {
                        assert_eq!(
                            e.normal[k].to_bits(),
                            a.normal[k].to_bits(),
                            "normal[{k}] bits differ"
                        );
                    }
                }
                (e, a) => panic!("hit disagreement: {e:?} vs {a:?}"),
            }
        }
    }

    #[test]
    fn bvh_any_hit_matches_brute_force() {
        let mut rng = Rng::new(0x0d15_c000_5678_9abcu64);
        let rectangles = random_scene(&mut rng, 48);
        let bvh = RectangleBvh::build(&rectangles);

        for _ in 0..3_000 {
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
            assert_eq!(bvh.any_hit(&ray), brute_closest(&rectangles, &ray).is_some());
        }
    }
}
