//! Analytic oriented disk primitive and its single-level `BVH`.
//!
//! Like [`super::sphere`], [`super::cylinder`], and [`super::aabb_primitive`],
//! this is a procedural primitive for the `DXR`/Vulkan `AABB` path: the `BLAS`
//! stores one axis-aligned box per disk and an intersection shader refines the
//! hit. A flat disk is the natural proxy for a round area light, a spot-light
//! emitter face, a coin/washer, or the end cap of a tube, so a path tracer wants
//! a closed-form test rather than a tessellated fan.
//!
//! A [`Disk`] is the set of points within `radius` of `center` lying in the
//! plane through `center` with unit `normal`. The intersection solves the single
//! ray/plane equation `t = (center − origin) · n / (d · n)` and then rejects the
//! hit when it lands outside the radius. Every step is add/sub/mul/div/`sqrt`
//! and comparisons, so it is bit-reproducible on the `GPU` and free of any
//! transcendental call.

use super::bvh::{build_linear_bvh, Aabb, BvhBuildConfig, LinearBvhNode};
use super::traversal::Ray;

/// An analytic oriented disk in world space.
///
/// The surface is the set of points within `radius` of `center` that lie in the
/// plane through `center` with unit `normal`. `primitive` is the caller's stable
/// id (mirroring [`super::bvh::Triangle`] and [`super::sphere::Sphere`]): the
/// [`DiskBvh`] builder reorders disks internally but always reports hits by this
/// id. The normal is stored normalized and the radius is stored non-negative so
/// the derived [`Aabb`] and the plane test stay well formed; a zero-length
/// caller normal is kept as-is and marks the disk degenerate (it never reports a
/// hit).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Disk {
    /// Center of the disk (a point on its plane).
    center: [f32; 3],
    /// Unit plane normal (zero only for a degenerate disk).
    normal: [f32; 3],
    /// Non-negative radius.
    radius: f32,
    /// Caller's stable primitive id, reported unchanged on every hit.
    primitive: u32,
}

impl Disk {
    /// Builds a disk at `center` with plane `normal` (normalized to unit length)
    /// and `radius` (folded to its magnitude), tagged with stable id
    /// `primitive`. A zero-length `normal` is stored unchanged and yields a
    /// degenerate disk that never reports a hit.
    #[must_use]
    pub fn new(center: [f32; 3], normal: [f32; 3], radius: f32, primitive: u32) -> Self {
        let len2 = dot(normal, normal);
        let normal = if len2 > 0.0 {
            let inv = 1.0 / len2.sqrt();
            [normal[0] * inv, normal[1] * inv, normal[2] * inv]
        } else {
            normal
        };
        Self {
            center,
            normal,
            radius: radius.abs(),
            primitive,
        }
    }

    /// Center of the disk (a point on its plane).
    #[must_use]
    pub fn center(&self) -> [f32; 3] {
        self.center
    }

    /// Unit plane normal (zero only for a degenerate disk).
    #[must_use]
    pub fn normal(&self) -> [f32; 3] {
        self.normal
    }

    /// Non-negative radius.
    #[must_use]
    pub fn radius(&self) -> f32 {
        self.radius
    }

    /// Caller's stable primitive id.
    #[must_use]
    pub fn primitive(&self) -> u32 {
        self.primitive
    }

    /// Tight axis-aligned bounds of the disk.
    ///
    /// This is the procedural-primitive `AABB` the hardware `BLAS` stores. A
    /// disk with unit normal `n` projects onto world axis `i` with half extent
    /// `radius · √(1 − nᵢ²)` — the exact projected radius of the circle — so the
    /// box hugs the disk rather than using the loose `± radius` cube. A
    /// degenerate (zero) normal falls back to that loose cube.
    #[must_use]
    pub fn aabb(&self) -> Aabb {
        let n = self.normal;
        let mut min = [0.0f32; 3];
        let mut max = [0.0f32; 3];
        for (axis, slot) in min.iter_mut().zip(max.iter_mut()).enumerate() {
            // Clamp guards tiny negative round-off before the sqrt.
            let frac = 1.0 - n[axis] * n[axis];
            let e = self.radius * frac.max(0.0).sqrt();
            *slot.0 = self.center[axis] - e;
            *slot.1 = self.center[axis] + e;
        }
        Aabb::new(min, max)
    }

    /// Nearest ray/disk intersection inside `ray`'s `[t_min, t_max]` interval,
    /// or `None` when the ray misses.
    ///
    /// [`DiskHit::normal`] is the unit plane normal oriented *against* the
    /// incident ray, and [`DiskHit::front_face`] is `true` when the ray struck
    /// the side the stored normal faces (a ray arriving from behind reports
    /// `front_face == false` with a flipped normal). A zero-radius disk, a
    /// degenerate (zero) normal, a ray parallel to the plane, or a zero-length
    /// ray direction never reports a hit.
    #[must_use]
    pub fn intersect(&self, ray: &Ray) -> Option<DiskHit> {
        if self.radius <= 0.0 {
            return None;
        }
        let nn = dot(self.normal, self.normal);
        if nn <= 0.0 {
            return None;
        }
        let direction = ray.direction();
        // `denom == 0` means the ray runs parallel to the plane (or has zero
        // length) and can never cross it.
        let denom = dot(direction, self.normal);
        if denom == 0.0 {
            return None;
        }
        let oc = sub(self.center, ray.origin());
        let t = dot(oc, self.normal) / denom;
        if t < ray.t_min() || t > ray.t_max() {
            return None;
        }
        // Reject the hit when it lands outside the disk radius.
        let offset = sub(ray.at(t), self.center);
        if dot(offset, offset) > self.radius * self.radius {
            return None;
        }
        let front_face = denom < 0.0;
        let normal = if front_face {
            self.normal
        } else {
            [-self.normal[0], -self.normal[1], -self.normal[2]]
        };
        Some(DiskHit {
            t,
            primitive: self.primitive,
            normal,
            front_face,
        })
    }
}

/// A ray/disk intersection result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiskHit {
    /// Ray parameter at the intersection (distance in `direction` lengths).
    pub t: f32,
    /// Stable id of the disk that was hit.
    pub primitive: u32,
    /// Unit surface normal oriented against the incident ray.
    pub normal: [f32; 3],
    /// `true` when the ray struck the side the stored normal faces; `false` when
    /// it arrived from behind, in which case `normal` is flipped to oppose it.
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

/// A single-level `BVH` over analytic [`Disk`] primitives.
///
/// Empty input yields an empty hierarchy ([`DiskBvh::is_empty`]); traversal of
/// an empty hierarchy simply never reports a hit. The layout and ordered slab
/// walk mirror the triangle [`super::bvh::Bvh`], [`super::sphere::SphereBvh`],
/// and [`super::cylinder::CylinderBvh`] so every primitive kind shares one
/// acceleration-structure contract.
#[derive(Clone, Debug, PartialEq)]
pub struct DiskBvh {
    /// Flattened `BVH` nodes; the root (when present) is index `0`.
    nodes: Vec<LinearBvhNode>,
    /// Disks reordered so each leaf owns a contiguous slice.
    disks: Vec<Disk>,
}

impl DiskBvh {
    /// Builds a `BVH` over `disks` with [`BvhBuildConfig::default`].
    #[must_use]
    pub fn build(disks: &[Disk]) -> Self {
        Self::build_with(disks, BvhBuildConfig::default())
    }

    /// Builds a `BVH` over `disks` with the given binned-`SAH` `config`.
    ///
    /// The builder runs over each disk's [`Disk::aabb`] and reorders the disks
    /// by the returned primitive order so every leaf's
    /// `[first_primitive, first_primitive + primitive_count)` slice indexes
    /// directly into [`DiskBvh::disks`].
    #[must_use]
    pub fn build_with(disks: &[Disk], config: BvhBuildConfig) -> Self {
        let bounds: Vec<Aabb> = disks.iter().map(Disk::aabb).collect();
        let (nodes, order) = build_linear_bvh(&bounds, config);
        let disks = order.iter().map(|&i| disks[i as usize]).collect();
        Self { nodes, disks }
    }

    /// Number of flattened `BVH` nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of disks in the hierarchy.
    #[must_use]
    pub fn primitive_count(&self) -> usize {
        self.disks.len()
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

    /// Disks in leaf-contiguous order.
    #[must_use]
    pub fn disks(&self) -> &[Disk] {
        &self.disks
    }

    /// Nearest intersection along `ray`, or `None` if the ray hits nothing.
    ///
    /// Walks the flattened nodes with an explicit stack, visiting the child on
    /// the near side of the split axis first so the running `t_max` shrinks as
    /// fast as possible and far subtrees are culled by the slab test.
    #[must_use]
    pub fn closest_hit(&self, ray: &Ray) -> Option<DiskHit> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut ray = *ray;
        let mut best: Option<DiskHit> = None;

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
                    for disk in &self.disks[start..end] {
                        if let Some(hit) = disk.intersect(&ray) {
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

    /// True when *any* disk intersects `ray` inside its interval.
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
                    for disk in &self.disks[start..end] {
                        if disk.intersect(ray).is_some() {
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

    /// Unit-radius disk at the origin whose normal faces `+z`.
    fn unit_disk(primitive: u32) -> Disk {
        Disk::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1.0, primitive)
    }

    #[test]
    fn hit_from_the_front() {
        let disk = unit_disk(3);
        // Ray from +z toward -z hits the disk plane at the center.
        let ray = Ray::infinite([0.0, 0.0, 5.0], [0.0, 0.0, -1.0]);
        let hit = disk.intersect(&ray).expect("front hit");
        assert_eq!(hit.primitive, 3);
        assert!(approx(hit.t, 5.0, 1e-4), "t = {}", hit.t);
        assert!(hit.front_face);
        assert!(approx(hit.normal[2], 1.0, 1e-4), "normal = {:?}", hit.normal);
    }

    #[test]
    fn hit_from_behind_reports_back_face() {
        let disk = unit_disk(4);
        // Ray arriving from -z toward +z strikes the back of the disk.
        let ray = Ray::infinite([0.0, 0.0, -5.0], [0.0, 0.0, 1.0]);
        let hit = disk.intersect(&ray).expect("back hit");
        assert!(approx(hit.t, 5.0, 1e-4), "t = {}", hit.t);
        assert!(!hit.front_face);
        // Normal is flipped to oppose the incident ray.
        assert!(approx(hit.normal[2], -1.0, 1e-4), "normal = {:?}", hit.normal);
    }

    #[test]
    fn ray_outside_radius_misses() {
        let disk = unit_disk(0);
        // Crosses the plane at (2, 0, 0), outside the unit radius.
        let ray = Ray::infinite([2.0, 0.0, 5.0], [0.0, 0.0, -1.0]);
        assert!(disk.intersect(&ray).is_none());
    }

    #[test]
    fn ray_parallel_to_plane_misses() {
        let disk = unit_disk(0);
        // Direction lies in the plane, so it never crosses it.
        let ray = Ray::infinite([0.0, 0.0, 0.5], [1.0, 0.0, 0.0]);
        assert!(disk.intersect(&ray).is_none());
    }

    #[test]
    fn behind_origin_is_missed() {
        let disk = unit_disk(0);
        // The plane is behind the origin along the ray direction.
        let ray = Ray::infinite([0.0, 0.0, 5.0], [0.0, 0.0, 1.0]);
        assert!(disk.intersect(&ray).is_none());
    }

    #[test]
    fn t_max_excludes_far_hit() {
        let disk = unit_disk(0);
        // The plane crossing is at t = 5; a shorter interval must miss.
        let ray = Ray::new([0.0, 0.0, 5.0], [0.0, 0.0, -1.0], 0.0, 4.0);
        assert!(disk.intersect(&ray).is_none());
        let ray = Ray::new([0.0, 0.0, 5.0], [0.0, 0.0, -1.0], 0.0, 6.0);
        assert!(disk.intersect(&ray).is_some());
    }

    #[test]
    fn zero_radius_never_hits() {
        let disk = Disk::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.0, 0);
        let ray = Ray::infinite([0.0, 0.0, 5.0], [0.0, 0.0, -1.0]);
        assert!(disk.intersect(&ray).is_none());
    }

    #[test]
    fn degenerate_normal_never_hits() {
        let disk = Disk::new([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1.0, 0);
        let ray = Ray::infinite([0.0, 0.0, 5.0], [0.0, 0.0, -1.0]);
        assert!(disk.intersect(&ray).is_none());
    }

    #[test]
    fn aabb_is_flat_for_axis_aligned_disk() {
        let disk = Disk::new([1.0, 2.0, 3.0], [0.0, 0.0, 1.0], 2.0, 0);
        let aabb = disk.aabb();
        // Extent is 2 in x/y (in-plane) and 0 in z (the disk's normal axis).
        assert!(approx(aabb.min[0], -1.0, 1e-4));
        assert!(approx(aabb.max[0], 3.0, 1e-4));
        assert!(approx(aabb.min[1], 0.0, 1e-4));
        assert!(approx(aabb.max[1], 4.0, 1e-4));
        assert!(approx(aabb.min[2], 3.0, 1e-4));
        assert!(approx(aabb.max[2], 3.0, 1e-4));
    }

    fn random_disk(rng: &mut Rng, primitive: u32) -> Disk {
        let center = [
            rng.range(-5.0, 5.0),
            rng.range(-5.0, 5.0),
            rng.range(-5.0, 5.0),
        ];
        let mut normal = [
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
        ];
        // Guard against a near-zero (degenerate) random normal.
        if dot(normal, normal) < 1e-3 {
            normal = [0.0, 0.0, 1.0];
        }
        Disk::new(center, normal, rng.range(0.2, 1.2), primitive)
    }

    fn random_scene(rng: &mut Rng, count: u32) -> Vec<Disk> {
        (0..count).map(|i| random_disk(rng, i)).collect()
    }

    fn brute_closest(disks: &[Disk], ray: &Ray) -> Option<DiskHit> {
        let mut best: Option<DiskHit> = None;
        let mut ray = *ray;
        for disk in disks {
            if let Some(hit) = disk.intersect(&ray) {
                ray = Ray::new(ray.origin(), ray.direction(), ray.t_min(), hit.t);
                best = Some(hit);
            }
        }
        best
    }

    #[test]
    fn empty_bvh_never_hits() {
        let bvh = DiskBvh::build(&[]);
        assert!(bvh.is_empty());
        let ray = Ray::infinite([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
        assert!(bvh.closest_hit(&ray).is_none());
        assert!(!bvh.any_hit(&ray));
    }

    #[test]
    fn bvh_closest_hit_matches_brute_force_bit_for_bit() {
        let mut rng = Rng::new(0x0d15_c000_9abc_1234u64);
        let disks = random_scene(&mut rng, 64);
        let bvh = DiskBvh::build(&disks);

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

            let expected = brute_closest(&disks, &ray);
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
        let disks = random_scene(&mut rng, 48);
        let bvh = DiskBvh::build(&disks);

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
            assert_eq!(bvh.any_hit(&ray), brute_closest(&disks, &ray).is_some());
        }
    }
}
