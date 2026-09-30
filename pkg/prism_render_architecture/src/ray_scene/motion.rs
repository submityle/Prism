//! Two-level *matrix motion blur* acceleration.
//!
//! A [`MotionTlas`] is the temporal counterpart of [`super::tlas::Tlas`]: every
//! instance carries two key object→world transforms, a start pose `from` and an
//! end pose `to`, and each ray is traced at a normalized shutter `time` in
//! `[0, 1]`. This mirrors Direct3D 12 *matrix motion* instances
//! (`D3D12_RAYTRACING_INSTANCE_DESC` motion variant), where the runtime linearly
//! interpolates the 3×4 instance matrix per ray and re-inverts it to trace in
//! object space. It is the primitive behind motion-blurred reflections, shadows
//! and global illumination: a stochastic renderer jitters each sample's `time`
//! across the shutter interval so moving geometry smears correctly in
//! secondary rays, not just in the raster G-buffer.
//!
//! # Why component-wise matrix interpolation
//! The interpolated pose at `time` is the per-component linear blend of the two
//! key matrices, `M(t) = (1 - t)·from + t·to`, applied to the three linear
//! columns and the translation. Because a transformed corner
//! `p(t) = M(t)·corner` is then linear in `t`, every corner of an instance's
//! `BLAS` box travels a straight world-space segment, so the union of the two
//! endpoint boxes is a *correct conservative* swept bound — no mid-shutter
//! sample can escape it. The top-level `BVH` is built over those swept bounds
//! once; only the per-ray pose blend and its inverse are recomputed at trace
//! time. Interpolating the matrix (rather than decomposing to
//! translation/rotation/scale and slerping) keeps this module `libm`-free and
//! bit-for-bit reproducible on the `GPU`, matching the DXR matrix-motion
//! contract (which also blends matrix components, not decomposed rotations).
//!
//! # Relationship to the static `TLAS`
//! With `from == to` a [`MotionInstance`] is static and a [`MotionTlas`]
//! reproduces [`super::tlas::Tlas`] at every `time`; at `time == 0` it matches a
//! static `TLAS` built from the `from` poses and at `time == 1` from the `to`
//! poses. The traversal reuses the same explicit-stack top-level walk, DXR-style
//! 8-bit [`instance inclusion masks`](MotionInstance::mask), object-space ray
//! transform, and cross-instance nearest-hit pruning, and reports the shared
//! [`TlasHit`].

use super::bvh::{build_linear_bvh, Aabb, Bvh, BvhBuildConfig, LinearBvhNode};
use super::tlas::{Affine3, TlasHit};
use super::traversal::Ray;

/// One motion-blurred placement of a `BLAS`: two key object→world poses plus a
/// stable id and a DXR-style visibility mask.
///
/// The start pose `from` and end pose `to` are blended per ray by
/// [`MotionInstance::pose_at`]; a static instance simply uses `from == to`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotionInstance {
    from: Affine3,
    to: Affine3,
    blas: usize,
    instance_id: u32,
    mask: u8,
}

impl MotionInstance {
    /// The all-ones (`0xFF`) visibility mask assigned by [`MotionInstance::new`].
    ///
    /// Matches [`super::tlas::Instance::MASK_ALL`]: an instance built with this
    /// mask is visible to every ray regardless of the ray's inclusion mask.
    pub const MASK_ALL: u8 = 0xFF;

    /// Builds a motion instance interpolating from pose `from` to pose `to`.
    ///
    /// Both key poses must be invertible (non-degenerate scale), since each is a
    /// valid object→world placement on its own; a `None` return matches
    /// [`super::tlas::Instance::new`]. The instance defaults to
    /// [`MotionInstance::MASK_ALL`]. A static instance uses `from == to`.
    #[must_use]
    pub fn new(from: Affine3, to: Affine3, blas: usize, instance_id: u32) -> Option<Self> {
        Self::with_mask(from, to, blas, instance_id, Self::MASK_ALL)
    }

    /// Builds a motion instance carrying an explicit 8-bit visibility mask.
    ///
    /// The `mask` follows the same DXR `InstanceMask` semantics as
    /// [`super::tlas::Instance::with_mask`]: a ray traced with inclusion mask
    /// `ray_mask` tests this instance only when `(mask & ray_mask) != 0`, and a
    /// `mask` of `0` makes it invisible to every ray. Returns `None` when either
    /// key pose is non-invertible.
    #[must_use]
    pub fn with_mask(
        from: Affine3,
        to: Affine3,
        blas: usize,
        instance_id: u32,
        mask: u8,
    ) -> Option<Self> {
        // Both endpoints must be valid object→world placements.
        from.inverse()?;
        to.inverse()?;
        Some(Self {
            from,
            to,
            blas,
            instance_id,
            mask,
        })
    }

    /// Start (`time == 0`) object→world pose.
    #[must_use]
    pub const fn from_pose(&self) -> Affine3 {
        self.from
    }

    /// End (`time == 1`) object→world pose.
    #[must_use]
    pub const fn to_pose(&self) -> Affine3 {
        self.to
    }

    /// `BLAS` pool index this instance references.
    #[must_use]
    pub const fn blas(&self) -> usize {
        self.blas
    }

    /// Stable user-facing instance id, echoed on every [`TlasHit`].
    #[must_use]
    pub const fn instance_id(&self) -> u32 {
        self.instance_id
    }

    /// The instance's 8-bit DXR-style visibility mask.
    ///
    /// A ray with inclusion mask `ray_mask` tests this instance only when
    /// `(mask() & ray_mask) != 0`.
    #[must_use]
    pub const fn mask(&self) -> u8 {
        self.mask
    }

    /// The interpolated object→world pose at shutter `time`.
    ///
    /// `time` is clamped to `[0, 1]` (DXR clamps the shutter parameter), then the
    /// three linear columns and the translation are blended component-wise as
    /// `(1 - t)·from + t·to`. Uses only multiply/add, so it is deterministic and
    /// `libm`-free.
    #[must_use]
    pub fn pose_at(&self, time: f32) -> Affine3 {
        let t = time.clamp(0.0, 1.0);
        let s = 1.0 - t;
        let a = self.from.columns();
        let b = self.to.columns();
        let mut cols = [[0.0f32; 3]; 3];
        for c in 0..3 {
            for r in 0..3 {
                cols[c][r] = s * a[c][r] + t * b[c][r];
            }
        }
        let ta = self.from.translation();
        let tb = self.to.translation();
        let translation = [
            s * ta[0] + t * tb[0],
            s * ta[1] + t * tb[1],
            s * ta[2] + t * tb[2],
        ];
        Affine3::from_cols(cols, translation)
    }

    /// The interpolated world→object inverse at shutter `time`, or `None` when
    /// the blended pose is degenerate (non-invertible).
    ///
    /// Although both key poses are invertible by construction, an intermediate
    /// blend can in principle collapse (for example blending a pose with its
    /// mirror image passes through a singular matrix); such an instance is
    /// skipped by traversal at that `time` rather than corrupting the trace.
    #[must_use]
    pub fn world_to_object_at(&self, time: f32) -> Option<Affine3> {
        self.pose_at(time).inverse()
    }

    /// World-space bounds swept across the whole shutter for `blases`.
    ///
    /// The union of the `BLAS` root box mapped through the `from` and `to`
    /// poses. Because each transformed corner moves linearly in `time`, this
    /// union conservatively bounds every intermediate pose, so the top-level
    /// `BVH` built over it never misses a mid-shutter intersection.
    #[must_use]
    fn swept_world_bounds(&self, blases: &[Bvh]) -> Aabb {
        let local = blases[self.blas].bounds();
        if local.is_empty() {
            return Aabb::empty();
        }
        let mut out = Aabb::empty();
        for pose in [&self.from, &self.to] {
            for &cx in &[local.min[0], local.max[0]] {
                for &cy in &[local.min[1], local.max[1]] {
                    for &cz in &[local.min[2], local.max[2]] {
                        out = out.enclose(pose.transform_point([cx, cy, cz]));
                    }
                }
            }
        }
        out
    }
}

/// A built two-level acceleration structure with per-instance matrix motion.
///
/// Empty input yields an empty structure; traversal never reports a hit.
#[derive(Clone, Debug, PartialEq)]
pub struct MotionTlas {
    nodes: Vec<LinearBvhNode>,
    instances: Vec<MotionInstance>,
}

impl MotionTlas {
    /// Builds a `MotionTLAS` over `instances` with the default builder config.
    #[must_use]
    pub fn build(instances: &[MotionInstance], blases: &[Bvh]) -> Self {
        Self::build_with(instances, blases, BvhBuildConfig::default())
    }

    /// Builds a `MotionTLAS` with an explicit builder configuration.
    ///
    /// The top-level `BVH` is built once over each instance's swept world bounds
    /// (see [`MotionInstance::swept_world_bounds`]); it is valid for every
    /// shutter `time` without rebuilding.
    #[must_use]
    pub fn build_with(
        instances: &[MotionInstance],
        blases: &[Bvh],
        config: BvhBuildConfig,
    ) -> Self {
        let bounds: Vec<Aabb> = instances
            .iter()
            .map(|inst| inst.swept_world_bounds(blases))
            .collect();
        let (nodes, order) = build_linear_bvh(&bounds, config);
        let instances = order.iter().map(|&i| instances[i as usize]).collect();
        Self { nodes, instances }
    }

    /// Number of nodes in the flattened top-level hierarchy.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// True when the `MotionTLAS` holds no instances.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Read-only view of the flattened nodes (depth-first order).
    #[must_use]
    pub fn nodes(&self) -> &[LinearBvhNode] {
        &self.nodes
    }

    /// Read-only view of the reordered instance table; [`TlasHit::instance_index`]
    /// indexes into this slice.
    #[must_use]
    pub fn instances(&self) -> &[MotionInstance] {
        &self.instances
    }

    /// Swept root world-space bounds, or [`Aabb::empty`] when empty.
    #[must_use]
    pub fn bounds(&self) -> Aabb {
        self.nodes.first().map_or(Aabb::empty(), |n| n.bounds)
    }

    /// Nearest intersection along the world-space `ray` at shutter `time`.
    #[must_use]
    pub fn closest_hit(&self, ray: &Ray, time: f32, blases: &[Bvh]) -> Option<TlasHit> {
        self.closest_hit_masked(ray, time, blases, MotionInstance::MASK_ALL)
    }

    /// Nearest intersection at `time` restricted to `ray_mask`-included instances.
    ///
    /// Applies DXR-style instance inclusion: an instance participates only when
    /// `(instance.mask() & ray_mask) != 0`. [`MotionInstance::MASK_ALL`]
    /// reproduces [`MotionTlas::closest_hit`]; a `ray_mask` of `0` returns
    /// `None`.
    #[must_use]
    pub fn closest_hit_masked(
        &self,
        ray: &Ray,
        time: f32,
        blases: &[Bvh],
        ray_mask: u8,
    ) -> Option<TlasHit> {
        self.walk_closest(ray, time, blases, ray_mask, false)
    }

    /// Watertight nearest intersection at shutter `time`.
    #[must_use]
    pub fn closest_hit_watertight(
        &self,
        ray: &Ray,
        time: f32,
        blases: &[Bvh],
    ) -> Option<TlasHit> {
        self.walk_closest(ray, time, blases, MotionInstance::MASK_ALL, true)
    }

    /// Watertight nearest intersection at `time` restricted to `ray_mask`.
    #[must_use]
    pub fn closest_hit_watertight_masked(
        &self,
        ray: &Ray,
        time: f32,
        blases: &[Bvh],
        ray_mask: u8,
    ) -> Option<TlasHit> {
        self.walk_closest(ray, time, blases, ray_mask, true)
    }

    /// True when *any* instance intersects `ray` at shutter `time`.
    #[must_use]
    pub fn any_hit(&self, ray: &Ray, time: f32, blases: &[Bvh]) -> bool {
        self.any_hit_masked(ray, time, blases, MotionInstance::MASK_ALL)
    }

    /// Occlusion query at `time` restricted to `ray_mask`-included instances.
    #[must_use]
    pub fn any_hit_masked(&self, ray: &Ray, time: f32, blases: &[Bvh], ray_mask: u8) -> bool {
        self.walk_any(ray, time, blases, ray_mask, false)
    }

    /// Watertight occlusion query at shutter `time`.
    #[must_use]
    pub fn any_hit_watertight(&self, ray: &Ray, time: f32, blases: &[Bvh]) -> bool {
        self.walk_any(ray, time, blases, MotionInstance::MASK_ALL, true)
    }

    /// Watertight occlusion query at `time` restricted to `ray_mask`.
    #[must_use]
    pub fn any_hit_watertight_masked(
        &self,
        ray: &Ray,
        time: f32,
        blases: &[Bvh],
        ray_mask: u8,
    ) -> bool {
        self.walk_any(ray, time, blases, ray_mask, true)
    }

    /// Shared explicit-stack closest-hit walk for every closest-hit entry point.
    ///
    /// `watertight` selects [`Bvh::closest_hit_watertight`] over
    /// [`Bvh::closest_hit`] for the per-instance object-space query; `ray_mask`
    /// gates each instance by `(inst.mask & ray_mask) != 0`. Each leaf blends the
    /// instance pose at `time`, inverts it to object space (skipping the
    /// instance if that blend is degenerate), transforms the ray, and keeps the
    /// globally nearest hit while `best_t` shrinks across instances.
    fn walk_closest(
        &self,
        ray: &Ray,
        time: f32,
        blases: &[Bvh],
        ray_mask: u8,
        watertight: bool,
    ) -> Option<TlasHit> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut best: Option<TlasHit> = None;
        let mut best_t = ray.t_max();
        let t_min = ray.t_min();

        let mut stack = [0u32; 64];
        let mut sp = 0usize;
        let mut node_index = 0u32;
        loop {
            let node = &self.nodes[node_index as usize];
            if ray.aabb_interval(&node.bounds, t_min, best_t).is_some() {
                if node.is_leaf() {
                    let start = node.first_primitive as usize;
                    let end = start + node.primitive_count as usize;
                    for (offset, inst) in self.instances[start..end].iter().enumerate() {
                        if inst.mask & ray_mask == 0 {
                            continue;
                        }
                        let Some(world_to_object) = inst.world_to_object_at(time) else {
                            continue;
                        };
                        let obj_origin = world_to_object.transform_point(ray.origin());
                        let obj_dir = world_to_object.transform_vector(ray.direction());
                        let obj_ray = Ray::new(obj_origin, obj_dir, t_min, best_t);
                        let hit = if watertight {
                            blases[inst.blas].closest_hit_watertight(&obj_ray)
                        } else {
                            blases[inst.blas].closest_hit(&obj_ray)
                        };
                        if let Some(hit) = hit
                            && hit.t < best_t
                        {
                            best_t = hit.t;
                            best = Some(TlasHit {
                                t: hit.t,
                                u: hit.u,
                                v: hit.v,
                                primitive: hit.primitive,
                                instance_id: inst.instance_id,
                                instance_index: (start + offset) as u32,
                            });
                        }
                    }
                    match pop(&mut stack, &mut sp) {
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
                match pop(&mut stack, &mut sp) {
                    Some(n) => node_index = n,
                    None => break,
                }
            }
        }
        best
    }

    /// Shared explicit-stack any-hit walk for every occlusion entry point.
    ///
    /// Returns on the first instance for which `(inst.mask & ray_mask) != 0`, a
    /// non-degenerate pose blend at `time`, and an object-space hit (watertight
    /// or not per `watertight`); never shrinks `t_max`.
    fn walk_any(
        &self,
        ray: &Ray,
        time: f32,
        blases: &[Bvh],
        ray_mask: u8,
        watertight: bool,
    ) -> bool {
        if self.nodes.is_empty() {
            return false;
        }
        let t_min = ray.t_min();
        let t_max = ray.t_max();

        let mut stack = [0u32; 64];
        let mut sp = 0usize;
        let mut node_index = 0u32;
        loop {
            let node = &self.nodes[node_index as usize];
            if ray.aabb_interval(&node.bounds, t_min, t_max).is_some() {
                if node.is_leaf() {
                    let start = node.first_primitive as usize;
                    let end = start + node.primitive_count as usize;
                    for inst in &self.instances[start..end] {
                        if inst.mask & ray_mask == 0 {
                            continue;
                        }
                        let Some(world_to_object) = inst.world_to_object_at(time) else {
                            continue;
                        };
                        let obj_origin = world_to_object.transform_point(ray.origin());
                        let obj_dir = world_to_object.transform_vector(ray.direction());
                        let obj_ray = Ray::new(obj_origin, obj_dir, t_min, t_max);
                        let occluded = if watertight {
                            blases[inst.blas].any_hit_watertight(&obj_ray)
                        } else {
                            blases[inst.blas].any_hit(&obj_ray)
                        };
                        if occluded {
                            return true;
                        }
                    }
                    match pop(&mut stack, &mut sp) {
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
                match pop(&mut stack, &mut sp) {
                    Some(n) => node_index = n,
                    None => break,
                }
            }
        }
        false
    }
}

#[inline]
fn pop(stack: &mut [u32; 64], sp: &mut usize) -> Option<u32> {
    if *sp == 0 {
        return None;
    }
    *sp -= 1;
    Some(stack[*sp])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ray_scene::bvh::Triangle;
    use crate::ray_scene::tlas::{Instance, Tlas};

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

    /// A small non-trivial `BLAS`: two axis-facing quads (4 triangles), matching
    /// the `tlas.rs` reference so both suites exercise identical object-space
    /// geometry.
    fn sample_blas() -> Bvh {
        let tris = vec![
            Triangle::new([-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], 0),
            Triangle::new([-1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0], 1),
            Triangle::new([0.5, -1.0, -1.0], [0.5, 1.0, -1.0], [0.5, 1.0, 1.0], 2),
            Triangle::new([0.5, -1.0, -1.0], [0.5, 1.0, 1.0], [0.5, -1.0, 1.0], 3),
        ];
        Bvh::build(&tris)
    }

    fn random_affine(rng: &mut Rng) -> Affine3 {
        let quat = [
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
        ];
        let rot = Affine3::from_quaternion(quat);
        let scale = Affine3::from_scale([
            rng.range(0.4, 2.5),
            rng.range(0.4, 2.5),
            rng.range(0.4, 2.5),
        ]);
        let trans = Affine3::from_translation([
            rng.range(-5.0, 5.0),
            rng.range(-5.0, 5.0),
            rng.range(-5.0, 5.0),
        ]);
        trans.compose(&rot.compose(&scale))
    }

    /// Brute-force reference: interpolate each instance's pose at `time` exactly
    /// as [`MotionInstance::pose_at`], invert it (skipping degenerate blends and
    /// masked-out instances), query the object-space `BLAS`, and keep the
    /// globally nearest hit as a stable `(instance_id, primitive, t)` tuple.
    fn brute_closest_at(
        instances: &[MotionInstance],
        blases: &[Bvh],
        ray: &Ray,
        time: f32,
        ray_mask: u8,
        watertight: bool,
    ) -> Option<(u32, u32, f32)> {
        let mut best: Option<(u32, u32, f32)> = None;
        let mut best_t = ray.t_max();
        for inst in instances {
            if inst.mask() & ray_mask == 0 {
                continue;
            }
            let Some(world_to_object) = inst.world_to_object_at(time) else {
                continue;
            };
            let obj_origin = world_to_object.transform_point(ray.origin());
            let obj_dir = world_to_object.transform_vector(ray.direction());
            let obj_ray = Ray::new(obj_origin, obj_dir, ray.t_min(), ray.t_max());
            let hit = if watertight {
                blases[inst.blas()].closest_hit_watertight(&obj_ray)
            } else {
                blases[inst.blas()].closest_hit(&obj_ray)
            };
            if let Some(hit) = hit
                && hit.t < best_t
            {
                best_t = hit.t;
                best = Some((inst.instance_id(), hit.primitive, hit.t));
            }
        }
        best
    }

    /// Brute-force occlusion reference mirroring [`MotionTlas::walk_any`].
    fn brute_any_at(
        instances: &[MotionInstance],
        blases: &[Bvh],
        ray: &Ray,
        time: f32,
        ray_mask: u8,
        watertight: bool,
    ) -> bool {
        for inst in instances {
            if inst.mask() & ray_mask == 0 {
                continue;
            }
            let Some(world_to_object) = inst.world_to_object_at(time) else {
                continue;
            };
            let obj_origin = world_to_object.transform_point(ray.origin());
            let obj_dir = world_to_object.transform_vector(ray.direction());
            let obj_ray = Ray::new(obj_origin, obj_dir, ray.t_min(), ray.t_max());
            let occluded = if watertight {
                blases[inst.blas()].any_hit_watertight(&obj_ray)
            } else {
                blases[inst.blas()].any_hit(&obj_ray)
            };
            if occluded {
                return true;
            }
        }
        false
    }

    #[test]
    fn non_invertible_endpoint_has_no_instance() {
        let flat = Affine3::from_scale([1.0, 0.0, 1.0]);
        let ok = Affine3::identity();
        assert!(MotionInstance::new(flat, ok, 0, 0).is_none());
        assert!(MotionInstance::new(ok, flat, 0, 0).is_none());
        assert!(MotionInstance::new(ok, ok, 0, 0).is_some());
    }

    #[test]
    fn pose_at_clamps_and_blends_component_wise() {
        let from = Affine3::from_translation([0.0, 0.0, 0.0]);
        let to = Affine3::from_translation([10.0, -4.0, 2.0]);
        let inst = MotionInstance::new(from, to, 0, 0).unwrap();
        // Endpoints are exact.
        assert_eq!(inst.pose_at(0.0), from);
        assert_eq!(inst.pose_at(1.0), to);
        // Below/above the shutter clamps to the endpoints.
        assert_eq!(inst.pose_at(-3.0), from);
        assert_eq!(inst.pose_at(5.0), to);
        // Midpoint is the arithmetic mean of the translations.
        let mid = inst.pose_at(0.5).translation();
        assert!(approx(mid[0], 5.0, 1e-6));
        assert!(approx(mid[1], -2.0, 1e-6));
        assert!(approx(mid[2], 1.0, 1e-6));
    }

    #[test]
    fn static_instance_matches_static_tlas_at_every_time() {
        let blases = vec![sample_blas()];
        let mut rng = Rng::new(0xABCD_1234);
        let mut motion = Vec::new();
        let mut statics = Vec::new();
        for id in 0..6u32 {
            let pose = random_affine(&mut rng);
            motion.push(MotionInstance::new(pose, pose, 0, id).unwrap());
            statics.push(Instance::new(pose, 0, id).unwrap());
        }
        let mtlas = MotionTlas::build(&motion, &blases);
        let stlas = Tlas::build(&statics, &blases);
        // With from == to the swept bounds equal the single-pose bounds, so the
        // build reorders identically: instance_index must match too.
        for &time in &[0.0f32, 0.25, 0.5, 0.75, 1.0] {
            for _ in 0..1500 {
                let origin =
                    [rng.range(-6.0, 6.0), rng.range(-6.0, 6.0), rng.range(-6.0, 6.0)];
                let target =
                    [rng.range(-2.0, 2.0), rng.range(-2.0, 2.0), rng.range(-2.0, 2.0)];
                let dir = [
                    target[0] - origin[0],
                    target[1] - origin[1],
                    target[2] - origin[2],
                ];
                let ray = Ray::infinite(origin, dir);
                let m = mtlas.closest_hit(&ray, time, &blases);
                let s = stlas.closest_hit(&ray, &blases);
                match (m, s) {
                    (None, None) => {}
                    (Some(m), Some(s)) => {
                        assert_eq!(m.instance_id, s.instance_id);
                        assert_eq!(m.primitive, s.primitive);
                        assert_eq!(m.instance_index, s.instance_index);
                        assert!(approx(m.t, s.t, 1e-5), "t {} != {}", m.t, s.t);
                    }
                    (a, b) => panic!("static mismatch at t={time}: {a:?} vs {b:?}"),
                }
            }
        }
    }

    #[test]
    fn endpoints_match_static_tlas_from_and_to() {
        let blases = vec![sample_blas()];
        let mut rng = Rng::new(0x55AA_7799);
        let mut motion = Vec::new();
        let mut from_static = Vec::new();
        let mut to_static = Vec::new();
        for id in 0..5u32 {
            let from = random_affine(&mut rng);
            let to = random_affine(&mut rng);
            motion.push(MotionInstance::new(from, to, 0, id).unwrap());
            from_static.push(Instance::new(from, 0, id).unwrap());
            to_static.push(Instance::new(to, 0, id).unwrap());
        }
        let mtlas = MotionTlas::build(&motion, &blases);
        let from_tlas = Tlas::build(&from_static, &blases);
        let to_tlas = Tlas::build(&to_static, &blases);
        // Swept bounds differ from single-pose bounds, so instance_index can be
        // reordered; compare the stable (instance_id, primitive, t) tuple only.
        for _ in 0..3000 {
            let origin = [rng.range(-6.0, 6.0), rng.range(-6.0, 6.0), rng.range(-6.0, 6.0)];
            let target = [rng.range(-2.0, 2.0), rng.range(-2.0, 2.0), rng.range(-2.0, 2.0)];
            let dir = [
                target[0] - origin[0],
                target[1] - origin[1],
                target[2] - origin[2],
            ];
            let ray = Ray::infinite(origin, dir);
            let at0 = mtlas.closest_hit(&ray, 0.0, &blases).map(|h| (h.instance_id, h.primitive));
            let s0 = from_tlas.closest_hit(&ray, &blases).map(|h| (h.instance_id, h.primitive));
            assert_eq!(at0, s0, "time=0 should match the from-pose TLAS");
            let at1 = mtlas.closest_hit(&ray, 1.0, &blases).map(|h| (h.instance_id, h.primitive));
            let s1 = to_tlas.closest_hit(&ray, &blases).map(|h| (h.instance_id, h.primitive));
            assert_eq!(at1, s1, "time=1 should match the to-pose TLAS");
        }
    }

    #[test]
    fn translating_instance_hit_distance_tracks_the_pose() {
        // A single quad translating along +x; a ray fired down -z at x=0 hits the
        // z=0 quad only while its swept x-extent still covers the origin. Verify
        // the hit distance equals the ray's z travel at three shutter times.
        let blases = vec![sample_blas()];
        let from = Affine3::from_translation([0.0, 0.0, 0.0]);
        let to = Affine3::from_translation([0.0, 0.0, -4.0]);
        let inst = MotionInstance::new(from, to, 0, 0).unwrap();
        let tlas = MotionTlas::build(&[inst], &blases);
        // Ray from z=+5 pointing toward -z through the origin column.
        let ray = Ray::infinite([0.0, 0.0, 5.0], [0.0, 0.0, -1.0]);
        for (time, expected_z) in [(0.0f32, 0.0f32), (0.5, -2.0), (1.0, -4.0)] {
            let hit = tlas.closest_hit(&ray, time, &blases).expect("quad hit");
            // The z=0 quad sits at world z = expected_z; ray travels |5 - z| units.
            assert!(
                approx(hit.t, 5.0 - expected_z, 1e-4),
                "time={time}: t={} expected {}",
                hit.t,
                5.0 - expected_z
            );
        }
    }

    #[test]
    fn mask_gating_semantics() {
        let blases = vec![sample_blas()];
        let pose = Affine3::identity();
        let a = MotionInstance::with_mask(pose, pose, 0, 1, 0x01).unwrap();
        let b = MotionInstance::with_mask(pose, pose, 0, 2, 0x02).unwrap();
        let tlas = MotionTlas::build(&[a, b], &blases);
        let ray = Ray::infinite([0.0, 0.0, 5.0], [0.0, 0.0, -1.0]);

        // ray_mask 0 excludes everything.
        assert!(tlas.closest_hit_masked(&ray, 0.5, &blases, 0x00).is_none());
        assert!(!tlas.any_hit_masked(&ray, 0.5, &blases, 0x00));
        // A ray mask overlapping only instance 1 hits instance 1.
        let hit = tlas.closest_hit_masked(&ray, 0.5, &blases, 0x01).expect("hit");
        assert_eq!(hit.instance_id, 1);
        // Overlapping only instance 2 hits instance 2 (same geometry, id differs).
        let hit = tlas.closest_hit_masked(&ray, 0.5, &blases, 0x02).expect("hit");
        assert_eq!(hit.instance_id, 2);
        // MASK_ALL is equivalent to the unmasked entry point.
        let full = tlas.closest_hit_masked(&ray, 0.5, &blases, MotionInstance::MASK_ALL);
        let plain = tlas.closest_hit(&ray, 0.5, &blases);
        assert_eq!(full.map(|h| h.instance_id), plain.map(|h| h.instance_id));
        // A disjoint mask (no bit overlap with any instance) never hits.
        assert!(tlas.closest_hit_masked(&ray, 0.5, &blases, 0x04).is_none());
    }

    #[test]
    fn random_scene_matches_brute_force_all_four_variants() {
        let blases = vec![sample_blas()];
        let mut rng = Rng::new(0xF00D_CAFE);
        let mut motion = Vec::new();
        for id in 0..7u32 {
            let from = random_affine(&mut rng);
            let to = random_affine(&mut rng);
            let mask = (rng.next_u32() as u8) | 0x01; // never zero
            motion.push(MotionInstance::with_mask(from, to, 0, id, mask).unwrap());
        }
        let tlas = MotionTlas::build(&motion, &blases);
        for _ in 0..5000 {
            let time = rng.range(-0.2, 1.2); // exercise the clamp too
            let ray_mask = rng.next_u32() as u8;
            let origin = [rng.range(-7.0, 7.0), rng.range(-7.0, 7.0), rng.range(-7.0, 7.0)];
            let target = [rng.range(-2.0, 2.0), rng.range(-2.0, 2.0), rng.range(-2.0, 2.0)];
            let dir = [
                target[0] - origin[0],
                target[1] - origin[1],
                target[2] - origin[2],
            ];
            let ray = Ray::infinite(origin, dir);

            for &wt in &[false, true] {
                let got = if wt {
                    tlas.closest_hit_watertight_masked(&ray, time, &blases, ray_mask)
                } else {
                    tlas.closest_hit_masked(&ray, time, &blases, ray_mask)
                };
                let want = brute_closest_at(&motion, &blases, &ray, time, ray_mask, wt);
                match (got, want) {
                    (None, None) => {}
                    (Some(g), Some((iid, prim, t))) => {
                        assert_eq!(g.instance_id, iid, "wt={wt}");
                        assert_eq!(g.primitive, prim, "wt={wt}");
                        assert!(approx(g.t, t, 1e-4), "wt={wt}: {} != {}", g.t, t);
                    }
                    (a, b) => panic!("closest mismatch wt={wt}: {a:?} vs {b:?}"),
                }

                let occ = if wt {
                    tlas.any_hit_watertight_masked(&ray, time, &blases, ray_mask)
                } else {
                    tlas.any_hit_masked(&ray, time, &blases, ray_mask)
                };
                let occ_want = brute_any_at(&motion, &blases, &ray, time, ray_mask, wt);
                assert_eq!(occ, occ_want, "any wt={wt}");
            }
        }
    }

    #[test]
    fn traversal_is_deterministic_bit_for_bit() {
        let blases = vec![sample_blas()];
        let mut rng = Rng::new(0x0BAD_F00D);
        let mut motion = Vec::new();
        for id in 0..5u32 {
            let from = random_affine(&mut rng);
            let to = random_affine(&mut rng);
            motion.push(MotionInstance::new(from, to, 0, id).unwrap());
        }
        let a = MotionTlas::build(&motion, &blases);
        let b = MotionTlas::build(&motion, &blases);
        assert_eq!(a, b);
        for _ in 0..2000 {
            let time = rng.range(0.0, 1.0);
            let origin = [rng.range(-6.0, 6.0), rng.range(-6.0, 6.0), rng.range(-6.0, 6.0)];
            let dir = [rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(-1.0, 1.0)];
            let ray = Ray::infinite(origin, dir);
            let h1 = a.closest_hit(&ray, time, &blases);
            let h2 = b.closest_hit(&ray, time, &blases);
            match (h1, h2) {
                (None, None) => {}
                (Some(x), Some(y)) => {
                    assert_eq!(x.t.to_bits(), y.t.to_bits());
                    assert_eq!(x.u.to_bits(), y.u.to_bits());
                    assert_eq!(x.v.to_bits(), y.v.to_bits());
                    assert_eq!(x.primitive, y.primitive);
                    assert_eq!(x.instance_id, y.instance_id);
                    assert_eq!(x.instance_index, y.instance_index);
                }
                (a, b) => panic!("nondeterministic: {a:?} vs {b:?}"),
            }
        }
    }

    #[test]
    fn swept_bounds_contain_every_mid_shutter_pose() {
        let blases = vec![sample_blas()];
        let mut rng = Rng::new(0x1357_9BDF);
        for _ in 0..400 {
            let from = random_affine(&mut rng);
            let to = random_affine(&mut rng);
            let inst = MotionInstance::new(from, to, 0, 0).unwrap();
            let swept = inst.swept_world_bounds(&blases);
            let local = blases[0].bounds();
            for step in 0..=10u32 {
                let time = step as f32 / 10.0;
                let pose = inst.pose_at(time);
                for &cx in &[local.min[0], local.max[0]] {
                    for &cy in &[local.min[1], local.max[1]] {
                        for &cz in &[local.min[2], local.max[2]] {
                            let p = pose.transform_point([cx, cy, cz]);
                            let eps = 1e-4;
                            assert!(
                                p[0] >= swept.min[0] - eps
                                    && p[0] <= swept.max[0] + eps
                                    && p[1] >= swept.min[1] - eps
                                    && p[1] <= swept.max[1] + eps
                                    && p[2] >= swept.min[2] - eps
                                    && p[2] <= swept.max[2] + eps,
                                "corner {p:?} escaped swept bounds {swept:?} at t={time}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn empty_motion_tlas_never_hits() {
        let blases = vec![sample_blas()];
        let tlas = MotionTlas::build(&[], &blases);
        assert!(tlas.is_empty());
        assert_eq!(tlas.node_count(), 0);
        let ray = Ray::infinite([0.0, 0.0, 5.0], [0.0, 0.0, -1.0]);
        assert!(tlas.closest_hit(&ray, 0.5, &blases).is_none());
        assert!(!tlas.any_hit(&ray, 0.5, &blases));
    }
}
