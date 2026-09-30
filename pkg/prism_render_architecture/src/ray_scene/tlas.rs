//! Two-level acceleration: a top-level `BVH` (`TLAS`) over affine-transformed
//! instances of bottom-level `BVH`s (`BLAS`).
//!
//! This is the `CPU`-verifiable golden reference for the two-level traversal a
//! `GPU` kernel mirrors. A [`Tlas`] holds its own flattened [`LinearBvhNode`]
//! array (built by the *same* [`build_linear_bvh`] `SAH` builder the `BLAS`
//! uses, over each instance's world-space bounds) plus a reordered instance
//! table. It does **not** own the [`Bvh`] geometry: instances reference a
//! shared `BLAS` pool by index, so a scene keeps one `Bvh` per unique mesh and
//! instances it many times with different transforms — the standard `AAA`
//! layout that keeps the acceleration memory bounded by unique geometry, not
//! by instance count.
//!
//! Traversal transforms the world ray into each instance's object space with
//! the cached inverse ([`Affine3::inverse`]). Because the direction is carried
//! through the *linear* part without renormalizing, the ray parameter `t` is
//! identical in both spaces, so object-space hit distances compare directly in
//! world space and the running `t_max` prunes across instances exactly like a
//! single-level walk.

use super::bvh::{build_linear_bvh, linear_sah_cost, Aabb, Bvh, BvhBuildConfig, LinearBvhNode};
use super::traversal::Ray;

/// An affine transform: a 3×3 linear map (stored column-major) plus a
/// translation applied after it.
///
/// `cols[c]` is the image of basis vector `c`, so a point `p` maps to
/// `cols[0]*p.x + cols[1]*p.y + cols[2]*p.z + translation`. Column-major
/// storage matches the `GPU`-side `mat3` convention the kernel consumes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Affine3 {
    cols: [[f32; 3]; 3],
    translation: [f32; 3],
}

impl Affine3 {
    /// Builds a transform from explicit columns and a translation.
    #[must_use]
    pub const fn from_cols(cols: [[f32; 3]; 3], translation: [f32; 3]) -> Self {
        Self { cols, translation }
    }

    /// The identity transform.
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            cols: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translation: [0.0, 0.0, 0.0],
        }
    }

    /// A pure translation.
    #[must_use]
    pub const fn from_translation(t: [f32; 3]) -> Self {
        Self {
            cols: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translation: t,
        }
    }

    /// A pure (possibly non-uniform) scale about the origin.
    #[must_use]
    pub const fn from_scale(s: [f32; 3]) -> Self {
        Self {
            cols: [[s[0], 0.0, 0.0], [0.0, s[1], 0.0], [0.0, 0.0, s[2]]],
            translation: [0.0, 0.0, 0.0],
        }
    }

    /// A rotation from a quaternion `[w, x, y, z]`.
    ///
    /// The quaternion is normalized defensively (a zero quaternion yields the
    /// identity), then converted with the standard arithmetic form — no
    /// trigonometry, keeping this module's `libm`-free determinism so a `GPU`
    /// kernel reproduces the matrix bit-for-bit.
    #[must_use]
    pub fn from_quaternion(q: [f32; 4]) -> Self {
        let len = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
        if len <= f32::MIN_POSITIVE {
            return Self::identity();
        }
        let (w, x, y, z) = (q[0] / len, q[1] / len, q[2] / len, q[3] / len);
        // Column-major: cols[c][r] = R[r][c] of the quaternion rotation matrix.
        Self {
            cols: [
                [
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y + w * z),
                    2.0 * (x * z - w * y),
                ],
                [
                    2.0 * (x * y - w * z),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z + w * x),
                ],
                [
                    2.0 * (x * z + w * y),
                    2.0 * (y * z - w * x),
                    1.0 - 2.0 * (x * x + y * y),
                ],
            ],
            translation: [0.0, 0.0, 0.0],
        }
    }

    /// Read-only columns of the linear part.
    #[must_use]
    pub const fn columns(&self) -> [[f32; 3]; 3] {
        self.cols
    }

    /// Read-only translation.
    #[must_use]
    pub const fn translation(&self) -> [f32; 3] {
        self.translation
    }

    /// Applies the linear part to a direction/vector (ignores translation).
    #[must_use]
    pub fn transform_vector(&self, v: [f32; 3]) -> [f32; 3] {
        [
            self.cols[0][0] * v[0] + self.cols[1][0] * v[1] + self.cols[2][0] * v[2],
            self.cols[0][1] * v[0] + self.cols[1][1] * v[1] + self.cols[2][1] * v[2],
            self.cols[0][2] * v[0] + self.cols[1][2] * v[1] + self.cols[2][2] * v[2],
        ]
    }

    /// Applies the *transpose* of the linear part to a vector (`Lᵀ · v`).
    ///
    /// Each output component is the dot of `v` with a column of `L`, which is a
    /// row of `Lᵀ`. This is the building block for transforming surface normals:
    /// composed with the cached world→object inverse it yields the
    /// inverse-transpose `(L⁻¹)ᵀ` that normals require under non-uniform scale or
    /// shear (see [`Instance::transform_normal_to_world`]).
    #[must_use]
    pub fn transpose_transform_vector(&self, v: [f32; 3]) -> [f32; 3] {
        [
            self.cols[0][0] * v[0] + self.cols[0][1] * v[1] + self.cols[0][2] * v[2],
            self.cols[1][0] * v[0] + self.cols[1][1] * v[1] + self.cols[1][2] * v[2],
            self.cols[2][0] * v[0] + self.cols[2][1] * v[1] + self.cols[2][2] * v[2],
        ]
    }

    /// Applies the full affine map to a point.
    #[must_use]
    pub fn transform_point(&self, p: [f32; 3]) -> [f32; 3] {
        let v = self.transform_vector(p);
        [
            v[0] + self.translation[0],
            v[1] + self.translation[1],
            v[2] + self.translation[2],
        ]
    }

    /// Composition `self ∘ inner`: the transform that applies `inner` first,
    /// then `self`.
    #[must_use]
    pub fn compose(&self, inner: &Affine3) -> Affine3 {
        Affine3 {
            cols: [
                self.transform_vector(inner.cols[0]),
                self.transform_vector(inner.cols[1]),
                self.transform_vector(inner.cols[2]),
            ],
            translation: self.transform_point(inner.translation),
        }
    }

    /// Inverse transform, or `None` when the linear part is singular.
    ///
    /// Uses the closed-form 3×3 adjugate over the determinant; the inverse
    /// translation is `-L⁻¹ · t` so that `inv.transform_point(self.transform_point(p)) == p`.
    #[must_use]
    pub fn inverse(&self) -> Option<Affine3> {
        // Row-form entries m[r][c] = cols[c][r].
        let a = self.cols[0][0];
        let d = self.cols[0][1];
        let g = self.cols[0][2];
        let b = self.cols[1][0];
        let e = self.cols[1][1];
        let h = self.cols[1][2];
        let c = self.cols[2][0];
        let f = self.cols[2][1];
        let i = self.cols[2][2];

        let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
        if det.abs() <= f32::MIN_POSITIVE {
            return None;
        }
        let inv_det = 1.0 / det;

        // Inverse in row form.
        let inv00 = (e * i - f * h) * inv_det;
        let inv01 = (c * h - b * i) * inv_det;
        let inv02 = (b * f - c * e) * inv_det;
        let inv10 = (f * g - d * i) * inv_det;
        let inv11 = (a * i - c * g) * inv_det;
        let inv12 = (c * d - a * f) * inv_det;
        let inv20 = (d * h - e * g) * inv_det;
        let inv21 = (b * g - a * h) * inv_det;
        let inv22 = (a * e - b * d) * inv_det;

        // Back to column-major: inv_cols[c][r] = inv[r][c].
        let inv_cols = [
            [inv00, inv10, inv20],
            [inv01, inv11, inv21],
            [inv02, inv12, inv22],
        ];
        let t = self.translation;
        let inv_translation = [
            -(inv00 * t[0] + inv01 * t[1] + inv02 * t[2]),
            -(inv10 * t[0] + inv11 * t[1] + inv12 * t[2]),
            -(inv20 * t[0] + inv21 * t[1] + inv22 * t[2]),
        ];
        Some(Affine3 {
            cols: inv_cols,
            translation: inv_translation,
        })
    }
}

/// One placement of a `BLAS` into the scene.
///
/// Holds both the object→world transform and its cached inverse (world→object,
/// computed once at construction) plus the `BLAS` pool index the instance
/// references and a stable user-facing `instance_id`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Instance {
    object_to_world: Affine3,
    world_to_object: Affine3,
    blas: usize,
    instance_id: u32,
    mask: u8,
}

impl Instance {
    /// Builds an instance of `BLAS` pool entry `blas` placed by `object_to_world`.
    ///
    /// Returns `None` when the transform is non-invertible (degenerate scale),
    /// since such an instance has no well-defined object space to trace in.
    #[must_use]
    pub fn new(object_to_world: Affine3, blas: usize, instance_id: u32) -> Option<Self> {
        Self::with_mask(object_to_world, blas, instance_id, Self::MASK_ALL)
    }

    /// The all-ones (`0xFF`) visibility mask assigned by [`Instance::new`].
    ///
    /// An instance built with this mask is visible to every ray regardless of
    /// the ray's inclusion mask, matching the DXR default where an instance
    /// with `InstanceMask = 0xFF` participates in all `TraceRay` calls.
    pub const MASK_ALL: u8 = 0xFF;

    /// Builds an instance carrying an explicit 8-bit DXR-style visibility mask.
    ///
    /// The `mask` follows Direct3D 12 `InstanceMask` semantics: a ray traced
    /// with inclusion mask `ray_mask` tests this instance only when
    /// `(mask & ray_mask) != 0`. A `mask` of `0` makes the instance invisible
    /// to every ray (it can never satisfy the bitwise-AND predicate), which is
    /// useful for temporarily disabling an instance without removing it from
    /// the `TLAS`. Selective categories (for example "casts shadows" versus
    /// "seen by reflections") are encoded as distinct bits so a shadow ray and
    /// a reflection ray can each include a different subset of the scene.
    ///
    /// Returns `None` when the transform is non-invertible (degenerate scale),
    /// exactly like [`Instance::new`].
    #[must_use]
    pub fn with_mask(
        object_to_world: Affine3,
        blas: usize,
        instance_id: u32,
        mask: u8,
    ) -> Option<Self> {
        let world_to_object = object_to_world.inverse()?;
        Some(Self {
            object_to_world,
            world_to_object,
            blas,
            instance_id,
            mask,
        })
    }

    /// Object→world transform.
    #[must_use]
    pub const fn object_to_world(&self) -> Affine3 {
        self.object_to_world
    }

    /// Cached world→object transform.
    #[must_use]
    pub const fn world_to_object(&self) -> Affine3 {
        self.world_to_object
    }

    /// Transforms an object-space surface normal into world space.
    ///
    /// Directions transform by the linear part `L`, but normals do not: a normal
    /// must stay perpendicular to the surface it describes, which under
    /// non-uniform scale or shear requires the inverse-transpose `(L⁻¹)ᵀ` rather
    /// than `L` (transforming a normal by `L` tilts it off the surface — the
    /// classic instanced-normal bug that shows up as wrong shading on stretched
    /// or sheared instances). Because the world→object inverse is already cached,
    /// this applies its transpose (`(L⁻¹)ᵀ`) with
    /// [`Affine3::transpose_transform_vector`] and renormalises, so the result is
    /// the unit world-space normal for a `BLAS` geometric or shading normal (for
    /// example [`super::bvh::Triangle::geometric_normal`]). A zero or degenerate
    /// input, or a result that collapses to zero length, yields `[0, 0, 0]`.
    #[must_use]
    pub fn transform_normal_to_world(&self, object_normal: [f32; 3]) -> [f32; 3] {
        let n = self.world_to_object.transpose_transform_vector(object_normal);
        let len_sq = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
        if len_sq <= 0.0 {
            return [0.0, 0.0, 0.0];
        }
        let inv_len = 1.0 / len_sq.sqrt();
        [n[0] * inv_len, n[1] * inv_len, n[2] * inv_len]
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
    /// Defaults to [`Instance::MASK_ALL`] for instances built via
    /// [`Instance::new`]; set explicitly through [`Instance::with_mask`]. A ray
    /// with inclusion mask `ray_mask` tests this instance only when
    /// `(mask() & ray_mask) != 0`.
    #[must_use]
    pub const fn mask(&self) -> u8 {
        self.mask
    }
}

/// A ray/instance intersection reported by the `TLAS`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TlasHit {
    /// Ray parameter at the hit (identical in world and object space).
    pub t: f32,
    /// Barycentric `u` (weight of the hit triangle's `v1`).
    pub u: f32,
    /// Barycentric `v` (weight of the hit triangle's `v2`).
    pub v: f32,
    /// Stable primitive id from the hit `BLAS` triangle.
    pub primitive: u32,
    /// Stable id of the instance that was hit.
    pub instance_id: u32,
    /// Index of the hit instance in [`Tlas::instances`] (reordered order).
    pub instance_index: u32,
}

/// A built top-level acceleration structure over a set of [`Instance`]s.
///
/// Empty input yields an empty structure; traversal never reports a hit.
#[derive(Clone, Debug, PartialEq)]
pub struct Tlas {
    nodes: Vec<LinearBvhNode>,
    instances: Vec<Instance>,
}

/// World-space bounds of `inst`: the `BLAS` root box's eight corners mapped
/// through `object_to_world` and unioned. An empty `BLAS` yields empty bounds.
fn instance_world_bounds(inst: &Instance, blases: &[Bvh]) -> Aabb {
    let local = blases[inst.blas].bounds();
    if local.is_empty() {
        return Aabb::empty();
    }
    let mut out = Aabb::empty();
    for &cx in &[local.min[0], local.max[0]] {
        for &cy in &[local.min[1], local.max[1]] {
            for &cz in &[local.min[2], local.max[2]] {
                out = out.enclose(inst.object_to_world.transform_point([cx, cy, cz]));
            }
        }
    }
    out
}

impl Tlas {
    /// Builds a `TLAS` over `instances` (referencing `blases`) with the default
    /// [`BvhBuildConfig`].
    #[must_use]
    pub fn build(instances: &[Instance], blases: &[Bvh]) -> Self {
        Self::build_with(instances, blases, BvhBuildConfig::default())
    }

    /// Builds a `TLAS` with an explicit builder configuration.
    #[must_use]
    pub fn build_with(instances: &[Instance], blases: &[Bvh], config: BvhBuildConfig) -> Self {
        let bounds: Vec<Aabb> = instances
            .iter()
            .map(|inst| instance_world_bounds(inst, blases))
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

    /// True when the `TLAS` holds no instances.
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
    pub fn instances(&self) -> &[Instance] {
        &self.instances
    }

    /// Root world-space bounds, or [`Aabb::empty`] when the `TLAS` is empty.
    #[must_use]
    pub fn bounds(&self) -> Aabb {
        self.nodes.first().map_or(Aabb::empty(), |n| n.bounds)
    }

    /// Refits every top-level node's bounds in place after instance transforms
    /// changed, preserving the existing topology (node and leaf structure and
    /// the reordered instance table).
    ///
    /// `updated(instance_id)` returns the instance's new object→world transform.
    /// Each instance's cached world→object inverse is recomputed; if the new
    /// transform is singular (non-invertible) the instance keeps its previous
    /// transform, so a degenerate frame never corrupts the table or leaves an
    /// instance without an object space to trace in. This is the top-level
    /// executor for
    /// [`AccelerationUpdate::Refit`](super::acceleration::AccelerationUpdate::Refit):
    /// valid only while the instance set is intact — none added, removed, or
    /// reordered — which the policy guards by bounding motion and the moved
    /// ratio. A refit is `O(nodes)` versus a rebuild's `O(n log n)`, trading
    /// gradually looser (still conservative) bounds under large motion for a far
    /// cheaper per-frame update.
    ///
    /// `blases` must be the same pool passed to [`Tlas::build`], used to
    /// recompute each moved instance's world-space bounds. Bounds are
    /// recomputed bottom-up in a single reverse pass, correct because the
    /// depth-first flattening guarantees both children of an interior node sit
    /// at a strictly greater array index than the node itself.
    pub fn refit(&mut self, updated: impl Fn(u32) -> Affine3, blases: &[Bvh]) {
        for inst in &mut self.instances {
            let object_to_world = updated(inst.instance_id);
            if let Some(world_to_object) = object_to_world.inverse() {
                inst.object_to_world = object_to_world;
                inst.world_to_object = world_to_object;
            }
        }
        for i in (0..self.nodes.len()).rev() {
            let node = self.nodes[i];
            let bounds = if node.is_leaf() {
                let start = node.first_primitive as usize;
                let end = start + node.primitive_count as usize;
                self.instances[start..end]
                    .iter()
                    .fold(Aabb::empty(), |acc, inst| {
                        acc.union(&instance_world_bounds(inst, blases))
                    })
            } else {
                let first = self.nodes[i + 1].bounds;
                let second = self.nodes[node.second_child as usize].bounds;
                first.union(&second)
            };
            self.nodes[i].bounds = bounds;
        }
    }

    /// Rebuilds a fresh, maximally compact top-level hierarchy from the current
    /// (possibly refit-moved) instance transforms, over the same `blases` pool.
    ///
    /// This is the executor for
    /// [`AccelerationUpdate::Rebuild`](super::acceleration::AccelerationUpdate::Rebuild)
    /// and, because the flattened top-level array is contiguous by construction
    /// with no inter-node fragmentation, also for
    /// [`AccelerationUpdate::BuildAndCompact`](super::acceleration::AccelerationUpdate::BuildAndCompact):
    /// the rebuild *is* the compaction. Use this after instance motion has
    /// loosened refit bounds enough that the policy escalates from
    /// [`refit`](Self::refit) to a rebuild.
    #[must_use]
    pub fn rebuilt(&self, blases: &[Bvh]) -> Tlas {
        Tlas::build(&self.instances, blases)
    }

    /// Surface-area-heuristic expected traversal cost of the current top-level
    /// hierarchy, scored with the same shared model as
    /// [`Bvh::sah_cost`](super::bvh::Bvh::sah_cost).
    ///
    /// Each top-level leaf's primitive count is the number of *instances* it
    /// forces a ray to transform-and-test, so the score is the expected number
    /// of node visits plus instance descents for a uniform ray, normalised by
    /// the root surface area. `traversal_cost` weights an interior-node visit
    /// relative to one instance test (fixed at `1.0`); pass the same value the
    /// `TLAS` was built with to compare a tree against its own build weight. An
    /// empty `TLAS` scores `0.0`.
    #[must_use]
    pub fn sah_cost(&self, traversal_cost: f32) -> f64 {
        linear_sah_cost(&self.nodes, traversal_cost)
    }

    /// Ratio of the current top-level hierarchy's [`sah_cost`](Self::sah_cost) to
    /// that of a fresh [`rebuilt`](Self::rebuilt) tree over the same instances.
    ///
    /// A [`refit`](Self::refit) keeps every top-level box tight for the *existing*
    /// topology but never re-partitions, so as instances translate across the
    /// scene the original split planes stop matching their world-space layout and
    /// a ray descends more instance leaves per query even though every box is
    /// still snug. Comparing the refit tree's cost against a rebuild isolates that
    /// topological degradation: the value is `1.0` right after a build and climbs
    /// above `1.0` as instance motion accumulates. This is the top-level twin of
    /// [`Bvh::refit_quality`](super::bvh::Bvh::refit_quality) and feeds the same
    /// [`AccelerationUpdatePolicy::should_rebuild_after_refit`](super::acceleration::AccelerationUpdatePolicy::should_rebuild_after_refit)
    /// decision so many-instance dynamic scenes escalate a cheap top-level refit
    /// to a full rebuild once the hierarchy has drifted too far.
    ///
    /// Both trees are scored with the same `traversal_cost`; `blases` must be the
    /// same pool passed to [`Tlas::build`]. An empty `TLAS`, whose costs are both
    /// `0.0`, reports `1.0` (no degradation).
    #[must_use]
    pub fn refit_quality(&self, blases: &[Bvh], traversal_cost: f32) -> f64 {
        let current = self.sah_cost(traversal_cost);
        let ideal = self.rebuilt(blases).sah_cost(traversal_cost);
        if ideal <= 0.0 {
            return 1.0;
        }
        current / ideal
    }

    /// Nearest intersection along the world-space `ray`, or `None`.
    ///
    /// `blases` must be the same pool passed to [`Tlas::build`]. Walks the
    /// top-level nodes with an explicit stack, and in each leaf transforms the
    /// ray into every instance's object space to query its `BLAS`, keeping the
    /// globally nearest hit. The running `t_max` shrinks across instances so
    /// far candidates are culled by the slab test.
    #[must_use]
    pub fn closest_hit(&self, ray: &Ray, blases: &[Bvh]) -> Option<TlasHit> {
        self.closest_hit_masked(ray, blases, Instance::MASK_ALL)
    }

    /// Nearest intersection restricted to instances the `ray_mask` includes.
    ///
    /// Applies DXR-style instance inclusion: an instance participates only when
    /// `(instance.mask() & ray_mask) != 0`, so a caller can trace, for example,
    /// a reflection ray that ignores instances excluded from reflections while
    /// the identical geometry still shows up for primary rays. Passing
    /// [`Instance::MASK_ALL`] reproduces [`Tlas::closest_hit`] exactly; a
    /// `ray_mask` of `0` matches nothing and always returns `None`. Traversal,
    /// `t_max` shrinking, and the returned [`TlasHit`] are otherwise identical.
    #[must_use]
    pub fn closest_hit_masked(&self, ray: &Ray, blases: &[Bvh], ray_mask: u8) -> Option<TlasHit> {
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
                        let obj_origin = inst.world_to_object.transform_point(ray.origin());
                        let obj_dir = inst.world_to_object.transform_vector(ray.direction());
                        let obj_ray = Ray::new(obj_origin, obj_dir, t_min, best_t);
                        if let Some(hit) = blases[inst.blas].closest_hit(&obj_ray)
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

    /// True when *any* instance intersects the world-space `ray` inside its
    /// interval. Returns on the first hit; the cheap shadow/AO query.
    #[must_use]
    pub fn any_hit(&self, ray: &Ray, blases: &[Bvh]) -> bool {
        self.any_hit_masked(ray, blases, Instance::MASK_ALL)
    }

    /// Occlusion query restricted to instances the `ray_mask` includes.
    ///
    /// The masked counterpart of [`Tlas::any_hit`]: an instance can occlude the
    /// ray only when `(instance.mask() & ray_mask) != 0`. This is the primitive
    /// behind selective shadows — a shadow ray traced with a "casts shadows"
    /// bit skips instances that are visible to the camera but excluded from
    /// shadow casting. [`Instance::MASK_ALL`] reproduces [`Tlas::any_hit`]; a
    /// `ray_mask` of `0` is never blocked.
    #[must_use]
    pub fn any_hit_masked(&self, ray: &Ray, blases: &[Bvh], ray_mask: u8) -> bool {
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
                        let obj_origin = inst.world_to_object.transform_point(ray.origin());
                        let obj_dir = inst.world_to_object.transform_vector(ray.direction());
                        let obj_ray = Ray::new(obj_origin, obj_dir, t_min, t_max);
                        if blases[inst.blas].any_hit(&obj_ray) {
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

    /// Nearest world-space intersection using the watertight `BLAS` walk.
    ///
    /// Identical top-level traversal to [`Tlas::closest_hit`] but each instance
    /// is queried with [`Bvh::closest_hit_watertight`], so a primary/reflection
    /// ray that strikes a seam shared by two triangles of an instanced mesh is
    /// never lost between them.
    #[must_use]
    pub fn closest_hit_watertight(&self, ray: &Ray, blases: &[Bvh]) -> Option<TlasHit> {
        self.closest_hit_watertight_masked(ray, blases, Instance::MASK_ALL)
    }

    /// Watertight nearest intersection restricted to `ray_mask`-included instances.
    ///
    /// Combines the leak-free seam handling of [`Tlas::closest_hit_watertight`]
    /// with DXR-style instance inclusion: an instance is queried only when
    /// `(instance.mask() & ray_mask) != 0`. [`Instance::MASK_ALL`] reproduces
    /// [`Tlas::closest_hit_watertight`]; a `ray_mask` of `0` returns `None`.
    #[must_use]
    pub fn closest_hit_watertight_masked(
        &self,
        ray: &Ray,
        blases: &[Bvh],
        ray_mask: u8,
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
                        let obj_origin = inst.world_to_object.transform_point(ray.origin());
                        let obj_dir = inst.world_to_object.transform_vector(ray.direction());
                        let obj_ray = Ray::new(obj_origin, obj_dir, t_min, best_t);
                        if let Some(hit) = blases[inst.blas].closest_hit_watertight(&obj_ray)
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

    /// Occlusion query using the watertight `BLAS` walk.
    ///
    /// The watertight counterpart of [`Tlas::any_hit`]: a shadow/AO ray aimed
    /// along a seam of an instanced occluder is still reported as blocked, so
    /// closed instanced meshes cast leak-free shadows.
    #[must_use]
    pub fn any_hit_watertight(&self, ray: &Ray, blases: &[Bvh]) -> bool {
        self.any_hit_watertight_masked(ray, blases, Instance::MASK_ALL)
    }

    /// Watertight occlusion query restricted to `ray_mask`-included instances.
    ///
    /// The masked counterpart of [`Tlas::any_hit_watertight`]: a seam-safe
    /// shadow/AO ray is blocked only by instances for which
    /// `(instance.mask() & ray_mask) != 0`. [`Instance::MASK_ALL`] reproduces
    /// [`Tlas::any_hit_watertight`]; a `ray_mask` of `0` is never blocked.
    #[must_use]
    pub fn any_hit_watertight_masked(&self, ray: &Ray, blases: &[Bvh], ray_mask: u8) -> bool {
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
                        let obj_origin = inst.world_to_object.transform_point(ray.origin());
                        let obj_dir = inst.world_to_object.transform_vector(ray.direction());
                        let obj_ray = Ray::new(obj_origin, obj_dir, t_min, t_max);
                        if blases[inst.blas].any_hit_watertight(&obj_ray) {
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

    fn approx_pt(a: [f32; 3], b: [f32; 3], eps: f32) -> bool {
        approx(a[0], b[0], eps) && approx(a[1], b[1], eps) && approx(a[2], b[2], eps)
    }

    /// A small non-trivial BLAS: two axis-facing quads (4 triangles) offset so
    /// rays along -z and +x both find geometry.
    fn sample_blas() -> Bvh {
        let tris = vec![
            // Quad in the z = 0 plane spanning x,y in [-1,1] (primitives 0,1).
            Triangle::new([-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], 0),
            Triangle::new([-1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0], 1),
            // Quad in the x = 0.5 plane spanning y,z in [-1,1] (primitives 2,3).
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
        // Keep scales comfortably away from zero so the transform is invertible.
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
        // Apply scale, then rotation, then translation.
        trans.compose(&rot.compose(&scale))
    }

    /// Brute-force reference: query every instance's BLAS in object space and
    /// keep the globally nearest hit. Compared to `Tlas::closest_hit` by the
    /// stable `(instance_id, primitive, t)` tuple.
    fn brute_closest(
        instances: &[Instance],
        blases: &[Bvh],
        ray: &Ray,
    ) -> Option<(u32, u32, f32)> {
        let mut best: Option<(u32, u32, f32)> = None;
        let mut best_t = ray.t_max();
        for inst in instances {
            let obj_origin = inst.world_to_object().transform_point(ray.origin());
            let obj_dir = inst.world_to_object().transform_vector(ray.direction());
            let obj_ray = Ray::new(obj_origin, obj_dir, ray.t_min(), ray.t_max());
            if let Some(hit) = blases[inst.blas()].closest_hit(&obj_ray)
                && hit.t < best_t
            {
                best_t = hit.t;
                best = Some((inst.instance_id(), hit.primitive, hit.t));
            }
        }
        best
    }

    #[test]
    fn affine_inverse_roundtrips() {
        let mut rng = Rng::new(0x1234_5678);
        for _ in 0..2000 {
            let m = random_affine(&mut rng);
            let inv = m.inverse().expect("random affine is invertible");
            let p = [
                rng.range(-10.0, 10.0),
                rng.range(-10.0, 10.0),
                rng.range(-10.0, 10.0),
            ];
            let round = inv.transform_point(m.transform_point(p));
            assert!(approx_pt(round, p, 1e-4), "roundtrip {round:?} != {p:?}");
            // Direction roundtrip (linear part only).
            let v = [rng.range(-2.0, 2.0), rng.range(-2.0, 2.0), rng.range(-2.0, 2.0)];
            let vr = inv.transform_vector(m.transform_vector(v));
            assert!(approx_pt(vr, v, 1e-4), "vec roundtrip {vr:?} != {v:?}");
        }
    }

    #[test]
    fn singular_transform_has_no_inverse_and_no_instance() {
        let flat = Affine3::from_scale([1.0, 0.0, 1.0]);
        assert!(flat.inverse().is_none());
        assert!(Instance::new(flat, 0, 0).is_none());
    }

    #[test]
    fn identity_instance_matches_blas_directly() {
        let blas = sample_blas();
        let blases = vec![blas.clone()];
        let inst = Instance::new(Affine3::identity(), 0, 7).unwrap();
        let tlas = Tlas::build(&[inst], &blases);

        let mut rng = Rng::new(99);
        for _ in 0..3000 {
            let origin = [rng.range(-3.0, 3.0), rng.range(-3.0, 3.0), rng.range(2.0, 6.0)];
            let target = [rng.range(-2.0, 2.0), rng.range(-2.0, 2.0), rng.range(-2.0, 2.0)];
            let dir = [
                target[0] - origin[0],
                target[1] - origin[1],
                target[2] - origin[2],
            ];
            let ray = Ray::infinite(origin, dir);
            let direct = blas.closest_hit(&ray);
            let via = tlas.closest_hit(&ray, &blases);
            match (direct, via) {
                (None, None) => {}
                (Some(d), Some(v)) => {
                    assert_eq!(d.primitive, v.primitive);
                    assert_eq!(v.instance_id, 7);
                    assert!(approx(d.t, v.t, 1e-5), "t {} != {}", d.t, v.t);
                    assert!(approx(d.u, v.u, 1e-4));
                    assert!(approx(d.v, v.v, 1e-4));
                }
                (d, v) => panic!("existence mismatch: {d:?} vs {v:?}"),
            }
        }
    }

    #[test]
    fn transformed_hit_point_is_consistent_in_world_space() {
        // For any hit, world_point == object_to_world(object_point). Since t is
        // preserved, ray.at(t) in world must equal the transformed object hit.
        let blas = sample_blas();
        let blases = vec![blas];
        let mut rng = Rng::new(0xDEAD_BEEF);
        for _ in 0..200 {
            let m = random_affine(&mut rng);
            let inst = Instance::new(m, 0, 42).unwrap();
            let tlas = Tlas::build(&[inst], &blases);
            for _ in 0..40 {
                let origin = [
                    rng.range(-8.0, 8.0),
                    rng.range(-8.0, 8.0),
                    rng.range(-8.0, 8.0),
                ];
                let dir = [
                    rng.range(-1.0, 1.0),
                    rng.range(-1.0, 1.0),
                    rng.range(-1.0, 1.0),
                ];
                let ray = Ray::infinite(origin, dir);
                if let Some(hit) = tlas.closest_hit(&ray, &blases) {
                    let world_pt = ray.at(hit.t);
                    // Reconstruct object-space hit point and map it forward.
                    let obj_origin = inst.world_to_object().transform_point(origin);
                    let obj_dir = inst.world_to_object().transform_vector(dir);
                    let obj_pt = [
                        obj_origin[0] + hit.t * obj_dir[0],
                        obj_origin[1] + hit.t * obj_dir[1],
                        obj_origin[2] + hit.t * obj_dir[2],
                    ];
                    let mapped = inst.object_to_world().transform_point(obj_pt);
                    assert!(
                        approx_pt(world_pt, mapped, 2e-3),
                        "world {world_pt:?} != mapped {mapped:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn multi_instance_returns_global_nearest() {
        let blas = sample_blas();
        let blases = vec![blas];
        // Two z=0 quads translated to z=2 and z=5; a ray down -z from z=10 hits
        // the nearer (z=5) instance first.
        let near = Instance::new(Affine3::from_translation([0.0, 0.0, 5.0]), 0, 100).unwrap();
        let far = Instance::new(Affine3::from_translation([0.0, 0.0, 2.0]), 0, 200).unwrap();
        let tlas = Tlas::build(&[far, near], &blases);
        let ray = Ray::infinite([0.0, 0.0, 10.0], [0.0, 0.0, -1.0]);
        let hit = tlas.closest_hit(&ray, &blases).expect("hits nearer quad");
        assert_eq!(hit.instance_id, 100);
        assert!(approx(hit.t, 5.0, 1e-4), "t = {}", hit.t);
    }

    #[test]
    fn random_scene_matches_brute_force() {
        let blases = vec![sample_blas()];
        let mut rng = Rng::new(0xA5A5_1234);
        // Build a handful of random scenes, each cross-checked over many rays.
        for _ in 0..40 {
            let n = 1 + (rng.next_u32() % 12) as usize;
            let mut instances = Vec::with_capacity(n);
            for id in 0..n {
                let m = random_affine(&mut rng);
                instances.push(Instance::new(m, 0, id as u32).unwrap());
            }
            let tlas = Tlas::build(&instances, &blases);
            for _ in 0..300 {
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
                let ray = Ray::infinite(origin, dir);
                let brute = brute_closest(&instances, &blases, &ray);
                let via = tlas.closest_hit(&ray, &blases);
                match (brute, via) {
                    (None, None) => {}
                    (Some((bid, bprim, bt)), Some(v)) => {
                        assert_eq!(bid, v.instance_id, "instance mismatch");
                        assert_eq!(bprim, v.primitive, "primitive mismatch");
                        assert!(approx(bt, v.t, 1e-4), "t {bt} != {}", v.t);
                        // instance_index must resolve back to the same id.
                        assert_eq!(
                            tlas.instances()[v.instance_index as usize].instance_id(),
                            v.instance_id
                        );
                    }
                    (b, v) => panic!("existence mismatch: {b:?} vs {v:?}"),
                }
            }
        }
    }

    #[test]
    fn any_hit_agrees_with_closest_hit() {
        let blases = vec![sample_blas()];
        let mut rng = Rng::new(0x0BADF00D);
        for _ in 0..20 {
            let n = 1 + (rng.next_u32() % 8) as usize;
            let instances: Vec<Instance> = (0..n)
                .map(|id| Instance::new(random_affine(&mut rng), 0, id as u32).unwrap())
                .collect();
            let tlas = Tlas::build(&instances, &blases);
            for _ in 0..400 {
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
                let ray = Ray::infinite(origin, dir);
                assert_eq!(
                    tlas.any_hit(&ray, &blases),
                    tlas.closest_hit(&ray, &blases).is_some()
                );
            }
        }
    }

    #[test]
    fn empty_tlas_and_empty_blas_never_hit() {
        let blases: Vec<Bvh> = vec![Bvh::build(&[])];
        let empty = Tlas::build(&[], &blases);
        let ray = Ray::infinite([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
        assert!(empty.is_empty());
        assert!(empty.closest_hit(&ray, &blases).is_none());
        assert!(!empty.any_hit(&ray, &blases));

        // Instance of an empty BLAS: valid transform, but nothing to hit.
        let inst = Instance::new(Affine3::identity(), 0, 0).unwrap();
        let tlas = Tlas::build(&[inst], &blases);
        assert!(tlas.closest_hit(&ray, &blases).is_none());
        assert!(!tlas.any_hit(&ray, &blases));
    }

    #[test]
    fn scale_changes_hit_distance_predictably() {
        // A 2× uniform scale doubles the world-space distance to the same quad.
        let blases = vec![sample_blas()];
        let unit = Instance::new(Affine3::identity(), 0, 0).unwrap();
        let scaled = Instance::new(Affine3::from_scale([2.0, 2.0, 2.0]), 0, 0).unwrap();
        let tlas_unit = Tlas::build(&[unit], &blases);
        let tlas_scaled = Tlas::build(&[scaled], &blases);
        let ray = Ray::infinite([0.0, 0.0, 4.0], [0.0, 0.0, -1.0]);
        let h_unit = tlas_unit.closest_hit(&ray, &blases).unwrap();
        let h_scaled = tlas_scaled.closest_hit(&ray, &blases).unwrap();
        // Unit quad at z=0 -> t=4; scaled quad still at z=0 (scale about origin) -> t=4.
        assert!(approx(h_unit.t, 4.0, 1e-4));
        assert!(approx(h_scaled.t, 4.0, 1e-4));
        // Offsetting the scaled instance moves the plane to z=0 as well, but a
        // ray starting closer confirms t tracks world distance.
        let ray2 = Ray::infinite([0.0, 0.0, 4.0], [0.0, 0.0, -2.0]);
        let h2 = tlas_unit.closest_hit(&ray2, &blases).unwrap();
        assert!(approx(h2.t, 2.0, 1e-4), "t = {}", h2.t);
    }

    #[test]
    fn refit_after_instance_motion_matches_rebuild_and_preserves_topology() {
        let blases = vec![sample_blas()];
        let mut rng = Rng::new(0x5EED_0F17);
        for _ in 0..30 {
            let n = 1 + (rng.next_u32() % 12) as usize;
            // Stable ids 0..n so `updated(id)` can index a transform table.
            let start: Vec<Instance> = (0..n)
                .map(|id| Instance::new(random_affine(&mut rng), 0, id as u32).unwrap())
                .collect();
            let mut tlas = Tlas::build(&start, &blases);
            let nodes_before = tlas.node_count();
            let instances_before = tlas.instances().len();

            // New transforms per id; rebuild an independent reference TLAS with
            // them, then refit the existing one to the same target.
            let moved_transforms: Vec<Affine3> =
                (0..n).map(|_| random_affine(&mut rng)).collect();
            let moved_instances: Vec<Instance> = (0..n)
                .map(|id| Instance::new(moved_transforms[id], 0, id as u32).unwrap())
                .collect();
            let rebuilt = Tlas::build(&moved_instances, &blases);

            tlas.refit(|id| moved_transforms[id as usize], &blases);

            // Topology is preserved: same node and instance counts.
            assert_eq!(tlas.node_count(), nodes_before);
            assert_eq!(tlas.instances().len(), instances_before);

            for _ in 0..300 {
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
                let ray = Ray::infinite(origin, dir);
                let refit_hit = tlas.closest_hit(&ray, &blases);
                let ref_hit = rebuilt.closest_hit(&ray, &blases);
                match (refit_hit, ref_hit) {
                    (None, None) => {}
                    (Some(a), Some(b)) => {
                        assert_eq!(a.instance_id, b.instance_id, "instance mismatch");
                        assert_eq!(a.primitive, b.primitive, "primitive mismatch");
                        assert!(approx(a.t, b.t, 1e-4), "t {} != {}", a.t, b.t);
                    }
                    (a, b) => panic!("existence mismatch: {a:?} vs {b:?}"),
                }
            }
        }
    }

    #[test]
    fn rebuilt_matches_traversal_and_preserves_instance_set() {
        // `rebuilt` reclaims a compact TLAS from the current instance transforms.
        // It must trace identically to the live TLAS (same geometry) and keep
        // every instance (ids form the full set), though node ordering may differ.
        let blases = vec![sample_blas()];
        let mut rng = Rng::new(0xC0FF_EE42);
        for _ in 0..20 {
            let n = 1 + (rng.next_u32() % 10) as usize;
            let start: Vec<Instance> = (0..n)
                .map(|id| Instance::new(random_affine(&mut rng), 0, id as u32).unwrap())
                .collect();
            let mut tlas = Tlas::build(&start, &blases);

            // Move instances, refit, then rebuild from the refit state.
            let moved: Vec<Affine3> = (0..n).map(|_| random_affine(&mut rng)).collect();
            tlas.refit(|id| moved[id as usize], &blases);
            let rebuilt = tlas.rebuilt(&blases);

            assert_eq!(rebuilt.instances().len(), n);
            let mut ids: Vec<u32> =
                rebuilt.instances().iter().map(Instance::instance_id).collect();
            ids.sort_unstable();
            let expected: Vec<u32> = (0..n as u32).collect();
            assert_eq!(ids, expected, "rebuilt must keep every instance once");

            for _ in 0..200 {
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
                let ray = Ray::infinite(origin, dir);
                match (tlas.closest_hit(&ray, &blases), rebuilt.closest_hit(&ray, &blases)) {
                    (None, None) => {}
                    (Some(a), Some(b)) => {
                        assert_eq!(a.instance_id, b.instance_id, "instance mismatch");
                        assert_eq!(a.primitive, b.primitive, "primitive mismatch");
                        assert!(approx(a.t, b.t, 1e-4), "t {} != {}", a.t, b.t);
                    }
                    (a, b) => panic!("existence mismatch: {a:?} vs {b:?}"),
                }
            }
        }
    }

    #[test]
    fn refit_keeps_previous_transform_for_singular_update() {
        // A singular target transform must not corrupt the instance: the refit
        // keeps the prior (invertible) transform, so hits are unchanged.
        let blases = vec![sample_blas()];
        let inst = Instance::new(Affine3::from_translation([0.0, 0.0, 5.0]), 0, 0).unwrap();
        let mut tlas = Tlas::build(&[inst], &blases);
        let ray = Ray::infinite([0.0, 0.0, 10.0], [0.0, 0.0, -1.0]);
        let before = tlas.closest_hit(&ray, &blases).expect("hits the quad");

        // Flatten the transform (non-invertible) -> instance keeps its old one.
        tlas.refit(|_| Affine3::from_scale([1.0, 0.0, 1.0]), &blases);
        let after = tlas.closest_hit(&ray, &blases).expect("still hits the quad");
        assert_eq!(after.instance_id, before.instance_id);
        assert_eq!(after.primitive, before.primitive);
        assert!(approx(after.t, before.t, 1e-5), "t {} != {}", after.t, before.t);
    }

    /// Brute-force watertight reference: query every instance's BLAS with the
    /// watertight walk and keep the globally nearest hit. Mirror of
    /// [`brute_closest`] but exercising the leak-free triangle test.
    fn brute_closest_watertight(
        instances: &[Instance],
        blases: &[Bvh],
        ray: &Ray,
    ) -> Option<(u32, u32, f32)> {
        let mut best: Option<(u32, u32, f32)> = None;
        let mut best_t = ray.t_max();
        for inst in instances {
            let obj_origin = inst.world_to_object().transform_point(ray.origin());
            let obj_dir = inst.world_to_object().transform_vector(ray.direction());
            let obj_ray = Ray::new(obj_origin, obj_dir, ray.t_min(), ray.t_max());
            if let Some(hit) = blases[inst.blas()].closest_hit_watertight(&obj_ray)
                && hit.t < best_t
            {
                best_t = hit.t;
                best = Some((inst.instance_id(), hit.primitive, hit.t));
            }
        }
        best
    }

    #[test]
    fn watertight_identity_instance_matches_blas_directly() {
        // Under an identity transform the TLAS watertight walk must agree with
        // the BLAS watertight walk ray-for-ray (same t/primitive), confirming the
        // top-level traversal does not perturb the per-instance result.
        let blas = sample_blas();
        let blases = vec![blas.clone()];
        let inst = Instance::new(Affine3::identity(), 0, 7).unwrap();
        let tlas = Tlas::build(&[inst], &blases);

        let mut rng = Rng::new(0x1DEF_2266);
        for _ in 0..5000 {
            let origin = [rng.range(-3.0, 3.0), rng.range(-3.0, 3.0), rng.range(2.0, 6.0)];
            let dir = [rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(-2.0, -0.2)];
            let ray = Ray::infinite(origin, dir);
            let direct = blas.closest_hit_watertight(&ray);
            let via = tlas.closest_hit_watertight(&ray, &blases);
            match (direct, via) {
                (None, None) => {}
                (Some(d), Some(v)) => {
                    assert_eq!(d.primitive, v.primitive, "primitive mismatch");
                    assert!(approx(d.t, v.t, 1e-5), "t {} != {}", d.t, v.t);
                    assert_eq!(v.instance_id, 7);
                }
                (a, b) => panic!(
                    "existence mismatch: {:?} vs {:?}",
                    a.map(|h| h.t),
                    b.map(|h| h.t)
                ),
            }
        }
    }

    #[test]
    fn watertight_random_scene_matches_brute_force() {
        // The accelerated watertight TLAS walk must return exactly the globally
        // nearest watertight hit that a brute-force per-instance scan finds.
        let blases = vec![sample_blas()];
        let mut rng = Rng::new(0xC0FF_EE42);
        for _ in 0..40 {
            let n = 1 + (rng.next_u32() % 12) as usize;
            let mut instances = Vec::with_capacity(n);
            for id in 0..n {
                instances.push(Instance::new(random_affine(&mut rng), 0, id as u32).unwrap());
            }
            let tlas = Tlas::build(&instances, &blases);
            for _ in 0..300 {
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
                let ray = Ray::infinite(origin, dir);
                let brute = brute_closest_watertight(&instances, &blases, &ray);
                let via = tlas.closest_hit_watertight(&ray, &blases);
                match (brute, via) {
                    (None, None) => {}
                    (Some((bid, bprim, bt)), Some(v)) => {
                        assert_eq!(bid, v.instance_id, "instance mismatch");
                        assert_eq!(bprim, v.primitive, "primitive mismatch");
                        assert!(approx(bt, v.t, 1e-4), "t {bt} != {}", v.t);
                        assert_eq!(
                            tlas.instances()[v.instance_index as usize].instance_id(),
                            v.instance_id
                        );
                    }
                    (b, v) => panic!("existence mismatch: {b:?} vs {v:?}"),
                }
            }
        }
    }

    #[test]
    fn watertight_any_hit_agrees_with_closest_hit() {
        // Occlusion via the watertight walk must exactly match the existence of a
        // watertight nearest hit for the same ray.
        let blases = vec![sample_blas()];
        let mut rng = Rng::new(0x5EA1_ED00);
        for _ in 0..20 {
            let n = 1 + (rng.next_u32() % 8) as usize;
            let instances: Vec<Instance> = (0..n)
                .map(|id| Instance::new(random_affine(&mut rng), 0, id as u32).unwrap())
                .collect();
            let tlas = Tlas::build(&instances, &blases);
            for _ in 0..400 {
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
                let ray = Ray::infinite(origin, dir);
                assert_eq!(
                    tlas.any_hit_watertight(&ray, &blases),
                    tlas.closest_hit_watertight(&ray, &blases).is_some()
                );
            }
        }
    }

    #[test]
    fn watertight_instanced_seam_has_no_leaks() {
        // A ray marched across the shared diagonal of an instanced quad must
        // always report a hit through the watertight TLAS walk. The instance is
        // rotated so the seam is not axis-aligned, matching the standalone and
        // BLAS-level leak proofs but now exercised through the full TLAS path.
        let blases = vec![sample_blas()];
        // Rotate about z (in-plane) so the shared diagonal is a generic line
        // rather than axis-aligned. z-rotation keeps the quad in the z = 0
        // plane, so a -z ray still sees it. Exact 3-4-5 rotation (c^2+s^2=1)
        // avoids transcendental calls. Column-major z-rotation matrix.
        let (c, s) = (0.6_f32, 0.8_f32);
        let rot = Affine3::from_cols([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]], [0.0, 0.0, 0.0]);
        let inst = Instance::new(rot, 0, 0).unwrap();
        let tlas = Tlas::build(&[inst], &blases);

        // The BLAS seam is the diagonal from [-1,-1,0] to [1,1,0]; its world
        // image under `rot` is R*[t,t,0] = [t(c-s), t(c+s), 0]. March a -z ray
        // straight down that world line: every step must strike one of the two
        // triangles sharing the seam, so a watertight walk never leaks.
        let steps = 20_000u32;
        let mut leaks = 0u32;
        for i in 0..steps {
            let t = -0.95 + 1.9 * (i as f32 / steps as f32);
            let x = t * (c - s);
            let y = t * (c + s);
            let ray = Ray::infinite([x, y, 5.0], [0.0, 0.0, -1.0]);
            if tlas.closest_hit_watertight(&ray, &blases).is_none() {
                leaks += 1;
            }
        }
        assert_eq!(leaks, 0, "watertight TLAS leaked {leaks}/{steps} along seam");
    }

    fn vdot(a: [f32; 3], b: [f32; 3]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    fn vnorm(v: [f32; 3]) -> [f32; 3] {
        let inv = 1.0 / vdot(v, v).sqrt();
        [v[0] * inv, v[1] * inv, v[2] * inv]
    }

    #[test]
    fn normal_transform_matches_direction_under_pure_rotation() {
        // A rotation is orthonormal, so (L^-1)^T == L and the normal transform
        // coincides with the plain direction transform (up to normalisation).
        let rot = Affine3::from_cols(
            [[0.6, 0.8, 0.0], [-0.8, 0.6, 0.0], [0.0, 0.0, 1.0]],
            [0.0, 0.0, 0.0],
        );
        let inst = Instance::new(rot, 0, 0).unwrap();
        let n = vnorm([0.3, -0.5, 0.8]);
        let by_normal = inst.transform_normal_to_world(n);
        let by_dir = vnorm(rot.transform_vector(n));
        for k in 0..3 {
            assert!((by_normal[k] - by_dir[k]).abs() <= 1.0e-6, "axis {k}");
        }
    }

    #[test]
    fn normal_stays_perpendicular_to_surface_under_nonuniform_scale() {
        // Non-uniform scale is where transforming a normal by L breaks: the
        // correct (L^-1)^T keeps the normal orthogonal to every surface tangent,
        // while the naive L does not.
        let scale = Affine3::from_scale([2.0, 1.0, 1.0]);
        let inst = Instance::new(scale, 0, 0).unwrap();

        let n = vnorm([1.0, 1.0, 0.0]);
        // Two independent object-space tangents of the surface (both ⟂ n).
        let t1 = vnorm([1.0, -1.0, 0.0]);
        let t2 = [0.0, 0.0, 1.0];

        let world_n = inst.transform_normal_to_world(n);
        let world_t1 = scale.transform_vector(t1);
        let world_t2 = scale.transform_vector(t2);

        // Correct transform: normal remains orthogonal to the mapped surface.
        assert!(vdot(world_n, world_t1).abs() <= 1.0e-6);
        assert!(vdot(world_n, world_t2).abs() <= 1.0e-6);
        // The result is a unit vector.
        assert!((vdot(world_n, world_n) - 1.0).abs() <= 1.0e-6);

        // Naive direction transform of the normal is *not* perpendicular here —
        // this is exactly the bug the inverse-transpose fixes.
        let naive = vnorm(scale.transform_vector(n));
        assert!(vdot(naive, world_t1).abs() > 0.1);
    }

    #[test]
    fn degenerate_normal_maps_to_zero() {
        let inst = Instance::new(Affine3::from_scale([2.0, 3.0, 4.0]), 0, 0).unwrap();
        assert_eq!(inst.transform_normal_to_world([0.0, 0.0, 0.0]), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn transpose_transform_vector_is_the_matrix_transpose() {
        // (L^T v) . e_i == v . (L e_i): checking against every basis column.
        let m = Affine3::from_cols(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]],
            [0.0, 0.0, 0.0],
        );
        let v = [1.3, -2.1, 0.7];
        let tv = m.transpose_transform_vector(v);
        let cols = m.columns();
        for i in 0..3 {
            assert!((tv[i] - vdot(cols[i], v)).abs() <= 1.0e-6, "row {i}");
        }
    }
    #[test]
    fn empty_tlas_has_zero_sah_cost_and_unit_refit_quality() {
        let blases = vec![sample_blas()];
        let tlas = Tlas::build(&[], &blases);
        assert!(tlas.is_empty());
        assert_eq!(tlas.sah_cost(0.125), 0.0);
        // Both current and rebuilt cost are 0.0, so quality is the neutral 1.0.
        assert_eq!(tlas.refit_quality(&blases, 0.125), 1.0);
    }

    #[test]
    fn single_instance_tlas_scores_its_instance_count() {
        // One instance -> one leaf whose box equals the root box, so the score is
        // just that leaf's primitive count (one instance test) = 1.0, with no
        // interior traversal contributing.
        let blases = vec![sample_blas()];
        let inst = Instance::new(Affine3::identity(), 0, 0).unwrap();
        let tlas = Tlas::build(&[inst], &blases);
        assert!((tlas.sah_cost(0.125) - 1.0).abs() <= 1.0e-9);
        assert!((tlas.refit_quality(&blases, 0.125) - 1.0).abs() <= 1.0e-6);
    }

    #[test]
    fn freshly_built_tlas_has_unit_refit_quality() {
        // A tree scored against a rebuild of its own current instances is ideal.
        let blases = vec![sample_blas()];
        let mut rng = Rng::new(0xA11A_5EED);
        for _ in 0..20 {
            let n = 1 + (rng.next_u32() % 12) as usize;
            let instances: Vec<Instance> = (0..n)
                .map(|id| Instance::new(random_affine(&mut rng), 0, id as u32).unwrap())
                .collect();
            let tlas = Tlas::build(&instances, &blases);
            let q = tlas.refit_quality(&blases, 0.125);
            assert!((q - 1.0).abs() <= 1.0e-6, "fresh quality {q} != 1.0");
        }
    }

    #[test]
    fn scrambling_instances_degrades_refit_quality() {
        // Lay instances out on a line so the SAH build partitions them cleanly
        // along X, then *reverse* their positions and refit. Refitting keeps the
        // original (now wrong) split planes, so each leaf must enclose instances
        // that ended up far apart -> the top-level boxes balloon and the SAH cost
        // rises above a fresh rebuild of the same scrambled positions.
        let blases = vec![sample_blas()];
        let n = 9usize;
        let spacing = 8.0f32;
        let start: Vec<Instance> = (0..n)
            .map(|id| {
                let t = Affine3::from_translation([id as f32 * spacing, 0.0, 0.0]);
                Instance::new(t, 0, id as u32).unwrap()
            })
            .collect();
        let mut tlas = Tlas::build(&start, &blases);
        assert!(
            (tlas.refit_quality(&blases, 0.125) - 1.0).abs() <= 1.0e-6,
            "line layout should build ideally"
        );

        // Interleave the layout so instances the build grouped into the same
        // contiguous leaf are flung to opposite ends of the line. Mapping id to
        // `(id % 4) * n + (id / 4)` spreads each block-of-four leaf across the
        // whole extent, so the refit's preserved leaf boxes must span nearly the
        // entire scene while a rebuild would regroup by the new proximity.
        let moved: Vec<Affine3> = (0..n)
            .map(|id| {
                let slot = (id % 4) * n + (id / 4);
                Affine3::from_translation([slot as f32 * spacing, 0.0, 0.0])
            })
            .collect();
        tlas.refit(|id| moved[id as usize], &blases);

        let degraded = tlas.refit_quality(&blases, 0.125);
        assert!(
            degraded > 1.0 + 1.0e-3,
            "scrambled refit should degrade quality, got {degraded}"
        );

        // A rebuild over the scrambled positions restores an ideal tree.
        let rebuilt = tlas.rebuilt(&blases);
        assert!(
            (rebuilt.refit_quality(&blases, 0.125) - 1.0).abs() <= 1.0e-6,
            "rebuild should recover unit quality"
        );
        assert!(
            rebuilt.sah_cost(0.125) < tlas.sah_cost(0.125),
            "rebuilt tree must be cheaper than the degraded refit"
        );
    }

    #[test]
    fn tlas_sah_cost_and_refit_quality_are_deterministic() {
        let blases = vec![sample_blas()];
        let mut rng = Rng::new(0xD37E_2222);
        let n = 8usize;
        let instances: Vec<Instance> = (0..n)
            .map(|id| Instance::new(random_affine(&mut rng), 0, id as u32).unwrap())
            .collect();
        let tlas = Tlas::build(&instances, &blases);
        assert_eq!(tlas.sah_cost(0.125), tlas.sah_cost(0.125));
        assert_eq!(
            tlas.refit_quality(&blases, 0.125),
            tlas.refit_quality(&blases, 0.125)
        );
    }

    /// A ray whose inclusion mask shares no bit with an instance's mask must
    /// treat that instance as absent: no closest hit and no occlusion.
    #[test]
    fn masked_out_instance_never_hits() {
        let blases = vec![sample_blas()];
        // Instance visible only on bit 0; ray includes only bit 1 -> disjoint.
        let inst = Instance::with_mask(Affine3::identity(), 0, 42, 0b0000_0001).unwrap();
        let tlas = Tlas::build(&[inst], &blases);
        // Straight down -z at the z=0 quad, which an all-mask ray would hit at t=5.
        let ray = Ray::new([0.0, 0.0, 5.0], [0.0, 0.0, -1.0], 1.0e-4, 100.0);
        assert!(tlas.closest_hit(&ray, &blases).is_some(), "sanity: geometry is there");
        assert!(tlas.closest_hit_masked(&ray, &blases, 0b0000_0010).is_none());
        assert!(!tlas.any_hit_masked(&ray, &blases, 0b0000_0010));
        assert!(tlas.closest_hit_watertight_masked(&ray, &blases, 0b0000_0010).is_none());
        assert!(!tlas.any_hit_watertight_masked(&ray, &blases, 0b0000_0010));
    }

    /// `Instance::new` yields `MASK_ALL`, and tracing with `MASK_ALL` reproduces
    /// the unmasked traversal on every one of the four query variants.
    #[test]
    fn mask_all_matches_unmasked() {
        let blases = vec![sample_blas()];
        let mut rng = Rng::new(0x51DE_ABCD);
        let n = 12usize;
        let instances: Vec<Instance> = (0..n)
            .map(|id| Instance::new(random_affine(&mut rng), 0, id as u32).unwrap())
            .collect();
        assert!(instances.iter().all(|i| i.mask() == Instance::MASK_ALL));
        let tlas = Tlas::build(&instances, &blases);
        for _ in 0..500 {
            let origin = [rng.range(-8.0, 8.0), rng.range(-8.0, 8.0), rng.range(-8.0, 8.0)];
            let dir = [rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(-1.0, 1.0)];
            let ray = Ray::new(origin, dir, 1.0e-4, 50.0);

            let a = tlas.closest_hit(&ray, &blases);
            let b = tlas.closest_hit_masked(&ray, &blases, Instance::MASK_ALL);
            match (a, b) {
                (None, None) => {}
                (Some(x), Some(y)) => {
                    assert_eq!(x.instance_id, y.instance_id);
                    assert_eq!(x.primitive, y.primitive);
                    assert_eq!(x.t.to_bits(), y.t.to_bits());
                }
                _ => panic!("closest_hit vs mask_all disagree"),
            }
            assert_eq!(
                tlas.any_hit(&ray, &blases),
                tlas.any_hit_masked(&ray, &blases, Instance::MASK_ALL)
            );
            let cw = tlas.closest_hit_watertight(&ray, &blases);
            let cwm = tlas.closest_hit_watertight_masked(&ray, &blases, Instance::MASK_ALL);
            assert_eq!(cw.map(|h| h.instance_id), cwm.map(|h| h.instance_id));
            assert_eq!(
                tlas.any_hit_watertight(&ray, &blases),
                tlas.any_hit_watertight_masked(&ray, &blases, Instance::MASK_ALL)
            );
        }
    }

    /// Any shared bit between the instance mask and ray mask admits the instance.
    #[test]
    fn partial_bit_overlap_hits() {
        let blases = vec![sample_blas()];
        // Instance visible on bits 1 and 2; ray includes bit 2 only -> overlap.
        let inst = Instance::with_mask(Affine3::identity(), 0, 7, 0b0000_0110).unwrap();
        let tlas = Tlas::build(&[inst], &blases);
        let ray = Ray::new([0.0, 0.0, 5.0], [0.0, 0.0, -1.0], 1.0e-4, 100.0);
        let hit = tlas.closest_hit_masked(&ray, &blases, 0b0000_0100).expect("overlap hits");
        assert_eq!(hit.instance_id, 7);
        assert!(tlas.any_hit_masked(&ray, &blases, 0b0000_0100));
    }

    /// A zero inclusion mask can never satisfy the bitwise-AND predicate, so it
    /// matches nothing regardless of geometry, on all four query variants.
    #[test]
    fn zero_ray_mask_matches_nothing() {
        let blases = vec![sample_blas()];
        let inst = Instance::new(Affine3::identity(), 0, 1).unwrap();
        let tlas = Tlas::build(&[inst], &blases);
        let ray = Ray::new([0.0, 0.0, 5.0], [0.0, 0.0, -1.0], 1.0e-4, 100.0);
        assert!(tlas.closest_hit_masked(&ray, &blases, 0).is_none());
        assert!(!tlas.any_hit_masked(&ray, &blases, 0));
        assert!(tlas.closest_hit_watertight_masked(&ray, &blases, 0).is_none());
        assert!(!tlas.any_hit_watertight_masked(&ray, &blases, 0));
    }

    /// Selective visibility: two stacked instances carry different category
    /// bits, and a ray selects which one it can see. The nearer instance is
    /// excluded from the "shadow" category, so a shadow-masked ray skips it and
    /// reports the farther instance instead — the primitive behind per-ray
    /// shadow/reflection instance selection.
    #[test]
    fn selective_category_masks_pick_different_instances() {
        let blases = vec![sample_blas()];
        // Near quad at z=3, reflection-only (bit 0). Far quad at z=1, shadow-only (bit 1).
        let near = Instance::with_mask(
            Affine3::from_translation([0.0, 0.0, 3.0]),
            0,
            100,
            0b0000_0001,
        )
        .unwrap();
        let far = Instance::with_mask(
            Affine3::from_translation([0.0, 0.0, 1.0]),
            0,
            200,
            0b0000_0010,
        )
        .unwrap();
        let tlas = Tlas::build(&[near, far], &blases);
        let ray = Ray::new([0.0, 0.0, 5.0], [0.0, 0.0, -1.0], 1.0e-4, 100.0);

        // Reflection ray (bit 0) sees only the near quad.
        let refl = tlas.closest_hit_masked(&ray, &blases, 0b0000_0001).unwrap();
        assert_eq!(refl.instance_id, 100);
        // Shadow ray (bit 1) skips the near quad and lands on the far one.
        let shadow = tlas.closest_hit_masked(&ray, &blases, 0b0000_0010).unwrap();
        assert_eq!(shadow.instance_id, 200);
        // A ray including both bits still returns the globally nearest.
        let both = tlas.closest_hit_masked(&ray, &blases, 0b0000_0011).unwrap();
        assert_eq!(both.instance_id, 100);
    }

    /// Masked traversal is deterministic across repeated identical queries.
    #[test]
    fn masked_traversal_is_deterministic() {
        let blases = vec![sample_blas()];
        let mut rng = Rng::new(0x0FF1_CE55);
        let instances: Vec<Instance> = (0..10)
            .map(|id| {
                let mask = 1u8 << (id % 8);
                Instance::with_mask(random_affine(&mut rng), 0, id as u32, mask).unwrap()
            })
            .collect();
        let tlas = Tlas::build(&instances, &blases);
        let ray = Ray::new([0.5, 0.5, 6.0], [0.0, 0.0, -1.0], 1.0e-4, 50.0);
        for mask in 0u8..=0xFF {
            let a = tlas.closest_hit_masked(&ray, &blases, mask);
            let b = tlas.closest_hit_masked(&ray, &blases, mask);
            assert_eq!(a.map(|h| (h.instance_id, h.primitive, h.t.to_bits())),
                       b.map(|h| (h.instance_id, h.primitive, h.t.to_bits())));
        }
    }
}

