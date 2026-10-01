//! Analytic pbrt-style bilinear patch carrying per-corner **shading normals**
//! and **texture `UV`s**, with a single-level `BVH` that serves as the `CPU`
//! golden reference for the smooth-shaded quad-patch path.
//!
//! The geometric [`super::bilinear_patch::BilinearPatch`] reports only the flat
//! surface normal `∂P/∂u × ∂P/∂v`, which is correct for visibility but gives
//! faceted shading and no surface parameterization beyond the intrinsic patch
//! `(u, v)`. Production bilinear-patch meshes (pbrt's `BilinearPatchMesh`,
//! subdivision cages, displaced quads, hair/cloth cards) store a shading normal
//! and a texture coordinate at every corner and interpolate them across the
//! patch so curved surfaces read smooth and textures map correctly. This
//! primitive is that smooth-shaded patch: it keeps the four corner positions,
//! four corner shading normals, and four corner `UV`s, intersects with the
//! Reshetov "Cool Patches" analytic solve (identical to the geometric patch, so
//! the hit `t`/`(u, v)` bits match), and additionally returns the bilinearly
//! interpolated shading normal (re-normalized and flipped into the geometric
//! hemisphere, exactly as pbrt does) and the interpolated `UV`.
//!
//! Corner attributes are stored **verbatim** (not re-normalized at
//! construction) so a flat `GPU` layout decodes to a bit-identical primitive
//! and the packed traversal reproduces every hit's bits exactly. The ray
//! direction is never assumed unit: `t` comes out in `direction`-length units,
//! matching [`Ray::at`].

use super::bvh::{build_linear_bvh, Aabb, BvhBuildConfig, LinearBvhNode};
use super::traversal::Ray;

/// A bilinear patch with per-corner shading normals and texture `UV`s.
///
/// Corners follow the `(u, v)` convention `p00 = P(0, 0)`, `p10 = P(1, 0)`,
/// `p11 = P(1, 1)`, `p01 = P(0, 1)` (indices `0, 1, 2, 3`), identical to
/// [`super::bilinear_patch::BilinearPatch`]: `p00→p10` is the `v = 0` edge and
/// `p00→p01` is the `u = 0` edge. The matching corner normal and `UV` share
/// each index. `primitive` is the caller's stable id, reported unchanged on
/// every hit; the [`ShadedBilinearPatchBvh`] builder reorders primitives
/// internally but always reports hits by this id.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadedBilinearPatch {
    /// Corner positions `[p00, p10, p11, p01]`.
    positions: [[f32; 3]; 4],
    /// Per-corner shading normals `[n00, n10, n11, n01]`, stored verbatim (not
    /// re-normalized) so the `GPU` decode is bit-identical; interpolation
    /// re-normalizes the blended result at the hit.
    normals: [[f32; 3]; 4],
    /// Per-corner texture coordinates `[uv00, uv10, uv11, uv01]`, stored
    /// verbatim and bilinearly interpolated at the hit.
    uvs: [[f32; 2]; 4],
    /// Caller's stable primitive id, reported unchanged on every hit.
    primitive: u32,
}

impl ShadedBilinearPatch {
    /// Builds a shaded bilinear patch from corner `positions`, matching
    /// per-corner shading `normals`, per-corner `uvs`, and stable id
    /// `primitive`.
    ///
    /// All corner arrays use the `(u, v)` index order `0 = (0, 0)`,
    /// `1 = (1, 0)`, `2 = (1, 1)`, `3 = (0, 1)`. Normals are kept verbatim and
    /// need not be unit length on input because the interpolated normal is
    /// re-normalized at each hit.
    #[must_use]
    pub fn new(
        positions: [[f32; 3]; 4],
        normals: [[f32; 3]; 4],
        uvs: [[f32; 2]; 4],
        primitive: u32,
    ) -> Self {
        Self {
            positions,
            normals,
            uvs,
            primitive,
        }
    }

    /// Corner positions `[p00, p10, p11, p01]`.
    #[must_use]
    pub fn positions(&self) -> [[f32; 3]; 4] {
        self.positions
    }

    /// Per-corner shading normals `[n00, n10, n11, n01]` (verbatim).
    #[must_use]
    pub fn normals(&self) -> [[f32; 3]; 4] {
        self.normals
    }

    /// Per-corner texture coordinates `[uv00, uv10, uv11, uv01]` (verbatim).
    #[must_use]
    pub fn uvs(&self) -> [[f32; 2]; 4] {
        self.uvs
    }

    /// Caller's stable primitive id.
    #[must_use]
    pub fn primitive(&self) -> u32 {
        self.primitive
    }

    /// Surface point `P(u, v)` for parameters `u, v ∈ [0, 1]`.
    #[must_use]
    pub fn point(&self, u: f32, v: f32) -> [f32; 3] {
        let [p00, p10, p11, p01] = self.positions;
        let bottom = mix3(p00, p10, u);
        let top = mix3(p01, p11, u);
        mix3(bottom, top, v)
    }

    /// Axis-aligned bounds of the four corners.
    ///
    /// A bilinear patch is contained in the convex hull of its corners, so the
    /// corner `AABB` is a tight, correct bound (and the procedural-primitive
    /// `AABB` the hardware `BLAS` would store).
    #[must_use]
    pub fn aabb(&self) -> Aabb {
        let [p00, p10, p11, p01] = self.positions;
        let mut min = p00;
        let mut max = p00;
        for c in [p10, p11, p01] {
            for axis in 0..3 {
                min[axis] = min[axis].min(c[axis]);
                max[axis] = max[axis].max(c[axis]);
            }
        }
        Aabb::new(min, max)
    }

    /// Nearest ray/patch intersection inside the ray interval, or `None`.
    ///
    /// Reduces the ray to a quadratic in `u` (Reshetov 2019), back-substitutes
    /// each root in `[0, 1]` to recover `v` and `t`, keeps the nearest valid
    /// hit, then interpolates the corner attributes. The returned
    /// [`ShadedBilinearPatchHit::geometric_normal`] is the unit `∂P/∂u × ∂P/∂v`
    /// oriented against the ray, [`ShadedBilinearPatchHit::shading_normal`] is
    /// the bilinearly blended corner normal forced into that hemisphere, and
    /// [`ShadedBilinearPatchHit::uv`] is the blended texture coordinate.
    #[must_use]
    pub fn intersect(&self, ray: &Ray) -> Option<ShadedBilinearPatchHit> {
        let [p00, p10, p11, p01] = self.positions;
        let ro = ray.origin();
        let rd = ray.direction();

        // Translation-invariant edge vectors (quadratic + analytic normal).
        let e10 = sub(p10, p00); // ∂/∂u along the v = 0 edge.
        let e11 = sub(p11, p10); // v = 1 - side u edge direction term.
        let e00 = sub(p01, p00); // ∂/∂v along the u = 0 edge.
        let qn = cross(e10, sub(p01, p11));

        // Corners relative to the ray origin.
        let q00 = sub(p00, ro);
        let q10 = sub(p10, ro);

        // Quadratic a·u² + b·u + c = 0 (Reshetov's formulation).
        let a = dot(cross(q00, rd), e00);
        let c = dot(qn, rd);
        let b = dot(cross(q10, rd), e11) - a - c;

        let (u1, u2) = {
            let det = b * b - 4.0 * a * c;
            if det < 0.0 {
                return None;
            }
            let sq = det.sqrt();
            if c == 0.0 {
                // Degenerate (planar in u): linear equation b·u + a = 0.
                if b == 0.0 {
                    return None;
                }
                (-a / b, -1.0)
            } else {
                // Numerically stable roots: the large root uses a same-sign
                // addition, the small one via the product a/c.
                let big = (-b - sq.copysign(b)) * 0.5;
                (big / c, a / big)
            }
        };

        let mut best: Option<ShadedBilinearPatchHit> = None;
        let mut t_hi = ray.t_max();
        for u in [u1, u2] {
            if !(0.0..=1.0).contains(&u) {
                continue;
            }
            if let Some((t, v)) = self.solve_v(ray, q00, q10, e00, e11, u, t_hi)
                && let Some((geometric_normal, front_face)) =
                    self.oriented_normal(e10, e00, e11, u, v, rd)
            {
                t_hi = t;
                let shading_normal = self.shading_normal(u, v, geometric_normal);
                let uv = self.uv(u, v);
                best = Some(ShadedBilinearPatchHit {
                    t,
                    primitive: self.primitive,
                    geometric_normal,
                    shading_normal,
                    front_face,
                    u,
                    v,
                    uv,
                });
            }
        }
        best
    }

    /// Back-substitutes a `u` root to recover `(t, v)` for the vertical line at
    /// that `u`, returning `None` when the ray misses that line, the hit is
    /// behind the current nearest `t_hi`, or `v` leaves `[0, 1]`.
    fn solve_v(
        &self,
        ray: &Ray,
        q00: [f32; 3],
        q10: [f32; 3],
        e00: [f32; 3],
        e11: [f32; 3],
        u: f32,
        t_hi: f32,
    ) -> Option<(f32, f32)> {
        let rd = ray.direction();
        // Bottom point (relative to origin) and the vertical edge direction at u.
        let pa = mix3(q00, q10, u);
        let pb = mix3(e00, e11, u);
        let n0 = cross(rd, pb);
        let det = dot(n0, n0);
        if det <= 0.0 {
            return None;
        }
        let m = cross(n0, pa);
        let t = dot(m, pb) / det;
        let v = dot(m, rd) / det;
        if !(0.0..=1.0).contains(&v) {
            return None;
        }
        if t < ray.t_min() || t > t_hi {
            return None;
        }
        Some((t, v))
    }

    /// Unit geometric surface normal at `(u, v)`, oriented against the ray.
    ///
    /// The geometric normal is `∂P/∂u × ∂P/∂v`; `front_face` records whether the
    /// ray approached that outward side before the normal is flipped to face the
    /// ray. Returns `None` for a degenerate (zero-area) tangent frame.
    fn oriented_normal(
        &self,
        e10: [f32; 3],
        e00: [f32; 3],
        e11: [f32; 3],
        u: f32,
        v: f32,
        rd: [f32; 3],
    ) -> Option<([f32; 3], bool)> {
        let [_p00, _p10, p11, p01] = self.positions;
        let f = sub(p11, p01);
        let dpdu = add(scale(e10, 1.0 - v), scale(f, v));
        let dpdv = add(scale(e00, 1.0 - u), scale(e11, u));
        let g = cross(dpdu, dpdv);
        let len2 = dot(g, g);
        if len2 <= 0.0 {
            return None;
        }
        let inv_len = 1.0 / len2.sqrt();
        let outward = scale(g, inv_len);
        let front_face = dot(rd, outward) < 0.0;
        let normal = if front_face {
            outward
        } else {
            negate(outward)
        };
        Some((normal, front_face))
    }

    /// Bilinearly interpolated unit shading normal at `(u, v)`, forced into the
    /// `geometric_normal`'s hemisphere (pbrt convention); falls back to the
    /// geometric normal when the blend collapses to zero length.
    fn shading_normal(&self, u: f32, v: f32, geometric_normal: [f32; 3]) -> [f32; 3] {
        let [n00, n10, n11, n01] = self.normals;
        let bottom = mix3(n00, n10, u);
        let top = mix3(n01, n11, u);
        let ns_raw = mix3(bottom, top, v);
        let ns_oriented = if dot(ns_raw, geometric_normal) < 0.0 {
            negate(ns_raw)
        } else {
            ns_raw
        };
        normalize_or(ns_oriented, geometric_normal)
    }

    /// Bilinearly interpolated texture coordinate at `(u, v)`.
    fn uv(&self, u: f32, v: f32) -> [f32; 2] {
        let [uv00, uv10, uv11, uv01] = self.uvs;
        let bottom = mix2(uv00, uv10, u);
        let top = mix2(uv01, uv11, u);
        mix2(bottom, top, v)
    }
}

/// A ray/[`ShadedBilinearPatch`] intersection result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadedBilinearPatchHit {
    /// Ray parameter at the intersection (distance in `direction` lengths).
    pub t: f32,
    /// Stable id of the patch that was hit.
    pub primitive: u32,
    /// Unit geometric normal `∂P/∂u × ∂P/∂v`, oriented toward the incoming ray.
    pub geometric_normal: [f32; 3],
    /// Unit interpolated shading normal, in the geometric normal's hemisphere.
    pub shading_normal: [f32; 3],
    /// `true` when the ray struck the outward-facing (geometric normal) side.
    pub front_face: bool,
    /// Patch `u` parameter of the hit, in `[0, 1]`.
    pub u: f32,
    /// Patch `v` parameter of the hit, in `[0, 1]`.
    pub v: f32,
    /// Bilinearly interpolated texture coordinate at the hit.
    pub uv: [f32; 2],
}

/// A single-level `BVH` over [`ShadedBilinearPatch`] primitives.
///
/// Empty input yields an empty hierarchy ([`ShadedBilinearPatchBvh::is_empty`]);
/// traversal of an empty hierarchy never reports a hit. The layout and ordered
/// slab walk mirror [`super::bilinear_patch::BilinearPatchBvh`] and
/// [`super::shaded_triangle::ShadedTriangleBvh`] so every primitive kind shares
/// one acceleration-structure contract.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ShadedBilinearPatchBvh {
    /// Flattened `BVH` nodes; the root (when present) is index `0`.
    nodes: Vec<LinearBvhNode>,
    /// Patches reordered so each leaf owns a contiguous slice.
    patches: Vec<ShadedBilinearPatch>,
}

impl ShadedBilinearPatchBvh {
    /// Builds a `BVH` over `patches` with [`BvhBuildConfig::default`].
    #[must_use]
    pub fn build(patches: &[ShadedBilinearPatch]) -> Self {
        Self::build_with(patches, BvhBuildConfig::default())
    }

    /// Builds a `BVH` over `patches` with the given binned-`SAH` `config`.
    ///
    /// Each patch's [`ShadedBilinearPatch::aabb`] feeds the builder; the patches
    /// are then reordered by the returned primitive order so every leaf's
    /// `[first_primitive, first_primitive + primitive_count)` slice indexes
    /// directly into [`ShadedBilinearPatchBvh::patches`].
    #[must_use]
    pub fn build_with(patches: &[ShadedBilinearPatch], config: BvhBuildConfig) -> Self {
        let bounds: Vec<Aabb> = patches.iter().map(ShadedBilinearPatch::aabb).collect();
        let (nodes, order) = build_linear_bvh(&bounds, config);
        let patches = order.iter().map(|&i| patches[i as usize]).collect();
        Self { nodes, patches }
    }

    /// Number of flattened `BVH` nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of patches in the hierarchy.
    #[must_use]
    pub fn primitive_count(&self) -> usize {
        self.patches.len()
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

    /// Patches in leaf-contiguous order.
    #[must_use]
    pub fn patches(&self) -> &[ShadedBilinearPatch] {
        &self.patches
    }

    /// Nearest intersection along `ray`, or `None` if the ray hits nothing.
    ///
    /// Walks the flattened nodes with an explicit stack, visiting the child on
    /// the near side of the split axis first so the running `t_max` shrinks as
    /// fast as possible and far subtrees are culled by the slab test.
    #[must_use]
    pub fn closest_hit(&self, ray: &Ray) -> Option<ShadedBilinearPatchHit> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut ray = *ray;
        let mut best: Option<ShadedBilinearPatchHit> = None;

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
                    for patch in &self.patches[start..end] {
                        if let Some(hit) = patch.intersect(&ray) {
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

    /// True when *any* patch intersects `ray` inside its interval.
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
                    for patch in &self.patches[start..end] {
                        if patch.intersect(ray).is_some() {
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

/// Subtracts `b` from `a` componentwise.
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Adds `a` and `b` componentwise.
fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Scales `a` by scalar `s`.
fn scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Negates `a` componentwise.
fn negate(a: [f32; 3]) -> [f32; 3] {
    [-a[0], -a[1], -a[2]]
}

/// Euclidean dot product of two vectors.
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Cross product `a × b`.
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Linear interpolation `a + t·(b − a)` for a 3-vector.
fn mix3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + t * (b[0] - a[0]),
        a[1] + t * (b[1] - a[1]),
        a[2] + t * (b[2] - a[2]),
    ]
}

/// Linear interpolation `a + t·(b − a)` for a 2-vector.
fn mix2(a: [f32; 2], b: [f32; 2], t: f32) -> [f32; 2] {
    [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]
}

/// Returns the unit vector along `v`, or `fallback` when `v` is near zero.
fn normalize_or(v: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let len2 = dot(v, v);
    if len2 > 0.0 {
        scale(v, 1.0 / len2.sqrt())
    } else {
        fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal deterministic xorshift `RNG` for property-style tests.
    struct Rng(u64);

    impl Rng {
        /// Seeds the generator, forcing a non-zero state.
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }

        /// Advances the state and returns the high 32 bits.
        fn next_u32(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            (x >> 32) as u32
        }

        /// Uniform sample in `[0, 1)`.
        fn unit(&mut self) -> f32 {
            self.next_u32() as f32 / u32::MAX as f32
        }

        /// Uniform sample in `[lo, hi)`.
        fn range(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * self.unit()
        }
    }

    /// A planar unit quad in the `z = 0` plane with standard corner `UV`s and
    /// gently tilted corner shading normals.
    fn planar_patch() -> ShadedBilinearPatch {
        ShadedBilinearPatch::new(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            [
                [0.0, 0.0, 1.0],
                [0.2, 0.0, 1.0],
                [0.2, 0.2, 1.0],
                [0.0, 0.2, 1.0],
            ],
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            7,
        )
    }

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn front_hit_interpolates_normal_and_uv() {
        let patch = planar_patch();
        let ray = Ray::infinite([0.3, 0.4, 5.0], [0.0, 0.0, -1.0]);
        let hit = patch.intersect(&ray).expect("front ray should hit");

        assert_eq!(hit.primitive, 7);
        assert!(hit.front_face);
        assert!(approx(hit.t, 5.0, 1e-5), "t = {}", hit.t);
        assert!(approx(hit.u, 0.3, 1e-4) && approx(hit.v, 0.4, 1e-4));

        // Flat quad -> geometric normal is +z, oriented against the -z ray.
        assert!(approx(hit.geometric_normal[0], 0.0, 1e-5));
        assert!(approx(hit.geometric_normal[1], 0.0, 1e-5));
        assert!(approx(hit.geometric_normal[2], 1.0, 1e-5));

        // Bilinear blend of the corner normals at (0.3, 0.4) = [0.06, 0.08, 1].
        let raw = [0.06f32, 0.08, 1.0];
        let inv = 1.0 / (raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2]).sqrt();
        let want = [raw[0] * inv, raw[1] * inv, raw[2] * inv];
        for (got, expected) in hit.shading_normal.iter().zip(want.iter()) {
            assert!(approx(*got, *expected, 1e-4), "shading normal = {got}");
        }

        // Planar axis-aligned UVs track (u, v) exactly.
        assert!(approx(hit.uv[0], 0.3, 1e-4) && approx(hit.uv[1], 0.4, 1e-4));
    }

    #[test]
    fn back_hit_flips_both_normals() {
        let patch = planar_patch();
        let ray = Ray::infinite([0.3, 0.4, -5.0], [0.0, 0.0, 1.0]);
        let hit = patch.intersect(&ray).expect("back ray should hit");

        assert!(!hit.front_face);
        // Geometric normal flipped to face the +z-travelling ray.
        assert!(approx(hit.geometric_normal[2], -1.0, 1e-5));
        // Shading normal forced into the same (negative-z) hemisphere.
        assert!(hit.shading_normal[2] < 0.0);
        // Interpolated UV is independent of the viewing side.
        assert!(approx(hit.uv[0], 0.3, 1e-4) && approx(hit.uv[1], 0.4, 1e-4));
    }

    #[test]
    fn corner_sample_recovers_corner_attributes() {
        let patch = planar_patch();
        // Aim at the (u, v) = (1, 1) corner = p11 = [1, 1, 0], uv11 = [1, 1].
        let ray = Ray::infinite([1.0, 1.0, 3.0], [0.0, 0.0, -1.0]);
        let hit = patch.intersect(&ray).expect("corner ray should hit");
        assert!(approx(hit.u, 1.0, 1e-3) && approx(hit.v, 1.0, 1e-3));
        assert!(approx(hit.uv[0], 1.0, 1e-3) && approx(hit.uv[1], 1.0, 1e-3));

        // Shading normal at that corner = normalize(n11) = normalize([0.2,0.2,1]).
        let n = [0.2f32, 0.2, 1.0];
        let inv = 1.0 / (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        let want = [n[0] * inv, n[1] * inv, n[2] * inv];
        for (got, expected) in hit.shading_normal.iter().zip(want.iter()) {
            assert!(approx(*got, *expected, 2e-3));
        }
    }

    #[test]
    fn parallel_ray_misses() {
        let patch = planar_patch();
        // A ray travelling inside the z = 0 plane never crosses the patch.
        let ray = Ray::infinite([-1.0, 0.5, 0.0], [1.0, 0.0, 0.0]);
        assert!(patch.intersect(&ray).is_none());
    }

    #[test]
    fn shading_normal_is_unit_length() {
        let patch = planar_patch();
        let mut rng = Rng::new(0x51AD_E12A);
        for _ in 0..512 {
            let origin = [rng.range(0.1, 0.9), rng.range(0.1, 0.9), 4.0];
            let ray = Ray::infinite(origin, [0.0, 0.0, -1.0]);
            if let Some(hit) = patch.intersect(&ray) {
                let n = hit.shading_normal;
                let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                assert!(approx(len, 1.0, 1e-4), "shading normal not unit: {len}");
                let g = hit.geometric_normal;
                let glen = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
                assert!(approx(glen, 1.0, 1e-4), "geo normal not unit: {glen}");
            }
        }
    }

    #[test]
    fn empty_bvh_never_hits() {
        let bvh = ShadedBilinearPatchBvh::build(&[]);
        assert!(bvh.is_empty());
        assert_eq!(bvh.node_count(), 0);
        assert_eq!(bvh.primitive_count(), 0);
        let ray = Ray::infinite([0.0, 0.0, 1.0], [0.0, 0.0, -1.0]);
        assert!(bvh.closest_hit(&ray).is_none());
        assert!(!bvh.any_hit(&ray));
    }

    #[test]
    fn bvh_closest_hit_matches_brute_force() {
        let mut rng = Rng::new(0xB117_A14C);
        let mut patches = Vec::new();
        for i in 0..24u32 {
            let cx = rng.range(-5.0, 5.0);
            let cy = rng.range(-5.0, 5.0);
            let cz = rng.range(-5.0, 5.0);
            let s = rng.range(0.3, 1.0);
            // A slightly non-planar (saddle) quad centred at (cx, cy, cz).
            let positions = [
                [cx - s, cy - s, cz],
                [cx + s, cy - s, cz + rng.range(-0.3, 0.3)],
                [cx + s, cy + s, cz],
                [cx - s, cy + s, cz + rng.range(-0.3, 0.3)],
            ];
            let normals = [
                [rng.range(-0.3, 0.3), rng.range(-0.3, 0.3), 1.0],
                [rng.range(-0.3, 0.3), rng.range(-0.3, 0.3), 1.0],
                [rng.range(-0.3, 0.3), rng.range(-0.3, 0.3), 1.0],
                [rng.range(-0.3, 0.3), rng.range(-0.3, 0.3), 1.0],
            ];
            let uvs = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
            patches.push(ShadedBilinearPatch::new(positions, normals, uvs, i));
        }
        let bvh = ShadedBilinearPatchBvh::build(&patches);
        assert_eq!(bvh.primitive_count(), patches.len());

        let mut tested = 0u32;
        for _ in 0..4_000 {
            let origin = [
                rng.range(-8.0, 8.0),
                rng.range(-8.0, 8.0),
                rng.range(-8.0, 8.0),
            ];
            let target = [rng.range(-5.0, 5.0), rng.range(-5.0, 5.0), rng.range(-5.0, 5.0)];
            let dir = [
                target[0] - origin[0],
                target[1] - origin[1],
                target[2] - origin[2],
            ];
            if dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2] < 1e-6 {
                continue;
            }
            let ray = Ray::infinite(origin, dir);

            // Brute-force nearest over the original (unordered) patch list.
            let mut brute: Option<ShadedBilinearPatchHit> = None;
            for patch in &patches {
                if let Some(hit) = patch.intersect(&ray)
                    && brute.is_none_or(|b| hit.t < b.t)
                {
                    brute = Some(hit);
                }
            }

            let got = bvh.closest_hit(&ray);
            match (brute, got) {
                (None, None) => {}
                (Some(b), Some(g)) => {
                    assert_eq!(b.t.to_bits(), g.t.to_bits(), "nearest t differs");
                    assert_eq!(b.primitive, g.primitive, "nearest primitive differs");
                    assert_eq!(b.front_face, g.front_face);
                    tested += 1;
                }
                (b, g) => panic!("closest_hit disagreed with brute force: {b:?} vs {g:?}"),
            }

            assert_eq!(brute.is_some(), bvh.any_hit(&ray), "any_hit disagreed");
        }
        assert!(tested > 200, "too few nearest-hit comparisons: {tested}");
    }
}
