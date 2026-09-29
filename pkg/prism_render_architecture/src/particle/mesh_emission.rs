//! Mesh and skeletal-mesh surface emission sampling (design §8).
//!
//! This layer is the `CPU` reference *mathematics* behind the two mesh
//! sampling `DataInterface`s declared in [`super::modules`]:
//! [`super::modules::BuiltinDataInterface::SampleMesh`] (aligned with `Niagara`
//! "Static Mesh Location" and `VFX Graph` "Sample Mesh") and
//! [`super::modules::BuiltinDataInterface::SampleSkeletalMesh`] (aligned with
//! `Niagara` "Skeletal Mesh Location" and `VFX Graph` "Sample Skinned Mesh").
//!
//! Division of labor: [`super::modules`] owns the *declaration* side — the
//! binding tables (`B_MESH` / `B_SKELETAL`), the shader function-name lists
//! (`sample_mesh_barycentric`, `sample_skeletal_position`,
//! `sample_skeletal_velocity`), and the `WESL` codegen metadata a `GPU` kernel
//! consumes. This module is orthogonal to that: it implements the actual
//! area-weighted triangle picking, uniform barycentric sampling, attribute
//! interpolation, and linear-blend skinning (`LBS`) that both the `CPU` path
//! and a future `GPU` kernel must agree on bit for bit.
//!
//! Determinism: sampling consumes unit-interval random draws from a shared
//! [`super::emitter::UnitCursor`], the same cursor abstraction the emitter
//! shapes use, so the mesh spawn stream is reproducible against the hash `RNG`
//! (design §29) and lands bit-identically on the `GPU` kernel.
//!
//! Math budget: only basic arithmetic, `sqrt` (triangle area and
//! normalization), and `f32::floor`-free binary search over a prefix-sum `CDF`
//! are used. No transcendental function (`sin`/`cos`/`exp`/`ln`/`pow`) appears,
//! keeping the reference reproducible across backends.

use alloc::vec::Vec;

use super::emitter::UnitCursor;
use super::Vec3;

/// Absolute tolerance for `f32` comparisons in this module.
///
/// Direct `==` / `!=` on `f32` is forbidden here; equality is always expressed
/// as `(a - b).abs() < EPS`. A triangle whose area is below this threshold is
/// treated as degenerate and is never selected by the area-weighted sampler.
pub const EPS: f32 = 1e-6;

/// One mesh vertex: position plus the attributes interpolated at a sample.
///
/// The shading `normal` is optional data; when a mesh supplies no normals (all
/// zero) the sampler falls back to the per-triangle geometric normal so a
/// sampled emission direction is always well defined. `uv` carries a texture
/// coordinate that is barycentrically interpolated alongside the position.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshVertex {
    /// Object-space position of the vertex.
    pub position: Vec3,
    /// Object-space shading normal (need not be unit length; interpolation
    /// renormalizes). [`Vec3::ZERO`] means "no authored normal".
    pub normal: Vec3,
    /// Texture coordinate, interpolated with the same barycentric weights.
    pub uv: [f32; 2],
}

impl MeshVertex {
    /// Builds a fully specified vertex.
    #[must_use]
    pub const fn new(position: Vec3, normal: Vec3, uv: [f32; 2]) -> Self {
        Self {
            position,
            normal,
            uv,
        }
    }

    /// Builds a position-only vertex (zero normal, zero `UV`).
    ///
    /// Useful for geometry-only meshes where the geometric normal suffices.
    #[must_use]
    pub const fn at(position: Vec3) -> Self {
        Self {
            position,
            normal: Vec3::ZERO,
            uv: [0.0, 0.0],
        }
    }

    /// The degenerate vertex at the origin, returned for out-of-range indices.
    const DEGENERATE: Self = Self {
        position: Vec3::ZERO,
        normal: Vec3::ZERO,
        uv: [0.0, 0.0],
    };
}

/// Three resolved vertices forming one triangle of a mesh.
///
/// Produced by [`Mesh::triangle`] / [`SkeletalMesh::posed_triangle`]; it carries
/// enough information to compute the surface area, the geometric normal, and
/// any barycentrically interpolated attribute.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshTriangle {
    /// First vertex.
    pub a: MeshVertex,
    /// Second vertex.
    pub b: MeshVertex,
    /// Third vertex.
    pub c: MeshVertex,
}

impl MeshTriangle {
    /// Surface area `0.5 * |(b - a) × (c - a)|`.
    ///
    /// Uses only a cross product and one `sqrt`. A collinear or coincident
    /// triangle returns (near) zero and is excluded from area-weighted picking.
    #[must_use]
    pub fn area(&self) -> f32 {
        let e1 = self.b.position.sub(self.a.position);
        let e2 = self.c.position.sub(self.a.position);
        e1.cross(e2).length() * 0.5
    }

    /// Unit geometric normal `normalize((b - a) × (c - a))`.
    ///
    /// Returns [`Vec3::ZERO`] for a degenerate triangle (never `NaN`).
    #[must_use]
    pub fn geometric_normal(&self) -> Vec3 {
        let e1 = self.b.position.sub(self.a.position);
        let e2 = self.c.position.sub(self.a.position);
        e1.cross(e2).normalize_or_zero()
    }

    /// Interpolates the position at barycentric weights `(w_a, w_b, w_c)`.
    #[must_use]
    pub fn position_at(&self, bary: Vec3) -> Vec3 {
        self.a
            .position
            .scale(bary.x)
            .add(self.b.position.scale(bary.y))
            .add(self.c.position.scale(bary.z))
    }

    /// Interpolates and renormalizes the shading normal at `bary`.
    ///
    /// Falls back to the geometric normal when the mesh has no authored
    /// normals (interpolated normal is numerically zero).
    #[must_use]
    pub fn normal_at(&self, bary: Vec3) -> Vec3 {
        let blended = self
            .a
            .normal
            .scale(bary.x)
            .add(self.b.normal.scale(bary.y))
            .add(self.c.normal.scale(bary.z));
        let unit = blended.normalize_or_zero();
        if unit == Vec3::ZERO {
            self.geometric_normal()
        } else {
            unit
        }
    }

    /// Interpolates the texture coordinate at `bary`.
    #[must_use]
    pub fn uv_at(&self, bary: Vec3) -> [f32; 2] {
        [
            self.a.uv[0] * bary.x + self.b.uv[0] * bary.y + self.c.uv[0] * bary.z,
            self.a.uv[1] * bary.x + self.b.uv[1] * bary.y + self.c.uv[1] * bary.z,
        ]
    }
}

/// One sampled emission point on (or in) a mesh surface.
///
/// `barycentric` stores `(w_a, w_b, w_c)` which always sum to `1`, letting a
/// caller re-derive any other per-vertex attribute or re-evaluate the same
/// surface point on a different pose (used by skeletal velocity sampling).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshSample {
    /// Object-space (or posed) sample position.
    pub position: Vec3,
    /// Unit surface normal at the sample.
    pub normal: Vec3,
    /// Index of the selected triangle.
    pub tri_index: usize,
    /// Barycentric weights `(w_a, w_b, w_c)`; components sum to `1`.
    pub barycentric: Vec3,
    /// Interpolated texture coordinate.
    pub uv: [f32; 2],
}

impl MeshSample {
    /// The fallback sample for an empty or fully degenerate mesh: the origin,
    /// a zero normal, triangle `0`, and the first-vertex barycentric.
    const DEGENERATE: Self = Self {
        position: Vec3::ZERO,
        normal: Vec3::ZERO,
        tri_index: 0,
        barycentric: Vec3::new(1.0, 0.0, 0.0),
        uv: [0.0, 0.0],
    };
}

/// Converts a raw `u32` index into a `usize`, or `usize::MAX` on overflow.
///
/// An out-of-range result makes the guarded `slice::get` return `None`, which
/// the resolver turns into a degenerate (zero-area) triangle rather than a
/// panic — keeping sampling total on malformed index buffers.
#[inline]
#[must_use]
fn as_index(raw: u32) -> usize {
    usize::try_from(raw).unwrap_or(usize::MAX)
}

/// Folds two unit draws into uniform barycentric weights `(w_a, w_b, w_c)`.
///
/// The standard reflection `if u + v > 1 { u = 1 - u; v = 1 - v }` maps the
/// unit square uniformly onto the triangle; the returned components are each in
/// `0..=1` and sum to `1` (checked in the unit tests).
#[must_use]
fn fold_barycentric(u: f32, v: f32) -> Vec3 {
    let (su, sv) = if u + v > 1.0 {
        (1.0 - u, 1.0 - v)
    } else {
        (u, v)
    };
    Vec3::new(1.0 - su - sv, su, sv)
}

/// A triangle mesh with a precomputed area `CDF` for `O(log n)` area-weighted
/// surface sampling.
///
/// Positions and attributes live in `vertices`; `indices` lists triangles as
/// vertex triples. The prefix-sum `cdf` (built once at construction) stores the
/// running sum of triangle areas, so a triangle is chosen in proportion to its
/// area and zero-area (degenerate or out-of-range) triangles occupy no `CDF`
/// width and are never selected.
#[derive(Clone, Debug, PartialEq)]
pub struct Mesh {
    vertices: Vec<MeshVertex>,
    indices: Vec<[u32; 3]>,
    cdf: Vec<f32>,
    total_area: f32,
}

impl Mesh {
    /// Builds a mesh and its area `CDF` from vertices and triangle indices.
    #[must_use]
    pub fn new(vertices: Vec<MeshVertex>, indices: Vec<[u32; 3]>) -> Self {
        let mut mesh = Self {
            vertices,
            indices,
            cdf: Vec::new(),
            total_area: 0.0,
        };
        mesh.rebuild_cdf();
        mesh
    }

    /// Recomputes the area prefix-sum `CDF`; call after mutating geometry.
    fn rebuild_cdf(&mut self) {
        let mut cdf = Vec::with_capacity(self.indices.len());
        let mut running = 0.0_f32;
        for i in 0..self.indices.len() {
            running += self.triangle(i).area();
            cdf.push(running);
        }
        self.total_area = running;
        self.cdf = cdf;
    }

    /// Number of triangles.
    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.indices.len()
    }

    /// Total surface area (sum of all triangle areas).
    #[must_use]
    pub fn total_area(&self) -> f32 {
        self.total_area
    }

    /// Resolves a vertex by raw index, or the degenerate origin vertex when the
    /// index is out of range.
    #[must_use]
    fn vertex(&self, raw: u32) -> MeshVertex {
        match self.vertices.get(as_index(raw)) {
            Some(v) => *v,
            None => MeshVertex::DEGENERATE,
        }
    }

    /// Resolves triangle `i` into its three vertices.
    ///
    /// An out-of-range `i` or any out-of-range vertex index yields a degenerate
    /// (zero-area) triangle, never a panic.
    #[must_use]
    pub fn triangle(&self, i: usize) -> MeshTriangle {
        let Some(&[ia, ib, ic]) = self.indices.get(i) else {
            return MeshTriangle {
                a: MeshVertex::DEGENERATE,
                b: MeshVertex::DEGENERATE,
                c: MeshVertex::DEGENERATE,
            };
        };
        MeshTriangle {
            a: self.vertex(ia),
            b: self.vertex(ib),
            c: self.vertex(ic),
        }
    }

    /// Area of triangle `i`.
    #[must_use]
    pub fn triangle_area(&self, i: usize) -> f32 {
        self.triangle(i).area()
    }

    /// Axis-aligned bounds `(min, max)` over all vertices, or `(ZERO, ZERO)`
    /// for an empty mesh.
    #[must_use]
    pub fn aabb(&self) -> (Vec3, Vec3) {
        let Some((first, rest)) = self.vertices.split_first() else {
            return (Vec3::ZERO, Vec3::ZERO);
        };
        let mut lo = first.position;
        let mut hi = first.position;
        for v in rest {
            lo = lo.min(v.position);
            hi = hi.max(v.position);
        }
        (lo, hi)
    }

    /// Picks a triangle in proportion to its area from a unit draw `u`.
    ///
    /// Binary-searches the prefix-sum `CDF` for the first interval that
    /// contains `u * total_area`, then walks back over any trailing zero-area
    /// triangle so the result always has positive area when the mesh does.
    /// Returns `0` for an empty or fully degenerate mesh.
    #[must_use]
    pub fn select_triangle(&self, u: f32) -> usize {
        let count = self.cdf.len();
        if count == 0 || self.total_area < EPS {
            return 0;
        }
        let target = u.clamp(0.0, 1.0) * self.total_area;
        let mut idx = self.cdf.partition_point(|&c| c <= target).min(count - 1);
        while idx > 0 && self.triangle_area(idx) < EPS {
            idx -= 1;
        }
        idx
    }

    /// Samples a uniform surface point, consuming three unit draws from
    /// `cursor` (one to pick the triangle, two for the barycentric fold).
    ///
    /// Fully determined by the mesh and the cursor contents, so it reproduces
    /// against the hash `RNG` stream and a `GPU` kernel.
    #[must_use]
    pub fn sample_surface(&self, cursor: &mut UnitCursor<'_>) -> MeshSample {
        if self.triangle_count() == 0 {
            return MeshSample::DEGENERATE;
        }
        let tri_index = self.select_triangle(cursor.next_unit());
        let bary = fold_barycentric(cursor.next_unit(), cursor.next_unit());
        let tri = self.triangle(tri_index);
        MeshSample {
            position: tri.position_at(bary),
            normal: tri.normal_at(bary),
            tri_index,
            barycentric: bary,
            uv: tri.uv_at(bary),
        }
    }

    /// Tests whether `p` is inside this (assumed closed) mesh.
    ///
    /// Casts a fixed non-axis-aligned ray from `p` and counts forward triangle
    /// crossings with a Möller–Trumbore intersection; an odd count means
    /// inside. The skewed direction avoids the edge/vertex singularities an
    /// axis-aligned ray hits on axis-aligned meshes (a ray grazing a shared
    /// diagonal would otherwise be double-counted). Only basic arithmetic and
    /// one division per triangle are used.
    #[must_use]
    pub fn contains_point(&self, p: Vec3) -> bool {
        let dir = RAY_DIR;
        let mut crossings = 0_u32;
        for i in 0..self.triangle_count() {
            if ray_hits_triangle(p, dir, &self.triangle(i)) {
                crossings += 1;
            }
        }
        crossings % 2 == 1
    }

    /// Samples a point uniformly inside the mesh volume by axis-aligned
    /// rejection, consuming three unit draws per attempt.
    ///
    /// Returns `Some(point)` on the first accepted sample within
    /// [`VOLUME_MAX_TRIES`] attempts, or `None` when every attempt is rejected
    /// (a thin or open mesh); the caller can then fall back to surface
    /// sampling.
    #[must_use]
    pub fn sample_volume(&self, cursor: &mut UnitCursor<'_>) -> Option<Vec3> {
        if self.triangle_count() == 0 {
            return None;
        }
        let (lo, hi) = self.aabb();
        let extent = hi.sub(lo);
        for _ in 0..VOLUME_MAX_TRIES {
            let p = Vec3::new(
                lo.x + extent.x * cursor.next_unit(),
                lo.y + extent.y * cursor.next_unit(),
                lo.z + extent.z * cursor.next_unit(),
            );
            if self.contains_point(p) {
                return Some(p);
            }
        }
        None
    }
}

/// Fixed non-axis-aligned ray direction used by [`Mesh::contains_point`].
///
/// The irrational-looking, non-axis-aligned components make the parity ray miss
/// shared triangle edges and vertices on typical (including axis-aligned)
/// meshes, so a boundary sample is counted once rather than twice.
const RAY_DIR: Vec3 = Vec3::new(1.0, 0.372_133_1, 0.211_907_3);

/// Maximum axis-aligned rejection attempts for volume sampling before the
/// sampler reports failure (leaving the fallback to the caller).
pub const VOLUME_MAX_TRIES: u32 = 32;

/// Möller–Trumbore ray/triangle test: does the ray from `origin` along `dir`
/// cross `tri` in the forward (`t > EPS`) direction?
///
/// Returns `false` for a back-parallel or degenerate triangle. Used by
/// [`Mesh::contains_point`]'s parity test; pure arithmetic plus one division.
#[must_use]
fn ray_hits_triangle(origin: Vec3, dir: Vec3, tri: &MeshTriangle) -> bool {
    let e1 = tri.b.position.sub(tri.a.position);
    let e2 = tri.c.position.sub(tri.a.position);
    let h = dir.cross(e2);
    let det = e1.dot(h);
    if det.abs() < EPS {
        return false;
    }
    let inv_det = 1.0 / det;
    let s = origin.sub(tri.a.position);
    let u = inv_det * s.dot(h);
    if !(0.0..=1.0).contains(&u) {
        return false;
    }
    let q = s.cross(e1);
    let v = inv_det * dir.dot(q);
    if v < 0.0 || u + v > 1.0 {
        return false;
    }
    let t = inv_det * e2.dot(q);
    t > EPS
}

/// Unified entry point aligned with `Niagara` "Static Mesh Location" and
/// `VFX Graph` "Sample Mesh": one area-weighted, barycentric surface sample.
///
/// Deterministic in `mesh` and the `cursor` stream.
#[must_use]
pub fn sample_static_mesh(mesh: &Mesh, cursor: &mut UnitCursor<'_>) -> MeshSample {
    mesh.sample_surface(cursor)
}

/// A `3x4` affine bone transform: three basis columns plus a translation.
///
/// Applying it is a plain matrix-times-vector, so no transcendental function is
/// needed. This is the per-bone matrix consumed by linear-blend skinning
/// (`LBS`); authoring code composes rotation/scale into the basis columns.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoneTransform {
    /// Image of the object-space `+X` axis (first matrix column).
    pub basis_x: Vec3,
    /// Image of the object-space `+Y` axis (second matrix column).
    pub basis_y: Vec3,
    /// Image of the object-space `+Z` axis (third matrix column).
    pub basis_z: Vec3,
    /// Translation (fourth matrix column).
    pub translation: Vec3,
}

impl BoneTransform {
    /// The identity transform (leaves points and vectors unchanged).
    pub const IDENTITY: Self = Self {
        basis_x: Vec3::new(1.0, 0.0, 0.0),
        basis_y: Vec3::new(0.0, 1.0, 0.0),
        basis_z: Vec3::new(0.0, 0.0, 1.0),
        translation: Vec3::ZERO,
    };

    /// Builds a transform from its four columns.
    #[must_use]
    pub const fn new(basis_x: Vec3, basis_y: Vec3, basis_z: Vec3, translation: Vec3) -> Self {
        Self {
            basis_x,
            basis_y,
            basis_z,
            translation,
        }
    }

    /// A pure translation (identity basis).
    #[must_use]
    pub const fn from_translation(translation: Vec3) -> Self {
        Self {
            basis_x: Vec3::new(1.0, 0.0, 0.0),
            basis_y: Vec3::new(0.0, 1.0, 0.0),
            basis_z: Vec3::new(0.0, 0.0, 1.0),
            translation,
        }
    }

    /// Transforms a position (applies the basis and the translation).
    #[must_use]
    pub fn transform_point(&self, p: Vec3) -> Vec3 {
        self.basis_x
            .scale(p.x)
            .add(self.basis_y.scale(p.y))
            .add(self.basis_z.scale(p.z))
            .add(self.translation)
    }

    /// Transforms a direction (applies the basis only, no translation).
    ///
    /// This is the correct transform for skinning normals under rigid or
    /// uniform-scale bones; non-uniform scale would require the inverse
    /// transpose, which is out of scope for this reference.
    #[must_use]
    pub fn transform_vector(&self, v: Vec3) -> Vec3 {
        self.basis_x
            .scale(v.x)
            .add(self.basis_y.scale(v.y))
            .add(self.basis_z.scale(v.z))
    }
}

/// Number of bone influences per skinned vertex (a standard four-bone `LBS`).
pub const MAX_BONE_INFLUENCES: usize = 4;

/// A skinned mesh vertex: bind-pose attributes plus four-bone influences.
///
/// `weights` should sum to `1` for volume-preserving skinning but this is not
/// enforced; near-zero weights are skipped. `bones` indexes the pose's
/// [`BoneTransform`] palette.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SkinnedVertex {
    /// Bind-pose position.
    pub position: Vec3,
    /// Bind-pose shading normal ([`Vec3::ZERO`] means "no authored normal").
    pub normal: Vec3,
    /// Texture coordinate.
    pub uv: [f32; 2],
    /// Per-influence blend weights.
    pub weights: [f32; MAX_BONE_INFLUENCES],
    /// Per-influence bone-palette indices.
    pub bones: [u16; MAX_BONE_INFLUENCES],
}

impl SkinnedVertex {
    /// Builds a rigidly bound vertex (a single bone, unit weight).
    #[must_use]
    pub const fn rigid(position: Vec3, normal: Vec3, uv: [f32; 2], bone: u16) -> Self {
        Self {
            position,
            normal,
            uv,
            weights: [1.0, 0.0, 0.0, 0.0],
            bones: [bone, 0, 0, 0],
        }
    }
}

/// Linear-blend skinning (`LBS`) of a vertex position under a bone palette.
///
/// Computes `Σ w_i · (B_i · p)` over the (up to four) influences, skipping
/// near-zero weights and out-of-range bone indices. Pure matrix-vector math.
#[must_use]
pub fn skin_position(vertex: &SkinnedVertex, bones: &[BoneTransform]) -> Vec3 {
    let mut acc = Vec3::ZERO;
    for (&w, &bone_idx) in vertex.weights.iter().zip(vertex.bones.iter()) {
        if w.abs() < EPS {
            continue;
        }
        let Some(bone) = bones.get(usize::from(bone_idx)) else {
            continue;
        };
        acc = acc.add(bone.transform_point(vertex.position).scale(w));
    }
    acc
}

/// Linear-blend skinning of a vertex normal (basis only), renormalized.
///
/// Falls back to [`Vec3::ZERO`] when the vertex has no authored normal.
#[must_use]
pub fn skin_normal(vertex: &SkinnedVertex, bones: &[BoneTransform]) -> Vec3 {
    let mut acc = Vec3::ZERO;
    for (&w, &bone_idx) in vertex.weights.iter().zip(vertex.bones.iter()) {
        if w.abs() < EPS {
            continue;
        }
        let Some(bone) = bones.get(usize::from(bone_idx)) else {
            continue;
        };
        acc = acc.add(bone.transform_vector(vertex.normal).scale(w));
    }
    acc.normalize_or_zero()
}

/// A skeletal mesh: bind-pose skinned vertices plus triangle indices.
///
/// Posing it under a [`BoneTransform`] palette yields a static [`Mesh`] whose
/// surface can be area-weighted sampled exactly like a rigid mesh — the basis
/// for `Niagara` "Skeletal Mesh Location" / `VFX Graph` "Sample Skinned Mesh".
#[derive(Clone, Debug, PartialEq)]
pub struct SkeletalMesh {
    vertices: Vec<SkinnedVertex>,
    indices: Vec<[u32; 3]>,
}

impl SkeletalMesh {
    /// Builds a skeletal mesh from skinned vertices and triangle indices.
    #[must_use]
    pub fn new(vertices: Vec<SkinnedVertex>, indices: Vec<[u32; 3]>) -> Self {
        Self { vertices, indices }
    }

    /// Number of triangles.
    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.indices.len()
    }

    /// Skins vertex `raw` to its posed position under `bones` (degenerate
    /// origin for an out-of-range index).
    #[must_use]
    fn posed_position(&self, raw: u32, bones: &[BoneTransform]) -> Vec3 {
        match self.vertices.get(as_index(raw)) {
            Some(v) => skin_position(v, bones),
            None => Vec3::ZERO,
        }
    }

    /// Skins vertex `raw` into a posed [`MeshVertex`] under `bones`.
    #[must_use]
    fn posed_vertex(&self, raw: u32, bones: &[BoneTransform]) -> MeshVertex {
        match self.vertices.get(as_index(raw)) {
            Some(v) => MeshVertex::new(skin_position(v, bones), skin_normal(v, bones), v.uv),
            None => MeshVertex::DEGENERATE,
        }
    }

    /// Resolves triangle `i` into its posed vertices under `bones`.
    #[must_use]
    pub fn posed_triangle(&self, i: usize, bones: &[BoneTransform]) -> MeshTriangle {
        let Some(&[ia, ib, ic]) = self.indices.get(i) else {
            return MeshTriangle {
                a: MeshVertex::DEGENERATE,
                b: MeshVertex::DEGENERATE,
                c: MeshVertex::DEGENERATE,
            };
        };
        MeshTriangle {
            a: self.posed_vertex(ia, bones),
            b: self.posed_vertex(ib, bones),
            c: self.posed_vertex(ic, bones),
        }
    }

    /// Bakes this skeletal mesh into a static [`Mesh`] under `bones`.
    ///
    /// All vertices are skinned once and the area `CDF` rebuilt, so repeated
    /// sampling of the same pose should reuse the returned mesh.
    #[must_use]
    pub fn pose(&self, bones: &[BoneTransform]) -> Mesh {
        let mut posed = Vec::with_capacity(self.vertices.len());
        for v in &self.vertices {
            posed.push(MeshVertex::new(
                skin_position(v, bones),
                skin_normal(v, bones),
                v.uv,
            ));
        }
        Mesh::new(posed, self.indices.clone())
    }
}

/// A skeletal surface sample paired with the surface velocity at that point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SkeletalVelocitySample {
    /// The surface sample on the current pose.
    pub sample: MeshSample,
    /// Surface velocity from finite-differencing the two poses.
    pub velocity: Vec3,
}

/// Unified entry point aligned with `Niagara` "Skeletal Mesh Location" and
/// `VFX Graph` "Sample Skinned Mesh": pose the mesh, then area-weighted sample.
///
/// Deterministic in the mesh, the bone palette, and the `cursor` stream.
#[must_use]
pub fn sample_skeletal_mesh(
    mesh: &SkeletalMesh,
    bones: &[BoneTransform],
    cursor: &mut UnitCursor<'_>,
) -> MeshSample {
    let posed = mesh.pose(bones);
    posed.sample_surface(cursor)
}

/// Samples a skeletal surface point on the current pose and finite-differences
/// the previous pose to recover the surface velocity (the `sample_skeletal_
/// velocity` `DataInterface` function).
///
/// The same triangle and barycentric weights are re-evaluated on `prev_bones`
/// so the velocity reflects the exact material point's motion; `dt` below
/// [`EPS`] yields a zero velocity rather than a division by zero.
#[must_use]
pub fn sample_skeletal_velocity(
    mesh: &SkeletalMesh,
    prev_bones: &[BoneTransform],
    curr_bones: &[BoneTransform],
    dt: f32,
    cursor: &mut UnitCursor<'_>,
) -> SkeletalVelocitySample {
    let sample = sample_skeletal_mesh(mesh, curr_bones, cursor);
    let bary = sample.barycentric;
    let prev_pos = match mesh.indices.get(sample.tri_index) {
        Some(&[ia, ib, ic]) => mesh
            .posed_position(ia, prev_bones)
            .scale(bary.x)
            .add(mesh.posed_position(ib, prev_bones).scale(bary.y))
            .add(mesh.posed_position(ic, prev_bones).scale(bary.z)),
        None => sample.position,
    };
    let velocity = if dt.abs() < EPS {
        Vec3::ZERO
    } else {
        sample.position.sub(prev_pos).scale(1.0 / dt)
    };
    SkeletalVelocitySample { sample, velocity }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// A unit right triangle in the `z = 0` plane with `+Z` normals.
    fn unit_right_triangle() -> Mesh {
        let verts = vec![
            MeshVertex::new(
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                [0.0, 0.0],
            ),
            MeshVertex::new(
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                [1.0, 0.0],
            ),
            MeshVertex::new(
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                [0.0, 1.0],
            ),
        ];
        Mesh::new(verts, vec![[0, 1, 2]])
    }

    /// An axis-aligned unit cube `[0,1]^3` as 12 triangles (outward winding).
    fn unit_cube() -> Mesh {
        let p = |x: f32, y: f32, z: f32| MeshVertex::at(Vec3::new(x, y, z));
        let verts = vec![
            p(0.0, 0.0, 0.0),
            p(1.0, 0.0, 0.0),
            p(1.0, 1.0, 0.0),
            p(0.0, 1.0, 0.0),
            p(0.0, 0.0, 1.0),
            p(1.0, 0.0, 1.0),
            p(1.0, 1.0, 1.0),
            p(0.0, 1.0, 1.0),
        ];
        let indices = vec![
            [0, 2, 1],
            [0, 3, 2], // -Z
            [4, 5, 6],
            [4, 6, 7], // +Z
            [0, 1, 5],
            [0, 5, 4], // -Y
            [3, 6, 2],
            [3, 7, 6], // +Y
            [0, 4, 7],
            [0, 7, 3], // -X
            [1, 2, 6],
            [1, 6, 5], // +X
        ];
        Mesh::new(verts, indices)
    }

    #[test]
    fn triangle_area_matches_analytic() {
        let mesh = unit_right_triangle();
        assert!((mesh.triangle_area(0) - 0.5).abs() < EPS);
        assert!((mesh.total_area() - 0.5).abs() < EPS);
    }

    #[test]
    fn geometric_normal_points_along_plus_z() {
        let n = unit_right_triangle().triangle(0).geometric_normal();
        assert!(n.distance(Vec3::new(0.0, 0.0, 1.0)) < EPS);
    }

    #[test]
    fn degenerate_triangle_has_zero_area() {
        // Three collinear points -> zero area.
        let verts = vec![
            MeshVertex::at(Vec3::new(0.0, 0.0, 0.0)),
            MeshVertex::at(Vec3::new(1.0, 0.0, 0.0)),
            MeshVertex::at(Vec3::new(2.0, 0.0, 0.0)),
        ];
        let mesh = Mesh::new(verts, vec![[0, 1, 2]]);
        assert!(mesh.triangle_area(0) < EPS);
    }

    #[test]
    fn out_of_range_triangle_is_degenerate_not_panic() {
        let verts = vec![MeshVertex::at(Vec3::ZERO)];
        let mesh = Mesh::new(verts, vec![[0, 9, 42]]);
        assert!(mesh.triangle_area(0) < EPS);
        assert!(mesh.total_area() < EPS);
    }

    #[test]
    fn barycentric_weights_sum_to_one_and_stay_in_range() {
        // Sweep the unit square, including the folded region u + v > 1.
        for i in 0..17 {
            for j in 0..17 {
                let u = i as f32 / 16.0;
                let v = j as f32 / 16.0;
                let b = fold_barycentric(u, v);
                assert!((b.x + b.y + b.z - 1.0).abs() < EPS);
                assert!(b.x >= -EPS && b.y >= -EPS && b.z >= -EPS);
                assert!(b.x <= 1.0 + EPS && b.y <= 1.0 + EPS && b.z <= 1.0 + EPS);
            }
        }
    }

    #[test]
    fn sampled_point_lies_in_triangle_plane() {
        let mesh = unit_right_triangle();
        let tri = mesh.triangle(0);
        let n = tri.geometric_normal();
        for i in 0..64 {
            let u = (i as f32 + 0.3) / 64.0;
            let v = (i as f32 + 0.7) / 64.0 * 0.5;
            let samples = [0.0, u, v];
            let mut cursor = UnitCursor::new(&samples);
            let s = mesh.sample_surface(&mut cursor);
            // Point is in the z = 0 plane: distance along the normal is ~0.
            let offset = s.position.sub(tri.a.position);
            assert!(n.dot(offset).abs() < 1e-4);
        }
    }

    #[test]
    fn interpolated_normal_is_unit_length() {
        let mesh = unit_right_triangle();
        let samples = [0.0, 0.25, 0.25];
        let mut cursor = UnitCursor::new(&samples);
        let s = mesh.sample_surface(&mut cursor);
        assert!((s.normal.length() - 1.0).abs() < EPS);
    }

    #[test]
    fn missing_normals_fall_back_to_geometric_normal() {
        // Position-only mesh (zero authored normals).
        let verts = vec![
            MeshVertex::at(Vec3::new(0.0, 0.0, 0.0)),
            MeshVertex::at(Vec3::new(1.0, 0.0, 0.0)),
            MeshVertex::at(Vec3::new(0.0, 1.0, 0.0)),
        ];
        let mesh = Mesh::new(verts, vec![[0, 1, 2]]);
        let samples = [0.0, 0.2, 0.2];
        let mut cursor = UnitCursor::new(&samples);
        let s = mesh.sample_surface(&mut cursor);
        assert!(s.normal.distance(Vec3::new(0.0, 0.0, 1.0)) < EPS);
    }

    #[test]
    fn area_weighting_matches_area_ratio() {
        // Triangle 0 has area 0.5; triangle 1 has area 2.0 -> 1:4 selection.
        let verts = vec![
            MeshVertex::at(Vec3::new(0.0, 0.0, 0.0)),
            MeshVertex::at(Vec3::new(1.0, 0.0, 0.0)),
            MeshVertex::at(Vec3::new(0.0, 1.0, 0.0)),
            MeshVertex::at(Vec3::new(10.0, 0.0, 0.0)),
            MeshVertex::at(Vec3::new(12.0, 0.0, 0.0)),
            MeshVertex::at(Vec3::new(10.0, 2.0, 0.0)),
        ];
        let mesh = Mesh::new(verts, vec![[0, 1, 2], [3, 4, 5]]);
        assert!((mesh.triangle_area(0) - 0.5).abs() < EPS);
        assert!((mesh.triangle_area(1) - 2.0).abs() < EPS);

        // Stratified sweep of the selection draw makes the ratio near-exact.
        let n = 10_000_u32;
        let mut count1 = 0_u32;
        for i in 0..n {
            let u = (i as f32 + 0.5) / n as f32;
            if mesh.select_triangle(u) == 1 {
                count1 += 1;
            }
        }
        let frac1 = count1 as f32 / n as f32;
        // Expected 2.0 / 2.5 = 0.8.
        assert!((frac1 - 0.8).abs() < 0.02, "fraction was {frac1}");
    }

    #[test]
    fn degenerate_triangle_is_never_selected() {
        // Triangle 0 degenerate (collinear), triangle 1 valid.
        let verts = vec![
            MeshVertex::at(Vec3::new(0.0, 0.0, 0.0)),
            MeshVertex::at(Vec3::new(1.0, 0.0, 0.0)),
            MeshVertex::at(Vec3::new(2.0, 0.0, 0.0)),
            MeshVertex::at(Vec3::new(0.0, 0.0, 0.0)),
            MeshVertex::at(Vec3::new(1.0, 0.0, 0.0)),
            MeshVertex::at(Vec3::new(0.0, 1.0, 0.0)),
        ];
        let mesh = Mesh::new(verts, vec![[0, 1, 2], [3, 4, 5]]);
        let n = 5_000_u32;
        for i in 0..n {
            let u = (i as f32 + 0.5) / n as f32;
            assert_eq!(mesh.select_triangle(u), 1);
        }
        // Boundary draws must not select the degenerate triangle either.
        assert_eq!(mesh.select_triangle(0.0), 1);
        assert_eq!(mesh.select_triangle(1.0), 1);
    }

    #[test]
    fn sampling_is_deterministic() {
        let mesh = unit_right_triangle();
        let samples = [0.42, 0.13, 0.71];
        let mut a = UnitCursor::new(&samples);
        let mut b = UnitCursor::new(&samples);
        assert_eq!(mesh.sample_surface(&mut a), mesh.sample_surface(&mut b));
    }

    #[test]
    fn empty_mesh_samples_degenerate_without_panic() {
        let mesh = Mesh::new(Vec::new(), Vec::new());
        let mut cursor = UnitCursor::new(&[]);
        let s = mesh.sample_surface(&mut cursor);
        assert_eq!(s, MeshSample::DEGENERATE);
        assert_eq!(mesh.triangle_count(), 0);
    }

    #[test]
    fn aabb_bounds_all_vertices() {
        let (lo, hi) = unit_cube().aabb();
        assert!(lo.distance(Vec3::ZERO) < EPS);
        assert!(hi.distance(Vec3::new(1.0, 1.0, 1.0)) < EPS);
    }

    #[test]
    fn point_in_mesh_parity_test() {
        let cube = unit_cube();
        assert!(cube.contains_point(Vec3::new(0.5, 0.5, 0.5)));
        assert!(cube.contains_point(Vec3::new(0.1, 0.9, 0.5)));
        assert!(!cube.contains_point(Vec3::new(1.5, 0.5, 0.5)));
        assert!(!cube.contains_point(Vec3::new(-0.5, 0.5, 0.5)));
        assert!(!cube.contains_point(Vec3::new(0.5, 0.5, 2.0)));
    }

    #[test]
    fn volume_sampling_lands_inside_the_mesh() {
        let cube = unit_cube();
        // A long deterministic stream of interior-friendly coordinates.
        let stream = [
            0.5, 0.5, 0.5, 0.25, 0.75, 0.1, 0.9, 0.2, 0.6, 0.33, 0.44, 0.55,
        ];
        let mut cursor = UnitCursor::new(&stream);
        let p = cube.sample_volume(&mut cursor).expect("interior sample");
        assert!(cube.contains_point(p));
        assert!(p.x >= -EPS && p.x <= 1.0 + EPS);
        assert!(p.y >= -EPS && p.y <= 1.0 + EPS);
        assert!(p.z >= -EPS && p.z <= 1.0 + EPS);
    }

    #[test]
    fn bone_identity_leaves_position_unchanged() {
        let vertex = SkinnedVertex::rigid(Vec3::new(3.0, -2.0, 5.0), Vec3::ZERO, [0.0, 0.0], 0);
        let bones = [BoneTransform::IDENTITY];
        let out = skin_position(&vertex, &bones);
        assert!(out.distance(Vec3::new(3.0, -2.0, 5.0)) < EPS);
    }

    #[test]
    fn bone_translation_applies() {
        let vertex = SkinnedVertex::rigid(Vec3::new(1.0, 1.0, 1.0), Vec3::ZERO, [0.0, 0.0], 0);
        let bones = [BoneTransform::from_translation(Vec3::new(10.0, 0.0, -4.0))];
        let out = skin_position(&vertex, &bones);
        assert!(out.distance(Vec3::new(11.0, 1.0, -3.0)) < EPS);
    }

    #[test]
    fn two_bones_fifty_fifty_gives_midpoint() {
        // Same bind position, two bones translating in opposite directions,
        // 50/50 weights -> the midpoint of the two transformed positions.
        let vertex = SkinnedVertex {
            position: Vec3::new(0.0, 0.0, 0.0),
            normal: Vec3::ZERO,
            uv: [0.0, 0.0],
            weights: [0.5, 0.5, 0.0, 0.0],
            bones: [0, 1, 0, 0],
        };
        let bones = [
            BoneTransform::from_translation(Vec3::new(-2.0, 0.0, 0.0)),
            BoneTransform::from_translation(Vec3::new(4.0, 0.0, 0.0)),
        ];
        let out = skin_position(&vertex, &bones);
        // Midpoint of (-2,0,0) and (4,0,0) is (1,0,0).
        assert!(out.distance(Vec3::new(1.0, 0.0, 0.0)) < EPS);
    }

    #[test]
    fn skin_normal_is_unit_or_zero() {
        let vertex = SkinnedVertex::rigid(Vec3::ZERO, Vec3::new(0.0, 3.0, 0.0), [0.0, 0.0], 0);
        let bones = [BoneTransform::IDENTITY];
        let n = skin_normal(&vertex, &bones);
        assert!((n.length() - 1.0).abs() < EPS);
        assert!(n.distance(Vec3::new(0.0, 1.0, 0.0)) < EPS);

        let no_normal = SkinnedVertex::rigid(Vec3::ZERO, Vec3::ZERO, [0.0, 0.0], 0);
        assert_eq!(skin_normal(&no_normal, &bones), Vec3::ZERO);
    }

    /// A skeletal quad (two triangles) rigidly bound to bone 0.
    fn skeletal_quad() -> SkeletalMesh {
        let verts = vec![
            SkinnedVertex::rigid(
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                [0.0, 0.0],
                0,
            ),
            SkinnedVertex::rigid(
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                [1.0, 0.0],
                0,
            ),
            SkinnedVertex::rigid(
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                [1.0, 1.0],
                0,
            ),
            SkinnedVertex::rigid(
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                [0.0, 1.0],
                0,
            ),
        ];
        SkeletalMesh::new(verts, vec![[0, 1, 2], [0, 2, 3]])
    }

    #[test]
    fn skeletal_identity_pose_matches_bind_pose_sample() {
        let mesh = skeletal_quad();
        let bones = [BoneTransform::IDENTITY];
        let samples = [0.6, 0.3, 0.2];
        let mut cursor = UnitCursor::new(&samples);
        let s = sample_skeletal_mesh(&mesh, &bones, &mut cursor);
        // The bind quad lies in z = 0, so any sample must too.
        assert!(s.position.z.abs() < EPS);
        assert!((s.normal.length() - 1.0).abs() < EPS);
    }

    #[test]
    fn skeletal_translation_pose_shifts_sample() {
        let mesh = skeletal_quad();
        let identity = [BoneTransform::IDENTITY];
        let shifted = [BoneTransform::from_translation(Vec3::new(5.0, 0.0, 0.0))];
        let samples = [0.6, 0.3, 0.2];

        let mut c0 = UnitCursor::new(&samples);
        let base = sample_skeletal_mesh(&mesh, &identity, &mut c0);
        let mut c1 = UnitCursor::new(&samples);
        let moved = sample_skeletal_mesh(&mesh, &shifted, &mut c1);

        let delta = moved.position.sub(base.position);
        assert!(delta.distance(Vec3::new(5.0, 0.0, 0.0)) < EPS);
    }

    #[test]
    fn skeletal_sampling_is_deterministic() {
        let mesh = skeletal_quad();
        let bones = [BoneTransform::IDENTITY];
        let samples = [0.11, 0.22, 0.33];
        let mut a = UnitCursor::new(&samples);
        let mut b = UnitCursor::new(&samples);
        assert_eq!(
            sample_skeletal_mesh(&mesh, &bones, &mut a),
            sample_skeletal_mesh(&mesh, &bones, &mut b)
        );
    }

    #[test]
    fn skeletal_velocity_matches_finite_difference() {
        let mesh = skeletal_quad();
        let prev = [BoneTransform::from_translation(Vec3::new(0.0, 0.0, 0.0))];
        let curr = [BoneTransform::from_translation(Vec3::new(2.0, 0.0, 0.0))];
        let dt = 0.5;
        let samples = [0.4, 0.35, 0.25];
        let mut cursor = UnitCursor::new(&samples);
        let out = sample_skeletal_velocity(&mesh, &prev, &curr, dt, &mut cursor);
        // Whole surface translated by 2 over dt = 0.5 -> velocity (4, 0, 0).
        assert!(out.velocity.distance(Vec3::new(4.0, 0.0, 0.0)) < 1e-4);
    }

    #[test]
    fn skeletal_velocity_zero_dt_is_zero() {
        let mesh = skeletal_quad();
        let prev = [BoneTransform::IDENTITY];
        let curr = [BoneTransform::from_translation(Vec3::new(9.0, 9.0, 9.0))];
        let samples = [0.4, 0.35, 0.25];
        let mut cursor = UnitCursor::new(&samples);
        let out = sample_skeletal_velocity(&mesh, &prev, &curr, 0.0, &mut cursor);
        assert_eq!(out.velocity, Vec3::ZERO);
    }

    #[test]
    fn static_mesh_entry_point_agrees_with_method() {
        let mesh = unit_right_triangle();
        let samples = [0.5, 0.2, 0.3];
        let mut a = UnitCursor::new(&samples);
        let mut b = UnitCursor::new(&samples);
        assert_eq!(
            sample_static_mesh(&mesh, &mut a),
            mesh.sample_surface(&mut b)
        );
    }
}
