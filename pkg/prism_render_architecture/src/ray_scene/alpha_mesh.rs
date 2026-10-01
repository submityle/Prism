//! Alpha-tested (cutout) indexed triangle mesh: foliage, fence, and decal
//! geometry whose silhouette lives in an alpha mask rather than in triangles.
//!
//! A real-time renderer draws leaves, grass blades, chain-link fences, and
//! grates as a handful of flat quads whose *shape* comes from an alpha texture:
//! the triangle is geometrically solid, but a ray that strikes a texel below
//! the cutoff passes straight through as if the surface were not there. This is
//! pbrt's alpha-texture test and the hardware `RT` "any-hit alpha" callback.
//!
//! [`AlphaMesh`] layers that gate on top of the proven [`super::triangle_mesh`]
//! primitive: it keeps a [`TriangleMesh`] verbatim (so the flat `GPU` layout
//! still decodes bit-identically), samples an [`AlphaTexture`] at each
//! candidate hit's interpolated `UV`, and keeps only hits whose alpha is at or
//! above the [`AlphaMesh::cutoff`]. The ray math — Möller–Trumbore, the
//! barycentric position/normal/`UV` reconstruction, and the `BVH` traversal —
//! is reused unchanged from [`super::triangle_mesh`]; the only new behavior is
//! that a sub-cutoff hit does **not** shrink the ray interval (closest hit) and
//! does **not** satisfy an occlusion query (any hit), so the ray continues to
//! the geometry behind the hole.
//!
//! Because the alpha test keys off the hit's `UV`, a mesh carrying no `UV` pool
//! falls back to the barycentric `(u, v)` coordinate (see
//! [`TriangleMesh::intersect_triangle`]); meaningful cutouts therefore require a
//! `UV`-mapped mesh. The on-device variant adds only the mask texels and the
//! cutoff to the triangle-mesh buffers, so it is handled in a sibling
//! `alpha_mesh_gpu_layout` module.

use super::bvh::{Aabb, BvhBuildConfig, LinearBvhNode};
use super::traversal::Ray;
use super::triangle_mesh::{MeshHit, TriangleMesh, TriangleMeshBvh};

/// Why [`AlphaTexture::new`] rejected its inputs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AlphaTextureError {
    /// Either dimension was zero, so the texel grid would be empty.
    EmptyDimensions {
        /// Requested width.
        width: usize,
        /// Requested height.
        height: usize,
    },
    /// The texel count did not equal `width * height`.
    TexelCountMismatch {
        /// Number of texels supplied.
        texels: usize,
        /// Number of texels required (`width * height`).
        expected: usize,
    },
}

impl core::fmt::Display for AlphaTextureError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyDimensions { width, height } => write!(
                f,
                "alpha texture dimensions must be non-zero but were {width} x {height}"
            ),
            Self::TexelCountMismatch { texels, expected } => write!(
                f,
                "alpha texture has {texels} texels but {expected} were expected"
            ),
        }
    }
}

impl std::error::Error for AlphaTextureError {}

/// A single-channel alpha mask sampled with clamp-addressed bilinear filtering.
///
/// Texels are stored row-major (`texels[y * width + x]`) and clamped to the
/// `[0, 1]` range at construction, matching how a cutout mask is authored and
/// how the `GPU` layout will decode it. Bilinear filtering uses the pixel-center
/// convention (texel `(x, y)` sits at `UV` `((x + 0.5) / width, (y + 0.5) /
/// height)`) and clamps out-of-range taps to the edge texel.
#[derive(Clone, Debug, PartialEq)]
pub struct AlphaTexture {
    /// Texel columns; always at least one.
    width: usize,
    /// Texel rows; always at least one.
    height: usize,
    /// Row-major alpha texels, each clamped to `[0, 1]`.
    texels: Vec<f32>,
}

impl AlphaTexture {
    /// Builds an alpha mask from `width * height` row-major `texels`.
    ///
    /// Texels are clamped to `[0, 1]`; the clamp is idempotent, so the `GPU`
    /// layout re-decodes the same values bit-for-bit.
    ///
    /// # Errors
    ///
    /// Returns [`AlphaTextureError`] when a dimension is zero or the texel count
    /// does not equal `width * height`.
    pub fn new(width: usize, height: usize, texels: Vec<f32>) -> Result<Self, AlphaTextureError> {
        if width == 0 || height == 0 {
            return Err(AlphaTextureError::EmptyDimensions { width, height });
        }
        let expected = width * height;
        if texels.len() != expected {
            return Err(AlphaTextureError::TexelCountMismatch {
                texels: texels.len(),
                expected,
            });
        }
        let texels = texels.into_iter().map(|a| a.clamp(0.0, 1.0)).collect();
        Ok(Self {
            width,
            height,
            texels,
        })
    }

    /// Texel columns.
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Texel rows.
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    /// The row-major alpha texels, each in `[0, 1]`.
    #[must_use]
    pub fn texels(&self) -> &[f32] {
        &self.texels
    }

    /// The clamped texel at integer coordinate `(x, y)`.
    #[must_use]
    pub fn texel(&self, x: usize, y: usize) -> f32 {
        self.texels[y * self.width + x]
    }

    /// Bilinearly samples the mask at `uv`, clamping both the coordinate and the
    /// four taps to the texture edges.
    #[must_use]
    pub fn sample(&self, uv: [f32; 2]) -> f32 {
        let u = uv[0].clamp(0.0, 1.0);
        let v = uv[1].clamp(0.0, 1.0);
        // Pixel-center convention: shift so texel centers land on integers.
        let fx = u * self.width as f32 - 0.5;
        let fy = v * self.height as f32 - 0.5;
        let x0f = fx.floor();
        let y0f = fy.floor();
        let tx = fx - x0f;
        let ty = fy - y0f;
        let x0 = clamp_index(x0f, self.width);
        let x1 = clamp_index(x0f + 1.0, self.width);
        let y0 = clamp_index(y0f, self.height);
        let y1 = clamp_index(y0f + 1.0, self.height);
        let a = self.texel(x0, y0);
        let b = self.texel(x1, y0);
        let c = self.texel(x0, y1);
        let d = self.texel(x1, y1);
        let top = a + (b - a) * tx;
        let bottom = c + (d - c) * tx;
        top + (bottom - top) * ty
    }
}

/// An alpha-tested triangle mesh: a [`TriangleMesh`] plus the mask and cutoff
/// that decide which hits survive.
///
/// The mesh is stored verbatim so a flat `GPU` layout decodes a bit-identical
/// primitive; the alpha gate is applied only at hit evaluation.
#[derive(Clone, Debug, PartialEq)]
pub struct AlphaMesh {
    /// The underlying indexed triangle geometry (kept intact).
    mesh: TriangleMesh,
    /// The cutout mask sampled at each candidate hit's `UV`.
    alpha: AlphaTexture,
    /// Hits with sampled alpha below this value are treated as misses; stored
    /// clamped to `[0, 1]`.
    cutoff: f32,
}

impl AlphaMesh {
    /// Builds an alpha-tested mesh from `mesh`, its `alpha` mask, and the
    /// `cutoff` below which hits are discarded.
    ///
    /// `cutoff` is clamped to `[0, 1]` (the clamp is idempotent). A cutoff of
    /// `0` keeps every hit (fully opaque); `1` keeps only fully opaque texels.
    #[must_use]
    pub fn new(mesh: TriangleMesh, alpha: AlphaTexture, cutoff: f32) -> Self {
        Self {
            mesh,
            alpha,
            cutoff: cutoff.clamp(0.0, 1.0),
        }
    }

    /// The underlying triangle mesh.
    #[must_use]
    pub fn mesh(&self) -> &TriangleMesh {
        &self.mesh
    }

    /// The cutout alpha mask.
    #[must_use]
    pub fn alpha(&self) -> &AlphaTexture {
        &self.alpha
    }

    /// The alpha cutoff in `[0, 1]`.
    #[must_use]
    pub fn cutoff(&self) -> f32 {
        self.cutoff
    }

    /// Axis-aligned bounds of the underlying mesh.
    #[must_use]
    pub fn aabb(&self) -> Aabb {
        self.mesh.aabb()
    }

    /// True when the mask at `uv` is at or above the cutoff (an opaque texel).
    #[must_use]
    pub fn is_opaque_at(&self, uv: [f32; 2]) -> bool {
        self.alpha.sample(uv) >= self.cutoff
    }

    /// Nearest *opaque* intersection along `ray` via a linear scan, or `None`.
    ///
    /// This is the brute-force reference for [`AlphaMeshBvh::closest_hit`]: a
    /// sub-cutoff hit does not shrink the search interval, so the ray continues
    /// to any geometry visible through the hole.
    #[must_use]
    pub fn intersect(&self, ray: &Ray) -> Option<MeshHit> {
        let mut ray = *ray;
        let mut best: Option<MeshHit> = None;
        for tri in 0..self.mesh.triangle_count() {
            if let Some(hit) = self.mesh.intersect_triangle(tri, &ray)
                && self.is_opaque_at(hit.uv)
            {
                ray = Ray::new(ray.origin(), ray.direction(), ray.t_min(), hit.t);
                best = Some(hit);
            }
        }
        best
    }

    /// True when any *opaque* texel of any triangle intersects `ray`.
    #[must_use]
    pub fn any_hit(&self, ray: &Ray) -> bool {
        for tri in 0..self.mesh.triangle_count() {
            if let Some(hit) = self.mesh.intersect_triangle(tri, ray)
                && self.is_opaque_at(hit.uv)
            {
                return true;
            }
        }
        false
    }
}

/// A single-level `BVH` over an [`AlphaMesh`] whose traversal honours the alpha
/// cutout.
///
/// The acceleration structure is identical to [`TriangleMeshBvh`] (the mask does
/// not change triangle bounds), so this type wraps one and layers the alpha gate
/// into closest-hit and any-hit traversal: only an opaque hit shrinks the ray
/// interval or satisfies an occlusion query.
#[derive(Clone, Debug)]
pub struct AlphaMeshBvh {
    /// The underlying triangle-mesh `BVH` (nodes, order, and the mesh itself).
    inner: TriangleMeshBvh,
    /// The cutout mask sampled during traversal.
    alpha: AlphaTexture,
    /// Alpha cutoff in `[0, 1]`.
    cutoff: f32,
}

impl AlphaMeshBvh {
    /// Builds the `BVH` with [`BvhBuildConfig::default`].
    #[must_use]
    pub fn build(mesh: TriangleMesh, alpha: AlphaTexture, cutoff: f32) -> Self {
        Self::build_with(mesh, alpha, cutoff, BvhBuildConfig::default())
    }

    /// Builds the `BVH` with the given binned-`SAH` `config`.
    #[must_use]
    pub fn build_with(
        mesh: TriangleMesh,
        alpha: AlphaTexture,
        cutoff: f32,
        config: BvhBuildConfig,
    ) -> Self {
        Self {
            inner: TriangleMeshBvh::build_with(mesh, config),
            alpha,
            cutoff: cutoff.clamp(0.0, 1.0),
        }
    }

    /// The underlying triangle mesh.
    #[must_use]
    pub fn mesh(&self) -> &TriangleMesh {
        self.inner.mesh()
    }

    /// The cutout alpha mask.
    #[must_use]
    pub fn alpha(&self) -> &AlphaTexture {
        &self.alpha
    }

    /// The alpha cutoff in `[0, 1]`.
    #[must_use]
    pub fn cutoff(&self) -> f32 {
        self.cutoff
    }

    /// Number of flattened `BVH` nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    /// Number of pooled triangles.
    #[must_use]
    pub fn primitive_count(&self) -> usize {
        self.inner.primitive_count()
    }

    /// True when the hierarchy holds no triangles.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Root bounds, or the empty box when the hierarchy is empty.
    #[must_use]
    pub fn bounds(&self) -> Aabb {
        self.inner.bounds()
    }

    /// The flattened node array.
    #[must_use]
    pub fn nodes(&self) -> &[LinearBvhNode] {
        self.inner.nodes()
    }

    /// Maps each `BVH` primitive slot back to its original triangle index.
    #[must_use]
    pub fn order(&self) -> &[u32] {
        self.inner.order()
    }

    /// True when the mask at `uv` is at or above the cutoff.
    #[must_use]
    fn is_opaque_at(&self, uv: [f32; 2]) -> bool {
        self.alpha.sample(uv) >= self.cutoff
    }

    /// Nearest opaque intersection along `ray`, or `None` if the ray reaches
    /// only transparent texels or empty space.
    #[must_use]
    pub fn closest_hit(&self, ray: &Ray) -> Option<MeshHit> {
        let nodes = self.inner.nodes();
        if nodes.is_empty() {
            return None;
        }
        let mesh = self.inner.mesh();
        let order = self.inner.order();
        let mut ray = *ray;
        let mut best: Option<MeshHit> = None;

        let mut stack = [0u32; 64];
        let mut sp = 0usize;
        let mut node_index = 0u32;
        loop {
            let node = &nodes[node_index as usize];
            if ray
                .aabb_interval(&node.bounds, ray.t_min(), ray.t_max())
                .is_some()
            {
                if node.is_leaf() {
                    let start = node.first_primitive as usize;
                    let end = start + node.primitive_count as usize;
                    for &slot in &order[start..end] {
                        if let Some(hit) = mesh.intersect_triangle(slot as usize, &ray)
                            && self.is_opaque_at(hit.uv)
                        {
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

    /// True when any opaque texel of any triangle intersects `ray`.
    #[must_use]
    pub fn any_hit(&self, ray: &Ray) -> bool {
        let nodes = self.inner.nodes();
        if nodes.is_empty() {
            return false;
        }
        let mesh = self.inner.mesh();
        let order = self.inner.order();
        let mut stack = [0u32; 64];
        let mut sp = 0usize;
        let mut node_index = 0u32;
        loop {
            let node = &nodes[node_index as usize];
            if ray
                .aabb_interval(&node.bounds, ray.t_min(), ray.t_max())
                .is_some()
            {
                if node.is_leaf() {
                    let start = node.first_primitive as usize;
                    let end = start + node.primitive_count as usize;
                    for &slot in &order[start..end] {
                        if let Some(hit) = mesh.intersect_triangle(slot as usize, ray)
                            && self.is_opaque_at(hit.uv)
                        {
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

/// Clamps a (floored) texel coordinate into `[0, dim - 1]`.
///
/// `coord` is always an integer-valued float produced by [`f32::floor`] (or that
/// value plus one), so the `as usize` truncation is exact for the non-negative
/// branch; `dim` is guaranteed non-zero by [`AlphaTexture::new`].
fn clamp_index(coord: f32, dim: usize) -> usize {
    if coord < 0.0 {
        return 0;
    }
    let i = coord as usize;
    if i >= dim {
        dim - 1
    } else {
        i
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

    /// Small deterministic xorshift RNG (shared `ray_scene` test generator).
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
        fn point(&mut self, lo: f32, hi: f32) -> [f32; 3] {
            [self.range(lo, hi), self.range(lo, hi), self.range(lo, hi)]
        }
    }

    /// Independent normalize for oracle rays.
    fn o_norm(v: [f32; 3]) -> [f32; 3] {
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if len < 1e-12 {
            [0.0, 0.0, 1.0]
        } else {
            [v[0] / len, v[1] / len, v[2] / len]
        }
    }

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    /// A `verts`-vertex, `tris`-triangle `UV`-mapped mesh.
    fn random_mesh(rng: &mut Rng, verts: usize, tris: usize) -> TriangleMesh {
        let positions: Vec<[f32; 3]> = (0..verts).map(|_| rng.point(-5.0, 5.0)).collect();
        let uvs: Vec<[f32; 2]> = (0..verts)
            .map(|_| [rng.range(0.0, 1.0), rng.range(0.0, 1.0)])
            .collect();
        let mut indices = Vec::with_capacity(tris);
        while indices.len() < tris {
            let a = rng.next_u32() as usize % verts;
            let b = rng.next_u32() as usize % verts;
            let c = rng.next_u32() as usize % verts;
            if a == b || b == c || a == c {
                continue;
            }
            indices.push([a as u32, b as u32, c as u32]);
        }
        TriangleMesh::new(positions, vec![], uvs, indices).expect("valid mesh")
    }

    /// Independent alpha-aware nearest-hit oracle: gather every opaque triangle
    /// hit over the full ray interval and keep the smallest `t`.
    fn brute_closest(mesh: &AlphaMesh, ray: &Ray) -> Option<MeshHit> {
        let mut best: Option<MeshHit> = None;
        for tri in 0..mesh.mesh().triangle_count() {
            if let Some(hit) = mesh.mesh().intersect_triangle(tri, ray)
                && mesh.is_opaque_at(hit.uv)
            {
                let take = match &best {
                    None => true,
                    Some(b) => hit.t < b.t,
                };
                if take {
                    best = Some(hit);
                }
            }
        }
        best
    }

    #[test]
    fn texture_rejects_empty_and_mismatch() {
        assert_eq!(
            AlphaTexture::new(0, 4, vec![]),
            Err(AlphaTextureError::EmptyDimensions {
                width: 0,
                height: 4
            })
        );
        assert_eq!(
            AlphaTexture::new(2, 2, vec![1.0, 0.0, 1.0]),
            Err(AlphaTextureError::TexelCountMismatch {
                texels: 3,
                expected: 4,
            })
        );
        // Out-of-range texels are clamped into [0, 1].
        let tex = AlphaTexture::new(1, 2, vec![-0.5, 2.0]).unwrap();
        assert_eq!(tex.texel(0, 0), 0.0);
        assert_eq!(tex.texel(0, 1), 1.0);
    }

    #[test]
    fn sample_reproduces_texel_centers() {
        let mut rng = Rng::new(0x0A1B_2C3D);
        let (w, h) = (5usize, 3usize);
        let texels: Vec<f32> = (0..w * h).map(|_| rng.unit()).collect();
        let tex = AlphaTexture::new(w, h, texels).unwrap();
        for y in 0..h {
            for x in 0..w {
                let uv = [
                    (x as f32 + 0.5) / w as f32,
                    (y as f32 + 0.5) / h as f32,
                ];
                assert!(
                    approx(tex.sample(uv), tex.texel(x, y), 1e-6),
                    "center ({x},{y}) sample {} vs texel {}",
                    tex.sample(uv),
                    tex.texel(x, y)
                );
            }
        }
    }

    #[test]
    fn sample_bilinear_midpoint_and_clamp() {
        // 2x1 mask: left 0.0, right 1.0. The midpoint averages the two texels,
        // and sampling past the edges clamps to the end texels.
        let tex = AlphaTexture::new(2, 1, vec![0.0, 1.0]).unwrap();
        assert!(approx(tex.sample([0.5, 0.5]), 0.5, 1e-6));
        assert!(approx(tex.sample([0.0, 0.5]), 0.0, 1e-6));
        assert!(approx(tex.sample([1.0, 0.5]), 1.0, 1e-6));
        // Out-of-range UV clamps rather than wrapping.
        assert!(approx(tex.sample([-3.0, 0.5]), 0.0, 1e-6));
        assert!(approx(tex.sample([5.0, 0.5]), 1.0, 1e-6));
    }

    #[test]
    fn opaque_gate_matches_cutoff() {
        let tex = AlphaTexture::new(2, 1, vec![0.2, 0.8]).unwrap();
        let mesh = TriangleMesh::new(vec![], vec![], vec![], vec![]).unwrap();
        let alpha = AlphaMesh::new(mesh, tex, 0.5);
        assert_eq!(alpha.cutoff(), 0.5);
        // Left texel center (0.25, _) is 0.2 < 0.5 -> transparent.
        assert!(!alpha.is_opaque_at([0.25, 0.5]));
        // Right texel center (0.75, _) is 0.8 >= 0.5 -> opaque.
        assert!(alpha.is_opaque_at([0.75, 0.5]));
    }

    #[test]
    fn intersect_sees_through_transparent_front_face() {
        // Two parallel quads in the XY plane facing -Z, stacked in Z. The near
        // quad (z = 1) is fully transparent, the far quad (z = 3) fully opaque,
        // so a -Z ray from z = 5 must skip the near quad and hit the far one.
        let near_z = 1.0;
        let far_z = 3.0;
        let positions = vec![
            // near quad (triangles 0,1)
            [-1.0, -1.0, near_z],
            [1.0, -1.0, near_z],
            [1.0, 1.0, near_z],
            [-1.0, 1.0, near_z],
            // far quad (triangles 2,3)
            [-1.0, -1.0, far_z],
            [1.0, -1.0, far_z],
            [1.0, 1.0, far_z],
            [-1.0, 1.0, far_z],
        ];
        // All vertices map to the same UV corner so the whole quad reads one
        // mask texel: near quad -> texel (0,0), far quad -> texel (1,0).
        let uvs = vec![
            [0.25, 0.5],
            [0.25, 0.5],
            [0.25, 0.5],
            [0.25, 0.5],
            [0.75, 0.5],
            [0.75, 0.5],
            [0.75, 0.5],
            [0.75, 0.5],
        ];
        let indices = vec![
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
        ];
        let mesh = TriangleMesh::new(positions, vec![], uvs, indices).unwrap();
        // Mask: texel (0,0) = 0 (transparent), texel (1,0) = 0.8 (opaque at
        // the 0.5 cutoff, but transparent once the cutoff rises above 0.8).
        let tex = AlphaTexture::new(2, 1, vec![0.0, 0.8]).unwrap();
        let alpha = AlphaMesh::new(mesh.clone(), tex.clone(), 0.5);
        let bvh = AlphaMeshBvh::build(mesh, tex, 0.5);

        let ray = Ray::new([0.0, 0.0, 5.0], [0.0, 0.0, -1.0], 1e-3, 1e6);
        let hit = alpha.intersect(&ray).expect("ray must reach the far quad");
        assert!(approx(hit.t, 5.0 - far_z, 1e-4), "t was {}", hit.t);
        let bhit = bvh.closest_hit(&ray).expect("bvh must reach the far quad");
        assert!(approx(bhit.t, 5.0 - far_z, 1e-4));
        assert_eq!(hit.triangle, bhit.triangle);
        assert!(alpha.any_hit(&ray));
        assert!(bvh.any_hit(&ray));

        // With a cutoff above the opaque texel's value (0.8) nothing survives.
        let opaque_blocked = AlphaMesh::new(alpha.mesh().clone(), alpha.alpha().clone(), 0.9);
        assert!(opaque_blocked.intersect(&ray).is_none());
        assert!(!opaque_blocked.any_hit(&ray));
    }

    #[test]
    fn bvh_matches_alpha_aware_brute_force() {
        let mut rng = Rng::new(0xA1F0_C0DE);
        let mesh = random_mesh(&mut rng, 40, 110);
        let (w, h) = (8usize, 8usize);
        let texels: Vec<f32> = (0..w * h).map(|_| rng.unit()).collect();
        let tex = AlphaTexture::new(w, h, texels).unwrap();
        let cutoff = 0.5;
        let alpha = AlphaMesh::new(mesh.clone(), tex.clone(), cutoff);
        let bvh = AlphaMeshBvh::build(mesh, tex, cutoff);
        assert_eq!(bvh.primitive_count(), alpha.mesh().triangle_count());

        let mut shared = 0usize;
        for _ in 0..6000 {
            let origin = rng.point(-8.0, 8.0);
            let dir = o_norm(rng.point(-1.0, 1.0));
            let ray = Ray::new(origin, dir, 1e-3, 1e6);
            let brute = brute_closest(&alpha, &ray);
            let fast = bvh.closest_hit(&ray);
            match (brute, fast) {
                (None, None) => {}
                (Some(a), Some(b)) => {
                    assert!(approx(a.t, b.t, 1e-3), "t {} vs {}", a.t, b.t);
                    if a.triangle == b.triangle {
                        assert_eq!(a.front_face, b.front_face);
                        for (pa, pb) in a.position.iter().zip(b.position.iter()) {
                            assert!(approx(*pa, *pb, 2e-3));
                        }
                    }
                    shared += 1;
                }
                (a, b) => panic!("brute/bvh disagree: {a:?} vs {b:?}"),
            }
            assert_eq!(brute_closest(&alpha, &ray).is_some(), bvh.any_hit(&ray));
            assert_eq!(alpha.any_hit(&ray), bvh.any_hit(&ray));
        }
        assert!(shared > 50, "too few shared opaque hits: {shared}");
    }

    #[test]
    fn empty_bvh_never_hits() {
        let mesh = TriangleMesh::new(vec![], vec![], vec![], vec![]).unwrap();
        let tex = AlphaTexture::new(1, 1, vec![1.0]).unwrap();
        let bvh = AlphaMeshBvh::build(mesh, tex, 0.5);
        assert!(bvh.is_empty());
        assert_eq!(bvh.primitive_count(), 0);
        assert_eq!(bvh.bounds(), Aabb::empty());
        let ray = Ray::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.0, 1e6);
        assert!(bvh.closest_hit(&ray).is_none());
        assert!(!bvh.any_hit(&ray));
    }
}
