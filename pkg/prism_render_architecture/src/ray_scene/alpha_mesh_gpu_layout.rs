//! `GPU`-ready flat buffer layout for the [`AlphaMeshBvh`], plus a packed
//! traversal that reads those buffers and reproduces the alpha-gated in-memory
//! walk bit-for-bit.
//!
//! The alpha-tested mesh is the proven indexed [`super::triangle_mesh`]
//! primitive with one extra gate: a candidate hit survives only when the mask
//! sampled at its interpolated `UV` is at or above the cutoff. The on-device
//! representation therefore reuses the full [`GpuTriangleMeshBvhBuffers`]
//! layout verbatim (nodes, vertices, indices, order, pool flags) and adds just
//! two things a kernel needs to apply the hardware any-hit alpha callback: the
//! row-major mask texels and the cutoff.
//!
//! Nothing about the acceleration structure changes — the mask does not move a
//! triangle — so [`GpuAlphaMeshBvhBuffers::closest_hit`]/[`GpuAlphaMeshBvhBuffers::any_hit`]
//! mirror [`AlphaMeshBvh`] exactly: the same slab rejection, the same near/far
//! child ordering by split-axis sign, the same running `t_max` shrink, and the
//! same [`super::triangle_mesh::TriangleMesh::intersect_triangle`] test, with
//! the alpha gate re-decoded through [`AlphaTexture::sample`] so a sub-cutoff
//! hit neither shrinks the ray nor satisfies an occlusion query. Decoding the
//! mask routes through [`AlphaTexture::new`], whose `[0, 1]` clamp is idempotent
//! on already-clamped texels, so the mask round-trips bit-for-bit.
//!
//! Encoding stays dependency-free: the mesh half is the shared `Vec<u32>`
//! layout, the mask texels are stored as their `to_bits` pattern, and the small
//! header (`width`, `height`, cutoff bits, pad) is a multiple of four words so
//! every record stays 16-byte aligned.

use super::alpha_mesh::{AlphaMeshBvh, AlphaTexture};
use super::bvh::Aabb;
use super::gpu_layout::NODE_WORDS;
use super::traversal::Ray;
use super::triangle_mesh::{MeshHit, TriangleMesh};
use super::triangle_mesh_gpu_layout::GpuTriangleMeshBvhBuffers;

/// `u32` words in the packed alpha header (16 bytes, 16-byte aligned).
///
/// Layout: mask width (0), mask height (1), cutoff `to_bits` (2), padding (3).
pub const ALPHA_HEADER_WORDS: usize = 4;

/// Reads three `from_bits` words at `words[base..base + 3]` back into `[f32; 3]`.
fn read_vec3(words: &[u32], base: usize) -> [f32; 3] {
    [
        f32::from_bits(words[base]),
        f32::from_bits(words[base + 1]),
        f32::from_bits(words[base + 2]),
    ]
}

/// Pops the top node index off the traversal stack, or `None` when empty.
fn pop(stack: &mut [u32; 64], sp: &mut usize) -> Option<u32> {
    if *sp == 0 {
        None
    } else {
        *sp -= 1;
        Some(stack[*sp])
    }
}

/// Flattened, `GPU`-uploadable buffers for a single [`AlphaMeshBvh`].
///
/// `mesh` is the verbatim triangle-mesh packing (nodes with the shared
/// [`NODE_WORDS`] stride, vertices, indices, order, pool flags); `alpha` holds
/// the row-major mask texels as `to_bits` words; and `header` carries the mask
/// dimensions and the cutoff. A kernel binds `mesh` exactly as a plain
/// triangle-mesh `BLAS` and additionally binds `alpha` + `header` to gate every
/// candidate hit through the mask, walking them exactly as
/// [`GpuAlphaMeshBvhBuffers::closest_hit`] does here.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct GpuAlphaMeshBvhBuffers {
    /// The underlying triangle-mesh buffers, reused without duplication.
    pub mesh: GpuTriangleMeshBvhBuffers,
    /// Row-major mask texels, each stored as its `to_bits` pattern.
    pub alpha: Vec<u32>,
    /// Packed header: `[width, height, cutoff_bits, pad]`.
    pub header: [u32; ALPHA_HEADER_WORDS],
}

impl GpuAlphaMeshBvhBuffers {
    /// Serializes a built [`AlphaMeshBvh`] into flat buffers.
    ///
    /// The triangle-mesh half is packed by [`GpuTriangleMeshBvhBuffers::from_bvh`]
    /// so nodes, vertices, indices, and the `order` map are preserved exactly.
    /// The mask texels are copied verbatim (already clamped to `[0, 1]` by
    /// [`AlphaTexture::new`]) as `to_bits` words, and the header records the
    /// mask dimensions and the cutoff bits, so decoding rebuilds a bit-identical
    /// primitive.
    #[must_use]
    pub fn from_bvh(bvh: &AlphaMeshBvh) -> Self {
        let mesh = GpuTriangleMeshBvhBuffers::from_bvh(bvh.triangle_bvh());
        let tex = bvh.alpha();
        let alpha = tex.texels().iter().map(|a| a.to_bits()).collect();
        let header = [
            tex.width() as u32,
            tex.height() as u32,
            bvh.cutoff().to_bits(),
            0,
        ];
        Self {
            mesh,
            alpha,
            header,
        }
    }

    /// Number of packed nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.mesh.node_count()
    }

    /// Number of packed triangles.
    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.mesh.triangle_count()
    }

    /// Number of packed vertices.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.mesh.vertex_count()
    }

    /// Mask columns, as recorded in the header.
    #[must_use]
    pub fn alpha_width(&self) -> usize {
        self.header[0] as usize
    }

    /// Mask rows, as recorded in the header.
    #[must_use]
    pub fn alpha_height(&self) -> usize {
        self.header[1] as usize
    }

    /// The alpha cutoff decoded from the header.
    #[must_use]
    pub fn cutoff(&self) -> f32 {
        f32::from_bits(self.header[2])
    }

    /// True when there are no nodes to traverse.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.mesh.is_empty()
    }

    /// Decoded bounds of packed node `i`.
    fn node_bounds(&self, i: usize) -> Aabb {
        let b = i * NODE_WORDS;
        Aabb::new(
            read_vec3(&self.mesh.nodes, b),
            read_vec3(&self.mesh.nodes, b + 3),
        )
    }

    /// Rebuilds the triangle mesh from the packed buffers (bit-identical to the
    /// source mesh; see [`GpuTriangleMeshBvhBuffers::decode_mesh`]).
    #[must_use]
    pub fn decode_mesh(&self) -> TriangleMesh {
        self.mesh.decode_mesh()
    }

    /// Rebuilds the mask from the header + `alpha` words.
    ///
    /// Decoding routes through [`AlphaTexture::new`] with the exact stored
    /// texels; the `[0, 1]` clamp is idempotent on the already-clamped source,
    /// so the reconstructed mask equals the original bit-for-bit and
    /// [`AlphaTexture::sample`] reproduces every gate decision.
    #[must_use]
    pub fn decode_alpha(&self) -> AlphaTexture {
        let texels = self.alpha.iter().map(|&w| f32::from_bits(w)).collect();
        AlphaTexture::new(self.alpha_width(), self.alpha_height(), texels)
            .expect("packed header encodes a valid alpha mask")
    }

    /// Nearest opaque intersection along `ray` by walking the packed buffers.
    ///
    /// Decodes the mesh and mask, then mirrors [`AlphaMeshBvh::closest_hit`]
    /// exactly — same slab rejection, same near/far child ordering by split-axis
    /// sign, same running `t_max` shrink, and the same `order`-indexed
    /// [`TriangleMesh::intersect_triangle`] test gated by
    /// [`AlphaTexture::sample`] at or above the cutoff — so the result is
    /// bit-for-bit identical and a `GPU` kernel binding these buffers can be
    /// diffed against it.
    #[must_use]
    pub fn closest_hit(&self, ray: &Ray) -> Option<MeshHit> {
        if self.mesh.nodes.is_empty() {
            return None;
        }
        let mesh = self.decode_mesh();
        let alpha = self.decode_alpha();
        let cutoff = self.cutoff();
        let mut ray = *ray;
        let mut best: Option<MeshHit> = None;

        let mut stack = [0u32; 64];
        let mut sp = 0usize;
        let mut node_index = 0u32;
        loop {
            let ni = node_index as usize;
            let bounds = self.node_bounds(ni);
            if ray.aabb_interval(&bounds, ray.t_min(), ray.t_max()).is_some() {
                let base = ni * NODE_WORDS;
                let primitive_count = self.mesh.nodes[base + 8];
                if primitive_count > 0 {
                    let start = self.mesh.nodes[base + 6] as usize;
                    let end = start + primitive_count as usize;
                    for &slot in &self.mesh.order[start..end] {
                        if let Some(hit) = mesh.intersect_triangle(slot as usize, &ray)
                            && alpha.sample(hit.uv) >= cutoff
                        {
                            ray = Ray::new(ray.origin(), ray.direction(), ray.t_min(), hit.t);
                            best = Some(hit);
                        }
                    }
                    match pop(&mut stack, &mut sp) {
                        Some(n) => node_index = n,
                        None => break,
                    }
                } else {
                    let first_child = node_index + 1;
                    let second_child = self.mesh.nodes[base + 7];
                    let axis = self.mesh.nodes[base + 9] as usize;
                    let neg = ray.direction()[axis] < 0.0;
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

    /// True when any opaque texel of any triangle intersects `ray`; mirrors
    /// [`AlphaMeshBvh::any_hit`].
    #[must_use]
    pub fn any_hit(&self, ray: &Ray) -> bool {
        if self.mesh.nodes.is_empty() {
            return false;
        }
        let mesh = self.decode_mesh();
        let alpha = self.decode_alpha();
        let cutoff = self.cutoff();
        let mut stack = [0u32; 64];
        let mut sp = 0usize;
        let mut node_index = 0u32;
        loop {
            let ni = node_index as usize;
            let bounds = self.node_bounds(ni);
            if ray.aabb_interval(&bounds, ray.t_min(), ray.t_max()).is_some() {
                let base = ni * NODE_WORDS;
                let primitive_count = self.mesh.nodes[base + 8];
                if primitive_count > 0 {
                    let start = self.mesh.nodes[base + 6] as usize;
                    let end = start + primitive_count as usize;
                    for &slot in &self.mesh.order[start..end] {
                        if let Some(hit) = mesh.intersect_triangle(slot as usize, ray)
                            && alpha.sample(hit.uv) >= cutoff
                        {
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
                        stack[sp] = self.mesh.nodes[base + 7];
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

    /// Builds a `UV`-mapped triangle mesh with random positions and texcoords.
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

    /// Builds a random mask with texels across `[0, 1]`.
    fn random_mask(rng: &mut Rng, width: usize, height: usize) -> AlphaTexture {
        let texels: Vec<f32> = (0..width * height).map(|_| rng.range(0.0, 1.0)).collect();
        AlphaTexture::new(width, height, texels).expect("valid mask")
    }

    #[test]
    fn header_stride_is_sixteen_byte_aligned() {
        assert_eq!(ALPHA_HEADER_WORDS % 4, 0);
        let mut rng = Rng::new(0xABCD_1234);
        let mesh = random_mesh(&mut rng, 30, 70);
        let mask = random_mask(&mut rng, 8, 6);
        let bvh = AlphaMeshBvh::build(mesh, mask, 0.5);
        let gpu = GpuAlphaMeshBvhBuffers::from_bvh(&bvh);
        assert_eq!(gpu.alpha.len(), gpu.alpha_width() * gpu.alpha_height());
    }

    #[test]
    fn from_bvh_preserves_shape_dims_and_cutoff() {
        let mut rng = Rng::new(0xABCD_1234);
        let mesh = random_mesh(&mut rng, 40, 90);
        let mask = random_mask(&mut rng, 12, 9);
        let bvh = AlphaMeshBvh::build(mesh, mask, 0.37);
        let gpu = GpuAlphaMeshBvhBuffers::from_bvh(&bvh);
        assert_eq!(gpu.node_count(), bvh.node_count());
        assert_eq!(gpu.triangle_count(), bvh.mesh().triangle_count());
        assert_eq!(gpu.vertex_count(), bvh.mesh().vertex_count());
        assert_eq!(gpu.alpha_width(), 12);
        assert_eq!(gpu.alpha_height(), 9);
        assert_eq!(gpu.alpha.len(), 12 * 9);
        assert_eq!(gpu.cutoff().to_bits(), bvh.cutoff().to_bits());
    }

    #[test]
    fn decoded_mesh_mask_and_cutoff_round_trip() {
        let mut rng = Rng::new(0x5EED_F00D);
        let mesh = random_mesh(&mut rng, 24, 50);
        let mask = random_mask(&mut rng, 10, 7);
        let bvh = AlphaMeshBvh::build(mesh, mask, 0.42);
        let gpu = GpuAlphaMeshBvhBuffers::from_bvh(&bvh);
        assert_eq!(gpu.decode_mesh(), *bvh.mesh());
        assert_eq!(gpu.decode_alpha(), *bvh.alpha());
        assert_eq!(gpu.cutoff().to_bits(), bvh.cutoff().to_bits());
    }

    #[test]
    fn empty_buffers_never_hit() {
        let mesh = TriangleMesh::new(vec![], vec![], vec![], vec![]).unwrap();
        let mask = AlphaTexture::new(1, 1, vec![1.0]).unwrap();
        let bvh = AlphaMeshBvh::build(mesh, mask, 0.5);
        let gpu = GpuAlphaMeshBvhBuffers::from_bvh(&bvh);
        assert!(gpu.is_empty());
        assert_eq!(gpu.node_count(), 0);
        assert_eq!(gpu.triangle_count(), 0);
        assert_eq!(gpu.vertex_count(), 0);
        let ray = Ray::infinite([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
        assert!(gpu.closest_hit(&ray).is_none());
        assert!(!gpu.any_hit(&ray));
    }

    #[test]
    fn packed_walk_matches_in_memory_bvh_bit_for_bit() {
        for seed in [0xABCD_1234u64, 0x5EED_F00D] {
            let mut rng = Rng::new(seed);
            let mesh = random_mesh(&mut rng, 64, 160);
            let mask = random_mask(&mut rng, 16, 16);
            let bvh = AlphaMeshBvh::build(mesh, mask, 0.5);
            let gpu = GpuAlphaMeshBvhBuffers::from_bvh(&bvh);

            let mut shared = 0usize;
            for _ in 0..2_000 {
                let origin = rng.point(-10.0, 10.0);
                let dir = rng.point(-1.0, 1.0);
                if dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2] < 1e-6 {
                    continue;
                }
                let ray = Ray::infinite(origin, dir);

                let cpu = bvh.closest_hit(&ray);
                let packed = gpu.closest_hit(&ray);
                match (cpu, packed) {
                    (None, None) => {}
                    (Some(c), Some(p)) => {
                        assert_eq!(c.triangle, p.triangle);
                        assert_eq!(c.t.to_bits(), p.t.to_bits(), "t bits differ");
                        assert_eq!(c.u.to_bits(), p.u.to_bits(), "u bits differ");
                        assert_eq!(c.v.to_bits(), p.v.to_bits(), "v bits differ");
                        assert_eq!(c.front_face, p.front_face);
                        for k in 0..3 {
                            assert_eq!(
                                c.position[k].to_bits(),
                                p.position[k].to_bits(),
                                "position[{k}] bits differ"
                            );
                            assert_eq!(
                                c.normal[k].to_bits(),
                                p.normal[k].to_bits(),
                                "normal[{k}] bits differ"
                            );
                        }
                        assert_eq!(c.uv[0].to_bits(), p.uv[0].to_bits(), "uv.x bits differ");
                        assert_eq!(c.uv[1].to_bits(), p.uv[1].to_bits(), "uv.y bits differ");
                        shared += 1;
                    }
                    (c, p) => panic!("hit disagreement: {c:?} vs {p:?}"),
                }
                assert_eq!(gpu.any_hit(&ray), bvh.any_hit(&ray));
            }
            assert!(shared > 50, "too few shared hits for seed {seed:#x}: {shared}");
        }
    }
}
