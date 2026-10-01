//! `GPU`-ready flat buffer layout for the indexed [`TriangleMeshBvh`], plus a
//! packed traversal that reads those buffers and reproduces the in-memory walk
//! bit-for-bit.
//!
//! This is the indexed-mesh counterpart of [`super::shaded_triangle_gpu_layout`]:
//! the `CPU` [`TriangleMeshBvh`] is convenient for building and testing, but a
//! compute kernel binds plain storage buffers. This module pins the exact word
//! layout a `BLAS` traversal kernel consumes and proves — with the same
//! [`TriangleMesh::intersect_triangle`] and [`Ray::aabb_interval`] arithmetic
//! the in-memory walk uses — that a walk over the flattened buffers returns the
//! identical hit, including the interpolated shading normal and texture
//! coordinate.
//!
//! Four buffers mirror the mesh's shared-pool layout: `nodes` reuse the shared
//! [`NODE_WORDS`] record; `vertices` pack position + normal + `UV` per vertex;
//! `indices` pack the vertex triples in their **original** order; and `order`
//! maps each `BVH` primitive slot back to its original triangle index (exactly
//! as [`TriangleMeshBvh::order`] does). The `has_normals`/`has_uvs` flags record
//! whether the mesh carried those pools so decoding rebuilds an identical mesh
//! (empty pools reproduce geometric shading / barycentric `UV` fallback).
//! Encoding is dependency-free: every buffer is a `Vec<u32>` with `f32` fields
//! stored as their `to_bits` pattern, and both the vertex and index strides are
//! multiples of four words so each record stays 16-byte aligned.

use super::bvh::Aabb;
use super::gpu_layout::NODE_WORDS;
use super::traversal::Ray;
use super::triangle_mesh::{MeshHit, TriangleMesh, TriangleMeshBvh};

/// `u32` words per packed vertex (32 bytes, 16-byte aligned).
///
/// Layout: position (0..3), shading normal (3..6), texture coordinate (6..8).
pub const MESH_VERTEX_WORDS: usize = 8;

/// `u32` words per packed index triple (16 bytes, 16-byte aligned).
///
/// Layout: vertex indices (0..3), padding (3).
pub const MESH_INDEX_WORDS: usize = 4;

/// Writes an `[f32; 3]` as three `to_bits` words at `out[base..base + 3]`.
fn write_vec3(out: &mut [u32], base: usize, v: [f32; 3]) {
    out[base] = v[0].to_bits();
    out[base + 1] = v[1].to_bits();
    out[base + 2] = v[2].to_bits();
}

/// Reads three `from_bits` words at `words[base..base + 3]` back into `[f32; 3]`.
fn read_vec3(words: &[u32], base: usize) -> [f32; 3] {
    [
        f32::from_bits(words[base]),
        f32::from_bits(words[base + 1]),
        f32::from_bits(words[base + 2]),
    ]
}

/// Reads two `from_bits` words at `words[base..base + 2]` back into `[f32; 2]`.
fn read_vec2(words: &[u32], base: usize) -> [f32; 2] {
    [f32::from_bits(words[base]), f32::from_bits(words[base + 1])]
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

/// Flattened, `GPU`-uploadable buffers for a single [`TriangleMeshBvh`].
///
/// `nodes` are packed with the shared [`NODE_WORDS`] stride; `vertices` with the
/// [`MESH_VERTEX_WORDS`] stride; `indices` with the [`MESH_INDEX_WORDS`] stride
/// in the mesh's original triangle order; and `order` is the `BVH`-slot →
/// original-triangle-index map that leaf primitive ranges index into. A kernel
/// binds them as read-only storage and walks them exactly as
/// [`GpuTriangleMeshBvhBuffers::closest_hit`] does here.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct GpuTriangleMeshBvhBuffers {
    /// Packed node records, [`NODE_WORDS`] words each, in depth-first order.
    pub nodes: Vec<u32>,
    /// Packed vertex records, [`MESH_VERTEX_WORDS`] words each.
    pub vertices: Vec<u32>,
    /// Packed index triples, [`MESH_INDEX_WORDS`] words each, original order.
    pub indices: Vec<u32>,
    /// `BVH`-slot → original-triangle-index map; leaf ranges index this.
    pub order: Vec<u32>,
    /// Whether the source mesh carried per-vertex shading normals.
    pub has_normals: bool,
    /// Whether the source mesh carried per-vertex texture coordinates.
    pub has_uvs: bool,
}

impl GpuTriangleMeshBvhBuffers {
    /// Serializes a built [`TriangleMeshBvh`] into flat buffers.
    ///
    /// Node, vertex, and index order are preserved exactly, and the `order` map
    /// is copied verbatim, so the packed leaf ranges index `order` the same way
    /// the in-memory [`TriangleMeshBvh`] does. When the mesh lacks a normal or
    /// `UV` pool the corresponding vertex words are zero-filled and the matching
    /// flag is cleared, so decoding rebuilds the empty pool.
    #[must_use]
    pub fn from_bvh(bvh: &TriangleMeshBvh) -> Self {
        let mesh = bvh.mesh();
        let mut nodes = vec![0u32; bvh.nodes().len() * NODE_WORDS];
        for (i, node) in bvh.nodes().iter().enumerate() {
            let b = i * NODE_WORDS;
            write_vec3(&mut nodes, b, node.bounds.min);
            write_vec3(&mut nodes, b + 3, node.bounds.max);
            nodes[b + 6] = node.first_primitive;
            nodes[b + 7] = node.second_child;
            nodes[b + 8] = u32::from(node.primitive_count);
            nodes[b + 9] = u32::from(node.axis);
        }

        let has_normals = mesh.has_normals();
        let has_uvs = mesh.has_uvs();
        let mut vertices = vec![0u32; mesh.vertex_count() * MESH_VERTEX_WORDS];
        for (i, &pos) in mesh.positions().iter().enumerate() {
            let b = i * MESH_VERTEX_WORDS;
            write_vec3(&mut vertices, b, pos);
            if has_normals {
                write_vec3(&mut vertices, b + 3, mesh.normals()[i]);
            }
            if has_uvs {
                let uv = mesh.uvs()[i];
                vertices[b + 6] = uv[0].to_bits();
                vertices[b + 7] = uv[1].to_bits();
            }
        }

        let mut indices = vec![0u32; mesh.triangle_count() * MESH_INDEX_WORDS];
        for (i, tri) in mesh.indices().iter().enumerate() {
            let b = i * MESH_INDEX_WORDS;
            indices[b] = tri[0];
            indices[b + 1] = tri[1];
            indices[b + 2] = tri[2];
        }

        Self {
            nodes,
            vertices,
            indices,
            order: bvh.order().to_vec(),
            has_normals,
            has_uvs,
        }
    }

    /// Number of packed nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len() / NODE_WORDS
    }

    /// Number of packed vertices.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.vertices.len() / MESH_VERTEX_WORDS
    }

    /// Number of packed triangles.
    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / MESH_INDEX_WORDS
    }

    /// True when there are no nodes to traverse.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Decoded bounds of packed node `i`.
    fn node_bounds(&self, i: usize) -> Aabb {
        let b = i * NODE_WORDS;
        Aabb::new(read_vec3(&self.nodes, b), read_vec3(&self.nodes, b + 3))
    }

    /// Rebuilds the shared-pool [`TriangleMesh`] from the packed buffers.
    ///
    /// Decoding routes through [`TriangleMesh::new`] with the exact stored words
    /// (positions, normals, and uvs verbatim; empty pools when the flags are
    /// clear), so the reconstructed mesh is bit-identical to the one the `CPU`
    /// [`TriangleMeshBvh`] holds and `intersect_triangle` reproduces every hit.
    #[must_use]
    pub fn decode_mesh(&self) -> TriangleMesh {
        let vertex_count = self.vertex_count();
        let mut positions = Vec::with_capacity(vertex_count);
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        if self.has_normals {
            normals.reserve(vertex_count);
        }
        if self.has_uvs {
            uvs.reserve(vertex_count);
        }
        for i in 0..vertex_count {
            let b = i * MESH_VERTEX_WORDS;
            positions.push(read_vec3(&self.vertices, b));
            if self.has_normals {
                normals.push(read_vec3(&self.vertices, b + 3));
            }
            if self.has_uvs {
                uvs.push(read_vec2(&self.vertices, b + 6));
            }
        }
        let mut indices = Vec::with_capacity(self.triangle_count());
        for i in 0..self.triangle_count() {
            let b = i * MESH_INDEX_WORDS;
            indices.push([self.indices[b], self.indices[b + 1], self.indices[b + 2]]);
        }
        TriangleMesh::new(positions, normals, uvs, indices)
            .expect("packed buffers encode a valid mesh")
    }

    /// Nearest intersection along `ray` by walking the packed buffers.
    ///
    /// Decodes the mesh, then mirrors [`TriangleMeshBvh::closest_hit`] exactly —
    /// same slab rejection, same near/far child ordering by split-axis sign,
    /// same running `t_max` shrink, and the same `order`-indexed
    /// [`TriangleMesh::intersect_triangle`] test — so the result is bit-for-bit
    /// identical and a `GPU` kernel binding these buffers can be diffed against
    /// it.
    #[must_use]
    pub fn closest_hit(&self, ray: &Ray) -> Option<MeshHit> {
        if self.nodes.is_empty() {
            return None;
        }
        let mesh = self.decode_mesh();
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
                let primitive_count = self.nodes[base + 8];
                if primitive_count > 0 {
                    let start = self.nodes[base + 6] as usize;
                    let end = start + primitive_count as usize;
                    for &slot in &self.order[start..end] {
                        if let Some(hit) = mesh.intersect_triangle(slot as usize, &ray) {
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
                    let second_child = self.nodes[base + 7];
                    let axis = self.nodes[base + 9] as usize;
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

    /// True when *any* triangle intersects `ray`; mirrors
    /// [`TriangleMeshBvh::any_hit`].
    #[must_use]
    pub fn any_hit(&self, ray: &Ray) -> bool {
        if self.nodes.is_empty() {
            return false;
        }
        let mesh = self.decode_mesh();
        let mut stack = [0u32; 64];
        let mut sp = 0usize;
        let mut node_index = 0u32;
        loop {
            let ni = node_index as usize;
            let bounds = self.node_bounds(ni);
            if ray.aabb_interval(&bounds, ray.t_min(), ray.t_max()).is_some() {
                let base = ni * NODE_WORDS;
                let primitive_count = self.nodes[base + 8];
                if primitive_count > 0 {
                    let start = self.nodes[base + 6] as usize;
                    let end = start + primitive_count as usize;
                    for &slot in &self.order[start..end] {
                        if mesh.intersect_triangle(slot as usize, ray).is_some() {
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
                        stack[sp] = self.nodes[base + 7];
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
        fn point(&mut self, lo: f32, hi: f32) -> [f32; 3] {
            [self.range(lo, hi), self.range(lo, hi), self.range(lo, hi)]
        }
    }

    fn norm(v: [f32; 3]) -> [f32; 3] {
        let inv = 1.0 / (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] * inv, v[1] * inv, v[2] * inv]
    }

    /// Random indexed mesh; `normals`/`uvs` toggle whether those pools exist.
    fn random_mesh(
        rng: &mut Rng,
        verts: usize,
        tris: usize,
        normals: bool,
        uvs: bool,
    ) -> TriangleMesh {
        let positions: Vec<[f32; 3]> = (0..verts).map(|_| rng.point(-6.0, 6.0)).collect();
        let normal_pool: Vec<[f32; 3]> = if normals {
            (0..verts).map(|_| norm(rng.point(-1.0, 1.0))).collect()
        } else {
            Vec::new()
        };
        let uv_pool: Vec<[f32; 2]> = if uvs {
            (0..verts)
                .map(|_| [rng.range(0.0, 1.0), rng.range(0.0, 1.0)])
                .collect()
        } else {
            Vec::new()
        };
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
        TriangleMesh::new(positions, normal_pool, uv_pool, indices).expect("valid mesh")
    }

    #[test]
    fn strides_are_sixteen_byte_aligned() {
        assert_eq!(MESH_VERTEX_WORDS % 4, 0);
        assert_eq!(MESH_INDEX_WORDS % 4, 0);
    }

    #[test]
    fn from_bvh_preserves_counts_and_buffer_shape() {
        let mut rng = Rng::new(0xABCD_1234);
        let mesh = random_mesh(&mut rng, 40, 90, true, true);
        let bvh = TriangleMeshBvh::build(mesh);
        let gpu = GpuTriangleMeshBvhBuffers::from_bvh(&bvh);
        assert_eq!(gpu.node_count(), bvh.node_count());
        assert_eq!(gpu.triangle_count(), bvh.mesh().triangle_count());
        assert_eq!(gpu.vertex_count(), bvh.mesh().vertex_count());
        assert_eq!(gpu.order, bvh.order());
        assert_eq!(gpu.nodes.len(), gpu.node_count() * NODE_WORDS);
        assert_eq!(gpu.vertices.len(), gpu.vertex_count() * MESH_VERTEX_WORDS);
        assert_eq!(gpu.indices.len(), gpu.triangle_count() * MESH_INDEX_WORDS);
    }

    #[test]
    fn decoded_mesh_round_trips_every_pool_combination() {
        let mut rng = Rng::new(0xABCD_1234);
        for (normals, uvs) in [(true, true), (true, false), (false, true), (false, false)] {
            let mesh = random_mesh(&mut rng, 24, 50, normals, uvs);
            let bvh = TriangleMeshBvh::build(mesh.clone());
            let gpu = GpuTriangleMeshBvhBuffers::from_bvh(&bvh);
            assert_eq!(gpu.has_normals, normals);
            assert_eq!(gpu.has_uvs, uvs);
            // The decoded mesh must equal the original bit-for-bit (PartialEq
            // over the verbatim pools), including empty pools when absent.
            assert_eq!(gpu.decode_mesh(), mesh);
        }
    }

    #[test]
    fn empty_buffers_never_hit() {
        let mesh = TriangleMesh::new(vec![], vec![], vec![], vec![]).unwrap();
        let bvh = TriangleMeshBvh::build(mesh);
        let gpu = GpuTriangleMeshBvhBuffers::from_bvh(&bvh);
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
            let mesh = random_mesh(&mut rng, 64, 160, true, true);
            let bvh = TriangleMeshBvh::build(mesh);
            let gpu = GpuTriangleMeshBvhBuffers::from_bvh(&bvh);

            let mut shared = 0usize;
            for _ in 0..3_000 {
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
            assert!(shared > 100, "too few shared hits for seed {seed:#x}: {shared}");
        }
    }
}
