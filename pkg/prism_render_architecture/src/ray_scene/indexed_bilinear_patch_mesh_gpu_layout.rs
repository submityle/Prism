//! `GPU`-ready flat buffer layout for the indexed
//! [`IndexedBilinearPatchMeshBvh`], plus a packed traversal that reads those
//! buffers and reproduces the in-memory walk bit-for-bit.
//!
//! This is the quad-patch counterpart of [`super::triangle_mesh_gpu_layout`]:
//! the `CPU` [`IndexedBilinearPatchMeshBvh`] is convenient for building and
//! testing, but a compute kernel binds plain storage buffers. This module pins
//! the exact word layout a `BLAS` traversal kernel consumes and proves — with
//! the same [`IndexedBilinearPatchMesh::intersect_patch`] and
//! [`Ray::aabb_interval`] arithmetic the in-memory walk uses — that a walk over
//! the flattened buffers returns the identical hit, including the interpolated
//! shading normal and texture coordinate.
//!
//! Four buffers mirror the mesh's shared-pool layout: `nodes` reuse the shared
//! [`NODE_WORDS`] record; `vertices` pack position + shading normal + `UV` per
//! vertex; `indices` pack the four corner ids per patch in the mesh's
//! **original** patch order; and `order` maps each `BVH` primitive slot back to
//! its original patch index (exactly as [`IndexedBilinearPatchMeshBvh::order`]
//! does). Because the mesh always carries a full normal and `UV` pool, no
//! presence flags are needed. Encoding is dependency-free: every buffer is a
//! `Vec<u32>` with `f32` fields stored as their `to_bits` pattern, and both the
//! vertex and index strides are multiples of four words so each record stays
//! 16-byte aligned.

use super::bvh::Aabb;
use super::gpu_layout::NODE_WORDS;
use super::indexed_bilinear_patch_mesh::{IndexedBilinearPatchMesh, IndexedBilinearPatchMeshBvh};
use super::shaded_bilinear_patch::ShadedBilinearPatchHit;
use super::traversal::Ray;

/// `u32` words per packed vertex (32 bytes, 16-byte aligned).
///
/// Layout: position (0..3), shading normal (3..6), texture coordinate (6..8).
pub const PATCH_MESH_VERTEX_WORDS: usize = 8;

/// `u32` words per packed patch index record (16 bytes, 16-byte aligned).
///
/// Layout: corner vertex indices `[i00, i10, i11, i01]` (0..4).
pub const PATCH_MESH_INDEX_WORDS: usize = 4;

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

/// Flattened, `GPU`-uploadable buffers for a single
/// [`IndexedBilinearPatchMeshBvh`].
///
/// `nodes` are packed with the shared [`NODE_WORDS`] stride; `vertices` with the
/// [`PATCH_MESH_VERTEX_WORDS`] stride; `indices` with the
/// [`PATCH_MESH_INDEX_WORDS`] stride in the mesh's original patch order; and
/// `order` is the `BVH`-slot → original-patch-index map that leaf primitive
/// ranges index into. A kernel binds them as read-only storage and walks them
/// exactly as [`GpuIndexedBilinearPatchMeshBvhBuffers::closest_hit`] does here.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct GpuIndexedBilinearPatchMeshBvhBuffers {
    /// Packed node records, [`NODE_WORDS`] words each, in depth-first order.
    pub nodes: Vec<u32>,
    /// Packed vertex records, [`PATCH_MESH_VERTEX_WORDS`] words each.
    pub vertices: Vec<u32>,
    /// Packed patch index records, [`PATCH_MESH_INDEX_WORDS`] words each,
    /// original patch order.
    pub indices: Vec<u32>,
    /// `BVH`-slot → original-patch-index map; leaf ranges index this.
    pub order: Vec<u32>,
}

impl GpuIndexedBilinearPatchMeshBvhBuffers {
    /// Serializes a built [`IndexedBilinearPatchMeshBvh`] into flat buffers.
    ///
    /// Node, vertex, and index order are preserved exactly, and the `order` map
    /// is copied verbatim, so the packed leaf ranges index `order` the same way
    /// the in-memory [`IndexedBilinearPatchMeshBvh`] does.
    #[must_use]
    pub fn from_bvh(bvh: &IndexedBilinearPatchMeshBvh) -> Self {
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

        let mut vertices = vec![0u32; mesh.vertex_count() * PATCH_MESH_VERTEX_WORDS];
        for (i, &pos) in mesh.positions().iter().enumerate() {
            let b = i * PATCH_MESH_VERTEX_WORDS;
            write_vec3(&mut vertices, b, pos);
            write_vec3(&mut vertices, b + 3, mesh.normals()[i]);
            let uv = mesh.uvs()[i];
            vertices[b + 6] = uv[0].to_bits();
            vertices[b + 7] = uv[1].to_bits();
        }

        let mut indices = vec![0u32; mesh.patch_count() * PATCH_MESH_INDEX_WORDS];
        for (i, quad) in mesh.indices().iter().enumerate() {
            let b = i * PATCH_MESH_INDEX_WORDS;
            indices[b] = quad[0];
            indices[b + 1] = quad[1];
            indices[b + 2] = quad[2];
            indices[b + 3] = quad[3];
        }

        Self {
            nodes,
            vertices,
            indices,
            order: bvh.order().to_vec(),
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
        self.vertices.len() / PATCH_MESH_VERTEX_WORDS
    }

    /// Number of packed patches.
    #[must_use]
    pub fn patch_count(&self) -> usize {
        self.indices.len() / PATCH_MESH_INDEX_WORDS
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

    /// Rebuilds the shared-pool [`IndexedBilinearPatchMesh`] from the packed
    /// buffers.
    ///
    /// Decoding routes through [`IndexedBilinearPatchMesh::new`] with the exact
    /// stored words (positions, normals, and uvs verbatim), so the
    /// reconstructed mesh is bit-identical to the one the `CPU`
    /// [`IndexedBilinearPatchMeshBvh`] holds and `intersect_patch` reproduces
    /// every hit.
    #[must_use]
    pub fn decode_mesh(&self) -> IndexedBilinearPatchMesh {
        let vertex_count = self.vertex_count();
        let mut positions = Vec::with_capacity(vertex_count);
        let mut normals = Vec::with_capacity(vertex_count);
        let mut uvs = Vec::with_capacity(vertex_count);
        for i in 0..vertex_count {
            let b = i * PATCH_MESH_VERTEX_WORDS;
            positions.push(read_vec3(&self.vertices, b));
            normals.push(read_vec3(&self.vertices, b + 3));
            uvs.push(read_vec2(&self.vertices, b + 6));
        }
        let mut indices = Vec::with_capacity(self.patch_count());
        for i in 0..self.patch_count() {
            let b = i * PATCH_MESH_INDEX_WORDS;
            indices.push([
                self.indices[b],
                self.indices[b + 1],
                self.indices[b + 2],
                self.indices[b + 3],
            ]);
        }
        IndexedBilinearPatchMesh::new(positions, normals, uvs, indices)
            .expect("packed buffers encode a valid mesh")
    }

    /// Nearest intersection along `ray` by walking the packed buffers.
    ///
    /// Decodes the mesh, then mirrors
    /// [`IndexedBilinearPatchMeshBvh::closest_hit`] exactly — same slab
    /// rejection, same near/far child ordering by split-axis sign, same running
    /// `t_max` shrink, and the same `order`-indexed
    /// [`IndexedBilinearPatchMesh::intersect_patch`] test — so the result is
    /// bit-for-bit identical and a `GPU` kernel binding these buffers can be
    /// diffed against it.
    #[must_use]
    pub fn closest_hit(&self, ray: &Ray) -> Option<ShadedBilinearPatchHit> {
        if self.nodes.is_empty() {
            return None;
        }
        let mesh = self.decode_mesh();
        let mut ray = *ray;
        let mut best: Option<ShadedBilinearPatchHit> = None;

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
                        if let Some(hit) = mesh.intersect_patch(slot as usize, &ray) {
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

    /// True when *any* patch intersects `ray`; mirrors
    /// [`IndexedBilinearPatchMeshBvh::any_hit`].
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
                        if mesh.intersect_patch(slot as usize, ray).is_some() {
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
    }

    /// Builds an `n`×`n` welded grid mesh matching the `CPU` suite's generator.
    fn grid_mesh(n: usize, rng: &mut Rng) -> IndexedBilinearPatchMesh {
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        for j in 0..n {
            for i in 0..n {
                positions.push([i as f32, rng.range(-0.3, 0.3), j as f32]);
                normals.push([rng.range(-0.2, 0.2), 1.0, rng.range(-0.2, 0.2)]);
                uvs.push([i as f32 / (n - 1) as f32, j as f32 / (n - 1) as f32]);
            }
        }
        let mut indices = Vec::new();
        for j in 0..n - 1 {
            for i in 0..n - 1 {
                let v = |ii: usize, jj: usize| (jj * n + ii) as u32;
                indices.push([v(i, j), v(i + 1, j), v(i + 1, j + 1), v(i, j + 1)]);
            }
        }
        IndexedBilinearPatchMesh::new(positions, normals, uvs, indices).unwrap()
    }

    /// Asserts two hits are bit-identical across every reported field.
    fn assert_hit_bits(a: &ShadedBilinearPatchHit, b: &ShadedBilinearPatchHit) {
        assert_eq!(a.t.to_bits(), b.t.to_bits());
        assert_eq!(a.primitive, b.primitive);
        assert_eq!(a.front_face, b.front_face);
        assert_eq!(a.u.to_bits(), b.u.to_bits());
        assert_eq!(a.v.to_bits(), b.v.to_bits());
        for (x, y) in a.geometric_normal.iter().zip(b.geometric_normal.iter()) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
        for (x, y) in a.shading_normal.iter().zip(b.shading_normal.iter()) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
        for (x, y) in a.uv.iter().zip(b.uv.iter()) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
    }

    #[test]
    fn strides_are_sixteen_byte_aligned() {
        assert_eq!(PATCH_MESH_VERTEX_WORDS % 4, 0);
        assert_eq!(PATCH_MESH_INDEX_WORDS % 4, 0);
    }

    #[test]
    fn empty_buffers_never_hit() {
        let gpu = GpuIndexedBilinearPatchMeshBvhBuffers::default();
        assert!(gpu.is_empty());
        let ray = Ray::infinite([0.0, 1.0, 0.0], [0.0, -1.0, 0.0]);
        assert!(gpu.closest_hit(&ray).is_none());
        assert!(!gpu.any_hit(&ray));
    }

    #[test]
    fn from_bvh_preserves_counts_and_buffer_shape() {
        let mut rng = Rng::new(0x1357);
        let mesh = grid_mesh(5, &mut rng);
        let vertex_count = mesh.vertex_count();
        let patch_count = mesh.patch_count();
        let bvh = IndexedBilinearPatchMeshBvh::build(mesh);
        let gpu = GpuIndexedBilinearPatchMeshBvhBuffers::from_bvh(&bvh);
        assert_eq!(gpu.node_count(), bvh.node_count());
        assert_eq!(gpu.vertex_count(), vertex_count);
        assert_eq!(gpu.patch_count(), patch_count);
        assert_eq!(gpu.vertices.len(), vertex_count * PATCH_MESH_VERTEX_WORDS);
        assert_eq!(gpu.indices.len(), patch_count * PATCH_MESH_INDEX_WORDS);
        assert_eq!(gpu.order, bvh.order());
    }

    #[test]
    fn decoded_mesh_round_trips_fields() {
        let mut rng = Rng::new(0x2468);
        let mesh = grid_mesh(4, &mut rng);
        let bvh = IndexedBilinearPatchMeshBvh::build(mesh.clone());
        let gpu = GpuIndexedBilinearPatchMeshBvhBuffers::from_bvh(&bvh);
        let decoded = gpu.decode_mesh();
        assert_eq!(decoded, mesh);
    }

    #[test]
    fn packed_walk_matches_in_memory_bvh_bit_for_bit() {
        for seed in [0x5EED_1234u64, 0x0C0F_FEE1] {
            let mut rng = Rng::new(seed);
            let mesh = grid_mesh(5, &mut rng);
            let bvh = IndexedBilinearPatchMeshBvh::build(mesh);
            let gpu = GpuIndexedBilinearPatchMeshBvhBuffers::from_bvh(&bvh);
            let mut hits = 0;
            for _ in 0..600 {
                let o = [rng.range(-1.0, 5.0), rng.range(1.0, 3.0), rng.range(-1.0, 5.0)];
                let target = [rng.range(0.0, 4.0), 0.0, rng.range(0.0, 4.0)];
                let d = [target[0] - o[0], target[1] - o[1], target[2] - o[2]];
                let ray = Ray::infinite(o, d);
                match (bvh.closest_hit(&ray), gpu.closest_hit(&ray)) {
                    (Some(a), Some(b)) => {
                        assert_hit_bits(&a, &b);
                        hits += 1;
                    }
                    (None, None) => {}
                    (a, b) => panic!("closest_hit disagreement: {a:?} vs {b:?}"),
                }
                assert_eq!(bvh.any_hit(&ray), gpu.any_hit(&ray));
            }
            assert!(hits > 100, "expected many hits, got {hits}");
        }
    }
}
