//! Indexed, smooth-shaded bilinear-patch mesh: a shared vertex pool plus
//! four-index quad records, with a single-level `BVH` that is the `CPU` golden
//! reference for welded quad-patch surfaces.
//!
//! The standalone [`super::shaded_bilinear_patch::ShadedBilinearPatch`] stores
//! its four corner positions, shading normals, and `UV`s inline, which is ideal
//! for a handful of isolated patches but wastes memory and breaks shading
//! continuity across a surface: a shared edge is stored twice, and nothing
//! guarantees the two copies carry the same normal, so a welded cage reads with
//! visible seams. Production quad surfaces — Catmull-Clark subdivision cages,
//! pbrt's `BilinearPatchMesh`, displaced quad grids, cloth/hair cards — instead
//! store **one** position/normal/`UV` per vertex and reference them from quad
//! index records, so a shared corner is a single sample that every incident
//! patch reuses. This primitive is that indexed form: a vertex pool
//! (`positions`, `normals`, `uvs` of equal length) and a list of `[u32; 4]`
//! corner indices, one per patch.
//!
//! Each patch is materialised on demand as a
//! [`super::shaded_bilinear_patch::ShadedBilinearPatch`] gathered from the pool
//! and intersected with the identical Reshetov "Cool Patches" analytic solve,
//! so a hit's `t`/`(u, v)`/normals/`UV` bits are **identical** to the
//! standalone patch built from the same four corners. Corner attributes are
//! gathered verbatim (normals re-normalized only at the hit, exactly as the
//! standalone patch does), the ray direction is never assumed unit, and the
//! [`IndexedBilinearPatchMeshBvh`] reorders patches internally but always
//! reports hits by their original patch index via
//! [`super::shaded_bilinear_patch::ShadedBilinearPatchHit::primitive`].

use super::bvh::{build_linear_bvh, Aabb, BvhBuildConfig, LinearBvhNode};
use super::shaded_bilinear_patch::{ShadedBilinearPatch, ShadedBilinearPatchHit};
use super::traversal::Ray;

/// Why an [`IndexedBilinearPatchMesh`] could not be constructed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexedBilinearPatchMeshError {
    /// A patch corner referenced a vertex outside the pool.
    IndexOutOfRange {
        /// Patch record that held the bad index.
        patch: usize,
        /// Corner slot `0..4` within that patch.
        corner: usize,
        /// The offending vertex index.
        index: u32,
        /// Number of vertices actually present in the pool.
        vertex_count: usize,
    },
    /// The shading-normal pool length did not match the position pool length.
    NormalCountMismatch {
        /// Number of shading normals supplied.
        normals: usize,
        /// Number of positions supplied.
        positions: usize,
    },
    /// The texture-coordinate pool length did not match the position pool
    /// length.
    UvCountMismatch {
        /// Number of texture coordinates supplied.
        uvs: usize,
        /// Number of positions supplied.
        positions: usize,
    },
}

impl core::fmt::Display for IndexedBilinearPatchMeshError {
    /// Formats the error as a single human-readable diagnostic line.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::IndexOutOfRange {
                patch,
                corner,
                index,
                vertex_count,
            } => write!(
                f,
                "patch {patch} corner {corner} references vertex {index}, \
                 but the pool holds only {vertex_count} vertices"
            ),
            Self::NormalCountMismatch { normals, positions } => write!(
                f,
                "shading-normal count {normals} does not match position count {positions}"
            ),
            Self::UvCountMismatch { uvs, positions } => write!(
                f,
                "texture-coordinate count {uvs} does not match position count {positions}"
            ),
        }
    }
}

impl std::error::Error for IndexedBilinearPatchMeshError {}

/// A smooth-shaded quad-patch mesh over a shared vertex pool.
///
/// `positions`, `normals`, and `uvs` are parallel arrays indexed by vertex id
/// and must share the same length. Each entry of `indices` names the four
/// corner vertices of one patch in the `(u, v)` order `0 = (0, 0)`,
/// `1 = (1, 0)`, `2 = (1, 1)`, `3 = (0, 1)`, matching
/// [`super::shaded_bilinear_patch::ShadedBilinearPatch`]. Normals are stored
/// verbatim (not re-normalized here) so the gather is bit-identical to the
/// standalone patch; they are re-normalized at the hit.
#[derive(Clone, Debug, PartialEq)]
pub struct IndexedBilinearPatchMesh {
    /// Vertex positions, one per vertex id.
    positions: Vec<[f32; 3]>,
    /// Per-vertex shading normals (verbatim), parallel to `positions`.
    normals: Vec<[f32; 3]>,
    /// Per-vertex texture coordinates, parallel to `positions`.
    uvs: Vec<[f32; 2]>,
    /// Patch corner indices `[i00, i10, i11, i01]` into the vertex pool.
    indices: Vec<[u32; 4]>,
}

impl IndexedBilinearPatchMesh {
    /// Builds a mesh from a shared vertex pool and quad `indices`.
    ///
    /// Returns [`IndexedBilinearPatchMeshError`] when `normals` or `uvs` do not
    /// match `positions` in length, or when any corner index is out of range.
    /// An empty mesh (no positions, no patches) is valid and never hits.
    pub fn new(
        positions: Vec<[f32; 3]>,
        normals: Vec<[f32; 3]>,
        uvs: Vec<[f32; 2]>,
        indices: Vec<[u32; 4]>,
    ) -> Result<Self, IndexedBilinearPatchMeshError> {
        if normals.len() != positions.len() {
            return Err(IndexedBilinearPatchMeshError::NormalCountMismatch {
                normals: normals.len(),
                positions: positions.len(),
            });
        }
        if uvs.len() != positions.len() {
            return Err(IndexedBilinearPatchMeshError::UvCountMismatch {
                uvs: uvs.len(),
                positions: positions.len(),
            });
        }
        let vertex_count = positions.len();
        for (patch, quad) in indices.iter().enumerate() {
            for (corner, &index) in quad.iter().enumerate() {
                if index as usize >= vertex_count {
                    return Err(IndexedBilinearPatchMeshError::IndexOutOfRange {
                        patch,
                        corner,
                        index,
                        vertex_count,
                    });
                }
            }
        }
        Ok(Self {
            positions,
            normals,
            uvs,
            indices,
        })
    }

    /// The vertex positions pool.
    #[must_use]
    pub fn positions(&self) -> &[[f32; 3]] {
        &self.positions
    }

    /// The per-vertex shading normals pool.
    #[must_use]
    pub fn normals(&self) -> &[[f32; 3]] {
        &self.normals
    }

    /// The per-vertex texture coordinates pool.
    #[must_use]
    pub fn uvs(&self) -> &[[f32; 2]] {
        &self.uvs
    }

    /// The patch corner-index records.
    #[must_use]
    pub fn indices(&self) -> &[[u32; 4]] {
        &self.indices
    }

    /// Number of vertices in the shared pool.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    /// Number of patches in the mesh.
    #[must_use]
    pub fn patch_count(&self) -> usize {
        self.indices.len()
    }

    /// Materialises patch `patch` as a standalone
    /// [`super::shaded_bilinear_patch::ShadedBilinearPatch`] whose
    /// `primitive` id is `patch`.
    ///
    /// The four corners are gathered verbatim from the vertex pool; indices
    /// were range-checked at construction, so this never panics for a patch id
    /// in `0..patch_count`.
    #[must_use]
    pub fn patch(&self, patch: usize) -> ShadedBilinearPatch {
        let [i00, i10, i11, i01] = self.indices[patch];
        let (i00, i10, i11, i01) =
            (i00 as usize, i10 as usize, i11 as usize, i01 as usize);
        let positions = [
            self.positions[i00],
            self.positions[i10],
            self.positions[i11],
            self.positions[i01],
        ];
        let normals = [
            self.normals[i00],
            self.normals[i10],
            self.normals[i11],
            self.normals[i01],
        ];
        let uvs = [
            self.uvs[i00],
            self.uvs[i10],
            self.uvs[i11],
            self.uvs[i01],
        ];
        ShadedBilinearPatch::new(positions, normals, uvs, patch as u32)
    }

    /// Axis-aligned bounds of patch `patch` (its four corners).
    #[must_use]
    pub fn patch_aabb(&self, patch: usize) -> Aabb {
        self.patch(patch).aabb()
    }

    /// Axis-aligned bounds of the whole mesh (union of every vertex position),
    /// or the empty box when the pool is empty.
    #[must_use]
    pub fn aabb(&self) -> Aabb {
        let mut iter = self.positions.iter();
        let Some(&first) = iter.next() else {
            return Aabb::empty();
        };
        let mut lo = first;
        let mut hi = first;
        for v in iter {
            for k in 0..3 {
                if v[k] < lo[k] {
                    lo[k] = v[k];
                }
                if v[k] > hi[k] {
                    hi[k] = v[k];
                }
            }
        }
        Aabb::new(lo, hi)
    }

    /// Nearest intersection of `ray` with patch `patch`, or `None` on a miss.
    ///
    /// Delegates to the standalone patch's analytic solve, so the hit bits
    /// match the inline [`super::shaded_bilinear_patch::ShadedBilinearPatch`]
    /// built from the same four corners.
    #[must_use]
    pub fn intersect_patch(&self, patch: usize, ray: &Ray) -> Option<ShadedBilinearPatchHit> {
        self.patch(patch).intersect(ray)
    }

    /// Nearest intersection of `ray` with any patch via a linear scan.
    ///
    /// This is the brute-force reference for
    /// [`IndexedBilinearPatchMeshBvh::closest_hit`]; production traversal
    /// should go through the `BVH`.
    #[must_use]
    pub fn intersect(&self, ray: &Ray) -> Option<ShadedBilinearPatchHit> {
        let mut ray = *ray;
        let mut best: Option<ShadedBilinearPatchHit> = None;
        for patch in 0..self.patch_count() {
            if let Some(hit) = self.intersect_patch(patch, &ray) {
                ray = Ray::new(ray.origin(), ray.direction(), ray.t_min(), hit.t);
                best = Some(hit);
            }
        }
        best
    }
}

/// A single-level `BVH` over an [`IndexedBilinearPatchMesh`]'s patches.
#[derive(Clone, Debug)]
pub struct IndexedBilinearPatchMeshBvh {
    /// The owned mesh, kept intact so hits gather bit-identical corners.
    mesh: IndexedBilinearPatchMesh,
    /// Flattened `BVH` nodes; the root (when present) is index `0`.
    nodes: Vec<LinearBvhNode>,
    /// Maps each `BVH` primitive slot to its original patch index; leaf slices
    /// index into this so hits report the original patch id.
    order: Vec<u32>,
}

impl IndexedBilinearPatchMeshBvh {
    /// Builds a `BVH` over `mesh`'s patches with [`BvhBuildConfig::default`].
    #[must_use]
    pub fn build(mesh: IndexedBilinearPatchMesh) -> Self {
        Self::build_with(mesh, BvhBuildConfig::default())
    }

    /// Builds a `BVH` over `mesh`'s patches with the given binned-`SAH`
    /// `config`.
    ///
    /// Each patch's [`IndexedBilinearPatchMesh::patch_aabb`] feeds the builder;
    /// the returned primitive order is kept as
    /// [`IndexedBilinearPatchMeshBvh::order`] so leaf slices map back to
    /// original patch indices without disturbing the shared pool.
    #[must_use]
    pub fn build_with(mesh: IndexedBilinearPatchMesh, config: BvhBuildConfig) -> Self {
        let bounds: Vec<Aabb> = (0..mesh.patch_count())
            .map(|patch| mesh.patch_aabb(patch))
            .collect();
        let (nodes, order) = build_linear_bvh(&bounds, config);
        Self { mesh, nodes, order }
    }

    /// The owned mesh.
    #[must_use]
    pub fn mesh(&self) -> &IndexedBilinearPatchMesh {
        &self.mesh
    }

    /// Number of flattened `BVH` nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of patches in the hierarchy.
    #[must_use]
    pub fn primitive_count(&self) -> usize {
        self.order.len()
    }

    /// True when the hierarchy holds no patches.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Root bounds, or the empty box when the hierarchy is empty.
    #[must_use]
    pub fn bounds(&self) -> Aabb {
        self.nodes.first().map_or(Aabb::empty(), |node| node.bounds)
    }

    /// The flattened node array.
    #[must_use]
    pub fn nodes(&self) -> &[LinearBvhNode] {
        &self.nodes
    }

    /// The `BVH`-slot → original-patch-index map (leaf slices index this).
    #[must_use]
    pub fn order(&self) -> &[u32] {
        &self.order
    }

    /// Nearest intersection along `ray`, or `None` if the ray hits nothing.
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
                    for &slot in &self.order[start..end] {
                        if let Some(hit) = self.mesh.intersect_patch(slot as usize, &ray) {
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
                    for &slot in &self.order[start..end] {
                        if self.mesh.intersect_patch(slot as usize, ray).is_some() {
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

/// Pops the next deferred `BVH` node index from the traversal `stack`.
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
    use crate::ray_scene::shaded_bilinear_patch::ShadedBilinearPatch;

    /// Minimal xorshift `RNG` for deterministic test rays and meshes.
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

        /// Uniform sample in `[0, 1]`.
        fn unit(&mut self) -> f32 {
            self.next_u32() as f32 / u32::MAX as f32
        }

        /// Uniform sample in `[lo, hi]`.
        fn range(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * self.unit()
        }
    }

    /// Builds an `n`×`n` vertex grid welded into `(n-1)²` quad patches, with
    /// per-vertex normals pointing roughly up and planar `UV`s.
    fn grid_mesh(n: usize, rng: &mut Rng) -> IndexedBilinearPatchMesh {
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        for j in 0..n {
            for i in 0..n {
                let x = i as f32;
                let z = j as f32;
                let y = rng.range(-0.3, 0.3);
                positions.push([x, y, z]);
                normals.push([rng.range(-0.2, 0.2), 1.0, rng.range(-0.2, 0.2)]);
                uvs.push([i as f32 / (n - 1) as f32, j as f32 / (n - 1) as f32]);
            }
        }
        let mut indices = Vec::new();
        for j in 0..n - 1 {
            for i in 0..n - 1 {
                let v = |ii: usize, jj: usize| (jj * n + ii) as u32;
                // (u,v) order 0=(0,0),1=(1,0),2=(1,1),3=(0,1).
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
    fn normal_count_mismatch_is_rejected() {
        let err = IndexedBilinearPatchMesh::new(
            vec![[0.0; 3]; 4],
            vec![[0.0, 1.0, 0.0]; 3],
            vec![[0.0; 2]; 4],
            vec![[0, 1, 2, 3]],
        )
        .unwrap_err();
        assert_eq!(
            err,
            IndexedBilinearPatchMeshError::NormalCountMismatch {
                normals: 3,
                positions: 4,
            }
        );
    }

    #[test]
    fn uv_count_mismatch_is_rejected() {
        let err = IndexedBilinearPatchMesh::new(
            vec![[0.0; 3]; 4],
            vec![[0.0, 1.0, 0.0]; 4],
            vec![[0.0; 2]; 5],
            vec![[0, 1, 2, 3]],
        )
        .unwrap_err();
        assert_eq!(
            err,
            IndexedBilinearPatchMeshError::UvCountMismatch {
                uvs: 5,
                positions: 4,
            }
        );
    }

    #[test]
    fn out_of_range_index_is_rejected() {
        let err = IndexedBilinearPatchMesh::new(
            vec![[0.0; 3]; 4],
            vec![[0.0, 1.0, 0.0]; 4],
            vec![[0.0; 2]; 4],
            vec![[0, 1, 9, 3]],
        )
        .unwrap_err();
        assert_eq!(
            err,
            IndexedBilinearPatchMeshError::IndexOutOfRange {
                patch: 0,
                corner: 2,
                index: 9,
                vertex_count: 4,
            }
        );
    }

    #[test]
    fn empty_mesh_never_hits() {
        let mesh = IndexedBilinearPatchMesh::new(vec![], vec![], vec![], vec![]).unwrap();
        assert_eq!(mesh.patch_count(), 0);
        assert_eq!(mesh.vertex_count(), 0);
        let bvh = IndexedBilinearPatchMeshBvh::build(mesh);
        assert!(bvh.is_empty());
        let ray = Ray::infinite([0.0, 1.0, 0.0], [0.0, -1.0, 0.0]);
        assert!(bvh.closest_hit(&ray).is_none());
        assert!(!bvh.any_hit(&ray));
        assert_eq!(bvh.bounds(), Aabb::empty());
    }

    #[test]
    fn single_patch_matches_standalone_bit_for_bit() {
        let positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ];
        let normals = [
            [0.1, 1.0, 0.0],
            [-0.1, 1.0, 0.0],
            [0.0, 1.0, 0.2],
            [0.0, 1.0, -0.2],
        ];
        let uvs = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let standalone = ShadedBilinearPatch::new(positions, normals, uvs, 0);
        let mesh = IndexedBilinearPatchMesh::new(
            positions.to_vec(),
            normals.to_vec(),
            uvs.to_vec(),
            vec![[0, 1, 2, 3]],
        )
        .unwrap();

        let mut rng = Rng::new(0xA11CE);
        let mut hits = 0;
        for _ in 0..400 {
            let o = [rng.range(-0.5, 1.5), rng.range(0.5, 2.0), rng.range(-0.5, 1.5)];
            let target = [rng.range(0.0, 1.0), 0.0, rng.range(0.0, 1.0)];
            let d = [target[0] - o[0], target[1] - o[1], target[2] - o[2]];
            let ray = Ray::infinite(o, d);
            match (standalone.intersect(&ray), mesh.intersect_patch(0, &ray)) {
                (Some(a), Some(b)) => {
                    assert_hit_bits(&a, &b);
                    hits += 1;
                }
                (None, None) => {}
                (a, b) => panic!("hit disagreement: {a:?} vs {b:?}"),
            }
        }
        assert!(hits > 50, "expected many hits, got {hits}");
    }

    #[test]
    fn bvh_closest_hit_matches_brute_force() {
        let mut rng = Rng::new(0x5EED_1234);
        let mesh = grid_mesh(5, &mut rng);
        let patch_count = mesh.patch_count();
        let bvh = IndexedBilinearPatchMeshBvh::build(mesh.clone());
        assert_eq!(bvh.primitive_count(), patch_count);

        let mut hits = 0;
        for _ in 0..600 {
            let o = [rng.range(-1.0, 5.0), rng.range(1.0, 3.0), rng.range(-1.0, 5.0)];
            let target = [rng.range(0.0, 4.0), 0.0, rng.range(0.0, 4.0)];
            let d = [target[0] - o[0], target[1] - o[1], target[2] - o[2]];
            let ray = Ray::infinite(o, d);
            let brute = mesh.intersect(&ray);
            let fast = bvh.closest_hit(&ray);
            match (brute, fast) {
                (Some(a), Some(b)) => {
                    assert_hit_bits(&a, &b);
                    hits += 1;
                }
                (None, None) => {}
                (a, b) => panic!("closest_hit disagreement: {a:?} vs {b:?}"),
            }
            assert_eq!(mesh.intersect(&ray).is_some(), bvh.any_hit(&ray));
        }
        assert!(hits > 100, "expected many hits, got {hits}");
    }

    #[test]
    fn welded_edge_shares_one_normal() {
        // Two patches sharing the edge (v10, v11) read the identical corner
        // normal because it is a single pooled vertex.
        let positions = vec![
            [0.0, 0.0, 0.0], // 0
            [1.0, 0.0, 0.0], // 1 (shared)
            [1.0, 0.0, 1.0], // 2 (shared)
            [0.0, 0.0, 1.0], // 3
            [2.0, 0.0, 0.0], // 4
            [2.0, 0.0, 1.0], // 5
        ];
        let normals = vec![
            [0.0, 1.0, 0.0],
            [0.3, 1.0, 0.0], // shared vertex 1
            [-0.3, 1.0, 0.1], // shared vertex 2
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let uvs = vec![[0.0; 2]; 6];
        let mesh = IndexedBilinearPatchMesh::new(
            positions,
            normals.clone(),
            uvs,
            vec![[0, 1, 2, 3], [1, 4, 5, 2]],
        )
        .unwrap();
        // Patch 0 corner 1 and patch 1 corner 0 are both pooled vertex 1.
        assert_eq!(mesh.patch(0).normals()[1], normals[1]);
        assert_eq!(mesh.patch(1).normals()[0], normals[1]);
        // Patch 0 corner 2 and patch 1 corner 3 are both pooled vertex 2.
        assert_eq!(mesh.patch(0).normals()[2], normals[2]);
        assert_eq!(mesh.patch(1).normals()[3], normals[2]);
    }

    #[test]
    fn hit_reports_original_patch_index() {
        let mut rng = Rng::new(0xBEEF_0007);
        let mesh = grid_mesh(4, &mut rng);
        let bvh = IndexedBilinearPatchMeshBvh::build(mesh.clone());
        for _ in 0..300 {
            let o = [rng.range(0.0, 3.0), 2.0, rng.range(0.0, 3.0)];
            let ray = Ray::infinite(o, [0.0, -1.0, 0.0]);
            if let Some(hit) = bvh.closest_hit(&ray) {
                // The reported primitive id must index a real patch and its
                // gathered corners must reproduce the same hit.
                let id = hit.primitive as usize;
                assert!(id < mesh.patch_count());
                let direct = mesh.intersect_patch(id, &ray).expect("patch must hit");
                assert_hit_bits(&hit, &direct);
            }
        }
    }
}
