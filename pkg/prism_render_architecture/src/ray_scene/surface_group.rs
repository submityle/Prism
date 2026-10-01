//! Merge many heterogeneous triangle-mesh parts into one bottom-level
//! acceleration structure while preserving per-triangle *source* attribution.
//!
//! This is the single-`BLAS` counterpart to [`super::tlas`]: a `TLAS`
//! *instances* one shared `BLAS` under many transforms, whereas a
//! [`SurfaceGroup`] *bakes* several distinct meshes — say the triangle meshes
//! produced by [`super::trimmed_surface`], [`super::displaced_surface`], or any
//! other [`super::triangle_mesh::TriangleMesh`] source — into **one** merged
//! vertex pool and index buffer, exactly as an engine welds the sub-meshes of
//! one static object into a single `BLAS`. Because the merged primitive
//! ordering is flattened, a ray hit reports a global triangle index; the group
//! keeps the per-part triangle ranges so [`SurfaceGroup::source_of`] (and
//! [`SurfaceGroupBvh::closest_hit`]) attribute that hit back to the originating
//! part — the hook a renderer uses to pick the right material per sub-mesh.
//!
//! Vertex-attribute presence is **all-or-nothing**: every non-empty part must
//! agree on whether it carries shading normals and whether it carries texture
//! coordinates, so the merged pools stay rectangular and a flat `GPU` layout
//! decodes unambiguously. Empty parts (no vertices) are allowed and simply
//! occupy a zero-length triangle range.

use super::bvh::Aabb;
use super::traversal::Ray;
use super::triangle_mesh::{MeshHit, TriangleMesh, TriangleMeshBvh};

/// Why [`SurfaceGroup::new`] rejected its parts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceGroupError {
    /// No parts were supplied; a group needs at least one.
    NoParts,
    /// A part disagreed with the rest on shading-normal presence.
    NormalPresenceMismatch {
        /// Zero-based index of the offending part.
        part: usize,
        /// Whether that part carried normals.
        part_has: bool,
        /// Whether the group (its first non-empty part) carried normals.
        group_has: bool,
    },
    /// A part disagreed with the rest on texture-coordinate presence.
    UvPresenceMismatch {
        /// Zero-based index of the offending part.
        part: usize,
        /// Whether that part carried texture coordinates.
        part_has: bool,
        /// Whether the group (its first non-empty part) carried coordinates.
        group_has: bool,
    },
}

impl core::fmt::Display for SurfaceGroupError {
    /// Formats the merge error for diagnostics.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoParts => write!(f, "a surface group requires at least one part"),
            Self::NormalPresenceMismatch {
                part,
                part_has,
                group_has,
            } => write!(
                f,
                "part {part} normal presence ({part_has}) disagrees with the group ({group_has})"
            ),
            Self::UvPresenceMismatch {
                part,
                part_has,
                group_has,
            } => write!(
                f,
                "part {part} UV presence ({part_has}) disagrees with the group ({group_has})"
            ),
        }
    }
}

impl std::error::Error for SurfaceGroupError {}

/// Several triangle-mesh parts merged into one mesh with per-part attribution.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceGroup {
    /// The merged mesh: all parts' vertices concatenated and indices rebased.
    mesh: TriangleMesh,
    /// Prefix sums of per-part triangle counts; `triangle_offsets[i]` is the
    /// first global triangle index of part `i` and the array has
    /// `part_count + 1` entries (the last equals the total triangle count).
    triangle_offsets: Vec<usize>,
}

impl SurfaceGroup {
    /// Merges `parts` into one group.
    ///
    /// # Errors
    ///
    /// Returns [`SurfaceGroupError::NoParts`] when `parts` is empty, or a
    /// `*PresenceMismatch` when a non-empty part disagrees with the rest on
    /// whether it carries normals or texture coordinates.
    pub fn new(parts: Vec<TriangleMesh>) -> Result<Self, SurfaceGroupError> {
        if parts.is_empty() {
            return Err(SurfaceGroupError::NoParts);
        }

        // Decide attribute presence from the first non-empty part, then require
        // every other non-empty part to agree.
        let mut group_normals: Option<bool> = None;
        let mut group_uvs: Option<bool> = None;
        for (part, mesh) in parts.iter().enumerate() {
            if mesh.vertex_count() == 0 {
                continue;
            }
            match group_normals {
                None => group_normals = Some(mesh.has_normals()),
                Some(group_has) if group_has != mesh.has_normals() => {
                    return Err(SurfaceGroupError::NormalPresenceMismatch {
                        part,
                        part_has: mesh.has_normals(),
                        group_has,
                    });
                }
                _ => {}
            }
            match group_uvs {
                None => group_uvs = Some(mesh.has_uvs()),
                Some(group_has) if group_has != mesh.has_uvs() => {
                    return Err(SurfaceGroupError::UvPresenceMismatch {
                        part,
                        part_has: mesh.has_uvs(),
                        group_has,
                    });
                }
                _ => {}
            }
        }
        let group_has_normals = group_normals.unwrap_or(false);
        let group_has_uvs = group_uvs.unwrap_or(false);

        let mut positions: Vec<[f32; 3]> = Vec::new();
        let mut normals: Vec<[f32; 3]> = Vec::new();
        let mut uvs: Vec<[f32; 2]> = Vec::new();
        let mut indices: Vec<[u32; 3]> = Vec::new();
        let mut triangle_offsets: Vec<usize> = Vec::with_capacity(parts.len() + 1);
        triangle_offsets.push(0);

        for mesh in &parts {
            let base = positions.len() as u32;
            positions.extend_from_slice(mesh.positions());
            if group_has_normals {
                normals.extend_from_slice(mesh.normals());
            }
            if group_has_uvs {
                uvs.extend_from_slice(mesh.uvs());
            }
            for tri in mesh.indices() {
                indices.push([tri[0] + base, tri[1] + base, tri[2] + base]);
            }
            triangle_offsets.push(indices.len());
        }

        let mesh = TriangleMesh::new(positions, normals, uvs, indices)
            .expect("rebased part indices stay within the merged vertex pool");
        Ok(Self {
            mesh,
            triangle_offsets,
        })
    }

    /// The merged triangle mesh.
    #[must_use]
    pub fn mesh(&self) -> &TriangleMesh {
        &self.mesh
    }

    /// Number of parts the group was built from.
    #[must_use]
    pub fn part_count(&self) -> usize {
        self.triangle_offsets.len() - 1
    }

    /// Total number of triangles across all parts.
    #[must_use]
    pub fn triangle_count(&self) -> usize {
        *self.triangle_offsets.last().unwrap_or(&0)
    }

    /// The half-open global triangle range `[start, end)` owned by `part`.
    ///
    /// Returns `None` when `part` is out of range. An empty part yields an
    /// empty range (`start == end`).
    #[must_use]
    pub fn part_triangle_range(&self, part: usize) -> Option<(usize, usize)> {
        if part >= self.part_count() {
            return None;
        }
        Some((self.triangle_offsets[part], self.triangle_offsets[part + 1]))
    }

    /// Maps a global `triangle` index back to the part that produced it.
    ///
    /// Returns `None` when `triangle` is out of range. Empty parts are never
    /// returned because no triangle falls inside their zero-length range.
    #[must_use]
    pub fn source_of(&self, triangle: usize) -> Option<usize> {
        if triangle >= self.triangle_count() {
            return None;
        }
        // Largest part index whose start offset is <= triangle. With tied
        // offsets (empty parts) this lands on the non-empty owner because its
        // start equals triangle and its end exceeds it.
        let p = self.triangle_offsets.partition_point(|&o| o <= triangle);
        Some(p - 1)
    }

    /// Conservative axis-aligned bounds over every merged vertex.
    #[must_use]
    pub fn aabb(&self) -> Aabb {
        let mut aabb = Aabb::empty();
        for p in self.mesh.positions() {
            aabb.min[0] = aabb.min[0].min(p[0]);
            aabb.min[1] = aabb.min[1].min(p[1]);
            aabb.min[2] = aabb.min[2].min(p[2]);
            aabb.max[0] = aabb.max[0].max(p[0]);
            aabb.max[1] = aabb.max[1].max(p[1]);
            aabb.max[2] = aabb.max[2].max(p[2]);
        }
        aabb
    }

    /// Builds a [`SurfaceGroupBvh`] over the merged mesh for ray intersection
    /// with source attribution. Clones the merged mesh so the group remains
    /// usable afterwards.
    #[must_use]
    pub fn build_bvh(&self) -> SurfaceGroupBvh {
        SurfaceGroupBvh {
            inner: TriangleMeshBvh::build(self.mesh.clone()),
            triangle_offsets: self.triangle_offsets.clone(),
        }
    }
}

/// A nearest-hit on a [`SurfaceGroup`] plus the part that owns the triangle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceGroupHit {
    /// The underlying triangle-mesh hit (global triangle index in `.triangle`).
    pub hit: MeshHit,
    /// Zero-based index of the part that produced the hit triangle.
    pub source: usize,
}

/// A [`TriangleMeshBvh`] over a merged [`SurfaceGroup`] that reports which part
/// each hit belongs to.
#[derive(Clone, Debug)]
pub struct SurfaceGroupBvh {
    /// The acceleration structure over the merged mesh.
    inner: TriangleMeshBvh,
    /// Per-part triangle prefix sums, mirrored from the source group.
    triangle_offsets: Vec<usize>,
}

impl SurfaceGroupBvh {
    /// The underlying triangle-mesh `BVH`.
    #[must_use]
    pub fn inner(&self) -> &TriangleMeshBvh {
        &self.inner
    }

    /// Number of parts represented.
    #[must_use]
    pub fn part_count(&self) -> usize {
        self.triangle_offsets.len() - 1
    }

    /// Conservative bounds of the merged mesh.
    #[must_use]
    pub fn bounds(&self) -> Aabb {
        self.inner.bounds()
    }

    /// Maps a global `triangle` index to its owning part (see
    /// [`SurfaceGroup::source_of`]).
    #[must_use]
    pub fn source_of(&self, triangle: usize) -> Option<usize> {
        let total = *self.triangle_offsets.last().unwrap_or(&0);
        if triangle >= total {
            return None;
        }
        let p = self.triangle_offsets.partition_point(|&o| o <= triangle);
        Some(p - 1)
    }

    /// Nearest hit along `ray`, carrying the owning part index.
    #[must_use]
    pub fn closest_hit(&self, ray: &Ray) -> Option<SurfaceGroupHit> {
        let hit = self.inner.closest_hit(ray)?;
        let source = self.source_of(hit.triangle as usize)?;
        Some(SurfaceGroupHit { hit, source })
    }

    /// Whether any triangle is hit along `ray` (occlusion/shadow query).
    #[must_use]
    pub fn any_hit(&self, ray: &Ray) -> bool {
        self.inner.any_hit(ray)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds an axis-aligned quad (two triangles) in the `z = height` plane
    /// spanning `[x0, x1] × [y0, y1]`, with `+Z` normals and corner `UV`s.
    fn quad(x0: f32, x1: f32, y0: f32, y1: f32, height: f32) -> TriangleMesh {
        let positions = vec![
            [x0, y0, height],
            [x1, y0, height],
            [x1, y1, height],
            [x0, y1, height],
        ];
        let normals = vec![[0.0, 0.0, 1.0]; 4];
        let uvs = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let indices = vec![[0, 1, 2], [0, 2, 3]];
        TriangleMesh::new(positions, normals, uvs, indices).unwrap()
    }

    /// A quad with no normals/uvs (geometric shading only).
    fn bare_quad(x0: f32, x1: f32, y0: f32, y1: f32) -> TriangleMesh {
        let positions = vec![
            [x0, y0, 0.0],
            [x1, y0, 0.0],
            [x1, y1, 0.0],
            [x0, y1, 0.0],
        ];
        TriangleMesh::new(positions, Vec::new(), Vec::new(), vec![[0, 1, 2], [0, 2, 3]]).unwrap()
    }

    /// A ray straight down `-Z` at world `(x, y)`.
    fn down_ray(x: f32, y: f32) -> Ray {
        Ray::infinite([x, y, 1.0], [0.0, 0.0, -1.0])
    }

    #[test]
    fn rejects_empty_part_list() {
        assert_eq!(
            SurfaceGroup::new(Vec::new()).unwrap_err(),
            SurfaceGroupError::NoParts
        );
    }

    #[test]
    fn merges_vertex_and_triangle_counts() {
        let a = quad(0.0, 1.0, 0.0, 1.0, 0.0);
        let b = quad(2.0, 3.0, 0.0, 1.0, 0.0);
        let group = SurfaceGroup::new(vec![a, b]).unwrap();
        assert_eq!(group.part_count(), 2);
        assert_eq!(group.triangle_count(), 4);
        assert_eq!(group.mesh().vertex_count(), 8);
        assert_eq!(group.part_triangle_range(0), Some((0, 2)));
        assert_eq!(group.part_triangle_range(1), Some((2, 4)));
        assert_eq!(group.part_triangle_range(2), None);
    }

    #[test]
    fn rebased_indices_stay_in_range() {
        let a = quad(0.0, 1.0, 0.0, 1.0, 0.0);
        let b = quad(2.0, 3.0, 0.0, 1.0, 0.0);
        let group = SurfaceGroup::new(vec![a, b]).unwrap();
        let vcount = group.mesh().vertex_count() as u32;
        for tri in group.mesh().indices() {
            assert!(tri.iter().all(|&idx| idx < vcount));
        }
        // The second part's triangles must reference the second vertex block.
        assert_eq!(group.mesh().indices()[2], [4, 5, 6]);
    }

    #[test]
    fn source_of_maps_triangles_to_parts() {
        let a = quad(0.0, 1.0, 0.0, 1.0, 0.0);
        let b = quad(2.0, 3.0, 0.0, 1.0, 0.0);
        let group = SurfaceGroup::new(vec![a, b]).unwrap();
        assert_eq!(group.source_of(0), Some(0));
        assert_eq!(group.source_of(1), Some(0));
        assert_eq!(group.source_of(2), Some(1));
        assert_eq!(group.source_of(3), Some(1));
        assert_eq!(group.source_of(4), None);
    }

    #[test]
    fn empty_part_occupies_zero_length_range() {
        let a = quad(0.0, 1.0, 0.0, 1.0, 0.0);
        let empty = TriangleMesh::new(Vec::new(), Vec::new(), Vec::new(), Vec::new()).unwrap();
        let c = quad(2.0, 3.0, 0.0, 1.0, 0.0);
        let group = SurfaceGroup::new(vec![a, empty, c]).unwrap();
        assert_eq!(group.part_count(), 3);
        assert_eq!(group.part_triangle_range(1), Some((2, 2))); // empty
        // Triangle 2 belongs to the third part, never the empty middle one.
        assert_eq!(group.source_of(2), Some(2));
        assert_eq!(group.source_of(3), Some(2));
    }

    #[test]
    fn normal_presence_mismatch_is_rejected() {
        let a = quad(0.0, 1.0, 0.0, 1.0, 0.0); // has normals
        let b = bare_quad(2.0, 3.0, 0.0, 1.0); // no normals
        let err = SurfaceGroup::new(vec![a, b]).unwrap_err();
        assert_eq!(
            err,
            SurfaceGroupError::NormalPresenceMismatch {
                part: 1,
                part_has: false,
                group_has: true,
            }
        );
    }

    #[test]
    fn uv_presence_mismatch_is_rejected() {
        // First part has UVs but not normals; second has neither -> UV mismatch.
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]];
        let uvs = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]];
        let a = TriangleMesh::new(positions, Vec::new(), uvs, vec![[0, 1, 2]]).unwrap();
        let b = bare_quad(2.0, 3.0, 0.0, 1.0);
        let err = SurfaceGroup::new(vec![a, b]).unwrap_err();
        assert_eq!(
            err,
            SurfaceGroupError::UvPresenceMismatch {
                part: 1,
                part_has: false,
                group_has: true,
            }
        );
    }

    #[test]
    fn aabb_covers_all_parts() {
        let a = quad(0.0, 1.0, 0.0, 1.0, -2.0);
        let b = quad(2.0, 3.0, 0.0, 1.0, 5.0);
        let group = SurfaceGroup::new(vec![a, b]).unwrap();
        let aabb = group.aabb();
        assert!((aabb.min[0] - 0.0).abs() < 1e-6);
        assert!((aabb.max[0] - 3.0).abs() < 1e-6);
        assert!((aabb.min[2] - (-2.0)).abs() < 1e-6);
        assert!((aabb.max[2] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn bvh_attributes_hits_to_the_right_part() {
        let a = quad(0.0, 1.0, 0.0, 1.0, 0.0); // part 0 near origin
        let b = quad(5.0, 6.0, 0.0, 1.0, 0.0); // part 1 far along +X
        let group = SurfaceGroup::new(vec![a, b]).unwrap();
        let bvh = group.build_bvh();
        let hit_a = bvh.closest_hit(&down_ray(0.5, 0.5)).expect("hit part 0");
        assert_eq!(hit_a.source, 0);
        assert!((hit_a.hit.t - 1.0).abs() < 1e-4);
        let hit_b = bvh.closest_hit(&down_ray(5.5, 0.5)).expect("hit part 1");
        assert_eq!(hit_b.source, 1);
        // A ray into empty space misses entirely.
        assert!(bvh.closest_hit(&down_ray(3.0, 0.5)).is_none());
        assert!(bvh.any_hit(&down_ray(0.5, 0.5)));
    }

    #[test]
    fn merge_is_deterministic() {
        let a = quad(0.0, 1.0, 0.0, 1.0, 0.0);
        let b = quad(2.0, 3.0, 0.0, 1.0, 0.0);
        let g0 = SurfaceGroup::new(vec![a.clone(), b.clone()]).unwrap();
        let g1 = SurfaceGroup::new(vec![a, b]).unwrap();
        assert_eq!(g0, g1);
    }

    #[test]
    fn bare_parts_merge_without_attributes() {
        let a = bare_quad(0.0, 1.0, 0.0, 1.0);
        let b = bare_quad(2.0, 3.0, 0.0, 1.0);
        let group = SurfaceGroup::new(vec![a, b]).unwrap();
        assert!(!group.mesh().has_normals());
        assert!(!group.mesh().has_uvs());
        assert_eq!(group.triangle_count(), 4);
        let bvh = group.build_bvh();
        let hit = bvh.closest_hit(&down_ray(0.5, 0.5)).expect("hit");
        assert_eq!(hit.source, 0);
    }
}
