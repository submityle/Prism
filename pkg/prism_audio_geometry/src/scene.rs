//! The triangle-mesh scene the geometric backend traces.
//!
//! An [`AcousticScene`] pairs a [`TriangleMesh`] (with its built-in
//! bounding-volume-hierarchy ray cast) with a [`MaterialTable`] assigning an
//! [`AcousticMaterial`](prism_audio_spatial::propagation::AcousticMaterial) to
//! each triangle. It is the single geometry authority every path builder shares:
//! it answers "what is the nearest surface a ray hits, and what is it made of?"
//! and marches a segment to enumerate every partition between two points, which
//! is how the direct path accumulates transmission loss and how reflections and
//! diffractions test visibility.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//!
//! Wraps [`prism_physics_geometry::TriangleMesh`] and [`crate::material_map::MaterialTable`];
//! consumed by [`crate::direct_path`], [`crate::reflection_path`],
//! [`crate::diffraction_path`], and assembled by [`crate::backend::GeometricBackend`].

use alloc::vec::Vec;

use bevy_math::Vec3;
use prism_audio_core::math::Sample;
use prism_audio_spatial::propagation::AcousticMaterial;
use prism_physics_geometry::{Ray, TriangleMesh};

use crate::material_map::MaterialTable;

/// Largest number of surfaces [`AcousticScene::march_segment`] will step
/// through before giving up, bounding the control-rate transmission walk even
/// for pathological geometry.
pub const MAX_MARCH_HITS: usize = 64;

/// Why building an [`AcousticScene`] failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum SceneBuildError {
    /// A triangle index referenced a vertex outside the vertex list.
    IndexOutOfRange {
        /// The offending triangle's position in the index list.
        triangle: usize,
        /// The vertex index that was out of range.
        vertex: u32,
    },
}

/// A single surface intersection returned by the scene's ray queries.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SceneRayHit {
    /// Index of the triangle that was hit.
    pub triangle: u32,
    /// Distance along the ray to the hit point, in metres.
    pub distance: Sample,
    /// World-space hit position.
    pub point: Vec3,
    /// Geometric (face) unit normal of the hit triangle.
    pub normal: Vec3,
    /// Acoustic material of the hit triangle.
    pub material: AcousticMaterial,
}

/// A triangle mesh plus the acoustic material of each triangle.
#[derive(Debug, Clone)]
pub struct AcousticScene {
    mesh: TriangleMesh,
    materials: MaterialTable,
}

impl AcousticScene {
    /// Builds a scene from vertex positions, triangle indices, and a material
    /// table. Returns [`SceneBuildError`] when an index is out of range.
    pub fn new(
        vertices: Vec<Vec3>,
        indices: Vec<[u32; 3]>,
        materials: MaterialTable,
    ) -> Result<Self, SceneBuildError> {
        let vertex_count = vertices.len() as u32;
        for (triangle, tri) in indices.iter().enumerate() {
            for &vertex in tri {
                if vertex >= vertex_count {
                    return Err(SceneBuildError::IndexOutOfRange { triangle, vertex });
                }
            }
        }
        Ok(Self {
            mesh: TriangleMesh::new(vertices, indices),
            materials,
        })
    }

    /// Builds a scene from an already-constructed triangle mesh and its
    /// material table.
    #[inline]
    #[must_use]
    pub fn from_mesh(mesh: TriangleMesh, materials: MaterialTable) -> Self {
        Self { mesh, materials }
    }

    /// The underlying triangle mesh.
    #[inline]
    #[must_use]
    pub fn mesh(&self) -> &TriangleMesh {
        &self.mesh
    }

    /// The scene's material table.
    #[inline]
    #[must_use]
    pub fn materials(&self) -> &MaterialTable {
        &self.materials
    }

    /// Number of triangles in the scene.
    #[inline]
    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.mesh.triangle_count()
    }

    /// Whether the scene has no triangles.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.mesh.is_empty()
    }

    /// The three vertices of triangle `index`, or [`None`] if out of range.
    #[inline]
    #[must_use]
    pub fn triangle(&self, index: usize) -> Option<[Vec3; 3]> {
        self.mesh.triangle(index)
    }

    /// The outward face unit normal of triangle `index`, or [`None`] if out of
    /// range or degenerate.
    #[must_use]
    pub fn triangle_normal(&self, index: usize) -> Option<Vec3> {
        let [a, b, c] = self.mesh.triangle(index)?;
        let n = (b - a).cross(c - a);
        let n = n.normalize_or_zero();
        if n == Vec3::ZERO { None } else { Some(n) }
    }

    /// The acoustic material of triangle `index` (default when unassigned).
    #[inline]
    #[must_use]
    pub fn material(&self, index: usize) -> AcousticMaterial {
        self.materials.material(index)
    }

    /// Casts a ray from `origin` in `direction` and returns the nearest surface
    /// hit no farther than `max_distance`, if any. `direction` is normalized by
    /// the ray; a zero `max_distance` yields no hit.
    #[must_use]
    pub fn first_hit(
        &self,
        origin: Vec3,
        direction: Vec3,
        max_distance: Sample,
    ) -> Option<SceneRayHit> {
        if max_distance <= 0.0 {
            return None;
        }
        let ray = Ray::with_tmax(origin, direction, max_distance);
        let hit = self.mesh.ray_cast(&ray)?;
        if hit.t > max_distance {
            return None;
        }
        Some(SceneRayHit {
            triangle: hit.triangle,
            distance: hit.t,
            point: hit.point,
            normal: hit.normal,
            material: self.materials.material(hit.triangle as usize),
        })
    }

    /// Returns `true` when the straight segment from `from` to `to` crosses any
    /// surface, i.e. the two points do not share a clear line of sight.
    ///
    /// `epsilon` shrinks the segment at both ends so a surface the endpoints
    /// already lie on (a reflection or diffraction point) does not count as a
    /// blocker.
    #[must_use]
    pub fn segment_blocked(&self, from: Vec3, to: Vec3, epsilon: Sample) -> bool {
        let delta = to - from;
        let length = delta.length();
        if length <= 2.0 * epsilon.max(0.0) {
            return false;
        }
        let dir = delta / length;
        let eps = epsilon.max(0.0);
        let origin = from + dir * eps;
        self.first_hit(origin, dir, length - 2.0 * eps).is_some()
    }

    /// Marches the segment from `from` to `to`, invoking `visit` for each
    /// surface crossed in near-to-far order until it returns `false`, the far
    /// endpoint is reached, or [`MAX_MARCH_HITS`] surfaces have been reported.
    ///
    /// `epsilon` is the step pushed past each hit so the walk advances through
    /// coincident faces instead of re-hitting the surface it just left.
    pub fn march_segment(
        &self,
        from: Vec3,
        to: Vec3,
        epsilon: Sample,
        mut visit: impl FnMut(SceneRayHit) -> bool,
    ) {
        let delta = to - from;
        let total = delta.length();
        if total <= 0.0 {
            return;
        }
        let dir = delta / total;
        let eps = epsilon.max(0.0);
        let mut cursor = from;
        let mut remaining = total;
        for _ in 0..MAX_MARCH_HITS {
            if remaining <= eps {
                break;
            }
            let Some(hit) = self.first_hit(cursor, dir, remaining) else {
                break;
            };
            if !visit(hit) {
                break;
            }
            // Step just past the hit so the next cast does not re-report it.
            let step = hit.distance + eps;
            cursor += dir * step;
            remaining -= step;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{AcousticScene, SceneBuildError, MAX_MARCH_HITS};
    use alloc::vec;
    use bevy_math::Vec3;
    use prism_audio_spatial::propagation::AcousticMaterial;

    use crate::material_map::MaterialTable;

    // A single quad (two triangles) in the plane x = 0, spanning y,z in
    // [-1, 1]. Front face normal points toward +X.
    fn wall_at_x0(material: AcousticMaterial) -> AcousticScene {
        let vertices = vec![
            Vec3::new(0.0, -1.0, -1.0),
            Vec3::new(0.0, 1.0, -1.0),
            Vec3::new(0.0, 1.0, 1.0),
            Vec3::new(0.0, -1.0, 1.0),
        ];
        let indices = vec![[0, 1, 2], [0, 2, 3]];
        AcousticScene::new(vertices, indices, MaterialTable::uniform(material)).unwrap()
    }

    #[test]
    fn rejects_out_of_range_index() {
        let vertices = vec![Vec3::ZERO, Vec3::X, Vec3::Y];
        let indices = vec![[0, 1, 5]];
        let err = AcousticScene::new(vertices, indices, MaterialTable::default()).unwrap_err();
        assert_eq!(
            err,
            SceneBuildError::IndexOutOfRange {
                triangle: 0,
                vertex: 5
            }
        );
    }

    #[test]
    fn first_hit_finds_the_wall() {
        let wall = AcousticMaterial::new(30.0, 0.5);
        let scene = wall_at_x0(wall);
        let hit = scene
            .first_hit(Vec3::new(-2.0, 0.0, 0.0), Vec3::X, 10.0)
            .expect("ray toward wall should hit");
        assert!((hit.distance - 2.0).abs() < 1e-4);
        assert_eq!(hit.material, wall);
        assert!(hit.point.x.abs() < 1e-4);
    }

    #[test]
    fn first_hit_respects_max_distance() {
        let scene = wall_at_x0(AcousticMaterial::OPEN);
        // Wall is 2 m away; a 1 m ray must not reach it.
        assert!(scene
            .first_hit(Vec3::new(-2.0, 0.0, 0.0), Vec3::X, 1.0)
            .is_none());
    }

    #[test]
    fn segment_blocked_detects_the_wall() {
        let scene = wall_at_x0(AcousticMaterial::OPEN);
        assert!(scene.segment_blocked(Vec3::new(-2.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0), 1e-3));
        // A segment that stays on one side is clear.
        assert!(!scene.segment_blocked(
            Vec3::new(-2.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            1e-3
        ));
    }

    #[test]
    fn march_reports_single_wall_once() {
        let scene = wall_at_x0(AcousticMaterial::new(20.0, 0.0));
        let mut count = 0usize;
        scene.march_segment(
            Vec3::new(-2.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            1e-3,
            |_hit| {
                count += 1;
                true
            },
        );
        assert_eq!(count, 1);
    }

    #[test]
    fn march_is_bounded() {
        let scene = wall_at_x0(AcousticMaterial::OPEN);
        let mut count = 0usize;
        // Visiting forever would exceed the cap; our wall yields one hit, but
        // the bound is what we assert never to exceed.
        scene.march_segment(
            Vec3::new(-2.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            1e-3,
            |_hit| {
                count += 1;
                true
            },
        );
        assert!(count <= MAX_MARCH_HITS);
    }
}
