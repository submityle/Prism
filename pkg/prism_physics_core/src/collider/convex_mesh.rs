//! Bounded convex-mesh collider data for the shared [`ShapeRegistry`].
//!
//! A convex mesh is stored as its surface triangle set, from which this module
//! derives everything the physics pipeline needs:
//!
//! - a vertex cloud for support mapping (GJK/EPA via
//!   [`prism_physics_geometry::SupportMap`]),
//! - a set of outward face half-spaces for exact ray clipping and
//!   least-penetration point projection,
//! - a local-space AABB and bounding radius for broad-phase,
//! - closed-form mass properties via signed-tetrahedron integration.
//!
//! The convex mesh is referenced from bodies through a [`ConvexMeshHandle`], so
//! many bodies can share one immutable definition. The geometry here is kept in
//! the collider layer (rather than the geometry crate) because it also carries
//! the physics-specific cached mass tensor.
//!
//! The integration formulas are standard rigid-body results (divergence theorem
//! over the surface triangles) and are not derived from Unreal Engine source.

use crate::state::body::MassProperties;
use glam::Vec3;
use prism_physics_geometry::{gjk_closest_points, SupportMap};

use super::mass_props_from_diagonal;

/// A handle into the convex-mesh arena of a [`ShapeRegistry`](super::ShapeRegistry).
///
/// This is a plain index handle; convex meshes are immutable once inserted, so
/// no generation counter is required for correctness.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ConvexMeshHandle(pub u32);

/// An outward-facing support plane of a convex polytope.
///
/// The interior half-space is `dot(normal, x) <= offset`; `normal` is unit
/// length and points away from the polytope interior.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
struct FacePlane {
    /// Unit outward normal.
    normal: Vec3,
    /// Signed distance of the plane from the local origin along `normal`.
    offset: f32,
}

/// The result of casting a ray against a convex mesh.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ConvexRayHit {
    /// Ray parameter at the first intersection (`origin + dir * time`).
    pub time: f32,
    /// World/local-space intersection point.
    pub point: Vec3,
    /// Unit outward surface normal at the intersection.
    pub normal: Vec3,
}

/// The result of projecting a point onto a convex mesh.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ConvexProjection {
    /// Closest point on (or inside) the convex mesh.
    pub point: Vec3,
    /// Unit outward surface normal at `point`.
    pub normal: Vec3,
    /// Distance from the query point to `point`; zero when the query is inside.
    pub distance: f32,
    /// `true` when the query point lies inside the convex mesh.
    pub is_inside: bool,
}

/// Immutable, bounded convex-mesh collider geometry plus its cached mass tensor.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ConvexMeshData {
    /// Hull vertices in local space (used for support mapping).
    vertices: Vec<Vec3>,
    /// Surface triangles as indices into `vertices` (used for mass integration).
    triangles: Vec<[u32; 3]>,
    /// Deduplicated outward face half-spaces (used for ray clip / projection).
    faces: Vec<FacePlane>,
    /// Cached local-space AABB minimum.
    aabb_min: Vec3,
    /// Cached local-space AABB maximum.
    aabb_max: Vec3,
    /// Cached bounding-sphere radius about the local origin.
    bounding_radius: f32,
    /// Enclosed volume (zero or negative for degenerate / non-closed hulls).
    volume: f32,
    /// Diagonal inertia about the local axes for unit density.
    unit_density_inertia: Vec3,
}

impl ConvexMeshData {
    /// Builds a convex mesh from its surface triangle set.
    ///
    /// `triangles` index into `vertices` and are expected to form a closed
    /// convex surface. Face normals are oriented outward relative to the
    /// vertex centroid; coplanar duplicate faces are merged for the half-space
    /// set. Mass is computed under a closed-manifold solid assumption; a
    /// degenerate (near-zero volume) hull yields zero mass with a `None`-style
    /// zero inertia, matching the immovable-shape precedent.
    #[must_use]
    pub fn from_surface(vertices: Vec<Vec3>, triangles: Vec<[u32; 3]>) -> ConvexMeshData {
        let (aabb_min, aabb_max) = aabb_of(&vertices);
        let centroid = centroid_of(&vertices);
        let bounding_radius = vertices.iter().map(|v| v.length()).fold(0.0_f32, f32::max);

        let faces = derive_faces(&vertices, &triangles, centroid);
        let (volume, inertia) = integrate_mass(&vertices, &triangles);

        ConvexMeshData {
            vertices,
            triangles,
            faces,
            aabb_min,
            aabb_max,
            bounding_radius,
            volume,
            unit_density_inertia: inertia,
        }
    }

    /// Builds an axis-aligned box convex mesh with the given half-extents.
    ///
    /// This is a convenience constructor (8 vertices, 12 triangles) that is
    /// handy for tests and for treating boxes through the convex-mesh path.
    #[must_use]
    pub fn from_box(half_extents: Vec3) -> ConvexMeshData {
        let h = half_extents;
        let vertices = vec![
            Vec3::new(-h.x, -h.y, -h.z),
            Vec3::new(h.x, -h.y, -h.z),
            Vec3::new(h.x, h.y, -h.z),
            Vec3::new(-h.x, h.y, -h.z),
            Vec3::new(-h.x, -h.y, h.z),
            Vec3::new(h.x, -h.y, h.z),
            Vec3::new(h.x, h.y, h.z),
            Vec3::new(-h.x, h.y, h.z),
        ];
        // Outward-facing triangles (CCW seen from outside).
        let triangles = vec![
            [0, 2, 1],
            [0, 3, 2], // -Z
            [4, 5, 6],
            [4, 6, 7], // +Z
            [0, 1, 5],
            [0, 5, 4], // -Y
            [3, 7, 6],
            [3, 6, 2], // +Y
            [0, 4, 7],
            [0, 7, 3], // -X
            [1, 2, 6],
            [1, 6, 5], // +X
        ];
        ConvexMeshData::from_surface(vertices, triangles)
    }

    /// Returns the hull vertices in local space.
    #[must_use]
    pub fn vertices(&self) -> &[Vec3] {
        &self.vertices
    }

    /// Returns the surface triangles as indices into [`Self::vertices`].
    #[must_use]
    pub fn triangles(&self) -> &[[u32; 3]] {
        &self.triangles
    }

    /// Returns the local-space axis-aligned bounding box as `(min, max)`.
    #[must_use]
    pub fn local_aabb(&self) -> (Vec3, Vec3) {
        (self.aabb_min, self.aabb_max)
    }

    /// Returns the bounding-sphere radius about the local origin.
    #[must_use]
    pub fn bounding_radius(&self) -> f32 {
        self.bounding_radius
    }

    /// Returns the enclosed volume of the hull.
    #[must_use]
    pub fn volume(&self) -> f32 {
        self.volume
    }

    /// Returns the mass properties for the given uniform `density`.
    ///
    /// Mass scales with volume and the inertia tensor scales linearly with
    /// density. The inertia is reported as a diagonal about the local axes; the
    /// off-diagonal (product-of-inertia) terms are dropped, matching the
    /// documented capsule approximation. A degenerate hull reports zero mass
    /// (an immovable body).
    #[must_use]
    pub fn mass_properties(&self, density: f32) -> MassProperties {
        if self.volume <= 0.0 || density <= 0.0 {
            return MassProperties::zero();
        }
        let mass = density * self.volume;
        mass_props_from_diagonal(mass, self.unit_density_inertia * density)
    }

    /// Returns the farthest hull vertex along `dir` (support mapping).
    #[must_use]
    pub fn support_point(&self, dir: Vec3) -> Vec3 {
        let mut best = self.vertices.first().copied().unwrap_or(Vec3::ZERO);
        let mut best_dot = best.dot(dir);
        for &v in self.vertices.iter().skip(1) {
            let d = v.dot(dir);
            if d > best_dot {
                best_dot = d;
                best = v;
            }
        }
        best
    }

    /// Casts a ray `origin + dir * t` for `t in [0, max_time]` against the hull.
    ///
    /// Returns the first entry intersection, or `None` when the ray misses. A
    /// ray starting inside the hull reports `time == 0` with the exit-adjacent
    /// entry normal.
    #[must_use]
    pub fn ray_cast(&self, origin: Vec3, dir: Vec3, max_time: f32) -> Option<ConvexRayHit> {
        if self.faces.is_empty() {
            return None;
        }
        let mut t_enter = 0.0_f32;
        let mut t_exit = max_time;
        let mut enter_normal = Vec3::ZERO;

        for face in &self.faces {
            // f(t) = normal . (origin + t*dir) - offset, interior is f <= 0.
            let num = face.normal.dot(origin) - face.offset;
            let denom = face.normal.dot(dir);
            if denom.abs() <= 1e-9 {
                // Ray is parallel to the slab; reject if it starts outside.
                if num > 0.0 {
                    return None;
                }
                continue;
            }
            let t = -num / denom;
            if denom < 0.0 {
                // Entering this half-space.
                if t > t_enter {
                    t_enter = t;
                    enter_normal = face.normal;
                }
            } else {
                // Leaving this half-space.
                if t < t_exit {
                    t_exit = t;
                }
            }
            if t_enter > t_exit {
                return None;
            }
        }

        if t_enter > max_time || t_exit < 0.0 {
            return None;
        }
        // Origin already inside: no face recorded an entry normal.
        if enter_normal == Vec3::ZERO {
            enter_normal = self.outward_normal_at(origin + dir * t_enter);
        }
        Some(ConvexRayHit {
            time: t_enter,
            point: origin + dir * t_enter,
            normal: enter_normal,
        })
    }

    /// Projects `point` onto the convex mesh, returning the closest surface
    /// point and whether the query lies inside.
    #[must_use]
    pub fn project_point(&self, point: Vec3) -> ConvexProjection {
        // Inside test: the point satisfies every interior half-space.
        let mut max_signed = f32::NEG_INFINITY;
        let mut deepest_normal = Vec3::Y;
        let mut inside = true;
        for face in &self.faces {
            let signed = face.normal.dot(point) - face.offset;
            if signed > 1e-6 {
                inside = false;
            }
            if signed > max_signed {
                max_signed = signed;
                deepest_normal = face.normal;
            }
        }

        if inside {
            // Push out along the least-penetrating face.
            let depth = -max_signed; // non-negative distance to the nearest face
            return ConvexProjection {
                point: point + deepest_normal * depth,
                normal: deepest_normal,
                distance: 0.0,
                is_inside: true,
            };
        }

        // Outside: exact closest point via GJK between a point and the hull.
        match gjk_closest_points(&PointSupport(point), self) {
            Some(cp) => ConvexProjection {
                point: cp.point_b,
                normal: if cp.normal.length_squared() > 0.0 {
                    cp.normal
                } else {
                    deepest_normal
                },
                distance: cp.distance,
                is_inside: false,
            },
            None => ConvexProjection {
                point,
                normal: deepest_normal,
                distance: 0.0,
                is_inside: true,
            },
        }
    }

    /// Returns the outward normal of the face closest to `p` (used when a ray
    /// starts inside the hull).
    fn outward_normal_at(&self, p: Vec3) -> Vec3 {
        let mut best = Vec3::Y;
        let mut best_signed = f32::NEG_INFINITY;
        for face in &self.faces {
            let signed = face.normal.dot(p) - face.offset;
            if signed > best_signed {
                best_signed = signed;
                best = face.normal;
            }
        }
        best
    }
}

impl SupportMap for ConvexMeshData {
    fn support_point(&self, dir: Vec3) -> Vec3 {
        ConvexMeshData::support_point(self, dir)
    }
}

/// A degenerate convex shape consisting of a single point, used as the second
/// operand to [`gjk_closest_points`] when projecting a point onto a hull.
struct PointSupport(Vec3);

impl SupportMap for PointSupport {
    fn support_point(&self, _dir: Vec3) -> Vec3 {
        self.0
    }
}

/// Returns the axis-aligned bounds of a vertex cloud as `(min, max)`.
fn aabb_of(vertices: &[Vec3]) -> (Vec3, Vec3) {
    if vertices.is_empty() {
        return (Vec3::ZERO, Vec3::ZERO);
    }
    let mut min = vertices[0];
    let mut max = vertices[0];
    for &v in &vertices[1..] {
        min = min.min(v);
        max = max.max(v);
    }
    (min, max)
}

/// Returns the arithmetic mean of a vertex cloud.
fn centroid_of(vertices: &[Vec3]) -> Vec3 {
    if vertices.is_empty() {
        return Vec3::ZERO;
    }
    let sum: Vec3 = vertices.iter().copied().fold(Vec3::ZERO, |a, b| a + b);
    sum / vertices.len() as f32
}

/// Derives the deduplicated outward face half-spaces from the surface triangles.
fn derive_faces(vertices: &[Vec3], triangles: &[[u32; 3]], centroid: Vec3) -> Vec<FacePlane> {
    let mut faces: Vec<FacePlane> = Vec::new();
    for tri in triangles {
        let a = vertices[tri[0] as usize];
        let b = vertices[tri[1] as usize];
        let c = vertices[tri[2] as usize];
        let mut normal = (b - a).cross(c - a);
        let len = normal.length();
        if len <= 1e-12 {
            continue; // degenerate triangle
        }
        normal /= len;
        // Orient outward relative to the hull centroid.
        if normal.dot(a - centroid) < 0.0 {
            normal = -normal;
        }
        let offset = normal.dot(a);
        // Merge coplanar duplicates (same normal and offset within tolerance).
        let dup = faces
            .iter()
            .any(|f| f.normal.dot(normal) > 0.999_9 && (f.offset - offset).abs() < 1e-5);
        if !dup {
            faces.push(FacePlane { normal, offset });
        }
    }
    faces
}

/// Integrates the enclosed volume and the diagonal inertia (unit density) using
/// signed tetrahedra formed by each surface triangle and the local origin.
fn integrate_mass(vertices: &[Vec3], triangles: &[[u32; 3]]) -> (f32, Vec3) {
    let mut volume = 0.0_f32;
    // Second moments accumulated about the local origin.
    let mut ixx = 0.0_f32;
    let mut iyy = 0.0_f32;
    let mut izz = 0.0_f32;

    for tri in triangles {
        let a = vertices[tri[0] as usize];
        let b = vertices[tri[1] as usize];
        let c = vertices[tri[2] as usize];
        // Signed volume of the tetrahedron (origin, a, b, c).
        let det = a.dot(b.cross(c));
        let vol = det / 6.0;
        volume += vol;

        // Second-moment contribution of this tetrahedron about the origin.
        // Integral of x_i*x_j over a tetrahedron with one vertex at the origin.
        for axis in 0..3 {
            let (pa, pb, pc) = (a[axis], b[axis], c[axis]);
            let moment = (pa * pa + pb * pb + pc * pc + pa * pb + pb * pc + pc * pa) * (vol / 10.0);
            match axis {
                0 => ixx += moment,
                1 => iyy += moment,
                _ => izz += moment,
            }
        }
    }

    if volume <= 1e-9 {
        return (0.0_f32.max(volume), Vec3::ZERO);
    }

    // Inertia about principal axes for unit density: I_xx = integral(y^2+z^2).
    let inertia = Vec3::new(iyy + izz, ixx + izz, ixx + iyy);
    (volume, inertia)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PI: f32 = std::f32::consts::PI;

    #[test]
    fn box_aabb_matches_half_extents() {
        let h = Vec3::new(1.0, 2.0, 3.0);
        let hull = ConvexMeshData::from_box(h);
        let (min, max) = hull.local_aabb();
        assert!((min - (-h)).length() < 1e-6);
        assert!((max - h).length() < 1e-6);
    }

    #[test]
    fn box_volume_and_mass_match_closed_form() {
        let h = Vec3::new(1.0, 2.0, 3.0);
        let hull = ConvexMeshData::from_box(h);
        let full = h * 2.0;
        let expected_vol = full.x * full.y * full.z;
        assert!((hull.volume() - expected_vol).abs() < 1e-3);

        let density = 2.5;
        let mp = hull.mass_properties(density);
        let mass = density * expected_vol;
        assert!((mp.mass() - mass).abs() < 1e-2);

        // Box diagonal inertia closed form.
        let ix = (1.0 / 12.0) * mass * (full.y * full.y + full.z * full.z);
        let iy = (1.0 / 12.0) * mass * (full.x * full.x + full.z * full.z);
        let iz = (1.0 / 12.0) * mass * (full.x * full.x + full.y * full.y);
        assert!((mp.inv_inertia.x - 1.0 / ix).abs() < 1e-3);
        assert!((mp.inv_inertia.y - 1.0 / iy).abs() < 1e-3);
        assert!((mp.inv_inertia.z - 1.0 / iz).abs() < 1e-3);
    }

    #[test]
    fn support_point_picks_extreme_vertex() {
        let hull = ConvexMeshData::from_box(Vec3::ONE);
        let s = hull.support_point(Vec3::new(1.0, 1.0, 1.0));
        assert!((s - Vec3::ONE).length() < 1e-6);
        let s = hull.support_point(Vec3::new(-1.0, -1.0, -1.0));
        assert!((s - Vec3::splat(-1.0)).length() < 1e-6);
    }

    #[test]
    fn ray_hits_box_front_face() {
        let hull = ConvexMeshData::from_box(Vec3::ONE);
        let hit = hull
            .ray_cast(Vec3::new(0.0, 0.0, -5.0), Vec3::Z, 100.0)
            .expect("ray should hit the box");
        assert!((hit.time - 4.0).abs() < 1e-4);
        assert!((hit.point - Vec3::new(0.0, 0.0, -1.0)).length() < 1e-4);
        assert!(hit.normal.dot(Vec3::NEG_Z) > 0.99);
    }

    #[test]
    fn ray_misses_box() {
        let hull = ConvexMeshData::from_box(Vec3::ONE);
        assert!(hull
            .ray_cast(Vec3::new(5.0, 5.0, -5.0), Vec3::Z, 100.0)
            .is_none());
    }

    #[test]
    fn project_inside_pushes_to_nearest_face() {
        let hull = ConvexMeshData::from_box(Vec3::ONE);
        let proj = hull.project_point(Vec3::new(0.9, 0.0, 0.0));
        assert!(proj.is_inside);
        assert!((proj.point - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-4);
        assert!(proj.normal.dot(Vec3::X) > 0.99);
        assert!(proj.distance.abs() < 1e-6);
    }

    #[test]
    fn project_outside_returns_surface_point() {
        let hull = ConvexMeshData::from_box(Vec3::ONE);
        let proj = hull.project_point(Vec3::new(3.0, 0.0, 0.0));
        assert!(!proj.is_inside);
        assert!((proj.point.x - 1.0).abs() < 1e-3);
        assert!((proj.distance - 2.0).abs() < 1e-3);
    }

    #[test]
    fn tetrahedron_volume_matches_closed_form() {
        // Regular-ish tetrahedron with one vertex at origin.
        let vertices = vec![
            Vec3::ZERO,
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(0.0, 0.0, 2.0),
        ];
        let triangles = vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
        let hull = ConvexMeshData::from_surface(vertices, triangles);
        // Volume of tetra with legs 2,2,2 = (1/6)*2*2*2 = 8/6.
        assert!((hull.volume() - 8.0 / 6.0).abs() < 1e-4);
    }

    #[test]
    fn degenerate_hull_is_massless() {
        let hull = ConvexMeshData::from_surface(Vec::new(), Vec::new());
        let mp = hull.mass_properties(5.0);
        assert_eq!(mp.inv_mass, 0.0);
    }

    #[test]
    fn bounding_radius_covers_box() {
        let hull = ConvexMeshData::from_box(Vec3::ONE);
        assert!((hull.bounding_radius() - 3.0_f32.sqrt()).abs() < 1e-5);
        // Sanity: the box corner distance uses sqrt(3) ~ 1.732, not PI.
        assert!(PI > hull.bounding_radius());
    }
}
