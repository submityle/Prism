//! Triangle-mesh collider data for the shared shape registry.
//!
//! The analytic [`ColliderShape`](super::ColliderShape) primitives (sphere,
//! cuboid, capsule, half-space) cover the bread-and-butter rigid-body proxies,
//! but production engines (`PhysX`, Havok, Chaos) also carry arbitrary indexed
//! triangle meshes for static environment collision and detailed convex props.
//! This module wraps [`prism_physics_geometry::TriangleMesh`] into a first-class
//! piece of collider data, [`TriMeshData`], that an arena-style registry can own
//! and reference by [`TriMeshHandle`].
//!
//! [`TriMeshData`] eagerly caches the two quantities the broad phase and the
//! solver ask for most: a local-space axis-aligned bounding box and uniform
//! unit-density mass properties. The mesh itself (and its BVH) stays available
//! through [`TriMeshData::mesh`] for narrow-phase queries.
//!
//! # Mass properties
//!
//! Mass is computed under a **closed, solid-manifold assumption**: the mesh is
//! treated as the boundary of a watertight solid of uniform density. We use the
//! signed-tetrahedron decomposition (a divergence-theorem result): each triangle
//! `(v0, v1, v2)` forms a tetrahedron with the local-frame origin, whose signed
//! volume is `dot(v0, cross(v1, v2)) / 6`. Summing the signed volumes yields the
//! enclosed volume, and summing the per-tetrahedron second-moment (covariance)
//! integrals yields the full inertia integral about the local-frame origin
//! (Blow & Binstock, "How to find the inertia tensor of an arbitrary mesh",
//! 2004; Mirtich, "Fast and Accurate Computation of Polyhedral Mass
//! Properties", 1996).
//!
//! Only the **diagonal** of the resulting inertia tensor is retained — the same
//! documented diagonal approximation the capsule arm uses — so the stored
//! inertia is exact for a mesh whose local axes are (near) principal axes (a
//! box, a centred symmetric prop) and an approximation when strong products of
//! inertia exist. The inertia is taken about the local-frame origin, so callers
//! that want it about the centre of mass must apply the parallel-axis theorem
//! themselves.
//!
//! If the signed volume is effectively zero (an empty mesh, an open shell, or a
//! self-cancelling non-manifold), the solid assumption does not hold and we
//! fall back to **zero mass / zero inertia** (an immovable, infinite-mass body),
//! which is the safe choice for the static environment meshes this most often
//! represents. Reversed winding (negative enclosed volume) is handled by
//! negating the accumulated integrals so the reported mass stays non-negative.
//!
//! # Provenance
//!
//! The volume and inertia formulas are standard public results from the
//! references above. This module contains **no Unreal Engine source or derived
//! code**.

use crate::state::body::MassProperties;
use glam::{Mat3, Vec3};
use prism_physics_geometry::TriangleMesh;

/// A handle into a triangle-mesh arena.
///
/// Like [`ColliderHandle`](super::ColliderHandle) this is a plain index handle;
/// mesh data is immutable once inserted, so no generation counter is required
/// for correctness.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct TriMeshHandle(pub u32);

/// Immutable triangle-mesh collider data with cached bounds and mass.
///
/// Construct with [`TriMeshData::new`]; the local AABB and unit-density mass
/// properties are computed once and cached. Scale the cached unit-density
/// values to any density linearly via [`TriMeshData::mass_properties`].
#[derive(Clone, Debug)]
pub struct TriMeshData {
    mesh: TriangleMesh,
    local_aabb: (Vec3, Vec3),
    /// Enclosed mass at unit density (equal to the enclosed volume).
    unit_density_mass: f32,
    /// Diagonal of the inertia tensor about the local-frame origin at unit
    /// density.
    unit_density_inertia: Vec3,
}

impl TriMeshData {
    /// Builds mesh data from a vertex buffer and triangle index triples.
    ///
    /// The vertices and indices are handed to
    /// [`TriangleMesh::new`](prism_physics_geometry::TriangleMesh::new), which
    /// skips malformed triangles while building its BVH. The local AABB is the
    /// tight box over every valid triangle corner; the mass properties use the
    /// closed-solid assumption documented on the module.
    #[must_use]
    pub fn new(vertices: Vec<Vec3>, indices: Vec<[u32; 3]>) -> Self {
        let mesh = TriangleMesh::new(vertices, indices);
        let local_aabb = compute_local_aabb(&mesh);
        let (unit_density_mass, unit_density_inertia) = compute_unit_density_mass(&mesh);
        Self {
            mesh,
            local_aabb,
            unit_density_mass,
            unit_density_inertia,
        }
    }

    /// Returns the underlying geometry mesh (for narrow-phase queries).
    #[must_use]
    pub fn mesh(&self) -> &TriangleMesh {
        &self.mesh
    }

    /// Returns the cached local-space axis-aligned bounding box as `(min, max)`.
    ///
    /// An empty mesh reports a degenerate zero-extent box at the origin.
    #[must_use]
    pub fn local_aabb(&self) -> (Vec3, Vec3) {
        self.local_aabb
    }

    /// Returns the enclosed mass at unit density (equal to the enclosed
    /// volume), or `0.0` when the mesh is not a usable closed solid.
    #[must_use]
    pub fn unit_density_mass(&self) -> f32 {
        self.unit_density_mass
    }

    /// Returns the diagonal inertia about the local-frame origin at unit
    /// density.
    #[must_use]
    pub fn unit_density_inertia(&self) -> Vec3 {
        self.unit_density_inertia
    }

    /// Computes mass properties for the mesh at uniform `density`.
    ///
    /// Mass and inertia scale linearly with density. A non-positive density, or
    /// a mesh that is not a usable closed solid, yields
    /// [`MassProperties::zero`] (an immovable body).
    #[must_use]
    pub fn mass_properties(&self, density: f32) -> MassProperties {
        if density <= 0.0 {
            return MassProperties::zero();
        }
        let mass = density * self.unit_density_mass;
        let inertia = self.unit_density_inertia * density;
        mass_props_from_diagonal(mass, inertia)
    }
}

/// Builds [`MassProperties`] from a total mass and a diagonal inertia tensor,
/// inverting each nonzero component and leaving zero (infinite) components at
/// zero.
///
/// Mirrors the private helper in the parent module so this file stays
/// self-contained without widening that function's visibility.
fn mass_props_from_diagonal(mass: f32, inertia: Vec3) -> MassProperties {
    let inv_mass = if mass > 0.0 { 1.0 / mass } else { 0.0 };
    let inv = |i: f32| if i > 0.0 { 1.0 / i } else { 0.0 };
    MassProperties {
        inv_mass,
        inv_inertia: Vec3::new(inv(inertia.x), inv(inertia.y), inv(inertia.z)),
    }
}

/// Computes the tight local AABB over every valid triangle corner.
fn compute_local_aabb(mesh: &TriangleMesh) -> (Vec3, Vec3) {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    let mut seen = false;
    for i in 0..mesh.triangle_count() {
        if let Some([a, b, c]) = mesh.triangle(i) {
            min = min.min(a).min(b).min(c);
            max = max.max(a).max(b).max(c);
            seen = true;
        }
    }
    if seen {
        (min, max)
    } else {
        (Vec3::ZERO, Vec3::ZERO)
    }
}

/// Computes `(unit_density_mass, unit_density_inertia_diagonal)` under the
/// closed-solid assumption via signed-tetrahedron decomposition.
///
/// Returns `(0.0, Vec3::ZERO)` for a degenerate (near-zero-volume) mesh.
fn compute_unit_density_mass(mesh: &TriangleMesh) -> (f32, Vec3) {
    // Covariance integral of the canonical tetrahedron (origin + unit basis),
    // from Blow & Binstock. The full tetrahedron covariance is
    // `det(A) * A * C_canonical * A^T`, where `A` has the three triangle
    // corners as its columns.
    let c_canonical = Mat3::from_cols(
        Vec3::new(2.0, 1.0, 1.0),
        Vec3::new(1.0, 2.0, 1.0),
        Vec3::new(1.0, 1.0, 2.0),
    ) * (1.0 / 120.0);

    let mut signed_volume = 0.0f32;
    let mut covariance = Mat3::ZERO;
    for i in 0..mesh.triangle_count() {
        let Some([a, b, c]) = mesh.triangle(i) else {
            continue;
        };
        let a_mat = Mat3::from_cols(a, b, c);
        let det = a_mat.determinant();
        signed_volume += det / 6.0;
        covariance += (a_mat * c_canonical * a_mat.transpose()) * det;
    }

    // A reversed global winding flips the sign of both accumulators together;
    // negate so the reported mass stays non-negative while the inertia stays
    // consistent with that mass.
    if signed_volume < 0.0 {
        signed_volume = -signed_volume;
        covariance = -covariance;
    }

    // Below this the solid assumption is meaningless (empty / open / planar).
    const VOLUME_EPSILON: f32 = 1e-9;
    if signed_volume <= VOLUME_EPSILON {
        return (0.0, Vec3::ZERO);
    }

    // Inertia diagonal about the local-frame origin: `I_xx = C_yy + C_zz`, etc.
    let c_xx = covariance.x_axis.x;
    let c_yy = covariance.y_axis.y;
    let c_zz = covariance.z_axis.z;
    let inertia = Vec3::new(c_yy + c_zz, c_xx + c_zz, c_xx + c_yy);

    (signed_volume, inertia)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds the 8 corners and 12 triangles of an axis-aligned cube centred at
    /// the origin with the given half-extent, wound counter-clockwise (outward).
    fn unit_cube(half: f32) -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let h = half;
        let vertices = vec![
            Vec3::new(-h, -h, -h), // 0
            Vec3::new(h, -h, -h),  // 1
            Vec3::new(h, h, -h),   // 2
            Vec3::new(-h, h, -h),  // 3
            Vec3::new(-h, -h, h),  // 4
            Vec3::new(h, -h, h),   // 5
            Vec3::new(h, h, h),    // 6
            Vec3::new(-h, h, h),   // 7
        ];
        // Outward-facing (CCW) triangles for each of the six faces.
        let indices = vec![
            [0, 3, 2],
            [0, 2, 1], // -Z
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
        (vertices, indices)
    }

    #[test]
    fn unit_cube_aabb_is_symmetric() {
        let (v, i) = unit_cube(0.5);
        let data = TriMeshData::new(v, i);
        let (min, max) = data.local_aabb();
        assert!((min - Vec3::splat(-0.5)).length() < 1e-6);
        assert!((max - Vec3::splat(0.5)).length() < 1e-6);
    }

    #[test]
    fn unit_cube_volume_is_one() {
        let (v, i) = unit_cube(0.5);
        let data = TriMeshData::new(v, i);
        assert!((data.unit_density_mass() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn unit_cube_mass_scales_with_density() {
        let (v, i) = unit_cube(0.5);
        let data = TriMeshData::new(v, i);
        let density = 7.0;
        let mp = data.mass_properties(density);
        assert!((mp.mass() - density).abs() < 1e-4);
    }

    #[test]
    fn unit_cube_inertia_matches_closed_form() {
        // Side s = 1, mass m = density. I = (1/6) m s² for each principal axis.
        let (v, i) = unit_cube(0.5);
        let data = TriMeshData::new(v, i);
        let density = 3.0;
        let mp = data.mass_properties(density);
        let mass = density;
        let expected = (1.0 / 6.0) * mass; // s = 1
        let expected_inv = 1.0 / expected;
        assert!((mp.inv_inertia.x - expected_inv).abs() < 1e-3);
        assert!((mp.inv_inertia.y - expected_inv).abs() < 1e-3);
        assert!((mp.inv_inertia.z - expected_inv).abs() < 1e-3);
    }

    #[test]
    fn unit_density_inertia_is_cube_diagonal() {
        // At unit density the cube's mass is 1 and each diagonal moment is 1/6.
        let (v, i) = unit_cube(0.5);
        let data = TriMeshData::new(v, i);
        let diag = data.unit_density_inertia();
        assert!((diag.x - 1.0 / 6.0).abs() < 1e-5);
        assert!((diag.y - 1.0 / 6.0).abs() < 1e-5);
        assert!((diag.z - 1.0 / 6.0).abs() < 1e-5);
    }

    #[test]
    fn larger_cube_volume_cubes_with_side() {
        // Half-extent 1 → side 2 → volume 8.
        let (v, i) = unit_cube(1.0);
        let data = TriMeshData::new(v, i);
        assert!((data.unit_density_mass() - 8.0).abs() < 1e-4);
    }

    #[test]
    fn larger_cube_inertia_matches_closed_form() {
        // Side s = 2, unit density → mass 8, I = (1/6)·8·4 = 16/3.
        let (v, i) = unit_cube(1.0);
        let data = TriMeshData::new(v, i);
        let expected = (1.0 / 6.0) * 8.0 * 4.0;
        let diag = data.unit_density_inertia();
        assert!((diag.x - expected).abs() < 1e-3);
        assert!((diag.y - expected).abs() < 1e-3);
        assert!((diag.z - expected).abs() < 1e-3);
    }

    #[test]
    fn reversed_winding_still_reports_positive_mass() {
        // Flip every triangle's winding; the enclosed volume must stay positive.
        let (v, i) = unit_cube(0.5);
        let flipped: Vec<[u32; 3]> = i.into_iter().map(|[a, b, c]| [a, c, b]).collect();
        let data = TriMeshData::new(v, flipped);
        assert!((data.unit_density_mass() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn empty_mesh_has_zero_mass() {
        let data = TriMeshData::new(Vec::new(), Vec::new());
        assert_eq!(data.unit_density_mass(), 0.0);
        assert_eq!(data.unit_density_inertia(), Vec3::ZERO);
        let (min, max) = data.local_aabb();
        assert_eq!(min, Vec3::ZERO);
        assert_eq!(max, Vec3::ZERO);
    }

    #[test]
    fn empty_mesh_mass_properties_are_immovable() {
        let data = TriMeshData::new(Vec::new(), Vec::new());
        let mp = data.mass_properties(10.0);
        assert_eq!(mp.inv_mass, 0.0);
        assert_eq!(mp.inv_inertia, Vec3::ZERO);
    }

    #[test]
    fn open_shell_is_treated_as_massless() {
        // A single triangle encloses no volume → not a solid.
        let v = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let data = TriMeshData::new(v, vec![[0, 1, 2]]);
        assert_eq!(data.unit_density_mass(), 0.0);
    }

    #[test]
    fn non_positive_density_is_immovable() {
        let (v, i) = unit_cube(0.5);
        let data = TriMeshData::new(v, i);
        let mp = data.mass_properties(0.0);
        assert_eq!(mp.inv_mass, 0.0);
        assert_eq!(mp.inv_inertia, Vec3::ZERO);
    }

    #[test]
    fn offset_cube_volume_is_origin_independent() {
        // Translate the cube off the origin; enclosed volume must not change.
        let (mut v, i) = unit_cube(0.5);
        for p in &mut v {
            *p += Vec3::new(5.0, -3.0, 2.0);
        }
        let data = TriMeshData::new(v, i);
        assert!((data.unit_density_mass() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn handle_is_copy_and_hashable() {
        use std::collections::HashSet;
        let h = TriMeshHandle(7);
        let copy = h;
        assert_eq!(h, copy);
        let mut set = HashSet::new();
        set.insert(h);
        assert!(set.contains(&TriMeshHandle(7)));
    }
}
