//! Analytic collider shapes and a shared shape registry.
//!
//! [`ColliderShape`] enumerates the analytic primitives M0 understands. Each
//! shape can report a local-space axis-aligned bounding box and closed-form (or
//! documented-approximate) mass properties. [`ShapeRegistry`] lets many bodies
//! share a single shape definition by [`ColliderHandle`].
//!
//! The inertia formulas here are standard closed-form results and are not
//! derived from Unreal Engine source.

use crate::state::body::MassProperties;
use glam::Vec3;

pub mod material;

pub use material::PhysicsMaterial;
pub mod convex_mesh;

pub use convex_mesh::{ConvexMeshData, ConvexMeshHandle, ConvexProjection, ConvexRayHit};
pub mod tri_mesh;

pub use tri_mesh::{TriMeshData, TriMeshHandle};
pub mod hull;

pub use hull::convex_hull;
pub mod decompose;

pub use decompose::{convex_decompose, DecompositionParams};

/// A handle into a [`ShapeRegistry`].
///
/// This is a plain index handle; shapes are immutable once inserted, so no
/// generation counter is required for correctness in M0.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ColliderHandle(pub u32);

/// An analytic collision shape.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum ColliderShape {
    /// A solid sphere.
    Sphere {
        /// Sphere radius.
        radius: f32,
    },
    /// An axis-aligned box given by its half-extents in local space.
    Cuboid {
        /// Half the box size along each local axis.
        half_extents: Vec3,
    },
    /// A capsule aligned with the local Y axis: a cylinder of the given
    /// half-height capped by two hemispheres of the given radius.
    Capsule {
        /// Half the length of the central cylinder (excluding the caps).
        half_height: f32,
        /// Radius of the cylinder and the hemispherical caps.
        radius: f32,
    },
    /// An infinite half-space plane `dot(normal, x) = offset`.
    Plane {
        /// Unit outward normal of the plane.
        normal: Vec3,
        /// Signed distance of the plane from the origin along `normal`.
        offset: f32,
    },
    /// A bounded convex hull, stored in the owning [`ShapeRegistry`]'s
    /// convex-mesh arena and referenced here by [`ConvexMeshHandle`].
    ///
    /// The geometric quantities needed for the broad phase and mass setup
    /// ([`local_aabb`](ColliderShape::local_aabb),
    /// [`mass_properties`](ColliderShape::mass_properties)) are cached inline so
    /// those two methods stay registry-free and `Copy`; the full vertex/face
    /// data (needed by the narrow phase, scene queries, and CCD) is resolved
    /// from the arena via `mesh`. Build one with
    /// [`ShapeRegistry::convex_hull_shape`], which fills the cache from the
    /// stored [`ConvexMeshData`].
    ConvexHull {
        /// Handle into the registry's convex-mesh arena.
        mesh: ConvexMeshHandle,
        /// Cached local-space AABB minimum.
        local_aabb_min: Vec3,
        /// Cached local-space AABB maximum.
        local_aabb_max: Vec3,
        /// Cached enclosed volume (cubic metres).
        volume: f32,
        /// Cached unit-density diagonal inertia (scales linearly with density).
        unit_density_inertia: Vec3,
    },
    /// A static triangle mesh (concave allowed), stored in the owning
    /// [`ShapeRegistry`]'s triangle-mesh arena and referenced by
    /// [`TriMeshHandle`].
    ///
    /// Like [`ConvexHull`](ColliderShape::ConvexHull) the AABB and mass cache is
    /// inline so the two registry-free methods stay `Copy`; the triangle soup
    /// (needed by queries and the narrow phase) is resolved from the arena via
    /// `mesh`. Build one with [`ShapeRegistry::tri_mesh_shape`]. A triangle mesh
    /// is intended as immovable scene geometry: its cached mass is the
    /// closed-solid mass of the (possibly open) surface and callers typically
    /// pair it with a static body.
    TriangleMesh {
        /// Handle into the registry's triangle-mesh arena.
        mesh: TriMeshHandle,
        /// Cached local-space AABB minimum.
        local_aabb_min: Vec3,
        /// Cached local-space AABB maximum.
        local_aabb_max: Vec3,
        /// Cached unit-density mass (scales linearly with density).
        unit_density_mass: f32,
        /// Cached unit-density diagonal inertia (scales linearly with density).
        unit_density_inertia: Vec3,
    },
}

impl ColliderShape {
    /// Returns the local-space axis-aligned bounding box as `(min, max)`.
    ///
    /// A [`ColliderShape::Plane`] is unbounded; it reports a degenerate box
    /// that spans the full finite range so that broad-phase code can treat it
    /// as "covers everything" without producing NaNs.
    #[must_use]
    pub fn local_aabb(&self) -> (Vec3, Vec3) {
        match *self {
            ColliderShape::Sphere { radius } => (Vec3::splat(-radius), Vec3::splat(radius)),
            ColliderShape::Cuboid { half_extents } => (-half_extents, half_extents),
            ColliderShape::Capsule {
                half_height,
                radius,
            } => {
                let ext = Vec3::new(radius, half_height + radius, radius);
                (-ext, ext)
            }
            ColliderShape::Plane { .. } => {
                let big = Vec3::splat(f32::MAX);
                (-big, big)
            }
            ColliderShape::ConvexHull {
                local_aabb_min,
                local_aabb_max,
                ..
            }
            | ColliderShape::TriangleMesh {
                local_aabb_min,
                local_aabb_max,
                ..
            } => (local_aabb_min, local_aabb_max),
        }
    }

    /// Computes mass properties for this shape at uniform `density`.
    ///
    /// - [`ColliderShape::Sphere`] and [`ColliderShape::Cuboid`] use exact
    ///   closed-form inertia.
    /// - [`ColliderShape::Capsule`] uses a documented approximation: a solid
    ///   cylinder plus a solid sphere (the two caps), with the sphere's
    ///   perpendicular contribution offset by the half-height via the parallel
    ///   axis theorem. This is accurate for the axial moment and a good
    ///   approximation for the perpendicular moments.
    /// - [`ColliderShape::Plane`] is treated as immovable and returns
    ///   [`MassProperties::zero`].
    #[must_use]
    pub fn mass_properties(&self, density: f32) -> MassProperties {
        match *self {
            ColliderShape::Sphere { radius } => {
                let mass =
                    density * (4.0 / 3.0) * crate::math::scalar::PI * radius * radius * radius;
                let inertia = 0.4 * mass * radius * radius;
                mass_props_from_diagonal(mass, Vec3::splat(inertia))
            }
            ColliderShape::Cuboid { half_extents } => {
                let full = half_extents * 2.0;
                let mass = density * full.x * full.y * full.z;
                let ix = (1.0 / 12.0) * mass * (full.y * full.y + full.z * full.z);
                let iy = (1.0 / 12.0) * mass * (full.x * full.x + full.z * full.z);
                let iz = (1.0 / 12.0) * mass * (full.x * full.x + full.y * full.y);
                mass_props_from_diagonal(mass, Vec3::new(ix, iy, iz))
            }
            ColliderShape::Capsule {
                half_height,
                radius,
            } => {
                let pi = crate::math::scalar::PI;
                let cyl_h = 2.0 * half_height;
                let cyl_mass = density * pi * radius * radius * cyl_h;
                let sphere_mass = density * (4.0 / 3.0) * pi * radius * radius * radius;
                let total = cyl_mass + sphere_mass;

                // Axial moment (about the Y capsule axis).
                let iy = 0.5 * cyl_mass * radius * radius + 0.4 * sphere_mass * radius * radius;

                // Perpendicular moment (about X/Z), cylinder plus offset sphere.
                let cyl_perp = cyl_mass * (cyl_h * cyl_h / 12.0 + radius * radius / 4.0);
                let sphere_perp =
                    0.4 * sphere_mass * radius * radius + sphere_mass * half_height * half_height;
                let iperp = cyl_perp + sphere_perp;

                mass_props_from_diagonal(total, Vec3::new(iperp, iy, iperp))
            }
            ColliderShape::Plane { .. } => MassProperties::zero(),
            ColliderShape::ConvexHull {
                volume,
                unit_density_inertia,
                ..
            } => {
                if volume <= 0.0 || density <= 0.0 {
                    return MassProperties::zero();
                }
                mass_props_from_diagonal(density * volume, unit_density_inertia * density)
            }
            ColliderShape::TriangleMesh {
                unit_density_mass,
                unit_density_inertia,
                ..
            } => {
                if unit_density_mass <= 0.0 || density <= 0.0 {
                    return MassProperties::zero();
                }
                mass_props_from_diagonal(
                    density * unit_density_mass,
                    unit_density_inertia * density,
                )
            }
        }
    }
}

/// Builds [`MassProperties`] from a total mass and a diagonal inertia tensor,
/// inverting each nonzero component and leaving zero (infinite) components at
/// zero.
fn mass_props_from_diagonal(mass: f32, inertia: Vec3) -> MassProperties {
    let inv_mass = if mass > 0.0 { 1.0 / mass } else { 0.0 };
    let inv = |i: f32| if i > 0.0 { 1.0 / i } else { 0.0 };
    MassProperties {
        inv_mass,
        inv_inertia: Vec3::new(inv(inertia.x), inv(inertia.y), inv(inertia.z)),
    }
}

/// A registry of shared, immutable collision shapes.
#[derive(Clone, Debug, Default)]
pub struct ShapeRegistry {
    shapes: Vec<ColliderShape>,
    convex_meshes: Vec<ConvexMeshData>,
    tri_meshes: Vec<TriMeshData>,
}

impl ShapeRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> ShapeRegistry {
        ShapeRegistry::default()
    }

    /// Inserts a shape and returns a handle referring to it.
    pub fn insert(&mut self, shape: ColliderShape) -> ColliderHandle {
        let handle = ColliderHandle(self.shapes.len() as u32);
        self.shapes.push(shape);
        handle
    }

    /// Returns a reference to the shape for `handle`, or `None` if the handle
    /// is out of range.
    #[must_use]
    pub fn get(&self, handle: ColliderHandle) -> Option<&ColliderShape> {
        self.shapes.get(handle.0 as usize)
    }

    /// Returns the number of registered shapes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.shapes.len()
    }

    /// Returns `true` if the registry holds no shapes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.shapes.is_empty()
    }

    /// Inserts a convex mesh into the convex-mesh arena and returns a handle.
    pub fn insert_convex_mesh(&mut self, mesh: ConvexMeshData) -> ConvexMeshHandle {
        let handle = ConvexMeshHandle(self.convex_meshes.len() as u32);
        self.convex_meshes.push(mesh);
        handle
    }

    /// Returns a reference to the convex mesh for `handle`, or `None` if the
    /// handle is out of range.
    #[must_use]
    pub fn convex_mesh(&self, handle: ConvexMeshHandle) -> Option<&ConvexMeshData> {
        self.convex_meshes.get(handle.0 as usize)
    }

    /// Returns the number of convex meshes stored in the arena.
    #[must_use]
    pub fn convex_mesh_count(&self) -> usize {
        self.convex_meshes.len()
    }

    /// Inserts a triangle mesh into the triangle-mesh arena and returns a handle.
    pub fn insert_tri_mesh(&mut self, mesh: TriMeshData) -> TriMeshHandle {
        let handle = TriMeshHandle(self.tri_meshes.len() as u32);
        self.tri_meshes.push(mesh);
        handle
    }

    /// Returns a reference to the triangle mesh for `handle`, or `None` if the
    /// handle is out of range.
    #[must_use]
    pub fn tri_mesh(&self, handle: TriMeshHandle) -> Option<&TriMeshData> {
        self.tri_meshes.get(handle.0 as usize)
    }

    /// Returns the number of triangle meshes stored in the arena.
    #[must_use]
    pub fn tri_mesh_count(&self) -> usize {
        self.tri_meshes.len()
    }

    /// Builds a [`ColliderShape::ConvexHull`] referring to the convex mesh at
    /// `handle`, filling the inline broad-phase / mass cache from the stored
    /// [`ConvexMeshData`].
    ///
    /// Returns [`None`] when `handle` is out of range. The cached AABB, volume,
    /// and unit-density inertia are read once here so that
    /// [`ColliderShape::local_aabb`] and [`ColliderShape::mass_properties`] stay
    /// registry-free and `Copy`.
    #[must_use]
    pub fn convex_hull_shape(&self, handle: ConvexMeshHandle) -> Option<ColliderShape> {
        let mesh = self.convex_meshes.get(handle.0 as usize)?;
        let (local_aabb_min, local_aabb_max) = mesh.local_aabb();
        Some(ColliderShape::ConvexHull {
            mesh: handle,
            local_aabb_min,
            local_aabb_max,
            volume: mesh.volume(),
            unit_density_inertia: mesh.unit_density_inertia(),
        })
    }

    /// Builds a [`ColliderShape::TriangleMesh`] referring to the triangle mesh
    /// at `handle`, filling the inline broad-phase / mass cache from the stored
    /// [`TriMeshData`].
    ///
    /// Returns [`None`] when `handle` is out of range. A triangle mesh is
    /// intended as immovable scene geometry; the cached closed-solid mass is
    /// kept so callers that make it dynamic still receive a finite inertia.
    #[must_use]
    pub fn tri_mesh_shape(&self, handle: TriMeshHandle) -> Option<ColliderShape> {
        let mesh = self.tri_meshes.get(handle.0 as usize)?;
        let (local_aabb_min, local_aabb_max) = mesh.local_aabb();
        Some(ColliderShape::TriangleMesh {
            mesh: handle,
            local_aabb_min,
            local_aabb_max,
            unit_density_mass: mesh.unit_density_mass(),
            unit_density_inertia: mesh.unit_density_inertia(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_inertia_matches_closed_form() {
        let density = 3.0;
        let radius = 2.0;
        let shape = ColliderShape::Sphere { radius };
        let mp = shape.mass_properties(density);

        let mass = density * (4.0 / 3.0) * crate::math::scalar::PI * radius * radius * radius;
        let inertia = 0.4 * mass * radius * radius;
        assert!((mp.mass() - mass).abs() < 1e-3);
        let expected_inv = 1.0 / inertia;
        assert!((mp.inv_inertia.x - expected_inv).abs() < 1e-6);
        assert!((mp.inv_inertia.y - expected_inv).abs() < 1e-6);
        assert!((mp.inv_inertia.z - expected_inv).abs() < 1e-6);
    }

    #[test]
    fn cuboid_inertia_matches_closed_form() {
        let density = 2.0;
        let he = Vec3::new(1.0, 2.0, 3.0);
        let shape = ColliderShape::Cuboid { half_extents: he };
        let mp = shape.mass_properties(density);

        let full = he * 2.0;
        let mass = density * full.x * full.y * full.z;
        let ix = (1.0 / 12.0) * mass * (full.y * full.y + full.z * full.z);
        let iz = (1.0 / 12.0) * mass * (full.x * full.x + full.y * full.y);
        assert!((mp.mass() - mass).abs() < 1e-3);
        assert!((mp.inv_inertia.x - 1.0 / ix).abs() < 1e-6);
        assert!((mp.inv_inertia.z - 1.0 / iz).abs() < 1e-6);
    }

    #[test]
    fn plane_is_immovable() {
        let shape = ColliderShape::Plane {
            normal: Vec3::Y,
            offset: 0.0,
        };
        let mp = shape.mass_properties(5.0);
        assert_eq!(mp.inv_mass, 0.0);
        assert_eq!(mp.inv_inertia, Vec3::ZERO);
    }

    #[test]
    fn aabb_bounds_are_correct() {
        let (min, max) = ColliderShape::Sphere { radius: 1.5 }.local_aabb();
        assert_eq!(min, Vec3::splat(-1.5));
        assert_eq!(max, Vec3::splat(1.5));
    }

    #[test]
    fn registry_shares_shapes() {
        let mut reg = ShapeRegistry::new();
        assert!(reg.is_empty());
        let h = reg.insert(ColliderShape::Sphere { radius: 1.0 });
        assert_eq!(reg.len(), 1);
        assert!(matches!(
            reg.get(h),
            Some(ColliderShape::Sphere { radius }) if (*radius - 1.0).abs() < 1e-6
        ));
        assert!(reg.get(ColliderHandle(99)).is_none());
    }
}
