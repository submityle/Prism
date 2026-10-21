//! Mapping from rigid [`ColliderShape`] bodies to soft-body [`BodyCollider`]
//! coupling proxies, plus the lightweight AABB overlap used to cull proxies.
//!
//! The two-way coupling kernel ([`crate::soft::resolve_two_way_coupling`])
//! works against the soft side's analytic [`BodyCollider`] primitives, not the
//! rigid-body [`ColliderShape`] registry. This module is the adapter: it turns
//! a rigid body's shape and world pose into the matching [`BodyCollider`], and
//! derives the proxy's inverse mass from its [`BodyKind`] and mass properties.
//!
//! # Shape mapping
//!
//! * [`ColliderShape::Sphere`] → [`BodyCollider::Sphere`] at the body's
//!   world position.
//! * [`ColliderShape::Capsule`] (local `Y` axis) → [`BodyCollider::Capsule`]
//!   whose endpoints are the body position offset by `±half_height` along the
//!   oriented local `Y` axis.
//! * [`ColliderShape::Plane`] (feasible region `dot(normal, x) >= offset` in the
//!   rigid convention, with the world plane shifted by the body position) →
//!   [`BodyCollider::HalfSpace`] with the same feasible-region convention.
//! * [`ColliderShape::Cuboid`] → [`BodyCollider::Obb`] at the body's world pose
//!   (center = position, orientation = body orientation, matching half-extents),
//!   so a box prop collides against cloth via its true oriented-box faces.
//! * [`ColliderShape::Compound`] → multiple real leaf proxies through
//!   [`body_colliders_from_shape_in`]. Single-proxy converters deliberately
//!   return `None` rather than fill the union's concave gaps.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The shape
//! conversions are elementary rigid-transform algebra.

use glam::{Quat, Vec3};

use crate::collider::{ColliderShape, ShapeRegistry};
use crate::math::scalar::Real;
use crate::soft::{BodyCollider, ConvexProxy, CouplingBody, Plane};
use crate::state::body::BodyKind;
use crate::state::handle::BodyHandle;

/// A rigid body paired with the soft-side coupling proxy built from it.
///
/// The [`handle`](RigidProxy::handle) identifies which rigid body the proxy
/// stands in for, so the driver can write the accumulated reaction impulse back
/// onto the correct body after the coupling pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RigidProxy {
    /// The rigid body this proxy represents.
    pub handle: BodyHandle,
    /// Root body's pre-pass world COM, not the child collider's center.
    pub center_of_mass: Vec3,
    /// The coupling body (collider + inverse mass + reaction accumulator) the
    /// two-way pass operates on.
    pub body: CouplingBody,
}

/// Why a collision leaf cannot be represented by this analytic soft bridge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CouplingProxyError {
    /// Collider or backing geometry is absent from the registry.
    InvalidGeometry,
    /// A zero-thickness triangle surface needs swept particles/contact thickness,
    /// neither of which is supplied to the penetration-only normal bridge.
    TriangleSurface,
    /// The bounded analytic proxy cannot store all convex face planes.
    ConvexPlaneCapacity,
}

/// Converts the entire leaf collection, with composed world transforms.
/// Failure is atomic: unsupported leaves are reported, never silently dropped.
/// The normal bridge supports precisely the same solid primitives/convex hulls
/// as before; compounds do not introduce a hull/AABB approximation.
pub fn body_colliders_from_shape_in(
    shape: &ColliderShape,
    position: Vec3,
    orientation: Quat,
    shapes: &ShapeRegistry,
) -> Result<Vec<BodyCollider>, CouplingProxyError> {
    use crate::math::transform::Isometry;
    if let ColliderShape::Compound { compound, .. } = *shape {
        let root = Isometry::new(position, orientation);
        let mut proxies = Vec::new();
        for child in shapes
            .compound(compound)
            .ok_or(CouplingProxyError::InvalidGeometry)?
            .children()
        {
            let pose = root.mul(&child.local_pose);
            let shape = shapes
                .get(child.collider)
                .ok_or(CouplingProxyError::InvalidGeometry)?;
            proxies.extend(body_colliders_from_shape_in(
                shape,
                pose.translation,
                pose.rotation,
                shapes,
            )?);
        }
        return Ok(proxies);
    }
    let error = match *shape {
        ColliderShape::TriangleMesh { .. } => CouplingProxyError::TriangleSurface,
        ColliderShape::ConvexHull { mesh, .. } if shapes.convex_mesh(mesh).is_some() => {
            CouplingProxyError::ConvexPlaneCapacity
        }
        _ => CouplingProxyError::InvalidGeometry,
    };
    body_collider_from_shape_in(shape, position, orientation, shapes)
        .map(|p| vec![p])
        .ok_or(error)
}

/// Builds the soft-side [`BodyCollider`] for a rigid [`ColliderShape`] at the
/// given world pose, or [`None`] when the shape has no analytic proxy.
///
/// `position` is the body's world-space origin and `orientation` its world
/// orientation. See the module docs for the per-shape mapping. A
/// [`ColliderShape::Cuboid`] maps to a [`BodyCollider::Obb`] carrying the body's
/// orientation and half-extents. The function still returns [`None`] for any
/// future shape that has no analytic proxy.
#[must_use]
pub fn body_collider_from_shape(
    shape: &ColliderShape,
    position: Vec3,
    orientation: Quat,
) -> Option<BodyCollider> {
    match *shape {
        ColliderShape::Sphere { radius } => Some(BodyCollider::Sphere {
            center: position,
            radius,
        }),
        ColliderShape::Capsule {
            half_height,
            radius,
        } => {
            let axis = orientation * Vec3::Y;
            Some(BodyCollider::Capsule {
                p0: position - axis * half_height,
                p1: position + axis * half_height,
                radius,
            })
        }
        ColliderShape::Plane { normal, offset } => {
            // Rotate the local plane normal into the world and shift the offset
            // so the plane passes through the body position: a point `x` on the
            // world plane satisfies `world_normal.dot(x) == offset +
            // world_normal.dot(position)`.
            let world_normal = orientation * normal;
            Some(BodyCollider::HalfSpace {
                normal: world_normal,
                offset: offset + world_normal.dot(position),
            })
        }
        ColliderShape::Cuboid { half_extents } => Some(BodyCollider::Obb {
            center: position,
            orientation,
            half_extents,
        }),
        // Convex-hull and triangle-mesh shapes carry their geometry in the
        // registry's arenas, not inline, so they have no analytic proxy from
        // this registry-free function. They are dispatched by the
        // registry-aware [`body_collider_from_shape_in`]; this function handles
        // the analytic (self-describing) shapes only.
        ColliderShape::ConvexHull { .. } | ColliderShape::TriangleMesh { .. } => None,
        // A compound is a potentially concave union, not one convex proxy.
        ColliderShape::Compound { .. } => None,
    }
}

/// Registry-aware variant of [`body_collider_from_shape`] that can resolve the
/// arena-backed convex-hull and triangle-mesh shapes.
///
/// Analytic shapes (sphere, capsule, plane, cuboid) are delegated verbatim to
/// [`body_collider_from_shape`], so this function is a strict superset of it.
/// The arena-backed shapes are resolved through `shapes`:
///
/// * [`ColliderShape::ConvexHull`] → [`BodyCollider::ConvexHull`]: the hull's
///   local outward face half-spaces (from
///   [`ConvexMeshData::face_half_spaces`](crate::collider::ConvexMeshData::face_half_spaces))
///   are transformed into world space (normal rotated by `orientation`, offset
///   shifted by `world_normal · position`) and fed to
///   [`ConvexProxy::from_planes`]. A hull with more than
///   [`crate::soft::MAX_CONVEX_PLANES`] faces has no bounded proxy, so
///   [`ConvexProxy::from_planes`] returns [`None`] and so does this function —
///   an honest, documented coverage gap rather than a silent truncation.
/// * [`ColliderShape::TriangleMesh`] → [`None`]: a triangle mesh is immovable
///   scene geometry with no analytic (bounded, `Copy`) convex proxy, so it is
///   skipped exactly like any other shape without a proxy.
/// * [`ColliderShape::Compound`] → [`None`]: even with its registry available,
///   a potentially concave union cannot be represented by one convex proxy.
///   Use [`body_colliders_from_shape_in`] to expand the real leaves.
///
/// Returns [`None`] when the shape has no coupling proxy or when the referenced
/// mesh handle is not present in `shapes`.
///
/// # Provenance
///
/// The world transform of the face half-spaces is elementary rigid-transform
/// algebra; this contains **no Unreal Engine source or derived code**.
#[must_use]
pub fn body_collider_from_shape_in(
    shape: &ColliderShape,
    position: Vec3,
    orientation: Quat,
    shapes: &ShapeRegistry,
) -> Option<BodyCollider> {
    match *shape {
        // Analytic, self-describing shapes need no registry lookup.
        ColliderShape::Sphere { .. }
        | ColliderShape::Capsule { .. }
        | ColliderShape::Plane { .. }
        | ColliderShape::Cuboid { .. } => body_collider_from_shape(shape, position, orientation),
        ColliderShape::ConvexHull { mesh, .. } => {
            let data = shapes.convex_mesh(mesh)?;
            let planes: Vec<Plane> = data
                .face_half_spaces()
                .into_iter()
                .map(|(n_local, off_local)| {
                    // Transform the local face plane `n_local · x = off_local`
                    // into world space. A world point `y = R·x + position`
                    // satisfies `(R·n_local) · y = off_local + (R·n_local) ·
                    // position`, so the world normal is `orientation * n_local`
                    // and the world offset shifts by `world_normal · position`.
                    let world_normal = (orientation * n_local).normalize_or_zero();
                    let world_offset = off_local + world_normal.dot(position);
                    Plane {
                        normal: world_normal,
                        offset: world_offset,
                    }
                })
                .collect();
            ConvexProxy::from_planes(position, data.bounding_radius(), &planes)
                .map(BodyCollider::ConvexHull)
        }
        // A triangle mesh is static scene geometry with no bounded analytic
        // proxy; it is skipped, matching the "no proxy -> None" contract.
        ColliderShape::TriangleMesh { .. } => None,
        // Reject rather than fill the union's gaps with a hull or an AABB.
        ColliderShape::Compound { .. } => None,
    }
}

/// Builds a bounded [`ConvexProxy`] for a cuboid at the given world pose.
///
/// This is the opt-in convex mapping of [`ColliderShape::Cuboid`]: it produces
/// the box as the intersection of its six oriented face half-spaces, which
/// projects interior particles identically to the default
/// [`BodyCollider::Obb`] mapping. It exists so callers (and the cross-validation
/// goldens) can exercise the convex arm against a shape whose oriented-box
/// answer is already known; the default [`body_collider_from_shape`] mapping is
/// unchanged, so existing goldens stay bit-identical.
#[must_use]
pub fn convex_proxy_from_cuboid(
    half_extents: Vec3,
    position: Vec3,
    orientation: Quat,
) -> ConvexProxy {
    ConvexProxy::from_box(position, orientation, half_extents)
}

/// Returns the inverse mass to give a coupling proxy for a rigid body.
///
/// Only [`BodyKind::Dynamic`] bodies get a movable proxy (their clamped
/// `inv_mass`); [`BodyKind::Static`] and [`BodyKind::Kinematic`] bodies get a
/// zero inverse mass so the proxy never moves but still records the reaction
/// impulse the cloth applies to it.
#[must_use]
pub fn proxy_inverse_mass(kind: BodyKind, inv_mass: Real) -> Real {
    match kind {
        BodyKind::Dynamic => inv_mass.max(0.0),
        BodyKind::Static | BodyKind::Kinematic => 0.0,
    }
}

/// Returns the representative anchor point of a coupling collider.
///
/// This is the point whose displacement over the coupling pass equals the rigid
/// translation the pass applied to the proxy: a sphere's center, a capsule's
/// segment midpoint, or the origin for a half-space (which the pass shifts by
/// its offset rather than translating a point).
#[must_use]
pub fn collider_anchor(collider: BodyCollider) -> Vec3 {
    match collider {
        BodyCollider::Sphere { center, .. } | BodyCollider::Obb { center, .. } => center,
        BodyCollider::Capsule { p0, p1, .. } => (p0 + p1) * 0.5,
        BodyCollider::HalfSpace { .. } => Vec3::ZERO,
        BodyCollider::ConvexHull(proxy) => proxy.center(),
    }
}

/// An axis-aligned bounding box used to cull coupling proxies against a soft
/// body's extent.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    /// Minimum corner (component-wise).
    pub min: Vec3,
    /// Maximum corner (component-wise).
    pub max: Vec3,
}

impl Aabb {
    /// Builds the tight AABB enclosing `points`, or [`None`] when the slice is
    /// empty (an empty soft body has no extent to cull against).
    #[must_use]
    pub fn of_points(points: &[Vec3]) -> Option<Aabb> {
        let mut iter = points.iter();
        let first = *iter.next()?;
        let mut min = first;
        let mut max = first;
        for &p in iter {
            min = min.min(p);
            max = max.max(p);
        }
        Some(Aabb { min, max })
    }

    /// Returns `true` when this box overlaps `other` on all three axes
    /// (closed intervals, so touching counts as overlap).
    #[must_use]
    pub fn intersects(&self, other: &Aabb) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }
}

/// Returns the world-space AABB of a coupling collider, or [`None`] for an
/// unbounded half-space (which overlaps every finite box).
#[must_use]
pub fn collider_world_aabb(collider: BodyCollider) -> Option<Aabb> {
    match collider {
        BodyCollider::Sphere { center, radius } => {
            let r = Vec3::splat(radius.max(0.0));
            Some(Aabb {
                min: center - r,
                max: center + r,
            })
        }
        BodyCollider::Capsule { p0, p1, radius } => {
            let r = Vec3::splat(radius.max(0.0));
            Some(Aabb {
                min: p0.min(p1) - r,
                max: p0.max(p1) + r,
            })
        }
        BodyCollider::Obb {
            center,
            orientation,
            half_extents,
        } => {
            // World AABB of the oriented box: the world half-extent along each
            // world axis is the sum over the local axes of
            // `|rotation_component| * half_extent`, i.e. `|R| * he` with the
            // component-wise absolute rotation matrix. Negative extents are
            // clamped to zero so a collapsed axis contributes nothing.
            let m = glam::Mat3::from_quat(orientation);
            let abs = glam::Mat3::from_cols(m.x_axis.abs(), m.y_axis.abs(), m.z_axis.abs());
            let world_half = abs * half_extents.max(Vec3::ZERO);
            Some(Aabb {
                min: center - world_half,
                max: center + world_half,
            })
        }
        BodyCollider::ConvexHull(proxy) => {
            // Conservative bounding box from the proxy's cached bounding sphere.
            let r = Vec3::splat(proxy.bounding_radius().max(0.0));
            let center = proxy.center();
            Some(Aabb {
                min: center - r,
                max: center + r,
            })
        }
        BodyCollider::HalfSpace { .. } => None,
    }
}

/// Returns `true` when a coupling collider overlaps the soft body's AABB.
///
/// A bounded collider is tested with [`Aabb::intersects`]; an unbounded
/// half-space always overlaps (it extends to infinity), so it is never culled.
#[must_use]
pub fn collider_overlaps(collider: BodyCollider, soft_aabb: &Aabb) -> bool {
    match collider_world_aabb(collider) {
        Some(box_aabb) => box_aabb.intersects(soft_aabb),
        None => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compound_is_explicitly_rejected_by_both_single_proxy_converters() {
        use crate::collider::CompoundChild;
        use crate::math::transform::Isometry;

        let mut shapes = ShapeRegistry::new();
        let sphere = shapes.insert(ColliderShape::Sphere { radius: 0.5 });
        let children = [-2.0, 2.0].map(|x| CompoundChild {
            collider: sphere,
            local_pose: Isometry::new(Vec3::new(x, 0.0, 0.0), Quat::IDENTITY),
        });
        let handle = shapes.insert_compound(&children).expect("valid union");
        let shape = shapes.get(handle).expect("registered compound");
        let position = Vec3::new(1.0, 2.0, 3.0);
        let rotation = Quat::from_rotation_y(0.7);

        // Registry access must not silently turn two disjoint spheres into a
        // solid box/hull spanning the empty space between them.
        assert_eq!(body_collider_from_shape(shape, position, rotation), None);
        assert_eq!(
            body_collider_from_shape_in(shape, position, rotation, &shapes),
            None
        );
    }

    #[test]
    fn sphere_shape_maps_to_sphere_at_position() {
        let shape = ColliderShape::Sphere { radius: 0.5 };
        let got = body_collider_from_shape(&shape, Vec3::new(1.0, 2.0, 3.0), Quat::IDENTITY);
        assert_eq!(
            got,
            Some(BodyCollider::Sphere {
                center: Vec3::new(1.0, 2.0, 3.0),
                radius: 0.5,
            })
        );
    }

    #[test]
    fn capsule_shape_maps_endpoints_along_oriented_axis() {
        let shape = ColliderShape::Capsule {
            half_height: 1.0,
            radius: 0.25,
        };
        // Identity orientation: axis is +Y.
        let got =
            body_collider_from_shape(&shape, Vec3::ZERO, Quat::IDENTITY).expect("capsule proxy");
        let BodyCollider::Capsule { p0, p1, radius } = got else {
            panic!("expected capsule");
        };
        assert!((p0 - Vec3::new(0.0, -1.0, 0.0)).length() < 1e-6);
        assert!((p1 - Vec3::new(0.0, 1.0, 0.0)).length() < 1e-6);
        assert!((radius - 0.25).abs() < 1e-9);
    }

    #[test]
    fn plane_shape_maps_to_half_space_through_position() {
        let shape = ColliderShape::Plane {
            normal: Vec3::Y,
            offset: 0.0,
        };
        let got = body_collider_from_shape(&shape, Vec3::new(0.0, 2.0, 0.0), Quat::IDENTITY)
            .expect("plane proxy");
        let BodyCollider::HalfSpace { normal, offset } = got else {
            panic!("expected half-space");
        };
        assert!((normal - Vec3::Y).length() < 1e-6);
        // Plane now passes through y = 2.
        assert!((offset - 2.0).abs() < 1e-6);
    }

    #[test]
    fn cuboid_shape_maps_to_obb_at_pose() {
        let shape = ColliderShape::Cuboid {
            half_extents: Vec3::new(0.5, 0.25, 1.0),
        };
        let orientation = Quat::from_rotation_y(std::f32::consts::FRAC_PI_4);
        let got = body_collider_from_shape(&shape, Vec3::new(1.0, 2.0, 3.0), orientation)
            .expect("cuboid proxy");
        assert_eq!(
            got,
            BodyCollider::Obb {
                center: Vec3::new(1.0, 2.0, 3.0),
                orientation,
                half_extents: Vec3::new(0.5, 0.25, 1.0),
            }
        );
    }

    #[test]
    fn inverse_mass_zero_for_non_dynamic() {
        assert_eq!(proxy_inverse_mass(BodyKind::Static, 5.0), 0.0);
        assert_eq!(proxy_inverse_mass(BodyKind::Kinematic, 5.0), 0.0);
        assert_eq!(proxy_inverse_mass(BodyKind::Dynamic, 5.0), 5.0);
        // Negative clamps to zero.
        assert_eq!(proxy_inverse_mass(BodyKind::Dynamic, -1.0), 0.0);
    }

    #[test]
    fn anchor_is_center_or_midpoint() {
        assert_eq!(
            collider_anchor(BodyCollider::Sphere {
                center: Vec3::new(1.0, 2.0, 3.0),
                radius: 1.0,
            }),
            Vec3::new(1.0, 2.0, 3.0)
        );
        assert_eq!(
            collider_anchor(BodyCollider::Capsule {
                p0: Vec3::new(0.0, 0.0, 0.0),
                p1: Vec3::new(0.0, 2.0, 0.0),
                radius: 1.0,
            }),
            Vec3::new(0.0, 1.0, 0.0)
        );
        assert_eq!(
            collider_anchor(BodyCollider::Obb {
                center: Vec3::new(-2.0, 1.0, 4.0),
                orientation: Quat::from_rotation_x(0.3),
                half_extents: Vec3::new(0.5, 0.5, 0.5),
            }),
            Vec3::new(-2.0, 1.0, 4.0)
        );
    }

    #[test]
    fn aabb_of_points_handles_empty_and_nonempty() {
        assert_eq!(Aabb::of_points(&[]), None);
        let aabb = Aabb::of_points(&[Vec3::new(-1.0, 0.0, 2.0), Vec3::new(1.0, 3.0, -1.0)])
            .expect("non-empty");
        assert_eq!(aabb.min, Vec3::new(-1.0, 0.0, -1.0));
        assert_eq!(aabb.max, Vec3::new(1.0, 3.0, 2.0));
    }

    #[test]
    fn aabb_intersection_is_closed() {
        let a = Aabb {
            min: Vec3::ZERO,
            max: Vec3::splat(1.0),
        };
        let touching = Aabb {
            min: Vec3::new(1.0, 0.0, 0.0),
            max: Vec3::new(2.0, 1.0, 1.0),
        };
        let apart = Aabb {
            min: Vec3::new(1.5, 0.0, 0.0),
            max: Vec3::new(2.0, 1.0, 1.0),
        };
        assert!(a.intersects(&touching));
        assert!(!a.intersects(&apart));
    }

    #[test]
    fn half_space_never_culled() {
        let soft = Aabb {
            min: Vec3::splat(100.0),
            max: Vec3::splat(101.0),
        };
        let hs = BodyCollider::HalfSpace {
            normal: Vec3::Y,
            offset: 0.0,
        };
        assert_eq!(collider_world_aabb(hs), None);
        assert!(collider_overlaps(hs, &soft));
    }

    #[test]
    fn obb_world_aabb_is_axis_aligned_when_unrotated() {
        let obb = BodyCollider::Obb {
            center: Vec3::new(1.0, 2.0, 3.0),
            orientation: Quat::IDENTITY,
            half_extents: Vec3::new(0.5, 0.25, 2.0),
        };
        let aabb = collider_world_aabb(obb).expect("bounded");
        assert!((aabb.min - Vec3::new(0.5, 1.75, 1.0)).length() < 1e-6);
        assert!((aabb.max - Vec3::new(1.5, 2.25, 5.0)).length() < 1e-6);
    }

    #[test]
    fn obb_world_aabb_grows_when_rotated() {
        // A 45-degree yaw grows the XZ footprint of a square base from
        // half-extent 1 to half-extent sqrt(2); the Y extent is unchanged.
        let obb = BodyCollider::Obb {
            center: Vec3::ZERO,
            orientation: Quat::from_rotation_y(std::f32::consts::FRAC_PI_4),
            half_extents: Vec3::new(1.0, 0.5, 1.0),
        };
        let aabb = collider_world_aabb(obb).expect("bounded");
        let expected = 2.0_f32.sqrt();
        assert!((aabb.max.x - expected).abs() < 1e-6, "x: {}", aabb.max.x);
        assert!((aabb.max.z - expected).abs() < 1e-6, "z: {}", aabb.max.z);
        assert!((aabb.max.y - 0.5).abs() < 1e-6, "y: {}", aabb.max.y);
        assert!(
            (aabb.min + aabb.max).length() < 1e-6,
            "symmetric about center"
        );
    }

    #[test]
    fn obb_overlaps_soft_box_when_near() {
        let soft = Aabb {
            min: Vec3::splat(-1.0),
            max: Vec3::splat(1.0),
        };
        let near = BodyCollider::Obb {
            center: Vec3::new(1.4, 0.0, 0.0),
            orientation: Quat::IDENTITY,
            half_extents: Vec3::splat(0.5),
        };
        let far = BodyCollider::Obb {
            center: Vec3::new(10.0, 0.0, 0.0),
            orientation: Quat::IDENTITY,
            half_extents: Vec3::splat(0.5),
        };
        assert!(collider_overlaps(near, &soft));
        assert!(!collider_overlaps(far, &soft));
    }

    #[test]
    fn sphere_culled_when_far() {
        let soft = Aabb {
            min: Vec3::splat(-1.0),
            max: Vec3::splat(1.0),
        };
        let near = BodyCollider::Sphere {
            center: Vec3::new(1.5, 0.0, 0.0),
            radius: 1.0,
        };
        let far = BodyCollider::Sphere {
            center: Vec3::new(10.0, 0.0, 0.0),
            radius: 1.0,
        };
        assert!(collider_overlaps(near, &soft));
        assert!(!collider_overlaps(far, &soft));
    }

    // --- Convex-hull proxy coupling goldens -------------------------------
    //
    // These exercise the `BodyCollider::ConvexHull` arm through the shared
    // per-particle coupling kernel (`couple_particle_against_body`), the same
    // entry point the linear/angular/friction drivers all funnel through, so a
    // passing cross-check here means those three drivers auto-work for convex
    // props without any driver change.
    use crate::soft::couple_particle_against_body;

    /// An oriented box and its convex-from-box proxy at the same pose must
    /// produce bit-identical coupling contributions for interior particles, so
    /// a convex box couples exactly like the existing OBB arm.
    #[test]
    fn convex_hull_couples_identically_to_obb() {
        let center = Vec3::new(0.4, 1.0, -0.7);
        let orientation =
            Quat::from_rotation_y(0.7) * Quat::from_rotation_x(0.3) * Quat::from_rotation_z(-0.2);
        let he = Vec3::new(0.8, 0.5, 1.2);
        let obb = BodyCollider::Obb {
            center,
            orientation,
            half_extents: he,
        };
        let hull = BodyCollider::ConvexHull(ConvexProxy::from_box(center, orientation, he));
        let samples = [
            center + orientation * Vec3::new(0.1, 0.3, -0.4),
            center + orientation * Vec3::new(-0.5, 0.1, 0.6),
            center + orientation * Vec3::new(0.2, -0.35, 0.1),
        ];
        for pos in samples {
            let a = couple_particle_against_body(pos, 1.0, obb, 0.5, 1.0 / 60.0);
            let b = couple_particle_against_body(pos, 1.0, hull, 0.5, 1.0 / 60.0);
            // The two arms share the same geometry but reach it via different
            // arithmetic (box-local clamp vs. plane dot products), so they agree
            // to float tolerance rather than bit-for-bit on a rotated box.
            assert!(
                (a.particle_delta - b.particle_delta).length() < 1e-5,
                "particle delta mismatch: {a:?} vs {b:?}"
            );
            assert!((a.body_delta - b.body_delta).length() < 1e-5);
            assert!((a.impulse - b.impulse).length() < 1e-3);
            // And the interior particle actually moved (a real contact).
            assert!(b.particle_delta.length() > 1e-6);
        }
    }

    /// The opt-in `convex_proxy_from_cuboid` builder must reproduce the default
    /// `Cuboid -> Obb` mapping's projection, the cross-validation entry point.
    #[test]
    fn convex_proxy_from_cuboid_matches_obb_mapping() {
        let shape = ColliderShape::Cuboid {
            half_extents: Vec3::new(0.6, 0.3, 0.9),
        };
        let position = Vec3::new(-1.0, 2.0, 0.5);
        let orientation = Quat::from_rotation_z(std::f32::consts::FRAC_PI_3);
        let obb = body_collider_from_shape(&shape, position, orientation).expect("obb proxy");
        let hull = BodyCollider::ConvexHull(convex_proxy_from_cuboid(
            Vec3::new(0.6, 0.3, 0.9),
            position,
            orientation,
        ));
        let pos = position + orientation * Vec3::new(0.1, 0.2, -0.3);
        // Same surface via different arithmetic => float-tolerance equal.
        assert!((obb.project(pos) - hull.project(pos)).length() < 1e-5);
    }

    /// An interior particle under a tilted face is pushed out along that face's
    /// world normal (so the follow-up solve lets it slide along the face).
    #[test]
    fn convex_hull_push_is_along_tilted_face_normal() {
        let orientation = Quat::from_rotation_z(std::f32::consts::FRAC_PI_6);
        let hull = BodyCollider::ConvexHull(ConvexProxy::from_box(
            Vec3::ZERO,
            orientation,
            Vec3::new(1.0, 0.4, 1.0),
        ));
        // Just inside the +Y face in box-local coordinates.
        let pos = orientation * Vec3::new(0.1, 0.35, -0.2);
        let contrib = couple_particle_against_body(pos, 1.0, hull, 0.0, 1.0 / 60.0);
        // Body is infinite-mass (w_body 0) so the particle takes the full push.
        let correction = contrib.particle_delta;
        assert!(correction.length() > 1e-4, "expected a push");
        let expected_normal = orientation * Vec3::Y;
        let along = correction.normalize_or_zero().dot(expected_normal);
        assert!(
            (along - 1.0).abs() < 1e-5,
            "push not along face normal: {along}"
        );
    }

    /// A particle outside the convex solid produces no coupling (zero
    /// contribution), the back-face / outside miss case.
    #[test]
    fn convex_hull_outside_particle_is_no_op() {
        let hull = BodyCollider::ConvexHull(ConvexProxy::from_box(
            Vec3::ZERO,
            Quat::IDENTITY,
            Vec3::new(1.0, 0.5, 2.0),
        ));
        let outside = Vec3::new(5.0, 0.0, 0.0);
        let contrib = couple_particle_against_body(outside, 1.0, hull, 1.0, 1.0 / 60.0);
        assert_eq!(contrib, crate::soft::CouplingContribution::ZERO);
    }

    /// A degenerate (empty) convex proxy is inert: no projection, no coupling.
    #[test]
    fn convex_hull_empty_proxy_is_inert() {
        let hull = BodyCollider::ConvexHull(ConvexProxy::EMPTY);
        let pos = Vec3::new(0.1, 0.2, 0.3);
        assert_eq!(hull.project(pos), pos);
        let contrib = couple_particle_against_body(pos, 1.0, hull, 1.0, 1.0 / 60.0);
        assert_eq!(contrib, crate::soft::CouplingContribution::ZERO);
    }

    /// Identical inputs produce identical contributions on repeat (determinism).
    #[test]
    fn convex_hull_coupling_is_deterministic() {
        let hull = BodyCollider::ConvexHull(ConvexProxy::from_box(
            Vec3::new(0.0, 1.0, 0.0),
            Quat::from_rotation_x(0.4),
            Vec3::new(0.7, 0.5, 0.9),
        ));
        let pos = Vec3::new(0.05, 1.1, 0.1);
        let a = couple_particle_against_body(pos, 1.0, hull, 0.5, 1.0 / 60.0);
        let b = couple_particle_against_body(pos, 1.0, hull, 0.5, 1.0 / 60.0);
        assert_eq!(a, b);
    }

    /// A registry-backed [`ColliderShape::ConvexHull`] resolves to a bounded
    /// [`BodyCollider::ConvexHull`] through [`body_collider_from_shape_in`]:
    /// a box hull yields a six-face proxy centered at the body position.
    #[test]
    fn convex_hull_shape_maps_to_bounded_proxy() {
        use crate::collider::{ConvexMeshData, ShapeRegistry};

        let mut reg = ShapeRegistry::new();
        let half = Vec3::new(0.5, 0.75, 1.25);
        let mesh = reg.insert_convex_mesh(ConvexMeshData::from_box(half));
        let shape = reg
            .convex_hull_shape(mesh)
            .expect("convex hull shape from box");

        let position = Vec3::new(2.0, -1.0, 0.5);
        let got = body_collider_from_shape_in(&shape, position, Quat::IDENTITY, &reg)
            .expect("convex hull proxy");
        let BodyCollider::ConvexHull(proxy) = got else {
            panic!("expected a convex-hull proxy");
        };
        // Center is the body position and a box has exactly six faces.
        assert!((proxy.center() - position).length() < 1e-6);
        assert!(proxy.planes().len() == 6);
    }

    /// The world transform of a hull's face planes tracks the body pose: an
    /// off-origin, rotated box hull resolves to the same projection surface as
    /// the equivalent [`BodyCollider::Obb`] mapping (cross-validation).
    #[test]
    fn convex_hull_shape_tracks_world_pose() {
        use crate::collider::{ConvexMeshData, ShapeRegistry};

        let mut reg = ShapeRegistry::new();
        let half = Vec3::new(0.6, 0.3, 0.9);
        let mesh = reg.insert_convex_mesh(ConvexMeshData::from_box(half));
        let shape = reg.convex_hull_shape(mesh).expect("convex hull shape");

        let position = Vec3::new(-1.0, 2.0, 0.5);
        let orientation = Quat::from_rotation_z(std::f32::consts::FRAC_PI_3);
        let hull = body_collider_from_shape_in(&shape, position, orientation, &reg)
            .expect("convex hull proxy");
        let obb = BodyCollider::Obb {
            center: position,
            orientation,
            half_extents: half,
        };
        // An interior particle is pushed to the same surface by both arms.
        let probe = position + orientation * Vec3::new(0.1, 0.2, -0.3);
        assert!((hull.project(probe) - obb.project(probe)).length() < 1e-5);
    }

    /// A [`ColliderShape::TriangleMesh`] has no analytic bounded proxy, so the
    /// registry-aware mapping returns [`None`] (static scene geometry contract).
    #[test]
    fn triangle_mesh_shape_has_no_proxy() {
        use crate::collider::{ShapeRegistry, TriMeshData};

        let mut reg = ShapeRegistry::new();
        // A unit cube as a triangle mesh (8 vertices, 12 triangles).
        let verts = vec![
            Vec3::new(-0.5, -0.5, -0.5),
            Vec3::new(0.5, -0.5, -0.5),
            Vec3::new(0.5, 0.5, -0.5),
            Vec3::new(-0.5, 0.5, -0.5),
            Vec3::new(-0.5, -0.5, 0.5),
            Vec3::new(0.5, -0.5, 0.5),
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(-0.5, 0.5, 0.5),
        ];
        let tris = vec![
            [0u32, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ];
        let mesh = reg.insert_tri_mesh(TriMeshData::new(verts, tris));
        let shape = reg.tri_mesh_shape(mesh).expect("tri mesh shape");
        let got =
            body_collider_from_shape_in(&shape, Vec3::new(1.0, 2.0, 3.0), Quat::IDENTITY, &reg);
        assert!(got.is_none());
    }

    /// A convex-hull handle that is not present in the registry yields [`None`]
    /// rather than panicking (the `?` on the arena lookup).
    #[test]
    fn convex_hull_missing_handle_is_none() {
        use crate::collider::{ConvexMeshHandle, ShapeRegistry};

        let reg = ShapeRegistry::new();
        let shape = ColliderShape::ConvexHull {
            mesh: ConvexMeshHandle(7),
            local_aabb_min: Vec3::splat(-0.5),
            local_aabb_max: Vec3::splat(0.5),
            volume: 1.0,
            unit_density_inertia: Vec3::splat(1.0 / 6.0),
        };
        let got = body_collider_from_shape_in(&shape, Vec3::ZERO, Quat::IDENTITY, &reg);
        assert!(got.is_none());
    }

    /// The convex anchor is its center and its world AABB encloses that center.
    #[test]
    fn convex_hull_anchor_and_aabb() {
        let center = Vec3::new(1.0, -2.0, 3.0);
        let proxy = ConvexProxy::from_box(center, Quat::IDENTITY, Vec3::new(0.5, 0.5, 0.5));
        let hull = BodyCollider::ConvexHull(proxy);
        assert_eq!(collider_anchor(hull), center);
        let aabb = collider_world_aabb(hull).expect("convex hull is bounded");
        assert!(aabb.min.x <= center.x && aabb.max.x >= center.x);
        assert!(aabb.min.y <= center.y && aabb.max.y >= center.y);
        assert!(aabb.min.z <= center.z && aabb.max.z >= center.z);
    }
}
