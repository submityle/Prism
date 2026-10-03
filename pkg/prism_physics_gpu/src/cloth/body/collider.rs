//! Host-side packing of authored [`BodyCollider`]s and [`Backstop`]s into the
//! fixed-layout records the `GPU` body-collision kernel reads.
//!
//! The solver authors body proxies as a small Rust enum ([`BodyCollider`]:
//! sphere, capsule, half-space) and per-particle [`Backstop`] planes. A compute
//! shader cannot read a Rust tagged union, so this module flattens each into a
//! `#[repr(C)]` `Pod` record with an explicit `kind` discriminant and padded
//! `vec4` slots, exactly mirroring the `WGSL` `Collider` / `Backstop` structs.
//! Keeping the packing here (rather than inline in the kernel wrapper) makes the
//! field mapping a single, independently testable unit.
//!
//! # Provenance
//!
//! Plain struct-of-arrays flattening of standard analytic collision primitives.
//! No Unreal Engine source or derived code.

use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use prism_physics_core::{Backstop, BodyCollider};

/// Discriminant for a packed sphere collider.
pub const COLLIDER_SPHERE: u32 = 0;
/// Discriminant for a packed capsule collider.
pub const COLLIDER_CAPSULE: u32 = 1;
/// Discriminant for a packed half-space collider.
pub const COLLIDER_HALF_SPACE: u32 = 2;
/// Discriminant for a packed oriented-box (`OBB`) collider.
pub const COLLIDER_OBB: u32 = 3;
/// Discriminant for a packed convex-hull collider.
///
/// Unlike the analytic primitives the hull's face planes do not fit in the
/// fixed 64-byte record, so a convex record stores only its anchor and
/// bounding radius inline and points at a run of [`GpuConvexPlane`]s in the
/// shared plane buffer via [`GpuBodyCollider::plane_offset`] /
/// [`GpuBodyCollider::plane_count`].
pub const COLLIDER_CONVEX: u32 = 4;

/// A body collider flattened for the `GPU`, mirroring the `WGSL` `Collider`
/// struct (64 bytes, 16-byte aligned).
///
/// The `kind` discriminant selects how the slots are read:
/// - [`COLLIDER_SPHERE`]: `p0.xyz` = center, `radius` = radius, `p1`/`p2`
///   unused.
/// - [`COLLIDER_CAPSULE`]: `p0.xyz`/`p1.xyz` = segment endpoints, `radius` =
///   inflation radius, `p2` unused.
/// - [`COLLIDER_HALF_SPACE`]: `p0.xyz` = plane normal, `radius` = plane offset,
///   `p1`/`p2` unused.
/// - [`COLLIDER_OBB`]: `p0.xyz` = center, `p1.xyz` = half-extents, `p2.xyzw` =
///   orientation quaternion `(x, y, z, w)`, `radius` unused.
/// - [`COLLIDER_CONVEX`]: `p0.xyz` = anchor center, `radius` = bounding
///   radius, [`plane_offset`](Self::plane_offset) / [`plane_count`](Self::plane_count)
///   index the shared [`GpuConvexPlane`] buffer; `p1`/`p2` unused.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuBodyCollider {
    /// Which primitive this record encodes (see the `COLLIDER_*` constants).
    pub kind: u32,
    /// Sphere/capsule radius, half-space offset, or (for
    /// [`COLLIDER_CONVEX`]) the hull's conservative bounding radius.
    pub radius: f32,
    /// Index of this collider's first face plane in the shared
    /// [`GpuConvexPlane`] buffer; `0` and unused for the analytic
    /// primitives, meaningful only for [`COLLIDER_CONVEX`].
    pub plane_offset: u32,
    /// Number of live face planes for a [`COLLIDER_CONVEX`] record; `0`
    /// (unused) for the analytic primitives.
    pub plane_count: u32,
    /// Sphere center / capsule endpoint 0 / half-space normal / box center
    /// (`w` unused).
    pub p0: [f32; 4],
    /// Capsule endpoint 1 / box half-extents (`w` unused; unused for
    /// sphere/half-space).
    pub p1: [f32; 4],
    /// Box orientation quaternion `(x, y, z, w)`; unused (zeroed) for the other
    /// primitives.
    pub p2: [f32; 4],
}

const _: () = assert!(size_of::<GpuBodyCollider>() == 64);

impl GpuBodyCollider {
    /// Flattens one authored [`BodyCollider`] into its `GPU` record.
    #[must_use]
    pub fn from_collider(collider: BodyCollider) -> GpuBodyCollider {
        match collider {
            BodyCollider::Sphere { center, radius } => GpuBodyCollider {
                kind: COLLIDER_SPHERE,
                radius,
                plane_offset: 0,
                plane_count: 0,
                p0: [center.x, center.y, center.z, 0.0],
                p1: [0.0; 4],
                p2: [0.0; 4],
            },
            BodyCollider::Capsule { p0, p1, radius } => GpuBodyCollider {
                kind: COLLIDER_CAPSULE,
                radius,
                plane_offset: 0,
                plane_count: 0,
                p0: [p0.x, p0.y, p0.z, 0.0],
                p1: [p1.x, p1.y, p1.z, 0.0],
                p2: [0.0; 4],
            },
            BodyCollider::HalfSpace { normal, offset } => GpuBodyCollider {
                kind: COLLIDER_HALF_SPACE,
                radius: offset,
                plane_offset: 0,
                plane_count: 0,
                p0: [normal.x, normal.y, normal.z, 0.0],
                p1: [0.0; 4],
                p2: [0.0; 4],
            },
            BodyCollider::Obb {
                center,
                orientation,
                half_extents,
            } => GpuBodyCollider {
                kind: COLLIDER_OBB,
                radius: 0.0,
                plane_offset: 0,
                plane_count: 0,
                p0: [center.x, center.y, center.z, 0.0],
                p1: [half_extents.x, half_extents.y, half_extents.z, 0.0],
                p2: [orientation.x, orientation.y, orientation.z, orientation.w],
            },
            BodyCollider::ConvexHull(proxy) => {
                let center = proxy.center();
                GpuBodyCollider {
                    kind: COLLIDER_CONVEX,
                    radius: proxy.bounding_radius(),
                    // `plane_offset` is a placeholder here: a standalone record
                    // assumes its planes start at the head of a dedicated buffer.
                    // `pack_body_scene` overwrites it with the real slice offset
                    // when several colliders share one plane buffer.
                    plane_offset: 0,
                    plane_count: u32::try_from(proxy.planes().len()).unwrap_or(u32::MAX),
                    p0: [center.x, center.y, center.z, 0.0],
                    p1: [0.0; 4],
                    p2: [0.0; 4],
                }
            }
        }
    }
}

/// One convex-hull face plane flattened for the `GPU`, mirroring the `WGSL`
/// `ConvexPlane` struct (16 bytes, one `vec4`).
///
/// `plane.xyz` is the outward unit face normal and `plane.w` the signed plane
/// offset: the solid lies on the `normal \u{b7} x <= offset` side, exactly as in
/// [`prism_physics_core::ConvexProxy`].
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuConvexPlane {
    /// Outward unit normal (`xyz`) and signed plane offset (`w`).
    pub plane: [f32; 4],
}

const _: () = assert!(size_of::<GpuConvexPlane>() == 16);

impl GpuConvexPlane {
    /// Flattens one `(normal, offset)` face into its `GPU` record.
    #[must_use]
    fn new(normal: glam::Vec3, offset: f32) -> GpuConvexPlane {
        GpuConvexPlane {
            plane: [normal.x, normal.y, normal.z, offset],
        }
    }
}

/// A body-collider slice flattened for the `GPU`: the fixed-size collider
/// records plus the variable-length convex face-plane pool the convex records
/// index into.
///
/// The two vectors are uploaded to separate storage buffers; a convex record's
/// [`GpuBodyCollider::plane_offset`] / [`GpuBodyCollider::plane_count`] address
/// its contiguous run inside [`planes`](Self::planes).
#[derive(Clone, Debug, Default)]
pub struct PackedBodyColliders {
    /// One fixed-layout record per authored collider, in slice order.
    pub records: Vec<GpuBodyCollider>,
    /// The concatenated convex face planes, in collider slice order.
    pub planes: Vec<GpuConvexPlane>,
}

/// Packs an authored collider slice into the fixed-size collider records plus
/// the shared convex face-plane pool, preserving slice order.
///
/// Analytic primitives contribute no planes; each [`BodyCollider::ConvexHull`]
/// appends its live faces to the pool and records their start offset and count
/// so the kernel can walk exactly that run.
#[must_use]
pub fn pack_body_scene(colliders: &[BodyCollider]) -> PackedBodyColliders {
    let mut records = Vec::with_capacity(colliders.len());
    let mut planes = Vec::new();
    for collider in colliders.iter().copied() {
        let mut record = GpuBodyCollider::from_collider(collider);
        if let BodyCollider::ConvexHull(proxy) = collider {
            let offset = u32::try_from(planes.len()).unwrap_or(u32::MAX);
            record.plane_offset = offset;
            for face in proxy.planes() {
                planes.push(GpuConvexPlane::new(face.normal, face.offset));
            }
        }
        records.push(record);
    }
    PackedBodyColliders { records, planes }
}

/// Packs an authored collider slice into the `GPU` record array in order.
///
/// This is [`pack_body_scene`] projected onto just the fixed-size records; use
/// [`pack_body_scene`] directly when the convex face-plane pool is also needed.
#[must_use]
pub fn pack_body_colliders(colliders: &[BodyCollider]) -> Vec<GpuBodyCollider> {
    pack_body_scene(colliders).records
}

/// Packs the convex face-plane pool for an authored collider slice, with the
/// same ordering and per-collider offsets as [`pack_body_scene`].
#[must_use]
pub fn pack_convex_planes(colliders: &[BodyCollider]) -> Vec<GpuConvexPlane> {
    pack_body_scene(colliders).planes
}

/// A per-particle backstop plane flattened for the `GPU`, mirroring the `WGSL`
/// `Backstop` struct (32 bytes, 16-byte aligned).
///
/// `origin.xyz` is the anchor point and `origin.w` the maximum behind-plane
/// travel distance; `normal.xyz` is the (not necessarily unit) outward normal.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuBackstop {
    /// Anchor point (`xyz`) and max behind-plane distance (`w`).
    pub origin: [f32; 4],
    /// Outward plane normal (`w` unused).
    pub normal: [f32; 4],
}

impl GpuBackstop {
    /// Flattens one authored [`Backstop`] into its `GPU` record.
    #[must_use]
    pub fn from_backstop(backstop: Backstop) -> GpuBackstop {
        GpuBackstop {
            origin: [
                backstop.origin.x,
                backstop.origin.y,
                backstop.origin.z,
                backstop.distance,
            ],
            normal: [backstop.normal.x, backstop.normal.y, backstop.normal.z, 0.0],
        }
    }
}

/// Packs an authored backstop slice into the `GPU` record array in order.
#[must_use]
pub fn pack_backstops(backstops: &[Backstop]) -> Vec<GpuBackstop> {
    backstops
        .iter()
        .copied()
        .map(GpuBackstop::from_backstop)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;
    use prism_physics_core::ConvexProxy;

    #[test]
    fn sphere_packs_center_and_radius() {
        let c = GpuBodyCollider::from_collider(BodyCollider::Sphere {
            center: Vec3::new(1.0, 2.0, 3.0),
            radius: 0.5,
        });
        assert_eq!(c.kind, COLLIDER_SPHERE);
        assert_eq!(c.radius, 0.5);
        assert_eq!(c.p0, [1.0, 2.0, 3.0, 0.0]);
    }

    #[test]
    fn capsule_packs_both_endpoints() {
        let c = GpuBodyCollider::from_collider(BodyCollider::Capsule {
            p0: Vec3::new(-1.0, 0.0, 0.0),
            p1: Vec3::new(1.0, 0.0, 0.0),
            radius: 0.25,
        });
        assert_eq!(c.kind, COLLIDER_CAPSULE);
        assert_eq!(c.radius, 0.25);
        assert_eq!(c.p0, [-1.0, 0.0, 0.0, 0.0]);
        assert_eq!(c.p1, [1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn half_space_packs_normal_and_offset() {
        let c = GpuBodyCollider::from_collider(BodyCollider::HalfSpace {
            normal: Vec3::new(0.0, 1.0, 0.0),
            offset: -2.0,
        });
        assert_eq!(c.kind, COLLIDER_HALF_SPACE);
        assert_eq!(c.radius, -2.0);
        assert_eq!(c.p0, [0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn obb_packs_center_half_extents_and_orientation() {
        use glam::Quat;
        let orientation = Quat::from_rotation_z(core::f32::consts::FRAC_PI_2);
        let c = GpuBodyCollider::from_collider(BodyCollider::Obb {
            center: Vec3::new(1.0, 2.0, 3.0),
            orientation,
            half_extents: Vec3::new(0.5, 0.25, 0.75),
        });
        assert_eq!(c.kind, COLLIDER_OBB);
        assert_eq!(c.radius, 0.0);
        assert_eq!(c.p0, [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(c.p1, [0.5, 0.25, 0.75, 0.0]);
        assert_eq!(
            c.p2,
            [orientation.x, orientation.y, orientation.z, orientation.w]
        );
    }

    #[test]
    fn backstop_packs_distance_into_origin_w() {
        let b = GpuBackstop::from_backstop(Backstop {
            origin: Vec3::new(1.0, 0.0, 0.0),
            normal: Vec3::new(0.0, 0.0, 1.0),
            distance: 0.1,
        });
        assert_eq!(b.origin, [1.0, 0.0, 0.0, 0.1]);
        assert_eq!(b.normal, [0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn pack_preserves_order_and_count() {
        let colliders = [
            BodyCollider::Sphere {
                center: Vec3::ZERO,
                radius: 1.0,
            },
            BodyCollider::HalfSpace {
                normal: Vec3::Y,
                offset: 0.0,
            },
        ];
        let packed = pack_body_colliders(&colliders);
        assert_eq!(packed.len(), 2);
        assert_eq!(packed[0].kind, COLLIDER_SPHERE);
        assert_eq!(packed[1].kind, COLLIDER_HALF_SPACE);
    }

    #[test]
    fn convex_record_carries_anchor_radius_and_plane_run() {
        use glam::Quat;
        let proxy = ConvexProxy::from_box(
            Vec3::new(1.0, 2.0, 3.0),
            Quat::IDENTITY,
            Vec3::new(0.5, 0.25, 0.75),
        );
        let record = GpuBodyCollider::from_collider(BodyCollider::ConvexHull(proxy));
        assert_eq!(record.kind, COLLIDER_CONVEX);
        assert_eq!(record.p0, [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(record.radius, proxy.bounding_radius());
        // A fully-extended box has six faces.
        assert_eq!(record.plane_count, 6);
        // A standalone record places its planes at the head of its own buffer.
        assert_eq!(record.plane_offset, 0);
    }

    #[test]
    fn scene_packs_convex_planes_with_matching_offsets() {
        use glam::Quat;
        let box_a = ConvexProxy::from_box(Vec3::ZERO, Quat::IDENTITY, Vec3::splat(1.0));
        let box_b = ConvexProxy::from_box(Vec3::X, Quat::IDENTITY, Vec3::new(2.0, 0.0, 2.0));
        let colliders = [
            BodyCollider::Sphere {
                center: Vec3::ZERO,
                radius: 1.0,
            },
            BodyCollider::ConvexHull(box_a),
            BodyCollider::ConvexHull(box_b),
        ];
        let scene = pack_body_scene(&colliders);
        assert_eq!(scene.records.len(), 3);
        // The sphere contributes no planes and leaves its indices zeroed.
        assert_eq!(scene.records[0].plane_count, 0);
        // box_a (an axis-collapsed-free unit box) has six faces starting at 0.
        let a_count = box_a.planes().len() as u32;
        assert_eq!(scene.records[1].plane_offset, 0);
        assert_eq!(scene.records[1].plane_count, a_count);
        // box_b (y-extent zero => four faces) starts right after box_a's run.
        let b_count = box_b.planes().len() as u32;
        assert_eq!(scene.records[2].plane_offset, a_count);
        assert_eq!(scene.records[2].plane_count, b_count);
        assert_eq!(scene.planes.len() as u32, a_count + b_count);
        // The first face of box_a round-trips its (normal, offset).
        let first = box_a.planes()[0];
        assert_eq!(
            scene.planes[0].plane,
            [first.normal.x, first.normal.y, first.normal.z, first.offset]
        );
    }
}
