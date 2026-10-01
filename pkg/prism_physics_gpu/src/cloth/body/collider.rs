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

/// A body collider flattened for the `GPU`, mirroring the `WGSL` `Collider`
/// struct (48 bytes, 16-byte aligned).
///
/// The `kind` discriminant selects how the slots are read:
/// - [`COLLIDER_SPHERE`]: `p0.xyz` = center, `radius` = radius, `p1` unused.
/// - [`COLLIDER_CAPSULE`]: `p0.xyz`/`p1.xyz` = segment endpoints, `radius` =
///   inflation radius.
/// - [`COLLIDER_HALF_SPACE`]: `p0.xyz` = plane normal, `radius` = plane offset,
///   `p1` unused.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuBodyCollider {
    /// Which primitive this record encodes (see the `COLLIDER_*` constants).
    pub kind: u32,
    /// Sphere/capsule radius, or half-space offset.
    pub radius: f32,
    /// Padding to a 16-byte word.
    pub _pad0: u32,
    /// Padding to a 16-byte word.
    pub _pad1: u32,
    /// Sphere center / capsule endpoint 0 / half-space normal (`w` unused).
    pub p0: [f32; 4],
    /// Capsule endpoint 1 (`w` unused; unused for sphere/half-space).
    pub p1: [f32; 4],
}

impl GpuBodyCollider {
    /// Flattens one authored [`BodyCollider`] into its `GPU` record.
    #[must_use]
    pub fn from_collider(collider: BodyCollider) -> GpuBodyCollider {
        match collider {
            BodyCollider::Sphere { center, radius } => GpuBodyCollider {
                kind: COLLIDER_SPHERE,
                radius,
                _pad0: 0,
                _pad1: 0,
                p0: [center.x, center.y, center.z, 0.0],
                p1: [0.0; 4],
            },
            BodyCollider::Capsule { p0, p1, radius } => GpuBodyCollider {
                kind: COLLIDER_CAPSULE,
                radius,
                _pad0: 0,
                _pad1: 0,
                p0: [p0.x, p0.y, p0.z, 0.0],
                p1: [p1.x, p1.y, p1.z, 0.0],
            },
            BodyCollider::HalfSpace { normal, offset } => GpuBodyCollider {
                kind: COLLIDER_HALF_SPACE,
                radius: offset,
                _pad0: 0,
                _pad1: 0,
                p0: [normal.x, normal.y, normal.z, 0.0],
                p1: [0.0; 4],
            },
        }
    }
}

/// Packs an authored collider slice into the `GPU` record array in order.
#[must_use]
pub fn pack_body_colliders(colliders: &[BodyCollider]) -> Vec<GpuBodyCollider> {
    colliders
        .iter()
        .copied()
        .map(GpuBodyCollider::from_collider)
        .collect()
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
}
