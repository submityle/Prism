//! Host-side packing from the architecture-layer solver types into the
//! `#[repr(C)]` `GPU` records the resident cloth buffers upload.
//!
//! The architecture crate (`prism_render_architecture::cloth`) owns the
//! `CPU`-golden solver types ([`Constraint`], [`BendingConstraint`]) and the
//! class-separated upload planning
//! ([`plan_constraint_upload`](prism_render_architecture::cloth::gpu::upload::plan_constraint_upload),
//! [`color_bending`](prism_render_architecture::cloth::gpu::upload::color_bending)).
//! This module is the thin, `GPU`-free bridge that turns those planned,
//! by-color reorderings into the byte-compatible mirrors in [`super::abi`] that
//! [`super::bind_groups::ClothPieceUpload`] uploads.
//!
//! It is deliberately pure (slice-in / `Vec`-out, no device handles) so the
//! packing is deterministic and unit-testable without a `GPU`. The contract it
//! preserves is the shared-buffer layout the dispatch planner addresses: the
//! distance section is packed first (buffer base `0`), then the long-range
//! section (buffer base = distance count), so
//! [`prism_render_architecture::cloth::gpu::pipeline::prepare`] can offset a
//! long-range color's slice past the whole distance section. Bending hinges
//! pack into their own buffer.

use prism_render_architecture::cloth::bending::BendingConstraint;
use prism_render_architecture::cloth::collision::BodyCollider;
use prism_render_architecture::cloth::gpu::upload::{BendingUploadPlan, ConstraintUploadPlan};
use prism_render_architecture::cloth::{Constraint, ConstraintKind};

use super::abi::{
    GpuClothBendingConstraint, GpuClothCollider, GpuClothConstraint, CLOTH_COLLIDER_CAPSULE,
    CLOTH_COLLIDER_HALF_SPACE, CLOTH_COLLIDER_SPHERE, CLOTH_CONSTRAINT_BEND, CLOTH_CONSTRAINT_LRA,
    CLOTH_CONSTRAINT_SHEAR, CLOTH_CONSTRAINT_STRETCH, CLOTH_CONSTRAINT_TETHER,
};

/// Maps an architecture-layer [`ConstraintKind`] to its stable `GPU` tag
/// ([`CLOTH_CONSTRAINT_*`](super::abi)), mirroring the `cloth_sim.wesl`
/// constants. The encoding follows the enum's declaration order and must stay
/// in lockstep with the shader-side constants: the strain limiter reads it to
/// clamp only structural (stretch) edges.
#[must_use]
fn gpu_constraint_kind(kind: ConstraintKind) -> u32 {
    match kind {
        ConstraintKind::Stretch => CLOTH_CONSTRAINT_STRETCH,
        ConstraintKind::Bend => CLOTH_CONSTRAINT_BEND,
        ConstraintKind::Shear => CLOTH_CONSTRAINT_SHEAR,
        ConstraintKind::Lra => CLOTH_CONSTRAINT_LRA,
        ConstraintKind::Tether => CLOTH_CONSTRAINT_TETHER,
    }
}

/// Packs one architecture-layer [`Constraint`] into its byte-compatible
/// [`GpuClothConstraint`] mirror.
///
/// The record carries the two endpoints, the rest length, the compliance and a
/// [`ConstraintKind`] tag. The projection kernels treat every two-sided
/// distance edge alike and the plan routes one-sided long-range edges into
/// their own buffer section, so the kind is *not* needed for projection — but
/// the strain limiter must clamp only structural (stretch) edges to mirror the
/// CPU golden `apply_strain_limit`, so the kind travels with each record. The
/// compliance is read through
/// [`Compliance::value`](prism_render_architecture::cloth::Compliance::value),
/// clamping any negative authored value to the rigid `0.0` the shader expects.
#[must_use]
pub(crate) fn pack_constraint(constraint: &Constraint) -> GpuClothConstraint {
    GpuClothConstraint {
        a: constraint.a,
        b: constraint.b,
        rest_length: constraint.rest_length,
        compliance: constraint.compliance.value(),
        kind: gpu_constraint_kind(constraint.kind),
    }
}

/// Packs a [`ConstraintUploadPlan`] into the single contiguous
/// distance-then-long-range constraint buffer content.
///
/// The distance section (buffer base `0`) is emitted first in its by-color
/// order, immediately followed by the long-range section (buffer base =
/// [`ConstraintUploadPlan::long_range_base`]) in its own by-color order. The
/// returned length is exactly [`ConstraintUploadPlan::total`], and the split
/// point is the distance count, matching the dispatch planner's `class_base`.
#[must_use]
pub(crate) fn pack_constraints(plan: &ConstraintUploadPlan) -> Vec<GpuClothConstraint> {
    let mut out: Vec<GpuClothConstraint> = Vec::with_capacity(plan.total());
    out.extend(plan.distance.iter().map(pack_constraint));
    out.extend(plan.long_range.iter().map(pack_constraint));
    out
}

/// Packs one architecture-layer [`BendingConstraint`] into its byte-compatible
/// [`GpuClothBendingConstraint`] mirror (four stencil indices, four weights, an
/// area scale and the compliance).
#[must_use]
pub(crate) fn pack_bending_constraint(hinge: &BendingConstraint) -> GpuClothBendingConstraint {
    GpuClothBendingConstraint {
        v0: hinge.vertices[0],
        v1: hinge.vertices[1],
        v2: hinge.vertices[2],
        v3: hinge.vertices[3],
        w0: hinge.weights[0],
        w1: hinge.weights[1],
        w2: hinge.weights[2],
        w3: hinge.weights[3],
        scale: hinge.scale,
        compliance: hinge.compliance.value(),
    }
}

/// Packs a [`BendingUploadPlan`] into its own contiguous bending buffer content
/// (buffer base `0`), preserving the plan's by-color ordering.
#[must_use]
pub(crate) fn pack_bending(plan: &BendingUploadPlan) -> Vec<GpuClothBendingConstraint> {
    plan.bending.iter().map(pack_bending_constraint).collect()
}

/// Packs one authored architecture-layer [`BodyCollider`] into its
/// byte-compatible [`GpuClothCollider`] mirror, the flat 32-byte `std430`
/// record `cloth_collision.wesl` reads.
///
/// This is the missing host bridge between the `CPU`-golden authoring type
/// (`BodyCollider`, the tagged union a garment fits to its skeleton) and the
/// device record the body-collision kernel projects against. The field mapping
/// mirrors the shader's tagged-union reading exactly, so the same collider
/// resolves identically on both paths:
///
/// - [`BodyCollider::Sphere`]: `kind = `[`CLOTH_COLLIDER_SPHERE`], `a` = centre,
///   `radius` = radius, `b` unused (left zero).
/// - [`BodyCollider::Capsule`]: `kind = `[`CLOTH_COLLIDER_CAPSULE`], `a` = `p0`,
///   `b` = `p1`, `radius` = inflation radius.
/// - [`BodyCollider::HalfSpace`]: `kind = `[`CLOTH_COLLIDER_HALF_SPACE`], `a` =
///   plane normal (need not be unit), `radius` = signed offset, `b` unused.
///
/// No value is clamped or normalised here: the projection kernels
/// (`cloth_project_out_of_sphere` / `cloth_project_out_of_half_space`) reproduce the CPU
/// degeneracy rules (non-positive radius / near-zero normal are inert) on read,
/// so the packed record stays a faithful, lossless copy of the authored proxy.
#[must_use]
pub(crate) fn pack_collider(collider: &BodyCollider) -> GpuClothCollider {
    match *collider {
        BodyCollider::Sphere { center, radius } => GpuClothCollider {
            kind: CLOTH_COLLIDER_SPHERE,
            ax: center.x,
            ay: center.y,
            az: center.z,
            bx: 0.0,
            by: 0.0,
            bz: 0.0,
            radius,
        },
        BodyCollider::Capsule { p0, p1, radius } => GpuClothCollider {
            kind: CLOTH_COLLIDER_CAPSULE,
            ax: p0.x,
            ay: p0.y,
            az: p0.z,
            bx: p1.x,
            by: p1.y,
            bz: p1.z,
            radius,
        },
        BodyCollider::HalfSpace { normal, offset } => GpuClothCollider {
            kind: CLOTH_COLLIDER_HALF_SPACE,
            ax: normal.x,
            ay: normal.y,
            az: normal.z,
            bx: 0.0,
            by: 0.0,
            bz: 0.0,
            radius: offset,
        },
    }
}

/// Packs an authored collider slice into the contiguous device buffer content
/// [`super::bind_groups::ClothPieceUpload::colliders`] uploads.
///
/// Order is preserved one-to-one: the body-collision kernel applies colliders
/// in array order per particle (the last to push wins), matching the CPU
/// [`resolve_body_collisions`](prism_render_architecture::cloth::collision::resolve_body_collisions)
/// slice-order projection, so the packed order *is* load-bearing and must not be
/// reshuffled. An empty input packs to an empty `Vec` (an honest no-op collider
/// set).
#[must_use]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "authored-BodyCollider slice -> GPU buffer host bridge; exercised now by the body-collision on-device parity test and wired into the garment spawn path once main-world authoring lands"
    )
)]
pub(crate) fn pack_colliders(colliders: &[BodyCollider]) -> Vec<GpuClothCollider> {
    colliders.iter().map(pack_collider).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_architecture::cloth::gpu::upload::{color_bending, plan_constraint_upload};
    use prism_render_architecture::cloth::{Compliance, ConstraintKind, Vec3};

    /// Builds a distance constraint for the packing tests.
    fn constraint(a: u32, b: u32, rest: f32, compliance: f32, kind: ConstraintKind) -> Constraint {
        Constraint::new(a, b, rest, Compliance(compliance), kind)
    }

    /// Builds a bending hinge for the packing tests.
    fn hinge(vertices: [u32; 4]) -> BendingConstraint {
        BendingConstraint {
            vertices,
            weights: [1.0, -2.0, 3.0, -4.0],
            scale: 0.25,
            compliance: Compliance(0.5),
        }
    }

    #[test]
    fn single_constraint_maps_every_field() {
        let c = constraint(3, 7, 1.5, 0.01, ConstraintKind::Stretch);
        let packed = pack_constraint(&c);
        assert_eq!(packed.a, 3);
        assert_eq!(packed.b, 7);
        assert!((packed.rest_length - 1.5).abs() <= f32::EPSILON);
        assert!((packed.compliance - 0.01).abs() <= f32::EPSILON);
    }

    #[test]
    fn negative_compliance_clamps_to_rigid() {
        let c = constraint(0, 1, 1.0, -5.0, ConstraintKind::Stretch);
        let packed = pack_constraint(&c);
        assert!(packed.compliance.abs() <= f32::EPSILON);
    }

    #[test]
    fn packed_constraints_are_distance_then_long_range() {
        let input = vec![
            constraint(0, 1, 1.0, 0.0, ConstraintKind::Stretch),
            constraint(2, 3, 1.0, 0.0, ConstraintKind::Lra),
            constraint(4, 5, 1.0, 0.0, ConstraintKind::Shear),
            constraint(6, 7, 1.0, 0.0, ConstraintKind::Tether),
        ];
        let plan = plan_constraint_upload(&input);
        let packed = pack_constraints(&plan);
        assert_eq!(packed.len(), plan.total());
        // The prefix must be exactly the distance section, in plan order.
        let split = plan.long_range_base() as usize;
        assert_eq!(split, plan.distance.len());
        for (i, c) in plan.distance.iter().enumerate() {
            assert_eq!(packed[i], pack_constraint(c));
        }
        for (i, c) in plan.long_range.iter().enumerate() {
            assert_eq!(packed[split + i], pack_constraint(c));
        }
    }

    #[test]
    fn empty_plan_packs_to_empty() {
        let plan = plan_constraint_upload(&[]);
        assert!(pack_constraints(&plan).is_empty());
        let bending = color_bending(&[]);
        assert!(pack_bending(&bending).is_empty());
    }

    #[test]
    fn single_bending_maps_every_field() {
        let h = hinge([2, 4, 6, 8]);
        let packed = pack_bending_constraint(&h);
        assert_eq!([packed.v0, packed.v1, packed.v2, packed.v3], [2, 4, 6, 8]);
        assert!((packed.w0 - 1.0).abs() <= f32::EPSILON);
        assert!((packed.w1 + 2.0).abs() <= f32::EPSILON);
        assert!((packed.w2 - 3.0).abs() <= f32::EPSILON);
        assert!((packed.w3 + 4.0).abs() <= f32::EPSILON);
        assert!((packed.scale - 0.25).abs() <= f32::EPSILON);
        assert!((packed.compliance - 0.5).abs() <= f32::EPSILON);
    }

    #[test]
    fn packed_bending_preserves_plan_order_and_length() {
        let input = vec![
            hinge([0, 1, 2, 3]),
            hinge([4, 5, 6, 7]),
            hinge([0, 1, 8, 9]),
        ];
        let plan = color_bending(&input);
        let packed = pack_bending(&plan);
        assert_eq!(packed.len(), plan.total());
        for (i, h) in plan.bending.iter().enumerate() {
            assert_eq!(packed[i], pack_bending_constraint(h));
        }
    }

    #[test]
    fn sphere_collider_maps_center_and_radius() {
        let packed = pack_collider(&BodyCollider::Sphere {
            center: Vec3::new(1.0, 2.0, 3.0),
            radius: 0.5,
        });
        assert_eq!(packed.kind, CLOTH_COLLIDER_SPHERE);
        assert_eq!([packed.ax, packed.ay, packed.az], [1.0, 2.0, 3.0]);
        assert!((packed.radius - 0.5).abs() <= f32::EPSILON);
        // The second point is unused for a sphere and must be left zeroed.
        assert_eq!([packed.bx, packed.by, packed.bz], [0.0, 0.0, 0.0]);
    }

    #[test]
    fn capsule_collider_maps_both_endpoints_and_radius() {
        let packed = pack_collider(&BodyCollider::Capsule {
            p0: Vec3::new(-1.0, 0.0, 0.0),
            p1: Vec3::new(1.0, 4.0, -2.0),
            radius: 0.25,
        });
        assert_eq!(packed.kind, CLOTH_COLLIDER_CAPSULE);
        assert_eq!([packed.ax, packed.ay, packed.az], [-1.0, 0.0, 0.0]);
        assert_eq!([packed.bx, packed.by, packed.bz], [1.0, 4.0, -2.0]);
        assert!((packed.radius - 0.25).abs() <= f32::EPSILON);
    }

    #[test]
    fn half_space_collider_maps_normal_into_a_and_offset_into_radius() {
        let packed = pack_collider(&BodyCollider::HalfSpace {
            normal: Vec3::new(0.0, 1.0, 0.0),
            offset: -0.75,
        });
        assert_eq!(packed.kind, CLOTH_COLLIDER_HALF_SPACE);
        assert_eq!([packed.ax, packed.ay, packed.az], [0.0, 1.0, 0.0]);
        // The offset lands in the shared radius/offset word.
        assert!((packed.radius + 0.75).abs() <= f32::EPSILON);
        assert_eq!([packed.bx, packed.by, packed.bz], [0.0, 0.0, 0.0]);
    }

    #[test]
    fn packed_colliders_preserve_author_order_and_length() {
        let input = vec![
            BodyCollider::Sphere {
                center: Vec3::new(0.0, 0.0, 0.0),
                radius: 1.0,
            },
            BodyCollider::HalfSpace {
                normal: Vec3::new(0.0, 1.0, 0.0),
                offset: 0.0,
            },
            BodyCollider::Capsule {
                p0: Vec3::new(0.0, 0.0, 0.0),
                p1: Vec3::new(0.0, 1.0, 0.0),
                radius: 0.3,
            },
        ];
        let packed = pack_colliders(&input);
        assert_eq!(packed.len(), input.len());
        for (i, c) in input.iter().enumerate() {
            assert_eq!(packed[i], pack_collider(c));
        }
    }

    #[test]
    fn empty_collider_set_packs_to_empty() {
        assert!(pack_colliders(&[]).is_empty());
    }
}
