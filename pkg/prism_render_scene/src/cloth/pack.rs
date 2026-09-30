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
use prism_render_architecture::cloth::gpu::upload::{BendingUploadPlan, ConstraintUploadPlan};
use prism_render_architecture::cloth::{Constraint, ConstraintKind};

use super::abi::{
    GpuClothBendingConstraint, GpuClothConstraint, CLOTH_CONSTRAINT_BEND, CLOTH_CONSTRAINT_LRA,
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

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_architecture::cloth::gpu::upload::{color_bending, plan_constraint_upload};
    use prism_render_architecture::cloth::{Compliance, ConstraintKind};

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
}
