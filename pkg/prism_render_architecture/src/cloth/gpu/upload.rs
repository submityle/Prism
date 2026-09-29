//! Class-separated constraint upload planning for the `GPU` cloth pipeline.
//!
//! The `CPU` graph coloring in [`super::super::constraints::color_constraints`]
//! colors an arbitrary constraint list as a single mixed set. The `GPU`
//! dispatch contract in [`super::pipeline`], however, drives three *separate*
//! color-serial projection kernels — distance, long-range and bending — each of
//! which needs its own per-color constraint count and its own contiguous,
//! by-color buffer layout ([`super::pipeline::ClothGpuExtract`] carries
//! `distance_colors`, `long_range_colors` and `bending_colors`).
//!
//! This module is the bridge that the extract stage was missing: it partitions
//! authored constraints into the distance and long-range classes, colors each
//! class independently, and repacks each class into the exact contiguous
//! by-color layout the dispatch planner addresses. Dihedral bending
//! constraints (four particles each) live in their own buffer and get a
//! dedicated four-vertex greedy coloring here as well.
//!
//! Everything is pure array-in / array-out integer bookkeeping so it is
//! deterministic and golden-testable on the `CPU` with no `GPU` handles. It
//! mirrors how `Chaos` Cloth and `NvCloth` batch independent constraints for a
//! parallel solve while keeping distance and attachment work in distinct
//! passes.

use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;

use super::super::bending::BendingConstraint;
use super::super::constraints::color_constraints;
use super::super::Constraint;

/// The distance and long-range constraint upload, colored per class and
/// repacked into the contiguous by-color layout the `GPU` dispatch planner
/// addresses.
///
/// The distance and long-range classes share one device `constraints` buffer.
/// The layout is `[all distance colors ..., all long-range colors ...]`: every
/// distance color first (buffer base `0`), then every long-range color (buffer
/// base [`ConstraintUploadPlan::long_range_base`], which equals the distance
/// count). This matches [`super::pipeline::prepare`], which offsets long-range
/// color slices past the whole distance section.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ConstraintUploadPlan {
    /// Distance-class constraints (`!kind.is_one_sided()`), reordered so each
    /// color batch is contiguous. Occupies the shared buffer from base `0`.
    pub distance: Vec<Constraint>,
    /// Per-color distance-constraint counts, in color order.
    pub distance_colors: Vec<u32>,
    /// Long-range-class constraints (`kind.is_one_sided()`: LRA and tether),
    /// reordered so each color batch is contiguous. Occupies the shared buffer
    /// starting at [`ConstraintUploadPlan::long_range_base`].
    pub long_range: Vec<Constraint>,
    /// Per-color long-range-constraint counts, in color order.
    pub long_range_colors: Vec<u32>,
}

impl ConstraintUploadPlan {
    /// The element offset of the long-range section within the shared
    /// `constraints` buffer, which is exactly the distance-constraint count.
    ///
    /// A long-range color addresses `[long_range_base + local .. ]`, so this is
    /// the `class_base` the dispatch planner applies to the long-range pass.
    #[must_use]
    pub fn long_range_base(&self) -> u32 {
        self.distance.len() as u32
    }

    /// Total constraints across both classes (distance then long-range).
    #[must_use]
    pub fn total(&self) -> usize {
        self.distance.len() + self.long_range.len()
    }
}

/// The dihedral bending upload, colored so that no two hinges in a color share
/// any of their four particles, and repacked contiguous by color.
///
/// Bending constraints live in their own device buffer (base `0`), so the
/// coloring is independent of the distance/long-range coloring.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BendingUploadPlan {
    /// Bending constraints reordered so each color batch is contiguous.
    pub bending: Vec<BendingConstraint>,
    /// Per-color bending-constraint counts, in color order.
    pub bending_colors: Vec<u32>,
}

impl BendingUploadPlan {
    /// Total bending constraints across every color.
    #[must_use]
    pub fn total(&self) -> usize {
        self.bending.len()
    }
}

/// Splits a mixed constraint list into the distance and long-range classes,
/// preserving the input order within each class.
///
/// Distance-class constraints are the two-sided positional edges (stretch,
/// shear and distance-based bend); long-range-class constraints are the
/// one-sided attachments (LRA and tether) reported by
/// [`super::super::ConstraintKind::is_one_sided`].
fn partition_by_sidedness(constraints: &[Constraint]) -> (Vec<Constraint>, Vec<Constraint>) {
    let mut distance: Vec<Constraint> = Vec::new();
    let mut long_range: Vec<Constraint> = Vec::new();
    for constraint in constraints {
        if constraint.kind.is_one_sided() {
            long_range.push(*constraint);
        } else {
            distance.push(*constraint);
        }
    }
    (distance, long_range)
}

/// Colors one class of constraints and returns the by-color contiguous
/// reordering together with its per-color counts.
///
/// Delegates to the deterministic greedy [`color_constraints`], then reads each
/// color batch's length as the per-color count. The reordered constraint list
/// is exactly [`super::super::ConstraintGraph::constraints`], already laid out
/// color by color.
fn color_class(constraints: &[Constraint]) -> (Vec<Constraint>, Vec<u32>) {
    let graph = color_constraints(constraints);
    let colors: Vec<u32> = graph.batches.iter().map(|batch| batch.len).collect();
    (graph.constraints, colors)
}

/// Partitions authored constraints by sidedness, colors each class
/// independently, and repacks them into the contiguous by-color layout the
/// `GPU` dispatch planner expects.
///
/// The returned plan feeds [`super::pipeline::extract`]: `distance_colors` and
/// `long_range_colors` become the per-color count vectors, while `distance`
/// followed by `long_range` is the contiguous constraint buffer content (the
/// long-range section starting at
/// [`ConstraintUploadPlan::long_range_base`]). Empty input yields an empty
/// plan and never panics.
#[must_use]
pub fn plan_constraint_upload(constraints: &[Constraint]) -> ConstraintUploadPlan {
    let (distance_in, long_range_in) = partition_by_sidedness(constraints);
    let (distance, distance_colors) = color_class(&distance_in);
    let (long_range, long_range_colors) = color_class(&long_range_in);
    ConstraintUploadPlan {
        distance,
        distance_colors,
        long_range,
        long_range_colors,
    }
}

/// Greedy four-vertex coloring of dihedral bending constraints.
///
/// Each hinge touches four particles `[edge0, edge1, apex_a, apex_b]`. Two
/// hinges may share a color only when their eight particle slots are pairwise
/// disjoint, so that a color's projections are independent (Jacobi within a
/// color, Gauss-Seidel across colors), matching the distance coloring
/// invariant. The scan is deterministic (first fit in input order) and runs in
/// `O(n * colors)` with no hidden quadratic over the particle sets.
///
/// Returns the by-color contiguous reordering and its per-color counts. Empty
/// input yields an empty plan and never panics.
#[must_use]
pub fn color_bending(bending: &[BendingConstraint]) -> BendingUploadPlan {
    let count = bending.len();
    let mut color_of = vec![0usize; count];
    let mut used_particles: Vec<BTreeSet<u32>> = Vec::new();

    for (i, hinge) in bending.iter().enumerate() {
        let mut chosen: Option<usize> = None;
        for (color, particles) in used_particles.iter().enumerate() {
            if hinge
                .vertices
                .iter()
                .all(|vertex| !particles.contains(vertex))
            {
                chosen = Some(color);
                break;
            }
        }
        let color = match chosen {
            Some(color) => color,
            None => {
                used_particles.push(BTreeSet::new());
                used_particles.len() - 1
            }
        };
        for vertex in hinge.vertices {
            used_particles[color].insert(vertex);
        }
        color_of[i] = color;
    }

    let mut plan = BendingUploadPlan::default();
    for color in 0..used_particles.len() {
        let start = plan.bending.len();
        for (i, hinge) in bending.iter().enumerate() {
            if color_of[i] == color {
                plan.bending.push(*hinge);
            }
        }
        let len = plan.bending.len() - start;
        plan.bending_colors.push(len as u32);
    }
    plan
}

#[cfg(test)]
mod tests {
    use super::super::super::{Compliance, ConstraintKind};
    use super::*;

    /// Builds a two-sided distance constraint between two particles.
    fn distance(a: u32, b: u32, kind: ConstraintKind) -> Constraint {
        Constraint::new(a, b, 1.0, Compliance::RIGID, kind)
    }

    /// Builds a one-sided attachment constraint between two particles.
    fn attach(a: u32, b: u32, kind: ConstraintKind) -> Constraint {
        Constraint::new(a, b, 1.0, Compliance::RIGID, kind)
    }

    /// Builds a dihedral bending constraint over four particles.
    fn hinge(vertices: [u32; 4]) -> BendingConstraint {
        BendingConstraint {
            vertices,
            weights: [1.0, -1.0, 0.5, -0.5],
            scale: 1.0,
            compliance: Compliance::RIGID,
        }
    }

    /// The two color batches of `color_constraints` slice a class into
    /// pairwise-disjoint runs; assert that within each returned color the
    /// distance constraints share no particle.
    fn colors_are_disjoint(constraints: &[Constraint], colors: &[u32]) {
        let mut offset = 0usize;
        for &count in colors {
            let slice = &constraints[offset..offset + count as usize];
            let mut seen: BTreeSet<u32> = BTreeSet::new();
            for constraint in slice {
                assert!(seen.insert(constraint.a), "duplicate particle in color");
                assert!(seen.insert(constraint.b), "duplicate particle in color");
            }
            offset += count as usize;
        }
        assert_eq!(offset, constraints.len());
    }

    #[test]
    fn partition_splits_by_sidedness() {
        let input = vec![
            distance(0, 1, ConstraintKind::Stretch),
            attach(2, 3, ConstraintKind::Lra),
            distance(1, 2, ConstraintKind::Shear),
            attach(4, 5, ConstraintKind::Tether),
        ];
        let plan = plan_constraint_upload(&input);
        assert_eq!(plan.distance.len(), 2);
        assert_eq!(plan.long_range.len(), 2);
        assert!(plan.distance.iter().all(|c| !c.kind.is_one_sided()));
        assert!(plan.long_range.iter().all(|c| c.kind.is_one_sided()));
    }

    #[test]
    fn long_range_base_equals_distance_count() {
        let input = vec![
            distance(0, 1, ConstraintKind::Stretch),
            distance(2, 3, ConstraintKind::Stretch),
            distance(4, 5, ConstraintKind::Bend),
            attach(6, 7, ConstraintKind::Lra),
        ];
        let plan = plan_constraint_upload(&input);
        assert_eq!(plan.long_range_base(), plan.distance.len() as u32);
        assert_eq!(plan.long_range_base(), 3);
        assert_eq!(plan.total(), 4);
    }

    #[test]
    fn per_color_counts_sum_to_class_totals() {
        // Two distance colors expected: (0,1)&(2,3) share nothing so they can
        // share a color; (1,2) collides with (0,1) so it needs a second color.
        let input = vec![
            distance(0, 1, ConstraintKind::Stretch),
            distance(2, 3, ConstraintKind::Stretch),
            distance(1, 2, ConstraintKind::Shear),
            attach(4, 5, ConstraintKind::Lra),
            attach(5, 6, ConstraintKind::Tether),
        ];
        let plan = plan_constraint_upload(&input);
        let distance_sum: u32 = plan.distance_colors.iter().copied().sum();
        let long_range_sum: u32 = plan.long_range_colors.iter().copied().sum();
        assert_eq!(distance_sum as usize, plan.distance.len());
        assert_eq!(long_range_sum as usize, plan.long_range.len());
        colors_are_disjoint(&plan.distance, &plan.distance_colors);
        colors_are_disjoint(&plan.long_range, &plan.long_range_colors);
    }

    #[test]
    fn plan_is_deterministic() {
        let input = vec![
            distance(0, 1, ConstraintKind::Stretch),
            distance(1, 2, ConstraintKind::Shear),
            distance(2, 3, ConstraintKind::Bend),
            attach(3, 4, ConstraintKind::Lra),
        ];
        let a = plan_constraint_upload(&input);
        let b = plan_constraint_upload(&input);
        assert_eq!(a, b);
    }

    #[test]
    fn empty_constraint_input_yields_empty_plan() {
        let plan = plan_constraint_upload(&[]);
        assert!(plan.distance.is_empty());
        assert!(plan.long_range.is_empty());
        assert!(plan.distance_colors.is_empty());
        assert!(plan.long_range_colors.is_empty());
        assert_eq!(plan.long_range_base(), 0);
        assert_eq!(plan.total(), 0);
    }

    #[test]
    fn bending_colors_cover_all_hinges_disjointly() {
        // First two hinges share no vertex → one color; third reuses vertex 0
        // and 1, forcing a second color.
        let input = vec![
            hinge([0, 1, 2, 3]),
            hinge([4, 5, 6, 7]),
            hinge([0, 1, 8, 9]),
        ];
        let plan = color_bending(&input);
        assert_eq!(plan.total(), 3);
        let sum: u32 = plan.bending_colors.iter().copied().sum();
        assert_eq!(sum as usize, plan.bending.len());

        // Within each color, the four-particle slots must be pairwise disjoint.
        let mut offset = 0usize;
        for &count in &plan.bending_colors {
            let slice = &plan.bending[offset..offset + count as usize];
            let mut seen: BTreeSet<u32> = BTreeSet::new();
            for h in slice {
                for v in h.vertices {
                    assert!(seen.insert(v), "shared particle within a bending color");
                }
            }
            offset += count as usize;
        }
        assert_eq!(offset, plan.bending.len());
    }

    #[test]
    fn bending_coloring_is_deterministic() {
        let input = vec![
            hinge([0, 1, 2, 3]),
            hinge([2, 3, 4, 5]),
            hinge([0, 1, 6, 7]),
        ];
        let a = color_bending(&input);
        let b = color_bending(&input);
        assert_eq!(a, b);
    }

    #[test]
    fn empty_bending_input_yields_empty_plan() {
        let plan = color_bending(&[]);
        assert!(plan.bending.is_empty());
        assert!(plan.bending_colors.is_empty());
        assert_eq!(plan.total(), 0);
    }
}
