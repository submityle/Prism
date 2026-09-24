//! Constraints: the reserved M1 XPBD constraint interface.
//!
//! This module defines the trait and taxonomy that the next milestone's
//! Extended Position-Based Dynamics (XPBD) constraint solver will implement.
//! M0 deliberately ships **no** constraint-solving logic: providing hollow
//! implementations that pretend to solve would be misleading. What is provided
//! here is a genuine, documented interface plus the constraint-kind
//! enumeration, ready to be built upon.

/// The category of a constraint, used for grouping and solver dispatch.
///
/// These mirror the standard XPBD constraint families that M1 will implement.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum ConstraintKind {
    /// A non-penetration contact constraint between two bodies.
    Contact,
    /// A fixed-distance constraint between two anchor points.
    Distance,
    /// A ball-and-socket joint (shared point, free rotation).
    Ball,
    /// A hinge joint (one rotational degree of freedom).
    Hinge,
    /// A prismatic joint (one translational degree of freedom).
    Prismatic,
    /// A volume-preservation constraint over a set of bodies/particles.
    Volume,
    /// A bending constraint between adjacent elements (e.g. cloth/rods).
    Bending,
}

/// A simulation constraint to be resolved by the XPBD solver in M1.
///
/// This is the reserved interface only: M0 defines the shape of a constraint
/// (its [`kind`](Constraint::kind) and how many bodies it couples) without
/// implementing any projection or solving. Later milestones add the position
/// projection methods and reference implementations.
pub trait Constraint {
    /// Returns the category of this constraint.
    fn kind(&self) -> ConstraintKind;

    /// Returns the number of bodies this constraint couples.
    fn body_count(&self) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;

    // A minimal descriptor used only to exercise the reserved interface in
    // tests. It carries no solving logic, matching the M0 scope.
    struct DistanceDescriptor;

    impl Constraint for DistanceDescriptor {
        fn kind(&self) -> ConstraintKind {
            ConstraintKind::Distance
        }

        fn body_count(&self) -> usize {
            2
        }
    }

    #[test]
    fn interface_reports_kind_and_body_count() {
        let c = DistanceDescriptor;
        assert_eq!(c.kind(), ConstraintKind::Distance);
        assert_eq!(c.body_count(), 2);
    }
}
