//! XPBD constraint primitives coupling soft-body particles.
//!
//! Every deformable behaviour in the unified kernel is expressed as a set of
//! *constraints* that the substep solver projects onto the particle positions.
//! Each constraint exposes a scalar constraint function `C` and its gradients;
//! the solver applies the compliant XPBD position correction
//!
//! ```text
//! alpha_tilde = compliance / dt^2
//! delta_lambda = (-C - alpha_tilde * lambda) / (sum_i w_i * |grad C_i|^2 + alpha_tilde)
//! delta_x_i    = w_i * grad C_i * delta_lambda
//! ```
//!
//! where `w_i` is a particle's inverse mass (`0` for pinned particles, which are
//! therefore never moved). A [`compliance`](ParticleConstraint::compliance) of
//! `0` yields a perfectly rigid constraint; larger values yield softer,
//! stretchier behaviour that is independent of the iteration count and time
//! step. The accumulated Lagrange multiplier `lambda` is reset once per substep
//! via [`reset`](ParticleConstraint::reset).
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! compliant XPBD projection above is the standard formulation from Müller et
//! al., "XPBD: Position-Based Simulation of Compliant Constrained Dynamics".

pub mod attachment;
pub mod bending;
pub mod distance;
pub mod long_range;
pub mod pressure;
pub mod set;
pub mod strain_limit;
pub mod volume;

pub use attachment::AttachmentConstraint;
pub use bending::{project_bending, BendingConstraint};
pub use distance::DistanceConstraint;
pub use long_range::{project_long_range, LongRangeConstraint};
pub use pressure::{mesh_volume, PressureConstraint};
pub use set::ConstraintSet;
pub use strain_limit::StrainLimitConstraint;
pub use volume::TetraVolumeConstraint;

use glam::Vec3;

use crate::math::scalar::Real;

/// The category of a soft-body constraint, used for reporting and dispatch.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum SoftConstraintKind {
    /// A distance (stretch) constraint holding two particles at a rest length.
    Distance,
    /// A bending constraint resisting curvature between adjacent elements.
    Bending,
    /// A volume-preservation constraint over a tetrahedron.
    Volume,
    /// An attachment pinning a particle toward a fixed world-space point.
    Attachment,
    /// A one-sided long-range-attachment leash to a fixed anchor.
    LongRange,
    /// A closed-mesh pressure (enclosed-volume) constraint for inflatable cloth.
    Pressure,
    /// A hard biphasic length clamp (strain limiter) over a stretch edge.
    StrainLimit,
}

/// A projectable XPBD constraint over soft-body particles.
///
/// Implementors store the particle indices they couple, a rest configuration,
/// a compliance, and an internal Lagrange multiplier. The solver calls
/// [`reset`](Self::reset) once at the start of each substep, then
/// [`project`](Self::project) once per solver iteration.
pub trait ParticleConstraint {
    /// Returns the category of this constraint.
    fn kind(&self) -> SoftConstraintKind;

    /// Returns the compliance (inverse stiffness) of this constraint. A value
    /// of `0` is perfectly rigid.
    fn compliance(&self) -> Real;

    /// Resets the accumulated Lagrange multiplier to zero. The solver calls
    /// this once at the start of every substep.
    fn reset(&mut self);

    /// Applies one compliant XPBD projection iteration, mutating `positions`
    /// in place.
    ///
    /// `inverse_masses` is index-aligned with `positions`; a value of `0`
    /// marks a pinned particle that must not move. `dt` is the substep
    /// duration (seconds) used to scale the compliance.
    fn project(&mut self, positions: &mut [Vec3], inverse_masses: &[Real], dt: Real);
}
