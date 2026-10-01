//! Per-frame external contact inputs for the soft-body collide stage.
//!
//! Body-proxy colliders and per-particle backstops are *skinned, animated*
//! data: they move every frame with the character pose, so—unlike the static
//! tunables in [`SoftSolverConfig`](crate::soft::solver::SoftSolverConfig)—they
//! are supplied to each [`SoftSolver::step_with_contacts`] call by borrow rather
//! than stored on the solver or its config. [`SoftContacts`] is the small
//! borrowed bundle that carries them, plus the body-contact friction
//! coefficient, into one substep's collide stage.
//!
//! The empty bundle ([`SoftContacts::EMPTY`]) drives the plain
//! [`SoftSolver::step`], so a garment with no body proxy pays no collide cost
//! beyond the optional self-collision pass.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. It is a
//! plain borrowed parameter bundle.

use crate::math::scalar::Real;
use crate::soft::collision::{Backstop, BodyCollider};

/// Per-frame external contact inputs threaded into one solver step.
///
/// All fields borrow caller-owned, per-frame data. [`body_colliders`] are the
/// analytic body-proxy primitives the garment is kept out of; [`backstops`] are
/// per-particle one-sided planes (`backstops[i]` constrains particle `i`,
/// shorter slices leave trailing particles unconstrained); and
/// [`body_friction`] is the Coulomb coefficient for the body-contact pass
/// (clamped to `0..=1` by the resolver, `0` meaning frictionless).
///
/// [`body_colliders`]: Self::body_colliders
/// [`backstops`]: Self::backstops
/// [`body_friction`]: Self::body_friction
#[derive(Clone, Copy, Debug)]
pub struct SoftContacts<'a> {
    /// Analytic body-proxy colliders the garment is projected out of, applied
    /// in slice order per particle. An empty slice disables body collision.
    pub body_colliders: &'a [BodyCollider],
    /// Per-particle backstop planes; `backstops[i]` constrains particle `i`. A
    /// slice shorter than the particle count leaves trailing particles
    /// unconstrained, and an empty slice disables backstops.
    pub backstops: &'a [Backstop],
    /// Coulomb friction coefficient for the body-contact pass, clamped to
    /// `0..=1` by the resolver; `0` gives a frictionless (pure normal) slide.
    pub body_friction: Real,
}

impl SoftContacts<'_> {
    /// An empty contact bundle: no body colliders, no backstops, no friction.
    /// This is what [`SoftSolver::step`](crate::soft::solver::SoftSolver::step)
    /// feeds the collide stage, so a garment without a body proxy pays nothing
    /// extra.
    pub const EMPTY: SoftContacts<'static> = SoftContacts {
        body_colliders: &[],
        backstops: &[],
        body_friction: 0.0,
    };
}

impl<'a> SoftContacts<'a> {
    /// Creates a frictionless contact bundle from the given body colliders and
    /// per-particle backstops.
    #[must_use]
    pub const fn new(
        body_colliders: &'a [BodyCollider],
        backstops: &'a [Backstop],
    ) -> SoftContacts<'a> {
        SoftContacts {
            body_colliders,
            backstops,
            body_friction: 0.0,
        }
    }

    /// Returns a copy of this bundle with the given body-contact Coulomb
    /// `friction` coefficient (clamped to `0..=1` by the resolver at use time).
    #[must_use]
    pub const fn with_body_friction(mut self, friction: Real) -> SoftContacts<'a> {
        self.body_friction = friction;
        self
    }

    /// Returns `true` when the bundle would run no body-contact or backstop
    /// work (both slices empty), so the collide stage can skip it entirely.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.body_colliders.is_empty() && self.backstops.is_empty()
    }
}

impl Default for SoftContacts<'_> {
    fn default() -> Self {
        SoftContacts::EMPTY
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn empty_bundle_reports_empty() {
        assert!(SoftContacts::EMPTY.is_empty());
        assert!(SoftContacts::default().is_empty());
        assert_eq!(SoftContacts::EMPTY.body_friction, 0.0);
    }

    #[test]
    fn new_bundle_carries_slices_and_zero_friction() {
        let colliders = [BodyCollider::Sphere {
            center: Vec3::ZERO,
            radius: 1.0,
        }];
        let backstops = [Backstop {
            origin: Vec3::ZERO,
            normal: Vec3::Y,
            distance: 0.1,
        }];
        let c = SoftContacts::new(&colliders, &backstops);
        assert!(!c.is_empty());
        assert_eq!(c.body_colliders.len(), 1);
        assert_eq!(c.backstops.len(), 1);
        assert_eq!(c.body_friction, 0.0);
    }

    #[test]
    fn with_body_friction_sets_coefficient() {
        let colliders = [BodyCollider::Sphere {
            center: Vec3::ZERO,
            radius: 1.0,
        }];
        let c = SoftContacts::new(&colliders, &[]).with_body_friction(0.4);
        assert_eq!(c.body_friction, 0.4);
        assert!(!c.is_empty());
    }
}
