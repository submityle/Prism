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
use crate::soft::collision::{Backstop, BodyCollider, VirtualParticle};

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
    /// `NvCloth`-style virtual particles sampling the garment's triangle
    /// interiors, folded into the self-collision tier so a vertex cannot tunnel
    /// through a triangle's face between its three corners. These are a
    /// *topology-derived*, per-frame input (regenerated when the mesh retopos or
    /// tears), so—like the body colliders—they are borrowed into each step
    /// rather than stored on the config. An empty slice disables the virtual
    /// tier regardless of the config gate. Gate the pass with
    /// [`SoftSolverConfig::virtual_self_collision`](crate::soft::solver::SoftSolverConfig::virtual_self_collision).
    pub virtual_particles: &'a [VirtualParticle],
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
        virtual_particles: &[],
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
            virtual_particles: &[],
        }
    }

    /// Returns a copy of this bundle with the given body-contact Coulomb
    /// `friction` coefficient (clamped to `0..=1` by the resolver at use time).
    #[must_use]
    pub const fn with_body_friction(mut self, friction: Real) -> SoftContacts<'a> {
        self.body_friction = friction;
        self
    }

    /// Returns a copy of this bundle carrying the given `NvCloth`-style
    /// `virtual_particles` for the self-collision tier. The virtual pass still
    /// only runs when
    /// [`SoftSolverConfig::virtual_self_collision`](crate::soft::solver::SoftSolverConfig::virtual_self_collision)
    /// is set; an empty slice (the default) leaves the garment's vertex-only
    /// self-collision behaviour unchanged.
    #[must_use]
    pub const fn with_virtual_particles(
        mut self,
        virtual_particles: &'a [VirtualParticle],
    ) -> SoftContacts<'a> {
        self.virtual_particles = virtual_particles;
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
        assert!(SoftContacts::EMPTY.virtual_particles.is_empty());
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

    #[test]
    fn with_virtual_particles_carries_slice_without_changing_emptiness() {
        let vps = [VirtualParticle {
            verts: [0, 1, 2],
            weights: [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        }];
        // Virtual particles alone do not flip `is_empty`, which only gates the
        // body-contact/backstop work; the virtual tier is gated separately by
        // `SoftSolverConfig::virtual_self_collision`.
        let c = SoftContacts::new(&[], &[]).with_virtual_particles(&vps);
        assert!(c.is_empty());
        assert_eq!(c.virtual_particles.len(), 1);
        assert_eq!(c.virtual_particles[0].verts, [0, 1, 2]);
    }
}
