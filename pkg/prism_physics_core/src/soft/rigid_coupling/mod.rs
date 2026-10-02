//! The cloth↔rigid two-way coupling bridge.
//!
//! The soft-body kernel already has a tested two-way coupling *math core*
//! ([`crate::soft::resolve_two_way_coupling`]) that exchanges a mass-weighted
//! push between particles and analytic [`crate::soft::BodyCollider`] proxies
//! and records the reaction impulse each proxy accrues. What was missing was
//! the *pipeline*: turning a [`crate::world::PhysicsWorld`]'s rigid bodies into
//! those proxies, running the pass, and writing the reaction back onto the
//! rigid bodies. This module is that bridge.
//!
//! It is split one-concept-per-file:
//!
//! * [`config`] — the opt-in [`ClothRigidCouplingConfig`] flag (default off,
//!   so rigid-only goldens stay bit-identical).
//! * [`proxy`] — the [`ColliderShape`](crate::collider::ColliderShape) →
//!   [`BodyCollider`](crate::soft::BodyCollider) mapping, inverse-mass rule, and
//!   the AABB overlap cull.
//! * [`driver`] — the per-substep gather / resolve / write-back driver.
//!
//! Coupling is **linear only**; angular response is an explicit, documented
//! `TODO(angular)` hook in [`driver`], never faked.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. It is
//! pipeline wiring over textbook position-based-dynamics coupling.

pub mod config;
pub mod driver;
pub mod proxy;

pub use config::ClothRigidCouplingConfig;
pub use driver::{couple_cloth_to_rigid, gather_rigid_proxies, CouplingReport};
pub use proxy::{
    body_collider_from_shape, collider_anchor, collider_overlaps, collider_world_aabb,
    proxy_inverse_mass, Aabb, RigidProxy,
};
