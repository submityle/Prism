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
//! * [`angular`] — an opt-in per-contact driver
//!   ([`couple_cloth_to_rigid_angular`]) that layers a real analytic torque
//!   (`Σ arm × impulse` mapped through the body's world inverse inertia) on top
//!   of the linear bridge, replacing the old `TODO(angular)` stub. It is gated
//!   on [`ClothRigidCouplingConfig::angular`] and off by default, so the linear
//!   bridge stays bit-identical.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. It is
//! pipeline wiring over textbook position-based-dynamics coupling.

pub mod angular;
pub mod config;
pub mod driver;
pub mod friction;
pub mod proxy;

pub use angular::{
    couple_cloth_to_rigid_angular, world_inverse_inertia_apply, AngularCouplingReport,
};
pub use config::ClothRigidCouplingConfig;
pub use driver::{couple_cloth_to_rigid, gather_rigid_proxies, CouplingReport};
pub use friction::{couple_cloth_to_rigid_friction, FrictionCouplingReport};
pub use proxy::{
    body_collider_from_shape, collider_anchor, collider_overlaps, collider_world_aabb,
    convex_proxy_from_cuboid, proxy_inverse_mass, Aabb, RigidProxy,
};
