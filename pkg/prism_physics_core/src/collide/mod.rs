//! Narrow-phase collision detection.
//!
//! The narrow phase turns a candidate pair of posed
//! [`ColliderShape`](crate::collider::ColliderShape)s into an optional
//! [`ContactManifold`] describing their overlap. This module currently exposes
//! the shared manifold data types in [`contact`]; the per-pair closed-form and
//! polytope routines and the [`generate_contact`](primitives::generate_contact)
//! dispatch live in [`primitives`].
//!
//! # Contract
//!
//! The narrow-phase entry point takes shapes `a` and `b` with their world-space
//! poses and returns `Some(manifold)` when they overlap (or exactly touch
//! within a small tolerance) and `None` otherwise. The returned manifold
//! follows the sign and witness-point conventions documented on [`contact`]:
//! the normal points from `a` toward `b`, penetration is non-negative, and body
//! handles are left as
//! [`BodyHandle::INVALID`](crate::state::handle::BodyHandle::INVALID) for the
//! caller to stamp via [`ContactManifold::with_bodies`].
//!
//! # Provenance
//!
//! The primitive tests (closed-form sphere/capsule/plane distances, the
//! separating-axis test for boxes, and Sutherland-Hodgman face clipping) are
//! standard, publicly documented computational-geometry techniques and contain
//! no Unreal Engine source or derived code.

pub mod contact;
pub mod primitives;

pub use contact::{ContactManifold, ContactPoint, MAX_MANIFOLD_POINTS};
pub use primitives::generate_contact;
