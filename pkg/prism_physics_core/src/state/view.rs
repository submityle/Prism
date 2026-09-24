//! A borrowing Structure-of-Arrays view for position-based solvers.
//!
//! [`BodySolverView`] hands a solver mutable access to the hot per-body columns
//! (pose, previous pose, and velocities) alongside immutable descriptive
//! columns (mass, kind, collider, material, occupancy). All slices are indexed
//! by the same slot index, so a solver can visit body `i` by reading position
//! `positions[i]`, mass `mass_props[i]`, and so on.
//!
//! The view is produced by
//! [`BodyStorage::solver_view_mut`](crate::state::storage::BodyStorage::solver_view_mut)
//! and borrows the storage for its lifetime.

use crate::collider::{ColliderHandle, PhysicsMaterial};
use crate::state::body::{BodyKind, MassProperties};
use glam::{Quat, Vec3};

/// A mutable Structure-of-Arrays view over the columns a position-based solver
/// needs.
///
/// Every slice has the same length (the storage slot count). Freed slots are
/// still present; call [`BodySolverView::is_active`] before touching a slot.
pub struct BodySolverView<'a> {
    /// Current world-space positions.
    pub positions: &'a mut [Vec3],
    /// Current world-space orientations.
    pub orientations: &'a mut [Quat],
    /// Positions saved at the start of the current sub-step.
    pub prev_positions: &'a mut [Vec3],
    /// Orientations saved at the start of the current sub-step.
    pub prev_orientations: &'a mut [Quat],
    /// Current linear velocities.
    pub linear_velocities: &'a mut [Vec3],
    /// Current angular velocities (world-space axis-angle rate).
    pub angular_velocities: &'a mut [Vec3],
    /// Inverse mass and inverse principal inertia per body.
    pub mass_props: &'a [MassProperties],
    /// Simulation category per body.
    pub kinds: &'a [BodyKind],
    /// Optional collider handle per body.
    pub colliders: &'a [Option<ColliderHandle>],
    /// Contact material per body.
    pub materials: &'a [PhysicsMaterial],
    /// Linear velocity damping coefficient per body (per second).
    pub linear_damping: &'a [f32],
    /// Angular velocity damping coefficient per body (per second).
    pub angular_damping: &'a [f32],
    /// Slot occupancy flags.
    pub active: &'a [bool],
}

impl BodySolverView<'_> {
    /// Returns the number of slots (including freed ones) in the view.
    #[must_use]
    pub fn slot_count(&self) -> usize {
        self.positions.len()
    }

    /// Returns `true` if slot `i` holds a live body.
    #[must_use]
    pub fn is_active(&self, i: usize) -> bool {
        self.active.get(i).copied().unwrap_or(false)
    }

    /// Returns `true` if slot `i` is a live, fully simulated dynamic body.
    #[must_use]
    pub fn is_dynamic(&self, i: usize) -> bool {
        self.is_active(i) && self.kinds[i] == BodyKind::Dynamic
    }
}
