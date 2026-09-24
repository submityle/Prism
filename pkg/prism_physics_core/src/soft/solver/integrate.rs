//! Substep prediction and velocity finalization for the XPBD soft-body solver.
//!
//! These are the un-constrained halves of a substep. [`predict`] applies
//! external acceleration and velocity damping, snapshots the pre-step positions
//! into `prev_positions`, and advances positions by the (damped) velocity.
//! After the constraint projection sweeps have moved the positions,
//! [`finalize_velocities`] recovers the velocity implied by that motion,
//! `v = (x - x_prev) / h`, which is the defining feature of position-based
//! dynamics: constraints act on positions and velocities simply follow.
//!
//! Pinned particles (inverse mass `0`) are never moved by [`predict`]; their
//! previous position is still snapshotted so [`finalize_velocities`] leaves
//! their velocity at zero.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. Symplectic
//! position prediction and the `v = (x - x_prev) / h` velocity update are the
//! standard position-based-dynamics substep, publicly documented by Müller et
//! al.

use glam::Vec3;

use crate::math::scalar::Real;
use crate::soft::particle::storage::ParticleColumnsMut;

/// Predicts positions for one substep of duration `h` seconds.
///
/// For every non-pinned particle this applies the uniform `acceleration`,
/// damps the velocity by `(1 - damping * h).max(0)`, snapshots the current
/// position into `prev_positions`, then advances the position by the damped
/// velocity. Pinned particles keep their position but still have their previous
/// position snapshotted.
pub fn predict(columns: &mut ParticleColumnsMut<'_>, acceleration: Vec3, damping: Real, h: Real) {
    let damping_scale = (1.0 - damping * h).max(0.0);
    for i in 0..columns.positions.len() {
        let inverse_mass = columns.inverse_masses[i];
        columns.prev_positions[i] = columns.positions[i];
        if inverse_mass <= 0.0 {
            // Pinned: immovable anchor, velocity stays whatever it was (zero).
            continue;
        }
        let mut velocity = columns.velocities[i];
        velocity += acceleration * h;
        velocity *= damping_scale;
        columns.velocities[i] = velocity;
        columns.positions[i] += velocity * h;
    }
}

/// Recovers velocities from the net position change over the substep.
///
/// For each particle `v = (position - prev_position) / h`. Pinned particles did
/// not move, so this yields zero for them.
pub fn finalize_velocities(columns: &mut ParticleColumnsMut<'_>, h: Real) {
    if h <= 0.0 {
        return;
    }
    let inv_h = 1.0 / h;
    for i in 0..columns.positions.len() {
        columns.velocities[i] = (columns.positions[i] - columns.prev_positions[i]) * inv_h;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::soft::particle::ParticleStorage;

    #[test]
    fn predict_advances_free_particle_under_gravity() {
        let mut store = ParticleStorage::new();
        let h = store.spawn(Vec3::ZERO, 1.0);
        let mut cols = store.columns_mut();
        predict(&mut cols, Vec3::new(0.0, -10.0, 0.0), 0.0, 0.5);
        // v = 0 + (-10)*0.5 = -5; x = 0 + (-5)*0.5 = -2.5
        assert_eq!(cols.velocities[h.index()], Vec3::new(0.0, -5.0, 0.0));
        assert_eq!(cols.positions[h.index()], Vec3::new(0.0, -2.5, 0.0));
        assert_eq!(cols.prev_positions[h.index()], Vec3::ZERO);
    }

    #[test]
    fn predict_leaves_pinned_particle_in_place() {
        let mut store = ParticleStorage::new();
        let h = store.spawn_pinned(Vec3::new(1.0, 2.0, 3.0));
        let mut cols = store.columns_mut();
        predict(&mut cols, Vec3::new(0.0, -10.0, 0.0), 0.0, 0.5);
        assert_eq!(cols.positions[h.index()], Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(cols.velocities[h.index()], Vec3::ZERO);
    }

    #[test]
    fn damping_reduces_velocity() {
        let mut store = ParticleStorage::new();
        store.spawn(Vec3::ZERO, 1.0);
        store.set_velocity(
            crate::soft::particle::ParticleHandle::from_index(0),
            Vec3::X,
        );
        let mut cols = store.columns_mut();
        predict(&mut cols, Vec3::ZERO, 1.0, 0.5);
        // damping_scale = 1 - 1*0.5 = 0.5, so v = 1*0.5 = 0.5
        assert_eq!(cols.velocities[0], Vec3::new(0.5, 0.0, 0.0));
    }

    #[test]
    fn finalize_recovers_velocity_from_displacement() {
        let mut store = ParticleStorage::new();
        store.spawn(Vec3::ZERO, 1.0);
        {
            let mut cols = store.columns_mut();
            cols.prev_positions[0] = Vec3::ZERO;
            cols.positions[0] = Vec3::new(0.0, -1.0, 0.0);
            finalize_velocities(&mut cols, 0.5);
        }
        assert_eq!(
            store.velocity(crate::soft::particle::ParticleHandle::from_index(0)),
            Some(Vec3::new(0.0, -2.0, 0.0))
        );
    }
}
