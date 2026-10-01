//! The aero velocity pre-pass over a triangle mesh.
//!
//! [`apply_aero_forces`] walks the mesh triangles, computes each face's wind
//! force with [`triangle_aero_force`], and spreads it evenly across the face's
//! three vertices as a velocity increment `(force / 3) * inverse_mass * dt`.
//! This mirrors the external pre-pass style of the XPBD substep: it nudges
//! velocities before [`crate::soft::solver::integrate::predict`] advances
//! positions, so the subsequent constraint sweeps see the wind-driven motion.
//!
//! Velocities are read and written in triangle order, so a vertex shared by
//! several faces feeds each later face its already-updated velocity (a
//! Gauss-Seidel coupling). This is the sequential reference; a race-free
//! parallel GPU gather would instead read one frozen snapshot per face (a
//! Jacobi update) and the two agree exactly when no vertex is shared.
//!
//! Everything is deterministic and finite: pinned vertices (inverse mass `0`)
//! are skipped, triangles indexing outside the particle set are ignored rather
//! than panicking, and a non-positive or non-finite `dt` is a no-op.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**.

use glam::Vec3;

use crate::math::scalar::Real;
use crate::soft::particle::storage::ParticleColumnsMut;

use super::field::{AeroParams, WindField};
use super::triangle::{triangle_aero_force, turbulence_offset};

/// Applies wind-driven aerodynamic forces to the mesh over `dt` seconds.
///
/// `positions`, `velocities`, and `inverse_masses` are index-aligned particle
/// columns; `velocities` is mutated in place. For each triangle in `triangles`
/// the per-face force from [`triangle_aero_force`] is spread evenly across its
/// three vertices: each free vertex gains `(force / 3) * inverse_mass * dt`.
/// Pinned vertices (inverse mass `0`) are skipped, and triangles indexing
/// outside the particle set are ignored. A non-positive or non-finite `dt`, an
/// empty particle set, and an empty triangle set are all no-ops. `wind` and
/// `aero` are sanitized once up front, and the optional turbulence adds a
/// deterministic per-triangle jitter derived from the vertex indices, so the
/// whole pass is reproducible.
pub fn apply_aero_forces(
    positions: &[Vec3],
    velocities: &mut [Vec3],
    inverse_masses: &[Real],
    triangles: &[[u32; 3]],
    wind: &WindField,
    aero: AeroParams,
    dt: Real,
) {
    let count = positions.len();
    if dt <= 0.0
        || !dt.is_finite()
        || count == 0
        || triangles.is_empty()
        || velocities.len() != count
        || inverse_masses.len() != count
    {
        return;
    }
    let field = wind.sanitized();
    let aero = aero.sanitized();

    for indices in triangles {
        let i0 = indices[0] as usize;
        let i1 = indices[1] as usize;
        let i2 = indices[2] as usize;
        if i0 >= count || i1 >= count || i2 >= count {
            continue;
        }

        let wind_vec = field.velocity + turbulence_offset(*indices, field.turbulence);
        let force = triangle_aero_force(
            positions[i0],
            positions[i1],
            positions[i2],
            velocities[i0],
            velocities[i1],
            velocities[i2],
            wind_vec,
            aero,
        );
        let per_vertex = force * (1.0 / 3.0);

        for &index in &[i0, i1, i2] {
            let w = inverse_masses[index];
            if w <= 0.0 {
                continue;
            }
            velocities[index] += per_vertex * (w * dt);
        }
    }
}

/// Applies the aero pre-pass directly to a [`ParticleColumnsMut`] borrow.
///
/// This is the convenience entry point for the solver path: it forwards the
/// particle columns to [`apply_aero_forces`]. Call it once per frame before the
/// solver step so the wind-driven velocity feeds the substep prediction.
pub fn apply_aero_to_columns(
    columns: &mut ParticleColumnsMut<'_>,
    triangles: &[[u32; 3]],
    wind: &WindField,
    aero: AeroParams,
    dt: Real,
) {
    apply_aero_forces(
        columns.positions,
        columns.velocities,
        columns.inverse_masses,
        triangles,
        wind,
        aero,
        dt,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::soft::particle::ParticleStorage;

    #[test]
    fn single_triangle_pushes_free_vertices_downwind() {
        // A unit XY triangle hit by +Z wind; all three vertices should gain
        // +Z velocity (pure drag spread evenly).
        let positions = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let mut velocities = [Vec3::ZERO; 3];
        let inv = [1.0, 1.0, 1.0];
        let tris = [[0u32, 1, 2]];
        apply_aero_forces(
            &positions,
            &mut velocities,
            &inv,
            &tris,
            &WindField::new(Vec3::new(0.0, 0.0, 2.0), 0.0),
            AeroParams::new(1.0, 0.0),
            1.0,
        );
        for v in &velocities {
            assert!(v.z > 0.0, "velocity {v:?}");
            assert!(v.x.abs() < 1e-6 && v.y.abs() < 1e-6);
        }
        // Symmetric masses => equal share.
        assert!((velocities[0] - velocities[1]).length() < 1e-6);
        assert!((velocities[1] - velocities[2]).length() < 1e-6);
    }

    #[test]
    fn pinned_vertex_is_not_pushed() {
        let positions = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let mut velocities = [Vec3::ZERO; 3];
        let inv = [0.0, 1.0, 1.0]; // vertex 0 pinned
        let tris = [[0u32, 1, 2]];
        apply_aero_forces(
            &positions,
            &mut velocities,
            &inv,
            &tris,
            &WindField::new(Vec3::new(0.0, 0.0, 2.0), 0.0),
            AeroParams::new(1.0, 0.0),
            1.0,
        );
        assert_eq!(velocities[0], Vec3::ZERO);
        assert!(velocities[1].z > 0.0);
    }

    #[test]
    fn non_positive_dt_is_noop() {
        let positions = [Vec3::ZERO, Vec3::X, Vec3::Y];
        let mut velocities = [Vec3::ZERO; 3];
        let inv = [1.0; 3];
        let tris = [[0u32, 1, 2]];
        apply_aero_forces(
            &positions,
            &mut velocities,
            &inv,
            &tris,
            &WindField::new(Vec3::new(0.0, 0.0, 2.0), 0.0),
            AeroParams::new(1.0, 0.0),
            0.0,
        );
        assert_eq!(velocities, [Vec3::ZERO; 3]);
    }

    #[test]
    fn out_of_range_triangle_is_skipped() {
        let positions = [Vec3::ZERO, Vec3::X, Vec3::Y];
        let mut velocities = [Vec3::ZERO; 3];
        let inv = [1.0; 3];
        let tris = [[0u32, 1, 9]]; // index 9 out of range
        apply_aero_forces(
            &positions,
            &mut velocities,
            &inv,
            &tris,
            &WindField::new(Vec3::new(0.0, 0.0, 2.0), 0.0),
            AeroParams::new(1.0, 0.0),
            1.0,
        );
        assert_eq!(velocities, [Vec3::ZERO; 3]);
    }

    #[test]
    fn mismatched_column_lengths_is_noop() {
        let positions = [Vec3::ZERO, Vec3::X, Vec3::Y];
        let mut velocities = [Vec3::ZERO; 2]; // wrong length
        let inv = [1.0; 3];
        let tris = [[0u32, 1, 2]];
        apply_aero_forces(
            &positions,
            &mut velocities,
            &inv,
            &tris,
            &WindField::new(Vec3::new(0.0, 0.0, 2.0), 0.0),
            AeroParams::new(1.0, 0.0),
            1.0,
        );
        assert_eq!(velocities, [Vec3::ZERO; 2]);
    }

    #[test]
    fn columns_entry_point_matches_slice_form() {
        let mut store = ParticleStorage::new();
        store.spawn(Vec3::new(0.0, 0.0, 0.0), 1.0);
        store.spawn(Vec3::new(1.0, 0.0, 0.0), 1.0);
        store.spawn(Vec3::new(0.0, 1.0, 0.0), 1.0);
        let tris = [[0u32, 1, 2]];
        {
            let mut cols = store.columns_mut();
            apply_aero_to_columns(
                &mut cols,
                &tris,
                &WindField::new(Vec3::new(0.0, 0.0, 2.0), 0.0),
                AeroParams::new(1.0, 0.0),
                1.0,
            );
        }
        for i in 0..3 {
            let v = store
                .velocity(crate::soft::particle::ParticleHandle::from_index(i))
                .unwrap();
            assert!(v.z > 0.0, "velocity {v:?}");
        }
    }
}
