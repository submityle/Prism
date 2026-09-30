//! The `CPU` golden twins of the per-face grid operators that run between the
//! particle-to-grid scatter and the pressure projection in the full fluid step:
//! the body-force integration ([`add_gravity`]) and the no-through-flow solid
//! boundary condition ([`enforce_solid_faces`]).
//!
//! Both operate on the concatenated `[u | v | w]` staggered face array so they
//! share one memory layout with the device kernels in
//! `shaders/fluid_grid_ops.wgsl`. Every arithmetic operation here is a single
//! floating-point add of a host-computed constant or a plain store of `0.0`, so
//! the twin and the device agree bit-for-bit and the real-device parity test
//! can assert exact equality rather than a tolerance.
//!
//! These mirror the reference operators in
//! [`prism_physics_core`](prism_physics_core::fluid::mac_grid): `add_gravity`
//! adds `g * dt` to every face, and `enforce_solid_faces` zeroes the six faces
//! bordering each solid cell so no flow passes through a wall.
//!
//! # Provenance
//!
//! Explicit body-force integration on a staggered `MAC` grid and the solid
//! no-through-flow boundary are standard `CFD` constructs (Harlow and Welch
//! 1965; Bridson, *Fluid Simulation for Computer Graphics*). No Unreal Engine
//! source or derived code.

use glam::Vec3;

use crate::fluid::grid::{CellType, GridDims};

/// Adds the constant velocity increment `gravity * dt` to every face of the
/// concatenated `[u | v | w]` array.
///
/// The `x` component is added to the `u` faces, `y` to the `v` faces, and `z`
/// to the `w` faces, matching the axis each staggered field is normal to. This
/// is the exact per-face store the device `add_gravity` kernel performs, so the
/// two engines produce identical values.
///
/// # Panics
///
/// Panics if `velocity` is not exactly [`GridDims::face_total`] long.
pub fn add_gravity(dims: GridDims, velocity: &mut [f32], gravity: Vec3, dt: f32) {
    assert_eq!(
        velocity.len(),
        dims.face_total(),
        "velocity must hold every concatenated face"
    );
    let dv = gravity * dt;
    let v_base = dims.v_offset();
    let w_base = dims.w_offset();
    for (idx, face) in velocity.iter_mut().enumerate() {
        if idx < v_base {
            *face += dv.x;
        } else if idx < w_base {
            *face += dv.y;
        } else {
            *face += dv.z;
        }
    }
}

/// Zeroes the normal velocity on every face bordering a solid cell, enforcing
/// the no-through-flow wall condition.
///
/// This mirrors the reference scatter exactly: for each solid cell `(i, j, k)`
/// its six bounding faces `u(i)`, `u(i + 1)`, `v(j)`, `v(j + 1)`, `w(k)`,
/// `w(k + 1)` are set to zero. Because the writes only ever store `0.0`, the
/// operation is idempotent and order-independent, so the device kernel is free
/// to evaluate it as a per-face gather (a face is zeroed when either adjacent
/// cell is solid) and still land on the identical field.
///
/// # Panics
///
/// Panics if `cell_types` is not [`GridDims::cell_count`] long or `velocity` is
/// not [`GridDims::face_total`] long.
pub fn enforce_solid_faces(dims: GridDims, cell_types: &[CellType], velocity: &mut [f32]) {
    assert_eq!(
        cell_types.len(),
        dims.cell_count(),
        "cell_types must hold every cell"
    );
    assert_eq!(
        velocity.len(),
        dims.face_total(),
        "velocity must hold every concatenated face"
    );
    let u_base = dims.u_offset();
    let v_base = dims.v_offset();
    let w_base = dims.w_offset();
    for k in 0..dims.nz {
        for j in 0..dims.ny {
            for i in 0..dims.nx {
                if cell_types[dims.cell_idx(i, j, k)] != CellType::Solid {
                    continue;
                }
                velocity[u_base + dims.u_idx(i, j, k)] = 0.0;
                velocity[u_base + dims.u_idx(i + 1, j, k)] = 0.0;
                velocity[v_base + dims.v_idx(i, j, k)] = 0.0;
                velocity[v_base + dims.v_idx(i, j + 1, k)] = 0.0;
                velocity[w_base + dims.w_idx(i, j, k)] = 0.0;
                velocity[w_base + dims.w_idx(i, j, k + 1)] = 0.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Gravity adds the matching axis component to each staggered region and
    /// leaves the orthogonal components untouched.
    #[test]
    fn gravity_adds_per_axis_increment() {
        let dims = GridDims::new(3, 4, 5, 0.1, Vec3::ZERO);
        let mut velocity = vec![1.0f32; dims.face_total()];
        let dt = 0.5;
        let gravity = Vec3::new(2.0, -10.0, 4.0);
        add_gravity(dims, &mut velocity, gravity, dt);

        let v_base = dims.v_offset();
        let w_base = dims.w_offset();
        for (idx, face) in velocity.iter().enumerate() {
            let expected = if idx < v_base {
                1.0 + gravity.x * dt
            } else if idx < w_base {
                1.0 + gravity.y * dt
            } else {
                1.0 + gravity.z * dt
            };
            assert!((face - expected).abs() < 1e-6, "face {idx}: {face}");
        }
    }

    /// A lone solid cell in the interior has all six of its faces forced to
    /// zero while its non-adjacent neighbours keep their values.
    #[test]
    fn solid_cell_zeroes_its_six_faces() {
        let dims = GridDims::new(3, 3, 3, 0.1, Vec3::ZERO);
        let mut cells = vec![CellType::Fluid; dims.cell_count()];
        cells[dims.cell_idx(1, 1, 1)] = CellType::Solid;
        let mut velocity = vec![1.0f32; dims.face_total()];
        enforce_solid_faces(dims, &cells, &mut velocity);

        let u_base = dims.u_offset();
        let v_base = dims.v_offset();
        let w_base = dims.w_offset();
        assert_eq!(velocity[u_base + dims.u_idx(1, 1, 1)], 0.0);
        assert_eq!(velocity[u_base + dims.u_idx(2, 1, 1)], 0.0);
        assert_eq!(velocity[v_base + dims.v_idx(1, 1, 1)], 0.0);
        assert_eq!(velocity[v_base + dims.v_idx(1, 2, 1)], 0.0);
        assert_eq!(velocity[w_base + dims.w_idx(1, 1, 1)], 0.0);
        assert_eq!(velocity[w_base + dims.w_idx(1, 1, 2)], 0.0);

        // A face far from the solid cell is untouched.
        assert_eq!(velocity[u_base + dims.u_idx(0, 0, 0)], 1.0);
    }

    /// Two adjacent solid cells share the face between them; zeroing is
    /// idempotent, so the shared face is simply zero regardless of order.
    #[test]
    fn adjacent_solids_share_a_zeroed_face() {
        let dims = GridDims::new(4, 1, 1, 0.1, Vec3::ZERO);
        let mut cells = vec![CellType::Fluid; dims.cell_count()];
        cells[dims.cell_idx(1, 0, 0)] = CellType::Solid;
        cells[dims.cell_idx(2, 0, 0)] = CellType::Solid;
        let mut velocity = vec![1.0f32; dims.face_total()];
        enforce_solid_faces(dims, &cells, &mut velocity);

        let u_base = dims.u_offset();
        // The shared face u(2) between the two solids, plus their outer faces.
        assert_eq!(velocity[u_base + dims.u_idx(1, 0, 0)], 0.0);
        assert_eq!(velocity[u_base + dims.u_idx(2, 0, 0)], 0.0);
        assert_eq!(velocity[u_base + dims.u_idx(3, 0, 0)], 0.0);
        // The far face u(0) borders only fluid cell 0, so it survives.
        assert_eq!(velocity[u_base + dims.u_idx(0, 0, 0)], 1.0);
    }
}
