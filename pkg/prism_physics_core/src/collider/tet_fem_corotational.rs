//! Corotational (warped) stiffness for large-rotation tetrahedral `FEM`.
//!
//! Linear tetrahedral elasticity (see [`element_stiffness`]) is only valid for
//! small displacements: a rigid rotation of the element produces large spurious
//! forces because the linear strain measure cannot tell rotation apart from
//! stretch. The corotational method of Müller & Gross removes that error by
//! extracting the per-element rotation `R` with a polar decomposition of the
//! deformation gradient `F`, measuring the displacement in the unrotated frame,
//! applying the linear stiffness there, and rotating the result back:
//!
//! ```text
//! f = -R Ke (Rᵀ x - x0)        (restoring force)
//! K = R Ke Rᵀ                  (warped tangent stiffness)
//! ```
//!
//! Here `x` and `x0` are the current and rest nodal positions stacked into a
//! 12-vector, `Ke` is the linear element stiffness, and `R` acts block-wise on
//! each vertex (the same `3x3` rotation on all four nodes). The warped tangent
//! `K` deliberately ignores the derivative `dR/dx`, which is the standard
//! Müller approximation: it keeps the matrix symmetric positive semidefinite
//! and cheap to assemble while remaining exact at the rest configuration.
//!
//! This module holds no simulation state and performs no time integration. The
//! polar rotation is reused from the crate-level [`polar_rotation`]; nothing
//! here is derived from Unreal Engine source.

use super::tet_fem_basis::TetFemElement;
use super::tet_fem_stiffness::TetStiffness;
use crate::mpm::polar_rotation;
use glam::{Mat3, Vec3};

/// Stacks four nodal vectors into a 12-component column vector.
fn flatten(v: [Vec3; 4]) -> [f32; 12] {
    let mut out = [0.0f32; 12];
    for (i, p) in v.iter().enumerate() {
        out[3 * i] = p.x;
        out[3 * i + 1] = p.y;
        out[3 * i + 2] = p.z;
    }
    out
}

/// Applies a per-vertex `3x3` rotation `r` to each 3-component block of `v`.
fn rotate_blocks(r: Mat3, v: &[f32; 12]) -> [f32; 12] {
    let mut out = [0.0f32; 12];
    for i in 0..4 {
        let block = Vec3::new(v[3 * i], v[3 * i + 1], v[3 * i + 2]);
        let rotated = r * block;
        out[3 * i] = rotated.x;
        out[3 * i + 1] = rotated.y;
        out[3 * i + 2] = rotated.z;
    }
    out
}

/// Reads the `3x3` sub-block `(bi, bj)` of a 12x12 stiffness as a [`Mat3`].
///
/// The returned matrix satisfies `m[row][col] == k[3*bi + row][3*bj + col]`.
fn read_block(k: &[[f32; 12]; 12], bi: usize, bj: usize) -> Mat3 {
    Mat3::from_cols(
        Vec3::new(
            k[3 * bi][3 * bj],
            k[3 * bi + 1][3 * bj],
            k[3 * bi + 2][3 * bj],
        ),
        Vec3::new(
            k[3 * bi][3 * bj + 1],
            k[3 * bi + 1][3 * bj + 1],
            k[3 * bi + 2][3 * bj + 1],
        ),
        Vec3::new(
            k[3 * bi][3 * bj + 2],
            k[3 * bi + 1][3 * bj + 2],
            k[3 * bi + 2][3 * bj + 2],
        ),
    )
}

/// Writes the [`Mat3`] `m` into the `3x3` sub-block `(bi, bj)` of `k`.
fn write_block(k: &mut [[f32; 12]; 12], bi: usize, bj: usize, m: Mat3) {
    let cols = [
        m.x_axis.to_array(),
        m.y_axis.to_array(),
        m.z_axis.to_array(),
    ];
    for row in 0..3 {
        for col in 0..3 {
            k[3 * bi + row][3 * bj + col] = cols[col][row];
        }
    }
}

/// Returns the element rotation `R` as the polar factor of the deformation
/// gradient from the rest configuration to `current`.
///
/// `R` is the closest proper rotation to `F` and is the frame in which the
/// corotational force and stiffness are evaluated.
#[must_use]
pub fn element_rotation(element: &TetFemElement, current: [Vec3; 4]) -> Mat3 {
    let f = element.deformation_gradient(current[0], current[1], current[2], current[3]);
    polar_rotation(f)
}

/// Computes the corotational restoring force on each node.
///
/// Given the precomputed linear element stiffness `linear`, the `rest`
/// configuration used to build it, and the `current` deformed positions, this
/// returns `f = -R Ke (Rᵀ x - x0)` split per vertex. The force vanishes exactly
/// under rigid translation and rigid rotation and reduces to the linear
/// restoring force `-Ke u` for small displacements.
#[must_use]
pub fn corotational_internal_force(
    element: &TetFemElement,
    linear: &TetStiffness,
    rest: [Vec3; 4],
    current: [Vec3; 4],
) -> [Vec3; 4] {
    let r = element_rotation(element, current);
    let rt = r.transpose();
    let x = flatten(current);
    let x0 = flatten(rest);
    let rtx = rotate_blocks(rt, &x);
    let mut u_warp = [0.0f32; 12];
    for i in 0..12 {
        u_warp[i] = rtx[i] - x0[i];
    }
    let ku = linear.apply(&u_warp);
    let r_ku = rotate_blocks(r, &ku);
    [
        Vec3::new(-r_ku[0], -r_ku[1], -r_ku[2]),
        Vec3::new(-r_ku[3], -r_ku[4], -r_ku[5]),
        Vec3::new(-r_ku[6], -r_ku[7], -r_ku[8]),
        Vec3::new(-r_ku[9], -r_ku[10], -r_ku[11]),
    ]
}

/// Computes the warped tangent stiffness `K = R Ke Rᵀ` for `current`.
///
/// The result is a 12x12 [`TetStiffness`] suitable for implicit integration.
/// It is symmetric whenever `linear` is symmetric, equals `linear` at the rest
/// configuration (`R = I`), and is invariant to the choice of world frame.
#[must_use]
pub fn corotational_stiffness(
    element: &TetFemElement,
    linear: &TetStiffness,
    current: [Vec3; 4],
) -> TetStiffness {
    let r = element_rotation(element, current);
    let rt = r.transpose();
    let mut k = [[0.0f32; 12]; 12];
    for bi in 0..4 {
        for bj in 0..4 {
            let block = read_block(&linear.k, bi, bj);
            let warped = r * block * rt;
            write_block(&mut k, bi, bj, warped);
        }
    }
    TetStiffness { k }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::tet_fem_stiffness::{element_stiffness, IsotropicElasticity};

    const REST: [Vec3; 4] = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    ];

    fn element() -> TetFemElement {
        TetFemElement::from_rest(REST[0], REST[1], REST[2], REST[3], 1e-9).unwrap()
    }

    fn material() -> IsotropicElasticity {
        IsotropicElasticity::new(1.0e5, 0.3).unwrap()
    }

    fn max_force(f: &[Vec3; 4]) -> f32 {
        f.iter().map(|v| v.length()).fold(0.0, f32::max)
    }

    #[test]
    fn rest_stiffness_equals_linear() {
        let e = element();
        let linear = element_stiffness(&e, &material());
        let warped = corotational_stiffness(&e, &linear, REST);
        for i in 0..12 {
            for j in 0..12 {
                assert!(
                    (warped.get(i, j) - linear.get(i, j)).abs()
                        <= 1e-3 * (1.0 + linear.get(i, j).abs())
                );
            }
        }
    }

    #[test]
    fn rigid_translation_is_force_free() {
        let e = element();
        let linear = element_stiffness(&e, &material());
        let t = Vec3::new(3.0, -2.0, 5.0);
        let current = [REST[0] + t, REST[1] + t, REST[2] + t, REST[3] + t];
        let f = corotational_internal_force(&e, &linear, REST, current);
        assert!(max_force(&f) <= 1e-2, "max force {}", max_force(&f));
    }

    #[test]
    fn rigid_rotation_is_force_free() {
        let e = element();
        let linear = element_stiffness(&e, &material());
        let q = Mat3::from_axis_angle(Vec3::new(0.3, 0.7, -0.5).normalize(), 1.2);
        let current = [q * REST[0], q * REST[1], q * REST[2], q * REST[3]];
        let f_corot = corotational_internal_force(&e, &linear, REST, current);
        // Linear FEM would produce a large spurious force for the same rotation.
        let x = flatten(current);
        let x0 = flatten(REST);
        let mut u = [0.0f32; 12];
        for i in 0..12 {
            u[i] = x[i] - x0[i];
        }
        let ku = linear.apply(&u);
        let linear_force = ku.iter().map(|&c| c.abs()).fold(0.0, f32::max);
        assert!(
            max_force(&f_corot) <= 1e-2,
            "corot force {}",
            max_force(&f_corot)
        );
        assert!(
            linear_force > 1.0,
            "linear force should be large: {linear_force}"
        );
    }

    #[test]
    fn small_displacement_matches_linear_force() {
        let e = element();
        let linear = element_stiffness(&e, &material());
        let d = 1e-4;
        let current = [REST[0], REST[1] + Vec3::new(d, 0.0, 0.0), REST[2], REST[3]];
        let f_corot = corotational_internal_force(&e, &linear, REST, current);
        let x = flatten(current);
        let x0 = flatten(REST);
        let mut u = [0.0f32; 12];
        for i in 0..12 {
            u[i] = x[i] - x0[i];
        }
        let ku = linear.apply(&u);
        let f_lin = [
            Vec3::new(-ku[0], -ku[1], -ku[2]),
            Vec3::new(-ku[3], -ku[4], -ku[5]),
            Vec3::new(-ku[6], -ku[7], -ku[8]),
            Vec3::new(-ku[9], -ku[10], -ku[11]),
        ];
        for i in 0..4 {
            assert!((f_corot[i] - f_lin[i]).length() <= 1e-2 * (1.0 + f_lin[i].length()));
        }
    }

    #[test]
    fn warped_stiffness_is_symmetric() {
        let e = element();
        let linear = element_stiffness(&e, &material());
        let q = Mat3::from_axis_angle(Vec3::Y, 0.8);
        let current = [q * REST[0], q * REST[1], q * REST[2], q * REST[3]];
        let k = corotational_stiffness(&e, &linear, current);
        for i in 0..12 {
            for j in 0..12 {
                let diff = (k.get(i, j) - k.get(j, i)).abs();
                assert!(
                    diff <= 1e-2 * (1.0 + k.get(i, j).abs()),
                    "asym {i},{j}: {diff}"
                );
            }
        }
    }

    #[test]
    fn warped_stiffness_equals_r_k_rt() {
        // Independent reconstruction of R Ke Rᵀ block by block.
        let e = element();
        let linear = element_stiffness(&e, &material());
        let q = Mat3::from_axis_angle(Vec3::new(1.0, 1.0, 1.0).normalize(), 0.6);
        let current = [q * REST[0], q * REST[1], q * REST[2], q * REST[3]];
        let r = element_rotation(&e, current);
        let rt = r.transpose();
        let k = corotational_stiffness(&e, &linear, current);
        for bi in 0..4 {
            for bj in 0..4 {
                let block = read_block(&linear.k, bi, bj);
                let expected = r * block * rt;
                for row in 0..3 {
                    for col in 0..3 {
                        let got = k.get(3 * bi + row, 3 * bj + col);
                        let want = expected.to_cols_array_2d()[col][row];
                        assert!((got - want).abs() <= 1e-2 * (1.0 + want.abs()));
                    }
                }
            }
        }
    }

    #[test]
    fn warped_energy_is_non_negative() {
        let e = element();
        let linear = element_stiffness(&e, &material());
        let q = Mat3::from_axis_angle(Vec3::X, 0.4);
        let current = [
            q * (REST[0] * 1.1),
            q * (REST[1] * 1.1),
            q * (REST[2] * 1.1),
            q * (REST[3] * 1.1),
        ];
        let k = corotational_stiffness(&e, &linear, current);
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut rand = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        };
        let u: [f32; 12] = core::array::from_fn(|_| rand());
        assert!(k.energy(&u) >= -1e-3, "energy {}", k.energy(&u));
    }

    #[test]
    fn is_deterministic() {
        let e = element();
        let linear = element_stiffness(&e, &material());
        let q = Mat3::from_axis_angle(Vec3::Z, 0.9);
        let current = [q * REST[0], q * REST[1], q * REST[2], q * REST[3]];
        let a = corotational_stiffness(&e, &linear, current);
        let b = corotational_stiffness(&e, &linear, current);
        for i in 0..12 {
            for j in 0..12 {
                assert_eq!(a.get(i, j).to_bits(), b.get(i, j).to_bits());
            }
        }
        let fa = corotational_internal_force(&e, &linear, REST, current);
        let fb = corotational_internal_force(&e, &linear, REST, current);
        for i in 0..4 {
            assert_eq!(fa[i].to_array(), fb[i].to_array());
        }
    }
}
