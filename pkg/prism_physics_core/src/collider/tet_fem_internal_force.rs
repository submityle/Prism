//! Nonlinear elastic internal forces for a tetrahedral `FEM` element.
//!
//! For a constant-strain tetrahedron the deformation gradient `F` is constant
//! across the cell, so the elastic energy is `U = V * Psi(F)` with `V` the
//! absolute rest volume and `Psi` a hyperelastic
//! [`HyperelasticModel`](crate::collider::HyperelasticModel) strain energy
//! density. The nodal elastic force is the negative energy gradient
//! `f_i = -dU/dx_i = -V * P(F) * g_i`, where `P` is the first Piola-Kirchhoff
//! stress and `g_i` is the reference-space shape-function gradient stored on
//! the element.
//!
//! These functions are pure and stateless: they map rest geometry plus a set
//! of deformed node positions to a scalar energy or a set of nodal forces,
//! holding no solver state and performing no time integration.

use crate::collider::tet_fem_basis::TetFemElement;
use crate::collider::tet_fem_constitutive::{HyperelasticModel, LameParameters};
use glam::Vec3;

/// Elastic potential energy `U = V * Psi(F)` stored in a single tetrahedral
/// element at the supplied deformed node positions.
///
/// `V` is the absolute rest volume, so inverted rest elements contribute a
/// non-negative volume weight.
#[must_use]
pub fn element_energy(
    element: &TetFemElement,
    positions: [Vec3; 4],
    model: HyperelasticModel,
    lame: &LameParameters,
) -> f32 {
    let [p0, p1, p2, p3] = positions;
    let f = element.deformation_gradient(p0, p1, p2, p3);
    let volume = element.rest_volume.abs();
    volume * model.strain_energy_density(f, lame)
}

/// Nodal elastic forces `f_i = -V * P(F) * g_i` for a single tetrahedral
/// element at the supplied deformed node positions.
///
/// The returned forces are ordered to match `positions`. Because the shape
/// gradients sum to zero, the total force is zero to within rounding, so the
/// element exerts no net force on its centre of mass.
#[must_use]
pub fn element_internal_force(
    element: &TetFemElement,
    positions: [Vec3; 4],
    model: HyperelasticModel,
    lame: &LameParameters,
) -> [Vec3; 4] {
    let [p0, p1, p2, p3] = positions;
    let f = element.deformation_gradient(p0, p1, p2, p3);
    let piola = model.first_piola(f, lame);
    let volume = element.rest_volume.abs();
    let g = element.shape_gradients;
    [
        -volume * (piola * g[0]),
        -volume * (piola * g[1]),
        -volume * (piola * g[2]),
        -volume * (piola * g[3]),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::tet_fem_constitutive::HyperelasticModel as HM;
    use crate::collider::tet_fem_stiffness::IsotropicElasticity;
    use glam::Mat3;

    const REST: [Vec3; 4] = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    ];

    const SKEW: [Vec3; 4] = [
        Vec3::new(0.2, 0.1, -0.1),
        Vec3::new(1.3, 0.2, 0.0),
        Vec3::new(0.1, 1.1, 0.3),
        Vec3::new(-0.2, 0.3, 1.2),
    ];

    const MODELS: [HM; 3] = [HM::Linear, HM::StVenantKirchhoff, HM::StableNeoHookean];

    fn lame() -> LameParameters {
        LameParameters::from_isotropic(&IsotropicElasticity::new(1.0e6, 0.4).unwrap())
    }

    fn make(rest: [Vec3; 4]) -> TetFemElement {
        TetFemElement::from_rest(rest[0], rest[1], rest[2], rest[3], 1e-12).unwrap()
    }

    fn total(forces: [Vec3; 4]) -> Vec3 {
        forces[0] + forces[1] + forces[2] + forces[3]
    }

    #[test]
    fn forces_sum_to_zero() {
        let el = make(SKEW);
        let lame = lame();
        let deformed = [
            SKEW[0] + Vec3::new(0.05, -0.02, 0.03),
            SKEW[1] + Vec3::new(-0.01, 0.04, 0.0),
            SKEW[2] + Vec3::new(0.02, 0.01, -0.03),
            SKEW[3] + Vec3::new(0.0, -0.02, 0.05),
        ];
        for model in MODELS {
            let f = element_internal_force(&el, deformed, model, &lame);
            assert!(
                total(f).length() <= 1e-1,
                "{model:?} net force not zero: {:?}",
                total(f)
            );
        }
    }

    #[test]
    fn rest_configuration_is_force_free() {
        let el = make(SKEW);
        let lame = lame();
        for model in MODELS {
            let f = element_internal_force(&el, SKEW, model, &lame);
            for (i, fi) in f.iter().enumerate() {
                assert!(
                    fi.length() <= 1e-1,
                    "{model:?} node {i} not force free: {fi:?}"
                );
            }
            let u = element_energy(&el, SKEW, model, &lame);
            if model != HM::StableNeoHookean {
                assert!(u.abs() <= 1e-2, "{model:?} rest energy nonzero: {u}");
            }
        }
    }

    #[test]
    fn rigid_rotation_of_objective_models_is_force_free() {
        // Objective models (St.VK, stable Neo-Hookean) produce zero internal
        // force when the whole element is rigidly rotated about its rest state.
        let el = make(REST);
        let lame = lame();
        let r = Mat3::from_axis_angle(Vec3::new(0.2, 0.9, -0.3).normalize(), 0.8);
        let rotated = [r * REST[0], r * REST[1], r * REST[2], r * REST[3]];
        for model in [HM::StVenantKirchhoff, HM::StableNeoHookean] {
            let f = element_internal_force(&el, rotated, model, &lame);
            for fi in f {
                assert!(
                    fi.length() <= 1.0,
                    "{model:?} spurious rotation force: {fi:?}"
                );
            }
        }
        // Linear elasticity is not objective, so a rotation induces force.
        let lin = element_internal_force(&el, rotated, HM::Linear, &lame);
        let max = lin.iter().map(|v| v.length()).fold(0.0_f32, f32::max);
        assert!(max > 1.0, "linear rotation force should be large: {max}");
    }

    #[test]
    fn forces_match_numerical_energy_gradient() {
        // Independent cross-check: f_i must equal the negative numerical
        // gradient of the total element energy with respect to node i.
        let el = make(SKEW);
        let lame = lame();
        let deformed = [
            SKEW[0] + Vec3::new(0.08, -0.03, 0.04),
            SKEW[1] + Vec3::new(-0.02, 0.06, 0.01),
            SKEW[2] + Vec3::new(0.03, 0.02, -0.05),
            SKEW[3] + Vec3::new(-0.01, -0.04, 0.07),
        ];
        let eps = 1.0e-3_f32;
        for model in MODELS {
            let f = element_internal_force(&el, deformed, model, &lame);
            for node in 0..4 {
                for axis in 0..3 {
                    let mut up = deformed;
                    let mut dn = deformed;
                    up[node][axis] += eps;
                    dn[node][axis] -= eps;
                    let num = -(element_energy(&el, up, model, &lame)
                        - element_energy(&el, dn, model, &lame))
                        / (2.0 * eps);
                    let ana = f[node][axis];
                    let scale = 1.0 + ana.abs();
                    assert!(
                        (num - ana).abs() <= 2.0 * scale,
                        "{model:?} f[{node}][{axis}]: num={num} ana={ana}"
                    );
                }
            }
        }
    }

    #[test]
    fn stretch_produces_restoring_force() {
        // Pulling node 1 in +x must create a restoring force in -x on it.
        let el = make(REST);
        let lame = lame();
        let mut deformed = REST;
        deformed[1] += Vec3::new(0.3, 0.0, 0.0);
        for model in MODELS {
            let f = element_internal_force(&el, deformed, model, &lame);
            assert!(
                f[1].x < 0.0,
                "{model:?} node 1 force should oppose stretch: {:?}",
                f[1]
            );
        }
    }

    #[test]
    fn evaluation_is_deterministic() {
        let el = make(SKEW);
        let lame = lame();
        let deformed = [
            SKEW[0] + Vec3::new(0.05, 0.0, 0.0),
            SKEW[1],
            SKEW[2] + Vec3::new(0.0, -0.02, 0.0),
            SKEW[3],
        ];
        for model in MODELS {
            let a = element_internal_force(&el, deformed, model, &lame);
            let b = element_internal_force(&el, deformed, model, &lame);
            for i in 0..4 {
                assert_eq!(a[i].x.to_bits(), b[i].x.to_bits());
                assert_eq!(a[i].y.to_bits(), b[i].y.to_bits());
                assert_eq!(a[i].z.to_bits(), b[i].z.to_bits());
            }
            let e0 = element_energy(&el, deformed, model, &lame);
            let e1 = element_energy(&el, deformed, model, &lame);
            assert_eq!(e0.to_bits(), e1.to_bits());
        }
    }
}
