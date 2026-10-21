//! Objective orthotropic membrane stress and closed-form principal directions.

use glam::{Mat3, Vec3};

use crate::math::scalar::Real;

/// Plane-stress material in the projected warp/weft frame. Moduli and tear
/// thresholds must use the same units. Reciprocal Poisson coupling is derived
/// from the two Young moduli, so the elastic energy remains symmetric.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct MembraneMaterial {
    /// Young modulus along the warp axis.
    pub young_warp: Real,
    /// Young modulus along the weft axis.
    pub young_weft: Real,
    /// Transverse strain in weft under uniaxial warp loading.
    pub poisson: Real,
    /// In-plane shear modulus.
    pub shear: Real,
    /// Rest-space warp direction, projected onto each triangle.
    pub warp: Vec3,
}

impl Default for MembraneMaterial {
    fn default() -> Self {
        Self {
            young_warp: 1.0,
            young_weft: 1.0,
            poisson: 0.3,
            shear: 1.0 / 2.6,
            warp: Vec3::X,
        }
    }
}

impl MembraneMaterial {
    /// Checks positivity of the orthotropic compliance matrix. Invalid
    /// material input disables stress evaluation instead of injecting NaNs.
    #[must_use]
    pub fn is_valid(self) -> bool {
        self.young_warp.is_finite()
            && self.young_warp > 0.0
            && self.young_weft.is_finite()
            && self.young_weft > 0.0
            && self.shear.is_finite()
            && self.shear > 0.0
            && self.poisson.is_finite()
            && 1.0 - self.poisson * self.poisson * self.young_weft / self.young_warp > 1.0e-6
            && self.warp.is_finite()
            && self.warp.length_squared() > 1.0e-12
    }
}

/// Current-space Cauchy membrane stress with rest-area integration weight.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TriangleMembraneStress {
    /// Symmetric, tangent stress embedded in three dimensions.
    pub tensor: Mat3,
    /// Rest area used for vertex aggregation.
    pub rest_area: Real,
    /// Oriented current surface normal.
    pub normal: Vec3,
    /// Current material warp tangent.
    pub warp: Vec3,
}

/// Area-weighted vertex stress and analytic maximum tensile eigenpair.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrincipalMembraneStress {
    /// Maximum in-plane principal stress (compression remains negative).
    pub maximum: Real,
    /// Minimum in-plane principal stress.
    pub minimum: Real,
    /// Unit current-space maximum principal direction, used as crack normal.
    pub direction: Vec3,
    /// Area-weighted current material tangent for directional toughness.
    pub warp: Vec3,
}

fn outer(a: Vec3, b: Vec3) -> Mat3 {
    Mat3::from_cols(a * b.x, a * b.y, a * b.z)
}

/// Computes Green strain, orthotropic second Piola stress, then its Cauchy
/// push-forward. Rigid translations/rotations do not produce membrane strain.
/// Collapsed, non-finite, or degenerate elements return `None`.
#[must_use]
pub fn triangle_membrane_stress(
    rest: [Vec3; 3],
    current: [Vec3; 3],
    material: MembraneMaterial,
) -> Option<TriangleMembraneStress> {
    if !material.is_valid() || rest.iter().chain(&current).any(|p| !p.is_finite()) {
        return None;
    }
    let r1 = rest[1] - rest[0];
    let r2 = rest[2] - rest[0];
    let rest_cross = r1.cross(r2);
    let area2 = rest_cross.length();
    if !area2.is_finite() || area2 <= 1.0e-12 {
        return None;
    }
    let rest_normal = rest_cross / area2;
    let warp = (material.warp - rest_normal * material.warp.dot(rest_normal))
        .try_normalize()
        .or_else(|| r1.try_normalize())?;
    let weft = rest_normal.cross(warp);
    let (a, b, c, d) = (r1.dot(warp), r2.dot(warp), r1.dot(weft), r2.dot(weft));
    let determinant = a * d - b * c;
    if determinant.abs() <= 1.0e-12 || !determinant.is_finite() {
        return None;
    }
    let x1 = current[1] - current[0];
    let x2 = current[2] - current[0];
    let f0 = (x1 * d - x2 * c) / determinant;
    let f1 = (x2 * a - x1 * b) / determinant;
    let cross = f0.cross(f1);
    let jacobian = cross.length();
    if !jacobian.is_finite() || jacobian <= 1.0e-8 {
        return None;
    }
    let e00 = 0.5 * (f0.length_squared() - 1.0);
    let e11 = 0.5 * (f1.length_squared() - 1.0);
    let engineering_shear = f0.dot(f1);
    let nu21 = material.poisson * material.young_weft / material.young_warp;
    let denominator = 1.0 - material.poisson * nu21;
    let s00 = material.young_warp * (e00 + nu21 * e11) / denominator;
    let s11 = material.young_weft * (e11 + material.poisson * e00) / denominator;
    let s01 = material.shear * engineering_shear;
    let tensor =
        (outer(f0, f0) * s00 + outer(f1, f1) * s11 + (outer(f0, f1) + outer(f1, f0)) * s01)
            / jacobian;
    if !tensor.is_finite() {
        return None;
    }
    Some(TriangleMembraneStress {
        tensor,
        rest_area: area2 * 0.5,
        normal: cross / jacobian,
        warp: f0.try_normalize()?,
    })
}

/// Aggregates a vertex one-ring in a common spatial frame before projecting
/// onto its tangent plane. Eigenvalues use a closed-form symmetric 2x2 solve;
/// isotropic stress has a stable material-axis tie break.
#[must_use]
pub fn principal_membrane_stress(
    stresses: impl IntoIterator<Item = TriangleMembraneStress>,
) -> Option<PrincipalMembraneStress> {
    let mut tensor = Mat3::ZERO;
    let mut normal = Vec3::ZERO;
    let mut warp = Vec3::ZERO;
    let mut area = 0.0;
    for stress in stresses {
        tensor += stress.tensor * stress.rest_area;
        normal += stress.normal * stress.rest_area;
        warp += stress.warp * stress.rest_area;
        area += stress.rest_area;
    }
    if area <= 1.0e-12 || !area.is_finite() {
        return None;
    }
    let normal = normal.try_normalize()?;
    let u = (warp - normal * normal.dot(warp)).try_normalize()?;
    let v = normal.cross(u);
    tensor /= area;
    let a = u.dot(tensor * u);
    let b = u.dot(tensor * v);
    let d = v.dot(tensor * v);
    let mean = 0.5 * (a + d);
    let radius = (0.25 * (a - d) * (a - d) + b * b).sqrt();
    let maximum = mean + radius;
    let minimum = mean - radius;
    let direction = if b.abs() > 1.0e-7 * maximum.abs().max(1.0) {
        (u * b + v * (maximum - a)).try_normalize()?
    } else if a >= d {
        u
    } else {
        v
    };
    (maximum.is_finite() && minimum.is_finite()).then_some(PrincipalMembraneStress {
        maximum,
        minimum,
        direction,
        warp: u,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;

    const REST: [Vec3; 3] = [Vec3::ZERO, Vec3::X, Vec3::Y];

    #[test]
    fn rigid_motion_is_objective_and_rotation_transports_principal_axis() {
        let rotation = Quat::from_euler(glam::EulerRot::XYZ, 0.7, -0.4, 1.2);
        let translate = Vec3::new(3.0, -2.0, 4.0);
        let principal = |positions| {
            principal_membrane_stress([triangle_membrane_stress(
                REST,
                positions,
                MembraneMaterial::default(),
            )
            .unwrap()])
            .unwrap()
        };
        let rigid = principal(REST.map(|p| rotation * p + translate));
        assert!(rigid.maximum.abs() < 2.0e-6);
        let stretched = REST.map(|p| p * Vec3::new(1.5, 1.0, 1.0));
        let base = principal(stretched);
        let moved = principal(stretched.map(|p| rotation * p + translate));
        assert!((base.maximum - moved.maximum).abs() < 3.0e-6);
        assert!(moved.direction.dot(rotation * Vec3::X).abs() > 0.99999);
    }

    #[test]
    fn uniaxial_constitutive_response_and_anisotropy() {
        let material = MembraneMaterial {
            poisson: 0.0,
            young_warp: 10.0,
            young_weft: 2.0,
            ..Default::default()
        };
        let warp =
            triangle_membrane_stress(REST, [Vec3::ZERO, Vec3::X * 1.2, Vec3::Y], material).unwrap();
        let weft =
            triangle_membrane_stress(REST, [Vec3::ZERO, Vec3::X, Vec3::Y * 1.2], material).unwrap();
        // Sxx=E*(1.2^2-1)/2; sigma_xx=1.2*Sxx for transverse stretch 1.
        assert!((warp.tensor.x_axis.x - 2.64).abs() < 2.0e-6);
        assert!((weft.tensor.y_axis.y - 0.528).abs() < 1.0e-6);
    }

    #[test]
    fn compression_is_not_tension_and_invalid_elements_are_rejected() {
        let stress =
            triangle_membrane_stress(REST, REST.map(|p| p * 0.7), Default::default()).unwrap();
        assert!(principal_membrane_stress([stress]).unwrap().maximum < 0.0);
        assert!(triangle_membrane_stress(REST, [Vec3::ZERO; 3], Default::default()).is_none());
        assert!(triangle_membrane_stress([Vec3::ZERO; 3], REST, Default::default()).is_none());
        let invalid = MembraneMaterial {
            poisson: 2.0,
            ..Default::default()
        };
        assert!(triangle_membrane_stress(REST, REST, invalid).is_none());
    }
}
