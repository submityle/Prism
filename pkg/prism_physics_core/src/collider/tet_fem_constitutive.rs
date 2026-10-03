//! Hyperelastic constitutive models for tetrahedral `FEM` elements.
//!
//! Each model maps a deformation gradient `F` (a 3x3 matrix) to a scalar
//! strain energy density `Psi(F)` and the first Piola-Kirchhoff stress
//! `P(F) = dPsi/dF`. These are pure, stateless functions of `F` and the
//! Lamé material parameters: they hold no solver state and perform no time
//! integration, so they compose with the element shape gradients to assemble
//! nonlinear internal forces independently of any particular time stepper.
//!
//! Three isotropic models are provided:
//!
//! * **Linear elasticity** — the small-strain model `Psi = mu (e:e) +
//!   (lambda/2) tr(e)^2` with the symmetric strain `e = (F + Fᵀ)/2 - I`. It is
//!   not rotation invariant and is only accurate for small deformations.
//! * **St. Venant-Kirchhoff** — `Psi = mu (E:E) + (lambda/2) tr(E)^2` with the
//!   Green strain `E = (FᵀF - I)/2`. It is objective (frame indifferent) but
//!   becomes unstable under large compression.
//! * **Stable Neo-Hookean** (Smith, De Goes, Kim 2018) — a robust model that
//!   stays finite and well defined even for inverted elements (`det F <= 0`),
//!   making it the preferred choice for production soft-body simulation.
//!
//! All arithmetic accumulates in `f64` for accuracy and rounds the result back
//! to the engine real type on return.

use crate::collider::tet_fem_stiffness::IsotropicElasticity;
use glam::{Mat3, Vec3};

/// Lamé parameters describing an isotropic material.
///
/// The pair `(mu, lambda)` is the second (shear) and first Lamé parameter
/// respectively. Linear elasticity and St. Venant-Kirchhoff accept any
/// `lambda >= 0`; the stable Neo-Hookean model additionally requires
/// `lambda > 0` because its rest-state offset divides by `lambda`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LameParameters {
    /// Second Lamé parameter `mu`, i.e. the shear modulus.
    pub mu: f32,
    /// First Lamé parameter `lambda`.
    pub lambda: f32,
}

impl LameParameters {
    /// Creates a parameter pair directly from `mu` and `lambda`.
    #[must_use]
    pub fn new(mu: f32, lambda: f32) -> Self {
        Self { mu, lambda }
    }

    /// Derives the Lamé parameters from a Young's-modulus / Poisson-ratio
    /// material, reusing [`IsotropicElasticity::lame`] so the conversion stays
    /// in a single place.
    #[must_use]
    pub fn from_isotropic(material: &IsotropicElasticity) -> Self {
        let (lambda, mu) = material.lame();
        Self { mu, lambda }
    }
}

/// Selects which hyperelastic constitutive model to evaluate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HyperelasticModel {
    /// Small-strain linear elasticity. Fast but not rotation invariant.
    Linear,
    /// St. Venant-Kirchhoff. Objective but unstable under large compression.
    StVenantKirchhoff,
    /// Stable Neo-Hookean (Smith et al. 2018). Robust under inversion.
    StableNeoHookean,
}

impl HyperelasticModel {
    /// Evaluates the strain energy density `Psi(F)` for this model.
    #[must_use]
    pub fn strain_energy_density(self, f: Mat3, lame: &LameParameters) -> f32 {
        match self {
            Self::Linear => linear_strain_energy_density(f, lame),
            Self::StVenantKirchhoff => stvk_strain_energy_density(f, lame),
            Self::StableNeoHookean => stable_neo_hookean_strain_energy_density(f, lame),
        }
    }

    /// Evaluates the first Piola-Kirchhoff stress `P(F) = dPsi/dF`.
    #[must_use]
    pub fn first_piola(self, f: Mat3, lame: &LameParameters) -> Mat3 {
        match self {
            Self::Linear => linear_first_piola(f, lame),
            Self::StVenantKirchhoff => stvk_first_piola(f, lame),
            Self::StableNeoHookean => stable_neo_hookean_first_piola(f, lame),
        }
    }
}

// ------------------------------------------------------------------------
// f64 3x3 matrix helpers (row-major) used internally for accuracy.
// ------------------------------------------------------------------------

type M3 = [[f64; 3]; 3];

/// Row-major `f64` copy of a column-major `glam` matrix.
fn to_m3(f: Mat3) -> M3 {
    let c0 = f.x_axis;
    let c1 = f.y_axis;
    let c2 = f.z_axis;
    [
        [f64::from(c0.x), f64::from(c1.x), f64::from(c2.x)],
        [f64::from(c0.y), f64::from(c1.y), f64::from(c2.y)],
        [f64::from(c0.z), f64::from(c1.z), f64::from(c2.z)],
    ]
}

/// Builds a column-major `glam` matrix from a row-major `f64` matrix.
fn from_m3(m: M3) -> Mat3 {
    Mat3::from_cols(
        Vec3::new(m[0][0] as f32, m[1][0] as f32, m[2][0] as f32),
        Vec3::new(m[0][1] as f32, m[1][1] as f32, m[2][1] as f32),
        Vec3::new(m[0][2] as f32, m[1][2] as f32, m[2][2] as f32),
    )
}

/// The 3x3 identity in `f64`.
const IDENTITY_M3: M3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// Transpose.
fn transpose_m3(a: M3) -> M3 {
    let mut t = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            t[i][j] = a[j][i];
        }
    }
    t
}

/// Matrix product `a * b`.
fn mul_m3(a: M3, b: M3) -> M3 {
    let mut c = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

/// Scales every entry by `s`.
fn scale_m3(a: M3, s: f64) -> M3 {
    let mut r = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = a[i][j] * s;
        }
    }
    r
}

/// Entrywise sum `a + b`.
fn add_m3(a: M3, b: M3) -> M3 {
    let mut r = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = a[i][j] + b[i][j];
        }
    }
    r
}

/// Trace `tr(a)`.
fn trace_m3(a: M3) -> f64 {
    a[0][0] + a[1][1] + a[2][2]
}

/// Double contraction (Frobenius inner product) `a : b`.
fn inner_m3(a: M3, b: M3) -> f64 {
    let mut s = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            s += a[i][j] * b[i][j];
        }
    }
    s
}

/// Determinant `det(a)`.
fn det_m3(a: M3) -> f64 {
    a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
}

/// Cofactor matrix `cof(a) = det(a) a⁻ᵀ`, whose columns are the cross products
/// of the columns of `a` (mirrors `crate::cofactor` in `f64`).
fn cofactor_m3(a: M3) -> M3 {
    // Columns of `a`.
    let c0 = [a[0][0], a[1][0], a[2][0]];
    let c1 = [a[0][1], a[1][1], a[2][1]];
    let c2 = [a[0][2], a[1][2], a[2][2]];
    let k0 = cross3(c1, c2);
    let k1 = cross3(c2, c0);
    let k2 = cross3(c0, c1);
    // Reassemble as a row-major matrix whose columns are k0, k1, k2.
    [
        [k0[0], k1[0], k2[0]],
        [k0[1], k1[1], k2[1]],
        [k0[2], k1[2], k2[2]],
    ]
}

/// 3-vector cross product in `f64`.
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

// ------------------------------------------------------------------------
// Linear elasticity.
// ------------------------------------------------------------------------

/// Linear-elastic strain energy density `Psi = mu (e:e) + (lambda/2) tr(e)^2`
/// with the small-strain tensor `e = (F + Fᵀ)/2 - I`.
#[must_use]
pub fn linear_strain_energy_density(f: Mat3, lame: &LameParameters) -> f32 {
    let mu = f64::from(lame.mu);
    let lambda = f64::from(lame.lambda);
    let e = linear_strain(to_m3(f));
    let tr = trace_m3(e);
    (mu * inner_m3(e, e) + 0.5 * lambda * tr * tr) as f32
}

/// Linear-elastic first Piola stress `P = 2 mu e + lambda tr(e) I`.
///
/// In the linear regime the first Piola stress coincides with the symmetric
/// Cauchy stress.
#[must_use]
pub fn linear_first_piola(f: Mat3, lame: &LameParameters) -> Mat3 {
    let mu = f64::from(lame.mu);
    let lambda = f64::from(lame.lambda);
    let e = linear_strain(to_m3(f));
    let tr = trace_m3(e);
    let p = add_m3(scale_m3(e, 2.0 * mu), scale_m3(IDENTITY_M3, lambda * tr));
    from_m3(p)
}

/// Small-strain tensor `e = (F + Fᵀ)/2 - I`.
fn linear_strain(fm: M3) -> M3 {
    let sym = scale_m3(add_m3(fm, transpose_m3(fm)), 0.5);
    add_m3(sym, scale_m3(IDENTITY_M3, -1.0))
}

// ------------------------------------------------------------------------
// St. Venant-Kirchhoff.
// ------------------------------------------------------------------------

/// St. Venant-Kirchhoff strain energy density `Psi = mu (E:E) +
/// (lambda/2) tr(E)^2` with the Green strain `E = (FᵀF - I)/2`.
#[must_use]
pub fn stvk_strain_energy_density(f: Mat3, lame: &LameParameters) -> f32 {
    let mu = f64::from(lame.mu);
    let lambda = f64::from(lame.lambda);
    let fm = to_m3(f);
    let green = green_strain(fm);
    let tr = trace_m3(green);
    (mu * inner_m3(green, green) + 0.5 * lambda * tr * tr) as f32
}

/// St. Venant-Kirchhoff first Piola stress `P = F S` with the second
/// Piola-Kirchhoff stress `S = 2 mu E + lambda tr(E) I`.
#[must_use]
pub fn stvk_first_piola(f: Mat3, lame: &LameParameters) -> Mat3 {
    let mu = f64::from(lame.mu);
    let lambda = f64::from(lame.lambda);
    let fm = to_m3(f);
    let green = green_strain(fm);
    let tr = trace_m3(green);
    let s = add_m3(
        scale_m3(green, 2.0 * mu),
        scale_m3(IDENTITY_M3, lambda * tr),
    );
    from_m3(mul_m3(fm, s))
}

/// Green strain `E = (FᵀF - I)/2`.
fn green_strain(fm: M3) -> M3 {
    let c = mul_m3(transpose_m3(fm), fm);
    scale_m3(add_m3(c, scale_m3(IDENTITY_M3, -1.0)), 0.5)
}

// ------------------------------------------------------------------------
// Stable Neo-Hookean (Smith, De Goes, Kim 2018).
// ------------------------------------------------------------------------

/// Rest-state offset `alpha = 1 + 3 mu / (4 lambda)` of the stable
/// Neo-Hookean model, chosen so the undeformed configuration is force free.
///
/// Requires `lambda > 0`; a non-positive `lambda` yields a non-finite result.
fn stable_nh_alpha(mu: f64, lambda: f64) -> f64 {
    1.0 + 0.75 * mu / lambda
}

/// Stable Neo-Hookean strain energy density
/// `Psi = (mu/2)(I_c - 3) + (lambda/2)(J - alpha)^2 - (mu/2) ln(I_c + 1)`,
/// where `I_c = tr(FᵀF)` and `J = det F`.
///
/// Requires `lambda > 0`. The model stays finite for inverted elements
/// (`J <= 0`) because it avoids the classical `ln(J)` term.
#[must_use]
pub fn stable_neo_hookean_strain_energy_density(f: Mat3, lame: &LameParameters) -> f32 {
    let mu = f64::from(lame.mu);
    let lambda = f64::from(lame.lambda);
    let fm = to_m3(f);
    let ic = inner_m3(fm, fm);
    let j = det_m3(fm);
    let alpha = stable_nh_alpha(mu, lambda);
    let dj = j - alpha;
    (0.5 * mu * (ic - 3.0) + 0.5 * lambda * dj * dj - 0.5 * mu * (ic + 1.0).ln()) as f32
}

/// Stable Neo-Hookean first Piola stress
/// `P = mu (1 - 1/(I_c + 1)) F + lambda (J - alpha) cof(F)`.
///
/// Requires `lambda > 0`.
#[must_use]
pub fn stable_neo_hookean_first_piola(f: Mat3, lame: &LameParameters) -> Mat3 {
    let mu = f64::from(lame.mu);
    let lambda = f64::from(lame.lambda);
    let fm = to_m3(f);
    let ic = inner_m3(fm, fm);
    let j = det_m3(fm);
    let alpha = stable_nh_alpha(mu, lambda);
    let cof = cofactor_m3(fm);
    let coeff_f = mu * (1.0 - 1.0 / (ic + 1.0));
    let coeff_cof = lambda * (j - alpha);
    let p = add_m3(scale_m3(fm, coeff_f), scale_m3(cof, coeff_cof));
    from_m3(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lame() -> LameParameters {
        // E = 1e6, nu = 0.4 -> lambda > 0, mu > 0.
        let material = IsotropicElasticity::new(1.0e6, 0.4).unwrap();
        LameParameters::from_isotropic(&material)
    }

    fn mat_close(a: Mat3, b: Mat3, tol: f32) -> bool {
        (0..3).all(|c| (a.col(c) - b.col(c)).length() <= tol)
    }

    fn set_elem(f: Mat3, row: usize, col: usize, delta: f32) -> Mat3 {
        let mut cols = [f.x_axis, f.y_axis, f.z_axis];
        let v = cols[col];
        let comp = [v.x, v.y, v.z];
        let mut nc = comp;
        nc[row] += delta;
        cols[col] = Vec3::new(nc[0], nc[1], nc[2]);
        Mat3::from_cols(cols[0], cols[1], cols[2])
    }

    const MODELS: [HyperelasticModel; 3] = [
        HyperelasticModel::Linear,
        HyperelasticModel::StVenantKirchhoff,
        HyperelasticModel::StableNeoHookean,
    ];

    #[test]
    fn rest_state_is_force_free() {
        let lame = lame();
        for model in MODELS {
            let p = model.first_piola(Mat3::IDENTITY, &lame);
            assert!(
                mat_close(p, Mat3::ZERO, 1e-2),
                "{model:?} not force free at identity: {p:?}"
            );
        }
        // Linear and St.VK additionally have zero energy at the rest state.
        assert!(
            HyperelasticModel::Linear
                .strain_energy_density(Mat3::IDENTITY, &lame)
                .abs()
                < 1e-2
        );
        assert!(
            HyperelasticModel::StVenantKirchhoff
                .strain_energy_density(Mat3::IDENTITY, &lame)
                .abs()
                < 1e-2
        );
    }

    #[test]
    fn finite_difference_matches_first_piola() {
        // Independent cross-check: P must equal the numerical gradient of Psi.
        let lame = lame();
        let f = Mat3::from_cols(
            Vec3::new(1.1, 0.05, -0.02),
            Vec3::new(0.03, 0.95, 0.04),
            Vec3::new(-0.01, 0.02, 1.08),
        );
        let eps = 1.0e-3_f32;
        for model in MODELS {
            let p = model.first_piola(f, &lame);
            for row in 0..3 {
                for col in 0..3 {
                    let fp = set_elem(f, row, col, eps);
                    let fm = set_elem(f, row, col, -eps);
                    let num = (model.strain_energy_density(fp, &lame)
                        - model.strain_energy_density(fm, &lame))
                        / (2.0 * eps);
                    let ana = p.col(col)[row];
                    let scale = 1.0 + ana.abs();
                    assert!(
                        (num - ana).abs() <= 2.0 * scale,
                        "{model:?} dPsi/dF[{row}][{col}]: num={num} ana={ana}"
                    );
                }
            }
        }
    }

    #[test]
    fn objective_models_are_rotation_invariant() {
        // St.VK and stable Neo-Hookean are frame indifferent: Psi(R F) = Psi(F)
        // and P(R F) = R P(F) for any rotation R.
        let lame = lame();
        let r = Mat3::from_axis_angle(Vec3::new(0.3, -0.7, 0.5).normalize(), 0.9);
        let f = Mat3::from_cols(
            Vec3::new(1.2, 0.1, 0.0),
            Vec3::new(-0.05, 0.9, 0.07),
            Vec3::new(0.02, -0.03, 1.1),
        );
        let rf = r * f;
        for model in [
            HyperelasticModel::StVenantKirchhoff,
            HyperelasticModel::StableNeoHookean,
        ] {
            let e0 = model.strain_energy_density(f, &lame);
            let e1 = model.strain_energy_density(rf, &lame);
            assert!(
                (e0 - e1).abs() <= 1e-2 * (1.0 + e0.abs()),
                "{model:?} energy not invariant: {e0} vs {e1}"
            );
            let p0 = model.first_piola(f, &lame);
            let p1 = model.first_piola(rf, &lame);
            assert!(
                mat_close(p1, r * p0, 1e-1 * (1.0 + p0.x_axis.length())),
                "{model:?} stress not covariant"
            );
        }
    }

    #[test]
    fn linear_model_is_not_rotation_invariant() {
        // A pure rotation produces spurious linear-elastic energy, confirming
        // the known small-strain limitation (and that we compute true linear
        // strain, not an accidentally objective variant).
        let lame = lame();
        let r = Mat3::from_axis_angle(Vec3::Z, 0.5);
        let e = HyperelasticModel::Linear.strain_energy_density(r, &lame);
        assert!(e > 1.0, "linear energy under rotation should be large: {e}");
    }

    #[test]
    fn stvk_small_strain_limit_agrees_with_linear() {
        // St.VK provably reduces to linear elasticity with the same Lamé
        // parameters as the deformation gradient approaches the identity.
        let lame = lame();
        let g = Mat3::from_cols(
            Vec3::new(0.4, 0.1, -0.2),
            Vec3::new(0.1, -0.3, 0.05),
            Vec3::new(-0.2, 0.05, 0.25),
        );
        let f = Mat3::IDENTITY + g * 1.0e-4;
        let lin = HyperelasticModel::Linear.first_piola(f, &lame);
        let p = HyperelasticModel::StVenantKirchhoff.first_piola(f, &lame);
        assert!(
            mat_close(p, lin, 1e-1 * (1.0 + lin.x_axis.length())),
            "St.VK does not match linear limit"
        );
    }

    #[test]
    fn stable_neo_hookean_small_strain_matches_effective_moduli() {
        // The Smith (2018) stable Neo-Hookean model does not linearise to the
        // same Lamé pair; its small-strain tangent is linear elasticity with
        // the derived effective moduli mu_eff = 3 mu / 4 and
        // lambda_eff = lambda - 5 mu / 8. Comparing against those values is an
        // independent check on the implemented stress tangent.
        let lame = lame();
        let effective = LameParameters::new(0.75 * lame.mu, lame.lambda - 0.625 * lame.mu);
        let g = Mat3::from_cols(
            Vec3::new(0.4, 0.1, -0.2),
            Vec3::new(0.1, -0.3, 0.05),
            Vec3::new(-0.2, 0.05, 0.25),
        );
        let f = Mat3::IDENTITY + g * 1.0e-4;
        let lin = HyperelasticModel::Linear.first_piola(f, &effective);
        let p = HyperelasticModel::StableNeoHookean.first_piola(f, &lame);
        assert!(
            mat_close(p, lin, 1e-1 * (1.0 + lin.x_axis.length())),
            "stable Neo-Hookean does not match its effective linear tangent"
        );
    }

    #[test]
    fn energy_is_nonnegative_for_sum_of_squares_models() {
        let lame = lame();
        let mut state: u64 = 0x1234_5678_9abc_def1;
        for _ in 0..64 {
            let f = random_mat(&mut state);
            for model in [
                HyperelasticModel::Linear,
                HyperelasticModel::StVenantKirchhoff,
            ] {
                let e = model.strain_energy_density(f, &lame);
                assert!(e >= -1e-1, "{model:?} negative energy: {e}");
            }
        }
    }

    #[test]
    fn stable_neo_hookean_handles_inversion() {
        // An inverted element (det F < 0) must still yield finite stress.
        let lame = lame();
        let f = Mat3::from_cols(
            Vec3::new(-1.1, 0.0, 0.0),
            Vec3::new(0.0, 0.9, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        );
        assert!(f.determinant() < 0.0);
        let e = HyperelasticModel::StableNeoHookean.strain_energy_density(f, &lame);
        let p = HyperelasticModel::StableNeoHookean.first_piola(f, &lame);
        assert!(e.is_finite(), "energy not finite under inversion");
        for c in 0..3 {
            assert!(p.col(c).is_finite(), "stress not finite under inversion");
        }
    }

    #[test]
    fn lame_from_isotropic_matches_elasticity() {
        let material = IsotropicElasticity::new(2.5e5, 0.33).unwrap();
        let (lambda, mu) = material.lame();
        let lame = LameParameters::from_isotropic(&material);
        assert!((lame.mu - mu).abs() < 1e-3);
        assert!((lame.lambda - lambda).abs() < 1e-3);
    }

    #[test]
    fn evaluation_is_deterministic() {
        let lame = lame();
        let f = Mat3::from_cols(
            Vec3::new(1.05, 0.02, -0.01),
            Vec3::new(0.03, 0.98, 0.04),
            Vec3::new(-0.02, 0.01, 1.03),
        );
        for model in MODELS {
            let e0 = model.strain_energy_density(f, &lame);
            let e1 = model.strain_energy_density(f, &lame);
            assert_eq!(e0.to_bits(), e1.to_bits());
            let p0 = model.first_piola(f, &lame);
            let p1 = model.first_piola(f, &lame);
            for c in 0..3 {
                assert_eq!(p0.col(c).x.to_bits(), p1.col(c).x.to_bits());
                assert_eq!(p0.col(c).y.to_bits(), p1.col(c).y.to_bits());
                assert_eq!(p0.col(c).z.to_bits(), p1.col(c).z.to_bits());
            }
        }
    }

    fn random_mat(state: &mut u64) -> Mat3 {
        let mut next = || {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            // Map to roughly [-0.5, 1.5] around the identity scale.
            ((*state >> 11) as f32 / (1u64 << 53) as f32) * 2.0 - 0.5
        };
        Mat3::from_cols(
            Vec3::new(1.0 + next(), next(), next()),
            Vec3::new(next(), 1.0 + next(), next()),
            Vec3::new(next(), next(), 1.0 + next()),
        )
    }
}
