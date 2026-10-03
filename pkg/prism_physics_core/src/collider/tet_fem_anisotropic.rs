//! Transversely-isotropic and orthotropic fiber-reinforced hyperelastic
//! constitutive terms for tetrahedral `FEM` elements.
//!
//! Many biological and manufactured soft materials are *not* isotropic: muscle,
//! tendon, ligament, and woven or knitted cloth are far stiffer along embedded
//! fiber directions than across them. This module augments the isotropic
//! [`HyperelasticModel`] base with additive fiber terms, yielding a
//! *transversely isotropic* material (one preferred fiber family) or an
//! *orthotropic* material (two or more fiber families, e.g. the warp and weft
//! of a woven fabric).
//!
//! # Kinematics
//!
//! A fiber family is described by a unit direction `a0` in the undeformed
//! (material) configuration. Under a deformation gradient `F` the fiber vector
//! becomes `fa = F a0`, and the fourth pseudo-invariant
//!
//! ```text
//! I4 = a0 . (C a0) = (F a0) . (F a0) = |fa|^2
//! ```
//!
//! is the square of the fiber stretch `lambda_f = sqrt(I4)`, where
//! `C = Fᵀ F` is the right Cauchy-Green tensor. `I4 = 1` means the fiber keeps
//! its rest length; `I4 > 1` is tension and `I4 < 1` is compression along the
//! fiber.
//!
//! # Energy and stress
//!
//! Each family adds the quadratic reinforcement energy
//!
//! ```text
//! Psi_f = (c / 2) (I4 - 1)^2 ,
//! ```
//!
//! with fiber modulus `c >= 0`. Its first Piola-Kirchhoff stress follows from
//! `dI4/dF = 2 (F a0) ⊗ a0`:
//!
//! ```text
//! P_f = dPsi_f/dF = 2 c (I4 - 1) (F a0) ⊗ a0 .
//! ```
//!
//! Fibers can respond [`FiberResponse::Bidirectional`] (resisting both tension
//! and compression) or [`FiberResponse::TensionOnly`] (the physically common
//! case where slender fibers buckle under compression and carry no load until
//! `I4 > 1`).
//!
//! These terms are *objective*: `I4` depends only on `C`, so a superposed
//! rigid rotation `F -> R F` leaves both the energy and the stress magnitude
//! unchanged. They are purely elastic additive contributions and compose with
//! any isotropic base model and any time integrator.
//!
//! All arithmetic accumulates in `f64` and rounds back to the engine real type
//! on return, matching [`crate::collider::tet_fem_constitutive`].

use crate::collider::tet_fem_constitutive::{HyperelasticModel, LameParameters};
use glam::{Mat3, Vec3};

/// Smallest squared length a candidate fiber direction may have before it is
/// rejected as degenerate during normalization.
const MIN_DIRECTION_LENGTH_SQ: f32 = 1.0e-12;

/// A unit fiber direction in the undeformed (material) configuration.
///
/// Construction normalizes the supplied vector, so the stored direction is
/// always of unit length and the downstream invariants stay well defined.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FiberDirection {
    dir: Vec3,
}

impl FiberDirection {
    /// Normalizes `direction` into a unit fiber direction.
    ///
    /// Returns [`None`] when `direction` is shorter than
    /// `sqrt(MIN_DIRECTION_LENGTH_SQ)`, i.e. too close to zero to define a
    /// meaningful orientation.
    #[must_use]
    pub fn new(direction: Vec3) -> Option<Self> {
        let len_sq = direction.length_squared();
        if !len_sq.is_finite() || len_sq < MIN_DIRECTION_LENGTH_SQ {
            return None;
        }
        Some(Self {
            dir: direction / len_sq.sqrt(),
        })
    }

    /// Returns the stored unit direction.
    #[must_use]
    pub fn get(self) -> Vec3 {
        self.dir
    }
}

/// Selects whether a fiber family carries load in compression.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FiberResponse {
    /// The fiber resists both tension (`I4 > 1`) and compression (`I4 < 1`).
    ///
    /// Appropriate for materials where the reinforcement is bonded and cannot
    /// buckle, so the quadratic term is active for every state.
    Bidirectional,
    /// The fiber resists tension only; it carries no load while `I4 <= 1`.
    ///
    /// Models slender fibers (muscle, tendon, yarn) that buckle under
    /// compression. The energy and stress are clamped to zero when the fiber
    /// is at or below its rest length.
    TensionOnly,
}

impl FiberResponse {
    /// Returns `true` when the family is active at the given invariant `i4`.
    #[must_use]
    fn is_active(self, i4: f64) -> bool {
        match self {
            Self::Bidirectional => true,
            Self::TensionOnly => i4 > 1.0,
        }
    }
}

/// A single reinforcing fiber family: a direction, a modulus, and a response.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FiberFamily {
    /// Unit rest-space fiber direction.
    pub direction: FiberDirection,
    /// Fiber modulus `c >= 0` scaling the quadratic reinforcement energy.
    pub modulus: f32,
    /// Whether the family carries load in compression.
    pub response: FiberResponse,
}

impl FiberFamily {
    /// Builds a fiber family, normalizing `direction` and validating `modulus`.
    ///
    /// Returns [`None`] when `direction` is degenerate (see
    /// [`FiberDirection::new`]) or when `modulus` is negative or non-finite.
    #[must_use]
    pub fn new(direction: Vec3, modulus: f32, response: FiberResponse) -> Option<Self> {
        if !modulus.is_finite() || modulus < 0.0 {
            return None;
        }
        let direction = FiberDirection::new(direction)?;
        Some(Self {
            direction,
            modulus,
            response,
        })
    }

    /// Returns the fourth pseudo-invariant `I4 = |F a0|^2`.
    #[must_use]
    pub fn invariant_i4(self, f: Mat3) -> f32 {
        let fa = fiber_vector(f, self.direction.get());
        (fa[0] * fa[0] + fa[1] * fa[1] + fa[2] * fa[2]) as f32
    }

    /// Returns the fiber stretch `lambda_f = sqrt(I4)`.
    #[must_use]
    pub fn fiber_stretch(self, f: Mat3) -> f32 {
        let fa = fiber_vector(f, self.direction.get());
        (fa[0] * fa[0] + fa[1] * fa[1] + fa[2] * fa[2]).sqrt() as f32
    }

    /// Evaluates the fiber strain energy density `Psi_f(F)`.
    #[must_use]
    pub fn strain_energy_density(self, f: Mat3) -> f32 {
        let c = f64::from(self.modulus);
        let fa = fiber_vector(f, self.direction.get());
        let i4 = fa[0] * fa[0] + fa[1] * fa[1] + fa[2] * fa[2];
        if !self.response.is_active(i4) {
            return 0.0;
        }
        let d = i4 - 1.0;
        (0.5 * c * d * d) as f32
    }

    /// Evaluates the fiber first Piola-Kirchhoff stress
    /// `P_f = 2 c (I4 - 1) (F a0) ⊗ a0`.
    #[must_use]
    pub fn first_piola(self, f: Mat3) -> Mat3 {
        let c = f64::from(self.modulus);
        let a0 = self.direction.get();
        let fa = fiber_vector(f, a0);
        let i4 = fa[0] * fa[0] + fa[1] * fa[1] + fa[2] * fa[2];
        if !self.response.is_active(i4) {
            return Mat3::ZERO;
        }
        let coeff = 2.0 * c * (i4 - 1.0);
        outer_product(coeff, fa, a0)
    }
}

/// A transversely-isotropic material: an isotropic base plus one fiber family.
///
/// This is the canonical single-fiber reinforcement (e.g. a muscle belly with
/// one dominant fiber direction). Energy and stress are the additive sum of the
/// isotropic base term and the fiber term.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransverselyIsotropicMaterial {
    /// Isotropic ground-matrix model.
    pub base: HyperelasticModel,
    /// Lamé parameters for the isotropic base.
    pub lame: LameParameters,
    /// The reinforcing fiber family.
    pub fiber: FiberFamily,
}

impl TransverselyIsotropicMaterial {
    /// Creates a transversely-isotropic material from its parts.
    #[must_use]
    pub fn new(base: HyperelasticModel, lame: LameParameters, fiber: FiberFamily) -> Self {
        Self { base, lame, fiber }
    }

    /// Evaluates the total strain energy density `Psi_iso + Psi_f`.
    #[must_use]
    pub fn strain_energy_density(self, f: Mat3) -> f32 {
        let iso = f64::from(self.base.strain_energy_density(f, &self.lame));
        let fiber = f64::from(self.fiber.strain_energy_density(f));
        (iso + fiber) as f32
    }

    /// Evaluates the total first Piola-Kirchhoff stress `P_iso + P_f`.
    #[must_use]
    pub fn first_piola(self, f: Mat3) -> Mat3 {
        self.base.first_piola(f, &self.lame) + self.fiber.first_piola(f)
    }
}

/// Sums the fiber strain energy density over an orthotropic set of families.
///
/// Each family contributes independently and additively, so two orthogonal
/// families reproduce the warp/weft reinforcement of a woven fabric. An empty
/// slice yields zero.
#[must_use]
pub fn orthotropic_fiber_energy(f: Mat3, families: &[FiberFamily]) -> f32 {
    let mut acc = 0.0_f64;
    for family in families {
        acc += f64::from(family.strain_energy_density(f));
    }
    acc as f32
}

/// Sums the fiber first Piola-Kirchhoff stress over an orthotropic set of
/// families. An empty slice yields the zero matrix.
#[must_use]
pub fn orthotropic_fiber_first_piola(f: Mat3, families: &[FiberFamily]) -> Mat3 {
    let mut acc = Mat3::ZERO;
    for family in families {
        acc += family.first_piola(f);
    }
    acc
}

/// Computes the spatial fiber vector `fa = F a0` with `f64` accumulation.
///
/// glam stores `Mat3` in column-major order, so `F a0` is the linear
/// combination `a0.x * col0 + a0.y * col1 + a0.z * col2` of the columns.
fn fiber_vector(f: Mat3, a0: Vec3) -> [f64; 3] {
    let a = [f64::from(a0.x), f64::from(a0.y), f64::from(a0.z)];
    let c0 = f.x_axis;
    let c1 = f.y_axis;
    let c2 = f.z_axis;
    [
        f64::from(c0.x) * a[0] + f64::from(c1.x) * a[1] + f64::from(c2.x) * a[2],
        f64::from(c0.y) * a[0] + f64::from(c1.y) * a[1] + f64::from(c2.y) * a[2],
        f64::from(c0.z) * a[0] + f64::from(c1.z) * a[1] + f64::from(c2.z) * a[2],
    ]
}

/// Builds the scaled outer product `coeff * (fa ⊗ a0)` as a `Mat3`.
///
/// Element `(row, col)` is `coeff * fa[row] * a0[col]`; the result is assembled
/// column by column to honor glam's column-major layout.
fn outer_product(coeff: f64, fa: [f64; 3], a0: Vec3) -> Mat3 {
    let a = [f64::from(a0.x), f64::from(a0.y), f64::from(a0.z)];
    let col = |j: usize| {
        Vec3::new(
            (coeff * fa[0] * a[j]) as f32,
            (coeff * fa[1] * a[j]) as f32,
            (coeff * fa[2] * a[j]) as f32,
        )
    };
    Mat3::from_cols(col(0), col(1), col(2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::tet_fem_stiffness::IsotropicElasticity;

    fn lame() -> LameParameters {
        let material = IsotropicElasticity::new(1.0e6, 0.4).unwrap();
        LameParameters::from_isotropic(&material)
    }

    fn mat_close(a: Mat3, b: Mat3, tol: f32) -> bool {
        (0..3).all(|c| (a.col(c) - b.col(c)).length() <= tol)
    }

    fn set_elem(f: Mat3, row: usize, col: usize, delta: f32) -> Mat3 {
        let mut cols = [f.x_axis, f.y_axis, f.z_axis];
        let v = cols[col];
        let mut comp = [v.x, v.y, v.z];
        comp[row] += delta;
        cols[col] = Vec3::new(comp[0], comp[1], comp[2]);
        Mat3::from_cols(cols[0], cols[1], cols[2])
    }

    fn diag(sx: f32, sy: f32, sz: f32) -> Mat3 {
        Mat3::from_cols(
            Vec3::new(sx, 0.0, 0.0),
            Vec3::new(0.0, sy, 0.0),
            Vec3::new(0.0, 0.0, sz),
        )
    }

    #[test]
    fn direction_normalizes_and_rejects_degenerate() {
        let d = FiberDirection::new(Vec3::new(0.0, 3.0, 0.0)).unwrap();
        assert!((d.get() - Vec3::Y).length() < 1.0e-6);
        assert!(FiberDirection::new(Vec3::ZERO).is_none());
        assert!(FiberDirection::new(Vec3::new(1.0e-7, 0.0, 0.0)).is_none());
    }

    #[test]
    fn family_rejects_bad_modulus() {
        assert!(FiberFamily::new(Vec3::X, -1.0, FiberResponse::Bidirectional).is_none());
        assert!(FiberFamily::new(Vec3::X, f32::NAN, FiberResponse::Bidirectional).is_none());
        assert!(FiberFamily::new(Vec3::X, 10.0, FiberResponse::Bidirectional).is_some());
    }

    #[test]
    fn undeformed_state_is_force_free() {
        for response in [FiberResponse::Bidirectional, FiberResponse::TensionOnly] {
            let fam = FiberFamily::new(Vec3::X, 5.0e5, response).unwrap();
            assert!((fam.invariant_i4(Mat3::IDENTITY) - 1.0).abs() < 1.0e-5);
            assert!(fam.strain_energy_density(Mat3::IDENTITY).abs() < 1.0e-3);
            assert!(mat_close(
                fam.first_piola(Mat3::IDENTITY),
                Mat3::ZERO,
                1.0e-2
            ));
        }
    }

    #[test]
    fn stretch_along_fiber_matches_closed_form() {
        let c = 4.0e5_f32;
        let fam = FiberFamily::new(Vec3::X, c, FiberResponse::Bidirectional).unwrap();
        let s = 1.2_f32;
        let f = diag(s, 1.0, 1.0);
        let i4 = s * s;
        assert!((fam.invariant_i4(f) - i4).abs() < 1.0e-2);
        assert!((fam.fiber_stretch(f) - s).abs() < 1.0e-5);
        let expected = 0.5 * c * (i4 - 1.0) * (i4 - 1.0);
        assert!((fam.strain_energy_density(f) - expected).abs() <= expected * 1.0e-4 + 1.0);
    }

    #[test]
    fn transverse_deformation_leaves_fiber_inert() {
        // Fiber along X; stretch only Y and Z. The fiber length is unchanged,
        // so I4 stays 1 and the fiber term vanishes.
        let fam = FiberFamily::new(Vec3::X, 7.0e5, FiberResponse::Bidirectional).unwrap();
        let f = diag(1.0, 1.3, 0.8);
        assert!((fam.invariant_i4(f) - 1.0).abs() < 1.0e-5);
        assert!(fam.strain_energy_density(f).abs() < 1.0e-2);
        assert!(mat_close(fam.first_piola(f), Mat3::ZERO, 1.0e-1));
    }

    #[test]
    fn tension_only_is_inactive_under_compression() {
        let fam = FiberFamily::new(Vec3::X, 5.0e5, FiberResponse::TensionOnly).unwrap();
        let compressed = diag(0.8, 1.0, 1.0);
        assert_eq!(fam.strain_energy_density(compressed), 0.0);
        assert!(mat_close(fam.first_piola(compressed), Mat3::ZERO, 1.0e-6));
        let stretched = diag(1.2, 1.0, 1.0);
        assert!(fam.strain_energy_density(stretched) > 0.0);
        assert!(fam.first_piola(stretched).col(0).length() > 0.0);
    }

    #[test]
    fn bidirectional_is_active_under_compression() {
        let fam = FiberFamily::new(Vec3::X, 5.0e5, FiberResponse::Bidirectional).unwrap();
        let compressed = diag(0.8, 1.0, 1.0);
        assert!(fam.strain_energy_density(compressed) > 0.0);
        assert!(fam.first_piola(compressed).col(0).length() > 0.0);
    }

    #[test]
    fn first_piola_matches_energy_gradient() {
        // Central finite differences of Psi_f must recover P_f = dPsi/dF.
        let fam = FiberFamily::new(
            Vec3::new(1.0, 2.0, -1.0),
            3.0e4,
            FiberResponse::Bidirectional,
        )
        .unwrap();
        let f = Mat3::from_cols(
            Vec3::new(1.1, 0.05, -0.03),
            Vec3::new(0.02, 0.95, 0.04),
            Vec3::new(-0.01, 0.03, 1.08),
        );
        let analytic = fam.first_piola(f);
        let h = 1.0e-3_f32;
        for row in 0..3 {
            for col in 0..3 {
                let plus = fam.strain_energy_density(set_elem(f, row, col, h));
                let minus = fam.strain_energy_density(set_elem(f, row, col, -h));
                let fd = (plus - minus) / (2.0 * h);
                let a = analytic.col(col)[row];
                assert!(
                    (fd - a).abs() <= a.abs() * 2.0e-2 + 2.0,
                    "row {row} col {col}: fd={fd} analytic={a}"
                );
            }
        }
    }

    #[test]
    fn fiber_energy_is_rotation_invariant() {
        // A superposed rotation F -> R F must not change I4 or the energy.
        let fam = FiberFamily::new(Vec3::X, 6.0e5, FiberResponse::Bidirectional).unwrap();
        let f = diag(1.25, 0.9, 1.05);
        let r = Mat3::from_rotation_y(0.7) * Mat3::from_rotation_z(-0.4);
        let rotated = r * f;
        assert!((fam.invariant_i4(f) - fam.invariant_i4(rotated)).abs() < 1.0e-1);
        let e0 = fam.strain_energy_density(f);
        let e1 = fam.strain_energy_density(rotated);
        assert!((e0 - e1).abs() <= e0.abs() * 1.0e-3 + 1.0);
    }

    #[test]
    fn transversely_isotropic_is_additive() {
        let fiber = FiberFamily::new(Vec3::X, 2.0e5, FiberResponse::Bidirectional).unwrap();
        let mat =
            TransverselyIsotropicMaterial::new(HyperelasticModel::StableNeoHookean, lame(), fiber);
        let f = diag(1.15, 0.95, 1.02);
        let iso = HyperelasticModel::StableNeoHookean.strain_energy_density(f, &lame());
        let expected = iso + fiber.strain_energy_density(f);
        assert!((mat.strain_energy_density(f) - expected).abs() <= expected.abs() * 1.0e-4 + 1.0);
        let p_expected =
            HyperelasticModel::StableNeoHookean.first_piola(f, &lame()) + fiber.first_piola(f);
        assert!(mat_close(mat.first_piola(f), p_expected, 1.0e-1));
    }

    #[test]
    fn orthotropic_sum_matches_manual_sum() {
        let warp = FiberFamily::new(Vec3::X, 3.0e5, FiberResponse::TensionOnly).unwrap();
        let weft = FiberFamily::new(Vec3::Z, 2.0e5, FiberResponse::TensionOnly).unwrap();
        let families = [warp, weft];
        let f = diag(1.2, 0.9, 1.1);
        let e = orthotropic_fiber_energy(f, &families);
        assert!(
            (e - (warp.strain_energy_density(f) + weft.strain_energy_density(f))).abs() < 1.0e-1
        );
        let p = orthotropic_fiber_first_piola(f, &families);
        assert!(mat_close(
            p,
            warp.first_piola(f) + weft.first_piola(f),
            1.0e-3
        ));
        assert_eq!(orthotropic_fiber_energy(f, &[]), 0.0);
        assert!(mat_close(
            orthotropic_fiber_first_piola(f, &[]),
            Mat3::ZERO,
            1.0e-9
        ));
    }
}
