//! Fixed-corotated elasticity and snow plasticity for MPM.
//!
//! The stress model is the fixed-corotated hyperelastic energy of Stomakhin et
//! al. 2013. Its first Piola–Kirchhoff stress is
//! `P = 2μ(F − R) + λ(J − 1) J F⁻ᵀ`, where `R` is the polar-rotation of `F`,
//! `J = det F`, and `μ, λ` are the Lamé parameters (optionally hardened by the
//! plastic determinant). The MLS-MPM force term needs `P Fᵀ`, which simplifies
//! to `2μ(F − R)Fᵀ + λ(J − 1) J I` and is what [`corotated_pf`] returns.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! fixed-corotated energy and the snow return-mapping are from Stomakhin et
//! al. 2013.

use glam::{Mat3, Vec3};

use crate::math::scalar::Real;

use super::config::SnowPlasticity;
use super::expf::exp_stable;
use super::svd::{polar_rotation, svd3};

/// Returns `P Fᵀ` for the fixed-corotated model — the quantity needed by the
/// MLS-MPM momentum scatter.
///
/// `P Fᵀ = 2μ(F − R)Fᵀ + λ(J − 1) J I`, where `R` is the polar rotation of
/// `F` and `J = det F`.
#[must_use]
pub fn corotated_pf(f: Mat3, mu: Real, lambda: Real) -> Mat3 {
    let r = polar_rotation(f);
    let j = f.determinant();
    let term_shear = (f - r) * f.transpose() * (2.0 * mu);
    let term_vol = lambda * (j - 1.0) * j;
    term_shear + Mat3::from_diagonal(Vec3::splat(term_vol))
}

/// Returns the full first Piola–Kirchhoff stress `P` for the fixed-corotated
/// model: `P = 2μ(F − R) + λ(J − 1) · cof(F)`, where `cof(F) = J F⁻ᵀ` is the
/// cofactor matrix (computed from column cross products, so it is robust even
/// when `F` is singular).
#[must_use]
pub fn corotated_piola(f: Mat3, mu: Real, lambda: Real) -> Mat3 {
    let r = polar_rotation(f);
    let j = f.determinant();
    let cof = cofactor(f);
    (f - r) * (2.0 * mu) + cof * (lambda * (j - 1.0))
}

/// Returns the cofactor matrix `cof(F) = J F⁻ᵀ`, whose columns are the cross
/// products of the columns of `F`.
#[must_use]
pub fn cofactor(f: Mat3) -> Mat3 {
    let c0 = f.y_axis.cross(f.z_axis);
    let c1 = f.z_axis.cross(f.x_axis);
    let c2 = f.x_axis.cross(f.y_axis);
    Mat3::from_cols(c0, c1, c2)
}

/// The hardening multiplier `exp(ξ(1 − Jp))` applied to the Lamé parameters.
///
/// A hardening coefficient of zero returns exactly `1.0` (no hardening).
#[must_use]
pub fn hardening_factor(hardening: Real, plastic_det: Real) -> Real {
    if hardening == 0.0 {
        return 1.0;
    }
    exp_stable(hardening * (1.0 - plastic_det))
}

/// The result of the snow plastic return-mapping applied to a trial
/// deformation gradient.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlasticUpdate {
    /// The corrected (elastic) deformation gradient after clamping.
    pub deformation: Mat3,
    /// The updated plastic determinant `Jp`.
    pub plastic_det: Real,
}

/// Applies the snow return-mapping to a trial deformation gradient.
///
/// The singular values of `f_trial` are clamped into
/// `[1 − θ_c, 1 + θ_s]`; the clamped-off volumetric part is accumulated into
/// the plastic determinant `Jp` so that total volume is conserved. Returns the
/// corrected elastic deformation gradient and the new `Jp`.
#[must_use]
pub fn snow_return_mapping(f_trial: Mat3, prev_jp: Real, params: &SnowPlasticity) -> PlasticUpdate {
    let svd = svd3(f_trial);
    let lo = 1.0 - params.critical_compression;
    let hi = 1.0 + params.critical_stretch;
    let clamped = Vec3::new(
        svd.sigma.x.clamp(lo, hi),
        svd.sigma.y.clamp(lo, hi),
        svd.sigma.z.clamp(lo, hi),
    );
    // Elastic part uses the clamped singular values.
    let sig_elastic = Mat3::from_diagonal(clamped);
    let f_elastic = svd.u * sig_elastic * svd.v.transpose();
    // Track total J and the elastic J to fold the difference into Jp.
    let j_total = svd.sigma.x * svd.sigma.y * svd.sigma.z;
    let j_elastic = clamped.x * clamped.y * clamped.z;
    // Jp_new * Je = J_total  =>  Jp_new = prev_jp * (J_total / J_elastic).
    let jp_new = if j_elastic.abs() > 1.0e-12 {
        prev_jp * j_total / j_elastic
    } else {
        prev_jp
    };
    PlasticUpdate {
        deformation: f_elastic,
        plastic_det: jp_new,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cofactor_matches_j_finv_transpose() {
        let f = Mat3::from_cols(
            Vec3::new(1.2, 0.1, 0.0),
            Vec3::new(0.0, 0.9, 0.05),
            Vec3::new(0.1, 0.0, 1.1),
        );
        let j = f.determinant();
        let expected = f.inverse().transpose() * j;
        let cof = cofactor(f);
        assert!((cof.x_axis - expected.x_axis).length() < 1.0e-5);
        assert!((cof.y_axis - expected.y_axis).length() < 1.0e-5);
        assert!((cof.z_axis - expected.z_axis).length() < 1.0e-5);
    }

    #[test]
    fn stress_is_zero_at_identity() {
        let pf = corotated_pf(Mat3::IDENTITY, 100.0, 50.0);
        assert!(pf.x_axis.length() < 1.0e-5);
        assert!(pf.y_axis.length() < 1.0e-5);
        assert!(pf.z_axis.length() < 1.0e-5);
    }

    #[test]
    fn pf_matches_p_times_ft() {
        let f = Mat3::from_cols(
            Vec3::new(1.1, 0.05, 0.0),
            Vec3::new(-0.03, 1.02, 0.01),
            Vec3::new(0.0, 0.0, 0.97),
        );
        let mu = 80.0;
        let lambda = 120.0;
        let pf = corotated_pf(f, mu, lambda);
        let p = corotated_piola(f, mu, lambda);
        let pft = p * f.transpose();
        assert!((pf.x_axis - pft.x_axis).length() < 1.0e-3);
        assert!((pf.y_axis - pft.y_axis).length() < 1.0e-3);
        assert!((pf.z_axis - pft.z_axis).length() < 1.0e-3);
    }

    #[test]
    fn hardening_is_one_when_disabled() {
        assert_eq!(hardening_factor(0.0, 0.5), 1.0);
        assert!(
            hardening_factor(10.0, 1.0) == 1.0
                || (hardening_factor(10.0, 1.0) - 1.0).abs() < 1.0e-5
        );
        // Compaction (Jp < 1) stiffens the material.
        assert!(hardening_factor(10.0, 0.8) > 1.0);
    }

    #[test]
    fn return_mapping_clamps_singular_values() {
        let params = SnowPlasticity::new(0.1, 0.1, 0.0);
        // Strong compression along x.
        let f = Mat3::from_cols(Vec3::new(0.5, 0.0, 0.0), Vec3::Y, Vec3::Z);
        let out = snow_return_mapping(f, 1.0, &params);
        let svd = svd3(out.deformation);
        let lo = 0.9;
        assert!(svd.sigma.x >= lo - 1.0e-4 && svd.sigma.y >= lo - 1.0e-4);
        // Compression pushed material into plasticity: Jp shrinks below 1.
        assert!(out.plastic_det < 1.0);
    }
}
