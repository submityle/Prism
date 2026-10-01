//! Cloth / fabric BRDF CPU golden references.
//!
//! Deterministic, GPU-free evaluation of fabric sheen lobes.  This module
//! evaluates the actual BRDF lobes (velvet / microfiber) and cloth diffuse,
//! and is distinct from [`crate::gi::env_brdf::sheen_clearcoat`], which only
//! bakes the split-sum sheen DFG lookup used for energy compensation.
//!
//! * [`charlie`] — Estevez-Kulla "Charlie" inverted-Gaussian sheen NDF plus
//!   the Ashikhmin "no-closed-form" / Neubelt visibility term.
//! * [`velvet`] — Ashikhmin-Shirley velvet (inverted) distribution lobe.
//! * [`fabric`] — combined cloth BRDF: sheen lobe + subsurface-tinted diffuse.
//!
//! # Conventions
//! * Every item is a deterministic pure function (no RNG / I/O / GPU / globals /
//!   `unsafe`); transcendental maths goes through [`bevy_math::ops`].
//! * Lobe helpers take cosines against the shading normal and are therefore
//!   frame-independent; the vector-domain helpers assume a local `+Z` normal.
//! * Defensive clamping everywhere: back-facing and degenerate inputs return
//!   zero and no routine ever yields `NaN` or infinity.

pub mod charlie;
pub mod fabric;
pub mod velvet;

pub use charlie::{
    d_charlie, lambda_sheen, sheen_lobe_charlie, sheen_lobe_neubelt, v_charlie, v_neubelt,
};
pub use fabric::{
    cloth_brdf, cloth_diffuse_wrap, cloth_shade, wrap_energy_norm, ClothParams, SheenModel,
};
pub use velvet::{grazing_rim, velvet_brdf, velvet_lobe, velvet_ndf, velvet_visibility};

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::Vec3;

    /// The combined cloth BRDF reduces to the exported sheen lobes: with the
    /// diffuse disabled, `cloth_brdf` equals `sheen_color · D·V` for the chosen
    /// model, tying the `fabric` composition to the `charlie` / `velvet` lobes.
    #[test]
    fn cloth_brdf_matches_sheen_lobes() {
        let n = Vec3::Z;
        let wi = Vec3::new(0.3, 0.1, 0.948).normalize();
        let wo = Vec3::new(-0.2, 0.2, 0.959).normalize();
        let h = (wi + wo).normalize();
        let (mu_v, mu_l, mu_h) = (n.dot(wo), n.dot(wi), n.dot(h));
        let r = 0.5;

        let mut params = ClothParams {
            diffuse_color: Vec3::ZERO,
            subsurface_color: Vec3::ZERO,
            subsurface_blend: 0.0,
            sheen_color: Vec3::ONE,
            sheen_roughness: r,
            wrap: 0.0,
            model: SheenModel::Charlie,
        };

        let charlie_val = cloth_brdf(wi, wo, n, &params);
        let expect_charlie = sheen_lobe_neubelt(mu_v, mu_l, mu_h, r);
        assert!(
            (charlie_val.x - expect_charlie).abs() < 1.0e-6,
            "charlie {charlie_val:?} vs {expect_charlie}"
        );

        params.model = SheenModel::Velvet;
        let velvet_val = cloth_brdf(wi, wo, n, &params);
        let expect_velvet = velvet_lobe(mu_v, mu_l, mu_h, r);
        assert!(
            (velvet_val.x - expect_velvet).abs() < 1.0e-6,
            "velvet {velvet_val:?} vs {expect_velvet}"
        );
    }

    /// Both sheen distributions keep the physical trait that drives this module:
    /// a brighter response toward grazing half angles than toward the normal.
    #[test]
    fn both_ndfs_brighten_at_grazing() {
        let r = 0.5;
        assert!(d_charlie(r, 0.03) > d_charlie(r, 0.97));
        assert!(velvet_ndf(0.03, r) > velvet_ndf(0.97, r));
    }

    /// The shared Neubelt visibility re-export is finite, non-negative, and
    /// consistent between the `charlie` origin and the `velvet` wrapper.
    #[test]
    fn shared_visibility_is_consistent() {
        for &(a, b) in &[(0.2f32, 0.9f32), (0.5, 0.5), (0.05, 0.3)] {
            let via_charlie = v_neubelt(a, b);
            let via_velvet = velvet_visibility(a, b);
            assert!((via_charlie - via_velvet).abs() < 1.0e-7);
            assert!(via_charlie.is_finite() && via_charlie >= 0.0);
        }
    }

    /// End-to-end shading stays finite, non-negative, and energy-sane across the
    /// full parameter surface and all three sheen models.
    #[test]
    fn shade_is_well_behaved_across_params() {
        let n = Vec3::Z;
        let wo = Vec3::new(0.25, 0.0, 0.968).normalize();
        for model in [SheenModel::Charlie, SheenModel::CharlieSoft, SheenModel::Velvet] {
            for &wrap in &[0.0f32, 0.5, 1.0] {
                for &rough in &[0.05f32, 0.5, 1.0] {
                    let p = ClothParams {
                        diffuse_color: Vec3::new(0.7, 0.5, 0.4),
                        subsurface_color: Vec3::new(0.9, 0.3, 0.2),
                        subsurface_blend: 0.5,
                        sheen_color: Vec3::splat(0.4),
                        sheen_roughness: rough,
                        wrap,
                        model,
                    };
                    for i in 0..16 {
                        let a = i as f32 / 16.0 * core::f32::consts::PI;
                        let (s, c) = bevy_math::ops::sin_cos(a);
                        let wi = Vec3::new(s, 0.0, c);
                        let out = cloth_shade(wi, wo, n, &p);
                        assert!(
                            out.is_finite() && out.min_element() >= 0.0,
                            "{model:?} wrap={wrap} rough={rough} out={out:?}"
                        );
                        let fr = cloth_brdf(wi, wo, n, &p);
                        assert!(fr.is_finite() && fr.min_element() >= 0.0);
                    }
                }
            }
        }
    }
}
