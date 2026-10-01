//! The water-surface pass's **specular ambient-occlusion** remap: the `CPU`
//! twin that locks the numeric contract of the `water_spec_occlusion` helper
//! added to `water_surface_raster.wesl`.
//!
//! ## Why the water surface needs a specular occlusion remap
//!
//! The scalar ambient occlusion ([`super::surface_gtao`]'s horizon search, read
//! back as `SurfaceSample::ambient_occlusion`) measures how much of the *full*
//! cosine hemisphere is visible — it is the correct weight for the diffuse
//! response. Feeding that same scalar straight into the **specular** weight (as
//! the previous `water_ibl` did, `spec_weight = (f0*dfg.x + dfg.y) * ao`) double
//! darkens tight reflections, because a glossy lobe only integrates a narrow
//! cone around the mirror direction, not the whole hemisphere. Lagarde & de
//! Rousiers' *Moving Frostbite to Physically Based Rendering* (SIGGRAPH 2014)
//! give the cheap analytic remap this slice wires into the water indirect
//! specular, recovering a view- and roughness-dependent specular occlusion from
//! the diffuse `AO`:
//!
//! ```text
//! SO = saturate(pow(NoV + ao, exp2(-16 * roughness - 1)) - 1 + ao)
//! ```
//!
//! ## Why this file carries no production code
//!
//! The remap lives entirely inside the shader (`water_spec_occlusion`): it reads
//! the view cosine, the already-resolved `GTAO` scalar, and the perceptual
//! roughness that `water_ibl` already has in hand, so it needs **no** new bind
//! group, uniform, or `CPU`-side upload. This module therefore exists only to
//! pin the shader formula against the engine's backend-neutral golden
//! ([`prism_render_shading::gi::occlusion::specular_occlusion`]) so the `WESL`
//! twin cannot silently drift. All of its contents are `#[cfg(test)]`.

#[cfg(test)]
mod tests {
    use bevy_math::ops;
    use prism_render_shading::gi::occlusion::specular_occlusion as golden;

    /// Byte-for-byte re-statement of the shader's `water_spec_occlusion` helper
    /// (same clamps and `exp2`/`pow` algebra), evaluated with the engine's
    /// libm-deterministic [`bevy_math::ops`] so it reproduces the golden
    /// bit-for-bit under identical `f32` arithmetic.
    fn water_spec_occlusion(n_dot_v: f32, ao: f32, roughness: f32) -> f32 {
        let nov = n_dot_v.clamp(0.0, 1.0);
        let a = ao.clamp(0.0, 1.0);
        let r = roughness.clamp(0.0, 1.0);
        let exponent = ops::exp2(-16.0 * r - 1.0);
        let base = (nov + a).max(0.0);
        (ops::powf(base, exponent) - 1.0 + a).clamp(0.0, 1.0)
    }

    /// The shader twin must match the engine golden exactly over a dense grid,
    /// so the water specular occlusion stays identical to every other surface's.
    #[test]
    fn twin_matches_shading_golden() {
        for i in 0..=10u32 {
            for j in 0..=10u32 {
                for k in 0..=10u32 {
                    let nov = i as f32 / 10.0;
                    let ao = j as f32 / 10.0;
                    let r = k as f32 / 10.0;
                    let twin = water_spec_occlusion(nov, ao, r);
                    let g = golden(nov, ao, r);
                    assert_eq!(
                        twin.to_bits(),
                        g.to_bits(),
                        "nov {nov} ao {ao} r {r}: twin {twin} golden {g}"
                    );
                }
            }
        }
    }

    /// `ao = 1` (nothing occluded) never darkens the specular, for any view
    /// angle or roughness.
    #[test]
    fn full_visibility_never_darkens() {
        for &nov in &[0.0, 0.1, 0.5, 0.9, 1.0] {
            for &r in &[0.0, 0.25, 0.5, 0.75, 1.0] {
                let so = water_spec_occlusion(nov, 1.0, r);
                assert!((so - 1.0).abs() < 1e-6, "nov {nov} r {r} so {so}");
            }
        }
    }

    /// As roughness grows the lobe widens until the diffuse `AO` is already the
    /// right answer, so `SO -> ao`.
    #[test]
    fn rough_lobe_approaches_ao() {
        for &ao in &[0.0, 0.2, 0.5, 0.8, 1.0] {
            for &nov in &[0.0, 0.3, 0.7, 1.0] {
                let so = water_spec_occlusion(nov, ao, 1.0);
                assert!((so - ao).abs() < 1e-3, "ao {ao} nov {nov} so {so}");
            }
        }
    }

    /// A sharp (low-roughness) lobe is *view dependent*, unlike the raw diffuse
    /// `AO`: specular occlusion rises as the surface turns to face the viewer,
    /// and a head-on sharp lobe recovers at least the raw `AO` (grazing angles
    /// may legitimately crush it further, which the diffuse term never does).
    #[test]
    fn sharp_lobe_is_view_dependent() {
        for &ao in &[0.2, 0.4, 0.6, 0.8] {
            let mut prev = -1.0_f32;
            for i in 0..=20u32 {
                let nov = i as f32 / 20.0;
                let so = water_spec_occlusion(nov, ao, 0.0);
                assert!(so >= prev - 1e-6, "ao {ao} nov {nov} so {so} < prev {prev}");
                prev = so;
            }
            let head_on = water_spec_occlusion(1.0, ao, 0.0);
            assert!(head_on >= ao - 1e-6, "ao {ao} head_on {head_on} < ao");
        }
    }

    /// Monotonically non-decreasing in the input occlusion: more visibility can
    /// never yield less specular.
    #[test]
    fn monotonic_in_ao() {
        for &nov in &[0.0, 0.5, 1.0] {
            for &r in &[0.0, 0.5, 1.0] {
                let mut prev = -1.0_f32;
                for j in 0..=20u32 {
                    let ao = j as f32 / 20.0;
                    let so = water_spec_occlusion(nov, ao, r);
                    assert!(
                        so >= prev - 1e-6,
                        "nov {nov} r {r} ao {ao} so {so} < prev {prev}"
                    );
                    prev = so;
                }
            }
        }
    }

    /// Every result is a finite factor in `[0, 1]`, including the degenerate
    /// zero-occlusion / zero-view corner.
    #[test]
    fn always_bounded_and_finite() {
        for i in 0..=8u32 {
            for j in 0..=8u32 {
                for k in 0..=8u32 {
                    let so = water_spec_occlusion(i as f32 / 8.0, j as f32 / 8.0, k as f32 / 8.0);
                    assert!(so.is_finite(), "non-finite SO at {i} {j} {k}");
                    assert!((0.0..=1.0).contains(&so), "out of range SO {so}");
                }
            }
        }
    }

    /// Out-of-range inputs are defensively clamped, matching the shader.
    #[test]
    fn clamps_out_of_range_inputs() {
        let lo = water_spec_occlusion(-5.0, -5.0, -5.0);
        let hi = water_spec_occlusion(5.0, 5.0, 5.0);
        assert_eq!(lo.to_bits(), water_spec_occlusion(0.0, 0.0, 0.0).to_bits());
        assert_eq!(hi.to_bits(), water_spec_occlusion(1.0, 1.0, 1.0).to_bits());
    }

    /// Deterministic: identical inputs reproduce identical bits.
    #[test]
    fn deterministic() {
        let a = water_spec_occlusion(0.4, 0.3, 0.2);
        let b = water_spec_occlusion(0.4, 0.3, 0.2);
        assert_eq!(a.to_bits(), b.to_bits());
    }
}
