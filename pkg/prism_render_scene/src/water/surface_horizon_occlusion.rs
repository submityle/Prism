//! The water-surface pass's **bent-normal horizon occlusion**: the `CPU` twins
//! that lock the numeric contract of the `water_cone_cone_intersection`,
//! `water_horizon_occlusion`, and bent-normal cone-fit helpers added to
//! `water_surface_raster.wesl`.
//!
//! ## Why the water surface needs a horizon-occluded reflection
//!
//! The image-based reflection sampled in `water_ibl` sees the *whole* upper
//! hemisphere, but a point tucked under a wave crest or against a steep swell
//! only sees a sliver of the sky. Multiplying the environment prefilter by a
//! flat ambient-occlusion scalar is the diffuse answer; the mirror lobe instead
//! needs to know *which directions* are still visible, not just how much of the
//! hemisphere survives. [`super::surface_gtao`]'s horizon march already yields a
//! view-space **bent normal** (the mean unoccluded direction) and a visibility
//! **cone aperture** (recovered from the mean-resultant length via the uniform
//! cone identity `r = (1 + cos aperture) / 2`, i.e. `aperture = acos(2r - 1)`,
//! matching the engine `BentNormalCone` convention in
//! [`prism_render_shading::gi::occlusion`]). The shader models the reflection as
//! a thin cone and intersects it with that visibility cone, so a reflection
//! pointing into an occluded direction fades out smoothly.
//!
//! This is the specular counterpart to the Lagarde & de Rousiers specular
//! occlusion (SIGGRAPH 2014 `Frostbite`) already wired into the pass: the
//! bent-normal cone comes from the same `GTAO` horizon search (`McGuire` et al.),
//! and the soft cone-overlap approximation follows Oat & Sander's *Ambient
//! Aperture Lighting*.
//!
//! ## Why this file carries no production code
//!
//! The reflection occlusion lives entirely inside the shader
//! (`water_horizon_occlusion` -> `water_cone_cone_intersection` ->
//! `water_smoothstep01`): it consumes the bent normal and aperture that
//! `water_gtao` already returns, so it needs **no** new bind group, uniform, or
//! `CPU`-side upload. This module therefore exists only to pin the shader math
//! against the engine's backend-neutral golden
//! ([`prism_render_shading::gi::occlusion::cone_cone_intersection`] and
//! [`prism_render_shading::gi::occlusion::horizon_occlusion`]) so the `WESL`
//! twins cannot silently drift. All of its contents are `#[cfg(test)]`.

#[cfg(test)]
mod tests {
    use bevy_math::{ops, Vec3};
    use core::f32::consts::PI;
    use prism_render_shading::gi::occlusion::{
        cone_cone_intersection as golden_cone, horizon_occlusion as golden_horizon,
    };

    /// SIGGRAPH 2014 `Frostbite` specular-rim fraction (shader literal `0.5`).
    const SPECULAR_RIM_FRACTION: f32 = 0.5;
    /// Minimum rim / closed-cone epsilon (shader literal `0.02`).
    const MIN_SPECULAR_RIM: f32 = 0.02;

    /// Byte-for-byte re-statement of the shader's `water_smoothstep01` helper:
    /// the single-parameter Hermite ramp, evaluated with the engine's
    /// libm-deterministic [`bevy_math::ops`] arithmetic.
    fn water_smoothstep01(t: f32) -> f32 {
        let x = t.clamp(0.0, 1.0);
        x * x * (3.0 - 2.0 * x)
    }

    /// Byte-for-byte re-statement of the shader's `water_cone_cone_intersection`
    /// helper (same clamps, `1.0 / sqrt` normalisation with +Y fallback, and
    /// cap-ratio containment), so it reproduces the golden bit-for-bit under
    /// identical `f32` arithmetic.
    fn water_cone_cone_intersection(
        dir_a: Vec3,
        aperture_a: f32,
        dir_b: Vec3,
        aperture_b: f32,
    ) -> f32 {
        let a = aperture_a.clamp(0.0, PI);
        let b = aperture_b.clamp(0.0, PI);

        let da = normalize_or_y(dir_a);
        let db = normalize_or_y(dir_b);

        let cos_angle = da.dot(db).clamp(-1.0, 1.0);
        let angle = ops::acos(cos_angle);

        let inner = (a - b).abs();
        let outer = a + b;

        let cap_a = (1.0 - ops::cos(a)).max(0.0);
        let cap_b = (1.0 - ops::cos(b)).max(0.0);
        let contained_fraction = if a <= b {
            1.0
        } else if cap_a > 1.0e-12 {
            (cap_b / cap_a).clamp(0.0, 1.0)
        } else {
            0.0
        };

        if angle <= inner {
            return contained_fraction;
        }
        if angle >= outer {
            return 0.0;
        }
        let span = outer - inner;
        if span <= 1.0e-12 {
            return 0.0;
        }
        let t = ((outer - angle) / span).clamp(0.0, 1.0);
        water_smoothstep01(t) * contained_fraction
    }

    /// Byte-for-byte re-statement of the shader's `water_horizon_occlusion`
    /// helper (closed-/open-cone early-outs and the aperture-scaled rim).
    fn water_horizon_occlusion(
        reflection_dir: Vec3,
        bent_normal_view: Vec3,
        cone_aperture: f32,
    ) -> f32 {
        let aperture = cone_aperture.clamp(0.0, PI);
        if aperture <= MIN_SPECULAR_RIM {
            return 0.0;
        }
        if aperture >= PI - 1.0e-3 {
            return 1.0;
        }
        let rim = (aperture * SPECULAR_RIM_FRACTION).clamp(MIN_SPECULAR_RIM, PI);
        water_cone_cone_intersection(reflection_dir, rim, bent_normal_view, aperture)
    }

    /// The shader's `1.0 / sqrt(len_sq)` normalisation with the +Y degenerate
    /// fallback. `1.0 / x` and the golden's `x.recip()` are both correctly
    /// rounded `f32` divisions of `1.0`, so they agree bit-for-bit.
    fn normalize_or_y(v: Vec3) -> Vec3 {
        let len_sq = v.length_squared();
        if len_sq > 1.0e-12 {
            v * (1.0 / len_sq.sqrt())
        } else {
            Vec3::Y
        }
    }

    /// The shader's bent-normal cone fit: recover the half-angle from the
    /// mean-resultant length through the uniform-cone centroid identity
    /// `aperture = acos(2r - 1)` (engine `BentNormalCone` convention).
    fn aperture_from_resultant(r: f32) -> f32 {
        ops::acos((2.0 * r - 1.0).clamp(-1.0, 1.0))
    }

    /// A spread of exact and normalised unit directions whose `length_squared`
    /// clears the `1e-12` normalisation threshold, so the twin and golden take
    /// identical normalisation branches.
    fn unit_dirs() -> [Vec3; 7] {
        [
            Vec3::X,
            Vec3::Y,
            Vec3::Z,
            Vec3::new(1.0, 1.0, 0.0).normalize(),
            Vec3::new(-1.0, 2.0, 0.5).normalize(),
            Vec3::new(0.3, -0.7, 1.2).normalize(),
            Vec3::new(-0.5, -0.5, -0.5).normalize(),
        ]
    }

    /// An aperture sweep over `[0, PI]` that avoids the degenerate band where
    /// `span` or `cap_a` would fall between the `1e-12` twin threshold and the
    /// golden's `f32::MIN_POSITIVE`, so both resolve the same branch.
    fn apertures() -> [f32; 8] {
        [0.0, 0.1, 0.4, 0.8, 1.2, 1.8, 2.6, PI]
    }

    /// The shader twin must match the engine golden exactly over a dense grid of
    /// direction pairs and apertures, so the water reflection occlusion stays
    /// identical to every other surface's.
    #[test]
    fn twin_cone_matches_golden() {
        let dirs = unit_dirs();
        for da in dirs {
            for db in dirs {
                for &ap_a in &apertures() {
                    for &ap_b in &apertures() {
                        let twin = water_cone_cone_intersection(da, ap_a, db, ap_b);
                        let g = golden_cone(da, ap_a, db, ap_b);
                        assert_eq!(
                            twin.to_bits(),
                            g.to_bits(),
                            "cone da {da:?} ap_a {ap_a} db {db:?} ap_b {ap_b}: twin {twin} golden {g}"
                        );
                    }
                }
            }
        }
    }

    /// The horizon twin must match the engine golden exactly over the same grid.
    #[test]
    fn twin_horizon_matches_golden() {
        let dirs = unit_dirs();
        for refl in dirs {
            for bent in dirs {
                for &ap in &apertures() {
                    let twin = water_horizon_occlusion(refl, bent, ap);
                    let g = golden_horizon(refl, bent, ap);
                    assert_eq!(
                        twin.to_bits(),
                        g.to_bits(),
                        "horizon refl {refl:?} bent {bent:?} ap {ap}: twin {twin} golden {g}"
                    );
                }
            }
        }
    }

    /// A reflection aligned with the bent normal, well inside an open cone, is
    /// essentially unoccluded.
    #[test]
    fn aligned_reflection_inside_open_cone_survives() {
        let occ = water_horizon_occlusion(Vec3::Y, Vec3::Y, 2.5);
        assert!(occ > 0.99, "aligned open-cone occlusion {occ} should be ~1");
    }

    /// A reflection pointing opposite the bent normal of a narrow cone is fully
    /// occluded.
    #[test]
    fn opposed_reflection_narrow_cone_is_occluded() {
        let occ = water_horizon_occlusion(-Vec3::Y, Vec3::Y, 0.3);
        assert!(
            occ < 1.0e-6,
            "opposed narrow-cone occlusion {occ} should be ~0"
        );
    }

    /// A fully open cone never occludes; a fully closed cone always occludes,
    /// regardless of reflection direction.
    #[test]
    fn open_cone_passes_closed_cone_blocks() {
        for refl in unit_dirs() {
            let open = water_horizon_occlusion(refl, Vec3::Y, PI);
            assert!((open - 1.0).abs() < 1.0e-6, "open cone {open} should be 1");
            let closed = water_horizon_occlusion(refl, Vec3::Y, 0.0);
            assert!(closed.abs() < 1.0e-6, "closed cone {closed} should be 0");
        }
    }

    /// Every result is a finite factor in `[0, 1]`, including degenerate inputs.
    #[test]
    fn always_bounded_and_finite() {
        let dirs = unit_dirs();
        for refl in dirs {
            for bent in dirs {
                for &ap in &[-1.0, 0.0, 0.5, 1.5, 3.0, 4.0] {
                    let occ = water_horizon_occlusion(refl, bent, ap);
                    assert!(
                        occ.is_finite(),
                        "non-finite occlusion refl {refl:?} ap {ap}"
                    );
                    assert!((0.0..=1.0).contains(&occ), "out of range occlusion {occ}");
                }
            }
        }
    }

    /// A zero-length direction normalises to the +Y fallback rather than
    /// producing a NaN, matching the shader.
    #[test]
    fn degenerate_direction_uses_fallback() {
        let via_zero = water_cone_cone_intersection(Vec3::ZERO, 1.0, Vec3::Y, 1.5);
        let via_y = water_cone_cone_intersection(Vec3::Y, 1.0, Vec3::Y, 1.5);
        assert!(via_zero.is_finite());
        assert_eq!(via_zero.to_bits(), via_y.to_bits());
    }

    /// Deterministic: identical inputs reproduce identical bits.
    #[test]
    fn deterministic() {
        let a = water_horizon_occlusion(Vec3::new(0.2, 0.9, 0.1).normalize(), Vec3::Y, 0.7);
        let b = water_horizon_occlusion(Vec3::new(0.2, 0.9, 0.1).normalize(), Vec3::Y, 0.7);
        assert_eq!(a.to_bits(), b.to_bits());
    }

    /// The bent-normal cone fit maps the mean-resultant length to the half-angle
    /// through `acos(2r - 1)`: a fully coherent resultant (`r = 1`) collapses the
    /// cone to a point, a fully cancelled one (`r = 0`) opens it to `PI`, the
    /// midpoint maps to `PI / 2`, and the map is monotonically decreasing in `r`.
    #[test]
    fn aperture_from_resultant_mapping() {
        assert!(aperture_from_resultant(1.0).abs() < 1.0e-6, "r=1 -> 0");
        assert!(
            (aperture_from_resultant(0.0) - PI).abs() < 1.0e-6,
            "r=0 -> PI"
        );
        assert!(
            (aperture_from_resultant(0.5) - PI / 2.0).abs() < 1.0e-6,
            "r=0.5 -> PI/2"
        );
        let mut prev = f32::INFINITY;
        for i in 0..=20u32 {
            let r = i as f32 / 20.0;
            let ap = aperture_from_resultant(r);
            assert!(
                ap <= prev + 1.0e-6,
                "aperture must be non-increasing in r at r {r}"
            );
            prev = ap;
        }
    }
}
