//! Janssen silo wall-pressure profile.
//!
//! Granular material stored in a tall silo does not behave like a liquid:
//! wall friction carries a growing fraction of the overburden, so the
//! vertical stress saturates with depth instead of rising linearly. Janssen's
//! classic 1895 analysis models this screening effect for a prismatic column
//! whose cross-section is characterised by its hydraulic radius `R = A / P`
//! (area over wetted perimeter).
//!
//! With bulk density `ρ`, gravitational acceleration `g`, wall friction
//! coefficient `μ_w`, and lateral-to-vertical stress ratio `K`, the vertical
//! stress at depth `z` below the free surface is
//!
//! ```text
//! σ_v(z) = σ_∞ · (1 − exp(−z / z_c)),   z_c = R / (μ_w · K),   σ_∞ = ρ g z_c
//! ```
//!
//! The horizontal (wall-normal) stress is `σ_h = K · σ_v` and the wall shear
//! stress is `τ_w = μ_w · σ_h`. As `z → ∞` the vertical stress saturates at
//! `σ_∞ = ρ g R / (μ_w K)`, independent of the fill height — the signature of
//! the Janssen effect.
//!
//! This module is a pure analytic correlation with no coupling to the
//! simulation pipeline; it mirrors the style of the Beverloo discharge model.

/// Cross-sectional geometry of a silo, reduced to its hydraulic radius.
///
/// The hydraulic radius `R = A / P` is the only cross-section descriptor that
/// enters Janssen's equation, so circular and rectangular shells collapse onto
/// the same one-dimensional profile once `R` is known.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SiloCrossSection {
    hydraulic_radius: f32,
}

impl SiloCrossSection {
    /// Builds a cross-section directly from a hydraulic radius `R > 0`.
    pub fn from_hydraulic_radius(hydraulic_radius: f32) -> Option<Self> {
        if !hydraulic_radius.is_finite() || hydraulic_radius <= 0.0 {
            return None;
        }
        Some(Self { hydraulic_radius })
    }

    /// Circular silo of the given inner `radius > 0`.
    ///
    /// `A = π r²`, `P = 2π r`, so `R = r / 2`.
    pub fn circular(radius: f32) -> Option<Self> {
        if !radius.is_finite() || radius <= 0.0 {
            return None;
        }
        Self::from_hydraulic_radius(0.5 * radius)
    }

    /// Rectangular silo with inner plan dimensions `width` and `depth` (both
    /// strictly positive).
    ///
    /// `A = w · d`, `P = 2 (w + d)`, so `R = w d / (2 (w + d))`.
    pub fn rectangular(width: f32, depth: f32) -> Option<Self> {
        if !width.is_finite() || !depth.is_finite() || width <= 0.0 || depth <= 0.0 {
            return None;
        }
        let r = (width * depth) / (2.0 * (width + depth));
        Self::from_hydraulic_radius(r)
    }

    /// The hydraulic radius `R = A / P`.
    pub fn hydraulic_radius(&self) -> f32 {
        self.hydraulic_radius
    }
}

/// A fully parameterised Janssen vertical/horizontal/shear stress profile.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JanssenProfile {
    hydraulic_radius: f32,
    bulk_density: f32,
    gravity: f32,
    wall_friction: f32,
    k_ratio: f32,
    characteristic_depth: f32,
    saturation_vertical_stress: f32,
}

impl JanssenProfile {
    /// Builds a profile from the silo cross-section and material parameters.
    ///
    /// * `bulk_density` — ρ, aggregate (not grain) density, `> 0`.
    /// * `gravity` — g, magnitude of gravitational acceleration, `> 0`.
    /// * `wall_friction` — `μ_w`, grain/wall friction coefficient, `> 0`.
    /// * `k_ratio` — K, lateral-to-vertical stress ratio, `> 0` (typically
    ///   around 0.4–0.6 for granular solids).
    ///
    /// Returns `None` for any non-finite or non-positive argument.
    pub fn new(
        cross_section: SiloCrossSection,
        bulk_density: f32,
        gravity: f32,
        wall_friction: f32,
        k_ratio: f32,
    ) -> Option<Self> {
        let hydraulic_radius = cross_section.hydraulic_radius();
        for v in [bulk_density, gravity, wall_friction, k_ratio] {
            if !v.is_finite() || v <= 0.0 {
                return None;
            }
        }
        // z_c = R / (μ_w K); σ_∞ = ρ g z_c.
        let characteristic_depth = hydraulic_radius / (wall_friction * k_ratio);
        if !characteristic_depth.is_finite() || characteristic_depth <= 0.0 {
            return None;
        }
        let saturation_vertical_stress = bulk_density * gravity * characteristic_depth;
        if !saturation_vertical_stress.is_finite() {
            return None;
        }
        Some(Self {
            hydraulic_radius,
            bulk_density,
            gravity,
            wall_friction,
            k_ratio,
            characteristic_depth,
            saturation_vertical_stress,
        })
    }

    /// Hydraulic radius `R`.
    pub fn hydraulic_radius(&self) -> f32 {
        self.hydraulic_radius
    }

    /// Bulk density `ρ`.
    pub fn bulk_density(&self) -> f32 {
        self.bulk_density
    }

    /// Gravitational acceleration `g`.
    pub fn gravity(&self) -> f32 {
        self.gravity
    }

    /// Wall friction coefficient `μ_w`.
    pub fn wall_friction(&self) -> f32 {
        self.wall_friction
    }

    /// Lateral-to-vertical stress ratio `K`.
    pub fn k_ratio(&self) -> f32 {
        self.k_ratio
    }

    /// Characteristic Janssen depth `z_c = R / (μ_w K)`.
    ///
    /// At `z = z_c` the vertical stress reaches `1 − 1/e ≈ 63.2 %` of its
    /// saturation value.
    pub fn characteristic_depth(&self) -> f32 {
        self.characteristic_depth
    }

    /// Saturation (asymptotic) vertical stress `σ_∞ = ρ g R / (μ_w K)`.
    pub fn saturation_vertical_stress(&self) -> f32 {
        self.saturation_vertical_stress
    }

    /// Saturation horizontal (wall-normal) stress `K · σ_∞`.
    pub fn saturation_horizontal_stress(&self) -> f32 {
        self.k_ratio * self.saturation_vertical_stress
    }

    /// Saturation wall shear stress `μ_w K · σ_∞`.
    pub fn saturation_wall_shear_stress(&self) -> f32 {
        self.wall_friction * self.saturation_horizontal_stress()
    }

    /// Vertical stress `σ_v(z) = σ_∞ (1 − exp(−z / z_c))` at `depth ≥ 0` below
    /// the free surface. Negative depths clamp to zero (no overburden above the
    /// surface).
    pub fn vertical_stress(&self, depth: f32) -> f32 {
        if !depth.is_finite() || depth <= 0.0 {
            return 0.0;
        }
        // exp() is banned on f32; evaluate the saturation factor in f64.
        let ratio = (depth / self.characteristic_depth) as f64;
        let factor = 1.0 - (-ratio).exp();
        self.saturation_vertical_stress * factor as f32
    }

    /// Horizontal (wall-normal) stress `σ_h(z) = K · σ_v(z)`.
    pub fn horizontal_stress(&self, depth: f32) -> f32 {
        self.k_ratio * self.vertical_stress(depth)
    }

    /// Wall shear stress `τ_w(z) = μ_w · σ_h(z)`.
    pub fn wall_shear_stress(&self, depth: f32) -> f32 {
        self.wall_friction * self.horizontal_stress(depth)
    }

    /// Fraction of the hydrostatic (frictionless) vertical stress that friction
    /// screens away at `depth`: `1 − σ_v(z) / (ρ g z)`.
    ///
    /// At the surface the frictionless limit applies (screening `0`); deep in
    /// the silo the screening approaches `1`. Returns `0` for non-positive
    /// depths.
    pub fn screening_fraction(&self, depth: f32) -> f32 {
        if !depth.is_finite() || depth <= 0.0 {
            return 0.0;
        }
        let hydrostatic = self.bulk_density * self.gravity * depth;
        if hydrostatic <= 0.0 {
            return 0.0;
        }
        let screened = 1.0 - self.vertical_stress(depth) / hydrostatic;
        screened.clamp(0.0, 1.0)
    }

    /// Uniformly samples the profile over `[0, max_depth]` into `samples`
    /// points (inclusive of both ends), yielding `(depth, σ_v, σ_h, τ_w)`.
    ///
    /// Returns `None` if `max_depth` is not finite/positive or `samples < 2`.
    pub fn sample(&self, max_depth: f32, samples: usize) -> Option<Vec<StressSample>> {
        if !max_depth.is_finite() || max_depth <= 0.0 || samples < 2 {
            return None;
        }
        let mut out = Vec::with_capacity(samples);
        let last = (samples - 1) as f32;
        for i in 0..samples {
            let depth = max_depth * (i as f32 / last);
            out.push(StressSample {
                depth,
                vertical_stress: self.vertical_stress(depth),
                horizontal_stress: self.horizontal_stress(depth),
                wall_shear_stress: self.wall_shear_stress(depth),
            });
        }
        Some(out)
    }
}

/// A single depth sample of the Janssen stress state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StressSample {
    /// Depth below the free surface.
    pub depth: f32,
    /// Vertical stress `σ_v`.
    pub vertical_stress: f32,
    /// Horizontal (wall-normal) stress `σ_h`.
    pub horizontal_stress: f32,
    /// Wall shear stress `τ_w`.
    pub wall_shear_stress: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cross_section_hydraulic_radii() {
        // Circle: R = r / 2.
        let c = SiloCrossSection::circular(2.0).unwrap();
        assert!((c.hydraulic_radius() - 1.0).abs() < 1e-6);
        // Square side 2: A = 4, P = 8, R = 0.5.
        let s = SiloCrossSection::rectangular(2.0, 2.0).unwrap();
        assert!((s.hydraulic_radius() - 0.5).abs() < 1e-6);
        // Direct constructor.
        let d = SiloCrossSection::from_hydraulic_radius(0.75).unwrap();
        assert!((d.hydraulic_radius() - 0.75).abs() < 1e-6);
    }

    #[test]
    fn cross_section_rejects_bad_input() {
        assert!(SiloCrossSection::circular(0.0).is_none());
        assert!(SiloCrossSection::circular(-1.0).is_none());
        assert!(SiloCrossSection::rectangular(0.0, 1.0).is_none());
        assert!(SiloCrossSection::rectangular(1.0, f32::NAN).is_none());
        assert!(SiloCrossSection::from_hydraulic_radius(f32::INFINITY).is_none());
    }

    #[test]
    fn profile_rejects_bad_input() {
        let cs = SiloCrossSection::circular(1.0).unwrap();
        assert!(JanssenProfile::new(cs, 0.0, 9.81, 0.5, 0.5).is_none());
        assert!(JanssenProfile::new(cs, 1000.0, -9.81, 0.5, 0.5).is_none());
        assert!(JanssenProfile::new(cs, 1000.0, 9.81, 0.0, 0.5).is_none());
        assert!(JanssenProfile::new(cs, 1000.0, 9.81, 0.5, f32::NAN).is_none());
    }

    #[test]
    fn characteristic_depth_and_saturation() {
        // R = 0.5, μ_w = 0.5, K = 0.5 → z_c = 0.5 / 0.25 = 2.
        let cs = SiloCrossSection::from_hydraulic_radius(0.5).unwrap();
        let p = JanssenProfile::new(cs, 1000.0, 10.0, 0.5, 0.5).unwrap();
        assert!((p.characteristic_depth() - 2.0).abs() < 1e-4);
        // σ_∞ = ρ g z_c = 1000 · 10 · 2 = 20000.
        assert!((p.saturation_vertical_stress() - 20_000.0).abs() < 1.0);
        assert!((p.saturation_horizontal_stress() - 10_000.0).abs() < 1.0);
        assert!((p.saturation_wall_shear_stress() - 5_000.0).abs() < 1.0);
    }

    #[test]
    fn vertical_stress_hits_63_percent_at_characteristic_depth() {
        let cs = SiloCrossSection::from_hydraulic_radius(0.5).unwrap();
        let p = JanssenProfile::new(cs, 1000.0, 10.0, 0.5, 0.5).unwrap();
        let zc = p.characteristic_depth();
        // 1 - 1/e ≈ 0.632120559.
        let expected = p.saturation_vertical_stress() * 0.632_120_6;
        assert!((p.vertical_stress(zc) - expected).abs() < 1.0);
    }

    #[test]
    fn vertical_stress_monotone_and_bounded() {
        let cs = SiloCrossSection::circular(1.0).unwrap();
        let p = JanssenProfile::new(cs, 1500.0, 9.81, 0.45, 0.5).unwrap();
        let mut prev = 0.0_f32;
        for i in 0..=50 {
            let z = i as f32 * 0.5;
            let sv = p.vertical_stress(z);
            assert!(sv >= prev - 1e-3, "monotone non-decreasing");
            assert!(sv <= p.saturation_vertical_stress() + 1.0, "bounded by σ_∞");
            prev = sv;
        }
    }

    #[test]
    fn shallow_depth_approaches_hydrostatic() {
        // For z ≪ z_c, σ_v(z) ≈ ρ g z (friction has not engaged yet).
        let cs = SiloCrossSection::from_hydraulic_radius(0.5).unwrap();
        let p = JanssenProfile::new(cs, 1000.0, 10.0, 0.5, 0.5).unwrap();
        let z = 0.01; // z_c = 2, so z/z_c = 0.005.
        let hydrostatic = 1000.0 * 10.0 * z;
        let sv = p.vertical_stress(z);
        let rel = (sv - hydrostatic).abs() / hydrostatic;
        assert!(
            rel < 0.01,
            "near-surface stress is nearly hydrostatic: {rel}"
        );
        assert!(p.screening_fraction(z) < 0.01);
    }

    #[test]
    fn stress_components_scale_with_k_and_mu() {
        let cs = SiloCrossSection::circular(1.0).unwrap();
        let p = JanssenProfile::new(cs, 1200.0, 9.81, 0.4, 0.5).unwrap();
        let z = 3.0;
        let sv = p.vertical_stress(z);
        let sh = p.horizontal_stress(z);
        let tau = p.wall_shear_stress(z);
        assert!((sh - 0.5 * sv).abs() < 1e-2);
        assert!((tau - 0.4 * sh).abs() < 1e-2);
    }

    #[test]
    fn negative_and_zero_depth_give_zero() {
        let cs = SiloCrossSection::circular(1.0).unwrap();
        let p = JanssenProfile::new(cs, 1000.0, 9.81, 0.5, 0.5).unwrap();
        assert_eq!(p.vertical_stress(0.0), 0.0);
        assert_eq!(p.vertical_stress(-5.0), 0.0);
        assert_eq!(p.horizontal_stress(-1.0), 0.0);
        assert_eq!(p.wall_shear_stress(0.0), 0.0);
        assert_eq!(p.screening_fraction(0.0), 0.0);
    }

    #[test]
    fn sample_spans_endpoints() {
        let cs = SiloCrossSection::circular(1.0).unwrap();
        let p = JanssenProfile::new(cs, 1000.0, 9.81, 0.5, 0.5).unwrap();
        let samples = p.sample(10.0, 11).unwrap();
        assert_eq!(samples.len(), 11);
        assert!((samples[0].depth - 0.0).abs() < 1e-6);
        assert_eq!(samples[0].vertical_stress, 0.0);
        assert!((samples[10].depth - 10.0).abs() < 1e-5);
        // Monotone non-decreasing depth and vertical stress.
        for w in samples.windows(2) {
            assert!(w[1].depth >= w[0].depth);
            assert!(w[1].vertical_stress >= w[0].vertical_stress - 1e-3);
        }
        assert!(p.sample(10.0, 1).is_none());
        assert!(p.sample(0.0, 5).is_none());
    }
}
