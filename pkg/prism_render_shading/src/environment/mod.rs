//! Backend-neutral image-based lighting (IBL) reference.
//!
//! Direct lights cover analytic emitters; the indirect term is supplied by an
//! environment.  This module models the low-frequency environment with an order
//! two (L2) spherical-harmonic radiance probe and evaluates the two split-sum
//! halves of Karis's real-time IBL:
//!
//! * **Diffuse** convolves the radiance SH with the clamped cosine lobe to get
//!   irradiance (Ramamoorthi & Hanrahan 2001), matching Unreal's `FSHVectorRGB3`
//!   diffuse path.
//! * **Specular** reconstructs the environment radiance along the reflection
//!   vector and scales it by the analytic environment-BRDF (`EnvBRDFApprox`
//!   from "Physically Based Shading on Mobile"), which stands in for the
//!   prefiltered mip / DFG LUT the GPU path samples.
//!
//! Everything is polynomial except the single `exp2` in the BRDF fit, which
//! routes through `bevy_math::ops` for cross-platform determinism.

use bevy_math::ops;

use crate::{ShadingFrame, SurfaceSample};

mod brdf_lut;
mod cubemap;
mod prefilter;
mod sampling;
pub use brdf_lut::{integrate_brdf, DfgLut};
pub use cubemap::{project_cubemap_to_sh, CubemapFaces};
pub use prefilter::{prefilter_radiance, PrefilteredEnvMap};

/// `2 * sqrt(pi)`, the projection weight of a constant function onto the SH DC
/// band.  A constant radiance `c` therefore stores `c * SQRT_PI_4` in band 0.
const SQRT_PI_4: f32 = 3.5449077;

/// Cosine-lobe convolution factors per SH band (Ramamoorthi & Hanrahan).
const A0: f32 = core::f32::consts::PI; // l = 0
const A1: f32 = 2.094_395_2; // 2*pi/3, l = 1
const A2: f32 = core::f32::consts::FRAC_PI_4; // pi/4,   l = 2

/// Real SH basis constants for bands 0..2.
const K0: f32 = 0.282_094_8; // 0.5 * sqrt(1/pi)
const K1: f32 = 0.488_602_5; // 0.5 * sqrt(3/pi)
const K2_XY: f32 = 1.092_548_4; // 0.5 * sqrt(15/pi)
const K2_Z2: f32 = 0.315_391_57; // 0.25 * sqrt(5/pi)
const K2_X2: f32 = 0.546_274_2; // 0.25 * sqrt(15/pi)

/// Evaluates the nine real SH basis functions for a unit direction.
fn sh_basis(direction: [f32; 3]) -> [f32; 9] {
    let [x, y, z] = direction;
    [
        K0,
        K1 * y,
        K1 * z,
        K1 * x,
        K2_XY * x * y,
        K2_XY * y * z,
        K2_Z2 * (3.0 * z * z - 1.0),
        K2_XY * x * z,
        K2_X2 * (x * x - y * y),
    ]
}

/// An order-two spherical-harmonic radiance probe with one RGB coefficient per
/// band.  The layout mirrors the GPU probe that will feed the resolve shader.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphericalHarmonicsL2 {
    /// Nine RGB coefficients following the [`sh_basis`] ordering.
    pub coefficients: [[f32; 3]; 9],
}

impl Default for SphericalHarmonicsL2 {
    fn default() -> Self {
        Self::ZERO
    }
}

impl SphericalHarmonicsL2 {
    /// A probe that radiates nothing.
    pub const ZERO: Self = Self {
        coefficients: [[0.0; 3]; 9],
    };

    /// Builds a probe describing a uniform radiance `color` over the sphere.
    pub fn from_constant(color: [f32; 3]) -> Self {
        let mut probe = Self::ZERO;
        probe.coefficients[0] = [
            color[0] * SQRT_PI_4,
            color[1] * SQRT_PI_4,
            color[2] * SQRT_PI_4,
        ];
        probe
    }

    /// Accumulates a directional radiance sample into the probe.
    ///
    /// `direction` points toward the incoming radiance and is normalized; the
    /// `weight` is the solid angle the sample represents.  This is the baking
    /// primitive used to assemble an environment from analytic lights or from
    /// integrated cube-map texels.
    pub fn add_directional_radiance(
        &mut self,
        direction: [f32; 3],
        radiance: [f32; 3],
        weight: f32,
    ) {
        let basis = sh_basis(normalize_or(direction, [0.0, 1.0, 0.0]));
        for (coefficient, basis_value) in self.coefficients.iter_mut().zip(basis) {
            let scaled = basis_value * weight;
            coefficient[0] += radiance[0] * scaled;
            coefficient[1] += radiance[1] * scaled;
            coefficient[2] += radiance[2] * scaled;
        }
    }

    /// Reconstructs the (unconvolved) radiance leaving the probe along
    /// `direction`.  Ringing can drive individual channels negative, so the
    /// result is clamped to the non-negative range for use as specular radiance.
    pub fn radiance(&self, direction: [f32; 3]) -> [f32; 3] {
        let basis = sh_basis(normalize_or(direction, [0.0, 1.0, 0.0]));
        let mut sum = [0.0; 3];
        for (coefficient, basis_value) in self.coefficients.iter().zip(basis) {
            sum[0] += coefficient[0] * basis_value;
            sum[1] += coefficient[1] * basis_value;
            sum[2] += coefficient[2] * basis_value;
        }
        [sum[0].max(0.0), sum[1].max(0.0), sum[2].max(0.0)]
    }

    /// Convolves the probe with the clamped cosine lobe to produce irradiance
    /// at a surface `normal`.  Dividing by `PI` yields Lambertian diffuse.
    pub fn irradiance(&self, normal: [f32; 3]) -> [f32; 3] {
        let basis = sh_basis(normalize_or(normal, [0.0, 1.0, 0.0]));
        let band = [A0, A1, A1, A1, A2, A2, A2, A2, A2];
        let mut sum = [0.0; 3];
        for index in 0..9 {
            let factor = band[index] * basis[index];
            sum[0] += self.coefficients[index][0] * factor;
            sum[1] += self.coefficients[index][1] * factor;
            sum[2] += self.coefficients[index][2] * factor;
        }
        [sum[0].max(0.0), sum[1].max(0.0), sum[2].max(0.0)]
    }
}

/// An image-based light: a low-frequency radiance probe plus an intensity
/// scale applied to both the diffuse and specular contributions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImageBasedLight {
    /// Environment radiance encoded as an L2 SH probe.
    pub radiance: SphericalHarmonicsL2,
    /// Linear, pre-exposed multiplier applied to the probe contribution.
    pub intensity: f32,
}

impl Default for ImageBasedLight {
    fn default() -> Self {
        Self {
            radiance: SphericalHarmonicsL2::ZERO,
            intensity: 1.0,
        }
    }
}

impl ImageBasedLight {
    /// Wraps a probe with unit intensity.
    pub fn new(radiance: SphericalHarmonicsL2) -> Self {
        Self {
            radiance,
            intensity: 1.0,
        }
    }
}

/// Analytic environment BRDF (`EnvBRDFApprox`, Karis 2014) returning the split
/// sum `(scale, bias)` used as `F0 * scale + bias`.
pub fn env_brdf_approx(n_dot_v: f32, perceptual_roughness: f32) -> [f32; 2] {
    let n_dot_v = n_dot_v.clamp(0.0, 1.0);
    let roughness = perceptual_roughness.clamp(0.0, 1.0);
    const C0: [f32; 4] = [-1.0, -0.0275, -0.572, 0.022];
    const C1: [f32; 4] = [1.0, 0.0425, 1.04, -0.04];
    let r = [
        roughness * C0[0] + C1[0],
        roughness * C0[1] + C1[1],
        roughness * C0[2] + C1[2],
        roughness * C0[3] + C1[3],
    ];
    let a004 = (r[0] * r[0]).min(ops::exp2(-9.28 * n_dot_v)) * r[0] + r[1];
    [-1.04 * a004 + r[2], 1.04 * a004 + r[3]]
}

/// Evaluates the indirect (image-based) contribution for a surface.
///
/// The diffuse half uses SH irradiance scaled by albedo and occlusion; the
/// specular half reflects the view about the normal, samples the probe there,
/// and weights it by the analytic environment BRDF.  Emissive is intentionally
/// excluded so the resolve pass can add it exactly once.
pub fn evaluate_image_based_light(
    surface: SurfaceSample,
    frame: ShadingFrame,
    light: &ImageBasedLight,
) -> [f32; 3] {
    let normal = normalize_or(frame.normal, [0.0, 1.0, 0.0]);
    let view = normalize_or(frame.view, normal);
    let n_dot_v = dot(normal, view).max(1.0e-4);

    let metallic = surface.metallic.clamp(0.0, 1.0);
    let roughness = surface.perceptual_roughness.clamp(0.0, 1.0);
    let occlusion = surface.ambient_occlusion.clamp(0.0, 1.0);
    let reflectance = surface.reflectance.clamp(0.0, 1.0);
    let f0_dielectric = 0.16 * reflectance * reflectance;
    let f0 = [
        mix(f0_dielectric, surface.base_color[0], metallic),
        mix(f0_dielectric, surface.base_color[1], metallic),
        mix(f0_dielectric, surface.base_color[2], metallic),
    ];

    // Diffuse: Lambertian albedo lit by cosine-convolved irradiance.
    let irradiance = light.radiance.irradiance(normal);
    let diffuse_weight = 1.0 - metallic;
    let inverse_pi = core::f32::consts::FRAC_1_PI;
    let diffuse = [
        surface.base_color[0] * diffuse_weight * irradiance[0] * inverse_pi,
        surface.base_color[1] * diffuse_weight * irradiance[1] * inverse_pi,
        surface.base_color[2] * diffuse_weight * irradiance[2] * inverse_pi,
    ];

    // Specular: prefiltered radiance along the reflection vector weighted by
    // the analytic environment BRDF.
    let reflection = reflect(mul_scalar(view, -1.0), normal);
    let prefiltered = light.radiance.radiance(reflection);
    let dfg = env_brdf_approx(n_dot_v, roughness);
    let specular = [
        prefiltered[0] * (f0[0] * dfg[0] + dfg[1]),
        prefiltered[1] * (f0[1] * dfg[0] + dfg[1]),
        prefiltered[2] * (f0[2] * dfg[0] + dfg[1]),
    ];

    [
        (diffuse[0] + specular[0]) * occlusion * light.intensity,
        (diffuse[1] + specular[1]) * occlusion * light.intensity,
        (diffuse[2] + specular[2]) * occlusion * light.intensity,
    ]
}

/// A high-frequency specular environment: a GGX-prefiltered radiance mip chain
/// paired with the split-sum environment-BRDF ("DFG") lookup table.
///
/// Passing this to [`evaluate_image_based_light_specular`] replaces the
/// low-frequency SH-radiance specular hack with the real prefiltered map
/// sampled at the roughness-selected mip and the integrated DFG term, matching
/// the GPU resolve path once Slice C binds the precomputed textures.
#[derive(Clone, Copy, Debug)]
pub struct SpecularEnvironment<'a> {
    /// GGX-prefiltered environment radiance, one cube per roughness mip.
    pub prefiltered: &'a PrefilteredEnvMap,
    /// Integrated split-sum environment BRDF, indexed by `(n_dot_v, roughness)`.
    pub dfg: &'a DfgLut,
}

/// Evaluates the indirect contribution using a prefiltered specular source.
///
/// The diffuse half is identical to [`evaluate_image_based_light`] (SH
/// irradiance scaled by albedo and occlusion).  The specular half samples the
/// prefiltered radiance mip chain along the reflection vector at the surface
/// roughness and weights it by the integrated DFG table (`F0 * scale + bias`),
/// so glossy reflections read a sharp mip and rough reflections read a
/// pre-blurred one.  Emissive is intentionally excluded so the resolve pass can
/// add it exactly once.
pub fn evaluate_image_based_light_specular(
    surface: SurfaceSample,
    frame: ShadingFrame,
    light: &ImageBasedLight,
    specular_env: &SpecularEnvironment<'_>,
) -> [f32; 3] {
    let normal = normalize_or(frame.normal, [0.0, 1.0, 0.0]);
    let view = normalize_or(frame.view, normal);
    let n_dot_v = dot(normal, view).max(1.0e-4);

    let metallic = surface.metallic.clamp(0.0, 1.0);
    let roughness = surface.perceptual_roughness.clamp(0.0, 1.0);
    let occlusion = surface.ambient_occlusion.clamp(0.0, 1.0);
    let reflectance = surface.reflectance.clamp(0.0, 1.0);
    let f0_dielectric = 0.16 * reflectance * reflectance;
    let f0 = [
        mix(f0_dielectric, surface.base_color[0], metallic),
        mix(f0_dielectric, surface.base_color[1], metallic),
        mix(f0_dielectric, surface.base_color[2], metallic),
    ];

    // Diffuse: Lambertian albedo lit by cosine-convolved irradiance.
    let irradiance = light.radiance.irradiance(normal);
    let diffuse_weight = 1.0 - metallic;
    let inverse_pi = core::f32::consts::FRAC_1_PI;
    let diffuse = [
        surface.base_color[0] * diffuse_weight * irradiance[0] * inverse_pi,
        surface.base_color[1] * diffuse_weight * irradiance[1] * inverse_pi,
        surface.base_color[2] * diffuse_weight * irradiance[2] * inverse_pi,
    ];

    // Specular: prefiltered radiance sampled at the roughness-selected mip,
    // weighted by the integrated DFG table (F0 * scale + bias).
    let reflection = reflect(mul_scalar(view, -1.0), normal);
    let prefiltered = specular_env.prefiltered.sample(reflection, roughness);
    let dfg = specular_env.dfg.sample(n_dot_v, roughness);
    let specular = [
        prefiltered[0] * (f0[0] * dfg[0] + dfg[1]),
        prefiltered[1] * (f0[1] * dfg[0] + dfg[1]),
        prefiltered[2] * (f0[2] * dfg[0] + dfg[1]),
    ];

    [
        (diffuse[0] + specular[0]) * occlusion * light.intensity,
        (diffuse[1] + specular[1]) * occlusion * light.intensity,
        (diffuse[2] + specular[2]) * occlusion * light.intensity,
    ]
}

fn reflect(incident: [f32; 3], normal: [f32; 3]) -> [f32; 3] {
    let d = 2.0 * dot(incident, normal);
    [
        incident[0] - d * normal[0],
        incident[1] - d * normal[1],
        incident[2] - d * normal[2],
    ]
}

fn mix(a: f32, b: f32, factor: f32) -> f32 {
    a * (1.0 - factor) + b * factor
}

fn normalize_or(value: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let length_squared = dot(value, value);
    if length_squared > 1.0e-12 && length_squared.is_finite() {
        mul_scalar(value, length_squared.sqrt().recip())
    } else {
        fallback
    }
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn mul_scalar(value: [f32; 3], scalar: f32) -> [f32; 3] {
    [value[0] * scalar, value[1] * scalar, value[2] * scalar]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: [f32; 3], b: [f32; 3], eps: f32) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() <= eps)
    }

    #[test]
    fn constant_probe_reconstructs_uniform_radiance_and_irradiance() {
        let probe = SphericalHarmonicsL2::from_constant([2.0, 3.0, 4.0]);
        // Uniform radiance everywhere.
        for direction in [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]] {
            assert!(approx(probe.radiance(direction), [2.0, 3.0, 4.0], 1.0e-4));
        }
        // Irradiance over a uniform environment is PI * radiance.
        let pi = core::f32::consts::PI;
        assert!(approx(
            probe.irradiance([0.0, 1.0, 0.0]),
            [2.0 * pi, 3.0 * pi, 4.0 * pi],
            1.0e-3
        ));
    }

    #[test]
    fn constant_environment_diffuse_matches_albedo_times_radiance() {
        let light = ImageBasedLight::new(SphericalHarmonicsL2::from_constant([0.5; 3]));
        let surface = SurfaceSample {
            base_color: [0.6, 0.6, 0.6],
            metallic: 0.0,
            perceptual_roughness: 1.0,
            reflectance: 0.0,
            ..Default::default()
        };
        let frame = ShadingFrame {
            normal: [0.0, 1.0, 0.0],
            view: [0.0, 1.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
            bitangent: [0.0, 0.0, -1.0],
        };
        let value = evaluate_image_based_light(surface, frame, &light);
        // Rough non-reflective dielectric: diffuse dominates at albedo*radiance
        // (0.6 * 0.5 = 0.3) plus a small specular tail from the BRDF bias.
        assert!(value.iter().all(|c| *c > 0.28 && *c < 0.4));
    }

    #[test]
    fn directional_sample_biases_irradiance_toward_the_light() {
        let mut probe = SphericalHarmonicsL2::ZERO;
        probe.add_directional_radiance([0.0, 1.0, 0.0], [1.0; 3], 1.0);
        let toward = probe.irradiance([0.0, 1.0, 0.0]);
        let away = probe.irradiance([0.0, -1.0, 0.0]);
        assert!(toward[0] > away[0], "irradiance should peak toward the sample");
        assert!(toward.iter().all(|c| c.is_finite()));
    }

    #[test]
    fn env_brdf_scale_and_bias_stay_in_unit_range() {
        for &n_dot_v in &[0.05_f32, 0.25, 0.5, 0.75, 1.0] {
            for &roughness in &[0.0_f32, 0.25, 0.5, 0.75, 1.0] {
                let [scale, bias] = env_brdf_approx(n_dot_v, roughness);
                assert!(scale.is_finite() && bias.is_finite());
                assert!((-0.05..=1.05).contains(&scale), "scale {scale}");
                assert!((-0.05..=1.05).contains(&bias), "bias {bias}");
            }
        }
    }

    #[test]
    fn metallic_surface_reflects_environment_color() {
        let light = ImageBasedLight::new(SphericalHarmonicsL2::from_constant([1.0, 0.0, 0.0]));
        let surface = SurfaceSample {
            base_color: [1.0, 1.0, 1.0],
            metallic: 1.0,
            perceptual_roughness: 0.1,
            reflectance: 0.5,
            ..Default::default()
        };
        let frame = ShadingFrame {
            normal: [0.0, 1.0, 0.0],
            view: [0.0, 1.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
            bitangent: [0.0, 0.0, -1.0],
        };
        let value = evaluate_image_based_light(surface, frame, &light);
        // A red environment seen by a metal reflects red, not green/blue.
        assert!(value[0] > value[1] && value[0] > value[2]);
        assert!(value.iter().all(|c| c.is_finite() && *c >= 0.0));
    }

    #[test]
    fn occlusion_scales_the_indirect_term() {
        let light = ImageBasedLight::new(SphericalHarmonicsL2::from_constant([0.8; 3]));
        let make = |ao: f32| SurfaceSample {
            base_color: [0.5; 3],
            ambient_occlusion: ao,
            perceptual_roughness: 0.8,
            ..Default::default()
        };
        let frame = ShadingFrame {
            normal: [0.0, 1.0, 0.0],
            view: [0.0, 1.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
            bitangent: [0.0, 0.0, -1.0],
        };
        let full = evaluate_image_based_light(make(1.0), frame, &light);
        let half = evaluate_image_based_light(make(0.5), frame, &light);
        for channel in 0..3 {
            assert!((half[channel] - full[channel] * 0.5).abs() < 1.0e-5);
        }
    }

    fn constant_cube(size: u32, color: [f32; 3]) -> CubemapFaces {
        let face = alloc::vec![color; (size as usize) * (size as usize)];
        CubemapFaces::new(size, core::array::from_fn(|_| face.clone())).unwrap()
    }

    fn surface_for_specular(roughness: f32, metallic: f32) -> SurfaceSample {
        SurfaceSample {
            base_color: [0.9, 0.8, 0.7],
            perceptual_roughness: roughness,
            metallic,
            reflectance: 0.5,
            ambient_occlusion: 1.0,
            ..Default::default()
        }
    }

    #[test]
    fn prefiltered_specular_matches_analytic_on_constant_environment() {
        // A constant environment: the prefiltered map is constant at every mip
        // and the DFG LUT tracks the analytic fit, so the prefiltered specular
        // path must land close to the analytic `evaluate_image_based_light`.
        let color = [0.4, 0.5, 0.6];
        let probe = SphericalHarmonicsL2::from_constant(color);
        let light = ImageBasedLight::new(probe);
        let cube = constant_cube(8, color);
        let prefiltered = PrefilteredEnvMap::generate(&cube, 5, 8, 64).unwrap();
        let dfg = DfgLut::generate(64, 256).unwrap();
        let env = SpecularEnvironment {
            prefiltered: &prefiltered,
            dfg: &dfg,
        };
        let frame = ShadingFrame {
            normal: [0.0, 1.0, 0.0],
            view: [0.0, 1.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
            bitangent: [0.0, 0.0, -1.0],
        };
        for roughness in [0.1, 0.4, 0.7, 1.0] {
            let surface = surface_for_specular(roughness, 0.3);
            let analytic = evaluate_image_based_light(surface, frame, &light);
            let real = evaluate_image_based_light_specular(surface, frame, &light, &env);
            // The DFG LUT and analytic fit differ slightly; require the same
            // ballpark rather than bit equality.
            assert!(approx(analytic, real, 0.05), "r={roughness} {analytic:?} vs {real:?}");
        }
    }

    #[test]
    fn prefiltered_specular_reflects_the_bright_face() {
        // Only +X is bright. A mirror metal looking along -X reflects toward +X
        // and must pick up that radiance; looking along +X reflects to -X (dark).
        let mut colors = [[0.0f32; 3]; 6];
        colors[0] = [6.0, 6.0, 6.0]; // +X
        let cube = CubemapFaces::new(
            16,
            core::array::from_fn(|i| alloc::vec![colors[i]; 16 * 16]),
        )
        .unwrap();
        let prefiltered = PrefilteredEnvMap::generate(&cube, 6, 16, 128).unwrap();
        let dfg = DfgLut::generate(64, 128).unwrap();
        let env = SpecularEnvironment {
            prefiltered: &prefiltered,
            dfg: &dfg,
        };
        let light = ImageBasedLight::new(SphericalHarmonicsL2::ZERO);
        let mirror = surface_for_specular(0.02, 1.0);
        // View looking along -X: reflection about +X normal points to +X.
        let toward = ShadingFrame {
            normal: [1.0, 0.0, 0.0],
            view: [1.0, 0.0, 0.0],
            tangent: [0.0, 1.0, 0.0],
            bitangent: [0.0, 0.0, 1.0],
        };
        let away = ShadingFrame {
            normal: [-1.0, 0.0, 0.0],
            view: [-1.0, 0.0, 0.0],
            tangent: [0.0, 1.0, 0.0],
            bitangent: [0.0, 0.0, 1.0],
        };
        let hit = evaluate_image_based_light_specular(mirror, toward, &light, &env);
        let miss = evaluate_image_based_light_specular(mirror, away, &light, &env);
        assert!(hit[0] > miss[0] + 1.0, "hit={hit:?} miss={miss:?}");
        assert!(hit.iter().all(|c| c.is_finite() && *c >= 0.0));
    }

    #[test]
    fn rougher_specular_blurs_the_bright_face_highlight() {
        // A sharp mirror sees the bright +X face at full strength; a rough
        // surface averages it with dark neighbours, dimming the peak.
        let mut colors = [[0.0f32; 3]; 6];
        colors[0] = [8.0, 8.0, 8.0];
        let cube = CubemapFaces::new(
            16,
            core::array::from_fn(|i| alloc::vec![colors[i]; 16 * 16]),
        )
        .unwrap();
        let prefiltered = PrefilteredEnvMap::generate(&cube, 6, 16, 256).unwrap();
        let dfg = DfgLut::generate(64, 128).unwrap();
        let env = SpecularEnvironment {
            prefiltered: &prefiltered,
            dfg: &dfg,
        };
        let light = ImageBasedLight::new(SphericalHarmonicsL2::ZERO);
        let frame = ShadingFrame {
            normal: [1.0, 0.0, 0.0],
            view: [1.0, 0.0, 0.0],
            tangent: [0.0, 1.0, 0.0],
            bitangent: [0.0, 0.0, 1.0],
        };
        let sharp = evaluate_image_based_light_specular(surface_for_specular(0.05, 1.0), frame, &light, &env);
        let rough = evaluate_image_based_light_specular(surface_for_specular(0.9, 1.0), frame, &light, &env);
        assert!(rough[0] < sharp[0], "sharp={sharp:?} rough={rough:?}");
    }
}
