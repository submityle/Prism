//! Adapter from the production material response to the path-traced BSDF oracle.
//!
//! The adapter is deliberately narrow.  `reference_pt::Bsdf` is a collection
//! of explicit lobe models, while the production `SurfaceSample` is the full
//! über surface contract.  A parity fixture must reject fields that cannot be
//! represented without changing their meaning; silently dropping them would
//! make a passing image comparison misleading.
//!
//! This maps the dedicated `evaluate_clearcoat_direct` model, NOT every
//! production LIT closure. In particular the principled coat's Schlick/slab
//! energy redistribution is not the reference's exact-Fresnel/two-pass model.
//! A production-resolve comparison must expose that difference, not replace
//! the independent reference evaluation with the realtime BSDF.

use prism_render_architecture::reference_pt::bsdf::Bsdf;
use prism_render_architecture::reference_pt::Vec3;

use crate::SurfaceSample;

/// A production material configuration that cannot be represented by the
/// reference BSDF adapter without changing semantics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReferencePtAdapterError {
    /// Every input must be finite, including parameters of disabled lobes.
    NonFiniteParameter,
    /// The adapter accepts authored values, not implicit clamp conversions.
    ParameterOutOfRange,
    /// The adapter currently covers the diffuse substrate clear-coat path only.
    UnsupportedMetallic,
    /// The reference clear-coat lobe currently tints its own highlight with
    /// `coat_color`, while the production coat reserves that tint for the base.
    NonWhiteCoatColor,
    /// Base-layer scaling is not a parameter of the reference clear-coat BSDF.
    UnsupportedBaseWeight,
    /// The reference adapter does not model these production response lobes.
    UnsupportedResponseLobe,
    /// This adapter's comparison domain is limited to a Lambert substrate.
    UnsupportedDiffuseRoughness,
    /// The production coat normal/tangent extensions have no corresponding
    /// field in the isotropic reference clear-coat BSDF.
    UnsupportedCoatFrame,
    /// The production evaluator floors coat roughness at this value, while the
    /// path-traced distribution has a different lower alpha floor.
    CoatRoughnessBelowParityFloor,
    /// No shade-time specular-AA variance is represented by this mapping.
    UnsupportedNormalVariance,
    /// Emission is radiance, not part of the reference BRDF value.
    UnsupportedEmission,
}

/// A validated production-to-reference BSDF mapping.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReferencePtMaterialAdapter {
    bsdf: Bsdf,
}

impl ReferencePtMaterialAdapter {
    /// Builds the supported reference mapping for a production surface.
    ///
    /// The supported slice is a Lambert diffuse substrate under the production
    /// dedicated clear-coat layer: albedo, coat weight and darkening in [0, 1],
    /// coat roughness in [0.045, 1], coat IOR in [1.001, 3], white coat, unit
    /// base weight, zero emission/normal variance and zero additional lobes.
    /// Unsupported weights and frame sentinels must be EXACTLY zero: using an
    /// epsilon here silently discards authored energy, especially at grazing.
    ///
    /// Finite parameters of disabled lobes are dormant, not mapped. AO and
    /// geometry opacity are not BRDF factors; coverage must be checked by the
    /// caller against the response ABI separately. Lights, textures, dynamic
    /// channels, measured tables and the integrator are outside SurfaceSample
    /// and are not validated by this adapter. Acceptance is not a claim of
    /// parity with the production principled coat energy-closure model.
    pub fn from_surface(surface: SurfaceSample) -> Result<Self, ReferencePtAdapterError> {
        // Check dormant fields too: production arithmetic can read them before
        // multiplying by zero, and NaN comparisons must never bypass a guard.
        let scalars = [
            surface.base_weight,
            surface.diffuse_roughness,
            surface.metallic,
            surface.perceptual_roughness,
            surface.specular_weight,
            surface.specular_ior,
            surface.transmission,
            surface.index_of_refraction,
            surface.dispersion,
            surface.thin_walled,
            surface.geometry_opacity,
            surface.transmission_scatter_anisotropy,
            surface.transmission_dispersion_abbe_number,
            surface.coat_darkening,
            surface.scatter_anisotropy,
            surface.ambient_occlusion,
            surface.clearcoat,
            surface.clearcoat_roughness,
            surface.coat_ior,
            surface.coat_roughness_anisotropy,
            surface.sheen,
            surface.fuzz_roughness,
            surface.subsurface,
            surface.subsurface_radius,
            surface.thickness,
            surface.anisotropy,
            surface.anisotropy_rotation,
            surface.normal_variance,
            surface.iridescence,
            surface.iridescence_thickness_nm,
            surface.iridescence_ior,
            surface.glint_weight,
            surface.glint_roughness,
            surface.glint_density,
            surface.glint_bin_half_width,
            surface.glint_state,
            surface.glint_footprint_area,
        ];
        let vectors: &[&[f32]] = &[
            &surface.base_color,
            &surface.specular_color,
            &surface.transmission_color,
            &surface.transmission_scatter,
            &surface.emissive,
            &surface.coat_color,
            &surface.coat_normal,
            &surface.coat_tangent,
            &surface.fuzz_color,
            &surface.subsurface_color,
            &surface.subsurface_radius_scale,
            &surface.glint_cell,
        ];
        if scalars
            .iter()
            .chain(vectors.iter().flat_map(|values| values.iter()))
            .any(|value| !value.is_finite())
        {
            return Err(ReferencePtAdapterError::NonFiniteParameter);
        }
        if !surface
            .base_color
            .iter()
            .all(|value| (0.0..=1.0).contains(value))
            || !(0.0..=1.0).contains(&surface.clearcoat)
            || !(0.0..=1.0).contains(&surface.coat_darkening)
            || !(0.0..=1.0).contains(&surface.geometry_opacity)
            || !(0.0..=1.0).contains(&surface.clearcoat_roughness)
            || !(1.001..=3.0).contains(&surface.coat_ior)
        {
            return Err(ReferencePtAdapterError::ParameterOutOfRange);
        }
        if surface.metallic != 0.0 {
            return Err(ReferencePtAdapterError::UnsupportedMetallic);
        }
        if surface.base_weight != 1.0 {
            return Err(ReferencePtAdapterError::UnsupportedBaseWeight);
        }
        if surface.specular_weight != 0.0
            || surface.transmission != 0.0
            || surface.sheen != 0.0
            || surface.subsurface != 0.0
            || surface.iridescence != 0.0
            || surface.glint_weight != 0.0
        {
            return Err(ReferencePtAdapterError::UnsupportedResponseLobe);
        }
        if surface.diffuse_roughness != 0.0 {
            return Err(ReferencePtAdapterError::UnsupportedDiffuseRoughness);
        }
        if surface.coat_normal.iter().any(|value| *value != 0.0)
            || surface.coat_tangent.iter().any(|value| *value != 0.0)
            || surface.coat_roughness_anisotropy != 0.0
        {
            return Err(ReferencePtAdapterError::UnsupportedCoatFrame);
        }
        if surface.clearcoat_roughness < 0.045 {
            return Err(ReferencePtAdapterError::CoatRoughnessBelowParityFloor);
        }
        if surface.coat_color.iter().any(|value| *value != 1.0) {
            return Err(ReferencePtAdapterError::NonWhiteCoatColor);
        }
        if surface.normal_variance != 0.0 {
            return Err(ReferencePtAdapterError::UnsupportedNormalVariance);
        }
        if surface.emissive.iter().any(|value| *value != 0.0) {
            return Err(ReferencePtAdapterError::UnsupportedEmission);
        }

        Ok(Self {
            bsdf: Bsdf::ClearcoatDiffuse {
                albedo: Vec3::from_array(surface.base_color),
                sigma: 0.0,
                coat_roughness: surface.clearcoat_roughness,
                coat_ior: surface.coat_ior,
                coat_weight: surface.clearcoat,
                coat_color: Vec3::ONE,
                coat_darkening: surface.coat_darkening,
            },
        })
    }

    /// Returns the validated reference BSDF.
    #[must_use]
    pub const fn bsdf(&self) -> Bsdf {
        self.bsdf
    }

    /// Evaluates the independent reference BRDF (no cosine, light or coverage).
    /// Directions and the normal must be finite unit vectors pointing away
    /// from the surface. This calls only architecture/reference_pt code.
    #[must_use]
    pub fn evaluate(&self, view: [f32; 3], light: [f32; 3], normal: [f32; 3]) -> [f32; 3] {
        self.bsdf
            .evaluate(
                Vec3::from_array(view),
                Vec3::from_array(light),
                Vec3::from_array(normal),
            )
            .to_array()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_architecture::reference_pt::compare::compare_pixels;

    use crate::{evaluate_clearcoat_direct, DirectLightSample, ShadingFrame};

    const NORMAL: [f32; 3] = [0.0, 1.0, 0.0];

    fn supported_surface() -> SurfaceSample {
        SurfaceSample {
            base_color: [0.25, 0.5, 0.75],
            specular_weight: 0.0,
            clearcoat: 1.0,
            clearcoat_roughness: 0.2,
            ..Default::default()
        }
    }

    fn normalize(vector: [f32; 3]) -> [f32; 3] {
        let length = (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt();
        [vector[0] / length, vector[1] / length, vector[2] / length]
    }

    fn frame(view: [f32; 3]) -> ShadingFrame {
        ShadingFrame {
            normal: NORMAL,
            view,
            tangent: [1.0, 0.0, 0.0],
            bitangent: [0.0, 0.0, -1.0],
        }
    }

    const FIXTURE_WIDTH: usize = 16;
    const FIXTURE_HEIGHT: usize = 16;
    const FIXTURE_PIXELS: usize = FIXTURE_WIDTH * FIXTURE_HEIGHT;

    fn fixture_pixels(surface: SurfaceSample) -> ([Vec3; FIXTURE_PIXELS], [Vec3; FIXTURE_PIXELS]) {
        let adapter = ReferencePtMaterialAdapter::from_surface(surface).expect("supported fixture");
        let mut production = [Vec3::ZERO; FIXTURE_PIXELS];
        let mut reference = [Vec3::ZERO; FIXTURE_PIXELS];
        for y in 0..FIXTURE_HEIGHT {
            // Sweep both hemispheres away from the near-tangent singularity.
            let view_cosine = 0.12 + 0.84 * y as f32 / (FIXTURE_HEIGHT - 1) as f32;
            let view = normalize([0.6, view_cosine, 0.2]);
            for x in 0..FIXTURE_WIDTH {
                let light_cosine = 0.12 + 0.84 * x as f32 / (FIXTURE_WIDTH - 1) as f32;
                let light = normalize([
                    0.45 * (1.0 - 2.0 * x as f32 / (FIXTURE_WIDTH - 1) as f32),
                    light_cosine,
                    -0.25 + 0.5 * y as f32 / (FIXTURE_HEIGHT - 1) as f32,
                ]);
                let index = y * FIXTURE_WIDTH + x;
                let direct = evaluate_clearcoat_direct(
                    surface,
                    frame(view),
                    DirectLightSample {
                        direction: light,
                        illuminance: [1.0; 3],
                        visibility: 1.0,
                    },
                );
                let cos_i = light[1].max(0.0);
                production[index] = Vec3::from_array(direct);
                reference[index] =
                    Vec3::from_array(adapter.evaluate(view, light, NORMAL)).scale(cos_i);
            }
        }
        (production, reference)
    }

    #[test]
    fn adapter_rejects_unrepresentable_production_lobes() {
        assert_eq!(
            ReferencePtMaterialAdapter::from_surface(SurfaceSample {
                metallic: 1.0,
                ..SurfaceSample::default()
            }),
            Err(ReferencePtAdapterError::UnsupportedMetallic)
        );
        assert_eq!(
            ReferencePtMaterialAdapter::from_surface(SurfaceSample {
                coat_color: [1.0, 0.8, 1.0],
                specular_weight: 0.0,
                clearcoat_roughness: 0.2,
                ..SurfaceSample::default()
            }),
            Err(ReferencePtAdapterError::NonWhiteCoatColor)
        );
        assert_eq!(
            ReferencePtMaterialAdapter::from_surface(SurfaceSample {
                diffuse_roughness: 0.35,
                specular_weight: 0.0,
                clearcoat_roughness: 0.2,
                ..SurfaceSample::default()
            }),
            Err(ReferencePtAdapterError::UnsupportedDiffuseRoughness)
        );
    }

    #[test]
    fn adapter_rejects_nonfinite_active_and_dormant_parameters() {
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for surface in [
                SurfaceSample {
                    base_color: [invalid, 0.5, 0.75],
                    ..supported_surface()
                },
                SurfaceSample {
                    clearcoat: invalid,
                    ..supported_surface()
                },
                SurfaceSample {
                    coat_darkening: invalid,
                    ..supported_surface()
                },
                SurfaceSample {
                    coat_ior: invalid,
                    ..supported_surface()
                },
                SurfaceSample {
                    coat_normal: [0.0, invalid, 0.0],
                    ..supported_surface()
                },
                SurfaceSample {
                    specular_weight: invalid,
                    ..supported_surface()
                },
                SurfaceSample {
                    glint_cell: [invalid, 0.0],
                    ..supported_surface()
                },
                SurfaceSample {
                    transmission_color: [invalid; 3],
                    ..supported_surface()
                },
                SurfaceSample {
                    ambient_occlusion: invalid,
                    ..supported_surface()
                },
            ] {
                assert_eq!(
                    ReferencePtMaterialAdapter::from_surface(surface),
                    Err(ReferencePtAdapterError::NonFiniteParameter)
                );
            }
        }
    }

    #[test]
    fn adapter_rejects_implicit_clamps_and_small_unsupported_contributions() {
        use ReferencePtAdapterError as E;
        let base = supported_surface();
        for (surface, error) in [
            (
                SurfaceSample {
                    base_color: [-0.01, 0.5, 0.75],
                    ..base
                },
                E::ParameterOutOfRange,
            ),
            (
                SurfaceSample {
                    base_color: [1.01, 0.5, 0.75],
                    ..base
                },
                E::ParameterOutOfRange,
            ),
            (
                SurfaceSample {
                    clearcoat: 1.01,
                    ..base
                },
                E::ParameterOutOfRange,
            ),
            (
                SurfaceSample {
                    coat_darkening: -0.01,
                    ..base
                },
                E::ParameterOutOfRange,
            ),
            (
                SurfaceSample {
                    clearcoat_roughness: 1.01,
                    ..base
                },
                E::ParameterOutOfRange,
            ),
            (
                SurfaceSample {
                    clearcoat_roughness: 0.044,
                    ..base
                },
                E::CoatRoughnessBelowParityFloor,
            ),
            (
                SurfaceSample {
                    coat_ior: 1.0,
                    ..base
                },
                E::ParameterOutOfRange,
            ),
            (
                SurfaceSample {
                    coat_ior: 3.01,
                    ..base
                },
                E::ParameterOutOfRange,
            ),
            (
                SurfaceSample {
                    geometry_opacity: -0.01,
                    ..base
                },
                E::ParameterOutOfRange,
            ),
            (
                SurfaceSample {
                    metallic: 1.0e-8,
                    ..base
                },
                E::UnsupportedMetallic,
            ),
            (
                SurfaceSample {
                    base_weight: 0.5,
                    ..base
                },
                E::UnsupportedBaseWeight,
            ),
            (
                SurfaceSample {
                    specular_weight: 1.0e-8,
                    ..base
                },
                E::UnsupportedResponseLobe,
            ),
            (
                SurfaceSample {
                    transmission: 1.0e-8,
                    ..base
                },
                E::UnsupportedResponseLobe,
            ),
            (
                SurfaceSample {
                    sheen: 1.0e-8,
                    ..base
                },
                E::UnsupportedResponseLobe,
            ),
            (
                SurfaceSample {
                    subsurface: 1.0e-8,
                    ..base
                },
                E::UnsupportedResponseLobe,
            ),
            (
                SurfaceSample {
                    iridescence: 1.0e-8,
                    ..base
                },
                E::UnsupportedResponseLobe,
            ),
            (
                SurfaceSample {
                    glint_weight: 1.0e-8,
                    ..base
                },
                E::UnsupportedResponseLobe,
            ),
            (
                SurfaceSample {
                    diffuse_roughness: 1.0e-8,
                    ..base
                },
                E::UnsupportedDiffuseRoughness,
            ),
            (
                SurfaceSample {
                    coat_normal: [1.0e-8, 0.0, 0.0],
                    ..base
                },
                E::UnsupportedCoatFrame,
            ),
            (
                SurfaceSample {
                    coat_tangent: [1.0e-8, 0.0, 0.0],
                    ..base
                },
                E::UnsupportedCoatFrame,
            ),
            (
                SurfaceSample {
                    coat_roughness_anisotropy: 1.0e-8,
                    ..base
                },
                E::UnsupportedCoatFrame,
            ),
            (
                SurfaceSample {
                    coat_color: [0.5, 1.0, 1.0],
                    ..base
                },
                E::NonWhiteCoatColor,
            ),
            (
                SurfaceSample {
                    normal_variance: 1.0e-8,
                    ..base
                },
                E::UnsupportedNormalVariance,
            ),
            (
                SurfaceSample {
                    emissive: [0.0, 1.0e-8, 0.0],
                    ..base
                },
                E::UnsupportedEmission,
            ),
        ] {
            assert_eq!(
                ReferencePtMaterialAdapter::from_surface(surface),
                Err(error),
                "{surface:?}"
            );
        }
    }

    #[test]
    fn adapter_maps_parameters_without_reusing_a_production_bsdf() {
        let surface = SurfaceSample {
            clearcoat: 0.6,
            clearcoat_roughness: 0.35,
            coat_ior: 1.7,
            coat_darkening: 0.4,
            geometry_opacity: 0.375,
            ..supported_surface()
        };
        let adapter = ReferencePtMaterialAdapter::from_surface(surface).unwrap();
        assert_eq!(
            adapter.bsdf(),
            Bsdf::ClearcoatDiffuse {
                albedo: Vec3::from_array(surface.base_color),
                sigma: 0.0,
                coat_roughness: 0.35,
                coat_ior: 1.7,
                coat_weight: 0.6,
                coat_color: Vec3::ONE,
                coat_darkening: 0.4,
            }
        );
        // Coverage must not accidentally scale the BRDF.
        let opaque = ReferencePtMaterialAdapter::from_surface(SurfaceSample {
            geometry_opacity: 1.0,
            ..surface
        })
        .unwrap();
        assert_eq!(
            adapter.evaluate(NORMAL, NORMAL, NORMAL),
            opaque.evaluate(NORMAL, NORMAL, NORMAL)
        );
    }

    #[test]
    fn reference_coat_off_has_an_analytic_lambert_anchor() {
        for darkening in [0.0, 1.0] {
            let surface = SurfaceSample {
                clearcoat: 0.0,
                coat_darkening: darkening,
                ..supported_surface()
            };
            let adapter = ReferencePtMaterialAdapter::from_surface(surface).unwrap();
            for view in [NORMAL, normalize([0.8, 0.2, 0.3])] {
                for light in [NORMAL, normalize([-0.4, 0.7, 0.2])] {
                    let value = adapter.evaluate(view, light, NORMAL);
                    for channel in 0..3 {
                        assert!(
                            (value[channel] - surface.base_color[channel] / std::f32::consts::PI)
                                .abs()
                                < 1.0e-6
                        );
                    }
                }
            }
            assert_eq!(adapter.evaluate(NORMAL, [0.0, -1.0, 0.0], NORMAL), [0.0; 3]);
        }
    }

    #[test]
    fn dedicated_cpu_clearcoat_candidate_matches_reference_on_a_small_image() {
        // CPU-only 16x16 local-model fixture. This does NOT execute WESL or
        // validate the full shading_resolve path, whose principled coat closure
        // is a different model.
        for (coat, darkening) in [
            (0.0, 0.0),
            (0.0, 1.0),
            (0.35, 0.0),
            (0.35, 0.5),
            (0.35, 1.0),
            (1.0, 0.0),
            (1.0, 0.5),
            (1.0, 1.0),
        ] {
            let surface = SurfaceSample {
                base_color: [0.25, 0.5, 0.75],
                diffuse_roughness: 0.0,
                clearcoat: coat,
                clearcoat_roughness: 0.2,
                coat_ior: 1.6,
                coat_darkening: darkening,
                specular_weight: 0.0,
                ..SurfaceSample::default()
            };
            let (candidate, reference) = fixture_pixels(surface);
            for pixel in candidate.iter().chain(&reference) {
                assert!(pixel
                    .to_array()
                    .iter()
                    .all(|value| value.is_finite() && *value >= 0.0));
            }
            let metrics = compare_pixels(&reference, &candidate).expect("same fixture shape");
            assert!(metrics.max_abs < 5.0e-5,
                "16x16 coat/base/transmission image mismatch: coat={coat} darkening={darkening} metrics={metrics:?}");
        }
    }

    #[test]
    fn clearcoat_wesl_keeps_the_same_layering_and_darkening_contract() {
        // This is a source contract, not shader execution or GPU image parity.
        // The CPU/reference image fixture above is restricted to the dedicated
        // clearcoat path; ensure the corresponding WESL path retains the same
        // independent coat lobe and base-only double-pass/darkening operations.
        let wesl = include_str!("../../prism_render_scene/src/shaders/clearcoat.wesl");
        let throughput = wesl
            .find("let throughput = (1.0 - coat * fresnel_dielectric(n_dot_v, coat_ior))")
            .expect("WESL view/light double-pass throughput exists");
        let darkening = wesl
            .find("let darkening = cc_modulated_darkening(")
            .expect("WESL coat darkening factor exists");
        let base_attenuation = wesl
            .find("let reflected = tinted_base * (throughput * darkening) + coat_specular")
            .expect("WESL applies throughput and darkening to base, not coat lobe");
        assert!(throughput < darkening && darkening < base_attenuation,
            "WESL clearcoat evaluation order changed; inspect the shader before trusting the CPU/reference fixture");
    }

    #[test]
    fn material_transmission_is_rejected_instead_of_dropped_from_reference_image() {
        for transmission in [f32::EPSILON, 0.25, 1.0] {
            let surface = SurfaceSample {
                transmission,
                clearcoat_roughness: 0.2,
                ..SurfaceSample::default()
            };
            assert_eq!(
                ReferencePtMaterialAdapter::from_surface(surface),
                Err(ReferencePtAdapterError::UnsupportedResponseLobe),
                "transmission={transmission} must not be silently omitted from a reference image"
            );
        }
    }
}
