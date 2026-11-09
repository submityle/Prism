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
use prism_render_architecture::reference_pt::openpbr::OpenPbrMaterial;
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
    /// The adapter has no tangent frame for anisotropic base response.
    UnsupportedAnisotropy,
    /// The production coat normal/tangent extensions have no corresponding
    /// field in the isotropic reference clear-coat BSDF.
    UnsupportedCoatFrame,
    /// The production evaluator floors coat roughness at this value, while the
    /// path-traced distribution has a different lower alpha floor.
    CoatRoughnessBelowParityFloor,
    /// No shade-time specular-AA variance is represented by this mapping.
    UnsupportedNormalVariance,
    /// Coverage is owned by the caller's visibility/alpha path, not the BRDF.
    UnsupportedGeometryOpacity,
    /// Ambient occlusion is an external visibility factor, not a BRDF term.
    UnsupportedAmbientOcclusion,
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
    /// The supported slice is the isotropic, front-facing OpenPBR response:
    /// base/EON, metallic/specular, transmission, coat, sheen, subsurface,
    /// iridescence and glint. Emission, normal variance, anisotropic frames,
    /// coverage and AO remain outside the BRDF oracle and are rejected.
    /// Unsupported frame sentinels must be EXACTLY zero: using an epsilon here
    /// silently changes authored energy, especially at grazing.
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
            || !(0.0..=1.0).contains(&surface.base_weight)
            || !(0.0..=1.0).contains(&surface.diffuse_roughness)
            || !surface
                .coat_color
                .iter()
                .all(|value| (0.0..=1.0).contains(value))
            || !(0.0..=1.0).contains(&surface.clearcoat)
            || !(0.0..=1.0).contains(&surface.coat_darkening)
            || !(0.0..=1.0).contains(&surface.geometry_opacity)
            || !(0.0..=1.0).contains(&surface.clearcoat_roughness)
            || !(1.001..=3.0).contains(&surface.coat_ior)
            || !(0.0..=1.0).contains(&surface.metallic)
            || !(0.0..=1.0).contains(&surface.perceptual_roughness)
            || !(0.0..=1.0).contains(&surface.specular_weight)
            || !surface
                .specular_color
                .iter()
                .all(|value| (0.0..=1.0).contains(value))
            || !(1.001..=3.0).contains(&surface.specular_ior)
            || !(0.0..=1.0).contains(&surface.transmission)
            || !(1.0..=3.0).contains(&surface.index_of_refraction)
            || !(0.0..=1.0).contains(&surface.dispersion)
            || !(0.0..=1.0).contains(&surface.thin_walled)
            || !surface
                .transmission_color
                .iter()
                .all(|value| (0.0..=1.0).contains(value))
            || !surface
                .transmission_scatter
                .iter()
                .all(|value| (0.0..=1.0).contains(value))
            || !(-1.0..=1.0).contains(&surface.transmission_scatter_anisotropy)
            || !(1.0..=200.0).contains(&surface.transmission_dispersion_abbe_number)
            || !(-1.0..=1.0).contains(&surface.scatter_anisotropy)
            || !(0.0..=1.0).contains(&surface.sheen)
            || !(0.0..=1.0).contains(&surface.fuzz_roughness)
            || !surface
                .fuzz_color
                .iter()
                .all(|value| (0.0..=1.0).contains(value))
            || !(0.0..=1.0).contains(&surface.subsurface)
            || !surface
                .subsurface_color
                .iter()
                .all(|value| (0.0..=1.0).contains(value))
            || surface.subsurface_radius < 0.0
            || !surface
                .subsurface_radius_scale
                .iter()
                .all(|value| (0.0..=1.0).contains(value))
            || surface.thickness < 0.0
            || !(0.0..=1.0).contains(&surface.iridescence)
            || surface.iridescence_thickness_nm < 0.0
            || !(1.0..=3.0).contains(&surface.iridescence_ior)
            || !(0.0..=1.0).contains(&surface.glint_weight)
            || !(0.0..=1.0).contains(&surface.glint_roughness)
            || surface.glint_density < 0.0
            || !(0.0..=core::f32::consts::PI).contains(&surface.glint_bin_half_width)
            || surface.glint_footprint_area < 0.0
            || (surface.glint_state >= 0.0 && surface.glint_state > 1.0)
        {
            return Err(ReferencePtAdapterError::ParameterOutOfRange);
        }
        if surface.geometry_opacity != 1.0 {
            return Err(ReferencePtAdapterError::UnsupportedGeometryOpacity);
        }
        if surface.ambient_occlusion != 1.0 {
            return Err(ReferencePtAdapterError::UnsupportedAmbientOcclusion);
        }
        if surface.anisotropy != 0.0 || surface.anisotropy_rotation != 0.0 {
            return Err(ReferencePtAdapterError::UnsupportedAnisotropy);
        }
        if surface.coat_normal.iter().any(|value| *value != 0.0)
            || surface.coat_tangent.iter().any(|value| *value != 0.0)
            || surface.coat_roughness_anisotropy != 0.0
        {
            return Err(ReferencePtAdapterError::UnsupportedCoatFrame);
        }
        if surface.clearcoat > 0.0 && surface.clearcoat_roughness < 0.045 {
            return Err(ReferencePtAdapterError::CoatRoughnessBelowParityFloor);
        }
        if surface.normal_variance != 0.0 {
            return Err(ReferencePtAdapterError::UnsupportedNormalVariance);
        }
        if surface.emissive.iter().any(|value| *value != 0.0) {
            return Err(ReferencePtAdapterError::UnsupportedEmission);
        }

        let uses_openpbr = surface.metallic != 0.0
            || surface.specular_weight != 0.0
            || surface.transmission != 0.0
            || surface.sheen != 0.0
            || surface.subsurface != 0.0
            || surface.iridescence != 0.0
            || surface.glint_weight != 0.0;
        let bsdf = if uses_openpbr {
            Bsdf::OpenPbr {
                material: OpenPbrMaterial {
                    base_color: Vec3::from_array(surface.base_color),
                    base_weight: surface.base_weight,
                    diffuse_roughness: surface.diffuse_roughness,
                    metallic: surface.metallic,
                    perceptual_roughness: surface.perceptual_roughness,
                    specular_weight: surface.specular_weight,
                    specular_color: Vec3::from_array(surface.specular_color),
                    specular_ior: surface.specular_ior,
                    transmission: surface.transmission,
                    index_of_refraction: surface.index_of_refraction,
                    dispersion: surface.dispersion,
                    thin_walled: surface.thin_walled,
                    transmission_color: Vec3::from_array(surface.transmission_color),
                    transmission_scatter: Vec3::from_array(surface.transmission_scatter),
                    transmission_scatter_anisotropy: surface.transmission_scatter_anisotropy,
                    transmission_dispersion_abbe_number: surface.transmission_dispersion_abbe_number,
                    coat_darkening: surface.coat_darkening,
                    scatter_anisotropy: surface.scatter_anisotropy,
                    clearcoat: surface.clearcoat,
                    clearcoat_roughness: surface.clearcoat_roughness,
                    coat_ior: surface.coat_ior,
                    coat_color: Vec3::from_array(surface.coat_color),
                    sheen: surface.sheen,
                    fuzz_roughness: surface.fuzz_roughness,
                    fuzz_color: Vec3::from_array(surface.fuzz_color),
                    subsurface: surface.subsurface,
                    subsurface_color: Vec3::from_array(surface.subsurface_color),
                    subsurface_radius: surface.subsurface_radius,
                    subsurface_radius_scale: Vec3::from_array(surface.subsurface_radius_scale),
                    thickness: surface.thickness,
                    iridescence: surface.iridescence,
                    iridescence_thickness_nm: surface.iridescence_thickness_nm,
                    iridescence_ior: surface.iridescence_ior,
                    glint_weight: surface.glint_weight,
                    glint_roughness: surface.glint_roughness,
                    glint_density: surface.glint_density,
                    glint_bin_half_width: surface.glint_bin_half_width,
                    glint_state: surface.glint_state,
                    glint_cell: surface.glint_cell,
                    glint_footprint_area: surface.glint_footprint_area,
                },
            }
        } else {
            Bsdf::ClearcoatDiffuse {
                albedo: Vec3::from_array(surface.base_color),
                sigma: surface.diffuse_roughness * core::f32::consts::FRAC_PI_2,
                base_weight: surface.base_weight,
                coat_roughness: surface.clearcoat_roughness.max(0.045),
                coat_ior: surface.coat_ior,
                coat_weight: surface.clearcoat,
                coat_color: Vec3::from_array(surface.coat_color),
                coat_darkening: surface.coat_darkening,
            }
        };
        Ok(Self { bsdf })
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
    fn eon_rough_diffuse_standalone_matches_the_independent_reference() {
        // Keep the coat disabled so this gate isolates the EON compensation
        // from clearcoat transmission, darkening, and the coat highlight.
        let surface = SurfaceSample {
            base_color: [0.91, 0.63, 0.27],
            base_weight: 0.73,
            diffuse_roughness: 0.82,
            metallic: 0.0,
            specular_weight: 0.0,
            clearcoat: 0.0,
            clearcoat_roughness: 0.2,
            ..SurfaceSample::default()
        };
        let adapter = ReferencePtMaterialAdapter::from_surface(surface).expect("EON slice");
        let view = normalize([0.42, 0.71, 0.56]);
        let light = normalize([-0.37, 0.86, 0.35]);
        let production = evaluate_clearcoat_direct(
            surface,
            frame(view),
            DirectLightSample {
                direction: light,
                illuminance: [1.0; 3],
                visibility: 1.0,
            },
        );
        let cosine_weighted_reference =
            Vec3::from_array(adapter.evaluate(view, light, NORMAL)).scale(light[1].max(0.0));
        let production = Vec3::from_array(production);
        let error = production.sub(cosine_weighted_reference);
        assert!(
            error.length() < 1.0e-5,
            "standalone EON parity mismatch: production={production:?} reference={cosine_weighted_reference:?} error={error:?}"
        );

        let oren_nayar = Bsdf::OrenNayar {
            albedo: Vec3::from_array(surface.base_color),
            sigma: surface.diffuse_roughness * core::f32::consts::FRAC_PI_2,
        };
        let legacy = oren_nayar
            .evaluate(
                Vec3::from_array(view),
                Vec3::from_array(light),
                Vec3::from_array(NORMAL),
            )
            .scale(light[1].max(0.0) * surface.base_weight);
        assert!(
            production.max_component() > legacy.max_component(),
            "rough diffuse parity must include a non-zero EON compensation: production={production:?} legacy={legacy:?}"
        );
    }

    #[test]
    fn adapter_accepts_metallic_and_rejects_unrepresentable_production_lobes() {
        assert!(ReferencePtMaterialAdapter::from_surface(SurfaceSample {
            metallic: 1.0,
            clearcoat_roughness: 0.2,
            ..SurfaceSample::default()
        })
        .is_ok());
        assert!(ReferencePtMaterialAdapter::from_surface(SurfaceSample {
            coat_color: [1.0, 0.8, 1.0],
            specular_weight: 0.0,
            clearcoat_roughness: 0.2,
            ..SurfaceSample::default()
        })
        .is_ok());
        assert!(ReferencePtMaterialAdapter::from_surface(SurfaceSample {
            diffuse_roughness: 0.37,
            specular_weight: 0.0,
            clearcoat_roughness: 0.2,
            ..SurfaceSample::default()
        })
        .is_ok());
        assert!(ReferencePtMaterialAdapter::from_surface(SurfaceSample {
            diffuse_roughness: 0.35,
            specular_weight: 0.0,
            clearcoat_roughness: 0.2,
            ..SurfaceSample::default()
        })
        .is_ok());
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
    fn adapter_rejects_implicit_clamps_and_unsupported_context() {
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
            base_weight: 0.7,
            diffuse_roughness: 0.0,
            coat_color: [0.9, 0.8, 0.7],
            clearcoat: 0.6,
            clearcoat_roughness: 0.35,
            coat_ior: 1.7,
            coat_darkening: 0.4,
            geometry_opacity: 1.0,
            ..supported_surface()
        };
        let adapter = ReferencePtMaterialAdapter::from_surface(surface).unwrap();
        assert_eq!(
            adapter.bsdf(),
            Bsdf::ClearcoatDiffuse {
                albedo: Vec3::from_array(surface.base_color),
                sigma: surface.diffuse_roughness * core::f32::consts::FRAC_PI_2,
                base_weight: surface.base_weight,
                coat_roughness: 0.35,
                coat_ior: 1.7,
                coat_weight: 0.6,
                coat_color: Vec3::from_array(surface.coat_color),
                coat_darkening: 0.4,
            }
        );
        assert!(adapter.evaluate(NORMAL, NORMAL, NORMAL).iter().all(|value| value.is_finite()));
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
    fn layered_reference_consumes_base_weight_and_coat_medium_tint() {
        let surface = SurfaceSample {
            base_color: [0.6, 0.45, 0.3],
            base_weight: 0.7,
            diffuse_roughness: 0.0,
            clearcoat: 0.8,
            clearcoat_roughness: 0.2,
            coat_color: [0.25, 0.7, 1.0],
            specular_weight: 0.0,
            ..Default::default()
        };
        let adapter = ReferencePtMaterialAdapter::from_surface(surface).unwrap();
        let value = adapter.evaluate(
            normalize([0.3, 0.8, 0.2]),
            normalize([-0.2, 0.9, 0.3]),
            NORMAL,
        );
        assert!(value
            .iter()
            .all(|channel| channel.is_finite() && *channel >= 0.0));

        let no_base = ReferencePtMaterialAdapter::from_surface(SurfaceSample {
            base_weight: 0.0,
            ..surface
        })
        .unwrap();
        let white_coat = ReferencePtMaterialAdapter::from_surface(SurfaceSample {
            coat_color: [1.0; 3],
            ..surface
        })
        .unwrap();
        let no_base_value = no_base.evaluate(NORMAL, NORMAL, NORMAL);
        let white_value = white_coat.evaluate(NORMAL, NORMAL, NORMAL);
        let tinted_value = adapter.evaluate(NORMAL, NORMAL, NORMAL);
        assert!(no_base_value.iter().all(|channel| *channel > 0.0));
        assert!(tinted_value[0] < white_value[0]);
        assert!(tinted_value[1] < white_value[1]);
        assert!((tinted_value[2] - white_value[2]).abs() < 1.0e-6);
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
                diffuse_roughness: 0.08,
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
    fn cpu_reference_parity_covers_every_mapped_material_parameter() {
        // This is the parity gate for the complete adapter-supported material
        // slice.  The fixture is deterministic, but it is not a fallback
        // material: every row changes at least one authored parameter that is
        // copied into Bsdf::ClearcoatDiffuse, and every pixel evaluates the
        // production clearcoat candidate against the independent reference
        // BSDF with the same directions and unit illuminance.
        let materials = [
            SurfaceSample {
                base_color: [0.11, 0.37, 0.83],
                base_weight: 0.23,
                diffuse_roughness: 0.31,
                clearcoat: 0.0,
                clearcoat_roughness: 0.045,
                coat_ior: 1.001,
                coat_color: [0.91, 0.73, 0.42],
                coat_darkening: 0.0,
                specular_weight: 0.0,
                ..SurfaceSample::default()
            },
            SurfaceSample {
                base_color: [0.72, 0.18, 0.09],
                base_weight: 0.61,
                diffuse_roughness: 0.67,
                clearcoat: 0.31,
                clearcoat_roughness: 0.19,
                coat_ior: 1.37,
                coat_color: [0.86, 0.94, 0.67],
                coat_darkening: 0.28,
                specular_weight: 0.0,
                ..SurfaceSample::default()
            },
            SurfaceSample {
                base_color: [0.93, 0.64, 0.21],
                base_weight: 1.0,
                diffuse_roughness: 1.0,
                clearcoat: 0.67,
                clearcoat_roughness: 0.52,
                coat_ior: 1.76,
                coat_color: [0.39, 0.78, 0.97],
                coat_darkening: 0.64,
                specular_weight: 0.0,
                ..SurfaceSample::default()
            },
            SurfaceSample {
                base_color: [0.04, 0.06, 0.08],
                base_weight: 0.88,
                diffuse_roughness: 0.0,
                clearcoat: 1.0,
                clearcoat_roughness: 1.0,
                coat_ior: 2.87,
                coat_color: [0.22, 0.51, 0.79],
                coat_darkening: 1.0,
                specular_weight: 0.0,
                ..SurfaceSample::default()
            },
        ];

        for (material_index, surface) in materials.into_iter().enumerate() {
            let (candidate, reference) = fixture_pixels(surface);
            for (pixel_index, (candidate, reference)) in
                candidate.iter().zip(reference.iter()).enumerate()
            {
                assert!(
                    candidate
                        .to_array()
                        .into_iter()
                        .chain(reference.to_array())
                        .all(|value| value.is_finite() && value >= 0.0),
                    "non-finite/negative response at material={material_index} pixel={pixel_index}"
                );
            }
            let metrics = compare_pixels(&reference, &candidate)
                .expect("production/reference fixtures must have identical dimensions");
            assert!(
                metrics.max_abs < 1.0e-4,
                "mapped material parameter parity mismatch: material={material_index} metrics={metrics:?} surface={surface:?}"
            );
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
    fn material_transmission_is_mapped_instead_of_dropped_from_reference_image() {
        for transmission in [f32::EPSILON, 0.25, 1.0] {
            let surface = SurfaceSample {
                transmission,
                clearcoat_roughness: 0.2,
                ..SurfaceSample::default()
            };
            let adapter = ReferencePtMaterialAdapter::from_surface(surface)
                .expect("transmission has an independent reference implementation");
            assert!(adapter.evaluate(NORMAL, NORMAL, NORMAL).iter().all(|value| value.is_finite()));
        }
    }
}
