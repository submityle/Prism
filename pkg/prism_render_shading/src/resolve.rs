//! Backend-neutral CPU reference for the compute resolve pass.
//!
//! This closes the visibility -> classify -> shade chain: it reconstructs a
//! surface from one [`VisibilityPixel`], samples the material parameters, and
//! evaluates direct lighting per shading class.  It is deliberately free of any
//! GPU or Bevy dependency so it can act as the golden reference the WESL
//! resolve shader must match bit-for-bit within tolerance.

use prism_render_material::{GpuMaterialHeader, GpuSurfaceParameters};

use crate::{
    apply_tangent_space_normal, classify_material_header, evaluate_image_based_light,
    reconstruct_surface, sample_material, ClassificationError, DirectLightSample, GpuShadingPrimitive,
    GpuShadingVertex, ImageBasedLight, MaterialModulationParams, MaterialShadingClass, PunctualLight,
    SampledTextureBinding, ShadingFrame, SurfaceReconstructionError, SurfaceReconstructionFlags,
    SurfaceReconstructionInput, SurfaceSample, TangentBasis, VisibilityPixel,
};

/// One analytic directional light expressed in world space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DirectionalLight {
    /// Unit vector pointing from the surface toward the light.
    pub direction: [f32; 3],
    /// Linear illuminance (already pre-exposed) contributed at the surface.
    pub illuminance: [f32; 3],
    /// Analytic shadow visibility term in `[0, 1]`.
    pub visibility: f32,
}

impl Default for DirectionalLight {
    fn default() -> Self {
        Self {
            direction: [0.0, 1.0, 0.0],
            illuminance: [0.0; 3],
            visibility: 1.0,
        }
    }
}

/// The lighting environment sampled by [`resolve_pixel`].
#[derive(Clone, Copy, Debug)]
pub struct LightingEnvironment<'a> {
    /// Analytic directional lights accumulated for every shading class.
    pub directional: &'a [DirectionalLight],
    /// Punctual (point and spot) lights accumulated for every shading class.
    pub punctual: &'a [PunctualLight],
    /// Optional image-based light supplying the indirect term.  When present it
    /// replaces the constant [`Self::ambient`] approximation.
    pub image_based: Option<ImageBasedLight>,
    /// Constant ambient irradiance approximating unresolved indirect light.
    pub ambient: [f32; 3],
    /// Quantization band count used by the non-photoreal toon path.
    pub toon_bands: u32,
}

impl Default for LightingEnvironment<'_> {
    fn default() -> Self {
        Self {
            directional: &[],
            punctual: &[],
            image_based: None,
            ambient: [0.0; 3],
            toon_bands: 4,
        }
    }
}

/// Everything required to resolve a single shaded pixel.
#[derive(Clone, Copy, Debug)]
pub struct ResolveInput<'a> {
    pub pixel: VisibilityPixel,
    pub primitives: &'a [GpuShadingPrimitive],
    pub vertices: &'a [GpuShadingVertex],
    /// Generation of the geometry table the primitives were built from.
    pub geometry_generation: u32,
    /// Generation the resolving view expects; guards against recycled slots.
    pub expected_geometry_generation: u32,
    pub header: GpuMaterialHeader,
    pub parameters: GpuSurfaceParameters,
    /// Sampled texels for every texture the material binds, in the same
    /// `texture_offset..texture_offset + texture_count` order the GPU heap
    /// iterates.  Each entry carries its `TextureSemantic` and the RGBA texel
    /// sampled at the interpolated surface UV; an empty slice shades from the
    /// authored factors alone (the bindless heap's white/flat-normal defaults).
    pub textures: &'a [SampledTextureBinding],
    /// World-space camera position used to derive the view vector.
    pub view_position: [f32; 3],
}

/// The linear HDR result of resolving one pixel plus its provenance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResolvedPixel {
    pub color: [f32; 3],
    pub shading_class: MaterialShadingClass,
    pub surface_flags: SurfaceReconstructionFlags,
}

/// Conditions that prevent a pixel from being shaded.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResolveError {
    /// The visibility pixel did not reference live scene/geometry/material ids.
    InvalidPixel,
    /// The material generation the pixel captured no longer matches the header.
    StaleMaterial,
    /// The material could not be mapped to a lit surface shading class.
    Classification(ClassificationError),
    /// The surface could not be reconstructed from the geometry tables.
    Surface(SurfaceReconstructionError),
}

/// Maps stable GPU surface parameters onto the analytic BSDF sample.
pub fn surface_sample_from_parameters(parameters: &GpuSurfaceParameters) -> SurfaceSample {
    SurfaceSample {
        base_color: [
            parameters.base_color[0],
            parameters.base_color[1],
            parameters.base_color[2],
        ],
        metallic: parameters.metallic,
        perceptual_roughness: parameters.perceptual_roughness,
        reflectance: parameters.reflectance,
        ambient_occlusion: parameters.ambient_occlusion,
        emissive: [
            parameters.emissive[0],
            parameters.emissive[1],
            parameters.emissive[2],
        ],
        clearcoat: parameters.clearcoat,
        clearcoat_roughness: parameters.clearcoat_roughness,
        sheen: parameters.sheen,
        subsurface: parameters.subsurface,
        thickness: parameters.thickness,
        anisotropy: parameters.anisotropy,
        anisotropy_rotation: parameters.anisotropy_rotation,
    }
}

/// Projects the stable GPU surface parameters onto the texture-modulation
/// factors consumed by [`sample_material`].  Only the fields a texture can
/// scale are carried; the rest of the surface (reflectance, sheen, subsurface,
/// anisotropy, ...) is read straight from the parameters.  Mirrors
/// `sampled_material_defaults`'s view of `PrismSurfaceParameters` in
/// `material_sample.wesl`.
fn modulation_params_from(parameters: &GpuSurfaceParameters) -> MaterialModulationParams {
    MaterialModulationParams {
        base_color: parameters.base_color,
        emissive: [
            parameters.emissive[0],
            parameters.emissive[1],
            parameters.emissive[2],
        ],
        metallic: parameters.metallic,
        perceptual_roughness: parameters.perceptual_roughness,
        ambient_occlusion: parameters.ambient_occlusion,
        normal_scale: parameters.normal_scale,
        clearcoat: parameters.clearcoat,
        clearcoat_roughness: parameters.clearcoat_roughness,
    }
}

/// Resolves one pixel into linear HDR scene color.
///
/// Emissive and ambient are added exactly once; the per-light BSDF evaluations
/// run against an emissive-free copy of the surface so accumulation over many
/// lights never multiplies self-illumination.
pub fn resolve_pixel(
    input: ResolveInput<'_>,
    lights: LightingEnvironment<'_>,
) -> Result<ResolvedPixel, ResolveError> {
    if !input.pixel.is_valid() {
        return Err(ResolveError::InvalidPixel);
    }
    if input.header.generation != input.pixel.material_generation {
        return Err(ResolveError::StaleMaterial);
    }
    let shading_class =
        classify_material_header(&input.header).map_err(ResolveError::Classification)?;
    let geometry = reconstruct_surface(
        SurfaceReconstructionInput {
            primitive_id: input.pixel.primitive_id,
            barycentrics: input.pixel.barycentrics(),
            geometry_generation: input.geometry_generation,
            expected_geometry_generation: input.expected_geometry_generation,
        },
        input.primitives,
        input.vertices,
    )
    .map_err(ResolveError::Surface)?;

    // Fold every bound texture over the authored factors *before* assembling
    // the BSDF sample so the indirect/ambient terms below see texture-modulated
    // base color, emissive and occlusion, exactly like the GPU resolve.
    let modulation = modulation_params_from(&input.parameters);
    let sampled = sample_material(&modulation, input.textures);

    let mut surface = surface_sample_from_parameters(&input.parameters);
    surface.base_color = sampled.base_color;
    surface.metallic = sampled.metallic;
    surface.perceptual_roughness = sampled.perceptual_roughness;
    surface.ambient_occlusion = sampled.occlusion;
    surface.emissive = sampled.emissive;
    surface.clearcoat = sampled.clearcoat;
    surface.clearcoat_roughness = sampled.clearcoat_roughness;

    let emissive = surface.emissive;
    let base_color = surface.base_color;
    let ambient_occlusion = surface.ambient_occlusion.clamp(0.0, 1.0);
    let metallic = surface.metallic.clamp(0.0, 1.0);

    // A bound normal map rotates the tangent-space normal into world space
    // against the reconstructed basis; otherwise the geometric normal stands.
    let shading_normal = if sampled.has_normal_map {
        apply_tangent_space_normal(
            TangentBasis {
                tangent: geometry.tangent,
                bitangent: geometry.bitangent,
                degenerate: false,
            },
            geometry.normal,
            sampled.normal_tangent,
        )
    } else {
        geometry.normal
    };

    let frame = ShadingFrame {
        normal: shading_normal,
        view: normalize_or(sub(input.view_position, geometry.position), geometry.normal),
        tangent: geometry.tangent,
        bitangent: geometry.bitangent,
    };
    // Per-light evaluation must not re-add self-illumination.
    let lit_surface = SurfaceSample {
        emissive: [0.0; 3],
        ..surface
    };

    // Indirect term: prefer the image-based light when the view supplies an
    // environment probe, otherwise fall back to the constant ambient term.
    let indirect = match lights.image_based {
        Some(image_based) => evaluate_image_based_light(lit_surface, frame, &image_based),
        None => ambient_term(base_color, ambient_occlusion, lights.ambient, metallic),
    };

    // Integrates the principled GGX lobe over every analytic light, then adds
    // the shared indirect + emissive terms once.  Shared by the physically
    // based classes (Principled and, until their specialized lobes land,
    // Subsurface/ClearCoat/Cloth/Hair/Water/Custom).
    let shade_principled = || {
        let mut accumulated = [0.0; 3];
        for light in lights.directional {
            accumulated = add(
                accumulated,
                crate::evaluate_principled_direct(lit_surface, frame, direct_sample(*light)),
            );
        }
        for light in lights.punctual {
            if let Some(sample) = light.sample(geometry.position) {
                accumulated = add(
                    accumulated,
                    crate::evaluate_principled_direct(lit_surface, frame, sample),
                );
            }
        }
        add(add(accumulated, indirect), emissive)
    };

    // Integrates the banded toon lobe over every analytic light, then adds the
    // shared indirect + emissive terms once.  Used by the NPR class.
    let shade_toon = || {
        let mut accumulated = [0.0; 3];
        for light in lights.directional {
            accumulated = add(
                accumulated,
                crate::evaluate_toon_direct(
                    lit_surface,
                    frame,
                    direct_sample(*light),
                    lights.toon_bands,
                ),
            );
        }
        for light in lights.punctual {
            if let Some(sample) = light.sample(geometry.position) {
                accumulated = add(
                    accumulated,
                    crate::evaluate_toon_direct(lit_surface, frame, sample, lights.toon_bands),
                );
            }
        }
        add(add(accumulated, indirect), emissive)
    };

    // Integrates the cloth (fabric) lobe over every analytic light, then adds
    // the shared indirect + emissive terms once.  Used by the Cloth class.
    let shade_cloth = || {
        let mut accumulated = [0.0; 3];
        for light in lights.directional {
            accumulated = add(
                accumulated,
                crate::evaluate_cloth_direct(lit_surface, frame, direct_sample(*light)),
            );
        }
        for light in lights.punctual {
            if let Some(sample) = light.sample(geometry.position) {
                accumulated = add(
                    accumulated,
                    crate::evaluate_cloth_direct(lit_surface, frame, sample),
                );
            }
        }
        add(add(accumulated, indirect), emissive)
    };

    // Integrates the subsurface (SSS) lobe over every analytic light, then
    // adds the shared indirect + emissive terms once.  Used by the Subsurface
    // class; the lobe intentionally accepts back-facing lights so radiance can
    // transmit through thin geometry.
    let shade_subsurface = || {
        let mut accumulated = [0.0; 3];
        for light in lights.directional {
            accumulated = add(
                accumulated,
                crate::evaluate_subsurface_direct(lit_surface, frame, direct_sample(*light)),
            );
        }
        for light in lights.punctual {
            if let Some(sample) = light.sample(geometry.position) {
                accumulated = add(
                    accumulated,
                    crate::evaluate_subsurface_direct(lit_surface, frame, sample),
                );
            }
        }
        add(add(accumulated, indirect), emissive)
    };

    // Integrates the hair (strand) lobe over every analytic light, then adds
    // the shared indirect + emissive terms once.  Used by the Hair class; the
    // lobe keeps a translucent diffuse so it intentionally survives backward
    // grazing light rather than hard-cutting at `N.L = 0`.
    let shade_hair = || {
        let mut accumulated = [0.0; 3];
        for light in lights.directional {
            accumulated = add(
                accumulated,
                crate::evaluate_hair_direct(lit_surface, frame, direct_sample(*light)),
            );
        }
        for light in lights.punctual {
            if let Some(sample) = light.sample(geometry.position) {
                accumulated = add(
                    accumulated,
                    crate::evaluate_hair_direct(lit_surface, frame, sample),
                );
            }
        }
        add(add(accumulated, indirect), emissive)
    };

    // Every class is handled explicitly so this branch stays byte-for-byte in
    // step with the `switch` in `shading_resolve.wesl` (9 arms, no wildcard).
    // Cloth, Subsurface and Hair now have dedicated lobes; the remaining
    // specialized classes (ClearCoat/Water) still share `shade_principled()`
    // until their lobes land, so the lint that would collapse those equal arms
    // is suppressed to preserve the 1:1 GPU switch mapping.
    #[expect(
        clippy::match_same_arms,
        reason = "each class keeps its own arm to mirror the GPU `switch`; specialized lobes replace the shared fallback per class later"
    )]
    let color = match shading_class {
        // Unlit surfaces bypass the lighting integrator entirely.
        MaterialShadingClass::Unlit => add(base_color, emissive),
        MaterialShadingClass::Npr => shade_toon(),
        MaterialShadingClass::Principled => shade_principled(),
        MaterialShadingClass::Subsurface => shade_subsurface(),
        MaterialShadingClass::ClearCoat => shade_principled(),
        MaterialShadingClass::Cloth => shade_cloth(),
        MaterialShadingClass::Hair => shade_hair(),
        MaterialShadingClass::Water => shade_principled(),
        MaterialShadingClass::Custom => shade_principled(),
    };

    Ok(ResolvedPixel {
        color,
        shading_class,
        surface_flags: geometry.flags,
    })
}

fn direct_sample(light: DirectionalLight) -> DirectLightSample {
    DirectLightSample {
        direction: light.direction,
        illuminance: light.illuminance,
        visibility: light.visibility,
    }
}

/// Lambertian ambient approximation weighted by occlusion and metalness.
fn ambient_term(
    base_color: [f32; 3],
    ambient_occlusion: f32,
    ambient: [f32; 3],
    metallic: f32,
) -> [f32; 3] {
    let diffuse = 1.0 - metallic;
    [
        base_color[0] * ambient[0] * ambient_occlusion * diffuse,
        base_color[1] * ambient[1] * ambient_occlusion * diffuse,
        base_color[2] * ambient[2] * ambient_occlusion * diffuse,
    ]
}

fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn normalize_or(value: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let length_squared = value[0] * value[0] + value[1] * value[1] + value[2] * value[2];
    if length_squared > 1.0e-12 && length_squared.is_finite() {
        let inverse = length_squared.sqrt().recip();
        [value[0] * inverse, value[1] * inverse, value[2] * inverse]
    } else {
        fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{encode_barycentrics, MaterialShadingClass};
    use prism_render_material::{
        fallback_material_header, GpuMaterialHeader, MaterialRenderClass, MaterialShadingModel,
    };

    fn unit_triangle() -> ([GpuShadingPrimitive; 1], [GpuShadingVertex; 3]) {
        let vertex = |position: [f32; 3], uv: [f32; 2]| GpuShadingVertex {
            position,
            _position_padding: 0.0,
            normal: [0.0, 0.0, 1.0],
            _normal_padding: 0.0,
            tangent: [1.0, 0.0, 0.0, 1.0],
            uv,
            flags: 0,
            _padding: 0,
        };
        (
            [GpuShadingPrimitive { indices: [0, 1, 2], flags: 0 }],
            [
                vertex([0.0, 0.0, 0.0], [0.0, 0.0]),
                vertex([1.0, 0.0, 0.0], [1.0, 0.0]),
                vertex([0.0, 1.0, 0.0], [0.0, 1.0]),
            ],
        )
    }

    fn pixel(material_generation: u32, barycentrics: [f32; 3]) -> VisibilityPixel {
        VisibilityPixel {
            scene_index: 3,
            scene_generation: 1,
            primitive_id: 0,
            geometry_lod_or_cluster: 0,
            material_index: 1,
            material_generation,
            barycentrics_unorm16: encode_barycentrics(barycentrics).unwrap(),
            coverage_and_flags: 255,
        }
    }

    fn principled_header() -> GpuMaterialHeader {
        let mut header = fallback_material_header(1);
        header.generation = 4;
        header.active = 1;
        header.render_class = MaterialRenderClass::Opaque as u32;
        header.shading_model = MaterialShadingModel::Principled as u32;
        header
    }

    fn base_input<'a>(
        primitives: &'a [GpuShadingPrimitive],
        vertices: &'a [GpuShadingVertex],
        header: GpuMaterialHeader,
        parameters: GpuSurfaceParameters,
    ) -> ResolveInput<'a> {
        ResolveInput {
            pixel: pixel(header.generation, [0.2, 0.3, 0.5]),
            primitives,
            vertices,
            geometry_generation: 9,
            expected_geometry_generation: 9,
            header,
            parameters,
            textures: &[],
            view_position: [0.0, 0.0, 4.0],
        }
    }

    #[test]
    fn unlit_bypasses_lighting_and_returns_base_plus_emissive() {
        let (primitives, vertices) = unit_triangle();
        let mut header = principled_header();
        header.shading_model = MaterialShadingModel::Unlit as u32;
        let parameters = GpuSurfaceParameters {
            base_color: [0.2, 0.4, 0.6, 1.0],
            emissive: [0.1, 0.0, 0.0, 0.0],
            ..Default::default()
        };
        let light = DirectionalLight {
            direction: [0.0, 0.0, 1.0],
            illuminance: [10.0; 3],
            visibility: 1.0,
        };
        let resolved = resolve_pixel(
            base_input(&primitives, &vertices, header, parameters),
            LightingEnvironment { directional: &[light], punctual: &[], image_based: None, ambient: [5.0; 3], toon_bands: 4 },
        )
        .unwrap();
        assert_eq!(resolved.shading_class, MaterialShadingClass::Unlit);
        assert_eq!(resolved.color, [0.3, 0.4, 0.6]);
    }

    #[test]
    fn principled_adds_emissive_exactly_once_across_lights() {
        let (primitives, vertices) = unit_triangle();
        let header = principled_header();
        let parameters = GpuSurfaceParameters {
            base_color: [0.5, 0.5, 0.5, 1.0],
            emissive: [2.0, 2.0, 2.0, 0.0],
            ..Default::default()
        };
        let light = DirectionalLight {
            direction: [0.0, 0.0, 1.0],
            illuminance: [3.0; 3],
            visibility: 1.0,
        };
        let one = resolve_pixel(
            base_input(&primitives, &vertices, header, parameters),
            LightingEnvironment { directional: &[light], punctual: &[], image_based: None, ambient: [0.0; 3], toon_bands: 4 },
        )
        .unwrap()
        .color;
        let two = resolve_pixel(
            base_input(&primitives, &vertices, header, parameters),
            LightingEnvironment { directional: &[light, light], punctual: &[], image_based: None, ambient: [0.0; 3], toon_bands: 4 },
        )
        .unwrap()
        .color;
        // Doubling identical lights doubles only the lit term; emissive (2.0)
        // is added once, so the delta equals a single light's lit contribution.
        for channel in 0..3 {
            let lit_once = one[channel] - 2.0;
            let lit_twice = two[channel] - 2.0;
            assert!((lit_twice - 2.0 * lit_once).abs() < 1.0e-4, "channel {channel}");
            assert!(lit_once > 0.0, "expected positive direct lighting");
        }
    }

    #[test]
    fn rejects_stale_material_and_invalid_pixels() {
        let (primitives, vertices) = unit_triangle();
        let header = principled_header();
        let mut stale = base_input(&primitives, &vertices, header, GpuSurfaceParameters::default());
        stale.pixel.material_generation = header.generation + 1;
        assert_eq!(
            resolve_pixel(stale, LightingEnvironment::default()),
            Err(ResolveError::StaleMaterial)
        );

        let mut invalid = base_input(&primitives, &vertices, header, GpuSurfaceParameters::default());
        invalid.pixel = VisibilityPixel::INVALID;
        assert_eq!(
            resolve_pixel(invalid, LightingEnvironment::default()),
            Err(ResolveError::InvalidPixel)
        );
    }

    #[test]
    fn npr_uses_quantized_toon_response() {
        let (primitives, vertices) = unit_triangle();
        let mut header = principled_header();
        header.shading_model = MaterialShadingModel::Npr as u32;
        header.render_class = MaterialRenderClass::NprOpaque as u32;
        let parameters = GpuSurfaceParameters {
            base_color: [1.0, 1.0, 1.0, 1.0],
            ..Default::default()
        };
        let light = DirectionalLight {
            direction: [0.0, 0.0, 1.0],
            illuminance: [1.0; 3],
            visibility: 1.0,
        };
        let resolved = resolve_pixel(
            base_input(&primitives, &vertices, header, parameters),
            LightingEnvironment { directional: &[light], punctual: &[], image_based: None, ambient: [0.0; 3], toon_bands: 4 },
        )
        .unwrap();
        assert_eq!(resolved.shading_class, MaterialShadingClass::Npr);
        // Facing light with 4 bands snaps N.L (=1.0) to the top band -> 1.0.
        for channel in 0..3 {
            assert!((resolved.color[channel] - 1.0).abs() < 1.0e-4);
        }
    }

    #[test]
    fn punctual_point_light_illuminates_and_respects_range() {
        use crate::PunctualLight;
        let (primitives, vertices) = unit_triangle();
        let header = principled_header();
        let parameters = GpuSurfaceParameters {
            base_color: [0.8, 0.8, 0.8, 1.0],
            perceptual_roughness: 0.6,
            ..Default::default()
        };
        // The unit triangle sits on the z=0 plane; place the light above it so
        // it faces the reconstructed geometry normal.
        let lit = resolve_pixel(
            base_input(&primitives, &vertices, header, parameters),
            LightingEnvironment {
                directional: &[],
                punctual: &[PunctualLight::point([0.25, 0.25, 1.0], [5.0; 3], 0.0)],
                image_based: None,
                ambient: [0.0; 3],
                toon_bands: 4,
            },
        )
        .unwrap();
        assert!(lit.color.iter().all(|c| c.is_finite() && *c >= 0.0));
        assert!(lit.color.iter().any(|c| *c > 0.0), "point light must add energy");

        // A light whose range window closes before it reaches the surface adds
        // nothing, so the resolved pixel collapses to the (zero) ambient term.
        let dark = resolve_pixel(
            base_input(&primitives, &vertices, header, parameters),
            LightingEnvironment {
                directional: &[],
                punctual: &[PunctualLight::point([0.25, 0.25, 10.0], [5.0; 3], 1.0)],
                image_based: None,
                ambient: [0.0; 3],
                toon_bands: 4,
            },
        )
        .unwrap();
        assert_eq!(dark.color, [0.0; 3]);
    }

    #[test]
    fn image_based_light_replaces_ambient_for_indirect_term() {
        use crate::{ImageBasedLight, SphericalHarmonicsL2};
        let (primitives, vertices) = unit_triangle();
        let header = principled_header();
        let parameters = GpuSurfaceParameters {
            base_color: [0.8, 0.8, 0.8, 1.0],
            metallic: 0.0,
            perceptual_roughness: 1.0,
            ambient_occlusion: 1.0,
            ..Default::default()
        };
        // No direct lights: the resolved color is purely the indirect term, so a
        // constant environment probe must brighten the surface above the unlit
        // (zero-ambient, no-probe) baseline.
        let dark = resolve_pixel(
            base_input(&primitives, &vertices, header, parameters),
            LightingEnvironment::default(),
        )
        .unwrap();
        assert_eq!(dark.color, [0.0; 3]);

        let probe = ImageBasedLight::new(SphericalHarmonicsL2::from_constant([0.5; 3]));
        let lit = resolve_pixel(
            base_input(&primitives, &vertices, header, parameters),
            LightingEnvironment {
                directional: &[],
                punctual: &[],
                image_based: Some(probe),
                ambient: [0.0; 3],
                toon_bands: 4,
            },
        )
        .unwrap();
        assert!(lit.color.iter().all(|c| c.is_finite() && *c > 0.0));
        // A grey albedo under a 0.5 constant environment lands near albedo*env.
        for channel in 0..3 {
            assert!((0.3..0.5).contains(&lit.color[channel]), "channel {channel} = {}", lit.color[channel]);
        }
    }

    #[test]
    fn base_color_texture_srgb_decodes_and_modulates_unlit_output() {
        use crate::{srgb_channel_to_linear, SEMANTIC_BASE_COLOR};
        let (primitives, vertices) = unit_triangle();
        let mut header = principled_header();
        header.shading_model = MaterialShadingModel::Unlit as u32;
        let parameters = GpuSurfaceParameters {
            base_color: [0.2, 0.4, 0.6, 1.0],
            emissive: [0.1, 0.0, 0.0, 0.0],
            ..Default::default()
        };
        // A mid-grey sRGB base-color texel decodes to linear ~0.214 and scales
        // the authored factor; the unlit path then adds emissive exactly once.
        let textures = [SampledTextureBinding { semantic: SEMANTIC_BASE_COLOR, texel: [0.5, 0.5, 0.5, 1.0] }];
        let input = ResolveInput {
            textures: &textures,
            ..base_input(&primitives, &vertices, header, parameters)
        };
        let resolved = resolve_pixel(
            input,
            LightingEnvironment { directional: &[], punctual: &[], image_based: None, ambient: [0.0; 3], toon_bands: 4 },
        )
        .unwrap();
        let decoded = srgb_channel_to_linear(0.5);
        let expected = [0.2 * decoded + 0.1, 0.4 * decoded, 0.6 * decoded];
        for c in 0..3 {
            assert!(
                (resolved.color[c] - expected[c]).abs() < 1.0e-6,
                "channel {c}: {} vs {}",
                resolved.color[c],
                expected[c]
            );
        }
    }

    #[test]
    fn normal_map_reorients_shading_normal_and_changes_lit_output() {
        use crate::SEMANTIC_NORMAL;
        let (primitives, vertices) = unit_triangle();
        let header = principled_header();
        let parameters = GpuSurfaceParameters {
            base_color: [0.8, 0.8, 0.8, 1.0],
            metallic: 0.0,
            perceptual_roughness: 0.5,
            ..Default::default()
        };
        // Offset the light from the geometric normal so a tilt in the shading
        // normal moves N.L (and therefore the lit response). The direction is
        // already unit length (0.6^2 + 0.8^2 == 1).
        let light = DirectionalLight { direction: [0.6, 0.0, 0.8], illuminance: [4.0; 3], visibility: 1.0 };
        let flat = resolve_pixel(
            base_input(&primitives, &vertices, header, parameters),
            LightingEnvironment { directional: &[light], punctual: &[], image_based: None, ambient: [0.0; 3], toon_bands: 4 },
        )
        .unwrap()
        .color;
        // Tangent-space normal tilted toward +X (world +X here) via the R > 0.5
        // channel; the resolve must rotate it into world space and shift N.L.
        let textures = [SampledTextureBinding { semantic: SEMANTIC_NORMAL, texel: [0.9, 0.5, 0.85, 1.0] }];
        let input = ResolveInput {
            textures: &textures,
            ..base_input(&primitives, &vertices, header, parameters)
        };
        let tilted = resolve_pixel(
            input,
            LightingEnvironment { directional: &[light], punctual: &[], image_based: None, ambient: [0.0; 3], toon_bands: 4 },
        )
        .unwrap()
        .color;
        assert!(
            tilted.iter().zip(flat).any(|(a, b)| (a - b).abs() > 1.0e-4),
            "normal map must change the lit output: {tilted:?} vs {flat:?}"
        );
        assert!(tilted.iter().all(|c| c.is_finite() && *c >= 0.0));
    }

    #[test]
    fn metallic_roughness_texture_modulates_via_gltf_bg_channels() {
        use crate::SEMANTIC_METALLIC_ROUGHNESS;
        let (primitives, vertices) = unit_triangle();
        let header = principled_header();
        let parameters = GpuSurfaceParameters {
            base_color: [0.8, 0.8, 0.8, 1.0],
            metallic: 1.0,
            perceptual_roughness: 1.0,
            ..Default::default()
        };
        let untextured = resolve_pixel(
            base_input(&primitives, &vertices, header, parameters),
            LightingEnvironment { directional: &[DirectionalLight { direction: [0.0, 0.0, 1.0], illuminance: [4.0; 3], visibility: 1.0 }], punctual: &[], image_based: None, ambient: [0.0; 3], toon_bands: 4 },
        )
        .unwrap()
        .color;
        // glTF packs roughness in G and metalness in B; (G=0.25, B=0.5) must
        // quarter the authored roughness and halve metalness, shifting the
        // specular response away from the untextured baseline.
        let textures = [SampledTextureBinding { semantic: SEMANTIC_METALLIC_ROUGHNESS, texel: [0.0, 0.25, 0.5, 1.0] }];
        let input = ResolveInput {
            textures: &textures,
            ..base_input(&primitives, &vertices, header, parameters)
        };
        let textured = resolve_pixel(
            input,
            LightingEnvironment { directional: &[DirectionalLight { direction: [0.0, 0.0, 1.0], illuminance: [4.0; 3], visibility: 1.0 }], punctual: &[], image_based: None, ambient: [0.0; 3], toon_bands: 4 },
        )
        .unwrap()
        .color;
        assert!(
            textured.iter().zip(untextured).any(|(a, b)| (a - b).abs() > 1.0e-4),
            "metallic-roughness texture must change the output: {textured:?} vs {untextured:?}"
        );
        assert!(textured.iter().all(|c| c.is_finite() && *c >= 0.0));
    }

    #[test]
    fn identity_texels_match_the_untextured_resolve_exactly() {
        use crate::{SEMANTIC_BASE_COLOR, SEMANTIC_METALLIC_ROUGHNESS, SEMANTIC_OCCLUSION};
        let (primitives, vertices) = unit_triangle();
        let header = principled_header();
        let parameters = GpuSurfaceParameters {
            base_color: [0.6, 0.7, 0.8, 1.0],
            metallic: 0.3,
            perceptual_roughness: 0.4,
            ambient_occlusion: 1.0,
            ..Default::default()
        };
        let light = DirectionalLight { direction: [0.0, 0.0, 1.0], illuminance: [3.0; 3], visibility: 1.0 };
        let untextured = resolve_pixel(
            base_input(&primitives, &vertices, header, parameters),
            LightingEnvironment { directional: &[light], punctual: &[], image_based: None, ambient: [0.5; 3], toon_bands: 4 },
        )
        .unwrap()
        .color;
        // White base color (sRGB 1 -> linear 1), unit metallic-roughness
        // (G=B=1) and unit occlusion (R=1) are exact identities under the fold,
        // so the resolve must reproduce the untextured path bit for bit.
        let textures = [
            SampledTextureBinding { semantic: SEMANTIC_BASE_COLOR, texel: [1.0, 1.0, 1.0, 1.0] },
            SampledTextureBinding { semantic: SEMANTIC_METALLIC_ROUGHNESS, texel: [0.0, 1.0, 1.0, 0.0] },
            SampledTextureBinding { semantic: SEMANTIC_OCCLUSION, texel: [1.0, 0.0, 0.0, 0.0] },
        ];
        let input = ResolveInput {
            textures: &textures,
            ..base_input(&primitives, &vertices, header, parameters)
        };
        let textured = resolve_pixel(
            input,
            LightingEnvironment { directional: &[light], punctual: &[], image_based: None, ambient: [0.5; 3], toon_bands: 4 },
        )
        .unwrap()
        .color;
        assert_eq!(textured, untextured, "identity texels must not perturb the resolve");
    }
}
