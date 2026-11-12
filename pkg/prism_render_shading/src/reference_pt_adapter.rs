//! Adapter from the production material response to the path-traced BSDF oracle.
//!
//! The adapter is deliberately narrow.  `reference_pt::Bsdf` is a collection
//! of explicit lobe models, while the production `SurfaceSample` is the full
//! über surface contract.  A parity fixture must reject fields that cannot be
//! represented without changing their meaning; silently dropping them would
//! make a passing image comparison misleading.
//!
//! The adapter has two deliberately separate mappings. Clearcoat/diffuse-only
//! inputs use the dedicated `ClearcoatDiffuse` oracle; any active OpenPBR
//! response lobe uses `OpenPbrMaterial`. Neither mapping is the realtime
//! evaluator: the reference has its own Fresnel, slab, sampling, and path
//! transport code. A production-resolve comparison must expose differences,
//! not replace the independent reference evaluation with the realtime BSDF.

use prism_render_architecture::reference_pt::bsdf::{Bsdf, BsdfSample};
use prism_render_architecture::reference_pt::openpbr::OpenPbrMaterial;
use prism_render_architecture::reference_pt::sampler::SampleSource;
use prism_render_architecture::reference_pt::Vec3;
use prism_render_material::{
    DynamicChannels, GpuMaterialTexture, GpuSurfaceParameters, NO_VIRTUAL_TEXTURE,
};

use crate::{sample_material, SampledTextureBinding, SurfaceSample};

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
    /// The canonical runtime record carries a measured lobe, which requires a
    /// table-backed evaluator and cannot be reduced to the analytic oracle.
    UnsupportedMeasuredLobe,
    /// A GPU record with an active glint lobe needs the per-hit cell and
    /// footprint that are supplied by the shade-time context.
    MissingShadeContext,
    /// A runtime record contains a non-default normal-map scale but no sampled
    /// normal/footprint from which the reference normal can be reconstructed.
    UnsupportedNormalMapContext,
    /// Stylized response controls belong to the NPR execution family, not the
    /// analytic OpenPBR reference closure.
    UnsupportedNprLobe,
    /// The material row count and sampled payload count do not describe one
    /// bindless range.
    TextureRowCountMismatch,
    /// A sampled payload is paired with a different material semantic.
    TextureSemanticMismatch,
    /// A texture semantic has no analytic reference consumer.
    UnsupportedTextureSemantic(u32),
    /// A sampled texel or its normal-map footprint contains a non-finite value.
    NonFiniteTextureSample,
    /// Virtual-texture metadata is incomplete or inconsistent with the row.
    InvalidVirtualTextureMetadata,
}

/// Shade-time data needed to extract a complete reference material from the
/// canonical GPU surface record. These values are not material ABI fields.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReferencePtShadeContext {
    /// Optional coat normal override; zero means the base normal is inherited.
    pub coat_normal: [f32; 3],
    /// Optional coat tangent override; zero means the base tangent is inherited.
    pub coat_tangent: [f32; 3],
    /// Screen/texture normal variance used for specular anti-aliasing.
    pub normal_variance: f32,
    /// Per-hit glint mixture state; negative means authored mixture weight.
    pub glint_state: f32,
    /// Surface-cell coordinates used by the glint seed.
    pub glint_cell: [f32; 2],
    /// Surface-cell footprint area used by the glint count estimator.
    pub glint_footprint_area: f32,
}

impl Default for ReferencePtShadeContext {
    fn default() -> Self {
        Self {
            coat_normal: [0.0; 3],
            coat_tangent: [0.0; 3],
            normal_variance: 0.0,
            glint_state: crate::lighting::GLINT_STATE_AUTHORED,
            glint_cell: [0.0; 2],
            glint_footprint_area: 0.0,
        }
    }
}

/// A validated production-to-reference material mapping.
///
/// Non-BRDF outputs stay beside the oracle so emission and coverage are not
/// silently folded into a BRDF comparison.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReferencePtMaterialAdapter {
    bsdf: Bsdf,
    emission: Vec3,
    geometry_opacity: f32,
    ambient_occlusion: f32,
    alpha_cutoff: f32,
    coverage_alpha: f32,
}

impl ReferencePtMaterialAdapter {
    /// Extracts the analytic OpenPBR portion of the canonical GPU material
    /// record using the neutral shade-time context.
    ///
    /// The method is intentionally fail-closed for measured and active glint
    /// data: callers that have the per-hit data must use
    /// [`Self::from_gpu_parameters_with_context`]. It maps emission using the
    /// ABI contract `emission_color * emission_luminance` and converts the
    /// thin-film ABI thickness from micrometres to the reference's nanometres.
    pub fn from_gpu_parameters(
        parameters: GpuSurfaceParameters,
    ) -> Result<Self, ReferencePtAdapterError> {
        Self::from_gpu_parameters_with_context(parameters, ReferencePtShadeContext::default())
    }

    /// Extracts a canonical GPU surface record plus its shade-time context.
    ///
    /// This is the material extraction boundary used by conformance harnesses:
    /// it consumes the same packed surface values as the runtime and keeps
    /// values that are outside that ABI explicit instead of silently dropping
    /// them.
    pub fn from_gpu_parameters_with_context(
        parameters: GpuSurfaceParameters,
        context: ReferencePtShadeContext,
    ) -> Result<Self, ReferencePtAdapterError> {
        Self::from_gpu_parameters_with_textures_and_dynamic(
            parameters,
            &[],
            &[],
            DynamicChannels::NONE,
            context,
        )
    }

    /// Extracts the canonical material record, sampled texture rows, and
    /// runtime dynamic state into the independent reference closure.
    ///
    /// `texture_rows` and `sampled_textures` are deliberately separate: the
    /// former proves the ABI row/VT metadata and the latter is the CPU twin of
    /// the texel sampled at the same row. A mismatch is an error, never a
    /// white/default-texture fallback. Normal maps are also rejected here
    /// because this material adapter has no per-hit tangent frame; callers must
    /// provide that frame through a future hit-normal adapter rather than lose
    /// the authored normal silently.
    pub fn from_gpu_parameters_with_textures_and_dynamic(
        parameters: GpuSurfaceParameters,
        texture_rows: &[GpuMaterialTexture],
        sampled_textures: &[SampledTextureBinding],
        dynamic: DynamicChannels,
        context: ReferencePtShadeContext,
    ) -> Result<Self, ReferencePtAdapterError> {
        validate_gpu_parameters(parameters)?;
        validate_shade_context(context)?;
        validate_dynamic_channels(dynamic)?;
        if parameters.glint_weight != 0.0 && context.glint_footprint_area == 0.0 {
            return Err(ReferencePtAdapterError::MissingShadeContext);
        }
        if texture_rows.len() != sampled_textures.len() {
            return Err(ReferencePtAdapterError::TextureRowCountMismatch);
        }
        for (row, sample) in texture_rows.iter().zip(sampled_textures) {
            validate_texture_input(*row, *sample)?;
        }

        let modulation = crate::MaterialModulationParams {
            base_color: parameters.base_color,
            emissive: rgb(parameters.emission_color),
            emission_luminance: parameters.emission_luminance,
            metallic: parameters.base_metalness,
            perceptual_roughness: parameters.specular_roughness,
            ambient_occlusion: parameters.ambient_occlusion,
            normal_scale: parameters.normal_scale,
            clearcoat: parameters.coat_weight,
            clearcoat_roughness: parameters.coat_roughness,
            base_weight: parameters.base_weight,
            base_diffuse_roughness: parameters.base_diffuse_roughness,
            specular_weight: parameters.specular_weight,
            specular_color: rgb(parameters.specular_color),
            specular_ior: parameters.specular_ior,
            coat_ior: parameters.coat_ior,
            coat_color: rgb(parameters.coat_color),
            coat_roughness_anisotropy: parameters.coat_roughness_anisotropy,
            coat_darkening: parameters.coat_darkening,
            anisotropy: parameters.anisotropy,
            anisotropy_rotation: parameters.anisotropy_rotation,
            fuzz_weight: parameters.fuzz_weight,
            fuzz_roughness: parameters.fuzz_roughness,
            fuzz_color: rgb(parameters.fuzz_color),
            subsurface_weight: parameters.subsurface_weight,
            subsurface_color: rgb(parameters.subsurface_color),
            subsurface_radius: parameters.subsurface_radius,
            subsurface_radius_scale: [
                parameters.subsurface_radius_scale[0],
                parameters.subsurface_radius_scale[1],
                parameters.subsurface_radius_scale[2],
            ],
            subsurface_scatter_anisotropy: parameters.subsurface_scatter_anisotropy,
            transmission_weight: parameters.transmission_weight,
            thickness: parameters.thickness,
            index_of_refraction: parameters.index_of_refraction,
            transmission_dispersion_scale: parameters.transmission_dispersion_scale,
            transmission_color: rgb(parameters.transmission_color),
            transmission_scatter: rgb(parameters.transmission_scatter),
            transmission_scatter_anisotropy: parameters.transmission_scatter_anisotropy,
            transmission_dispersion_abbe_number: parameters.transmission_dispersion_abbe_number,
            thin_film_weight: parameters.thin_film_weight,
            thin_film_thickness: parameters.thin_film_thickness,
            thin_film_ior: parameters.thin_film_ior,
        };
        let sampled = sample_material(&modulation, sampled_textures);
        let mut surface = surface_from_gpu_parameters(parameters, context);
        surface.base_color = sampled.base_color;
        surface.base_weight = sampled.base_weight;
        surface.diffuse_roughness = sampled.base_diffuse_roughness;
        surface.metallic = sampled.metallic;
        surface.perceptual_roughness = sampled.perceptual_roughness;
        surface.specular_weight = sampled.specular_weight;
        surface.specular_color = sampled.specular_color;
        surface.specular_ior = sampled.specular_ior;
        surface.anisotropy = sampled.anisotropy;
        surface.anisotropy_rotation = sampled.anisotropy_rotation;
        surface.ambient_occlusion = sampled.occlusion;
        surface.emissive = [
            sampled.emissive[0] * sampled.emission_luminance,
            sampled.emissive[1] * sampled.emission_luminance,
            sampled.emissive[2] * sampled.emission_luminance,
        ];
        surface.clearcoat = sampled.clearcoat;
        surface.clearcoat_roughness = sampled.clearcoat_roughness;
        surface.coat_ior = sampled.coat_ior;
        surface.coat_color = sampled.coat_color;
        surface.coat_roughness_anisotropy = sampled.coat_roughness_anisotropy;
        surface.coat_darkening = sampled.coat_darkening;
        surface.sheen = sampled.fuzz_weight;
        surface.fuzz_roughness = sampled.fuzz_roughness;
        surface.fuzz_color = sampled.fuzz_color;
        surface.subsurface = sampled.subsurface_weight;
        surface.subsurface_color = sampled.subsurface_color;
        surface.subsurface_radius = sampled.subsurface_radius;
        surface.subsurface_radius_scale = sampled.subsurface_radius_scale;
        surface.scatter_anisotropy = sampled.subsurface_scatter_anisotropy;
        surface.transmission = sampled.transmission_weight;
        surface.thickness = sampled.thickness;
        surface.index_of_refraction = sampled.index_of_refraction;
        surface.dispersion = sampled.transmission_dispersion_scale;
        surface.transmission_color = sampled.transmission_color;
        surface.transmission_scatter = sampled.transmission_scatter;
        surface.transmission_scatter_anisotropy = sampled.transmission_scatter_anisotropy;
        surface.transmission_dispersion_abbe_number = sampled.transmission_dispersion_abbe_number;
        surface.iridescence = sampled.thin_film_weight;
        surface.iridescence_thickness_nm = sampled.thin_film_thickness * 1000.0;
        surface.iridescence_ior = sampled.thin_film_ior;
        crate::apply_dynamic_channels(&mut surface, dynamic, context.glint_cell);

        let adapter = Self::from_surface(surface)?;
        Ok(Self {
            alpha_cutoff: parameters.alpha_cutoff,
            coverage_alpha: sampled.base_alpha
                * crate::dynamic::dissolve_coverage(dynamic.dissolve, context.glint_cell),
            ..adapter
        })
    }

    /// Builds the supported reference mapping for a production surface.
    ///
    /// The supported slice is the isotropic, front-facing OpenPBR response:
    /// base/EON, metallic/specular, transmission, coat, sheen, subsurface,
    /// iridescence and glint. Emission, coverage and AO remain separate
    /// material outputs rather than being mixed into the BRDF oracle.
    /// Unsupported frame sentinels must be EXACTLY zero: using an epsilon here
    /// silently changes authored energy, especially at grazing.
    ///
    /// Finite parameters of disabled lobes are dormant, not mapped. AO and
    /// geometry opacity are not BRDF factors; coverage must be checked by the
    /// caller against the response ABI separately. Texture rows, dynamic
    /// channels, measured tables and the integrator require the GPU-input
    /// entrypoint above. Acceptance is not a claim of
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
            || (surface.glint_state != crate::lighting::GLINT_STATE_AUTHORED
                && !(0.0..=1.0).contains(&surface.glint_state))
        {
            return Err(ReferencePtAdapterError::ParameterOutOfRange);
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
                    transmission_dispersion_abbe_number: surface
                        .transmission_dispersion_abbe_number,
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
        Ok(Self {
            bsdf,
            emission: Vec3::from_array(surface.emissive),
            geometry_opacity: surface.geometry_opacity,
            ambient_occlusion: surface.ambient_occlusion,
            alpha_cutoff: 0.5,
            coverage_alpha: 1.0,
        })
    }

    /// Returns the validated reference BSDF.
    #[must_use]
    pub const fn bsdf(&self) -> Bsdf {
        self.bsdf
    }

    /// Returns authored emission in linear radiance units.
    #[must_use]
    pub const fn emission(&self) -> Vec3 {
        self.emission
    }

    /// Returns authored geometric coverage for the visibility/alpha path.
    #[must_use]
    pub const fn geometry_opacity(&self) -> f32 {
        self.geometry_opacity
    }

    /// Returns authored ambient-occlusion weight for the lighting integrator.
    #[must_use]
    pub const fn ambient_occlusion(&self) -> f32 {
        self.ambient_occlusion
    }

    /// Returns the canonical alpha-test threshold kept outside the BRDF.
    #[must_use]
    pub const fn alpha_cutoff(&self) -> f32 {
        self.alpha_cutoff
    }

    /// Returns base-alpha coverage after the sampled texture and dissolve
    /// coverage have been applied. This stays outside the BRDF oracle.
    #[must_use]
    pub const fn coverage_alpha(&self) -> f32 {
        self.coverage_alpha
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

    /// Evaluates path transport, including the opposite-hemisphere
    /// transmission event supported by the OpenPBR reference closure.
    #[must_use]
    pub fn evaluate_path(&self, view: [f32; 3], light: [f32; 3], normal: [f32; 3]) -> [f32; 3] {
        match self.bsdf {
            Bsdf::OpenPbr { material } => material
                .evaluate_path(
                    Vec3::from_array(view),
                    Vec3::from_array(light),
                    Vec3::from_array(normal),
                )
                .to_array(),
            _ => self.evaluate(view, light, normal),
        }
    }

    /// Returns the solid-angle density used by [`Self::sample`].
    #[must_use]
    pub fn pdf(&self, view: [f32; 3], light: [f32; 3], normal: [f32; 3]) -> f32 {
        self.bsdf.pdf(
            Vec3::from_array(view),
            Vec3::from_array(light),
            Vec3::from_array(normal),
        )
    }

    /// Samples the complete reference closure using its matching proposal/PDF.
    #[must_use]
    pub fn sample(
        &self,
        view: [f32; 3],
        normal: [f32; 3],
        rng: &mut impl SampleSource,
    ) -> Option<BsdfSample> {
        self.bsdf
            .sample(Vec3::from_array(view), Vec3::from_array(normal), rng)
    }
}

fn surface_from_gpu_parameters(
    parameters: GpuSurfaceParameters,
    context: ReferencePtShadeContext,
) -> SurfaceSample {
    SurfaceSample {
        base_color: rgb(parameters.base_color),
        base_weight: parameters.base_weight,
        diffuse_roughness: parameters.base_diffuse_roughness,
        metallic: parameters.base_metalness,
        perceptual_roughness: parameters.specular_roughness,
        specular_weight: parameters.specular_weight,
        specular_color: rgb(parameters.specular_color),
        specular_ior: parameters.specular_ior,
        transmission: parameters.transmission_weight,
        index_of_refraction: parameters.index_of_refraction,
        dispersion: parameters.transmission_dispersion_scale,
        thin_walled: parameters.thin_walled,
        geometry_opacity: parameters.geometry_opacity,
        ambient_occlusion: parameters.ambient_occlusion,
        emissive: [
            parameters.emission_color[0] * parameters.emission_luminance,
            parameters.emission_color[1] * parameters.emission_luminance,
            parameters.emission_color[2] * parameters.emission_luminance,
        ],
        transmission_color: rgb(parameters.transmission_color),
        transmission_scatter: rgb(parameters.transmission_scatter),
        transmission_scatter_anisotropy: parameters.transmission_scatter_anisotropy,
        transmission_dispersion_abbe_number: parameters.transmission_dispersion_abbe_number,
        thickness: parameters.thickness,
        clearcoat: parameters.coat_weight,
        clearcoat_roughness: parameters.coat_roughness,
        coat_ior: parameters.coat_ior,
        coat_color: rgb(parameters.coat_color),
        coat_roughness_anisotropy: parameters.coat_roughness_anisotropy,
        coat_darkening: parameters.coat_darkening,
        anisotropy: parameters.anisotropy,
        anisotropy_rotation: parameters.anisotropy_rotation,
        coat_normal: context.coat_normal,
        coat_tangent: context.coat_tangent,
        normal_variance: context.normal_variance,
        sheen: parameters.fuzz_weight,
        fuzz_roughness: parameters.fuzz_roughness,
        fuzz_color: rgb(parameters.fuzz_color),
        subsurface: parameters.subsurface_weight,
        subsurface_color: rgb(parameters.subsurface_color),
        subsurface_radius: parameters.subsurface_radius,
        subsurface_radius_scale: rgb3(parameters.subsurface_radius_scale),
        scatter_anisotropy: parameters.subsurface_scatter_anisotropy,
        iridescence: parameters.thin_film_weight,
        iridescence_thickness_nm: parameters.thin_film_thickness * 1000.0,
        iridescence_ior: parameters.thin_film_ior,
        glint_weight: parameters.glint_weight,
        glint_roughness: parameters.glint_roughness,
        glint_density: parameters.glint_density,
        glint_bin_half_width: parameters.glint_bin_half_width,
        glint_state: context.glint_state,
        glint_cell: context.glint_cell,
        glint_footprint_area: context.glint_footprint_area,
        ..Default::default()
    }
}

fn validate_gpu_parameters(
    parameters: GpuSurfaceParameters,
) -> Result<(), ReferencePtAdapterError> {
    let vectors = [
        parameters.base_color,
        parameters.emission_color,
        parameters.specular_color,
        parameters.coat_color,
        parameters.fuzz_color,
        parameters.subsurface_color,
        parameters.subsurface_radius_scale,
        parameters.transmission_color,
        parameters.transmission_scatter,
    ];
    let scalars = [
        parameters.emission_luminance,
        parameters.base_metalness,
        parameters.specular_roughness,
        parameters.specular_weight,
        parameters.ambient_occlusion,
        parameters.normal_scale,
        parameters.alpha_cutoff,
        parameters.base_weight,
        parameters.base_diffuse_roughness,
        parameters.specular_ior,
        parameters.thin_walled,
        parameters.geometry_opacity,
        parameters.transmission_weight,
        parameters.thickness,
        parameters.coat_weight,
        parameters.coat_roughness,
        parameters.coat_ior,
        parameters.coat_roughness_anisotropy,
        parameters.coat_darkening,
        parameters.anisotropy,
        parameters.anisotropy_rotation,
        parameters.fuzz_weight,
        parameters.fuzz_roughness,
        parameters.subsurface_weight,
        parameters.subsurface_radius,
        parameters.subsurface_scatter_anisotropy,
        parameters.index_of_refraction,
        parameters.transmission_dispersion_scale,
        parameters.thin_film_weight,
        parameters.thin_film_thickness,
        parameters.thin_film_ior,
        parameters.transmission_scatter_anisotropy,
        parameters.transmission_dispersion_abbe_number,
        parameters.npr_face_softness,
        parameters.npr_stylized_ramp_softness,
        parameters.npr_stylized_specular_scale,
        parameters.npr_stylized_rim_scale,
        parameters.glint_weight,
        parameters.glint_roughness,
        parameters.glint_density,
        parameters.glint_bin_half_width,
        parameters.measured_weight,
        parameters.measured_phi_d,
        parameters.measured_scale,
    ];
    if vectors
        .into_iter()
        .flat_map(|value| value.into_iter())
        .chain(scalars)
        .any(|value| !value.is_finite())
    {
        return Err(ReferencePtAdapterError::NonFiniteParameter);
    }
    let in_unit = |value: f32| (0.0..=1.0).contains(&value);
    if !parameters.base_color.into_iter().all(in_unit)
        || !parameters.emission_color.into_iter().all(in_unit)
        || !parameters.specular_color.into_iter().all(in_unit)
        || !parameters.coat_color.into_iter().all(in_unit)
        || !parameters.fuzz_color.into_iter().all(in_unit)
        || !parameters.subsurface_color.into_iter().all(in_unit)
        || !parameters.subsurface_radius_scale.into_iter().all(in_unit)
        || !parameters.transmission_color.into_iter().all(in_unit)
        || !parameters.transmission_scatter.into_iter().all(in_unit)
        || !in_unit(parameters.base_metalness)
        || !in_unit(parameters.specular_roughness)
        || !in_unit(parameters.specular_weight)
        || !in_unit(parameters.ambient_occlusion)
        || !in_unit(parameters.alpha_cutoff)
        || !in_unit(parameters.base_weight)
        || !in_unit(parameters.base_diffuse_roughness)
        || !in_unit(parameters.thin_walled)
        || !in_unit(parameters.geometry_opacity)
        || !in_unit(parameters.transmission_weight)
        || !in_unit(parameters.coat_weight)
        || !in_unit(parameters.coat_roughness)
        || !in_unit(parameters.coat_roughness_anisotropy)
        || !in_unit(parameters.coat_darkening)
        || !in_unit(parameters.fuzz_weight)
        || !in_unit(parameters.fuzz_roughness)
        || !in_unit(parameters.subsurface_weight)
        || !in_unit(parameters.transmission_dispersion_scale)
        || !in_unit(parameters.thin_film_weight)
        || !in_unit(parameters.glint_weight)
        || !in_unit(parameters.glint_roughness)
        || parameters.emission_luminance < 0.0
        || parameters.normal_scale < 0.0
        || !(1.001..=3.0).contains(&parameters.specular_ior)
        || !(1.0..=3.0).contains(&parameters.index_of_refraction)
        || parameters.thickness < 0.0
        || !(1.001..=3.0).contains(&parameters.coat_ior)
        || parameters.subsurface_radius < 0.0
        || !(-1.0..=1.0).contains(&parameters.subsurface_scatter_anisotropy)
        || parameters.thin_film_thickness < 0.0
        || !(1.0..=3.0).contains(&parameters.thin_film_ior)
        || !(-1.0..=1.0).contains(&parameters.transmission_scatter_anisotropy)
        || !(1.0..=200.0).contains(&parameters.transmission_dispersion_abbe_number)
        || !(0.0..=0.5).contains(&parameters.npr_face_softness)
        || parameters.npr_stylized_ramp_softness < 0.0
        || parameters.npr_stylized_specular_scale < 0.0
        || parameters.npr_stylized_rim_scale < 0.0
        || parameters.glint_density < 0.0
        || !(0.0..=core::f32::consts::PI).contains(&parameters.glint_bin_half_width)
        || parameters.measured_phi_d <= 0.0
        || parameters.measured_scale < 0.0
    {
        return Err(ReferencePtAdapterError::ParameterOutOfRange);
    }
    if parameters.normal_scale != 1.0 {
        return Err(ReferencePtAdapterError::UnsupportedNormalMapContext);
    }
    if parameters.measured_weight != 0.0 {
        return Err(ReferencePtAdapterError::UnsupportedMeasuredLobe);
    }
    if parameters.npr_face_softness != 0.1
        || parameters.npr_stylized_ramp_softness != 1.0
        || parameters.npr_stylized_specular_scale != 1.0
        || parameters.npr_stylized_rim_scale != 1.0
    {
        return Err(ReferencePtAdapterError::UnsupportedNprLobe);
    }
    Ok(())
}

fn validate_shade_context(context: ReferencePtShadeContext) -> Result<(), ReferencePtAdapterError> {
    if context
        .coat_normal
        .into_iter()
        .chain(context.coat_tangent)
        .chain([
            context.normal_variance,
            context.glint_state,
            context.glint_cell[0],
            context.glint_cell[1],
            context.glint_footprint_area,
        ])
        .any(|value| !value.is_finite())
    {
        return Err(ReferencePtAdapterError::NonFiniteParameter);
    }
    if context.glint_footprint_area < 0.0 {
        return Err(ReferencePtAdapterError::ParameterOutOfRange);
    }
    if context.glint_state != crate::lighting::GLINT_STATE_AUTHORED
        && !(0.0..=1.0).contains(&context.glint_state)
    {
        return Err(ReferencePtAdapterError::ParameterOutOfRange);
    }
    Ok(())
}

fn validate_dynamic_channels(channels: DynamicChannels) -> Result<(), ReferencePtAdapterError> {
    let values = [
        channels.wetness,
        channels.snow,
        channels.damage,
        channels.dissolve,
    ];
    if values.iter().any(|value| !value.is_finite()) {
        return Err(ReferencePtAdapterError::NonFiniteParameter);
    }
    if values.iter().any(|value| !(0.0..=1.0).contains(value)) {
        return Err(ReferencePtAdapterError::ParameterOutOfRange);
    }
    Ok(())
}

fn validate_texture_input(
    row: GpuMaterialTexture,
    sample: SampledTextureBinding,
) -> Result<(), ReferencePtAdapterError> {
    if row.semantic != sample.semantic {
        return Err(ReferencePtAdapterError::TextureSemanticMismatch);
    }
    if sample
        .texel
        .into_iter()
        .chain(
            sample
                .normal_map_forward_texels
                .into_iter()
                .flatten()
                .flat_map(|texel| texel.into_iter()),
        )
        .any(|value| !value.is_finite())
    {
        return Err(ReferencePtAdapterError::NonFiniteTextureSample);
    }
    if row.virtual_texture_id == NO_VIRTUAL_TEXTURE {
        if row.virtual_texture_width != 0
            || row.virtual_texture_height != 0
            || row.virtual_texture_page_size != 0
        {
            return Err(ReferencePtAdapterError::InvalidVirtualTextureMetadata);
        }
    } else if row.virtual_texture_width == 0
        || row.virtual_texture_height == 0
        || row.virtual_texture_page_size == 0
        || row.virtual_texture_page_size > row.virtual_texture_width.max(row.virtual_texture_height)
    {
        return Err(ReferencePtAdapterError::InvalidVirtualTextureMetadata);
    }
    match row.semantic {
        crate::SEMANTIC_NORMAL | crate::SEMANTIC_CLEAR_COAT_NORMAL => {
            Err(ReferencePtAdapterError::UnsupportedNormalMapContext)
        }
        crate::gi::specular_aa::normal_variance::SEMANTIC_NORMAL_VARIANCE => {
            Err(ReferencePtAdapterError::UnsupportedNormalVariance)
        }
        crate::texture_sample::SEMANTIC_MEASURED_BRDF => {
            Err(ReferencePtAdapterError::UnsupportedMeasuredLobe)
        }
        semantic if semantic == 8 || semantic == 10 => {
            Err(ReferencePtAdapterError::UnsupportedNprLobe)
        }
        semantic => match crate::texture_sample::texture_semantic_route(semantic) {
            crate::texture_sample::TextureSemanticRoute::SurfaceFold => Ok(()),
            crate::texture_sample::TextureSemanticRoute::DedicatedConsumer => Err(
                ReferencePtAdapterError::UnsupportedTextureSemantic(semantic),
            ),
            crate::texture_sample::TextureSemanticRoute::Unsupported => Err(
                ReferencePtAdapterError::UnsupportedTextureSemantic(semantic),
            ),
        },
    }
}

fn rgb(value: [f32; 4]) -> [f32; 3] {
    [value[0], value[1], value[2]]
}

fn rgb3(value: [f32; 4]) -> [f32; 3] {
    rgb(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_architecture::reference_pt::compare::compare_pixels;
    use prism_render_architecture::reference_pt::sampler::Rng;

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

    fn openpbr_surface() -> SurfaceSample {
        SurfaceSample {
            base_color: [0.42, 0.18, 0.07],
            base_weight: 0.83,
            diffuse_roughness: 0.37,
            metallic: 0.27,
            perceptual_roughness: 0.41,
            specular_weight: 0.63,
            specular_color: [0.71, 0.82, 0.93],
            specular_ior: 1.77,
            transmission: 0.38,
            index_of_refraction: 1.52,
            dispersion: 0.72,
            thin_walled: 0.25,
            geometry_opacity: 1.0,
            transmission_color: [0.86, 0.73, 0.61],
            transmission_scatter: [0.12, 0.23, 0.34],
            transmission_scatter_anisotropy: -0.31,
            transmission_dispersion_abbe_number: 47.0,
            coat_darkening: 0.44,
            scatter_anisotropy: 0.28,
            ambient_occlusion: 1.0,
            emissive: [0.0; 3],
            clearcoat: 0.22,
            clearcoat_roughness: 0.28,
            coat_ior: 1.61,
            coat_color: [0.93, 0.81, 0.69],
            coat_normal: [0.0; 3],
            coat_roughness_anisotropy: 0.0,
            coat_tangent: [0.0; 3],
            sheen: 0.44,
            fuzz_roughness: 0.63,
            fuzz_color: [0.51, 0.67, 0.79],
            subsurface: 0.31,
            subsurface_color: [0.72, 0.41, 0.26],
            subsurface_radius: 0.83,
            subsurface_radius_scale: [0.91, 0.57, 0.23],
            thickness: 0.67,
            anisotropy: 0.0,
            anisotropy_rotation: 0.0,
            normal_variance: 0.0,
            iridescence: 0.55,
            iridescence_thickness_nm: 430.0,
            iridescence_ior: 1.38,
            glint_weight: 0.2,
            glint_roughness: 0.24,
            glint_density: 2.5,
            glint_bin_half_width: 0.07,
            glint_state: -1.0,
            glint_cell: [3.0, 5.0],
            glint_footprint_area: 0.32,
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
    fn adapter_accepts_supported_metallic_and_clearcoat_parameters() {
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
                    anisotropy: 0.5,
                    ..base
                },
                E::UnsupportedAnisotropy,
            ),
            (
                SurfaceSample {
                    anisotropy_rotation: 0.5,
                    ..base
                },
                E::UnsupportedAnisotropy,
            ),
            (
                SurfaceSample {
                    coat_normal: [1.0e-8, 0.0, 0.0],
                    ..base
                },
                E::UnsupportedCoatFrame,
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
    fn adapter_copies_every_supported_openpbr_field_into_the_independent_bsdf() {
        let surface = openpbr_surface();
        let adapter = ReferencePtMaterialAdapter::from_surface(surface).unwrap();
        let Bsdf::OpenPbr { material } = adapter.bsdf() else {
            panic!("a non-zero OpenPBR lobe must not fall back to ClearcoatDiffuse");
        };
        assert_eq!(
            material,
            OpenPbrMaterial {
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
            }
        );
    }

    #[test]
    fn supported_openpbr_lobes_change_the_reference_response_instead_of_being_dropped() {
        let view = normalize([0.37, 0.82, 0.43]);
        let light = normalize([-0.48, 0.76, 0.43]);
        let baseline_surface = openpbr_surface();
        let baseline = ReferencePtMaterialAdapter::from_surface(baseline_surface)
            .unwrap()
            .evaluate(view, light, NORMAL);
        let cases = [
            (
                "metallic",
                SurfaceSample {
                    metallic: 0.91,
                    ..baseline_surface
                },
            ),
            (
                "specular",
                SurfaceSample {
                    specular_weight: 0.07,
                    ..baseline_surface
                },
            ),
            (
                "transmission",
                SurfaceSample {
                    transmission: 0.91,
                    ..baseline_surface
                },
            ),
            (
                "sheen",
                SurfaceSample {
                    sheen: 0.02,
                    ..baseline_surface
                },
            ),
            (
                "subsurface",
                SurfaceSample {
                    subsurface: 0.0,
                    ..baseline_surface
                },
            ),
            (
                "iridescence",
                SurfaceSample {
                    iridescence: 0.0,
                    ..baseline_surface
                },
            ),
        ];
        for (name, surface) in cases {
            let value = ReferencePtMaterialAdapter::from_surface(surface)
                .unwrap()
                .evaluate(view, light, NORMAL);
            let delta = value
                .into_iter()
                .zip(baseline)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0, f32::max);
            assert!(
                delta > 1.0e-6,
                "{name} changed SurfaceSample but not the reference response: baseline={baseline:?} value={value:?}"
            );
        }
    }

    #[test]
    fn mapped_transmission_reaches_the_opposite_hemisphere_through_the_path_sampler() {
        let surface = SurfaceSample {
            base_weight: 0.0,
            metallic: 0.0,
            specular_weight: 0.0,
            transmission: 1.0,
            clearcoat: 0.0,
            sheen: 0.0,
            subsurface: 0.0,
            iridescence: 0.0,
            glint_weight: 0.0,
            perceptual_roughness: 0.35,
            ..openpbr_surface()
        };
        let adapter = ReferencePtMaterialAdapter::from_surface(surface).unwrap();
        let mut saw_transmission = false;
        for seed in 0..64 {
            let mut rng = Rng::seed(seed);
            if let Some(sample) = adapter.bsdf().sample(
                Vec3::from_array(normalize([0.2, 0.95, 0.1])),
                Vec3::from_array(NORMAL),
                &mut rng,
            ) {
                assert!(sample.direction.is_finite());
                assert!(sample.value.is_finite());
                assert!(sample.pdf.is_finite() && sample.pdf > 0.0);
                if sample.direction.y < 0.0 {
                    saw_transmission = true;
                }
            }
        }
        assert!(
            saw_transmission,
            "mapped transmission never produced an opposite-hemisphere path sample"
        );
    }

    #[test]
    fn adapter_rejects_unrepresentable_brdf_fields_explicitly() {
        use ReferencePtAdapterError as E;
        let base = openpbr_surface();
        for (surface, error) in [
            (
                SurfaceSample {
                    anisotropy: 0.2,
                    ..base
                },
                E::UnsupportedAnisotropy,
            ),
            (
                SurfaceSample {
                    anisotropy_rotation: 0.2,
                    ..base
                },
                E::UnsupportedAnisotropy,
            ),
            (
                SurfaceSample {
                    coat_roughness_anisotropy: 0.2,
                    ..base
                },
                E::UnsupportedCoatFrame,
            ),
            (
                SurfaceSample {
                    coat_normal: [0.0, 1.0, 0.0],
                    ..base
                },
                E::UnsupportedCoatFrame,
            ),
            (
                SurfaceSample {
                    coat_tangent: [1.0, 0.0, 0.0],
                    ..base
                },
                E::UnsupportedCoatFrame,
            ),
            (
                SurfaceSample {
                    normal_variance: 0.2,
                    ..base
                },
                E::UnsupportedNormalVariance,
            ),
        ] {
            assert_eq!(
                ReferencePtMaterialAdapter::from_surface(surface),
                Err(error),
                "unsupported field was accepted: {surface:?}"
            );
        }
    }

    #[test]
    fn adapter_preserves_non_brdf_outputs_and_path_transport() {
        let surface = SurfaceSample {
            geometry_opacity: 0.37,
            ambient_occlusion: 0.61,
            emissive: [1.0, 0.25, 0.0],
            transmission: 0.7,
            ..openpbr_surface()
        };
        let adapter = ReferencePtMaterialAdapter::from_surface(surface).unwrap();
        assert_eq!(adapter.emission().to_array(), surface.emissive);
        assert_eq!(adapter.geometry_opacity(), surface.geometry_opacity);
        assert_eq!(adapter.ambient_occlusion(), surface.ambient_occlusion);

        let view = normalize([0.2, 0.95, 0.1]);
        let transmitted = normalize([0.1, -0.98, 0.05]);
        let path = adapter.evaluate_path(view, transmitted, NORMAL);
        assert!(path.iter().all(|value| value.is_finite() && *value >= 0.0));

        let density = adapter.pdf(view, transmitted, NORMAL);
        assert!(density.is_finite() && density >= 0.0);
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
        assert!(adapter
            .evaluate(NORMAL, NORMAL, NORMAL)
            .iter()
            .all(|value| value.is_finite()));
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
    fn cpu_reference_parity_covers_every_clearcoat_diffuse_parameter() {
        // This fixture intentionally covers only the dedicated clearcoat
        // candidate. The complete OpenPBR mapping is gated separately by the
        // field-copy, lobe-sensitivity, and path-sampling tests above.
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
            assert!(adapter
                .evaluate(NORMAL, NORMAL, NORMAL)
                .iter()
                .all(|value| value.is_finite()));
        }
    }

    #[test]
    fn openpbr_cpu_golden_matrix_is_distinct_from_clearcoat_fixture() {
        // This matrix is intentionally separate from the clearcoat-only 16x16
        // fixture above. It pins three genuinely multi-lobe OpenPBR mappings
        // over three view/light pairs, including dispersion/scatter inputs.
        let mut materials = [openpbr_surface(); 3];
        materials[1].metallic = 0.86;
        materials[1].transmission = 0.0;
        materials[1].subsurface = 0.0;
        materials[1].sheen = 0.62;
        materials[1].iridescence = 0.0;
        materials[1].glint_weight = 0.0;
        materials[2].metallic = 0.08;
        materials[2].transmission = 0.77;
        materials[2].transmission_color = [0.77, 0.61, 0.42];
        materials[2].transmission_scatter = [0.13, 0.27, 0.46];
        materials[2].transmission_scatter_anisotropy = 0.54;
        materials[2].thickness = 0.71;
        materials[2].dispersion = 1.0;
        materials[2].transmission_dispersion_abbe_number = 23.0;
        let directions = [
            (normalize([0.2, 0.97, 0.1]), normalize([0.4, 0.9, -0.1])),
            (normalize([-0.45, 0.84, 0.3]), normalize([0.2, 0.96, 0.17])),
            (
                normalize([0.65, 0.72, -0.2]),
                normalize([-0.35, 0.88, 0.32]),
            ),
        ];
        let expected = [
            [
                [0.074073195, 0.06629787, 0.07675927],
                [0.071048126, 0.05813247, 0.06294348],
                [0.06986547, 0.054551564, 0.052732833],
            ],
            [
                [0.030591026, 0.019127227, 0.015000621],
                [0.038387436, 0.021924024, 0.015549844],
                [0.094673246, 0.045642767, 0.02472153],
            ],
            [
                [0.06257485, 0.03767831, 0.035851326],
                [0.06295971, 0.038641453, 0.0371988],
                [0.066668905, 0.04551283, 0.04692006],
            ],
        ];
        let expected_pdf = [
            [0.13999282, 0.1581955, 0.23138532],
            [0.20105067, 0.23128083, 0.3604911],
            [0.108905345, 0.122773655, 0.1798538],
        ];
        for (material_index, surface) in materials.into_iter().enumerate() {
            let adapter = ReferencePtMaterialAdapter::from_surface(surface).unwrap();
            for (direction_index, (view, light)) in directions.into_iter().enumerate() {
                let actual = adapter.evaluate(view, light, NORMAL);
                for channel in 0..3 {
                    assert!(
                        (actual[channel] - expected[material_index][direction_index][channel])
                            .abs()
                            <= 3.0e-6,
                        "OpenPBR golden drift: material={material_index} direction={direction_index} channel={channel} actual={actual:?} expected={:?}",
                        expected[material_index][direction_index]
                    );
                }
                let pdf = adapter.pdf(view, light, NORMAL);
                assert!((pdf - expected_pdf[material_index][direction_index]).abs() <= 3.0e-6);
            }
        }
    }

    #[test]
    fn openpbr_path_golden_matrix_covers_transmission_transport() {
        let mut surface = openpbr_surface();
        surface.base_weight = 0.08;
        surface.transmission = 0.77;
        surface.transmission_color = [0.77, 0.61, 0.42];
        surface.transmission_scatter = [0.13, 0.27, 0.46];
        surface.transmission_scatter_anisotropy = 0.54;
        surface.thickness = 0.71;
        surface.dispersion = 1.0;
        surface.transmission_dispersion_abbe_number = 23.0;
        let adapter = ReferencePtMaterialAdapter::from_surface(surface).unwrap();
        let directions = [
            (normalize([0.2, 0.97, 0.1]), normalize([-0.2, -0.97, -0.1])),
            (
                normalize([-0.45, 0.84, 0.3]),
                normalize([0.45, -0.84, -0.3]),
            ),
            (
                normalize([0.65, 0.72, -0.2]),
                normalize([-0.65, -0.72, 0.2]),
            ),
        ];
        let expected = [
            [4.213745, 8.117438, 13.534451],
            [0.3424032, 0.6870612, 1.1815869],
            [0.18799822, 0.39065626, 0.6898017],
        ];
        let expected_pdf = [5.8719397, 0.4138248, 0.19754508];
        for (index, (view, light)) in directions.into_iter().enumerate() {
            let actual = adapter.evaluate_path(view, light, NORMAL);
            for channel in 0..3 {
                assert!(
                    (actual[channel] - expected[index][channel]).abs() <= 3.0e-5,
                    "OpenPBR path golden drift: index={index} channel={channel} actual={actual:?} expected={:?}",
                    expected[index]
                );
            }
            assert!((adapter.pdf(view, light, NORMAL) - expected_pdf[index]).abs() <= 3.0e-5);
        }
    }

    #[test]
    fn gpu_surface_extraction_maps_the_canonical_record_without_dropping_fields() {
        let mut parameters = prism_render_material::GpuSurfaceParameters::default();
        parameters.base_color = [0.21, 0.42, 0.63, 1.0];
        parameters.base_weight = 0.73;
        parameters.base_diffuse_roughness = 0.27;
        parameters.base_metalness = 0.18;
        parameters.specular_roughness = 0.41;
        parameters.specular_weight = 0.82;
        parameters.specular_color = [0.91, 0.83, 0.72, 1.0];
        parameters.specular_ior = 1.72;
        parameters.transmission_weight = 0.64;
        parameters.index_of_refraction = 1.46;
        parameters.transmission_dispersion_scale = 0.35;
        parameters.thin_walled = 1.0;
        parameters.geometry_opacity = 0.77;
        parameters.alpha_cutoff = 0.37;
        parameters.ambient_occlusion = 0.68;
        parameters.emission_color = [0.4, 0.5, 0.6, 1.0];
        parameters.emission_luminance = 12.0;
        parameters.transmission_color = [0.8, 0.7, 0.6, 1.0];
        parameters.transmission_scatter = [0.1, 0.2, 0.3, 1.0];
        parameters.transmission_scatter_anisotropy = -0.25;
        parameters.transmission_dispersion_abbe_number = 28.0;
        parameters.thickness = 0.9;
        parameters.coat_weight = 0.43;
        parameters.coat_roughness = 0.22;
        parameters.coat_ior = 1.61;
        parameters.coat_color = [0.9, 0.8, 0.7, 1.0];
        parameters.coat_darkening = 0.31;
        parameters.fuzz_weight = 0.19;
        parameters.fuzz_roughness = 0.37;
        parameters.fuzz_color = [0.7, 0.8, 0.9, 1.0];
        parameters.subsurface_weight = 0.24;
        parameters.subsurface_color = [0.6, 0.5, 0.4, 1.0];
        parameters.subsurface_radius = 1.4;
        parameters.subsurface_radius_scale = [1.0, 0.7, 0.35, 1.0];
        parameters.subsurface_scatter_anisotropy = 0.21;
        parameters.thin_film_weight = 0.32;
        parameters.thin_film_thickness = 0.42;
        parameters.thin_film_ior = 1.33;

        let extracted = ReferencePtMaterialAdapter::from_gpu_parameters(parameters)
            .expect("canonical analytic material record is representable");
        let expected = ReferencePtMaterialAdapter::from_surface(SurfaceSample {
            base_color: [0.21, 0.42, 0.63],
            base_weight: 0.73,
            diffuse_roughness: 0.27,
            metallic: 0.18,
            perceptual_roughness: 0.41,
            specular_weight: 0.82,
            specular_color: [0.91, 0.83, 0.72],
            specular_ior: 1.72,
            transmission: 0.64,
            index_of_refraction: 1.46,
            dispersion: 0.35,
            thin_walled: 1.0,
            geometry_opacity: 0.77,
            ambient_occlusion: 0.68,
            emissive: [4.8, 6.0, 7.2],
            transmission_color: [0.8, 0.7, 0.6],
            transmission_scatter: [0.1, 0.2, 0.3],
            transmission_scatter_anisotropy: -0.25,
            transmission_dispersion_abbe_number: 28.0,
            thickness: 0.9,
            clearcoat: 0.43,
            clearcoat_roughness: 0.22,
            coat_ior: 1.61,
            coat_color: [0.9, 0.8, 0.7],
            coat_darkening: 0.31,
            sheen: 0.19,
            fuzz_roughness: 0.37,
            fuzz_color: [0.7, 0.8, 0.9],
            subsurface: 0.24,
            subsurface_color: [0.6, 0.5, 0.4],
            subsurface_radius: 1.4,
            subsurface_radius_scale: [1.0, 0.7, 0.35],
            scatter_anisotropy: 0.21,
            iridescence: 0.32,
            iridescence_thickness_nm: 420.0,
            iridescence_ior: 1.33,
            ..SurfaceSample::default()
        })
        .expect("manual reference extraction is representable");
        assert_eq!(extracted.bsdf(), expected.bsdf());
        let emission = extracted.emission().to_array();
        assert!(
            emission
                .into_iter()
                .zip([4.8, 6.0, 7.2])
                .all(|(actual, expected)| (actual - expected).abs() <= 1.0e-6),
            "GPU emission extraction drifted: {emission:?}"
        );
        assert_eq!(extracted.geometry_opacity(), 0.77);
        assert_eq!(extracted.ambient_occlusion(), 0.68);
        assert_eq!(extracted.alpha_cutoff(), 0.37);
    }

    #[test]
    fn gpu_surface_extraction_requires_context_for_non_abi_lobes() {
        let mut parameters = prism_render_material::GpuSurfaceParameters::default();
        parameters.glint_weight = 0.4;
        assert_eq!(
            ReferencePtMaterialAdapter::from_gpu_parameters(parameters),
            Err(ReferencePtAdapterError::MissingShadeContext)
        );
        let context = ReferencePtShadeContext {
            glint_cell: [3.0, -2.0],
            glint_footprint_area: 0.25,
            ..ReferencePtShadeContext::default()
        };
        assert!(
            ReferencePtMaterialAdapter::from_gpu_parameters_with_context(parameters, context)
                .is_ok()
        );

        parameters.glint_weight = 0.0;
        parameters.measured_weight = 0.1;
        assert_eq!(
            ReferencePtMaterialAdapter::from_gpu_parameters(parameters),
            Err(ReferencePtAdapterError::UnsupportedMeasuredLobe)
        );
        parameters.measured_weight = 0.0;
        parameters.normal_scale = 0.75;
        assert_eq!(
            ReferencePtMaterialAdapter::from_gpu_parameters(parameters),
            Err(ReferencePtAdapterError::UnsupportedNormalMapContext)
        );
        parameters.normal_scale = 1.0;
        parameters.npr_stylized_rim_scale = 1.2;
        assert_eq!(
            ReferencePtMaterialAdapter::from_gpu_parameters(parameters),
            Err(ReferencePtAdapterError::UnsupportedNprLobe)
        );

        let bad_state = ReferencePtShadeContext {
            glint_state: -2.0,
            ..ReferencePtShadeContext::default()
        };
        assert_eq!(
            ReferencePtMaterialAdapter::from_gpu_parameters_with_context(
                GpuSurfaceParameters::default(),
                bad_state,
            ),
            Err(ReferencePtAdapterError::ParameterOutOfRange)
        );
    }

    #[test]
    fn gpu_anisotropy_is_preserved_through_direct_and_texture_fold_paths() {
        let mut parameters = GpuSurfaceParameters::default();
        parameters.anisotropy = 0.4;
        assert_eq!(
            ReferencePtMaterialAdapter::from_gpu_parameters(parameters),
            Err(ReferencePtAdapterError::UnsupportedAnisotropy)
        );

        parameters.anisotropy = 0.0;
        parameters.anisotropy_rotation = 0.6;
        assert_eq!(
            ReferencePtMaterialAdapter::from_gpu_parameters(parameters),
            Err(ReferencePtAdapterError::UnsupportedAnisotropy)
        );

        let fold_parameters = GpuSurfaceParameters {
            anisotropy: 0.8,
            ..GpuSurfaceParameters::default()
        };
        let row = GpuMaterialTexture::non_virtual(
            7,
            1,
            crate::texture_sample::SEMANTIC_ANISOTROPY,
            0,
        );
        let sample = SampledTextureBinding {
            semantic: crate::texture_sample::SEMANTIC_ANISOTROPY,
            texel: [0.5, 1.0, 1.0, 1.0],
            normal_map_forward_texels: None,
        };
        assert_eq!(
            ReferencePtMaterialAdapter::from_gpu_parameters_with_textures_and_dynamic(
                fold_parameters,
                &[row],
                &[sample],
                DynamicChannels::NONE,
                ReferencePtShadeContext::default(),
            ),
            Err(ReferencePtAdapterError::UnsupportedAnisotropy)
        );
    }

    #[test]
    fn gpu_texture_and_dynamic_inputs_reach_the_reference_and_coverage_outputs() {
        let parameters = GpuSurfaceParameters::default();
        let baseline = ReferencePtMaterialAdapter::from_gpu_parameters(parameters)
            .expect("default GPU material is representable");
        let row = GpuMaterialTexture::non_virtual(4, 9, crate::SEMANTIC_BASE_COLOR, 2);
        let sample = SampledTextureBinding {
            semantic: crate::SEMANTIC_BASE_COLOR,
            texel: [0.5, 0.25, 0.75, 0.6],
            normal_map_forward_texels: None,
        };
        let textured = ReferencePtMaterialAdapter::from_gpu_parameters_with_textures_and_dynamic(
            parameters,
            &[row],
            &[sample],
            DynamicChannels {
                wetness: 0.5,
                ..DynamicChannels::NONE
            },
            ReferencePtShadeContext::default(),
        )
        .expect("analytic texture and dynamic inputs are representable");
        let baseline_value = baseline.evaluate([0.0, 1.0, 0.0], [0.0, 1.0, 0.0], NORMAL);
        let textured_value = textured.evaluate([0.0, 1.0, 0.0], [0.0, 1.0, 0.0], NORMAL);
        assert!(
            baseline_value
                .into_iter()
                .zip(textured_value)
                .any(|(before, after)| (before - after).abs() > 1.0e-5),
            "texture/dynamic inputs were silently dropped"
        );
        assert!((textured.coverage_alpha() - 0.6).abs() <= 1.0e-6);
    }

    #[test]
    fn gpu_openpbr_texture_semantics_reach_every_reference_material_field() {
        use prism_render_material::GpuSurfaceParameters;

        let mut parameters = GpuSurfaceParameters::default();
        parameters.base_color = [0.62, 0.48, 0.35, 1.0];
        parameters.emission_color = [0.7, 0.5, 0.3, 1.0];
        parameters.emission_luminance = 2.0;
        parameters.base_metalness = 0.6;
        parameters.specular_roughness = 0.55;
        parameters.base_weight = 0.82;
        parameters.base_diffuse_roughness = 0.58;
        parameters.specular_weight = 0.74;
        parameters.specular_color = [0.86, 0.72, 0.64, 1.0];
        parameters.specular_ior = 1.62;
        parameters.coat_weight = 0.65;
        parameters.coat_roughness = 0.31;
        parameters.coat_ior = 1.58;
        parameters.coat_color = [0.92, 0.8, 0.68, 1.0];
        parameters.coat_darkening = 0.64;
        parameters.fuzz_weight = 0.48;
        parameters.fuzz_roughness = 0.43;
        parameters.fuzz_color = [0.77, 0.66, 0.55, 1.0];
        parameters.subsurface_weight = 0.37;
        parameters.subsurface_color = [0.7, 0.58, 0.46, 1.0];
        parameters.subsurface_radius = 0.9;
        parameters.subsurface_radius_scale = [0.9, 0.65, 0.4, 1.0];
        parameters.subsurface_scatter_anisotropy = 0.35;
        parameters.transmission_weight = 0.61;
        parameters.thickness = 0.75;
        parameters.index_of_refraction = 1.52;
        parameters.transmission_dispersion_scale = 0.7;
        parameters.transmission_color = [0.88, 0.74, 0.59, 1.0];
        parameters.transmission_scatter = [0.34, 0.26, 0.18, 1.0];
        parameters.transmission_scatter_anisotropy = -0.22;
        parameters.transmission_dispersion_abbe_number = 43.0;
        parameters.thin_film_weight = 0.52;
        parameters.thin_film_thickness = 0.43;
        parameters.thin_film_ior = 1.37;

        let baseline = ReferencePtMaterialAdapter::from_gpu_parameters(parameters)
            .expect("untextured OpenPBR parameters are representable");
        let baseline_material = match baseline.bsdf() {
            Bsdf::OpenPbr { material } => material,
            _ => panic!("OpenPBR parameters must use the OpenPBR reference closure"),
        };

        let cases = [
            (crate::SEMANTIC_BASE_COLOR, [0.5, 0.42, 0.34, 1.0]),
            (crate::SEMANTIC_METALLIC_ROUGHNESS, [1.0, 0.72, 0.83, 1.0]),
            (crate::SEMANTIC_CLEAR_COAT, [0.71, 1.0, 1.0, 1.0]),
            (crate::SEMANTIC_CLEAR_COAT_ROUGHNESS, [1.0, 0.77, 1.0, 1.0]),
            (
                crate::texture_sample::SEMANTIC_TRANSMISSION_WEIGHT,
                [0.81, 1.0, 1.0, 1.0],
            ),
            (
                crate::texture_sample::SEMANTIC_TRANSMISSION_COLOR,
                [0.82, 0.9, 0.76, 1.0],
            ),
            (
                crate::texture_sample::SEMANTIC_TRANSMISSION_SCATTER,
                [0.82, 0.71, 0.63, 1.0],
            ),
            (
                crate::texture_sample::SEMANTIC_TRANSMISSION_SCATTER_ANISOTROPY,
                [0.76, 1.0, 1.0, 1.0],
            ),
            (
                crate::texture_sample::SEMANTIC_TRANSMISSION_DISPERSION_SCALE,
                [0.81, 1.0, 1.0, 1.0],
            ),
            (
                crate::texture_sample::SEMANTIC_TRANSMISSION_DISPERSION_ABBE_NUMBER,
                [0.86, 1.0, 1.0, 1.0],
            ),
            (
                crate::texture_sample::SEMANTIC_FUZZ_WEIGHT,
                [0.79, 1.0, 1.0, 1.0],
            ),
            (
                crate::texture_sample::SEMANTIC_FUZZ_ROUGHNESS,
                [0.82, 1.0, 1.0, 1.0],
            ),
            (
                crate::texture_sample::SEMANTIC_FUZZ_COLOR,
                [0.83, 0.74, 0.65, 1.0],
            ),
            (
                crate::texture_sample::SEMANTIC_SUBSURFACE_WEIGHT,
                [0.78, 1.0, 1.0, 1.0],
            ),
            (
                crate::texture_sample::SEMANTIC_SUBSURFACE_COLOR,
                [0.81, 0.72, 0.63, 1.0],
            ),
            (
                crate::texture_sample::SEMANTIC_SUBSURFACE_RADIUS,
                [0.83, 1.0, 1.0, 1.0],
            ),
            (
                crate::texture_sample::SEMANTIC_SUBSURFACE_RADIUS_SCALE,
                [0.86, 0.74, 0.61, 1.0],
            ),
            (
                crate::texture_sample::SEMANTIC_SUBSURFACE_SCATTER_ANISOTROPY,
                [0.78, 1.0, 1.0, 1.0],
            ),
            (
                crate::texture_sample::SEMANTIC_THIN_FILM_WEIGHT,
                [0.81, 1.0, 1.0, 1.0],
            ),
            (
                crate::texture_sample::SEMANTIC_THIN_FILM_THICKNESS,
                [0.77, 1.0, 1.0, 1.0],
            ),
            (
                crate::texture_sample::SEMANTIC_THIN_FILM_IOR,
                [0.94, 1.0, 1.0, 1.0],
            ),
            (
                crate::texture_sample::SEMANTIC_THICKNESS,
                [0.79, 1.0, 1.0, 1.0],
            ),
            (
                crate::texture_sample::SEMANTIC_BASE_WEIGHT,
                [0.77, 1.0, 1.0, 1.0],
            ),
            (
                crate::texture_sample::SEMANTIC_BASE_DIFFUSE_ROUGHNESS,
                [0.74, 1.0, 1.0, 1.0],
            ),
            (
                crate::texture_sample::SEMANTIC_SPECULAR_WEIGHT,
                [0.79, 1.0, 1.0, 1.0],
            ),
            (
                crate::texture_sample::SEMANTIC_SPECULAR_COLOR,
                [0.84, 0.73, 0.62, 1.0],
            ),
            (
                crate::texture_sample::SEMANTIC_SPECULAR_IOR,
                [0.93, 1.0, 1.0, 1.0],
            ),
            (
                crate::texture_sample::SEMANTIC_COAT_IOR,
                [0.94, 1.0, 1.0, 1.0],
            ),
            (
                crate::texture_sample::SEMANTIC_COAT_COLOR,
                [0.87, 0.76, 0.65, 1.0],
            ),
            (
                crate::texture_sample::SEMANTIC_COAT_DARKENING,
                [0.79, 1.0, 1.0, 1.0],
            ),
        ];

        for (semantic, texel) in cases {
            let row = GpuMaterialTexture::non_virtual(semantic + 1, 9, semantic, 2);
            let sample = SampledTextureBinding {
                semantic,
                texel,
                normal_map_forward_texels: None,
            };
            let candidate =
                ReferencePtMaterialAdapter::from_gpu_parameters_with_textures_and_dynamic(
                    parameters,
                    &[row],
                    &[sample],
                    DynamicChannels::NONE,
                    ReferencePtShadeContext::default(),
                )
                .unwrap_or_else(|error| {
                    panic!("texture semantic {semantic} was rejected: {error:?}")
                });
            let candidate_material = match candidate.bsdf() {
                Bsdf::OpenPbr { material } => material,
                _ => panic!("texture semantic {semantic} changed the closure family"),
            };
            assert_ne!(
                candidate_material, baseline_material,
                "texture semantic {semantic} was accepted but did not reach the reference material"
            );
        }

        let emission_row = GpuMaterialTexture::non_virtual(
            101,
            9,
            crate::texture_sample::SEMANTIC_EMISSION_LUMINANCE,
            2,
        );
        let emission_sample = SampledTextureBinding {
            semantic: crate::texture_sample::SEMANTIC_EMISSION_LUMINANCE,
            texel: [0.5, 1.0, 1.0, 1.0],
            normal_map_forward_texels: None,
        };
        let textured_emission =
            ReferencePtMaterialAdapter::from_gpu_parameters_with_textures_and_dynamic(
                parameters,
                &[emission_row],
                &[emission_sample],
                DynamicChannels::NONE,
                ReferencePtShadeContext::default(),
            )
            .expect("emission luminance texture is representable");
        assert_ne!(textured_emission.emission(), baseline.emission());

        let occlusion_row = GpuMaterialTexture::non_virtual(102, 9, crate::SEMANTIC_OCCLUSION, 2);
        let occlusion_sample = SampledTextureBinding {
            semantic: crate::SEMANTIC_OCCLUSION,
            texel: [0.43, 1.0, 1.0, 1.0],
            normal_map_forward_texels: None,
        };
        let textured_occlusion =
            ReferencePtMaterialAdapter::from_gpu_parameters_with_textures_and_dynamic(
                parameters,
                &[occlusion_row],
                &[occlusion_sample],
                DynamicChannels::NONE,
                ReferencePtShadeContext::default(),
            )
            .expect("occlusion texture is representable");
        assert!((textured_occlusion.ambient_occlusion() - 0.43).abs() <= 1.0e-6);
    }

    #[test]
    fn gpu_texture_and_dynamic_inputs_fail_closed() {
        let parameters = GpuSurfaceParameters::default();
        let row = GpuMaterialTexture::non_virtual(4, 9, crate::SEMANTIC_BASE_COLOR, 2);
        let sample = SampledTextureBinding {
            semantic: crate::SEMANTIC_BASE_COLOR,
            texel: [1.0; 4],
            normal_map_forward_texels: None,
        };
        assert_eq!(
            ReferencePtMaterialAdapter::from_gpu_parameters_with_textures_and_dynamic(
                parameters,
                &[],
                &[sample],
                DynamicChannels::NONE,
                ReferencePtShadeContext::default(),
            ),
            Err(ReferencePtAdapterError::TextureRowCountMismatch)
        );

        let mut mismatched = sample;
        mismatched.semantic = crate::SEMANTIC_EMISSIVE;
        assert_eq!(
            ReferencePtMaterialAdapter::from_gpu_parameters_with_textures_and_dynamic(
                parameters,
                &[row],
                &[mismatched],
                DynamicChannels::NONE,
                ReferencePtShadeContext::default(),
            ),
            Err(ReferencePtAdapterError::TextureSemanticMismatch)
        );

        let mut invalid_row = row;
        invalid_row.virtual_texture_id = 17;
        assert_eq!(
            ReferencePtMaterialAdapter::from_gpu_parameters_with_textures_and_dynamic(
                parameters,
                &[invalid_row],
                &[sample],
                DynamicChannels::NONE,
                ReferencePtShadeContext::default(),
            ),
            Err(ReferencePtAdapterError::InvalidVirtualTextureMetadata)
        );

        let mut unknown = row;
        unknown.semantic = 42;
        let mut unknown_sample = sample;
        unknown_sample.semantic = 42;
        assert_eq!(
            ReferencePtMaterialAdapter::from_gpu_parameters_with_textures_and_dynamic(
                parameters,
                &[unknown],
                &[unknown_sample],
                DynamicChannels::NONE,
                ReferencePtShadeContext::default(),
            ),
            Err(ReferencePtAdapterError::UnsupportedTextureSemantic(42))
        );

        let mut non_finite = sample;
        non_finite.texel[0] = f32::NAN;
        assert_eq!(
            ReferencePtMaterialAdapter::from_gpu_parameters_with_textures_and_dynamic(
                parameters,
                &[row],
                &[non_finite],
                DynamicChannels::NONE,
                ReferencePtShadeContext::default(),
            ),
            Err(ReferencePtAdapterError::NonFiniteTextureSample)
        );

        assert_eq!(
            ReferencePtMaterialAdapter::from_gpu_parameters_with_textures_and_dynamic(
                parameters,
                &[],
                &[],
                DynamicChannels {
                    wetness: f32::NAN,
                    ..DynamicChannels::NONE
                },
                ReferencePtShadeContext::default(),
            ),
            Err(ReferencePtAdapterError::NonFiniteParameter)
        );
        assert_eq!(
            ReferencePtMaterialAdapter::from_gpu_parameters_with_textures_and_dynamic(
                parameters,
                &[],
                &[],
                DynamicChannels {
                    snow: 1.1,
                    ..DynamicChannels::NONE
                },
                ReferencePtShadeContext::default(),
            ),
            Err(ReferencePtAdapterError::ParameterOutOfRange)
        );

        let normal_row = GpuMaterialTexture::non_virtual(4, 9, crate::SEMANTIC_NORMAL, 2);
        let normal_sample = SampledTextureBinding {
            semantic: crate::SEMANTIC_NORMAL,
            texel: [0.5, 0.5, 1.0, 1.0],
            normal_map_forward_texels: None,
        };
        assert_eq!(
            ReferencePtMaterialAdapter::from_gpu_parameters_with_textures_and_dynamic(
                parameters,
                &[normal_row],
                &[normal_sample],
                DynamicChannels::NONE,
                ReferencePtShadeContext::default(),
            ),
            Err(ReferencePtAdapterError::UnsupportedNormalMapContext)
        );
    }
}
