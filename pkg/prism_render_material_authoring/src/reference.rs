//! Authoring-to-reference material conversion.
//!
//! The reference path consumes an independent [`OpenPbrMaterial`] value. This
//! boundary validates every runtime field that is either outside that value or
//! requires shade-time data, so a reference render can never pass by silently
//! dropping an authored feature.

use prism_render_architecture::reference_pt::{
    openpbr::{OpenPbrMaterial, OpenPbrSample},
    sampler::SampleSource,
    Vec3,
};
use prism_render_material::GpuSurfaceParameters;

use crate::RuntimeSurface;

/// A reference material plus non-BRDF outputs that must remain outside the
/// OpenPBR closure.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReferenceMaterial {
    /// Independent path-traced OpenPBR response inputs.
    pub openpbr: OpenPbrMaterial,
    /// Linear self-emission retained beside the BRDF.
    pub emission: [f32; 3],
    /// Static base-layer coverage from `GpuSurfaceParameters::base_color.w`.
    /// Texture coverage is intentionally not synthesized here: a textured
    /// surface is rejected until its texel is supplied by a shade-time adapter.
    pub base_alpha: f32,
    /// Geometric coverage multiplier.
    pub geometry_opacity: f32,
    /// Ambient-occlusion weight retained for the integrator policy.
    pub ambient_occlusion: f32,
    /// Alpha-test threshold retained for visibility policy.
    pub alpha_cutoff: f32,
}

/// Shade-time values that are not part of the canonical material ABI but are
/// required by the independent reference closure's glint estimator.
///
/// The default is the authored glint state with no footprint. An active glint
/// material therefore fails closed through [`ReferenceMaterialError::MissingShadeContext`]
/// unless the caller supplies the per-hit footprint explicitly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReferenceShadeContext {
    /// Negative one selects the authored glint weight; otherwise this is the
    /// per-hit mixture state in `[0, 1]`.
    pub glint_state: f32,
    /// Surface-cell coordinates used by the deterministic glint seed.
    pub glint_cell: [f32; 2],
    /// Surface-cell footprint area used by the glint count estimator.
    pub glint_footprint_area: f32,
}

impl Default for ReferenceShadeContext {
    fn default() -> Self {
        Self {
            glint_state: -1.0,
            glint_cell: [0.0; 2],
            glint_footprint_area: 0.0,
        }
    }
}

/// Structured failure when a runtime material cannot be represented by the
/// authoring-side path-traced reference contract.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReferenceMaterialError {
    /// A runtime parameter or component is not finite.
    NonFiniteParameter {
        /// Canonical runtime field name.
        field: &'static str,
    },
    /// A parameter is outside the reference contract's authored range.
    ParameterOutOfRange {
        /// Canonical runtime field name.
        field: &'static str,
    },
    /// An authored field has no equivalent in the independent reference value.
    UnsupportedField {
        /// Canonical runtime field name.
        field: &'static str,
    },
    /// A texture needs a sampled value and cannot be reduced to a scalar here.
    UnsupportedTexture {
        /// Canonical runtime field name.
        field: String,
    },
    /// The reference needs per-hit data that authoring does not carry.
    MissingShadeContext {
        /// Canonical runtime field name.
        field: &'static str,
    },
}

impl core::fmt::Display for ReferenceMaterialError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NonFiniteParameter { field } => {
                write!(
                    formatter,
                    "reference material field '{field}' is not finite"
                )
            }
            Self::ParameterOutOfRange { field } => {
                write!(
                    formatter,
                    "reference material field '{field}' is out of range"
                )
            }
            Self::UnsupportedField { field } => {
                write!(
                    formatter,
                    "reference material field '{field}' is unsupported"
                )
            }
            Self::UnsupportedTexture { field } => write!(
                formatter,
                "reference material field '{field}' requires a sampled texture"
            ),
            Self::MissingShadeContext { field } => write!(
                formatter,
                "reference material field '{field}' requires shade-time context"
            ),
        }
    }
}

impl std::error::Error for ReferenceMaterialError {}

/// Converts an adapted surface to the independent path-traced material value.
pub fn reference_material(
    surface: &RuntimeSurface,
) -> Result<ReferenceMaterial, ReferenceMaterialError> {
    reference_material_with_context(surface, ReferenceShadeContext::default())
}

/// Converts an adapted surface with the shade-time data required by active
/// glint. Every texture input is rejected as a complete conversion failure;
/// no authored field is replaced with a default texel.
pub fn reference_material_with_context(
    surface: &RuntimeSurface,
    context: ReferenceShadeContext,
) -> Result<ReferenceMaterial, ReferenceMaterialError> {
    if let Some(input) = surface.texture_inputs.iter().min_by(|left, right| {
        (left.field.as_str(), left.graph.as_str(), left.node.as_str()).cmp(&(
            right.field.as_str(),
            right.graph.as_str(),
            right.node.as_str(),
        ))
    }) {
        return Err(ReferenceMaterialError::UnsupportedTexture {
            field: input.field.clone(),
        });
    }
    let parameters = surface.parameters;
    validate_parameters(parameters)?;
    validate_shade_context(context)?;
    if parameters.normal_scale != 1.0 {
        return Err(ReferenceMaterialError::UnsupportedField {
            field: "normal_scale",
        });
    }
    if parameters.anisotropy != 0.0 || parameters.anisotropy_rotation != 0.0 {
        return Err(ReferenceMaterialError::UnsupportedField {
            field: "anisotropy",
        });
    }
    if parameters.coat_roughness_anisotropy != 0.0 {
        return Err(ReferenceMaterialError::UnsupportedField {
            field: "coat_roughness_anisotropy",
        });
    }
    if parameters.coat_weight != 0.0 && parameters.coat_roughness < 0.045 {
        return Err(ReferenceMaterialError::ParameterOutOfRange {
            field: "coat_roughness",
        });
    }
    if parameters.npr_face_softness != 0.1
        || parameters.npr_stylized_ramp_softness != 1.0
        || parameters.npr_stylized_specular_scale != 1.0
        || parameters.npr_stylized_rim_scale != 1.0
    {
        return Err(ReferenceMaterialError::UnsupportedField {
            field: "npr_response",
        });
    }
    if parameters.measured_weight != 0.0 {
        return Err(ReferenceMaterialError::UnsupportedField {
            field: "measured_brdf",
        });
    }
    if parameters.glint_weight != 0.0 && context.glint_footprint_area == 0.0 {
        return Err(ReferenceMaterialError::MissingShadeContext { field: "glint" });
    }

    Ok(ReferenceMaterial {
        openpbr: OpenPbrMaterial {
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
            transmission_color: rgb(parameters.transmission_color),
            transmission_scatter: rgb(parameters.transmission_scatter),
            transmission_scatter_anisotropy: parameters.transmission_scatter_anisotropy,
            transmission_dispersion_abbe_number: parameters.transmission_dispersion_abbe_number,
            coat_darkening: parameters.coat_darkening,
            scatter_anisotropy: parameters.subsurface_scatter_anisotropy,
            clearcoat: parameters.coat_weight,
            clearcoat_roughness: parameters.coat_roughness,
            coat_ior: parameters.coat_ior,
            coat_color: rgb(parameters.coat_color),
            sheen: parameters.fuzz_weight,
            fuzz_roughness: parameters.fuzz_roughness,
            fuzz_color: rgb(parameters.fuzz_color),
            subsurface: parameters.subsurface_weight,
            subsurface_color: rgb(parameters.subsurface_color),
            subsurface_radius: parameters.subsurface_radius,
            subsurface_radius_scale: rgb(parameters.subsurface_radius_scale),
            thickness: parameters.thickness,
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
        },
        emission: [
            parameters.emission_color[0] * parameters.emission_luminance,
            parameters.emission_color[1] * parameters.emission_luminance,
            parameters.emission_color[2] * parameters.emission_luminance,
        ],
        base_alpha: parameters.base_color[3],
        geometry_opacity: parameters.geometry_opacity,
        ambient_occlusion: parameters.ambient_occlusion,
        alpha_cutoff: parameters.alpha_cutoff,
    })
}

/// Converts canonical parameters directly, for CPU conformance fixtures that
/// already own their texture/sample boundary. This is the canonical ABI
/// adapter; [`reference_material_with_context`] adds authoring texture policy.
pub fn reference_material_from_parameters(
    parameters: GpuSurfaceParameters,
) -> Result<ReferenceMaterial, ReferenceMaterialError> {
    reference_material_from_parameters_with_context(parameters, ReferenceShadeContext::default())
}

/// Canonical-parameter adapter with explicit shade-time glint context.
pub fn reference_material_from_parameters_with_context(
    parameters: GpuSurfaceParameters,
    context: ReferenceShadeContext,
) -> Result<ReferenceMaterial, ReferenceMaterialError> {
    let surface = RuntimeSurface {
        source_category: "canonical_gpu_surface".to_owned(),
        parameters,
        texture_inputs: Vec::new(),
        report: Default::default(),
    };
    reference_material_with_context(&surface, context)
}

impl ReferenceMaterial {
    /// Evaluates the independent front-facing response before the light cosine.
    #[must_use]
    pub fn evaluate(&self, wo: Vec3, wi: Vec3, normal: Vec3) -> Vec3 {
        self.openpbr.evaluate(wo, wi, normal)
    }

    /// Evaluates the path-transport event, including the opposite-hemisphere
    /// transmission branch used by the reference integrator.
    #[must_use]
    pub fn evaluate_path(&self, wo: Vec3, wi: Vec3, normal: Vec3) -> Vec3 {
        self.openpbr.evaluate_path(wo, wi, normal)
    }

    /// Returns the exact mixture density paired with [`Self::sample`].
    #[must_use]
    pub fn pdf(&self, wo: Vec3, wi: Vec3, normal: Vec3) -> f32 {
        self.openpbr.pdf(wo, wi, normal)
    }

    /// Samples the independent reference closure with its matching PDF.
    #[must_use]
    pub fn sample(
        &self,
        wo: Vec3,
        normal: Vec3,
        rng: &mut impl SampleSource,
    ) -> Option<OpenPbrSample> {
        self.openpbr.sample(wo, normal, rng)
    }
}

fn validate_shade_context(context: ReferenceShadeContext) -> Result<(), ReferenceMaterialError> {
    if context
        .glint_cell
        .into_iter()
        .chain([context.glint_state, context.glint_footprint_area])
        .any(|value| !value.is_finite())
    {
        return Err(ReferenceMaterialError::NonFiniteParameter {
            field: "glint_shade_context",
        });
    }
    if context.glint_footprint_area < 0.0
        || (context.glint_state != -1.0 && !(0.0..=1.0).contains(&context.glint_state))
    {
        return Err(ReferenceMaterialError::ParameterOutOfRange {
            field: "glint_shade_context",
        });
    }
    Ok(())
}

fn validate_parameters(parameters: GpuSurfaceParameters) -> Result<(), ReferenceMaterialError> {
    let scalars = [
        ("emission_luminance", parameters.emission_luminance),
        ("base_metalness", parameters.base_metalness),
        ("specular_roughness", parameters.specular_roughness),
        ("specular_weight", parameters.specular_weight),
        ("ambient_occlusion", parameters.ambient_occlusion),
        ("normal_scale", parameters.normal_scale),
        ("alpha_cutoff", parameters.alpha_cutoff),
        ("base_weight", parameters.base_weight),
        ("base_diffuse_roughness", parameters.base_diffuse_roughness),
        ("specular_ior", parameters.specular_ior),
        ("thin_walled", parameters.thin_walled),
        ("geometry_opacity", parameters.geometry_opacity),
        ("transmission_weight", parameters.transmission_weight),
        ("thickness", parameters.thickness),
        ("coat_weight", parameters.coat_weight),
        ("coat_roughness", parameters.coat_roughness),
        ("coat_ior", parameters.coat_ior),
        (
            "coat_roughness_anisotropy",
            parameters.coat_roughness_anisotropy,
        ),
        ("coat_darkening", parameters.coat_darkening),
        ("anisotropy", parameters.anisotropy),
        ("anisotropy_rotation", parameters.anisotropy_rotation),
        ("fuzz_weight", parameters.fuzz_weight),
        ("fuzz_roughness", parameters.fuzz_roughness),
        ("subsurface_weight", parameters.subsurface_weight),
        ("subsurface_radius", parameters.subsurface_radius),
        (
            "subsurface_scatter_anisotropy",
            parameters.subsurface_scatter_anisotropy,
        ),
        ("index_of_refraction", parameters.index_of_refraction),
        (
            "transmission_dispersion_scale",
            parameters.transmission_dispersion_scale,
        ),
        (
            "transmission_scatter_anisotropy",
            parameters.transmission_scatter_anisotropy,
        ),
        (
            "transmission_dispersion_abbe_number",
            parameters.transmission_dispersion_abbe_number,
        ),
        ("thin_film_weight", parameters.thin_film_weight),
        ("thin_film_thickness", parameters.thin_film_thickness),
        ("thin_film_ior", parameters.thin_film_ior),
        ("glint_weight", parameters.glint_weight),
        ("glint_roughness", parameters.glint_roughness),
        ("glint_density", parameters.glint_density),
        ("glint_bin_half_width", parameters.glint_bin_half_width),
        ("npr_face_softness", parameters.npr_face_softness),
        (
            "npr_stylized_ramp_softness",
            parameters.npr_stylized_ramp_softness,
        ),
        (
            "npr_stylized_specular_scale",
            parameters.npr_stylized_specular_scale,
        ),
        ("npr_stylized_rim_scale", parameters.npr_stylized_rim_scale),
        ("measured_weight", parameters.measured_weight),
        ("measured_phi_d", parameters.measured_phi_d),
        ("measured_scale", parameters.measured_scale),
    ];
    if let Some((field, _)) = scalars.iter().find(|(_, value)| !value.is_finite()) {
        return Err(ReferenceMaterialError::NonFiniteParameter { field });
    }
    for (field, values) in [
        ("base_color", parameters.base_color),
        ("emission_color", parameters.emission_color),
        ("specular_color", parameters.specular_color),
        ("coat_color", parameters.coat_color),
        ("fuzz_color", parameters.fuzz_color),
        ("subsurface_color", parameters.subsurface_color),
        (
            "subsurface_radius_scale",
            parameters.subsurface_radius_scale,
        ),
        ("transmission_color", parameters.transmission_color),
        ("transmission_scatter", parameters.transmission_scatter),
    ] {
        if values.iter().any(|value| !value.is_finite()) {
            return Err(ReferenceMaterialError::NonFiniteParameter { field });
        }
    }
    for (field, value, min, max) in [
        ("base_metalness", parameters.base_metalness, 0.0, 1.0),
        (
            "specular_roughness",
            parameters.specular_roughness,
            0.0,
            1.0,
        ),
        ("specular_weight", parameters.specular_weight, 0.0, 1.0),
        ("alpha_cutoff", parameters.alpha_cutoff, 0.0, 1.0),
        ("base_weight", parameters.base_weight, 0.0, 1.0),
        (
            "base_diffuse_roughness",
            parameters.base_diffuse_roughness,
            0.0,
            1.0,
        ),
        ("specular_ior", parameters.specular_ior, 1.001, 3.0),
        ("thin_walled", parameters.thin_walled, 0.0, 1.0),
        ("geometry_opacity", parameters.geometry_opacity, 0.0, 1.0),
        (
            "transmission_weight",
            parameters.transmission_weight,
            0.0,
            1.0,
        ),
        ("coat_weight", parameters.coat_weight, 0.0, 1.0),
        ("coat_roughness", parameters.coat_roughness, 0.0, 1.0),
        ("coat_ior", parameters.coat_ior, 1.001, 3.0),
        ("coat_darkening", parameters.coat_darkening, 0.0, 1.0),
        ("fuzz_weight", parameters.fuzz_weight, 0.0, 1.0),
        ("fuzz_roughness", parameters.fuzz_roughness, 0.0, 1.0),
        ("subsurface_weight", parameters.subsurface_weight, 0.0, 1.0),
        (
            "index_of_refraction",
            parameters.index_of_refraction,
            1.0,
            3.0,
        ),
        (
            "transmission_dispersion_scale",
            parameters.transmission_dispersion_scale,
            0.0,
            1.0,
        ),
        ("thin_film_weight", parameters.thin_film_weight, 0.0, 1.0),
        ("thin_film_ior", parameters.thin_film_ior, 1.0, 3.0),
        ("glint_weight", parameters.glint_weight, 0.0, 1.0),
        ("glint_roughness", parameters.glint_roughness, 0.0, 1.0),
        ("measured_weight", parameters.measured_weight, 0.0, 1.0),
        ("npr_face_softness", parameters.npr_face_softness, 0.0, 0.5),
        (
            "glint_bin_half_width",
            parameters.glint_bin_half_width,
            0.0,
            core::f32::consts::PI,
        ),
    ] {
        if !(min..=max).contains(&value) {
            return Err(ReferenceMaterialError::ParameterOutOfRange { field });
        }
    }
    for (field, value) in [
        ("emission_luminance", parameters.emission_luminance),
        ("normal_scale", parameters.normal_scale),
        ("thickness", parameters.thickness),
        ("subsurface_radius", parameters.subsurface_radius),
        ("thin_film_thickness", parameters.thin_film_thickness),
        ("glint_density", parameters.glint_density),
        (
            "npr_stylized_ramp_softness",
            parameters.npr_stylized_ramp_softness,
        ),
        (
            "npr_stylized_specular_scale",
            parameters.npr_stylized_specular_scale,
        ),
        ("npr_stylized_rim_scale", parameters.npr_stylized_rim_scale),
        ("measured_scale", parameters.measured_scale),
    ] {
        if value < 0.0 {
            return Err(ReferenceMaterialError::ParameterOutOfRange { field });
        }
    }
    if !(-1.0..=1.0).contains(&parameters.subsurface_scatter_anisotropy)
        || !(-1.0..=1.0).contains(&parameters.transmission_scatter_anisotropy)
        || !(-1.0..=1.0).contains(&parameters.anisotropy)
    {
        return Err(ReferenceMaterialError::ParameterOutOfRange {
            field: "anisotropy",
        });
    }
    if !(1.0..=200.0).contains(&parameters.transmission_dispersion_abbe_number) {
        return Err(ReferenceMaterialError::ParameterOutOfRange {
            field: "transmission_dispersion_abbe_number",
        });
    }
    if parameters.measured_phi_d <= 0.0 {
        return Err(ReferenceMaterialError::ParameterOutOfRange {
            field: "measured_phi_d",
        });
    }
    for (field, values) in [
        ("base_color", parameters.base_color),
        ("specular_color", parameters.specular_color),
        ("coat_color", parameters.coat_color),
        ("fuzz_color", parameters.fuzz_color),
        ("subsurface_color", parameters.subsurface_color),
        (
            "subsurface_radius_scale",
            parameters.subsurface_radius_scale,
        ),
        ("transmission_color", parameters.transmission_color),
        ("transmission_scatter", parameters.transmission_scatter),
    ] {
        if values.iter().any(|value| !(0.0..=1.0).contains(value)) {
            return Err(ReferenceMaterialError::ParameterOutOfRange { field });
        }
    }
    Ok(())
}

fn rgb(value: [f32; 4]) -> Vec3 {
    Vec3::new(value[0], value[1], value[2])
}
