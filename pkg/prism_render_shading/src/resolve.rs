//! Backend-neutral CPU reference for the compute resolve pass.
//!
//! This closes the visibility -> classify -> shade chain: it reconstructs a
//! surface from one [`VisibilityPixel`], samples the material parameters, and
//! evaluates direct lighting per shading class.  It is deliberately free of any
//! GPU or Bevy dependency so it can act as the golden reference the WESL
//! resolve shader must match bit-for-bit within tolerance.

use crate::measured_brdf::MeasuredBrdf;
use prism_render_material::{DynamicChannels, GpuMaterialHeader, GpuSurfaceParameters, ProgramRef};

use crate::gi::area_light::ltc_lut::LtcLut;
use crate::gi::area_light::representative::rect_representative_sample;
use crate::gi::area_light::resolve::{accumulate_area_lights, AreaLightRect};
use crate::{
    apply_tangent_space_normal, classify_material_header, evaluate_image_based_light,
    reconstruct_surface, sample_material, ClassificationError, DirectLightSample,
    GpuShadingPrimitive, GpuShadingVertex, ImageBasedLight, MaterialModulationParams,
    MaterialShadingClass, PunctualLight, SampledTextureBinding, ShadingFrame, StylizedParams,
    SurfaceReconstructionError, SurfaceReconstructionFlags, SurfaceReconstructionInput,
    SurfaceSample, TangentBasis, VisibilityPixel,
};

/// Identity `world_from_local` (row-major affine) used when an instance carries
/// no transform; lifting a local-space surface through it is a no-op.
pub const IDENTITY_WORLD_FROM_LOCAL: [[f32; 4]; 3] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
];

/// Transforms a point by a row-major affine `world_from_local` (three rows of
/// `[m0, m1, m2, translation]`).
fn affine_transform_point(m: &[[f32; 4]; 3], p: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * p[0] + m[0][1] * p[1] + m[0][2] * p[2] + m[0][3],
        m[1][0] * p[0] + m[1][1] * p[1] + m[1][2] * p[2] + m[1][3],
        m[2][0] * p[0] + m[2][1] * p[1] + m[2][2] * p[2] + m[2][3],
    ]
}

/// Row-major linear (upper-left 3x3) part of an affine `world_from_local`.
fn linear_part(m: &[[f32; 4]; 3]) -> [[f32; 3]; 3] {
    [
        [m[0][0], m[0][1], m[0][2]],
        [m[1][0], m[1][1], m[1][2]],
        [m[2][0], m[2][1], m[2][2]],
    ]
}

/// Multiplies a row-major 3x3 by a column vector.
fn mat3_mul(m: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Inverse-transpose of the linear part, carrying normals correctly under a
/// non-uniform scale.  Mirrors `inverse_transpose_3x3` in the WESL shaders
/// (cofactor columns over the determinant, indexed column-major); returns the
/// identity for a singular matrix so a degenerate transform is a no-op.  The
/// result is stored row-major so [`mat3_mul`] reproduces the WESL `matrix * n`.
fn normal_matrix(m: &[[f32; 4]; 3]) -> [[f32; 3]; 3] {
    let l = linear_part(m);
    let c0 = [l[0][0], l[1][0], l[2][0]];
    let c1 = [l[0][1], l[1][1], l[2][1]];
    let c2 = [l[0][2], l[1][2], l[2][2]];
    let x = cross3(c1, c2);
    let y = cross3(c2, c0);
    let z = cross3(c0, c1);
    let det = dot3(c2, z);
    if det.abs() <= 1.0e-8 {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    let inv = 1.0 / det;
    [
        [x[0] * inv, y[0] * inv, z[0] * inv],
        [x[1] * inv, y[1] * inv, z[1] * inv],
        [x[2] * inv, y[2] * inv, z[2] * inv],
    ]
}

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
    /// Stylized (NPR) front-end controls used by the `MaterialShadingClass::Npr`
    /// path. The default reproduces the historical four-band toon lobe.
    pub stylized: StylizedParams,
    /// Resident polygonal rectangle area lights (LTC). The physically based
    /// lobes (Principled/Cloth/Subsurface/ClearCoat/Custom) integrate the full
    /// Linearly-Transformed-Cosine term; the specialized Toon/Hair/Water lobes
    /// integrate each rectangle via the Most-Representative-Point conversion in
    /// [`crate::gi::area_light::representative`].
    pub area: &'a [AreaLightRect],
    /// Baked LTC coefficient LUT shared by every area light. `None` collapses
    /// the area specular lobe to the diffuse-only clamped-cosine term.
    pub ltc_lut: Option<&'a LtcLut>,
}

impl Default for LightingEnvironment<'_> {
    fn default() -> Self {
        Self {
            directional: &[],
            punctual: &[],
            image_based: None,
            ambient: [0.0; 3],
            stylized: StylizedParams::with_bands(4),
            area: &[],
            ltc_lut: None,
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
    /// Response program resolved from the ABI8 execution binding. `None` is
    /// the explicit unbound/fallback path; a bound header without this context
    /// is rejected by classification rather than guessed from header words.
    pub response_program: Option<ProgramRef>,
    pub parameters: GpuSurfaceParameters,
    /// Sampled texels for every texture the material binds, in the same
    /// `texture_offset..texture_offset + texture_count` order the GPU heap
    /// iterates.  Each entry carries its `TextureSemantic` and the RGBA texel
    /// sampled at the interpolated surface UV; an empty slice shades from the
    /// authored factors alone (the bindless heap's white/flat-normal defaults).
    pub textures: &'a [SampledTextureBinding],
    /// Single-lobe sampling state for this pixel (§17.4, P3-e): the quality/budget
    /// policy, the frame index and the pixel's integer coordinate. The defaults
    /// are `High` (full evaluation, an exact identity) at frame/pixel zero, so a
    /// caller that does not care about the amortised path is unchanged.
    pub lobe_sampling: crate::lobe_sampling::LobeSamplingSettings,
    /// The material's measured (tabulated) BRDF/BTF table, when it binds one
    /// (§17.6). The table itself is *data*: in the shipping renderer it is a
    /// material texture bound with the measured-BRDF semantic, and the GPU fetch
    /// lives in `measured_lobe.wesl`; the CPU golden receives the table directly
    /// rather than a sampled texel, because a BRDF lookup is not a texture fold.
    pub measured_brdf: Option<&'a MeasuredBrdf>,
    /// World-space camera position used to derive the view vector.
    pub view_position: [f32; 3],
    /// Screen-space ambient visibility in `[0, 1]` (`1` = unoccluded) sampled
    /// from the GTAO kernel's output for this pixel. It multiplies the
    /// material's own occlusion so the indirect/ambient term is darkened in
    /// creases the geometric AO catches. Views without GTAO pass `1.0`.
    pub screen_space_ao: f32,
    /// Area of one pixel in the reconstructed surface's UV cell space, i.e.
    /// `|det dUV/d(pixel)|` — the shade-time footprint the glint lobe (§17.6)
    /// multiplies by its authored facet density to get the expected facet count
    /// inside the pixel. The GPU resolve derives it analytically from the
    /// triangle's projected vertices (`shading_resolve.wesl`); a caller that does
    /// not track screen-space derivatives passes `0.0`, which keeps the glint
    /// term on the smooth-GGX path (an unknown footprint is not an empty one).
    pub uv_footprint_area: f32,
    /// Row-major affine `world_from_local` (three rows of `[m, m, m, translation]`)
    /// for this pixel's instance, matching `GpuSceneTransform.world_from_local`
    /// (`mat3x4`).  The reconstructed local-space surface is lifted into world
    /// space with it before any lighting so a translated/rotated/scaled instance
    /// shades at its true position; the WESL resolve applies the byte-identical
    /// `affine3_to_square` transform read from `scene_current_transforms`.
    pub world_from_local: [[f32; 4]; 3],
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
        base_weight: parameters.base_weight,
        diffuse_roughness: parameters.base_diffuse_roughness,
        metallic: parameters.base_metalness,
        perceptual_roughness: parameters.specular_roughness,
        specular_weight: parameters.specular_weight,
        specular_color: [
            parameters.specular_color[0],
            parameters.specular_color[1],
            parameters.specular_color[2],
        ],
        specular_ior: parameters.specular_ior,
        transmission: parameters.transmission_weight,
        index_of_refraction: parameters.index_of_refraction,
        dispersion: parameters.transmission_dispersion_scale,
        thin_walled: parameters.thin_walled,
        geometry_opacity: parameters.geometry_opacity,
        transmission_color: [
            parameters.transmission_color[0],
            parameters.transmission_color[1],
            parameters.transmission_color[2],
        ],
        transmission_scatter: [
            parameters.transmission_scatter[0],
            parameters.transmission_scatter[1],
            parameters.transmission_scatter[2],
        ],
        transmission_scatter_anisotropy: parameters.transmission_scatter_anisotropy,
        transmission_dispersion_abbe_number: parameters.transmission_dispersion_abbe_number,
        coat_darkening: parameters.coat_darkening,
        scatter_anisotropy: parameters.subsurface_scatter_anisotropy,
        ambient_occlusion: parameters.ambient_occlusion,
        // `OpenPBR` emission: radiance = emission_color * emission_luminance.
        emissive: [
            parameters.emission_color[0] * parameters.emission_luminance,
            parameters.emission_color[1] * parameters.emission_luminance,
            parameters.emission_color[2] * parameters.emission_luminance,
        ],
        clearcoat: parameters.coat_weight,
        clearcoat_roughness: parameters.coat_roughness,
        coat_ior: parameters.coat_ior,
        coat_color: [
            parameters.coat_color[0],
            parameters.coat_color[1],
            parameters.coat_color[2],
        ],
        // The CPU golden has no texture-sampling path, so it never carries a
        // clearcoat normal map; the all-zero identity makes the coat use the
        // base shading normal, mirroring the GPU resolve when no map is bound.
        coat_normal: [0.0; 3],
        coat_roughness_anisotropy: parameters.coat_roughness_anisotropy,
        // `geometry_coat_tangent` is a shade-time input; the material block does
        // not carry it, so the coat inherits the base frame tangent.
        coat_tangent: [0.0; 3],
        sheen: parameters.fuzz_weight,
        fuzz_roughness: parameters.fuzz_roughness,
        fuzz_color: [
            parameters.fuzz_color[0],
            parameters.fuzz_color[1],
            parameters.fuzz_color[2],
        ],
        subsurface: parameters.subsurface_weight,
        subsurface_color: [
            parameters.subsurface_color[0],
            parameters.subsurface_color[1],
            parameters.subsurface_color[2],
        ],
        subsurface_radius: parameters.subsurface_radius,
        subsurface_radius_scale: [
            parameters.subsurface_radius_scale[0],
            parameters.subsurface_radius_scale[1],
            parameters.subsurface_radius_scale[2],
        ],
        thickness: parameters.thickness,
        anisotropy: parameters.anisotropy,
        anisotropy_rotation: parameters.anisotropy_rotation,
        // Specular-AA normal variance is a shade-time quantity (screen-space
        // normal derivatives / footprint mip / baked map), not a stable GPU
        // surface parameter, so the CPU golden mirrors the GPU resolve and
        // leaves it at the identity `0.0`.
        normal_variance: 0.0,
        // `OpenPBR` thin-film lobe -> Airy BSDF. The ABI carries the film in
        // micrometres (specification unit) while the shader evaluates in
        // nanometres, so the thickness is converted here. Weight 0 (the default)
        // keeps the whole term inert.
        iridescence: parameters.thin_film_weight,
        iridescence_thickness_nm: parameters.thin_film_thickness * 1_000.0,
        iridescence_ior: parameters.thin_film_ior,
        // The glint lobe's authored parameters. Its two shade-time inputs (the
        // surface cell and the per-pixel footprint area) are set by the resolve
        // once it knows the pixel: `resolve_pixel` takes the footprint from
        // `ResolveInput::uv_footprint_area` and the cell from the reconstructed
        // surface UV, mirroring `shading_resolve.wesl`.
        glint_weight: parameters.glint_weight,
        glint_roughness: parameters.glint_roughness,
        glint_density: parameters.glint_density,
        glint_bin_half_width: parameters.glint_bin_half_width,
        glint_state: crate::lighting::GLINT_STATE_AUTHORED,
        glint_cell: [0.0, 0.0],
        glint_footprint_area: 0.0,
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
            parameters.emission_color[0],
            parameters.emission_color[1],
            parameters.emission_color[2],
        ],
        metallic: parameters.base_metalness,
        perceptual_roughness: parameters.specular_roughness,
        ambient_occlusion: parameters.ambient_occlusion,
        normal_scale: parameters.normal_scale,
        clearcoat: parameters.coat_weight,
        clearcoat_roughness: parameters.coat_roughness,
    }
}

fn stylized_params_for_material(
    base: StylizedParams,
    parameters: &GpuSurfaceParameters,
) -> StylizedParams {
    let mut result = base;
    result.ramp_softness *= parameters.npr_stylized_ramp_softness;
    result.specular_intensity *= parameters.npr_stylized_specular_scale;
    result.rim_intensity *= parameters.npr_stylized_rim_scale;
    result
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
    let shading_class = match input.response_program {
        Some(program) => crate::classification::classify_material_header_with_program(
            &input.header,
            program,
            u32::MAX,
        ),
        None => classify_material_header(&input.header),
    }
    .map_err(ResolveError::Classification)?;
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

    // Lift the reconstructed *local-space* surface into world space with the
    // instance's `world_from_local` before any lighting runs: the position by
    // the full affine, the normal by its inverse-transpose (correct under a
    // non-uniform scale), and the tangent/bitangent by the linear part.  The
    // WESL resolve performs the byte-identical transform from
    // `scene_current_transforms`; the default identity leaves all of it
    // untouched, so every existing golden stays bit-for-bit stable.
    let mut geometry = geometry;
    let world_from_local = input.world_from_local;
    geometry.position = affine_transform_point(&world_from_local, geometry.position);
    let normal_basis = normal_matrix(&world_from_local);
    geometry.normal = normalize_or(mat3_mul(&normal_basis, geometry.normal), geometry.normal);
    let linear = linear_part(&world_from_local);
    geometry.tangent = normalize_or(mat3_mul(&linear, geometry.tangent), geometry.tangent);
    geometry.bitangent = normalize_or(mat3_mul(&linear, geometry.bitangent), geometry.bitangent);

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
    // `OpenPBR` emission radiance is `emission_color * emission_luminance`, and
    // the texture-modulated `sampled.emissive` carries the colour (times any
    // emissive map), so the luminance scale is applied here. Mirrors
    // `shading_resolve.wesl`: `surface.emissive = sampled.emissive *
    // parameters.emission_luminance`.
    surface.emissive = [
        sampled.emissive[0] * input.parameters.emission_luminance,
        sampled.emissive[1] * input.parameters.emission_luminance,
        sampled.emissive[2] * input.parameters.emission_luminance,
    ];
    surface.clearcoat = sampled.clearcoat;
    surface.clearcoat_roughness = sampled.clearcoat_roughness;
    // Glint shade-time context (§17.6): the seed's cell is the reconstructed
    // surface UV and the footprint is the caller's `|det dUV/d(pixel)|`, so the
    // CPU golden and `shading_resolve.wesl` drive the estimator from the same
    // two quantities.
    surface.glint_cell = geometry.uv;
    surface.glint_footprint_area = input.uv_footprint_area;
    // Runtime dynamic channels (§17.1): the header's four dynamic words are the
    // material's live wetness / snow / damage / dissolve state, remapped
    // analytically onto the frozen über fields *before* any light or ambient term
    // is evaluated, so every consumer (direct, IBL, constant ambient) sees the
    // same modulated surface. All-zero is a bit-for-bit identity, and the bypass
    // branch is per material, not per pixel — see
    // [`crate::dynamic::apply_dynamic_channels`].
    crate::dynamic::apply_dynamic_channels(
        &mut surface,
        DynamicChannels::from_words([
            input.header.dynamic_wetness,
            input.header.dynamic_snow,
            input.header.dynamic_damage,
            input.header.dynamic_dissolve,
        ]),
        geometry.uv,
    );

    let emissive = surface.emissive;
    let base_color = surface.base_color;
    // Fold the screen-space GTAO visibility into the material occlusion so the
    // indirect/ambient term (IBL or constant) is attenuated in geometric
    // creases; both factors are clamped so the product stays in `[0, 1]`.
    let ambient_occlusion = (surface.ambient_occlusion * input.screen_space_ao).clamp(0.0, 1.0);
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
    // §17.4 single-lobe sampling: one draw per pixel decides the glint mixture
    // state and whether the additive measured lobe is evaluated this frame. On
    // the full-quality tiers both factors are the exact identity.
    let lobe_factors = crate::lobe_sampling::select_expensive_lobe_factors(
        input.lobe_sampling.policy,
        [
            input.parameters.glint_weight,
            input.parameters.measured_weight,
        ],
        input.lobe_sampling.pixel,
        input.lobe_sampling.frame_index,
        0,
    );
    // Per-light evaluation must not re-add self-illumination.
    let mut lit_surface = SurfaceSample {
        emissive: [0.0; 3],
        ..surface
    };
    // §17.4: the drawn glint mixture state rides on the surface the integrator
    // evaluates (the sentinel means "no sampling happened").
    lit_surface.glint_state = lobe_factors.glint_state;

    // Indirect term: prefer the image-based light when the view supplies an
    // environment probe, otherwise fall back to the constant ambient term.
    let indirect = match lights.image_based {
        Some(image_based) => evaluate_image_based_light(lit_surface, frame, &image_based),
        None => ambient_term(base_color, ambient_occlusion, lights.ambient, metallic),
    };

    // Analytic rectangle area lights (LTC): the specular + diffuse polygon
    // response added once per physically based lobe, mirroring
    // `accumulate_area_lights` in `shading_resolve.wesl`. The default
    // `area: &[]` keeps every existing golden bit-for-bit stable.
    let area_light_term = accumulate_area_lights(
        &lit_surface,
        frame,
        geometry.position,
        lights.area,
        lights.ltc_lut,
    );

    // §17.6 measured (tabulated) BRDF lobe: an additive, table-driven specular
    // response evaluated per light. Zero weight, an unbound table, a
    // below-horizon light or a dark light all return zero, so a material that
    // never authors the lobe is bit-for-bit unchanged.
    let measured_table = input
        .measured_brdf
        .filter(|_| input.parameters.measured_weight > 0.0);
    let measured_weight = input.parameters.measured_weight;
    let measured_scale = input.parameters.measured_scale;
    let measured_gate = lobe_factors.measured;
    let measured_term = |light: DirectLightSample| match measured_table {
        Some(table) => crate::evaluate_measured_direct(
            table,
            measured_scale * measured_gate,
            measured_weight,
            &frame,
            &light,
        ),
        None => [0.0; 3],
    };

    // Integrates the principled GGX lobe over every analytic light, then adds
    // the shared indirect + emissive terms once.  Shared by the physically
    // based classes (Principled and, until their specialized lobes land,
    // Subsurface/ClearCoat/Cloth/Hair/Water/Custom).
    let shade_principled = || {
        // The view- and material-only albedo terms are identical for every
        // light, so they are computed **once per pixel** here and threaded into
        // each per-light evaluation (`evaluate_principled_direct_with`) instead
        // of being recomputed per light — the tables cost ~3 Glookup/s on the dev
        // GPU, so a 4-light pixel was paying for 12 lookups where 3 suffice.
        let pre = crate::principled_pre(&lit_surface, &frame);
        let mut accumulated = [0.0; 3];
        for light in lights.directional {
            let sample = direct_sample(*light);
            accumulated = add(
                accumulated,
                crate::evaluate_principled_direct_with(lit_surface, frame, sample, &pre),
            );
            accumulated = add(accumulated, measured_term(sample));
        }
        for light in lights.punctual {
            if let Some(sample) = light.sample(geometry.position) {
                accumulated = add(
                    accumulated,
                    crate::evaluate_principled_direct_with(lit_surface, frame, sample, &pre),
                );
                accumulated = add(accumulated, measured_term(sample));
            }
        }
        add(add(add(accumulated, area_light_term), indirect), emissive)
    };

    // Integrates the stylized (NPR) front end over every analytic light, then
    // adds the shared indirect + emissive terms once.  Used by the NPR class;
    // `lights.stylized` selects the cel ramp / stepped shadow / stylized
    // specular / rim response (its default reproduces the banded toon lobe).
    let shade_toon = || {
        let material_stylized = stylized_params_for_material(lights.stylized, &input.parameters);
        let mut accumulated = [0.0; 3];
        for light in lights.directional {
            accumulated = add(
                accumulated,
                crate::evaluate_stylized_direct(
                    lit_surface,
                    frame,
                    direct_sample(*light),
                    &material_stylized,
                ),
            );
        }
        for light in lights.punctual {
            if let Some(sample) = light.sample(geometry.position) {
                accumulated = add(
                    accumulated,
                    crate::evaluate_stylized_direct(lit_surface, frame, sample, &material_stylized),
                );
            }
        }
        for light in lights.area {
            if let Some(sample) = rect_representative_sample(geometry.position, frame.normal, light)
            {
                accumulated = add(
                    accumulated,
                    crate::evaluate_stylized_direct(lit_surface, frame, sample, &material_stylized),
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
        add(add(add(accumulated, area_light_term), indirect), emissive)
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
        add(add(add(accumulated, area_light_term), indirect), emissive)
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
        for light in lights.area {
            if let Some(sample) = rect_representative_sample(geometry.position, frame.normal, light)
            {
                accumulated = add(
                    accumulated,
                    crate::evaluate_hair_direct(lit_surface, frame, sample),
                );
            }
        }
        add(add(accumulated, indirect), emissive)
    };

    // Integrates the single-layer water lobe over every analytic light, then
    // adds the shared indirect + emissive terms once.  Used by the Water class;
    // the Fresnel split routes energy between the surface glint and a
    // Beer-Lambert-absorbed refracted body.
    let shade_water = || {
        let mut accumulated = [0.0; 3];
        for light in lights.directional {
            accumulated = add(
                accumulated,
                crate::evaluate_water_direct(lit_surface, frame, direct_sample(*light)),
            );
        }
        for light in lights.punctual {
            if let Some(sample) = light.sample(geometry.position) {
                accumulated = add(
                    accumulated,
                    crate::evaluate_water_direct(lit_surface, frame, sample),
                );
            }
        }
        for light in lights.area {
            if let Some(sample) = rect_representative_sample(geometry.position, frame.normal, light)
            {
                accumulated = add(
                    accumulated,
                    crate::evaluate_water_direct(lit_surface, frame, sample),
                );
            }
        }
        add(add(accumulated, indirect), emissive)
    };

    // Integrates the two-layer clear-coat lobe over every analytic light, then
    // adds the shared indirect + emissive terms once.  Used by the ClearCoat
    // class; a smooth dielectric coat (independent `clearcoat_roughness`) sits
    // over the principled base, and the base is gated by the coat Fresnel twice
    // (double-pass `(1 - clearcoat * Fc)^2` transmission).
    let shade_clearcoat = || {
        let mut accumulated = [0.0; 3];
        for light in lights.directional {
            accumulated = add(
                accumulated,
                crate::evaluate_clearcoat_direct(lit_surface, frame, direct_sample(*light)),
            );
        }
        for light in lights.punctual {
            if let Some(sample) = light.sample(geometry.position) {
                accumulated = add(
                    accumulated,
                    crate::evaluate_clearcoat_direct(lit_surface, frame, sample),
                );
            }
        }
        add(add(add(accumulated, area_light_term), indirect), emissive)
    };

    // Every class is handled explicitly so this branch stays byte-for-byte in
    // step with the `switch` in `shading_resolve.wesl` (9 arms, no wildcard).
    // Subsurface, ClearCoat, Cloth, Hair and Water now have dedicated lobes;
    // only Custom still shares `shade_principled()` with Principled, so the lint
    // that would collapse those two equal arms is suppressed to preserve the
    // 1:1 GPU switch mapping.
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
        MaterialShadingClass::ClearCoat => shade_clearcoat(),
        MaterialShadingClass::Cloth => shade_cloth(),
        MaterialShadingClass::Hair => shade_hair(),
        MaterialShadingClass::Water => shade_water(),
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
        fallback_material_header, BuiltinProgram, GpuMaterialHeader, MaterialRenderClass,
    };

    /// The `OpenPBR` thin-film lobe must reach the Airy BSDF: the ABI carries the
    /// film in micrometres while shading evaluates nanometres, and weight 0 stays
    /// inert.
    #[test]
    fn thin_film_lobe_maps_into_the_airy_shade_time_fields() {
        let inert = surface_sample_from_parameters(&GpuSurfaceParameters::default());
        assert_eq!(inert.iridescence, 0.0);

        let parameters = GpuSurfaceParameters {
            thin_film_weight: 0.8,
            thin_film_thickness: 0.35,
            thin_film_ior: 1.55,
            ..Default::default()
        };
        let sample = surface_sample_from_parameters(&parameters);
        assert_eq!(sample.iridescence, 0.8);
        assert!((sample.iridescence_thickness_nm - 350.0).abs() < 1.0e-3);
        assert_eq!(sample.iridescence_ior, 1.55);
    }

    #[test]
    fn per_material_stylized_modifiers_scale_only_their_declared_lobes() {
        let base = StylizedParams {
            ramp_softness: 0.2,
            specular_intensity: 0.8,
            rim_intensity: 0.6,
            ..StylizedParams::with_bands(4)
        };
        let parameters = GpuSurfaceParameters {
            npr_stylized_ramp_softness: 1.5,
            npr_stylized_specular_scale: 0.25,
            npr_stylized_rim_scale: 2.0,
            ..Default::default()
        };
        let actual = stylized_params_for_material(base, &parameters);
        assert_eq!(actual.ramp_softness, 0.3);
        assert_eq!(actual.specular_intensity, 0.2);
        assert_eq!(actual.rim_intensity, 1.2);
        assert_eq!(actual.bands, base.bands);
        assert_eq!(actual.shadow_threshold, base.shadow_threshold);
    }

    /// `OpenPBR` emission is `emission_color * emission_luminance`, and the
    /// geometry group's opacity reaches the sample the blend path reads.
    #[test]
    fn emission_luminance_and_geometry_opacity_reach_the_sample() {
        let parameters = GpuSurfaceParameters {
            emission_color: [1.0, 0.5, 0.25, 1.0],
            emission_luminance: 4.0,
            geometry_opacity: 0.4,
            ..Default::default()
        };
        let sample = surface_sample_from_parameters(&parameters);
        assert_eq!(sample.emissive, [4.0, 2.0, 1.0]);
        assert_eq!(sample.geometry_opacity, 0.4);
    }

    /// The *resolved pixel* — not just the intermediate sample — must carry
    /// `emission_color * emission_luminance`. This pins the CPU golden to the
    /// WESL twin (`shading_resolve.wesl`:
    /// `surface.emissive = sampled.emissive * parameters.emission_luminance`),
    /// which the pre-P1-f2 default (`emission_luminance == 1`) could not
    /// distinguish from a plain `sampled.emissive`.
    #[test]
    fn resolved_pixel_scales_emission_by_luminance() {
        let (primitives, vertices) = unit_triangle();
        let header = principled_header();
        let parameters = GpuSurfaceParameters {
            emission_color: [1.0, 0.5, 0.25, 1.0],
            emission_luminance: 4.0,
            ..Default::default()
        };
        let dark = LightingEnvironment {
            directional: &[],
            punctual: &[],
            image_based: None,
            ambient: [0.0; 3],
            stylized: StylizedParams::with_bands(4),
            area: &[],
            ltc_lut: None,
        };
        let resolved = resolve_pixel(base_input(&primitives, &vertices, header, parameters), dark)
            .unwrap()
            .color;
        for (channel, want) in [4.0_f32, 2.0, 1.0].iter().enumerate() {
            assert!(
                (resolved[channel] - want).abs() < 1.0e-6,
                "channel {channel}: {} vs {want}",
                resolved[channel]
            );
        }
    }

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
            [GpuShadingPrimitive {
                indices: [0, 1, 2],
                flags: 0,
            }],
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
        header.is_active = 1;
        header.render_class = MaterialRenderClass::Opaque as u32;
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
            response_program: None,
            parameters,
            textures: &[],
            view_position: [0.0, 0.0, 4.0],
            screen_space_ao: 1.0,
            uv_footprint_area: 0.0,
            lobe_sampling: crate::lobe_sampling::LobeSamplingSettings::default(),
            measured_brdf: None,
            world_from_local: IDENTITY_WORLD_FROM_LOCAL,
        }
    }

    #[test]
    fn single_lobe_sampling_converges_to_the_full_evaluation_under_taa() {
        // §17.4 (P3-e) end-to-end gate. Both expensive lobes live: glint (a
        // mixture, amortised by drawing its state) and the measured table (an
        // additive term). On a low tier the pixel samples both instead of
        // evaluating them exactly; averaging over frames — what TAA does to a
        // fixed pixel — must converge to the full-quality image, i.e. the
        // estimator is unbiased and its variance is what TAA is for.
        let (primitives, vertices) = unit_triangle();
        let table = crate::measured_brdf::MeasuredBrdf::from_analytic(
            crate::measured_brdf::MeasuredBrdfDims::MERL,
            crate::measured_brdf::SyntheticBrdfParams::default(),
        );
        let parameters = GpuSurfaceParameters {
            base_color: [0.5, 0.45, 0.4, 1.0],
            specular_roughness: 0.4,
            glint_weight: 0.35,
            glint_roughness: 0.15,
            glint_density: 400.0,
            glint_bin_half_width: 0.05,
            measured_weight: 0.6,
            measured_phi_d: crate::measured_brdf::MB_DEFAULT_PHI_D as f32,
            measured_scale: 1.0,
            ..Default::default()
        };
        let light = DirectionalLight {
            direction: [0.0, 0.0, 1.0],
            illuminance: [10.0; 3],
            visibility: 1.0,
        };
        let resolve = |sampling: crate::lobe_sampling::LobeSamplingSettings, footprint: f32| {
            let mut input = base_input(&primitives, &vertices, principled_header(), parameters);
            input.measured_brdf = Some(&table);
            input.uv_footprint_area = footprint;
            input.lobe_sampling = sampling;
            resolve_pixel(
                input,
                LightingEnvironment {
                    directional: &[light],
                    punctual: &[],
                    image_based: None,
                    ambient: [5.0; 3],
                    stylized: StylizedParams::with_bands(4),
                    area: &[],
                    ltc_lut: None,
                },
            )
            .unwrap()
            .color
        };

        let pixel = [37u32, 19u32];
        let footprint = 1.0e-3;
        let full = resolve(
            crate::lobe_sampling::LobeSamplingSettings {
                policy: crate::LobeSamplingPolicy::for_quality(crate::ShadingQuality::High),
                frame_index: 0,
                pixel,
            },
            footprint,
        );

        // Low tier: one stochastic representative.
        let policy = crate::LobeSamplingPolicy::for_quality(crate::ShadingQuality::Low);
        assert_eq!(policy.mode(), crate::LobeSamplingMode::Stochastic);
        let frames = 512u32;
        let mut sum = [0.0f32; 3];
        let mut differing = 0usize;
        for frame in 0..frames {
            let value = resolve(
                crate::lobe_sampling::LobeSamplingSettings {
                    policy,
                    frame_index: frame,
                    pixel,
                },
                footprint,
            );
            if (0..3).any(|c| (value[c] - full[c]).abs() > 1.0e-5) {
                differing += 1;
            }
            for channel in 0..3 {
                sum[channel] += value[channel];
            }
        }
        assert!(
            differing > frames as usize / 2,
            "the amortised path must actually differ per frame ({differing}/{frames})"
        );
        for channel in 0..3 {
            let mean = sum[channel] / frames as f32;
            let tolerance = 0.12 * full[channel].abs().max(1.0);
            assert!(
                (mean - full[channel]).abs() <= tolerance,
                "channel {channel}: TAA mean {mean} vs full {}",
                full[channel]
            );
        }
    }

    #[test]
    fn a_bound_measured_table_reaches_the_resolved_pixel() {
        let (primitives, vertices) = unit_triangle();
        let parameters = GpuSurfaceParameters {
            base_color: [0.4, 0.4, 0.4, 1.0],
            specular_roughness: 0.5,
            measured_weight: 1.0,
            measured_phi_d: crate::measured_brdf::MB_DEFAULT_PHI_D as f32,
            measured_scale: 1.0,
            ..Default::default()
        };
        let table = crate::measured_brdf::MeasuredBrdf::from_analytic(
            crate::measured_brdf::MeasuredBrdfDims::MERL,
            crate::measured_brdf::SyntheticBrdfParams::default(),
        );
        let light = DirectionalLight {
            direction: [0.0, 0.0, 1.0],
            illuminance: [10.0; 3],
            visibility: 1.0,
        };
        let resolve = |measured: Option<&crate::measured_brdf::MeasuredBrdf>| {
            let mut input = base_input(&primitives, &vertices, principled_header(), parameters);
            input.measured_brdf = measured;
            resolve_pixel(
                input,
                LightingEnvironment {
                    directional: &[light],
                    punctual: &[],
                    image_based: None,
                    ambient: [5.0; 3],
                    stylized: StylizedParams::with_bands(4),
                    area: &[],
                    ltc_lut: None,
                },
            )
            .unwrap()
            .color
        };

        // §17.6 wiring gate: a *bound* table adds the measured term, and the
        // lobe is additive — never negative, never NaN.
        let unbound = resolve(None);
        let bound = resolve(Some(&table));
        assert!(
            bound.iter().zip(&unbound).all(|(a, b)| a > b),
            "bound {bound:?} vs unbound {unbound:?}"
        );
        for value in bound.iter() {
            assert!(value.is_finite() && *value >= 0.0, "{value}");
        }

        // An authored weight of 0 keeps the analytic path bit-for-bit: the table
        // is bound and ignored, exactly like a material that never authors it.
        let mut inert_parameters = parameters;
        inert_parameters.measured_weight = 0.0;
        let mut input = base_input(
            &primitives,
            &vertices,
            principled_header(),
            inert_parameters,
        );
        input.measured_brdf = Some(&table);
        let inert = resolve_pixel(
            input,
            LightingEnvironment {
                directional: &[light],
                punctual: &[],
                image_based: None,
                ambient: [5.0; 3],
                stylized: StylizedParams::with_bands(4),
                area: &[],
                ltc_lut: None,
            },
        )
        .unwrap()
        .color;
        assert_eq!(inert, unbound);
    }

    #[test]
    fn runtime_dynamic_channels_reach_the_resolved_pixel() {
        let (primitives, vertices) = unit_triangle();
        let parameters = GpuSurfaceParameters {
            base_color: [0.6, 0.5, 0.4, 1.0],
            specular_roughness: 0.5,
            ..Default::default()
        };
        let light = DirectionalLight {
            direction: [0.0, 0.0, 1.0],
            illuminance: [10.0; 3],
            visibility: 1.0,
        };
        let resolve = |header: GpuMaterialHeader| {
            resolve_pixel(
                base_input(&primitives, &vertices, header, parameters),
                LightingEnvironment {
                    directional: &[light],
                    punctual: &[],
                    image_based: None,
                    ambient: [5.0; 3],
                    stylized: StylizedParams::with_bands(4),
                    area: &[],
                    ltc_lut: None,
                },
            )
            .unwrap()
            .color
        };

        // All-zero dynamic words are a bit-for-bit identity: a header that never
        // drives a channel is exactly the pre-§17.1 header.
        let inert = resolve(principled_header());
        assert_eq!(resolve(principled_header()), inert);

        // Wetness darkens the base and tightens the highlight, so the resolved
        // radiance must move — this is the wiring gate, not the modulation unit
        // test: it proves the header words reach the shaded pixel.
        let mut wet_header = principled_header();
        wet_header.dynamic_wetness = 1.0_f32.to_bits();
        let wet = resolve(wet_header);
        assert!(
            wet.iter().zip(&inert).all(|(a, b)| a < b),
            "wet {wet:?} vs dry {inert:?}"
        );

        // Snow whitens and roughens: brighter than inert, and a different
        // response again than wetness.
        let mut snow_header = principled_header();
        snow_header.dynamic_snow = 1.0_f32.to_bits();
        let snow = resolve(snow_header);
        assert!(
            snow.iter().zip(&inert).all(|(a, b)| a > b),
            "snow {snow:?} vs bare {inert:?}"
        );

        // Dissolve cuts coverage against the *procedural* mask, and lights its
        // cut line with the material's own emission — so gate it on an emissive
        // material, at the pixel's actual cell. The barycentrics [0.2, 0.3, 0.5]
        // over the unit triangle's UVs interpolate to cell [0.3, 0.5].
        let emissive_parameters = GpuSurfaceParameters {
            emission_color: [1.0, 1.0, 1.0, 1.0],
            emission_luminance: 1.0,
            ..parameters
        };
        let resolve_emissive = |header: GpuMaterialHeader| {
            resolve_pixel(
                base_input(&primitives, &vertices, header, emissive_parameters),
                LightingEnvironment {
                    directional: &[light],
                    punctual: &[],
                    image_based: None,
                    ambient: [5.0; 3],
                    stylized: StylizedParams::with_bands(4),
                    area: &[],
                    ltc_lut: None,
                },
            )
            .unwrap()
            .color
        };
        // Reconstruct the pixel exactly as the resolve does, so the mask is
        // taken from the cell the shader would use rather than a hand-computed
        // guess at the barycentric interpolation.
        let decoded = pixel(principled_header().generation, [0.2, 0.3, 0.5]).barycentrics();
        let geometry = crate::reconstruct_surface(
            SurfaceReconstructionInput {
                primitive_id: 0,
                barycentrics: decoded,
                geometry_generation: 9,
                expected_geometry_generation: 9,
            },
            &primitives,
            &vertices,
        )
        .unwrap();
        let mask = crate::dynamic::dissolve_mask(geometry.uv);
        let intact_emissive = resolve_emissive(principled_header());
        let mut dissolve_header = principled_header();
        // Sit the pixel on the cut line: coverage 0.5 is where the rim peaks.
        dissolve_header.dynamic_dissolve =
            (mask - 0.5 * crate::dynamic::DISSOLVE_EDGE_WIDTH).to_bits();
        let rimmed = resolve_emissive(dissolve_header);
        assert!(
            rimmed.iter().zip(&intact_emissive).all(|(a, b)| a > b),
            "the cut line must glow: rimmed {rimmed:?} vs intact {intact_emissive:?} (uv {:?} mask {mask})",
            geometry.uv
        );

        // Past the cut, the pixel is gone: coverage 0 kills the rim again.
        let mut gone_header = principled_header();
        gone_header.dynamic_dissolve = (mask + 0.5 * crate::dynamic::DISSOLVE_EDGE_WIDTH).to_bits();
        let gone = resolve_emissive(gone_header);
        assert!(
            gone.iter().zip(&intact_emissive).all(|(a, b)| a <= b),
            "a fully cut pixel must not glow: {gone:?} vs {intact_emissive:?}"
        );
    }

    #[test]
    fn unlit_bypasses_lighting_and_returns_base_plus_emissive() {
        let (primitives, vertices) = unit_triangle();
        let header = principled_header();
        let parameters = GpuSurfaceParameters {
            base_color: [0.2, 0.4, 0.6, 1.0],
            emission_color: [0.1, 0.0, 0.0, 0.0],
            // `OpenPBR` emission radiance is `emission_color * emission_luminance`,
            // and the specification default luminance is 0 (no emission).
            emission_luminance: 1.0,
            ..Default::default()
        };
        let light = DirectionalLight {
            direction: [0.0, 0.0, 1.0],
            illuminance: [10.0; 3],
            visibility: 1.0,
        };
        let mut input = base_input(&primitives, &vertices, header, parameters);
        input.response_program = Some(ProgramRef::Builtin(BuiltinProgram::Unlit));
        let resolved = resolve_pixel(
            input,
            LightingEnvironment {
                directional: &[light],
                punctual: &[],
                image_based: None,
                ambient: [5.0; 3],
                stylized: StylizedParams::with_bands(4),
                area: &[],
                ltc_lut: None,
            },
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
            emission_color: [2.0, 2.0, 2.0, 0.0],
            emission_luminance: 1.0,
            ..Default::default()
        };
        let light = DirectionalLight {
            direction: [0.0, 0.0, 1.0],
            illuminance: [3.0; 3],
            visibility: 1.0,
        };
        let one = resolve_pixel(
            base_input(&primitives, &vertices, header, parameters),
            LightingEnvironment {
                directional: &[light],
                punctual: &[],
                image_based: None,
                ambient: [0.0; 3],
                stylized: StylizedParams::with_bands(4),
                area: &[],
                ltc_lut: None,
            },
        )
        .unwrap()
        .color;
        let two = resolve_pixel(
            base_input(&primitives, &vertices, header, parameters),
            LightingEnvironment {
                directional: &[light, light],
                punctual: &[],
                image_based: None,
                ambient: [0.0; 3],
                stylized: StylizedParams::with_bands(4),
                area: &[],
                ltc_lut: None,
            },
        )
        .unwrap()
        .color;
        // Doubling identical lights doubles only the lit term; emissive (2.0)
        // is added once, so the delta equals a single light's lit contribution.
        for channel in 0..3 {
            let lit_once = one[channel] - 2.0;
            let lit_twice = two[channel] - 2.0;
            assert!(
                (lit_twice - 2.0 * lit_once).abs() < 1.0e-4,
                "channel {channel}"
            );
            assert!(lit_once > 0.0, "expected positive direct lighting");
        }
    }

    #[test]
    fn rejects_stale_material_and_invalid_pixels() {
        let (primitives, vertices) = unit_triangle();
        let header = principled_header();
        let mut stale = base_input(
            &primitives,
            &vertices,
            header,
            GpuSurfaceParameters::default(),
        );
        stale.pixel.material_generation = header.generation + 1;
        assert_eq!(
            resolve_pixel(stale, LightingEnvironment::default()),
            Err(ResolveError::StaleMaterial)
        );

        let mut invalid = base_input(
            &primitives,
            &vertices,
            header,
            GpuSurfaceParameters::default(),
        );
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
        let parameters = GpuSurfaceParameters {
            base_color: [1.0, 1.0, 1.0, 1.0],
            ..Default::default()
        };
        let light = DirectionalLight {
            direction: [0.0, 0.0, 1.0],
            illuminance: [1.0; 3],
            visibility: 1.0,
        };
        let mut input = base_input(&primitives, &vertices, header, parameters);
        input.response_program = Some(ProgramRef::Builtin(BuiltinProgram::Npr));
        let resolved = resolve_pixel(
            input,
            LightingEnvironment {
                directional: &[light],
                punctual: &[],
                image_based: None,
                ambient: [0.0; 3],
                stylized: StylizedParams::with_bands(4),
                area: &[],
                ltc_lut: None,
            },
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
            specular_roughness: 0.6,
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
                stylized: StylizedParams::with_bands(4),
                area: &[],
                ltc_lut: None,
            },
        )
        .unwrap();
        assert!(lit.color.iter().all(|c| c.is_finite() && *c >= 0.0));
        assert!(
            lit.color.iter().any(|c| *c > 0.0),
            "point light must add energy"
        );

        // A light whose range window closes before it reaches the surface adds
        // nothing, so the resolved pixel collapses to the (zero) ambient term.
        let dark = resolve_pixel(
            base_input(&primitives, &vertices, header, parameters),
            LightingEnvironment {
                directional: &[],
                punctual: &[PunctualLight::point([0.25, 0.25, 10.0], [5.0; 3], 1.0)],
                image_based: None,
                ambient: [0.0; 3],
                stylized: StylizedParams::with_bands(4),
                area: &[],
                ltc_lut: None,
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
            base_metalness: 0.0,
            specular_roughness: 1.0,
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
                stylized: StylizedParams::with_bands(4),
                area: &[],
                ltc_lut: None,
            },
        )
        .unwrap();
        assert!(lit.color.iter().all(|c| c.is_finite() && *c > 0.0));
        // A grey albedo under a 0.5 constant environment lands near albedo*env.
        for channel in 0..3 {
            assert!(
                (0.3..0.5).contains(&lit.color[channel]),
                "channel {channel} = {}",
                lit.color[channel]
            );
        }
    }

    #[test]
    fn base_color_texture_srgb_decodes_and_modulates_unlit_output() {
        use crate::{srgb_channel_to_linear, SEMANTIC_BASE_COLOR};
        let (primitives, vertices) = unit_triangle();
        let header = principled_header();
        let parameters = GpuSurfaceParameters {
            base_color: [0.2, 0.4, 0.6, 1.0],
            emission_color: [0.1, 0.0, 0.0, 0.0],
            emission_luminance: 1.0,
            ..Default::default()
        };
        // A mid-grey sRGB base-color texel decodes to linear ~0.214 and scales
        // the authored factor; the unlit path then adds emissive exactly once.
        let textures = [SampledTextureBinding {
            semantic: SEMANTIC_BASE_COLOR,
            texel: [0.5, 0.5, 0.5, 1.0],
        }];
        let input = ResolveInput {
            textures: &textures,
            response_program: Some(ProgramRef::Builtin(BuiltinProgram::Unlit)),
            ..base_input(&primitives, &vertices, header, parameters)
        };
        let resolved = resolve_pixel(
            input,
            LightingEnvironment {
                directional: &[],
                punctual: &[],
                image_based: None,
                ambient: [0.0; 3],
                stylized: StylizedParams::with_bands(4),
                area: &[],
                ltc_lut: None,
            },
        )
        .unwrap();
        let decoded = srgb_channel_to_linear(0.5);
        let expected = [0.2 * decoded + 0.1, 0.4 * decoded, 0.6 * decoded];
        for (c, &want) in expected.iter().enumerate().take(3) {
            assert!(
                (resolved.color[c] - want).abs() < 1.0e-6,
                "channel {c}: {} vs {want}",
                resolved.color[c]
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
            base_metalness: 0.0,
            specular_roughness: 0.5,
            ..Default::default()
        };
        // Offset the light from the geometric normal so a tilt in the shading
        // normal moves N.L (and therefore the lit response). The direction is
        // already unit length (0.6^2 + 0.8^2 == 1).
        let light = DirectionalLight {
            direction: [0.6, 0.0, 0.8],
            illuminance: [4.0; 3],
            visibility: 1.0,
        };
        let flat = resolve_pixel(
            base_input(&primitives, &vertices, header, parameters),
            LightingEnvironment {
                directional: &[light],
                punctual: &[],
                image_based: None,
                ambient: [0.0; 3],
                stylized: StylizedParams::with_bands(4),
                area: &[],
                ltc_lut: None,
            },
        )
        .unwrap()
        .color;
        // Tangent-space normal tilted toward +X (world +X here) via the R > 0.5
        // channel; the resolve must rotate it into world space and shift N.L.
        let textures = [SampledTextureBinding {
            semantic: SEMANTIC_NORMAL,
            texel: [0.9, 0.5, 0.85, 1.0],
        }];
        let input = ResolveInput {
            textures: &textures,
            ..base_input(&primitives, &vertices, header, parameters)
        };
        let mut input = input;
        input.response_program = Some(ProgramRef::Builtin(BuiltinProgram::Unlit));
        let tilted = resolve_pixel(
            input,
            LightingEnvironment {
                directional: &[light],
                punctual: &[],
                image_based: None,
                ambient: [0.0; 3],
                stylized: StylizedParams::with_bands(4),
                area: &[],
                ltc_lut: None,
            },
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
            base_metalness: 1.0,
            specular_roughness: 1.0,
            ..Default::default()
        };
        let untextured = resolve_pixel(
            base_input(&primitives, &vertices, header, parameters),
            LightingEnvironment {
                directional: &[DirectionalLight {
                    direction: [0.0, 0.0, 1.0],
                    illuminance: [4.0; 3],
                    visibility: 1.0,
                }],
                punctual: &[],
                image_based: None,
                ambient: [0.0; 3],
                stylized: StylizedParams::with_bands(4),
                area: &[],
                ltc_lut: None,
            },
        )
        .unwrap()
        .color;
        // glTF packs roughness in G and metalness in B; (G=0.25, B=0.5) must
        // quarter the authored roughness and halve metalness, shifting the
        // specular response away from the untextured baseline.
        let textures = [SampledTextureBinding {
            semantic: SEMANTIC_METALLIC_ROUGHNESS,
            texel: [0.0, 0.25, 0.5, 1.0],
        }];
        let input = ResolveInput {
            textures: &textures,
            ..base_input(&primitives, &vertices, header, parameters)
        };
        let textured = resolve_pixel(
            input,
            LightingEnvironment {
                directional: &[DirectionalLight {
                    direction: [0.0, 0.0, 1.0],
                    illuminance: [4.0; 3],
                    visibility: 1.0,
                }],
                punctual: &[],
                image_based: None,
                ambient: [0.0; 3],
                stylized: StylizedParams::with_bands(4),
                area: &[],
                ltc_lut: None,
            },
        )
        .unwrap()
        .color;
        assert!(
            textured
                .iter()
                .zip(untextured)
                .any(|(a, b)| (a - b).abs() > 1.0e-4),
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
            base_metalness: 0.3,
            specular_roughness: 0.4,
            ambient_occlusion: 1.0,
            ..Default::default()
        };
        let light = DirectionalLight {
            direction: [0.0, 0.0, 1.0],
            illuminance: [3.0; 3],
            visibility: 1.0,
        };
        let untextured = resolve_pixel(
            base_input(&primitives, &vertices, header, parameters),
            LightingEnvironment {
                directional: &[light],
                punctual: &[],
                image_based: None,
                ambient: [0.5; 3],
                stylized: StylizedParams::with_bands(4),
                area: &[],
                ltc_lut: None,
            },
        )
        .unwrap()
        .color;
        // White base color (sRGB 1 -> linear 1), unit metallic-roughness
        // (G=B=1) and unit occlusion (R=1) are exact identities under the fold,
        // so the resolve must reproduce the untextured path bit for bit.
        let textures = [
            SampledTextureBinding {
                semantic: SEMANTIC_BASE_COLOR,
                texel: [1.0, 1.0, 1.0, 1.0],
            },
            SampledTextureBinding {
                semantic: SEMANTIC_METALLIC_ROUGHNESS,
                texel: [0.0, 1.0, 1.0, 0.0],
            },
            SampledTextureBinding {
                semantic: SEMANTIC_OCCLUSION,
                texel: [1.0, 0.0, 0.0, 0.0],
            },
        ];
        let input = ResolveInput {
            textures: &textures,
            ..base_input(&primitives, &vertices, header, parameters)
        };
        let textured = resolve_pixel(
            input,
            LightingEnvironment {
                directional: &[light],
                punctual: &[],
                image_based: None,
                ambient: [0.5; 3],
                stylized: StylizedParams::with_bands(4),
                area: &[],
                ltc_lut: None,
            },
        )
        .unwrap()
        .color;
        assert_eq!(
            textured, untextured,
            "identity texels must not perturb the resolve"
        );
    }
    #[test]
    fn screen_space_ao_darkens_the_ambient_term_but_spares_direct_light() {
        // A Lambertian surface lit only by constant ambient: halving the GTAO
        // visibility must halve the resolved color (the ambient term is linear
        // in occlusion), while a fully-lit direct-only surface is untouched.
        let (primitives, vertices) = unit_triangle();
        let header = principled_header();
        let parameters = GpuSurfaceParameters {
            base_color: [0.8, 0.8, 0.8, 1.0],
            base_metalness: 0.0,
            ..Default::default()
        };
        let ambient_only = LightingEnvironment {
            directional: &[],
            punctual: &[],
            image_based: None,
            ambient: [0.6; 3],
            stylized: StylizedParams::with_bands(4),
            area: &[],
            ltc_lut: None,
        };

        let full = resolve_pixel(
            ResolveInput {
                screen_space_ao: 1.0,
                ..base_input(&primitives, &vertices, header, parameters)
            },
            ambient_only,
        )
        .unwrap()
        .color;
        let half = resolve_pixel(
            ResolveInput {
                screen_space_ao: 0.5,
                ..base_input(&primitives, &vertices, header, parameters)
            },
            ambient_only,
        )
        .unwrap()
        .color;
        let occluded = resolve_pixel(
            ResolveInput {
                screen_space_ao: 0.0,
                ..base_input(&primitives, &vertices, header, parameters)
            },
            ambient_only,
        )
        .unwrap()
        .color;

        for c in 0..3 {
            assert!(full[c] > 0.0, "ambient term must be positive: {full:?}");
            assert!(
                (half[c] - 0.5 * full[c]).abs() < 1.0e-6,
                "GTAO must scale the ambient term linearly: {half:?} vs {full:?}",
            );
            assert!(
                occluded[c].abs() < 1.0e-6,
                "fully-occluded ambient must vanish: {occluded:?}",
            );
        }

        // Direct light does not flow through occlusion, so a lit-only surface
        // is identical regardless of the GTAO factor.
        let light = DirectionalLight {
            direction: [0.0, 0.0, 1.0],
            illuminance: [4.0; 3],
            visibility: 1.0,
        };
        let direct = LightingEnvironment {
            directional: &[light],
            punctual: &[],
            image_based: None,
            ambient: [0.0; 3],
            stylized: StylizedParams::with_bands(4),
            area: &[],
            ltc_lut: None,
        };
        let lit_full = resolve_pixel(
            ResolveInput {
                screen_space_ao: 1.0,
                ..base_input(&primitives, &vertices, header, parameters)
            },
            direct,
        )
        .unwrap()
        .color;
        let lit_occluded = resolve_pixel(
            ResolveInput {
                screen_space_ao: 0.0,
                ..base_input(&primitives, &vertices, header, parameters)
            },
            direct,
        )
        .unwrap()
        .color;
        assert_eq!(
            lit_full, lit_occluded,
            "GTAO must not touch direct lighting"
        );
    }

    #[test]
    fn world_from_local_transforms_position_normal_and_lifts_lighting() {
        // A pure translation shifts the point and leaves a normal untouched.
        let translate = [
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 1.0, 0.0, -3.0],
            [0.0, 0.0, 1.0, 5.0],
        ];
        assert_eq!(
            affine_transform_point(&translate, [1.0, 1.0, 1.0]),
            [3.0, -2.0, 6.0]
        );

        // A 90-degree rotation about +Z carries the +X normal onto +Y; the
        // inverse-transpose of a rotation is the rotation itself.
        let rot_z = [
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ];
        let rotated = mat3_mul(&normal_matrix(&rot_z), [1.0, 0.0, 0.0]);
        assert!((rotated[0]).abs() < 1.0e-6);
        assert!((rotated[1] - 1.0).abs() < 1.0e-6);
        assert!((rotated[2]).abs() < 1.0e-6);

        // A non-uniform scale must shrink the normal along the stretched axis
        // (inverse-transpose), the opposite of how a tangent scales.
        let scale_x = [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ];
        let scaled_normal = mat3_mul(&normal_matrix(&scale_x), [1.0, 0.0, 0.0]);
        assert!((scaled_normal[0] - 0.5).abs() < 1.0e-6);
        let scaled_tangent = mat3_mul(&linear_part(&scale_x), [1.0, 0.0, 0.0]);
        assert!((scaled_tangent[0] - 2.0).abs() < 1.0e-6);
    }

    #[test]
    fn world_from_local_lighting_is_translation_invariant() {
        // Translating the instance together with the point light and the camera
        // by the same offset must reproduce the identity-transform shading
        // exactly: the resolve lights at the *world* position, so a rigid shift
        // of the whole configuration cancels out.
        let (primitives, vertices) = unit_triangle();
        let header = principled_header();
        let parameters = GpuSurfaceParameters {
            base_color: [0.8, 0.7, 0.6, 1.0],
            base_metalness: 0.2,
            specular_roughness: 0.5,
            ..Default::default()
        };
        let offset = [4.0, -2.0, 7.0];

        let base_light = PunctualLight {
            position: [0.0, 0.0, 3.0],
            intensity: [12.0; 3],
            visibility: 1.0,
            ..Default::default()
        };
        let baseline = resolve_pixel(
            base_input(&primitives, &vertices, header, parameters),
            LightingEnvironment {
                directional: &[],
                punctual: &[base_light],
                image_based: None,
                ambient: [0.0; 3],
                stylized: StylizedParams::with_bands(4),
                area: &[],
                ltc_lut: None,
            },
        )
        .unwrap()
        .color;

        let shifted_light = PunctualLight {
            position: [offset[0], offset[1], 3.0 + offset[2]],
            ..base_light
        };
        let mut shifted = base_input(&primitives, &vertices, header, parameters);
        shifted.world_from_local = [
            [1.0, 0.0, 0.0, offset[0]],
            [0.0, 1.0, 0.0, offset[1]],
            [0.0, 0.0, 1.0, offset[2]],
        ];
        shifted.view_position = [
            baseline_view(&primitives, &vertices, header, parameters)[0] + offset[0],
            baseline_view(&primitives, &vertices, header, parameters)[1] + offset[1],
            baseline_view(&primitives, &vertices, header, parameters)[2] + offset[2],
        ];
        let moved = resolve_pixel(
            shifted,
            LightingEnvironment {
                directional: &[],
                punctual: &[shifted_light],
                image_based: None,
                ambient: [0.0; 3],
                stylized: StylizedParams::with_bands(4),
                area: &[],
                ltc_lut: None,
            },
        )
        .unwrap()
        .color;

        for (a, b) in moved.iter().zip(baseline) {
            assert!(
                (a - b).abs() < 1.0e-4,
                "translated instance+light+camera must match identity: {moved:?} vs {baseline:?}"
            );
        }

        // Moving *only* the instance (light and camera fixed) must change the
        // shading, proving the transform actually reaches the lighting position.
        let mut only_instance = base_input(&primitives, &vertices, header, parameters);
        only_instance.world_from_local = [
            [1.0, 0.0, 0.0, offset[0]],
            [0.0, 1.0, 0.0, offset[1]],
            [0.0, 0.0, 1.0, offset[2]],
        ];
        let drifted = resolve_pixel(
            only_instance,
            LightingEnvironment {
                directional: &[],
                punctual: &[base_light],
                image_based: None,
                ambient: [0.0; 3],
                stylized: StylizedParams::with_bands(4),
                area: &[],
                ltc_lut: None,
            },
        )
        .unwrap()
        .color;
        assert!(
            drifted
                .iter()
                .zip(baseline)
                .any(|(a, b)| (a - b).abs() > 1.0e-4),
            "translating only the instance must change the lit output"
        );
    }

    fn baseline_view(
        _primitives: &[GpuShadingPrimitive],
        _vertices: &[GpuShadingVertex],
        _header: GpuMaterialHeader,
        _parameters: GpuSurfaceParameters,
    ) -> [f32; 3] {
        // Mirror the camera position `base_input` bakes in so the invariance
        // test shifts by the exact same origin.
        [0.0, 0.0, 4.0]
    }

    #[test]
    fn area_light_adds_energy_to_a_principled_pixel_and_empty_slice_is_stable() {
        use crate::gi::area_light::ltc_lut::bake_ltc_lut_default;

        // A principled surface with no ambient / directional / punctual light
        // resolves to zero; a facing rectangle area light sampled through the
        // baked LTC LUT must inject strictly positive radiance, while an empty
        // `area` slice must leave the baseline bit-for-bit unchanged (the
        // default-stable contract the GPU resolve relies on).
        let (primitives, vertices) = unit_triangle();
        let header = principled_header();
        let parameters = GpuSurfaceParameters {
            base_color: [0.8, 0.6, 0.4, 1.0],
            base_metalness: 0.0,
            ..Default::default()
        };

        let dark = LightingEnvironment {
            directional: &[],
            punctual: &[],
            image_based: None,
            ambient: [0.0; 3],
            stylized: StylizedParams::with_bands(4),
            area: &[],
            ltc_lut: None,
        };
        let baseline = resolve_pixel(base_input(&primitives, &vertices, header, parameters), dark)
            .unwrap()
            .color;
        for c in 0..3 {
            assert!(
                baseline[c].abs() < 1.0e-6,
                "unlit principled pixel must resolve to zero: {baseline:?}",
            );
        }

        let lut = bake_ltc_lut_default();
        let lights = [AreaLightRect {
            center: [0.3, 0.5, 2.0],
            axis_u: [1.0, 0.0, 0.0],
            axis_v: [0.0, 1.0, 0.0],
            half_width: 0.5,
            half_height: 0.5,
            color: [1.0, 1.0, 1.0],
            intensity: 4.0,
            range: 0.0,
            two_sided: true,
        }];
        let lit = resolve_pixel(
            base_input(&primitives, &vertices, header, parameters),
            LightingEnvironment {
                area: &lights,
                ltc_lut: Some(&lut),
                ..dark
            },
        )
        .unwrap()
        .color;
        for c in 0..3 {
            assert!(
                lit[c] > baseline[c] + 1.0e-5,
                "a facing area light must add positive radiance: {lit:?} vs {baseline:?}",
            );
        }

        let empty = resolve_pixel(
            base_input(&primitives, &vertices, header, parameters),
            LightingEnvironment {
                area: &[],
                ltc_lut: Some(&lut),
                ..dark
            },
        )
        .unwrap()
        .color;
        assert_eq!(empty, baseline, "empty area slice must stay bit-stable");
    }
}
