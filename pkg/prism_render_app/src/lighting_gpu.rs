//! GPU execution for the deterministic direct-light and initial GI recipes.

use bytemuck::{bytes_of, cast_slice, Pod, Zeroable};
use prism_ecs::prelude::Resource;
use prism_render_architecture::lighting::contract::{GiCandidateSource, LightingPlan, RecipeId};
use prism_render_architecture::lighting::{
    CompletionProof, GpuSubmissionDomain, GpuTableHandle, GpuTablePayload, IncrementalGpuTable,
    LightSceneHandle, SubmissionProof, UploadPlan,
};
use prism_render_driver::{
    BindGroupDescriptor, BindGroupEntry, BindGroupId, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindGroupLayoutId, BindingResource, BindingType, BufferBindingType,
    BufferDescriptor, BufferUsages, ComputePipelineDescriptor, FrameSync, PipelineLayoutDescriptor,
    RenderQueue, ShaderModuleDescriptor, ShaderStages, StorageTextureAccess, TextureFormat,
    TextureSampleType, TextureViewDimension, TextureViewId,
};
use prism_render_runtime::{RenderRuntime, ReplayHandle};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use crate::lighting::{
    ExtractedLightingInputs, LightingAreaInput, LightingDirectionalInput, LightingGiCandidateInput,
    LightingPunctualInput,
};
use crate::lighting_trace_gpu::TracePreparationError;

impl GpuTablePayload for LightingDirectionalInput {
    fn neutral() -> Self {
        Self::default()
    }
}

impl GpuTablePayload for LightingPunctualInput {
    fn neutral() -> Self {
        Self::default()
    }
}

impl GpuTablePayload for LightingAreaInput {
    fn neutral() -> Self {
        Self::default()
    }
}

const MAX_DIRECTIONALS: usize = 1024;
const MAX_PUNCTUALS: usize = 1024;
const MAX_AREAS: usize = 1024;
/// The render runtime currently exposes one ordered submission stream.
/// Device recovery is separated by `device_generation`; this logical id keeps
/// the stable-table contract ready for additional queues without accepting a
/// proof from another stream by accident.
pub(crate) const LIGHTING_QUEUE_DOMAIN: u64 = 0;
pub(crate) const GI_CANDIDATES_PER_SURFACE: usize = 64;
pub(crate) const MAX_LIGHTING_SURFACES: usize = 1024;
const WORKGROUP_SIZE: u32 = 64;
/// X dimension of the per-target screen-space composite workgroup.
pub(crate) const COMPOSITE_WORKGROUP_X: u32 = 8;
/// Y dimension of the per-target screen-space composite workgroup.
pub(crate) const COMPOSITE_WORKGROUP_Y: u32 = 8;
/// The required format for the world-position GBuffer view.
pub(crate) const COMPOSITE_POSITION_FORMAT: TextureFormat = TextureFormat::Rgba32Float;
/// The required format for the world-normal GBuffer view.
pub(crate) const COMPOSITE_NORMAL_FORMAT: TextureFormat = TextureFormat::Rgba16Float;
/// The required format for the material/albedo GBuffer view.
pub(crate) const COMPOSITE_ALBEDO_FORMAT: TextureFormat = TextureFormat::Rgba16Float;
/// The required format for the surface-id GBuffer view.
pub(crate) const COMPOSITE_SURFACE_ID_FORMAT: TextureFormat = TextureFormat::R32Uint;
/// The required format for the binary/normalized surface coverage view.
pub(crate) const COMPOSITE_COVERAGE_FORMAT: TextureFormat = TextureFormat::R8Unorm;
/// The scene-linear HDR format written by the composite.
pub(crate) const COMPOSITE_HDR_FORMAT: TextureFormat = TextureFormat::Rgba16Float;
pub(crate) const OUTPUT_BYTES: u64 =
    (MAX_LIGHTING_SURFACES * core::mem::size_of::<[f32; 4]>()) as u64;
const DIRECTIONAL_BYTES: u64 =
    (MAX_DIRECTIONALS * core::mem::size_of::<LightingDirectionalInput>()) as u64;
const PUNCTUAL_BYTES: u64 = (MAX_PUNCTUALS * core::mem::size_of::<LightingPunctualInput>()) as u64;
const AREA_BYTES: u64 = (MAX_AREAS * core::mem::size_of::<LightingAreaInput>()) as u64;
const GI_CANDIDATE_BYTES: u64 = (MAX_LIGHTING_SURFACES
    * GI_CANDIDATES_PER_SURFACE
    * core::mem::size_of::<LightingGiCandidateInput>()) as u64;
pub(crate) const GI_CANDIDATE_BUFFER_BYTES: u64 = GI_CANDIDATE_BYTES;
pub(crate) const TRACE_RADIANCE_BYTES: u64 = (2_000_000 * core::mem::size_of::<[f32; 4]>()) as u64;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuLightingSurface {
    position: [f32; 4],
    normal: [f32; 4],
    albedo: [f32; 4],
    ambient: [f32; 4],
    candidate_range: [u32; 4],
}

const SURFACE_BYTES: u64 =
    (MAX_LIGHTING_SURFACES * core::mem::size_of::<GpuLightingSurface>()) as u64;
pub(crate) const LIGHTING_SURFACE_BUFFER_BYTES: u64 = SURFACE_BYTES;
pub(crate) const LIGHTING_PARAMS_BYTES: u64 = PARAM_BYTES;

/// Persistent lighting buffers shared by trace candidate-build and GI.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct LightingGiCandidateTarget {
    pub candidates: prism_render_driver::BufferId,
    pub candidates_size_bytes: u64,
    pub surface_count: u32,
    pub rays_per_surface: u32,
    pub surfaces: prism_render_driver::BufferId,
    pub surfaces_size_bytes: u64,
    pub params: prism_render_driver::BufferId,
    pub params_size_bytes: u64,
    pub radiance: prism_render_driver::BufferId,
    pub radiance_size_bytes: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuLightingParams {
    counts: [u32; 4],
    flags: [u32; 4],
}

const PARAM_BYTES: u64 = core::mem::size_of::<GpuLightingParams>() as u64;
const ZERO_OUTPUT: [u8; OUTPUT_BYTES as usize] = [0; OUTPUT_BYTES as usize];

const DIRECT_SHADER: &str = r#"
struct DirectionalLight {
    direction_to_light: vec3<f32>,
    visibility: f32,
    illuminance: vec3<f32>,
    padding: f32,
};

struct PunctualLight {
    position_range: vec4<f32>,
    intensity_spot_scale: vec4<f32>,
    direction_spot_offset: vec4<f32>,
    visibility_padding: vec4<f32>,
    reserved_padding: vec4<f32>,
};

struct AreaLight {
    origin: vec4<f32>,
    vector_a: vec4<f32>,
    vector_b: vec4<f32>,
    vector_c: vec4<f32>,
    radiance: vec4<f32>,
    shape: vec4<f32>,
    shape_kind: vec4<u32>,
};

struct Params {
    counts: vec4<u32>,
    flags: vec4<u32>,
};

struct Surface {
    position: vec4<f32>,
    normal: vec4<f32>,
    albedo: vec4<f32>,
    ambient: vec4<f32>,
    _candidate_range: vec4<u32>,
};

@group(0) @binding(0)
var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(1)
var<uniform> params: Params;
@group(0) @binding(2)
var<storage, read> directionals: array<DirectionalLight, 1024>;
@group(0) @binding(3)
var<storage, read> punctuals: array<PunctualLight, 1024>;
@group(0) @binding(4)
var<storage, read> surfaces: array<Surface, 1024>;
@group(0) @binding(5)
var<storage, read> areas: array<AreaLight, 1024>;

fn normalize_or(value: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let length_squared = dot(value, value);
    if (length_squared > 1.0e-12) {
        return value * inverseSqrt(length_squared);
    }
    return fallback;
}

fn finite(value: f32) -> bool {
    let exponent = (bitcast<u32>(value) >> 23u) & 0xffu;
    return exponent != 0xffu;
}

fn punctual_attenuation(light: PunctualLight, position: vec3<f32>) -> f32 {
    let to_light = light.position_range.xyz - position;
    let distance_squared = dot(to_light, to_light);
    if (!finite(distance_squared) || distance_squared <= 0.0) {
        return 0.0;
    }
    let distance = sqrt(distance_squared);
    let direction = to_light / distance;
    var attenuation = 1.0 / max(distance_squared, 1.0e-4);
    if (finite(light.position_range.w) && light.position_range.w > 0.0) {
        let ratio = min(distance_squared / (light.position_range.w * light.position_range.w), 1.0);
        let window = max(1.0 - ratio * ratio, 0.0);
        attenuation = attenuation * window * window;
    }
    if (light.intensity_spot_scale.w != 0.0) {
        let cone = clamp(
            dot(light.direction_spot_offset.xyz, -direction) * light.intensity_spot_scale.w
                + light.direction_spot_offset.w,
            0.0,
            1.0,
        );
        attenuation = attenuation * cone * cone;
    }
    return attenuation * clamp(light.visibility_padding.x, 0.0, 1.0);
}

fn basis_tangent(direction: vec3<f32>) -> vec3<f32> {
    let n = normalize_or(direction, vec3<f32>(0.0, 1.0, 0.0));
    let reference = select(
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(1.0, 0.0, 0.0),
        abs(n.y) > 0.9,
    );
    return normalize_or(cross(reference, n), vec3<f32>(1.0, 0.0, 0.0));
}

fn basis_bitangent(direction: vec3<f32>) -> vec3<f32> {
    let n = normalize_or(direction, vec3<f32>(0.0, 1.0, 0.0));
    return normalize_or(cross(n, basis_tangent(n)), vec3<f32>(0.0, 0.0, 1.0));
}

fn area_sample(light: AreaLight, sample_index: u32) -> vec3<f32> {
    let kind = light.shape_kind.x;
    if (kind == 0u) {
        let uv = array<vec2<f32>, 4>(
            vec2<f32>(-0.57735026, -0.57735026),
            vec2<f32>(0.57735026, -0.57735026),
            vec2<f32>(-0.57735026, 0.57735026),
            vec2<f32>(0.57735026, 0.57735026),
        )[sample_index];
        return light.origin.xyz + light.vector_b.xyz * (light.shape.x * uv.x)
            + light.vector_c.xyz * (light.shape.y * uv.y);
    }
    if (kind == 1u) {
        let offsets = array<vec2<f32>, 4>(
            vec2<f32>(0.5, 0.0),
            vec2<f32>(-0.5, 0.0),
            vec2<f32>(0.0, 0.5),
            vec2<f32>(0.0, -0.5),
        )[sample_index];
        return light.origin.xyz + basis_tangent(light.vector_a.xyz) * (light.shape.x * offsets.x)
            + basis_bitangent(light.vector_a.xyz) * (light.shape.x * offsets.y);
    }
    let segment = light.vector_a.xyz - light.origin.xyz;
    let axis = normalize_or(segment, vec3<f32>(0.0, 1.0, 0.0));
    let t = array<f32, 4>(0.125, 0.375, 0.625, 0.875)[sample_index];
    let angle = array<f32, 4>(0.0, 1.5707963, 3.1415927, 4.712389)[sample_index];
    let center = mix(light.origin.xyz, light.vector_a.xyz, t);
    let radial = basis_tangent(axis) * cos(angle) + basis_bitangent(axis) * sin(angle);
    return center + radial * light.shape.x;
}

fn area_measure(light: AreaLight) -> f32 {
    if (light.shape_kind.x == 0u) {
        return 4.0 * light.shape.x * light.shape.y;
    }
    if (light.shape_kind.x == 1u) {
        return 3.141592653589793 * light.shape.x * light.shape.x;
    }
    let length = distance(light.origin.xyz, light.vector_a.xyz);
    return 2.0 * 3.141592653589793 * light.shape.x * length
        + 4.0 * 3.141592653589793 * light.shape.x * light.shape.x;
}

fn area_contribution(light: AreaLight, surface: Surface, normal: vec3<f32>) -> vec3<f32> {
    let measure = area_measure(light);
    if (!finite(measure) || measure <= 0.0) {
        return vec3<f32>(0.0);
    }
    var result = vec3<f32>(0.0);
    for (var sample_index = 0u; sample_index < 4u; sample_index = sample_index + 1u) {
        let point = area_sample(light, sample_index);
        let to_light = point - surface.position.xyz;
        let distance_squared = dot(to_light, to_light);
        if (!finite(distance_squared) || distance_squared <= 1.0e-8) {
            continue;
        }
        let direction = to_light * inverseSqrt(distance_squared);
        let cosine_surface = max(dot(normal, direction), 0.0);
        var cosine_source = 1.0;
        if (light.shape_kind.x == 0u || light.shape_kind.x == 1u) {
            cosine_source = max(
                dot(
                    normalize_or(light.vector_a.xyz, vec3<f32>(0.0, -1.0, 0.0)),
                    -direction,
                ),
                0.0,
            );
        } else {
            let axis = normalize_or(light.vector_a.xyz - light.origin.xyz, vec3<f32>(0.0, 1.0, 0.0));
            let center = mix(light.origin.xyz, light.vector_a.xyz, 0.5);
            let radial = normalize_or(point - center, basis_tangent(axis));
            cosine_source = max(dot(radial, -direction), 0.0);
        }
        result = result + light.radiance.xyz
            * (cosine_surface * cosine_source / distance_squared);
    }
    return result * (measure * 0.25) * clamp(f32(light.shape_kind.y), 0.0, 1.0);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.counts.x || id.x >= arrayLength(&output)) {
        return;
    }
    let surface = surfaces[id.x];
    let normal = normalize_or(surface.normal.xyz, vec3<f32>(0.0, 1.0, 0.0));
    let diffuse_brdf = surface.albedo.xyz * (1.0 / 3.141592653589793);
    var radiance = diffuse_brdf * surface.ambient.xyz;
    let directional_count = min(params.counts.y, 1024u);
    let punctual_count = min(params.counts.z, 1024u);
    let area_count = min(params.flags.y, 1024u);
    for (var index = 0u; index < directional_count; index = index + 1u) {
        let light = directionals[index];
        let direction = normalize_or(light.direction_to_light, vec3<f32>(0.0, 1.0, 0.0));
        let cosine = max(dot(normal, direction), 0.0) * clamp(light.visibility, 0.0, 1.0);
        radiance = radiance + diffuse_brdf * light.illuminance * cosine;
    }
    for (var index = 0u; index < punctual_count; index = index + 1u) {
        let light = punctuals[index];
        let to_light = light.position_range.xyz - surface.position.xyz;
        let distance_squared = dot(to_light, to_light);
        if (distance_squared > 0.0) {
            let direction = to_light * inverseSqrt(distance_squared);
            let cosine = max(dot(normal, direction), 0.0);
            let attenuation = punctual_attenuation(light, surface.position.xyz);
            radiance = radiance
                + diffuse_brdf * light.intensity_spot_scale.xyz * (cosine * attenuation);
        }
    }
    for (var index = 0u; index < area_count; index = index + 1u) {
        radiance = radiance + diffuse_brdf * area_contribution(areas[index], surface, normal);
    }
    output[id.x] = vec4<f32>(radiance, 1.0);
}
"#;

const GI_SHADER: &str = r#"
struct GiCandidate {
    position_source_pdf: vec4<f32>,
    normal_target_pdf: vec4<f32>,
    radiance_selection_u01: vec4<f32>,
    visibility_padding: vec4<f32>,
};

struct Params {
    counts: vec4<u32>,
    flags: vec4<u32>,
};

struct Surface {
    position: vec4<f32>,
    normal: vec4<f32>,
    albedo: vec4<f32>,
    ambient: vec4<f32>,
    candidate_range: vec4<u32>,
};

@group(0) @binding(0)
var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(1)
var<uniform> params: Params;
@group(0) @binding(2)
var<storage, read> candidates: array<GiCandidate>;
@group(0) @binding(3)
var<storage, read> surfaces: array<Surface, 1024>;

fn normalize_or(value: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let length_squared = dot(value, value);
    if (length_squared > 1.0e-12) {
        return value * inverseSqrt(length_squared);
    }
    return fallback;
}

fn finite(value: f32) -> bool {
    let exponent = (bitcast<u32>(value) >> 23u) & 0xffu;
    return exponent != 0xffu;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.counts.x || id.x >= arrayLength(&output)) {
        return;
    }
    let surface = surfaces[id.x];
    let primary_normal = normalize_or(surface.normal.xyz, vec3<f32>(0.0, 1.0, 0.0));
    let candidate_count = min(surface.candidate_range.y, 64u);
    var weight_sum = 0.0;
    var selected_position = vec3<f32>(0.0);
    var selected_radiance = vec3<f32>(0.0);
    var selected_target_pdf = 0.0;
    var selected_visibility = 0.0;
    for (var index = 0u; index < candidate_count; index = index + 1u) {
        let candidate = candidates[surface.candidate_range.x + index];
        if (!finite(candidate.position_source_pdf.w) || candidate.position_source_pdf.w <= 0.0 ||
            !finite(candidate.normal_target_pdf.w) || candidate.normal_target_pdf.w <= 0.0) {
            continue;
        }
        let weight = candidate.normal_target_pdf.w / candidate.position_source_pdf.w;
        if (!finite(weight) || weight <= 0.0) {
            continue;
        }
        weight_sum = weight_sum + weight;
        if (candidate.normal_target_pdf.w == 0.0) {
            continue;
        }
        if (candidate.radiance_selection_u01.w * weight_sum < weight) {
            selected_position = candidate.position_source_pdf.xyz;
            selected_radiance = candidate.radiance_selection_u01.xyz;
            selected_target_pdf = candidate.normal_target_pdf.w;
            selected_visibility = candidate.visibility_padding.x;
        }
    }
    if (candidate_count == 0u || weight_sum <= 0.0 || selected_target_pdf <= 0.0) {
        output[id.x] = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        return;
    }
    let to_sample = selected_position - surface.position.xyz;
    let distance_squared = dot(to_sample, to_sample);
    if (!finite(distance_squared) || distance_squared <= 1.0e-8) {
        output[id.x] = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        return;
    }
    let direction = to_sample * inverseSqrt(distance_squared);
    let cosine_primary = max(dot(primary_normal, direction), 0.0);
    let visibility = clamp(selected_visibility, 0.0, 1.0);
    let reservoir_weight = weight_sum / (f32(candidate_count) * selected_target_pdf);
    let result = surface.albedo.xyz * selected_radiance * (cosine_primary / 3.141592653589793)
        * visibility * reservoir_weight;
    output[id.x] = vec4<f32>(result, 1.0);
}
"#;

pub(crate) const LIGHTING_COMPOSITE_WGSL: &str = r#"
struct Params {
    counts: vec4<u32>,
    flags: vec4<u32>,
};

@group(0) @binding(0)
var<storage, read> direct: array<vec4<f32>>;
@group(0) @binding(1)
var<storage, read> gi: array<vec4<f32>>;
@group(0) @binding(2)
var<uniform> params: Params;

@group(1) @binding(0)
var position_gbuffer: texture_2d<f32>;
@group(1) @binding(1)
var normal_gbuffer: texture_2d<f32>;
@group(1) @binding(2)
var albedo_gbuffer: texture_2d<f32>;
@group(1) @binding(3)
var surface_id_gbuffer: texture_2d<u32>;
@group(1) @binding(4)
var coverage_gbuffer: texture_2d<f32>;
@group(1) @binding(5)
var hdr_target: texture_storage_2d<rgba16float, write>;

fn finite(value: f32) -> bool {
    let exponent = (bitcast<u32>(value) >> 23u) & 0xffu;
    return exponent != 0xffu;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let extent = textureDimensions(hdr_target);
    if (id.x >= extent.x || id.y >= extent.y) {
        return;
    }

    let pixel = vec2<i32>(id.xy);
    let coverage = textureLoad(coverage_gbuffer, pixel, 0).x;
    let surface_id = textureLoad(surface_id_gbuffer, pixel, 0).x;
    if (coverage <= 0.0 || surface_id == 0xffffffffu || surface_id >= params.counts.x) {
        textureStore(hdr_target, pixel, vec4<f32>(0.0));
        return;
    }

    // These reads are part of the contract even when the current direct/GI
    // providers only consume the packed surface table. They make the
    // composite a real per-pixel GBuffer consumer and keep the ABI ready for
    // material-aware reconstruction and shadow/contact terms.
    let position = textureLoad(position_gbuffer, pixel, 0).xyz;
    let normal = textureLoad(normal_gbuffer, pixel, 0).xyz;
    let albedo = textureLoad(albedo_gbuffer, pixel, 0).xyz;
    if (!finite(position.x) || !finite(position.y) || !finite(position.z) ||
        !finite(normal.x) || !finite(normal.y) || !finite(normal.z) ||
        !finite(albedo.x) || !finite(albedo.y) || !finite(albedo.z)) {
        textureStore(hdr_target, pixel, vec4<f32>(0.0));
        return;
    }

    var value = vec3<f32>(0.0);
    if (params.flags.x == 1u || params.flags.x == 3u) {
        value = value + direct[surface_id].xyz;
    }
    if (params.flags.x == 2u || params.flags.x == 3u) {
        value = value + gi[surface_id].xyz;
    }

    // Keep the GBuffer contract live in the current path. The factors are
    // neutral for the canonical packed-light outputs and prevent dead-input
    // optimization from hiding a missing resource in backend validation.
    let normal_length = max(length(normal), 1.0e-6);
    let position_finite = finite(position.x) && finite(position.y) && finite(position.z);
    let albedo_finite = finite(albedo.x) && finite(albedo.y) && finite(albedo.z);
    let gbuffer_valid = select(0.0, 1.0, position_finite && albedo_finite && normal_length > 0.0);
    textureStore(hdr_target, pixel, vec4<f32>(value * gbuffer_valid, 1.0));
}
"#;

/// Per-target resources consumed by the lighting composite.
///
/// The producer must create all views from one 2D, single-sample target with
/// identical width and height. Formats are intentionally fixed by this
/// contract so the pipeline layout and WGSL stay device-independent:
/// position `Rgba32Float`, normal/albedo `Rgba16Float`, surface id `R32Uint`,
/// coverage `R8Unorm`, and HDR output `Rgba16Float`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct LightingCompositeTarget {
    pub position_view: TextureViewId,
    pub normal_view: TextureViewId,
    pub albedo_view: TextureViewId,
    pub surface_id_view: TextureViewId,
    pub coverage_view: TextureViewId,
    pub hdr_target_view: TextureViewId,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LightingCompositeTargetError {
    ZeroExtent,
    DuplicateView { view: TextureViewId },
}

impl core::fmt::Display for LightingCompositeTargetError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ZeroExtent => {
                formatter.write_str("lighting composite target extent must be nonzero")
            }
            Self::DuplicateView { view } => {
                write!(formatter, "lighting composite target reuses view {view:?}")
            }
        }
    }
}

impl std::error::Error for LightingCompositeTargetError {}

fn texture_layout(binding: u32, sample_type: TextureSampleType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Texture {
            sample_type,
            view_dimension: TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn storage_texture_layout(binding: u32, format: TextureFormat) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::StorageTexture {
            access: StorageTextureAccess::WriteOnly,
            format,
            view_dimension: TextureViewDimension::D2,
        },
        count: None,
    }
}

fn validate_composite_target(
    target: LightingCompositeTarget,
) -> Result<(), LightingCompositeTargetError> {
    if target.width == 0 || target.height == 0 {
        return Err(LightingCompositeTargetError::ZeroExtent);
    }
    let views = [
        target.position_view,
        target.normal_view,
        target.albedo_view,
        target.surface_id_view,
        target.coverage_view,
        target.hdr_target_view,
    ];
    for (index, view) in views.iter().copied().enumerate() {
        if views[..index].contains(&view) {
            return Err(LightingCompositeTargetError::DuplicateView { view });
        }
    }
    Ok(())
}

fn composite_static_layout_descriptor() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor {
        label: Some("lighting-composite-static-layout".into()),
        entries: vec![
            storage_layout(0, OUTPUT_BYTES, true),
            storage_layout(1, OUTPUT_BYTES, true),
            uniform_layout(2),
        ],
    }
}

fn composite_target_layout_descriptor() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor {
        label: Some("lighting-composite-target-layout".into()),
        entries: vec![
            texture_layout(0, COMPOSITE_POSITION_FORMAT.sample_type()),
            texture_layout(1, COMPOSITE_NORMAL_FORMAT.sample_type()),
            texture_layout(2, COMPOSITE_ALBEDO_FORMAT.sample_type()),
            texture_layout(3, COMPOSITE_SURFACE_ID_FORMAT.sample_type()),
            texture_layout(4, COMPOSITE_COVERAGE_FORMAT.sample_type()),
            storage_texture_layout(5, COMPOSITE_HDR_FORMAT),
        ],
    }
}

#[cfg(test)]
mod composite_contract_tests {
    use super::*;

    fn target() -> LightingCompositeTarget {
        LightingCompositeTarget {
            position_view: TextureViewId::from_parts(1, 1),
            normal_view: TextureViewId::from_parts(2, 1),
            albedo_view: TextureViewId::from_parts(3, 1),
            surface_id_view: TextureViewId::from_parts(4, 1),
            coverage_view: TextureViewId::from_parts(5, 1),
            hdr_target_view: TextureViewId::from_parts(6, 1),
            width: 1920,
            height: 1080,
        }
    }

    #[test]
    fn composite_target_requires_a_real_extent() {
        let mut zero = target();
        zero.width = 0;
        assert_eq!(
            validate_composite_target(zero),
            Err(LightingCompositeTargetError::ZeroExtent)
        );
    }

    #[test]
    fn composite_target_rejects_view_aliasing() {
        let mut aliased = target();
        aliased.hdr_target_view = aliased.position_view;
        assert_eq!(
            validate_composite_target(aliased),
            Err(LightingCompositeTargetError::DuplicateView {
                view: aliased.position_view
            })
        );
    }

    #[test]
    fn composite_dispatch_covers_every_target_pixel() {
        assert_eq!(
            [
                target().width.div_ceil(COMPOSITE_WORKGROUP_X),
                target().height.div_ceil(COMPOSITE_WORKGROUP_Y),
                1,
            ],
            [240, 135, 1]
        );
    }

    #[test]
    fn composite_layout_has_two_explicit_groups_without_placeholder_slots() {
        let static_layout = composite_static_layout_descriptor();
        assert_eq!(
            static_layout
                .entries
                .iter()
                .map(|entry| entry.binding)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert!(static_layout
            .entries
            .iter()
            .all(|entry| entry.validate().is_ok()));

        let target_layout = composite_target_layout_descriptor();
        assert_eq!(
            target_layout
                .entries
                .iter()
                .map(|entry| entry.binding)
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4, 5]
        );
        assert!(target_layout
            .entries
            .iter()
            .all(|entry| entry.validate().is_ok()));
        assert_eq!(
            target_layout.entries[5].ty,
            BindingType::StorageTexture {
                access: StorageTextureAccess::WriteOnly,
                format: COMPOSITE_HDR_FORMAT,
                view_dimension: TextureViewDimension::D2,
            }
        );
    }
}

/// The GPU objects used by one lighting execution bundle.
pub struct LightingGpuBundle {
    /// Final HDR radiance output for the surface samples.
    pub output: prism_render_driver::BufferId,
    /// Direct-light intermediate output.
    pub direct_output: prism_render_driver::BufferId,
    /// Initial GI reservoir output.
    pub gi_output: prism_render_driver::BufferId,
    /// Uniform surface and light counts.
    pub params: prism_render_driver::BufferId,
    /// Directional light storage buffer.
    pub directionals: prism_render_driver::BufferId,
    /// Point and spot light storage buffer.
    pub punctuals: prism_render_driver::BufferId,
    /// Rect, disk, and capsule storage buffer.
    pub areas: prism_render_driver::BufferId,
    /// GI candidate storage buffer.
    pub candidates: prism_render_driver::BufferId,
    /// Packed, shader-aligned surface sample storage buffer.
    pub surfaces: prism_render_driver::BufferId,
    /// Per-packed-triangle outgoing radiance used by trace candidate build.
    pub radiance: prism_render_driver::BufferId,
    /// Direct-light pipeline and bind group.
    pub direct_pipeline: prism_render_driver::ComputePipelineId,
    pub direct_bind_group: prism_render_driver::BindGroupId,
    /// GI pipeline and bind group.
    pub gi_pipeline: prism_render_driver::ComputePipelineId,
    pub gi_bind_group: prism_render_driver::BindGroupId,
    /// Composite pipeline and bind group.
    pub composite_pipeline: prism_render_driver::ComputePipelineId,
    /// Static composite group-0 layout: direct, GI, and params.
    pub composite_static_layout: BindGroupLayoutId,
    /// Per-target composite group-1 layout: GBuffer reads and HDR write.
    pub composite_target_layout: BindGroupLayoutId,
    pub composite_bind_group: prism_render_driver::BindGroupId,
}

/// A graph-ready view of the current lighting GPU bundle.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LightingGpuPass {
    pub output: prism_render_driver::BufferId,
    pub direct_output: prism_render_driver::BufferId,
    pub gi_output: prism_render_driver::BufferId,
    pub params: prism_render_driver::BufferId,
    pub directionals: prism_render_driver::BufferId,
    pub punctuals: prism_render_driver::BufferId,
    pub areas: prism_render_driver::BufferId,
    pub candidates: prism_render_driver::BufferId,
    pub surfaces: prism_render_driver::BufferId,
    pub radiance: prism_render_driver::BufferId,
    pub direct_pipeline: prism_render_driver::ComputePipelineId,
    pub direct_bind_group: prism_render_driver::BindGroupId,
    pub gi_pipeline: prism_render_driver::ComputePipelineId,
    pub gi_bind_group: prism_render_driver::BindGroupId,
    pub composite_pipeline: prism_render_driver::ComputePipelineId,
    pub composite_static_layout: BindGroupLayoutId,
    pub composite_target_layout: BindGroupLayoutId,
    pub composite_bind_group: prism_render_driver::BindGroupId,
    pub run_direct: bool,
    pub run_gi: bool,
    pub candidate_source: GiCandidateSource,
    pub candidate_surface_count: u32,
    pub candidate_rays_per_surface: u32,
    pub workgroups: u32,
}

impl LightingGpuPass {
    #[must_use]
    pub(crate) const fn pass_count(self) -> u64 {
        (self.run_direct as u64) + (self.run_gi as u64) + 1
    }

    /// Returns the persistent buffers shared by trace candidate-build and GI.
    pub(crate) const fn gi_candidate_target(self) -> LightingGiCandidateTarget {
        LightingGiCandidateTarget {
            candidates: self.candidates,
            candidates_size_bytes: GI_CANDIDATE_BUFFER_BYTES,
            surface_count: self.candidate_surface_count,
            rays_per_surface: self.candidate_rays_per_surface,
            surfaces: self.surfaces,
            surfaces_size_bytes: LIGHTING_SURFACE_BUFFER_BYTES,
            params: self.params,
            params_size_bytes: LIGHTING_PARAMS_BYTES,
            radiance: self.radiance,
            radiance_size_bytes: TRACE_RADIANCE_BYTES,
        }
    }

    /// Returns the exact group-1 layout required by the per-target factory.
    #[must_use]
    pub(crate) const fn composite_target_layout(self) -> BindGroupLayoutId {
        self.composite_target_layout
    }

    /// Creates the real screen-space composite bindings for one target.
    ///
    /// The backend validates each view against the fixed layout. This method
    /// never manufactures a missing view, substitutes a buffer, or reuses the
    /// static lighting group. The returned group must be bound at group 1;
    /// [`Self::composite_bind_group`] is bound at group 0.
    pub(crate) fn create_composite_target_bind_group<D: prism_render_driver::RenderDevice>(
        self,
        device: &D,
        target: LightingCompositeTarget,
    ) -> Result<BindGroupId, LightingCompositeTargetError> {
        validate_composite_target(target)?;
        Ok(device.create_bind_group(&BindGroupDescriptor {
            label: Some("lighting-composite-target-bind-group".into()),
            layout: Some(self.composite_target_layout),
            entries: vec![
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(target.position_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(target.normal_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(target.albedo_view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(target.surface_id_view),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::TextureView(target.coverage_view),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: BindingResource::TextureView(target.hdr_target_view),
                },
            ],
        }))
    }

    /// Returns the dispatch dimensions for the target's exact pixel extent.
    #[must_use]
    pub(crate) const fn composite_workgroups(target: LightingCompositeTarget) -> [u32; 3] {
        [
            target.width.div_ceil(COMPOSITE_WORKGROUP_X),
            target.height.div_ceil(COMPOSITE_WORKGROUP_Y),
            1,
        ]
    }
}

/// Render-world ownership of lighting GPU resources with device-loss replay.
pub struct LightingGpuState<I>
where
    I: prism_render_driver::Instance + 'static,
{
    bundle: ReplayHandle<LightingGpuBundle>,
    prepared: Mutex<PreparedLightingFrame>,
    radiance_upload: Mutex<RadianceUploadCache>,
    light_upload: Mutex<LightUploadCache>,
    marker: core::marker::PhantomData<fn() -> I>,
}

#[derive(Default)]
struct PreparedLightingFrame {
    // Cleared before validation, without changing the last published counts.
    // Holding this lock across prepare also prevents pass from observing a
    // mixture of the previous frame's counts and the current frame's writes.
    bundle: Option<Arc<LightingGpuBundle>>,
    surface_count: u32,
    candidate_rays_per_surface: u32,
    candidate_source: GiCandidateSource,
    execution_flags: u32,
}

#[derive(Default)]
struct RadianceUploadCache {
    bundle: Option<Arc<LightingGpuBundle>>,
    bvh: Option<Arc<prism_render_architecture::ray_scene::GpuBvhBuffers>>,
    packed: Option<Arc<[[f32; 4]]>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct LightUploadToken(u64);

#[derive(Clone, Debug)]
struct PendingLightUpload {
    token: LightUploadToken,
    expected_submission: SubmissionProof,
    directional: UploadPlan,
    punctual: UploadPlan,
    area: UploadPlan,
}

#[derive(Default)]
struct LightUploadCache {
    bundle: Option<Arc<LightingGpuBundle>>,
    directionals: IncrementalGpuTable<LightingDirectionalInput>,
    punctuals: IncrementalGpuTable<LightingPunctualInput>,
    areas: IncrementalGpuTable<LightingAreaInput>,
    directional_slots: BTreeMap<LightSceneHandle, GpuTableHandle>,
    punctual_slots: BTreeMap<LightSceneHandle, GpuTableHandle>,
    area_slots: BTreeMap<LightSceneHandle, GpuTableHandle>,
    next_upload_token: u64,
    pending: Option<PendingLightUpload>,
}

pub(crate) const fn lighting_submission_domain(device_generation: u64) -> GpuSubmissionDomain {
    GpuSubmissionDomain::new(device_generation, LIGHTING_QUEUE_DOMAIN)
}

/// A frame could not prepare a valid lighting upload. No queue writes or new
/// counts are published on error, and `pass` remains unavailable until retry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LightingGpuPrepareError {
    /// Surface samples exceed the persistent GPU allocation.
    SurfaceCapacityExceeded { requested: usize, capacity: usize },
    /// Host batches must be one-to-one with surface samples.
    HostCandidateBatchCountMismatch { surfaces: usize, batches: usize },
    /// A host batch exceeds the candidate allocation for one surface.
    HostCandidateCapacityExceeded {
        surface_index: usize,
        requested: usize,
        capacity: usize,
    },
    /// Software tracing requires between one and `capacity` rays per surface.
    SoftwareRayCapacityExceeded { requested: u32, capacity: u32 },
    /// The selected trace input could not be packed into a valid radiance table.
    TracePreparation(TracePreparationError),
    /// The packed radiance table exceeds the resident allocation (in records).
    RadianceCapacityExceeded { requested: usize, capacity: usize },
    /// The light ABI payload and its canonical identities were misaligned.
    IdentityPayloadMismatch,
    /// A rare capacity-pressure compaction could not drain the queue safely.
    QueueCompletionUnconfirmed,
}

struct StagedLightingInputs {
    surfaces: Vec<GpuLightingSurface>,
    candidates: Vec<LightingGiCandidateInput>,
    radiance: Option<Arc<[[f32; 4]]>>,
    rays_per_surface: u32,
}

fn lighting_execution_flags(plan: &LightingPlan) -> u32 {
    u32::from(plan.recipes.contains(&RecipeId::SpecializedDeterministic))
        | (u32::from(plan.recipes.contains(&RecipeId::WorldRestir)) << 1)
}

fn validate_radiance_capacity(
    requested: usize,
    capacity: usize,
) -> Result<(), LightingGpuPrepareError> {
    if requested > capacity {
        return Err(LightingGpuPrepareError::RadianceCapacityExceeded {
            requested,
            capacity,
        });
    }
    Ok(())
}

// Stage every bounded payload before touching light-table state or the queue.
// The explicit radiance capacity also allows small boundary tests without
// allocating a multi-million-triangle scene.
fn stage_lighting_inputs(
    plan: &LightingPlan,
    inputs: &ExtractedLightingInputs,
    radiance_capacity: usize,
) -> Result<StagedLightingInputs, LightingGpuPrepareError> {
    let surface_count = inputs.surfaces.len();
    if surface_count > MAX_LIGHTING_SURFACES {
        return Err(LightingGpuPrepareError::SurfaceCapacityExceeded {
            requested: surface_count,
            capacity: MAX_LIGHTING_SURFACES,
        });
    }
    let trace = if plan.candidate_source == GiCandidateSource::SoftwareBvhGpu {
        let trace =
            inputs
                .trace_scene
                .as_ref()
                .ok_or(LightingGpuPrepareError::TracePreparation(
                    TracePreparationError::NoScene,
                ))?;
        if trace.rays_per_surface == 0 || trace.rays_per_surface > GI_CANDIDATES_PER_SURFACE as u32
        {
            return Err(LightingGpuPrepareError::SoftwareRayCapacityExceeded {
                requested: trace.rays_per_surface,
                capacity: GI_CANDIDATES_PER_SURFACE as u32,
            });
        }
        if trace.surface_ids.len() != surface_count {
            return Err(LightingGpuPrepareError::TracePreparation(
                TracePreparationError::SurfaceIdCountMismatch {
                    surfaces: surface_count,
                    ids: trace.surface_ids.len(),
                },
            ));
        }
        Some(trace)
    } else {
        None
    };
    if plan.candidate_source == GiCandidateSource::HostProvided {
        if inputs.gi_candidates.len() != surface_count {
            return Err(LightingGpuPrepareError::HostCandidateBatchCountMismatch {
                surfaces: surface_count,
                batches: inputs.gi_candidates.len(),
            });
        }
        for (surface_index, candidates) in inputs.gi_candidates.iter().enumerate() {
            if candidates.len() > GI_CANDIDATES_PER_SURFACE {
                return Err(LightingGpuPrepareError::HostCandidateCapacityExceeded {
                    surface_index,
                    requested: candidates.len(),
                    capacity: GI_CANDIDATES_PER_SURFACE,
                });
            }
        }
    }
    let radiance = trace
        .map(|trace| trace.packed_outgoing_radiance())
        .transpose()
        .map_err(LightingGpuPrepareError::TracePreparation)?;
    if let Some(packed) = &radiance {
        validate_radiance_capacity(packed.len(), radiance_capacity)?;
    }
    let rays_per_surface = trace.map_or(0, |trace| trace.rays_per_surface);
    let mut candidates = Vec::new();
    let surfaces = inputs
        .surfaces
        .iter()
        .enumerate()
        .map(|(surface_index, surface)| {
            let candidate_start = surface_index * GI_CANDIDATES_PER_SURFACE;
            let candidate_count = match plan.candidate_source {
                GiCandidateSource::HostProvided => {
                    let batch = &inputs.gi_candidates[surface_index];
                    candidates.extend_from_slice(batch);
                    candidates.resize(
                        candidate_start + GI_CANDIDATES_PER_SURFACE,
                        LightingGiCandidateInput::default(),
                    );
                    batch.len() as u32
                }
                GiCandidateSource::SoftwareBvhGpu => rays_per_surface,
                GiCandidateSource::Disabled => 0,
            };
            let stable_id = trace.map_or(0, |trace| trace.surface_ids[surface_index]);
            GpuLightingSurface {
                position: [
                    surface.position[0],
                    surface.position[1],
                    surface.position[2],
                    0.0,
                ],
                normal: [surface.normal[0], surface.normal[1], surface.normal[2], 0.0],
                albedo: [surface.albedo[0], surface.albedo[1], surface.albedo[2], 0.0],
                ambient: [
                    surface.ambient[0],
                    surface.ambient[1],
                    surface.ambient[2],
                    0.0,
                ],
                candidate_range: [
                    candidate_start as u32,
                    candidate_count,
                    stable_id as u32,
                    (stable_id >> 32) as u32,
                ],
            }
        })
        .collect();
    Ok(StagedLightingInputs {
        surfaces,
        candidates,
        radiance,
        rays_per_surface,
    })
}

impl<I> LightingGpuState<I>
where
    I: prism_render_driver::Instance + 'static,
{
    /// Creates the initial resource bundle and registers its replay recipe.
    pub fn new(
        runtime: &mut RenderRuntime<I>,
    ) -> Result<Self, prism_render_runtime::ReplayFailure> {
        let bundle = runtime.register_replay("lighting-kernels", |device, _queue| {
            Ok(build_bundle(device))
        })?;
        Ok(Self {
            bundle,
            prepared: Mutex::new(PreparedLightingFrame::default()),
            radiance_upload: Mutex::new(RadianceUploadCache::default()),
            light_upload: Mutex::new(LightUploadCache::default()),
            marker: core::marker::PhantomData,
        })
    }

    /// Uploads the current immutable lighting snapshot before dispatch.
    pub fn prepare(
        &self,
        plan: &LightingPlan,
        inputs: &ExtractedLightingInputs,
        frame_index: u64,
        runtime: &RenderRuntime<I>,
    ) -> Result<(), LightingGpuPrepareError>
    where
        I::Queue: FrameSync,
    {
        let mut prepared = self
            .prepared
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        prepared.bundle = None;
        let staged = stage_lighting_inputs(
            plan,
            inputs,
            TRACE_RADIANCE_BYTES as usize / core::mem::size_of::<[f32; 4]>(),
        )?;
        let bundle = self.bundle.current();
        // This is the final fallible operation. It validates all three light
        // tables before issuing their writes; no payload writes precede it.
        let (directional_count, punctual_count, area_count) =
            self.prepare_light_tables(runtime, &bundle, inputs)?;
        let queue = runtime.queue();
        if let Some(packed) = &staged.radiance {
            let trace = inputs.trace_scene.as_ref().expect("staged trace scene");
            let mut cache = self
                .radiance_upload
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let unchanged = cache
                .bundle
                .as_ref()
                .is_some_and(|cached| Arc::ptr_eq(cached, &bundle))
                && cache
                    .bvh
                    .as_ref()
                    .is_some_and(|bvh| Arc::ptr_eq(bvh, &trace.bvh))
                && cache
                    .packed
                    .as_ref()
                    .is_some_and(|table| Arc::ptr_eq(table, packed));
            if !unchanged {
                queue.write_buffer(bundle.radiance, 0, cast_slice(packed));
                cache.bvh = Some(Arc::clone(&trace.bvh));
                cache.bundle = Some(Arc::clone(&bundle));
                cache.packed = Some(Arc::clone(packed));
            }
        }
        let surface_count = staged.surfaces.len() as u32;
        let execution_flags = lighting_execution_flags(plan);
        let params = GpuLightingParams {
            counts: [
                surface_count,
                directional_count as u32,
                punctual_count as u32,
                staged.rays_per_surface,
            ],
            flags: [
                execution_flags,
                area_count as u32,
                frame_index as u32,
                staged
                    .radiance
                    .as_ref()
                    .map_or(0, |packed| packed.len() as u32),
            ],
        };
        queue.write_buffer(bundle.params, 0, bytes_of(&params));
        queue.write_buffer(bundle.surfaces, 0, cast_slice(&staged.surfaces));
        if !staged.candidates.is_empty() {
            queue.write_buffer(bundle.candidates, 0, cast_slice(&staged.candidates));
        }
        queue.write_buffer(bundle.output, 0, &ZERO_OUTPUT);
        queue.write_buffer(bundle.direct_output, 0, &ZERO_OUTPUT);
        queue.write_buffer(bundle.gi_output, 0, &ZERO_OUTPUT);
        // Publish one coherent frame only after every validation and write.
        *prepared = PreparedLightingFrame {
            bundle: Some(bundle),
            surface_count,
            candidate_rays_per_surface: staged.rays_per_surface,
            candidate_source: plan.candidate_source,
            execution_flags,
        };
        Ok(())
    }

    fn prepare_light_tables(
        &self,
        runtime: &RenderRuntime<I>,
        bundle: &Arc<LightingGpuBundle>,
        inputs: &ExtractedLightingInputs,
    ) -> Result<(usize, usize, usize), LightingGpuPrepareError>
    where
        I::Queue: FrameSync,
    {
        if inputs.directional_handles.len() != inputs.directionals.len()
            || inputs.punctual_handles.len() != inputs.punctuals.len()
            || inputs.area_handles.len() != inputs.areas.len()
        {
            return Err(LightingGpuPrepareError::IdentityPayloadMismatch);
        }
        // The bundle owns fixed-size buffers. Preserve the renderer's existing
        // deterministic first-N admission policy before stable-slot
        // reconciliation; compaction below may remove historical holes, but it
        // must never rebuild more live entries than the physical buffer holds.
        let directional_count = inputs.directionals.len().min(MAX_DIRECTIONALS);
        let punctual_count = inputs.punctuals.len().min(MAX_PUNCTUALS);
        let area_count = inputs.areas.len().min(MAX_AREAS);
        let directional_handles = &inputs.directional_handles[..directional_count];
        let directionals = &inputs.directionals[..directional_count];
        let punctual_handles = &inputs.punctual_handles[..punctual_count];
        let punctuals = &inputs.punctuals[..punctual_count];
        let area_handles = &inputs.area_handles[..area_count];
        let areas = &inputs.areas[..area_count];

        let mut cache = self
            .light_upload
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let same_bundle = cache
            .bundle
            .as_ref()
            .is_some_and(|cached| Arc::ptr_eq(cached, bundle));
        let domain = lighting_submission_domain(runtime.device_generation());
        let reset_gpu_copy = !same_bundle
            || cache.directionals.submission_domain() != domain
            || cache.punctuals.submission_domain() != domain;
        let reset_gpu_copy = reset_gpu_copy || cache.areas.submission_domain() != domain;
        let completed = domain.completion(runtime.completed_value());
        let expected_submission = domain
            .submission(runtime.next_completion_value())
            .expect("the runtime's next completion value is nonzero");
        let (directional_count, punctual_count, area_count, token) =
            prepare_light_upload_transaction(
                &mut cache,
                domain,
                completed,
                reset_gpu_copy,
                expected_submission,
                directional_handles,
                directionals,
                punctual_handles,
                punctuals,
                area_handles,
                areas,
                || runtime.wait_idle(),
            )?;
        cache.bundle = Some(Arc::clone(bundle));
        let pending = cache
            .pending
            .as_ref()
            .filter(|pending| pending.token == token)
            .expect("the prepared light upload transaction is current");
        let directional_plan = pending.directional.clone();
        let punctual_plan = pending.punctual.clone();
        let area_plan = pending.area.clone();
        write_table_plan(
            runtime.queue(),
            bundle.directionals,
            &cache.directionals,
            &directional_plan,
        );
        write_table_plan(
            runtime.queue(),
            bundle.punctuals,
            &cache.punctuals,
            &punctual_plan,
        );
        write_table_plan(runtime.queue(), bundle.areas, &cache.areas, &area_plan);
        Ok((directional_count, punctual_count, area_count))
    }

    /// Confirms the current table writes with the exact successful submission.
    pub(crate) fn acknowledge_light_upload(&self, submission: SubmissionProof) {
        let mut cache = self
            .light_upload
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(token) = cache
            .pending
            .as_ref()
            .filter(|pending| pending.expected_submission == submission)
            .map(|pending| pending.token)
        else {
            return;
        };
        acknowledge_pending_light_uploads(&mut cache, token, submission);
    }

    /// Returns kernels only for a successfully prepared matching plan/bundle.
    /// Callers must discard any previously captured pass when prepare fails.
    #[must_use]
    pub(crate) fn pass(&self, plan: &LightingPlan) -> Option<LightingGpuPass> {
        let run_direct = plan.recipes.contains(&RecipeId::SpecializedDeterministic);
        let run_gi = plan.recipes.contains(&RecipeId::WorldRestir)
            && plan.candidate_source != GiCandidateSource::Disabled;
        if !run_direct && !run_gi {
            return None;
        }
        let prepared = self
            .prepared
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let prepared_bundle = prepared.bundle.as_ref()?;
        let surface_count = prepared.surface_count;
        if surface_count == 0 {
            return None;
        }
        let bundle = self.bundle.current();
        if !Arc::ptr_eq(prepared_bundle, &bundle)
            || prepared.candidate_source != plan.candidate_source
            || prepared.execution_flags != lighting_execution_flags(plan)
        {
            return None;
        }
        Some(LightingGpuPass {
            output: bundle.output,
            direct_output: bundle.direct_output,
            gi_output: bundle.gi_output,
            params: bundle.params,
            directionals: bundle.directionals,
            punctuals: bundle.punctuals,
            areas: bundle.areas,
            candidates: bundle.candidates,
            surfaces: bundle.surfaces,
            radiance: bundle.radiance,
            direct_pipeline: bundle.direct_pipeline,
            direct_bind_group: bundle.direct_bind_group,
            gi_pipeline: bundle.gi_pipeline,
            gi_bind_group: bundle.gi_bind_group,
            composite_pipeline: bundle.composite_pipeline,
            composite_static_layout: bundle.composite_static_layout,
            composite_target_layout: bundle.composite_target_layout,
            composite_bind_group: bundle.composite_bind_group,
            run_direct,
            run_gi,
            candidate_source: plan.candidate_source,
            candidate_surface_count: surface_count,
            candidate_rays_per_surface: prepared.candidate_rays_per_surface,
            workgroups: surface_count.div_ceil(WORKGROUP_SIZE),
        })
    }

    /// Returns the final output buffer for diagnostics and readback tests.
    #[must_use]
    pub fn output_buffer(&self) -> prism_render_driver::BufferId {
        self.bundle.current().output
    }

    /// Returns the direct-light intermediate buffer for diagnostics and readback tests.
    #[must_use]
    pub fn direct_output_buffer(&self) -> prism_render_driver::BufferId {
        self.bundle.current().direct_output
    }

    /// Returns the initial GI intermediate buffer for diagnostics and readback tests.
    #[must_use]
    pub fn gi_output_buffer(&self) -> prism_render_driver::BufferId {
        self.bundle.current().gi_output
    }
}

impl<I> Resource for LightingGpuState<I> where I: prism_render_driver::Instance + 'static {}

#[derive(Debug)]
struct LightTableReconciliation {
    removed: Vec<LightSceneHandle>,
    insertions: usize,
}

fn validate_light_table_reconciliation<T>(
    table: &IncrementalGpuTable<T>,
    slots: &BTreeMap<LightSceneHandle, GpuTableHandle>,
    handles: &[LightSceneHandle],
    payloads: &[T],
) -> Result<LightTableReconciliation, LightingGpuPrepareError>
where
    T: GpuTablePayload,
{
    if handles.len() != payloads.len() {
        return Err(LightingGpuPrepareError::IdentityPayloadMismatch);
    }
    let incoming = handles.iter().copied().collect::<BTreeSet<_>>();
    if incoming.len() != handles.len()
        || slots
            .values()
            .copied()
            .any(|gpu_handle| table.get(gpu_handle).is_none())
    {
        return Err(LightingGpuPrepareError::IdentityPayloadMismatch);
    }
    Ok(LightTableReconciliation {
        removed: slots
            .keys()
            .filter(|handle| !incoming.contains(handle))
            .copied()
            .collect(),
        insertions: handles
            .iter()
            .filter(|handle| !slots.contains_key(handle))
            .count(),
    })
}

fn apply_light_table_reconciliation<T>(
    table: &mut IncrementalGpuTable<T>,
    slots: &mut BTreeMap<LightSceneHandle, GpuTableHandle>,
    reconciliation: LightTableReconciliation,
    handles: &[LightSceneHandle],
    payloads: &[T],
) where
    T: GpuTablePayload + Copy,
{
    for handle in reconciliation.removed {
        let gpu_handle = slots
            .remove(&handle)
            .expect("validated light slot must remain present until commit");
        assert!(
            table.retire(gpu_handle),
            "validated light slot must remain live until commit"
        );
    }
    for (&handle, &payload) in handles.iter().zip(payloads) {
        if let Some(&gpu_handle) = slots.get(&handle) {
            assert!(
                table.update(gpu_handle, payload),
                "validated light slot must remain live until commit"
            );
        } else {
            slots.insert(handle, table.insert(payload));
        }
    }
}

fn reconcile_light_table<T>(
    table: &mut IncrementalGpuTable<T>,
    slots: &mut BTreeMap<LightSceneHandle, GpuTableHandle>,
    handles: &[LightSceneHandle],
    payloads: &[T],
) -> Result<(), LightingGpuPrepareError>
where
    T: GpuTablePayload + Copy,
{
    let reconciliation = validate_light_table_reconciliation(table, slots, handles, payloads)?;
    apply_light_table_reconciliation(table, slots, reconciliation, handles, payloads);
    Ok(())
}

fn rebuild_light_tables(
    cache: &mut LightUploadCache,
    domain: GpuSubmissionDomain,
    directional_handles: &[LightSceneHandle],
    directionals: &[LightingDirectionalInput],
    punctual_handles: &[LightSceneHandle],
    punctuals: &[LightingPunctualInput],
    area_handles: &[LightSceneHandle],
    areas: &[LightingAreaInput],
) {
    cache.directionals = IncrementalGpuTable::default();
    cache.punctuals = IncrementalGpuTable::default();
    cache.areas = IncrementalGpuTable::default();
    cache.directionals.invalidate_gpu_copy(domain);
    cache.punctuals.invalidate_gpu_copy(domain);
    cache.areas.invalidate_gpu_copy(domain);
    cache.directional_slots.clear();
    cache.punctual_slots.clear();
    cache.area_slots.clear();
    for (&handle, &payload) in directional_handles.iter().zip(directionals) {
        let gpu_handle = cache.directionals.insert(payload);
        cache.directional_slots.insert(handle, gpu_handle);
    }
    for (&handle, &payload) in punctual_handles.iter().zip(punctuals) {
        let gpu_handle = cache.punctuals.insert(payload);
        cache.punctual_slots.insert(handle, gpu_handle);
    }
    for (&handle, &payload) in area_handles.iter().zip(areas) {
        let gpu_handle = cache.areas.insert(payload);
        cache.area_slots.insert(handle, gpu_handle);
    }
}

#[allow(clippy::too_many_arguments)]
fn prepare_light_upload_transaction<F>(
    cache: &mut LightUploadCache,
    domain: GpuSubmissionDomain,
    completed: CompletionProof,
    reset_gpu_copy: bool,
    expected_submission: SubmissionProof,
    directional_handles: &[LightSceneHandle],
    directionals: &[LightingDirectionalInput],
    punctual_handles: &[LightSceneHandle],
    punctuals: &[LightingPunctualInput],
    area_handles: &[LightSceneHandle],
    areas: &[LightingAreaInput],
    wait_idle: F,
) -> Result<(usize, usize, usize, LightUploadToken), LightingGpuPrepareError>
where
    F: FnOnce() -> bool,
{
    assert_eq!(expected_submission.domain(), domain);
    let directional_reconciliation = validate_light_table_reconciliation(
        &cache.directionals,
        &cache.directional_slots,
        directional_handles,
        directionals,
    )?;
    let punctual_reconciliation = validate_light_table_reconciliation(
        &cache.punctuals,
        &cache.punctual_slots,
        punctual_handles,
        punctuals,
    )?;
    let area_reconciliation =
        validate_light_table_reconciliation(&cache.areas, &cache.area_slots, area_handles, areas)?;

    let compact = !reset_gpu_copy
        && (cache
            .directionals
            .projected_capacity_after_reclaim(completed, directional_reconciliation.insertions)
            > MAX_DIRECTIONALS
            || cache
                .punctuals
                .projected_capacity_after_reclaim(completed, punctual_reconciliation.insertions)
                > MAX_PUNCTUALS
            || cache
                .areas
                .projected_capacity_after_reclaim(completed, area_reconciliation.insertions)
                > MAX_AREAS);
    if compact && !wait_idle() {
        return Err(LightingGpuPrepareError::QueueCompletionUnconfirmed);
    }

    if reset_gpu_copy || compact {
        rebuild_light_tables(
            cache,
            domain,
            directional_handles,
            directionals,
            punctual_handles,
            punctuals,
            area_handles,
            areas,
        );
    } else {
        cache.directionals.reclaim_completed(completed);
        cache.punctuals.reclaim_completed(completed);
        cache.areas.reclaim_completed(completed);
        apply_light_table_reconciliation(
            &mut cache.directionals,
            &mut cache.directional_slots,
            directional_reconciliation,
            directional_handles,
            directionals,
        );
        apply_light_table_reconciliation(
            &mut cache.punctuals,
            &mut cache.punctual_slots,
            punctual_reconciliation,
            punctual_handles,
            punctuals,
        );
        apply_light_table_reconciliation(
            &mut cache.areas,
            &mut cache.area_slots,
            area_reconciliation,
            area_handles,
            areas,
        );
    }

    cache.next_upload_token = cache
        .next_upload_token
        .checked_add(1)
        .expect("light upload token exhausted");
    let token = LightUploadToken(cache.next_upload_token);
    cache.pending = Some(PendingLightUpload {
        token,
        expected_submission,
        directional: cache.directionals.plan_upload(),
        punctual: cache.punctuals.plan_upload(),
        area: cache.areas.plan_upload(),
    });
    Ok((
        cache.directionals.capacity() as usize,
        cache.punctuals.capacity() as usize,
        cache.areas.capacity() as usize,
        token,
    ))
}

fn acknowledge_pending_light_uploads(
    cache: &mut LightUploadCache,
    token: LightUploadToken,
    submission: SubmissionProof,
) -> bool {
    let Some(pending) = cache.pending.as_ref() else {
        return false;
    };
    if pending.token != token
        || pending.expected_submission != submission
        || !cache
            .directionals
            .can_acknowledge_upload(&pending.directional, submission)
        || !cache
            .punctuals
            .can_acknowledge_upload(&pending.punctual, submission)
        || !cache
            .areas
            .can_acknowledge_upload(&pending.area, submission)
    {
        return false;
    }

    let pending = cache
        .pending
        .take()
        .expect("validated light upload transaction must remain pending until commit");
    assert!(cache
        .directionals
        .acknowledge_upload(&pending.directional, submission));
    assert!(cache
        .punctuals
        .acknowledge_upload(&pending.punctual, submission));
    assert!(cache.areas.acknowledge_upload(&pending.area, submission));
    true
}

#[cfg(test)]
mod prepare_tests {
    use super::*;
    use crate::lighting::{
        LightingSurfaceInput, LightingTraceSceneInput, LightingTriangleRadiance,
    };
    use prism_render_architecture::lighting::contract::{FrameResourceBudget, Tier};
    use prism_render_architecture::ray_scene::{Bvh, GpuBvhBuffers, Triangle};
    use prism_render_driver::{self as rhi, Instance};
    use prism_render_driver_null::{NullAdapter, NullDevice, NullInstance, NullQueue, NullSurface};

    struct BufferWrite {
        buffer: rhi::BufferId,
        offset: u64,
        data: Vec<u8>,
    }

    // Observe host uploads only: these tests do not execute any GPU shader.
    #[derive(Default)]
    struct RecordingQueue {
        writes: Mutex<Vec<BufferWrite>>,
        reject_idle: core::sync::atomic::AtomicBool,
    }

    impl RenderQueue for RecordingQueue {
        fn write_buffer(&self, buffer: rhi::BufferId, offset: u64, data: &[u8]) {
            self.writes.lock().unwrap().push(BufferWrite {
                buffer,
                offset,
                data: data.to_vec(),
            });
        }

        fn write_texture(
            &self,
            destination: rhi::TextureWrite,
            data: &[u8],
            layout: rhi::ImageDataLayout,
        ) {
            NullQueue::new().write_texture(destination, data, layout);
        }

        fn submit(&self, buffers: &[rhi::CommandBuffer]) {
            NullQueue::new().submit(buffers);
        }
    }

    impl FrameSync for RecordingQueue {
        fn submit_tracked(&self, buffers: &[rhi::CommandBuffer]) -> rhi::CompletionToken {
            NullQueue::new().submit_tracked(buffers)
        }

        fn wait_for(&self, token: &(dyn core::any::Any + Send + Sync)) -> bool {
            NullQueue::new().wait_for(token)
        }

        fn wait_idle(&self) -> bool {
            !self.reject_idle.load(core::sync::atomic::Ordering::Relaxed)
                && NullQueue::new().wait_idle()
        }

        fn on_submitted_work_done(&self, callback: Box<dyn FnOnce() + Send + 'static>) {
            NullQueue::new().on_submitted_work_done(callback);
        }
    }

    struct RecordingInstance;

    impl Instance for RecordingInstance {
        type Adapter = NullAdapter;
        type Device = NullDevice;
        type Queue = RecordingQueue;
        type Surface = NullSurface;

        fn enumerate_adapters(&self) -> Vec<Self::Adapter> {
            NullInstance::new().enumerate_adapters()
        }

        fn request_adapter(&self, options: &rhi::RequestAdapterOptions) -> Option<Self::Adapter> {
            NullInstance::new().request_adapter(options)
        }

        fn request_adapter_for_surface(
            &self,
            options: &rhi::RequestAdapterOptions,
            surface: &Self::Surface,
        ) -> Option<Self::Adapter> {
            NullInstance::new().request_adapter_for_surface(options, surface)
        }

        fn create_device(
            &self,
            adapter: &Self::Adapter,
            descriptor: &rhi::DeviceDescriptor,
        ) -> Result<(Self::Device, Self::Queue), rhi::DeviceError> {
            let (device, _) = NullInstance::new().create_device(adapter, descriptor)?;
            Ok((device, RecordingQueue::default()))
        }

        fn create_surface<W>(
            &self,
            window: Arc<W>,
            device: &Self::Device,
            _queue: &Self::Queue,
        ) -> Result<Self::Surface, rhi::SurfaceError>
        where
            W: raw_window_handle::HasWindowHandle
                + raw_window_handle::HasDisplayHandle
                + Send
                + Sync
                + 'static,
        {
            NullInstance::new().create_surface(window, device, &NullQueue::new())
        }
    }

    fn setup() -> (
        RenderRuntime<RecordingInstance>,
        LightingGpuState<RecordingInstance>,
    ) {
        let mut runtime =
            RenderRuntime::headless(RecordingInstance, rhi::Extent3d::new_2d(4, 4)).unwrap();
        let state = LightingGpuState::new(&mut runtime).unwrap();
        runtime.queue().writes.lock().unwrap().clear();
        (runtime, state)
    }

    fn plan(candidate_source: GiCandidateSource) -> LightingPlan {
        LightingPlan {
            tier: Tier::Balanced,
            recipes: vec![RecipeId::SpecializedDeterministic, RecipeId::WorldRestir],
            candidate_source,
            partition: Default::default(),
            budget: FrameResourceBudget::default(),
            rejected: Vec::new(),
        }
    }

    fn surfaces(count: usize) -> ExtractedLightingInputs {
        ExtractedLightingInputs {
            surfaces: vec![LightingSurfaceInput::default(); count],
            ..Default::default()
        }
    }

    fn trace_inputs(surface_count: usize, rays_per_surface: u32) -> ExtractedLightingInputs {
        let triangle = Triangle::new([-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [0.0, 1.0, -1.0], 7);
        let mut inputs = surfaces(surface_count);
        inputs.trace_scene = Some(LightingTraceSceneInput::new(
            Arc::new(GpuBvhBuffers::from_bvh(&Bvh::build(&[triangle]))),
            (0..surface_count)
                .map(|index| (1_u64 << 40) + index as u64)
                .collect(),
            rays_per_surface,
            0.001,
            10.0,
            Arc::from([LightingTriangleRadiance {
                primitive_id: 7,
                rgb: [2.0, 3.0, 4.0],
            }]),
            1,
        ));
        inputs
    }

    fn assert_rejected_atomically(
        rejected_plan: &LightingPlan,
        rejected_inputs: &ExtractedLightingInputs,
        expected: LightingGpuPrepareError,
    ) {
        let (runtime, state) = setup();
        let valid_plan = plan(GiCandidateSource::SoftwareBvhGpu);
        let valid_inputs = trace_inputs(2, 3);
        state
            .prepare(&valid_plan, &valid_inputs, 1, &runtime)
            .unwrap();
        assert!(state.pass(&valid_plan).is_some());
        let previous_token = state
            .light_upload
            .lock()
            .unwrap()
            .pending
            .as_ref()
            .unwrap()
            .token;
        let previous_radiance = state
            .radiance_upload
            .lock()
            .unwrap()
            .packed
            .clone()
            .unwrap();
        runtime.queue().writes.lock().unwrap().clear();

        assert_eq!(
            state.prepare(rejected_plan, rejected_inputs, 2, &runtime),
            Err(expected)
        );
        assert!(runtime.queue().writes.lock().unwrap().is_empty());
        {
            let prepared = state.prepared.lock().unwrap();
            assert_eq!(prepared.surface_count, 2);
            assert_eq!(prepared.candidate_rays_per_surface, 3);
            assert!(prepared.bundle.is_none());
        }
        assert!(state.pass(&valid_plan).is_none());
        assert!(state.pass(rejected_plan).is_none());
        assert_eq!(
            state
                .light_upload
                .lock()
                .unwrap()
                .pending
                .as_ref()
                .unwrap()
                .token,
            previous_token
        );
        assert!(Arc::ptr_eq(
            state
                .radiance_upload
                .lock()
                .unwrap()
                .packed
                .as_ref()
                .unwrap(),
            &previous_radiance,
        ));

        state
            .prepare(&valid_plan, &valid_inputs, 3, &runtime)
            .unwrap();
        assert!(state.pass(&valid_plan).is_some());
        assert!(!runtime.queue().writes.lock().unwrap().is_empty());
    }

    #[test]
    fn prepare_surface_capacity_accepts_neighbors_and_rejects_overflow_atomically() {
        let plan = plan(GiCandidateSource::Disabled);
        let (runtime, state) = setup();
        assert!(state.pass(&plan).is_none());
        for count in [0, 1, MAX_LIGHTING_SURFACES - 1, MAX_LIGHTING_SURFACES] {
            state.prepare(&plan, &surfaces(count), 1, &runtime).unwrap();
            assert_eq!(state.prepared.lock().unwrap().surface_count, count as u32);
            assert_eq!(
                state.pass(&plan).map(|pass| pass.candidate_surface_count),
                (count != 0).then_some(count as u32)
            );
        }
        assert_rejected_atomically(
            &plan,
            &surfaces(MAX_LIGHTING_SURFACES + 1),
            LightingGpuPrepareError::SurfaceCapacityExceeded {
                requested: MAX_LIGHTING_SURFACES + 1,
                capacity: MAX_LIGHTING_SURFACES,
            },
        );
    }

    #[test]
    fn prepare_host_candidate_capacity_preserves_tail_and_valid_neighbors() {
        let plan = plan(GiCandidateSource::HostProvided);
        let (runtime, state) = setup();
        for count in [
            0,
            1,
            GI_CANDIDATES_PER_SURFACE - 1,
            GI_CANDIDATES_PER_SURFACE,
        ] {
            let mut inputs = surfaces(3);
            inputs.gi_candidates = vec![
                vec![],
                vec![LightingGiCandidateInput::default(); count],
                vec![LightingGiCandidateInput::default()],
            ];
            if let Some(last) = inputs.gi_candidates[1].last_mut() {
                last.radiance = [17.0, 19.0, 23.0];
            }
            let staged = stage_lighting_inputs(&plan, &inputs, 0).unwrap();
            assert_eq!(staged.candidates.len(), 3 * GI_CANDIDATES_PER_SURFACE);
            assert_eq!(staged.surfaces[0].candidate_range[..2], [0, 0]);
            assert_eq!(staged.surfaces[1].candidate_range[..2], [64, count as u32]);
            assert_eq!(staged.surfaces[2].candidate_range[..2], [128, 1]);
            if count > 0 {
                assert_eq!(
                    staged.candidates[GI_CANDIDATES_PER_SURFACE + count - 1].radiance,
                    [17.0, 19.0, 23.0]
                );
            }
            runtime.queue().writes.lock().unwrap().clear();
            state.prepare(&plan, &inputs, 1, &runtime).unwrap();
            let bundle = state.bundle.current();
            let writes = runtime.queue().writes.lock().unwrap();
            let uploaded = writes
                .iter()
                .find(|write| write.buffer == bundle.candidates)
                .unwrap();
            assert_eq!(uploaded.offset, 0);
            assert_eq!(
                uploaded.data.as_slice(),
                cast_slice::<_, u8>(&staged.candidates)
            );
        }
        let mut overflow = surfaces(3);
        overflow.gi_candidates = vec![
            vec![],
            vec![LightingGiCandidateInput::default(); 65],
            vec![LightingGiCandidateInput::default()],
        ];
        assert_rejected_atomically(
            &plan,
            &overflow,
            LightingGpuPrepareError::HostCandidateCapacityExceeded {
                surface_index: 1,
                requested: 65,
                capacity: 64,
            },
        );
    }

    #[test]
    fn prepare_host_candidate_batch_alignment_rejects_missing_and_extra_batches() {
        let plan = plan(GiCandidateSource::HostProvided);
        for batches in [0, 2] {
            let mut inputs = surfaces(1);
            inputs.gi_candidates = vec![vec![]; batches];
            assert_rejected_atomically(
                &plan,
                &inputs,
                LightingGpuPrepareError::HostCandidateBatchCountMismatch {
                    surfaces: 1,
                    batches,
                },
            );
        }
    }

    #[test]
    fn prepare_software_ray_capacity_accepts_neighbors_and_rejects_zero_or_overflow() {
        let plan = plan(GiCandidateSource::SoftwareBvhGpu);
        let (runtime, state) = setup();
        for rays in [1, 63, 64] {
            let inputs = trace_inputs(1, rays);
            let staged = stage_lighting_inputs(&plan, &inputs, 1).unwrap();
            assert_eq!(staged.surfaces[0].candidate_range, [0, rays, 0, 256]);
            state.prepare(&plan, &inputs, 1, &runtime).unwrap();
            assert_eq!(state.pass(&plan).unwrap().candidate_rays_per_surface, rays);
        }
        for rays in [0, 65] {
            assert_rejected_atomically(
                &plan,
                &trace_inputs(1, rays),
                LightingGpuPrepareError::SoftwareRayCapacityExceeded {
                    requested: rays,
                    capacity: 64,
                },
            );
        }
    }

    #[test]
    fn prepare_trace_requires_scene_and_aligned_surface_ids() {
        let plan = plan(GiCandidateSource::SoftwareBvhGpu);
        assert_rejected_atomically(
            &plan,
            &surfaces(1),
            LightingGpuPrepareError::TracePreparation(TracePreparationError::NoScene),
        );
        let mut inputs = trace_inputs(1, 1);
        inputs.trace_scene.as_mut().unwrap().surface_ids.clear();
        assert_rejected_atomically(
            &plan,
            &inputs,
            LightingGpuPrepareError::TracePreparation(
                TracePreparationError::SurfaceIdCountMismatch {
                    surfaces: 1,
                    ids: 0,
                },
            ),
        );
    }

    #[test]
    fn prepare_radiance_pack_errors_are_structured_and_atomic() {
        let plan = plan(GiCandidateSource::SoftwareBvhGpu);
        let mut malformed = trace_inputs(1, 1);
        Arc::make_mut(&mut malformed.trace_scene.as_mut().unwrap().bvh)
            .triangles
            .push(0);
        assert_rejected_atomically(
            &plan,
            &malformed,
            LightingGpuPrepareError::TracePreparation(TracePreparationError::Trace(
                crate::lighting_trace_gpu::TraceGpuError::MalformedPackedLength,
            )),
        );
        let mut missing = trace_inputs(1, 1);
        missing.trace_scene.as_mut().unwrap().outgoing_radiance = Arc::from([]);
        assert_rejected_atomically(
            &plan,
            &missing,
            LightingGpuPrepareError::TracePreparation(
                TracePreparationError::RadianceCountMismatch {
                    triangles: 1,
                    radiance: 0,
                },
            ),
        );
        for invalid in [f32::NAN, f32::INFINITY, -1.0] {
            let mut inputs = trace_inputs(1, 1);
            inputs.trace_scene.as_mut().unwrap().outgoing_radiance =
                Arc::from([LightingTriangleRadiance {
                    primitive_id: 7,
                    rgb: [invalid, 0.0, 0.0],
                }]);
            assert_rejected_atomically(
                &plan,
                &inputs,
                LightingGpuPrepareError::TracePreparation(TracePreparationError::InvalidRadiance(
                    0,
                )),
            );
        }
        let mut wrong_id = trace_inputs(1, 1);
        wrong_id.trace_scene.as_mut().unwrap().outgoing_radiance =
            Arc::from([LightingTriangleRadiance {
                primitive_id: 9,
                rgb: [1.0; 3],
            }]);
        assert_rejected_atomically(
            &plan,
            &wrong_id,
            LightingGpuPrepareError::TracePreparation(
                TracePreparationError::RadiancePrimitiveSetMismatch {
                    radiance_id: 9,
                    packed_id: 7,
                },
            ),
        );
    }

    #[test]
    fn prepare_radiance_capacity_checks_exact_limit_and_cached_packing() {
        let capacity = TRACE_RADIANCE_BYTES as usize / core::mem::size_of::<[f32; 4]>();
        for count in [0, capacity - 1, capacity] {
            assert_eq!(validate_radiance_capacity(count, capacity), Ok(()));
        }
        assert_eq!(
            validate_radiance_capacity(capacity + 1, capacity),
            Err(LightingGpuPrepareError::RadianceCapacityExceeded {
                requested: capacity + 1,
                capacity
            },)
        );
        let plan = plan(GiCandidateSource::SoftwareBvhGpu);
        let inputs = trace_inputs(1, 1);
        let packed = inputs
            .trace_scene
            .as_ref()
            .unwrap()
            .packed_outgoing_radiance()
            .unwrap();
        assert!(Arc::ptr_eq(
            &packed,
            stage_lighting_inputs(&plan, &inputs, 1)
                .unwrap()
                .radiance
                .as_ref()
                .unwrap()
        ));
        assert!(matches!(
            stage_lighting_inputs(&plan, &inputs, 0),
            Err(LightingGpuPrepareError::RadianceCapacityExceeded {
                requested: 1,
                capacity: 0
            },)
        ));
        assert!(stage_lighting_inputs(&plan, &inputs, 1).is_ok());
    }

    #[test]
    fn prepare_light_identity_error_does_not_upload_valid_radiance_first() {
        let mut inputs = trace_inputs(1, 1);
        inputs
            .directionals
            .push(LightingDirectionalInput::default());
        assert_rejected_atomically(
            &plan(GiCandidateSource::SoftwareBvhGpu),
            &inputs,
            LightingGpuPrepareError::IdentityPayloadMismatch,
        );
        inputs.directionals.clear();
        inputs.punctuals.push(LightingPunctualInput::default());
        assert_rejected_atomically(
            &plan(GiCandidateSource::SoftwareBvhGpu),
            &inputs,
            LightingGpuPrepareError::IdentityPayloadMismatch,
        );
        inputs.punctuals.clear();
        inputs.areas.push(LightingAreaInput::default());
        assert_rejected_atomically(
            &plan(GiCandidateSource::SoftwareBvhGpu),
            &inputs,
            LightingGpuPrepareError::IdentityPayloadMismatch,
        );
        // Equal lengths still require canonical identity validation inside the
        // light-table transaction, before even a valid radiance upload.
        inputs.areas.push(LightingAreaInput::default());
        inputs.area_handles = vec![LightSceneHandle::new(5, 1); 2];
        assert_rejected_atomically(
            &plan(GiCandidateSource::SoftwareBvhGpu),
            &inputs,
            LightingGpuPrepareError::IdentityPayloadMismatch,
        );
    }

    #[test]
    fn prepare_compaction_wait_failure_blocks_stale_pass_without_uploads() {
        let (runtime, state) = setup();
        let plan = plan(GiCandidateSource::SoftwareBvhGpu);
        let mut full = trace_inputs(2, 3);
        full.directionals = vec![LightingDirectionalInput::default(); MAX_DIRECTIONALS];
        full.directional_handles = (0..MAX_DIRECTIONALS)
            .map(|index| LightSceneHandle::new(index as u32, 1))
            .collect();
        state.prepare(&plan, &full, 1, &runtime).unwrap();
        let previous_token = state
            .light_upload
            .lock()
            .unwrap()
            .pending
            .as_ref()
            .unwrap()
            .token;
        let mut replacement = trace_inputs(1, 1);
        replacement.directionals = vec![LightingDirectionalInput::default()];
        replacement.directional_handles = vec![LightSceneHandle::new(10_000, 1)];
        runtime.queue().writes.lock().unwrap().clear();
        runtime
            .queue()
            .reject_idle
            .store(true, core::sync::atomic::Ordering::Relaxed);
        assert_eq!(
            state.prepare(&plan, &replacement, 2, &runtime),
            Err(LightingGpuPrepareError::QueueCompletionUnconfirmed),
        );
        assert!(runtime.queue().writes.lock().unwrap().is_empty());
        assert!(state.pass(&plan).is_none());
        {
            let prepared = state.prepared.lock().unwrap();
            assert_eq!(prepared.surface_count, 2);
            assert_eq!(prepared.candidate_rays_per_surface, 3);
            assert!(prepared.bundle.is_none());
        }
        {
            let cache = state.light_upload.lock().unwrap();
            assert_eq!(cache.pending.as_ref().unwrap().token, previous_token);
            assert_eq!(cache.directional_slots.len(), MAX_DIRECTIONALS);
            assert!(cache
                .directional_slots
                .contains_key(&full.directional_handles[0]));
        }
        runtime
            .queue()
            .reject_idle
            .store(false, core::sync::atomic::Ordering::Relaxed);
        state.prepare(&plan, &replacement, 3, &runtime).unwrap();
        assert_eq!(state.pass(&plan).unwrap().candidate_surface_count, 1);
        assert_eq!(state.prepared.lock().unwrap().candidate_rays_per_surface, 1);
    }

    #[test]
    fn prepare_ignores_inactive_candidate_sources_and_reuses_valid_radiance_upload() {
        let (runtime, state) = setup();
        let mut inputs = trace_inputs(1, 65);
        inputs.gi_candidates = vec![vec![LightingGiCandidateInput::default(); 65]];
        let disabled = plan(GiCandidateSource::Disabled);
        state.prepare(&disabled, &inputs, 1, &runtime).unwrap();
        let host = plan(GiCandidateSource::HostProvided);
        inputs.gi_candidates[0].clear();
        state.prepare(&host, &inputs, 2, &runtime).unwrap();
        let software = plan(GiCandidateSource::SoftwareBvhGpu);
        inputs.gi_candidates[0].resize(65, LightingGiCandidateInput::default());
        inputs.trace_scene.as_mut().unwrap().rays_per_surface = 1;
        state.prepare(&software, &inputs, 3, &runtime).unwrap();
        let bundle = state.bundle.current();
        assert!(runtime
            .queue()
            .writes
            .lock()
            .unwrap()
            .iter()
            .any(|write| write.buffer == bundle.radiance));
        runtime.queue().writes.lock().unwrap().clear();
        state.prepare(&software, &inputs, 4, &runtime).unwrap();
        assert!(!runtime
            .queue()
            .writes
            .lock()
            .unwrap()
            .iter()
            .any(|write| write.buffer == bundle.radiance));
    }

    #[test]
    fn prepare_pass_rejects_mismatched_plan_and_replayed_bundle_until_reprepare() {
        let (mut runtime, state) = setup();
        let inputs = trace_inputs(1, 1);
        let software = plan(GiCandidateSource::SoftwareBvhGpu);
        state.prepare(&software, &inputs, 1, &runtime).unwrap();
        assert!(state.pass(&software).is_some());
        assert!(state.pass(&plan(GiCandidateSource::HostProvided)).is_none());
        let mut other_recipes = software.clone();
        other_recipes.recipes = vec![RecipeId::WorldRestir];
        assert!(state.pass(&other_recipes).is_none());
        runtime.recover_device().unwrap();
        assert!(state.pass(&software).is_none());
        state.prepare(&software, &inputs, 2, &runtime).unwrap();
        assert!(state.pass(&software).is_some());
        let bundle = state.bundle.current();
        assert!(runtime
            .queue()
            .writes
            .lock()
            .unwrap()
            .iter()
            .any(|write| write.buffer == bundle.radiance));
    }
}

#[cfg(test)]
mod light_table_tests {
    use super::*;
    use prism_render_architecture::lighting::UploadPolicy;

    fn canonical(slot: u32, generation: u32) -> LightSceneHandle {
        LightSceneHandle::new(slot, generation)
    }

    fn submission(value: u64) -> SubmissionProof {
        lighting_submission_domain(0)
            .submission(value)
            .expect("test submission is nonzero")
    }

    fn directional(value: f32) -> LightingDirectionalInput {
        LightingDirectionalInput {
            illuminance: [value, value, value],
            ..LightingDirectionalInput::default()
        }
    }

    fn area(value: f32) -> LightingAreaInput {
        LightingAreaInput {
            origin: [value, 1.0, 0.0, 0.0],
            vector_a: [0.0, -1.0, 0.0, 0.0],
            vector_b: [1.0, 0.0, 0.0, 0.0],
            vector_c: [0.0, 0.0, 1.0, 0.0],
            radiance: [value, value, value, 0.0],
            shape: [0.5, 0.5, 0.0, 0.0],
            shape_kind: [0, 1, 0, 0],
        }
    }

    #[test]
    fn area_payload_abi_matches_the_wgsl_vec4_layout() {
        assert_eq!(
            core::mem::size_of::<LightingAreaInput>(),
            7 * core::mem::size_of::<[f32; 4]>()
        );
        assert_eq!(core::mem::align_of::<LightingAreaInput>(), 4);
        assert_eq!(
            bytemuck::bytes_of(&area(2.0)).len(),
            AREA_BYTES as usize / MAX_AREAS
        );
    }

    #[test]
    fn area_edits_preserve_gpu_slots_and_remove_to_neutral() {
        let mut table = IncrementalGpuTable::new(UploadPolicy::default()).unwrap();
        let mut slots = BTreeMap::new();
        let first = canonical(20, 1);
        let second = canonical(21, 1);
        reconcile_light_table(
            &mut table,
            &mut slots,
            &[first, second],
            &[area(1.0), area(2.0)],
        )
        .unwrap();
        let first_gpu = slots[&first];
        let second_gpu = slots[&second];
        assert_eq!((first_gpu.slot(), second_gpu.slot()), (0, 1));
        let initial = table.plan_upload();
        assert_eq!(initial.ranges()[0].elements, 0..2);
        assert!(table.acknowledge_upload(&initial, submission(1)));

        reconcile_light_table(
            &mut table,
            &mut slots,
            &[first, second],
            &[area(3.0), area(2.0)],
        )
        .unwrap();
        let update = table.plan_upload();
        assert_eq!(update.ranges()[0].elements, 0..1);
        assert!(table.acknowledge_upload(&update, submission(2)));

        reconcile_light_table(&mut table, &mut slots, &[second], &[area(2.0)]).unwrap();
        assert!(!slots.contains_key(&first));
        assert_eq!(
            table.payloads()[first_gpu.slot() as usize],
            LightingAreaInput::default()
        );
        assert_eq!(slots[&second], second_gpu);
    }

    #[test]
    fn malformed_area_table_does_not_mutate_directional_or_punctual_tables() {
        let mut cache = LightUploadCache::default();
        let directional_handle = canonical(30, 1);
        let punctual_handle = canonical(31, 1);
        let area_handle = canonical(32, 1);
        let result = prepare_light_upload_transaction(
            &mut cache,
            lighting_submission_domain(0),
            lighting_submission_domain(0).completion(0),
            false,
            submission(1),
            &[directional_handle],
            &[directional(1.0)],
            &[punctual_handle],
            &[LightingPunctualInput::default()],
            &[area_handle],
            &[],
            || panic!("malformed area input must fail before compaction"),
        );
        assert_eq!(
            result,
            Err(LightingGpuPrepareError::IdentityPayloadMismatch)
        );
        assert!(cache.directional_slots.is_empty());
        assert!(cache.punctual_slots.is_empty());
        assert!(cache.area_slots.is_empty());
    }

    #[test]
    fn area_table_rejection_keeps_the_three_table_transaction_pending() {
        let mut cache = LightUploadCache::default();
        cache.directionals.insert(directional(1.0));
        cache.punctuals.insert(LightingPunctualInput::default());
        cache.areas.insert(area(1.0));
        let token = LightUploadToken(1);
        cache.pending = Some(PendingLightUpload {
            token,
            expected_submission: submission(1),
            directional: cache.directionals.plan_upload(),
            punctual: cache.punctuals.plan_upload(),
            area: cache.areas.plan_upload(),
        });
        cache
            .areas
            .invalidate_gpu_copy(GpuSubmissionDomain::new(1, LIGHTING_QUEUE_DOMAIN));

        assert!(!acknowledge_pending_light_uploads(
            &mut cache,
            token,
            submission(1)
        ));
        assert!(cache.pending.is_some());
        assert!(!cache.directionals.plan_upload().is_empty());
        assert!(!cache.punctuals.plan_upload().is_empty());
        assert!(!cache.areas.plan_upload().is_empty());
    }

    #[test]
    fn canonical_edits_preserve_gpu_slots_and_remove_to_neutral() {
        let mut table = IncrementalGpuTable::new(UploadPolicy::default()).unwrap();
        let mut slots = BTreeMap::new();
        let first = canonical(10, 1);
        let second = canonical(11, 1);
        reconcile_light_table(
            &mut table,
            &mut slots,
            &[first, second],
            &[directional(1.0), directional(2.0)],
        )
        .unwrap();
        let first_gpu = slots[&first];
        let second_gpu = slots[&second];
        assert_eq!((first_gpu.slot(), second_gpu.slot()), (0, 1));

        let initial = table.plan_upload();
        assert_eq!(initial.ranges()[0].elements, 0..2);
        assert!(table.acknowledge_upload(&initial, submission(1)));
        reconcile_light_table(
            &mut table,
            &mut slots,
            &[first, second],
            &[directional(1.0), directional(3.0)],
        )
        .unwrap();
        let update = table.plan_upload();
        assert_eq!(update.ranges()[0].elements, 1..2);
        assert!(table.acknowledge_upload(&update, submission(2)));

        reconcile_light_table(&mut table, &mut slots, &[second], &[directional(3.0)]).unwrap();
        assert!(!slots.contains_key(&first));
        assert_eq!(table.live_len(), 1);
        assert_eq!(
            table.payloads()[first_gpu.slot() as usize],
            LightingDirectionalInput::default()
        );
        assert_eq!(slots[&second], second_gpu);
    }

    #[test]
    fn malformed_identity_payload_pairs_are_rejected_atomically() {
        let mut table = IncrementalGpuTable::default();
        let mut slots = BTreeMap::new();
        let handle = canonical(3, 1);
        assert_eq!(
            reconcile_light_table(&mut table, &mut slots, &[handle], &[]),
            Err(LightingGpuPrepareError::IdentityPayloadMismatch)
        );
        assert_eq!(table.live_len(), 0);
        assert_eq!(
            reconcile_light_table(
                &mut table,
                &mut slots,
                &[handle, handle],
                &[directional(1.0), directional(2.0)],
            ),
            Err(LightingGpuPrepareError::IdentityPayloadMismatch)
        );
        assert_eq!(table.live_len(), 0);
    }

    #[test]
    fn malformed_second_table_does_not_mutate_the_first_table() {
        let mut cache = LightUploadCache::default();
        let directional_handle = canonical(10, 1);
        let punctual_handle = canonical(11, 1);
        let result = prepare_light_upload_transaction(
            &mut cache,
            lighting_submission_domain(0),
            lighting_submission_domain(0).completion(0),
            false,
            submission(1),
            &[directional_handle],
            &[directional(1.0)],
            &[punctual_handle],
            &[],
            &[],
            &[],
            || panic!("malformed input must fail before compaction"),
        );
        assert_eq!(
            result,
            Err(LightingGpuPrepareError::IdentityPayloadMismatch)
        );
        assert!(cache.directional_slots.is_empty());
        assert!(cache.punctual_slots.is_empty());
        assert_eq!(cache.directionals.capacity(), 0);
        assert_eq!(cache.punctuals.capacity(), 0);
    }

    #[test]
    fn light_upload_acknowledgement_is_all_or_nothing_when_one_table_rejects() {
        let mut cache = LightUploadCache::default();
        cache.directionals.insert(directional(1.0));
        cache.punctuals.insert(LightingPunctualInput::default());
        let directional = cache.directionals.plan_upload();
        let punctual = cache.punctuals.plan_upload();
        let token = LightUploadToken(1);
        cache.pending = Some(PendingLightUpload {
            token,
            expected_submission: submission(1),
            directional,
            punctual,
            area: cache.areas.plan_upload(),
        });

        cache
            .punctuals
            .invalidate_gpu_copy(GpuSubmissionDomain::new(1, LIGHTING_QUEUE_DOMAIN));
        assert!(!acknowledge_pending_light_uploads(
            &mut cache,
            token,
            submission(1)
        ));

        assert!(cache.pending.is_some());
        assert!(!cache.directionals.plan_upload().is_empty());
        assert!(!cache.punctuals.plan_upload().is_empty());
    }

    #[test]
    fn acknowledgement_validates_both_tables_before_committing_either() {
        let mut cache = LightUploadCache::default();
        cache.directionals.insert(directional(1.0));
        cache.punctuals.insert(LightingPunctualInput::default());
        let token = LightUploadToken(1);
        cache.pending = Some(PendingLightUpload {
            token,
            expected_submission: submission(1),
            directional: cache.directionals.plan_upload(),
            punctual: cache.punctuals.plan_upload(),
            area: cache.areas.plan_upload(),
        });
        cache
            .punctuals
            .invalidate_gpu_copy(GpuSubmissionDomain::new(1, LIGHTING_QUEUE_DOMAIN));

        assert!(!acknowledge_pending_light_uploads(
            &mut cache,
            token,
            submission(1)
        ));
        assert!(!cache.directionals.plan_upload().is_empty());
        assert!(cache.pending.is_some());
    }

    #[test]
    fn old_submission_cannot_acknowledge_a_new_upload_token() {
        let mut cache = LightUploadCache::default();
        cache.directionals.insert(directional(1.0));
        let first = cache.directionals.plan_upload();
        cache.pending = Some(PendingLightUpload {
            token: LightUploadToken(1),
            expected_submission: submission(1),
            directional: first,
            punctual: cache.punctuals.plan_upload(),
            area: cache.areas.plan_upload(),
        });
        cache.pending = Some(PendingLightUpload {
            token: LightUploadToken(2),
            expected_submission: submission(1),
            directional: cache.directionals.plan_upload(),
            punctual: cache.punctuals.plan_upload(),
            area: cache.areas.plan_upload(),
        });

        assert!(
            acknowledge_pending_light_uploads(&mut cache, LightUploadToken(1), submission(1))
                == false
        );
        assert!(cache.pending.is_some());
        assert!(!cache.directionals.plan_upload().is_empty());
    }

    #[test]
    fn compaction_wait_failure_keeps_both_tables_and_slot_maps_unchanged() {
        let mut cache = LightUploadCache::default();
        let old_directional_handles = (0..MAX_DIRECTIONALS)
            .map(|index| canonical(index as u32, 1))
            .collect::<Vec<_>>();
        let old_punctual_handles = (MAX_DIRECTIONALS..MAX_PUNCTUALS * 2)
            .map(|index| canonical(index as u32, 1))
            .collect::<Vec<_>>();
        let directional_payloads = vec![directional(1.0); MAX_DIRECTIONALS];
        let punctual_payloads = vec![LightingPunctualInput::default(); MAX_PUNCTUALS];
        reconcile_light_table(
            &mut cache.directionals,
            &mut cache.directional_slots,
            &old_directional_handles,
            &directional_payloads,
        )
        .unwrap();
        reconcile_light_table(
            &mut cache.punctuals,
            &mut cache.punctual_slots,
            &old_punctual_handles,
            &punctual_payloads,
        )
        .unwrap();
        let new_directional = canonical(10_000, 1);
        let new_punctual = canonical(20_000, 1);
        let result = prepare_light_upload_transaction(
            &mut cache,
            lighting_submission_domain(0),
            lighting_submission_domain(0).completion(0),
            false,
            submission(1),
            &[new_directional],
            &[directional(2.0)],
            &[new_punctual],
            &[LightingPunctualInput::default()],
            &[],
            &[],
            || false,
        );
        assert_eq!(
            result,
            Err(LightingGpuPrepareError::QueueCompletionUnconfirmed)
        );
        assert_eq!(cache.directionals.capacity() as usize, MAX_DIRECTIONALS);
        assert_eq!(cache.punctuals.capacity() as usize, MAX_PUNCTUALS);
        assert!(cache
            .directional_slots
            .contains_key(&old_directional_handles[0]));
        assert!(cache.punctual_slots.contains_key(&old_punctual_handles[0]));
        assert!(!cache.directional_slots.contains_key(&new_directional));
        assert!(!cache.punctual_slots.contains_key(&new_punctual));
    }

    #[test]
    fn empty_table_recovery_stages_an_empty_rebuild_transaction() {
        let mut cache = LightUploadCache::default();
        let (_, _, _, token) = prepare_light_upload_transaction(
            &mut cache,
            lighting_submission_domain(1),
            lighting_submission_domain(1).completion(0),
            true,
            lighting_submission_domain(1).submission(1).unwrap(),
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            || panic!("recovery must not wait for idle"),
        )
        .unwrap();
        let pending = cache.pending.as_ref().unwrap();
        assert!(pending.directional.is_empty());
        assert!(pending.punctual.is_empty());
        assert!(acknowledge_pending_light_uploads(
            &mut cache,
            token,
            lighting_submission_domain(1).submission(1).unwrap(),
        ));
        assert!(cache.pending.is_none());
    }

    #[test]
    fn replacing_pending_plans_after_a_skipped_submit_preserves_dirty_work() {
        let mut cache = LightUploadCache::default();
        cache.directionals.insert(directional(1.0));
        cache.pending = Some(PendingLightUpload {
            token: LightUploadToken(1),
            expected_submission: submission(1),
            directional: cache.directionals.plan_upload(),
            punctual: cache.punctuals.plan_upload(),
            area: cache.areas.plan_upload(),
        });
        cache.pending = Some(PendingLightUpload {
            token: LightUploadToken(2),
            expected_submission: submission(1),
            directional: cache.directionals.plan_upload(),
            punctual: cache.punctuals.plan_upload(),
            area: cache.areas.plan_upload(),
        });

        assert!(!acknowledge_pending_light_uploads(
            &mut cache,
            LightUploadToken(1),
            submission(1),
        ));
        assert!(cache.pending.is_some());
        assert!(!cache.directionals.plan_upload().is_empty());
    }
}

fn write_table_plan<T, Q>(
    queue: &Q,
    buffer: prism_render_driver::BufferId,
    table: &IncrementalGpuTable<T>,
    plan: &UploadPlan,
) where
    T: GpuTablePayload,
    Q: RenderQueue,
{
    for range in plan.ranges() {
        let bytes = table
            .payload_bytes(range.elements.clone())
            .expect("upload plan range must be within the table payload");
        debug_assert_eq!(bytes.len() as u64, range.byte_len);
        queue.write_buffer(buffer, range.byte_offset, bytes);
    }
}

fn storage_layout(binding: u32, min_binding_size: u64, read_only: bool) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: Some(min_binding_size),
        },
        count: None,
    }
}

fn uniform_layout(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: Some(PARAM_BYTES),
        },
        count: None,
    }
}

fn buffer_binding(
    binding: u32,
    buffer: prism_render_driver::BufferId,
    size: u64,
) -> BindGroupEntry {
    BindGroupEntry {
        binding,
        resource: BindingResource::Buffer {
            buffer,
            offset: 0,
            size: Some(size),
        },
    }
}

fn build_bundle<D: prism_render_driver::RenderDevice>(device: &D) -> LightingGpuBundle {
    let output = device.create_buffer(&BufferDescriptor {
        label: Some("lighting-final-output".into()),
        size: OUTPUT_BYTES,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let direct_output = device.create_buffer(&BufferDescriptor {
        label: Some("lighting-direct-output".into()),
        size: OUTPUT_BYTES,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let gi_output = device.create_buffer(&BufferDescriptor {
        label: Some("lighting-gi-output".into()),
        size: OUTPUT_BYTES,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params = device.create_buffer(&BufferDescriptor {
        label: Some("lighting-params".into()),
        size: PARAM_BYTES,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let directionals = device.create_buffer(&BufferDescriptor {
        label: Some("lighting-directional-lights".into()),
        size: DIRECTIONAL_BYTES,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let punctuals = device.create_buffer(&BufferDescriptor {
        label: Some("lighting-punctual-lights".into()),
        size: PUNCTUAL_BYTES,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let areas = device.create_buffer(&BufferDescriptor {
        label: Some("lighting-area-lights".into()),
        size: AREA_BYTES,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let candidates = device.create_buffer(&BufferDescriptor {
        label: Some("lighting-gi-candidates".into()),
        size: GI_CANDIDATE_BYTES,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let surfaces = device.create_buffer(&BufferDescriptor {
        label: Some("lighting-surfaces".into()),
        size: SURFACE_BYTES,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let radiance = device.create_buffer(&BufferDescriptor {
        label: Some("lighting-trace-radiance".into()),
        size: TRACE_RADIANCE_BYTES,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let direct_shader = device.create_shader_module(&ShaderModuleDescriptor::wgsl(DIRECT_SHADER));
    let direct_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("lighting-direct-layout".into()),
        entries: vec![
            storage_layout(0, OUTPUT_BYTES, false),
            uniform_layout(1),
            storage_layout(2, DIRECTIONAL_BYTES, true),
            storage_layout(3, PUNCTUAL_BYTES, true),
            storage_layout(4, SURFACE_BYTES, true),
            storage_layout(5, AREA_BYTES, true),
        ],
    });
    let direct_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("lighting-direct-pipeline-layout".into()),
        bind_group_layouts: vec![direct_layout],
        push_constant_ranges: Vec::new(),
    });
    let direct_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("lighting-direct-compute".into()),
        layout: Some(direct_pipeline_layout),
        module: direct_shader,
        entry_point: "main".into(),
    });
    let direct_bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("lighting-direct-bind-group".into()),
        layout: Some(direct_layout),
        entries: vec![
            buffer_binding(0, direct_output, OUTPUT_BYTES),
            buffer_binding(1, params, PARAM_BYTES),
            buffer_binding(2, directionals, DIRECTIONAL_BYTES),
            buffer_binding(3, punctuals, PUNCTUAL_BYTES),
            buffer_binding(4, surfaces, SURFACE_BYTES),
            buffer_binding(5, areas, AREA_BYTES),
        ],
    });

    let gi_shader = device.create_shader_module(&ShaderModuleDescriptor::wgsl(GI_SHADER));
    let gi_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("lighting-gi-layout".into()),
        entries: vec![
            storage_layout(0, OUTPUT_BYTES, false),
            uniform_layout(1),
            storage_layout(2, GI_CANDIDATE_BYTES, true),
            storage_layout(3, SURFACE_BYTES, true),
        ],
    });
    let gi_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("lighting-gi-pipeline-layout".into()),
        bind_group_layouts: vec![gi_layout],
        push_constant_ranges: Vec::new(),
    });
    let gi_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("lighting-gi-compute".into()),
        layout: Some(gi_pipeline_layout),
        module: gi_shader,
        entry_point: "main".into(),
    });
    let gi_bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("lighting-gi-bind-group".into()),
        layout: Some(gi_layout),
        entries: vec![
            buffer_binding(0, gi_output, OUTPUT_BYTES),
            buffer_binding(1, params, PARAM_BYTES),
            buffer_binding(2, candidates, GI_CANDIDATE_BYTES),
            buffer_binding(3, surfaces, SURFACE_BYTES),
        ],
    });

    let composite_shader =
        device.create_shader_module(&ShaderModuleDescriptor::wgsl(LIGHTING_COMPOSITE_WGSL));
    let composite_static_layout =
        device.create_bind_group_layout(&composite_static_layout_descriptor());
    let composite_target_layout =
        device.create_bind_group_layout(&composite_target_layout_descriptor());
    let composite_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("lighting-composite-pipeline-layout".into()),
        bind_group_layouts: vec![composite_static_layout, composite_target_layout],
        push_constant_ranges: Vec::new(),
    });
    let composite_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("lighting-composite-compute".into()),
        layout: Some(composite_pipeline_layout),
        module: composite_shader,
        entry_point: "main".into(),
    });
    let composite_bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("lighting-composite-static-bind-group".into()),
        layout: Some(composite_static_layout),
        entries: vec![
            buffer_binding(0, direct_output, OUTPUT_BYTES),
            buffer_binding(1, gi_output, OUTPUT_BYTES),
            buffer_binding(2, params, PARAM_BYTES),
        ],
    });

    LightingGpuBundle {
        output,
        direct_output,
        gi_output,
        params,
        directionals,
        punctuals,
        areas,
        candidates,
        surfaces,
        radiance,
        direct_pipeline,
        direct_bind_group,
        gi_pipeline,
        gi_bind_group,
        composite_pipeline,
        composite_static_layout,
        composite_target_layout,
        composite_bind_group,
    }
}
