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
    BufferDescriptor, BufferId, BufferUsages, ComputePipelineDescriptor, FrameSync,
    PipelineLayoutDescriptor, RenderDevice, RenderQueue, ShaderModuleDescriptor, ShaderModuleId,
    ShaderSource, ShaderStages, StorageTextureAccess, TextureFormat, TextureSampleType,
    TextureViewDimension, TextureViewId,
};
use prism_render_runtime::{RenderRuntime, ReplayHandle};
use prism_render_shading::{LobeSampleBudget, LobeSamplingPolicy, ShadingQuality};
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use crate::lighting::{
    ExtractedLightingInputs, LightingAreaInput, LightingDirectionalInput, LightingGiCandidateInput,
    LightingPunctualInput, LightingSurfaceInput, NativeStylizedParams,
};
use crate::lighting_trace_gpu::TracePreparationError;
use crate::target::RenderTarget;

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
pub(crate) const INVALID_LIGHTING_SURFACE_INDEX: u32 = u32::MAX;
const WORKGROUP_SIZE: u32 = 64;
/// X dimension of the per-target screen-space composite workgroup.
pub(crate) const COMPOSITE_WORKGROUP_X: u32 = 8;
/// Y dimension of the per-target screen-space composite workgroup.
pub(crate) const COMPOSITE_WORKGROUP_Y: u32 = 8;
/// The required formats for world-position components.
pub(crate) const COMPOSITE_POSITION_X_FORMAT: TextureFormat = TextureFormat::R32Float;
pub(crate) const COMPOSITE_POSITION_Y_FORMAT: TextureFormat = TextureFormat::R32Float;
pub(crate) const COMPOSITE_POSITION_Z_FORMAT: TextureFormat = TextureFormat::R32Float;
/// Binary16 world normal in RGB; alpha is binary opaque coverage (clear to 0,
/// raster writes 1), not material opacity or a world-position component.
pub(crate) const COMPOSITE_NORMAL_FORMAT: TextureFormat = TextureFormat::Rgba16Float;
/// Linear shader albedo is encoded to sRGB8 on attachment writes and decoded
/// back to linear by textureLoad in the composite. RGB is clamped to [0, 1]
/// and quantized, not binary16/HDR-preserving; alpha is linear UNORM8 and is
/// independent of coverage. This stores 4 bytes/texel but costs 8 WebGPU
/// color-attachment bytes/sample, just like Rgba16Float.
pub(crate) const COMPOSITE_ALBEDO_FORMAT: TextureFormat = TextureFormat::Rgba8UnormSrgb;
/// The required format for the surface-id GBuffer view.
pub(crate) const COMPOSITE_SURFACE_ID_FORMAT: TextureFormat = TextureFormat::R32Uint;
/// The canonical per-pixel UV/footprint/tangent-angle surface input.
pub(crate) const COMPOSITE_SURFACE_DATA_FORMAT: TextureFormat = TextureFormat::Rgba16Float;
/// The scene-linear HDR format written by the composite.
pub(crate) const COMPOSITE_HDR_FORMAT: TextureFormat = TextureFormat::Rgba16Float;
pub(crate) const OUTPUT_BYTES: u64 =
    (MAX_LIGHTING_SURFACES * core::mem::size_of::<[f32; 4]>()) as u64;
const DIRECTIONAL_BYTES: u64 =
    (MAX_DIRECTIONALS * core::mem::size_of::<LightingDirectionalInput>()) as u64;
const PUNCTUAL_BYTES: u64 = (MAX_PUNCTUALS * core::mem::size_of::<LightingPunctualInput>()) as u64;
const AREA_BYTES: u64 = (MAX_AREAS * core::mem::size_of::<LightingAreaInput>()) as u64;
pub(crate) const CAMERA_VIEW_BYTES: u64 = core::mem::size_of::<GpuCameraView>() as u64;
pub(crate) const LOBE_SAMPLE_BUDGET_BYTES: u64 = 8 * core::mem::size_of::<u32>() as u64;
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

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuMaterialLightingSurface {
    material_index: u32,
    lighting_surface_index: u32,
}

pub(crate) const MATERIAL_LIGHTING_SURFACE_BYTES: u64 =
    (MAX_LIGHTING_SURFACES * core::mem::size_of::<GpuMaterialLightingSurface>()) as u64;

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
    stylized: NativeStylizedParams,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuCameraView {
    position: [f32; 3],
    _padding: f32,
}

/// Native composite copy of the shading-quality budget. The field order is a
/// stable eight-word ABI shared with `native/lobe_budget.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Pod, Zeroable)]
struct GpuLobeSampleBudget {
    sss_separable_taps: u32,
    sss_preintegration_steps: u32,
    translucency_profile_samples: u32,
    multiscatter_dir_samples: u32,
    multiscatter_march_samples: u32,
    total_samples: u32,
    revision: u32,
    max_full_lobes: u32,
}

impl GpuLobeSampleBudget {
    const fn for_quality(quality: ShadingQuality) -> Self {
        let budget = quality.budget();
        Self::from_budget(quality, budget, 0)
    }

    const fn from_budget(quality: ShadingQuality, budget: LobeSampleBudget, revision: u32) -> Self {
        let budget = budget.clamped();
        Self {
            sss_separable_taps: budget.sss_separable_taps,
            sss_preintegration_steps: budget.sss_preintegration_steps,
            translucency_profile_samples: budget.translucency_profile_samples,
            multiscatter_dir_samples: budget.multiscatter_dir_samples,
            multiscatter_march_samples: budget.multiscatter_march_samples,
            total_samples: budget.total_samples(),
            revision,
            max_full_lobes: LobeSamplingPolicy::for_budget(quality, budget).max_full_lobes(),
        }
    }
}

const PARAM_BYTES: u64 = core::mem::size_of::<GpuLightingParams>() as u64;
const MATERIAL_HEADER_BYTES: u64 =
    core::mem::size_of::<prism_render_material::GpuMaterialHeader>() as u64;
const MATERIAL_PLAN_ROW_BYTES: u64 =
    (prism_render_material::EXECUTION_PLAN_ROW_WORDS * core::mem::size_of::<u32>()) as u64;
const MATERIAL_PROGRAM_ROW_BYTES: u64 = (5 * core::mem::size_of::<u32>()) as u64;
const MATERIAL_PARAMETER_WORD_BYTES: u64 = core::mem::size_of::<u32>() as u64;
const ZERO_OUTPUT: [u8; OUTPUT_BYTES as usize] = [0; OUTPUT_BYTES as usize];

const DIRECT_LIGHTING_COMMON_WGSL: &str = r#"
struct Params {
    counts: vec4<u32>,
    flags: vec4<u32>,
    stylized: NativeStylizedParams,
};

struct NativeStylizedParams {
    bands: u32,
    wrap: f32,
    ramp_softness: f32,
    shadow_threshold: f32,
    shadow_softness: f32,
    specular_intensity: f32,
    specular_threshold: f32,
    specular_softness: f32,
    specular_color: vec3<f32>,
    rim_intensity: f32,
    rim_color: vec3<f32>,
    rim_power: f32,
};

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

@group(0) @binding(0)
var<uniform> params: Params;
@group(0) @binding(1)
var<storage, read> directionals: array<DirectionalLight, 1024>;
@group(0) @binding(2)
var<storage, read> punctuals: array<PunctualLight, 1024>;
@group(0) @binding(3)
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
// WESL's EscapeMangler keeps imported BRDF types module-qualified. These
// aliases bridge the raw runtime entry code to that linked, canonical ABI.
alias SurfaceSample = _2prism_render_app_shaders_brdf_SurfaceSample;
alias ShadingFrame = _2prism_render_app_shaders_brdf_ShadingFrame;
alias DirectLightSample = _2prism_render_app_shaders_brdf_DirectLightSample;
alias PrincipledPre = _2prism_render_app_shaders_brdf_PrincipledPre;
alias PrismSurfaceParameters = _2prism_render_material_shaders_material_PrismSurfaceParameters;
alias PrismMaterialHeader = _2prism_render_material_shaders_material_PrismMaterialHeader;

struct CameraView {
    position: vec4<f32>,
};

@group(0) @binding(5)
var<uniform> camera_view: CameraView;

@group(0) @binding(4)
var<storage, read> gi: array<vec4<f32>>;

@group(1) @binding(0)
var position_x_gbuffer: texture_2d<f32>;
@group(1) @binding(1)
var position_y_gbuffer: texture_2d<f32>;
@group(1) @binding(2)
var position_z_gbuffer: texture_2d<f32>;
@group(1) @binding(3)
var normal_gbuffer: texture_2d<f32>;
@group(1) @binding(4)
var albedo_gbuffer: texture_2d<f32>;
@group(1) @binding(5)
var surface_id_gbuffer: texture_2d<u32>;
@group(1) @binding(6)
var<storage, read> surface_identity: array<SurfaceIdentity>;
@group(1) @binding(7)
var hdr_target: texture_storage_2d<rgba16float, write>;
@group(1) @binding(8)
var<storage, read_write> response_outputs: array<CompositeResponseOutput>;
@group(1) @binding(9)
var surface_data_gbuffer: texture_2d<f32>;
@group(1) @binding(10)
var sampled_material_a_gbuffer: texture_2d<f32>;
@group(1) @binding(11)
var sampled_material_b_gbuffer: texture_2d<f32>;
@group(1) @binding(12)
var sampled_material_c_gbuffer: texture_2d<u32>;

fn composite_unpack_material_c(encoded: u32) -> vec4<f32> {
    return vec4<f32>(
        f32(encoded & 0xffu) / 255.0,
        f32((encoded >> 8u) & 0xffu) / 255.0,
        f32((encoded >> 16u) & 0xffu) / 255.0,
        f32((encoded >> 24u) & 0xffu) / 255.0,
    );
}

fn composite_decode_oct_normal(encoded: vec2<f32>) -> vec3<f32> {
    let p = encoded * 2.0 - vec2<f32>(1.0);
    var n = vec3<f32>(p.x, p.y, 1.0 - abs(p.x) - abs(p.y));
    if (n.z < 0.0) {
        let reflected = (1.0 - abs(n.yx))
            * vec2<f32>(select(-1.0, 1.0, n.x >= 0.0), select(-1.0, 1.0, n.y >= 0.0));
        n.x = reflected.x;
        n.y = reflected.y;
    }
    return normalize_or(n, vec3<f32>(0.0, 1.0, 0.0));
}

fn composite_basis_tangent(normal_input: vec3<f32>) -> vec3<f32> {
    let normal = normalize_or(normal_input, vec3<f32>(0.0, 1.0, 0.0));
    let reference = select(
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(1.0, 0.0, 0.0),
        abs(normal.y) > 0.9,
    );
    return normalize_or(cross(reference, normal), vec3<f32>(1.0, 0.0, 0.0));
}

fn composite_basis_bitangent(normal_input: vec3<f32>, tangent: vec3<f32>) -> vec3<f32> {
    let normal = normalize_or(normal_input, vec3<f32>(0.0, 1.0, 0.0));
    return normalize_or(cross(normal, tangent), vec3<f32>(0.0, 0.0, 1.0));
}

fn composite_area_source_cosine(light: AreaLight, point: vec3<f32>, direction: vec3<f32>) -> f32 {
    if (light.shape_kind.x == 0u || light.shape_kind.x == 1u) {
        return max(
            dot(
                normalize_or(light.vector_a.xyz, vec3<f32>(0.0, -1.0, 0.0)),
                -direction,
            ),
            0.0,
        );
    }
    let axis = normalize_or(light.vector_a.xyz - light.origin.xyz, vec3<f32>(0.0, 1.0, 0.0));
    let center = mix(light.origin.xyz, light.vector_a.xyz, 0.5);
    let radial = normalize_or(point - center, basis_tangent(axis));
    return max(dot(radial, -direction), 0.0);
}

struct CompositeMaskValue {
    valid: bool,
    value: f32,
}

fn composite_mask_value(identity: SurfaceIdentity) -> CompositeMaskValue {
    if (identity.right_program_index == 0xffffffffu) {
        return CompositeMaskValue(true, 0.0);
    }
    // The material helper is linked separately from this raw WGSL module, so
    // keep its generated header type out of the public composite signature.
    let header = composite_material_headers[identity.material_index];
    // Dynamic channels are the only mask source with a complete Native
    // per-pixel producer in this target. Other sources must remain explicit
    // failures until their UV/vertex/texture/region payload is part of the
    // SurfaceData contract; they must never silently reuse alpha or albedo.
    if (identity.mask_kind != 5u || identity.mask_index > 3u) {
        return CompositeMaskValue(false, 0.0);
    }
    var raw = 0.0;
    if (identity.mask_index == 0u) {
        raw = bitcast<f32>(header.dynamic_wetness);
    } else if (identity.mask_index == 1u) {
        raw = bitcast<f32>(header.dynamic_snow);
    } else if (identity.mask_index == 2u) {
        raw = bitcast<f32>(header.dynamic_damage);
    } else {
        raw = bitcast<f32>(header.dynamic_dissolve);
    }
    if (!finite(raw) || raw < 0.0 || raw > 1.0) {
        return CompositeMaskValue(false, 0.0);
    }
    return CompositeMaskValue(true, raw);
}

fn composite_lerp(left: f32, right: f32, mask: f32) -> f32 {
    if (mask == 0.0) { return left; }
    if (mask == 1.0) { return right; }
    return left + (right - left) * mask;
}

fn composite_lerp3(left: vec3<f32>, right: vec3<f32>, mask: f32) -> vec3<f32> {
    if (mask == 0.0) { return left; }
    if (mask == 1.0) { return right; }
    return left + (right - left) * mask;
}

fn composite_blend_response(
    left: CompositeResponseOutput,
    right: CompositeResponseOutput,
    identity: SurfaceIdentity,
    mask: f32,
    alpha: f32,
) -> CompositeResponseOutput {
    var result = left;
    let channels = identity.blended_channels;
    if ((channels & 1u) != 0u) {
        let left_direct = vec3<f32>(left.words[0], left.words[1], left.words[2]);
        let right_direct = vec3<f32>(right.words[0], right.words[1], right.words[2]);
        let direct = composite_lerp3(left_direct, right_direct, mask);
        result.words[0] = direct.x;
        result.words[1] = direct.y;
        result.words[2] = direct.z;
    }
    if ((channels & 2u) != 0u) {
        result.words[3] = composite_lerp(left.words[3], right.words[3], mask);
        result.words[4] = composite_lerp(left.words[4], right.words[4], mask);
        result.words[5] = composite_lerp(left.words[5], right.words[5], mask);
    }
    if ((channels & 4u) != 0u) {
        result.words[6] = composite_lerp(left.words[6], right.words[6], mask);
        result.words[7] = composite_lerp(left.words[7], right.words[7], mask);
        result.words[8] = composite_lerp(left.words[8], right.words[8], mask);
    }
    if ((channels & 8u) != 0u) {
        result.words[9] = composite_lerp(left.words[9], right.words[9], mask);
        result.words[10] = composite_lerp(left.words[10], right.words[10], mask);
        result.words[11] = composite_lerp(left.words[11], right.words[11], mask);
    }
    if ((channels & 16u) != 0u) {
        result.words[12] = composite_lerp(left.words[12], right.words[12], mask);
        result.words[13] = composite_lerp(left.words[13], right.words[13], mask);
        result.words[14] = composite_lerp(left.words[14], right.words[14], mask);
    }
    // Native opaque coverage is the shared alpha contract. It is not part of
    // the radiance channel mask and therefore never lerps from authored alpha.
    result.words[15] = alpha;
    return result;
}

struct CompositeProgramEvaluation {
    valid: bool,
    response: CompositeResponseOutput,
}

fn composite_evaluate_program(
    program_kind: u32,
    program_id: u32,
    surface: SurfaceSample,
    frame: ShadingFrame,
    world_position: vec3<f32>,
    identity: SurfaceIdentity,
    npr_modifiers: vec3<f32>,
    parameters: PrismSurfaceParameters,
    uv: vec2<f32>,
    alpha: f32,
) -> CompositeProgramEvaluation {
    var result = CompositeProgramEvaluation(false, composite_response_zero());
    if (program_kind == 1u) {
        var direct = vec3<f32>(0.0);
        if (params.flags.x == 1u || params.flags.x == 3u) {
            let directional_count = min(params.counts.y, 1024u);
            for (var index = 0u; index < directional_count; index += 1u) {
                let light = directionals[index];
                let sample = composite_make_direct_light(
                    normalize_or(light.direction_to_light, frame.normal),
                    light.illuminance,
                    clamp(light.visibility, 0.0, 1.0),
                );
                let channels = composite_evaluate_builtin_npr(surface, frame, sample, npr_modifiers);
                if (!channels.valid) { return result; }
                direct = direct + channels.direct;
            }

            let punctual_count = min(params.counts.z, 1024u);
            for (var index = 0u; index < punctual_count; index += 1u) {
                let light = punctuals[index];
                let to_light = light.position_range.xyz - world_position;
                let distance_squared = dot(to_light, to_light);
                if (finite(distance_squared) && distance_squared > 1.0e-8) {
                    let direction = to_light * inverseSqrt(distance_squared);
                    let attenuation = punctual_attenuation(light, world_position);
                    let sample = composite_make_direct_light(
                        direction,
                        light.intensity_spot_scale.xyz * attenuation,
                        1.0,
                    );
                    let channels = composite_evaluate_builtin_npr(surface, frame, sample, npr_modifiers);
                    if (!channels.valid) { return result; }
                    direct = direct + channels.direct;
                }
            }

            let area_count = min(params.flags.y, 1024u);
            for (var index = 0u; index < area_count; index += 1u) {
                let light = areas[index];
                let measure = area_measure(light);
                if (!finite(measure) || measure <= 0.0) { continue; }
                for (var sample_index = 0u; sample_index < 4u; sample_index += 1u) {
                    let point = area_sample(light, sample_index);
                    let to_light = point - world_position;
                    let distance_squared = dot(to_light, to_light);
                    if (!finite(distance_squared) || distance_squared <= 1.0e-8) { continue; }
                    let direction = to_light * inverseSqrt(distance_squared);
                    let source_cosine = composite_area_source_cosine(light, point, direction);
                    let illuminance = light.radiance.xyz
                        * (measure * 0.25 * source_cosine / distance_squared);
                    let sample = composite_make_direct_light(
                        direction,
                        illuminance,
                        clamp(f32(light.shape_kind.y), 0.0, 1.0),
                    );
                    let channels = composite_evaluate_builtin_npr(surface, frame, sample, npr_modifiers);
                    if (!channels.valid) { return result; }
                    direct = direct + channels.direct;
                }
            }
        }
        result.response = composite_response_from_channels(
            direct,
            vec3<f32>(0.0),
            vec3<f32>(0.0),
            vec3<f32>(0.0),
            surface.emissive,
            alpha,
        );
        result.valid = true;
        return result;
    }
    if (program_kind == 2u) {
        let channels = composite_evaluate_builtin_unlit(surface);
        if (!channels.valid) { return result; }
        result.response = composite_response_from_channels(
            vec3<f32>(0.0),
            vec3<f32>(0.0),
            vec3<f32>(0.0),
            vec3<f32>(0.0),
            channels.radiance,
            alpha,
        );
        result.valid = true;
        return result;
    }
    if (program_kind == 3u) {
        return composite_evaluate_native_registered(
            surface,
            frame,
            world_position,
            identity,
            program_id,
            parameters,
            uv,
            alpha,
        );
    }
    if (program_kind != 0u) {
        return result;
    }

    let normal = frame.normal;
    let pre = composite_principled_pre(surface, frame);
    let emission = surface.emissive;
    var direct = vec3<f32>(0.0);
    var transmission = vec3<f32>(0.0);
    var indirect_diffuse = vec3<f32>(0.0);

    if (params.flags.x == 1u || params.flags.x == 3u) {
        let directional_count = min(params.counts.y, 1024u);
        for (var index = 0u; index < directional_count; index += 1u) {
            let light = directionals[index];
            let sample = composite_make_direct_light(
                normalize_or(light.direction_to_light, normal),
                light.illuminance,
                clamp(light.visibility, 0.0, 1.0),
            );
            let channels = composite_evaluate_builtin_direct(surface, frame, sample, pre);
            if (!channels.valid) { return result; }
            direct = direct + channels.direct;
            transmission = transmission + channels.transmission;
        }

        let punctual_count = min(params.counts.z, 1024u);
        for (var index = 0u; index < punctual_count; index += 1u) {
            let light = punctuals[index];
            let to_light = light.position_range.xyz - world_position;
            let distance_squared = dot(to_light, to_light);
            if (finite(distance_squared) && distance_squared > 1.0e-8) {
                let direction = to_light * inverseSqrt(distance_squared);
                let attenuation = punctual_attenuation(light, world_position);
                let sample = composite_make_direct_light(
                    direction,
                    light.intensity_spot_scale.xyz * attenuation,
                    1.0,
                );
                let channels = composite_evaluate_builtin_direct(surface, frame, sample, pre);
                if (!channels.valid) { return result; }
                direct = direct + channels.direct;
                transmission = transmission + channels.transmission;
            }
        }

        let area_count = min(params.flags.y, 1024u);
        for (var index = 0u; index < area_count; index += 1u) {
            let light = areas[index];
            let measure = area_measure(light);
            if (!finite(measure) || measure <= 0.0) { continue; }
            for (var sample_index = 0u; sample_index < 4u; sample_index += 1u) {
                let point = area_sample(light, sample_index);
                let to_light = point - world_position;
                let distance_squared = dot(to_light, to_light);
                if (!finite(distance_squared) || distance_squared <= 1.0e-8) { continue; }
                let direction = to_light * inverseSqrt(distance_squared);
                let source_cosine = composite_area_source_cosine(light, point, direction);
                let illuminance = light.radiance.xyz
                    * (measure * 0.25 * source_cosine / distance_squared);
                let sample = composite_make_direct_light(
                    direction,
                    illuminance,
                    clamp(f32(light.shape_kind.y), 0.0, 1.0),
                );
                let channels = composite_evaluate_builtin_direct(surface, frame, sample, pre);
                if (!channels.valid) { return result; }
                direct = direct + channels.direct;
                transmission = transmission + channels.transmission;
            }
        }
    }

    if ((params.flags.x == 2u || params.flags.x == 3u) &&
        identity.lighting_surface_index != 0xffffffffu &&
        identity.lighting_surface_index < params.counts.x &&
        identity.lighting_surface_index < arrayLength(&gi)) {
        // The GI producer already evaluates its diffuse transport term; keep
        // it in the indirect channel so composition never applies the base
        // weight/metalness scale twice.
        indirect_diffuse = gi[identity.lighting_surface_index].xyz;
    }
    result.response = composite_response_from_channels(
        direct,
        indirect_diffuse,
        vec3<f32>(0.0),
        transmission,
        emission,
        alpha,
    );
    result.valid = true;
    return result;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let extent = textureDimensions(hdr_target);
    if (id.x >= extent.x || id.y >= extent.y) {
        return;
    }

    let pixel = vec2<i32>(id.xy);
    let position = vec3<f32>(
        textureLoad(position_x_gbuffer, pixel, 0).x,
        textureLoad(position_y_gbuffer, pixel, 0).x,
        textureLoad(position_z_gbuffer, pixel, 0).x,
    );
    let surface_id = textureLoad(surface_id_gbuffer, pixel, 0).x;
    let identity_index = id.y * extent.x + id.x;
    response_outputs[identity_index] = composite_response_zero();
    let normal_sample = textureLoad(normal_gbuffer, pixel, 0);
    // Material slot zero is valid, so reject the explicitly cleared coverage
    // plane before consulting the slot.
    if (normal_sample.a <= 0.0 || surface_id == 0xffffffffu ||
        identity_index >= arrayLength(&surface_identity)) {
        textureStore(hdr_target, pixel, vec4<f32>(0.0));
        return;
    }
    let identity_words = surface_identity[identity_index];
    let identity = identity_words;
    if (identity.material_index != surface_id || identity.material_index == 0xffffffffu) {
        textureStore(hdr_target, pixel, vec4<f32>(0.0));
        return;
    }
    let material = prism_composite_material_surface(identity);
    if (!material.valid) {
        textureStore(hdr_target, pixel, vec4<f32>(0.0));
        return;
    }

    let world_position = position;
    let normal_input = normal_sample.xyz;
    let sampled_base_color = textureLoad(albedo_gbuffer, pixel, 0).xyz;
    let surface_data = textureLoad(surface_data_gbuffer, pixel, 0);
    let sampled_material_a = textureLoad(sampled_material_a_gbuffer, pixel, 0);
    let sampled_material_b = textureLoad(sampled_material_b_gbuffer, pixel, 0);
    let sampled_material_c = composite_unpack_material_c(
        textureLoad(sampled_material_c_gbuffer, pixel, 0).x,
    );
    if (!finite(material.surface.base_weight) || !finite(material.surface.base_metalness) ||
        !finite(material.surface.transmission_weight) || !finite(material.surface.emission_luminance) ||
        !finite(camera_view.position.x) || !finite(camera_view.position.y) ||
        !finite(camera_view.position.z)) {
        textureStore(hdr_target, pixel, vec4<f32>(0.0));
        return;
    }
    let normal_length_squared = dot(normal_input, normal_input);
    if (!finite(world_position.x) || !finite(world_position.y) || !finite(world_position.z) ||
        !finite(normal_input.x) || !finite(normal_input.y) || !finite(normal_input.z) ||
        !finite(normal_length_squared) || normal_length_squared <= 1.0e-12 ||
        !finite(sampled_base_color.x) || !finite(sampled_base_color.y) ||
        !finite(sampled_base_color.z) || !finite(surface_data.x) ||
        !finite(surface_data.y) || !finite(surface_data.z) ||
        !finite(surface_data.w) || surface_data.w <= 0.0 ||
        !finite(sampled_material_a.x) || !finite(sampled_material_a.y) ||
        !finite(sampled_material_a.z) || !finite(sampled_material_a.w) ||
        !finite(sampled_material_b.x) || !finite(sampled_material_b.y) ||
        !finite(sampled_material_b.z) || !finite(sampled_material_b.w) ||
        !finite(sampled_material_c.x) || !finite(sampled_material_c.y) ||
        !finite(sampled_material_c.z) || !finite(sampled_material_c.w) ||
        sampled_material_c.w <= 0.0) {
        textureStore(hdr_target, pixel, vec4<f32>(0.0));
        return;
    }
    let normal = normalize_or(normal_input, vec3<f32>(0.0, 1.0, 0.0));

    // surface_encode already folded the active base-color map/portable VT into
    // this texel. Replace only the authored base color in the canonical sample;
    // all other OpenPBR parameters still come from the validated material row.
    var surface = composite_surface_sample(material.surface);
    surface.base_color = sampled_base_color;
    surface.emissive = sampled_material_a.xyz * material.surface.emission_luminance;
    surface.metallic = sampled_material_a.w;
    surface.perceptual_roughness = sampled_material_b.x;
    surface.ambient_occlusion = sampled_material_b.y;
    surface.clearcoat = sampled_material_b.z;
    surface.normal_variance = sampled_material_b.w;
    surface.clearcoat_roughness = sampled_material_c.z;
    surface.coat_normal = composite_decode_oct_normal(sampled_material_c.xy);
    let tangent_reference = composite_basis_tangent(normal);
    let tangent_bitangent = composite_basis_bitangent(normal, tangent_reference);
    let tangent_angle = surface_data.z;
    let tangent = normalize_or(
        tangent_reference * cos(tangent_angle) + tangent_bitangent * sin(tangent_angle),
        tangent_reference,
    );
    let bitangent = composite_basis_bitangent(normal, tangent);
    let view = normalize_or(camera_view.position.xyz - world_position, normal);
    let frame = composite_make_shading_frame(normal, view, tangent, bitangent);
    // These are canonical shade-time inputs. Keep them in the resolved
    // surface, even when the current BRDF permutation does not use glints;
    // invalid UV/footprint data is rejected above instead of replaced.
    surface.glint_cell = surface_data.xy;
    surface.glint_footprint_area = surface_data.w * surface_data.w;
    let left_evaluation = composite_evaluate_program(
        identity.left_program_kind,
        identity.left_program_id,
        surface,
        frame,
        world_position,
        identity,
        material.npr_modifiers,
        material.surface,
        surface_data.xy,
        normal_sample.a,
    );
    if (!left_evaluation.valid) {
        textureStore(hdr_target, pixel, vec4<f32>(0.0));
        return;
    }

    var response = left_evaluation.response;
    if (identity.right_program_index != 0xffffffffu) {
        let header = composite_material_headers[identity.material_index];
        let mask = composite_mask_value(identity);
        if (!mask.valid) {
            textureStore(hdr_target, pixel, vec4<f32>(0.0));
            return;
        }
        let right_evaluation = composite_evaluate_program(
            identity.right_program_kind,
            identity.right_program_id,
            surface,
            frame,
            world_position,
            identity,
            material.npr_modifiers,
            material.surface,
            surface_data.xy,
            normal_sample.a,
        );
        if (!right_evaluation.valid) {
            textureStore(hdr_target, pixel, vec4<f32>(0.0));
            return;
        }
        response = composite_blend_response(
            left_evaluation.response,
            right_evaluation.response,
            identity,
            mask.value,
            normal_sample.a,
        );
    }

    if (!composite_response_is_finite(response)) {
        textureStore(hdr_target, pixel, vec4<f32>(0.0));
        return;
    }
    let value = composite_response_radiance(response);
    if (!finite(value.x) || !finite(value.y) || !finite(value.z)) {
        textureStore(hdr_target, pixel, vec4<f32>(0.0));
        return;
    }
    response_outputs[identity_index] = response;
    textureStore(hdr_target, pixel, vec4<f32>(value, response.words[15]));
}
"#;

/// Builds the complete composite module for one execution-table generation.
/// The registered link is finite and already validated by the material crate;
/// it is inserted into the same fixed resource ABI as builtin responses.
fn build_composite_shader_source(response_link: Option<&str>) -> String {
    let generated_material = include_str!(concat!(env!("OUT_DIR"), "/composite_material.wgsl"));
    match response_link {
        Some(response_link) if !response_link.trim().is_empty() => format!(
            "{DIRECT_LIGHTING_COMMON_WGSL}\n{generated_material}\n{response_link}\n{}\n{}\n{LIGHTING_COMPOSITE_WGSL}",
            NATIVE_REGISTERED_HELPERS_WGSL,
            NATIVE_REGISTERED_ADAPTER_WGSL,
        ),
        _ => format!(
            "{DIRECT_LIGHTING_COMMON_WGSL}\n{generated_material}\n{}\n{LIGHTING_COMPOSITE_WGSL}",
            NATIVE_REGISTERED_UNAVAILABLE_WGSL,
        ),
    }
}

/// The builtin pipeline must still link when no registered execution table is
/// published. It rejects kind 3 instead of silently evaluating a different
/// response; the material-aware path replaces this module with the validated
/// registered link before dispatch.
const NATIVE_REGISTERED_UNAVAILABLE_WGSL: &str = r#"
fn composite_evaluate_native_registered(
    surface: SurfaceSample,
    frame: ShadingFrame,
    world_position: vec3<f32>,
    identity: SurfaceIdentity,
    program_id: u32,
    parameters: PrismSurfaceParameters,
    uv: vec2<f32>,
    alpha: f32,
) -> CompositeProgramEvaluation {
    return CompositeProgramEvaluation(false, composite_response_zero());
}
"#;

/// Appends the validated registered-response dispatch to the fixed Native
/// composite. Registered bodies are function-scoped and receive only the
/// canonical response ABI plus the Native read-only lighting helpers below.
fn native_registered_shader_source(response_link: &str) -> String {
    build_composite_shader_source(Some(response_link))
}

const NATIVE_REGISTERED_HELPERS_WGSL: &str = r#"
fn registered_light_count(lighting: RegisteredLightingContext) -> u32 {
    return lighting.directional_count + lighting.punctual_count;
}

fn registered_light(lighting: RegisteredLightingContext, index: u32) -> RegisteredLight {
    var invalid: RegisteredLight;
    if (index >= registered_light_count(lighting)) { return invalid; }
    if (index < lighting.directional_count) {
        let light = directionals[index];
        return RegisteredLight(
            normalize_or(light.direction_to_light, lighting.frame.normal),
            light.illuminance,
            clamp(light.visibility, 0.0, 1.0),
            1.0,
        );
    }
    let punctual_index = index - lighting.directional_count;
    let light = punctuals[punctual_index];
    let to_light = light.position_range.xyz - lighting.world_position;
    let distance_squared = dot(to_light, to_light);
    if (!finite(distance_squared) || distance_squared <= 1.0e-8) { return invalid; }
    let direction = to_light * inverseSqrt(distance_squared);
    return RegisteredLight(
        direction,
        light.intensity_spot_scale.xyz * punctual_attenuation(light, lighting.world_position),
        1.0,
        1.0,
    );
}

fn registered_area(lighting: RegisteredLightingContext, index: u32) -> RegisteredArea {
    var invalid: RegisteredArea;
    if (index >= lighting.area_count) { return invalid; }
    let light = areas[index];
    return RegisteredArea(
        light.origin.x, light.origin.y, light.origin.z, light.shape.w,
        light.radiance.x, light.radiance.y, light.radiance.z, light.radiance.w,
        light.vector_b.x, light.vector_b.y, light.vector_b.z, light.shape.x,
        light.vector_c.x, light.vector_c.y, light.vector_c.z, light.shape.y,
        light.shape_kind.x, light.shape_kind.y, 0.0, 0u,
    );
}

fn registered_custom_word(custom: RegisteredCustomData, index: u32) -> u32 {
    let header = custom.header;
    let base = header.parameter_offset;
    if (index >= header.parameter_size / 4u
        || base >= arrayLength(&composite_material_parameters)
        || index >= arrayLength(&composite_material_parameters) - base) {
        return 0u;
    }
    return composite_material_parameters[base + index];
}
"#;

const NATIVE_REGISTERED_ADAPTER_WGSL: &str = r#"
fn composite_registered_surface(surface: SurfaceSample) -> RegisteredSurface {
    return RegisteredSurface(
        surface.base_color, surface.base_weight, surface.diffuse_roughness,
        surface.metallic, surface.perceptual_roughness, surface.specular_weight,
        surface.specular_color, surface.specular_ior, surface.transmission,
        surface.index_of_refraction, surface.dispersion, surface.thin_walled,
        surface.transmission_color, surface.transmission_scatter,
        surface.transmission_scatter_anisotropy,
        surface.transmission_dispersion_abbe_number, surface.coat_darkening,
        surface.scatter_anisotropy, surface.ambient_occlusion, surface.emissive,
        surface.clearcoat, surface.clearcoat_roughness, surface.coat_ior,
        surface.coat_color, surface.coat_normal, surface.coat_roughness_anisotropy,
        surface.coat_tangent, surface.sheen, surface.fuzz_roughness,
        surface.fuzz_color, surface.subsurface, surface.subsurface_color,
        surface.subsurface_radius, surface.subsurface_radius_scale,
        surface.thickness, surface.anisotropy, surface.anisotropy_rotation,
        surface.normal_variance, surface.iridescence,
        surface.iridescence_thickness_nm, surface.iridescence_ior,
        surface.glint_weight, surface.glint_roughness, surface.glint_density,
        surface.glint_bin_half_width, surface.glint_state, surface.glint_cell,
        surface.glint_footprint_area,
    );
}

fn composite_registered_parameters(p: PrismSurfaceParameters) -> RegisteredParameters {
    return RegisteredParameters(
        p.base_color, p.emission_color, p.emission_luminance, p.base_metalness,
        p.specular_roughness, p.specular_weight, p.specular_color,
        p.specular_ior, p.thin_walled, p.geometry_opacity, p.ambient_occlusion,
        p.normal_scale, p.alpha_cutoff, p.base_weight, p.base_diffuse_roughness,
        p.transmission_weight, p.thickness, p.coat_weight, p.coat_roughness,
        p.coat_roughness_anisotropy, p.coat_ior, p.coat_color,
        p.anisotropy, p.anisotropy_rotation, p.fuzz_weight, p.fuzz_roughness,
        p.fuzz_color, p.subsurface_weight, p.subsurface_color,
        p.subsurface_radius, p.subsurface_radius_scale, p.index_of_refraction,
        p.transmission_dispersion_scale, p.transmission_color,
        p.transmission_scatter, p.thin_film_weight, p.thin_film_thickness,
        p.thin_film_ior, p.transmission_scatter_anisotropy,
        p.transmission_dispersion_abbe_number, p.coat_darkening,
        p.subsurface_scatter_anisotropy, p.npr_face_softness,
        p.npr_stylized_ramp_softness, p.npr_stylized_specular_scale,
        p.npr_stylized_rim_scale, p.glint_weight, p.glint_roughness,
        p.glint_density, p.glint_bin_half_width, p.measured_weight,
        p.measured_phi_d, p.measured_scale,
    );
}

fn composite_registered_header(header: PrismMaterialHeader) -> RegisteredHeader {
    return RegisteredHeader(
        header.generation, header.revision, header.render_class, header.render_domain,
        header.feature_flags, header.closure_mask, header.parameter_offset,
        header.parameter_size, header.texture_offset, header.texture_count,
        header.sampler_offset, header.sampler_count, header.is_active,
        header.material_epoch_low, header.material_epoch_high, header.lobe_mask,
        header.extensions_used_low, header.extensions_used_high,
        header.extensions_required_low, header.extensions_required_high,
        header.dynamic_wetness, header.dynamic_snow, header.dynamic_damage,
        header.dynamic_dissolve, header.execution_plan_index,
        header.execution_program_index, header.execution_package_index,
        header.execution_permutation_low, header.execution_permutation_high,
    );
}

fn composite_evaluate_native_registered(
    surface: SurfaceSample,
    frame: ShadingFrame,
    world_position: vec3<f32>,
    identity: SurfaceIdentity,
    program_id: u32,
    parameters: PrismSurfaceParameters,
    uv: vec2<f32>,
    alpha: f32,
) -> CompositeProgramEvaluation {
    var result = CompositeProgramEvaluation(false, composite_response_zero());
    if (identity.material_index >= arrayLength(&composite_material_headers)) { return result; }
    let header = composite_material_headers[identity.material_index];
    let registered_surface = composite_registered_surface(surface);
    let registered_frame = RegisteredFrame(frame.normal, frame.view, frame.tangent, frame.bitangent);
    let registered_geometry = RegisteredGeometry(
        world_position, frame.normal, frame.tangent, frame.bitangent, uv, 0u,
    );
    let environment = RegisteredEnvironment(
        vec3<f32>(0.0), min(params.counts.y, 1024u),
        array<vec4<f32>, 9>(
            vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0),
            vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0),
            vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0),
        ),
        min(params.counts.z, 1024u), 0u, 1.0, params.flags.x,
        RegisteredStylizedParams(
            params.stylized.bands, params.stylized.wrap,
            params.stylized.ramp_softness, params.stylized.shadow_threshold,
            params.stylized.shadow_softness, params.stylized.specular_intensity,
            params.stylized.specular_threshold, params.stylized.specular_softness,
            params.stylized.specular_color, params.stylized.rim_intensity,
            params.stylized.rim_color, params.stylized.rim_power,
        ),
    );
    let lighting = RegisteredLightingContext(
        environment, registered_frame, world_position,
        min(params.counts.y, 1024u), min(params.counts.z, 1024u),
        min(params.flags.y, 1024u), 0u, 0u,
        vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(1.0),
    );
    let view = RegisteredView(
        camera_view.position.xyz, vec2<u32>(0u), vec2<u32>(0u),
        params.flags.z,
        mat4x4<f32>(
            vec4<f32>(1.0, 0.0, 0.0, 0.0),
            vec4<f32>(0.0, 1.0, 0.0, 0.0),
            vec4<f32>(0.0, 0.0, 1.0, 0.0),
            vec4<f32>(0.0, 0.0, 0.0, 1.0),
        ),
        mat4x4<f32>(
            vec4<f32>(1.0, 0.0, 0.0, 0.0),
            vec4<f32>(0.0, 1.0, 0.0, 0.0),
            vec4<f32>(0.0, 0.0, 1.0, 0.0),
            vec4<f32>(0.0, 0.0, 0.0, 1.0),
        ),
    );
    let custom = RegisteredCustomData(
        composite_registered_header(header),
        header.execution_permutation_low, header.execution_permutation_high,
    );
    let output = prism_native_registered_evaluate(
        RegisteredSurfaceData(
            registered_surface, registered_frame, registered_geometry,
            composite_registered_parameters(parameters),
        ),
        lighting,
        view,
        custom,
        program_id,
        header.execution_permutation_low,
        header.execution_permutation_high,
    );
    result.response = composite_response_from_channels(
        output.direct, output.indirect_diffuse, output.reflection,
        output.transmission, output.emission, alpha,
    );
    if (!finite(output.alpha) || output.alpha < 0.0 || output.alpha > 1.0) { return result; }
    result.valid = true;
    return result;
}
"#;

/// Per-target resources consumed by the lighting composite.
///
/// The producer must supply distinct 2D, single-sample planes with identical
/// width and height, clearing normal alpha to zero before opaque raster.
/// View-id validation below cannot detect two views of the same texture or
/// validate dimensions/formats; the owner and backend must enforce those.
/// Formats are intentionally fixed by this
/// contract so the pipeline layout and WGSL stay device-independent:
/// position `R32Float` x/y/z, normal `Rgba16Float` (alpha carries binary
/// coverage), sRGB albedo `Rgba8UnormSrgb`, surface id `R32Uint`, and HDR
/// output `Rgba16Float`. SurfaceData is an additional `Rgba16Float` plane
/// carrying UV, footprint, and tangent orientation. Two more `Rgba16Float`
/// planes carry sampled emissive/metallic and roughness/occlusion/coat/
/// normal-variance channels; C is a packed `R32Uint` word carrying
/// oct-encoded coat normal, coat roughness, and an opaque validity marker as
/// four UNORM8 lanes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct LightingCompositeTarget {
    pub position_x_view: TextureViewId,
    pub position_y_view: TextureViewId,
    pub position_z_view: TextureViewId,
    pub normal_view: TextureViewId,
    pub albedo_view: TextureViewId,
    pub surface_id_view: TextureViewId,
    pub surface_data_view: TextureViewId,
    pub sampled_material_a_view: TextureViewId,
    pub sampled_material_b_view: TextureViewId,
    pub sampled_material_c_view: TextureViewId,
    pub surface_identity_buffer: BufferId,
    pub surface_identity_size_bytes: u64,
    pub hdr_target_view: TextureViewId,
    pub response_output_buffer: BufferId,
    pub response_output_size_bytes: u64,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LightingCompositeTargetError {
    ZeroExtent,
    InvalidIdentitySize,
    InvalidResponseSize,
    DuplicateView { view: TextureViewId },
}

impl core::fmt::Display for LightingCompositeTargetError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ZeroExtent => {
                formatter.write_str("lighting composite target extent must be nonzero")
            }
            Self::InvalidIdentitySize => {
                formatter.write_str("lighting composite surface identity buffer is too small")
            }
            Self::InvalidResponseSize => {
                formatter.write_str("lighting composite response buffer is too small")
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
    let pixel_count = u64::from(target.width)
        .checked_mul(u64::from(target.height))
        .ok_or(LightingCompositeTargetError::InvalidResponseSize)?;
    let required_identity_bytes = pixel_count
        .checked_mul(crate::target::SURFACE_IDENTITY_BYTES_PER_PIXEL)
        .ok_or(LightingCompositeTargetError::InvalidIdentitySize)?;
    if target.surface_identity_size_bytes < required_identity_bytes {
        return Err(LightingCompositeTargetError::InvalidIdentitySize);
    }
    let required_response_bytes = pixel_count
        .checked_mul(COMPOSITE_RESPONSE_BYTES_PER_PIXEL)
        .ok_or(LightingCompositeTargetError::InvalidResponseSize)?;
    if target.response_output_size_bytes < required_response_bytes {
        return Err(LightingCompositeTargetError::InvalidResponseSize);
    }
    let views = [
        target.position_x_view,
        target.position_y_view,
        target.position_z_view,
        target.normal_view,
        target.albedo_view,
        target.surface_id_view,
        target.surface_data_view,
        target.sampled_material_a_view,
        target.sampled_material_b_view,
        target.sampled_material_c_view,
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
            uniform_layout(0),
            storage_layout(1, DIRECTIONAL_BYTES, true),
            storage_layout(2, PUNCTUAL_BYTES, true),
            storage_layout(3, AREA_BYTES, true),
            storage_layout(4, OUTPUT_BYTES, true),
            camera_view_layout(5),
            lobe_budget_layout(6),
        ],
    }
}

fn composite_target_layout_descriptor() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor {
        label: Some("lighting-composite-target-layout".into()),
        entries: vec![
            texture_layout(0, COMPOSITE_POSITION_X_FORMAT.sample_type()),
            texture_layout(1, COMPOSITE_POSITION_Y_FORMAT.sample_type()),
            texture_layout(2, COMPOSITE_POSITION_Z_FORMAT.sample_type()),
            texture_layout(3, COMPOSITE_NORMAL_FORMAT.sample_type()),
            texture_layout(4, COMPOSITE_ALBEDO_FORMAT.sample_type()),
            texture_layout(5, COMPOSITE_SURFACE_ID_FORMAT.sample_type()),
            storage_layout(6, crate::target::SURFACE_IDENTITY_BYTES_PER_PIXEL, true),
            storage_texture_layout(7, COMPOSITE_HDR_FORMAT),
            storage_layout(8, COMPOSITE_RESPONSE_BYTES_PER_PIXEL, false),
            texture_layout(9, COMPOSITE_SURFACE_DATA_FORMAT.sample_type()),
            texture_layout(
                10,
                crate::target::SURFACE_MATERIAL_A_GBUFFER_FORMAT.sample_type(),
            ),
            texture_layout(
                11,
                crate::target::SURFACE_MATERIAL_B_GBUFFER_FORMAT.sample_type(),
            ),
            texture_layout(
                12,
                crate::target::SURFACE_MATERIAL_C_GBUFFER_FORMAT.sample_type(),
            ),
        ],
    }
}

fn surface_identity_layout_descriptor() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor {
        label: Some("lighting-surface-identity-layout".into()),
        entries: vec![
            texture_layout(0, COMPOSITE_NORMAL_FORMAT.sample_type()),
            texture_layout(1, COMPOSITE_SURFACE_ID_FORMAT.sample_type()),
            storage_layout(2, MATERIAL_HEADER_BYTES, true),
            storage_layout(3, MATERIAL_PLAN_ROW_BYTES, true),
            storage_layout(4, MATERIAL_PROGRAM_ROW_BYTES, true),
            storage_layout(5, crate::target::SURFACE_IDENTITY_BYTES_PER_PIXEL, false),
            storage_layout(6, MATERIAL_LIGHTING_SURFACE_BYTES, true),
            uniform_layout(7),
        ],
    }
}

/// Pipeline group used by the Native composite for canonical material data.
/// Group 2 is required by the material resolver and separate from lighting and targets.
pub(crate) const COMPOSITE_MATERIAL_GROUP: u32 = 2;
pub(crate) const COMPOSITE_RESPONSE_BYTES_PER_PIXEL: u64 = 16 * 4;

/// Required members of one canonical material snapshot. The composite group
/// consumes only headers and parameters; execution plans/programs are bound by
/// the post-raster identity prepass so the composite stays within the adapter
/// storage-buffer binding limit.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct LightingCompositeMaterialBuffers {
    pub headers: Option<prism_render_driver::BufferId>,
    pub headers_size_bytes: u64,
    pub parameters: Option<prism_render_driver::BufferId>,
    pub parameters_size_bytes: u64,
    pub execution_plans: Option<prism_render_driver::BufferId>,
    pub execution_plans_size_bytes: u64,
    pub execution_programs: Option<prism_render_driver::BufferId>,
    pub execution_programs_size_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LightingCompositeMaterialBindingError {
    MissingHeaders,
    MissingParameters,
    MissingExecutionPlans,
    MissingExecutionPrograms,
    InvalidBindingSize {
        binding: u32,
        size_bytes: u64,
        minimum_bytes: u64,
        row_bytes: u64,
    },
}

impl core::fmt::Display for LightingCompositeMaterialBindingError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str(match self {
            Self::MissingHeaders => "canonical material header buffer is missing",
            Self::MissingParameters => "canonical material parameter heap is missing",
            Self::MissingExecutionPlans => "canonical material execution plan buffer is missing",
            Self::MissingExecutionPrograms => {
                "canonical material execution program buffer is missing"
            }
            Self::InvalidBindingSize { .. } => "canonical material storage binding size is invalid",
        })
    }
}

impl std::error::Error for LightingCompositeMaterialBindingError {}

fn composite_material_layout_descriptor() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor {
        label: Some("lighting-composite-material-layout".into()),
        entries: vec![
            storage_layout(0, MATERIAL_HEADER_BYTES, true),
            storage_layout(1, MATERIAL_PARAMETER_WORD_BYTES, true),
        ],
    }
}

fn validate_material_buffers(
    buffers: LightingCompositeMaterialBuffers,
) -> Result<[(prism_render_driver::BufferId, u64); 4], LightingCompositeMaterialBindingError> {
    let bindings = [
        (
            buffers
                .headers
                .ok_or(LightingCompositeMaterialBindingError::MissingHeaders)?,
            buffers.headers_size_bytes,
            MATERIAL_HEADER_BYTES,
            MATERIAL_HEADER_BYTES,
        ),
        (
            buffers
                .parameters
                .ok_or(LightingCompositeMaterialBindingError::MissingParameters)?,
            buffers.parameters_size_bytes,
            MATERIAL_PARAMETER_WORD_BYTES,
            MATERIAL_PARAMETER_WORD_BYTES,
        ),
        (
            buffers
                .execution_plans
                .ok_or(LightingCompositeMaterialBindingError::MissingExecutionPlans)?,
            buffers.execution_plans_size_bytes,
            MATERIAL_PLAN_ROW_BYTES,
            MATERIAL_PLAN_ROW_BYTES,
        ),
        (
            buffers
                .execution_programs
                .ok_or(LightingCompositeMaterialBindingError::MissingExecutionPrograms)?,
            buffers.execution_programs_size_bytes,
            MATERIAL_PROGRAM_ROW_BYTES,
            MATERIAL_PROGRAM_ROW_BYTES,
        ),
    ];
    for (binding, (_, size_bytes, minimum_bytes, row_bytes)) in bindings.iter().copied().enumerate()
    {
        if size_bytes < minimum_bytes || size_bytes % row_bytes != 0 {
            return Err(LightingCompositeMaterialBindingError::InvalidBindingSize {
                binding: binding as u32,
                size_bytes,
                minimum_bytes,
                row_bytes,
            });
        }
    }
    Ok(bindings.map(|(buffer, size_bytes, _, _)| (buffer, size_bytes)))
}

#[cfg(test)]
mod composite_contract_tests {
    use super::*;
    use crate::target::SURFACE_IDENTITY_BYTES_PER_PIXEL;

    fn target() -> LightingCompositeTarget {
        LightingCompositeTarget {
            position_x_view: TextureViewId::from_parts(1, 1),
            position_y_view: TextureViewId::from_parts(2, 1),
            position_z_view: TextureViewId::from_parts(3, 1),
            normal_view: TextureViewId::from_parts(4, 1),
            albedo_view: TextureViewId::from_parts(5, 1),
            surface_id_view: TextureViewId::from_parts(7, 1),
            surface_data_view: TextureViewId::from_parts(11, 1),
            sampled_material_a_view: TextureViewId::from_parts(12, 1),
            sampled_material_b_view: TextureViewId::from_parts(13, 1),
            sampled_material_c_view: TextureViewId::from_parts(14, 1),
            surface_identity_buffer: BufferId::from_parts(9, 1),
            surface_identity_size_bytes: 1920 * 1080 * SURFACE_IDENTITY_BYTES_PER_PIXEL,
            hdr_target_view: TextureViewId::from_parts(8, 1),
            response_output_buffer: BufferId::from_parts(10, 1),
            response_output_size_bytes: 1920 * 1080 * COMPOSITE_RESPONSE_BYTES_PER_PIXEL,
            width: 1920,
            height: 1080,
        }
    }

    #[test]
    fn gbuffer_formats_fit_the_webgpu_per_sample_attachment_budget() {
        let formats = [
            COMPOSITE_POSITION_X_FORMAT,
            COMPOSITE_POSITION_Y_FORMAT,
            COMPOSITE_POSITION_Z_FORMAT,
            COMPOSITE_NORMAL_FORMAT,
            COMPOSITE_ALBEDO_FORMAT,
            COMPOSITE_SURFACE_ID_FORMAT,
            COMPOSITE_SURFACE_DATA_FORMAT,
            crate::target::SURFACE_MATERIAL_A_GBUFFER_FORMAT,
            crate::target::SURFACE_MATERIAL_B_GBUFFER_FORMAT,
            crate::target::SURFACE_MATERIAL_C_GBUFFER_FORMAT,
        ];
        assert_eq!(
            formats
                .iter()
                .map(|format| format.bytes_per_texel())
                .sum::<u32>(),
            56
        );
        // WebGPU target pixel byte cost is NOT the stored texel size:
        // rgba8unorm(-srgb) costs 8, not 4. The actual GPU pipeline/pass tests
        // below remain the authority for backend acceptance of this profile.
        let bytes_per_sample = formats
            .into_iter()
            .map(|format| match format {
                TextureFormat::R32Float => 4,
                TextureFormat::Rgba16Float => 8,
                TextureFormat::Rgba8UnormSrgb => 8,
                TextureFormat::Rgba8Unorm => 4,
                TextureFormat::R32Uint => 4,
                _ => unreachable!("unexpected GBuffer format"),
            })
            .sum::<u32>();
        assert_eq!(bytes_per_sample, 60);
    }

    #[test]
    fn composite_target_requires_a_real_extent() {
        assert_eq!(validate_composite_target(target()), Ok(()));
        assert_eq!(
            validate_composite_target(LightingCompositeTarget {
                surface_identity_size_bytes: target().surface_identity_size_bytes - 1,
                ..target()
            }),
            Err(LightingCompositeTargetError::InvalidIdentitySize)
        );
        for (width, height) in [(0, 1080), (1920, 0), (0, 0)] {
            assert_eq!(
                validate_composite_target(LightingCompositeTarget {
                    width,
                    height,
                    ..target()
                }),
                Err(LightingCompositeTargetError::ZeroExtent)
            );
        }
    }

    #[test]
    fn composite_target_rejects_view_aliasing() {
        for left in 0..11 {
            for right in left + 1..11 {
                let mut views = core::array::from_fn::<_, 11, _>(|i| {
                    TextureViewId::from_parts(i as u32 + 1, 1)
                });
                views[right] = views[left];
                let aliased = LightingCompositeTarget {
                    position_x_view: views[0],
                    position_y_view: views[1],
                    position_z_view: views[2],
                    normal_view: views[3],
                    albedo_view: views[4],
                    surface_id_view: views[5],
                    surface_data_view: views[6],
                    sampled_material_a_view: views[7],
                    sampled_material_b_view: views[8],
                    sampled_material_c_view: views[9],
                    hdr_target_view: views[10],
                    ..target()
                };
                assert_eq!(
                    validate_composite_target(aliased),
                    Err(LightingCompositeTargetError::DuplicateView { view: views[left] })
                );
            }
        }
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
        for (width, height, expected) in [(1, 1, [1, 1, 1]), (9, 7, [2, 1, 1]), (8, 8, [1, 1, 1])] {
            assert_eq!(
                LightingGpuPass::composite_workgroups(LightingCompositeTarget {
                    width,
                    height,
                    ..target()
                }),
                expected
            );
        }
    }

    #[test]
    fn composite_layout_has_explicit_bindings_without_placeholder_slots() {
        let static_layout = composite_static_layout_descriptor();
        assert_eq!(
            static_layout
                .entries
                .iter()
                .map(|entry| entry.binding)
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4, 5, 6]
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
            vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        );
        assert!(target_layout
            .entries
            .iter()
            .all(|entry| entry.validate().is_ok()));
        for (entry, sample_type) in target_layout.entries[..6].iter().zip([
            TextureSampleType::UnfilterableFloat,
            TextureSampleType::UnfilterableFloat,
            TextureSampleType::UnfilterableFloat,
            TextureSampleType::Float,
            TextureSampleType::Float,
            TextureSampleType::Uint,
        ]) {
            assert_eq!(
                entry.ty,
                BindingType::Texture {
                    sample_type,
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                }
            );
        }
        assert!(target_layout
            .entries
            .iter()
            .all(|entry| entry.visibility == ShaderStages::COMPUTE && entry.count.is_none()));
        assert_eq!(
            target_layout.entries[6].ty,
            BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: Some(SURFACE_IDENTITY_BYTES_PER_PIXEL),
            }
        );
        assert_eq!(
            target_layout.entries[7].ty,
            BindingType::StorageTexture {
                access: StorageTextureAccess::WriteOnly,
                format: COMPOSITE_HDR_FORMAT,
                view_dimension: TextureViewDimension::D2,
            }
        );
        assert_eq!(
            target_layout.entries[9].ty,
            BindingType::Texture {
                sample_type: TextureSampleType::Float,
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            }
        );
    }

    #[test]
    fn composite_material_group_has_canonical_required_storage_contract() {
        let layout = composite_material_layout_descriptor();
        assert_eq!(COMPOSITE_MATERIAL_GROUP, 2);
        assert_eq!(
            layout
                .entries
                .iter()
                .map(|entry| entry.binding)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert!(layout.entries.iter().all(|entry| {
            entry.validate().is_ok()
                && entry.visibility == ShaderStages::COMPUTE
                && entry.count.is_none()
                && entry.ty
                    == BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(match entry.binding {
                            0 => MATERIAL_HEADER_BYTES,
                            1 => MATERIAL_PARAMETER_WORD_BYTES,
                            _ => unreachable!("material binding is outside the contract"),
                        }),
                    }
        }));

        let ids =
            [0, 1, 2, 3].map(|index| prism_render_driver::BufferId::from_parts(index + 10, 7));
        let sizes = [
            MATERIAL_HEADER_BYTES * 4,
            256,
            MATERIAL_PLAN_ROW_BYTES * 3,
            MATERIAL_PROGRAM_ROW_BYTES * 2,
        ];
        let valid = LightingCompositeMaterialBuffers {
            headers: Some(ids[0]),
            headers_size_bytes: sizes[0],
            parameters: Some(ids[1]),
            parameters_size_bytes: sizes[1],
            execution_plans: Some(ids[2]),
            execution_plans_size_bytes: sizes[2],
            execution_programs: Some(ids[3]),
            execution_programs_size_bytes: sizes[3],
        };
        assert_eq!(
            validate_material_buffers(valid),
            Ok(core::array::from_fn(|i| (ids[i], sizes[i])))
        );
        for (missing, expected) in [
            (0, LightingCompositeMaterialBindingError::MissingHeaders),
            (1, LightingCompositeMaterialBindingError::MissingParameters),
            (
                2,
                LightingCompositeMaterialBindingError::MissingExecutionPlans,
            ),
            (
                3,
                LightingCompositeMaterialBindingError::MissingExecutionPrograms,
            ),
        ] {
            let mut incomplete = valid;
            match missing {
                0 => incomplete.headers = None,
                1 => incomplete.parameters = None,
                2 => incomplete.execution_plans = None,
                3 => incomplete.execution_programs = None,
                _ => unreachable!(),
            }
            assert_eq!(validate_material_buffers(incomplete), Err(expected));
        }
        for (binding, invalid_size) in [
            (0, MATERIAL_HEADER_BYTES - 1),
            (1, 3),
            (2, MATERIAL_PLAN_ROW_BYTES + 1),
            (3, MATERIAL_PROGRAM_ROW_BYTES + 1),
        ] {
            let mut invalid = valid;
            match binding {
                0 => invalid.headers_size_bytes = invalid_size,
                1 => invalid.parameters_size_bytes = invalid_size,
                2 => invalid.execution_plans_size_bytes = invalid_size,
                3 => invalid.execution_programs_size_bytes = invalid_size,
                _ => unreachable!(),
            }
            assert!(matches!(
                validate_material_buffers(invalid),
                Err(LightingCompositeMaterialBindingError::InvalidBindingSize { binding: actual, .. }) if actual == binding
            ));
        }
        assert_eq!(
            validate_material_buffers(LightingCompositeMaterialBuffers::default()),
            Err(LightingCompositeMaterialBindingError::MissingHeaders)
        );
    }

    #[test]
    fn composite_rejects_non_finite_response_before_publication() {
        let shader = LIGHTING_COMPOSITE_WGSL;
        let blend = shader
            .find("response = composite_blend_response(")
            .expect("MaskBlend response blend must remain in the composite shader");
        let finite = shader
            .find("if (!composite_response_is_finite(response))")
            .expect("composite response must be finite-validated");
        let publish = shader
            .find("response_outputs[identity_index] = response;")
            .expect("response publication must remain explicit");
        assert!(
            blend < finite,
            "finite validation must follow response blending"
        );
        assert!(
            finite < publish,
            "finite validation must precede response publication"
        );
        assert!(
            shader[finite..].contains("textureStore(hdr_target, pixel, vec4<f32>(0.0));"),
            "invalid response must fail closed to a cleared HDR pixel"
        );
    }

    #[test]
    fn composite_registered_blend_arms_use_distinct_program_ids() {
        let shader = LIGHTING_COMPOSITE_WGSL;
        assert!(shader.contains("identity.left_program_kind,\n        identity.left_program_id,"));
        assert!(
            shader.contains("identity.right_program_kind,\n            identity.right_program_id,")
        );
        assert!(super::NATIVE_REGISTERED_ADAPTER_WGSL.contains("program_id: u32,"));
        assert!(super::NATIVE_REGISTERED_ADAPTER_WGSL.contains("        program_id,\n"));
    }

    #[test]
    fn composite_admission_is_canonical_and_mask_is_fail_closed() {
        let material = include_str!("shaders/composite_material.wesl");
        for contract in [
            "composite_execution_identity_is_valid",
            "header.execution_program_index == identity.left_program_index",
            "identity.program_kind != identity.left_program_kind",
            "builtin MaskBlend is the canonical fused Lit/Lit form",
            "identity.mask_kind != 5u || identity.mask_index > 3u",
        ] {
            assert!(
                material.contains(contract),
                "missing canonical material admission contract: {contract}"
            );
        }

        let shader = LIGHTING_COMPOSITE_WGSL;
        assert!(shader.contains("raw < 0.0 || raw > 1.0"));
        assert!(shader.contains("return CompositeMaskValue(true, raw);"));
        assert!(!shader.contains("mix(left.words"));
        assert!(shader.contains("composite_lerp(left.words[3]"));
    }

    #[test]
    fn composite_consumes_native_sampled_material_planes() {
        let shader = LIGHTING_COMPOSITE_WGSL;
        for declaration in [
            "@group(1) @binding(10)",
            "@group(1) @binding(11)",
            "@group(1) @binding(12)",
            "surface.emissive = sampled_material_a.xyz * material.surface.emission_luminance",
            "surface.metallic = sampled_material_a.w",
            "surface.perceptual_roughness = sampled_material_b.x",
            "surface.ambient_occlusion = sampled_material_b.y",
            "surface.clearcoat = sampled_material_b.z",
            "surface.normal_variance = sampled_material_b.w",
            "surface.clearcoat_roughness = sampled_material_c.z",
            "surface.coat_normal = composite_decode_oct_normal(sampled_material_c.xy)",
            "sampled_material_c.w <= 0.0",
            "composite_unpack_material_c",
        ] {
            assert!(
                shader.contains(declaration),
                "missing Native sampled channel consumer: {declaration}"
            );
        }
    }

    #[test]
    fn lobe_budget_native_uniform_matches_the_wesl_eight_word_abi() {
        assert_eq!(core::mem::size_of::<GpuLobeSampleBudget>(), 32);
        assert_eq!(LOBE_SAMPLE_BUDGET_BYTES, 32);
        let budget = GpuLobeSampleBudget::for_quality(ShadingQuality::High);
        assert_eq!(budget.sss_separable_taps, 13);
        assert_eq!(budget.sss_preintegration_steps, 160);
        assert_eq!(budget.translucency_profile_samples, 48);
        assert_eq!(budget.multiscatter_dir_samples, 64);
        assert_eq!(budget.multiscatter_march_samples, 16);
        assert_eq!(budget.total_samples, 301);
        assert_eq!(budget.revision, 0);
        assert_eq!(
            budget.max_full_lobes,
            LobeSamplingPolicy::for_quality(ShadingQuality::High).max_full_lobes()
        );
        assert_eq!(bytes_of(&budget).len(), 8 * core::mem::size_of::<u32>());
    }
}

#[cfg(test)]
mod gbuffer_gpu_contract_tests {
    use super::*;
    use crate::forward_lit::{
        SurfaceEncodeProvider, SurfaceFrame, SurfaceMaterial, SurfaceResourceSet,
    };
    use crate::mesh_pipeline::MeshPipelineProvider;
    use prism_render_architecture::ray_scene::{RayHitAttributeRecord, RayHitIdentity};
    use prism_render_driver::{
        self as rhi, CommandEncoder, ComputeCommand, Extent3d, RenderCommand, SamplerDescriptor,
        TextureDescriptor, TextureDimension, TextureUsages, TextureViewDescriptor,
    };
    use prism_render_driver_wgpu::{require_hardware_gpu, WgpuInstance};
    use prism_render_material::response_source::{
        linked_registered_response_wgsl, RegisteredResponseSource,
    };
    use prism_render_material::{
        GpuMaterialHeader, GpuMaterialTexture, GpuSurfaceParameters, LobeMask,
        SurfaceParameterBlock,
    };
    use prism_render_shading::{
        evaluate_principled_direct, evaluate_stylized_direct, surface_sample_from_parameters,
        DirectLightSample, ShadingFrame, StylizedParams,
    };

    const WIDTH: u32 = 9;
    const HEIGHT: u32 = 7;

    fn canonical_surface_for_program(
        albedo: [f32; 4],
        geometry_opacity: f32,
        program: u32,
    ) -> (GpuMaterialHeader, Vec<u32>) {
        let lobe_mask = if program == 1 {
            LobeMask::EMISSION.union(LobeMask::NPR_FACE)
        } else {
            LobeMask::EMISSION
        };
        let block = SurfaceParameterBlock::from_full(
            &GpuSurfaceParameters {
                base_color: albedo,
                geometry_opacity,
                npr_face_softness: 0.25,
                npr_stylized_ramp_softness: 1.0,
                npr_stylized_specular_scale: 1.0,
                npr_stylized_rim_scale: 1.0,
                ..GpuSurfaceParameters::default()
            },
            lobe_mask,
        );
        let packed = block.pack();
        // Poison a nonzero prefix: the consumer must honor word offsets, not
        // treat parameter_offset as a byte offset or a fixed-stride row index.
        let mut heap = vec![0x7fc0_0000; 7];
        let header = GpuMaterialHeader {
            generation: 0x8123_4567,
            revision: 0x9234_5678,
            is_active: 1,
            lobe_mask: block.lobe_mask.bits(),
            parameter_offset: heap.len() as u32,
            parameter_size: (packed.len() * 4) as u32,
            execution_plan_index: program,
            execution_program_index: program,
            execution_package_index: 0,
            execution_permutation_low: 0,
            execution_permutation_high: 0,
            ..GpuMaterialHeader::default()
        };
        heap.extend(packed);
        (header, heap)
    }

    fn canonical_surface(albedo: [f32; 4], geometry_opacity: f32) -> (GpuMaterialHeader, Vec<u32>) {
        canonical_surface_for_program(albedo, geometry_opacity, 0)
    }

    fn canonical_npr_surface(
        albedo: [f32; 4],
        geometry_opacity: f32,
    ) -> (GpuMaterialHeader, Vec<u32>) {
        canonical_surface_for_program(albedo, geometry_opacity, 1)
    }

    struct Fixture {
        runtime: RenderRuntime<WgpuInstance>,
        _lighting: LightingGpuState<WgpuInstance>,
        target: RenderTarget,
        pass: LightingGpuPass,
        headers: BufferId,
        parameters: BufferId,
    }

    fn fixture() -> Option<Fixture> {
        let extent = rhi::Extent3d::new_2d(WIDTH, HEIGHT);
        let mut runtime = match RenderRuntime::headless(WgpuInstance::new(), extent) {
            Ok(runtime) => runtime,
            Err(prism_render_runtime::RenderInitError::NoAdapter) => {
                assert!(
                    !require_hardware_gpu(),
                    "PRISM_REQUIRE_GPU=1 but no adapter was found"
                );
                eprintln!("SKIPPED GBuffer GPU contract: no adapter");
                return None;
            }
            Err(error) => panic!("GBuffer GPU device creation failed: {error}"),
        };
        if !matches!(
            runtime.adapter_info().device_type,
            rhi::DeviceType::IntegratedGpu | rhi::DeviceType::DiscreteGpu
        ) {
            assert!(
                !require_hardware_gpu(),
                "GBuffer acceptance requires a physical GPU"
            );
            eprintln!("SKIPPED GBuffer GPU contract: non-hardware adapter");
            return None;
        }
        let lighting = LightingGpuState::new(&mut runtime).unwrap();
        let plan = LightingPlan {
            tier: prism_render_architecture::lighting::contract::Tier::Balanced,
            recipes: vec![RecipeId::SpecializedDeterministic],
            candidate_source: GiCandidateSource::Disabled,
            partition: Default::default(),
            budget: Default::default(),
            rejected: Vec::new(),
        };
        let directional = LightingDirectionalInput {
            direction_to_light: [0.0, 1.0, 0.0],
            visibility: 1.0,
            illuminance: [4.0, 2.0, 1.0],
            _padding: 0.0,
        };
        lighting
            .prepare(
                &plan,
                &ExtractedLightingInputs {
                    directional_handles: vec![LightSceneHandle::new(0, 1)],
                    directionals: vec![directional],
                    surfaces: vec![
                        LightingSurfaceInput {
                            position: [0.0, 0.0, 0.0],
                            normal: [0.0, 1.0, 0.0],
                            albedo: [0.18, 0.32, 0.72],
                            ambient: [0.0, 0.0, 0.0],
                        },
                        LightingSurfaceInput {
                            position: [0.0, 0.0, 0.0],
                            normal: [0.0, 1.0, 0.0],
                            albedo: [0.18, 0.32, 0.72],
                            ambient: [0.0, 0.0, 0.0],
                        },
                        LightingSurfaceInput {
                            position: [0.0, 0.0, 0.0],
                            normal: [0.0, 1.0, 0.0],
                            albedo: [0.18, 0.32, 0.72],
                            ambient: [0.0, 0.0, 0.0],
                        },
                    ],
                    surface_material_indices: vec![1, 2, 3],
                    ..Default::default()
                },
                1,
                &runtime,
            )
            .unwrap();
        let target = RenderTarget::new(runtime.device(), extent);
        assert_eq!(
            [
                target.position_x_format(),
                target.position_y_format(),
                target.position_z_format(),
                target.normal_format(),
                target.albedo_format(),
                target.surface_id_format(),
                target.format(),
            ],
            [
                TextureFormat::R32Float,
                TextureFormat::R32Float,
                TextureFormat::R32Float,
                TextureFormat::Rgba16Float,
                TextureFormat::Rgba8UnormSrgb,
                TextureFormat::R32Uint,
                TextureFormat::Rgba16Float,
            ]
        );
        let (initial_header, initial_heap) = canonical_surface([0.18, 0.32, 0.72, 1.0], 1.0);
        let (_, npr_heap) = canonical_npr_surface([0.18, 0.32, 0.72, 1.0], 1.0);
        let headers = runtime.device().create_buffer(&BufferDescriptor::new(
            (4 * core::mem::size_of::<GpuMaterialHeader>()) as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        ));
        runtime
            .queue()
            .write_buffer(headers, 0, cast_slice(&[initial_header; 4]));
        let parameters = runtime.device().create_buffer(&BufferDescriptor::new(
            (core::mem::size_of::<u32>() * initial_heap.len().max(npr_heap.len())) as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        ));
        runtime
            .queue()
            .write_buffer(parameters, 0, cast_slice(&initial_heap));
        let execution_plans = runtime.device().create_buffer(&BufferDescriptor::new(
            MATERIAL_PLAN_ROW_BYTES * 3,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        ));
        runtime.queue().write_buffer(
            execution_plans,
            0,
            cast_slice(&[
                0u32,
                u32::MAX,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                u32::MAX,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                2,
                0,
                1,
                5,
                0,
                1,
                0,
            ]),
        );
        let execution_programs = runtime.device().create_buffer(&BufferDescriptor::new(
            MATERIAL_PROGRAM_ROW_BYTES * 3,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        ));
        runtime.queue().write_buffer(
            execution_programs,
            0,
            cast_slice(&[0u32, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0]),
        );
        let mut pass = lighting
            .target_pass(
                &plan,
                &target,
                &runtime,
                [0.0; 3],
                LightingCompositeMaterialBuffers {
                    headers: Some(headers),
                    headers_size_bytes: (4 * core::mem::size_of::<GpuMaterialHeader>()) as u64,
                    parameters: Some(parameters),
                    parameters_size_bytes: (core::mem::size_of::<u32>()
                        * initial_heap.len().max(npr_heap.len()))
                        as u64,
                    execution_plans: Some(execution_plans),
                    execution_plans_size_bytes: MATERIAL_PLAN_ROW_BYTES * 3,
                    execution_programs: Some(execution_programs),
                    execution_programs_size_bytes: MATERIAL_PROGRAM_ROW_BYTES * 3,
                },
                None,
            )
            .unwrap();
        pass.run_gi = false;
        Some(Fixture {
            runtime,
            _lighting: lighting,
            target,
            pass,
            headers,
            parameters,
        })
    }

    #[test]
    fn physical_gpu_registered_response_links_and_reads_canonical_inputs() {
        let Some(f) = fixture() else {
            return;
        };
        let device = f.runtime.device();
        let queue = f.runtime.queue();
        let target = f.target;

        let (mut header, heap) = canonical_surface([0.18, 0.32, 0.72, 1.0], 1.0);
        header.execution_plan_index = 0;
        header.execution_program_index = 0;
        header.execution_package_index = 0;
        let headers = device.create_buffer(&BufferDescriptor::new(
            (4 * core::mem::size_of::<GpuMaterialHeader>()) as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        ));
        queue.write_buffer(headers, 0, cast_slice(&[header; 4]));
        let parameters = device.create_buffer(&BufferDescriptor::new(
            (heap.len() * core::mem::size_of::<u32>()) as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        ));
        queue.write_buffer(parameters, 0, cast_slice(&heap));
        let execution_plans = device.create_buffer(&BufferDescriptor::new(
            MATERIAL_PLAN_ROW_BYTES,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        ));
        queue.write_buffer(
            execution_plans,
            0,
            cast_slice(&[0u32, u32::MAX, 2, 0, 0, 0, 0, 0]),
        );
        let execution_programs = device.create_buffer(&BufferDescriptor::new(
            MATERIAL_PROGRAM_ROW_BYTES,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        ));
        queue.write_buffer(execution_programs, 0, cast_slice(&[3u32, 0, 1, 0, 7]));

        // The body deliberately consumes all three ABI inputs. Its expected
        // output is simple enough to calculate independently on the CPU.
        let body = r#"
let light = registered_light(lighting, 0u);
let scale = 0.25 + light.visibility + light.illuminance.x * 0.125
    + f32(custom.header.execution_package_index);
return RegisteredResponse(
    surface.sample.base_color * scale,
    vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), 1.0);
"#;
        let response_link =
            linked_registered_response_wgsl(&[RegisteredResponseSource::new(7, vec![body.into()])]);
        let plan = LightingPlan {
            tier: prism_render_architecture::lighting::contract::Tier::Balanced,
            recipes: vec![RecipeId::SpecializedDeterministic],
            candidate_source: GiCandidateSource::Disabled,
            partition: Default::default(),
            budget: Default::default(),
            rejected: Vec::new(),
        };
        let buffers = LightingCompositeMaterialBuffers {
            headers: Some(headers),
            headers_size_bytes: (4 * core::mem::size_of::<GpuMaterialHeader>()) as u64,
            parameters: Some(parameters),
            parameters_size_bytes: (heap.len() * core::mem::size_of::<u32>()) as u64,
            execution_plans: Some(execution_plans),
            execution_plans_size_bytes: MATERIAL_PLAN_ROW_BYTES,
            execution_programs: Some(execution_programs),
            execution_programs_size_bytes: MATERIAL_PROGRAM_ROW_BYTES,
        };
        let mut pass = f
            ._lighting
            .target_pass(
                &plan,
                &target,
                &f.runtime,
                [0.0; 3],
                buffers,
                Some(&response_link),
            )
            .expect("registered material pass must be admitted");
        pass.run_gi = false;

        let write = |texture, bytes: &[u8]| {
            queue.write_texture(
                rhi::TextureWrite {
                    texture,
                    mip_level: 0,
                    origin: [0, 0, 0],
                    size: Extent3d::new_2d(1, 1),
                },
                bytes,
                rhi::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes.len() as u32),
                    rows_per_image: Some(1),
                },
            );
        };
        for texture in [
            target.position_x_texture(),
            target.position_y_texture(),
            target.position_z_texture(),
        ] {
            write(texture, &0.0_f32.to_le_bytes());
        }
        write(
            target.normal_texture(),
            &[0_u16, 0, 0x3c00, 0x3c00]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect::<Vec<_>>(),
        );
        // White makes the authored response non-zero while keeping the
        // expected scale independent of sRGB8 quantization.
        write(target.albedo_texture(), &[255, 255, 255, 255]);
        write(target.surface_id_texture(), &1_u32.to_le_bytes());
        write(
            target.surface_data_texture(),
            &[0_u16, 0, 0, 0x3c00]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect::<Vec<_>>(),
        );
        // The composite treats sampled material planes as a real producer
        // contract. Keep this direct-response probe explicit instead of
        // relying on newly allocated zeroed textures, especially for C's
        // validity marker.
        write(
            target.surface_material_a_texture(),
            &[0_u16, 0, 0, 0]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect::<Vec<_>>(),
        );
        write(
            target.surface_material_b_texture(),
            &[0x34cd_u16, 0x3c00, 0, 0]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect::<Vec<_>>(),
        );
        write(
            target.surface_material_c_texture(),
            &0xff008080_u32.to_le_bytes(),
        );

        let mut encoder = CommandEncoder::new();
        encoder.push_compute_pass(None, composite_commands(pass));
        queue.submit(&[encoder.finish()]);
        assert!(queue.wait_idle(), "registered response dispatch completes");
        let hdr = queue
            .read_texture_rgba16f(target.texture(), WIDTH, HEIGHT)
            .expect("registered response HDR readback must complete");
        let identity = queue
            .read_buffer(target.surface_identity_buffer(), 16 * 4)
            .expect("registered surface identity readback must complete");
        let identity_words = identity
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
            .collect::<Vec<_>>();
        assert_eq!(
            hdr[0],
            [1.75, 1.75, 1.75, 1.0],
            "identity={identity_words:?}"
        );
        assert!(hdr[1..].iter().all(|pixel| *pixel == [0.0; 4]));
    }

    fn half_rgba_bytes(values: [f32; 4]) -> Vec<u8> {
        values
            .into_iter()
            .flat_map(|value| {
                prism_render_architecture::particle::half_float_f16::f32_to_f16_bits(value)
                    .to_le_bytes()
            })
            .collect()
    }

    fn half_rgba_values(values: [f32; 4]) -> [f32; 4] {
        values.map(|value| {
            prism_render_architecture::particle::half_float_f16::f16_bits_to_f32(
                prism_render_architecture::particle::half_float_f16::f32_to_f16_bits(value),
            )
        })
    }

    fn srgb8_to_linear(value: u8) -> f32 {
        let value = f32::from(value) / 255.0;
        if value <= 0.04045 {
            value / 12.92
        } else {
            ((value + 0.055) / 1.055).powf(2.4)
        }
    }

    fn decode_sampled_coat_normal(encoded: [u8; 2]) -> [f32; 3] {
        let p = [
            f32::from(encoded[0]) / 255.0 * 2.0 - 1.0,
            f32::from(encoded[1]) / 255.0 * 2.0 - 1.0,
        ];
        let mut normal = [p[0], p[1], 1.0 - p[0].abs() - p[1].abs()];
        if normal[2] < 0.0 {
            let reflected = [
                (1.0 - normal[1].abs()) * if normal[0] >= 0.0 { 1.0 } else { -1.0 },
                (1.0 - normal[0].abs()) * if normal[1] >= 0.0 { 1.0 } else { -1.0 },
            ];
            normal[0] = reflected[0];
            normal[1] = reflected[1];
        }
        normalize(normal, [0.0, 1.0, 0.0])
    }

    #[test]
    fn physical_gpu_non_neutral_sampled_material_matches_cpu_golden() {
        let Some(f) = fixture() else {
            return;
        };
        let queue = f.runtime.queue();
        let target = f.target;

        // These are deliberately non-neutral across all three sampled planes.
        // A/B are written as the actual RGBA16F attachment payloads and C is
        // the actual R32Uint oct-normal/roughness/validity packing.
        let sampled_a = half_rgba_values([0.23, 0.61, 0.37, 0.65]);
        let sampled_b = half_rgba_values([0.58, 0.82, 0.72, 0.18]);
        let sampled_c = [160_u8, 192, 96, 255];
        let albedo = [128_u8, 96, 64, 255];
        let write = |texture, bytes: &[u8]| {
            queue.write_texture(
                rhi::TextureWrite {
                    texture,
                    mip_level: 0,
                    origin: [0, 0, 0],
                    size: Extent3d::new_2d(1, 1),
                },
                bytes,
                rhi::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes.len() as u32),
                    rows_per_image: Some(1),
                },
            );
        };
        for (texture, value) in [
            (target.position_x_texture(), 0.0_f32),
            (target.position_y_texture(), 0.0_f32),
            (target.position_z_texture(), 0.0_f32),
        ] {
            write(texture, &value.to_le_bytes());
        }
        write(
            target.normal_texture(),
            &half_rgba_bytes([0.0, 1.0, 0.0, 1.0]),
        );
        write(target.albedo_texture(), &albedo);
        write(target.surface_id_texture(), &1_u32.to_le_bytes());
        write(
            target.surface_data_texture(),
            &half_rgba_bytes([0.0, 0.0, 0.0, 1.0]),
        );
        write(
            target.surface_material_a_texture(),
            &half_rgba_bytes(sampled_a),
        );
        write(
            target.surface_material_b_texture(),
            &half_rgba_bytes(sampled_b),
        );
        write(
            target.surface_material_c_texture(),
            &u32::from_le_bytes(sampled_c).to_le_bytes(),
        );

        let mut encoder = CommandEncoder::new();
        encoder.push_compute_pass(None, composite_commands(f.pass));
        queue.submit(&[encoder.finish()]);
        assert!(
            queue.wait_idle(),
            "sampled material composite dispatch completes"
        );

        let hdr = queue
            .read_texture_rgba16f(target.texture(), WIDTH, HEIGHT)
            .expect("sampled material HDR readback must complete");
        let response_bytes = queue
            .read_buffer(f.pass.response_output, f.pass.response_output_size_bytes)
            .expect("sampled material ResponseOutput readback must complete");
        let response_words = response_bytes
            .chunks_exact(core::mem::size_of::<f32>())
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("f32 response word")))
            .collect::<Vec<_>>();
        let center = 0usize;

        let sampled_albedo = albedo.map(srgb8_to_linear);
        let mut surface = surface_sample_from_parameters(&GpuSurfaceParameters::default());
        surface.base_color = sampled_albedo[..3].try_into().unwrap();
        surface.metallic = sampled_a[3];
        surface.perceptual_roughness = sampled_b[0];
        surface.ambient_occlusion = sampled_b[1];
        surface.clearcoat = sampled_b[2];
        surface.normal_variance = sampled_b[3];
        surface.clearcoat_roughness = f32::from(sampled_c[2]) / 255.0;
        surface.coat_normal = decode_sampled_coat_normal([sampled_c[0], sampled_c[1]]);

        let normal = [0.0, 1.0, 0.0];
        let tangent = normalize(cross([1.0, 0.0, 0.0], normal), [1.0, 0.0, 0.0]);
        let bitangent = normalize(cross(normal, tangent), [0.0, 0.0, 1.0]);
        let frame = ShadingFrame {
            normal,
            view: normal,
            tangent,
            bitangent,
        };
        let light = DirectLightSample {
            direction: [0.0, 1.0, 0.0],
            illuminance: [4.0, 2.0, 1.0],
            visibility: 1.0,
        };
        let expected = evaluate_principled_direct(surface, frame, light);
        let neutral = evaluate_principled_direct(
            surface_sample_from_parameters(&GpuSurfaceParameters {
                base_color: [sampled_albedo[0], sampled_albedo[1], sampled_albedo[2], 1.0],
                ..GpuSurfaceParameters::default()
            }),
            frame,
            light,
        );
        assert!(
            expected
                .into_iter()
                .zip(neutral)
                .any(|(sampled, neutral)| (sampled - neutral).abs() > 0.01),
            "non-neutral sampled A/B/C payload must change the golden response"
        );
        for channel in 0..3 {
            assert!(
                (hdr[center][channel] - expected[channel]).abs() <= 0.003,
                "sampled material HDR channel {channel}: GPU {} != CPU {}",
                hdr[center][channel],
                expected[channel]
            );
            assert!(
                (response_words[channel] - expected[channel]).abs() <= 0.003,
                "sampled material ResponseOutput channel {channel}: GPU {} != CPU {}",
                response_words[channel],
                expected[channel]
            );
        }
        assert_eq!(hdr[center][3], 1.0, "sampled material HDR coverage");
        assert_eq!(
            response_words[15], 1.0,
            "sampled material response coverage"
        );
    }

    fn normalize(value: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
        let length_squared = value
            .iter()
            .map(|component| component * component)
            .sum::<f32>();
        if length_squared > 1.0e-12 {
            let inverse_length = length_squared.sqrt().recip();
            return value.map(|component| component * inverse_length);
        }
        fallback
    }

    fn cross(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
        [
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        ]
    }

    fn canonical_principled_reference(
        position: [f32; 3],
        normal: [f32; 3],
        base_color: [f32; 3],
    ) -> [f32; 3] {
        let reference = if normal[1].abs() > 0.9 {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        let tangent = normalize(cross(reference, normal), [1.0, 0.0, 0.0]);
        let bitangent = normalize(cross(normal, tangent), [0.0, 0.0, 1.0]);
        let view = normalize([-position[0], -position[1], -position[2]], normal);
        let frame = ShadingFrame {
            normal,
            view,
            tangent,
            bitangent,
        };
        let parameters = GpuSurfaceParameters {
            base_color: [base_color[0], base_color[1], base_color[2], 1.0],
            ..GpuSurfaceParameters::default()
        };
        evaluate_principled_direct(
            surface_sample_from_parameters(&parameters),
            frame,
            DirectLightSample {
                direction: [0.0, 1.0, 0.0],
                illuminance: [4.0, 2.0, 1.0],
                visibility: 1.0,
            },
        )
    }

    fn canonical_stylized_reference(
        position: [f32; 3],
        normal: [f32; 3],
        base_color: [f32; 3],
    ) -> [f32; 3] {
        let reference = if normal[1].abs() > 0.9 {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        let tangent = normalize(cross(reference, normal), [1.0, 0.0, 0.0]);
        let bitangent = normalize(cross(normal, tangent), [0.0, 0.0, 1.0]);
        let frame = ShadingFrame {
            normal,
            view: normalize([-position[0], -position[1], -position[2]], normal),
            tangent,
            bitangent,
        };
        let parameters = GpuSurfaceParameters {
            base_color: [base_color[0], base_color[1], base_color[2], 1.0],
            ..GpuSurfaceParameters::default()
        };
        evaluate_stylized_direct(
            surface_sample_from_parameters(&parameters),
            frame,
            DirectLightSample {
                direction: [0.0, 1.0, 0.0],
                illuminance: [4.0, 2.0, 1.0],
                visibility: 1.0,
            },
            &StylizedParams::default(),
        )
    }

    fn composite_commands(pass: LightingGpuPass) -> Vec<ComputeCommand> {
        let [x, y, z] = [
            WIDTH.div_ceil(COMPOSITE_WORKGROUP_X),
            HEIGHT.div_ceil(COMPOSITE_WORKGROUP_Y),
            1,
        ];
        vec![
            ComputeCommand::SetPipeline(pass.surface_identity_pipeline),
            ComputeCommand::SetBindGroup {
                index: 0,
                bind_group: pass.surface_identity_bind_group.unwrap(),
                dynamic_offsets: vec![],
            },
            ComputeCommand::Dispatch { x, y, z },
            ComputeCommand::SetPipeline(pass.composite_pipeline),
            ComputeCommand::SetBindGroup {
                index: 0,
                bind_group: pass.composite_bind_group,
                dynamic_offsets: vec![],
            },
            ComputeCommand::SetBindGroup {
                index: 1,
                bind_group: pass.composite_target_bind_group.unwrap(),
                dynamic_offsets: vec![],
            },
            ComputeCommand::SetBindGroup {
                index: 2,
                bind_group: pass.composite_material_bind_group.unwrap(),
                dynamic_offsets: vec![],
            },
            ComputeCommand::Dispatch { x, y, z },
        ]
    }

    // A GPU reader, not a string test or replacement for the production
    // composite. It uses the SAME target layout/group and preserves float bits
    // and u32 identity in a storage-buffer readback, including sRGB decoding.
    const READ_GBUFFER: &str = r#"
@group(0) @binding(0) var<storage, read_write> output: array<vec4<u32>>;
@group(1) @binding(0) var px: texture_2d<f32>;
@group(1) @binding(1) var py: texture_2d<f32>;
@group(1) @binding(2) var pz: texture_2d<f32>;
@group(1) @binding(3) var normal: texture_2d<f32>;
@group(1) @binding(4) var albedo: texture_2d<f32>;
@group(1) @binding(5) var surface_id: texture_2d<u32>;
@group(1) @binding(10) var sampled_material_a: texture_2d<f32>;
@group(1) @binding(11) var sampled_material_b: texture_2d<f32>;
@group(1) @binding(12) var sampled_material_c: texture_2d<u32>;
struct SurfaceIdentityProbe {
    material_index: u32,
    generation: u32,
    revision: u32,
    execution_plan_index: u32,
    program_kind: u32,
    lighting_surface_index: u32,
    left_program_index: u32,
    right_program_index: u32,
    left_program_kind: u32,
    right_program_kind: u32,
    blended_channels: u32,
    mask_kind: u32,
    mask_index: u32,
    blend_domain: u32,
    left_program_id: u32,
    right_program_id: u32,
};

@group(1) @binding(6) var<storage, read> surface_identity: array<SurfaceIdentityProbe>;
@group(1) @binding(7) var hdr: texture_storage_2d<rgba16float, write>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let extent = textureDimensions(hdr);
    if (id.x >= extent.x || id.y >= extent.y) { return; }
    let p = vec2<i32>(id.xy);
    let n = textureLoad(normal, p, 0);
    let position = vec3<f32>(textureLoad(px, p, 0).x,
        textureLoad(py, p, 0).x, textureLoad(pz, p, 0).x);
    let base = (id.y * extent.x + id.x) * 6u;
    output[base] = vec4<u32>(bitcast<vec3<u32>>(position), bitcast<u32>(n.w));
    output[base + 1u] = vec4<u32>(bitcast<vec3<u32>>(n.xyz), textureLoad(surface_id, p, 0).x);
    output[base + 2u] = bitcast<vec4<u32>>(textureLoad(albedo, p, 0));
    output[base + 3u] = bitcast<vec4<u32>>(textureLoad(sampled_material_a, p, 0));
    output[base + 4u] = bitcast<vec4<u32>>(textureLoad(sampled_material_b, p, 0));
    output[base + 5u] = vec4<u32>(textureLoad(sampled_material_c, p, 0).x, 0u, 0u, 0u);
}
"#;

    #[test]
    fn physical_gpu_surface_encode_gbuffer_readback_and_composite_close_the_contract() {
        let Some(f) = fixture() else {
            return;
        };
        let device = f.runtime.device();
        let queue = f.runtime.queue();
        let target = f.target;
        let provider = SurfaceEncodeProvider::new(device);
        let vertex_layout = rhi::VertexBufferLayout {
            array_stride: 32,
            step_mode: rhi::VertexStepMode::Vertex,
            attributes: vec![
                rhi::VertexAttribute {
                    format: rhi::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                rhi::VertexAttribute {
                    format: rhi::VertexFormat::Float32x3,
                    offset: 12,
                    shader_location: 1,
                },
                rhi::VertexAttribute {
                    format: rhi::VertexFormat::Float32x2,
                    offset: 24,
                    shader_location: 2,
                },
            ],
        };
        let key = crate::mesh_pipeline::MeshPipelineKey {
            execution_family:
                prism_render_architecture::shader_package::ExecutionFamily::FusedBuiltin,
            response_module_id: prism_render_architecture::shader_package::ModuleId::SPINE,
            permutation_index: prism_render_architecture::shader_package::PermutationIndex::ZERO,
            fallback_permutation: None,
            color_format: Some(target.normal_format()),
            depth_format: None,
            sample_count: 1,
            topology: rhi::PrimitiveTopology::TriangleList,
        };
        let mut geometry_descriptor = provider
            .try_descriptor(&key, &[vertex_layout.clone()])
            .unwrap();
        geometry_descriptor.vertex.buffers = vec![vertex_layout.clone()];
        let geometry_pipeline = device.create_render_pipeline(&geometry_descriptor);
        let mut material_descriptor = provider
            .secondary_descriptor(&key)
            .expect("surface encode must publish the material attachment pipeline");
        material_descriptor.vertex.buffers = vec![vertex_layout];
        let material_pipeline = device.create_render_pipeline(&material_descriptor);
        // All components exceed finite binary16 range; z also has fractional
        // bits. An f16 intermediate or component swizzle cannot pass readback.
        let origin = [131_072.0_f32, -262_144.0, 1_048_576.25];
        let vertices: Vec<[f32; 8]> = [
            ([-0.5, -0.5], [0.0, 0.0]),
            ([0.5, -0.5], [1.0, 0.0]),
            ([0.5, 0.5], [1.0, 1.0]),
            ([-0.5, -0.5], [0.0, 0.0]),
            ([0.5, 0.5], [1.0, 1.0]),
            ([-0.5, 0.5], [0.0, 1.0]),
        ]
        .into_iter()
        .map(|([x, y], [u, v])| [origin[0] + x, origin[1] + y, origin[2], 0.6, 0.8, 0.0, u, v])
        .collect();
        let vertex = device.create_buffer(&BufferDescriptor::new(
            core::mem::size_of_val(vertices.as_slice()) as u64,
            BufferUsages::VERTEX | BufferUsages::COPY_DST,
        ));
        queue.write_buffer(vertex, 0, cast_slice(&vertices));
        let frame = device.create_buffer(&BufferDescriptor::new(
            SurfaceFrame::SIZE_BYTES,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        ));
        queue.write_buffer(
            frame,
            0,
            bytes_of(&SurfaceFrame {
                clip_from_world: [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [-origin[0], -origin[1], 0.5, 1.0],
                ],
                previous_clip_from_world: [[0.0; 4]; 4],
                exposure_stops: 0.0,
                camera_history_valid: 0,
                _padding: [0.0; 2],
            }),
        );
        let material = device.create_buffer(&BufferDescriptor::new(
            SurfaceMaterial::SIZE_BYTES,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        ));
        let transparent_albedo = [0.003_130_8, 0.18, 0.72, 0.0];
        let headers = f.headers;
        let parameters = f.parameters;
        let textures = device.create_buffer(&BufferDescriptor::new(
            core::mem::size_of::<GpuMaterialTexture>() as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        ));
        // The surface pipeline has an unconditional portable-VT group-1 ABI.
        // Keep this fixture explicit: a disabled, one-page slice is the
        // smallest valid fallback and avoids relying on implicit bind-group
        // defaults that real materials are not allowed to use.
        let vt_page_table = device.create_buffer(&BufferDescriptor::new(
            16,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        ));
        let vt_config = device.create_buffer(&BufferDescriptor::new(
            32,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        ));
        queue.write_buffer(vt_page_table, 0, &[0; 16]);
        queue.write_buffer(vt_config, 0, &[0; 32]);
        let vt_atlas = device.create_texture(&TextureDescriptor {
            label: Some("gbuffer-contract-vt-atlas".into()),
            size: Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        });
        let vt_atlas_view = device.create_texture_view(
            vt_atlas,
            &TextureViewDescriptor {
                dimension: TextureViewDimension::D2Array,
                ..TextureViewDescriptor::default()
            },
        );
        let vt_sampler = device.create_sampler(&SamplerDescriptor::default());
        let vt_group = device.create_bind_group(&provider.vt_bind_group_descriptor(
            vt_page_table,
            vt_config,
            vt_atlas_view,
            vt_sampler,
        ));
        let group = device.create_bind_group(
            &provider
                .bind_group_descriptor(SurfaceResourceSet {
                    frame: Some(frame),
                    material: Some(material),
                    headers: Some(headers),
                    parameters: Some(parameters),
                    textures: Some(textures),
                })
                .unwrap(),
        );
        let probe_bytes = u64::from(WIDTH * HEIGHT) * 96;
        let probe = device.create_buffer(&BufferDescriptor::new(
            probe_bytes,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        ));
        let probe_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: vec![storage_layout(0, probe_bytes, false)],
        });
        let probe_group = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: Some(probe_layout),
            entries: vec![buffer_binding(0, probe, probe_bytes)],
        });
        let probe_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: vec![probe_layout, f.pass.composite_target_layout()],
            push_constant_ranges: vec![],
        });
        let probe_shader = device.create_shader_module(&ShaderModuleDescriptor::wgsl(READ_GBUFFER));
        let probe_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            layout: Some(probe_pipeline_layout),
            module: probe_shader,
            entry_point: "main".into(),
        });
        let mut encoder = CommandEncoder::new();
        encoder.push_render_pass(
            rhi::RenderPassDescriptor {
                label: Some("gbuffer-contract-geometry-raster".into()),
                color_attachments: [
                    target.position_x_view(),
                    target.position_y_view(),
                    target.position_z_view(),
                    target.normal_view(),
                    target.albedo_view(),
                ]
                .into_iter()
                .map(|view| {
                    Some(rhi::ColorAttachment {
                        view,
                        resolve_target: None,
                        load: rhi::LoadOp::Clear(rhi::Color::TRANSPARENT),
                        store: rhi::StoreOp::Store,
                    })
                })
                .collect(),
                depth_stencil_attachment: None,
            },
            vec![
                RenderCommand::SetPipeline(geometry_pipeline),
                RenderCommand::SetBindGroup {
                    index: 0,
                    bind_group: group,
                    dynamic_offsets: vec![0, 0],
                },
                RenderCommand::SetBindGroup {
                    index: 1,
                    bind_group: vt_group,
                    dynamic_offsets: vec![],
                },
                RenderCommand::SetVertexBuffer {
                    slot: 0,
                    buffer: vertex,
                    offset: 0,
                    size: None,
                },
                RenderCommand::Draw {
                    vertices: 0..6,
                    instances: 0..1,
                },
            ],
        );
        encoder.push_render_pass(
            rhi::RenderPassDescriptor {
                label: Some("gbuffer-contract-material-raster".into()),
                color_attachments: [
                    target.surface_id_view(),
                    target.surface_data_view(),
                    target.surface_material_a_view(),
                    target.surface_material_b_view(),
                    target.surface_material_c_view(),
                ]
                .into_iter()
                .map(|view| {
                    Some(rhi::ColorAttachment {
                        view,
                        resolve_target: None,
                        load: rhi::LoadOp::Clear(rhi::Color::TRANSPARENT),
                        store: rhi::StoreOp::Store,
                    })
                })
                .collect(),
                depth_stencil_attachment: None,
            },
            vec![
                RenderCommand::SetPipeline(material_pipeline),
                RenderCommand::SetBindGroup {
                    index: 0,
                    bind_group: group,
                    dynamic_offsets: vec![0, 0],
                },
                RenderCommand::SetBindGroup {
                    index: 1,
                    bind_group: vt_group,
                    dynamic_offsets: vec![],
                },
                RenderCommand::SetVertexBuffer {
                    slot: 0,
                    buffer: vertex,
                    offset: 0,
                    size: None,
                },
                RenderCommand::Draw {
                    vertices: 0..6,
                    instances: 0..1,
                },
            ],
        );
        encoder.push_compute_pass(
            None,
            vec![
                ComputeCommand::SetPipeline(probe_pipeline),
                ComputeCommand::SetBindGroup {
                    index: 0,
                    bind_group: probe_group,
                    dynamic_offsets: vec![],
                },
                ComputeCommand::SetBindGroup {
                    index: 1,
                    bind_group: f.pass.composite_target_bind_group.unwrap(),
                    dynamic_offsets: vec![],
                },
                ComputeCommand::Dispatch {
                    x: WIDTH.div_ceil(8),
                    y: HEIGHT.div_ceil(8),
                    z: 1,
                },
            ],
        );
        encoder.push_compute_pass(None, composite_commands(f.pass));
        let commands = encoder.finish();
        // These clears specify the producer precondition; the separate real
        // target_graph regression below checks the host clear (no substitution).
        // Raster identities must address actual header rows. Arbitrary high
        // u32 identities are still covered by the composite injection below.
        for (surface_id, albedo, geometry_opacity) in [
            (1, transparent_albedo, 1.0),
            (2, transparent_albedo, 1.0),
            (3, transparent_albedo, 1.0),
            (1, [-0.25, 1.25, 0.5, 0.37], 0.5),
        ] {
            let (header, heap) = canonical_surface(albedo, geometry_opacity);
            let mut rows = [GpuMaterialHeader::default(); 4];
            rows[surface_id as usize] = header;
            queue.write_buffer(headers, 0, cast_slice(&rows));
            queue.write_buffer(parameters, 0, cast_slice(&heap));
            queue.write_buffer(
                material,
                0,
                bytes_of(&SurfaceMaterial {
                    index: surface_id,
                    generation: header.generation,
                    revision: header.revision,
                    execution_plan_index: header.execution_plan_index,
                }),
            );
            queue.submit(core::slice::from_ref(&commands));
            assert!(queue.wait_idle());
            let bytes = queue
                .read_buffer(probe, probe_bytes)
                .expect("lighting probe GPU readback must complete");
            let words: Vec<u32> = bytes
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            let hdr = queue
                .read_texture_rgba16f(target.texture(), WIDTH, HEIGHT)
                .expect("lighting HDR GPU readback must complete");
            let response_bytes = queue
                .read_buffer(f.pass.response_output, f.pass.response_output_size_bytes)
                .expect("lighting ResponseOutput GPU readback must complete");
            let response_words = response_bytes
                .chunks_exact(core::mem::size_of::<f32>())
                .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("f32-sized response word")))
                .collect::<Vec<_>>();
            assert_eq!(
                response_words.len(),
                (WIDTH * HEIGHT * 16) as usize,
                "ResponseOutput storage has one canonical 16-word record per target pixel"
            );
            let encoded = queue
                .read_texture_rgba8(target.albedo_texture(), WIDTH, HEIGHT)
                .expect("lighting albedo GPU readback must complete");
            for y in 0..HEIGHT {
                for x in 0..WIDTH {
                    let index = (y * WIDTH + x) as usize;
                    let sample = &words[index * 24..(index + 1) * 24];
                    let response = &response_words[index * 16..(index + 1) * 16];
                    let covered = (2..=6).contains(&x) && (2..=4).contains(&y);
                    assert_eq!(
                        f32::from_bits(sample[3]),
                        if covered { 1.0 } else { 0.0 },
                        "coverage at {x},{y}"
                    );
                    if !covered {
                        assert_eq!(
                            hdr[index], [0.0; 4],
                            "background must not read surface zero"
                        );
                        assert!(
                            response.iter().all(|word| *word == 0.0),
                            "background ResponseOutput must be the canonical zero record at {x},{y}"
                        );
                        continue;
                    }
                    let clip_x = (x as f32 + 0.5) * 2.0 / WIDTH as f32 - 1.0;
                    let clip_y = 1.0 - (y as f32 + 0.5) * 2.0 / HEIGHT as f32;
                    for (channel, expected) in [origin[0] + clip_x, origin[1] + clip_y, origin[2]]
                        .into_iter()
                        .enumerate()
                    {
                        // Allow f32 raster interpolation rounding, never f16
                        // position quantization or a depth reconstruction.
                        assert!(
                            (f32::from_bits(sample[channel]) - expected).abs() <= 0.125,
                            "position channel {channel} at {x},{y}"
                        );
                    }
                    let expected_normal = [0.6, 0.8, 0.0];
                    for (channel, expected) in expected_normal.into_iter().enumerate() {
                        assert!((f32::from_bits(sample[4 + channel]) - expected).abs() <= 0.0005);
                    }
                    assert_eq!(sample[7], surface_id, "canonical header row identity");
                    // The neutral OpenPBR emission color is white; its zero
                    // luminance keeps the default material non-emissive.
                    for channel in 12..15 {
                        assert_eq!(f32::from_bits(sample[channel]), 1.0);
                    }
                    assert_eq!(f32::from_bits(sample[15]), 0.0, "sampled metallic");
                    // RGBA16Float stores the authored roughness with half-float
                    // quantization; validate the format error, not f32 identity.
                    assert!(
                        (f32::from_bits(sample[16]) - 0.3).abs() <= 0.001,
                        "sampled roughness"
                    );
                    assert_eq!(f32::from_bits(sample[17]), 1.0, "sampled occlusion");
                    assert_eq!(f32::from_bits(sample[18]), 0.0, "sampled coat");
                    assert_eq!(f32::from_bits(sample[19]), 0.0, "sampled normal variance");
                    // The default clearcoat normal is the geometry normal
                    // [0.6, 0.8, 0.0], quantized into four UNORM8 lanes in
                    // the R32Uint C plane and decoded by textureLoad.
                    let packed_c = sample[20];
                    assert_eq!(packed_c & 0xff, 182, "sampled coat normal x");
                    assert_eq!((packed_c >> 8) & 0xff, 200, "sampled coat normal y");
                    assert_eq!((packed_c >> 16) & 0xff, 0, "sampled clearcoat roughness");
                    assert_eq!((packed_c >> 24) & 0xff, 255, "sampled material C validity");
                    assert_eq!(sample[21], 0);
                    assert_eq!(sample[22], 0);
                    assert_eq!(sample[23], 0);
                    let mut linear_albedo = [0.0; 3];
                    for channel in 0..3 {
                        let linear = albedo[channel].clamp(0.0, 1.0);
                        let srgb = if linear <= 0.003_130_8 {
                            12.92 * linear
                        } else {
                            1.055 * linear.powf(1.0 / 2.4) - 0.055
                        };
                        let stored = encoded[index * 4 + channel];
                        assert!((f32::from(stored) - (srgb * 255.0).round()).abs() <= 1.0);
                        let v = f32::from(stored) / 255.0;
                        let decoded = if v <= 0.04045 {
                            v / 12.92
                        } else {
                            ((v + 0.055) / 1.055).powf(2.4)
                        };
                        assert!(
                            (f32::from_bits(sample[8 + channel]) - decoded).abs() <= 0.001,
                            "textureLoad must decode sRGB"
                        );
                        linear_albedo[channel] = decoded;
                    }
                    let stored_alpha = encoded[index * 4 + 3];
                    assert!(
                        (f32::from(stored_alpha) - (albedo[3] * geometry_opacity * 255.0).round())
                            .abs()
                            <= 1.0,
                        "opacity is linear UNORM8, not sRGB or coverage"
                    );
                    assert!(
                        (f32::from_bits(sample[11]) - f32::from(stored_alpha) / 255.0).abs()
                            <= f32::EPSILON,
                        "textureLoad alpha must stay linear, including opacity zero"
                    );
                    let cpu_reference = canonical_principled_reference(
                        [
                            f32::from_bits(sample[0]),
                            f32::from_bits(sample[1]),
                            f32::from_bits(sample[2]),
                        ],
                        [
                            f32::from_bits(sample[4]),
                            f32::from_bits(sample[5]),
                            f32::from_bits(sample[6]),
                        ],
                        linear_albedo,
                    );
                    for channel in 0..3 {
                        assert!(
                            (hdr[index][channel] - cpu_reference[channel]).abs() <= 0.002,
                            "per-pixel direct channel {channel}, surface id {surface_id}: GPU {} != CPU {}",
                            hdr[index][channel],
                            cpu_reference[channel]
                        );
                        assert!(
                            (response[channel] - cpu_reference[channel]).abs() <= 0.002,
                            "ResponseOutput direct channel {channel}, surface id {surface_id}: GPU {} != CPU {}",
                            response[channel],
                            cpu_reference[channel]
                        );
                    }
                    for channel in 3..15 {
                        assert_eq!(
                            response[channel],
                            0.0,
                            "canonical direct-only fixture must not populate response lane {channel}"
                        );
                    }
                    assert_eq!(response[15], 1.0, "opaque response alpha");
                    assert_eq!(hdr[index][3], 1.0);
                }
            }
        }

        let (valid_header, heap) = canonical_surface(transparent_albedo, 1.0);
        let identity = SurfaceMaterial {
            index: 1,
            generation: valid_header.generation,
            revision: valid_header.revision,
            execution_plan_index: valid_header.execution_plan_index,
        };
        let unknown_lobes = !LobeMask::all().bits();
        assert_ne!(unknown_lobes, 0, "fixture needs an unknown lobe bit");
        let unknown_lobe_bit = 1_u32 << unknown_lobes.trailing_zeros();
        let core_size = (SurfaceParameterBlock::from_full(
            &GpuSurfaceParameters::default(),
            LobeMask::default(),
        )
        .pack()
        .len()
            * 4) as u32;
        let upload_material = |header: GpuMaterialHeader, draw: SurfaceMaterial| {
            let mut rows = [GpuMaterialHeader::default(); 4];
            rows[1] = header;
            queue.write_buffer(headers, 0, cast_slice(&rows));
            queue.write_buffer(parameters, 0, cast_slice(&heap));
            queue.write_buffer(material, 0, bytes_of(&draw));
            queue.submit(core::slice::from_ref(&commands));
            assert!(queue.wait_idle());
        };
        upload_material(valid_header, identity);
        let valid_probe = queue
            .read_buffer(probe, probe_bytes)
            .expect("valid lighting probe readback must complete");
        let valid_hdr = queue
            .read_texture_rgba16f(target.texture(), WIDTH, HEIGHT)
            .expect("valid lighting HDR readback must complete");
        let center = ((HEIGHT / 2) * WIDTH + WIDTH / 2) as usize;
        let center_word = |word: usize| {
            u32::from_le_bytes(
                valid_probe[center * 96 + word * 4..center * 96 + (word + 1) * 4]
                    .try_into()
                    .unwrap(),
            )
        };
        assert_eq!(center_word(3), 1.0_f32.to_bits(), "live row coverage");
        assert_eq!(center_word(7), identity.index, "live row identity");
        // Match the composite's decoded GBuffer inputs, including binary16
        // normal quantization and the textureLoad sRGB-to-linear conversion.
        let cpu_reference = canonical_principled_reference(
            core::array::from_fn(|channel| f32::from_bits(center_word(channel))),
            core::array::from_fn(|channel| f32::from_bits(center_word(4 + channel))),
            core::array::from_fn(|channel| f32::from_bits(center_word(8 + channel))),
        );
        for (channel, expected) in cpu_reference.into_iter().enumerate() {
            assert!(
                (valid_hdr[center][channel] - expected).abs() <= 0.002,
                "live row direct channel {channel}: GPU {} != CPU {expected}",
                valid_hdr[center][channel]
            );
        }
        assert_eq!(valid_hdr[center][3], 1.0, "live row HDR alpha");

        // Switch the same live raster pixel to the builtin NPR execution row.
        // This exercises the real identity reconstruction and Native composite
        // path; it is not a source-string or synthetic response check.
        let (npr_header, npr_heap) = canonical_npr_surface(transparent_albedo, 1.0);
        let physical_heap_words = npr_heap.len() as u32;
        let npr_identity = SurfaceMaterial {
            index: 1,
            generation: npr_header.generation,
            revision: npr_header.revision,
            execution_plan_index: npr_header.execution_plan_index,
        };
        let mut npr_rows = [GpuMaterialHeader::default(); 4];
        npr_rows[1] = npr_header;
        queue.write_buffer(headers, 0, cast_slice(&npr_rows));
        queue.write_buffer(parameters, 0, cast_slice(&npr_heap));
        queue.write_buffer(material, 0, bytes_of(&npr_identity));
        queue.submit(core::slice::from_ref(&commands));
        assert!(queue.wait_idle());
        let npr_hdr = queue
            .read_texture_rgba16f(target.texture(), WIDTH, HEIGHT)
            .expect("NPR lighting HDR readback must complete");
        let npr_response = queue
            .read_buffer(f.pass.response_output, f.pass.response_output_size_bytes)
            .expect("NPR ResponseOutput readback must complete");
        let npr_response_words = npr_response
            .chunks_exact(core::mem::size_of::<f32>())
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("f32 response word")))
            .collect::<Vec<_>>();
        let npr_reference = canonical_stylized_reference(
            core::array::from_fn(|channel| f32::from_bits(center_word(channel))),
            core::array::from_fn(|channel| f32::from_bits(center_word(4 + channel))),
            core::array::from_fn(|channel| f32::from_bits(center_word(8 + channel))),
        );
        for (channel, expected) in npr_reference.into_iter().enumerate() {
            assert!(
                (npr_hdr[center][channel] - expected).abs() <= 0.002,
                "NPR direct channel {channel}: GPU {} != CPU {expected}",
                npr_hdr[center][channel]
            );
            assert!(
                (npr_response_words[center * 16 + channel] - expected).abs() <= 0.002,
                "NPR ResponseOutput direct channel {channel}: GPU {} != CPU {expected}",
                npr_response_words[center * 16 + channel]
            );
        }
        for channel in 3..15 {
            assert_eq!(
                npr_response_words[center * 16 + channel],
                0.0,
                "NPR must not populate unrelated response lane {channel}"
            );
        }
        assert_eq!(npr_response_words[center * 16 + 15], 1.0);
        assert_eq!(npr_hdr[center][3], 1.0);

        // Select a canonical fused-builtin MaskBlend plan. Both endpoints are
        // Lit programs, and the dynamic wetness channel supplies a 0.5 mask.
        // Readback proves that the identity prepass and the response-channel
        // composition both consume the same plan row; split-only NPR is not
        // allowed to masquerade as this family.
        let mut blend_header = npr_header;
        blend_header.execution_plan_index = 2;
        blend_header.execution_program_index = 0;
        blend_header.dynamic_wetness = 0.5_f32.to_bits();
        let blend_identity = SurfaceMaterial {
            index: 1,
            generation: blend_header.generation,
            revision: blend_header.revision,
            execution_plan_index: blend_header.execution_plan_index,
        };
        let mut blend_rows = [GpuMaterialHeader::default(); 4];
        blend_rows[1] = blend_header;
        queue.write_buffer(headers, 0, cast_slice(&blend_rows));
        queue.write_buffer(parameters, 0, cast_slice(&npr_heap));
        queue.write_buffer(material, 0, bytes_of(&blend_identity));
        queue.submit(core::slice::from_ref(&commands));
        assert!(queue.wait_idle());
        let blend_hdr = queue
            .read_texture_rgba16f(target.texture(), WIDTH, HEIGHT)
            .expect("MaskBlend lighting HDR readback must complete");
        let blend_response = queue
            .read_buffer(f.pass.response_output, f.pass.response_output_size_bytes)
            .expect("MaskBlend ResponseOutput readback must complete");
        let blend_response_words = blend_response
            .chunks_exact(core::mem::size_of::<f32>())
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("f32 response word")))
            .collect::<Vec<_>>();
        for channel in 0..3 {
            let expected = valid_hdr[center][channel];
            assert!(
                (blend_hdr[center][channel] - expected).abs() <= 0.002,
                "MaskBlend HDR channel {channel}: GPU {} != CPU {expected}",
                blend_hdr[center][channel]
            );
            assert!(
                (blend_response_words[center * 16 + channel] - expected).abs() <= 0.002,
                "MaskBlend ResponseOutput channel {channel}: GPU {} != CPU {expected}",
                blend_response_words[center * 16 + channel]
            );
        }
        for channel in 3..15 {
            assert_eq!(
                blend_response_words[center * 16 + channel],
                0.0,
                "direct-only MaskBlend must preserve unrelated response lane {channel}"
            );
        }
        assert_eq!(blend_response_words[center * 16 + 15], 1.0);
        assert_eq!(blend_hdr[center][3], 1.0);

        // An out-of-range dynamic mask is malformed input, not a value to
        // clamp into a plausible composition. The consumer must clear both
        // public outputs.
        blend_header.dynamic_wetness = 1.5_f32.to_bits();
        queue.write_buffer(
            headers,
            0,
            cast_slice(&[
                GpuMaterialHeader::default(),
                blend_header,
                GpuMaterialHeader::default(),
                GpuMaterialHeader::default(),
            ]),
        );
        queue.submit(core::slice::from_ref(&commands));
        assert!(queue.wait_idle());
        let invalid_mask_hdr = queue
            .read_texture_rgba16f(target.texture(), WIDTH, HEIGHT)
            .expect("invalid MaskBlend mask HDR readback must complete");
        let invalid_mask_response = queue
            .read_buffer(f.pass.response_output, f.pass.response_output_size_bytes)
            .expect("invalid MaskBlend mask ResponseOutput readback must complete");
        assert_eq!(invalid_mask_hdr[center], [0.0; 4]);
        assert!(invalid_mask_response
            .chunks_exact(core::mem::size_of::<f32>())
            .skip(center * 16)
            .take(16)
            .all(|word| f32::from_le_bytes(word.try_into().unwrap()) == 0.0));

        upload_material(
            GpuMaterialHeader {
                lobe_mask: 0,
                parameter_size: core_size,
                ..valid_header
            },
            identity,
        );
        assert_eq!(
            queue
                .read_buffer(probe, probe_bytes)
                .expect("core-only lighting probe readback must complete"),
            valid_probe,
            "a core-only row with trailing heap words is valid"
        );
        assert_eq!(
            queue
                .read_texture_rgba16f(target.texture(), WIDTH, HEIGHT)
                .expect("core-only lighting HDR readback must complete"),
            valid_hdr,
            "absent emission must not change the surface-encode core"
        );
        // These inputs must reject independently of rendering/clearing state.
        // Each malformed draw is followed by the same live row, so a stuck
        // discard path or stale buffer/bind-group contents cannot pass.
        for case in [
            "generation",
            "generation_high_bit",
            "revision",
            "revision_high_bit",
            "plan",
            "plan_high_bit",
            "plan_sentinel",
            "inactive",
            "index",
            "index_at_header_count",
            "dormant_header_row",
            "size",
            "size_oversized_word",
            "size_short_byte",
            "size_extra_byte",
            "size_zero",
            "size_max",
            "size_max_aligned",
            "offset",
            "heap_end",
            "past_heap_end",
            "heap_one_word_short",
            "unknown_lobe_with_valid_size",
            "unknown_high_lobe_with_valid_size",
            "unknown_lobe_only_with_core_size",
            "known_lobe_with_core_size",
        ] {
            let mut header = valid_header;
            let mut draw = identity;
            match case {
                "generation" => draw.generation ^= 1,
                "generation_high_bit" => draw.generation ^= 1 << 31,
                "revision" => draw.revision ^= 1,
                "revision_high_bit" => draw.revision ^= 1 << 31,
                "plan" => draw.execution_plan_index ^= 1,
                "plan_high_bit" => draw.execution_plan_index ^= 1 << 31,
                "plan_sentinel" => draw.execution_plan_index = u32::MAX,
                "inactive" => header.is_active = 0,
                "index" => draw.index = u32::MAX,
                "index_at_header_count" => draw.index = 4,
                "dormant_header_row" => draw.index = 0,
                "size" => header.parameter_size -= 4,
                "size_oversized_word" => header.parameter_size += 4,
                "size_short_byte" => header.parameter_size -= 1,
                "size_extra_byte" => header.parameter_size += 1,
                "size_zero" => header.parameter_size = 0,
                "size_max" => header.parameter_size = u32::MAX,
                "size_max_aligned" => header.parameter_size = u32::MAX - 3,
                "offset" => header.parameter_offset = u32::MAX,
                "heap_end" => header.parameter_offset = heap.len() as u32,
                "past_heap_end" => header.parameter_offset = heap.len() as u32 + 1,
                // This fixture's storage buffer is large enough for the NPR
                // variant, while the current row is Lit. Exercise the GPU
                // binding's physical tail; the CPU fail-closed validator
                // separately covers a logical heap shorter than that binding.
                "heap_one_word_short" => {
                    let packed_words = header.parameter_size / 4;
                    header.parameter_offset = physical_heap_words
                        .checked_sub(packed_words)
                        .expect("fixture physical heap must fit the Lit block")
                        + 1;
                }
                // Keep the known mask, exact packed size and in-range heap
                // untouched: only the unknown-bit guard can reject this row.
                "unknown_lobe_with_valid_size" => header.lobe_mask |= unknown_lobe_bit,
                "unknown_high_lobe_with_valid_size" => {
                    header.lobe_mask |= 1_u32 << (31 - unknown_lobes.leading_zeros());
                }
                // The recognized mask is now empty and its size is correct.
                // This catches consumers that silently mask unknown bits out.
                "unknown_lobe_only_with_core_size" => {
                    header.lobe_mask = unknown_lobe_bit;
                    header.parameter_size = core_size;
                }
                "known_lobe_with_core_size" => header.parameter_size = core_size,
                _ => unreachable!(),
            }
            upload_material(header, draw);
            let rejected_probe = queue
                .read_buffer(probe, probe_bytes)
                .expect("rejected lighting probe readback must complete");
            assert_eq!(rejected_probe.len(), probe_bytes as usize);
            assert!(
                rejected_probe.iter().all(|byte| *byte == 0),
                "canonical material {case} must leave every GBuffer attachment cleared"
            );
            let hdr = queue
                .read_texture_rgba16f(target.texture(), WIDTH, HEIGHT)
                .expect("rejected lighting HDR readback must complete");
            assert!(
                hdr.iter().all(|pixel| *pixel == [0.0; 4]),
                "canonical material {case} must discard and leave no coverage or lighting"
            );
            upload_material(valid_header, identity);
            assert_eq!(
                queue
                    .read_buffer(probe, probe_bytes)
                    .expect("recovered lighting probe readback must complete"),
                valid_probe,
                "canonical material must recover after {case}"
            );
            assert_eq!(
                queue
                    .read_texture_rgba16f(target.texture(), WIDTH, HEIGHT)
                    .expect("recovered lighting HDR readback must complete"),
                valid_hdr,
                "canonical material lighting must recover after {case}"
            );
        }

        // Origin is a valid covered position; id zero is valid too. Populate
        // every fixture header row so ids 0..=2 exercise valid material rows,
        // while the later large ids still test the bounds guard.
        queue.write_buffer(headers, 0, cast_slice(&[valid_header; 4]));

        // Coverage, GI count bounds and finite checks must be independent of
        // both position and material-id values.
        let cases = [
            (false, 0, [0.0; 3]),
            (true, 0, [0.0; 3]),
            (true, 1, [0.0; 3]),
            (true, 2, [0.0; 3]),
            (true, u32::MAX, [0.0; 3]),
            (true, 0xf123_4567, [0.0; 3]),
            (true, 0, [f32::NAN, 0.0, 0.0]),
            (true, 0, [0.0, f32::INFINITY, 0.0]),
            (true, 0, [0.0, 0.0, f32::NEG_INFINITY]),
        ];
        for (x, (covered, id, position)) in cases.into_iter().enumerate() {
            let upload = |texture, bytes: &[u8]| {
                queue.write_texture(
                    rhi::TextureWrite {
                        texture,
                        mip_level: 0,
                        origin: [x as u32, 0, 0],
                        size: rhi::Extent3d::new_2d(1, 1),
                    },
                    bytes,
                    rhi::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(bytes.len() as u32),
                        rows_per_image: Some(1),
                    },
                )
            };
            for (texture, value) in [
                target.position_x_texture(),
                target.position_y_texture(),
                target.position_z_texture(),
            ]
            .into_iter()
            .zip(position)
            {
                upload(texture, &value.to_le_bytes());
            }
            upload(
                target.normal_texture(),
                &[0_u16, 0, 0x3c00, if covered { 0x3c00 } else { 0 }]
                    .into_iter()
                    .flat_map(u16::to_le_bytes)
                    .collect::<Vec<_>>(),
            );
            upload(target.surface_id_texture(), &id.to_le_bytes());
            // The composite contract rejects missing SurfaceData and sampled
            // material planes independently of coverage. Keep these inputs
            // valid so this loop isolates coverage and material-id bounds.
            upload(
                target.surface_data_texture(),
                &[0_u16, 0, 0, 0x3c00]
                    .into_iter()
                    .flat_map(u16::to_le_bytes)
                    .collect::<Vec<_>>(),
            );
            upload(
                target.surface_material_a_texture(),
                &[0_u16, 0, 0, 0]
                    .into_iter()
                    .flat_map(u16::to_le_bytes)
                    .collect::<Vec<_>>(),
            );
            upload(
                target.surface_material_b_texture(),
                &[0x34cd_u16, 0x3c00, 0, 0]
                    .into_iter()
                    .flat_map(u16::to_le_bytes)
                    .collect::<Vec<_>>(),
            );
            upload(
                target.surface_material_c_texture(),
                &0xff008080_u32.to_le_bytes(),
            );
        }
        let mut composite = CommandEncoder::new();
        composite.push_compute_pass(None, composite_commands(f.pass));
        queue.submit(&[composite.finish()]);
        assert!(queue.wait_idle());
        let boundary_identity = queue
            .read_buffer(target.surface_identity_buffer(), 4 * 64)
            .expect("boundary identity readback must complete");
        let boundary_ids = boundary_identity
            .chunks_exact(64)
            .map(|row| u32::from_le_bytes(row[0..4].try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(
            boundary_ids,
            vec![u32::MAX, 0, 1, 2],
            "identity prepass must preserve valid zero-based rows and reject clear"
        );
        let hdr = queue
            .read_texture_rgba16f(target.texture(), WIDTH, HEIGHT)
            .expect("lighting composite HDR readback must complete");
        for (index, pixel) in hdr[..WIDTH as usize].iter().enumerate() {
            assert_eq!(
                *pixel,
                match index {
                    // These covered probes use a normal orthogonal to the
                    // configured directional light. Arbitrary/high surface
                    // IDs are still valid direct pixels and stay black.
                    1..=3 => [0.0, 0.0, 0.0, 1.0],
                    _ => [0.0; 4],
                },
                "composite boundary case {index}"
            );
        }
    }

    /// Intentionally uses the host's actual FrameGraph clear, not the explicit
    /// transparent producer clear above. A BLACK/alpha=1 clear must fail this
    /// regression; it cannot be repaired by changing only shader assertions.
    #[test]
    fn physical_gpu_frame_graph_clear_leaves_no_gbuffer_coverage_or_surface_zero_light() {
        let Some(f) = fixture() else {
            return;
        };
        // Poison existing contents so a missing clear/dispatch cannot pass by
        // observing newly allocated, zero-initialized textures. Position zero
        // and id zero are valid and select the nonzero lighting table row.
        for (texture, texel) in [
            (f.target.normal_texture(), [0_u16, 0, 0x3c00, 0x3c00]),
            (f.target.texture(), [0x3c00_u16; 4]),
        ] {
            let bytes: Vec<u8> = (0..WIDTH * HEIGHT)
                .flat_map(|_| texel)
                .flat_map(u16::to_le_bytes)
                .collect();
            f.runtime.queue().write_texture(
                rhi::TextureWrite {
                    texture,
                    mip_level: 0,
                    origin: [0, 0, 0],
                    size: rhi::Extent3d::new_2d(WIDTH, HEIGHT),
                },
                &bytes,
                rhi::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(WIDTH * 8),
                    rows_per_image: Some(HEIGHT),
                },
            );
        }
        let graph = crate::frame::target_graph(
            &f.target,
            &crate::RenderQueues::new(),
            &crate::DrawFunctions::new(),
            Some(f.pass),
        );
        let realized = graph
            .execute(f.runtime.device(), f.runtime.queue())
            .unwrap();
        assert!(f.runtime.queue().wait_idle());
        let normals = f
            .runtime
            .queue()
            .read_texture_rgba16f(f.target.normal_texture(), WIDTH, HEIGHT)
            .expect("lighting normal attachment readback must complete");
        let hdr = f
            .runtime
            .queue()
            .read_texture_rgba16f(f.target.texture(), WIDTH, HEIGHT)
            .expect("lighting HDR attachment readback must complete");
        realized.release(f.runtime.device());
        assert!(normals.iter().all(|pixel| *pixel == [0.0; 4]) && hdr.iter().all(|pixel| *pixel == [0.0; 4]),
            "empty GBuffer must clear normal/coverage to zero and reject valid surface 0: normal={:?}, HDR={:?}", normals[0], hdr[0]);
    }
}

/// The GPU objects used by one lighting execution bundle.
pub struct LightingGpuBundle {
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
    /// Sorted explicit material-slot to lighting-surface mapping.
    pub material_lighting_surfaces: prism_render_driver::BufferId,
    /// Per-packed-triangle outgoing radiance used by trace candidate build.
    pub radiance: prism_render_driver::BufferId,
    /// GI pipeline and bind group.
    pub gi_pipeline: prism_render_driver::ComputePipelineId,
    pub gi_bind_group: prism_render_driver::BindGroupId,
    /// Composite pipeline and bind group.
    pub composite_pipeline: prism_render_driver::ComputePipelineId,
    /// Shared pipeline layout for both the builtin and linked registered
    /// composite variants. Keeping the layout in the replay bundle lets a
    /// registered response pipeline swap only its shader module and pipeline.
    pub composite_pipeline_layout: prism_render_driver::PipelineLayoutId,
    /// Post-raster identity reconstruction pipeline.
    pub surface_identity_pipeline: prism_render_driver::ComputePipelineId,
    /// Static composite group-0 layout: direct, GI, and params.
    pub composite_static_layout: BindGroupLayoutId,
    /// Per-target composite group-1 layout: GBuffer reads and HDR write.
    pub composite_target_layout: BindGroupLayoutId,
    /// Layout used by the identity reconstruction pass.
    pub surface_identity_layout: BindGroupLayoutId,
    /// Canonical material group-2 layout consumed by the composite resolver.
    pub composite_material_layout: BindGroupLayoutId,
    /// Native shading-quality budget consumed by the composite lobe helpers.
    pub lobe_sample_budget: BufferId,
    pub camera_view: BufferId,
    pub composite_bind_group: prism_render_driver::BindGroupId,
}

/// A graph-ready view of the current lighting GPU bundle.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LightingGpuPass {
    pub gi_output: prism_render_driver::BufferId,
    pub params: prism_render_driver::BufferId,
    pub directionals: prism_render_driver::BufferId,
    pub punctuals: prism_render_driver::BufferId,
    pub areas: prism_render_driver::BufferId,
    pub candidates: prism_render_driver::BufferId,
    pub surfaces: prism_render_driver::BufferId,
    pub material_lighting_surfaces: prism_render_driver::BufferId,
    pub radiance: prism_render_driver::BufferId,
    pub gi_pipeline: prism_render_driver::ComputePipelineId,
    pub gi_bind_group: prism_render_driver::BindGroupId,
    pub composite_pipeline: prism_render_driver::ComputePipelineId,
    pub composite_pipeline_layout: prism_render_driver::PipelineLayoutId,
    pub surface_identity_pipeline: prism_render_driver::ComputePipelineId,
    pub composite_static_layout: BindGroupLayoutId,
    pub composite_target_layout: BindGroupLayoutId,
    pub surface_identity_layout: BindGroupLayoutId,
    pub composite_material_layout: BindGroupLayoutId,
    pub composite_bind_group: prism_render_driver::BindGroupId,
    pub lobe_sample_budget: BufferId,
    pub camera_view: BufferId,
    /// Target-specific group-1 binding. It is populated only by the
    /// canonical material-aware target pass.
    pub composite_target_bind_group: Option<BindGroupId>,
    /// Target-sized canonical response storage written by the composite.
    pub response_output: BufferId,
    pub response_output_size_bytes: u64,
    /// Target/material binding used to reconstruct the per-pixel identity
    /// tuple before the composite reads it.
    pub surface_identity_bind_group: Option<BindGroupId>,
    /// Group-2 bind group for the four canonical material buffers from one
    /// published generation.
    pub composite_material_bind_group: Option<BindGroupId>,
    composite_material_buffers: Option<LightingCompositeMaterialBuffers>,
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
        (self.run_gi as u64) + 1
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

    /// Returns the fixed group index for the optional canonical material set.
    #[must_use]
    pub(crate) const fn composite_material_group() -> u32 {
        COMPOSITE_MATERIAL_GROUP
    }

    /// Returns the exact layout required by the canonical material factory.
    #[must_use]
    pub(crate) const fn composite_material_layout(self) -> BindGroupLayoutId {
        self.composite_material_layout
    }

    /// Creates the group-2 binding for one coherent canonical material
    /// snapshot. A partially supplied set is rejected before backend calls.
    pub(crate) fn create_composite_material_bind_group<D: prism_render_driver::RenderDevice>(
        self,
        device: &D,
        buffers: LightingCompositeMaterialBuffers,
    ) -> Result<BindGroupId, LightingCompositeMaterialBindingError> {
        let [(headers, headers_size), (parameters, parameters_size), _, _] =
            validate_material_buffers(buffers)?;
        // Plans/programs are consumed by the separate identity pass. Keeping
        // them out of group 2 is required for devices exposing only eight
        // storage-buffer bindings across the composite pipeline.
        Ok(device.create_bind_group(&BindGroupDescriptor {
            label: Some("lighting-composite-material-bind-group".into()),
            layout: Some(self.composite_material_layout),
            entries: vec![
                buffer_binding(0, headers, headers_size),
                buffer_binding(1, parameters, parameters_size),
            ],
        }))
    }

    /// Returns a pass carrying both the group and the exact buffer IDs used to
    /// create it, so frame-graph declarations cannot accidentally describe a
    /// different generation than the bound group.
    pub(crate) fn with_composite_material_bindings<D: prism_render_driver::RenderDevice>(
        mut self,
        device: &D,
        buffers: LightingCompositeMaterialBuffers,
    ) -> Result<Self, LightingCompositeMaterialBindingError> {
        self.composite_material_bind_group =
            Some(self.create_composite_material_bind_group(device, buffers)?);
        self.composite_material_buffers = Some(buffers);
        Ok(self)
    }

    /// The material resources to declare as reads when this pass has a
    /// material group. The caller must retain the snapshot owner through GPU
    /// completion; this value intentionally carries IDs only.
    #[must_use]
    pub(crate) const fn composite_material_buffers(
        self,
    ) -> Option<LightingCompositeMaterialBuffers> {
        self.composite_material_buffers
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
                    resource: BindingResource::TextureView(target.position_x_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(target.position_y_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(target.position_z_view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(target.normal_view),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::TextureView(target.albedo_view),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: BindingResource::TextureView(target.surface_id_view),
                },
                BindGroupEntry {
                    binding: 9,
                    resource: BindingResource::TextureView(target.surface_data_view),
                },
                BindGroupEntry {
                    binding: 10,
                    resource: BindingResource::TextureView(target.sampled_material_a_view),
                },
                BindGroupEntry {
                    binding: 11,
                    resource: BindingResource::TextureView(target.sampled_material_b_view),
                },
                BindGroupEntry {
                    binding: 12,
                    resource: BindingResource::TextureView(target.sampled_material_c_view),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: BindingResource::Buffer {
                        buffer: target.surface_identity_buffer,
                        offset: 0,
                        size: Some(target.surface_identity_size_bytes),
                    },
                },
                BindGroupEntry {
                    binding: 7,
                    resource: BindingResource::TextureView(target.hdr_target_view),
                },
                BindGroupEntry {
                    binding: 8,
                    resource: BindingResource::Buffer {
                        buffer: target.response_output_buffer,
                        offset: 0,
                        size: Some(target.response_output_size_bytes),
                    },
                },
            ],
        }))
    }

    pub(crate) fn create_composite_static_bind_group<D: prism_render_driver::RenderDevice>(
        self,
        device: &D,
        camera_view: BufferId,
    ) -> BindGroupId {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("lighting-composite-static-bind-group".into()),
            layout: Some(self.composite_static_layout),
            entries: vec![
                buffer_binding(0, self.params, PARAM_BYTES),
                buffer_binding(1, self.directionals, DIRECTIONAL_BYTES),
                buffer_binding(2, self.punctuals, PUNCTUAL_BYTES),
                buffer_binding(3, self.areas, AREA_BYTES),
                buffer_binding(4, self.gi_output, OUTPUT_BYTES),
                buffer_binding(5, camera_view, CAMERA_VIEW_BYTES),
                buffer_binding(6, self.lobe_sample_budget, LOBE_SAMPLE_BUDGET_BYTES),
            ],
        })
    }

    /// Creates the post-raster identity reconstruction bindings for one target
    /// and one published material-header generation.
    pub(crate) fn create_surface_identity_bind_group<D: prism_render_driver::RenderDevice>(
        self,
        device: &D,
        target: LightingCompositeTarget,
        headers: BufferId,
        headers_size_bytes: u64,
        execution_plans: BufferId,
        execution_plans_size_bytes: u64,
        execution_programs: BufferId,
        execution_programs_size_bytes: u64,
    ) -> BindGroupId {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("lighting-surface-identity-bind-group".into()),
            layout: Some(self.surface_identity_layout),
            entries: vec![
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(target.normal_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(target.surface_id_view),
                },
                buffer_binding(2, headers, headers_size_bytes),
                buffer_binding(3, execution_plans, execution_plans_size_bytes),
                buffer_binding(4, execution_programs, execution_programs_size_bytes),
                BindGroupEntry {
                    binding: 5,
                    resource: BindingResource::Buffer {
                        buffer: target.surface_identity_buffer,
                        offset: 0,
                        size: Some(target.surface_identity_size_bytes),
                    },
                },
                // The identity shader resolves the material slot to the
                // lighting-surface index used by the GI response channel.
                // These resources belong to the identity ABI, not composite
                // group 2.
                buffer_binding(
                    6,
                    self.material_lighting_surfaces,
                    MATERIAL_LIGHTING_SURFACE_BYTES,
                ),
                buffer_binding(7, self.params, PARAM_BYTES),
            ],
        })
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
    composite_targets: Mutex<CompositeTargetCache>,
    marker: core::marker::PhantomData<fn() -> I>,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct CompositeTargetKey {
    position_x_view: TextureViewId,
    position_y_view: TextureViewId,
    position_z_view: TextureViewId,
    normal_view: TextureViewId,
    albedo_view: TextureViewId,
    surface_id_view: TextureViewId,
    surface_data_view: TextureViewId,
    sampled_material_a_view: TextureViewId,
    sampled_material_b_view: TextureViewId,
    sampled_material_c_view: TextureViewId,
    surface_identity_buffer: BufferId,
    hdr_target_view: TextureViewId,
    width: u32,
    height: u32,
}

impl From<&RenderTarget> for CompositeTargetKey {
    fn from(target: &RenderTarget) -> Self {
        Self {
            position_x_view: target.position_x_view(),
            position_y_view: target.position_y_view(),
            position_z_view: target.position_z_view(),
            normal_view: target.normal_view(),
            albedo_view: target.albedo_view(),
            surface_id_view: target.surface_id_view(),
            surface_data_view: target.surface_data_view(),
            sampled_material_a_view: target.surface_material_a_view(),
            sampled_material_b_view: target.surface_material_b_view(),
            sampled_material_c_view: target.surface_material_c_view(),
            surface_identity_buffer: target.surface_identity_buffer(),
            hdr_target_view: target.view(),
            width: target.viewport().extent().width,
            height: target.viewport().extent().height,
        }
    }
}

#[derive(Default)]
struct CompositeTargetCache {
    bundle: Option<Arc<LightingGpuBundle>>,
    device_generation: Option<u64>,
    device: Option<Arc<dyn RenderDevice + Send + Sync>>,
    groups: BTreeMap<CompositeTargetKey, BindGroupId>,
    response_buffers: BTreeMap<CompositeTargetKey, (BufferId, u64)>,
    camera_groups: BTreeMap<(CompositeTargetKey, [u32; 3]), (BindGroupId, BufferId)>,
    identity_groups: BTreeMap<(CompositeTargetKey, BufferId, BufferId, BufferId), BindGroupId>,
    material_buffers: Option<LightingCompositeMaterialBuffers>,
    material_group: Option<BindGroupId>,
    registered_pipeline: Option<RegisteredCompositePipeline>,
}

struct RegisteredCompositePipeline {
    source: Arc<str>,
    shader: ShaderModuleId,
    pipeline: prism_render_driver::ComputePipelineId,
}

impl CompositeTargetCache {
    fn retire_registered_pipeline<Q: FrameSync>(&mut self, queue: &Q) {
        let Some(pipeline) = self.registered_pipeline.take() else {
            return;
        };
        let Some(device) = self.device.as_ref().map(Arc::clone) else {
            return;
        };
        queue.on_submitted_work_done(Box::new(move || {
            device.destroy_compute_pipeline(pipeline.pipeline);
            device.destroy_shader_module(pipeline.shader);
        }));
    }

    fn evict(
        &mut self,
        key: CompositeTargetKey,
        device: &Arc<dyn RenderDevice + Send + Sync>,
        generation: u64,
    ) {
        // Resource IDs and generation counters are local to a device. Two
        // independent runtimes may produce identical values for both.
        if self.device_generation != Some(generation)
            || !self
                .device
                .as_ref()
                .is_some_and(|owner| Arc::ptr_eq(owner, device))
        {
            return;
        }
        if let Some(group) = self.groups.remove(&key) {
            device.destroy_bind_group(group);
        }
        if let Some((buffer, _)) = self.response_buffers.remove(&key) {
            device.destroy_buffer(buffer);
        }
        let camera_keys = self
            .camera_groups
            .keys()
            .copied()
            .filter(|(target_key, _)| *target_key == key)
            .collect::<Vec<_>>();
        for camera_key in camera_keys {
            if let Some((group, buffer)) = self.camera_groups.remove(&camera_key) {
                device.destroy_bind_group(group);
                device.destroy_buffer(buffer);
            }
        }
        let identity_keys = self
            .identity_groups
            .keys()
            .copied()
            .filter(|(target_key, _, _, _)| *target_key == key)
            .collect::<Vec<_>>();
        for identity_key in identity_keys {
            if let Some(group) = self.identity_groups.remove(&identity_key) {
                device.destroy_bind_group(group);
            }
        }
    }

    fn clear(&mut self) {
        if let Some(device) = &self.device {
            for group in self.groups.values().copied() {
                device.destroy_bind_group(group);
            }
            for (group, buffer) in self.camera_groups.values().copied() {
                device.destroy_bind_group(group);
                device.destroy_buffer(buffer);
            }
            for (buffer, _) in self.response_buffers.values().copied() {
                device.destroy_buffer(buffer);
            }
            for group in self.identity_groups.values().copied() {
                device.destroy_bind_group(group);
            }
            if let Some(group) = self.material_group.take() {
                device.destroy_bind_group(group);
            }
            if let Some(pipeline) = self.registered_pipeline.take() {
                device.destroy_compute_pipeline(pipeline.pipeline);
                device.destroy_shader_module(pipeline.shader);
            }
        } else {
            self.material_group = None;
            self.registered_pipeline = None;
        }
        self.groups.clear();
        self.response_buffers.clear();
        self.camera_groups.clear();
        self.identity_groups.clear();
        self.material_buffers = None;
        self.bundle = None;
        self.device_generation = None;
        self.device = None;
    }
}

fn ensure_registered_composite_pipeline<D, Q>(
    cache: &mut CompositeTargetCache,
    device: &D,
    queue: &Q,
    pass: LightingGpuPass,
    response_link: &str,
) -> prism_render_driver::ComputePipelineId
where
    D: RenderDevice,
    Q: FrameSync,
{
    if let Some(existing) = cache
        .registered_pipeline
        .as_ref()
        .filter(|existing| existing.source.as_ref() == response_link)
    {
        return existing.pipeline;
    }

    cache.retire_registered_pipeline(queue);
    let shader_source = native_registered_shader_source(response_link);
    let shader = device.create_shader_module(&ShaderModuleDescriptor {
        label: Some("lighting-composite-registered-shader".into()),
        source: ShaderSource::Wgsl(Cow::Owned(shader_source)),
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("lighting-composite-registered-compute".into()),
        layout: Some(pass.composite_pipeline_layout),
        module: shader,
        entry_point: "main".into(),
    });
    cache.registered_pipeline = Some(RegisteredCompositePipeline {
        source: Arc::<str>::from(response_link),
        shader,
        pipeline,
    });
    pipeline
}

impl Drop for CompositeTargetCache {
    fn drop(&mut self) {
        self.clear();
    }
}

#[cfg(test)]
#[path = "lighting_target_cache_tests.rs"]
mod target_cache_tests;

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
    /// The frame-level NPR controls cannot be represented by the native ABI.
    InvalidStylizedParams,
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
    /// The per-surface material mapping is not aligned with the GI surface
    /// table, so a material slot cannot be resolved to a lighting surface.
    SurfaceMaterialMappingCountMismatch { surfaces: usize, mappings: usize },
    /// A surface mapping used the invalid material sentinel.
    InvalidSurfaceMaterialMapping {
        surface_index: usize,
        material_index: u32,
    },
    /// One material slot mapped to more than one lighting surface. Choosing
    /// one would make the result depend on ordering, so preparation rejects it.
    DuplicateSurfaceMaterialMapping { material_index: u32 },
    /// A rare capacity-pressure compaction could not drain the queue safely.
    QueueCompletionUnconfirmed,
}

struct StagedLightingInputs {
    surfaces: Vec<GpuLightingSurface>,
    material_lighting_surfaces: Vec<GpuMaterialLightingSurface>,
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
    if !inputs.surface_material_indices.is_empty()
        && inputs.surface_material_indices.len() != surface_count
    {
        return Err(
            LightingGpuPrepareError::SurfaceMaterialMappingCountMismatch {
                surfaces: surface_count,
                mappings: inputs.surface_material_indices.len(),
            },
        );
    }
    let mapping_required = plan.candidate_source != GiCandidateSource::Disabled
        && plan.recipes.contains(&RecipeId::WorldRestir)
        && surface_count != 0;
    if mapping_required && inputs.surface_material_indices.len() != surface_count {
        return Err(
            LightingGpuPrepareError::SurfaceMaterialMappingCountMismatch {
                surfaces: surface_count,
                mappings: inputs.surface_material_indices.len(),
            },
        );
    }
    let mut material_lighting_surfaces = inputs
        .surface_material_indices
        .iter()
        .copied()
        .enumerate()
        .map(|(lighting_surface_index, material_index)| {
            if material_index == INVALID_LIGHTING_SURFACE_INDEX {
                return Err(LightingGpuPrepareError::InvalidSurfaceMaterialMapping {
                    surface_index: lighting_surface_index,
                    material_index,
                });
            }
            Ok(GpuMaterialLightingSurface {
                material_index,
                lighting_surface_index: lighting_surface_index as u32,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    material_lighting_surfaces.sort_unstable_by_key(|entry| entry.material_index);
    for pair in material_lighting_surfaces.windows(2) {
        if pair[0].material_index == pair[1].material_index {
            return Err(LightingGpuPrepareError::DuplicateSurfaceMaterialMapping {
                material_index: pair[0].material_index,
            });
        }
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
        material_lighting_surfaces,
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
        Self::new_with_quality(runtime, ShadingQuality::High)
    }

    /// Creates the lighting bundle with the native composite's initial
    /// expensive-lobe quality policy. The policy is part of the replay recipe
    /// so device replacement recreates both the buffer and its initial upload.
    pub fn new_with_quality(
        runtime: &mut RenderRuntime<I>,
        shading_quality: ShadingQuality,
    ) -> Result<Self, prism_render_runtime::ReplayFailure> {
        let bundle = runtime.register_replay("lighting-kernels", move |device, queue| {
            Ok(build_bundle(device, queue, shading_quality))
        })?;
        Ok(Self {
            bundle,
            prepared: Mutex::new(PreparedLightingFrame::default()),
            radiance_upload: Mutex::new(RadianceUploadCache::default()),
            light_upload: Mutex::new(LightUploadCache::default()),
            composite_targets: Mutex::new(CompositeTargetCache::default()),
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
        if !inputs.stylized.is_valid() {
            return Err(LightingGpuPrepareError::InvalidStylizedParams);
        }
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
            stylized: inputs.stylized,
        };
        queue.write_buffer(bundle.params, 0, bytes_of(&params));
        if !staged.surfaces.is_empty() {
            queue.write_buffer(bundle.surfaces, 0, cast_slice(&staged.surfaces));
        }
        if !staged.candidates.is_empty() {
            queue.write_buffer(bundle.candidates, 0, cast_slice(&staged.candidates));
        }
        let mut material_lighting_surfaces = [GpuMaterialLightingSurface {
            material_index: INVALID_LIGHTING_SURFACE_INDEX,
            lighting_surface_index: INVALID_LIGHTING_SURFACE_INDEX,
        }; MAX_LIGHTING_SURFACES];
        material_lighting_surfaces[..staged.material_lighting_surfaces.len()]
            .copy_from_slice(&staged.material_lighting_surfaces);
        queue.write_buffer(
            bundle.material_lighting_surfaces,
            0,
            cast_slice(&material_lighting_surfaces),
        );
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
            gi_output: bundle.gi_output,
            params: bundle.params,
            directionals: bundle.directionals,
            punctuals: bundle.punctuals,
            areas: bundle.areas,
            candidates: bundle.candidates,
            surfaces: bundle.surfaces,
            material_lighting_surfaces: bundle.material_lighting_surfaces,
            radiance: bundle.radiance,
            gi_pipeline: bundle.gi_pipeline,
            gi_bind_group: bundle.gi_bind_group,
            composite_pipeline: bundle.composite_pipeline,
            composite_pipeline_layout: bundle.composite_pipeline_layout,
            surface_identity_pipeline: bundle.surface_identity_pipeline,
            composite_static_layout: bundle.composite_static_layout,
            composite_target_layout: bundle.composite_target_layout,
            surface_identity_layout: bundle.surface_identity_layout,
            composite_material_layout: bundle.composite_material_layout,
            composite_bind_group: bundle.composite_bind_group,
            lobe_sample_budget: bundle.lobe_sample_budget,
            camera_view: bundle.camera_view,
            composite_target_bind_group: None,
            response_output: BufferId::from_parts(0, 0),
            response_output_size_bytes: 0,
            surface_identity_bind_group: None,
            composite_material_bind_group: None,
            composite_material_buffers: None,
            run_direct,
            run_gi,
            candidate_source: plan.candidate_source,
            candidate_surface_count: surface_count,
            candidate_rays_per_surface: prepared.candidate_rays_per_surface,
            workgroups: surface_count.div_ceil(WORKGROUP_SIZE),
        })
    }

    /// Returns a graph-ready pass with a cached group-1 binding for `target`.
    ///
    /// The cache is keyed by every view and the target extent. A resize or
    /// replacement target therefore creates one new binding. Its owner must
    /// call `evict_target` after completion and before destroying attachments.
    /// A device-generation change drops the old cache without
    /// issuing destruction calls against the replacement device.
    #[must_use]
    fn target_pass_core(
        &self,
        plan: &LightingPlan,
        target: &RenderTarget,
        runtime: &RenderRuntime<I>,
        camera_view_position: [f32; 3],
    ) -> Option<LightingGpuPass>
    where
        I::Device: Send + Sync + 'static,
    {
        if !camera_view_position.iter().all(|value| value.is_finite()) {
            return None;
        }
        let mut pass = self.pass(plan)?;
        let bundle = self.bundle.current();
        let device = runtime.device_handle();
        let device_dyn: Arc<dyn RenderDevice + Send + Sync> = device.clone();
        let generation = runtime.device_generation();
        let key = CompositeTargetKey::from(target);
        let camera_key = (key, camera_view_position.map(f32::to_bits));
        let mut cache = self
            .composite_targets
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let bundle_changed = cache
            .bundle
            .as_ref()
            .is_some_and(|cached| !Arc::ptr_eq(cached, &bundle));
        let generation_changed = cache.device_generation != Some(generation)
            || !cache
                .device
                .as_ref()
                .is_some_and(|owner| Arc::ptr_eq(owner, &device_dyn));
        if bundle_changed || generation_changed {
            if generation_changed {
                cache.groups.clear();
                cache.response_buffers.clear();
                cache.camera_groups.clear();
                cache.identity_groups.clear();
                // The old device owns the registered module/pipeline. Do not
                // reuse its IDs after recovery; device teardown owns cleanup.
                cache.registered_pipeline = None;
                cache.device = None;
            } else {
                cache.clear();
            }
            cache.bundle = Some(Arc::clone(&bundle));
            cache.device_generation = Some(generation);
            cache.device = Some(device_dyn);
        }
        let (camera_group, camera_buffer) =
            if let Some(binding) = cache.camera_groups.get(&camera_key).copied() {
                binding
            } else {
                let camera_buffer = device.create_buffer(&BufferDescriptor::new(
                    CAMERA_VIEW_BYTES,
                    BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                ));
                let payload = GpuCameraView {
                    position: camera_view_position,
                    _padding: 0.0,
                };
                runtime
                    .queue()
                    .write_buffer(camera_buffer, 0, bytes_of(&payload));
                let camera_group = pass.create_composite_static_bind_group(&*device, camera_buffer);
                cache
                    .camera_groups
                    .insert(camera_key, (camera_group, camera_buffer));
                (camera_group, camera_buffer)
            };
        pass.composite_bind_group = camera_group;
        pass.camera_view = camera_buffer;
        let response_output_size_bytes = u64::from(key.width)
            .checked_mul(u64::from(key.height))
            .and_then(|pixels| pixels.checked_mul(COMPOSITE_RESPONSE_BYTES_PER_PIXEL))?;
        let (response_output, response_output_size_bytes) =
            if let Some(binding) = cache.response_buffers.get(&key).copied() {
                binding
            } else {
                let response_output = device.create_buffer(&BufferDescriptor::new(
                    response_output_size_bytes,
                    BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                ));
                cache
                    .response_buffers
                    .insert(key, (response_output, response_output_size_bytes));
                (response_output, response_output_size_bytes)
            };
        let group = if let Some(group) = cache.groups.get(&key).copied() {
            group
        } else {
            let group = pass
                .create_composite_target_bind_group(
                    &*device,
                    LightingCompositeTarget {
                        position_x_view: key.position_x_view,
                        position_y_view: key.position_y_view,
                        position_z_view: key.position_z_view,
                        normal_view: key.normal_view,
                        albedo_view: key.albedo_view,
                        surface_id_view: key.surface_id_view,
                        surface_data_view: key.surface_data_view,
                        sampled_material_a_view: key.sampled_material_a_view,
                        sampled_material_b_view: key.sampled_material_b_view,
                        sampled_material_c_view: key.sampled_material_c_view,
                        surface_identity_buffer: key.surface_identity_buffer,
                        surface_identity_size_bytes: target.surface_identity_size_bytes(),
                        hdr_target_view: key.hdr_target_view,
                        response_output_buffer: response_output,
                        response_output_size_bytes,
                        width: key.width,
                        height: key.height,
                    },
                )
                .expect("validated render target must satisfy the composite contract");
            cache.groups.insert(key, group);
            group
        };
        pass.composite_target_bind_group = Some(group);
        pass.response_output = response_output;
        pass.response_output_size_bytes = response_output_size_bytes;
        Some(pass)
    }

    /// Returns the only target-aware production pass. Its material bindings
    /// must reference one completion-published canonical generation; the
    /// optional source is used only for registered response programs.
    pub(crate) fn target_pass(
        &self,
        plan: &LightingPlan,
        target: &RenderTarget,
        runtime: &RenderRuntime<I>,
        camera_view_position: [f32; 3],
        buffers: LightingCompositeMaterialBuffers,
        registered_response_wgsl: Option<&str>,
    ) -> Option<LightingGpuPass>
    where
        I::Device: Send + Sync + 'static,
        I::Queue: FrameSync,
    {
        let mut pass = self.target_pass_core(plan, target, runtime, camera_view_position)?;
        let device = runtime.device_handle();
        let device_owner: Arc<dyn RenderDevice + Send + Sync> = device.clone();
        let generation = runtime.device_generation();
        let key = CompositeTargetKey::from(target);
        let mut cache = self
            .composite_targets
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(response_link) =
            registered_response_wgsl.filter(|source| !source.trim().is_empty())
        {
            pass.composite_pipeline = ensure_registered_composite_pipeline(
                &mut cache,
                &*device,
                runtime.queue(),
                pass,
                response_link,
            );
        }
        let owner_matches = cache.device_generation == Some(generation)
            && cache
                .device
                .as_ref()
                .is_some_and(|owner| Arc::ptr_eq(owner, &device_owner));
        let group = if owner_matches && cache.material_buffers == Some(buffers) {
            cache
                .material_group
                .expect("cached material buffers always have a bind group")
        } else {
            let group = pass
                .create_composite_material_bind_group(&*device, buffers)
                .expect("published canonical material buffers satisfy the binding ABI");
            if owner_matches {
                if let Some(old_group) = cache.material_group.replace(group) {
                    let old_device = Arc::clone(cache.device.as_ref().expect("owner matches"));
                    runtime.queue().on_submitted_work_done(Box::new(move || {
                        old_device.destroy_bind_group(old_group);
                    }));
                }
            } else {
                cache.material_group = Some(group);
            }
            cache.material_buffers = Some(buffers);
            group
        };
        pass.composite_material_bind_group = Some(group);
        pass.composite_material_buffers = Some(buffers);

        let headers = buffers
            .headers
            .expect("validated canonical material headers");
        let execution_plans = buffers
            .execution_plans
            .expect("validated canonical material execution plans");
        let execution_programs = buffers
            .execution_programs
            .expect("validated canonical material execution programs");
        let identity_key = (key, headers, execution_plans, execution_programs);
        let identity_group = if let Some(group) = cache.identity_groups.get(&identity_key).copied()
        {
            group
        } else {
            let group = pass.create_surface_identity_bind_group(
                &*device,
                LightingCompositeTarget {
                    position_x_view: key.position_x_view,
                    position_y_view: key.position_y_view,
                    position_z_view: key.position_z_view,
                    normal_view: key.normal_view,
                    albedo_view: key.albedo_view,
                    surface_id_view: key.surface_id_view,
                    surface_data_view: key.surface_data_view,
                    sampled_material_a_view: key.sampled_material_a_view,
                    sampled_material_b_view: key.sampled_material_b_view,
                    sampled_material_c_view: key.sampled_material_c_view,
                    surface_identity_buffer: key.surface_identity_buffer,
                    surface_identity_size_bytes: target.surface_identity_size_bytes(),
                    hdr_target_view: key.hdr_target_view,
                    response_output_buffer: pass.response_output,
                    response_output_size_bytes: pass.response_output_size_bytes,
                    width: key.width,
                    height: key.height,
                },
                headers,
                buffers.headers_size_bytes,
                execution_plans,
                buffers.execution_plans_size_bytes,
                execution_programs,
                buffers.execution_programs_size_bytes,
            );
            let old_keys = cache
                .identity_groups
                .keys()
                .copied()
                .filter(|(target_key, old_headers, old_plans, old_programs)| {
                    *target_key == key
                        && (*old_headers != headers
                            || *old_plans != execution_plans
                            || *old_programs != execution_programs)
                })
                .collect::<Vec<_>>();
            for old_key in old_keys {
                if let Some(old_group) = cache.identity_groups.remove(&old_key) {
                    let old_device = Arc::clone(cache.device.as_ref().expect("cache owner"));
                    runtime.queue().on_submitted_work_done(Box::new(move || {
                        old_device.destroy_bind_group(old_group);
                    }));
                }
            }
            cache.identity_groups.insert(identity_key, group);
            group
        };
        pass.surface_identity_bind_group = Some(identity_group);
        Some(pass)
    }

    /// Retires only this target's binding on its creating device/generation.
    ///
    /// The caller must first prove that every submission referencing the
    /// target has completed. An already absent binding (including an old
    /// generation discarded during recovery) is an idempotent success. This
    /// never destroys resource IDs through a replacement or unrelated device.
    pub(crate) fn evict_target(
        &self,
        target: &RenderTarget,
        device: &Arc<I::Device>,
        generation: u64,
    ) -> bool
    where
        I::Device: Send + Sync + 'static,
    {
        let device: Arc<dyn RenderDevice + Send + Sync> = device.clone();
        self.composite_targets
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .evict(CompositeTargetKey::from(target), &device, generation);
        true
    }

    /// Returns the GI signal buffer for diagnostics and estimator tests.
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
            // WorldRestir now requires an explicit, one-to-one material slot
            // mapping. Keep the low-level lighting fixtures canonical instead
            // of making the production staging path silently accept an
            // unresolvable GI identity.
            surface_material_indices: (0..count as u32).collect(),
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
    fn stage_material_surface_mapping_is_sorted_for_shader_lookup() {
        let plan = plan(GiCandidateSource::HostProvided);
        let mut inputs = surfaces(3);
        inputs.surface_material_indices = vec![42, 7, 19];
        inputs.gi_candidates = vec![vec![]; 3];

        let staged = stage_lighting_inputs(&plan, &inputs, 0).unwrap();

        assert_eq!(
            staged
                .material_lighting_surfaces
                .iter()
                .map(|entry| (entry.material_index, entry.lighting_surface_index))
                .collect::<Vec<_>>(),
            vec![(7, 1), (19, 2), (42, 0)]
        );
    }

    #[test]
    fn stage_material_surface_mapping_requires_a_complete_active_gi_relation() {
        let plan = plan(GiCandidateSource::HostProvided);
        let mut inputs = surfaces(2);
        inputs.surface_material_indices.clear();
        inputs.gi_candidates = vec![Vec::new(); 2];

        match stage_lighting_inputs(&plan, &inputs, 0) {
            Err(error) => assert_eq!(
                error,
                LightingGpuPrepareError::SurfaceMaterialMappingCountMismatch {
                    surfaces: 2,
                    mappings: 0,
                }
            ),
            Ok(_) => panic!("mapping is required"),
        }
    }

    #[test]
    fn stage_material_surface_mapping_rejects_invalid_and_duplicate_slots() {
        let plan = plan(GiCandidateSource::Disabled);

        let mut invalid = surfaces(1);
        invalid.surface_material_indices = vec![INVALID_LIGHTING_SURFACE_INDEX];
        match stage_lighting_inputs(&plan, &invalid, 0) {
            Err(error) => assert_eq!(
                error,
                LightingGpuPrepareError::InvalidSurfaceMaterialMapping {
                    surface_index: 0,
                    material_index: INVALID_LIGHTING_SURFACE_INDEX,
                }
            ),
            Ok(_) => panic!("invalid sentinel must be rejected"),
        }

        let mut duplicate = surfaces(2);
        duplicate.surface_material_indices = vec![11, 11];
        match stage_lighting_inputs(&plan, &duplicate, 0) {
            Err(error) => assert_eq!(
                error,
                LightingGpuPrepareError::DuplicateSurfaceMaterialMapping { material_index: 11 }
            ),
            Ok(_) => panic!("duplicate slot must be rejected"),
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

    #[test]
    fn target_pass_rejects_non_finite_camera_view_position() {
        let (runtime, state) = setup();
        let plan = plan(GiCandidateSource::HostProvided);
        let mut inputs = surfaces(1);
        inputs.gi_candidates = vec![Vec::new()];
        state.prepare(&plan, &inputs, 1, &runtime).unwrap();
        let target = RenderTarget::new(runtime.device(), rhi::Extent3d::new_2d(4, 4));
        assert!(state
            .target_pass(
                &plan,
                &target,
                &runtime,
                [f32::NAN, 0.0, 0.0],
                LightingCompositeMaterialBuffers::default(),
                None,
            )
            .is_none());
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

fn camera_view_layout(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: Some(CAMERA_VIEW_BYTES),
        },
        count: None,
    }
}

fn lobe_budget_layout(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: Some(LOBE_SAMPLE_BUDGET_BYTES),
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

fn build_bundle<D: prism_render_driver::RenderDevice, Q: RenderQueue>(
    device: &D,
    queue: &Q,
    shading_quality: ShadingQuality,
) -> LightingGpuBundle {
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
    let camera_view = device.create_buffer(&BufferDescriptor {
        label: Some("lighting-camera-view".into()),
        size: CAMERA_VIEW_BYTES,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let lobe_sample_budget = device.create_buffer(&BufferDescriptor {
        label: Some("lighting-lobe-sample-budget".into()),
        size: LOBE_SAMPLE_BUDGET_BYTES,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let lobe_budget = GpuLobeSampleBudget::for_quality(shading_quality);
    queue.write_buffer(lobe_sample_budget, 0, bytes_of(&lobe_budget));
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
    let material_lighting_surfaces = device.create_buffer(&BufferDescriptor {
        label: Some("lighting-material-surface-map".into()),
        size: MATERIAL_LIGHTING_SURFACE_BYTES,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let radiance = device.create_buffer(&BufferDescriptor {
        label: Some("lighting-trace-radiance".into()),
        size: TRACE_RADIANCE_BYTES,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
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

    let composite_shader_source = build_composite_shader_source(None);
    let composite_shader = device.create_shader_module(&ShaderModuleDescriptor {
        label: Some("lighting-composite-shader".into()),
        source: ShaderSource::Wgsl(Cow::Owned(composite_shader_source)),
    });
    let composite_static_layout =
        device.create_bind_group_layout(&composite_static_layout_descriptor());
    let composite_target_layout =
        device.create_bind_group_layout(&composite_target_layout_descriptor());
    let composite_material_layout =
        device.create_bind_group_layout(&composite_material_layout_descriptor());
    let composite_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("lighting-composite-pipeline-layout".into()),
        bind_group_layouts: vec![
            composite_static_layout,
            composite_target_layout,
            composite_material_layout,
        ],
        push_constant_ranges: Vec::new(),
    });
    let composite_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("lighting-composite-compute".into()),
        layout: Some(composite_pipeline_layout),
        module: composite_shader,
        entry_point: "main".into(),
    });
    let surface_identity_shader = device.create_shader_module(&ShaderModuleDescriptor {
        label: Some("lighting-surface-identity-shader".into()),
        source: ShaderSource::Wgsl(Cow::Borrowed(include_str!(concat!(
            env!("OUT_DIR"),
            "/surface_identity.wgsl"
        )))),
    });
    let surface_identity_layout =
        device.create_bind_group_layout(&surface_identity_layout_descriptor());
    let surface_identity_pipeline_layout =
        device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("lighting-surface-identity-pipeline-layout".into()),
            bind_group_layouts: vec![surface_identity_layout],
            push_constant_ranges: Vec::new(),
        });
    let surface_identity_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("lighting-surface-identity-compute".into()),
        layout: Some(surface_identity_pipeline_layout),
        module: surface_identity_shader,
        entry_point: "main".into(),
    });
    let composite_bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("lighting-composite-static-bind-group".into()),
        layout: Some(composite_static_layout),
        entries: vec![
            buffer_binding(0, params, PARAM_BYTES),
            buffer_binding(1, directionals, DIRECTIONAL_BYTES),
            buffer_binding(2, punctuals, PUNCTUAL_BYTES),
            buffer_binding(3, areas, AREA_BYTES),
            buffer_binding(4, gi_output, OUTPUT_BYTES),
            buffer_binding(5, camera_view, CAMERA_VIEW_BYTES),
            buffer_binding(6, lobe_sample_budget, LOBE_SAMPLE_BUDGET_BYTES),
        ],
    });

    LightingGpuBundle {
        gi_output,
        params,
        directionals,
        punctuals,
        areas,
        candidates,
        surfaces,
        material_lighting_surfaces,
        radiance,
        gi_pipeline,
        gi_bind_group,
        composite_pipeline,
        composite_pipeline_layout,
        surface_identity_pipeline,
        composite_static_layout,
        composite_target_layout,
        surface_identity_layout,
        composite_material_layout,
        lobe_sample_budget,
        camera_view,
        composite_bind_group,
    }
}
