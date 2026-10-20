//! GPU execution for the deterministic direct-light and initial GI recipes.

use bytemuck::{bytes_of, cast_slice, Pod, Zeroable};
use prism_ecs::prelude::Resource;
use prism_render_architecture::lighting::contract::{GiCandidateSource, LightingPlan, RecipeId};
use prism_render_architecture::lighting::{
    CompletionProof, GpuSubmissionDomain, GpuTableHandle, GpuTablePayload, IncrementalGpuTable,
    LightSceneHandle, SubmissionProof, UploadPlan,
};
use prism_render_driver::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingResource, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    ComputePipelineDescriptor, FrameSync, PipelineLayoutDescriptor, RenderQueue,
    ShaderModuleDescriptor, ShaderStages,
};
use prism_render_runtime::{RenderRuntime, ReplayHandle};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use crate::lighting::{
    ExtractedLightingInputs, LightingDirectionalInput, LightingGiCandidateInput,
    LightingPunctualInput,
};

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

const MAX_DIRECTIONALS: usize = 1024;
const MAX_PUNCTUALS: usize = 1024;
/// The render runtime currently exposes one ordered submission stream.
/// Device recovery is separated by `device_generation`; this logical id keeps
/// the stable-table contract ready for additional queues without accepting a
/// proof from another stream by accident.
pub(crate) const LIGHTING_QUEUE_DOMAIN: u64 = 0;
pub(crate) const GI_CANDIDATES_PER_SURFACE: usize = 64;
pub(crate) const MAX_LIGHTING_SURFACES: usize = 1024;
const WORKGROUP_SIZE: u32 = 64;
pub(crate) const OUTPUT_BYTES: u64 =
    (MAX_LIGHTING_SURFACES * core::mem::size_of::<[f32; 4]>()) as u64;
const DIRECTIONAL_BYTES: u64 =
    (MAX_DIRECTIONALS * core::mem::size_of::<LightingDirectionalInput>()) as u64;
const PUNCTUAL_BYTES: u64 = (MAX_PUNCTUALS * core::mem::size_of::<LightingPunctualInput>()) as u64;
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

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.counts.x || id.x >= arrayLength(&output)) {
        return;
    }
    let surface = surfaces[id.x];
    let normal = normalize_or(surface.normal.xyz, vec3<f32>(0.0, 1.0, 0.0));
    var radiance = surface.albedo.xyz * surface.ambient.xyz;
    let directional_count = min(params.counts.y, 1024u);
    let punctual_count = min(params.counts.z, 1024u);
    for (var index = 0u; index < directional_count; index = index + 1u) {
        let light = directionals[index];
        let direction = normalize_or(light.direction_to_light, vec3<f32>(0.0, 1.0, 0.0));
        let cosine = max(dot(normal, direction), 0.0) * clamp(light.visibility, 0.0, 1.0);
        radiance = radiance + surface.albedo.xyz * light.illuminance * cosine;
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
                + surface.albedo.xyz * light.intensity_spot_scale.xyz * (cosine * attenuation);
        }
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

const COMPOSITE_SHADER: &str = r#"
struct Params {
    counts: vec4<u32>,
    flags: vec4<u32>,
};

@group(0) @binding(0)
var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(1)
var<storage, read> direct: array<vec4<f32>>;
@group(0) @binding(2)
var<storage, read> gi: array<vec4<f32>>;
@group(0) @binding(3)
var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.counts.x || id.x >= arrayLength(&output)) {
        return;
    }
    var value = vec3<f32>(0.0);
    if (params.flags.x == 1u || params.flags.x == 3u) {
        value = value + direct[id.x].xyz;
    }
    if (params.flags.x == 2u || params.flags.x == 3u) {
        value = value + gi[id.x].xyz;
    }
    output[id.x] = vec4<f32>(value, 1.0);
}
"#;

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
    pub candidates: prism_render_driver::BufferId,
    pub surfaces: prism_render_driver::BufferId,
    pub radiance: prism_render_driver::BufferId,
    pub direct_pipeline: prism_render_driver::ComputePipelineId,
    pub direct_bind_group: prism_render_driver::BindGroupId,
    pub gi_pipeline: prism_render_driver::ComputePipelineId,
    pub gi_bind_group: prism_render_driver::BindGroupId,
    pub composite_pipeline: prism_render_driver::ComputePipelineId,
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
}

/// Render-world ownership of lighting GPU resources with device-loss replay.
pub struct LightingGpuState<I>
where
    I: prism_render_driver::Instance + 'static,
{
    bundle: ReplayHandle<LightingGpuBundle>,
    surface_count: core::sync::atomic::AtomicU32,
    candidate_rays_per_surface: core::sync::atomic::AtomicU32,
    radiance_upload: Mutex<RadianceUploadCache>,
    light_upload: Mutex<LightUploadCache>,
    marker: core::marker::PhantomData<fn() -> I>,
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
}

#[derive(Default)]
struct LightUploadCache {
    bundle: Option<Arc<LightingGpuBundle>>,
    directionals: IncrementalGpuTable<LightingDirectionalInput>,
    punctuals: IncrementalGpuTable<LightingPunctualInput>,
    directional_slots: BTreeMap<LightSceneHandle, GpuTableHandle>,
    punctual_slots: BTreeMap<LightSceneHandle, GpuTableHandle>,
    next_upload_token: u64,
    pending: Option<PendingLightUpload>,
}

pub(crate) const fn lighting_submission_domain(device_generation: u64) -> GpuSubmissionDomain {
    GpuSubmissionDomain::new(device_generation, LIGHTING_QUEUE_DOMAIN)
}

/// A frame could not prepare a valid stable-slot light upload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LightingGpuPrepareError {
    /// The light ABI payload and its canonical identities were misaligned.
    IdentityPayloadMismatch,
    /// A rare capacity-pressure compaction could not drain the queue safely.
    QueueCompletionUnconfirmed,
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
            surface_count: core::sync::atomic::AtomicU32::new(0),
            candidate_rays_per_surface: core::sync::atomic::AtomicU32::new(0),
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
        let queue = runtime.queue();
        let bundle = self.bundle.current();
        let surface_count = inputs.surfaces.len().min(MAX_LIGHTING_SURFACES);
        self.surface_count
            .store(surface_count as u32, core::sync::atomic::Ordering::Release);
        let trace_input = (plan.candidate_source == GiCandidateSource::SoftwareBvhGpu)
            .then_some(inputs.trace_scene.as_ref())
            .flatten();
        let trace_radiance_valid = trace_input.is_some_and(|trace| {
            let mut cache = self
                .radiance_upload
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let packed = match trace.packed_outgoing_radiance() {
                Ok(packed) => packed,
                Err(_) => return false,
            };
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
                    .is_some_and(|table| Arc::ptr_eq(table, &packed));
            if unchanged {
                return true;
            }
            if packed.len() > TRACE_RADIANCE_BYTES as usize / 16 {
                return false;
            }
            cache.bvh = Some(Arc::clone(&trace.bvh));
            cache.bundle = Some(Arc::clone(&bundle));
            cache.packed = Some(Arc::clone(&packed));
            queue.write_buffer(bundle.radiance, 0, cast_slice(&packed));
            true
        });
        let candidate_rays_per_surface = if trace_radiance_valid {
            trace_input.map_or(0, |trace| {
                trace.rays_per_surface.min(GI_CANDIDATES_PER_SURFACE as u32)
            })
        } else {
            0
        };
        self.candidate_rays_per_surface.store(
            candidate_rays_per_surface,
            core::sync::atomic::Ordering::Release,
        );
        let (directional_count, punctual_count) =
            self.prepare_light_tables(runtime, &bundle, inputs)?;
        let mut execution_flags = 0_u32;
        if plan.recipes.contains(&RecipeId::SpecializedDeterministic) {
            execution_flags |= 1;
        }
        if plan.recipes.contains(&RecipeId::WorldRestir) {
            execution_flags |= 2;
        }
        let params = GpuLightingParams {
            counts: [
                surface_count as u32,
                directional_count as u32,
                punctual_count as u32,
                candidate_rays_per_surface,
            ],
            flags: [
                execution_flags,
                frame_index as u32,
                (frame_index >> 32) as u32,
                trace_input
                    .filter(|_| trace_radiance_valid)
                    .map_or(0, |trace| trace.bvh.triangles.len().div_ceil(12) as u32),
            ],
        };
        queue.write_buffer(bundle.params, 0, bytes_of(&params));
        let mut packed_candidates =
            Vec::with_capacity(surface_count.saturating_mul(GI_CANDIDATES_PER_SURFACE));
        let packed_surfaces = inputs.surfaces[..surface_count]
            .iter()
            .enumerate()
            .map(|(surface_index, surface)| {
                let candidate_start = surface_index * GI_CANDIDATES_PER_SURFACE;
                if plan.candidate_source == GiCandidateSource::HostProvided {
                    if let Some(candidates) = inputs.gi_candidates.get(surface_index) {
                        let count = candidates.len().min(GI_CANDIDATES_PER_SURFACE);
                        packed_candidates.extend_from_slice(&candidates[..count]);
                    }
                }
                if plan.candidate_source == GiCandidateSource::HostProvided {
                    packed_candidates.resize(
                        candidate_start + GI_CANDIDATES_PER_SURFACE,
                        LightingGiCandidateInput::default(),
                    );
                }
                let stable_id = trace_input
                    .and_then(|trace| trace.surface_ids.get(surface_index).copied())
                    .unwrap_or(0);
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
                        match plan.candidate_source {
                            GiCandidateSource::SoftwareBvhGpu => candidate_rays_per_surface,
                            GiCandidateSource::HostProvided => inputs
                                .gi_candidates
                                .get(surface_index)
                                .map_or(0, |candidates| {
                                    candidates.len().min(GI_CANDIDATES_PER_SURFACE) as u32
                                }),
                            GiCandidateSource::Disabled => 0,
                        },
                        stable_id as u32,
                        (stable_id >> 32) as u32,
                    ],
                }
            })
            .collect::<Vec<_>>();
        queue.write_buffer(bundle.surfaces, 0, cast_slice(&packed_surfaces));
        if !packed_candidates.is_empty() {
            queue.write_buffer(bundle.candidates, 0, cast_slice(&packed_candidates));
        }
        queue.write_buffer(bundle.output, 0, &ZERO_OUTPUT);
        queue.write_buffer(bundle.direct_output, 0, &ZERO_OUTPUT);
        queue.write_buffer(bundle.gi_output, 0, &ZERO_OUTPUT);
        Ok(())
    }

    fn prepare_light_tables(
        &self,
        runtime: &RenderRuntime<I>,
        bundle: &Arc<LightingGpuBundle>,
        inputs: &ExtractedLightingInputs,
    ) -> Result<(usize, usize), LightingGpuPrepareError>
    where
        I::Queue: FrameSync,
    {
        if inputs.directional_handles.len() != inputs.directionals.len()
            || inputs.punctual_handles.len() != inputs.punctuals.len()
        {
            return Err(LightingGpuPrepareError::IdentityPayloadMismatch);
        }
        // The bundle owns fixed-size buffers. Preserve the renderer's existing
        // deterministic first-N admission policy before stable-slot
        // reconciliation; compaction below may remove historical holes, but it
        // must never rebuild more live entries than the physical buffer holds.
        let directional_count = inputs.directionals.len().min(MAX_DIRECTIONALS);
        let punctual_count = inputs.punctuals.len().min(MAX_PUNCTUALS);
        let directional_handles = &inputs.directional_handles[..directional_count];
        let directionals = &inputs.directionals[..directional_count];
        let punctual_handles = &inputs.punctual_handles[..punctual_count];
        let punctuals = &inputs.punctuals[..punctual_count];

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
        let completed = domain.completion(runtime.completed_value());
        let expected_submission = domain
            .submission(runtime.next_completion_value())
            .expect("the runtime's next completion value is nonzero");
        let (directional_count, punctual_count, token) = prepare_light_upload_transaction(
            &mut cache,
            domain,
            completed,
            reset_gpu_copy,
            expected_submission,
            directional_handles,
            directionals,
            punctual_handles,
            punctuals,
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
        Ok((directional_count, punctual_count))
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

    /// Returns only the kernels admitted by the current plan.
    #[must_use]
    pub(crate) fn pass(&self, plan: &LightingPlan) -> Option<LightingGpuPass> {
        let run_direct = plan.recipes.contains(&RecipeId::SpecializedDeterministic);
        let run_gi = plan.recipes.contains(&RecipeId::WorldRestir)
            && plan.candidate_source != GiCandidateSource::Disabled;
        if !run_direct && !run_gi {
            return None;
        }
        let surface_count = self
            .surface_count
            .load(core::sync::atomic::Ordering::Acquire);
        if surface_count == 0 {
            return None;
        }
        let bundle = self.bundle.current();
        Some(LightingGpuPass {
            output: bundle.output,
            direct_output: bundle.direct_output,
            gi_output: bundle.gi_output,
            params: bundle.params,
            directionals: bundle.directionals,
            punctuals: bundle.punctuals,
            candidates: bundle.candidates,
            surfaces: bundle.surfaces,
            radiance: bundle.radiance,
            direct_pipeline: bundle.direct_pipeline,
            direct_bind_group: bundle.direct_bind_group,
            gi_pipeline: bundle.gi_pipeline,
            gi_bind_group: bundle.gi_bind_group,
            composite_pipeline: bundle.composite_pipeline,
            composite_bind_group: bundle.composite_bind_group,
            run_direct,
            run_gi,
            candidate_source: plan.candidate_source,
            candidate_surface_count: surface_count as u32,
            candidate_rays_per_surface: self
                .candidate_rays_per_surface
                .load(core::sync::atomic::Ordering::Acquire),
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
) {
    cache.directionals = IncrementalGpuTable::default();
    cache.punctuals = IncrementalGpuTable::default();
    cache.directionals.invalidate_gpu_copy(domain);
    cache.punctuals.invalidate_gpu_copy(domain);
    cache.directional_slots.clear();
    cache.punctual_slots.clear();
    for (&handle, &payload) in directional_handles.iter().zip(directionals) {
        let gpu_handle = cache.directionals.insert(payload);
        cache.directional_slots.insert(handle, gpu_handle);
    }
    for (&handle, &payload) in punctual_handles.iter().zip(punctuals) {
        let gpu_handle = cache.punctuals.insert(payload);
        cache.punctual_slots.insert(handle, gpu_handle);
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
    wait_idle: F,
) -> Result<(usize, usize, LightUploadToken), LightingGpuPrepareError>
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

    let compact = !reset_gpu_copy
        && (cache
            .directionals
            .projected_capacity_after_reclaim(completed, directional_reconciliation.insertions)
            > MAX_DIRECTIONALS
            || cache
                .punctuals
                .projected_capacity_after_reclaim(completed, punctual_reconciliation.insertions)
                > MAX_PUNCTUALS);
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
        );
    } else {
        cache.directionals.reclaim_completed(completed);
        cache.punctuals.reclaim_completed(completed);
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
    });
    Ok((
        cache.directionals.capacity() as usize,
        cache.punctuals.capacity() as usize,
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
    true
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
        });
        cache.pending = Some(PendingLightUpload {
            token: LightUploadToken(2),
            expected_submission: submission(1),
            directional: cache.directionals.plan_upload(),
            punctual: cache.punctuals.plan_upload(),
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
        let (_, _, token) = prepare_light_upload_transaction(
            &mut cache,
            lighting_submission_domain(1),
            lighting_submission_domain(1).completion(0),
            true,
            lighting_submission_domain(1).submission(1).unwrap(),
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
        });
        cache.pending = Some(PendingLightUpload {
            token: LightUploadToken(2),
            expected_submission: submission(1),
            directional: cache.directionals.plan_upload(),
            punctual: cache.punctuals.plan_upload(),
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
        device.create_shader_module(&ShaderModuleDescriptor::wgsl(COMPOSITE_SHADER));
    let composite_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("lighting-composite-layout".into()),
        entries: vec![
            storage_layout(0, OUTPUT_BYTES, false),
            storage_layout(1, OUTPUT_BYTES, true),
            storage_layout(2, OUTPUT_BYTES, true),
            uniform_layout(3),
        ],
    });
    let composite_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("lighting-composite-pipeline-layout".into()),
        bind_group_layouts: vec![composite_layout],
        push_constant_ranges: Vec::new(),
    });
    let composite_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("lighting-composite-compute".into()),
        layout: Some(composite_pipeline_layout),
        module: composite_shader,
        entry_point: "main".into(),
    });
    let composite_bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("lighting-composite-bind-group".into()),
        layout: Some(composite_layout),
        entries: vec![
            buffer_binding(0, output, OUTPUT_BYTES),
            buffer_binding(1, direct_output, OUTPUT_BYTES),
            buffer_binding(2, gi_output, OUTPUT_BYTES),
            buffer_binding(3, params, PARAM_BYTES),
        ],
    });

    LightingGpuBundle {
        output,
        direct_output,
        gi_output,
        params,
        directionals,
        punctuals,
        candidates,
        surfaces,
        radiance,
        direct_pipeline,
        direct_bind_group,
        gi_pipeline,
        gi_bind_group,
        composite_pipeline,
        composite_bind_group,
    }
}
