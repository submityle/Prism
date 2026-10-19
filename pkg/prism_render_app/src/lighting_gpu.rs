//! GPU execution for the deterministic direct-light and initial GI recipes.

use bytemuck::{bytes_of, cast_slice, Pod, Zeroable};
use prism_ecs::prelude::Resource;
use prism_render_architecture::lighting::contract::{LightingPlan, RecipeId};
use prism_render_driver::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingResource, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    ComputePipelineDescriptor, PipelineLayoutDescriptor, RenderQueue, ShaderModuleDescriptor,
    ShaderStages,
};
use prism_render_runtime::{RenderRuntime, ReplayHandle};

use crate::lighting::{
    ExtractedLightingInputs, LightingDirectionalInput, LightingGiCandidateInput,
    LightingPunctualInput,
};

const MAX_DIRECTIONALS: usize = 1024;
const MAX_PUNCTUALS: usize = 1024;
const MAX_GI_CANDIDATES: usize = 1024;
const OUTPUT_BYTES: u64 = 16;
const DIRECTIONAL_BYTES: u64 =
    (MAX_DIRECTIONALS * core::mem::size_of::<LightingDirectionalInput>()) as u64;
const PUNCTUAL_BYTES: u64 = (MAX_PUNCTUALS * core::mem::size_of::<LightingPunctualInput>()) as u64;
const GI_CANDIDATE_BYTES: u64 =
    (MAX_GI_CANDIDATES * core::mem::size_of::<LightingGiCandidateInput>()) as u64;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuLightingParams {
    position: [f32; 4],
    normal: [f32; 4],
    albedo: [f32; 4],
    ambient: [f32; 4],
    counts: [u32; 4],
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
    position: vec4<f32>,
    normal: vec4<f32>,
    albedo: vec4<f32>,
    ambient: vec4<f32>,
    counts: vec4<u32>,
};

@group(0) @binding(0)
var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(1)
var<uniform> params: Params;
@group(0) @binding(2)
var<storage, read> directionals: array<DirectionalLight, 1024>;
@group(0) @binding(3)
var<storage, read> punctuals: array<PunctualLight, 1024>;

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

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arrayLength(&output)) {
        return;
    }
    let normal = normalize_or(params.normal.xyz, vec3<f32>(0.0, 1.0, 0.0));
    var radiance = params.albedo.xyz * params.ambient.xyz;
    let directional_count = min(params.counts.x, 1024u);
    let punctual_count = min(params.counts.y, 1024u);
    for (var index = 0u; index < directional_count; index = index + 1u) {
        let light = directionals[index];
        let direction = normalize_or(light.direction_to_light, vec3<f32>(0.0, 1.0, 0.0));
        let cosine = max(dot(normal, direction), 0.0) * clamp(light.visibility, 0.0, 1.0);
        radiance = radiance + params.albedo.xyz * light.illuminance * cosine;
    }
    for (var index = 0u; index < punctual_count; index = index + 1u) {
        let light = punctuals[index];
        let to_light = light.position_range.xyz - params.position.xyz;
        let distance_squared = dot(to_light, to_light);
        if (distance_squared > 0.0) {
            let direction = to_light * inverseSqrt(distance_squared);
            let cosine = max(dot(normal, direction), 0.0);
            let attenuation = punctual_attenuation(light, params.position.xyz);
            radiance = radiance
                + params.albedo.xyz * light.intensity_spot_scale.xyz * (cosine * attenuation);
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
    position: vec4<f32>,
    normal: vec4<f32>,
    albedo: vec4<f32>,
    ambient: vec4<f32>,
    counts: vec4<u32>,
};

@group(0) @binding(0)
var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(1)
var<uniform> params: Params;
@group(0) @binding(2)
var<storage, read> candidates: array<GiCandidate, 1024>;

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

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arrayLength(&output)) {
        return;
    }
    let primary_normal = normalize_or(params.normal.xyz, vec3<f32>(0.0, 1.0, 0.0));
    let candidate_count = min(params.counts.z, 4u);
    var weight_sum = 0.0;
    var sample_count = 0u;
    var selected_position = vec3<f32>(0.0);
    var selected_normal = vec3<f32>(0.0, 1.0, 0.0);
    var selected_radiance = vec3<f32>(0.0);
    var selected_target_pdf = 0.0;
    var selected_visibility = 0.0;
    for (var index = 0u; index < candidate_count; index = index + 1u) {
        let candidate = candidates[index];
        if (!finite(candidate.position_source_pdf.w) || candidate.position_source_pdf.w <= 0.0 ||
            !finite(candidate.normal_target_pdf.w) || candidate.normal_target_pdf.w <= 0.0) {
            continue;
        }
        let weight = candidate.normal_target_pdf.w / candidate.position_source_pdf.w;
        if (!finite(weight) || weight <= 0.0) {
            continue;
        }
        weight_sum = weight_sum + weight;
        sample_count = sample_count + 1u;
        if (candidate.radiance_selection_u01.w * weight_sum < weight) {
            selected_position = candidate.position_source_pdf.xyz;
            selected_normal = candidate.normal_target_pdf.xyz;
            selected_radiance = candidate.radiance_selection_u01.xyz;
            selected_target_pdf = candidate.normal_target_pdf.w;
            selected_visibility = candidate.visibility_padding.x;
        }
    }
    if (sample_count == 0u || weight_sum <= 0.0 || selected_target_pdf <= 0.0) {
        output[id.x] = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        return;
    }
    let to_sample = selected_position - params.position.xyz;
    let distance_squared = dot(to_sample, to_sample);
    if (!finite(distance_squared) || distance_squared <= 1.0e-8) {
        output[id.x] = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        return;
    }
    let direction = to_sample * inverseSqrt(distance_squared);
    let sample_normal = normalize_or(selected_normal, vec3<f32>(0.0, 1.0, 0.0));
    let cosine_primary = max(dot(primary_normal, direction), 0.0);
    let cosine_secondary = max(dot(sample_normal, -direction), 0.0);
    let geometry = cosine_primary * cosine_secondary / distance_squared;
    let visibility = clamp(selected_visibility, 0.0, 1.0);
    let reservoir_weight = weight_sum / (f32(sample_count) * selected_target_pdf);
    let result = params.albedo.xyz * selected_radiance * geometry * visibility * reservoir_weight;
    output[id.x] = vec4<f32>(result, 1.0);
}
"#;

const COMPOSITE_SHADER: &str = r#"
struct Params {
    position: vec4<f32>,
    normal: vec4<f32>,
    albedo: vec4<f32>,
    ambient: vec4<f32>,
    counts: vec4<u32>,
};

@group(0) @binding(0)
var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(1)
var<storage, read> direct: array<vec4<f32>>;
@group(0) @binding(2)
var<storage, read> gi: array<vec4<f32>>;
@group(0) @binding(3)
var<uniform> params: Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arrayLength(&output)) {
        return;
    }
    var value = vec3<f32>(0.0);
    if (params.counts.w == 1u || params.counts.w == 3u) {
        value = value + direct[id.x].xyz;
    }
    if (params.counts.w == 2u || params.counts.w == 3u) {
        value = value + gi[id.x].xyz;
    }
    output[id.x] = vec4<f32>(value, 1.0);
}
"#;

/// The GPU objects used by one lighting execution bundle.
pub struct LightingGpuBundle {
    /// Final HDR radiance output for the surface sample.
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
    pub direct_pipeline: prism_render_driver::ComputePipelineId,
    pub direct_bind_group: prism_render_driver::BindGroupId,
    pub gi_pipeline: prism_render_driver::ComputePipelineId,
    pub gi_bind_group: prism_render_driver::BindGroupId,
    pub composite_pipeline: prism_render_driver::ComputePipelineId,
    pub composite_bind_group: prism_render_driver::BindGroupId,
    pub run_direct: bool,
    pub run_gi: bool,
}

impl LightingGpuPass {
    #[must_use]
    pub(crate) const fn pass_count(self) -> u64 {
        (self.run_direct as u64) + (self.run_gi as u64) + 1
    }
}

/// Render-world ownership of lighting GPU resources with device-loss replay.
pub struct LightingGpuState<I>
where
    I: prism_render_driver::Instance + 'static,
{
    bundle: ReplayHandle<LightingGpuBundle>,
    marker: core::marker::PhantomData<fn() -> I>,
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
            marker: core::marker::PhantomData,
        })
    }

    /// Uploads the current immutable lighting snapshot before dispatch.
    pub fn prepare(&self, plan: &LightingPlan, inputs: &ExtractedLightingInputs, queue: &I::Queue) {
        let bundle = self.bundle.current();
        let directional_count = inputs.directionals.len().min(MAX_DIRECTIONALS);
        let punctual_count = inputs.punctuals.len().min(MAX_PUNCTUALS);
        let candidate_count = inputs.gi_candidates.len().min(MAX_GI_CANDIDATES);
        let mut execution_flags = 0_u32;
        if plan.recipes.contains(&RecipeId::SpecializedDeterministic) {
            execution_flags |= 1;
        }
        if plan.recipes.contains(&RecipeId::WorldRestir) {
            execution_flags |= 2;
        }
        let params = GpuLightingParams {
            position: [
                inputs.surface.position[0],
                inputs.surface.position[1],
                inputs.surface.position[2],
                0.0,
            ],
            normal: [
                inputs.surface.normal[0],
                inputs.surface.normal[1],
                inputs.surface.normal[2],
                0.0,
            ],
            albedo: [
                inputs.surface.albedo[0],
                inputs.surface.albedo[1],
                inputs.surface.albedo[2],
                0.0,
            ],
            ambient: [
                inputs.surface.ambient[0],
                inputs.surface.ambient[1],
                inputs.surface.ambient[2],
                0.0,
            ],
            counts: [
                directional_count as u32,
                punctual_count as u32,
                candidate_count as u32,
                execution_flags,
            ],
        };
        queue.write_buffer(bundle.params, 0, bytes_of(&params));
        queue.write_buffer(
            bundle.directionals,
            0,
            cast_slice(&inputs.directionals[..directional_count]),
        );
        queue.write_buffer(
            bundle.punctuals,
            0,
            cast_slice(&inputs.punctuals[..punctual_count]),
        );
        queue.write_buffer(
            bundle.candidates,
            0,
            cast_slice(&inputs.gi_candidates[..candidate_count]),
        );
        if execution_flags & 1 == 0 {
            queue.write_buffer(bundle.direct_output, 0, &ZERO_OUTPUT);
        }
        if execution_flags & 2 == 0 {
            queue.write_buffer(bundle.gi_output, 0, &ZERO_OUTPUT);
        }
    }

    /// Returns only the kernels admitted by the current plan.
    #[must_use]
    pub(crate) fn pass(&self, plan: &LightingPlan) -> Option<LightingGpuPass> {
        let run_direct = plan.recipes.contains(&RecipeId::SpecializedDeterministic);
        let run_gi = plan.recipes.contains(&RecipeId::WorldRestir);
        if !run_direct && !run_gi {
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
            direct_pipeline: bundle.direct_pipeline,
            direct_bind_group: bundle.direct_bind_group,
            gi_pipeline: bundle.gi_pipeline,
            gi_bind_group: bundle.gi_bind_group,
            composite_pipeline: bundle.composite_pipeline,
            composite_bind_group: bundle.composite_bind_group,
            run_direct,
            run_gi,
        })
    }

    /// Returns the final output buffer for diagnostics and readback tests.
    #[must_use]
    pub fn output_buffer(&self) -> prism_render_driver::BufferId {
        self.bundle.current().output
    }
}

impl<I> Resource for LightingGpuState<I> where I: prism_render_driver::Instance + 'static {}

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
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let direct_output = device.create_buffer(&BufferDescriptor {
        label: Some("lighting-direct-output".into()),
        size: OUTPUT_BYTES,
        usage: BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let gi_output = device.create_buffer(&BufferDescriptor {
        label: Some("lighting-gi-output".into()),
        size: OUTPUT_BYTES,
        usage: BufferUsages::STORAGE,
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

    let direct_shader = device.create_shader_module(&ShaderModuleDescriptor::wgsl(DIRECT_SHADER));
    let direct_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("lighting-direct-layout".into()),
        entries: vec![
            storage_layout(0, OUTPUT_BYTES, false),
            uniform_layout(1),
            storage_layout(2, DIRECTIONAL_BYTES, true),
            storage_layout(3, PUNCTUAL_BYTES, true),
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
        ],
    });

    let gi_shader = device.create_shader_module(&ShaderModuleDescriptor::wgsl(GI_SHADER));
    let gi_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("lighting-gi-layout".into()),
        entries: vec![
            storage_layout(0, OUTPUT_BYTES, false),
            uniform_layout(1),
            storage_layout(2, GI_CANDIDATE_BYTES, true),
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
        direct_pipeline,
        direct_bind_group,
        gi_pipeline,
        gi_bind_group,
        composite_pipeline,
        composite_bind_group,
    }
}
