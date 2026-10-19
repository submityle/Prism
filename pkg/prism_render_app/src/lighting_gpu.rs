//! GPU execution for the deterministic direct-light recipe.

use bytemuck::{bytes_of, cast_slice, Pod, Zeroable};
use prism_ecs::prelude::Resource;
use prism_render_architecture::lighting::contract::LightingPlan;
use prism_render_driver::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingResource, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    ComputePipelineDescriptor, PipelineLayoutDescriptor, RenderQueue, ShaderModuleDescriptor,
    ShaderStages,
};
use prism_render_runtime::{RenderRuntime, ReplayHandle};

use crate::lighting::{ExtractedLightingInputs, LightingDirectionalInput, LightingPunctualInput};

const MAX_DIRECTIONALS: usize = 1024;
const MAX_PUNCTUALS: usize = 1024;
const OUTPUT_BYTES: u64 = 16;
const DIRECTIONAL_BYTES: u64 =
    (MAX_DIRECTIONALS * core::mem::size_of::<LightingDirectionalInput>()) as u64;
const PUNCTUAL_BYTES: u64 = (MAX_PUNCTUALS * core::mem::size_of::<LightingPunctualInput>()) as u64;

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

const LIGHTING_SHADER: &str = r#"
struct DirectionalLight {
    direction_to_light: vec3<f32>,
    visibility: f32,
    illuminance: vec3<f32>,
    padding: f32,
};

struct PunctualLight {
    position: vec3<f32>,
    range: f32,
    intensity: vec3<f32>,
    spot_scale: f32,
    direction: vec3<f32>,
    spot_offset: f32,
    visibility: f32,
    padding: vec3<f32>,
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

fn punctual_attenuation(light: PunctualLight, position: vec3<f32>) -> f32 {
    let to_light = light.position - position;
    let distance_squared = dot(to_light, to_light);
    if (distance_squared <= 0.0) {
        return 0.0;
    }
    let distance = sqrt(distance_squared);
    let direction = to_light / distance;
    var attenuation = 1.0 / max(distance_squared, 1.0e-4);
    if (light.range > 0.0) {
        let ratio = min(distance_squared / (light.range * light.range), 1.0);
        let window = max(1.0 - ratio * ratio, 0.0);
        attenuation = attenuation * window * window;
    }
    if (light.spot_scale != 0.0) {
        let cone = clamp(dot(light.direction, -direction) * light.spot_scale + light.spot_offset, 0.0, 1.0);
        attenuation = attenuation * cone * cone;
    }
    return attenuation * clamp(light.visibility, 0.0, 1.0);
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
        let to_light = light.position - params.position.xyz;
        let distance_squared = dot(to_light, to_light);
        if (distance_squared > 0.0) {
            let direction = to_light * inverseSqrt(distance_squared);
            let cosine = max(dot(normal, direction), 0.0);
            let attenuation = punctual_attenuation(light, params.position.xyz);
            radiance = radiance + params.albedo.xyz * light.intensity * (cosine * attenuation);
        }
    }
    output[id.x] = vec4<f32>(radiance, 1.0);
}
"#;

/// The GPU objects used by one lighting execution bundle.
pub struct LightingGpuBundle {
    /// Persistent HDR radiance output for the surface sample.
    pub output: prism_render_driver::BufferId,
    /// Uniform surface and light counts.
    pub params: prism_render_driver::BufferId,
    /// Directional light storage buffer.
    pub directionals: prism_render_driver::BufferId,
    /// Point and spot light storage buffer.
    pub punctuals: prism_render_driver::BufferId,
    /// Compute pipeline for the deterministic direct-light kernel.
    pub pipeline: prism_render_driver::ComputePipelineId,
    /// Bind group containing all kernel resources.
    pub bind_group: prism_render_driver::BindGroupId,
}

/// A graph-ready view of the current lighting GPU bundle.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LightingGpuPass {
    pub output: prism_render_driver::BufferId,
    pub params: prism_render_driver::BufferId,
    pub directionals: prism_render_driver::BufferId,
    pub punctuals: prism_render_driver::BufferId,
    pub pipeline: prism_render_driver::ComputePipelineId,
    pub bind_group: prism_render_driver::BindGroupId,
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
        let bundle = runtime.register_replay("lighting-direct-kernel", |device, _queue| {
            Ok(build_bundle(device))
        })?;
        Ok(Self {
            bundle,
            marker: core::marker::PhantomData,
        })
    }

    /// Uploads the current immutable lighting snapshot before dispatch.
    pub fn prepare(
        &self,
        _plan: &LightingPlan,
        inputs: &ExtractedLightingInputs,
        queue: &I::Queue,
    ) {
        let bundle = self.bundle.current();
        let directional_count = inputs.directionals.len().min(MAX_DIRECTIONALS);
        let punctual_count = inputs.punctuals.len().min(MAX_PUNCTUALS);
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
            counts: [directional_count as u32, punctual_count as u32, 0, 0],
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
    }

    /// Returns the current bundle as graph recording data.
    #[must_use]
    pub(crate) fn pass(&self) -> LightingGpuPass {
        let bundle = self.bundle.current();
        LightingGpuPass {
            output: bundle.output,
            params: bundle.params,
            directionals: bundle.directionals,
            punctuals: bundle.punctuals,
            pipeline: bundle.pipeline,
            bind_group: bundle.bind_group,
        }
    }

    /// Returns the current output buffer for diagnostics and readback tests.
    #[must_use]
    pub fn output_buffer(&self) -> prism_render_driver::BufferId {
        self.bundle.current().output
    }
}

impl<I> Resource for LightingGpuState<I> where I: prism_render_driver::Instance + 'static {}

fn build_bundle<D: prism_render_driver::RenderDevice>(device: &D) -> LightingGpuBundle {
    let output = device.create_buffer(&BufferDescriptor {
        label: Some("lighting-direct-output".into()),
        size: OUTPUT_BYTES,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params = device.create_buffer(&BufferDescriptor {
        label: Some("lighting-direct-params".into()),
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
    let shader = device.create_shader_module(&ShaderModuleDescriptor::wgsl(LIGHTING_SHADER));
    let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("lighting-direct-layout".into()),
        entries: vec![
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: Some(OUTPUT_BYTES),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(PARAM_BYTES),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: Some(DIRECTIONAL_BYTES),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: Some(PUNCTUAL_BYTES),
                },
                count: None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("lighting-direct-pipeline-layout".into()),
        bind_group_layouts: vec![layout],
        push_constant_ranges: Vec::new(),
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("lighting-direct-compute".into()),
        layout: Some(pipeline_layout),
        module: shader,
        entry_point: "main".into(),
    });
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("lighting-direct-bind-group".into()),
        layout: Some(layout),
        entries: vec![
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer {
                    buffer: output,
                    offset: 0,
                    size: Some(OUTPUT_BYTES),
                },
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Buffer {
                    buffer: params,
                    offset: 0,
                    size: Some(PARAM_BYTES),
                },
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::Buffer {
                    buffer: directionals,
                    offset: 0,
                    size: Some(DIRECTIONAL_BYTES),
                },
            },
            BindGroupEntry {
                binding: 3,
                resource: BindingResource::Buffer {
                    buffer: punctuals,
                    offset: 0,
                    size: Some(PUNCTUAL_BYTES),
                },
            },
        ],
    });
    LightingGpuBundle {
        output,
        params,
        directionals,
        punctuals,
        pipeline,
        bind_group,
    }
}
