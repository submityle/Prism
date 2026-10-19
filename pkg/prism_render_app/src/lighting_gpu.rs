//! Real GPU execution for the selected lighting plan.
//!
//! This is deliberately a small deterministic kernel: it proves the complete
//! plan -> bind group -> compute dispatch -> persistent output path without
//! inventing a second recipe selector. More elaborate lighting kernels can
//! replace the shader while retaining this ownership and graph contract.

use prism_ecs::prelude::Resource;
use prism_render_architecture::lighting::contract::{FieldEpoch, LightingPlan, RecipeId};
use prism_render_driver::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingResource, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    ComputePipelineDescriptor, PipelineLayoutDescriptor, RenderQueue, ShaderModuleDescriptor,
    ShaderStages,
};
use prism_render_runtime::{RenderRuntime, ReplayHandle};

const OUTPUT_BYTES: u64 = 16;
const PARAM_BYTES: u64 = 16;

const LIGHTING_SHADER: &str = r#"
struct Params {
    values: vec4<u32>,
};

@group(0) @binding(0)
var<storage, read_write> output: array<vec4<f32>>;

@group(0) @binding(1)
var<uniform> params: Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arrayLength(&output)) {
        return;
    }
    let values = params.values;
    output[id.x] = vec4<f32>(
        f32(values.x),
        f32(values.y),
        f32(values.z),
        f32(values.w),
    );
}
"#;

/// The GPU objects used by one lighting execution bundle.
pub struct LightingGpuBundle {
    /// Persistent output containing the last dispatched plan signature.
    pub output: prism_render_driver::BufferId,
    /// Uniform input updated once per extracted frame.
    pub params: prism_render_driver::BufferId,
    /// Compute pipeline for the deterministic lighting kernel.
    pub pipeline: prism_render_driver::ComputePipelineId,
    /// Bind group containing output and plan parameters.
    pub bind_group: prism_render_driver::BindGroupId,
}

/// A graph-ready view of the current lighting GPU bundle.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LightingGpuPass {
    pub output: prism_render_driver::BufferId,
    pub params: prism_render_driver::BufferId,
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
        let bundle = runtime.register_replay("lighting-plan-kernel", |device, _queue| {
            Ok(build_bundle(device))
        })?;
        Ok(Self {
            bundle,
            marker: core::marker::PhantomData,
        })
    }

    /// Updates the uniform signature consumed by the next dispatch.
    pub fn prepare(
        &self,
        plan: &LightingPlan,
        epoch: FieldEpoch,
        queue: &I::Queue,
        stats: prism_render_architecture::lighting::contract::SceneStats,
    ) {
        let bundle = self.bundle.current();
        let recipe = plan
            .recipes
            .first()
            .copied()
            .unwrap_or(RecipeId::SpecializedDeterministic);
        let values = [
            recipe.ordinal(),
            epoch
                .geometry
                .saturating_add(epoch.material)
                .saturating_add(epoch.lighting)
                .saturating_add(epoch.environment)
                .saturating_add(epoch.layout),
            stats.lights,
            plan.recipes.len() as u32,
        ];
        let mut bytes = [0_u8; PARAM_BYTES as usize];
        for (index, value) in values.into_iter().enumerate() {
            bytes[index * 4..index * 4 + 4].copy_from_slice(&value.to_ne_bytes());
        }
        queue.write_buffer(bundle.params, 0, &bytes);
    }

    /// Returns the current bundle as graph recording data.
    #[must_use]
    pub(crate) fn pass(&self) -> LightingGpuPass {
        let bundle = self.bundle.current();
        LightingGpuPass {
            output: bundle.output,
            params: bundle.params,
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
        label: Some("lighting-plan-output".into()),
        size: OUTPUT_BYTES,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params = device.create_buffer(&BufferDescriptor {
        label: Some("lighting-plan-params".into()),
        size: PARAM_BYTES,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let shader = device.create_shader_module(&ShaderModuleDescriptor::wgsl(LIGHTING_SHADER));
    let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("lighting-plan-layout".into()),
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
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("lighting-plan-pipeline-layout".into()),
        bind_group_layouts: vec![layout],
        push_constant_ranges: Vec::new(),
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("lighting-plan-compute".into()),
        layout: Some(pipeline_layout),
        module: shader,
        entry_point: "main".into(),
    });
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("lighting-plan-bind-group".into()),
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
        ],
    });
    LightingGpuBundle {
        output,
        params,
        pipeline,
        bind_group,
    }
}
