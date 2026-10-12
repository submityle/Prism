//! The reusable PBR surface pass: shader module, bind-group layout, pipeline
//! layout, compute pipeline, and the batch it evaluates.
//!
//! [`SurfacePass::new`] builds the GPU objects once; [`SurfacePass::evaluate`]
//! uploads a batch and returns the handles a caller records against;
//! [`SurfacePass::record`] produces the `SetPipeline` / `SetBindGroup` /
//! `Dispatch` command list. Nothing here names a backend — the RHI traits carry
//! every call.

use prism_render_driver::{
    BindGroupId, BufferId, ComputeCommand, ComputePipelineDescriptor, ComputePipelineId,
    PipelineLayoutDescriptor, PipelineLayoutId, RenderDevice, RenderQueue, ShaderModuleDescriptor,
    ShaderModuleId,
};

use crate::bindings::{
    create_result_buffer, surface_bind_group, surface_bind_group_layout, upload_surface_batch,
    RESULT_STRIDE,
};
use crate::params::GpuSurfaceParams;
use crate::shader::{
    EVALUATE_ENTRY_POINT, SURFACE_SHADER_LABEL, SURFACE_SHADER_WGSL, WORKGROUP_SIZE,
};

/// The compiled surface pass, owning every RHI object the shader needs.
///
/// The ids are plain values; call [`SurfacePass::destroy`] (or hand them to a
/// deferred-reclaim queue) when the pass is no longer needed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SurfacePass {
    shader: ShaderModuleId,
    bind_group_layout: prism_render_driver::BindGroupLayoutId,
    pipeline_layout: PipelineLayoutId,
    pipeline: ComputePipelineId,
}

/// A ready-to-record surface batch: the uniform buffer, the result buffer, the
/// bind group that pairs them, and the evaluation count.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SurfaceEvaluation {
    /// The uniform batch holding `count` [`GpuSurfaceParams`].
    pub params: BufferId,
    /// The storage buffer the shader writes `count` results into.
    pub results: BufferId,
    /// The bind group pairing `params` and `results`.
    pub bind_group: BindGroupId,
    /// How many surfaces are evaluated.
    pub count: u32,
}

impl SurfaceEvaluation {
    /// The number of result bytes to read back for this batch.
    #[must_use]
    pub fn result_bytes(&self) -> u64 {
        RESULT_STRIDE * u64::from(self.count)
    }
}

impl SurfacePass {
    /// Compiles the shader and creates the layouts, pipeline, and bind-group
    /// layout the surface pass uses.
    #[must_use]
    pub fn new<D: RenderDevice>(device: &D) -> Self {
        let shader = device.create_shader_module(&ShaderModuleDescriptor {
            label: Some(SURFACE_SHADER_LABEL.into()),
            source: prism_render_driver::ShaderSource::Wgsl(SURFACE_SHADER_WGSL.into()),
        });
        let bind_group_layout = surface_bind_group_layout(device);
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some(SURFACE_SHADER_LABEL.into()),
            bind_group_layouts: vec![bind_group_layout],
            push_constant_ranges: Vec::new(),
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some(SURFACE_SHADER_LABEL.into()),
            layout: Some(pipeline_layout),
            module: shader,
            entry_point: EVALUATE_ENTRY_POINT.into(),
        });
        Self {
            shader,
            bind_group_layout,
            pipeline_layout,
            pipeline,
        }
    }

    /// The bind-group layout new bind groups must conform to.
    #[must_use]
    pub fn bind_group_layout(&self) -> prism_render_driver::BindGroupLayoutId {
        self.bind_group_layout
    }

    /// The compiled compute pipeline.
    #[must_use]
    pub fn pipeline(&self) -> ComputePipelineId {
        self.pipeline
    }

    /// The pipeline layout.
    #[must_use]
    pub fn pipeline_layout(&self) -> PipelineLayoutId {
        self.pipeline_layout
    }

    /// The workgroup count the dispatch needs for `count` evaluations.
    #[must_use]
    pub fn dispatch_groups(&self, count: u32) -> u32 {
        count.div_ceil(WORKGROUP_SIZE)
    }

    /// Uploads `params` and pairs them with a fresh result buffer.
    ///
    /// # Panics
    ///
    /// Panics when `params` is longer than the batch capacity.
    #[must_use]
    pub fn evaluate<D: RenderDevice, Q: RenderQueue>(
        &self,
        device: &D,
        queue: &Q,
        params: &[GpuSurfaceParams],
    ) -> SurfaceEvaluation {
        let count = params.len() as u32;
        let uniform = upload_surface_batch(device, queue, params);
        let results = create_result_buffer(device, count);
        let bind_group = surface_bind_group(device, self.bind_group_layout, uniform, results);
        SurfaceEvaluation {
            params: uniform,
            results,
            bind_group,
            count,
        }
    }

    /// The compute commands that shade `evaluation`'s batch.
    #[must_use]
    pub fn record(&self, evaluation: SurfaceEvaluation) -> Vec<ComputeCommand> {
        vec![
            ComputeCommand::SetPipeline(self.pipeline),
            ComputeCommand::SetBindGroup {
                index: 0,
                bind_group: evaluation.bind_group,
                dynamic_offsets: Vec::new(),
            },
            ComputeCommand::Dispatch {
                x: self.dispatch_groups(evaluation.count),
                y: 1,
                z: 1,
            },
        ]
    }

    /// Releases every RHI object the pass owns.
    pub fn destroy<D: RenderDevice>(&self, device: &D) {
        device.destroy_compute_pipeline(self.pipeline);
        device.destroy_pipeline_layout(self.pipeline_layout);
        device.destroy_bind_group_layout(self.bind_group_layout);
        device.destroy_shader_module(self.shader);
    }
}
