//! Host orchestration of the scalar easing-function compute kernel (§24.1
//! twin; animation/UI/tween systems and timeline curves consume these
//! remaps on the GPU side).
//!
//! [`GpuEasing`] batch-evaluates one easing function over a slice of scalar
//! parameters **on a real device**, mirroring the CPU reference family
//! [`prism_math::curve::easing`]. The curve math is **not** duplicated here:
//! a single pipeline is composed at runtime by prefixing the single-sourced
//! fragment [`WGSL_EASING`](prism_math::shader_mirror::WGSL_EASING) ahead of a
//! thin compute wrapper that calls the `prism_ease(op, t)` dispatcher, so the
//! device helpers cannot silently drift from the CPU reference. The function
//! is selected per dispatch by the [`Ease`] op code carried in the uniform.
//!
//! # Parity contract (honest boundary)
//!
//! The polynomial easings (`smoothstep`, `smootherstep`, the quad/cubic
//! family) are bare multiply/add arithmetic and match the CPU path to a tight
//! FMA tolerance. The sinusoidal (`sine_*`) and exponential (`expo_*`) easings
//! call the WGSL `cos`/`sin`/`pow` builtins, which Metal compiles under
//! fast-math, whereas the CPU reference uses deterministic `libm`; the §24.1
//! contract on those is therefore a small absolute+relative tolerance rather
//! than a bit contract. The clamp and the `t <= 0`/`t >= 1`/`t < 0.5` branch
//! literals are identical on both sides, so a given input takes the same
//! branch regardless of device.
//!
//! Standard `wgpu` compute orchestration. No Unreal Engine or Unity source or
//! derived code.

use alloc::string::String;
use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use prism_math::shader_mirror::WGSL_EASING;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, CommandEncoder,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    Device, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModuleDescriptor,
    ShaderSource, ShaderStages,
};

use crate::buffer;
use crate::context::GpuContext;

/// Threads per workgroup; a standard 1D batch tiling.
const WORKGROUP: u32 = 64;

/// Easing-function selector, matching the `prism_ease` dispatcher op codes in
/// [`WGSL_EASING`] and the CPU function ordering in
/// [`prism_math::curve::easing`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Ease {
    /// `3t^2 - 2t^3`, clamped to `[0, 1]`.
    Smoothstep = 0,
    /// `6t^5 - 15t^4 + 10t^3`, clamped to `[0, 1]`.
    Smootherstep = 1,
    /// Quadratic ease-in `t^2`.
    QuadIn = 2,
    /// Quadratic ease-out.
    QuadOut = 3,
    /// Quadratic ease-in-out.
    QuadInOut = 4,
    /// Cubic ease-in `t^3`.
    CubicIn = 5,
    /// Cubic ease-out.
    CubicOut = 6,
    /// Cubic ease-in-out.
    CubicInOut = 7,
    /// Sinusoidal ease-in.
    SineIn = 8,
    /// Sinusoidal ease-out.
    SineOut = 9,
    /// Sinusoidal ease-in-out.
    SineInOut = 10,
    /// Exponential (base-2) ease-in.
    ExpoIn = 11,
    /// Exponential (base-2) ease-out.
    ExpoOut = 12,
    /// Exponential (base-2) ease-in-out.
    ExpoInOut = 13,
}

/// Compute wrapper that applies the selected easing op to each input scalar.
const WRAP: &str = "\
struct Params { count: u32, op: u32 };\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read> src: array<f32>;\n\
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    dst[i] = prism_ease(params.op, src[i]);\n\
}\n";

/// Uniform block carrying the valid element count and the easing op code.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    op: u32,
    _pad: [u32; 2],
}

/// Real-device twin of the scalar easing-function family.
///
/// Build it once per device with [`GpuEasing::new`]; the single pipeline and
/// bind-group layout are created up front and reused across batches and ops.
pub struct GpuEasing {
    pipeline: ComputePipeline,
    layout: BindGroupLayout,
}

impl GpuEasing {
    /// Compiles the easing pipeline on `ctx`'s device, embedding the
    /// single-sourced [`WGSL_EASING`] fragment verbatim.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let device = ctx.device();
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_math_easing_layout"),
            entries: &[
                buffer_layout(
                    0,
                    BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                buffer_layout(
                    1,
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                buffer_layout(
                    2,
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_math_easing_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let mut source = String::new();
        source.push_str(WGSL_EASING);
        source.push('\n');
        source.push_str(WRAP);
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_math_easing"),
            source: ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_math_easing"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuEasing { pipeline, layout }
    }

    /// Batch-applies `ease` to each element of `src` on the device, mirroring
    /// the matching CPU function in [`prism_math::curve::easing`].
    #[must_use]
    pub fn map(&self, ctx: &GpuContext, ease: Ease, src: &[f32]) -> Vec<f32> {
        let n = src.len();
        if n == 0 {
            return Vec::new();
        }
        let device = ctx.device();
        let in_buf = buffer::storage_read(device, "prism_math_easing_in", src);
        let out_bytes = size_of_val(src) as u64;
        let out_buf = buffer::storage_rw_zeroed(device, "prism_math_easing_out", out_bytes);
        let bind_group = self.bind(device, n, ease, &in_buf, &out_buf);

        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_math_easing_encoder"),
        });
        dispatch(&mut enc, &self.pipeline, &bind_group, n);
        let stage = buffer::staging(device, "prism_math_easing_stage", out_bytes);
        buffer::copy(&mut enc, &out_buf, &stage, out_bytes);
        ctx.queue().submit([enc.finish()]);

        buffer::read_back::<f32>(ctx, &stage)
    }

    /// Builds the three-entry bind group (params uniform, input, output).
    fn bind(
        &self,
        device: &Device,
        count: usize,
        ease: Ease,
        input: &Buffer,
        output: &Buffer,
    ) -> BindGroup {
        let params = buffer::uniform(
            device,
            "prism_math_easing_params",
            &Params {
                count: count as u32,
                op: ease as u32,
                _pad: [0; 2],
            },
        );
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_math_easing_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: input.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output.as_entire_binding(),
                },
            ],
        })
    }
}

/// Records a 1D batch dispatch covering `n` elements at [`WORKGROUP`] threads
/// per group.
fn dispatch(
    enc: &mut CommandEncoder,
    pipeline: &ComputePipeline,
    bind_group: &BindGroup,
    n: usize,
) {
    let groups = (n as u32).div_ceil(WORKGROUP);
    let mut pass = enc.begin_compute_pass(&ComputePassDescriptor {
        label: Some("prism_math_easing_pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups(groups, 1, 1);
}

/// One storage/uniform bind-group-layout entry visible to the compute stage.
fn buffer_layout(binding: u32, ty: BindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty,
        count: None,
    }
}
