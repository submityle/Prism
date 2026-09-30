//! `wgpu` compute twin of the adaptive volumetric shadow-map transmittance
//! lookup ([`AvsmCurve::transmittance_at`](prism_render_architecture::volumetric::avsm::AvsmCurve::transmittance_at)).
//!
//! An `AVSM` curve is a set of `(depth, transmittance)` control points kept
//! sorted by ascending depth with a monotone non-increasing transmittance
//! sequence. The `insert`/`compress` state machine that builds the curve is
//! inherently sequential and stays on the `CPU`; this kernel twins only the
//! pure, read-only sampling query, which is what a shadowing raymarch evaluates
//! once per froxel or sample. [`GpuAvsmTransmittance`] uploads the shared node
//! array once and samples every query depth in parallel.
//!
//! # Correctness model
//!
//! The kernel mirrors the `CPU` sampler exactly: before the first node (or on
//! an empty curve) the transmittance is `1`; at or beyond the last node it is
//! the last node's transmittance; between two nodes the bracketing pair is
//! linearly interpolated with the same `EPS`-guarded span and saturated into
//! `[0, 1]`. Every step is compare/select plus one multiply-add, so `CPU` and
//! `GPU` agree to a few `ULP`. The parity test builds real curves through
//! `insert`/`compress` and sweeps query depths across and beyond the node
//! range, so a wrong bracket or a dropped saturation could not pass.
//!
//! # Portability
//!
//! The kernel is compare/`clamp` plus multiply-add in the portable core-`WGSL`
//! subset, so it runs unmodified on Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard piecewise-linear shadow-curve lookup plus `wgpu`
//! compute dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// One `AVSM` curve control point: the surviving `transmittance` at a light-space
/// `depth`. Mirrors `AvsmNode` in the `CPU` golden.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AvsmSampleNode {
    /// Light-space depth of the control point.
    pub depth: f32,
    /// Surviving transmittance at `depth`, in `[0, 1]`.
    pub transmittance: f32,
}

/// One curve control point as uploaded. `8`-byte `repr(C)` matching `Node` in
/// `shaders/avsm_transmittance.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuNode {
    depth: f32,
    transmittance: f32,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/avsm_transmittance.wesl`: the query and node counts plus two pad
/// words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    query_count: u32,
    node_count: u32,
    pad0: u32,
    pad1: u32,
}

/// A compiled, reusable `AVSM` transmittance-sampling pipeline.
pub struct GpuAvsmTransmittance {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuAvsmTransmittance {
    /// Compiles the sampling kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuAvsmTransmittance {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_avsm_transmittance"),
            source: ShaderSource::Wgsl(include_str!("../shaders/avsm_transmittance.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_avsm_transmittance_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_avsm_transmittance_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_avsm_transmittance_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("avsm_transmittance_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuAvsmTransmittance {
            module,
            layout,
            pipeline,
        }
    }

    /// Samples the curve given by `nodes` at every depth in `depths`, returning
    /// one transmittance per query in input order.
    ///
    /// Each result equals the `CPU`
    /// [`AvsmCurve::transmittance_at`](prism_render_architecture::volumetric::avsm::AvsmCurve::transmittance_at)
    /// for the same curve and depth to within the tolerance documented on this
    /// module. `nodes` must be the curve's control points in ascending `depth`
    /// order (as returned by `AvsmCurve::nodes`). An empty `depths` slice yields
    /// an empty result; an empty `nodes` slice yields `1.0` for every query
    /// (nothing occludes yet) — both are handled on the host because storage
    /// buffers cannot be zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, nodes: &[AvsmSampleNode], depths: &[f32]) -> Vec<f32> {
        if depths.is_empty() {
            return Vec::new();
        }
        if nodes.is_empty() {
            // Matches the CPU sampler: an empty curve occludes nothing.
            return alloc_ones(depths.len());
        }
        let device = ctx.device();

        let gpu_nodes: Vec<GpuNode> = nodes
            .iter()
            .map(|n| GpuNode {
                depth: n.depth,
                transmittance: n.transmittance,
            })
            .collect();

        let gpu_params = Params {
            query_count: depths.len() as u32,
            node_count: nodes.len() as u32,
            pad0: 0,
            pad1: 0,
        };

        let out_bytes = (depths.len() as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_avsm_transmittance_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let nodes_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_avsm_transmittance_nodes"),
            contents: bytemuck::cast_slice(&gpu_nodes),
            usage: BufferUsages::STORAGE,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_avsm_transmittance_queries"),
            contents: bytemuck::cast_slice(depths),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_avsm_transmittance_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_avsm_transmittance_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_avsm_transmittance_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: nodes_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: queries_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_avsm_transmittance_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_avsm_transmittance_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (depths.len() as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&results_buf, 0, &results_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        results_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = results_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let raw = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(raw.len(), depths.len());
        raw
    }
}

/// Builds a `Vec<f32>` of `len` ones for the empty-curve host fast path.
fn alloc_ones(len: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(len);
    v.resize(len, 1.0);
    v
}

/// Builds a compute-visible buffer binding layout entry.
fn buffer_entry(binding: u32, ty: BufferBindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
