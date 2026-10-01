//! `wgpu` compute twin of Prism's adaptive variable-node transmittance curve
//! lookup
//! ([`sample_transmittance`](prism_render_architecture::hair::adaptive_transmittance::sample_transmittance),
//! which applies the private `sample_nodes_shadow` receiver-query rule).
//!
//! A deep-shadow groom compresses each light ray's occlusion into an adaptive
//! [`CompressedCurve`](prism_render_architecture::hair::adaptive_transmittance::CompressedCurve):
//! a variable number of `(depth, transmittance)` control nodes whose
//! piecewise-linear reconstruction stays within tolerance of the full curve.
//! Shading a receiver then samples that curve at the receiver's light-space
//! depth to recover how much light survives to it. This kernel is that sample
//! step evaluated for a whole batch of receiver depths against one shared
//! compressed curve.
//!
//! # What the kernel evaluates
//!
//! [`GpuHairAdaptiveTransmittance::eval`] takes one shared
//! [`CompressedCurve`](prism_render_architecture::hair::adaptive_transmittance::CompressedCurve)
//! plus a batch of receiver depths and returns one transmittance per depth,
//! preserving input order — the array-in/array-out form used to resolve a tile
//! of receivers against a cached shadow curve. The depth index is the
//! invocation id (`@compute @workgroup_size(64)`, one-dimensional dispatch over
//! `global_invocation_id.x`); invocations past the depth count early-return. The
//! lookup rule mirrors the golden exactly: fully lit (`1.0`) in front of the
//! frontmost node, the deepest node's value at or beyond the last node, and a
//! linear interpolation of the two bracketing nodes in between. An empty curve
//! reads fully transmissive (`1.0`) everywhere.
//!
//! # The curve lives on the host, the depths vary per thread
//!
//! The curve nodes are uploaded once as a shared read-only storage buffer,
//! flattened as `(depth, transmittance)` pairs (two floats per node, tightly
//! packed to avoid the `std430` `vec2` stride); the per-thread variation is the
//! receiver depth read from a flat storage array. The golden performs no
//! sanitisation of the node data or the query depth, so neither does the twin —
//! callers keep every uploaded value finite.
//!
//! # Distinct from the deep-opacity twin
//!
//! This is **not** the layered deep-opacity decode of
//! [`GpuHairDeepTransmittanceSample`](crate::deep_transmittance_sample::GpuHairDeepTransmittanceSample)
//! (which reads fixed equi-depth opacity layers). Here the curve is a
//! variable-length, adaptively thinned node list and the lookup is the
//! bracketing-pair linear interpolation with the "fully lit in front / last
//! value beyond" receiver rule — different input contract, different lookup.
//!
//! # Correctness model
//!
//! The interpolation is a subtract, a divide and a mul/add a `GPU` may fuse, so
//! `CPU` and `GPU` agree to within the documented fma tolerance
//! (`abs_diff < 1e-4` or `rel_diff < 1e-3`) rather than bit-for-bit. The
//! front/back guards and the `span == 0` collapse are exact integer-indexed
//! branches that match the golden on both sides.
//!
//! # Portability
//!
//! The kernel uses only comparisons, a subtract, a divide and a mul/add, with no
//! `exp`, `pow`, `sin` or optional device feature, so it runs unmodified on
//! Metal, Vulkan and DX12.
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard deep-shadow transmittance curve lookup plus `wgpu`
//! compute dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::hair::adaptive_transmittance::{
    sample_transmittance, CompressedCurve,
};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Uniform parameters for one lookup dispatch. Layout matches `Params` in
/// `shaders/adaptive_transmittance.wesl`: the node count and query count in a
/// single `16`-byte uniform slot (two `u32` plus padding).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    node_count: u32,
    query_count: u32,
    pad0: u32,
    pad1: u32,
}

/// A compiled, reusable per-depth adaptive transmittance lookup pipeline.
pub struct GpuHairAdaptiveTransmittance {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairAdaptiveTransmittance {
    /// Compiles the per-depth transmittance lookup kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset (comparisons plus a
    /// divide and a mul/add), so no optional device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairAdaptiveTransmittance {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_adaptive_transmittance"),
            source: ShaderSource::Wgsl(
                include_str!("../shaders/adaptive_transmittance.wesl").into(),
            ),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_adaptive_transmittance_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_adaptive_transmittance_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_adaptive_transmittance_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairAdaptiveTransmittance {
            module,
            layout,
            pipeline,
        }
    }

    /// Samples the shared `curve` at every receiver depth in `depths`, returning
    /// one transmittance per depth in input order.
    ///
    /// The value for depth `i` equals the `CPU` golden
    /// [`sample_transmittance`](prism_render_architecture::hair::adaptive_transmittance::sample_transmittance)
    /// of `curve` at `depths[i]` to within the module's documented fma
    /// tolerance. An empty batch yields an empty vector without a dispatch —
    /// storage buffers cannot be zero-sized. An empty curve is uploaded as a
    /// single padding node so the buffer stays non-empty; the kernel reads
    /// `node_count == 0` from the uniform and returns `1.0` regardless.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, curve: &CompressedCurve, depths: &[f32]) -> Vec<f32> {
        let query_count = depths.len();
        if query_count == 0 {
            return Vec::new();
        }

        let device = ctx.device();

        let node_count = curve.nodes.len();
        let uniforms = Params {
            node_count: node_count as u32,
            query_count: query_count as u32,
            pad0: 0,
            pad1: 0,
        };

        // Curve nodes, flattened (depth, transmittance) pairs shared by every
        // thread. A zero-node curve still needs a non-empty storage buffer, so
        // upload one padding pair; the kernel ignores it when node_count == 0.
        let mut node_values: Vec<f32> = Vec::with_capacity(node_count.max(1) * 2);
        for node in &curve.nodes {
            node_values.push(node.depth);
            node_values.push(node.transmittance);
        }
        if node_values.is_empty() {
            node_values.push(0.0);
            node_values.push(1.0);
        }

        // Output is one f32 (4 bytes) per query depth.
        let out_bytes = (query_count as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_adaptive_transmittance_params"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM,
        });
        let nodes_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_adaptive_transmittance_nodes"),
            contents: bytemuck::cast_slice(&node_values),
            usage: BufferUsages::STORAGE,
        });
        let depths_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_adaptive_transmittance_depths"),
            contents: bytemuck::cast_slice(depths),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_adaptive_transmittance_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_adaptive_transmittance_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_adaptive_transmittance_bind_group"),
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
                    resource: depths_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_adaptive_transmittance_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_adaptive_transmittance_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (query_count as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &out_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        out_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = out_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let out = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        out_stage.unmap();

        out
    }
}

/// The `CPU` golden adaptive-curve lookup for one depth, re-exported so the
/// parity test can assert the device twin against the identical reference it
/// mirrors.
#[must_use]
pub fn reference_adaptive_transmittance_sample(curve: &CompressedCurve, depth: f32) -> f32 {
    sample_transmittance(curve, depth)
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
