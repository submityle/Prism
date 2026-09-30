//! `wgpu` compute twin of the adaptive volumetric shadow-map curve-area
//! reduction ([`AvsmCurve::area`](prism_render_architecture::volumetric::avsm::AvsmCurve::area)).
//!
//! An `AVSM` curve is a set of `(depth, transmittance)` control points kept
//! sorted by ascending depth with a monotone non-increasing transmittance
//! sequence. The area under its piecewise-linear profile, computed by the
//! trapezoidal rule, is the compression error metric: comparing the area before
//! and after [`AvsmCurve::compress`] bounds how far the adaptive merge shifted
//! the curve. The `insert`/`compress` state machine that builds each curve is
//! inherently sequential and stays on the `CPU`; this kernel twins only the
//! pure whole-curve reduction, which a shadow-quality pass evaluates once per
//! froxel or shadow texel. [`GpuAvsmArea`] uploads every curve's nodes once as a
//! single flattened array and reduces all curves in parallel, one invocation
//! per curve.
//!
//! # Correctness model
//!
//! The kernel mirrors the `CPU` reduction exactly: a curve with fewer than two
//! nodes encloses no area and yields `0`; otherwise it sums
//! `0.5 * (lo.transmittance + hi.transmittance) * (hi.depth - lo.depth)` over
//! each adjacent node pair in ascending index order. The accumulation order and
//! per-segment form match the golden, so `CPU` and `GPU` agree to a few `ULP`.
//! The parity test builds real curves through `insert`/`compress` (including
//! sub-two-node curves and out-of-order, duplicate and negative inputs) and
//! compares each area against [`AvsmCurve::area`], so a wrong bracket or a
//! dropped segment could not pass.
//!
//! # Portability
//!
//! The kernel is multiply-add plus compare in the portable core-`WGSL` subset,
//! so it runs unmodified on Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard trapezoidal-rule integration of a shadow-curve profile
//! plus `wgpu` compute dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::avsm_transmittance::AvsmSampleNode;
use crate::context::GpuContext;

/// One curve control point as uploaded. `8`-byte `repr(C)` matching `Node` in
/// `shaders/avsm_area.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuNode {
    depth: f32,
    transmittance: f32,
}

/// Where one curve's nodes live in the shared flattened array. `8`-byte
/// `repr(C)` matching `CurveRange` in `shaders/avsm_area.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuCurveRange {
    offset: u32,
    count: u32,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/avsm_area.wesl`: the curve count plus three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    curve_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable `AVSM` curve-area pipeline.
pub struct GpuAvsmArea {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuAvsmArea {
    /// Compiles the curve-area kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuAvsmArea {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_avsm_area"),
            source: ShaderSource::Wgsl(include_str!("../shaders/avsm_area.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_avsm_area_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_avsm_area_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_avsm_area_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("avsm_area_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuAvsmArea {
            module,
            layout,
            pipeline,
        }
    }

    /// Reduces every curve in `curves` to its trapezoidal transmittance area,
    /// returning one area per curve in input order.
    ///
    /// Each result equals the `CPU`
    /// [`AvsmCurve::area`](prism_render_architecture::volumetric::avsm::AvsmCurve::area)
    /// for the same control points to within the tolerance documented on this
    /// module. Every inner slice must be a curve's control points in ascending
    /// `depth` order (as returned by `AvsmCurve::nodes`). A curve with fewer
    /// than two nodes contributes `0`. An empty `curves` slice yields an empty
    /// result; when every curve is empty (no nodes anywhere) the all-zero result
    /// is produced on the host because storage buffers cannot be zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, curves: &[Vec<AvsmSampleNode>]) -> Vec<f32> {
        if curves.is_empty() {
            return Vec::new();
        }

        // Flatten every curve's nodes into one array and record each curve's
        // [offset, count) window into it.
        let total_nodes: usize = curves.iter().map(Vec::len).sum();
        if total_nodes == 0 {
            // Nothing to integrate anywhere: every curve encloses zero area.
            return alloc_zeros(curves.len());
        }

        let device = ctx.device();

        let mut gpu_nodes: Vec<GpuNode> = Vec::with_capacity(total_nodes);
        let mut ranges: Vec<GpuCurveRange> = Vec::with_capacity(curves.len());
        for curve in curves {
            let offset = gpu_nodes.len() as u32;
            for n in curve {
                gpu_nodes.push(GpuNode {
                    depth: n.depth,
                    transmittance: n.transmittance,
                });
            }
            ranges.push(GpuCurveRange {
                offset,
                count: curve.len() as u32,
            });
        }

        let gpu_params = Params {
            curve_count: curves.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let out_bytes = (curves.len() as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_avsm_area_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let nodes_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_avsm_area_nodes"),
            contents: bytemuck::cast_slice(&gpu_nodes),
            usage: BufferUsages::STORAGE,
        });
        let ranges_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_avsm_area_ranges"),
            contents: bytemuck::cast_slice(&ranges),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_avsm_area_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_avsm_area_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_avsm_area_bind_group"),
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
                    resource: ranges_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_avsm_area_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_avsm_area_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (curves.len() as u32).div_ceil(64);
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
        debug_assert_eq!(raw.len(), curves.len());
        raw
    }
}

/// Builds a `Vec<f32>` of `len` zeros for the all-empty-curves host fast path.
fn alloc_zeros(len: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(len);
    v.resize(len, 0.0);
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
