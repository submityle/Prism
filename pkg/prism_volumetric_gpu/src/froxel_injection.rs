//! `wgpu` compute twin of the froxel injection weight
//! ([`froxel_injection_weight`](prism_render_architecture::volumetric::fog::froxel_injection_weight)).
//!
//! The unified volumetric fog (design section 9f) only *adds* energy into the
//! shared froxel volume, never overwriting deeper contributions: near slices
//! (small `depth_slice`) receive full weight and far slices taper linearly to
//! zero. [`froxel_injection_weight`] returns `0` when `slice_count` is zero and
//! otherwise `saturate(1 - min(depth_slice, slice_count) / slice_count)`, so
//! the weight always lands in `0..=1` and clamping `depth_slice` to the slice
//! count keeps an out-of-range index from producing a negative weight. The
//! `CPU` golden
//! [`froxel_injection_weight`](prism_render_architecture::volumetric::fog::froxel_injection_weight)
//! owns that math; [`GpuFroxelInjection`] is the on-device twin that runs one
//! thread per query and reproduces the same value.
//!
//! # Correctness model
//!
//! The kernel contains no transcendental call — an unsigned `min`, a division
//! and a saturating clamp — so `CPU` and `GPU` evaluate the same closed-form
//! algebra on the same unsigned-integer inputs. The only slack is a legal
//! multiply-add contraction of a few `ULP`, so the parity test asserts a tight
//! tolerance (`abs_diff < 1e-6` or `rel_diff < 1e-5`). The scenes also assert
//! the documented `[0, 1]` range, that the nearest slice gets full weight, that
//! a zero slice count yields `0`, that an out-of-range index clamps to `0`, and
//! that the weight decreases monotonically with depth, so a degenerate kernel
//! could not pass.
//!
//! # Portability
//!
//! The kernel is an unsigned `min`, an integer-to-float conversion, a divide
//! and a `clamp` in the portable core-`WGSL` subset — no `exp`, `pow` or
//! optional device feature — so it runs unmodified on Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard additive froxel-injection weighting plus `wgpu`
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

/// One froxel-injection query: the depth slice index and the total slice
/// count.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FroxelInjectionQuery {
    /// Depth slice index into the shared froxel volume.
    pub depth_slice: u32,
    /// Total number of depth slices in the froxel volume.
    pub slice_count: u32,
}

/// One query as uploaded. `16`-byte `repr(C)` matching `Query` in
/// `shaders/froxel_injection.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    depth_slice: u32,
    slice_count: u32,
    pad0: u32,
    pad1: u32,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/froxel_injection.wesl`: the query count plus three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable froxel-injection pipeline.
pub struct GpuFroxelInjection {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuFroxelInjection {
    /// Compiles the froxel-injection kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuFroxelInjection {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_froxel_injection"),
            source: ShaderSource::Wgsl(include_str!("../shaders/froxel_injection.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_froxel_injection_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_froxel_injection_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_froxel_injection_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("froxel_injection_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuFroxelInjection {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates the froxel-injection weight for every query in `queries`,
    /// returning one value per query in input order.
    ///
    /// The returned value for query `q` equals
    /// [`froxel_injection_weight`](prism_render_architecture::volumetric::fog::froxel_injection_weight)`(q.depth_slice, q.slice_count)`
    /// to within the tolerance documented on this module. An empty `queries`
    /// slice yields an empty result — storage buffers cannot be zero-sized, so
    /// it is handled by an early return.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[FroxelInjectionQuery]) -> Vec<f32> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_queries: Vec<GpuQuery> = queries
            .iter()
            .map(|q| GpuQuery {
                depth_slice: q.depth_slice,
                slice_count: q.slice_count,
                pad0: 0,
                pad1: 0,
            })
            .collect();

        let gpu_params = Params {
            count: queries.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let out_bytes = (queries.len() as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_froxel_injection_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_froxel_injection_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_froxel_injection_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_froxel_injection_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_froxel_injection_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: queries_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_froxel_injection_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_froxel_injection_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (queries.len() as u32).div_ceil(64);
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
        let values = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(values.len(), queries.len());
        values
    }
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
