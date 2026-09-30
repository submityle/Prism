//! `wgpu` compute twin of the virtual-geometry paging priority
//! ([`coverage_priority`](prism_render_architecture::virtual_geometry::coverage_priority)).
//!
//! The page streamer orders clusters by how much screen area their bounding
//! sphere covers, so the coarsest visible cut becomes resident first. The CPU
//! golden
//! [`coverage_priority`](prism_render_architecture::virtual_geometry::coverage_priority)
//! owns the closed form - squared projected coverage
//! `extent^2 / max(dx^2 + dy^2 + dz^2, f32::EPSILON)` with
//! `extent = max(radius, 0) * focal` and `(dx, dy, dz) = center - view_origin`.
//! [`GpuCoveragePriority`] is the on-device twin that runs one thread per query
//! and returns the same priority, so the term that drives paged-cluster
//! streaming order is validated against the reference rather than merely
//! compiled. The surrounding cut walk and residency policy stay on the CPU
//! golden; only the pure priority scalar is mirrored here.
//!
//! # Portability
//!
//! The kernel is three subtracts, three multiplies, two `max` clamps, one
//! multiply and one divide in the portable core-`WGSL` subset, so it needs no
//! optional device feature and runs unmodified on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! The value is `e * e / d` with `e = max(radius, 0) * focal` and
//! `d = max(dx^2 + dy^2 + dz^2, f32::EPSILON)`. `max` is exact and every
//! multiply is correctly rounded, but a `GPU` divide by a non-power-of-two `d`
//! is only guaranteed correctly rounded to within one ULP (Metal implements it
//! as a refined reciprocal), so the twin matches the golden within one ULP in
//! general and bit-for-bit whenever the clamped `d` is an exact power of two.
//! The sum of squares may be contracted into an FMA on-device, so the parity
//! domain restricts the difference operands to values whose squares and partial
//! sums are exactly representable `f32` - then FMA fusion and the sequential
//! `CPU` adds coincide to the bit and only the final divide carries the
//! one-ULP slack. The `f32::EPSILON = 2^-23` floor is the exact literal
//! `0x1p-23` on both sides.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard perspective screen-coverage streaming priority for a
//! paged cluster hierarchy plus `wgpu` compute dispatch; no Unreal Engine
//! source or derived code.

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

/// One paging-priority query: a cluster bounding-sphere centre and radius, the
/// camera view origin and the perspective focal length in pixels.
///
/// `focal_length_pixels` comes from a
/// [`LodProjection`](prism_render_architecture::virtual_geometry::LodProjection);
/// it is carried per query so each query is fully independent on-device.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoveragePriorityQuery {
    /// Bounding-sphere centre in world space.
    pub center: [f32; 3],
    /// Bounding-sphere radius, in world units. A negative radius clamps to `0`.
    pub radius: f32,
    /// Camera view origin in world space.
    pub view_origin: [f32; 3],
    /// Perspective factor: half viewport height / `tan(0.5 * vertical_fov)`.
    pub focal_length_pixels: f32,
}

/// Uniform parameters for one priority dispatch. Layout matches `Params` in
/// `shaders/coverage_priority.wesl`: the query count then three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One priority query upload. `32`-byte stride, matching `CoverageQuery` in the
/// shader (eight scalar `f32` fields, no vec3 alignment padding).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuCoverageInput {
    center_x: f32,
    center_y: f32,
    center_z: f32,
    radius: f32,
    origin_x: f32,
    origin_y: f32,
    origin_z: f32,
    focal: f32,
}

/// A compiled, reusable paging-priority pipeline.
pub struct GpuCoveragePriority {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuCoveragePriority {
    /// Compiles the paging-priority kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so it needs no
    /// optional device feature.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_coverage_priority"),
            source: ShaderSource::Wgsl(include_str!("../shaders/coverage_priority.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_coverage_priority_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_coverage_priority_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_coverage_priority_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuCoveragePriority {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates each query's squared projected screen coverage on-device,
    /// returning one priority per query in input order.
    ///
    /// Each returned value equals
    /// [`coverage_priority`](prism_render_architecture::virtual_geometry::coverage_priority)
    /// evaluated with the query's bounds, view origin and focal length. An
    /// empty `queries` slice yields an empty result - storage buffers cannot be
    /// zero-sized, so it is handled by an early return.
    #[must_use]
    pub fn evaluate(&self, ctx: &GpuContext, queries: &[CoveragePriorityQuery]) -> Vec<f32> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let params = Params {
            count: u32::try_from(queries.len())
                .expect("query count must fit in u32 for the GPU dispatch"),
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let gpu_inputs: Vec<GpuCoverageInput> = queries
            .iter()
            .map(|q| GpuCoverageInput {
                center_x: q.center[0],
                center_y: q.center[1],
                center_z: q.center[2],
                radius: q.radius,
                origin_x: q.view_origin[0],
                origin_y: q.view_origin[1],
                origin_z: q.view_origin[2],
                focal: q.focal_length_pixels,
            })
            .collect();

        let out_bytes = (queries.len() as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_coverage_priority_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let inputs_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_coverage_priority_inputs"),
            contents: bytemuck::cast_slice(&gpu_inputs),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_coverage_priority_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_coverage_priority_out_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_coverage_priority_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: inputs_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_coverage_priority_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_coverage_priority_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = params.count.div_ceil(64);
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
        let gpu_out = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        out_stage.unmap();
        debug_assert_eq!(gpu_out.len(), queries.len());
        gpu_out
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
