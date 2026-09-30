//! `wgpu` compute twin of the virtual-geometry frustum-plane signed-distance
//! test
//! ([`Plane::signed_distance`](prism_render_architecture::virtual_geometry::Plane::signed_distance)).
//!
//! Frustum culling keeps a cluster when its bounds' signed distance to every
//! inward-facing plane clears the negated support radius. The signed distance
//! itself - `dot(normal, point) + distance` - is the scalar every plane test
//! consumes. The CPU golden
//! [`Plane::signed_distance`](prism_render_architecture::virtual_geometry::Plane::signed_distance)
//! owns that closed form; [`GpuSignedDistance`] is the on-device twin that runs
//! one thread per query and returns the same scalar. The already-landed
//! cluster-cull twin only checks the boolean cull verdict; mirroring the
//! continuous distance here validates the two backends' plane arithmetic to the
//! bit and catches any floating-point drift a boolean verdict would hide.
//!
//! # Portability
//!
//! The kernel is a three-term dot product plus one add in the portable
//! core-`WGSL` subset, so it needs no optional device feature and runs
//! unmodified on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! The value is `n0*p0 + n1*p1 + n2*p2 + d`. A `GPU` may contract this into
//! fused-multiply-adds and reassociate the sum, so it is bit-exact against the
//! sequential `CPU` golden only when every product and partial sum is exactly
//! representable - then no rounding occurs and fusion / ordering are
//! immaterial. The parity suite drives it with small integer / dyadic operands
//! whose products and running sums stay inside the 24-bit mantissa, where the
//! result is bit-for-bit identical regardless of FMA, and asserts exact
//! equality rather than a tolerance.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard inward-plane signed-distance test for frustum culling
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

use crate::context::GpuContext;

/// One signed-distance query: an inward-facing plane (`normal`, `distance`) and
/// the `point` whose signed distance to it is wanted.
///
/// `normal` and `distance` map straight onto a
/// [`Plane`](prism_render_architecture::virtual_geometry::Plane); the twin does
/// not require a unit normal - it mirrors the raw arithmetic the golden
/// performs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SignedDistanceQuery {
    /// Inward-facing plane normal.
    pub normal: [f32; 3],
    /// Query point.
    pub point: [f32; 3],
    /// Signed plane distance to the origin.
    pub distance: f32,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/signed_distance.wesl`: the query count then three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One query upload. `28`-byte stride, matching `PlaneQuery` in the shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuPlaneQuery {
    nx: f32,
    ny: f32,
    nz: f32,
    px: f32,
    py: f32,
    pz: f32,
    d: f32,
}

/// A compiled, reusable signed-distance pipeline.
pub struct GpuSignedDistance {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSignedDistance {
    /// Compiles the signed-distance kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so it needs no
    /// optional device feature.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_signed_distance"),
            source: ShaderSource::Wgsl(include_str!("../shaders/signed_distance.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_signed_distance_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_signed_distance_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_signed_distance_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSignedDistance {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates each query's signed distance on-device, returning one scalar
    /// per query in input order.
    ///
    /// Each returned value equals
    /// [`Plane::signed_distance`](prism_render_architecture::virtual_geometry::Plane::signed_distance)`(point)`
    /// for the query's plane. An empty `queries` slice yields an empty result -
    /// storage buffers cannot be zero-sized, so it is handled by an early
    /// return.
    #[must_use]
    pub fn evaluate(&self, ctx: &GpuContext, queries: &[SignedDistanceQuery]) -> Vec<f32> {
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

        let gpu_inputs: Vec<GpuPlaneQuery> = queries
            .iter()
            .map(|q| GpuPlaneQuery {
                nx: q.normal[0],
                ny: q.normal[1],
                nz: q.normal[2],
                px: q.point[0],
                py: q.point[1],
                pz: q.point[2],
                d: q.distance,
            })
            .collect();

        let out_bytes = (queries.len() as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_signed_distance_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let inputs_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_signed_distance_inputs"),
            contents: bytemuck::cast_slice(&gpu_inputs),
            usage: BufferUsages::STORAGE,
        });
        let distances_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_signed_distance_distances"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let distances_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_signed_distance_distances_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_signed_distance_bind_group"),
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
                    resource: distances_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_signed_distance_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_signed_distance_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = params.count.div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&distances_buf, 0, &distances_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        distances_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = distances_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let gpu_distances = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        distances_stage.unmap();
        debug_assert_eq!(gpu_distances.len(), queries.len());
        gpu_distances
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
