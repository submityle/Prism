//! `wgpu` compute twin of the virtual-geometry LOD screen-space-error
//! projection
//! ([`LodProjection::projected_error_pixels`](prism_render_architecture::virtual_geometry::LodProjection::projected_error_pixels)).
//!
//! The screen-space-error LOD selector displays the coarsest cluster level
//! whose projected error still fits the pixel budget. The projection itself is
//! the hot inner term - it maps an object-space geometric error to screen
//! pixels at the current view distance. The CPU golden
//! [`LodProjection::projected_error_pixels`](prism_render_architecture::virtual_geometry::LodProjection::projected_error_pixels)
//! owns that closed form; [`GpuLodProjection`] is the on-device twin that runs
//! one thread per query and returns the same pixel size, so the projection term
//! that drives paged-cluster LOD selection is validated against the reference
//! rather than merely compiled. The surrounding selection policy (hysteresis,
//! prefetch, level walk) stays on the CPU golden; only the pure projection
//! scalar is mirrored here.
//!
//! # Portability
//!
//! The kernel is two `max` clamps, one multiply and one divide in the portable
//! core-`WGSL` subset, so it needs no optional device feature and runs
//! unmodified on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! The value is `(m * f) / d` with `m = max(error, 0)`,
//! `f = focal_length_pixels` and `d = max(distance, f32::EPSILON)`. `max` is
//! exact and the multiply is correctly rounded, but a `GPU` divide by a
//! non-power-of-two divisor is only guaranteed correctly rounded to within one
//! ULP (Metal implements it as a refined reciprocal), so the twin matches the
//! golden within one ULP in general and bit-for-bit whenever `d` is an exact
//! power of two (the divide degenerates to exponent scaling). The clamp
//! threshold `f32::EPSILON = 2^-23` is written on both sides as an exact value,
//! so the CPU and GPU clamps coincide to the bit. The parity test asserts a
//! `<= 1` ULP bound in general and bit-exact equality on power-of-two divisors.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard perspective screen-space-error projection for LOD
//! selection plus `wgpu` compute dispatch; no Unreal Engine source or derived
//! code.

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

/// One LOD-projection query: an object-space geometric error, the view-space
/// distance to the cluster and the perspective focal length in pixels.
///
/// `focal_length_pixels` comes from a
/// [`LodProjection`](prism_render_architecture::virtual_geometry::LodProjection)
/// (`from_focal_length_pixels` or `from_half_fov_tan`); it is carried per query
/// so each query is fully independent on-device.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProjectedErrorQuery {
    /// Object-space geometric deviation bound of the level, in world units.
    pub geometric_error: f32,
    /// View-space distance to the cluster, in world units.
    pub view_distance: f32,
    /// Perspective factor: half viewport height / `tan(0.5 * vertical_fov)`.
    pub focal_length_pixels: f32,
}

/// Uniform parameters for one projection dispatch. Layout matches `Params` in
/// `shaders/lod_projection.wesl`: the query count then three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One projection query upload. `12`-byte stride, matching `LodProjectionQuery`
/// in the shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuLodInput {
    geometric_error: f32,
    view_distance: f32,
    focal_length_pixels: f32,
}

/// A compiled, reusable LOD-projection pipeline.
pub struct GpuLodProjection {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuLodProjection {
    /// Compiles the LOD-projection kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so it needs no
    /// optional device feature.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_lod_projection"),
            source: ShaderSource::Wgsl(include_str!("../shaders/lod_projection.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_lod_projection_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_lod_projection_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_lod_projection_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("project"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuLodProjection {
            module,
            layout,
            pipeline,
        }
    }

    /// Projects each query's geometric error to screen pixels on-device,
    /// returning one pixel size per query in input order.
    ///
    /// Each returned value equals
    /// [`LodProjection::projected_error_pixels`](prism_render_architecture::virtual_geometry::LodProjection::projected_error_pixels)`(geometric_error, view_distance)`
    /// evaluated with the query's `focal_length_pixels`. An empty `queries`
    /// slice yields an empty result - storage buffers cannot be zero-sized, so
    /// it is handled by an early return.
    #[must_use]
    pub fn project(&self, ctx: &GpuContext, queries: &[ProjectedErrorQuery]) -> Vec<f32> {
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

        let gpu_inputs: Vec<GpuLodInput> = queries
            .iter()
            .map(|q| GpuLodInput {
                geometric_error: q.geometric_error,
                view_distance: q.view_distance,
                focal_length_pixels: q.focal_length_pixels,
            })
            .collect();

        let out_bytes = (queries.len() as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_lod_projection_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let inputs_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_lod_projection_inputs"),
            contents: bytemuck::cast_slice(&gpu_inputs),
            usage: BufferUsages::STORAGE,
        });
        let pixels_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_lod_projection_pixels"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let pixels_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_lod_projection_pixels_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_lod_projection_bind_group"),
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
                    resource: pixels_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_lod_projection_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_lod_projection_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = params.count.div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&pixels_buf, 0, &pixels_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        pixels_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = pixels_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let gpu_pixels = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        pixels_stage.unmap();
        debug_assert_eq!(gpu_pixels.len(), queries.len());
        gpu_pixels
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
