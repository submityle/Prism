//! `wgpu` compute twin of Prism's hair barrier-contact distance primitives
//! ([`point_plane_signed_distance`](prism_render_architecture::hair::barrier_contact::point_plane_signed_distance)
//! and
//! [`point_point_distance`](prism_render_architecture::hair::barrier_contact::point_point_distance)).
//!
//! For each query the kernel returns two closed-form distances used by the hair
//! barrier contact solver's plane and point colliders:
//!
//! * the **signed plane distance** from the query point `p` to the plane through
//!   `plane_point` oriented by `plane_normal` — positive on the side the normal
//!   points toward, negative behind it, and exactly `0` when the normal is
//!   (numerically) the zero vector and so has no orientation; and
//! * the **point-point distance** between `p` and `plane_point`.
//!
//! # Why one thread per query
//!
//! Each query reads only its own nine input scalars (`p`, `plane_point`,
//! `plane_normal`) and writes its own output row, with no shared mutable state,
//! so this is embarrassingly parallel: one thread owns one query.
//!
//! # What the kernel evaluates
//!
//! [`GpuHairContactDistance::eval`] takes a batch of `(p, plane_point,
//! plane_normal)` triples and returns, per query and in input order, the pair
//! `(signed_plane_distance, point_point_distance)`.
//!
//! # Correctness model
//!
//! The reference sanitizes every input vector (non-finite components become `0`)
//! and guards a (numerically) zero plane normal with the squared-length
//! threshold `EPS_LEN_SQ = 1e-24`, returning `0` for the signed distance in that
//! case. This twin supplies finite inputs to the device and asserts against the
//! same reference, so the kernel keeps only the zero-normal guard and omits the
//! per-component finite guards. Both primitives are a single closed-form
//! evaluation (a `dot` and a reciprocal `sqrt` for the plane distance, one
//! subtract-and-length for the point distance) with no chained recurrence, so
//! the only `CPU` vs `GPU` divergence is legal fused-multiply-add contraction
//! and correctly rounded `sqrt`/division; parity is asserted to within
//! `abs_diff < 1e-4` or `rel_diff < 1e-3`. The zero-normal signed-distance
//! branch yields a bit-exact `0` on both sides.
//!
//! # Portability
//!
//! The kernel uses only subtract/dot/mul/divide/sqrt in the portable
//! core-`WGSL` subset — no `exp`, `pow`, `log` or optional device feature — so
//! the twin runs unmodified on Metal, Vulkan and DX12.
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard analytic point/plane distance plus a `wgpu` compute
//! dispatch; no Unreal Engine source or derived code.

use core::mem::size_of;

use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use prism_render_architecture::hair::barrier_contact::{
    point_plane_signed_distance, point_point_distance, Vec3,
};

use crate::context::GpuContext;

/// Uniform parameters for one contact-distance dispatch. Layout matches
/// `Params` in `shaders/contact_distance.wesl`: just the query count (a single
/// `u32` padded to the `16`-byte uniform slot).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    query_count: u32,
}

/// Compiled per-query contact-distance compute twin: the shader module (kept
/// alive so its pipeline stays valid), the bind-group layout and the pipeline.
pub struct GpuHairContactDistance {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairContactDistance {
    /// Compiles the per-query contact-distance kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairContactDistance {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_contact_distance"),
            source: ShaderSource::Wgsl(include_str!("../shaders/contact_distance.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_contact_distance_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_contact_distance_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_contact_distance_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairContactDistance {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates both contact distances for every `(p, plane_point,
    /// plane_normal)` triple, returning one `(signed_plane_distance,
    /// point_point_distance)` pair per query, in input order.
    ///
    /// The pair for query `i` equals
    /// `(point_plane_signed_distance(p, plane_point, plane_normal),
    /// point_point_distance(p, plane_point))`, to within the single-evaluation
    /// tolerance documented on this module (`abs_diff < 1e-4` or
    /// `rel_diff < 1e-3`). The empty batch is handled without a dispatch —
    /// storage buffers cannot be zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[(Vec3, Vec3, Vec3)]) -> Vec<(f32, f32)> {
        if queries.is_empty() {
            return Vec::new();
        }

        let device = ctx.device();
        let gpu_params = GpuParams {
            query_count: queries.len() as u32,
        };

        // Flatten each query to nine `f32`: `p.xyz`, `plane_point.xyz`,
        // `plane_normal.xyz`.
        let mut flat: Vec<f32> = Vec::with_capacity(queries.len() * 9);
        for (p, pp, nn) in queries {
            flat.push(p.x);
            flat.push(p.y);
            flat.push(p.z);
            flat.push(pp.x);
            flat.push(pp.y);
            flat.push(pp.z);
            flat.push(nn.x);
            flat.push(nn.y);
            flat.push(nn.z);
        }

        let out_bytes = (queries.len() as u64) * 4 * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_contact_distance_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_contact_distance_queries"),
            contents: bytemuck::cast_slice(&flat),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_contact_distance_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_contact_distance_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_contact_distance_bind_group"),
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
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_contact_distance_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_contact_distance_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (queries.len() as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();

        let view = stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let data = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        stage.unmap();

        debug_assert_eq!(data.len(), queries.len() * 4);

        let mut out: Vec<(f32, f32)> = Vec::with_capacity(queries.len());
        for i in 0..queries.len() {
            let b = i * 4;
            out.push((data[b], data[b + 1]));
        }
        out
    }
}

/// The `CPU` golden signed plane distance, re-exported so the parity test can
/// assert the device twin against the identical reference it mirrors.
#[must_use]
pub fn reference_point_plane_signed_distance(
    p: Vec3,
    plane_point: Vec3,
    plane_normal: Vec3,
) -> f32 {
    point_plane_signed_distance(p, plane_point, plane_normal)
}

/// The `CPU` golden point-point distance, re-exported for the parity test.
#[must_use]
pub fn reference_point_point_distance(a: Vec3, b: Vec3) -> f32 {
    point_point_distance(a, b)
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
