//! `wgpu` compute twin of Prism's import-time arc-length resampler
//! ([`resample_strand`](prism_render_architecture::hair::groom_import::resample_strand)
//! / [`resample_groom`](prism_render_architecture::hair::groom_import::resample_groom)).
//!
//! Groom import reparameterizes every raw guide polyline — authored with an
//! arbitrary control-point count — into a fixed `target_points` stride spaced
//! evenly by arc length, so the whole engine (dynamics edge rest lengths,
//! interpolation, `LOD`, raster) consumes one uniform-stride buffer. `UE5` Groom
//! and `TressFX` both resample at import for exactly this reason. This crate is
//! that bake's isolated twin: one thread per strand walks the raw polyline's
//! cumulative arc length and emits `target_points` control points, so a passing
//! real-device parity test is direct evidence the arc-length walk and the
//! endpoint pinning port bit-faithfully — coverage no other hair twin provides,
//! since none of them reparameterizes a variable-length polyline by arc length.
//!
//! # Host-side filtering
//!
//! [`GpuHairResample::eval`] mirrors [`resample_groom`]'s strand skipping on the
//! host: a range whose `start + len` overflows or runs past the shared buffer,
//! or whose `len` is `0`, is dropped before dispatch, so only the surviving
//! strands are uploaded and the flat strand-major output matches
//! `ResampledGroom::positions` one for one. The single-point and all-coincident
//! degeneracies (which still survive as valid zero-length strands) are handled
//! inside the kernel.
//!
//! # Correctness model
//!
//! The endpoints are pinned exactly: output `0` lands at `t == 0` (the root
//! vertex) and output `target - 1` lands at `t == (total - d0) / (total - d0)`,
//! i.e. exactly `1.0` by IEEE `x / x`, so it is the tip vertex — both bit-exact
//! when the input coordinates make `a + (b - a)` exact (integer-valued test
//! vertices). Interior points divide `(d - d0) / span` and take a `sqrt` per
//! segment, either of which a `GPU` may fuse, so they can differ by a few
//! low-mantissa `ULP`; the parity test keeps interior samples inside segment
//! interiors and asserts them within a small point-distance and per-component
//! `abs < 1e-4` or `rel < 1e-3` tolerance while asserting the endpoints exactly.
//!
//! # Portability
//!
//! The kernel uses only subtraction, `dot`, `sqrt`, multiply, compare and one
//! divide in the portable core-`WGSL` subset — no atomics, no dynamic local
//! array, no optional device feature — so the twin runs unmodified on Metal,
//! Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard arc-length polyline resampling plus `wgpu` compute
//! dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::hair::dynamics::Vec3;
use prism_render_architecture::hair::groom_import::{
    resample_groom, resample_strand, RawStrandRange, ResampledGroom,
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

/// Strand count and per-strand target count, padded to `16` bytes to match
/// `Params` in `shaders/resample.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    strand_count: u32,
    target_points: u32,
    pad0: u32,
    pad1: u32,
}

/// One surviving strand's slice in the shared raw buffer, padded to `16` bytes
/// to match `Range` in `shaders/resample.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuRange {
    start: u32,
    len: u32,
    pad0: u32,
    pad1: u32,
}

/// One raw or output control point: `xyz` in a `vec4` (`w` unused) so the stride
/// is a `16`-byte multiple with no `vec3` alignment hazard.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuPoint {
    xyzw: [f32; 4],
}

/// A compiled, reusable arc-length resampling pipeline.
pub struct GpuHairResample {
    #[expect(
        dead_code,
        reason = "the shader module must outlive the pipeline that borrows it"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairResample {
    /// Compiles the resampling kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairResample {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_resample"),
            source: ShaderSource::Wgsl(include_str!("../shaders/resample.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_resample_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_resample_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_resample_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairResample {
            module,
            layout,
            pipeline,
        }
    }

    /// Resamples every surviving strand in `strands` to `target_points`
    /// arc-length-even control points, returning the flat strand-major buffer.
    ///
    /// The result equals the `CPU` golden
    /// [`resample_groom`](prism_render_architecture::hair::groom_import::resample_groom)'s
    /// `positions` field to within the tolerance documented on this module
    /// (bit-exact endpoints for integer-valued input). Ranges are pre-filtered
    /// with the same skip rules as the golden, so only surviving strands appear
    /// in the output, in input order. An input that leaves no surviving strand
    /// yields an empty vector without a dispatch — storage buffers cannot be
    /// zero-sized.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        points: &[Vec3],
        strands: &[RawStrandRange],
        target_points: u32,
    ) -> Vec<Vec3> {
        let points_per_strand = (target_points.max(2)) as usize;

        // Mirror `resample_groom`'s host-side skip rules so the surviving set
        // (and its order) matches the golden exactly.
        let survivors: Vec<GpuRange> = strands
            .iter()
            .filter_map(|range| {
                let end = range.start.checked_add(range.len)?;
                if range.len == 0 || end > points.len() {
                    return None;
                }
                Some(GpuRange {
                    start: range.start as u32,
                    len: range.len as u32,
                    pad0: 0,
                    pad1: 0,
                })
            })
            .collect();

        if survivors.is_empty() || points.is_empty() {
            return Vec::new();
        }

        let strand_count = survivors.len();
        let raw_uploads: Vec<GpuPoint> = points
            .iter()
            .map(|p| GpuPoint {
                xyzw: [p.x, p.y, p.z, 0.0],
            })
            .collect();

        let device = ctx.device();
        let uniforms = Params {
            strand_count: strand_count as u32,
            target_points,
            pad0: 0,
            pad1: 0,
        };
        let out_count = strand_count * points_per_strand;
        let out_bytes = (out_count as u64) * (size_of::<GpuPoint>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_resample_params"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM,
        });
        let ranges_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_resample_ranges"),
            contents: bytemuck::cast_slice(&survivors),
            usage: BufferUsages::STORAGE,
        });
        let raw_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_resample_raw"),
            contents: bytemuck::cast_slice(&raw_uploads),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_resample_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_resample_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_resample_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: ranges_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: raw_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_resample_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_resample_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (strand_count as u32).div_ceil(64);
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
        let raw = bytemuck::cast_slice::<u8, GpuPoint>(&view).to_vec();
        drop(view);
        out_stage.unmap();

        raw.into_iter()
            .map(|p| Vec3::new(p.xyzw[0], p.xyzw[1], p.xyzw[2]))
            .collect()
    }
}

/// Runs the golden single-strand resampler directly; a thin re-export so the
/// parity test can name one reference path.
#[must_use]
pub fn reference_resample_strand(raw: &[Vec3], target_points: u32) -> Vec<Vec3> {
    resample_strand(raw, target_points)
}

/// Runs the golden groom resampler directly; a thin re-export so the parity test
/// can name one reference path.
#[must_use]
pub fn reference_resample_groom(
    points: &[Vec3],
    strands: &[RawStrandRange],
    target_points: u32,
) -> ResampledGroom {
    resample_groom(points, strands, target_points)
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
