//! `wgpu` compute twin of the weather-map coverage-mass reduction
//! ([`WeatherField::total_coverage`](prism_render_architecture::volumetric::weather::WeatherField::total_coverage)).
//!
//! `total_coverage` sums the `coverage` channel over every cell of a weather
//! field. It is the conserved quantity the advection mass tests track: a
//! divergence-free wind should transport cloud without creating or destroying
//! coverage, so the whole-field sum stays near-constant across an advection
//! step. The advection step itself is twinned elsewhere
//! ([`GpuAdvectSemiLagrangian`](crate::advect_semi_lagrangian::GpuAdvectSemiLagrangian),
//! [`GpuAdvectWithWind`](crate::advect_with_wind::GpuAdvectWithWind)); this
//! kernel twins the pure whole-field reduction a mass check evaluates on each
//! snapshot. [`GpuTotalCoverage`] uploads every field's coverage channel once as
//! a single flattened array and reduces all fields in parallel, one invocation
//! per field.
//!
//! # Correctness model
//!
//! The kernel mirrors the `CPU` reduction exactly: each field sums its
//! `coverage` samples in ascending cell index (row-major) order, matching the
//! golden's left-to-right `while` loop, so `CPU` and `GPU` agree to a few `ULP`.
//! An empty field contributes `0`. The parity test builds real
//! [`WeatherField`](prism_render_architecture::volumetric::weather::WeatherField)s
//! (including a pre/post-advection pair) and compares each sum against
//! [`WeatherField::total_coverage`], so a dropped cell or a reordered
//! accumulation could not pass.
//!
//! # Portability
//!
//! The kernel is add plus compare in the portable core-`WGSL` subset, so it runs
//! unmodified on Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: straightforward channel reduction over a weather grid plus
//! `wgpu` compute dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::advect_semi_lagrangian::WeatherAdvectSample;
use crate::context::GpuContext;

/// Where one field's coverage samples live in the shared flattened array.
/// `8`-byte `repr(C)` matching `FieldRange` in `shaders/total_coverage.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuFieldRange {
    offset: u32,
    count: u32,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/total_coverage.wesl`: the field count plus three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    field_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable weather-map coverage-mass pipeline.
pub struct GpuTotalCoverage {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuTotalCoverage {
    /// Compiles the coverage-mass kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuTotalCoverage {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_total_coverage"),
            source: ShaderSource::Wgsl(include_str!("../shaders/total_coverage.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_total_coverage_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_total_coverage_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_total_coverage_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("total_coverage_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuTotalCoverage {
            module,
            layout,
            pipeline,
        }
    }

    /// Reduces every field in `fields` to its `coverage`-channel sum, returning
    /// one total per field in input order.
    ///
    /// Each result equals the `CPU`
    /// [`WeatherField::total_coverage`](prism_render_architecture::volumetric::weather::WeatherField::total_coverage)
    /// for the same cells to within the tolerance documented on this module.
    /// Each inner slice is a field's cells in row-major order (as returned by
    /// `WeatherField::cells`). An empty field contributes `0`. An empty `fields`
    /// slice yields an empty result; when every field is empty (no cells
    /// anywhere) the all-zero result is produced on the host because storage
    /// buffers cannot be zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, fields: &[Vec<WeatherAdvectSample>]) -> Vec<f32> {
        if fields.is_empty() {
            return Vec::new();
        }

        // Flatten every field's coverage channel into one array and record each
        // field's [offset, count) window into it.
        let total_cells: usize = fields.iter().map(Vec::len).sum();
        if total_cells == 0 {
            // No cells anywhere: every field sums to zero coverage.
            return alloc_zeros(fields.len());
        }

        let device = ctx.device();

        let mut coverage: Vec<f32> = Vec::with_capacity(total_cells);
        let mut ranges: Vec<GpuFieldRange> = Vec::with_capacity(fields.len());
        for field in fields {
            let offset = coverage.len() as u32;
            for c in field {
                coverage.push(c.coverage);
            }
            ranges.push(GpuFieldRange {
                offset,
                count: field.len() as u32,
            });
        }

        let gpu_params = Params {
            field_count: fields.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let out_bytes = (fields.len() as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_total_coverage_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let coverage_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_total_coverage_coverage"),
            contents: bytemuck::cast_slice(&coverage),
            usage: BufferUsages::STORAGE,
        });
        let ranges_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_total_coverage_ranges"),
            contents: bytemuck::cast_slice(&ranges),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_total_coverage_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_total_coverage_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_total_coverage_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: coverage_buf.as_entire_binding(),
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
            label: Some("prism_volumetric_total_coverage_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_total_coverage_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (fields.len() as u32).div_ceil(64);
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
        debug_assert_eq!(raw.len(), fields.len());
        raw
    }
}

/// Builds a `Vec<f32>` of `len` zeros for the all-empty-fields host fast path.
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
