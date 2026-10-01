//! `wgpu` compute twin of Prism's kept-strand count
//! ([`kept_strand_count`](prism_render_architecture::hair::cluster::kept_strand_count)).
//!
//! The continuous decimation ramp ([`crate::strand_keep_ratio`]) yields a keep
//! ratio in `[min_ratio, 1]`; this stage turns that ratio into an integer
//! strand count for a cluster of `total` members — clamp the ratio into
//! `[0, 1]`, scale by the member count, round to the nearest strand and clamp to
//! never exceed `total`. This twin evaluates that per-cluster map batch-wide on
//! the device, one thread per `(total, ratio)` pair — the array-in/array-out
//! form the LOD stage consumes per cluster.
//!
//! # What the kernel evaluates
//!
//! [`GpuHairKeptStrandCount::eval`] takes a slice of member counts and a
//! matching slice of keep ratios and returns one kept count per pair in order.
//! The pair index is the invocation id (`@compute @workgroup_size(64)`, a
//! one-dimensional dispatch over `global_invocation_id.x`); invocations past the
//! element count early-return.
//!
//! # Correctness model
//!
//! The only subtle step is rounding. Rust's [`f32::round`] rounds halves away
//! from zero (`2.5 -> 3`), whereas WGSL's built-in `round` rounds halves to even
//! (`2.5 -> 2`). To stay bit-identical to the golden the kernel never calls the
//! built-in `round`; it computes `floor(scaled + 0.5)`, which reproduces
//! round-half-away-from-zero exactly for the non-negative scaled value here. A
//! non-finite ratio collapses to `0` kept, bit-faithfully to the golden. The
//! counts are therefore asserted with integer equality (not a tolerance) for
//! member counts in the f32-exact integer range (strand counts, well under
//! `2^24`), the only regime a groom reaches.
//!
//! # Portability
//!
//! The kernel uses only clamp, compare, multiply, `floor` and conversions in the
//! portable core-`WGSL` subset — no `exp`, `pow`, built-in `round` or optional
//! device feature — so the twin runs unmodified on Metal, Vulkan and DX12.
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: screen-footprint strand decimation member count plus `wgpu`
//! compute dispatch; no Unreal Engine source or derived code.

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

use prism_render_architecture::hair::cluster::kept_strand_count;

use crate::context::GpuContext;

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/kept_strand_count.wesl`: the element count padded to one `16`-byte
/// uniform slot.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    elem_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable per-pair kept-strand-count pipeline.
pub struct GpuHairKeptStrandCount {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairKeptStrandCount {
    /// Compiles the per-pair kept-strand-count kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairKeptStrandCount {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_kept_strand_count"),
            source: ShaderSource::Wgsl(include_str!("../shaders/kept_strand_count.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_kept_strand_count_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_kept_strand_count_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_kept_strand_count_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairKeptStrandCount {
            module,
            layout,
            pipeline,
        }
    }

    /// Maps each `(total, ratio)` pair to its kept strand count, returning one
    /// `u32` per pair in order.
    ///
    /// The count for pair `i` matches the `CPU` golden
    /// [`kept_strand_count`](prism_render_architecture::hair::cluster::kept_strand_count)
    /// bit-for-bit for member counts in the f32-exact integer range. `totals`
    /// and `ratios` must have equal length. An empty batch yields an empty
    /// vector without a dispatch — storage buffers cannot be zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, totals: &[u32], ratios: &[f32]) -> Vec<u32> {
        assert_eq!(
            totals.len(),
            ratios.len(),
            "totals and ratios must pair up one-to-one (got {} and {})",
            totals.len(),
            ratios.len()
        );
        let elem_count = totals.len();
        if elem_count == 0 {
            return Vec::new();
        }

        let device = ctx.device();

        let uniforms = Params {
            elem_count: elem_count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let out_bytes = (elem_count as u64) * (size_of::<u32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_kept_strand_count_params"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM,
        });
        let totals_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_kept_strand_count_totals"),
            contents: bytemuck::cast_slice(totals),
            usage: BufferUsages::STORAGE,
        });
        let ratios_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_kept_strand_count_ratios"),
            contents: bytemuck::cast_slice(ratios),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_kept_strand_count_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_kept_strand_count_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_kept_strand_count_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: totals_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: ratios_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_kept_strand_count_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_kept_strand_count_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (elem_count as u32).div_ceil(64);
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
        let out = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
        drop(view);
        out_stage.unmap();

        out
    }
}

/// The `CPU` golden kept strand count for one `(total, ratio)` pair, re-exported
/// so the parity test can assert the device twin against the identical reference
/// it mirrors.
#[must_use]
pub fn reference_kept_strand_count(total: u32, ratio: f32) -> u32 {
    kept_strand_count(total, ratio)
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
