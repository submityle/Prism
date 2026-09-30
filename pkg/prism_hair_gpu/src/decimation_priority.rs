//! `wgpu` compute twin of Prism's importance-weighted decimation ranking key
//! ([`decimation_priority`](prism_render_architecture::hair::decimation::decimation_priority)).
//!
//! Continuous density LOD keeps fewer render strands as a groom recedes, and to
//! stay pop-free the kept set at a low count must nest inside the kept set at a
//! higher count. The reference builds that nested order by ranking every strand
//! by a scalar priority `= importance + jitter * (hash - 0.5)`: the importance
//! biases long / curly / authored strands to survive to the lowest counts (the
//! density-LOD bias `UE5` Groom and `HairWorks` apply), while a per-strand hash
//! jitter in `[-0.5, 0.5) * jitter` decorrelates the ranking spatially so heavy
//! thinning removes strands evenly instead of carving bald patches. The host
//! then sorts these priorities (descending, ties by ascending index) into the
//! decimation order via
//! [`build_decimation_order`](prism_render_architecture::hair::decimation::build_decimation_order);
//! this kernel is only the per-strand key evaluation, the part that maps cleanly
//! to one `GPU` thread per strand. The `CPU` golden is
//! [`decimation_priority`](prism_render_architecture::hair::decimation::decimation_priority);
//! this crate is the on-device twin that runs the identical key so a passing
//! real-device parity test is direct evidence the ported kernel ranks strands
//! the same way the reference does — not merely that its shader compiles.
//!
//! # What the kernel evaluates
//!
//! [`GpuDecimationPriority::eval`] takes a per-strand `seeds` slice, the parallel
//! `importances` slice and the scalar `jitter`, and returns one f32 ranking key
//! per strand. The strand index is simply the invocation id.
//!
//! # Portability
//!
//! The kernel uses only integer arithmetic, one `u32`-to-`f32` conversion and a
//! single multiply-add in the portable core-`WGSL` subset — no `exp`, `pow`,
//! `sin` or optional device feature — so the twin runs unmodified on Metal,
//! Vulkan and DX12.
//!
//! # Correctness model
//!
//! The jitter draws from the golden's hand-written `splitmix64` integer hash
//! reproduced bit-for-bit (a `vec2<u32>` (lo, hi) emulation stands in for the
//! 64-bit integer baseline `WGSL` lacks) at the same fixed
//! `DECIMATION_JITTER_KEY` sub-key. The hash is integer-exact; only the final
//! `importance + jitter * h` combine is a single multiply-add a `GPU` may fuse,
//! so `CPU` and `GPU` agree to within the documented fma tolerance rather than
//! bit-for-bit. At `jitter == 0` the key collapses to the importance itself and
//! the twin is exact.
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard importance-weighted stochastic-LOD decimation ranking
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

/// Uniform parameters for one ranking-key dispatch. Layout matches `Params` in
/// `shaders/decimation_priority.wesl`: the jitter amplitude and the strand count
/// bounding the dispatch, padded to one `16`-byte uniform slot.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    jitter: f32,
    strand_count: u32,
    pad0: u32,
    pad1: u32,
}

/// A compiled, reusable per-strand decimation-priority pipeline.
pub struct GpuDecimationPriority {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuDecimationPriority {
    /// Compiles the per-strand ranking-key kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuDecimationPriority {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_decimation_priority"),
            source: ShaderSource::Wgsl(include_str!("../shaders/decimation_priority.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_decimation_priority_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_decimation_priority_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_decimation_priority_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuDecimationPriority {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates the decimation ranking key for every strand, returning one f32
    /// priority per strand.
    ///
    /// `seeds` and `importances` are the parallel per-strand hash seed and
    /// already-folded importance; they must share a length. `jitter` is the peak
    /// decorrelation amplitude. The priority for strand `i` equals the `CPU`
    /// golden
    /// [`decimation_priority`](prism_render_architecture::hair::decimation::decimation_priority)
    /// with the same `seeds[i]`, `importances[i]` and `jitter` to within the
    /// module's documented fma tolerance (bit-identical at `jitter == 0`). A
    /// length mismatch or an empty batch yields an empty vector without a
    /// dispatch — storage buffers cannot be zero-sized.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        seeds: &[u32],
        importances: &[f32],
        jitter: f32,
    ) -> Vec<f32> {
        let strand_count = seeds.len();
        if strand_count == 0 || importances.len() != strand_count {
            return Vec::new();
        }

        let device = ctx.device();

        let uniforms = Params {
            jitter,
            strand_count: strand_count as u32,
            pad0: 0,
            pad1: 0,
        };

        let out_bytes = (strand_count as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_decimation_priority_params"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM,
        });
        let seeds_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_decimation_priority_seeds"),
            contents: bytemuck::cast_slice(seeds),
            usage: BufferUsages::STORAGE,
        });
        let importances_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_decimation_priority_importances"),
            contents: bytemuck::cast_slice(importances),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_decimation_priority_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_decimation_priority_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_decimation_priority_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: seeds_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: importances_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_decimation_priority_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_decimation_priority_pass"),
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
        let priorities = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        out_stage.unmap();

        priorities
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
