//! `wgpu` compute twin of the god-ray sample weight
//! ([`god_ray_weight`](prism_render_architecture::volumetric::shadow::god_ray_weight)).
//!
//! A radial god-ray march composites `sample_count` samples toward the light,
//! each weighted by `weight * decay^index` — a geometric decay away from the
//! light (design section 12). `decay` and `weight` are saturated into `[0, 1]`
//! and the product saturated again, so the result is in `[0, 1]`, monotone
//! non-increasing in the sample index, and its partial sums are bounded by
//! `weight / (1 - decay)`. The `CPU` golden
//! [`god_ray_weight`](prism_render_architecture::volumetric::shadow::god_ray_weight)
//! owns that math; [`GpuGodRayWeight`] is the on-device twin that runs one
//! thread per query and reproduces the same value.
//!
//! # Correctness model
//!
//! The power `decay^index` is evaluated with the *same* hand-rolled
//! `pow_approx` the reference uses — `exp_approx(index * ln_approx(decay))`,
//! with both the base-two `exp_approx` and the exponent-extraction / `atanh`
//! series `ln_approx` mirrored bit-for-bit — not the device-native `pow`.
//! Mirroring the whole chain keeps the twin close to the reference, so the
//! parity test asserts a tight tolerance (`abs_diff < 1e-5` or
//! `rel_diff < 1e-4`, slightly looser than the single-`exp` kernels because the
//! `ln`-then-`exp` chain admits a few more legal multiply-add contractions).
//! The scenes also assert the documented monotonicity (weights fall with the
//! sample index), the `[0, 1]` range, and the geometric-series partial-sum
//! bound, so a degenerate kernel could not pass.
//!
//! # Portability
//!
//! The kernel is `floor`, `bitcast`, integer ops and multiply/add in the
//! portable core-`WGSL` subset — no `exp`, `pow` or optional device feature —
//! so it runs unmodified on Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard radial god-ray geometric decay plus `wgpu` compute
//! dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::volumetric::shadow::GodRayConfig;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// One god-ray query: the radial sample index and the beam config it decays
/// under.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GodRayWeightQuery {
    /// Radial sample index counted from the light (index `0` is the beam root).
    pub sample_index: u32,
    /// The god-ray beam config; only `decay` and `weight` affect the result.
    pub config: GodRayConfig,
}

/// One query as uploaded. `32`-byte `repr(C)` matching `Query` in
/// `shaders/godray.wesl`. `sample_count` and `exposure` are carried for a
/// faithful mirror of the CPU config even though `god_ray_weight` reads only
/// `decay` and `weight`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    sample_index: u32,
    sample_count: u32,
    decay: f32,
    weight: f32,
    exposure: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/godray.wesl`: the query count plus three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable god-ray sample-weight pipeline.
pub struct GpuGodRayWeight {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuGodRayWeight {
    /// Compiles the god-ray weight kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuGodRayWeight {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_godray"),
            source: ShaderSource::Wgsl(include_str!("../shaders/godray.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_godray_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_godray_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_godray_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("godray_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuGodRayWeight {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates the god-ray sample weight for every query in `queries`,
    /// returning one value per query in input order.
    ///
    /// The returned value for query `q` equals
    /// [`god_ray_weight`](prism_render_architecture::volumetric::shadow::god_ray_weight)`(q.sample_index, q.config)`
    /// to within the tolerance documented on this module. An empty `queries`
    /// slice yields an empty result — storage buffers cannot be zero-sized, so
    /// it is handled by an early return.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[GodRayWeightQuery]) -> Vec<f32> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_queries: Vec<GpuQuery> = queries
            .iter()
            .map(|q| GpuQuery {
                sample_index: q.sample_index,
                sample_count: q.config.sample_count,
                decay: q.config.decay,
                weight: q.config.weight,
                exposure: q.config.exposure,
                pad0: 0,
                pad1: 0,
                pad2: 0,
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
            label: Some("prism_volumetric_godray_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_godray_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_godray_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_godray_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_godray_bind_group"),
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
            label: Some("prism_volumetric_godray_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_godray_pass"),
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
