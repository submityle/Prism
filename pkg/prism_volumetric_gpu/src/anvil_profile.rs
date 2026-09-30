//! `wgpu` compute twin of the cumulonimbus anvil-density profile
//! ([`anvil_profile`](prism_render_architecture::volumetric::storm::anvil_profile)).
//!
//! The cumulonimbus vertical-development model (design section 9b) flares a flat
//! anvil out of the storm top. `anvil_profile` returns the anvil density weight
//! at a normalized `height_fraction` for a maturity `spread`:
//! `base = lerp(0.85, 0.5, saturate(spread))` then
//! `saturate(saturate(spread) * smoothstep(base, 1, saturate(height_fraction)))`,
//! so the anvil concentrates in the upper band and a more mature spread flares
//! it lower and stronger, always bounded to `0..=1`. The `CPU` golden
//! [`anvil_profile`](prism_render_architecture::volumetric::storm::anvil_profile)
//! owns that math; [`GpuAnvilProfile`] is the on-device twin that runs one
//! thread per query and reproduces the same value.
//!
//! # Correctness model
//!
//! The `lerp` and `smoothstep` are expanded to the *same* closed forms the CPU
//! `math::lerp` / `math::smoothstep` use (`a + (b - a) * t`; a degenerate-span
//! guard, a saturated ramp, then the cubic `t*t*(3-2t)`), and the kernel
//! contains no transcendental call — saturate and a multiply-add — so `CPU` and
//! `GPU` evaluate the same closed-form algebra. The only slack is a legal
//! multiply-add contraction of a few `ULP`, so the parity test asserts a tight
//! tolerance (`abs_diff < 1e-6` or `rel_diff < 1e-5`). The scenes also assert
//! the value stays in `0..=1` and rises monotonically with both height and
//! spread, so a degenerate kernel could not pass.
//!
//! # Portability
//!
//! The kernel is clamp/saturate and multiply/add in the portable core-`WGSL`
//! subset — no `exp`, `pow` or optional device feature — so it runs unmodified
//! on Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard cumulonimbus anvil-spread profile plus `wgpu` compute
//! dispatch; no Unreal Engine source or derived code.
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

/// One anvil-profile query: the normalized height plus the anvil maturity spread.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnvilProfileQuery {
    /// Normalized height within the cloud, saturated to `0..=1` before use.
    pub height_fraction: f32,
    /// Anvil maturity spread, saturated to `0..=1` before use.
    pub spread: f32,
}

/// One query as uploaded. `16`-byte `repr(C)` matching `Query` in
/// `shaders/anvil_profile.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    height_fraction: f32,
    spread: f32,
    pad0: f32,
    pad1: f32,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/anvil_profile.wesl`: the query count plus three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable anvil-profile pipeline.
pub struct GpuAnvilProfile {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuAnvilProfile {
    /// Compiles the anvil-profile kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuAnvilProfile {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_anvil_profile"),
            source: ShaderSource::Wgsl(include_str!("../shaders/anvil_profile.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_anvil_profile_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_anvil_profile_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_anvil_profile_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("anvil_profile_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuAnvilProfile {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates the anvil-density weight for every query in `queries`,
    /// returning one value per query in input order.
    ///
    /// The returned value for query `q` equals
    /// [`anvil_profile`](prism_render_architecture::volumetric::storm::anvil_profile)`(q.height_fraction, q.spread)`
    /// to within the tolerance documented on this module. An empty `queries`
    /// slice yields an empty result — storage buffers cannot be zero-sized, so
    /// it is handled by an early return.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[AnvilProfileQuery]) -> Vec<f32> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_queries: Vec<GpuQuery> = queries
            .iter()
            .map(|q| GpuQuery {
                height_fraction: q.height_fraction,
                spread: q.spread,
                pad0: 0.0,
                pad1: 0.0,
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
            label: Some("prism_volumetric_anvil_profile_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_anvil_profile_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_anvil_profile_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_anvil_profile_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_anvil_profile_bind_group"),
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
            label: Some("prism_volumetric_anvil_profile_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_anvil_profile_pass"),
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
