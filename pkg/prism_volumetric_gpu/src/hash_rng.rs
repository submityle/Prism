//! `wgpu` compute twin of the stateless hash / RNG primitives
//! ([`hash_u32`](prism_render_architecture::volumetric::reference::hash_u32) and
//! [`rng_unit`](prism_render_architecture::volumetric::reference::rng_unit)).
//!
//! [`hash_u32`](prism_render_architecture::volumetric::reference::hash_u32) is a
//! `Wang`-style all-`wrapping` integer avalanche that turns a counter into a
//! well-distributed `u32`; `rng_unit(x)` maps that hash to `[0, 1)` using the
//! top `24` bits so the value lands exactly on the `f32` mantissa grid and can
//! never round up to `1.0`. Both are pure, deterministic functions — the
//! fixed-seed reproducibility contract of design section 16.
//!
//! The `CPU` goldens own that logic; [`GpuHashRng`] is the on-device twin that
//! runs one thread per query and returns both the raw hash and its unit-float
//! mapping.
//!
//! # Correctness model
//!
//! `WGSL` `u32` arithmetic is defined to wrap on overflow, exactly matching the
//! `CPU` golden's `wrapping_mul` / `wrapping_add`, so the twin reproduces the
//! bit pattern exactly (not merely close). The parity test asserts bit-exact
//! equality of the hash and exact equality of the unit float, and checks that
//! the unit float stays in `[0, 1)`.
//!
//! # Portability
//!
//! The kernel is integer arithmetic in the portable core-`WGSL` subset — no
//! optional device feature — so it runs unmodified on Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard `Wang`-style integer hash plus `wgpu` compute dispatch;
//! no Unreal Engine source or derived code.
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

/// One hashed draw: the raw avalanche hash and its unit-float mapping.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HashRngSample {
    /// The raw `Wang`-style avalanche hash of the input.
    pub hash: u32,
    /// The unit float in `[0, 1)` derived from the top `24` bits of `hash`.
    pub unit: f32,
}

/// One query as uploaded. `16`-byte `repr(C)` matching `Query` in
/// `shaders/hash_rng.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    x: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One result as read back. `8`-byte `repr(C)` matching `Res` in
/// `shaders/hash_rng.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuSample {
    hash: u32,
    unit: f32,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/hash_rng.wesl`: the query count plus three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable stateless hash / RNG pipeline.
pub struct GpuHashRng {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHashRng {
    /// Compiles the stateless hash / RNG kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHashRng {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_hash_rng_shader"),
            source: ShaderSource::Wgsl(include_str!("../shaders/hash_rng.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_hash_rng_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_hash_rng_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_hash_rng_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("hash_rng_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHashRng {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every seed in `seeds`, returning one [`HashRngSample`] per seed
    /// in input order.
    ///
    /// The returned sample for seed `x` has `hash ==`
    /// [`hash_u32`](prism_render_architecture::volumetric::reference::hash_u32)`(x)`
    /// and `unit ==`
    /// [`rng_unit`](prism_render_architecture::volumetric::reference::rng_unit)`(x)`,
    /// bit-exactly. An empty `seeds` slice yields an empty result — storage
    /// buffers cannot be zero-sized, so it is handled by an early return.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, seeds: &[u32]) -> Vec<HashRngSample> {
        if seeds.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_queries: Vec<GpuQuery> = seeds
            .iter()
            .map(|&x| GpuQuery {
                x,
                pad0: 0,
                pad1: 0,
                pad2: 0,
            })
            .collect();

        let gpu_params = Params {
            count: seeds.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let out_bytes = (seeds.len() as u64) * (size_of::<GpuSample>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_hash_rng_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_hash_rng_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_hash_rng_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_hash_rng_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_hash_rng_bind_group"),
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
            label: Some("prism_volumetric_hash_rng_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_hash_rng_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (seeds.len() as u32).div_ceil(64);
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
        let raw = bytemuck::cast_slice::<u8, GpuSample>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(raw.len(), seeds.len());
        raw.into_iter()
            .map(|s| HashRngSample {
                hash: s.hash,
                unit: s.unit,
            })
            .collect()
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
