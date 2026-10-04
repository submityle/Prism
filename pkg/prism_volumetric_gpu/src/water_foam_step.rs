//! `wgpu` compute twin of the dependency-free `CPU` golden full foam update
//! ([`step_foam`](prism_render_architecture::water::foam::step_foam)).
//!
//! Foam is a scalar coverage field in `0..=1` carried on the surface-flow grid.
//! One step advects the field along the flow, applies a flow-aware exponential
//! decay, adds this frame's foam sources, and clamps coverage back into
//! `0..=1` — the churn-persists / calm-fades foam law the `Niagara` and `Crest`
//! foam passes run on device. This twin fuses the whole update into one
//! dispatch so the foam field evolves in the same place the rest of the water
//! subsystem already lives, with no round trip to the host.
//!
//! # Correctness model
//!
//! The update is embarrassingly parallel: one invocation owns one output cell.
//! The advection stage reads only the *previous* field (a pure gather, no write
//! hazard), and the decay-plus-source stage depends solely on that cell's own
//! advected value and its own flow sample, so no barrier and no workgroup
//! memory are needed. Each cell performs the identical floating-point
//! operations the golden performs, in the same order:
//!
//! 1. backtrace `-dt * velocity` and bilinearly gather the previous field
//!    (mirrors [`sample_bilinear`](prism_render_architecture::water::foam::sample_bilinear)),
//! 2. `flow_speed = sqrt(u*u + v*v)`,
//! 3. flow-aware `rate`
//!    ([`foam_decay_rate`](prism_render_architecture::water::foam::foam_decay_rate)),
//! 4. `decayed = advected * exp_approx(-rate * dt)`
//!    ([`decay_foam`](prism_render_architecture::water::foam::decay_foam)),
//! 5. `clamp(decayed + max(source, 0), 0, 1)`.
//!
//! Step 4's `exp_approx` is the shared water transcendental
//! ([`exp_approx`](prism_render_architecture::water::exp_approx)); the shader
//! replicates it bit for bit — `base = 1 + x/4096`, clamp-to-zero, then twelve
//! squarings — using only multiply/add, so the readback is bit-identical to
//! the golden up to the last-place slack of a `GPU` fused multiply-add. A
//! degenerate request (empty grid or an input shorter than `nx * nz`) returns
//! the density unchanged, matching the twin's device-buffer contract.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset (multiply/add,
//! `clamp`, `sqrt`, `floor` via truncation, and index math) — no transcendental
//! intrinsic, no optional device feature — so it runs unmodified on Metal,
//! Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::foam`；无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

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

/// Workgroup width of the foam step; must match `@workgroup_size(256)` in
/// `shaders/water_foam_step.wesl`.
const WORKGROUP_SIZE: u32 = 256;

/// One advanced foam field, row-major over the `nx * nz` grid.
///
/// Mirrors the vector the `CPU`
/// [`step_foam`](prism_render_architecture::water::foam::step_foam) returns. A
/// degenerate request yields the density unchanged.
#[derive(Clone, Debug, PartialEq)]
pub struct WaterFoamStep {
    /// Row-major `nx * nz` foam coverage after advect, decay, source and clamp.
    pub density: Vec<f32>,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/water_foam_step.wesl` (`32` bytes, 16-byte aligned).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    nx: u32,
    nz: u32,
    dx: f32,
    dt: f32,
    base_decay: f32,
    persistence_floor: f32,
    reference_speed: f32,
    _pad: u32,
}

/// A compiled, reusable foam-step pipeline.
pub struct GpuWaterFoamStep {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWaterFoamStep {
    /// Compiles the foam-step kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWaterFoamStep {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_water_foam_step"),
            source: ShaderSource::Wgsl(include_str!("../shaders/water_foam_step.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_water_foam_step_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: true }),
                buffer_entry(4, BufferBindingType::Storage { read_only: true }),
                buffer_entry(5, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_water_foam_step_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_water_foam_step_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("step_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWaterFoamStep {
            module,
            layout,
            pipeline,
        }
    }

    /// Advances a row-major `nx x nz` foam field one step on device: advect
    /// along the flow `(u, v)`, flow-aware decay, add `sources`, clamp to
    /// `0..=1`.
    ///
    /// Returns the density unchanged for a degenerate request (empty grid or an
    /// input shorter than `nx * nz`), the twin's device-buffer contract.
    #[expect(clippy::too_many_arguments, reason = "mirrors the golden signature")]
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        density: &[f32],
        u: &[f32],
        v: &[f32],
        sources: &[f32],
        nx: u32,
        nz: u32,
        dx: f32,
        dt: f32,
        base_decay: f32,
        persistence_floor: f32,
        reference_speed: f32,
    ) -> WaterFoamStep {
        let Some(n) = (nx as usize).checked_mul(nz as usize) else {
            return WaterFoamStep {
                density: density.to_vec(),
            };
        };
        if n == 0 || density.len() < n || u.len() < n || v.len() < n || sources.len() < n {
            return WaterFoamStep {
                density: density.to_vec(),
            };
        }
        let device = ctx.device();

        let params = Params {
            nx,
            nz,
            dx,
            dt,
            base_decay,
            persistence_floor,
            reference_speed,
            _pad: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_foam_step_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let density_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_foam_step_density"),
            contents: bytemuck::cast_slice(&density[..n]),
            usage: BufferUsages::STORAGE,
        });
        let u_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_foam_step_u"),
            contents: bytemuck::cast_slice(&u[..n]),
            usage: BufferUsages::STORAGE,
        });
        let v_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_foam_step_v"),
            contents: bytemuck::cast_slice(&v[..n]),
            usage: BufferUsages::STORAGE,
        });
        let sources_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_foam_step_sources"),
            contents: bytemuck::cast_slice(&sources[..n]),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (n * size_of::<f32>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_foam_step_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_water_foam_step_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: density_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: u_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: v_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: sources_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let stage_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_foam_step_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_water_foam_step_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_water_foam_step_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per cell; round the last workgroup up (the shader
            // guards `idx >= nx*nz`).
            let groups = (n as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &stage_buf, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        stage_buf.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();

        let density_out = read_f32(&stage_buf);

        WaterFoamStep {
            density: density_out,
        }
    }
}

/// Reads back a mapped `f32` staging buffer into an owned vector, unmapping it.
fn read_f32(stage: &wgpu::Buffer) -> Vec<f32> {
    let view = stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped readback range should be available after poll");
    let out = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
    drop(view);
    stage.unmap();
    out
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
