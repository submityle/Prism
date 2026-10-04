//! `wgpu` compute twin of the dependency-free `CPU` golden semi-Lagrangian foam
//! advection
//! ([`advect_foam_field`](prism_render_architecture::water::foam::advect_foam_field)).
//!
//! Foam is a scalar coverage field in `0..=1` carried on the surface-flow grid.
//! Each frame every cell backtraces its centre by `-dt * velocity` (converted
//! to cell units) and bilinearly samples the previous field there — the
//! unconditionally stable, non-negativity-preserving advection `Niagara` and
//! `Crest` foam passes run on device. This twin evaluates that gather on the
//! `GPU` so the foam field advects in the same place the rest of the water
//! subsystem already lives, with no round trip to the host.
//!
//! # Correctness model
//!
//! The gather is embarrassingly parallel: one invocation owns one output cell
//! and reads only the *previous* field, so there is no write hazard, no barrier
//! and no workgroup memory. Each cell performs the identical floating-point
//! operations the golden performs, in the same order (backtrace, clamp, the
//! two-tap-then-one-tap bilinear blend of
//! [`sample_bilinear`](prism_render_architecture::water::foam::sample_bilinear)),
//! so the readback is bit-identical to the golden up to the last-place slack of
//! a `GPU` fused multiply-add. A degenerate request (empty grid or an input
//! shorter than `nx * nz`) returns the density unchanged, matching the golden.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset (multiply/add,
//! `clamp`, `floor` via truncation, and index math) — no transcendental, no
//! optional device feature — so it runs unmodified on Metal, Vulkan and DX12.
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

/// Workgroup width of the advection; must match `@workgroup_size(256)` in
/// `shaders/water_foam_advect.wesl`.
const WORKGROUP_SIZE: u32 = 256;

/// One advected foam field, row-major over the `nx * nz` grid.
///
/// Mirrors the vector the `CPU`
/// [`advect_foam_field`](prism_render_architecture::water::foam::advect_foam_field)
/// returns. A degenerate request yields the density unchanged.
#[derive(Clone, Debug, PartialEq)]
pub struct WaterFoamAdvect {
    /// Row-major `nx * nz` advected foam coverage.
    pub density: Vec<f32>,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/water_foam_advect.wesl` (`16` bytes, 16-byte aligned).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    nx: u32,
    nz: u32,
    dx: f32,
    dt: f32,
}

/// A compiled, reusable foam-advection pipeline.
pub struct GpuWaterFoamAdvect {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWaterFoamAdvect {
    /// Compiles the foam-advection kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWaterFoamAdvect {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_water_foam_advect"),
            source: ShaderSource::Wgsl(include_str!("../shaders/water_foam_advect.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_water_foam_advect_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: true }),
                buffer_entry(4, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_water_foam_advect_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_water_foam_advect_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("advect_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWaterFoamAdvect {
            module,
            layout,
            pipeline,
        }
    }

    /// Advects a row-major `nx x nz` foam field along the flow `(u, v)` by `dt`
    /// on device.
    ///
    /// Returns the density unchanged for a degenerate request (empty grid or an
    /// input shorter than `nx * nz`), matching the golden.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        density: &[f32],
        u: &[f32],
        v: &[f32],
        nx: u32,
        nz: u32,
        dx: f32,
        dt: f32,
    ) -> WaterFoamAdvect {
        let Some(n) = (nx as usize).checked_mul(nz as usize) else {
            return WaterFoamAdvect {
                density: density.to_vec(),
            };
        };
        if n == 0 || density.len() < n || u.len() < n || v.len() < n {
            return WaterFoamAdvect {
                density: density.to_vec(),
            };
        }
        let device = ctx.device();

        let params = Params { nx, nz, dx, dt };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_foam_advect_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let density_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_foam_advect_density"),
            contents: bytemuck::cast_slice(&density[..n]),
            usage: BufferUsages::STORAGE,
        });
        let u_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_foam_advect_u"),
            contents: bytemuck::cast_slice(&u[..n]),
            usage: BufferUsages::STORAGE,
        });
        let v_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_foam_advect_v"),
            contents: bytemuck::cast_slice(&v[..n]),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (n * size_of::<f32>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_foam_advect_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_water_foam_advect_bind_group"),
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
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let stage_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_foam_advect_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_water_foam_advect_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_water_foam_advect_pass"),
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

        WaterFoamAdvect {
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
