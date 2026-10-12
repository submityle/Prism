//! `wgpu` compute twin of the dependency-free `CPU` golden per-domain surface
//! reconstruction scalar parameters
//! ([`surface_isovalue`](prism_render_architecture::water::reconstruct::surface_isovalue),
//! [`narrow_band_width_cells`](prism_render_architecture::water::reconstruct::narrow_band_width_cells),
//! [`screen_space_smoothing_iterations`](prism_render_architecture::water::reconstruct::screen_space_smoothing_iterations)).
//!
//! Each fluid domain picks three scalar knobs before its surface is extracted:
//! the iso-surface level the mesh routes cross, the narrow-band half-width in
//! cells the `SDF` route evaluates within, and the screen-space smoothing
//! iteration count that fades with camera distance. All three are small,
//! branchy pure functions of per-domain inputs with no cross-domain coupling,
//! so this twin batches a whole array of domains into one dispatch — one
//! invocation per domain — keeping the parameter choice on device next to the
//! reconstruction passes that consume it.
//!
//! # Correctness model
//!
//! The batch is embarrassingly parallel: one invocation owns one domain and
//! reads only that domain's own inputs, so there is no write hazard, no barrier
//! and no workgroup memory. Each domain performs the identical clamp / `max` /
//! linear-fade / round-to-`u32` operations the golden performs, in the same
//! order. The isovalue is a single clamp-and-multiply, so it is exactly
//! bit-identical up to the last-place slack of a `GPU` fused multiply-add; the
//! two integer counts reduce through the identical `f32`-to-`u32` truncation
//! the golden's `as u32` cast performs on a non-negative value, so they are
//! exactly equal. A degenerate request (zero domains) returns empty parameter
//! vectors, the twin's device-buffer contract.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset (multiply/add, `clamp`,
//! `min`/`max`, a divide, and `f32`-to-`u32` conversion) — no transcendental
//! intrinsic, no optional device feature — so it runs unmodified on Metal,
//! Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::reconstruct`；无第三方引擎源码或衍生代码。
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

/// Workgroup width of the parameter batch; must match `@workgroup_size(256)` in
/// `shaders/water_reconstruct_params.wesl`.
const WORKGROUP_SIZE: u32 = 256;

/// Per-domain reconstruction inputs. One entry drives one domain. Layout
/// matches `DomainInput` in `shaders/water_reconstruct_params.wesl` (`48`
/// bytes).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReconstructDomain {
    /// Fluid rest density; the isovalue is a fraction of it.
    pub rest_density: f32,
    /// Iso level as a fraction of rest density, clamped to `0..=1`.
    pub iso_fraction: f32,
    /// Particle influence radius (meters); scales the narrow band.
    pub particle_radius: f32,
    /// Reconstruction cell size `dx` (meters).
    pub dx: f32,
    /// Narrow-band factor on the influence radius.
    pub band_factor: f32,
    /// Camera distance (meters); fades the smoothing iteration count.
    pub camera_distance: f32,
    /// Maximum screen-space smoothing iterations near the camera.
    pub max_iterations: u32,
    /// Distance at or below which smoothing runs at `max_iterations`.
    pub near_distance: f32,
    /// Distance at or beyond which smoothing runs a single pass.
    pub far_distance: f32,
}

/// Per-domain reconstruction scalar parameters, returned as parallel fields.
///
/// Mirrors the three `CPU` golden selectors evaluated per domain, in input
/// order. A degenerate request yields empty vectors.
#[derive(Clone, Debug, PartialEq)]
pub struct WaterReconstructParams {
    /// Iso-surface level per domain
    /// ([`surface_isovalue`](prism_render_architecture::water::reconstruct::surface_isovalue)).
    pub isovalue: Vec<f32>,
    /// Narrow-band half-width in cells per domain
    /// ([`narrow_band_width_cells`](prism_render_architecture::water::reconstruct::narrow_band_width_cells)).
    pub narrow_band: Vec<u32>,
    /// Screen-space smoothing iteration count per domain
    /// ([`screen_space_smoothing_iterations`](prism_render_architecture::water::reconstruct::screen_space_smoothing_iterations)).
    pub smoothing_iterations: Vec<u32>,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/water_reconstruct_params.wesl` (`16` bytes, 16-byte aligned).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Device layout of one domain's inputs. Matches `DomainInput` in the shader
/// (`48` bytes: nine `4`-byte scalars plus three padding words).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DomainInput {
    rest_density: f32,
    iso_fraction: f32,
    particle_radius: f32,
    dx: f32,
    band_factor: f32,
    camera_distance: f32,
    max_iterations: u32,
    near_distance: f32,
    far_distance: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Device layout of one domain's results. Matches `DomainResult` in the shader
/// (`16` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DomainResult {
    isovalue: f32,
    narrow_band: u32,
    smoothing_iterations: u32,
    _pad0: u32,
}

/// A compiled, reusable reconstruction-parameter batch pipeline.
pub struct GpuWaterReconstructParams {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWaterReconstructParams {
    /// Compiles the parameter batch pipeline on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_water_reconstruct_params"),
            source: ShaderSource::Wgsl(
                include_str!("../shaders/water_reconstruct_params.wesl").into(),
            ),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_water_reconstruct_params_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_water_reconstruct_params_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_water_reconstruct_params_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("params_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWaterReconstructParams {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates the three reconstruction scalars for a batch of domains.
    ///
    /// Returns empty vectors for a degenerate request (zero domains), the
    /// twin's device-buffer contract.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        domains: &[ReconstructDomain],
    ) -> WaterReconstructParams {
        let count = domains.len();
        if count == 0 {
            return WaterReconstructParams {
                isovalue: Vec::new(),
                narrow_band: Vec::new(),
                smoothing_iterations: Vec::new(),
            };
        }
        let device = ctx.device();

        let inputs: Vec<DomainInput> = domains
            .iter()
            .map(|d| DomainInput {
                rest_density: d.rest_density,
                iso_fraction: d.iso_fraction,
                particle_radius: d.particle_radius,
                dx: d.dx,
                band_factor: d.band_factor,
                camera_distance: d.camera_distance,
                max_iterations: d.max_iterations,
                near_distance: d.near_distance,
                far_distance: d.far_distance,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            })
            .collect();

        let params = Params {
            count: count as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_reconstruct_params_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_reconstruct_params_input"),
            contents: bytemuck::cast_slice(&inputs),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<DomainResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_reconstruct_params_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_water_reconstruct_params_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: input_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_reconstruct_params_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_water_reconstruct_params_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_water_reconstruct_params_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per domain; round the last workgroup up (the shader
            // guards `idx >= count`).
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();

        let results = read_results(&stage);
        let mut isovalue = Vec::with_capacity(count);
        let mut narrow_band = Vec::with_capacity(count);
        let mut smoothing_iterations = Vec::with_capacity(count);
        for r in results {
            isovalue.push(r.isovalue);
            narrow_band.push(r.narrow_band);
            smoothing_iterations.push(r.smoothing_iterations);
        }
        WaterReconstructParams {
            isovalue,
            narrow_band,
            smoothing_iterations,
        }
    }
}

/// Reads back a mapped `DomainResult` staging buffer into owned values,
/// unmapping it.
fn read_results(stage: &wgpu::Buffer) -> Vec<DomainResult> {
    let view = stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped readback range should be available after poll");
    let out = bytemuck::cast_slice::<u8, DomainResult>(&view).to_vec();
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
