//! `wgpu` compute twin of Prism's divergence-free curl-noise wind field
//! ([`curl_wind_map`](prism_render_architecture::hair::wind_field::curl_wind_map)).
//!
//! A believable groom under wind must gust and swirl without the strands ever
//! appearing to inflate or collapse: the velocity field must be
//! **divergence-free** (`div w = 0`, incompressible). Rather than solve a fluid,
//! grooms borrow `Bridson` 2007 `curl-noise`: take the curl of a smooth vector
//! potential `P`, `w = curl P`. Because `div(curl P) = 0` identically, the field
//! is divergence-free *by construction*, with no solve and no grid — just
//! evaluate the potential's derivatives at a point. This is the approach grooms
//! in `UE5` and offline pipelines use for cheap, art-directable wind.
//!
//! # What the kernel evaluates
//!
//! [`GpuHairCurlWind::eval`] takes a slice of world positions and returns one
//! wind-velocity vector per position in input order — the array-in/array-out
//! form used to drive per-strand-vertex external forces. One invocation owns one
//! position (`@compute @workgroup_size(64)`, one-dimensional dispatch over
//! `global_invocation_id.x`); invocations past the position count early-return.
//! Positions and velocities are uploaded flattened (three `f32` per point) to
//! avoid `vec3` storage alignment, exactly like the `self_collision_grid` twin.
//!
//! # Potential field
//!
//! The potential `P = (Px, Py, Pz)` is a deterministic integer **value noise**:
//! a `u32` hash of the integer lattice cell seeds a pseudo-random corner value,
//! and a cubic `smoothstep` fades a trilinear interpolation. Three independent
//! channels (distinct seed offsets) give the three components; the velocity is
//! `amplitude * curl(P)`, with the curl approximated by central **finite
//! differences**. Time is injected as a coordinate offset so the field animates
//! smoothly. No table constants live only on the host here — the field is
//! generated procedurally, and the host uploads only the raw authored
//! `frequency` / `amplitude` / `seed` / `time`, which the shader sanitizes
//! itself (bit-faithfully to the golden) so it cannot drift.
//!
//! # Distinct from the `wind` twin
//!
//! The sibling [`GpuWindField`](crate::wind) twin evaluates the analytic
//! steady + gust + turbulent `wind_acceleration` model. This twin is a
//! curl-of-value-noise divergence-free field: a hashed integer lattice, cubic
//! `smoothstep`, trilinear blend and a central-difference curl. The two share no
//! algorithm — analytic gust superposition versus procedural curl-noise — so
//! they are genuinely distinct ports.
//!
//! # Portability
//!
//! The kernel uses only `floor`, `bitcast`, dynamic array indexing and
//! multiply/add in the portable core-`WGSL` subset — no `exp`, `pow`, `sin` or
//! optional device feature — so the twin runs unmodified on Metal, Vulkan and
//! DX12.
//!
//! # Correctness model
//!
//! The lattice hash is pure `u32` integer math (`bitcast<u32>` reproduces Rust's
//! two's-complement `as u32`, and `u32` multiply/add wrap exactly like
//! `wrapping_mul` / `wrapping_add`), so the hashed corner values are exact. The
//! subsequent trilinear blend and the finite-difference curl are long
//! multiply-add chains a `GPU` may fuse, and the difference is scaled by
//! `inv_two_eps = 500`, amplifying low-`ULP` noise, so `CPU` and `GPU` agree to
//! within the documented fma tolerance (`abs_diff < 1e-4` or `rel_diff < 1e-3`)
//! rather than bit-for-bit. The `frequency` / `amplitude` sanitizers mirror the
//! golden's [`WindFieldParams::sanitized`] exactly, so negative / non-finite
//! params collapse to the same safe field the golden emits and every component
//! stays finite.
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: `Bridson` 2007 curl-noise and `wgpu` compute dispatch; no Unreal
//! Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::hair::wind_field::{curl_wind_map, WindFieldParams};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Uniform parameters for one curl-wind dispatch. Layout matches `Params` in
/// `shaders/wind_field.wesl`: the raw authored field controls plus the position
/// count, scalar-packed into one `32`-byte uniform slot (a multiple of 16).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    frequency: f32,
    amplitude: f32,
    seed: u32,
    time: f32,
    position_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable curl-noise wind-field pipeline.
pub struct GpuHairCurlWind {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairCurlWind {
    /// Compiles the curl-noise wind-field kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairCurlWind {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_wind_field"),
            source: ShaderSource::Wgsl(include_str!("../shaders/wind_field.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_wind_field_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_wind_field_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_wind_field_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairCurlWind {
            module,
            layout,
            pipeline,
        }
    }

    /// Maps each world position to its divergence-free wind velocity for the
    /// given field params and time, returning one velocity per position in input
    /// order.
    ///
    /// Element `i` equals the `CPU` golden
    /// [`curl_wind_map`](prism_render_architecture::hair::wind_field::curl_wind_map)
    /// of `positions[i]` to within the module's documented fma tolerance, with
    /// negative / non-finite `frequency` collapsing to `1.0` and negative /
    /// non-finite `amplitude` collapsing to `0.0` (no wind). An empty batch
    /// yields an empty vector without a dispatch — storage buffers cannot be
    /// zero-sized.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        params: WindFieldParams,
        positions: &[[f32; 3]],
        time: f32,
    ) -> Vec<[f32; 3]> {
        let position_count = positions.len();
        if position_count == 0 {
            return Vec::new();
        }

        let device = ctx.device();

        // The shader sanitizes frequency / amplitude itself (bit-faithfully to
        // the golden), so upload the raw authored values.
        let uniforms = Params {
            frequency: params.frequency,
            amplitude: params.amplitude,
            seed: params.seed,
            time,
            position_count: position_count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let out_bytes = (position_count as u64) * (size_of::<[f32; 3]>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_wind_field_params"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM,
        });
        let positions_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_wind_field_positions"),
            contents: bytemuck::cast_slice(positions),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_wind_field_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_wind_field_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_wind_field_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: positions_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_wind_field_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_wind_field_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (position_count as u32).div_ceil(64);
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
        let out = bytemuck::cast_slice::<u8, [f32; 3]>(&view).to_vec();
        drop(view);
        out_stage.unmap();

        out
    }
}

/// The `CPU` golden curl-noise wind-field map, re-exported so the parity test
/// can assert the device twin against the identical reference it mirrors.
#[must_use]
pub fn reference_curl_wind_map(
    params: WindFieldParams,
    positions: &[[f32; 3]],
    time: f32,
) -> Vec<[f32; 3]> {
    curl_wind_map(params, positions, time)
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
