//! `wgpu` compute twin of the guide-strand wind pre-pass
//! ([`apply_wind`](prism_render_architecture::hair::wind::apply_wind)).
//!
//! Ambient wind is injected as an external pre-pass before the `XPBD`
//! constraint solve (design 6.3): every *free* guide particle gains the wind
//! displacement `wind_acceleration(field, pos, time) * dt^2` — the same
//! semi-implicit `dt^2` scaling the dynamics integrator uses for gravity, so
//! the next solve reads it back as injected velocity. *Pinned* particles (the
//! skinned root, `inverse_mass <= 0`) are never moved. The `CPU` golden is
//! [`apply_wind`](prism_render_architecture::hair::wind::apply_wind); this crate
//! is the on-device twin that reproduces the full pre-pass — the pinned skip,
//! the `dt^2` integration scaling and the in-place position update — one thread
//! per particle over one shared wind field, so a passing real-device parity
//! test is direct evidence the ported kernel moves the particles exactly as the
//! reference does, not merely that its shader compiles.
//!
//! # What the kernel evaluates
//!
//! [`GpuHairApplyWind::eval`] advances a batch of [`StrandParticle`]s under one
//! [`WindField`] over `(time, dt)`. For each particle it computes the shared
//! wind acceleration — the steady push along the wind direction, the along-wind
//! gust pulse and the small cross-wind turbulent flutter, including the
//! hand-written range-reduced Taylor sine (`sin_turns`) the reference uses in
//! place of a hardware `sin` — then writes `position + acceleration * dt^2` for
//! free particles and the unchanged position for pinned ones. A non-positive or
//! non-finite `dt` is a host-side no-op (the reference's early return), returning
//! the input positions unchanged without a dispatch.
//!
//! # Distinction from the `wind` twin
//!
//! The sister [`GpuWindField`](crate::wind::GpuWindField) twin evaluates the raw
//! acceleration field for a batch of independent point/time/field triples and
//! never touches particles. This twin instead reproduces the *integration*
//! pre-pass over one shared field: the pinned skip, the `dt^2` scaling and the
//! in-place position update that actually moves the groom. They share the
//! `wind_acceleration` kernel but differ in dispatch shape and output.
//!
//! # Portability
//!
//! The kernel uses only `sqrt`, `min`/`max`, `dot`, `round`, `fma` and
//! multiply/add in the portable core-`WGSL` subset — no `exp`, `pow`, hardware
//! `sin` or optional device feature — so it runs unmodified on Metal, Vulkan
//! and DX12.
//!
//! # Correctness model
//!
//! Both the field (a closed-form polynomial, the reference deliberately avoids
//! `f32::sin`) and the `pos + accel * dt^2` update are the same expression on
//! both sides, diverging only through legal fused-multiply-add contraction: the
//! reference's `f32::mul_add` Horner chain maps to WGSL `fma`, and the
//! surrounding products may still be contracted by the driver. The parity test
//! asserts a tolerance (`abs_diff < 1e-4` or `rel_diff < 1e-3`) per free-particle
//! component, while pinned particles and a calm field are copied through
//! bit-for-bit, so a genuinely wrong port (a moved pinned root, a dropped `dt^2`,
//! a swapped axis) fails and legal fma contraction passes.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard steady+gust+turbulence wind pre-pass with a
//! hand-written Taylor sine plus `wgpu` compute dispatch; no Unreal Engine
//! source or derived code.

use core::mem::size_of;

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::hair::dynamics::StrandParticle;
use prism_render_architecture::hair::wind::{apply_wind, WindField};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Shared uniform parameters for one wind pre-pass dispatch. Layout matches
/// `Params` in `shaders/apply_wind.wesl`: the particle count (plus three pad
/// words for the `16`-byte uniform alignment), the shared wind field, and the
/// sample time plus the frame `dt` (plus pad to a `64`-byte stride).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    dx: f32,
    dy: f32,
    dz: f32,
    speed: f32,
    gust_amplitude: f32,
    gust_frequency: f32,
    turbulence: f32,
    time: f32,
    dt: f32,
    pad3: f32,
    pad4: f32,
    pad5: f32,
}

/// One guide particle uploaded to the kernel. `16`-byte `repr(C)` matching
/// `Particle` in `shaders/apply_wind.wesl`: the world-space position then the
/// reciprocal mass (`inv_mass <= 0` is pinned).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct WindParticle {
    px: f32,
    py: f32,
    pz: f32,
    inv_mass: f32,
}

impl WindParticle {
    /// Flattens a [`StrandParticle`] into its uploadable position + inverse
    /// mass. `prev_position` is unused by the wind pre-pass.
    fn from_particle(p: &StrandParticle) -> WindParticle {
        WindParticle {
            px: p.position.x,
            py: p.position.y,
            pz: p.position.z,
            inv_mass: p.inverse_mass,
        }
    }
}

/// Reference wrapper: applies the `CPU` golden
/// [`apply_wind`](prism_render_architecture::hair::wind::apply_wind) to a copy
/// of `particles` and returns the resulting positions, one `[x, y, z]` per
/// particle in input order. This is the exact value the device twin's
/// [`GpuHairApplyWind::eval`] must reproduce.
#[must_use]
pub fn reference_apply_wind(
    particles: &[StrandParticle],
    field: WindField,
    time: f32,
    dt: f32,
) -> Vec<[f32; 3]> {
    let mut scratch = particles.to_vec();
    apply_wind(&mut scratch, field, time, dt);
    scratch
        .iter()
        .map(|p| [p.position.x, p.position.y, p.position.z])
        .collect()
}

/// A compiled, reusable wind pre-pass pipeline.
pub struct GpuHairApplyWind {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairApplyWind {
    /// Compiles the wind pre-pass kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairApplyWind {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_apply_wind"),
            source: ShaderSource::Wgsl(include_str!("../shaders/apply_wind.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_apply_wind_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_apply_wind_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_apply_wind_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairApplyWind {
            module,
            layout,
            pipeline,
        }
    }

    /// Advances `particles` under `field` over `(time, dt)`, returning one
    /// `[x, y, z]` world-space position per particle in input order.
    ///
    /// The returned position for a free particle equals its input position plus
    /// `wind_acceleration(field, position, time) * dt^2`, matching
    /// [`apply_wind`](prism_render_architecture::hair::wind::apply_wind) to
    /// within the fused-multiply-add tolerance documented on this module;
    /// pinned particles are returned unchanged. A non-positive or non-finite
    /// `dt` is a no-op that returns the input positions unchanged, and an empty
    /// `particles` slice yields an empty result — storage buffers cannot be
    /// zero-sized, so both are handled by an early return without a dispatch.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        particles: &[StrandParticle],
        field: WindField,
        time: f32,
        dt: f32,
    ) -> Vec<[f32; 3]> {
        // Host-side mirror of the reference's early return: a non-positive or
        // non-finite `dt` moves nothing, so return the input positions as-is.
        if particles.is_empty() || dt <= 0.0 || !dt.is_finite() {
            return particles
                .iter()
                .map(|p| [p.position.x, p.position.y, p.position.z])
                .collect();
        }
        let device = ctx.device();

        let params = Params {
            count: particles.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
            dx: field.direction.x,
            dy: field.direction.y,
            dz: field.direction.z,
            speed: field.speed,
            gust_amplitude: field.gust_amplitude,
            gust_frequency: field.gust_frequency,
            turbulence: field.turbulence,
            time,
            dt,
            pad3: 0.0,
            pad4: 0.0,
            pad5: 0.0,
        };

        let uploaded: Vec<WindParticle> =
            particles.iter().map(WindParticle::from_particle).collect();

        // Three f32 (x, y, z) per particle.
        let out_bytes = (particles.len() as u64) * 3 * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_apply_wind_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let particles_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_apply_wind_particles"),
            contents: bytemuck::cast_slice(&uploaded),
            usage: BufferUsages::STORAGE,
        });
        let values_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_apply_wind_values"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let values_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_apply_wind_values_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_apply_wind_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: particles_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: values_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_apply_wind_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_apply_wind_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (particles.len() as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&values_buf, 0, &values_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        values_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = values_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let flat = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        values_stage.unmap();
        debug_assert_eq!(flat.len(), particles.len() * 3);
        flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
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
