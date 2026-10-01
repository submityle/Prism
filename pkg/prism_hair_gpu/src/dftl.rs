//! `wgpu` compute twin of Prism's Dynamic Follow-The-Leader strand solver
//! ([`simulate_guides_dftl`](prism_render_architecture::hair::dftl::simulate_guides_dftl)).
//!
//! A groom simulates only its sparse set of *guide* strands; the strand *fast
//! tier* advances each guide with the Dynamic Follow-The-Leader (`DFTL`) length
//! solver — a single-pass, root-to-tip position propagation that makes a guide
//! rigidly inextensible in one sweep, plus Müller's velocity correction that
//! keeps that one-sweep projection from pumping fake energy into the hair. The
//! `CPU` golden for that solve is
//! [`simulate_guides_dftl`](prism_render_architecture::hair::dftl::simulate_guides_dftl),
//! which slices a flat particle pool into per-strand ranges and runs
//! [`simulate_strand_dftl`](prism_render_architecture::hair::dftl::simulate_strand_dftl)
//! on each. This module is the on-device twin: one thread per strand walks the
//! same substep schedule over its own contiguous particle range, so a passing
//! real-device parity test is direct evidence the ported kernel advances the
//! strands to the same state as the reference — not merely that its shader
//! compiles.
//!
//! # What the kernel evaluates
//!
//! [`GpuHairDftl::eval`] takes the same flat inputs as `simulate_guides_dftl`
//! (the particle pool, the per-strand particle counts, the per-particle rest
//! lengths and the [`DftlParams`]) and returns the advanced particle pool.
//! Strands are independent, so the batch is embarrassingly parallel; because
//! each thread mutates a disjoint particle range (and a disjoint scratch
//! range), the in-place updates need no barrier and the read-after-write
//! ordering inside a strand matches the reference sweep for sweep.
//!
//! # Device-side sanitation
//!
//! Unlike the `XPBD` twin, the parameters are uploaded **raw** and sanitized on
//! the device exactly as
//! [`DftlParams::sanitized`](prism_render_architecture::hair::dftl::DftlParams::sanitized)
//! does on the host (non-finite / non-positive `dt` → `1/60`, `substeps` → at
//! least `1`, `damping` / `correction` clamped to `0..=1` with non-finite
//! falling back to `0` / `0.9`, `gravity` components made finite). A poisoned
//! parameter set therefore still dispatches and must parity-match the
//! reference's sanitized result rather than being short-circuited to a no-op on
//! the host. The per-strand rest-length companion slice is all-or-nothing in
//! the reference (`rest_lengths.get(offset..end)` is `Some` for the whole
//! strand or `None`), reproduced here as a `has_rest` flag per strand.
//!
//! Every other guard is reproduced: a no-op call (empty pool, or no strand that
//! fits the pool) returns the pool unchanged without a dispatch, a
//! `strand_lengths` entry that would run past the pool truncates the walk
//! exactly as the reference does, a strand of fewer than two particles is a
//! no-op, and pinned particles (`inverse_mass <= 0`) are never moved.
//!
//! # Portability
//!
//! The solve uses only `sqrt`, `dot`, `min`, `max`, `clamp` and multiply/add in
//! the portable core-`WGSL` subset — no `exp`, `pow` or optional device feature
//! — so the twin runs unmodified on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! The solve contains no transcendental call, so `CPU` and `GPU` evaluate the
//! same closed-form arithmetic. They are **not** bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, and the perturbation
//! compounds over the iterated substeps. The parity test therefore asserts a
//! per-component tolerance (`abs_diff < 1e-4` or `rel_diff < 1e-3`) rather than
//! exact equality.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard `DFTL` inextensible-strand integrator (single-pass
//! `FTL` position propagation + velocity correction) plus `wgpu` compute
//! dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::hair::dftl::{DftlParams, FtlParticle, Vec3};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Raw solve parameters uploaded to the kernel (sanitized on-device). `32`-byte
/// scalar-packed `repr(C)` matching `Params` in `shaders/dftl.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Raw frame time step (seconds); sanitized to `1/60` if non-finite / `<= 0`.
    dt: f32,
    /// Raw gravity acceleration components; each made finite on-device.
    gx: f32,
    gy: f32,
    gz: f32,
    /// Raw velocity damping; sanitized to a `0..=1` fraction.
    damping: f32,
    /// Raw `DFTL` velocity-correction fraction; sanitized to `0..=1`.
    correction: f32,
    /// Raw substep count; sanitized to at least `1`.
    substeps: u32,
    /// Number of strand descriptors in the strands buffer.
    strand_count: u32,
}

/// One guide-strand descriptor uploaded to the kernel. `16`-byte `repr(C)`
/// matching `Strand` in `shaders/dftl.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Strand {
    point_offset: u32,
    point_count: u32,
    has_rest: u32,
    pad: u32,
}

/// A compiled, reusable `DFTL` strand-solver pipeline.
pub struct GpuHairDftl {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairDftl {
    /// Compiles the `DFTL` strand-solver kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairDftl {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_dftl"),
            source: ShaderSource::Wgsl(include_str!("../shaders/dftl.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_dftl_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_dftl_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_dftl_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairDftl {
            module,
            layout,
            pipeline,
        }
    }

    /// Advances the guide-strand pool by one [`DftlParams`] step, returning the
    /// updated particle pool (same length as `particles`).
    ///
    /// The result equals
    /// [`simulate_guides_dftl`](prism_render_architecture::hair::dftl::simulate_guides_dftl)
    /// applied to a clone of `particles`, to within the fused-multiply-add
    /// tolerance documented on this module. Parameters are sanitized on-device,
    /// so a non-finite `dt`, zero `substeps`, or out-of-range tuning values are
    /// still dispatched and must match the reference's sanitized result. A
    /// no-op call (empty pool, or no strand that fits the pool) returns the pool
    /// unchanged without a dispatch — storage buffers cannot be zero-sized.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        particles: &[FtlParticle],
        strand_lengths: &[usize],
        rest_lengths: &[f32],
        params: DftlParams,
    ) -> Vec<FtlParticle> {
        // Only the structural guards short-circuit on the host; the parameter
        // sanitation is performed on-device, so a poisoned parameter set is
        // still dispatched and parity-matched against the sanitized reference.
        if particles.is_empty() {
            return particles.to_vec();
        }

        // Replicate `simulate_guides_dftl`'s offset-slicing: process strands in
        // order until one would run past the pool, then stop. Only fitted
        // strands get a descriptor (including sub-two-particle strands, which
        // the kernel no-ops); the rest of the pool is left untouched (uploaded
        // and read back unchanged).
        let mut descriptors: Vec<Strand> = Vec::with_capacity(strand_lengths.len());
        let mut offset = 0usize;
        for &length in strand_lengths {
            let Some(end) = offset.checked_add(length) else {
                break;
            };
            if end > particles.len() {
                break;
            }
            descriptors.push(Strand {
                point_offset: offset as u32,
                point_count: length as u32,
                has_rest: u32::from(rest_lengths.get(offset..end).is_some()),
                pad: 0,
            });
            offset = end;
        }
        if descriptors.is_empty() {
            return particles.to_vec();
        }

        // Flat per-particle state, stride 8: position, velocity, inverse_mass,
        // rest length (the length of the segment leaving this particle,
        // defaulting to 0 for a missing entry).
        let mut state: Vec<f32> = Vec::with_capacity(particles.len() * 8);
        for (gi, p) in particles.iter().enumerate() {
            state.push(p.position.x);
            state.push(p.position.y);
            state.push(p.position.z);
            state.push(p.velocity.x);
            state.push(p.velocity.y);
            state.push(p.velocity.z);
            state.push(p.inverse_mass);
            state.push(rest_lengths.get(gi).copied().unwrap_or(0.0));
        }

        // Per-particle substep scratch, stride 6: start_position and
        // predicted_position. Each thread writes only its own strand's range,
        // so the initial content is irrelevant; it is zeroed to keep the upload
        // deterministic. A non-empty pool guarantees a non-zero-sized buffer.
        let scratch = vec![0.0f32; particles.len() * 6];

        let uniform = Params {
            dt: params.dt,
            gx: params.gravity.x,
            gy: params.gravity.y,
            gz: params.gravity.z,
            damping: params.damping,
            correction: params.correction,
            substeps: params.substeps,
            strand_count: descriptors.len() as u32,
        };

        let device = ctx.device();
        let state_bytes = (state.len() * size_of::<f32>()) as u64;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_dftl_params"),
            contents: bytemuck::bytes_of(&uniform),
            usage: BufferUsages::UNIFORM,
        });
        let strands_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_dftl_strands"),
            contents: bytemuck::cast_slice(&descriptors),
            usage: BufferUsages::STORAGE,
        });
        let state_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_dftl_state"),
            contents: bytemuck::cast_slice(&state),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });
        let scratch_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_dftl_scratch"),
            contents: bytemuck::cast_slice(&scratch),
            usage: BufferUsages::STORAGE,
        });
        let state_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_dftl_state_stage"),
            size: state_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_dftl_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: strands_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: state_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: scratch_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_dftl_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_dftl_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (descriptors.len() as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&state_buf, 0, &state_stage, 0, state_bytes);
        ctx.queue().submit([encoder.finish()]);

        state_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = state_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let flat = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        state_stage.unmap();
        debug_assert_eq!(flat.len(), particles.len() * 8);

        // Rebuild the particle pool from the read-back state. Inverse mass is
        // never written by the kernel, so it is carried through from the input
        // (untouched particles thus reconstruct exactly as uploaded).
        particles
            .iter()
            .enumerate()
            .map(|(gi, p)| {
                let b = gi * 8;
                FtlParticle {
                    position: Vec3::new(flat[b], flat[b + 1], flat[b + 2]),
                    velocity: Vec3::new(flat[b + 3], flat[b + 4], flat[b + 5]),
                    inverse_mass: p.inverse_mass,
                }
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
