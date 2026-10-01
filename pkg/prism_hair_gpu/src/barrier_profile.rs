//! `wgpu` compute twin of Prism's soft-barrier profile goldens
//! ([`barrier_energy`](prism_render_architecture::hair::barrier_contact::barrier_energy)
//! and
//! [`barrier_force_magnitude`](prism_render_architecture::hair::barrier_contact::barrier_force_magnitude)).
//!
//! For each contact distance `d` the kernel returns the soft-barrier energy
//! `b(d)` and the repulsive force magnitude `-b'(d)` under a shared set of
//! [`BarrierParams`]. On the clamped distance `d_e = clamp(d, d_floor, dhat)`,
//!
//! ```text
//!     b(d)    = k * (dhat - d_e)^2 * (1/d_e - 1/dhat)
//!     -b'(d)  = k * (dhat - d_e) * [ 2 (1/d_e - 1/dhat) + (dhat - d_e)/d_e^2 ]
//! ```
//!
//! and both are `0` in the free region `d >= dhat`.
//!
//! # Why one thread per sample
//!
//! Each sample's `(energy, force)` depends only on its own distance and the
//! shared params, so this is embarrassingly parallel: one thread owns one
//! distance and writes its energy/force pair. The host lays the distances out
//! flat so threads never alias.
//!
//! # What the kernel evaluates
//!
//! [`GpuBarrierProfile::eval`] takes shared [`BarrierParams`] and a batch of
//! distances and returns one `(energy, force)` pair per distance, in input
//! order.
//!
//! # Correctness model
//!
//! The reference sanitizes its params (forcing `dhat > 0`, `d_floor` into
//! `(0, dhat)`, `stiffness >= 0`) and treats a non-finite distance as "far"
//! (energy/force `0`). This twin supplies the already-sanitized params to the
//! device (via [`BarrierParams::sanitized`]) and asserts against the same
//! reference, so the kernel can keep only the `d >= dhat` free-region early-out
//! and the `clamp` to `[d_floor, dhat]` and omit the non-finite guard. The
//! solve is a single closed-form evaluation (a handful of add/sub/mul/divide),
//! no chained recurrence, so the only `CPU` vs `GPU` divergence is legal
//! fused-multiply-add contraction and correctly rounded division; parity is
//! asserted to within `abs_diff < 1e-4` or `rel_diff < 1e-3`. Sample distances
//! stay clear of the `dhat`/`d_floor` branch boundaries so the branch taken is
//! identical on both sides.
//!
//! # Portability
//!
//! The kernel uses only add/sub/mul/divide/clamp/max in the portable
//! core-`WGSL` subset — no `exp`, `pow`, `log` or optional device feature — so
//! the twin runs unmodified on Metal, Vulkan and DX12.
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: Prism's own rational soft-barrier (IPC-style, no Unreal Engine
//! source or derived code) plus a `wgpu` compute dispatch.

use core::mem::size_of;

use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use prism_render_architecture::hair::barrier_contact::{
    barrier_energy, barrier_force_magnitude, BarrierParams,
};

use crate::context::GpuContext;

/// Uniform parameters for one barrier-profile dispatch. Layout matches `Params`
/// in `shaders/barrier_profile.wesl`: the sanitized barrier tuning plus the
/// sample count, in one `16`-byte uniform slot.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    dhat: f32,
    stiffness: f32,
    d_floor: f32,
    sample_count: u32,
}

/// Compiled per-sample barrier-profile compute twin: the shader module (kept
/// alive so its pipeline stays valid), the bind-group layout and the pipeline.
pub struct GpuBarrierProfile {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuBarrierProfile {
    /// Compiles the per-sample barrier-profile kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuBarrierProfile {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_barrier_profile"),
            source: ShaderSource::Wgsl(include_str!("../shaders/barrier_profile.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_barrier_profile_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_barrier_profile_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_barrier_profile_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuBarrierProfile {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates the barrier profile for every distance under the shared
    /// `params`, returning one `(energy, force)` pair per input distance, in
    /// input order.
    ///
    /// The pair for distance `i` equals
    /// `(barrier_energy(dists[i], params), barrier_force_magnitude(dists[i], params))`,
    /// to within the single-evaluation tolerance documented on this module
    /// (`abs_diff < 1e-4` or `rel_diff < 1e-3`). The params are sanitized once
    /// on the host so the device sees the same clamped tuning the reference
    /// uses. The empty batch is handled without a dispatch — storage buffers
    /// cannot be zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, params: BarrierParams, dists: &[f32]) -> Vec<(f32, f32)> {
        if dists.is_empty() {
            return Vec::new();
        }

        let p = params.sanitized();
        let device = ctx.device();
        let gpu_params = GpuParams {
            dhat: p.dhat,
            stiffness: p.stiffness,
            d_floor: p.d_floor,
            sample_count: dists.len() as u32,
        };
        let out_bytes = (dists.len() as u64) * 2 * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_barrier_profile_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let dists_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_barrier_profile_dists"),
            contents: bytemuck::cast_slice(dists),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_barrier_profile_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_barrier_profile_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_barrier_profile_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: dists_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_barrier_profile_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_barrier_profile_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (dists.len() as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();

        let view = stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let flat = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        stage.unmap();

        debug_assert_eq!(flat.len(), dists.len() * 2);

        let mut out: Vec<(f32, f32)> = Vec::with_capacity(dists.len());
        for i in 0..dists.len() {
            let b = i * 2;
            out.push((flat[b], flat[b + 1]));
        }
        out
    }
}

/// The `CPU` golden `(energy, force)` pair, re-exported so the parity test can
/// assert the device twin against the identical references it mirrors.
///
/// Runs
/// [`barrier_energy`](prism_render_architecture::hair::barrier_contact::barrier_energy)
/// and
/// [`barrier_force_magnitude`](prism_render_architecture::hair::barrier_contact::barrier_force_magnitude)
/// for the distance `d` under `params`.
#[must_use]
pub fn reference_barrier_profile(d: f32, params: BarrierParams) -> (f32, f32) {
    (
        barrier_energy(d, params),
        barrier_force_magnitude(d, params),
    )
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
