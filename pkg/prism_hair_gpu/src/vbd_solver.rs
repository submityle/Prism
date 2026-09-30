//! `wgpu` compute twin of Prism's Vertex Block Descent (VBD) strand solver
//! ([`simulate_strand_vbd`](prism_render_architecture::hair::solver::simulate_strand_vbd)).
//!
//! A groom simulates only its sparse *guide* strands; the many render strands
//! are interpolated from them and never simulate. `XPBD`
//! ([`GpuGuideSolver`](crate::guide_solver::GpuGuideSolver)) is the cheap
//! default, but *stiff* grooms — braids, dreadlocks, wax/gel-set styling — need
//! much higher effective stiffness than position-based projection delivers.
//! `VBD` minimizes the same backward-Euler incremental potential a full Newton
//! solve would, but block-locally: it sweeps the vertices in Gauss-Seidel order
//! and takes one exact per-vertex Newton step against that vertex's own 3x3
//! Hessian each iteration. The `CPU` golden for one strand is
//! [`simulate_strand_vbd`](prism_render_architecture::hair::solver::simulate_strand_vbd);
//! a batch runs it on each strand slice. This crate is the on-device twin: one
//! thread per strand walks the same substep/iteration schedule over its own
//! contiguous particle range, so a passing real-device parity test is direct
//! evidence the ported kernel advances the strands to the same state as the
//! reference — not merely that its shader compiles.
//!
//! # What the kernel evaluates
//!
//! [`GpuVbdSolver::eval`] takes the flat particle pool, the per-strand particle
//! counts, the per-particle rest lengths, the shared colliders and the
//! [`VbdParams`], and returns the advanced particle pool — equivalent to
//! slicing the pool into per-strand ranges and running `simulate_strand_vbd` on
//! each. Strands are independent, so the batch is embarrassingly parallel;
//! because each thread mutates a disjoint particle range (and a disjoint slice
//! of the inertial-target scratch), the in-place Gauss-Seidel updates need no
//! barrier and the read-after-write ordering inside a strand matches the
//! reference sweep for sweep.
//!
//! All host-derived scalars (`sub_dt^2`, `velocity_retain = 1 - clamp(damping)`,
//! the clamped `stretch`/`bending`, `gravity_step = gravity * sub_dt^2`) are
//! precomputed here in the reference's evaluation order and uploaded in the
//! uniform block, so the device never re-derives them. The per-strand
//! rest-length slice is all-or-nothing in the reference, reproduced here as a
//! `has_rest` flag per strand.
//!
//! Every guard is reproduced: a no-op call (empty pool, zero substeps,
//! non-positive or non-finite `dt`, or no strand that fits the pool) returns
//! the pool unchanged without a dispatch, a `strand_lengths` entry that would
//! run past the pool truncates the walk exactly as slicing does, pinned
//! particles (`inverse_mass <= 0`) are never moved, and a (near-)singular vertex
//! Hessian leaves the vertex in place for that sweep.
//!
//! # Portability
//!
//! The solve uses only `sqrt` (via `length`), `min`, `max`, `clamp`, `dot`, and
//! multiply/add in the portable core-`WGSL` subset — no `exp`, `pow` or
//! optional device feature — so the twin runs unmodified on Metal, Vulkan and
//! DX12.
//!
//! # Correctness model
//!
//! The solve contains no transcendental call, so `CPU` and `GPU` evaluate the
//! same closed-form arithmetic. They are **not** bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, and the perturbation
//! compounds over the iterated Newton sweep (plus a 3x3 cofactor inverse per
//! vertex). The parity test therefore asserts a per-component tolerance
//! (`abs_diff < 1e-4` or `rel_diff < 1e-3`) rather than exact equality.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard Vertex Block Descent strand solver (inertia + stretch +
//! bending block-local Newton) plus analytic sphere/capsule push-out and `wgpu`
//! compute dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::hair::collision::Collider;
use prism_render_architecture::hair::dynamics::{StrandParticle, Vec3};
use prism_render_architecture::hair::solver::VbdParams;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Collider-kind discriminant for a sphere, matching the kernel encoding.
const KIND_SPHERE: u32 = 0;
/// Collider-kind discriminant for a capsule, matching the kernel encoding.
const KIND_CAPSULE: u32 = 1;

/// Uniform solve parameters uploaded to the kernel. `48`-byte scalar-packed
/// `repr(C)` matching `Params` in `shaders/vbd_solver.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    gsx: f32,
    gsy: f32,
    gsz: f32,
    velocity_retain: f32,
    sub_dt_sq: f32,
    stretch: f32,
    bending: f32,
    pad0: f32,
    substeps: u32,
    iterations: u32,
    strand_count: u32,
    collider_count: u32,
}

/// One strand descriptor uploaded to the kernel. `16`-byte `repr(C)` matching
/// `Strand` in `shaders/vbd_solver.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Strand {
    point_offset: u32,
    point_count: u32,
    has_rest: u32,
    pad0: u32,
}

/// One analytic collider uploaded to the kernel. `32`-byte `repr(C)` matching
/// `Collider` in `shaders/vbd_solver.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuCollider {
    kind: u32,
    radius: f32,
    ax: f32,
    ay: f32,
    az: f32,
    bx: f32,
    by: f32,
    bz: f32,
}

impl GpuCollider {
    /// Encodes a `CPU` [`Collider`] into its device layout.
    fn encode(collider: Collider) -> GpuCollider {
        match collider {
            Collider::Sphere { center, radius } => GpuCollider {
                kind: KIND_SPHERE,
                radius,
                ax: center.x,
                ay: center.y,
                az: center.z,
                bx: 0.0,
                by: 0.0,
                bz: 0.0,
            },
            Collider::Capsule { a, b, radius } => GpuCollider {
                kind: KIND_CAPSULE,
                radius,
                ax: a.x,
                ay: a.y,
                az: a.z,
                bx: b.x,
                by: b.y,
                bz: b.z,
            },
        }
    }

    /// An inert padding collider, so the storage buffer is never zero-sized
    /// when the caller passes no colliders (`collider_count` stays `0`, so the
    /// kernel never reads it).
    fn dummy() -> GpuCollider {
        GpuCollider {
            kind: KIND_SPHERE,
            radius: 0.0,
            ax: 0.0,
            ay: 0.0,
            az: 0.0,
            bx: 0.0,
            by: 0.0,
            bz: 0.0,
        }
    }
}

/// A compiled, reusable VBD-solver pipeline.
pub struct GpuVbdSolver {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuVbdSolver {
    /// Compiles the VBD-solver kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuVbdSolver {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_vbd_solver"),
            source: ShaderSource::Wgsl(include_str!("../shaders/vbd_solver.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_vbd_solver_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
                buffer_entry(4, BufferBindingType::Storage { read_only: true }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_vbd_solver_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_vbd_solver_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuVbdSolver {
            module,
            layout,
            pipeline,
        }
    }

    /// Advances the strand pool by one [`VbdParams`] step, returning the updated
    /// particle pool (same length as `particles`).
    ///
    /// The result equals slicing `particles` into per-strand ranges and running
    /// [`simulate_strand_vbd`](prism_render_architecture::hair::solver::simulate_strand_vbd)
    /// on each, to within the fused-multiply-add tolerance documented on this
    /// module. A no-op call (empty pool, zero substeps, non-positive or
    /// non-finite `dt`, or no strand that fits the pool) returns the pool
    /// unchanged without a dispatch — storage buffers cannot be zero-sized.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        particles: &[StrandParticle],
        strand_lengths: &[usize],
        rest_lengths: &[f32],
        colliders: &[Collider],
        params: VbdParams,
    ) -> Vec<StrandParticle> {
        // No-op short-circuit, matching `simulate_strand_vbd`'s guard.
        if particles.is_empty()
            || params.substeps == 0
            || params.dt <= 0.0
            || !params.dt.is_finite()
        {
            return particles.to_vec();
        }

        // Slice the flat pool into per-strand ranges, processing strands in
        // order until one would run past the pool, then stop. Only fitted
        // strands get a descriptor; the rest of the pool is left untouched.
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
                pad0: 0,
            });
            offset = end;
        }
        if descriptors.is_empty() {
            return particles.to_vec();
        }

        // Host-derived scalars, computed in the reference's exact evaluation
        // order so the uploaded values are bit-identical to the golden's.
        let sub_dt = params.dt / params.substeps as f32;
        let sub_dt_sq = sub_dt * sub_dt;
        let velocity_retain = 1.0 - params.damping.clamp(0.0, 1.0);
        let stretch = params.stretch_stiffness.max(0.0);
        let bending = params.bending_stiffness.max(0.0);
        let gravity_step = params.gravity.scale(sub_dt_sq);

        // Flat per-particle state, stride 8: position, prev_position,
        // inverse_mass, rest length (defaulting to 0 for a missing entry).
        let mut state: Vec<f32> = Vec::with_capacity(particles.len() * 8);
        for (gi, p) in particles.iter().enumerate() {
            state.push(p.position.x);
            state.push(p.position.y);
            state.push(p.position.z);
            state.push(p.prev_position.x);
            state.push(p.prev_position.y);
            state.push(p.prev_position.z);
            state.push(p.inverse_mass);
            state.push(rest_lengths.get(gi).copied().unwrap_or(0.0));
        }

        let mut gpu_colliders: Vec<GpuCollider> =
            colliders.iter().map(|&c| GpuCollider::encode(c)).collect();
        let collider_count = gpu_colliders.len() as u32;
        if gpu_colliders.is_empty() {
            gpu_colliders.push(GpuCollider::dummy());
        }

        let device = ctx.device();
        let uniform = Params {
            gsx: gravity_step.x,
            gsy: gravity_step.y,
            gsz: gravity_step.z,
            velocity_retain,
            sub_dt_sq,
            stretch,
            bending,
            pad0: 0.0,
            substeps: params.substeps,
            iterations: params.iterations,
            strand_count: descriptors.len() as u32,
            collider_count,
        };

        let state_bytes = (state.len() * size_of::<f32>()) as u64;
        let targets_bytes = (particles.len() * 3 * size_of::<f32>()) as u64;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_vbd_solver_params"),
            contents: bytemuck::bytes_of(&uniform),
            usage: BufferUsages::UNIFORM,
        });
        let strands_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_vbd_solver_strands"),
            contents: bytemuck::cast_slice(&descriptors),
            usage: BufferUsages::STORAGE,
        });
        let state_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_vbd_solver_state"),
            contents: bytemuck::cast_slice(&state),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });
        // Inertial-target scratch, written then read by the owning thread each
        // substep. Its contents are recomputed on-device before any read, so it
        // only needs to be non-zero-sized; wgpu zero-initializes it.
        let targets_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_vbd_solver_targets"),
            size: targets_bytes,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let colliders_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_vbd_solver_colliders"),
            contents: bytemuck::cast_slice(&gpu_colliders),
            usage: BufferUsages::STORAGE,
        });
        let state_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_vbd_solver_state_stage"),
            size: state_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_vbd_solver_bind_group"),
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
                    resource: targets_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: colliders_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_vbd_solver_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_vbd_solver_pass"),
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
                StrandParticle {
                    position: Vec3::new(flat[b], flat[b + 1], flat[b + 2]),
                    prev_position: Vec3::new(flat[b + 3], flat[b + 4], flat[b + 5]),
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
