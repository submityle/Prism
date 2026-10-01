//! `wgpu` compute twin of Prism's `Cosserat` rod solver
//! ([`simulate_guides_cosserat`](prism_render_architecture::hair::cosserat::simulate_guides_cosserat)).
//!
//! A pure mass-spring strand (the guide solver) carries no material frame and
//! therefore no twist energy; tight curls and braids need the torsional
//! stiffness and natural `helix` rest shape only a position-and-orientation
//! based `Cosserat` rod expresses. Each rod is an edge-vector poly-line of
//! particles paired with one quaternion material frame per segment; the
//! relative rotation between two adjacent frames is a discrete `Darboux` vector
//! (two bending curvatures plus twist), and driving it toward a non-zero rest
//! `Darboux` vector encodes the curl/twist rest a spring network cannot. The
//! `CPU` golden for that solve is
//! [`simulate_guides_cosserat`](prism_render_architecture::hair::cosserat::simulate_guides_cosserat),
//! which slices a flat particle pool into per-rod ranges and runs
//! [`simulate_strand_cosserat`](prism_render_architecture::hair::cosserat::simulate_strand_cosserat)
//! on each. This is the on-device twin: one thread per rod walks the same
//! substep/iteration schedule over its own contiguous particle and orientation
//! ranges, so a passing real-device parity test is direct evidence the ported
//! kernel advances the rods to the same state as the reference — not merely
//! that its shader compiles.
//!
//! # What the kernel evaluates
//!
//! [`GpuCosserat::eval`] takes the same flat inputs as
//! `simulate_guides_cosserat` (the particle pool, the per-segment orientations,
//! the per-rod particle counts, the per-segment rest lengths and the per-joint
//! rest `Darboux` vectors, plus the [`CosseratParams`]) and returns the
//! advanced particle pool and orientations. Rods are independent, so the batch
//! is embarrassingly parallel; because each thread mutates a disjoint particle
//! and orientation range, the in-place Gauss-Seidel updates need no barrier and
//! the read-after-write ordering inside a rod matches the reference sweep for
//! sweep.
//!
//! All host-derived scalars (`sub_dt`, `inv_sub_dt = 1 / sub_dt`, `retain =
//! 1 - damping`, and the three `*_alpha = compliance / sub_dt^2` values) are
//! precomputed here in the reference's evaluation order and uploaded in the
//! uniform block, so the device never re-derives them. The per-rod rest-length,
//! rest-`Darboux` and orientation companion slices are all-or-nothing in the
//! reference (`get(offset..end)` is `Some` for the whole rod or `None`),
//! reproduced here as a `has_rest` / `has_darboux` flag plus an `o_count` that
//! is `0` when the orientation slice does not fit (bend-twist no-op).
//!
//! Every guard is reproduced: a no-op call (empty pool, or no rod that fits the
//! pool) returns the inputs unchanged without a dispatch, a `strand_lengths`
//! entry that would run past the pool truncates the walk exactly as the
//! reference does, a rod of fewer than two particles is skipped, and pinned
//! particles (`inverse_mass <= 0`) are never moved. The kernel re-normalizes
//! every orientation on entry, mirroring the reference's per-frame
//! `sanitized()` for the finite inputs the parity domain supplies.
//!
//! # Portability
//!
//! The solve uses only `sqrt` (via `length`), `min`, `max`, `dot` and
//! multiply/add in the portable core-`WGSL` subset — no `exp`, `pow`,
//! transcendental or optional device feature — so the twin runs unmodified on
//! Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! The solve contains no transcendental call, so `CPU` and `GPU` evaluate the
//! same closed-form arithmetic. They are **not** bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, and the perturbation
//! compounds over the iterated solve. The parity test therefore asserts a
//! per-component tolerance (`abs_diff < 1e-4` or `rel_diff < 1e-3`) rather than
//! exact equality.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard position-based `Cosserat` rod (compliant edge-length +
//! bend-twist `Darboux` projection) plus `wgpu` compute dispatch, following
//! `Bergou` 2008/2010 and `Kugelstadt` 2016 public formulations; no Unreal
//! Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::hair::cosserat::{CosseratParams, Quat, RodParticle, Vec3};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Uniform solve parameters uploaded to the kernel. `32`-byte scalar-packed
/// `repr(C)` matching `Params` in `shaders/cosserat.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    sub_dt: f32,
    inv_sub_dt: f32,
    retain: f32,
    stretch_alpha: f32,
    bend_alpha: f32,
    twist_alpha: f32,
    substeps: u32,
    strand_count: u32,
}

/// One rod descriptor uploaded to the kernel. `32`-byte `repr(C)` matching
/// `Strand` in `shaders/cosserat.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Strand {
    p_offset: u32,
    p_count: u32,
    o_offset: u32,
    o_count: u32,
    rd_offset: u32,
    rd_count: u32,
    has_rest: u32,
    has_darboux: u32,
}

/// A compiled, reusable `Cosserat` rod solver pipeline.
pub struct GpuCosserat {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuCosserat {
    /// Compiles the `Cosserat` solver kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuCosserat {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_cosserat"),
            source: ShaderSource::Wgsl(include_str!("../shaders/cosserat.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_cosserat_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
                buffer_entry(4, BufferBindingType::Storage { read_only: false }),
                buffer_entry(5, BufferBindingType::Storage { read_only: true }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_cosserat_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_cosserat_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuCosserat {
            module,
            layout,
            pipeline,
        }
    }

    /// Advances the concatenated `Cosserat` rods by one [`CosseratParams`] step,
    /// returning the updated particle pool and orientation array (same lengths
    /// as the inputs).
    ///
    /// The result equals
    /// [`simulate_guides_cosserat`](prism_render_architecture::hair::cosserat::simulate_guides_cosserat)
    /// applied to clones of `particles` and `orientations`, to within the
    /// fused-multiply-add tolerance documented on this module. A no-op call
    /// (empty pool, or no rod that fits the pool) returns the inputs unchanged
    /// without a dispatch — storage buffers cannot be zero-sized.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        particles: &[RodParticle],
        orientations: &[Quat],
        strand_lengths: &[usize],
        rest_lengths: &[f32],
        rest_darboux: &[Vec3],
        params: CosseratParams,
    ) -> (Vec<RodParticle>, Vec<Quat>) {
        // No-op short-circuit, matching the reference's `count < 2` guard at the
        // batch level (an empty pool can hold no rod).
        if particles.is_empty() {
            return (particles.to_vec(), orientations.to_vec());
        }

        // Flat per-particle state, stride 8: position, velocity, inverse_mass,
        // rest length (filled below for rods that carry rest lengths).
        let mut state: Vec<f32> = Vec::with_capacity(particles.len() * 8);
        for p in particles {
            state.push(p.position.x);
            state.push(p.position.y);
            state.push(p.position.z);
            state.push(p.velocity.x);
            state.push(p.velocity.y);
            state.push(p.velocity.z);
            state.push(p.inverse_mass);
            state.push(0.0);
        }

        // Replicate `simulate_guides_cosserat`' offset-slicing: walk rods in
        // order until one would run past the pool, then stop. For a rod of `L`
        // particles the solver consumes `L` particles, `L - 1` orientations and
        // rest lengths, and `L - 2` rest `Darboux` vectors; companion slices
        // that do not fit disable their constraint for the whole rod.
        let mut descriptors: Vec<Strand> = Vec::with_capacity(strand_lengths.len());
        let mut p_off = 0usize;
        let mut o_off = 0usize;
        let mut rl_off = 0usize;
        let mut rd_off = 0usize;
        for &length in strand_lengths {
            let Some(p_end) = p_off.checked_add(length) else {
                break;
            };
            if p_end > particles.len() {
                break;
            }
            let segments = length.saturating_sub(1);
            let o_end = o_off.saturating_add(segments);
            let rl_end = rl_off.saturating_add(segments);
            let rd_end = rd_off.saturating_add(segments.saturating_sub(1));

            let has_rest = rest_lengths.get(rl_off..rl_end).is_some();
            let has_darboux = rest_darboux.get(rd_off..rd_end).is_some();
            // The orientation slice is read all-or-nothing: when it does not fit
            // the reference substitutes an empty slice (bend-twist no-op).
            let o_count = if orientations.get(o_off..o_end).is_some() {
                segments
            } else {
                0
            };

            // Pack this rod's segment rest lengths into the owning particle's
            // state slot (segment `i` leaves particle `p_off + i`).
            if has_rest {
                for i in 0..segments {
                    state[(p_off + i) * 8 + 7] = rest_lengths[rl_off + i];
                }
            }

            descriptors.push(Strand {
                p_offset: p_off as u32,
                p_count: length as u32,
                o_offset: o_off as u32,
                o_count: o_count as u32,
                rd_offset: rd_off as u32,
                rd_count: segments.saturating_sub(1) as u32,
                has_rest: u32::from(has_rest),
                has_darboux: u32::from(has_darboux),
            });

            p_off = p_end;
            o_off = o_end;
            rl_off = rl_end;
            rd_off = rd_end;
        }
        if descriptors.is_empty() {
            return (particles.to_vec(), orientations.to_vec());
        }

        // Host-derived scalars, computed in the reference's exact evaluation
        // order (`params.sanitized()` then `sub_dt = dt / substeps`) so the
        // uploaded values are bit-identical to the golden's.
        let sanitized = params.sanitized();
        let substeps = sanitized.substeps;
        let sub_dt = sanitized.dt / substeps as f32;
        let sub_dt_sq = sub_dt * sub_dt;
        let retain = 1.0 - sanitized.damping;
        let inv_sub_dt = 1.0 / sub_dt;
        let stretch_alpha = sanitized.stretch_compliance / sub_dt_sq;
        let bend_alpha = sanitized.bend_compliance / sub_dt_sq;
        let twist_alpha = sanitized.twist_compliance / sub_dt_sq;

        // Flat orientations, stride 4: quaternion w, x, y, z. Padded with one
        // dummy quat when empty so the storage buffer is never zero-sized (no
        // descriptor references it, so the kernel never reads the pad).
        let mut orient: Vec<f32> = Vec::with_capacity(orientations.len().max(1) * 4);
        for q in orientations {
            orient.push(q.w);
            orient.push(q.x);
            orient.push(q.y);
            orient.push(q.z);
        }
        if orientations.is_empty() {
            orient.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);
        }

        // Flat rest `Darboux` vectors, stride 3, padded like `orient` so the
        // buffer is never zero-sized.
        let mut rd: Vec<f32> = Vec::with_capacity(rest_darboux.len().max(1) * 3);
        for v in rest_darboux {
            rd.push(v.x);
            rd.push(v.y);
            rd.push(v.z);
        }
        if rest_darboux.is_empty() {
            rd.extend_from_slice(&[0.0, 0.0, 0.0]);
        }

        // Scratch previous-position buffer, stride 3. Written by predict before
        // it is read, so its initial contents do not matter.
        let prev: Vec<f32> = alloc_zeroed(particles.len() * 3);

        let device = ctx.device();
        let uniform = Params {
            sub_dt,
            inv_sub_dt,
            retain,
            stretch_alpha,
            bend_alpha,
            twist_alpha,
            substeps,
            strand_count: descriptors.len() as u32,
        };

        let state_bytes = (state.len() * size_of::<f32>()) as u64;
        let orient_bytes = (orient.len() * size_of::<f32>()) as u64;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_cosserat_params"),
            contents: bytemuck::bytes_of(&uniform),
            usage: BufferUsages::UNIFORM,
        });
        let strands_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_cosserat_strands"),
            contents: bytemuck::cast_slice(&descriptors),
            usage: BufferUsages::STORAGE,
        });
        let state_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_cosserat_state"),
            contents: bytemuck::cast_slice(&state),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });
        let orient_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_cosserat_orient"),
            contents: bytemuck::cast_slice(&orient),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });
        let prev_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_cosserat_prev"),
            contents: bytemuck::cast_slice(&prev),
            usage: BufferUsages::STORAGE,
        });
        let rd_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_cosserat_rd"),
            contents: bytemuck::cast_slice(&rd),
            usage: BufferUsages::STORAGE,
        });
        let state_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_cosserat_state_stage"),
            size: state_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let orient_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_cosserat_orient_stage"),
            size: orient_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_cosserat_bind_group"),
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
                    resource: orient_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: prev_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: rd_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_cosserat_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_cosserat_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (descriptors.len() as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&state_buf, 0, &state_stage, 0, state_bytes);
        encoder.copy_buffer_to_buffer(&orient_buf, 0, &orient_stage, 0, orient_bytes);
        ctx.queue().submit([encoder.finish()]);

        state_stage.slice(..).map_async(MapMode::Read, |_| {});
        orient_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();

        let state_flat = read_mapped(&state_stage);
        let orient_flat = read_mapped(&orient_stage);
        debug_assert_eq!(state_flat.len(), particles.len() * 8);

        // Rebuild the particle pool. Inverse mass is never written by the
        // kernel, so it is carried through from the input.
        let out_particles = particles
            .iter()
            .enumerate()
            .map(|(gi, p)| {
                let b = gi * 8;
                RodParticle {
                    position: Vec3::new(state_flat[b], state_flat[b + 1], state_flat[b + 2]),
                    velocity: Vec3::new(state_flat[b + 3], state_flat[b + 4], state_flat[b + 5]),
                    inverse_mass: p.inverse_mass,
                }
            })
            .collect();

        // Rebuild only the real orientations (any dummy pad is ignored).
        let out_orientations = orientations
            .iter()
            .enumerate()
            .map(|(oi, _)| {
                let b = oi * 4;
                Quat::new(
                    orient_flat[b],
                    orient_flat[b + 1],
                    orient_flat[b + 2],
                    orient_flat[b + 3],
                )
            })
            .collect();

        (out_particles, out_orientations)
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

/// Allocates a zero-filled `f32` buffer of `len` elements (at least one, so the
/// backing storage buffer is never zero-sized).
fn alloc_zeroed(len: usize) -> Vec<f32> {
    vec![0.0f32; len.max(1)]
}

/// Reads a mapped staging buffer back into an owned `f32` vector and unmaps it.
fn read_mapped(stage: &wgpu::Buffer) -> Vec<f32> {
    let view = stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped readback range should be available after poll");
    let flat = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
    drop(view);
    stage.unmap();
    flat
}
