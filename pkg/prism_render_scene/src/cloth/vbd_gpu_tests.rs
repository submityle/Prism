//! Real-machine GPU coverage of the Vertex Block Descent cloth solver in
//! `cloth_vbd.wesl`: the `cloth_vbd_predict` / `cloth_vbd_sweep` /
//! `cloth_vbd_velocity` kernels.
//!
//! CPU golden: `prism_render_architecture::cloth::vbd::solve_cloth_vbd_colored`,
//! the color-major VBD sweep that is itself the bit-for-bit reference for the GPU
//! dispatch schedule (parallel within a color, serial across colors). This module
//! uploads the same particle / constraint / coloring fixture, drives one full
//! frame on a real Metal device the way the production dispatch will (predict,
//! then per color a Newton sweep, repeated per iteration, per substep, then
//! velocity recovery), reads positions and velocities back, and asserts parity
//! with the golden.
//!
//! ## Why these cases match the golden within float32 rounding
//!
//! The color partition guarantees no two vertices relaxed in the same dispatch
//! share a constraint, so a color is a Jacobi block whose per-vertex result is
//! independent of order; applying colors in sequence reproduces the golden's
//! Gauss-Seidel sweep. Both paths run the same `float32` Newton step
//! (identical accumulation order over each vertex's CSR adjacency, identical
//! cofactor 3x3 solve), so the only slack is a few ULPs of fused-multiply-add
//! contraction. VBD relaxation is contractive toward the same fixed point, so
//! that slack does not compound across sweeps; the small fixtures here stay well
//! inside the parity tolerance.
//!
//! Device acquisition is best-effort: a headless host with no wgpu adapter
//! returns `None` from `try_compute_device`, so each on-device test prints a skip
//! note and passes, keeping the suite green on any machine. A device-free compile
//! test still guards that the shader parses everywhere.

use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor, MapMode,
    PipelineCompilationOptions, PipelineLayoutDescriptor, PollType, ShaderModuleDescriptor,
    ShaderSource, ShaderStages,
};

use prism_render_architecture::cloth::vbd::{solve_cloth_vbd_colored, VbdParams};
use prism_render_architecture::cloth::vbd_coloring::color_cloth_vertices;
use prism_render_architecture::cloth::{
    ClothParticle, Compliance, Constraint, ConstraintKind, Vec3,
};

use super::abi::GpuClothConstraint;
use super::gpu_test_support::{compile_vbd_wgsl, find_entry_point, try_compute_device};
use super::pack::pack_constraint;

/// Per-component GPU-vs-CPU absolute tolerance for the VBD parity twin.
///
/// Wider than the single-pass `gpu_test_support::PARITY_EPS` (`1e-4`) because a VBD
/// frame is many chained passes (predict, then a Newton sweep per color per
/// iteration per substep, then velocity recovery), and each per-vertex 3x3
/// cofactor solve leaves a few ULPs of fused-multiply-add slack that the CPU
/// golden, running plain `mul`/`add`, does not. VBD relaxation is contractive
/// toward the same backward-Euler fixed point, so that slack stays bounded
/// rather than compounding; `2e-3` on `O(1)` positions absorbs it while staying
/// far tighter than any real kernel bug (which diverges at `O(0.1)`). This
/// mirrors the reasoning behind `sim_gpu_tests`' relaxed multi-pass tolerance.
const VBD_PARITY_EPS: f32 = 2.0e-3;

/// A free particle (unit inverse mass) at `position`.
fn free_particle(position: Vec3) -> ClothParticle {
    ClothParticle::new(position, 1.0)
}

/// A two-sided stretch edge between `a` and `b` with the given rest length and
/// compliance (larger compliance = softer edge).
fn stretch(a: u32, b: u32, rest: f32, compliance: f32) -> Constraint {
    Constraint::new(a, b, rest, Compliance(compliance), ConstraintKind::Stretch)
}

/// Builds a `rows` x `cols` grid of particles on the XY plane with unit spacing,
/// the top row (`y == 0`) pinned so the sheet hangs under gravity. Row-major
/// index `r * cols + c`.
fn hanging_grid(rows: u32, cols: u32) -> Vec<ClothParticle> {
    let mut particles = Vec::with_capacity((rows * cols) as usize);
    for r in 0..rows {
        for c in 0..cols {
            let position = Vec3::new(c as f32, -(r as f32), 0.0);
            if r == 0 {
                particles.push(ClothParticle::pinned(position));
            } else {
                particles.push(free_particle(position));
            }
        }
    }
    particles
}

/// Structural stretch edges for a `rows` x `cols` grid: every horizontal and
/// vertical neighbor pair, rest length 1 (the grid spacing), shared compliance.
fn grid_stretch_constraints(rows: u32, cols: u32, compliance: f32) -> Vec<Constraint> {
    let mut constraints = Vec::new();
    let idx = |r: u32, c: u32| r * cols + c;
    for r in 0..rows {
        for c in 0..cols {
            if c + 1 < cols {
                constraints.push(stretch(idx(r, c), idx(r, c + 1), 1.0, compliance));
            }
            if r + 1 < rows {
                constraints.push(stretch(idx(r, c), idx(r + 1, c), 1.0, compliance));
            }
        }
    }
    constraints
}

/// Host CSR adjacency mirroring the private `vbd::build_adjacency`: `entries`
/// lists, per vertex in ascending vertex order, the indices of every constraint
/// touching it, in ascending constraint index (the order the loop appends them).
/// A self-constraint (`a == b`) or an out-of-range endpoint is skipped exactly as
/// the solver skips it, so the GPU sweep sums each vertex's constraints in the
/// same order the golden does. Returns `(offsets, entries)` with
/// `offsets.len() == count + 1`.
fn build_csr(constraints: &[Constraint], count: usize) -> (Vec<u32>, Vec<u32>) {
    let mut adjacency: Vec<Vec<u32>> = vec![Vec::new(); count];
    for (index, constraint) in constraints.iter().enumerate() {
        let a = constraint.a as usize;
        let b = constraint.b as usize;
        if a == b || a >= count || b >= count {
            continue;
        }
        adjacency[a].push(index as u32);
        adjacency[b].push(index as u32);
    }
    let mut offsets = Vec::with_capacity(count + 1);
    let mut entries = Vec::new();
    offsets.push(0u32);
    for row in &adjacency {
        entries.extend_from_slice(row);
        offsets.push(entries.len() as u32);
    }
    (offsets, entries)
}

/// Host mirror of the shader `VbdParams` uniform (48 bytes, `std140`-compatible:
/// the leading `vec3` forces 16-byte alignment, and the trailing pads round the
/// struct to a 16-byte multiple).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuVbdParams {
    /// Gravity times `dt_sub` squared.
    gravity_step: [f32; 3],
    /// Velocity retention `1 - damping`.
    retain: f32,
    /// Substep timestep.
    dt_sub: f32,
    /// Reciprocal substep timestep.
    inv_dt_sub: f32,
    /// Substep timestep squared.
    dt_sub_sq: f32,
    /// Particle count.
    particle_count: u32,
    /// Current color slice start.
    color_begin: u32,
    /// Current color slice end.
    color_end: u32,
    /// Padding.
    pad0: u32,
    /// Padding.
    pad1: u32,
}

/// Host upload layout for a particle buffer: `xyz` = position, `w` = inverse mass.
fn upload_positions(particles: &[ClothParticle]) -> Vec<[f32; 4]> {
    particles
        .iter()
        .map(|p| [p.position.x, p.position.y, p.position.z, p.inverse_mass])
        .collect()
}

/// Host upload layout for a velocity buffer: `xyz` = velocity, `w` unused.
fn upload_velocities(particles: &[ClothParticle]) -> Vec<[f32; 4]> {
    particles
        .iter()
        .map(|p| [p.velocity.x, p.velocity.y, p.velocity.z, 0.0])
        .collect()
}

/// The group-0 layout: four read-write particle-state buffers (positions,
/// velocities, previous, targets), four read-only topology buffers (constraints,
/// CSR offsets, CSR entries, color order) and the `params` uniform, matching
/// `@binding(0..8)` in the WESL.
fn vbd_bind_group_layout(device: &wgpu::Device) -> BindGroupLayout {
    let storage = |binding: u32, read_only: bool| BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("cloth_vbd_parity_group0"),
        entries: &[
            storage(0, false),
            storage(1, false),
            storage(2, false),
            storage(3, false),
            storage(4, true),
            storage(5, true),
            storage(6, true),
            storage(7, true),
            BindGroupLayoutEntry {
                binding: 8,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

/// Uploads one VBD fixture and drives a full frame on the device with exactly the
/// production dispatch schedule: per substep run `cloth_vbd_predict`, then for
/// each iteration run `cloth_vbd_sweep` once per color (each color a separate
/// submit so a color observes the prior colors' updated positions, reproducing the
/// golden's Gauss-Seidel-across-colors order), then run `cloth_vbd_velocity`.
/// Each pass is its own submit so ordering is strict. Returns the read-back
/// `(positions, velocities)` as `vec4` rows (`xyz` plus the untouched `w`).
///
/// The host scalars (`dt_sub`, `dt_sub_sq`, `inv_dt_sub`, `retain`,
/// `gravity_step`) are formed with the same op order as
/// `solve_cloth_vbd_colored` so the uniform the shader reads is bit-identical to
/// the golden's internal scalars.
#[expect(
    clippy::too_many_arguments,
    reason = "a VBD dispatch fixture legitimately needs the four state buffers, three topology buffers, the coloring and the solver params; grouping them into a struct would only relabel the same wiring"
)]
fn run_vbd_on_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    coloring: &prism_render_architecture::cloth::vbd_coloring::VertexColoring,
    positions: &[[f32; 4]],
    velocities: &[[f32; 4]],
    constraints: &[GpuClothConstraint],
    csr_offsets: &[u32],
    csr_entries: &[u32],
    color_order: &[u32],
    params: VbdParams,
    dt: f32,
) -> (Vec<[f32; 4]>, Vec<[f32; 4]>) {
    let count = positions.len() as u32;
    let layout = vbd_bind_group_layout(device);
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("cloth_vbd_parity_pipeline_layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("cloth_vbd_parity"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });
    let make_pipeline = |entry: &str, label: &str| {
        device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some(entry),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        })
    };
    let predict_entry = find_entry_point(wgsl, "predict");
    let sweep_entry = find_entry_point(wgsl, "sweep");
    let velocity_entry = find_entry_point(wgsl, "velocity");
    let predict_pipeline = make_pipeline(&predict_entry, "cloth_vbd_predict_pipeline");
    let sweep_pipeline = make_pipeline(&sweep_entry, "cloth_vbd_sweep_pipeline");
    let velocity_pipeline = make_pipeline(&velocity_entry, "cloth_vbd_velocity_pipeline");

    // Read-write particle state.
    let positions_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_vbd_positions"),
        contents: bytemuck::cast_slice(positions),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });
    let velocities_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_vbd_velocities"),
        contents: bytemuck::cast_slice(velocities),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });
    let zero_state = vec![[0.0f32; 4]; count.max(1) as usize];
    let previous_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_vbd_previous"),
        contents: bytemuck::cast_slice(&zero_state),
        usage: BufferUsages::STORAGE,
    });
    let targets_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_vbd_targets"),
        contents: bytemuck::cast_slice(&zero_state),
        usage: BufferUsages::STORAGE,
    });

    // Read-only topology. Empty slices cannot back a zero-size binding, so fall
    // back to a single zero element; the kernels never index past the real ranges.
    let constraints_fallback = [GpuClothConstraint::default()];
    let constraints_src: &[GpuClothConstraint] = if constraints.is_empty() {
        &constraints_fallback
    } else {
        constraints
    };
    let constraints_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_vbd_constraints"),
        contents: bytemuck::cast_slice(constraints_src),
        usage: BufferUsages::STORAGE,
    });
    let offsets_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_vbd_csr_offsets"),
        contents: bytemuck::cast_slice(csr_offsets),
        usage: BufferUsages::STORAGE,
    });
    let entries_fallback = [0u32];
    let entries_src: &[u32] = if csr_entries.is_empty() {
        &entries_fallback
    } else {
        csr_entries
    };
    let entries_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_vbd_csr_entries"),
        contents: bytemuck::cast_slice(entries_src),
        usage: BufferUsages::STORAGE,
    });
    let order_fallback = [0u32];
    let order_src: &[u32] = if color_order.is_empty() {
        &order_fallback
    } else {
        color_order
    };
    let order_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_vbd_color_order"),
        contents: bytemuck::cast_slice(order_src),
        usage: BufferUsages::STORAGE,
    });
    let params_buf = device.create_buffer(&BufferDescriptor {
        label: Some("cloth_vbd_params"),
        size: size_of::<GpuVbdParams>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("cloth_vbd_parity_bind"),
        layout: &layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: positions_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: velocities_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: previous_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: targets_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: constraints_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 5,
                resource: offsets_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 6,
                resource: entries_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 7,
                resource: order_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 8,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    // Scalars formed exactly like solve_cloth_vbd_colored.
    let dt_sub = dt / params.substeps as f32;
    let dt_sub_sq = dt_sub * dt_sub;
    let inv_dt_sub = 1.0 / dt_sub;
    let retain = 1.0 - params.damping;
    let gravity_step = params.gravity.scale(dt_sub_sq);
    let base = GpuVbdParams {
        gravity_step: [gravity_step.x, gravity_step.y, gravity_step.z],
        retain,
        dt_sub,
        inv_dt_sub,
        dt_sub_sq,
        particle_count: count,
        color_begin: 0,
        color_end: 0,
        pad0: 0,
        pad1: 0,
    };

    let dispatch = |pipeline: &wgpu::ComputePipeline, groups: u32, label: &str| {
        let mut encoder =
            device.create_command_encoder(&CommandEncoderDescriptor { label: Some(label) });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(groups.max(1), 1, 1);
        }
        queue.submit([encoder.finish()]);
    };

    let per_particle_groups = count.div_ceil(64);
    for _ in 0..params.substeps {
        queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&base));
        dispatch(
            &predict_pipeline,
            per_particle_groups,
            "cloth_vbd_predict_submit",
        );
        for _ in 0..params.iterations {
            for color in 0..coloring.color_count() as usize {
                let range = coloring.color_range(color);
                if range.end <= range.start {
                    continue;
                }
                let mut p = base;
                p.color_begin = range.start as u32;
                p.color_end = range.end as u32;
                queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&p));
                let color_groups = ((range.end - range.start) as u32).div_ceil(64);
                dispatch(&sweep_pipeline, color_groups, "cloth_vbd_sweep_submit");
            }
        }
        queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&base));
        dispatch(
            &velocity_pipeline,
            per_particle_groups,
            "cloth_vbd_velocity_submit",
        );
    }

    (
        read_back_vec4(device, queue, &positions_buf, count),
        read_back_vec4(device, queue, &velocities_buf, count),
    )
}

/// Copies a device `vec4` storage buffer back to the host, blocking on
/// `device.poll` until the copy completes.
fn read_back_vec4(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    count: u32,
) -> Vec<[f32; 4]> {
    let bytes = (count as u64) * size_of::<[f32; 4]>() as u64;
    let stage = device.create_buffer(&BufferDescriptor {
        label: Some("cloth_vbd_readback_stage"),
        size: bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("cloth_vbd_readback_encoder"),
    });
    encoder.copy_buffer_to_buffer(source, 0, &stage, 0, bytes);
    queue.submit([encoder.finish()]);

    stage.slice(..).map_async(MapMode::Read, |_| {});
    device
        .poll(PollType::wait_indefinitely())
        .expect("device poll should complete the VBD readback copy");
    let view = stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped VBD readback range should be available after poll");
    let out = bytemuck::cast_slice::<u8, [f32; 4]>(&view).to_vec();
    drop(view);
    stage.unmap();
    out
}

/// Runs the golden `solve_cloth_vbd_colored` and the on-device twin over the same
/// particle / constraint / coloring fixture for one `DT` frame, then asserts every
/// particle's position and velocity agree within `VBD_PARITY_EPS` per component.
///
/// `params` is sanitised once so the host uniform and the golden's internal
/// `sanitized` see identical substep / iteration / damping scalars, and the coloring
/// and CSR adjacency are built from the same helpers the production upload uses.
fn assert_vbd_parity(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    particles: &[ClothParticle],
    constraints: &[Constraint],
    params: VbdParams,
) {
    let params = params.sanitized();
    let coloring = color_cloth_vertices(constraints, particles.len());
    let (offsets, entries) = build_csr(constraints, particles.len());
    let packed: Vec<GpuClothConstraint> = constraints.iter().map(pack_constraint).collect();

    let mut golden = particles.to_vec();
    solve_cloth_vbd_colored(&mut golden, constraints, params, DT, &coloring);

    let (gpu_positions, gpu_velocities) = run_vbd_on_gpu(
        device,
        queue,
        wgsl,
        &coloring,
        &upload_positions(particles),
        &upload_velocities(particles),
        &packed,
        &offsets,
        &entries,
        coloring.order(),
        params,
        DT,
    );

    assert_eq!(gpu_positions.len(), golden.len());
    assert_eq!(gpu_velocities.len(), golden.len());
    for (i, particle) in golden.iter().enumerate() {
        let gp = gpu_positions[i];
        let gv = gpu_velocities[i];
        let pos = particle.position;
        let vel = particle.velocity;
        let dp = [
            (gp[0] - pos.x).abs(),
            (gp[1] - pos.y).abs(),
            (gp[2] - pos.z).abs(),
        ];
        let dv = [
            (gv[0] - vel.x).abs(),
            (gv[1] - vel.y).abs(),
            (gv[2] - vel.z).abs(),
        ];
        assert!(
            dp[0] <= VBD_PARITY_EPS && dp[1] <= VBD_PARITY_EPS && dp[2] <= VBD_PARITY_EPS,
            "position parity broke at particle {i}: gpu {gp:?} golden ({}, {}, {}) delta {dp:?}",
            pos.x,
            pos.y,
            pos.z,
        );
        assert!(
            dv[0] <= VBD_PARITY_EPS && dv[1] <= VBD_PARITY_EPS && dv[2] <= VBD_PARITY_EPS,
            "velocity parity broke at particle {i}: gpu {gv:?} golden ({}, {}, {}) delta {dv:?}",
            vel.x,
            vel.y,
            vel.z,
        );
    }
}

/// Frame timestep shared by every parity fixture (one 60 Hz step).
const DT: f32 = 1.0 / 60.0;

/// The shader must parse and expose all three entry points on every host,
/// including headless CI with no GPU adapter. No device is needed, so this test
/// never skips.
#[test]
fn cloth_vbd_wesl_compiles() {
    let wgsl = compile_vbd_wgsl();
    let _ = find_entry_point(&wgsl, "predict");
    let _ = find_entry_point(&wgsl, "sweep");
    let _ = find_entry_point(&wgsl, "velocity");
}

/// A 6x5 sheet pinned along its top row relaxes under gravity exactly like the
/// golden. Moderate compliance keeps the per-vertex Hessian well conditioned so
/// the cofactor solve stays away from the fused-multiply-add-sensitive
/// near-singular regime, and multiple substeps / iterations exercise the full
/// predict / per-color sweep / velocity schedule.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "on a host with no wgpu device the skip note must reach the test log"
)]
fn hanging_grid_relaxes_like_golden() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("cloth_vbd hanging_grid: no wgpu adapter, skipping");
        return;
    };
    let wgsl = compile_vbd_wgsl();
    let particles = hanging_grid(6, 5);
    let constraints = grid_stretch_constraints(6, 5, 0.1);
    let params = VbdParams {
        substeps: 2,
        iterations: 3,
        gravity: Vec3::new(0.0, -9.81, 0.0),
        damping: 0.05,
    };
    assert_vbd_parity(&device, &queue, &wgsl, &particles, &constraints, params);
}

/// The pinned top row (inverse mass 0) must never move: the sweep skips it and
/// velocity recovery zeroes it. This checks the GPU keeps pinned positions
/// bit-identical to the input rather than merely close.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "on a host with no wgpu device the skip note must reach the test log"
)]
fn pinned_top_row_never_moves() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("cloth_vbd pinned_row: no wgpu adapter, skipping");
        return;
    };
    let wgsl = compile_vbd_wgsl();
    let cols = 4u32;
    let particles = hanging_grid(3, cols);
    let constraints = grid_stretch_constraints(3, cols, 0.08);
    let coloring = color_cloth_vertices(&constraints, particles.len());
    let (offsets, entries) = build_csr(&constraints, particles.len());
    let packed: Vec<GpuClothConstraint> = constraints.iter().map(pack_constraint).collect();
    let params = VbdParams {
        substeps: 2,
        iterations: 4,
        gravity: Vec3::new(0.0, -9.81, 0.0),
        damping: 0.0,
    }
    .sanitized();
    let inputs = upload_positions(&particles);
    let (gpu_positions, gpu_velocities) = run_vbd_on_gpu(
        &device,
        &queue,
        &wgsl,
        &coloring,
        &inputs,
        &upload_velocities(&particles),
        &packed,
        &offsets,
        &entries,
        coloring.order(),
        params,
        DT,
    );
    for c in 0..cols as usize {
        assert_eq!(
            gpu_positions[c], inputs[c],
            "pinned vertex {c} position must stay bit-identical",
        );
        assert_eq!(
            gpu_velocities[c],
            [0.0, 0.0, 0.0, 0.0],
            "pinned vertex {c} velocity must be zeroed",
        );
    }
}

/// A one-sided tether (only resists stretching past rest, never compression)
/// exercises the sweep's one-sided early-out branch. The anchor is pinned and the
/// free vertex starts slack (inside rest), so the tether contributes nothing this
/// frame and the free vertex falls under gravity exactly like the golden.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "on a host with no wgpu device the skip note must reach the test log"
)]
fn one_sided_tether_matches_golden() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("cloth_vbd tether: no wgpu adapter, skipping");
        return;
    };
    let wgsl = compile_vbd_wgsl();
    let particles = vec![
        ClothParticle::pinned(Vec3::new(0.0, 0.0, 0.0)),
        free_particle(Vec3::new(0.0, -0.5, 0.0)),
    ];
    // Rest length 2 while the vertices sit 0.5 apart: the tether is slack, so its
    // one-sided branch takes the len <= rest early-out every sweep.
    let constraints = vec![Constraint::new(
        0,
        1,
        2.0,
        Compliance(0.02),
        ConstraintKind::Tether,
    )];
    let params = VbdParams {
        substeps: 3,
        iterations: 2,
        gravity: Vec3::new(0.0, -9.81, 0.0),
        damping: 0.01,
    };
    assert_vbd_parity(&device, &queue, &wgsl, &particles, &constraints, params);
}

/// A single free vertex with no constraints must fall purely under the inertial
/// prediction, exercising the degenerate empty-CSR path (`row_begin == row_end`)
/// and the empty-coloring dispatch guard.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "on a host with no wgpu device the skip note must reach the test log"
)]
fn free_particle_falls_like_golden() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("cloth_vbd free_particle: no wgpu adapter, skipping");
        return;
    };
    let wgsl = compile_vbd_wgsl();
    let particles = vec![free_particle(Vec3::new(1.0, 2.0, -3.0))];
    let constraints: Vec<Constraint> = Vec::new();
    let params = VbdParams {
        substeps: 4,
        iterations: 2,
        gravity: Vec3::new(0.0, -9.81, 0.0),
        damping: 0.0,
    };
    assert_vbd_parity(&device, &queue, &wgsl, &particles, &constraints, params);
}
