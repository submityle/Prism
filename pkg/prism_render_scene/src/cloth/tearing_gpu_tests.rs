//! Real-machine GPU coverage of the cloth tearing (constraint-break) detection
//! kernel in `cloth_tearing.wesl`: the `cloth_tearing_flag` entry.
//!
//! CPU golden: `prism_render_architecture::cloth::tearing::tear_flags`, the per-edge
//! break decision that marks every valid two-sided fabric edge whose tensile
//! strain exceeds the break threshold. The kernel is embarrassingly parallel
//! (each invocation writes only its own edge flag and reads shared read-only
//! positions and constraints), so the on-device mask matches the golden
//! bit-for-bit regardless of dispatch order. This module uploads the same
//! particle / constraint fixture, runs one dispatch on a real device, reads the
//! `u32` flag buffer back, and asserts it equals the golden `Vec<bool>` lifted to
//! `u32`.
//!
//! Device acquisition is best-effort: a headless host with no wgpu adapter
//! returns `None` from `try_compute_device`, so each on-device test prints a skip
//! note and passes, keeping the suite green anywhere. A device-free compile test
//! still guards that the shader parses everywhere.

use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor, MapMode,
    PipelineCompilationOptions, PipelineLayoutDescriptor, PollType, ShaderModuleDescriptor,
    ShaderSource, ShaderStages,
};

use prism_render_architecture::cloth::tearing::{tear_flags, TearingParams};
use prism_render_architecture::cloth::{ClothParticle, Compliance, Constraint, ConstraintKind, Vec3};

use super::abi::{GpuClothConstraint, GpuClothTearingParams};
use super::gpu_test_support::{compile_tearing_wgsl, find_entry_point, try_compute_device};
use super::pack::pack_constraint;

/// A free particle (unit inverse mass) at `position`.
fn free_particle(position: Vec3) -> ClothParticle {
    ClothParticle::new(position, 1.0)
}

/// A two-sided rigid stretch edge between `a` and `b` with the given rest length.
fn stretch(a: u32, b: u32, rest: f32) -> Constraint {
    Constraint::new(a, b, rest, Compliance::RIGID, ConstraintKind::Stretch)
}

/// Host upload layout for a particle buffer: `xyz` = position, `w` = inverse mass.
fn upload_positions(particles: &[ClothParticle]) -> Vec<[f32; 4]> {
    particles
        .iter()
        .map(|p| [p.position.x, p.position.y, p.position.z, p.inverse_mass])
        .collect()
}

/// The group-0 layout: read-only positions and constraints, a read-write flag
/// buffer and the `params` uniform, matching `@binding(0..3)` in the WESL.
fn tearing_bind_group_layout(device: &wgpu::Device) -> BindGroupLayout {
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
        label: Some("cloth_tearing_parity_group0"),
        entries: &[
            storage(0, true),
            storage(1, true),
            storage(2, false),
            BindGroupLayoutEntry {
                binding: 3,
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

/// Uploads the fixture and runs one `cloth_tearing_flag` dispatch, returning the
/// per-edge break flags (`1` = torn, `0` = kept) read back from the device.
fn run_tearing_flags_on_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    entry: &str,
    positions: &[[f32; 4]],
    constraints: &[GpuClothConstraint],
    params: GpuClothTearingParams,
) -> Vec<u32> {
    let layout = tearing_bind_group_layout(device);
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("cloth_tearing_parity_pipeline_layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("cloth_tearing_parity"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("cloth_tearing_flag_pipeline"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some(entry),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });

    let positions_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_tearing_positions"),
        contents: bytemuck::cast_slice(positions),
        usage: BufferUsages::STORAGE,
    });
    // Empty slices cannot back a zero-size binding; fall back to one zero element.
    let constraints_fallback = [GpuClothConstraint::default()];
    let constraints_src: &[GpuClothConstraint] = if constraints.is_empty() {
        &constraints_fallback
    } else {
        constraints
    };
    let constraints_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_tearing_constraints"),
        contents: bytemuck::cast_slice(constraints_src),
        usage: BufferUsages::STORAGE,
    });
    let flag_count = constraints.len().max(1);
    let flag_init = vec![0u32; flag_count];
    let flags_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_tearing_flags"),
        contents: bytemuck::cast_slice(&flag_init),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });
    let params_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_tearing_params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("cloth_tearing_parity_bind"),
        layout: &layout,
        entries: &[
            BindGroupEntry { binding: 0, resource: positions_buf.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: constraints_buf.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: flags_buf.as_entire_binding() },
            BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
        ],
    });

    let stage_bytes = (flag_count * size_of::<u32>()) as u64;
    let stage = device.create_buffer(&BufferDescriptor {
        label: Some("cloth_tearing_flags_stage"),
        size: stage_bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("cloth_tearing_parity_encoder"),
    });
    let groups = (constraints.len() as u32).div_ceil(64).max(1);
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("cloth_tearing_flag_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&flags_buf, 0, &stage, 0, stage_bytes);
    queue.submit([encoder.finish()]);

    stage.slice(..).map_async(MapMode::Read, |_| {});
    device
        .poll(PollType::wait_indefinitely())
        .expect("device poll should complete the submitted tearing work");
    let view = stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped tearing flag readback range should be available after poll");
    let mut out = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
    drop(view);
    stage.unmap();
    // Trim the padding slot when the fixture had no constraints.
    out.truncate(constraints.len());
    out
}

/// Runs the golden `tear_flags` and the on-device twin over the same fixture, then
/// asserts every per-edge break flag agrees exactly (the decision is an integer
/// comparison, so parity is bit-exact, not tolerance-based). `params` is sanitised
/// once so the host uniform and the golden see the identical `break_strain`.
fn assert_tearing_parity(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    particles: &[ClothParticle],
    constraints: &[Constraint],
    params: TearingParams,
) {
    let params = params.sanitized();
    let golden: Vec<u32> = tear_flags(constraints, particles, params)
        .into_iter()
        .map(u32::from)
        .collect();

    let packed: Vec<GpuClothConstraint> = constraints.iter().map(pack_constraint).collect();
    let gpu_params = GpuClothTearingParams {
        break_strain: params.break_strain,
        constraint_count: constraints.len() as u32,
        particle_count: particles.len() as u32,
        pad0: 0,
    };
    let entry = find_entry_point(wgsl, "tearing_flag");
    let gpu = run_tearing_flags_on_gpu(
        device,
        queue,
        wgsl,
        &entry,
        &upload_positions(particles),
        &packed,
        gpu_params,
    );

    assert_eq!(gpu.len(), golden.len());
    for (i, (&g, &c)) in gpu.iter().zip(golden.iter()).enumerate() {
        assert_eq!(g, c, "tear flag parity broke at edge {i}: gpu {g} golden {c}");
    }
}

/// The shader must parse and expose its entry point on every host, including
/// headless CI with no GPU adapter. No device is needed, so this never skips.
#[test]
fn cloth_tearing_wesl_compiles() {
    let wgsl = compile_tearing_wgsl();
    let _ = find_entry_point(&wgsl, "tearing_flag");
}

/// A mix of an over-strained edge, a slack edge, a one-sided tether stretched
/// past threshold, and an out-of-range edge exercises every golden guard: only
/// the valid two-sided over-strained edge should tear.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "on a host with no wgpu device the skip note must reach the test log"
)]
fn mixed_edges_tear_like_golden() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("cloth_tearing mixed_edges: no wgpu adapter, skipping");
        return;
    };
    let wgsl = compile_tearing_wgsl();
    let particles = [
        free_particle(Vec3::ZERO),
        free_particle(Vec3::new(2.0, 0.0, 0.0)),
        free_particle(Vec3::new(2.0, 1.05, 0.0)),
    ];
    // Edge 0-1 strain 1.0 (tears), edge 1-2 strain ~0.05 (slack), a tether past
    // threshold (one-sided, never), and an out-of-range 0-9 edge (never).
    let constraints = vec![
        stretch(0, 1, 1.0),
        stretch(1, 2, 1.0),
        Constraint::new(0, 1, 1.0, Compliance::RIGID, ConstraintKind::Tether),
        stretch(0, 9, 1.0),
    ];
    assert_tearing_parity(
        &device,
        &queue,
        &wgsl,
        &particles,
        &constraints,
        TearingParams { break_strain: 0.5 },
    );
}

/// A fully slack sheet (every edge below threshold) must produce an all-zero
/// flag mask exactly like the golden, guarding against a false-positive tear.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "on a host with no wgpu device the skip note must reach the test log"
)]
fn all_slack_never_tears() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("cloth_tearing all_slack: no wgpu adapter, skipping");
        return;
    };
    let wgsl = compile_tearing_wgsl();
    let particles = [
        free_particle(Vec3::ZERO),
        free_particle(Vec3::new(1.02, 0.0, 0.0)),
        free_particle(Vec3::new(2.01, 0.0, 0.0)),
    ];
    let constraints = vec![stretch(0, 1, 1.0), stretch(1, 2, 1.0)];
    assert_tearing_parity(
        &device,
        &queue,
        &wgsl,
        &particles,
        &constraints,
        TearingParams { break_strain: 0.5 },
    );
}

/// A compressed edge (current length below rest, negative strain) must never
/// tear regardless of how far it is compressed, since `break_strain` is
/// non-negative. Mirrors the golden's compression-safe behaviour.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "on a host with no wgpu device the skip note must reach the test log"
)]
fn compressed_edge_never_tears() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("cloth_tearing compressed: no wgpu adapter, skipping");
        return;
    };
    let wgsl = compile_tearing_wgsl();
    let particles = [
        free_particle(Vec3::ZERO),
        free_particle(Vec3::new(0.1, 0.0, 0.0)),
    ];
    let constraints = vec![stretch(0, 1, 1.0)];
    assert_tearing_parity(
        &device,
        &queue,
        &wgsl,
        &particles,
        &constraints,
        TearingParams { break_strain: 0.5 },
    );
}
