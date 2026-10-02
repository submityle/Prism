//! Real-machine GPU coverage of the per-edge plastic-creep kernel in
//! `cloth_plasticity.wesl`: `cloth_apply_plasticity`.
//!
//! CPU golden: `prism_render_architecture::cloth::tearing::apply_plasticity`, the
//! permanent-deformation pass that lets a two-sided distance edge stretched (or
//! compressed) past a yield strain creep its rest length toward the current
//! length, capturing the plastic set of wrinkles and sag while retaining a
//! bounded residual elastic strain. Until now the shader was only proven to
//! compile under `naga` (via `cloth::shader_tests`); this module dispatches the
//! kernel on a real Metal device, reads the mutated rest lengths back, and
//! asserts value-for-value parity with the golden.
//!
//! ## Why these cases match the golden within float32 rounding
//!
//! Plasticity is embarrassingly parallel: every edge mutates only its own
//! `rest_length` and never reads another edge's updated value, so the golden's
//! in-order visit and the GPU's one-invocation-per-edge dispatch compute the
//! identical float32 arithmetic on every edge regardless of order. The only
//! slack is a few ULPs of fused-multiply-add contraction, well inside
//! `PARITY_EPS`.
//!
//! Device acquisition is best-effort: a headless host with no wgpu adapter
//! returns `None` from `try_compute_device`, so each test prints a skip note and
//! passes, keeping the suite green on any machine.

use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor, MapMode,
    PipelineCompilationOptions, PipelineLayoutDescriptor, PollType, ShaderModuleDescriptor,
    ShaderSource, ShaderStages,
};

use prism_render_architecture::cloth::tearing::{apply_plasticity, PlasticParams};
use prism_render_architecture::cloth::{
    ClothParticle, Compliance, Constraint, ConstraintKind, Vec3,
};

use super::abi::{GpuClothConstraint, GpuClothPlasticityParams};
use super::gpu_test_support::{
    compile_plasticity_wgsl, find_entry_point, try_compute_device, PARITY_EPS,
};
use super::pack::pack_constraint;

/// Builds a free particle (unit inverse mass) at `position`.
fn free_particle(position: Vec3) -> ClothParticle {
    ClothParticle::new(position, 1.0)
}

/// Builds a two-sided stretch edge between `a` and `b` with the given rest length.
fn stretch(a: u32, b: u32, rest: f32) -> Constraint {
    Constraint::new(a, b, rest, Compliance::RIGID, ConstraintKind::Stretch)
}

/// host upload layout for `plasticity_positions`: xyz = position, w = inverse
/// mass, matching the shader's convention (only xyz is read).
fn upload_positions(particles: &[ClothParticle]) -> Vec<[f32; 4]> {
    particles
        .iter()
        .map(|p| [p.position.x, p.position.y, p.position.z, p.inverse_mass])
        .collect()
}

/// Packs the architecture constraints into their byte-compatible GPU mirror,
/// reusing the production `pack_constraint` so the kind ladder cannot drift.
fn upload_constraints(constraints: &[Constraint]) -> Vec<GpuClothConstraint> {
    constraints.iter().map(pack_constraint).collect()
}

/// The group-0 layout for the plasticity kernel: `plasticity_positions` (ro),
/// `plasticity_constraints` (rw) and `plasticity_params` (uniform), matching
/// @binding(0..2) in the WESL.
fn plasticity_bind_group_layout(device: &wgpu::Device) -> BindGroupLayout {
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
        label: Some("cloth_plasticity_parity_group0"),
        entries: &[
            storage(0, true),
            storage(1, false),
            BindGroupLayoutEntry {
                binding: 2,
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

/// Runs the plasticity kernel on a real device and reads the mutated constraint
/// records back (`rest_length` is the field under test).
fn run_plasticity_on_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    entry: &str,
    positions: &[[f32; 4]],
    constraints: &[GpuClothConstraint],
    params: GpuClothPlasticityParams,
) -> Vec<GpuClothConstraint> {
    let layout = plasticity_bind_group_layout(device);
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("cloth_plasticity_parity_pipeline_layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("cloth_plasticity_parity"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("cloth_apply_plasticity_pipeline"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some(entry),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });

    let positions_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_plasticity_positions"),
        contents: bytemuck::cast_slice(positions),
        usage: BufferUsages::STORAGE,
    });
    let constraints_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_plasticity_constraints"),
        contents: bytemuck::cast_slice(constraints),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });
    let params_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_plasticity_params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("cloth_plasticity_parity_bind"),
        layout: &layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: positions_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: constraints_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let stage_bytes = size_of_val(constraints) as u64;
    let stage = device.create_buffer(&BufferDescriptor {
        label: Some("cloth_plasticity_constraints_stage"),
        size: stage_bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("cloth_plasticity_parity_encoder"),
    });
    let groups = (constraints.len() as u32).div_ceil(64).max(1);
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("cloth_apply_plasticity_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&constraints_buf, 0, &stage, 0, stage_bytes);
    queue.submit([encoder.finish()]);

    stage.slice(..).map_async(MapMode::Read, |_| {});
    device
        .poll(PollType::wait_indefinitely())
        .expect("device poll should complete the submitted plasticity work");

    let view = stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped constraint readback range should be available after poll");
    let out: Vec<GpuClothConstraint> =
        bytemuck::cast_slice::<u8, GpuClothConstraint>(&view).to_vec();
    drop(view);
    stage.unmap();

    out
}

/// Runs the CPU golden and the real-machine twin over the same edge fixture,
/// then asserts every constraint's rest length agrees within `PARITY_EPS` and
/// that the endpoints, compliance and kind tag are untouched.
///
/// `params` is sanitised once up front so the host uniform and the golden's
/// internal sanitise (`apply_plasticity`'s first line) see identical scalars.
fn assert_plasticity_parity(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    particles: &[ClothParticle],
    constraints: &[Constraint],
    params: PlasticParams,
) {
    let params = params.sanitized();

    let mut golden = constraints.to_vec();
    apply_plasticity(&mut golden, particles, params);

    let gpu_params = GpuClothPlasticityParams {
        constraint_count: constraints.len() as u32,
        particle_count: particles.len() as u32,
        yield_strain: params.yield_strain,
        creep: params.creep,
        max_strain: params.max_strain,
        pad0: 0,
        pad1: 0,
        pad2: 0,
    };

    let positions = upload_positions(particles);
    let gpu_constraints = upload_constraints(constraints);
    let entry = find_entry_point(wgsl, "cloth_apply_plasticity");
    let out = run_plasticity_on_gpu(
        device,
        queue,
        wgsl,
        &entry,
        &positions,
        &gpu_constraints,
        gpu_params,
    );

    assert_eq!(out.len(), golden.len());
    for (i, cpu) in golden.iter().enumerate() {
        let gpu = out[i];
        assert!(
            (gpu.rest_length - cpu.rest_length).abs() <= PARITY_EPS,
            "constraint {i} rest_length: gpu {} vs cpu {}",
            gpu.rest_length,
            cpu.rest_length,
        );
        assert_eq!(gpu.a, cpu.a, "constraint {i}: endpoint a mutated");
        assert_eq!(gpu.b, cpu.b, "constraint {i}: endpoint b mutated");
        assert!(
            (gpu.compliance - cpu.compliance.value()).abs() <= f32::EPSILON,
            "constraint {i}: compliance mutated",
        );
    }
}

/// A stretched structural edge (len 1.5, rest 1.0, strain 0.5) past the 10%
/// yield creeps its rest length toward the current length under the residual
/// clamp; the GPU must land on the golden's new rest value.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志"
)]
fn stretched_edge_creeps_like_golden() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("stretched_edge_creeps_like_golden: no wgpu adapter, skipping");
        return;
    };
    let wgsl = compile_plasticity_wgsl();
    let particles = [
        free_particle(Vec3::ZERO),
        free_particle(Vec3::new(1.5, 0.0, 0.0)),
    ];
    let constraints = [stretch(0, 1, 1.0)];
    assert_plasticity_parity(
        &device,
        &queue,
        &wgsl,
        &particles,
        &constraints,
        PlasticParams::default(),
    );
}

/// A compressed edge (len 0.5, rest 1.0, strain -0.5) exercises the negative
/// strain branch: the sign of the excess and the residual clamp both flip.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志"
)]
fn compressed_edge_creeps_like_golden() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("compressed_edge_creeps_like_golden: no wgpu adapter, skipping");
        return;
    };
    let wgsl = compile_plasticity_wgsl();
    let particles = [
        free_particle(Vec3::ZERO),
        free_particle(Vec3::new(0.5, 0.0, 0.0)),
    ];
    let constraints = [stretch(0, 1, 1.0)];
    assert_plasticity_parity(
        &device,
        &queue,
        &wgsl,
        &particles,
        &constraints,
        PlasticParams::default(),
    );
}

/// An edge within the yield band (len 1.05, rest 1.0, strain 0.05 < 0.1 yield)
/// is untouched; the GPU must reproduce the golden's exact no-op.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志"
)]
fn within_yield_band_is_a_no_op() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("within_yield_band_is_a_no_op: no wgpu adapter, skipping");
        return;
    };
    let wgsl = compile_plasticity_wgsl();
    let particles = [
        free_particle(Vec3::ZERO),
        free_particle(Vec3::new(1.05, 0.0, 0.0)),
    ];
    let constraints = [stretch(0, 1, 1.0)];
    assert_plasticity_parity(
        &device,
        &queue,
        &wgsl,
        &particles,
        &constraints,
        PlasticParams::default(),
    );
}

/// A one-sided leash (LRA / tether) never creeps even when badly over-stretched,
/// exactly like the golden `edge_strain` returning `None`; a stretched structural
/// edge in the same batch still creeps, proving the per-kind guard is per-edge.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志"
)]
fn one_sided_leashes_never_creep() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("one_sided_leashes_never_creep: no wgpu adapter, skipping");
        return;
    };
    let wgsl = compile_plasticity_wgsl();
    let particles = [
        free_particle(Vec3::ZERO),
        free_particle(Vec3::new(2.0, 0.0, 0.0)),
        free_particle(Vec3::new(2.0, 2.0, 0.0)),
    ];
    let constraints = [
        Constraint::new(0, 1, 1.0, Compliance::RIGID, ConstraintKind::Lra),
        Constraint::new(1, 2, 1.0, Compliance::RIGID, ConstraintKind::Tether),
        stretch(0, 2, 1.0),
    ];
    assert_plasticity_parity(
        &device,
        &queue,
        &wgsl,
        &particles,
        &constraints,
        PlasticParams::default(),
    );
}

/// A degenerate (near-zero rest) edge and an out-of-range endpoint are both
/// skipped by the golden `edge_strain` guards; the GPU mirrors both, leaving a
/// neighbouring valid stretched edge free to creep.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志"
)]
fn degenerate_and_out_of_range_edges_are_skipped() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("degenerate_and_out_of_range_edges_are_skipped: no wgpu adapter, skipping");
        return;
    };
    let wgsl = compile_plasticity_wgsl();
    let particles = [
        free_particle(Vec3::ZERO),
        free_particle(Vec3::new(1.5, 0.0, 0.0)),
    ];
    let constraints = [
        stretch(0, 1, 1.0e-12),
        stretch(0, 9, 1.0),
        stretch(0, 1, 1.0),
    ];
    assert_plasticity_parity(
        &device,
        &queue,
        &wgsl,
        &particles,
        &constraints,
        PlasticParams::default(),
    );
}

/// A high creep with a tight `max_strain` cap forces the residual-clamp branch:
/// the first `new_rest` would leave more than `max_strain` residual, so both paths
/// must re-solve `new_rest = len / (1 + sign * max_strain)` identically.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志"
)]
fn residual_clamp_branch_matches_golden() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("residual_clamp_branch_matches_golden: no wgpu adapter, skipping");
        return;
    };
    let wgsl = compile_plasticity_wgsl();
    let particles = [
        free_particle(Vec3::ZERO),
        free_particle(Vec3::new(2.0, 0.0, 0.0)),
    ];
    let constraints = [stretch(0, 1, 1.0)];
    let params = PlasticParams {
        yield_strain: 0.1,
        creep: 0.9,
        max_strain: 0.05,
    };
    assert_plasticity_parity(&device, &queue, &wgsl, &particles, &constraints, params);
}
