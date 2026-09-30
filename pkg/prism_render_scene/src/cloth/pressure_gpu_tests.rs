//! Real-machine GPU coverage of the two own-slot pressure (volume) kernels in
//! `cloth_pressure.wesl`: `cloth_pressure_solve` and `cloth_pressure_apply`.
//!
//! CPU golden: `prism_render_architecture::cloth::pressure::project_pressure`, the
//! single XPBD volume projection that keeps a closed, outward-wound triangle
//! mesh at a target enclosed volume by pushing its particles along the
//! accumulated per-vertex volume gradient (design section 6). Until now the
//! shader was only proven to compile under `naga` (via `cloth::shader_tests`);
//! this module dispatches the full solve -> apply chain on a real Metal device,
//! reads the positions back, and asserts value-for-value parity with the golden.
//!
//! ## Why these cases match the golden within float32 rounding
//!
//! The pressure constraint is a global reduction: every triangle scatters into
//! three shared vertices and a single scalar `d_lambda` couples the whole shell.
//! `cloth_pressure_solve` reproduces that reduction as ONE invocation walking the
//! triangles and vertices in the golden's exact order, so the volume, the
//! per-vertex gradient, the denominator and `d_lambda` are the same float32
//! arithmetic on both paths; the parallel apply pass then adds
//! `w * d_lambda * grad_i` to each free particle independently (order-free). The
//! only slack is a few ULPs of fused-multiply-add contraction, well inside
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

use prism_render_architecture::cloth::pressure::{project_pressure, PressureParams};
use prism_render_architecture::cloth::{ClothParticle, Compliance, Vec3};

use super::abi::GpuClothPressureParams;
use super::gpu_test_support::{compile_pressure_wgsl, find_entry_point, try_compute_device, PARITY_EPS};

/// The eight corners of the axis-aligned unit cube `[0, 1]^3`, matching the
/// golden's own pressure fixture.
fn unit_cube_positions() -> Vec<Vec3> {
    vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(1.0, 0.0, 1.0),
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(0.0, 1.0, 1.0),
    ]
}

/// The twelve outward-wound triangles of the unit cube, matching the golden.
fn unit_cube_triangles() -> Vec<[u32; 3]> {
    vec![
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [3, 6, 2],
        [3, 7, 6],
        [0, 4, 7],
        [0, 7, 3],
        [1, 2, 6],
        [1, 6, 5],
    ]
}

/// Builds a movable-particle array (unit inverse mass) from positions.
fn free_particles(positions: &[Vec3]) -> Vec<ClothParticle> {
    positions.iter().map(|p| ClothParticle::new(*p, 1.0)).collect()
}

/// host upload layout for `pressure_positions`: xyz = position, w = inverse
/// mass (<= 0 = pinned), matching the shader's convention.
fn upload_positions(particles: &[ClothParticle]) -> Vec<[f32; 4]> {
    particles
        .iter()
        .map(|p| [p.position.x, p.position.y, p.position.z, p.inverse_mass])
        .collect()
}

/// host upload layout for `pressure_triangles`: xyz = vertex indices, w unused.
fn upload_triangles(triangles: &[[u32; 3]]) -> Vec<[u32; 4]> {
    triangles.iter().map(|t| [t[0], t[1], t[2], 0]).collect()
}

/// The group-0 layout for the pressure kernels: `pressure_positions` (rw),
/// `pressure_triangles` (ro), `pressure_gradients` (rw), `pressure_lambda` (rw)
/// and `pressure_params` (uniform), matching @binding(0..4) in the WESL.
fn pressure_bind_group_layout(device: &wgpu::Device) -> BindGroupLayout {
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
        label: Some("cloth_pressure_parity_group0"),
        entries: &[
            storage(0, false),
            storage(1, true),
            storage(2, false),
            storage(3, false),
            BindGroupLayoutEntry {
                binding: 4,
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

/// The two pressure entry-point symbol names as they appear in the compiled
/// WESL output (located by substring, since WESL may prefix module-local names).
struct PressureEntryPoints {
    solve: String,
    apply: String,
}

fn pressure_entry_points(wgsl: &str) -> PressureEntryPoints {
    PressureEntryPoints {
        solve: find_entry_point(wgsl, "cloth_pressure_solve"),
        apply: find_entry_point(wgsl, "cloth_pressure_apply"),
    }
}

/// Runs solve -> apply on a real device and reads the final positions back
/// (xyzw each; position w = inverse mass).
fn run_pressure_on_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    entries: &PressureEntryPoints,
    positions: &[[f32; 4]],
    triangles: &[[u32; 4]],
    params: GpuClothPressureParams,
) -> Vec<[f32; 4]> {
    let count = positions.len();
    let layout = pressure_bind_group_layout(device);
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("cloth_pressure_parity_pipeline_layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("cloth_pressure_parity"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });
    let make_pipeline = |label: &str, entry: &str| {
        device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some(entry),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        })
    };
    let solve_pipeline = make_pipeline("cloth_pressure_solve_pipeline", &entries.solve);
    let apply_pipeline = make_pipeline("cloth_pressure_apply_pipeline", &entries.apply);

    let gradients_init = vec![[0.0f32; 4]; count];
    let lambda_init = [0.0f32];

    let positions_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_pressure_positions"),
        contents: bytemuck::cast_slice(positions),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });
    let triangles_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_pressure_triangles"),
        contents: bytemuck::cast_slice(triangles),
        usage: BufferUsages::STORAGE,
    });
    let gradients_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_pressure_gradients"),
        contents: bytemuck::cast_slice(&gradients_init),
        usage: BufferUsages::STORAGE,
    });
    let lambda_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_pressure_lambda"),
        contents: bytemuck::cast_slice(&lambda_init),
        usage: BufferUsages::STORAGE,
    });
    let params_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_pressure_params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("cloth_pressure_parity_bind"),
        layout: &layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: positions_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: triangles_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: gradients_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: lambda_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let stage_bytes = size_of_val(positions) as u64;
    let pos_stage = device.create_buffer(&BufferDescriptor {
        label: Some("cloth_pressure_positions_stage"),
        size: stage_bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("cloth_pressure_parity_encoder"),
    });
    let apply_groups = (count as u32).div_ceil(64).max(1);
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("cloth_pressure_solve_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&solve_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        // Single-invocation reduction: one workgroup, gid.x == 0 does the work.
        pass.dispatch_workgroups(1, 1, 1);
    }
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("cloth_pressure_apply_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&apply_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(apply_groups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&positions_buf, 0, &pos_stage, 0, stage_bytes);
    queue.submit([encoder.finish()]);

    pos_stage.slice(..).map_async(MapMode::Read, |_| {});
    device
        .poll(PollType::wait_indefinitely())
        .expect("device poll should complete the submitted pressure work");

    let pos_view = pos_stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped position readback range should be available after poll");
    let positions_out: Vec<[f32; 4]> = bytemuck::cast_slice::<u8, [f32; 4]>(&pos_view).to_vec();
    drop(pos_view);
    pos_stage.unmap();

    positions_out
}

/// Runs the CPU golden and the real-machine two-pass twin over the same
/// closed-mesh fixture, then asserts every particle's position agrees within
/// `PARITY_EPS` and that the inverse mass is untouched.
///
/// `params` is sanitised once up front so the host uniform and the golden's
/// internal sanitise (`project_pressure`'s first line) see identical scalars, and
/// the host derives `target_volume` / `alpha_tilde` exactly as the golden does.
fn assert_pressure_parity(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    particles: &[ClothParticle],
    triangles: &[[u32; 3]],
    params: PressureParams,
    dt: f32,
) {
    let params = params.sanitized();

    let mut golden = particles.to_vec();
    project_pressure(&mut golden, triangles, params, dt);

    let gpu_params = GpuClothPressureParams {
        particle_count: particles.len() as u32,
        triangle_count: triangles.len() as u32,
        target_volume: params.target_volume(),
        alpha_tilde: params.compliance.value() / (dt * dt),
    };

    let positions = upload_positions(particles);
    let tri_rows = upload_triangles(triangles);
    let entries = pressure_entry_points(wgsl);
    let pos_out = run_pressure_on_gpu(device, queue, wgsl, &entries, &positions, &tri_rows, gpu_params);

    assert_eq!(pos_out.len(), golden.len());
    for (i, cpu) in golden.iter().enumerate() {
        let gpu_pos = pos_out[i];
        assert!(
            (gpu_pos[0] - cpu.position.x).abs() <= PARITY_EPS
                && (gpu_pos[1] - cpu.position.y).abs() <= PARITY_EPS
                && (gpu_pos[2] - cpu.position.z).abs() <= PARITY_EPS,
            "particle {i} position: gpu [{}, {}, {}] vs cpu [{}, {}, {}]",
            gpu_pos[0],
            gpu_pos[1],
            gpu_pos[2],
            cpu.position.x,
            cpu.position.y,
            cpu.position.z,
        );
        assert!(
            (gpu_pos[3] - particles[i].inverse_mass).abs() <= f32::EPSILON,
            "particle {i}: inverse mass mutated {} -> {}",
            particles[i].inverse_mass,
            gpu_pos[3],
        );
    }
}

/// One inflation step (overpressure 2) drives every free corner outward along
/// its accumulated volume gradient; the GPU reduction must reproduce the
/// golden's per-vertex displacement value-for-value.
#[test]
#[expect(clippy::print_stderr, reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志")]
fn inflation_single_step_matches_golden() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("inflation_single_step_matches_golden: no wgpu adapter, skipping");
        return;
    };
    let wgsl = compile_pressure_wgsl();
    let particles = free_particles(&unit_cube_positions());
    let params = PressureParams::new(1.0, 2.0, Compliance::RIGID);
    assert_pressure_parity(&device, &queue, &wgsl, &particles, &unit_cube_triangles(), params, 1.0 / 60.0);
}

/// One deflation step (overpressure 0.5) pulls every free corner inward; the
/// sign of the volume error flips relative to inflation, exercising the other
/// branch of `d_lambda`.
#[test]
#[expect(clippy::print_stderr, reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志")]
fn deflation_single_step_matches_golden() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("deflation_single_step_matches_golden: no wgpu adapter, skipping");
        return;
    };
    let wgsl = compile_pressure_wgsl();
    let particles = free_particles(&unit_cube_positions());
    let params = PressureParams::new(1.0, 0.5, Compliance::RIGID);
    assert_pressure_parity(&device, &queue, &wgsl, &particles, &unit_cube_triangles(), params, 1.0 / 60.0);
}

/// A non-zero compliance softens the response: `alpha_tilde = compliance / dt^2`
/// enters the denominator, so the GPU must fold the same softened `d_lambda` the
/// golden computes.
#[test]
#[expect(clippy::print_stderr, reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志")]
fn compliant_pressure_matches_golden() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("compliant_pressure_matches_golden: no wgpu adapter, skipping");
        return;
    };
    let wgsl = compile_pressure_wgsl();
    let particles = free_particles(&unit_cube_positions());
    let params = PressureParams::new(1.0, 2.0, Compliance(0.01));
    assert_pressure_parity(&device, &queue, &wgsl, &particles, &unit_cube_triangles(), params, 1.0 / 60.0);
}

/// Pinned corners (inverse mass 0) must never move and must never contribute to
/// the denominator, exactly like the golden's zero-weight guard; the free
/// corners still track the golden displacement.
#[test]
#[expect(clippy::print_stderr, reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志")]
fn pinned_corners_never_move() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("pinned_corners_never_move: no wgpu adapter, skipping");
        return;
    };
    let wgsl = compile_pressure_wgsl();
    let positions = unit_cube_positions();
    let mut particles = free_particles(&positions);
    particles[0] = ClothParticle::pinned(positions[0]);
    particles[1] = ClothParticle::pinned(positions[1]);
    let params = PressureParams::new(1.0, 3.0, Compliance::RIGID);
    assert_pressure_parity(&device, &queue, &wgsl, &particles, &unit_cube_triangles(), params, 1.0 / 60.0);
}

/// A triangle indexing past the particle array is skipped by both paths (the
/// solve pass's `i >= count` guard mirrors the golden's range check), so the
/// junk face perturbs neither the volume nor the gradient.
#[test]
#[expect(clippy::print_stderr, reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志")]
fn out_of_range_triangle_is_skipped() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("out_of_range_triangle_is_skipped: no wgpu adapter, skipping");
        return;
    };
    let wgsl = compile_pressure_wgsl();
    let particles = free_particles(&unit_cube_positions());
    let mut triangles = unit_cube_triangles();
    triangles.push([99, 100, 101]);
    let params = PressureParams::new(1.0, 2.0, Compliance::RIGID);
    assert_pressure_parity(&device, &queue, &wgsl, &particles, &triangles, params, 1.0 / 60.0);
}

/// A shell already at its target volume (overpressure 1, rest volume 1) has zero
/// volume error, so `d_lambda` is zero and no particle moves; the GPU must land
/// on the golden's exact no-op.
#[test]
#[expect(clippy::print_stderr, reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志")]
fn at_target_volume_is_a_no_op() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("at_target_volume_is_a_no_op: no wgpu adapter, skipping");
        return;
    };
    let wgsl = compile_pressure_wgsl();
    let particles = free_particles(&unit_cube_positions());
    let params = PressureParams::new(1.0, 1.0, Compliance::RIGID);
    assert_pressure_parity(&device, &queue, &wgsl, &particles, &unit_cube_triangles(), params, 1.0 / 60.0);
}
