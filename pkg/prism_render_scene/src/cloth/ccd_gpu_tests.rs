//! Real-machine GPU coverage of the per-particle continuous-collision (CCD)
//! sweep in `cloth_ccd.wesl`: `cloth_resolve_ccd`.
//!
//! CPU golden: `prism_render_architecture::cloth::ccd::resolve_ccd`, the
//! tunnelling guard that sweeps every free particle's segment
//! `prev -> curr` against each analytic body collider, solves the earliest
//! closed-form time of impact (TOI), snaps the particle to the collider surface
//! plus a skin offset, reflects its normal velocity by a restitution coefficient
//! and damps the tangential slide with position-level Coulomb friction (Macklin
//! et al. 2014). Until now the shader was only proven to compile under `naga`
//! (via `cloth::shader_tests`); this module dispatches the kernel on a real
//! Metal device, reads back both the mutated `position` and `velocity`
//! buffers, and asserts value-for-value parity with the golden.
//!
//! ## Why these cases match the golden within float32 rounding
//!
//! The sweep is embarrassingly parallel: every particle only ever reads its own
//! `prev` / `curr` position and writes only its own position and
//! velocity, and the collider set is read-only. The golden visits particles in
//! index order purely for determinism, so the one-invocation-per-particle
//! dispatch reproduces the identical float32 arithmetic regardless of order.
//! Each invocation walks the colliders in slice order with a strict
//! `t < best` tie-break, exactly mirroring the golden's inner loop, and the
//! TOI solvers, the `normalize_or_zero` `1.0 / sqrt(len_sq)` form and
//! the friction cone are all transcribed line-for-line, so the only slack is a
//! few ULPs of fused-multiply-add contraction, well inside `PARITY_EPS`.
//!
//! Device acquisition is best-effort: a headless host with no wgpu adapter
//! returns `None` from `try_compute_device`, so each test prints a skip
//! note and passes, keeping the suite green on any machine.

use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor, MapMode,
    PipelineCompilationOptions, PipelineLayoutDescriptor, PollType, ShaderModuleDescriptor,
    ShaderSource, ShaderStages,
};

use prism_render_architecture::cloth::ccd::{resolve_ccd, CcdParams};
use prism_render_architecture::cloth::collision::BodyCollider;
use prism_render_architecture::cloth::{ClothParticle, Vec3};

use super::abi::GpuClothCcdSweepParams;
use super::gpu_test_support::{compile_ccd_wgsl, find_entry_point, try_compute_device, PARITY_EPS};
use super::pack::pack_colliders;

/// Builds a free particle (unit inverse mass, zero initial velocity) at
/// `position`.
fn free_particle(position: Vec3) -> ClothParticle {
    ClothParticle::new(position, 1.0)
}

/// Host upload layout for `ccd_positions`: xyz = position, w = inverse
/// mass. The kernel reads w only to skip pinned particles (w <= 0).
fn upload_positions(particles: &[ClothParticle]) -> Vec<[f32; 4]> {
    particles
        .iter()
        .map(|p| [p.position.x, p.position.y, p.position.z, p.inverse_mass])
        .collect()
}

/// Host upload layout for `ccd_velocities`: xyz = velocity, w carried
/// through untouched by the kernel.
fn upload_velocities(particles: &[ClothParticle]) -> Vec<[f32; 4]> {
    particles
        .iter()
        .map(|p| [p.velocity.x, p.velocity.y, p.velocity.z, 0.0])
        .collect()
}

/// Host upload layout for `ccd_prev_positions`: xyz = frame-start position,
/// w ignored by the kernel.
fn upload_prev(prev: &[Vec3]) -> Vec<[f32; 4]> {
    prev.iter().map(|p| [p.x, p.y, p.z, 0.0]).collect()
}

/// The group-0 layout for the CCD kernel, matching @binding(0..4) in the WESL:
/// `ccd_positions` (rw), `ccd_velocities` (rw),
/// `ccd_prev_positions` (ro), `ccd_colliders` (ro) and
/// `ccd_params` (uniform).
fn ccd_bind_group_layout(device: &wgpu::Device) -> BindGroupLayout {
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
        label: Some("cloth_ccd_parity_group0"),
        entries: &[
            storage(0, false),
            storage(1, false),
            storage(2, true),
            storage(3, true),
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

/// Runs the CCD kernel on a real device and reads the mutated position and
/// velocity buffers back (xyz of each is the value under test).
#[expect(
    clippy::too_many_arguments,
    reason = "一次 dispatch 需要位置/速度/prev/碰撞体/uniform 五路输入，铺平参数比引入临时聚合体更贴近内核绑定"
)]
fn run_ccd_on_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    entry: &str,
    positions: &[[f32; 4]],
    velocities: &[[f32; 4]],
    prev: &[[f32; 4]],
    colliders: &[super::abi::GpuClothCollider],
    params: GpuClothCcdSweepParams,
) -> (Vec<[f32; 4]>, Vec<[f32; 4]>) {
    let layout = ccd_bind_group_layout(device);
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("cloth_ccd_parity_pipeline_layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("cloth_ccd_parity"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("cloth_resolve_ccd_pipeline"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some(entry),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });

    let positions_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_ccd_positions"),
        contents: bytemuck::cast_slice(positions),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });
    let velocities_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_ccd_velocities"),
        contents: bytemuck::cast_slice(velocities),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });
    let prev_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_ccd_prev_positions"),
        contents: bytemuck::cast_slice(prev),
        usage: BufferUsages::STORAGE,
    });
    let colliders_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_ccd_colliders"),
        contents: bytemuck::cast_slice(colliders),
        usage: BufferUsages::STORAGE,
    });
    let params_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_ccd_params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("cloth_ccd_parity_bind"),
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
                resource: prev_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: colliders_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let pos_bytes = size_of_val(positions) as u64;
    let vel_bytes = size_of_val(velocities) as u64;
    let pos_stage = device.create_buffer(&BufferDescriptor {
        label: Some("cloth_ccd_positions_stage"),
        size: pos_bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let vel_stage = device.create_buffer(&BufferDescriptor {
        label: Some("cloth_ccd_velocities_stage"),
        size: vel_bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("cloth_ccd_parity_encoder"),
    });
    let groups = (positions.len() as u32).div_ceil(64).max(1);
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("cloth_resolve_ccd_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&positions_buf, 0, &pos_stage, 0, pos_bytes);
    encoder.copy_buffer_to_buffer(&velocities_buf, 0, &vel_stage, 0, vel_bytes);
    queue.submit([encoder.finish()]);

    pos_stage.slice(..).map_async(MapMode::Read, |_| {});
    vel_stage.slice(..).map_async(MapMode::Read, |_| {});
    device
        .poll(PollType::wait_indefinitely())
        .expect("device poll should complete the submitted ccd work");

    let pos_view = pos_stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped position readback range should be available after poll");
    let out_pos: Vec<[f32; 4]> = bytemuck::cast_slice::<u8, [f32; 4]>(&pos_view).to_vec();
    drop(pos_view);
    pos_stage.unmap();

    let vel_view = vel_stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped velocity readback range should be available after poll");
    let out_vel: Vec<[f32; 4]> = bytemuck::cast_slice::<u8, [f32; 4]>(&vel_view).to_vec();
    drop(vel_view);
    vel_stage.unmap();

    (out_pos, out_vel)
}

/// Runs the CPU golden and the real-machine twin over the same swept fixture,
/// then asserts every particle's corrected position and velocity agree within
/// `PARITY_EPS`.
///
/// The scalars are passed already in range so the golden's internal
/// `sanitized` (skin >= 0, restitution clamped) and the shader's own
/// re-clamp both see identical values; `particles` and `prev` carry
/// equal length so the golden's `min` equals the dispatched count.
fn assert_ccd_parity(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    particles: &[ClothParticle],
    prev: &[Vec3],
    colliders: &[BodyCollider],
    params: CcdParams,
    dt: f32,
    friction: f32,
) {
    assert_eq!(
        particles.len(),
        prev.len(),
        "fixture particle / prev length mismatch"
    );

    let mut golden = particles.to_vec();
    resolve_ccd(&mut golden, prev, colliders, params, dt, friction);

    let gpu_params = GpuClothCcdSweepParams {
        skin: params.skin,
        restitution: params.restitution,
        dt,
        friction,
        collider_count: colliders.len() as u32,
        particle_count: particles.len() as u32,
        pad0: 0,
        pad1: 0,
    };

    let positions = upload_positions(particles);
    let velocities = upload_velocities(particles);
    let prev_rows = upload_prev(prev);
    let gpu_colliders = pack_colliders(colliders);
    let entry = find_entry_point(wgsl, "cloth_resolve_ccd");
    let (out_pos, out_vel) = run_ccd_on_gpu(
        device,
        queue,
        wgsl,
        &entry,
        &positions,
        &velocities,
        &prev_rows,
        &gpu_colliders,
        gpu_params,
    );

    assert_eq!(out_pos.len(), golden.len());
    assert_eq!(out_vel.len(), golden.len());
    for (i, cpu) in golden.iter().enumerate() {
        let gp = out_pos[i];
        assert!(
            (gp[0] - cpu.position.x).abs() <= PARITY_EPS
                && (gp[1] - cpu.position.y).abs() <= PARITY_EPS
                && (gp[2] - cpu.position.z).abs() <= PARITY_EPS,
            "particle {i} position: gpu {gp:?} vs cpu ({}, {}, {})",
            cpu.position.x,
            cpu.position.y,
            cpu.position.z,
        );
        let gv = out_vel[i];
        assert!(
            (gv[0] - cpu.velocity.x).abs() <= PARITY_EPS
                && (gv[1] - cpu.velocity.y).abs() <= PARITY_EPS
                && (gv[2] - cpu.velocity.z).abs() <= PARITY_EPS,
            "particle {i} velocity: gpu {gv:?} vs cpu ({}, {}, {})",
            cpu.velocity.x,
            cpu.velocity.y,
            cpu.velocity.z,
        );
    }
}

/// A plane at y >= 0 with restitution 0: a particle tunnelling from (0, 1, 0)
/// to (0, -5, 0) is snapped to the skin offset and its downward normal velocity
/// cancelled. The GPU must land on the golden's inelastic stop.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志"
)]
fn half_space_drop_is_inelastic_like_golden() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("half_space_drop_is_inelastic_like_golden: no wgpu adapter, skipping");
        return;
    };
    let wgsl = compile_ccd_wgsl();
    let particles = [free_particle(Vec3::new(0.0, -5.0, 0.0))];
    let prev = [Vec3::new(0.0, 1.0, 0.0)];
    let colliders = [BodyCollider::HalfSpace {
        normal: Vec3::new(0.0, 1.0, 0.0),
        offset: 0.0,
    }];
    let params = CcdParams {
        skin: 0.01,
        restitution: 0.0,
        enabled: true,
    };
    assert_ccd_parity(
        &device,
        &queue,
        &wgsl,
        &particles,
        &prev,
        &colliders,
        params,
        1.0 / 60.0,
        0.0,
    );
}

/// The same plane with restitution 1: the inbound normal velocity is mirrored
/// to a perfect bounce. This exercises the velocity write-back path, so the GPU
/// velocity buffer must match the golden's reflected value.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志"
)]
fn half_space_bounce_is_elastic_like_golden() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("half_space_bounce_is_elastic_like_golden: no wgpu adapter, skipping");
        return;
    };
    let wgsl = compile_ccd_wgsl();
    let particles = [free_particle(Vec3::new(0.0, -5.0, 0.0))];
    let prev = [Vec3::new(0.0, 1.0, 0.0)];
    let colliders = [BodyCollider::HalfSpace {
        normal: Vec3::new(0.0, 1.0, 0.0),
        offset: 0.0,
    }];
    let params = CcdParams {
        skin: 0.0,
        restitution: 1.0,
        enabled: true,
    };
    assert_ccd_parity(
        &device,
        &queue,
        &wgsl,
        &particles,
        &prev,
        &colliders,
        params,
        1.0 / 60.0,
        0.0,
    );
}

/// A particle sweeping (-2, 0, 0) -> (2, 0, 0) straight through a unit sphere at
/// the origin: the quadratic `sphere_toi` entry root and radial outward
/// normal drive the snap. The GPU must reproduce the golden's projected surface.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志"
)]
fn sphere_penetration_matches_golden() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("sphere_penetration_matches_golden: no wgpu adapter, skipping");
        return;
    };
    let wgsl = compile_ccd_wgsl();
    let particles = [free_particle(Vec3::new(2.0, 0.0, 0.0))];
    let prev = [Vec3::new(-2.0, 0.0, 0.0)];
    let colliders = [BodyCollider::Sphere {
        center: Vec3::ZERO,
        radius: 1.0,
    }];
    let params = CcdParams {
        skin: 0.02,
        restitution: 0.0,
        enabled: true,
    };
    assert_ccd_parity(
        &device,
        &queue,
        &wgsl,
        &particles,
        &prev,
        &colliders,
        params,
        1.0 / 60.0,
        0.0,
    );
}

/// A particle crossing the cylindrical side of a capsule (axis (0,0,0)->(0,0,4),
/// radius 1): the infinite-cylinder-slab TOI and segment-closest-point normal
/// drive the snap. The GPU must match the golden's capsule side contact.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志"
)]
fn capsule_side_matches_golden() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("capsule_side_matches_golden: no wgpu adapter, skipping");
        return;
    };
    let wgsl = compile_ccd_wgsl();
    let particles = [free_particle(Vec3::new(3.0, 0.0, 2.0))];
    let prev = [Vec3::new(-3.0, 0.0, 2.0)];
    let colliders = [BodyCollider::Capsule {
        p0: Vec3::new(0.0, 0.0, 0.0),
        p1: Vec3::new(0.0, 0.0, 4.0),
        radius: 1.0,
    }];
    let params = CcdParams {
        skin: 0.02,
        restitution: 0.0,
        enabled: true,
    };
    assert_ccd_parity(
        &device,
        &queue,
        &wgsl,
        &particles,
        &prev,
        &colliders,
        params,
        1.0 / 60.0,
        0.0,
    );
}

/// A particle diving onto a capsule end cap (sweep (0,0,7)->(0,0,3) beyond the
/// p1 end at z=4): the end-cap sphere TOI wins over the cylinder slab. The GPU
/// must match the golden's spherical end-cap contact.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志"
)]
fn capsule_end_cap_matches_golden() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("capsule_end_cap_matches_golden: no wgpu adapter, skipping");
        return;
    };
    let wgsl = compile_ccd_wgsl();
    let particles = [free_particle(Vec3::new(0.0, 0.0, 3.0))];
    let prev = [Vec3::new(0.0, 0.0, 7.0)];
    let colliders = [BodyCollider::Capsule {
        p0: Vec3::new(0.0, 0.0, 0.0),
        p1: Vec3::new(0.0, 0.0, 4.0),
        radius: 1.0,
    }];
    let params = CcdParams {
        skin: 0.02,
        restitution: 0.0,
        enabled: true,
    };
    assert_ccd_parity(
        &device,
        &queue,
        &wgsl,
        &particles,
        &prev,
        &colliders,
        params,
        1.0 / 60.0,
        0.0,
    );
}

/// A diagonal sweep (0,1,0)->(2,-1,0) onto the y >= 0 plane hits at t = 0.5 with
/// a unit tangential slide along +X; Coulomb friction (mu = 0.5) shrinks that
/// slide to 0.5. This exercises the dynamic friction cone, so the GPU corrected
/// position must match the golden's friction-damped landing.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志"
)]
fn diagonal_plane_friction_matches_golden() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("diagonal_plane_friction_matches_golden: no wgpu adapter, skipping");
        return;
    };
    let wgsl = compile_ccd_wgsl();
    let particles = [free_particle(Vec3::new(2.0, -1.0, 0.0))];
    let prev = [Vec3::new(0.0, 1.0, 0.0)];
    let colliders = [BodyCollider::HalfSpace {
        normal: Vec3::new(0.0, 1.0, 0.0),
        offset: 0.0,
    }];
    let params = CcdParams {
        skin: 0.0,
        restitution: 0.0,
        enabled: true,
    };
    assert_ccd_parity(
        &device,
        &queue,
        &wgsl,
        &particles,
        &prev,
        &colliders,
        params,
        1.0 / 60.0,
        0.5,
    );
}
