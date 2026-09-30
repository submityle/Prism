//! Real-machine GPU coverage of the sleep kinetic-indicator reduction in
//! `cloth_sleep.wesl` (the `sleep_max_kinetic` entry).
//!
//! CPU golden: `prism_render_architecture::cloth::sleep::max_kinetic_indicator`
//! -- it walks every particle, skips the pinned ones (`is_pinned` is
//! `inverse_mass <= 0.0`), and keeps the largest velocity `length_squared` it
//! sees, seeded at `0.0`. Until now the reduction lived only on the CPU; this
//! module dispatches the workgroup tree reduction on a real Metal device, reads
//! the single folded scalar back, and asserts it agrees with the golden.
//!
//! ## Why the reduction matches the golden
//!
//! `max_kinetic_indicator` is a pure max over per-particle scalars, so the answer
//! is independent of the fold order: the workgroup tree and the cross-workgroup
//! atomic reach the identical maximum the sequential golden does. Each lane
//! recomputes the squared speed as `x*x + y*y + z*z`, matching `Vec3::dot(self,
//! self)` term for term, so the winning scalar is bit-identical to the golden and
//! the seed `0.0` floor is shared. The running max travels as the `u32` bit
//! pattern of the `f32` partial: every contributed value is non-negative, and
//! non-negative IEEE-754 floats keep their order under `bitcast`, so integer
//! `atomicMax` selects the same maximum. The host seeds the atomic with
//! `bitcast(0.0)` and reinterprets the readback with `f32::from_bits`.
//!
//! Device acquisition is best-effort: a headless host with no `wgpu` adapter
//! returns `None` from `try_compute_device`, so each on-device test prints a skip
//! note and passes. A device-free compile test still guards that the shader
//! parses.

use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor, MapMode,
    PipelineCompilationOptions, PipelineLayoutDescriptor, PollType, ShaderModuleDescriptor,
    ShaderSource, ShaderStages,
};

use prism_render_architecture::cloth::sleep::max_kinetic_indicator;
use prism_render_architecture::cloth::{ClothParticle, Vec3};

use super::abi::GpuClothSleepParams;
use super::gpu_test_support::{
    compile_sleep_wgsl, find_entry_point, storage_from_slice, try_compute_device, PARITY_EPS,
};

/// A particle with the given `velocity` and `inverse_mass`, parked at the origin
/// (the reduction never reads position).
fn particle(velocity: Vec3, inverse_mass: f32) -> ClothParticle {
    let mut p = ClothParticle::new(Vec3::ZERO, inverse_mass);
    p.velocity = velocity;
    p
}

/// A free particle (unit inverse mass) moving at `velocity`.
fn free(velocity: Vec3) -> ClothParticle {
    particle(velocity, 1.0)
}

/// A pinned particle (zero inverse mass) moving at `velocity`; the golden skips
/// it no matter how fast it is.
fn pinned(velocity: Vec3) -> ClothParticle {
    particle(velocity, 0.0)
}

/// Host upload layout for the velocity buffer: `xyz` = velocity, `w` = inverse
/// mass (read only to skip pinned particles).
fn upload_velocities(particles: &[ClothParticle]) -> Vec<[f32; 4]> {
    particles
        .iter()
        .map(|p| [p.velocity.x, p.velocity.y, p.velocity.z, p.inverse_mass])
        .collect()
}

/// The group-0 layout shared by the reduction: a read-write atomic scalar, the
/// read-only velocity array, and the uniform scalars.
fn sleep_bind_group_layout(device: &wgpu::Device) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("cloth_sleep_parity_group0"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
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

/// Runs the reduction on a real device and folds the mutated atomic back to an
/// `f32` via `f32::from_bits`, matching the shader `bitcast`.
fn run_sleep_on_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    velocities: &[[f32; 4]],
    params: GpuClothSleepParams,
) -> f32 {
    let entry = find_entry_point(wgsl, "sleep_max_kinetic");
    let layout = sleep_bind_group_layout(device);
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("cloth_sleep_parity_pipeline_layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("cloth_sleep_parity"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("cloth_sleep_pipeline"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some(&entry),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });

    // Seed the atomic with bitcast(0.0) = 0, matching the golden max seed.
    let max_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_sleep_max"),
        contents: bytemuck::bytes_of(&0u32),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });
    let velocities_buf =
        storage_from_slice(device, "cloth_sleep_velocities", velocities, [0.0f32; 4]);
    let params_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_sleep_params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("cloth_sleep_parity_bind"),
        layout: &layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: max_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: velocities_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let stage = device.create_buffer(&BufferDescriptor {
        label: Some("cloth_sleep_max_stage"),
        size: size_of::<u32>() as u64,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("cloth_sleep_parity_encoder"),
    });
    let groups = params.particle_count.div_ceil(64).max(1);
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("cloth_sleep_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&max_buf, 0, &stage, 0, size_of::<u32>() as u64);
    queue.submit([encoder.finish()]);

    stage.slice(..).map_async(MapMode::Read, |_| {});
    device
        .poll(PollType::wait_indefinitely())
        .expect("device poll should complete the submitted sleep work");

    let view = stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped sleep readback range should be available after poll");
    let bits: u32 = bytemuck::cast_slice::<u8, u32>(&view)[0];
    drop(view);
    stage.unmap();
    f32::from_bits(bits)
}

/// Runs the CPU golden and the real-machine twin over the same particles, then
/// asserts the folded indicator agrees within a value-scaled `PARITY_EPS`.
fn assert_sleep_parity(device: &wgpu::Device, queue: &wgpu::Queue, wgsl: &str, particles: &[ClothParticle]) {
    let golden = max_kinetic_indicator(particles);
    let velocities = upload_velocities(particles);
    let params = GpuClothSleepParams {
        particle_count: particles.len() as u32,
        pad0: 0,
        pad1: 0,
        pad2: 0,
    };
    let gpu = run_sleep_on_gpu(device, queue, wgsl, &velocities, params);
    let tol = PARITY_EPS * (1.0 + golden.abs());
    assert!(
        (gpu - golden).abs() <= tol,
        "sleep indicator mismatch: gpu={gpu} golden={golden} tol={tol}",
    );
}

/// The shader parses even on a headless host: a device-free guard so a syntax
/// regression fails the build everywhere, not just on machines with a `wgpu`
/// adapter.
#[test]
fn sleep_wesl_compiles() {
    let wgsl = compile_sleep_wgsl();
    let entry = find_entry_point(&wgsl, "sleep_max_kinetic");
    assert!(entry.contains("sleep_max_kinetic"));
}

/// An empty batch folds to the `0.0` seed on both paths.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志"
)]
fn empty_batch_is_zero_like_golden() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("empty_batch_is_zero_like_golden: no wgpu adapter, skipping");
        return;
    };
    let wgsl = compile_sleep_wgsl();
    assert_sleep_parity(&device, &queue, &wgsl, &[]);
}

/// A single free particle folds to its own squared speed.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志"
)]
fn single_free_particle_matches_golden() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("single_free_particle_matches_golden: no wgpu adapter, skipping");
        return;
    };
    let wgsl = compile_sleep_wgsl();
    let particles = [free(Vec3::new(0.3, -0.4, 1.2))];
    assert_sleep_parity(&device, &queue, &wgsl, &particles);
}

/// A fast pinned particle is excluded, so the slower free particle wins -- the
/// per-particle pin guard must match the golden `is_pinned` skip.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志"
)]
fn pinned_are_excluded_like_golden() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("pinned_are_excluded_like_golden: no wgpu adapter, skipping");
        return;
    };
    let wgsl = compile_sleep_wgsl();
    let particles = [
        pinned(Vec3::new(50.0, 0.0, 0.0)),
        free(Vec3::new(1.5, 0.0, 0.0)),
        free(Vec3::new(0.5, 0.5, 0.5)),
    ];
    assert_sleep_parity(&device, &queue, &wgsl, &particles);
}

/// Every particle pinned folds to the `0.0` seed, exactly like the golden.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志"
)]
fn all_pinned_is_zero_like_golden() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("all_pinned_is_zero_like_golden: no wgpu adapter, skipping");
        return;
    };
    let wgsl = compile_sleep_wgsl();
    let particles = [
        pinned(Vec3::new(3.0, 2.0, 1.0)),
        pinned(Vec3::new(0.0, 9.0, 0.0)),
    ];
    assert_sleep_parity(&device, &queue, &wgsl, &particles);
}

/// A batch spanning several workgroups with the maximum buried in the middle:
/// exercises the intra-workgroup tree fold and the cross-workgroup atomic fold
/// together, plus a pinned outlier that must not win.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志"
)]
fn dense_batch_across_workgroups_matches_golden() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("dense_batch_across_workgroups_matches_golden: no wgpu adapter, skipping");
        return;
    };
    let wgsl = compile_sleep_wgsl();
    let mut particles = Vec::new();
    // 200 particles => four workgroups (64 lanes each, last partly out of range).
    for i in 0..200u32 {
        let f = i as f32;
        // A smooth ramp of modest speeds so squared magnitudes stay O(10).
        let vel = Vec3::new(0.01 * f, 0.005 * f, 0.002 * f);
        particles.push(free(vel));
    }
    // Bury the true maximum at index 137, comfortably above the ramp.
    particles[137] = free(Vec3::new(2.5, -1.5, 0.75));
    // A pinned outlier that is faster than everything but must be skipped.
    particles[42] = pinned(Vec3::new(99.0, 99.0, 99.0));
    assert_sleep_parity(&device, &queue, &wgsl, &particles);
}
