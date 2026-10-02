//! `cloth_collision.wesl` 的 `cloth_backstop` 内核**真机 GPU** parity 覆盖。
//!
//! `ClothKernel::Backstop` 把每个自由粒子按其配对的 painted-backstop 平面回拉：当粒子
//! 沿 `-normal` 沉到 `origin` 后方超过 `distance` 时，被推回到限制平面上；否则原样保留。
//! 这是「painted backstop」——防止服装塌陷进身体、同时允许向外鼓起——的核心一段。
//! 此前它只被 `shader_tests` 证明「能被 `naga` 编译」，从未在真实设备上 dispatch 过；
//! 而 [`super::pack::pack_backstops`] 这条从作者态 [`Backstop`] 到设备记录
//! [`GpuClothBackstop`](super::abi::GpuClothBackstop) 的 host 打包桥，也需要一条端到端
//! 的真机对拍来证明字段映射与内核读取逐比特一致。本模块闭合这两条缺口。
//!
//! 做法忠实复用权威路径而非重写算法：
//! - **打包**走生产用的 [`super::pack::pack_backstops`]，把作者态平面转成设备记录，
//!   与 prepare 阶段上传的字节完全一致。
//! - **黄金**来自架构层
//!   [`resolve_backstops`](prism_render_architecture::cloth::collision::resolve_backstops)：
//!   `backstops[i]` 约束 `particles[i]`，pinned 粒子跳过，遍历两者较短长度，空集合 no-op。
//!
//! ## 为什么是 bit-parity（`float32` 舍入内）
//!
//! backstop 是一段纯逐粒子回拉：每个粒子的结果只依赖自己的位置与自己的只读平面记录，
//! 粒子间零共享 ⇒ GPU 的并行 invocation 与 CPU 的顺序遍历产生**完全相同**的写集。
//! 唯一的自由度是平面法线归一化里 CPU 的 `normalize_or_zero`（`1.0 / sqrt`）与 WESL 的
//! `inverseSqrt`（多数 GPU 是原生 `rsqrt`，可能差几个 ULP），[`PARITY_EPS`] 只吸收这一项。
//!
//! ## 配对长度即 dispatch 宽度
//!
//! CPU `resolve_backstops` 用 `zip` 遍历 `min(particles, backstops)`：backstops 短于
//! particles 时，尾部粒子不受约束。真机侧把 dispatch 的 `particle_count` 设为这个配对
//! 长度（[`replay_backstop_on_gpu`] 里的 `min`），于是尾部粒子根本不被 dispatch、原样
//! 返回——与 CPU 的短切片语义逐比特一致，也顺带把空 backstops（`particle_count == 0`，
//! 彻底 no-op）覆盖进来。
//!
//! 取设备是尽力而为：无 `wgpu` adapter 的无头机上 [`try_compute_device`] 返回
//! `None`，测试打印跳过提示而非失败，让套件在任何机器上保持绿。

use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor, MapMode,
    PipelineCompilationOptions, PipelineLayoutDescriptor, PollType, ShaderModuleDescriptor,
    ShaderSource, ShaderStages,
};

use prism_render_architecture::cloth::collision::{resolve_backstops, Backstop};
use prism_render_architecture::cloth::{ClothParticle, Vec3};

use super::abi::{GpuClothBackstop, GpuClothBackstopParams};
use super::gpu_test_support::{
    compile_collision_wgsl, find_entry_point, storage_from_slice, try_compute_device, PARITY_EPS,
};
use super::pack::pack_backstops;

/// `cloth_backstop` 的 group-0 布局：positions（rw storage）、backstops（ro storage）、
/// params（uniform）。与 WESL 的 `@binding(0..2)` 一一对应。
fn backstop_bind_group_layout(device: &wgpu::Device) -> BindGroupLayout {
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
        label: Some("cloth_backstop_parity_group0"),
        entries: &[
            storage(0, false),
            storage(1, true),
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

/// 在真机上跑一遍 `cloth_backstop`，读回回拉后的 positions（`xyzw`）。
///
/// `particle_count` 取 `min(positions, backstops)`，忠实镜像 CPU `resolve_backstops`
/// 的 `zip` 配对宽度：尾部无配对记录的粒子不被 dispatch，原样返回；空 backstops ⇒
/// `particle_count == 0` ⇒ 彻底 no-op。
fn replay_backstop_on_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    entry: &str,
    positions: &[[f32; 4]],
    backstops: &[GpuClothBackstop],
) -> Vec<[f32; 4]> {
    let paired = positions.len().min(backstops.len());
    let vec4_bytes = size_of_val(positions) as u64;

    let layout = backstop_bind_group_layout(device);
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("cloth_backstop_parity_layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("cloth_backstop_parity"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("cloth_backstop_parity_pipeline"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some(entry),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });

    let positions_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_backstop_positions"),
        contents: bytemuck::cast_slice(positions),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });
    let backstops_buf = storage_from_slice(
        device,
        "cloth_backstop_records",
        backstops,
        GpuClothBackstop::default(),
    );
    let params = GpuClothBackstopParams {
        particle_count: paired as u32,
        _pad: [0, 0, 0],
    };
    let params_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_backstop_params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("cloth_backstop_parity_bind"),
        layout: &layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: positions_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: backstops_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let stage = device.create_buffer(&BufferDescriptor {
        label: Some("cloth_backstop_positions_stage"),
        size: vec4_bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("cloth_backstop_parity_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("cloth_backstop_parity_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        // 每 invocation 一个配对粒子，@workgroup_size(64) ⇒ ceil(paired / 64) 组。
        let groups = paired.div_ceil(64).max(1) as u32;
        pass.dispatch_workgroups(groups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&positions_buf, 0, &stage, 0, vec4_bytes);
    queue.submit([encoder.finish()]);

    stage.slice(..).map_async(MapMode::Read, |_| {});
    device
        .poll(PollType::wait_indefinitely())
        .expect("device poll should complete the submitted work");
    let view = stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped positions readback range should be available after poll");
    let out: Vec<[f32; 4]> = bytemuck::cast_slice::<u8, [f32; 4]>(&view).to_vec();
    drop(view);
    stage.unmap();
    out
}

/// 对一组粒子与作者态 backstop 平面，同时跑 CPU 黄金 [`resolve_backstops`] 与真机
/// `cloth_backstop`，断言逐粒子 `xyz` 落在 [`PARITY_EPS`] 内、`w`（inverse mass）不被
/// 内核改动。
fn assert_backstop_parity(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    entry: &str,
    particles: &[ClothParticle],
    backstops: &[Backstop],
) {
    // --- CPU 黄金：在副本上原地回拉 ---
    let mut golden = particles.to_vec();
    resolve_backstops(&mut golden, backstops);

    // --- host 上传口径：positions.w = inverse mass（`<= 0` 即 pinned）---
    let positions: Vec<[f32; 4]> = particles
        .iter()
        .map(|p| [p.position.x, p.position.y, p.position.z, p.inverse_mass])
        .collect();
    let packed = pack_backstops(backstops);
    let out = replay_backstop_on_gpu(device, queue, wgsl, entry, &positions, &packed);

    assert_eq!(out.len(), golden.len());
    for (i, (gpu, cpu)) in out.iter().zip(golden.iter()).enumerate() {
        let expected = cpu.position;
        assert!(
            (gpu[0] - expected.x).abs() <= PARITY_EPS
                && (gpu[1] - expected.y).abs() <= PARITY_EPS
                && (gpu[2] - expected.z).abs() <= PARITY_EPS,
            "particle {i}: gpu [{}, {}, {}] vs cpu [{}, {}, {}]",
            gpu[0],
            gpu[1],
            gpu[2],
            expected.x,
            expected.y,
            expected.z,
        );
        // inverse mass 是内核直通字段，绝不能被回拉改动。
        assert!(
            (gpu[3] - particles[i].inverse_mass).abs() <= f32::EPSILON,
            "particle {i}: inverse mass mutated {} -> {}",
            particles[i].inverse_mass,
            gpu[3],
        );
    }
}

/// 建一个自由粒子（`inverse_mass = 1`）。
fn free(x: f32, y: f32, z: f32) -> ClothParticle {
    ClothParticle::new(Vec3::new(x, y, z), 1.0)
}

/// 建一个 pinned 粒子（`inverse_mass = 0`，回拉永不移动它）。
fn pinned(x: f32, y: f32, z: f32) -> ClothParticle {
    ClothParticle::pinned(Vec3::new(x, y, z))
}

/// 建一个 backstop 平面。
fn plane(origin: [f32; 3], normal: [f32; 3], distance: f32) -> Backstop {
    Backstop {
        origin: Vec3::new(origin[0], origin[1], origin[2]),
        normal: Vec3::new(normal[0], normal[1], normal[2]),
        distance,
    }
}

/// 越界点被推回限制平面、可行侧点不动、pinned 越界点不动——全部与 CPU 逐比特对齐。
#[test]
#[expect(
    clippy::print_stderr,
    reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志"
)]
fn backstop_pushes_behind_particles_onto_limit_plane() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!(
            "backstop_pushes_behind_particles_onto_limit_plane: no wgpu adapter, skipping on-device parity"
        );
        return;
    };
    let wgsl = compile_collision_wgsl();
    let entry = find_entry_point(&wgsl, "cloth_backstop");

    // 平面法线 +Y，origin 在原点，distance 0.3：可行域 s >= -0.3（s = y）。
    let backstops = [
        plane([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.3),
        plane([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.3),
        plane([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.3),
        plane([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.3),
    ];
    let particles = [
        free(0.2, -0.9, 0.1),   // 沉到 -0.9 < -0.3 → 推回到 y=-0.3
        free(-0.3, 0.7, 0.4),   // 可行侧（y>-0.3）→ 不动
        free(1.0, -0.3, -1.0),  // 恰在限制平面 → 不动
        pinned(0.5, -5.0, 0.5), // pinned 越界 → 永不移动
    ];
    assert_backstop_parity(&device, &queue, &wgsl, &entry, &particles, &backstops);
}

/// 非单位法线（修正按法线平方长归一，几何正确）与斜置平面下的回拉与 CPU 逐比特对齐。
#[test]
#[expect(
    clippy::print_stderr,
    reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志"
)]
fn backstop_handles_non_unit_and_oblique_normals() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!(
            "backstop_handles_non_unit_and_oblique_normals: no wgpu adapter, skipping on-device parity"
        );
        return;
    };
    let wgsl = compile_collision_wgsl();
    let entry = find_entry_point(&wgsl, "cloth_backstop");

    let backstops = [
        // 非单位法线（长度 3），origin 偏置。
        plane([0.0, 1.0, 0.0], [0.0, 3.0, 0.0], 0.5),
        // 斜置非单位法线。
        plane([1.0, 0.0, -1.0], [1.0, 1.0, 0.0], 0.25),
    ];
    let particles = [
        free(0.2, -2.0, 0.1), // 深在平面后 → 沿单位法线推回限制面
        free(2.0, 2.0, -1.0), // 斜面另一侧的一般点
    ];
    assert_backstop_parity(&device, &queue, &wgsl, &entry, &particles, &backstops);
}

/// 零法线平面无定义 ⇒ 惰性：任何点都不动，与 CPU 的 `normalize_or_zero` 早退一致。
#[test]
#[expect(
    clippy::print_stderr,
    reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志"
)]
fn backstop_zero_normal_is_inert_on_gpu() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!(
            "backstop_zero_normal_is_inert_on_gpu: no wgpu adapter, skipping on-device parity"
        );
        return;
    };
    let wgsl = compile_collision_wgsl();
    let entry = find_entry_point(&wgsl, "cloth_backstop");

    let backstops = [
        plane([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.5),
        plane([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], -1.0),
    ];
    let particles = [free(0.0, -10.0, 0.0), free(3.0, 2.0, -4.0)];
    assert_backstop_parity(&device, &queue, &wgsl, &entry, &particles, &backstops);
}

/// backstops 短于 particles：尾部无配对记录的粒子不受约束、原样返回，与 CPU 的
/// `zip` 短切片语义逐比特一致。
#[test]
#[expect(
    clippy::print_stderr,
    reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志"
)]
fn backstop_short_slice_leaves_trailing_particles_free() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!(
            "backstop_short_slice_leaves_trailing_particles_free: no wgpu adapter, skipping on-device parity"
        );
        return;
    };
    let wgsl = compile_collision_wgsl();
    let entry = find_entry_point(&wgsl, "cloth_backstop");

    // 只有一个 backstop 约束首粒子；后两个粒子无配对 → 原样返回。
    let backstops = [plane([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.3)];
    let particles = [
        free(0.0, -0.9, 0.0), // 被推回 y=-0.3
        free(0.0, -5.0, 0.0), // 无配对 → 不动（尽管远在后方）
        free(1.0, -9.0, 1.0), // 无配对 → 不动
    ];
    assert_backstop_parity(&device, &queue, &wgsl, &entry, &particles, &backstops);
}

/// 空 backstops 集合是彻底 no-op：`particle_count == 0` ⇒ 无 dispatch，所有粒子原样
/// 返回，与 CPU 的空集合早退一致。
#[test]
#[expect(
    clippy::print_stderr,
    reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志"
)]
fn empty_backstop_set_is_noop_on_gpu() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("empty_backstop_set_is_noop_on_gpu: no wgpu adapter, skipping on-device parity");
        return;
    };
    let wgsl = compile_collision_wgsl();
    let entry = find_entry_point(&wgsl, "cloth_backstop");

    let backstops: [Backstop; 0] = [];
    let particles = [
        free(0.3, -4.0, 0.5),
        pinned(1.0, 1.0, 1.0),
        free(-2.0, 0.0, 0.0),
    ];
    assert_backstop_parity(&device, &queue, &wgsl, &entry, &particles, &backstops);
}
