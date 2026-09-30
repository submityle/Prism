//! `cloth_collision.wesl` 的 `cloth_body_collision` 内核**真机 GPU** parity 覆盖。
//!
//! `ClothKernel::BodyCollision` 把每个自由粒子按数组顺序投影出每一个解析碰撞体
//! （sphere / capsule / half-space），是布料贴合骨架身体的核心一段。此前它只被
//! `shader_tests` 证明「能被 `naga` 编译」——从未在真实设备上 dispatch 过；而
//! [`super::pack::pack_colliders`] 这条从作者态 [`BodyCollider`] 到设备记录
//! [`GpuClothCollider`](super::abi::GpuClothCollider) 的 host 打包桥，也需要一条
//! 端到端的真机对拍来证明字段映射与内核读取逐比特一致。本模块闭合这两条缺口。
//!
//! 做法忠实复用权威路径而非重写算法：
//! - **打包**走生产用的 [`super::pack::pack_colliders`]，把作者态碰撞体转成设备
//!   记录，与 prepare 阶段上传的字节完全一致。
//! - **黄金**来自架构层
//!   [`resolve_body_collisions`](prism_render_architecture::cloth::collision::resolve_body_collisions)：
//!   逐粒子、逐碰撞体顺序投影，pinned 粒子跳过，空集合 no-op。
//!
//! ## 为什么是 bit-parity（`float32` 舍入内）
//!
//! body-collision 是一段纯投影：每个粒子的结果只依赖自己的位置与只读碰撞体切片，
//! 粒子间零共享 ⇒ GPU 的并行 invocation 与 CPU 的顺序遍历产生**完全相同**的写集，
//! 并行不改变任何一个粒子的算术。唯一的自由度是球面投影里 CPU 的 `1.0 / sqrt(x)`
//! 与 WESL 的 `inverseSqrt(x)`（多数 GPU 是原生 `rsqrt`，可能差几个 ULP），
//! [`PARITY_EPS`] 只吸收这一项。
//!
//! ## shipping 路径恒为 frictionless
//!
//! WESL `cloth_body_collision` 末步用净 correction 施一次 Coulomb 摩擦
//! （`cloth_damp_tangential_slip`），但场景层 [`GpuClothBodyParams`](super::abi::GpuClothBodyParams)
//! **没有 `friction` 字段**——host 恒把该 uniform 词置零 ⇒ `mu = 0` ⇒ 摩擦段整体
//! 早退（no-op）。这正是当前真实 shipping 路径，此时 GPU 与
//! `resolve_body_collisions`（零摩擦）逐比特对齐。故本模块覆盖 frictionless 路径；
//! 端到端接线 `friction`（补 abi + pack + extract + 对齐多碰撞体逐个施摩擦的 WESL
//! 结构）是更大的后续 feature，单列。
//!
//! 取设备是尽力而为：无 `wgpu` adapter 的无头机上 [`try_collision_device`] 返回
//! `None`，测试打印跳过提示而非失败，让套件在任何机器上保持绿，同时在有真实设备
//! （如 `Apple` `M` 系列 `GPU`）时跑满整条 dispatch。

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_platform::future::block_on;
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BackendOptions, Backends, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, BufferBindingType,
    BufferDescriptor, BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor,
    ComputePipelineDescriptor, DeviceDescriptor, Instance, InstanceDescriptor, InstanceFlags,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, PollType, RequestAdapterOptions,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use prism_render_architecture::cloth::collision::{resolve_body_collisions, BodyCollider};
use prism_render_architecture::cloth::{ClothParticle, Vec3};

use super::abi::{GpuClothBodyParams, GpuClothCollider};
use super::pack::pack_colliders;

/// `GPU`-对-`CPU` 逐分量绝对容差。
///
/// body-collision 是单 pass 纯投影，两条路径跑同一份 `float32` 算术，唯一自由度是
/// 球面投影里 `1.0 / sqrt`（CPU）与 `inverseSqrt`（WESL，多为原生 `rsqrt`）的几个
/// ULP 之差。位置量级为 `O(1)`，`1e-4` 既能吸收该 ULP 差，又远紧于任何真实内核 bug
/// 会产生的 `O(0.1)` 级发散。
const PARITY_EPS: f32 = 1.0e-4;

/// 本模块编译 `cloth_collision.wesl` 用的一次性 `AssetId`，只需在本次编译内唯一。
const CLOTH_COLLISION_WESL_UUID: u128 = 0x434c_4f54_485f_434f_4c4c_4244_5f42_4f01;

/// 把 `WESL` 源经 render-world 的 [`ShaderCache`] 编译回 `Wgsl` 字符串（不建
/// 设备），供本模块自建的裸 `wgpu` 设备使用。镜像 `sim_gpu_tests` 的编译闭包。
fn keep_wgsl(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("cloth shaders are WESL"),
    }
}

/// 经 `ShaderCache` 把嵌入式 `cloth_collision.wesl` 编译成 `Wgsl`。
fn compile_collision_wgsl() -> String {
    let mut cache = ShaderCache::new((), keep_wgsl);
    let id = AssetId::Uuid {
        uuid: Uuid::from_u128(CLOTH_COLLISION_WESL_UUID),
    };
    cache.set_shader(
        id,
        Shader::from_wesl(
            include_str!("../shaders/cloth_collision.wesl"),
            "shaders/cloth_collision.wesl",
        ),
    );
    let module = cache
        .get(0, id, &[])
        .unwrap_or_else(|error| panic!("cloth_collision.wesl failed to compile: {error}"));
    (*module).clone()
}

/// 在编译后的 `Wgsl` 里按子串定位 compute 入口的真实符号名（`WESL` 可能给模块内
/// 名字加前缀，故按子串而非固定符号查找）。
fn find_entry_point(wgsl: &str, needle: &str) -> String {
    for line in wgsl.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("fn ")
            && let Some(paren) = rest.find('(')
        {
            let name = &rest[..paren];
            if name.contains(needle) {
                return name.to_string();
            }
        }
    }
    panic!("no compute entry point containing `{needle}` in compiled Wgsl");
}

/// 尽力获取一个原生 compute 设备与队列。
///
/// body-collision 不用 `immediate`（push-constant），故只需一个默认能力的 compute
/// 设备——比求解核宽松，能在更多机器上跑满。无 adapter 时返回 `None`（不 panic），
/// 让无头机保持绿。
fn try_collision_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = Instance::new(InstanceDescriptor {
        backends: Backends::METAL | Backends::VULKAN | Backends::DX12,
        flags: InstanceFlags::default(),
        memory_budget_thresholds: Default::default(),
        display: None,
        backend_options: BackendOptions::default(),
    });
    let adapter = block_on(instance.request_adapter(&RequestAdapterOptions::default())).ok()?;
    let (device, queue) = block_on(adapter.request_device(&DeviceDescriptor::default())).ok()?;
    Some((device, queue))
}

/// `cloth_body_collision` 的 group-0 布局：positions（rw storage）、colliders（ro
/// storage）、`body_params`（uniform）、`prev_positions`（ro storage）。与 WESL 的
/// `@binding(0..3)` 一一对应。
fn body_bind_group_layout(device: &wgpu::Device) -> BindGroupLayout {
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
        label: Some("cloth_body_collision_parity_group0"),
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
            storage(3, true),
        ],
    })
}

/// 从一份 `Pod` 切片建只读存储缓冲；空输入回退到一个零元素，避免 runtime-sized
/// array 绑定非法（`0` 字节绑定被拒）。
fn storage_from_slice<T: bytemuck::Pod>(
    device: &wgpu::Device,
    label: &str,
    data: &[T],
    fallback: T,
) -> wgpu::Buffer {
    if data.is_empty() {
        device.create_buffer_init(&BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::bytes_of(&fallback),
            usage: BufferUsages::STORAGE,
        })
    } else {
        device.create_buffer_init(&BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: BufferUsages::STORAGE,
        })
    }
}

/// 在真机上跑一遍 `cloth_body_collision`，读回投影后的 positions（`xyzw`）。
///
/// 走 shipping 口径：`GpuClothBodyParams` 无 `friction` 字段 ⇒ uniform 的摩擦词恒
/// 为零 ⇒ 摩擦段早退，`prev_positions` 只为满足绑定合法（此处填初始 positions，
/// frictionless 下不被解引用为有效值）。
fn replay_body_collision_on_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    entry: &str,
    positions: &[[f32; 4]],
    colliders: &[GpuClothCollider],
) -> Vec<[f32; 4]> {
    let count = positions.len();
    let vec4_bytes = size_of_val(positions) as u64;

    let layout = body_bind_group_layout(device);
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("cloth_body_collision_parity_layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("cloth_body_collision_parity"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("cloth_body_collision_parity_pipeline"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some(entry),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });

    let positions_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_body_positions"),
        contents: bytemuck::cast_slice(positions),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });
    let colliders_buf = storage_from_slice(
        device,
        "cloth_body_colliders",
        colliders,
        GpuClothCollider::default(),
    );
    let params = GpuClothBodyParams {
        particle_count: count as u32,
        collider_count: colliders.len() as u32,
        _pad: [0, 0],
    };
    let params_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_body_params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });
    // frictionless：prev 只需绑定合法；填初始位置，摩擦段早退不会读出有效值。
    let prev_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_body_prev_positions"),
        contents: bytemuck::cast_slice(positions),
        usage: BufferUsages::STORAGE,
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("cloth_body_collision_parity_bind"),
        layout: &layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: positions_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: colliders_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: params_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: prev_buf.as_entire_binding(),
            },
        ],
    });

    let stage = device.create_buffer(&BufferDescriptor {
        label: Some("cloth_body_positions_stage"),
        size: vec4_bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("cloth_body_collision_parity_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("cloth_body_collision_parity_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        // 每 invocation 一个粒子，@workgroup_size(64) ⇒ ceil(count / 64) 组。
        let groups = count.div_ceil(64).max(1) as u32;
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

/// 对一组粒子与作者态碰撞体，同时跑 CPU 黄金 [`resolve_body_collisions`] 与真机
/// `cloth_body_collision`，断言逐粒子 `xyz` 落在 [`PARITY_EPS`] 内、`w`（inverse
/// mass）不被内核改动。
fn assert_body_collision_parity(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    entry: &str,
    particles: &[ClothParticle],
    colliders: &[BodyCollider],
) {
    // --- CPU 黄金：在副本上原地投影 ---
    let mut golden = particles.to_vec();
    resolve_body_collisions(&mut golden, colliders);

    // --- host 上传口径：positions.w = inverse mass（`<= 0` 即 pinned）---
    let positions: Vec<[f32; 4]> = particles
        .iter()
        .map(|p| [p.position.x, p.position.y, p.position.z, p.inverse_mass])
        .collect();
    let packed = pack_colliders(colliders);
    let out = replay_body_collision_on_gpu(device, queue, wgsl, entry, &positions, &packed);

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
        // inverse mass 是内核直通字段，绝不能被投影改动。
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

/// 建一个 pinned 粒子（`inverse_mass = 0`，投影永不移动它）。
fn pinned(x: f32, y: f32, z: f32) -> ClothParticle {
    ClothParticle::pinned(Vec3::new(x, y, z))
}

/// 单球：内部点被推到表面、表面/外部点不动、圆心退化沿 `+Y` 逃逸，且 pinned 内部点
/// 不动——全部与 CPU 逐比特对齐。
#[test]
#[expect(
    clippy::print_stderr,
    reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志"
)]
fn sphere_projection_matches_cpu_golden() {
    let Some((device, queue)) = try_collision_device() else {
        eprintln!(
            "sphere_projection_matches_cpu_golden: no wgpu adapter, skipping on-device parity"
        );
        return;
    };
    let wgsl = compile_collision_wgsl();
    let entry = find_entry_point(&wgsl, "cloth_body_collision");

    let colliders = [BodyCollider::Sphere {
        center: Vec3::new(0.0, 0.0, 0.0),
        radius: 1.0,
    }];
    let particles = [
        free(0.3, 0.1, -0.2),  // 深在内部 → 推到表面
        free(0.9, 0.0, 0.0),   // 恰在表面附近
        free(2.0, 0.0, 0.0),   // 外部 → 不动
        free(0.0, 0.0, 0.0),   // 圆心退化 → 沿 +Y 逃逸到 (0, r, 0)
        pinned(0.2, -0.1, 0.1),// pinned 内部 → 永不移动
    ];
    assert_body_collision_parity(&device, &queue, &wgsl, &entry, &particles, &colliders);
}

/// 胶囊：轴中段（径向推出）、端帽半球（超出端点后按端点球投影）、轴上退化（沿 `+Y`
/// 逃逸）、零长胶囊（`p0 == p1` 退化为球）——全部与 CPU 逐比特对齐。
#[test]
#[expect(
    clippy::print_stderr,
    reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志"
)]
fn capsule_projection_matches_cpu_golden() {
    let Some((device, queue)) = try_collision_device() else {
        eprintln!(
            "capsule_projection_matches_cpu_golden: no wgpu adapter, skipping on-device parity"
        );
        return;
    };
    let wgsl = compile_collision_wgsl();
    let entry = find_entry_point(&wgsl, "cloth_body_collision");

    let colliders = [BodyCollider::Capsule {
        p0: Vec3::new(-1.0, 0.0, 0.0),
        p1: Vec3::new(1.0, 0.0, 0.0),
        radius: 0.5,
    }];
    let particles = [
        free(0.0, 0.2, 0.0),   // 轴中段内部 → 径向推到 r
        free(0.0, 0.6, 0.0),   // 轴中段外部 → 不动
        free(1.3, 0.1, 0.0),   // 越过 p1 端 → 端帽半球投影
        free(1.0, 0.0, 0.0),   // 落在端点轴上退化 → 沿 +Y 逃逸
        free(-1.2, 0.0, 0.3),  // 越过 p0 端 → 另一端帽半球
    ];
    assert_body_collision_parity(&device, &queue, &wgsl, &entry, &particles, &colliders);

    // 零长胶囊退化为球（`p0 == p1`），单独一拍确认与 CPU 的退化分支一致。
    let degenerate = [BodyCollider::Capsule {
        p0: Vec3::new(0.5, 0.5, 0.5),
        p1: Vec3::new(0.5, 0.5, 0.5),
        radius: 0.4,
    }];
    let near = [free(0.6, 0.5, 0.5), free(0.5, 0.5, 0.5)];
    assert_body_collision_parity(&device, &queue, &wgsl, &entry, &near, &degenerate);
}

/// 半空间：非单位法线（修正按法线平方长归一，几何正确）下的越界点被推回平面，
/// 可行侧点不动，零法线惰性——全部与 CPU 逐比特对齐。
#[test]
#[expect(
    clippy::print_stderr,
    reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志"
)]
fn half_space_projection_matches_cpu_golden() {
    let Some((device, queue)) = try_collision_device() else {
        eprintln!(
            "half_space_projection_matches_cpu_golden: no wgpu adapter, skipping on-device parity"
        );
        return;
    };
    let wgsl = compile_collision_wgsl();
    let entry = find_entry_point(&wgsl, "cloth_body_collision");

    // 非单位法线（长度 3），可行域 `dot(n, x) >= offset`。
    let colliders = [BodyCollider::HalfSpace {
        normal: Vec3::new(0.0, 3.0, 0.0),
        offset: 0.0,
    }];
    let particles = [
        free(0.2, -0.5, 0.1),  // 越界（下方）→ 推回平面 y=0
        free(-0.3, 0.7, 0.4),  // 可行侧 → 不动
        free(1.0, 0.0, -1.0),  // 恰在平面上 → 不动
    ];
    assert_body_collision_parity(&device, &queue, &wgsl, &entry, &particles, &colliders);

    // 零法线惰性：碰撞体无定义平面，任何点都不动。
    let inert = [BodyCollider::HalfSpace {
        normal: Vec3::new(0.0, 0.0, 0.0),
        offset: 5.0,
    }];
    let pts = [free(0.0, -10.0, 0.0), free(1.0, 2.0, 3.0)];
    assert_body_collision_parity(&device, &queue, &wgsl, &entry, &pts, &inert);
}

/// 多碰撞体按数组顺序叠加：一个粒子先被球投影、再被半空间投影，后者覆盖前者
/// （「最后推的赢」）——GPU 的顺序循环与 CPU 的逐碰撞体顺序一致。
#[test]
#[expect(
    clippy::print_stderr,
    reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志"
)]
fn multi_collider_order_matches_cpu_golden() {
    let Some((device, queue)) = try_collision_device() else {
        eprintln!(
            "multi_collider_order_matches_cpu_golden: no wgpu adapter, skipping on-device parity"
        );
        return;
    };
    let wgsl = compile_collision_wgsl();
    let entry = find_entry_point(&wgsl, "cloth_body_collision");

    // 先球（把内部点推到球面），再地面半空间（把落在地面下的点抬回 y=0）。
    let colliders = [
        BodyCollider::Sphere {
            center: Vec3::new(0.0, 0.0, 0.0),
            radius: 1.0,
        },
        BodyCollider::HalfSpace {
            normal: Vec3::new(0.0, 1.0, 0.0),
            offset: -0.3,
        },
    ];
    let particles = [
        free(0.1, -0.9, 0.0),  // 球把它推到下半球面(y<-0.3) → 半空间再抬回 y=-0.3
        free(0.5, 0.5, 0.0),   // 球内 → 推到球面，位于 y>-0.3 → 半空间不动
        free(0.2, 0.1, 0.15),  // 顺序叠加的一般点
    ];
    assert_body_collision_parity(&device, &queue, &wgsl, &entry, &particles, &colliders);
}

/// 空碰撞体集合是彻底 no-op：内核 `collider_count == 0` ⇒ 循环不执行，所有粒子
/// 原样返回，与 CPU 的空集合早退一致。
#[test]
#[expect(
    clippy::print_stderr,
    reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志"
)]
fn empty_collider_set_is_noop_on_gpu() {
    let Some((device, queue)) = try_collision_device() else {
        eprintln!("empty_collider_set_is_noop_on_gpu: no wgpu adapter, skipping on-device parity");
        return;
    };
    let wgsl = compile_collision_wgsl();
    let entry = find_entry_point(&wgsl, "cloth_body_collision");

    let colliders: [BodyCollider; 0] = [];
    let particles = [free(0.3, -0.4, 0.5), pinned(1.0, 1.0, 1.0), free(-2.0, 0.0, 0.0)];
    assert_body_collision_parity(&device, &queue, &wgsl, &entry, &particles, &colliders);
}
