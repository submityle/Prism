//! `cloth_sim.wesl` 核心 XPBD 求解内核的**真机 GPU** parity 覆盖。
//!
//! 这是继气动两段式内核 [`aero_gpu_tests`](super::aero_gpu_tests) 之后，对布料
//! 主求解回路（predict → 距离投影 → 应变限制 → 速度回收）在真实设备上的对拍。
//! `shader_tests` 只证明 `cloth_sim.wesl` 能被 `naga` 编译，`solve_plan` 的单测
//! 只证明 dispatch 计划的装配正确；两者都**没有在真机上跑过内核**——一个能编译
//! 的 shader 仍可能因绑定顺序、`std430` 对齐、push-constant（`immediate`）窗口、
//! dispatch 覆盖或驱动代码生成而算错。本模块闭合这条缺口。
//!
//! 做法忠实复用权威路径而非重写算法：
//! - **计划**来自 [`build_solve_plan`]（`extract` + `prepare` 已把 substep 循环
//!   与图着色展平成一维 [`PlannedDispatch`] 列表），本模块只在裸 `wgpu` 上**重放**
//!   这份计划，算法本体跑在 `WESL` 里。
//! - **黄金**来自架构层 [`solve_cloth`]（内部固定 `dt = 1/60`，逐 substep
//!   predict → 按图着色 batch 投影 → 应变限制 → 速度回收）。
//!
//! 之所以能 bit-parity（`float32` 舍入内）：CPU 与 GPU 都按**图着色**分批投影，
//! 一个颜色内的约束互不共享粒子 ⇒ 该色内的投影可交换、与批内顺序无关；颜色之间
//! 都按升序 Gauss-Seidel ⇒ 两条路径的算术序一致，只差驱动的 `FMA` 收缩。容差
//! [`PARITY_EPS`] 只吸收这一自由度。
//!
//! 测试刻意把工况收敛到 sim 核心四内核：只喂网格 stretch/shear/bend **距离**约束
//! （皆两侧，走 `cloth_project_distance_batch`），不放 `LRA`/`tether`（会触发
//! 单侧 long-range 内核，需另配黄金）、不放 bending 铰链、风场、碰撞体、backstop、
//! skin-embed、自碰撞。位移量级（重力在 `dt=1/60`、2 substep 下约 1.4% 边长）远
//! 低于 10% 应变阈 ⇒ 应变限制内核在 CPU/GPU 两侧都逐约束早退（都为 no-op），从而
//! 规避 GPU 应变内核「全约束并行、无着色」的写竞态与顺序版黄金发散。
//!
//! 取设备是尽力而为：无 `wgpu` adapter、或设备不支持 `immediate`（push-constant）
//! 的无头机上 [`try_solver_device`] 返回 `None`，测试打印跳过提示而非失败，让套件
//! 在任何机器上保持绿，同时在有真实设备（如 `Apple` `M` 系列 `GPU`）时跑满整条
//! dispatch 计划。

use std::collections::HashMap;

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_math::ops::{cos, sin};
use bevy_platform::future::block_on;
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BackendOptions, Backends, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, BufferBindingType,
    BufferDescriptor, BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor,
    ComputePipeline, ComputePipelineDescriptor, DeviceDescriptor, Features, Instance,
    InstanceDescriptor, InstanceFlags, MapMode, PipelineCompilationOptions,
    PipelineLayoutDescriptor, PollType, RequestAdapterOptions, ShaderModuleDescriptor,
    ShaderSource, ShaderStages,
};

use prism_render_architecture::cloth::bending::{
    build_dihedral_bending, project_bending, BendingConstraint,
};
use prism_render_architecture::cloth::constraints::{
    build_grid_constraints, color_constraints, ClothGrid, GridConstraintParams,
};
use prism_render_architecture::cloth::dynamics::{solve_cloth, SolverParams};
use prism_render_architecture::cloth::gpu::kernels::ClothKernel;
use prism_render_architecture::cloth::gpu::upload::color_bending;
use prism_render_architecture::cloth::{ClothParticle, Compliance, Vec3};

use super::abi::{GpuClothBendingConstraint, GpuClothConstraint};
use super::solve_plan::{build_solve_plan, ClothSolveInput};

/// `GPU`-对-`CPU` 逐分量绝对容差。
///
/// 两条路径跑同一份 `float32` 算术，实测吻合远比此紧；此裕度只吸收多 substep 下
/// 驱动的融合乘加（`FMA`）收缩与投影求和重排自由度（气动单段用 `1e-3`，本模块两
/// substep 叠加故放宽到 `2e-3`，与 `water::gpu_tests` 同族口径）。
const PARITY_EPS: f32 = 2.0e-3;

/// `cloth_sim.wesl` 里 `ClothColorBatch { base: u32, count: u32 }` 的 `immediate`
/// 块字节数；三个逐颜色投影内核用它选取当前颜色的连续约束切片。
const CLOTH_COLOR_BATCH_SIZE: u32 = size_of::<[u32; 2]>() as u32;

/// 把 `WESL` 源经 render-world 的 [`ShaderCache`] 编译回 `Wgsl` 字符串（不建
/// 设备），供本模块自建的裸 `wgpu` 设备使用。镜像 `shader_tests` 的编译闭包。
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

/// 经 `ShaderCache` 把嵌入式 `WESL` 源编译成 `Wgsl`。`uuid` 只需在本次编译内唯一。
fn compile_wgsl(source: &'static str, path: &'static str, uuid: u128) -> String {
    let mut cache = ShaderCache::new((), keep_wgsl);
    let id = AssetId::Uuid {
        uuid: Uuid::from_u128(uuid),
    };
    cache.set_shader(id, Shader::from_wesl(source, path));
    let module = cache
        .get(0, id, &[])
        .unwrap_or_else(|error| panic!("{path} failed to compile: {error}"));
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

/// 尽力获取一个支持 `immediate`（push-constant）的原生 compute 设备与队列。
///
/// 逐颜色投影内核依赖 `var<immediate> batch`，故设备必须启用 [`Features::IMMEDIATES`]
/// 且 `max_immediate_size >= 8`；无 adapter 或不满足能力时返回 `None`（不 panic），
/// 让无头机保持绿，有真实 `GPU` 时给出可 dispatch 的活设备。
fn try_solver_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = Instance::new(InstanceDescriptor {
        backends: Backends::METAL | Backends::VULKAN | Backends::DX12,
        flags: InstanceFlags::default(),
        memory_budget_thresholds: Default::default(),
        display: None,
        backend_options: BackendOptions::default(),
    });
    let adapter = block_on(instance.request_adapter(&RequestAdapterOptions::default())).ok()?;
    if !adapter.features().contains(Features::IMMEDIATES) {
        return None;
    }
    let limits = adapter.limits();
    if limits.max_immediate_size < CLOTH_COLOR_BATCH_SIZE {
        return None;
    }
    let (device, queue) = block_on(adapter.request_device(&DeviceDescriptor {
        required_features: Features::IMMEDIATES,
        required_limits: limits,
        ..Default::default()
    }))
    .ok()?;
    Some((device, queue))
}

/// 一块 `rows × cols` 悬挂布料的确定性初值。
///
/// 粒子按行主序 `r * cols + c` 排布，行距/列距 `0.1`；**顶行**（`r == 0`）
/// `inverse_mass = 0`（pin，永不移动），其余自由粒子给随索引轻微变化的 inverse
/// mass 与一个位置相关的非零初速，以压满 predict/投影分支。位移量级远低于应变阈，
/// 故应变限制内核在两侧都早退。
fn build_hanging_grid(rows: u32, cols: u32) -> Vec<ClothParticle> {
    let mut particles = Vec::with_capacity((rows * cols) as usize);
    for r in 0..rows {
        for c in 0..cols {
            let px = c as f32 * 0.1 - (cols as f32) * 0.05;
            let py = 0.5 - r as f32 * 0.1;
            let pz = sin((r + c) as f32 * 0.013) * 0.02;
            let inv_mass = if r == 0 {
                0.0
            } else {
                0.8 + ((r * 7 + c * 13) % 5) as f32 * 0.1
            };
            let mut p = ClothParticle::new(Vec3::new(px, py, pz), inv_mass);
            if !p.is_pinned() {
                p.velocity = Vec3::new(
                    cos(c as f32 * 0.05) * 0.05,
                    sin(r as f32 * 0.04) * 0.03,
                    cos((r + c) as f32 * 0.03) * 0.02,
                );
            }
            particles.push(p);
        }
    }
    particles
}

/// `cloth_sim` 六内核共享的 group-0 布局：3 个 read-write 存储缓冲（positions /
/// velocities / `prev_positions`）、2 个 read-only 存储缓冲（constraints / bending）
/// 与 1 个 uniform（params）。逐内核只引用其中一个子集；布局是它们的超集，`WESL`
/// 不引用的绑定多出无碍。与场景层 `cloth::pipeline` 的显式布局一致。
fn sim_bind_group_layout(device: &wgpu::Device) -> BindGroupLayout {
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
    let uniform = BindGroupLayoutEntry {
        binding: 5,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("cloth_sim_parity_group0"),
        entries: &[
            storage(0, false),
            storage(1, false),
            storage(2, false),
            storage(3, true),
            storage(4, true),
            uniform,
        ],
    })
}

/// 逐颜色投影内核用 8 字节 `immediate`；逐粒子内核不读 `batch`，`immediate_size` 为 0。
fn immediate_size_for(kernel: ClothKernel) -> u32 {
    match kernel {
        ClothKernel::ProjectDistanceBatch
        | ClothKernel::ProjectBendingBatch
        | ClothKernel::ProjectLongRangeBatch => CLOTH_COLOR_BATCH_SIZE,
        _ => 0,
    }
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

/// 在真机上按 [`ClothSolvePlan`](super::solve_plan::ClothSolvePlan) 的 dispatch
/// 计划重放 `cloth_sim` 求解，读回最终的 positions 与 velocities。
///
/// 所有 dispatch 记录在**同一个** compute pass 内（与场景层 `cloth::dispatch`
/// 一致）：`wgpu` 的资源冒险跟踪会在读写同一存储缓冲的相邻 dispatch 间自动插入内存
/// 屏障，从而复刻 CPU 黄金逐 substep、逐颜色的严格先后序。
#[expect(
    clippy::too_many_lines,
    reason = "一条线性的建管线-建缓冲-重放-读回让 parity 路径整体可审计"
)]
fn replay_sim_on_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    positions: &[[f32; 4]],
    velocities: &[[f32; 4]],
    plan: &super::solve_plan::ClothSolvePlan,
) -> (Vec<[f32; 4]>, Vec<[f32; 4]>) {
    let count = positions.len();
    let vec4_bytes = size_of_val(positions) as u64;

    // --- 共享 group-0 与两个 pipeline layout（投影用 8 字节 immediate，其余为 0）---
    let layout = sim_bind_group_layout(device);
    let layout_ref: &[Option<&BindGroupLayout>] = &[Some(&layout)];
    let pl_imm = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("cloth_sim_parity_layout_imm"),
        bind_group_layouts: layout_ref,
        immediate_size: CLOTH_COLOR_BATCH_SIZE,
    });
    let pl_plain = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("cloth_sim_parity_layout_plain"),
        bind_group_layouts: layout_ref,
        immediate_size: 0,
    });

    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("cloth_sim_parity"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });

    // 只为计划里实际出现的内核建 pipeline（本工况：Predict / ProjectDistanceBatch /
    // StrainLimit / VelocityUpdate）。
    let mut pipelines: HashMap<ClothKernel, ComputePipeline> = HashMap::new();
    for dispatch in &plan.dispatches {
        pipelines.entry(dispatch.kernel).or_insert_with(|| {
            let entry = find_entry_point(wgsl, dispatch.kernel.wesl_entry_point());
            let pipeline_layout = if immediate_size_for(dispatch.kernel) > 0 {
                &pl_imm
            } else {
                &pl_plain
            };
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("cloth_sim_parity_pipeline"),
                layout: Some(pipeline_layout),
                module: &module,
                entry_point: Some(&entry),
                compilation_options: PipelineCompilationOptions::default(),
                cache: None,
            })
        });
    }

    // --- 缓冲 ---
    let positions_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_sim_positions"),
        contents: bytemuck::cast_slice(positions),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });
    let velocities_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_sim_velocities"),
        contents: bytemuck::cast_slice(velocities),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });
    // prev_positions 由 predict 先写后读，初值无所谓，置零。
    let prev_zero = vec![[0.0f32; 4]; count];
    let prev_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_sim_prev_positions"),
        contents: bytemuck::cast_slice(&prev_zero),
        usage: BufferUsages::STORAGE,
    });
    let constraints_buf = storage_from_slice(
        device,
        "cloth_sim_constraints",
        &plan.constraints,
        GpuClothConstraint::default(),
    );
    let bending_buf = storage_from_slice(
        device,
        "cloth_sim_bending",
        &plan.bending,
        GpuClothBendingConstraint::default(),
    );
    let params_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_sim_params"),
        contents: bytemuck::bytes_of(&plan.sim_params),
        usage: BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("cloth_sim_parity_bind"),
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
                resource: constraints_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: bending_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 5,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let pos_stage = device.create_buffer(&BufferDescriptor {
        label: Some("cloth_sim_positions_stage"),
        size: vec4_bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let vel_stage = device.create_buffer(&BufferDescriptor {
        label: Some("cloth_sim_velocities_stage"),
        size: vec4_bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("cloth_sim_parity_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("cloth_sim_parity_pass"),
            timestamp_writes: None,
        });
        for dispatch in &plan.dispatches {
            let pipeline = pipelines
                .get(&dispatch.kernel)
                .expect("every scheduled kernel has a compiled pipeline");
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            if let Some(batch) = dispatch.batch {
                let window: [u32; 2] = [batch.base, batch.count];
                pass.set_immediates(0, bytemuck::bytes_of(&window));
            }
            pass.dispatch_workgroups(dispatch.groups, 1, 1);
        }
    }
    encoder.copy_buffer_to_buffer(&positions_buf, 0, &pos_stage, 0, vec4_bytes);
    encoder.copy_buffer_to_buffer(&velocities_buf, 0, &vel_stage, 0, vec4_bytes);
    queue.submit([encoder.finish()]);

    pos_stage.slice(..).map_async(MapMode::Read, |_| {});
    vel_stage.slice(..).map_async(MapMode::Read, |_| {});
    device
        .poll(PollType::wait_indefinitely())
        .expect("device poll should complete the submitted work");

    let pos_view = pos_stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped positions readback range should be available after poll");
    let out_positions: Vec<[f32; 4]> = bytemuck::cast_slice::<u8, [f32; 4]>(&pos_view).to_vec();
    drop(pos_view);
    pos_stage.unmap();

    let vel_view = vel_stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped velocities readback range should be available after poll");
    let out_velocities: Vec<[f32; 4]> = bytemuck::cast_slice::<u8, [f32; 4]>(&vel_view).to_vec();
    drop(vel_view);
    vel_stage.unmap();

    (out_positions, out_velocities)
}

/// 核心 sim 内核在真机上重放整条 dispatch 计划后，必须与架构层黄金
/// [`solve_cloth`] 逐顶点落在 `float32` 舍入容差内（位置与速度同拍）。
#[test]
#[expect(
    clippy::print_stderr,
    reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志"
)]
fn sim_gpu_matches_cpu_golden() {
    let Some((device, queue)) = try_solver_device() else {
        eprintln!(
            "sim_gpu_matches_cpu_golden: no wgpu adapter with IMMEDIATES support, \
             skipping on-device parity"
        );
        return;
    };

    const ROWS: u32 = 8;
    const COLS: u32 = 6;
    let particles = build_hanging_grid(ROWS, COLS);
    let count = particles.len();

    // 约束集：网格 stretch/shear/bend 距离约束（皆两侧），rest length 取自初始位置。
    let grid = ClothGrid::new(ROWS, COLS);
    let rest_positions: Vec<Vec3> = particles.iter().map(|p| p.position).collect();
    let constraints =
        build_grid_constraints(grid, &rest_positions, GridConstraintParams::default());

    // 求解参数：CPU 与 GPU 完全对齐。solve_cloth 内部固定 dt = 1/60。
    let params = SolverParams {
        substeps: 2,
        iterations: 1,
        gravity: Vec3::new(0.0, -9.81, 0.0),
        damping: 0.02,
        strain_limit: 0.1,
    };

    // --- CPU 黄金：在副本上原地推进一帧 ---
    let mut golden = particles.clone();
    let graph = color_constraints(&constraints);
    solve_cloth(&mut golden, &graph, params);

    // --- host 上传口径：positions.w = inverse mass；velocities.w 塞可辨识载荷位 ---
    let positions: Vec<[f32; 4]> = particles
        .iter()
        .map(|p| [p.position.x, p.position.y, p.position.z, p.inverse_mass])
        .collect();
    let velocities: Vec<[f32; 4]> = particles
        .iter()
        .enumerate()
        .map(|(i, p)| [p.velocity.x, p.velocity.y, p.velocity.z, (i as f32) + 0.5])
        .collect();

    // --- GPU 计划：走权威 build_solve_plan（与黄金同参、同约束、同着色）---
    let input = ClothSolveInput {
        positions: &positions,
        velocities: &velocities,
        constraints: &constraints,
        bending: &[],
        triangles: &[],
        wind_velocity: [0.0, 0.0, 0.0],
        wind_turbulence: 0.0,
        aero_drag: 0.0,
        aero_lift: 0.0,
        colliders: &[],
        backstops: &[],
        embed_bindings: &[],
        render_vertex_count: 0,
        hash_cell_count: 0,
        gravity: [0.0, -9.81, 0.0],
        dt: 1.0 / 60.0,
        substeps: 2,
        iterations: 1,
        damping: 0.02,
        strain_limit: 0.1,
        self_thickness: 0.0,
        self_cell_size: 0.0,
    };
    let plan = build_solve_plan(&input);

    let wgsl = compile_wgsl(
        include_str!("../shaders/cloth_sim.wesl"),
        "embedded://prism_render_scene/shaders/cloth_sim.wesl",
        0x434c_4f54_485f_5349_4d5f_5041_5249_5401,
    );

    let (gpu_positions, gpu_velocities) =
        replay_sim_on_gpu(&device, &queue, &wgsl, &positions, &velocities, &plan);

    assert_eq!(gpu_positions.len(), count, "position readback length");
    assert_eq!(gpu_velocities.len(), count, "velocity readback length");

    for (i, g) in golden.iter().enumerate() {
        let p = gpu_positions[i];
        let px = (p[0] - g.position.x).abs();
        let py = (p[1] - g.position.y).abs();
        let pz = (p[2] - g.position.z).abs();
        assert!(
            px <= PARITY_EPS && py <= PARITY_EPS && pz <= PARITY_EPS,
            "vertex {i}: GPU position ({}, {}, {}) drifted from golden ({}, {}, {}) beyond {PARITY_EPS}",
            p[0], p[1], p[2], g.position.x, g.position.y, g.position.z
        );
        // positions.w 必须始终保持 inverse mass。
        assert_eq!(
            p[3].to_bits(),
            g.inverse_mass.to_bits(),
            "vertex {i}: positions.w (inverse mass) was mutated"
        );

        let v = gpu_velocities[i];
        let vx = (v[0] - g.velocity.x).abs();
        let vy = (v[1] - g.velocity.y).abs();
        let vz = (v[2] - g.velocity.z).abs();
        assert!(
            vx <= PARITY_EPS && vy <= PARITY_EPS && vz <= PARITY_EPS,
            "vertex {i}: GPU velocity ({}, {}, {}) drifted from golden ({}, {}, {}) beyond {PARITY_EPS}",
            v[0], v[1], v[2], g.velocity.x, g.velocity.y, g.velocity.z
        );
        // velocities.w 载荷位必须原样保留。
        assert_eq!(
            v[3].to_bits(),
            ((i as f32) + 0.5).to_bits(),
            "vertex {i}: velocities.w payload was mutated"
        );
    }
}

/// 为 `rows × cols` 行主序网格生成两三角/格的三角化（顺时针一致缠绕）。
///
/// 索引 `r * cols + c`。每个内部格 `(r, c)`（`r < rows-1`、`c < cols-1`）切成
/// `[i00, i10, i11]` 与 `[i00, i11, i01]` 两片；相邻格共享的内部边正是
/// [`build_dihedral_bending`] 要收集的二面角铰链所在。
fn build_grid_triangles(rows: u32, cols: u32) -> Vec<[u32; 3]> {
    let mut triangles = Vec::new();
    for r in 0..rows.saturating_sub(1) {
        for c in 0..cols.saturating_sub(1) {
            let i00 = r * cols + c;
            let i01 = r * cols + c + 1;
            let i10 = (r + 1) * cols + c;
            let i11 = (r + 1) * cols + c + 1;
            triangles.push([i00, i10, i11]);
            triangles.push([i00, i11, i01]);
        }
    }
    triangles
}

/// 一块 `rows × cols` 布料的**折叠**初值，外加与之对应的**平坦静止位形**。
///
/// 返回 `(particles, rest_positions)`：`rest_positions` 是完全共面（`z = 0`）的
/// 静止网格，交给 [`build_dihedral_bending`] 算铰链权重 ⇒ 平坦静止的弯曲测度
/// `S = 0`；`particles` 则把每个顶点沿 `z` 抬起一个位置相关的折量，使初始 `S ≠ 0`，
/// 从而弯曲投影内核有真实的非零修正可算（否则整条弯曲路径退化成 no-op，测不出
/// 任何东西）。顶行 `r == 0` pin 住，其余自由粒子带轻微初速以压满 predict 分支。
fn build_folded_grid(rows: u32, cols: u32) -> (Vec<ClothParticle>, Vec<Vec3>) {
    let mut particles = Vec::with_capacity((rows * cols) as usize);
    let mut rest = Vec::with_capacity((rows * cols) as usize);
    for r in 0..rows {
        for c in 0..cols {
            let px = c as f32 * 0.1 - (cols as f32) * 0.05;
            let py = 0.5 - r as f32 * 0.1;
            rest.push(Vec3::new(px, py, 0.0));

            // 位置相关的出平面折量：沿列一个正弦褶皱，沿行一个更缓的起伏，叠加成
            // 一个非平凡且确定的折叠面。幅度远小于边长，弯曲能量温和但非零。
            let fold_z = sin(c as f32 * 0.8) * 0.03 + sin(r as f32 * 0.6) * 0.02;
            let inv_mass = if r == 0 {
                0.0
            } else {
                0.8 + ((r * 7 + c * 13) % 5) as f32 * 0.1
            };
            let mut p = ClothParticle::new(Vec3::new(px, py, fold_z), inv_mass);
            if !p.is_pinned() {
                p.velocity = Vec3::new(
                    cos(c as f32 * 0.05) * 0.03,
                    sin(r as f32 * 0.04) * 0.02,
                    cos((r + c) as f32 * 0.03) * 0.02,
                );
            }
            particles.push(p);
        }
    }
    (particles, rest)
}

/// 二面角弯曲**专属黄金**：忠实复刻 GPU 计划在一帧内对弯曲铰链的推进算术序。
///
/// 场景 GPU 计划把每个 substep 展平成 `predict → 距离颜色 → 弯曲颜色 → long-range
/// 颜色 → 应变 → 碰撞 → 速度`（见 `architecture::cloth::gpu::pipeline::prepare`）。
/// 本工况只喂弯曲铰链、不放任何距离约束，故计划坍缩为
/// `predict → 弯曲颜色（× iterations）→ 速度`。这里的 predict / 速度回收逐字段镜像
/// [`solve_cloth_with_collision`](prism_render_architecture::cloth::dynamics)，弯曲
/// 修正则调用权威原语 [`project_bending`]——CPU 与 WESL 是两套独立实现的同一
/// XPBD 弯曲能量投影，因此这是真对拍而非自证。
///
/// `colored` 必须已是 [`color_bending`] 的按颜色连续重排：同色铰链八个粒子槽两两
/// 不交 ⇒ 色内投影可交换、原地 Gauss-Seidel 等价于 GPU 的色内 Jacobi；颜色之间
/// CPU 顺扫与 GPU 顺序 dispatch 都读上一色的写结果 ⇒ 算术序逐色对齐。
fn solve_bending_golden(
    particles: &mut [ClothParticle],
    colored: &[BendingConstraint],
    params: &SolverParams,
    dt: f32,
) {
    if dt <= 0.0 || particles.is_empty() {
        return;
    }
    let substeps = params.substeps.max(1);
    let iterations = params.iterations.max(1);
    let dt_sub = dt / substeps as f32;
    let retain = (1.0 - params.damping).clamp(0.0, 1.0);
    let gravity_step = params.gravity.scale(dt_sub);

    for _ in 0..substeps {
        // 1. predict：阻尼速度 + 重力，积分位置。pin 粒子不动。
        let mut previous: Vec<Vec3> = Vec::with_capacity(particles.len());
        for particle in particles.iter_mut() {
            previous.push(particle.position);
            if particle.is_pinned() {
                continue;
            }
            particle.velocity = particle.velocity.scale(retain).add(gravity_step);
            particle.position = particle.position.add(particle.velocity.scale(dt_sub));
        }

        // 2. 逐迭代、逐颜色投影弯曲铰链（色内互不相干，原地即 Jacobi）。
        for _ in 0..iterations {
            for hinge in colored {
                project_bending(particles, *hinge, dt_sub);
            }
        }

        // 3. 从位移增量回收速度。pin 粒子速度清零。
        for (particle, &prev) in particles.iter_mut().zip(previous.iter()) {
            if particle.is_pinned() {
                particle.velocity = Vec3::ZERO;
                continue;
            }
            particle.velocity = particle.position.sub(prev).scale(1.0 / dt_sub);
        }
    }
}

/// 二面角弯曲投影内核 `cloth_project_bending_batch` 在真机上按整条 dispatch 计划
/// 重放后，必须与弯曲专属黄金 [`solve_bending_golden`] 逐顶点落在 `float32` 容差内
/// （位置与速度同拍）。这补上 sim 核心四内核之外的第五个逐颜色内核覆盖。
#[test]
#[expect(
    clippy::print_stderr,
    reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志"
)]
fn bending_gpu_matches_cpu_golden() {
    let Some((device, queue)) = try_solver_device() else {
        eprintln!(
            "bending_gpu_matches_cpu_golden: no wgpu adapter with IMMEDIATES support, \
             skipping on-device parity"
        );
        return;
    };

    const ROWS: u32 = 7;
    const COLS: u32 = 5;
    let (particles, rest) = build_folded_grid(ROWS, COLS);
    let count = particles.len();

    // 从**平坦静止位形**建刚性二面角铰链：静止 `S = 0`，折叠初值 `S ≠ 0`。
    let triangles = build_grid_triangles(ROWS, COLS);
    let hinges = build_dihedral_bending(&rest, &triangles, Compliance::RIGID);
    assert!(!hinges.is_empty(), "折叠网格三角化应产出至少一条内部边铰链");

    // 求解参数：CPU 与 GPU 完全对齐。solve/plan 内部按 dt = 1/60 推进。
    let params = SolverParams {
        substeps: 2,
        iterations: 1,
        gravity: Vec3::new(0.0, -9.81, 0.0),
        damping: 0.02,
        strain_limit: 0.0,
    };

    // --- CPU 黄金：按 color_bending 的颜色序在副本上推进一帧 ---
    let colored = color_bending(&hinges).bending;
    let mut golden = particles.clone();
    solve_bending_golden(&mut golden, &colored, &params, 1.0 / 60.0);

    // --- host 上传口径：positions.w = inverse mass；velocities.w 塞可辨识载荷位 ---
    let positions: Vec<[f32; 4]> = particles
        .iter()
        .map(|p| [p.position.x, p.position.y, p.position.z, p.inverse_mass])
        .collect();
    let velocities: Vec<[f32; 4]> = particles
        .iter()
        .enumerate()
        .map(|(i, p)| [p.velocity.x, p.velocity.y, p.velocity.z, (i as f32) + 0.5])
        .collect();

    // --- GPU 计划：走权威 build_solve_plan（无距离约束 ⇒ 计划只含 predict / 弯曲
    // 颜色 / 应变(空,no-op) / 速度）---
    let input = ClothSolveInput {
        positions: &positions,
        velocities: &velocities,
        constraints: &[],
        bending: &hinges,
        triangles: &[],
        wind_velocity: [0.0, 0.0, 0.0],
        wind_turbulence: 0.0,
        aero_drag: 0.0,
        aero_lift: 0.0,
        colliders: &[],
        backstops: &[],
        embed_bindings: &[],
        render_vertex_count: 0,
        hash_cell_count: 0,
        gravity: [0.0, -9.81, 0.0],
        dt: 1.0 / 60.0,
        substeps: 2,
        iterations: 1,
        damping: 0.02,
        strain_limit: 0.0,
        self_thickness: 0.0,
        self_cell_size: 0.0,
    };
    let plan = build_solve_plan(&input);
    assert!(!plan.bending.is_empty(), "计划应打包非空的弯曲缓冲");

    let wgsl = compile_wgsl(
        include_str!("../shaders/cloth_sim.wesl"),
        "embedded://prism_render_scene/shaders/cloth_sim.wesl",
        0x434c_4f54_485f_4245_4e44_5f50_4152_5401,
    );

    let (gpu_positions, gpu_velocities) =
        replay_sim_on_gpu(&device, &queue, &wgsl, &positions, &velocities, &plan);

    assert_eq!(gpu_positions.len(), count, "position readback length");
    assert_eq!(gpu_velocities.len(), count, "velocity readback length");

    for (i, g) in golden.iter().enumerate() {
        let p = gpu_positions[i];
        let px = (p[0] - g.position.x).abs();
        let py = (p[1] - g.position.y).abs();
        let pz = (p[2] - g.position.z).abs();
        assert!(
            px <= PARITY_EPS && py <= PARITY_EPS && pz <= PARITY_EPS,
            "vertex {i}: GPU position ({}, {}, {}) drifted from golden ({}, {}, {}) beyond {PARITY_EPS}",
            p[0], p[1], p[2], g.position.x, g.position.y, g.position.z
        );
        assert_eq!(
            p[3].to_bits(),
            g.inverse_mass.to_bits(),
            "vertex {i}: positions.w (inverse mass) was mutated"
        );

        let v = gpu_velocities[i];
        let vx = (v[0] - g.velocity.x).abs();
        let vy = (v[1] - g.velocity.y).abs();
        let vz = (v[2] - g.velocity.z).abs();
        assert!(
            vx <= PARITY_EPS && vy <= PARITY_EPS && vz <= PARITY_EPS,
            "vertex {i}: GPU velocity ({}, {}, {}) drifted from golden ({}, {}, {}) beyond {PARITY_EPS}",
            v[0], v[1], v[2], g.velocity.x, g.velocity.y, g.velocity.z
        );
        assert_eq!(
            v[3].to_bits(),
            ((i as f32) + 0.5).to_bits(),
            "vertex {i}: velocities.w payload was mutated"
        );
    }
}
