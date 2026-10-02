//! 布料气动力两段式内核的**真机 GPU** parity 覆盖。
//!
//! 兄弟模块 [`aero_parity`](super::aero_parity) 把两个气动 `WESL` 内核逐条
//! 转写成 CPU 版本、断言其与架构层黄金 [`accumulate_aero_gather`] **逐位**一致；
//! 那证明了「转写 == 黄金」，也通过 `shader_tests` 证明了 `WESL` 能编译，但两者
//! 合起来仍**没有在真实设备上跑过内核**——一个 shader 可以编译通过、CPU 转写也
//! 对，真机上却因绑定顺序、`std430` 对齐、dispatch 覆盖或驱动代码生成而算错。
//! 本模块闭合这条缺口：在真实 `Metal` / `Vulkan` / `DX12` `wgpu` 设备上绑定
//! 真正的 `cloth_aerodynamics_snapshot` + `cloth_aerodynamics` 两个 compute
//! 管线，按 host 上传口径喂入一块非平凡布料网格（含 pin 顶点、非零初速、风场
//! 与湍流），两段 dispatch 后把 `velocities` 读回，逐顶点与 CPU 黄金
//! [`accumulate_aero_gather`] 对拍。
//!
//! 因为 `WESL` 内核与 CPU 参考共享 `float32` 算术，绿测即真机层面证明移植内核
//! 复刻了参考、精度落在 `float32` 舍入内，而非仅仅「能编译」。容差
//! [`PARITY_EPS`] 只用来吸收驱动的 `FMA` 收缩与求和重排自由度（与
//! `water::gpu_tests` 同款口径）。
//!
//! 取设备是尽力而为：无 `wgpu` adapter 的无头机（部分 `CI` 镜像）
//! [`try_solver_device`] 返回 `None`，测试打印跳过提示而非失败，让套件在任何机
//! 器上保持绿，同时在有真实设备（如 `Apple` `M` 系列 `GPU`）时跑满整条
//! dispatch。

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_math::ops::{cos, sin};
use bevy_platform::future::block_on;
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BackendOptions, Backends, BindGroupDescriptor, BindGroupEntry, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor, DeviceDescriptor,
    Instance, InstanceDescriptor, InstanceFlags, MapMode, PipelineCompilationOptions, PollType,
    RequestAdapterOptions, ShaderModuleDescriptor, ShaderSource,
};

use prism_render_architecture::cloth::aero_gather::{
    accumulate_aero_gather, VertexTriangleAdjacency,
};
use prism_render_architecture::cloth::wind::{AeroParams, WindField};
use prism_render_architecture::cloth::{ClothParticle, Vec3};

use super::abi::GpuClothAeroParams;

/// `GPU`-对-`CPU` 逐顶点比较的每分量绝对容差。
///
/// 两条路径跑同一份 `float32` 算术，实际吻合远比这紧；此裕度只吸收驱动的
/// 融合乘加（`FMA`）收缩与三角求和重排自由度。
const PARITY_EPS: f32 = 1.0e-3;

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

/// 经 `ShaderCache` 把一个嵌入式 `WESL` 源编译成 `Wgsl`。
///
/// `uuid` 只需在本次编译内唯一；用与 `shader_tests` 同族的稳定命名空间派生。
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

/// 在编译后的 `Wgsl` 里按子串定位 compute 入口的真实符号名。
///
/// `WESL` 编译器可能给模块内名字加前缀，故按子串而非固定符号查找。
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

/// 尽力获取一个原生 compute 设备与队列；无 adapter 时返回 `None`（不 panic），
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
    let (device, queue) = block_on(adapter.request_device(&DeviceDescriptor {
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .ok()?;
    Some((device, queue))
}

/// 一块 `nx × nz` 网格布料的确定性初值：世界坐标、非零初速、按行列部分 pin。
///
/// 每个四边形拆成两个三角（`(a,b,c)` / `(a,c,d)`），拓扑与 host 打包一致。角上
/// 若干顶点 `inverse_mass = 0`（pin，气动应跳过），其余给非零 inverse mass 与
/// 一个随位置缓变的初速，压满 gather 的分支。
fn build_grid(nx: usize, nz: usize) -> (Vec<ClothParticle>, Vec<[u32; 3]>) {
    let mut particles = Vec::with_capacity(nx * nz);
    let mut z = 0;
    while z < nz {
        let mut x = 0;
        while x < nx {
            let px = x as f32 * 0.1 - (nx as f32) * 0.05;
            let py = sin((x + z) as f32 * 0.017) * 0.03;
            let pz = z as f32 * 0.1 - (nz as f32) * 0.05;
            // 四角 pin；其余为自由顶点，inverse mass 随索引轻微变化。
            let corner = (x == 0 || x == nx - 1) && (z == 0 || z == nz - 1);
            let inv_mass = if corner {
                0.0
            } else {
                0.8 + ((x * 7 + z * 13) % 5) as f32 * 0.1
            };
            let mut p = ClothParticle::new(Vec3::new(px, py, pz), inv_mass);
            p.velocity = Vec3::new(
                cos(z as f32 * 0.05) * 0.2,
                sin(x as f32 * 0.04) * 0.15,
                cos((x + z) as f32 * 0.03) * 0.1,
            );
            particles.push(p);
            x += 1;
        }
        z += 1;
    }

    let mut triangles = Vec::new();
    let mut z = 0;
    while z + 1 < nz {
        let mut x = 0;
        while x + 1 < nx {
            let a = (z * nx + x) as u32;
            let b = (z * nx + x + 1) as u32;
            let c = ((z + 1) * nx + x + 1) as u32;
            let d = ((z + 1) * nx + x) as u32;
            triangles.push([a, b, c]);
            triangles.push([a, c, d]);
            x += 1;
        }
        z += 1;
    }
    (particles, triangles)
}

/// 在设备上顺序 dispatch `cloth_aerodynamics_snapshot` 与 `cloth_aerodynamics`，
/// 读回累加后的 `velocities`。两个 pass 的 bind group 均直接从各自管线反射出的
/// `group(0)` 布局构建，绑定序严格对齐 `WESL` 声明。
#[expect(
    clippy::too_many_lines,
    reason = "一条线性的两段 dispatch-and-readback 让 parity 路径整体可审计"
)]
fn dispatch_aero(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    snapshot_wgsl: &str,
    snapshot_entry: &str,
    gather_wgsl: &str,
    gather_entry: &str,
    positions: &[[f32; 4]],
    velocities: &[[f32; 4]],
    triangles: &[u32],
    csr_offsets: &[u32],
    csr_entries: &[u32],
    params: &GpuClothAeroParams,
) -> Vec<[f32; 4]> {
    let count = velocities.len();
    let vec4_bytes = size_of_val(velocities) as u64;

    // --- 共享缓冲 ---
    let positions_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("aero_positions"),
        contents: bytemuck::cast_slice(positions),
        usage: BufferUsages::STORAGE,
    });
    // velocities 会被 gather 就地读写，故 STORAGE | COPY_SRC。
    let velocities_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("aero_velocities"),
        contents: bytemuck::cast_slice(velocities),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });
    // snapshot 由第一段写、第二段读。
    let snapshot_buf = device.create_buffer(&BufferDescriptor {
        label: Some("aero_velocity_snapshot"),
        size: vec4_bytes,
        usage: BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let triangles_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("aero_triangles"),
        contents: bytemuck::cast_slice(triangles),
        usage: BufferUsages::STORAGE,
    });
    let offsets_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("aero_csr_offsets"),
        contents: bytemuck::cast_slice(csr_offsets),
        usage: BufferUsages::STORAGE,
    });
    let entries_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("aero_csr_entries"),
        contents: bytemuck::cast_slice(csr_entries),
        usage: BufferUsages::STORAGE,
    });
    let params_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("aero_params"),
        contents: bytemuck::bytes_of(params),
        usage: BufferUsages::UNIFORM,
    });

    // --- 第一段：快照 pipeline（bindings 0=velocities, 1=snapshot, 2=params） ---
    let snapshot_module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("cloth_aero_snapshot_parity"),
        source: ShaderSource::Wgsl(snapshot_wgsl.into()),
    });
    let snapshot_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("cloth_aero_snapshot_parity"),
        layout: None,
        module: &snapshot_module,
        entry_point: Some(snapshot_entry),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });
    let snapshot_layout = snapshot_pipeline.get_bind_group_layout(0);
    let snapshot_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("cloth_aero_snapshot_group0"),
        layout: &snapshot_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: velocities_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: snapshot_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    // --- 第二段：gather pipeline（bindings 0..=6） ---
    let gather_module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("cloth_aero_gather_parity"),
        source: ShaderSource::Wgsl(gather_wgsl.into()),
    });
    let gather_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("cloth_aero_gather_parity"),
        layout: None,
        module: &gather_module,
        entry_point: Some(gather_entry),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });
    let gather_layout = gather_pipeline.get_bind_group_layout(0);
    let gather_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("cloth_aero_gather_group0"),
        layout: &gather_layout,
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
                resource: snapshot_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: triangles_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: offsets_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 5,
                resource: entries_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 6,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let stage = device.create_buffer(&BufferDescriptor {
        label: Some("aero_velocities_stage"),
        size: vec4_bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let workgroups = (count as u32).div_ceil(64);
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("cloth_aero_parity_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("cloth_aero_snapshot_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&snapshot_pipeline);
        pass.set_bind_group(0, &snapshot_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("cloth_aero_gather_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&gather_pipeline);
        pass.set_bind_group(0, &gather_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&velocities_buf, 0, &stage, 0, vec4_bytes);
    queue.submit([encoder.finish()]);

    stage.slice(..).map_async(MapMode::Read, |_| {});
    device
        .poll(PollType::wait_indefinitely())
        .expect("device poll should complete the submitted work");

    let view = stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped readback range should be available after poll");
    let values: Vec<[f32; 4]> = bytemuck::cast_slice::<u8, [f32; 4]>(&view).to_vec();
    drop(view);
    stage.unmap();
    values
}

/// 在给定设备上跑两段式气动 dispatch，逐顶点与 CPU 黄金对拍（容差 `PARITY_EPS`）。
///
/// 抽出 `wind` / `aero` / `dt` 三个入参，供线性与二次（`air_density > 0`）两条
/// 模型分别驱动同一条真机 dispatch + 对拍路径，避免重复。
fn run_on_device_parity(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wind: WindField,
    aero: AeroParams,
    dt: f32,
) {
    const NX: usize = 12;
    const NZ: usize = 10;
    let (particles, triangles) = build_grid(NX, NZ);
    let count = particles.len();
    let adjacency = VertexTriangleAdjacency::build(count, &triangles);

    // CPU 黄金：在副本上原地累加。
    let mut golden = particles.clone();
    accumulate_aero_gather(&mut golden, &triangles, &adjacency, &wind, aero, dt);

    // host 上传口径：positions.w = inverse mass；velocities.w 塞可辨识载荷位以
    // 验证内核不动它；扁平三角 + CSR。
    let positions: Vec<[f32; 4]> = particles
        .iter()
        .map(|p| [p.position.x, p.position.y, p.position.z, p.inverse_mass])
        .collect();
    let velocities: Vec<[f32; 4]> = particles
        .iter()
        .enumerate()
        .map(|(i, p)| [p.velocity.x, p.velocity.y, p.velocity.z, (i as f32) + 0.5])
        .collect();
    let flat: Vec<u32> = triangles.iter().flat_map(|t| t.iter().copied()).collect();

    // 参数按 solve_plan 口径 sanitize 后打包（含二次模型的 air_density）。
    let field = wind.sanitized();
    let clean = aero.sanitized();
    let params = GpuClothAeroParams {
        wind: [field.velocity.x, field.velocity.y, field.velocity.z],
        turbulence: field.turbulence,
        drag: clean.drag,
        lift: clean.lift,
        dt,
        particle_count: count as u32,
        air_density: clean.air_density,
        _pad_aero: [0.0; 3],
    };

    let snapshot_wgsl = compile_wgsl(
        include_str!("../shaders/cloth_aerodynamics_snapshot.wesl"),
        "embedded://prism_render_scene/shaders/cloth_aerodynamics_snapshot.wesl",
        0x434c_4f54_485f_4145_524f_5f53_4e41_5001,
    );
    let snapshot_entry = find_entry_point(&snapshot_wgsl, "cloth_aerodynamics_snapshot");
    let gather_wgsl = compile_wgsl(
        include_str!("../shaders/cloth_aerodynamics.wesl"),
        "embedded://prism_render_scene/shaders/cloth_aerodynamics.wesl",
        0x434c_4f54_485f_4145_524f_5f47_4154_5001,
    );
    let gather_entry = find_entry_point(&gather_wgsl, "cloth_aerodynamics");

    let result = dispatch_aero(
        device,
        queue,
        &snapshot_wgsl,
        &snapshot_entry,
        &gather_wgsl,
        &gather_entry,
        &positions,
        &velocities,
        &flat,
        adjacency.offsets(),
        adjacency.entries(),
        &params,
    );

    assert_eq!(
        result.len(),
        count,
        "readback length must match vertex count"
    );
    for (i, g) in golden.iter().enumerate() {
        let r = result[i];
        let dx = (r[0] - g.velocity.x).abs();
        let dy = (r[1] - g.velocity.y).abs();
        let dz = (r[2] - g.velocity.z).abs();
        assert!(
            dx <= PARITY_EPS && dy <= PARITY_EPS && dz <= PARITY_EPS,
            "vertex {i}: GPU velocity ({}, {}, {}) drifted from golden ({}, {}, {}) beyond {PARITY_EPS}",
            r[0], r[1], r[2], g.velocity.x, g.velocity.y, g.velocity.z
        );
        // `.w` 载荷位必须原样保留。
        assert_eq!(
            r[3].to_bits(),
            ((i as f32) + 0.5).to_bits(),
            "vertex {i}: payload .w was mutated by the GPU kernels"
        );
    }
}

/// 线性气动模型（`air_density = 0`）在真机上必须与 CPU 黄金落在 `float32` 舍入容差内。
#[test]
#[expect(
    clippy::print_stderr,
    reason = "无 wgpu adapter 的主机上，跳过提示需要进入测试日志"
)]
fn aero_gpu_matches_cpu_golden() {
    let Some((device, queue)) = try_solver_device() else {
        eprintln!("aero_gpu_matches_cpu_golden: no wgpu adapter, skipping on-device parity");
        return;
    };
    let wind = WindField::new(Vec3::new(2.5, -0.4, 1.3), 0.35);
    let aero = AeroParams::new(1.2, 0.6);
    run_on_device_parity(&device, &queue, wind, aero, 1.0 / 60.0);
}

/// 二次（UE5 `Chaos` 风格 airspeed²）气动模型（`air_density > 0`）在真机上同样
/// 必须与 CPU 黄金逐顶点吻合，验证 `air_density` 经 48 字节 uniform 正确抵达
/// GPU 并驱动动压分支。
#[test]
#[expect(
    clippy::print_stderr,
    reason = "无 wgpu adapter 的主机上，跳过提示需要进入测试日志"
)]
fn aero_gpu_quadratic_matches_cpu_golden() {
    let Some((device, queue)) = try_solver_device() else {
        eprintln!(
            "aero_gpu_quadratic_matches_cpu_golden: no wgpu adapter, skipping on-device parity"
        );
        return;
    };
    let wind = WindField::new(Vec3::new(2.5, -0.4, 1.3), 0.35);
    let aero = AeroParams::new(1.2, 0.6).with_air_density(1.225);
    run_on_device_parity(&device, &queue, wind, aero, 1.0 / 60.0);
}
