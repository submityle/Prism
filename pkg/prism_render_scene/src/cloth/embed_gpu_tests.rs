//! `cloth_embed.wesl` 的 `cloth_skin_embed` 内核**真机 GPU** parity 覆盖。
//!
//! `ClothKernel::SkinEmbed` 把高分辨率渲染网格的每个顶点冻结进一枚 sim 三角形：读出该
//! 顶点的宿主三角索引、重心权重与带符号法线偏移，取回三枚当前 sim 位置，重建面内点
//! `p0*w0 + p1*w1 + p2*w2`，再沿重估的面法线按偏移推出，从而在跟随形变与朝向的同时
//! 复原服装厚度。此前它只被 `shader_tests` 证明「能被 `naga` 编译」，从未在真实设备上
//! dispatch 过；而 [`super::pack::pack_embed_bindings`] 这条从作者态
//! [`BarycentricBinding`](prism_render_architecture::cloth::embed::BarycentricBinding) 到设备
//! 记录 [`GpuClothEmbedBinding`](super::abi::GpuClothEmbedBinding) 的 host 打包桥，也需要一条
//! 端到端真机对拍证明字段映射与内核读取逐比特一致。本模块闭合这两条缺口。
//!
//! 做法忠实复用权威路径而非重写算法：
//! - **打包**走生产用的 [`super::pack::pack_embed_bindings`]，把作者态绑定转成设备记录，
//!   与 prepare 阶段上传的字节完全一致。
//! - **黄金**来自架构层
//!   [`embed_render_mesh`](prism_render_architecture::cloth::embed::embed_render_mesh)：
//!   `bindings[i]` 驱动渲染顶点 `i`，越界索引降级为 `Vec3::ZERO`，退化（零面积）宿主三角
//!   贡献零法线（偏移项自然消失），故无 `NaN` 能进入渲染流。
//!
//! ## 为什么是 bit-parity（`float32` 舍入内）
//!
//! embed 是一段纯逐顶点重建：每个渲染顶点的结果只依赖自己的绑定与其宿主三角的三枚
//! 只读 sim 位置，顶点间零共享 ⇒ GPU 并行 invocation 与 CPU 顺序遍历产生**完全相同**的
//! 写集。唯一自由度是面法线归一化里 CPU 的 `normalize_or_zero`（`1.0 / sqrt`）与 WESL 的
//! `inverseSqrt`（多数 GPU 是原生 `rsqrt`，可能差几个 ULP），[`PARITY_EPS`] 只吸收这一项。
//!
//! ## `w` 通道直通
//!
//! 内核只改写 `render_positions[i].xyz`，`.w` 原样保留（渲染网格常在 `w` 里携带如顶点
//! id / uv 打包等元数据）。各用例给渲染缓冲预置一枚哨兵 `w`，回读后断言其不被扰动。
//!
//! 取设备是尽力而为：无 `wgpu` adapter 的无头机上 [`try_compute_device`] 返回 `None`，
//! 测试打印跳过提示而非失败，让套件在任何机器上保持绿。

use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor, MapMode,
    PipelineCompilationOptions, PipelineLayoutDescriptor, PollType, ShaderModuleDescriptor,
    ShaderSource, ShaderStages,
};

use prism_render_architecture::cloth::embed::{embed_render_mesh, BarycentricBinding};
use prism_render_architecture::cloth::Vec3;

use super::abi::{GpuClothEmbedBinding, GpuClothEmbedParams};
use super::gpu_test_support::{
    compile_embed_wgsl, find_entry_point, storage_from_slice, try_compute_device, PARITY_EPS,
};
use super::pack::pack_embed_bindings;

/// `cloth_skin_embed` 的 group-0 布局：`sim_positions`（ro storage）、`render_positions`
/// （rw storage）、`embed_bindings`（ro storage）、`params`（uniform）。与 WESL 的
/// `@binding(0..3)` 一一对应。
fn embed_bind_group_layout(device: &wgpu::Device) -> BindGroupLayout {
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
        label: Some("cloth_embed_parity_group0"),
        entries: &[
            storage(0, true),
            storage(1, false),
            storage(2, true),
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

/// 在真机上跑一遍 `cloth_skin_embed`，读回嵌入后的 `render_positions`（`xyzw`）。
///
/// `render_vertex_count` 取渲染缓冲长度（也即绑定数，两者 index 对齐），作 dispatch
/// 宽度。`render_seed` 携带每个渲染顶点预置的 `xyzw`——`xyz` 会被内核覆写、`w` 应原样
/// 保留，供上层断言 `w` 直通。
fn replay_embed_on_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    entry: &str,
    sim_positions: &[[f32; 4]],
    render_seed: &[[f32; 4]],
    bindings: &[GpuClothEmbedBinding],
) -> Vec<[f32; 4]> {
    let render_bytes = size_of_val(render_seed) as u64;

    let layout = embed_bind_group_layout(device);
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("cloth_embed_parity_layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("cloth_embed_parity"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("cloth_embed_parity_pipeline"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some(entry),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });

    let sim_buf = storage_from_slice(
        device,
        "cloth_embed_sim_positions",
        sim_positions,
        [0.0_f32; 4],
    );
    let render_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_embed_render_positions"),
        contents: bytemuck::cast_slice(render_seed),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });
    let bindings_buf = storage_from_slice(
        device,
        "cloth_embed_bindings",
        bindings,
        GpuClothEmbedBinding::default(),
    );
    let params = GpuClothEmbedParams {
        render_vertex_count: render_seed.len() as u32,
        _pad: [0, 0, 0],
    };
    let params_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_embed_params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("cloth_embed_parity_bind"),
        layout: &layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: sim_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: render_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: bindings_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let stage = device.create_buffer(&BufferDescriptor {
        label: Some("cloth_embed_render_stage"),
        size: render_bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("cloth_embed_parity_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("cloth_embed_parity_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        // 每 invocation 一个渲染顶点，@workgroup_size(64) ⇒ ceil(count / 64) 组。
        let groups = render_seed.len().div_ceil(64).max(1) as u32;
        pass.dispatch_workgroups(groups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&render_buf, 0, &stage, 0, render_bytes);
    queue.submit([encoder.finish()]);

    stage.slice(..).map_async(MapMode::Read, |_| {});
    device
        .poll(PollType::wait_indefinitely())
        .expect("device poll should complete the submitted work");
    let view = stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped render readback range should be available after poll");
    let out: Vec<[f32; 4]> = bytemuck::cast_slice::<u8, [f32; 4]>(&view).to_vec();
    drop(view);
    stage.unmap();
    out
}

/// 对一组 sim 位置与作者态渲染绑定，同时跑 CPU 黄金 [`embed_render_mesh`] 与真机
/// `cloth_skin_embed`，断言逐渲染顶点 `xyz` 落在 [`PARITY_EPS`] 内、`w`（哨兵）不被
/// 内核改动。
fn assert_embed_parity(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    entry: &str,
    sim: &[Vec3],
    bindings: &[BarycentricBinding],
) {
    // --- CPU 黄金 ---
    let mut golden = Vec::new();
    embed_render_mesh(bindings, sim, &mut golden);

    // --- host 上传口径 ---
    let sim_positions: Vec<[f32; 4]> = sim.iter().map(|p| [p.x, p.y, p.z, 0.0]).collect();
    // 每个渲染顶点预置一枚可辨识的哨兵 w，回读后据此核验 w 直通。
    let render_seed: Vec<[f32; 4]> = (0..bindings.len())
        .map(|i| [0.0, 0.0, 0.0, 100.0 + i as f32])
        .collect();
    let packed = pack_embed_bindings(bindings);
    let out = replay_embed_on_gpu(
        device,
        queue,
        wgsl,
        entry,
        &sim_positions,
        &render_seed,
        &packed,
    );

    assert_eq!(out.len(), golden.len());
    for (i, (gpu, cpu)) in out.iter().zip(golden.iter()).enumerate() {
        assert!(
            (gpu[0] - cpu.x).abs() <= PARITY_EPS
                && (gpu[1] - cpu.y).abs() <= PARITY_EPS
                && (gpu[2] - cpu.z).abs() <= PARITY_EPS,
            "render vertex {i}: gpu [{}, {}, {}] vs cpu [{}, {}, {}]",
            gpu[0],
            gpu[1],
            gpu[2],
            cpu.x,
            cpu.y,
            cpu.z,
        );
        // w 是内核直通字段，绝不能被嵌入改动。
        let seed_w = 100.0 + i as f32;
        assert!(
            (gpu[3] - seed_w).abs() <= f32::EPSILON,
            "render vertex {i}: w channel mutated {seed_w} -> {}",
            gpu[3],
        );
    }
}

/// 一枚非退化宿主三角（`xy` 平面上的直角三角）。
fn host_triangle() -> [Vec3; 3] {
    [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(2.0, 0.0, 0.0),
        Vec3::new(0.0, 2.0, 0.0),
    ]
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "无 GPU 的无头机上打印跳过提示而非失败，让套件保持绿"
)]
fn embed_reconstructs_render_vertices_on_gpu() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("skip: no compute device for cloth_skin_embed parity");
        return;
    };
    let wgsl = compile_embed_wgsl();
    let entry = find_entry_point(&wgsl, "skin_embed");

    let sim = host_triangle().to_vec();
    // 面内质心（无偏移）、单顶点（w0=1）、以及带正/负法线偏移的一般点。
    let bindings = vec![
        BarycentricBinding::new([0, 1, 2], (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0), 0.0),
        BarycentricBinding::new([0, 1, 2], (1.0, 0.0, 0.0), 0.5),
        BarycentricBinding::new([0, 1, 2], (0.2, 0.3, 0.5), -0.75),
        BarycentricBinding::new([0, 1, 2], (0.5, 0.25, 0.25), 1.25),
    ];
    assert_embed_parity(&device, &queue, &wgsl, &entry, &sim, &bindings);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "无 GPU 的无头机上打印跳过提示而非失败，让套件保持绿"
)]
fn embed_degenerate_host_triangle_drops_offset_on_gpu() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("skip: no compute device for cloth_skin_embed parity");
        return;
    };
    let wgsl = compile_embed_wgsl();
    let entry = find_entry_point(&wgsl, "skin_embed");

    // 三点共线 ⇒ 零面积 ⇒ 面法线为零 ⇒ 偏移项消失，仅剩重心加权和（CPU/GPU 同）。
    let sim = vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(2.0, 0.0, 0.0),
    ];
    let bindings = vec![BarycentricBinding::new([0, 1, 2], (0.25, 0.25, 0.5), 5.0)];
    assert_embed_parity(&device, &queue, &wgsl, &entry, &sim, &bindings);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "无 GPU 的无头机上打印跳过提示而非失败，让套件保持绿"
)]
fn embed_out_of_range_index_degrades_to_zero_on_gpu() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("skip: no compute device for cloth_skin_embed parity");
        return;
    };
    let wgsl = compile_embed_wgsl();
    let entry = find_entry_point(&wgsl, "skin_embed");

    // 只有 3 枚 sim 位置；第二个绑定引用索引 9（越界）⇒ CPU/GPU 均写零 xyz、保 w。
    let sim = host_triangle().to_vec();
    let bindings = vec![
        BarycentricBinding::new([0, 1, 2], (0.3, 0.3, 0.4), 0.2),
        BarycentricBinding::new([0, 9, 2], (0.5, 0.25, 0.25), 0.2),
    ];
    assert_embed_parity(&device, &queue, &wgsl, &entry, &sim, &bindings);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "无 GPU 的无头机上打印跳过提示而非失败，让套件保持绿"
)]
fn embed_deformed_host_triangle_tracks_orientation_on_gpu() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("skip: no compute device for cloth_skin_embed parity");
        return;
    };
    let wgsl = compile_embed_wgsl();
    let entry = find_entry_point(&wgsl, "skin_embed");

    // 一枚倾斜（非轴对齐）三角，法线既非 +Z 也非单位轴，确保面法线重估路径被覆盖。
    let sim = vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 2.0, 0.5),
        Vec3::new(-0.5, 1.0, 1.5),
    ];
    let bindings = vec![
        BarycentricBinding::new([0, 1, 2], (0.4, 0.35, 0.25), 0.3),
        BarycentricBinding::new([0, 1, 2], (0.1, 0.1, 0.8), -0.6),
    ];
    assert_embed_parity(&device, &queue, &wgsl, &entry, &sim, &bindings);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "无 GPU 的无头机上打印跳过提示而非失败，让套件保持绿"
)]
fn empty_render_mesh_is_noop_on_gpu() {
    let Some((_device, _queue)) = try_compute_device() else {
        eprintln!("skip: no compute device for cloth_skin_embed parity");
        return;
    };
    // 空渲染网格：CPU 黄金输出空、GPU 无顶点可 dispatch，两侧同为诚实 no-op。
    // 直接核验 CPU 黄金为空即可（无渲染顶点则无字节可上传/回读）。
    let sim = host_triangle().to_vec();
    let bindings: Vec<BarycentricBinding> = Vec::new();
    let mut golden = Vec::new();
    embed_render_mesh(&bindings, &sim, &mut golden);
    assert!(golden.is_empty());
    assert!(pack_embed_bindings(&bindings).is_empty());
}
