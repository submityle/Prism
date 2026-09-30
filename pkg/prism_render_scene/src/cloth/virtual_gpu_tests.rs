//! `cloth_self_collision_virtual.wesl` 的三段虚拟粒子（`NvCloth` 风格）
//! 自碰撞内核的**真机 GPU** 覆盖：`cloth_vp_hash_build` / `cloth_vp_resolve` /
//! `cloth_vp_scatter`。
//!
//! 点对点自碰撞（`cloth_collision.wesl`）有一个结构性盲区（设计 §6.2）：一个
//! 孤立顶点可以从一枚大三角形的**内部**穿过，而始终不靠近该三角形的任一角点到
//! `thickness` 之内，于是点对点段永远检测不到它。虚拟粒子段在每个三角形上撒几枚
//! 重心采样点（此处用质心），把真实顶点与虚拟点折进**同一套**均匀空间哈希：一个
//! 下潜的顶点因此会在它正穿过的那个面上撞到一枚虚拟粒子，被推回、并按重心权重把
//! 修正散射回该面的三个真实顶点。
//!
//! 此前这套着色器只被 `shader_tests` 证明「能被 `naga` 编译」，从未在真实设备
//! 上 dispatch 过。本模块闭合该缺口：在真机 Metal 上跑完整的
//! build → resolve → scatter 三段，回读位置，与 CPU 黄金
//! [`resolve_self_collision_virtual_jacobi`] 逐顶点对拍。
//!
//! ## 为什么这些用例可对拍 CPU 黄金（`float32` 舍入内）
//!
//! CPU 黄金本身就是 **Jacobi**（own-slot：每个采样从冻结快照累加、只写自己那槽），
//! 与 GPU 内核同构。二者仅有的自由度是归约顺序——Phase 1 里一枚采样对其穿透邻居
//! 求和、Phase 2 里一个顶点对其入射采样求和。哈希 build 用 `atomicExchange`，桶内
//! 链表次序不定，故当某枚采样有 **多个** 穿透邻居时，两条路径的求和次序发散。为把
//! 断言锁在逐比特可复现的射程内，本模块只取**单接触**几何（每枚采样至多一个穿透
//! 邻居），此时求和只有一项、与次序无关，[`PARITY_EPS`] 只需吸收
//! `1.0 / sqrt` 与 `inverseSqrt` 之间的几个 ULP 之差。
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

use prism_render_architecture::cloth::virtual_particles::{
    generate_virtual_particles, VirtualParticle, VirtualParticlePattern,
};
use prism_render_architecture::cloth::virtual_particles_jacobi::{
    resolve_self_collision_virtual_augment_jacobi, resolve_self_collision_virtual_jacobi,
};
use prism_render_architecture::cloth::{ClothParticle, Vec3};

use super::abi::{GpuClothHashCell, GpuClothVpParams, GpuClothVpSample};
use super::gpu_test_support::{compile_virtual_wgsl, find_entry_point, try_compute_device, PARITY_EPS};

/// 链表终止哨兵，与 `cloth_self_collision_virtual.wesl` 的 `CLOTH_VP_SENTINEL` 一致。
const SENTINEL: u32 = 0xffff_ffffu32;

/// 建一个自由粒子（`inverse_mass = 1`）。
fn free(x: f32, y: f32, z: f32) -> ClothParticle {
    ClothParticle::new(Vec3::new(x, y, z), 1.0)
}

/// 建一个 pinned 粒子（`inverse_mass = 0`，永不移动）。
fn pinned(x: f32, y: f32, z: f32) -> ClothParticle {
    ClothParticle::pinned(Vec3::new(x, y, z))
}

/// host 上传口径：`positions.w = inverse mass`（`<= 0` 即 pinned）。
fn upload_positions(particles: &[ClothParticle]) -> Vec<[f32; 4]> {
    particles
        .iter()
        .map(|p| [p.position.x, p.position.y, p.position.z, p.inverse_mass])
        .collect()
}

/// 按 CPU 黄金的固定采样次序构建 GPU 采样表：先全体真实顶点
/// （`verts = (i, i, i)`、`weights = (1, 0, 0)`），再按生成序追加落在真实顶点
/// 范围内的虚拟粒子。与 `accumulate_virtual_jacobi_corrections` 的
/// reals-then-virtuals 打包完全一致。
fn build_samples(real_count: usize, virtuals: &[VirtualParticle]) -> Vec<GpuClothVpSample> {
    let mut samples = Vec::with_capacity(real_count.saturating_add(virtuals.len()));
    for i in 0..real_count {
        samples.push(GpuClothVpSample {
            v0: i as u32,
            v1: i as u32,
            v2: i as u32,
            w0: 1.0,
            w1: 0.0,
            w2: 0.0,
            _pad0: 0.0,
            _pad1: 0.0,
        });
    }
    for vp in virtuals {
        if vp.verts.iter().all(|&v| (v as usize) < real_count) {
            samples.push(GpuClothVpSample {
                v0: vp.verts[0],
                v1: vp.verts[1],
                v2: vp.verts[2],
                w0: vp.weights[0],
                w1: vp.weights[1],
                w2: vp.weights[2],
                _pad0: 0.0,
                _pad1: 0.0,
            });
        }
    }
    samples
}

/// 构建 scatter 段读取的 CSR 入射表：`offsets[v]..offsets[v + 1]` 索引进 `entries`，
/// 列出把顶点 `v` 作为**带正权重**活跃角点的每一枚采样。同一采样对同一顶点只入表
/// 一次（着色器在单次访问内自行把该采样多个匹配角点的权重求和），与
/// `cloth_aerodynamics.wesl` 的逐顶点三角形邻接同构。
fn build_csr(samples: &[GpuClothVpSample], vertex_count: usize) -> (Vec<u32>, Vec<u32>) {
    let mut per_vertex: Vec<Vec<u32>> = alloc_rows(vertex_count);
    for (a, s) in samples.iter().enumerate() {
        let corners = [(s.v0, s.w0), (s.v1, s.w1), (s.v2, s.w2)];
        let mut seen: Vec<u32> = Vec::new();
        for (v, w) in corners {
            if w > 0.0 && (v as usize) < vertex_count && !seen.contains(&v) {
                seen.push(v);
                per_vertex[v as usize].push(a as u32);
            }
        }
    }
    let mut offsets = Vec::with_capacity(vertex_count + 1);
    let mut entries = Vec::new();
    offsets.push(0u32);
    for row in per_vertex.iter().take(vertex_count) {
        entries.extend_from_slice(row);
        offsets.push(entries.len() as u32);
    }
    (offsets, entries)
}

/// `vec![Vec::new(); n]` 的等价物，避开对 `Vec<u32>: Clone` 逐行浅拷贝的歧义。
fn alloc_rows(n: usize) -> Vec<Vec<u32>> {
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        rows.push(Vec::new());
    }
    rows
}

/// 三段虚拟粒子自碰撞的 group-0 布局：`vp_positions`（rw）、`vp_samples`（ro）、
/// `vp_cell_table`（rw）、`vp_sample_next`（rw）、`vp_sample_dp`（rw）、
/// `vp_params`（uniform）、`vp_csr_offsets`（ro）、`vp_csr_entries`（ro），与 WESL
/// 的 `@binding(0..7)` 一一对应。
fn vp_bind_group_layout(device: &wgpu::Device) -> BindGroupLayout {
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
        label: Some("cloth_vp_parity_group0"),
        entries: &[
            storage(0, false),
            storage(1, true),
            storage(2, false),
            storage(3, false),
            storage(4, false),
            BindGroupLayoutEntry {
                binding: 5,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            storage(6, true),
            storage(7, true),
        ],
    })
}

/// 在真机上跑 build → resolve → scatter 三段，回读并返回最终的粒子位置
/// （`xyzw`，`w` = 逆质量）。
#[expect(
    clippy::too_many_arguments,
    reason = "parity replay 需要几何、采样与网格参数全部显式传入，聚成结构体反而分散阅读"
)]
fn run_vp_on_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    entries: &VpEntryPoints,
    positions: &[[f32; 4]],
    samples: &[GpuClothVpSample],
    csr_offsets: &[u32],
    csr_entries: &[u32],
    params: GpuClothVpParams,
) -> Vec<[f32; 4]> {
    let layout = vp_bind_group_layout(device);
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("cloth_vp_parity_pipeline_layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("cloth_vp_parity"),
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
    let build_pipeline = make_pipeline("cloth_vp_build_pipeline", &entries.build);
    let resolve_pipeline = make_pipeline("cloth_vp_resolve_pipeline", &entries.resolve);
    let scatter_pipeline = make_pipeline("cloth_vp_scatter_pipeline", &entries.scatter);

    let cell_init = vec![
        GpuClothHashCell {
            head: SENTINEL,
            count: 0,
        };
        params.table_size as usize
    ];
    let next_init = vec![SENTINEL; samples.len()];
    let dp_init = vec![[0.0f32; 4]; samples.len()];

    let positions_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_vp_positions"),
        contents: bytemuck::cast_slice(positions),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });
    let samples_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_vp_samples"),
        contents: bytemuck::cast_slice(samples),
        usage: BufferUsages::STORAGE,
    });
    let cell_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_vp_cell_table"),
        contents: bytemuck::cast_slice(&cell_init),
        usage: BufferUsages::STORAGE,
    });
    let next_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_vp_sample_next"),
        contents: bytemuck::cast_slice(&next_init),
        usage: BufferUsages::STORAGE,
    });
    let dp_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_vp_sample_dp"),
        contents: bytemuck::cast_slice(&dp_init),
        usage: BufferUsages::STORAGE,
    });
    let params_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_vp_params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });
    let offsets_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_vp_csr_offsets"),
        contents: bytemuck::cast_slice(csr_offsets),
        usage: BufferUsages::STORAGE,
    });
    let entries_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_vp_csr_entries"),
        contents: bytemuck::cast_slice(csr_entries),
        usage: BufferUsages::STORAGE,
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("cloth_vp_parity_bind"),
        layout: &layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: positions_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: samples_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: cell_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: next_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: dp_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 5,
                resource: params_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 6,
                resource: offsets_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 7,
                resource: entries_buf.as_entire_binding(),
            },
        ],
    });

    let pos_bytes = size_of_val(positions) as u64;
    let pos_stage = device.create_buffer(&BufferDescriptor {
        label: Some("cloth_vp_positions_stage"),
        size: pos_bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("cloth_vp_parity_encoder"),
    });
    let sample_groups = params.sample_count.div_ceil(64).max(1);
    let vertex_groups = params.vertex_count.div_ceil(64).max(1);
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("cloth_vp_build_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&build_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(sample_groups, 1, 1);
    }
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("cloth_vp_resolve_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&resolve_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(sample_groups, 1, 1);
    }
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("cloth_vp_scatter_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&scatter_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(vertex_groups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&positions_buf, 0, &pos_stage, 0, pos_bytes);
    queue.submit([encoder.finish()]);

    pos_stage.slice(..).map_async(MapMode::Read, |_| {});
    device
        .poll(PollType::wait_indefinitely())
        .expect("device poll should complete the submitted virtual-particle work");

    let view = pos_stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped position readback range should be available after poll");
    let out: Vec<[f32; 4]> = bytemuck::cast_slice::<u8, [f32; 4]>(&view).to_vec();
    drop(view);
    pos_stage.unmap();
    out
}

/// build / resolve / scatter 三个入口在编译产物里的真实符号名。
struct VpEntryPoints {
    build: String,
    resolve: String,
    scatter: String,
}

fn vp_entry_points(wgsl: &str) -> VpEntryPoints {
    VpEntryPoints {
        build: find_entry_point(wgsl, "cloth_vp_hash_build"),
        resolve: find_entry_point(wgsl, "cloth_vp_resolve"),
        scatter: find_entry_point(wgsl, "cloth_vp_scatter"),
    }
}

/// 是否为增量模式（`virtual_only`）：跳过真实-真实对，只解算触及虚拟粒子的对。
#[derive(Clone, Copy)]
enum Scope {
    All,
    VirtualOnly,
}

/// 同时跑 CPU 黄金与真机三段，断言逐顶点 `xyz` 落在 [`PARITY_EPS`] 内、
/// `w`（逆质量）不被内核改动。仅用于**单接触**几何。
fn assert_vp_parity(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    particles: &[ClothParticle],
    triangles: &[[u32; 3]],
    pattern: &VirtualParticlePattern,
    cell_size: f32,
    thickness: f32,
    scope: Scope,
) {
    let virtuals = generate_virtual_particles(triangles, pattern);

    let mut golden = particles.to_vec();
    match scope {
        Scope::All => {
            resolve_self_collision_virtual_jacobi(&mut golden, &virtuals, cell_size, thickness);
        }
        Scope::VirtualOnly => {
            resolve_self_collision_virtual_augment_jacobi(
                &mut golden,
                &virtuals,
                cell_size,
                thickness,
            );
        }
    }

    let real_count = particles.len();
    let samples = build_samples(real_count, &virtuals);
    let (csr_offsets, csr_entries) = build_csr(&samples, real_count);
    let params = GpuClothVpParams {
        sample_count: samples.len() as u32,
        real_count: real_count as u32,
        vertex_count: real_count as u32,
        table_size: 97,
        cell_size,
        thickness,
        virtual_only: match scope {
            Scope::All => 0,
            Scope::VirtualOnly => 1,
        },
        _pad: 0,
    };

    let positions = upload_positions(particles);
    let entry_points = vp_entry_points(wgsl);
    let readback = run_vp_on_gpu(
        device,
        queue,
        wgsl,
        &entry_points,
        &positions,
        &samples,
        &csr_offsets,
        &csr_entries,
        params,
    );

    assert_eq!(readback.len(), golden.len());
    for (i, (gpu, cpu)) in readback.iter().zip(golden.iter()).enumerate() {
        let expected = cpu.position;
        assert!(
            (gpu[0] - expected.x).abs() <= PARITY_EPS
                && (gpu[1] - expected.y).abs() <= PARITY_EPS
                && (gpu[2] - expected.z).abs() <= PARITY_EPS,
            "vertex {i}: gpu [{}, {}, {}] vs cpu [{}, {}, {}]",
            gpu[0],
            gpu[1],
            gpu[2],
            expected.x,
            expected.y,
            expected.z,
        );
        assert!(
            (gpu[3] - particles[i].inverse_mass).abs() <= f32::EPSILON,
            "vertex {i}: inverse mass mutated {} -> {}",
            particles[i].inverse_mass,
            gpu[3],
        );
    }
}

/// 只有质心一枚虚拟粒子的图案：把每三角形的采样锁成单枚，保证下潜顶点至多一个
/// 穿透邻居（单接触），从而 Jacobi 归约与次序无关、可逐比特对拍。
fn centroid_pattern() -> VirtualParticlePattern {
    VirtualParticlePattern::from_weights(&[[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]])
}

/// 一枚大三角形 + 正对其质心、`z` 方向仅 0.05 之隔的孤立顶点：点对点段会漏掉它，
/// 虚拟粒子段在质心撞到它并沿 `+z` 推开，修正按重心散射回三个角点。单接触。
fn vertex_through_triangle() -> [ClothParticle; 4] {
    [
        free(0.0, 0.0, 0.0),
        free(2.0, 0.0, 0.0),
        free(0.0, 2.0, 0.0),
        free(2.0 / 3.0, 2.0 / 3.0, 0.05),
    ]
}

/// 三段联跑：孤立顶点被三角形质心的虚拟粒子推开，散射回三角形三角点，逐顶点对拍
/// CPU Jacobi 黄金。
#[test]
#[expect(
    clippy::print_stderr,
    reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志"
)]
fn vertex_diving_through_triangle_is_pushed_out() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("vertex_diving_through_triangle_is_pushed_out: no wgpu adapter, skipping");
        return;
    };
    let wgsl = compile_virtual_wgsl();
    let particles = vertex_through_triangle();
    assert_vp_parity(
        &device,
        &queue,
        &wgsl,
        &particles,
        &[[0, 1, 2]],
        &centroid_pattern(),
        1.0,
        0.5,
        Scope::All,
    );
}

/// 增量模式：跳过真实-真实对，只解算触及虚拟粒子的对，结果仍与增量黄金逐顶点一致。
#[test]
#[expect(
    clippy::print_stderr,
    reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志"
)]
fn vertex_diving_through_triangle_augment_mode() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("vertex_diving_through_triangle_augment_mode: no wgpu adapter, skipping");
        return;
    };
    let wgsl = compile_virtual_wgsl();
    let particles = vertex_through_triangle();
    assert_vp_parity(
        &device,
        &queue,
        &wgsl,
        &particles,
        &[[0, 1, 2]],
        &centroid_pattern(),
        1.0,
        0.5,
        Scope::VirtualOnly,
    );
}

/// pinned 下潜顶点：`eff = 0` 令其在 Phase 1 不受力、Phase 2 被 scatter 跳过，
/// 全部分离量由三角形三个自由角点独吞，逐顶点对拍。
#[test]
#[expect(
    clippy::print_stderr,
    reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志"
)]
fn pinned_diving_vertex_pushes_only_the_triangle() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("pinned_diving_vertex_pushes_only_the_triangle: no wgpu adapter, skipping");
        return;
    };
    let wgsl = compile_virtual_wgsl();
    let particles = [
        free(0.0, 0.0, 0.0),
        free(2.0, 0.0, 0.0),
        free(0.0, 2.0, 0.0),
        pinned(2.0 / 3.0, 2.0 / 3.0, 0.05),
    ];
    assert_vp_parity(
        &device,
        &queue,
        &wgsl,
        &particles,
        &[[0, 1, 2]],
        &centroid_pattern(),
        1.0,
        0.5,
        Scope::All,
    );
}

/// 相距甚远的两枚三角形：所有采样对都超过 `thickness`，三段联跑是逐比特 no-op，
/// 回读位置必须与输入完全一致。
#[test]
#[expect(
    clippy::print_stderr,
    reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志"
)]
fn distant_triangles_are_a_no_op() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("distant_triangles_are_a_no_op: no wgpu adapter, skipping");
        return;
    };
    let wgsl = compile_virtual_wgsl();
    let particles = [
        free(0.0, 0.0, 0.0),
        free(1.0, 0.0, 0.0),
        free(0.0, 1.0, 0.0),
        free(50.0, 50.0, 50.0),
        free(51.0, 50.0, 50.0),
        free(50.0, 51.0, 50.0),
    ];
    assert_vp_parity(
        &device,
        &queue,
        &wgsl,
        &particles,
        &[[0, 1, 2], [3, 4, 5]],
        &centroid_pattern(),
        1.0,
        0.5,
        Scope::All,
    );
}
