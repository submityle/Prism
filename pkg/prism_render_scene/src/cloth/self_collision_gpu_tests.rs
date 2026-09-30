//! `cloth_collision.wesl` 的 `cloth_self_collision_hash_build` /
//! `cloth_self_collision_resolve` 两个内核的**真机 GPU** 覆盖。
//!
//! 自碰撞是布料防穿插的核心一段：先把每个粒子按均匀空间哈希插进它所在格子的链表
//! （build），再让每个粒子读遍 27 邻域链表、按逆质量把过近的邻居推开（resolve）。
//! 此前这两段只被 `shader_tests` 证明「能被 `naga` 编译」，从未在真实设备上
//! dispatch 过。本模块闭合该缺口，并锁死两处刚修好的正确性问题：
//!
//! - **build 必须逐粒子恰好插一次**：内核现为扁平 `@workgroup_size(64)` 的
//!   per-particle launch（`p = gid.x`）。旧的 `@workgroup_size(4, 4, 4)` 会把同一
//!   `gid.x` 映射到 16 个 `(y, z)` invocation，导致每个粒子被原子插链 16 次。
//!   [`hash_build_chains_each_particle_into_its_bucket_once`] 读回链表，断言每个桶
//!   的成员集合与 `count` 与期望一致，且全体粒子合计恰好各出现一次。
//! - **coincident 对称分离**：resolve 的重合分支现按索引定符号（低索引走 `-X`、
//!   高索引走 `+X`），复刻 CPU 黄金 `resolve_pair` 的带符号半分，两个重合自由粒子
//!   因此对称分开而非同向漂移仍旧重合。
//!
//! ## 为什么单接触场景可对拍 CPU 黄金（`float32` 舍入内）
//!
//! CPU 黄金 [`resolve_self_collision`] 是有序 Gauss-Seidel（原地更新），而 GPU
//! resolve 是 Jacobi（每粒子从原始位置累加所有邻居分离量、只写自己）——见着色器
//! 模块的 determinism note。二者只有在**每个粒子至多一个接触邻居**时逐比特一致，
//! 故本模块的 resolve 对拍全部取单对几何（复用 CPU 黄金自身的用例），此时 Jacobi
//! 与 Gauss-Seidel 写集完全相同，[`PARITY_EPS`] 只吸收归一化里 `1.0 / sqrt` 与
//! `inverseSqrt` 的几个 ULP 之差。多邻居会让两条路径发散，那不是内核 bug，故不在
//! 单 pass parity 断言的射程内。
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

use prism_render_architecture::cloth::collision::resolve_self_collision;
use prism_render_architecture::cloth::{ClothParticle, Vec3};

use super::abi::{GpuClothHashCell, GpuClothSelfParams};
use super::gpu_test_support::{
    compile_collision_wgsl, find_entry_point, try_compute_device, PARITY_EPS,
};

/// 链表终止哨兵，与 `cloth_collision.wesl` 的 `CLOTH_COL_SENTINEL` 一致。
const SENTINEL: u32 = 0xffff_ffffu32;

/// 复刻 `cloth_collision.wesl` 的 `cloth_cell_of`：`floor(pos / cell_size)` 后取整。
fn cell_of_cpu(pos: [f32; 3], cell_size: f32) -> [i32; 3] {
    let inv = 1.0 / cell_size;
    [
        (pos[0] * inv).floor() as i32,
        (pos[1] * inv).floor() as i32,
        (pos[2] * inv).floor() as i32,
    ]
}

/// 复刻 `cloth_collision.wesl` 的 `cloth_cell_hash`：大质数 multiply-xor 空间哈希，
/// 对 `table_size` 取模。整型 `i32 -> u32` 走位模式重解释（与 WESL `bitcast` 相同），
/// 乘法用 wrapping 复刻 GPU 的模 `2^32` 溢出语义。
fn cell_hash_cpu(cell: [i32; 3], table_size: u32) -> u32 {
    let x = (cell[0] as u32).wrapping_mul(73_856_093u32);
    let y = (cell[1] as u32).wrapping_mul(19_349_663u32);
    let z = (cell[2] as u32).wrapping_mul(83_492_791u32);
    (x ^ y ^ z) % table_size
}

/// 两个自碰撞内核共用的 group-0 布局：`self_positions`（rw storage）、`cell_table`
/// （rw storage）、`particle_next`（rw storage）、`self_params`（uniform）。与 WESL
/// 的 `@binding(0..3)` 一一对应。
fn self_bind_group_layout(device: &wgpu::Device) -> BindGroupLayout {
    let storage = |binding: u32| BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("cloth_self_collision_parity_group0"),
        entries: &[
            storage(0),
            storage(1),
            storage(2),
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

/// 一次自碰撞 dispatch 读回的三份缓冲。
struct SelfReadback {
    /// resolve 后的 `xyzw`（`w` 为逆质量直通字段）。
    positions: Vec<[f32; 4]>,
    /// build 后的每桶链表头 + 占用计数。resolve 只读不写它，故这份即 build 结果。
    cell_table: Vec<GpuClothHashCell>,
    /// build 后的每粒子「链上下一个」指针。
    particle_next: Vec<u32>,
}

/// 在真机上跑一遍自碰撞：先 dispatch `hash_build`，可选再 dispatch `resolve`，一次
/// 读回三份缓冲。
///
/// `cell_table` 建前清成 `{head: SENTINEL, count: 0}`（镜像 CPU 每次重建全新
/// `BTreeMap`）；`particle_next` 建前置零（build 会覆写每个粒子的槽）。resolve 只写
/// `self_positions`、不动 `cell_table` / `particle_next`，所以两个 pass 后读回的链表
/// 就是 build 的结果，而 `positions` 是 resolve 的结果。
fn run_self_collision_on_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    build_entry: &str,
    resolve_entry: &str,
    positions: &[[f32; 4]],
    table_size: u32,
    cell_size: f32,
    thickness: f32,
    run_resolve: bool,
) -> SelfReadback {
    let count = positions.len();
    let vec4_bytes = size_of_val(positions) as u64;
    let cell_init = vec![
        GpuClothHashCell {
            head: SENTINEL,
            count: 0,
        };
        table_size as usize
    ];
    let cell_bytes = (table_size as u64) * size_of::<GpuClothHashCell>() as u64;
    let next_init = vec![0u32; count];
    let next_bytes = (count as u64) * size_of::<u32>() as u64;

    let layout = self_bind_group_layout(device);
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("cloth_self_collision_parity_layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("cloth_self_collision_parity"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });
    let build_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("cloth_self_collision_build_pipeline"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some(build_entry),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });
    let resolve_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("cloth_self_collision_resolve_pipeline"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some(resolve_entry),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });

    let positions_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_self_positions"),
        contents: bytemuck::cast_slice(positions),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });
    let cell_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_self_cell_table"),
        contents: bytemuck::cast_slice(&cell_init),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });
    let next_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_self_particle_next"),
        contents: bytemuck::cast_slice(&next_init),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });
    let params = GpuClothSelfParams {
        particle_count: count as u32,
        table_size,
        cell_size,
        thickness,
    };
    let params_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_self_params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("cloth_self_collision_parity_bind"),
        layout: &layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: positions_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: cell_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: next_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let pos_stage = device.create_buffer(&BufferDescriptor {
        label: Some("cloth_self_positions_stage"),
        size: vec4_bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let cell_stage = device.create_buffer(&BufferDescriptor {
        label: Some("cloth_self_cell_stage"),
        size: cell_bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let next_stage = device.create_buffer(&BufferDescriptor {
        label: Some("cloth_self_next_stage"),
        size: next_bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("cloth_self_collision_parity_encoder"),
    });
    // 每 invocation 一个粒子，@workgroup_size(64) => ceil(count / 64) 组。
    let groups = (count as u32).div_ceil(64).max(1);
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("cloth_self_collision_build_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&build_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    if run_resolve {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("cloth_self_collision_resolve_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&resolve_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&positions_buf, 0, &pos_stage, 0, vec4_bytes);
    encoder.copy_buffer_to_buffer(&cell_buf, 0, &cell_stage, 0, cell_bytes);
    encoder.copy_buffer_to_buffer(&next_buf, 0, &next_stage, 0, next_bytes);
    queue.submit([encoder.finish()]);

    pos_stage.slice(..).map_async(MapMode::Read, |_| {});
    cell_stage.slice(..).map_async(MapMode::Read, |_| {});
    next_stage.slice(..).map_async(MapMode::Read, |_| {});
    device
        .poll(PollType::wait_indefinitely())
        .expect("device poll should complete the submitted work");

    let pos_view = pos_stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped positions readback range should be available after poll");
    let positions_out: Vec<[f32; 4]> = bytemuck::cast_slice::<u8, [f32; 4]>(&pos_view).to_vec();
    drop(pos_view);
    pos_stage.unmap();

    let cell_view = cell_stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped cell-table readback range should be available after poll");
    let cell_out: Vec<GpuClothHashCell> =
        bytemuck::cast_slice::<u8, GpuClothHashCell>(&cell_view).to_vec();
    drop(cell_view);
    cell_stage.unmap();

    let next_view = next_stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped particle-next readback range should be available after poll");
    let next_out: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&next_view).to_vec();
    drop(next_view);
    next_stage.unmap();

    SelfReadback {
        positions: positions_out,
        cell_table: cell_out,
        particle_next: next_out,
    }
}

/// 建一个自由粒子（`inverse_mass = 1`）。
fn free(x: f32, y: f32, z: f32) -> ClothParticle {
    ClothParticle::new(Vec3::new(x, y, z), 1.0)
}

/// 建一个 pinned 粒子（`inverse_mass = 0`，resolve 永不移动它）。
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

/// 对一组粒子同时跑 CPU 黄金 [`resolve_self_collision`] 与真机 build+resolve，断言
/// 逐粒子 `xyz` 落在 [`PARITY_EPS`] 内、`w`（逆质量）不被内核改动。仅用于**单接触**
/// 几何（每粒子至多一个接触邻居），此时 Jacobi 与 CPU 的 Gauss-Seidel 写集一致。
fn assert_self_resolve_parity(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    build_entry: &str,
    resolve_entry: &str,
    particles: &[ClothParticle],
    table_size: u32,
    cell_size: f32,
    thickness: f32,
) {
    let mut golden = particles.to_vec();
    resolve_self_collision(&mut golden, cell_size, thickness);

    let positions = upload_positions(particles);
    let readback = run_self_collision_on_gpu(
        device,
        queue,
        wgsl,
        build_entry,
        resolve_entry,
        &positions,
        table_size,
        cell_size,
        thickness,
        true,
    );

    assert_eq!(readback.positions.len(), golden.len());
    for (i, (gpu, cpu)) in readback.positions.iter().zip(golden.iter()).enumerate() {
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
        assert!(
            (gpu[3] - particles[i].inverse_mass).abs() <= f32::EPSILON,
            "particle {i}: inverse mass mutated {} -> {}",
            particles[i].inverse_mass,
            gpu[3],
        );
    }
}

/// build 阶段：每个粒子必须恰好插进它自己格子的桶一次。读回链表，按 WESL 的
/// `cell_of` + `cell_hash` 复算每个粒子的期望桶，断言每桶链上的成员集合与 `count`
/// 与期望一致，且全体粒子跨所有桶合计恰好各出现一次——这正是 per-particle launch
/// 修复（旧 `4x4x4` tiling 会重复插 16 次）要锁死的不变量。
#[test]
#[expect(
    clippy::print_stderr,
    reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志"
)]
fn hash_build_chains_each_particle_into_its_bucket_once() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!(
            "hash_build_chains_each_particle_into_its_bucket_once: no wgpu adapter, skipping"
        );
        return;
    };
    let wgsl = compile_collision_wgsl();
    let build_entry = find_entry_point(&wgsl, "cloth_self_collision_hash_build");
    let resolve_entry = find_entry_point(&wgsl, "cloth_self_collision_resolve");

    // 一组散落在不同格子的粒子（含一个与 [0] 同格的近邻，制造一个双成员桶）。
    let cell_size = 0.5f32;
    let table_size = 97u32;
    let particles = [
        free(0.1, 0.1, 0.1),
        free(0.2, 0.15, 0.05),
        free(3.4, -1.2, 2.7),
        free(-2.1, 4.9, -0.3),
        free(10.0, 10.0, 10.0),
        free(-5.5, -5.5, -5.5),
    ];
    let positions = upload_positions(&particles);

    // 只跑 build（resolve 不改链表，但省去无谓 dispatch 让意图更清晰）。
    let readback = run_self_collision_on_gpu(
        &device,
        &queue,
        &wgsl,
        &build_entry,
        &resolve_entry,
        &positions,
        table_size,
        cell_size,
        1.0,
        false,
    );

    // 期望：每个粒子按 cell_of + cell_hash 落进一个桶。
    let mut expected: Vec<Vec<u32>> = vec![Vec::new(); table_size as usize];
    for (i, p) in particles.iter().enumerate() {
        let cell = cell_of_cpu([p.position.x, p.position.y, p.position.z], cell_size);
        let bucket = cell_hash_cpu(cell, table_size) as usize;
        expected[bucket].push(i as u32);
    }

    // 逐桶：走链表收集成员集合，与期望集合（顺序无关）+ count 对齐。
    let mut total_seen = 0usize;
    for bucket in 0..table_size as usize {
        let mut chain = Vec::new();
        let mut j = readback.cell_table[bucket].head;
        let mut guard = 0u32;
        while j != SENTINEL {
            assert!(
                (j as usize) < particles.len(),
                "bucket {bucket}: chain index {j} out of range"
            );
            chain.push(j);
            j = readback.particle_next[j as usize];
            guard += 1;
            assert!(
                guard <= particles.len() as u32,
                "bucket {bucket}: chain longer than particle count (cycle?)"
            );
        }
        assert_eq!(
            readback.cell_table[bucket].count as usize,
            expected[bucket].len(),
            "bucket {bucket}: count mismatch"
        );
        let mut got = chain.clone();
        got.sort_unstable();
        let mut want = expected[bucket].clone();
        want.sort_unstable();
        assert_eq!(got, want, "bucket {bucket}: chain membership mismatch");
        total_seen += chain.len();
    }
    // 全体粒子跨所有桶合计恰好各出现一次（旧 bug 会让它变成 16x）。
    assert_eq!(
        total_seen,
        particles.len(),
        "every particle must be chained exactly once across all buckets"
    );
}

/// resolve：两个自由粒子 0.4 间距、thickness 1 => 对称推到 1.0 间距（-0.3 / 0.7）。
#[test]
#[expect(
    clippy::print_stderr,
    reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志"
)]
fn resolve_separates_two_close_free_particles() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("resolve_separates_two_close_free_particles: no wgpu adapter, skipping");
        return;
    };
    let wgsl = compile_collision_wgsl();
    let build_entry = find_entry_point(&wgsl, "cloth_self_collision_hash_build");
    let resolve_entry = find_entry_point(&wgsl, "cloth_self_collision_resolve");

    let particles = [free(0.0, 0.0, 0.0), free(0.4, 0.0, 0.0)];
    assert_self_resolve_parity(
        &device,
        &queue,
        &wgsl,
        &build_entry,
        &resolve_entry,
        &particles,
        97,
        1.0,
        1.0,
    );
}

/// resolve：pinned + free 0.4 间距 => 只有自由粒子移动，独吞全部 0.6 走到 1.0 间距。
#[test]
#[expect(
    clippy::print_stderr,
    reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志"
)]
fn resolve_pinned_partner_takes_no_correction() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("resolve_pinned_partner_takes_no_correction: no wgpu adapter, skipping");
        return;
    };
    let wgsl = compile_collision_wgsl();
    let build_entry = find_entry_point(&wgsl, "cloth_self_collision_hash_build");
    let resolve_entry = find_entry_point(&wgsl, "cloth_self_collision_resolve");

    let particles = [pinned(0.0, 0.0, 0.0), free(0.4, 0.0, 0.0)];
    assert_self_resolve_parity(
        &device,
        &queue,
        &wgsl,
        &build_entry,
        &resolve_entry,
        &particles,
        97,
        1.0,
        1.0,
    );
}

/// resolve：相距 5、thickness 1 => 不接触，逐比特 no-op。
#[test]
#[expect(
    clippy::print_stderr,
    reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志"
)]
fn resolve_leaves_distant_particles_untouched() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("resolve_leaves_distant_particles_untouched: no wgpu adapter, skipping");
        return;
    };
    let wgsl = compile_collision_wgsl();
    let build_entry = find_entry_point(&wgsl, "cloth_self_collision_hash_build");
    let resolve_entry = find_entry_point(&wgsl, "cloth_self_collision_resolve");

    let particles = [free(0.0, 0.0, 0.0), free(5.0, 0.0, 0.0)];
    assert_self_resolve_parity(
        &device,
        &queue,
        &wgsl,
        &build_entry,
        &resolve_entry,
        &particles,
        97,
        1.0,
        1.0,
    );
}

/// resolve：两个**重合**自由粒子 => 沿带符号 `X` 轴对称分开（0.5 / 1.5）。这是
/// coincident 分支修复（旧代码两粒子同推 `+X`、仍旧重合）要锁死的行为。
#[test]
#[expect(
    clippy::print_stderr,
    reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志"
)]
fn resolve_coincident_particles_separate_symmetrically() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("resolve_coincident_particles_separate_symmetrically: no wgpu adapter, skipping");
        return;
    };
    let wgsl = compile_collision_wgsl();
    let build_entry = find_entry_point(&wgsl, "cloth_self_collision_hash_build");
    let resolve_entry = find_entry_point(&wgsl, "cloth_self_collision_resolve");

    let particles = [free(1.0, 1.0, 1.0), free(1.0, 1.0, 1.0)];
    assert_self_resolve_parity(
        &device,
        &queue,
        &wgsl,
        &build_entry,
        &resolve_entry,
        &particles,
        97,
        1.0,
        1.0,
    );
}

/// resolve：跨格子边界（cell_size 0.5，x = 0.45 在格 0、x = 0.55 在格 1）仍被 27
/// 邻域检出并推到 thickness 0.5 间距。锁死邻域遍历跨边界的正确性。
#[test]
#[expect(
    clippy::print_stderr,
    reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志"
)]
fn resolve_detects_pairs_across_cell_boundaries() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("resolve_detects_pairs_across_cell_boundaries: no wgpu adapter, skipping");
        return;
    };
    let wgsl = compile_collision_wgsl();
    let build_entry = find_entry_point(&wgsl, "cloth_self_collision_hash_build");
    let resolve_entry = find_entry_point(&wgsl, "cloth_self_collision_resolve");

    let particles = [free(0.45, 0.0, 0.0), free(0.55, 0.0, 0.0)];
    assert_self_resolve_parity(
        &device,
        &queue,
        &wgsl,
        &build_entry,
        &resolve_entry,
        &particles,
        97,
        0.5,
        0.5,
    );
}
