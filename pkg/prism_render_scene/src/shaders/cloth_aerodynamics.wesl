//! `GPU` 布料气动力 gather 内核（arch `accumulate_aero_gather` 的孪生）。
//!
//! 这是 `prism_render_architecture::cloth::wind` 里 CPU 黄金参考
//! `accumulate_aero_gather` 的 `GPU` 孪生，逐字节对齐。气动力是每帧一次的预解
//! 速度冲量：在 XPBD 位置求解之前，按当前风场对布料网格施加一次风力，把结果直
//! 接累加进 `velocities`。这里的 `dt` 是整帧 dt（不是 XPBD 的 substep dt），因
//! 为气动冲量按整帧施加一次。
//!
//! 采用 per-vertex gather：每个顶点通过 `CSR` 邻接表 `csr_offsets` /
//! `csr_entries` 找到自己一环邻接的三角面，对每个面计算风力，再把面力平均摊到
//! 三个顶点（`× 1/3`），只取属于当前顶点的那一份累加到自身速度。相较 per-face
//! scatter，gather 无需原子写、写集合互不重叠，天然确定。
//!
//! 关键约束：面速度**必须**从只读快照 `velocity_snapshot` 读取，而不是从本 pass
//! 正在写入的 `velocities` 读取。`velocity_snapshot` 由前置 pass
//! `cloth_aerodynamics_snapshot` 冻结、本 pass 不写它，因此每个 invocation 看到
//! 的邻居面速度都是同一份帧初值，与 `GPU` warp 调度无关——这是确定性与无数据竞
//! 争的关键，也正是 CPU 黄金参考「先拍快照再 gather」的镜像。
//!
//! 沙盒环境没有 `GPU`，本模块仅通过 naga 编译（见 `shader_tests`）验证；它是气
//! 动力两段式 dispatch 中的第二段。
//!
//! 缓冲布局镜像 `prism_render_architecture::cloth::gpu::buffers`：
//! `positions[v].xyz` 是世界坐标、`.w` 是 inverse mass（`<= 0` = pin，跳过）。
//! `triangles` 每三角 3 个扁平 `u32`（三角 `t` 的顶点是 `triangles[t*3+0/1/2]`）。
//! `csr_offsets` 长度 `vertex_count + 1`，`csr_entries` 存三角形索引。

/// 叉积长度平方下限，低于它认为三角退化（零面积），跳过以避免除以约 0
/// （镜像 CPU 的 `EPS_LEN_SQ`）。
const CLOTH_EPS_LEN_SQ: f32 = 1.0e-12;

/// 气动力全局参数（两文件共用；此处独立声明一份，32 字节，字段偏移严格对齐
/// arch 的 `GpuClothAeroParams`）。
struct GpuClothAeroParams {
    /// 世界系风速，世界单位每秒（偏移 0）。
    wind_velocity: vec3<f32>,
    /// 每三角湍流扰动幅度（偏移 12）；`<= 0` 时禁用扰动。
    turbulence: f32,
    /// 法向（压差）阻力系数（偏移 16）；负值会被 `max(_, 0)` 夹为非负。
    drag: f32,
    /// 切向（升力）系数（偏移 20）；负值会被 `max(_, 0)` 夹为非负。
    lift: f32,
    /// 整帧 dt，秒（偏移 24）；`NaN` / `<= 0` 天然跳过。
    dt: f32,
    /// 顶点数量，界定 per-vertex dispatch（偏移 28）。
    vertex_count: u32,
}

/// 世界坐标 + inverse mass（`.w`），只读。
@binding(0) var<storage, read> positions: array<vec4<f32>>;
/// 顶点速度，本 pass 把气动冲量累加写回（读自身、不读作面速度）。
@binding(1) var<storage, read_write> velocities: array<vec4<f32>>;
/// 帧初速度快照，只读；面相对风速一律取自这里以保证确定性。
@binding(2) var<storage, read> velocity_snapshot: array<vec4<f32>>;
/// 三角顶点索引，扁平存储（每三角 3 个 `u32`）。
@binding(3) var<storage, read> triangles: array<u32>;
/// `CSR` 行偏移，长度 `vertex_count + 1`。
@binding(4) var<storage, read> csr_offsets: array<u32>;
/// `CSR` 邻接项，存三角形索引。
@binding(5) var<storage, read> csr_entries: array<u32>;
/// 气动力全局参数。
@binding(6) var<uniform> aero_params: GpuClothAeroParams;

/// 把 32 位种子雪崩混合成 `[-1, 1]` 区间内的单位标量。
///
/// 逐位复刻 CPU 侧的整数哈希：三轮「乘常数 + 右移异或」雪崩，取高 24 位映射到
/// `[0, 1)` 再线性映射到 `[-1, 1]`。`WESL` 的 `u32` 乘法与 CPU 端一样是自然
/// wrapping，因此结果逐位一致。
fn hash_to_unit(seed: u32) -> f32 {
    var h = seed * 0x9E3779B1u;
    h ^= h >> 15u;
    h = h * 0x85EBCA77u;
    h ^= h >> 13u;
    h = h * 0xC2B2AE3Du;
    h ^= h >> 16u;
    let unit = f32(h >> 8u) * (1.0 / 16777216.0);
    return unit * 2.0 - 1.0;
}

/// 由三角三个顶点索引确定性地生成一个方向扰动向量，用于模拟湍流。
///
/// `turb <= 0` 时直接返回零向量（禁用）。否则用三个大素数把顶点索引混成一个
/// per-triangle 基种子 `base`，再用三个不同掩码异或后分别哈希出 x/y/z 三个
/// `[-1, 1]` 分量并整体乘以 `turb`。同一三角每帧得到同一扰动，无需外部随机数。
fn turbulence_offset(i0: u32, i1: u32, i2: u32, turb: f32) -> vec3<f32> {
    if turb <= 0.0 {
        return vec3<f32>(0.0);
    }
    let base = (i0 * 73856093u) ^ (i1 * 19349663u) ^ (i2 * 83492791u);
    return vec3<f32>(
        hash_to_unit(base ^ 0x00A55A00u),
        hash_to_unit(base ^ 0x5A0000A5u),
        hash_to_unit(base ^ 0x00FF00FFu),
    ) * turb;
}

/// 计算单个三角面受到的风力（法向阻力 + 切向升力，按面积缩放）。
///
/// 先用两条边的叉积求面法向与面积：叉积长度平方低于 `CLOTH_EPS_LEN_SQ` 视为退
/// 化三角，返回零力。面速度取三顶点速度均值，相对风速 `relative = wind - 面速度`
/// 拆成法向分量与切向分量：法向分量乘 `drag`（迎风压差），切向分量乘 `lift`
/// （掠面升力），二者之和再乘以三角面积。返回的是整个面的力（调用方再摊到顶点）。
fn triangle_wind_force(
    p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>,
    v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>,
    wind: vec3<f32>, drag: f32, lift: f32,
) -> vec3<f32> {
    let cross_vec = cross(p1 - p0, p2 - p0);
    let cross_len_sq = dot(cross_vec, cross_vec);
    if cross_len_sq <= CLOTH_EPS_LEN_SQ {
        return vec3<f32>(0.0);
    }
    let area = 0.5 * sqrt(cross_len_sq);
    let normal = cross_vec * inverseSqrt(cross_len_sq);
    let face_vel = (v0 + v1 + v2) * (1.0 / 3.0);
    let relative = wind - face_vel;
    let normal_comp = normal * dot(relative, normal);
    let tangent = relative - normal_comp;
    return (normal_comp * drag + tangent * lift) * area;
}

/// 气动力 gather 主内核：读快照面速度，把每帧一次的风力冲量累加进自身速度。
///
/// 一个 invocation 负责一个顶点：越界或 pin（`inv_mass <= 0`）跳过；`dt` 为
/// `NaN` 或 `<= 0` 时经 `!(dt > 0.0)` 天然跳过；`drag` / `lift` 用 `max(_, 0)`
/// 夹为非负（镜像 CPU sanitized 参数）。随后沿 `CSR` 遍历本顶点一环邻接三角，
/// 每个面叠加确定性湍流扰动后求风力，摊 `1/3` 累加，最后按 `inv_mass * dt`
/// 转成速度增量写回，`.w` 载荷位保持不变。
@compute @workgroup_size(64)
fn cloth_aerodynamics(@builtin(global_invocation_id) gid: vec3<u32>) {
    let v = gid.x;
    if v >= aero_params.vertex_count {
        return;
    }
    let inv_mass = positions[v].w;
    if inv_mass <= 0.0 {
        return;
    }
    let dt = aero_params.dt;
    if !(dt > 0.0) {
        return;
    }
    let drag = max(aero_params.drag, 0.0);
    let lift = max(aero_params.lift, 0.0);
    var accum = vec3<f32>(0.0);
    let start = csr_offsets[v];
    let end = csr_offsets[v + 1u];
    for (var e = start; e < end; e = e + 1u) {
        let t = csr_entries[e];
        let b = t * 3u;
        let i0 = triangles[b + 0u];
        let i1 = triangles[b + 1u];
        let i2 = triangles[b + 2u];
        let wind_vec = aero_params.wind_velocity + turbulence_offset(i0, i1, i2, aero_params.turbulence);
        let force = triangle_wind_force(
            positions[i0].xyz, positions[i1].xyz, positions[i2].xyz,
            velocity_snapshot[i0].xyz, velocity_snapshot[i1].xyz, velocity_snapshot[i2].xyz,
            wind_vec, drag, lift,
        );
        accum = accum + force * (1.0 / 3.0);
    }
    velocities[v] = vec4<f32>(velocities[v].xyz + accum * inv_mass * dt, velocities[v].w);
}
